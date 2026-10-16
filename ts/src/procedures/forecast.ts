// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Forecasting: exponential smoothing, autoregression, model averaging, conformal intervals and comparison; mirrors procedures/forecast.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { logSumExp, neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, fromIdentity, identity, type Identity, type Output, register, requireState, type State } from "../procedure.js";
import type { SplitMix64 } from "../rng.js";
import { erfc, lgamma, tCdf, tIsfLog } from "./distributions.js";
import { normalInvCdf } from "./group_sequential.js";
import { BaseProcedure, bool, type Config, int, invalid, num, nums, obj, optNum, pyMax, pyMin, str } from "./util.js";

const LOG_2PI = Math.log(2.0 * Math.PI);
const SQRT2 = Math.SQRT2;

function checkCoverage(coverage: number): number {
  if (!(0.0 < coverage && coverage < 1.0)) invalid("coverage must lie in (0, 1)");
  return coverage;
}

function checkHorizon(horizon: number): number {
  if (!(Number.isInteger(horizon) && horizon >= 1)) invalid("horizon must be a positive int");
  return horizon;
}

/** Log density of a normal (df null) or Student t location-scale law at `error` from its location. */
export function logDensity(error: number, scale: number, df: number | null): number {
  const z = error / scale;
  if (df === null) return -0.5 * LOG_2PI - Math.log(scale) - 0.5 * z * z;
  return (lgamma(0.5 * (df + 1.0)) - lgamma(0.5 * df) - 0.5 * Math.log(df * Math.PI)
    - Math.log(scale) - 0.5 * (df + 1.0) * Math.log1p(z * z / df));
}

export function standardCdf(z: number, df: number | null): number {
  return df === null ? 0.5 * erfc(-z / SQRT2) : tCdf(z, df);
}

/** The forecast of the next 1..horizon values: location + scale * (standard normal or t_df). */
export interface Predictive {
  mean: number[];
  scale: number[];
  df: number | null;
}

/** (new state, error, log score) after one observation. */
export type Updated = [State, number | null, number | null];

function optStr(c: Config, key: string): string | null {
  return c[key] === null ? null : str(c, key);
}
function optInt(c: Config, key: string): number | null {
  return c[key] === null ? null : int(c, key);
}
const isNumber = (v: Json): v is number => typeof v === "number" && Number.isFinite(v);
const isCount = (v: Json): v is number => isNumber(v) && Number.isInteger(v) && v >= 0;

/** Common protocol: `update` scores and absorbs one value, `predictive` forecasts from a state. */
export abstract class Forecaster extends BaseProcedure {
  abstract readonly horizon: number;
  abstract readonly coverage: number;
  abstract readonly input: string;

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  abstract update(state: State, y: number): Updated;
  abstract predictive(state: State): Predictive | null;

  parameters(_state: State): { [key: string]: Json } {
    return {};
  }

  logLikelihood(state: State): number {
    return (state.log_likelihood as number) + (state.comp as number);
  }

  interval(pred: Predictive): [number[], number[]] {
    const q = this.quantile(pred.df);
    const lower = pred.mean.map((m, i) => m - q * pred.scale[i]);
    const upper = pred.mean.map((m, i) => m + q * pred.scale[i]);
    return [lower, upper];
  }

  quantile(df: number | null): number {
    const tail = 0.5 * (1.0 - this.coverage);
    if (df === null) return -normalInvCdf(tail);
    return tIsfLog(Math.log(tail), df);
  }

  step(state: State, x: StatInput, _rng: SplitMix64 | null = null): [State, Output] {
    const y = x[this.input] as number;
    const [next, error, score] = this.update(state, y);
    const pred = this.predictive(next);
    const n = next.n as number;
    const logLik = requireFinite(this.logLikelihood(next), "log likelihood");
    if (pred === null) {
      return [next, {
        n, mean: [], scale: [], df: null, lower: [], upper: [], error, log_score: score, log_likelihood: logLik,
        parameters: this.parameters(next),
      }];
    }
    const [lower, upper] = this.interval(pred);
    return [next, {
      n, mean: pred.mean, scale: pred.scale, df: pred.df, lower, upper, error, log_score: score, log_likelihood: logLik,
      parameters: this.parameters(next),
    }];
  }
}

// ---------------------------------------------------------------------------------------- ETS

type ETSState = {
  n: number; level: number; trend: number; seasonal: number[]; errors: number; sse: number; sse_comp: number;
  log_likelihood: number; comp: number;
};

const TRENDS: (string | null)[] = [null, "additive", "damped"];

/** Additive-error exponential smoothing (ETS) with optional trend and season; see forecast.py. */
export class ExponentialSmoothing extends Forecaster {
  readonly name = "exponential_smoothing";
  readonly version = "1";
  readonly initialSeasonal: number[];
  readonly horizon: number;
  readonly coverage: number;
  private readonly damping: number[];
  private readonly factors: number[];

  constructor(
    readonly alpha: number, readonly beta = 0.0, readonly gamma = 0.0, readonly phi = 1.0, readonly trend: string | null = null,
    readonly period: number | null = null, readonly initialLevel: number | null = null, readonly initialTrend = 0.0,
    initialSeasonal: number[] | null = null, readonly sigma: number | null = null, horizon = 1, coverage = 0.9, readonly input = "y",
  ) {
    super();
    if (!TRENDS.includes(trend)) invalid(`trend must be one of (None, 'additive', 'damped')`);
    if (!(0.0 < alpha && alpha <= 1.0)) invalid("alpha must lie in (0, 1]");
    if (!(0.0 <= beta && beta <= 1.0) || !(0.0 <= gamma && gamma <= 1.0)) invalid("beta and gamma must lie in [0, 1]");
    if (!(0.0 < phi && phi <= 1.0)) invalid("phi must lie in (0, 1]");
    if (trend === null && (beta !== 0.0 || initialTrend !== 0.0)) invalid("beta and initial_trend need a trend");
    if (trend !== "damped" && phi !== 1.0) invalid("phi < 1 needs trend='damped'");
    if (period === null) {
      if (gamma !== 0.0 || initialSeasonal !== null) invalid("gamma and initial_seasonal need a period");
    } else if (!(Number.isInteger(period) && period >= 2)) invalid("period must be an int >= 2");
    if (sigma !== null && !(0.0 < sigma && sigma < Infinity)) invalid("sigma must be positive");
    for (const v of [initialLevel, initialTrend]) if (v !== null && !Number.isFinite(v)) invalid("initial values must be finite");
    const seasonal = initialSeasonal !== null ? [...initialSeasonal] : period !== null ? Array.from({ length: period }, () => 0.0) : [];
    if (period !== null && seasonal.length !== period) invalid("initial_seasonal needs one value per season");
    if (!seasonal.every((v) => Number.isFinite(v))) invalid("initial values must be finite");
    this.initialSeasonal = seasonal;
    this.horizon = checkHorizon(horizon);
    this.coverage = checkCoverage(coverage);
    // phi_h = phi + ... + phi^h multiplies the trend; v_h is the variance factor of the h-step forecast
    const damping: number[] = [], factors: number[] = [];
    let acc = 0.0, power = 1.0, v = 1.0;
    for (let h = 1; h <= this.horizon; h++) {
      if (h > 1) {
        const c = this.alpha + this.beta * acc + (period !== null && (h - 1) % period === 0 ? this.gamma : 0.0);
        v += c * c;
      }
      power *= this.phi;
      acc += power;
      damping.push(acc);
      factors.push(Math.sqrt(v));
    }
    this.damping = damping;
    this.factors = factors;
  }

  static fromConfig(c: Config): ExponentialSmoothing {
    const period = optInt(c, "period");
    const seasonal = c.initial_seasonal === null ? null : nums(c, "initial_seasonal");
    return new ExponentialSmoothing(num(c, "alpha"), num(c, "beta"), num(c, "gamma"), num(c, "phi"), optStr(c, "trend"), period,
      optNum(c, "initial_level"), num(c, "initial_trend"), seasonal,
      optNum(c, "sigma"), int(c, "horizon"), num(c, "coverage"), str(c, "input"));
  }

  config(): Config {
    return {
      alpha: this.alpha, beta: this.beta, gamma: this.gamma, phi: this.phi, trend: this.trend, period: this.period,
      initial_level: this.initialLevel, initial_trend: this.initialTrend,
      initial_seasonal: this.period !== null ? [...this.initialSeasonal] : null, sigma: this.sigma,
      horizon: this.horizon, coverage: this.coverage, input: this.input,
    };
  }

  initialState(): ETSState {
    const level = this.initialLevel === null ? 0.0 : this.initialLevel;
    return {
      n: 0, level, trend: this.initialTrend, seasonal: [...this.initialSeasonal], errors: 0, sse: 0.0, sse_comp: 0.0,
      log_likelihood: 0.0, comp: 0.0,
    };
  }

  private started(s: ETSState): boolean {
    return s.n > 0 || this.initialLevel !== null;
  }

  private rotate(seasonal: number[], next: number): number[] {
    return seasonal.length ? [...seasonal.slice(1), next] : [];
  }

  /** [sigma, df] of the one-step forecast, or null while sigma cannot be estimated. */
  private noise(s: ETSState): [number, number | null] | null {
    if (this.sigma !== null) return [this.sigma, null];
    const sse = s.sse + s.sse_comp;
    if (s.errors < 1 || !(sse > 0.0)) return null;
    return [Math.sqrt(sse / s.errors), s.errors];
  }

  update(state: State, y: number): Updated {
    const s = state as ETSState;
    const season = s.seasonal.length ? s.seasonal[0] : 0.0;
    if (!this.started(s)) {
      const fresh: ETSState = {
        n: 1, level: y - season, trend: s.trend, seasonal: this.rotate(s.seasonal, season), errors: 0, sse: 0.0, sse_comp: 0.0,
        log_likelihood: s.log_likelihood, comp: s.comp,
      };
      return [fresh, null, null];
    }
    const error = y - (s.level + this.phi * s.trend + season);
    const noise = this.noise(s);
    const score = noise === null ? null : logDensity(error, noise[0], noise[1]);
    const level = requireFinite(s.level + this.phi * s.trend + this.alpha * error, "level");
    const trend = requireFinite(this.phi * s.trend + this.beta * error, "trend");
    const seasonal = this.rotate(s.seasonal, season + this.gamma * error);
    const [sse, sseComp] = neumaierAdd(s.sse, s.sse_comp, error * error);
    let logLik = s.log_likelihood, comp = s.comp;
    if (score !== null) [logLik, comp] = neumaierAdd(logLik, comp, score);
    const next: ETSState = {
      n: s.n + 1, level, trend, seasonal, errors: s.errors + 1, sse, sse_comp: sseComp, log_likelihood: logLik, comp,
    };
    return [next, error, score];
  }

  predictive(state: State): Predictive | null {
    const s = state as ETSState;
    const noise = this.started(s) ? this.noise(s) : null;
    if (noise === null) return null;
    const [sigma, df] = noise;
    const mean: number[] = [];
    for (let h = 1; h <= this.horizon; h++) {
      const season = s.seasonal.length ? s.seasonal[(h - 1) % s.seasonal.length] : 0.0;
      mean.push(s.level + this.damping[h - 1] * s.trend + season);
    }
    return { mean, scale: this.factors.map((f) => sigma * f), df };
  }

  override parameters(state: State): { [key: string]: Json } {
    const s = state as ETSState;
    const noise = this.started(s) ? this.noise(s) : null;
    return { level: s.level, trend: s.trend, seasonal: [...s.seasonal], sigma: noise === null ? null : noise[0] };
  }

  decodeState(data: Json): ETSState {
    const s = decodeFields({
      n: "count", level: "number", trend: "number", seasonal: "numbers", errors: "count", sse: "number", sse_comp: "number",
      log_likelihood: "number", comp: "number",
    }, data) as ETSState;
    requireState(s.seasonal.length === this.initialSeasonal.length, "one seasonal effect per season");
    requireState(0 <= s.errors && s.errors <= s.n && s.sse >= 0, "error statistics");
    requireState(s.n > 0 || (s.errors === 0 && s.sse === 0 && s.log_likelihood === 0), "statistics before data");
    requireState(this.trend !== null || s.trend === 0, "a trend without a trend model");
    return s;
  }
}

register("exponential_smoothing", "1", (c) => ExponentialSmoothing.fromConfig(c));

// ------------------------------------------------------------------------------------------ AR

/** Cholesky factor of a symmetric positive definite matrix, pivots floored at 1e-13 of the diagonal. */
function choleskyFloor(a: number[][]): number[][] {
  const k = a.length;
  const low = Array.from({ length: k }, () => Array.from({ length: k }, () => 0.0));
  for (let j = 0; j < k; j++) {
    let d = a[j][j];
    for (let i = 0; i < j; i++) d -= low[j][i] * low[j][i];
    d = pyMax(d, 1e-13 * a[j][j]);
    low[j][j] = Math.sqrt(d);
    for (let r = j + 1; r < k; r++) {
      let v = a[r][j];
      for (let i = 0; i < j; i++) v -= low[r][i] * low[j][i];
      low[r][j] = v / low[j][j];
    }
  }
  return low;
}

function forward(low: number[][], b: number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < b.length; i++) {
    let v = b[i];
    for (let j = 0; j < i; j++) v -= low[i][j] * out[j];
    out.push(v / low[i][i]);
  }
  return out;
}

function backward(low: number[][], b: number[]): number[] {
  const k = b.length;
  const out = Array.from({ length: k }, () => 0.0);
  for (let i = k - 1; i >= 0; i--) {
    let v = b[i];
    for (let j = i + 1; j < k; j++) v -= low[j][i] * out[j];
    out[i] = v / low[i][i];
  }
  return out;
}

type ARState = {
  n: number; lags: number[]; information: number[][]; moment: number[]; shape: number; rate: number; log_likelihood: number;
  comp: number;
};

/** AR(p) forecasts, with fixed coefficients or learnt by Bayesian regression on the lags; see forecast.py. */
export class AutoRegressive extends Forecaster {
  readonly name = "autoregressive";
  readonly version = "1";
  readonly phi: number[] | null;
  readonly horizon: number;
  readonly coverage: number;

  constructor(
    readonly order: number, coefficients: number[] | null = null, readonly intercept = 0.0, readonly sigma: number | null = null,
    readonly forgetting = 1.0, readonly ridge = 1e-6, horizon = 1, coverage = 0.9, readonly input = "y",
  ) {
    super();
    if (!(Number.isInteger(order) && order >= 1)) invalid("order must be a positive int");
    if (coefficients !== null) {
      const phi = [...coefficients];
      if (phi.length !== order || !phi.every((c) => Number.isFinite(c)) || !Number.isFinite(intercept))
        invalid("need `order` finite coefficients and a finite intercept");
      if (sigma === null || !(0.0 < sigma && sigma < Infinity)) invalid("fixed coefficients need a positive sigma");
      if (forgetting !== 1.0 || ridge !== 1e-6) invalid("forgetting and ridge are for learnt coefficients");
      this.phi = phi;
    } else {
      if (sigma !== null || intercept !== 0.0) invalid("sigma and intercept come with fixed coefficients");
      if (!(0.0 < forgetting && forgetting <= 1.0)) invalid("forgetting must lie in (0, 1]");
      if (!(0.0 < ridge && ridge < Infinity)) invalid("ridge must be positive");
      this.phi = null;
    }
    this.horizon = checkHorizon(horizon);
    this.coverage = checkCoverage(coverage);
  }

  static fromConfig(c: Config): AutoRegressive {
    return new AutoRegressive(int(c, "order"), c.coefficients === null ? null : nums(c, "coefficients"), num(c, "intercept"),
      optNum(c, "sigma"), num(c, "forgetting"), num(c, "ridge"), int(c, "horizon"), num(c, "coverage"), str(c, "input"));
  }

  config(): Config {
    return {
      order: this.order, coefficients: this.phi === null ? null : [...this.phi], intercept: this.intercept, sigma: this.sigma,
      forgetting: this.forgetting, ridge: this.ridge, horizon: this.horizon, coverage: this.coverage, input: this.input,
    };
  }

  initialState(): ARState {
    if (this.phi !== null) return { n: 0, lags: [], information: [], moment: [], shape: 0.0, rate: 0.0, log_likelihood: 0.0, comp: 0.0 };
    const k = this.order + 1;
    return {
      n: 0, lags: [], information: Array.from({ length: k }, () => Array.from({ length: k }, () => 0.0)),
      moment: Array.from({ length: k }, () => 0.0), shape: 0.0, rate: 0.0, log_likelihood: 0.0, comp: 0.0,
    };
  }

  private point(theta: number[], lags: number[]): number {
    let total = theta[0];
    for (let i = 0; i < this.order; i++) total += theta[i + 1] * lags[i];
    return total;
  }

  /** Coefficients for the next value, and x' M^-1 x with M = lambda (information) + ridge I. */
  private prior(s: ARState, x: number[]): [number[], number] {
    const lam = this.forgetting, k = this.order + 1;
    const precision = Array.from({ length: k }, (_, i) =>
      Array.from({ length: k }, (_, j) => lam * s.information[i][j] + (i === j ? this.ridge : 0.0)));
    const low = choleskyFloor(precision);
    const theta = backward(low, forward(low, s.moment.map((v) => lam * v)));
    const w = forward(low, x);
    let quad = 0.0;
    for (const v of w) quad += v * v;
    return [theta, quad];
  }

  update(state: State, y: number): Updated {
    const s = state as ARState;
    const lags = [y, ...s.lags].slice(0, this.order);
    if (s.lags.length < this.order) return [{ ...s, n: s.n + 1, lags }, null, null];
    let logLik = s.log_likelihood, comp = s.comp;
    if (this.phi !== null) {
      const sigma = this.sigma as number;
      const error = y - this.point([this.intercept, ...this.phi], s.lags);
      const fixedScore = logDensity(error, sigma, null);
      [logLik, comp] = neumaierAdd(logLik, comp, fixedScore);
      const next: ARState = { n: s.n + 1, lags, information: [], moment: [], shape: 0.0, rate: 0.0, log_likelihood: logLik, comp };
      return [next, error, fixedScore];
    }
    const x = [1.0, ...s.lags];
    const k = x.length;
    const [theta, quad] = this.prior(s, x);
    const error = y - this.point(theta, s.lags);
    const sv = 1.0 + quad;
    let score: number | null = null;
    if (s.shape > 0.0 && s.rate > 0.0) {
      score = logDensity(error, Math.sqrt(s.rate / s.shape * sv), 2.0 * this.forgetting * s.shape);
      [logLik, comp] = neumaierAdd(logLik, comp, score);
    }
    const lam = this.forgetting;
    const information = Array.from({ length: k }, (_, i) =>
      Array.from({ length: k }, (_, j) => requireFinite(lam * s.information[i][j] + x[i] * x[j], "information")));
    const moment = Array.from({ length: k }, (_, i) => requireFinite(lam * s.moment[i] + x[i] * y, "moment"));
    const shape = lam * s.shape + 0.5;
    const rate = requireFinite(lam * s.rate + error * error / (2.0 * sv), "noise statistic");
    return [{ n: s.n + 1, lags, information, moment, shape, rate, log_likelihood: logLik, comp }, error, score];
  }

  private coefficients(s: ARState): [number[], number] {
    if (this.phi !== null) return [[this.intercept, ...this.phi], 0.0];
    return this.prior(s, [1.0, ...s.lags]);
  }

  predictive(state: State): Predictive | null {
    const s = state as ARState;
    if (s.lags.length < this.order) return null;
    let sigma2: number, df: number | null;
    if (this.phi !== null) {
      const sigma = this.sigma as number;
      [sigma2, df] = [sigma * sigma, null];
    } else if (s.shape > 0.0 && s.rate > 0.0) {
      [sigma2, df] = [s.rate / s.shape, 2.0 * this.forgetting * s.shape];
    } else return null;
    const [theta, extra] = this.coefficients(s);
    let history = [...s.lags];
    const psi = [1.0];
    const mean: number[] = [], scale: number[] = [];
    let total = 0.0;
    for (let h = 1; h <= this.horizon; h++) {
      const value = this.point(theta, history);
      mean.push(value);
      history = [value, ...history.slice(0, -1)];
      total += psi[h - 1] * psi[h - 1];
      scale.push(Math.sqrt(sigma2 * (extra + total)));
      let nxt = 0.0;
      for (let i = 1; i <= Math.min(h, this.order); i++) nxt += theta[i] * psi[h - i];
      psi.push(nxt);
    }
    return { mean, scale, df };
  }

  /** The coefficients and noise level the next forecast uses. */
  override parameters(state: State): { [key: string]: Json } {
    const s = state as ARState;
    if (this.phi !== null) return { intercept: this.intercept, coefficients: [...this.phi], sigma: this.sigma };
    if (s.lags.length < this.order) return { intercept: null, coefficients: null, sigma: null };
    const [theta] = this.coefficients(s);
    const sigma = s.shape > 0.0 && s.rate > 0.0 ? Math.sqrt(s.rate / s.shape) : null;
    return { intercept: theta[0], coefficients: theta.slice(1), sigma };
  }

  decodeState(data: Json): ARState {
    const s = decodeFields({
      n: "count", lags: "numbers", information: "json", moment: "numbers", shape: "number", rate: "number", log_likelihood: "number",
      comp: "number",
    }, data) as ARState;
    const info = s.information as Json;
    if (!Array.isArray(info) || !info.every((row) => Array.isArray(row) && row.every(isNumber)))
      throw new Error(`state field 'information' must be a list of lists of numbers, got ${JSON.stringify(info)}`);
    requireState(s.lags.length === Math.min(s.n, this.order), "one lag per observation, up to the order");
    if (this.phi !== null) {
      requireState(s.information.length === 0 && s.moment.length === 0 && s.shape === 0 && s.rate === 0,
        "learnt statistics with fixed coefficients");
    } else {
      const k = this.order + 1;
      requireState(info.length === k && s.moment.length === k && s.information.every((row) => row.length === k), "statistics' sizes");
      let ok = true;
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) ok = ok && s.information[i][i] >= 0 && s.information[i][j] === s.information[j][i];
      requireState(ok, "information must be symmetric with a non-negative diagonal");
      requireState(s.shape >= 0 && s.rate >= 0, "noise statistics");
    }
    return s;
  }
}

register("autoregressive", "1", (c) => AutoRegressive.fromConfig(c));

// ----------------------------------------------------------------------------- model averaging

type ModelAveragingState = { n: number; states: State[]; log_weights: number[]; log_likelihood: number; comp: number };
type Component = [number, Predictive];

/** Bayesian model averaging of forecasters, with optional forgetting (dynamic model averaging); see forecast.py. */
export class ModelAveraging extends Forecaster {
  readonly name = "model_averaging";
  readonly version = "1";
  readonly models: Forecaster[];
  readonly prior: number[];
  readonly horizon: number;
  readonly input: string;
  readonly coverage: number;
  private readonly logPrior: number[];

  constructor(models: Forecaster[], weights: number[] | null = null, readonly forgetting = 1.0, coverage = 0.9) {
    super();
    if (!models.length) invalid("at least one model is needed");
    for (const m of models)
      if (!(m instanceof ExponentialSmoothing || m instanceof AutoRegressive))
        invalid("models must be ExponentialSmoothing or AutoRegressive forecasters");
    if (new Set(models.map((m) => m.horizon)).size !== 1 || new Set(models.map((m) => m.input)).size !== 1)
      invalid("the models need the same horizon and the same input");
    const prior = weights !== null ? [...weights] : models.map(() => 1.0);
    if (prior.length !== models.length || !prior.every((w) => 0.0 < w && w < Infinity)) invalid("need one positive weight per model");
    if (!(0.0 < forgetting && forgetting <= 1.0)) invalid("forgetting must lie in (0, 1]");
    this.models = [...models];
    this.prior = prior;
    this.horizon = models[0].horizon;
    this.input = models[0].input;
    this.coverage = checkCoverage(coverage);
    const logs = prior.map((w) => Math.log(w));
    const top = logSumExp(logs);
    this.logPrior = logs.map((v) => v - top);
  }

  static fromConfig(c: Config): ModelAveraging {
    const raw = c.models;
    if (!Array.isArray(raw) || !raw.every((m) => m !== null && typeof m === "object" && !Array.isArray(m)))
      invalid("config 'models' must be a list of identities");
    const models = raw.map((m) => {
      const p = fromIdentity(m as unknown as Identity);
      return p instanceof Forecaster ? p : invalid("models must be ExponentialSmoothing or AutoRegressive forecasters");
    });
    return new ModelAveraging(models, c.weights === null ? null : nums(c, "weights"), num(c, "forgetting"), num(c, "coverage"));
  }

  config(): Config {
    return {
      models: this.models.map((m) => identity(m) as unknown as Json), weights: [...this.prior], forgetting: this.forgetting,
      coverage: this.coverage,
    };
  }

  initialState(): ModelAveragingState {
    return { n: 0, states: this.models.map((m) => m.initialState()), log_weights: [...this.logPrior], log_likelihood: 0.0, comp: 0.0 };
  }

  private flattened(logWeights: number[]): number[] {
    const logs = logWeights.map((v) => this.forgetting * v);
    const top = logSumExp(logs);
    return logs.map((v) => v - top);
  }

  update(state: State, y: number): Updated {
    const s = state as ModelAveragingState;
    const states: State[] = [], scores: (number | null)[] = [];
    this.models.forEach((model, i) => {
      const [next, , score] = model.update(s.states[i], y);
      states.push(next);
      scores.push(score);
    });
    let logWeights = s.log_weights, logLik = s.log_likelihood, comp = s.comp;
    let score: number | null = null;
    const known = scores.filter((v): v is number => v !== null);
    if (known.length === scores.length) {
      const prior = this.flattened(s.log_weights);
      const joint = prior.map((w, i) => w + known[i]);
      const total = requireFinite(logSumExp(joint), "log score");
      score = total;
      logWeights = joint.map((v) => v - total);
      [logLik, comp] = neumaierAdd(logLik, comp, total);
    }
    const next: ModelAveragingState = { n: s.n + 1, states, log_weights: logWeights, log_likelihood: logLik, comp };
    return [next, null, score];
  }

  private components(s: ModelAveragingState): Component[] | null {
    const weights = this.flattened(s.log_weights);
    const out: Component[] = [];
    for (let i = 0; i < this.models.length; i++) {
      const pred = this.models[i].predictive(s.states[i]);
      if (pred === null) return null;
      out.push([Math.exp(weights[i]), pred]);
    }
    return out;
  }

  private mixture(s: ModelAveragingState): [number[], number[], number[]] | null {
    const comps = this.components(s);
    if (comps === null) return null;
    // the components that carry all but 1e-10 of the weight; the rest cannot move a quantile visibly
    // (Python's sorted is stable: ties keep their order)
    const ranked = comps.map((_, i) => i).sort((i, j) => comps[j][0] - comps[i][0] || i - j);
    const live: Component[] = [];
    let mass = 0.0;
    for (const i of ranked) {
      if (mass >= 1.0 - 1e-10) break;
      live.push(comps[i]);
      mass += comps[i][0];
    }
    const tail = 0.5 * (1.0 - this.coverage);
    const quantiles = new Map<number | null, number>();
    for (const [, p] of live) if (!quantiles.has(p.df)) quantiles.set(p.df, this.quantile(p.df));
    const mean: number[] = [], lower: number[] = [], upper: number[] = [];
    for (let h = 0; h < this.horizon; h++) {
      let total = 0.0;
      for (const [w, p] of comps) total += w * p.mean[h];
      mean.push(total);
      let lo = 0.0, hi = 0.0;
      live.forEach(([, p], i) => {
        const q = quantiles.get(p.df) as number;
        const a = p.mean[h] - q * p.scale[h], b = p.mean[h] + q * p.scale[h];
        lo = i === 0 ? a : pyMin(lo, a);
        hi = i === 0 ? b : pyMax(hi, b);
      });
      lower.push(ModelAveraging.solve(live, h, tail, lo, hi));
      upper.push(ModelAveraging.solve(live, h, 1.0 - tail, lo, hi));
    }
    return [mean, lower, upper];
  }

  /** The point where the mixture CDF reaches `target` in [lo, hi]: Newton steps, bisecting when they leave. */
  static solve(live: Component[], h: number, target: number, lo: number, hi: number): number {
    let x = 0.5 * (lo + hi);
    for (let it = 0; it < 100; it++) {
      let cdf = 0.0, pdf = 0.0;
      for (const [w, p] of live) {
        const z = (x - p.mean[h]) / p.scale[h];
        cdf += w * standardCdf(z, p.df);
        pdf += w * Math.exp(logDensity(z, 1.0, p.df)) / p.scale[h];
      }
      if (cdf === target) return x;
      if (cdf < target) lo = x;
      else hi = x;
      const step = pdf > 0.0 ? (cdf - target) / pdf : Infinity;
      let nxt = x - step;
      if (!(lo < nxt && nxt < hi)) nxt = 0.5 * (lo + hi);
      if (Math.abs(nxt - x) <= 1e-13 * (Math.abs(x) + hi - lo) || nxt === lo || nxt === hi) return nxt;
      x = nxt;
    }
    return x;
  }

  predictive(state: State): Predictive | null {
    const mix = this.mixture(state as ModelAveragingState);
    if (mix === null) return null;
    const [mean, lower, upper] = mix;
    const z = this.quantile(null);
    return { mean, scale: lower.map((lo, i) => (upper[i] - lo) / (2.0 * z)), df: null };
  }

  override step(state: State, x: StatInput, _rng: SplitMix64 | null = null): [State, Output] {
    const [next, , score] = this.update(state, x[this.input] as number);
    const s = next as ModelAveragingState;
    const mix = this.mixture(s);
    const logLik = requireFinite(s.log_likelihood + s.comp, "log likelihood");
    const params = this.parameters(s);
    if (mix === null) {
      return [s, {
        n: s.n, mean: [], scale: [], df: null, lower: [], upper: [], error: null, log_score: score, log_likelihood: logLik, parameters: params,
      }];
    }
    const [mean, lower, upper] = mix;
    const z = this.quantile(null);
    const scale = lower.map((lo, i) => (upper[i] - lo) / (2.0 * z));
    return [s, { n: s.n, mean, scale, df: null, lower, upper, error: null, log_score: score, log_likelihood: logLik, parameters: params }];
  }

  override parameters(state: State): { [key: string]: Json } {
    return { weights: (state as ModelAveragingState).log_weights.map((v) => Math.exp(v)) };
  }

  decodeState(data: Json): ModelAveragingState {
    requireState(data !== null && typeof data === "object" && !Array.isArray(data)
      && Object.keys(data).sort().join(",") === "comp,log_likelihood,log_weights,n,states", "fields");
    const d = data as { [key: string]: Json };
    const { n, states, log_weights: logs } = d;
    requireState(isCount(n), "n");
    requireState(Array.isArray(states) && states.length === this.models.length, "one state per model");
    requireState(Array.isArray(logs) && logs.length === this.models.length && logs.every((v) => typeof v === "number" && v <= 1e-9),
      "log weights");
    requireState(Math.abs(logSumExp(logs as number[])) < 1e-9, "weights must sum to 1");
    for (const key of ["log_likelihood", "comp"]) requireState(isNumber(d[key]), key);
    const decoded = this.models.map((m, i) => m.decodeState((states as Json[])[i]));
    requireState(decoded.every((s) => (s as { n: number }).n === n), "every model has seen every observation");
    return {
      n: n as number, states: decoded, log_weights: logs as number[],
      log_likelihood: d.log_likelihood as number, comp: d.comp as number,
    };
  }
}

register("model_averaging", "1", (c) => ModelAveraging.fromConfig(c));

export interface AveragedSmoothingOptions {
  trend?: string | null; period?: number | null; alphas?: number[]; betas?: number[]; gammas?: number[]; phis?: number[];
  forgetting?: number; horizon?: number; coverage?: number; input?: string;
}

/** Exponential smoothing that learns its smoothing parameters: model averaging over a grid (Python's averaged_smoothing). */
export function averagedSmoothing(options: AveragedSmoothingOptions = {}): ModelAveraging {
  const {
    trend = null, period = null, alphas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9], betas = [0.01, 0.1, 0.3], gammas = [0.01, 0.1, 0.3],
    phis = [0.9, 0.98], forgetting = 1.0, horizon = 1, coverage = 0.9, input = "y",
  } = options;
  const models: Forecaster[] = [];
  for (const a of alphas)
    for (const b of trend !== null ? betas : [0.0])
      for (const g of period !== null ? gammas : [0.0])
        for (const f of trend === "damped" ? phis : [1.0])
          models.push(new ExponentialSmoothing(a, a * b, (1.0 - a) * g, f, trend, period, null, 0.0, null, null, horizon, coverage, input));
  return new ModelAveraging(models, null, forgetting, coverage);
}

// ----------------------------------------------------------------------------------- conformal

type Interval = [number[], number[]];
type ConformalState = {
  n: number; inner: State; pending: (Interval | null)[]; quantiles: number[]; counts: number[]; misses: number[];
};

/** Distribution-free forecast intervals around any forecaster: online conformal quantile tracking; see forecast.py. */
export class ConformalForecast extends BaseProcedure {
  readonly name = "conformal_forecast";
  readonly version = "1";
  readonly initial: number;
  readonly horizon: number;
  readonly input: string;

  constructor(readonly forecaster: Forecaster, readonly alpha = 0.1, readonly stepSize = 0.05, readonly normalize = true,
    initial: number | null = null) {
    super();
    if (!(forecaster instanceof Forecaster)) invalid("forecaster must be a forecaster (ExponentialSmoothing, AutoRegressive, ...)");
    if (!(0.0 < alpha && alpha < 1.0)) invalid("alpha must lie in (0, 1)");
    if (!(0.0 < stepSize && stepSize < Infinity)) invalid("step_size must be positive");
    if (initial === null) {
      if (!normalize) invalid("without normalize, give the initial half width in the data's units");
      initial = -normalInvCdf(0.5 * alpha);
    }
    if (!Number.isFinite(initial)) invalid("initial must be finite");
    this.initial = initial;
    this.horizon = forecaster.horizon;
    this.input = forecaster.input;
  }

  static fromConfig(c: Config): ConformalForecast {
    const inner = fromIdentity(obj(c, "forecaster") as unknown as Identity);
    if (!(inner instanceof Forecaster)) invalid("forecaster must be a forecaster (ExponentialSmoothing, AutoRegressive, ...)");
    return new ConformalForecast(inner, num(c, "alpha"), num(c, "step_size"), bool(c, "normalize"), optNum(c, "initial"));
  }

  config(): Config {
    return {
      forecaster: identity(this.forecaster) as unknown as Json, alpha: this.alpha, step_size: this.stepSize, normalize: this.normalize,
      initial: this.initial,
    };
  }

  inputs(): Record<string, InputContract> {
    return this.forecaster.inputs();
  }

  initialState(): ConformalState {
    const h = this.horizon;
    return {
      n: 0, inner: this.forecaster.initialState(), pending: Array.from({ length: h }, () => null),
      quantiles: Array.from({ length: h }, () => this.initial), counts: Array.from({ length: h }, () => 0),
      misses: Array.from({ length: h }, () => 0),
    };
  }

  step(state: State, x: StatInput, rng: SplitMix64 | null = null): [ConformalState, Output] {
    const s = state as ConformalState;
    const y = x[this.input] as number;
    const [inner, out] = this.forecaster.step(s.inner, x, rng);
    const quantiles = [...s.quantiles], counts = [...s.counts], misses = [...s.misses];
    for (let h = 0; h < this.horizon; h++) {
      const issued = s.pending[h];
      if (issued === null) continue;
      const miss = !(issued[0][h] <= y && y <= issued[1][h]);
      quantiles[h] = requireFinite(quantiles[h] + this.stepSize * ((miss ? 1.0 : 0.0) - this.alpha), "conformal quantile");
      counts[h] += 1;
      misses[h] += miss ? 1 : 0;
    }
    const mean = out.mean as number[];
    let lower: number[], upper: number[], issuedNow: Interval | null;
    if (mean.length) {
      const unit = this.normalize ? out.scale as number[] : Array.from({ length: this.horizon }, () => 1.0);
      lower = mean.map((m, i) => m - quantiles[i] * unit[i]);
      upper = mean.map((m, i) => m + quantiles[i] * unit[i]);
      issuedNow = [lower, upper];
    } else {
      lower = upper = [];
      issuedNow = null;
    }
    const pending = [issuedNow, ...s.pending.slice(0, -1)];
    const next: ConformalState = { n: s.n + 1, inner, pending, quantiles, counts, misses };
    const rates = misses.map((m, i) => (counts[i] ? m / counts[i] : null));
    return [next, { n: next.n, mean, lower, upper, quantiles: [...quantiles], miscoverage: rates, counts: [...counts], forecaster: out }];
  }

  decodeState(data: Json): ConformalState {
    requireState(data !== null && typeof data === "object" && !Array.isArray(data)
      && Object.keys(data).sort().join(",") === "counts,inner,misses,n,pending,quantiles", "fields");
    const d = data as { [key: string]: Json };
    const h = this.horizon;
    const n = d.n;
    requireState(isCount(n), "n");
    const lists = ["pending", "quantiles", "counts", "misses"].map((k) => d[k]);
    requireState(lists.every((v) => Array.isArray(v) && v.length === h), "one entry per horizon");
    const quantiles = d.quantiles as Json[], counts = d.counts as Json[], misses = d.misses as Json[];
    requireState(quantiles.every(isNumber), "quantiles");
    requireState(counts.every((c, i) => {
      const m = misses[i];
      return typeof c === "number" && Number.isInteger(c) && typeof m === "number" && Number.isInteger(m) && 0 <= m && m <= c
        && c <= (n as number);
    }), "counts");
    const pending: (Interval | null)[] = [];
    for (const p of d.pending as Json[]) {
      if (p === null) {
        pending.push(null);
        continue;
      }
      requireState(Array.isArray(p) && p.length === 2 && p.every((side) => Array.isArray(side) && side.length === h && side.every(isNumber)),
        "pending interval");
      const [lo, hi] = p as number[][];
      pending.push([lo, hi]);
    }
    const inner = this.forecaster.decodeState(d.inner) as { n?: number };
    requireState(inner.n === undefined || inner.n === n, "the forecaster has seen every observation");
    return {
      n: n as number, inner, pending, quantiles: quantiles as number[], counts: counts as number[],
      misses: misses as number[],
    };
  }
}

register("conformal_forecast", "1", (c) => ConformalForecast.fromConfig(c));

// ---------------------------------------------------------------------------------- comparison

const ZETA_1_4 = 3.105547277977581; // Riemann zeta(1.4)
const STITCH_S = 1.4, STITCH_ETA = 2.0;
// CPython's 2.0 ** 0.25 and 2.0 ** -0.25 (glibc's pow); V8's Math.pow rounds the second one up in the last bit
const ETA_QUARTER = 1.189207115002721, ETA_MINUS_QUARTER = 0.8408964152537145;

/** Polynomial stitched boundary of Howard, Ramdas, McAuliffe and Sekhon (2021), with s = 1.4 and eta = 2. */
export function stitchedBoundary(v: number, c: number, alpha: number, m: number): number {
  v = pyMax(v, m);
  const eta = STITCH_ETA, s = STITCH_S;
  const ell = s * Math.log(Math.log(eta * v / m)) + Math.log(ZETA_1_4 / (alpha * Math.log(eta) ** s));
  const k1 = (ETA_QUARTER + ETA_MINUS_QUARTER) / SQRT2;
  const k2 = (Math.sqrt(eta) + 1.0) / 2.0;
  return Math.sqrt(k1 * k1 * v * ell + k2 * k2 * c * c * ell * ell) + c * k2 * ell;
}

type ComparisonState = { n: number; total: number; total_comp: number; variance: number; variance_comp: number };

/** Anytime-valid comparison of two forecasters by their losses (Choe and Ramdas 2024); see forecast.py. */
export class ForecastComparison extends BaseProcedure {
  readonly name = "forecast_comparison";
  readonly version = "1";
  readonly bounds: [number, number];
  readonly minVariance: number;
  private readonly width: number;

  constructor(bounds: number[], readonly alpha = 0.05, minVariance: number | null = null, readonly a = "a", readonly b = "b") {
    super();
    if (bounds.length !== 2) invalid("bounds must be a (lower, upper) pair");
    const [lo, hi] = bounds;
    if (!(Number.isFinite(lo) && Number.isFinite(hi) && lo < hi)) invalid("bounds must be finite, lower < upper");
    if (!(0.0 < alpha && alpha < 1.0)) invalid("alpha must lie in (0, 1)");
    const width = hi - lo;
    const m = minVariance === null ? width * width : minVariance;
    if (!(0.0 < m && m < Infinity)) invalid("min_variance must be positive");
    if (a === b) invalid("the two losses need different names");
    this.bounds = [lo, hi];
    this.minVariance = m;
    this.width = width;
  }

  static fromConfig(c: Config): ForecastComparison {
    return new ForecastComparison(nums(c, "bounds"), num(c, "alpha"), optNum(c, "min_variance"), str(c, "a"), str(c, "b"));
  }

  config(): Config {
    return { bounds: [...this.bounds], alpha: this.alpha, min_variance: this.minVariance, a: this.a, b: this.b };
  }

  inputs(): Record<string, InputContract> {
    const [lo, hi] = this.bounds;
    return { [this.a]: new InputContract(this.a, "real", [], lo, hi), [this.b]: new InputContract(this.b, "real", [], lo, hi) };
  }

  initialState(): ComparisonState {
    return { n: 0, total: 0.0, total_comp: 0.0, variance: 0.0, variance_comp: 0.0 };
  }

  step(state: State, x: StatInput): [ComparisonState, Output] {
    const s = state as ComparisonState;
    const d = (x[this.a] as number) - (x[this.b] as number);
    const guess = s.n ? (s.total + s.total_comp) / s.n : 0.0;
    const [total, totalComp] = neumaierAdd(s.total, s.total_comp, d);
    const dev = d - guess;
    const [variance, varianceComp] = neumaierAdd(s.variance, s.variance_comp, dev * dev);
    const n = s.n + 1;
    const next: ComparisonState = { n, total, total_comp: totalComp, variance, variance_comp: varianceComp };
    const mean = (total + totalComp) / n;
    const radius = stitchedBoundary(variance + varianceComp, 2.0 * this.width, 0.5 * this.alpha, this.minVariance) / n;
    const lower = mean - radius, upper = mean + radius;
    const better = upper < 0.0 ? "a" : lower > 0.0 ? "b" : null;
    return [next, { n, mean, lower, upper, better }];
  }

  decodeState(data: Json): ComparisonState {
    const s = decodeFields({ n: "count", total: "number", total_comp: "number", variance: "number", variance_comp: "number" },
      data) as ComparisonState;
    requireState(s.variance >= 0, "negative variance");
    requireState(s.n > 0 || (s.total === 0 && s.variance === 0), "statistics before data");
    return s;
  }
}

register("forecast_comparison", "1", (c) => ForecastComparison.fromConfig(c));
