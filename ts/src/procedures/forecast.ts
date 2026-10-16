// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Forecasting: smoothing, autoregression, counts, model averaging, conformal intervals and comparison; mirrors procedures/forecast.py.
import { type Json, sameJson } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { ContractViolation } from "../errors.js";
import { logSumExp, neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, fromIdentity, identity, type Identity, type Output, register, requireState, type State } from "../procedure.js";
import type { SplitMix64 } from "../rng.js";
import { erfc, lgamma, regularizedBeta, tCdf, tIsfLog } from "./distributions.js";
import { normalInvCdf } from "./group_sequential.js";
import { BaseProcedure, bool, type Config, int, invalid, num, nums, obj, optNum, pyMax, pyMin, str } from "./util.js";

const LOG_2PI = Math.log(2.0 * Math.PI);
const SQRT2 = Math.SQRT2;
const TRANSFORMS: (string | null)[] = [null, "log"];

function checkCoverage(coverage: number): number {
  if (!(0.0 < coverage && coverage < 1.0)) invalid("coverage must lie in (0, 1)");
  return coverage;
}

function checkHorizon(horizon: number): number {
  if (!(Number.isInteger(horizon) && horizon >= 1)) invalid("horizon must be a positive int");
  return horizon;
}

function checkTransform(transform: string | null): void {
  if (!TRANSFORMS.includes(transform)) invalid("transform must be one of (None, 'log')");
}

function checkNames(input: string, regressors: string[]): string[] {
  const names = [...regressors];
  if (!names.every((r) => typeof r === "string") || new Set(names).size !== names.length || names.includes(input))
    invalid("regressors need distinct names, different from the input");
  return names;
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

/** log P(Y = k) for Poisson(exposure mu) counts with mu ~ Gamma(size, rate). */
export function negativeBinomialLogPmf(k: number, size: number, rate: number, exposure: number): number {
  const logTotal = Math.log(rate + exposure);
  return (lgamma(size + k) - lgamma(size) - lgamma(k + 1.0)
    + size * (Math.log(rate) - logTotal) + k * (Math.log(exposure) - logTotal));
}

/** P(Y <= k) for the same law: I_q(size, k + 1) with q = rate / (rate + exposure). */
export function negativeBinomialCdf(k: number, size: number, rate: number, exposure: number): number {
  if (k < 0) return 0.0;
  const total = rate + exposure;
  const logTotal = Math.log(total);
  return regularizedBeta(rate / total, size, k + 1.0, exposure / total, [Math.log(rate) - logTotal, Math.log(exposure) - logTotal]);
}

/**
 * The forecast of the next 1..horizon values.  Continuous (`size` empty): `mean` + `scale` * (standard normal, or t
 * with `df`), on the log scale with `transform` "log".  Counts: negative binomial with `size`, `rate` and `exposure`
 * per horizon; `mean` and `scale` are its mean and standard deviation.
 */
export class Predictive {
  constructor(
    readonly mean: number[], readonly scale: number[], readonly df: number | null, readonly transform: string | null = null,
    readonly size: number[] = [], readonly rate: number[] = [], readonly exposure: number[] = [],
  ) {}

  get counts(): boolean {
    return this.size.length > 0;
  }

  /** P(Y <= x) for the forecast h + 1 steps ahead, in the data's units. */
  cdf(h: number, x: number): number {
    if (this.counts) return negativeBinomialCdf(Math.floor(x), this.size[h], this.rate[h], this.exposure[h]);
    if (this.transform === "log") {
      if (x <= 0.0) return 0.0;
      x = Math.log(x);
    }
    return standardCdf((x - this.mean[h]) / this.scale[h], this.df);
  }

  /** The density at x in the data's units (continuous forecasts only). */
  density(h: number, x: number): number {
    if (this.transform === "log") {
      if (x <= 0.0) return 0.0;
      return Math.exp(logDensity(Math.log(x) - this.mean[h], this.scale[h], this.df)) / x;
    }
    return Math.exp(logDensity(x - this.mean[h], this.scale[h], this.df));
  }
}

/** (new state, error, log score) after one record. */
export type Updated = [State, number | null, number | null];

function optStr(c: Config, key: string): string | null {
  return c[key] === null ? null : str(c, key);
}
function optInt(c: Config, key: string): number | null {
  return c[key] === null ? null : int(c, key);
}
function strs(c: Config, key: string): string[] {
  if (!(key in c)) invalid(`config is missing '${key}'`);
  const v = c[key];
  if (!Array.isArray(v) || !v.every((e) => typeof e === "string")) invalid(`config '${key}' must be a list of strings`);
  return v as string[];
}
const isNumber = (v: Json): v is number => typeof v === "number" && Number.isFinite(v);
const isCount = (v: Json): v is number => isNumber(v) && Number.isInteger(v) && v >= 0;

/** A state field holding a list of lists of numbers (Python's tuple[tuple[float, ...], ...]). */
function checkMatrix(name: string, value: Json): void {
  if (!Array.isArray(value) || !value.every((row) => Array.isArray(row) && row.every(isNumber)))
    throw new Error(`state field '${name}' must be a list of lists of numbers, got ${JSON.stringify(value)}`);
}

/** Python's == between two dicts of input contracts. */
function sameContracts(a: Record<string, InputContract>, b: Record<string, InputContract>): boolean {
  const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
  return ka.length === kb.length && ka.every((k, i) => k === kb[i] && sameJson(a[k].spec(), b[k].spec()));
}

/** Common protocol: `update` scores and absorbs one record, `predictive` forecasts from a state. */
export abstract class Forecaster extends BaseProcedure {
  abstract readonly horizon: number;
  abstract readonly coverage: number;
  abstract readonly input: string;
  abstract readonly regressors: string[];
  abstract readonly transform: string | null;

  inputs(): Record<string, InputContract> {
    const contracts: Record<string, InputContract> = { [this.input]: new InputContract(this.input) };
    for (const r of this.regressors) contracts[r] = new InputContract(r, "real", [this.horizon + 1]);
    return contracts;
  }

  /** (new state, error, log score) after observing the record x. */
  abstract update(state: State, x: StatInput): Updated;
  abstract predictive(state: State): Predictive | null;

  parameters(_state: State): { [key: string]: Json } {
    return {};
  }

  logLikelihood(state: State): number {
    return (state.log_likelihood as number) + (state.comp as number);
  }

  /** (the observation on the model's scale, the log Jacobian to subtract from its log density). */
  protected value(x: StatInput): [number, number] {
    const y = x[this.input] as number;
    if (this.transform === "log") {
      if (!(y > 0.0)) throw new ContractViolation(`${this.name}: the log transform needs positive values, got ${y}`);
      const logY = Math.log(y);
      return [logY, logY];
    }
    return [y, 0.0];
  }

  /** (the regressors' current values, their planned values for 1..horizon steps ahead). */
  protected windows(x: StatInput): [number[], number[][]] {
    const now = this.regressors.map((r) => (x[r] as number[])[0]);
    const plan = this.regressors.map((r) => (x[r] as number[]).slice(1));
    return [now, plan];
  }

  protected point(location: number): number {
    return this.transform === "log" ? Math.exp(location) : location;
  }

  interval(pred: Predictive): [number[], number[]] {
    if (pred.counts) {
      const tail = 0.5 * (1.0 - this.coverage);
      const lower = pred.mean.map((_, h) => countQuantile(pred, h, tail));
      const upper = pred.mean.map((_, h) => countQuantile(pred, h, 1.0 - tail));
      return [lower, upper];
    }
    const q = this.quantile(pred.df);
    const lower = pred.mean.map((m, i) => this.point(m - q * pred.scale[i]));
    const upper = pred.mean.map((m, i) => this.point(m + q * pred.scale[i]));
    return [lower, upper];
  }

  quantile(df: number | null): number {
    const tail = 0.5 * (1.0 - this.coverage);
    if (df === null) return -normalInvCdf(tail);
    return tIsfLog(Math.log(tail), df);
  }

  /** (point forecasts, scales) in the data's units: on the log scale, the median and m * s. */
  present(pred: Predictive): [number[], number[]] {
    if (this.transform === "log" && !pred.counts) {
      const mean = pred.mean.map((m) => Math.exp(m));
      return [mean, mean.map((m, i) => m * pred.scale[i])];
    }
    return [pred.mean, pred.scale];
  }

  step(state: State, x: StatInput, _rng: SplitMix64 | null = null): [State, Output] {
    const [next, error, score] = this.update(state, x);
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
    const [mean, scale] = this.present(pred);
    return [next, {
      n, mean, scale, df: pred.df, lower, upper, error, log_score: score, log_likelihood: logLik, parameters: this.parameters(next),
    }];
  }
}

/** The smallest count k with P(Y <= k) >= p. */
export function countQuantile(pred: Predictive, h: number, p: number): number {
  if (pred.cdf(h, 0.0) >= p) return 0;
  let lo = 0, hi = 1;
  while (pred.cdf(h, hi) < p) [lo, hi] = [hi, 2 * hi];
  while (hi - lo > 1) { // P(Y <= lo) < p <= P(Y <= hi)
    const mid = Math.floor((lo + hi) / 2);
    if (pred.cdf(h, mid) < p) lo = mid;
    else hi = mid;
  }
  return hi;
}

// ------------------------------------------------------------------ Bayesian regression helpers

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

/** (coefficients for the next value, Cholesky factor of M) with M = lambda information + ridge I. */
function posterior(information: number[][], moment: number[], forgetting: number, ridge: number): [number[], number[][]] {
  const k = moment.length;
  const precision = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => forgetting * information[i][j] + (i === j ? ridge : 0.0)));
  const low = choleskyFloor(precision);
  return [backward(low, forward(low, moment.map((v) => forgetting * v))), low];
}

/** u' M^-1 u from the Cholesky factor of M. */
function quad(low: number[][], u: number[]): number {
  let total = 0.0;
  for (const v of forward(low, u)) total += v * v;
  return total;
}

function accumulate(information: number[][], moment: number[], u: number[], target: number, forgetting: number): [number[][], number[]] {
  const k = u.length;
  const info = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => requireFinite(forgetting * information[i][j] + u[i] * u[j], "information")));
  const mom = Array.from({ length: k }, (_, i) => requireFinite(forgetting * moment[i] + u[i] * target, "moment"));
  return [info, mom];
}

function checkInformation(information: number[][], moment: number[], k: number): void {
  requireState(information.length === k && moment.length === k && information.every((row) => row.length === k), "statistics' sizes");
  let ok = true;
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) ok = ok && information[i][i] >= 0 && information[i][j] === information[j][i];
  requireState(ok, "information must be symmetric with a non-negative diagonal");
}

function checkLearning(forgetting: number, ridge: number): void {
  if (!(0.0 < forgetting && forgetting <= 1.0)) invalid("forgetting must lie in (0, 1]");
  if (!(0.0 < ridge && ridge < Infinity)) invalid("ridge must be positive");
}

// ---------------------------------------------------------------------------------------- ETS

type ETSState = {
  n: number; level: number; trend: number; seasonal: number[]; errors: number; sse: number; sse_comp: number;
  log_likelihood: number; comp: number; plan: number[][]; effects: number[][]; information: number[][]; moment: number[];
  shape: number; rate: number;
};

const TRENDS: (string | null)[] = [null, "additive", "damped"];

/** Additive-error exponential smoothing (ETS) with optional trend, season and regressors; see forecast.py. */
export class ExponentialSmoothing extends Forecaster {
  readonly name = "exponential_smoothing";
  readonly version = "1";
  readonly initialSeasonal: number[];
  readonly regressors: string[];
  readonly effects: number[] | null;
  readonly horizon: number;
  readonly coverage: number;
  private readonly damping: number[];
  private readonly factors: number[];

  constructor(
    readonly alpha: number, readonly beta = 0.0, readonly gamma = 0.0, readonly phi = 1.0, readonly trend: string | null = null,
    readonly period: number | null = null, readonly initialLevel: number | null = null, readonly initialTrend = 0.0,
    initialSeasonal: number[] | null = null, readonly sigma: number | null = null, regressors: string[] = [],
    coefficients: number[] | null = null, readonly forgetting = 1.0, readonly ridge = 1e-6, readonly transform: string | null = null,
    horizon = 1, coverage = 0.9, readonly input = "y",
  ) {
    super();
    if (!TRENDS.includes(trend)) invalid(`trend must be one of (None, 'additive', 'damped')`);
    checkTransform(transform);
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
    this.regressors = checkNames(input, regressors);
    this.effects = null;
    if (coefficients !== null) {
      this.effects = [...coefficients];
      if (this.effects.length !== this.regressors.length || !this.effects.every((c) => Number.isFinite(c)))
        invalid("need one finite coefficient per regressor");
      if (forgetting !== 1.0 || ridge !== 1e-6) invalid("forgetting and ridge are for learnt effects");
    } else if (this.regressors.length) {
      if (sigma !== null) invalid("learnt effects come with a learnt sigma (sigma=None)");
      checkLearning(forgetting, ridge);
    } else if (forgetting !== 1.0 || ridge !== 1e-6) invalid("forgetting and ridge are for learnt effects");
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

  private get learning(): boolean {
    return this.regressors.length > 0 && this.effects === null;
  }

  static fromConfig(c: Config): ExponentialSmoothing {
    const period = optInt(c, "period");
    const seasonal = c.initial_seasonal === null ? null : nums(c, "initial_seasonal");
    return new ExponentialSmoothing(num(c, "alpha"), num(c, "beta"), num(c, "gamma"), num(c, "phi"), optStr(c, "trend"), period,
      optNum(c, "initial_level"), num(c, "initial_trend"), seasonal, optNum(c, "sigma"), strs(c, "regressors"),
      c.coefficients === null ? null : nums(c, "coefficients"), num(c, "forgetting"), num(c, "ridge"), optStr(c, "transform"),
      int(c, "horizon"), num(c, "coverage"), str(c, "input"));
  }

  config(): Config {
    return {
      alpha: this.alpha, beta: this.beta, gamma: this.gamma, phi: this.phi, trend: this.trend, period: this.period,
      initial_level: this.initialLevel, initial_trend: this.initialTrend,
      initial_seasonal: this.period !== null ? [...this.initialSeasonal] : null, sigma: this.sigma, regressors: [...this.regressors],
      coefficients: this.effects === null ? null : [...this.effects], forgetting: this.forgetting, ridge: this.ridge,
      transform: this.transform, horizon: this.horizon, coverage: this.coverage, input: this.input,
    };
  }

  initialState(): ETSState {
    const level = this.initialLevel === null ? 0.0 : this.initialLevel;
    const state: ETSState = {
      n: 0, level, trend: this.initialTrend, seasonal: [...this.initialSeasonal], errors: 0, sse: 0.0, sse_comp: 0.0,
      log_likelihood: 0.0, comp: 0.0, plan: [], effects: [], information: [], moment: [], shape: 0.0, rate: 0.0,
    };
    if (!this.learning) return state;
    const r = this.regressors.length, width = 2 + this.initialSeasonal.length;
    return {
      ...state, effects: Array.from({ length: r }, () => Array.from({ length: width }, () => 0.0)),
      information: Array.from({ length: r }, () => Array.from({ length: r }, () => 0.0)), moment: Array.from({ length: r }, () => 0.0),
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
    if (this.learning) {
      if (!(s.shape > 0.0 && s.rate > 0.0)) return null;
      return [Math.sqrt(s.rate / s.shape), 2.0 * this.forgetting * s.shape];
    }
    const sse = s.sse + s.sse_comp;
    if (s.errors < 1 || !(sse > 0.0)) return null;
    return [Math.sqrt(sse / s.errors), s.errors];
  }

  private effect(values: number[]): number {
    let total = 0.0;
    if (this.effects !== null) for (let j = 0; j < this.effects.length; j++) total += this.effects[j] * values[j];
    return total;
  }

  /** One step of a state vector (level, trend, season...) given its innovation. */
  private advance(a: number[], innovation: number): number[] {
    const level = a[0], trend = a[1], season = a.slice(2);
    const first = season.length ? season[0] : 0.0;
    return [level + this.phi * trend + this.alpha * innovation, this.phi * trend + this.beta * innovation,
      ...this.rotate(season, first + this.gamma * innovation)];
  }

  /** What a state vector (level, trend, season...) predicts h steps ahead. */
  private predicted(a: number[], h: number): number {
    const season = a.length > 2 ? a[2 + (h - 1) % (a.length - 2)] : 0.0;
    return a[0] + this.damping[h - 1] * a[1] + season;
  }

  update(state: State, x: StatInput): Updated {
    const s = state as ETSState;
    const [y, jacobian] = this.value(x);
    const [now, plan] = this.windows(x);
    const season = s.seasonal.length ? s.seasonal[0] : 0.0;
    if (!this.started(s)) {
      // the first value sets the level: l = y - s - b'z, so A_j starts with level z_j
      const effects = this.learning ? now.map((z) => [z, 0.0, ...s.seasonal.map(() => 0.0)]) : [];
      const fresh: ETSState = {
        n: 1, level: y - season - this.effect(now), trend: s.trend, seasonal: this.rotate(s.seasonal, season), errors: 0, sse: 0.0,
        sse_comp: 0.0, log_likelihood: s.log_likelihood, comp: s.comp, plan, effects, information: s.information, moment: s.moment,
        shape: s.shape, rate: s.rate,
      };
      return [fresh, null, null];
    }
    const base = s.level + this.phi * s.trend + season;
    let logLik = s.log_likelihood, comp = s.comp;
    if (!this.learning) {
      const error = y - (base + this.effect(now));
      const noise = this.noise(s);
      const score = noise === null ? null : logDensity(error, noise[0], noise[1]) - jacobian;
      const level = requireFinite(s.level + this.phi * s.trend + this.alpha * error, "level");
      const trend = requireFinite(this.phi * s.trend + this.beta * error, "trend");
      const seasonal = this.rotate(s.seasonal, season + this.gamma * error);
      const [sse, sseComp] = neumaierAdd(s.sse, s.sse_comp, error * error);
      if (score !== null) [logLik, comp] = neumaierAdd(logLik, comp, score);
      const next: ETSState = {
        n: s.n + 1, level, trend, seasonal, errors: s.errors + 1, sse, sse_comp: sseComp, log_likelihood: logLik, comp, plan,
        effects: [], information: [], moment: [], shape: 0.0, rate: 0.0,
      };
      return [next, this.point(y) - this.point(base + this.effect(now)), score];
    }
    // learnt effects: regress the error of the run without regressors on u
    const u = now.map((z, j) => z - this.predicted(s.effects[j], 1));
    const [theta, low] = posterior(s.information, s.moment, this.forgetting, this.ridge);
    const raw = y - base;
    let fitted = 0.0;
    for (let j = 0; j < theta.length; j++) fitted += theta[j] * u[j];
    const error = raw - fitted;
    const sv = 1.0 + quad(low, u);
    let score: number | null = null;
    if (s.shape > 0.0 && s.rate > 0.0) {
      score = logDensity(error, Math.sqrt(s.rate / s.shape * sv), 2.0 * this.forgetting * s.shape) - jacobian;
      [logLik, comp] = neumaierAdd(logLik, comp, score);
    }
    const lam = this.forgetting;
    const [information, moment] = accumulate(s.information, s.moment, u, raw, lam);
    const level = requireFinite(s.level + this.phi * s.trend + this.alpha * raw, "level");
    const trend = requireFinite(this.phi * s.trend + this.beta * raw, "trend");
    const seasonal = this.rotate(s.seasonal, season + this.gamma * raw);
    const effects = s.effects.map((a, j) => this.advance(a, u[j]));
    const next: ETSState = {
      n: s.n + 1, level, trend, seasonal, errors: s.errors + 1, sse: 0.0, sse_comp: 0.0, log_likelihood: logLik, comp, plan, effects,
      information, moment, shape: lam * s.shape + 0.5, rate: requireFinite(lam * s.rate + error * error / (2.0 * sv), "noise statistic"),
    };
    return [next, this.point(y) - this.point(base + fitted), score];
  }

  predictive(state: State): Predictive | null {
    const s = state as ETSState;
    const noise = this.started(s) ? this.noise(s) : null;
    if (noise === null || (this.regressors.length > 0 && !s.plan.length)) return null;
    const [sigma, df] = noise;
    const x0 = [s.level, s.trend, ...s.seasonal];
    const mean: number[] = [], scale: number[] = [];
    if (!this.learning) {
      for (let h = 1; h <= this.horizon; h++) mean.push(this.predicted(x0, h) + this.effect(s.plan.map((p) => p[h - 1])));
      return new Predictive(mean, this.factors.map((f) => sigma * f), df, this.transform);
    }
    const [theta, low] = posterior(s.information, s.moment, this.forgetting, this.ridge);
    for (let h = 1; h <= this.horizon; h++) {
      const u = s.plan.map((p, j) => p[h - 1] - this.predicted(s.effects[j], h));
      let value = this.predicted(x0, h);
      for (let j = 0; j < theta.length; j++) value += theta[j] * u[j];
      mean.push(value);
      const f = this.factors[h - 1];
      scale.push(sigma * Math.sqrt(f * f + quad(low, u)));
    }
    return new Predictive(mean, scale, df, this.transform);
  }

  override parameters(state: State): { [key: string]: Json } {
    const s = state as ETSState;
    const noise = this.started(s) ? this.noise(s) : null;
    const sigma = noise === null ? null : noise[0];
    if (this.learning) {
      const [theta] = posterior(s.information, s.moment, this.forgetting, this.ridge);
      // the states of the model itself are x~ - sum_j b_j A_j
      let x = [s.level, s.trend, ...s.seasonal];
      theta.forEach((b, j) => {
        const a = s.effects[j];
        x = x.map((v, i) => v - b * a[i]);
      });
      return { level: x[0], trend: x[1], seasonal: x.slice(2), sigma, coefficients: theta };
    }
    const out: { [key: string]: Json } = { level: s.level, trend: s.trend, seasonal: [...s.seasonal], sigma };
    if (this.effects !== null) out.coefficients = [...this.effects];
    return out;
  }

  decodeState(data: Json): ETSState {
    const s = decodeFields({
      n: "count", level: "number", trend: "number", seasonal: "numbers", errors: "count", sse: "number", sse_comp: "number",
      log_likelihood: "number", comp: "number", plan: "json", effects: "json", information: "json", moment: "numbers", shape: "number",
      rate: "number",
    }, data) as ETSState;
    for (const key of ["plan", "effects", "information"] as const) checkMatrix(key, s[key]);
    requireState(s.seasonal.length === this.initialSeasonal.length, "one seasonal effect per season");
    requireState(0 <= s.errors && s.errors <= s.n && s.sse >= 0, "error statistics");
    requireState(s.n > 0 || (s.errors === 0 && s.sse === 0 && s.log_likelihood === 0), "statistics before data");
    requireState(this.trend !== null || s.trend === 0, "a trend without a trend model");
    const r = this.regressors.length;
    requireState(s.plan.length === (s.n > 0 ? r : 0) && s.plan.every((p) => p.length === this.horizon), "one plan per regressor");
    if (this.learning) {
      const width = 2 + this.initialSeasonal.length;
      requireState(s.effects.length === r && s.effects.every((a) => a.length === width), "effects' sizes");
      checkInformation(s.information, s.moment, r);
      requireState(s.shape >= 0 && s.rate >= 0 && s.sse === 0, "noise statistics");
    } else {
      requireState(s.effects.length === 0 && s.information.length === 0 && s.moment.length === 0 && s.shape === 0 && s.rate === 0,
        "learnt statistics without learnt effects");
    }
    return s;
  }
}

register("exponential_smoothing", "1", (c) => ExponentialSmoothing.fromConfig(c));

// ------------------------------------------------------------------------------------------ AR

type ARState = {
  n: number; lags: number[]; information: number[][]; moment: number[]; shape: number; rate: number; log_likelihood: number;
  comp: number; plan: number[][];
};

/** AR(p) forecasts, with fixed coefficients or learnt by Bayesian regression on the lags and regressors; see forecast.py. */
export class AutoRegressive extends Forecaster {
  readonly name = "autoregressive";
  readonly version = "1";
  readonly phi: number[] | null;
  readonly beta: number[];
  readonly regressors: string[];
  readonly horizon: number;
  readonly coverage: number;

  constructor(
    readonly order: number, coefficients: number[] | null = null, readonly intercept = 0.0, readonly sigma: number | null = null,
    regressors: string[] = [], effects: number[] | null = null, readonly forgetting = 1.0, readonly ridge = 1e-6,
    readonly transform: string | null = null, horizon = 1, coverage = 0.9, readonly input = "y",
  ) {
    super();
    if (!(Number.isInteger(order) && order >= 1)) invalid("order must be a positive int");
    checkTransform(transform);
    this.regressors = checkNames(input, regressors);
    if (coefficients !== null) {
      const phi = [...coefficients];
      if (phi.length !== order || !phi.every((c) => Number.isFinite(c)) || !Number.isFinite(intercept))
        invalid("need `order` finite coefficients and a finite intercept");
      if (sigma === null || !(0.0 < sigma && sigma < Infinity)) invalid("fixed coefficients need a positive sigma");
      if (forgetting !== 1.0 || ridge !== 1e-6) invalid("forgetting and ridge are for learnt coefficients");
      const beta = effects !== null ? [...effects] : [];
      if (beta.length !== this.regressors.length || !beta.every((b) => Number.isFinite(b)))
        invalid("fixed coefficients need one finite effect per regressor");
      this.phi = phi;
      this.beta = beta;
    } else {
      if (sigma !== null || intercept !== 0.0 || effects !== null) invalid("sigma, intercept and effects come with fixed coefficients");
      checkLearning(forgetting, ridge);
      this.phi = null;
      this.beta = [];
    }
    this.horizon = checkHorizon(horizon);
    this.coverage = checkCoverage(coverage);
  }

  static fromConfig(c: Config): AutoRegressive {
    return new AutoRegressive(int(c, "order"), c.coefficients === null ? null : nums(c, "coefficients"), num(c, "intercept"),
      optNum(c, "sigma"), strs(c, "regressors"), c.effects === null ? null : nums(c, "effects"), num(c, "forgetting"), num(c, "ridge"),
      optStr(c, "transform"), int(c, "horizon"), num(c, "coverage"), str(c, "input"));
  }

  config(): Config {
    return {
      order: this.order, coefficients: this.phi === null ? null : [...this.phi], intercept: this.intercept, sigma: this.sigma,
      regressors: [...this.regressors], effects: this.phi === null ? null : [...this.beta], forgetting: this.forgetting,
      ridge: this.ridge, transform: this.transform, horizon: this.horizon, coverage: this.coverage, input: this.input,
    };
  }

  initialState(): ARState {
    if (this.phi !== null) {
      return { n: 0, lags: [], information: [], moment: [], shape: 0.0, rate: 0.0, log_likelihood: 0.0, comp: 0.0, plan: [] };
    }
    const k = this.order + 1 + this.regressors.length;
    return {
      n: 0, lags: [], information: Array.from({ length: k }, () => Array.from({ length: k }, () => 0.0)),
      moment: Array.from({ length: k }, () => 0.0), shape: 0.0, rate: 0.0, log_likelihood: 0.0, comp: 0.0, plan: [],
    };
  }

  private pointAr(theta: number[], lags: number[], z: number[]): number {
    let total = theta[0];
    for (let i = 0; i < this.order; i++) total += theta[i + 1] * lags[i];
    for (let j = 0; j < z.length; j++) total += theta[this.order + 1 + j] * z[j];
    return total;
  }

  private theta(): number[] {
    return [this.intercept, ...(this.phi as number[]), ...this.beta];
  }

  update(state: State, x: StatInput): Updated {
    const s = state as ARState;
    const [y, jacobian] = this.value(x);
    const [now, plan] = this.windows(x);
    const lags = [y, ...s.lags].slice(0, this.order);
    if (s.lags.length < this.order) return [{ ...s, n: s.n + 1, lags, plan }, null, null];
    let logLik = s.log_likelihood, comp = s.comp;
    if (this.phi !== null) {
      const point = this.pointAr(this.theta(), s.lags, now);
      const fixedScore = logDensity(y - point, this.sigma as number, null) - jacobian;
      [logLik, comp] = neumaierAdd(logLik, comp, fixedScore);
      const next: ARState = {
        n: s.n + 1, lags, information: [], moment: [], shape: 0.0, rate: 0.0, log_likelihood: logLik, comp, plan,
      };
      return [next, this.point(y) - this.point(point), fixedScore];
    }
    const design = [1.0, ...s.lags, ...now];
    const [theta, low] = posterior(s.information, s.moment, this.forgetting, this.ridge);
    const point = this.pointAr(theta, s.lags, now);
    const error = y - point;
    const sv = 1.0 + quad(low, design);
    let score: number | null = null;
    if (s.shape > 0.0 && s.rate > 0.0) {
      score = logDensity(error, Math.sqrt(s.rate / s.shape * sv), 2.0 * this.forgetting * s.shape) - jacobian;
      [logLik, comp] = neumaierAdd(logLik, comp, score);
    }
    const lam = this.forgetting;
    const [information, moment] = accumulate(s.information, s.moment, design, y, lam);
    const rate = requireFinite(lam * s.rate + error * error / (2.0 * sv), "noise statistic");
    const next: ARState = {
      n: s.n + 1, lags, information, moment, shape: lam * s.shape + 0.5, rate, log_likelihood: logLik, comp, plan,
    };
    return [next, this.point(y) - this.point(point), score];
  }

  predictive(state: State): Predictive | null {
    const s = state as ARState;
    if (s.lags.length < this.order || (this.regressors.length > 0 && !s.plan.length)) return null;
    let sigma2: number, df: number | null, theta: number[], extra: number;
    if (this.phi !== null) {
      const sigma = this.sigma as number;
      [sigma2, df, theta, extra] = [sigma * sigma, null, this.theta(), 0.0];
    } else if (s.shape > 0.0 && s.rate > 0.0) {
      [sigma2, df] = [s.rate / s.shape, 2.0 * this.forgetting * s.shape];
      const [t, low] = posterior(s.information, s.moment, this.forgetting, this.ridge);
      theta = t;
      extra = quad(low, [1.0, ...s.lags, ...s.plan.map((p) => p[0])]);
    } else return null;
    let history = [...s.lags];
    const psi = [1.0];
    const mean: number[] = [], scale: number[] = [];
    let total = 0.0;
    for (let h = 1; h <= this.horizon; h++) {
      const value = this.pointAr(theta, history, s.plan.map((p) => p[h - 1]));
      mean.push(value);
      history = [value, ...history.slice(0, -1)];
      total += psi[h - 1] * psi[h - 1];
      scale.push(Math.sqrt(sigma2 * (extra + total)));
      let nxt = 0.0;
      for (let i = 1; i <= Math.min(h, this.order); i++) nxt += theta[i] * psi[h - i];
      psi.push(nxt);
    }
    return new Predictive(mean, scale, df, this.transform);
  }

  /** The coefficients and noise level the next forecast uses. */
  override parameters(state: State): { [key: string]: Json } {
    const s = state as ARState;
    const p = this.order;
    let out: { [key: string]: Json };
    if (this.phi !== null) out = { intercept: this.intercept, coefficients: [...this.phi], sigma: this.sigma };
    else if (s.lags.length < this.order) out = { intercept: null, coefficients: null, sigma: null };
    else {
      const [theta] = posterior(s.information, s.moment, this.forgetting, this.ridge);
      const sigma = s.shape > 0.0 && s.rate > 0.0 ? Math.sqrt(s.rate / s.shape) : null;
      out = { intercept: theta[0], coefficients: theta.slice(1, p + 1), sigma };
      if (this.regressors.length) out.effects = theta.slice(p + 1);
      return out;
    }
    if (this.regressors.length) out.effects = this.phi !== null ? [...this.beta] : null;
    return out;
  }

  decodeState(data: Json): ARState {
    const s = decodeFields({
      n: "count", lags: "numbers", information: "json", moment: "numbers", shape: "number", rate: "number", log_likelihood: "number",
      comp: "number", plan: "json",
    }, data) as ARState;
    for (const key of ["information", "plan"] as const) checkMatrix(key, s[key]);
    requireState(s.lags.length === Math.min(s.n, this.order), "one lag per observation, up to the order");
    const r = this.regressors.length;
    requireState(s.plan.length === (s.n > 0 ? r : 0) && s.plan.every((p) => p.length === this.horizon), "one plan per regressor");
    if (this.phi !== null) {
      requireState(s.information.length === 0 && s.moment.length === 0 && s.shape === 0 && s.rate === 0,
        "learnt statistics with fixed coefficients");
    } else {
      checkInformation(s.information, s.moment, this.order + 1 + r);
      requireState(s.shape >= 0 && s.rate >= 0, "noise statistics");
    }
    return s;
  }
}

register("autoregressive", "1", (c) => AutoRegressive.fromConfig(c));

// ------------------------------------------------------------------------------------- counts

type PoissonState = { n: number; shape: number; rate: number; plan: number[]; log_likelihood: number; comp: number };

/** Forecasts of counts: the Poisson local level model with a discounted gamma rate; see forecast.py. */
export class PoissonSmoothing extends Forecaster {
  readonly name = "poisson_smoothing";
  readonly version = "1";
  readonly regressors: string[] = [];
  readonly transform: string | null = null;
  readonly prior: [number, number];
  readonly horizon: number;
  readonly coverage: number;

  constructor(readonly discount: number, readonly exposure: string | null = null, prior: number[] = [0.0, 0.0], horizon = 1,
    coverage = 0.9, readonly input = "y") {
    super();
    if (!(0.0 < discount && discount <= 1.0)) invalid("discount must lie in (0, 1]");
    if (prior.length !== 2) invalid("prior must be a (shape, rate) pair");
    const [a0, b0] = prior;
    if (!(0.0 <= a0 && a0 < Infinity && 0.0 <= b0 && b0 < Infinity)) invalid("the prior's shape and rate must be finite and non-negative");
    if (exposure !== null && (typeof exposure !== "string" || exposure === input))
      invalid("exposure must name an input different from the counts'");
    this.prior = [a0, b0];
    this.horizon = checkHorizon(horizon);
    this.coverage = checkCoverage(coverage);
  }

  static fromConfig(c: Config): PoissonSmoothing {
    return new PoissonSmoothing(num(c, "discount"), optStr(c, "exposure"), nums(c, "prior"), int(c, "horizon"), num(c, "coverage"),
      str(c, "input"));
  }

  config(): Config {
    return {
      discount: this.discount, exposure: this.exposure, prior: [...this.prior], horizon: this.horizon, coverage: this.coverage,
      input: this.input,
    };
  }

  override inputs(): Record<string, InputContract> {
    const contracts: Record<string, InputContract> = { [this.input]: new InputContract(this.input, "integer", [], 0) };
    if (this.exposure !== null) contracts[this.exposure] = new InputContract(this.exposure, "real", [this.horizon + 1], 0.0);
    return contracts;
  }

  initialState(): PoissonState {
    return { n: 0, shape: this.prior[0], rate: this.prior[1], plan: [], log_likelihood: 0.0, comp: 0.0 };
  }

  private exposures(x: StatInput): [number, number[]] {
    if (this.exposure === null) return [1.0, []];
    const window = [...(x[this.exposure] as number[])];
    if (!window.every((v) => v > 0.0)) throw new ContractViolation(`${this.name}: exposures must be positive, got [${window.join(", ")}]`);
    return [window[0], window.slice(1)];
  }

  update(state: State, x: StatInput): Updated {
    const s = state as PoissonState;
    const y = x[this.input] as number;
    const [e, plan] = this.exposures(x);
    const a = this.discount * s.shape, b = this.discount * s.rate;
    let logLik = s.log_likelihood, comp = s.comp;
    let score: number | null = null, error: number | null = null;
    if (a > 0.0 && b > 0.0) {
      score = negativeBinomialLogPmf(y, a, b, e);
      [logLik, comp] = neumaierAdd(logLik, comp, score);
      error = y - e * a / b;
    }
    const next: PoissonState = {
      n: s.n + 1, shape: requireFinite(a + y, "shape"), rate: requireFinite(b + e, "rate"), plan, log_likelihood: logLik, comp,
    };
    return [next, error, score];
  }

  predictive(state: State): Predictive | null {
    const s = state as PoissonState;
    if (!(s.shape > 0.0 && s.rate > 0.0) || (this.exposure !== null && !s.plan.length)) return null;
    const mean: number[] = [], scale: number[] = [], size: number[] = [], rate: number[] = [], exposure: number[] = [];
    let factor = 1.0;
    for (let h = 1; h <= this.horizon; h++) {
      factor *= this.discount;
      const e = this.exposure !== null ? s.plan[h - 1] : 1.0;
      const a = factor * s.shape, b = factor * s.rate;
      const m = e * s.shape / s.rate;
      mean.push(m);
      scale.push(Math.sqrt(m * (1.0 + e / b)));
      size.push(a);
      rate.push(b);
      exposure.push(e);
    }
    return new Predictive(mean, scale, null, null, size, rate, exposure);
  }

  override parameters(state: State): { [key: string]: Json } {
    const s = state as PoissonState;
    const level = s.shape > 0.0 && s.rate > 0.0 ? s.shape / s.rate : null;
    return { level, shape: s.shape, rate: s.rate };
  }

  decodeState(data: Json): PoissonState {
    const s = decodeFields({
      n: "count", shape: "number", rate: "number", plan: "numbers", log_likelihood: "number", comp: "number",
    }, data) as PoissonState;
    requireState(s.shape >= 0 && s.rate >= 0, "the gamma distribution's parameters");
    requireState(s.plan.length === (this.exposure !== null && s.n > 0 ? this.horizon : 0), "one planned exposure per horizon");
    requireState(s.n > 0 || s.log_likelihood === 0, "statistics before data");
    return s;
  }
}

register("poisson_smoothing", "1", (c) => PoissonSmoothing.fromConfig(c));

// ----------------------------------------------------------------------------- model averaging

type ModelAveragingState = { n: number; states: State[]; log_weights: number[]; log_likelihood: number; comp: number };
/** (weight, model, its forecast) */
type Component = [number, Forecaster, Predictive];

const MODELS = "models must be ExponentialSmoothing or AutoRegressive forecasters, or all PoissonSmoothing ones";

/** Bayesian model averaging of forecasters, with optional forgetting (dynamic model averaging); see forecast.py. */
export class ModelAveraging extends Forecaster {
  readonly name = "model_averaging";
  readonly version = "1";
  readonly regressors: string[] = [];
  readonly transform: string | null = null;
  readonly models: Forecaster[];
  readonly prior: number[];
  readonly horizon: number;
  readonly input: string;
  readonly coverage: number;
  readonly counts: boolean;
  private readonly contracts: Record<string, InputContract>;
  private readonly logPrior: number[];

  constructor(models: Forecaster[], weights: number[] | null = null, readonly forgetting = 1.0, coverage = 0.9) {
    super();
    if (!models.length) invalid("at least one model is needed");
    const counts = models.every((m) => m instanceof PoissonSmoothing);
    for (const m of models) if (!(counts || m instanceof ExponentialSmoothing || m instanceof AutoRegressive)) invalid(MODELS);
    const first = models[0];
    if (new Set(models.map((m) => m.horizon)).size !== 1 || models.some((m) => !sameContracts(m.inputs(), first.inputs())))
      invalid("the models need the same horizon and the same inputs");
    const prior = weights !== null ? [...weights] : models.map(() => 1.0);
    if (prior.length !== models.length || !prior.every((w) => 0.0 < w && w < Infinity)) invalid("need one positive weight per model");
    if (!(0.0 < forgetting && forgetting <= 1.0)) invalid("forgetting must lie in (0, 1]");
    this.models = [...models];
    this.prior = prior;
    this.horizon = first.horizon;
    this.input = first.input;
    this.counts = counts;
    this.contracts = first.inputs();
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
      return p instanceof Forecaster ? p : invalid(MODELS);
    });
    return new ModelAveraging(models, c.weights === null ? null : nums(c, "weights"), num(c, "forgetting"), num(c, "coverage"));
  }

  config(): Config {
    return {
      models: this.models.map((m) => identity(m) as unknown as Json), weights: [...this.prior], forgetting: this.forgetting,
      coverage: this.coverage,
    };
  }

  override inputs(): Record<string, InputContract> {
    return { ...this.contracts };
  }

  initialState(): ModelAveragingState {
    return { n: 0, states: this.models.map((m) => m.initialState()), log_weights: [...this.logPrior], log_likelihood: 0.0, comp: 0.0 };
  }

  private flattened(logWeights: number[]): number[] {
    const logs = logWeights.map((v) => this.forgetting * v);
    const top = logSumExp(logs);
    return logs.map((v) => v - top);
  }

  update(state: State, x: StatInput): Updated {
    const s = state as ModelAveragingState;
    const states: State[] = [], scores: (number | null)[] = [];
    this.models.forEach((model, i) => {
      const [next, , score] = model.update(s.states[i], x);
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
      out.push([Math.exp(weights[i]), this.models[i], pred]);
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
    const points = comps.map(([, model, p]) => model.present(p)[0]);
    const quantiles = new Map<number | null, number>();
    for (const [, , p] of live) if (!p.counts && !quantiles.has(p.df)) quantiles.set(p.df, this.quantile(p.df));
    const mean: number[] = [], lower: number[] = [], upper: number[] = [];
    for (let h = 0; h < this.horizon; h++) {
      let total = 0.0;
      comps.forEach(([w], i) => {
        total += w * points[i][h];
      });
      mean.push(total);
      // the mixture's quantiles lie between the components' quantiles at the same levels
      const ends = live.map(([, , p]) => ModelAveraging.ends(p, h, tail, quantiles));
      let lo = ends[0][0], hi = ends[0][1];
      for (let i = 1; i < ends.length; i++) lo = pyMin(lo, ends[i][0]);
      for (let i = 1; i < ends.length; i++) hi = pyMax(hi, ends[i][1]);
      if (this.counts) {
        lower.push(ModelAveraging.countSolve(live, h, tail, lo, hi));
        upper.push(ModelAveraging.countSolve(live, h, 1.0 - tail, lo, hi));
      } else {
        lower.push(ModelAveraging.solve(live, h, tail, lo, hi));
        upper.push(ModelAveraging.solve(live, h, 1.0 - tail, lo, hi));
      }
    }
    return [mean, lower, upper];
  }

  /** A component's quantiles at `tail` and 1 - `tail`, in the data's units. */
  static ends(p: Predictive, h: number, tail: number, quantiles: Map<number | null, number>): [number, number] {
    if (p.counts) return [countQuantile(p, h, tail), countQuantile(p, h, 1.0 - tail)];
    const q = quantiles.get(p.df) as number;
    const lo = p.mean[h] - q * p.scale[h], hi = p.mean[h] + q * p.scale[h];
    if (p.transform === "log") return [Math.exp(lo), Math.exp(hi)];
    return [lo, hi];
  }

  /** The smallest count where the mixture CDF reaches `target`, between the components' quantiles. */
  static countSolve(live: Component[], h: number, target: number, lo: number, hi: number): number {
    const cdf = (k: number): number => {
      let total = 0.0;
      for (const [w, , p] of live) total += w * p.cdf(h, k);
      return total;
    };
    let a = Math.trunc(lo) - 1, b = Math.trunc(hi); // cdf(a) < target <= cdf(b), up to the weight left out
    if (cdf(b) < target) return b;
    while (b - a > 1) {
      const mid = Math.floor((a + b) / 2);
      if (cdf(mid) < target) a = mid;
      else b = mid;
    }
    return b;
  }

  /** The point where the mixture CDF reaches `target` in [lo, hi]: Newton steps, bisecting when they leave. */
  static solve(live: Component[], h: number, target: number, lo: number, hi: number): number {
    let x = 0.5 * (lo + hi);
    for (let it = 0; it < 100; it++) {
      let cdf = 0.0, pdf = 0.0;
      for (const [w, , p] of live) {
        cdf += w * p.cdf(h, x);
        pdf += w * p.density(h, x);
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
    return new Predictive(mean, lower.map((lo, i) => (upper[i] - lo) / (2.0 * z)), null);
  }

  override step(state: State, x: StatInput, _rng: SplitMix64 | null = null): [State, Output] {
    const [next, , score] = this.update(state, x);
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
  regressors?: string[]; transform?: string | null; forgetting?: number; horizon?: number; coverage?: number; input?: string;
}

/** Exponential smoothing that learns its smoothing parameters: model averaging over a grid (Python's averaged_smoothing). */
export function averagedSmoothing(options: AveragedSmoothingOptions = {}): ModelAveraging {
  const {
    trend = null, period = null, alphas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9], betas = [0.01, 0.1, 0.3], gammas = [0.01, 0.1, 0.3],
    phis = [0.9, 0.98], regressors = [], transform = null, forgetting = 1.0, horizon = 1, coverage = 0.9, input = "y",
  } = options;
  const models: Forecaster[] = [];
  for (const a of alphas)
    for (const b of trend !== null ? betas : [0.0])
      for (const g of period !== null ? gammas : [0.0])
        for (const f of trend === "damped" ? phis : [1.0])
          models.push(new ExponentialSmoothing(a, a * b, (1.0 - a) * g, f, trend, period, null, 0.0, null, null, regressors, null, 1.0, 1e-6,
            transform, horizon, coverage, input));
  return new ModelAveraging(models, null, forgetting, coverage);
}

export interface AveragedCountsOptions {
  discounts?: number[]; exposure?: string | null; forgetting?: number; horizon?: number; coverage?: number; input?: string;
}

/** Count forecasts that learn how fast the rate moves: model averaging over a grid of discounts (Python's averaged_counts). */
export function averagedCounts(options: AveragedCountsOptions = {}): ModelAveraging {
  const {
    discounts = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995, 1.0], exposure = null, forgetting = 1.0, horizon = 1, coverage = 0.9,
    input = "y",
  } = options;
  const models = discounts.map((d) => new PoissonSmoothing(d, exposure, [0.0, 0.0], horizon, coverage, input));
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
