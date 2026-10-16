// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Forward filtering in a hidden Markov model; mirrors procedures/hmm.py, the emission models operation for operation.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { NumericalError } from "../errors.js";
import * as la from "../linalg.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { fsum } from "./ab.js";
import { lgamma, lgammaRatio } from "./distributions.js";
import { BaseProcedure, type Config, invalid, nums, obj, pyMax, str } from "./util.js";

const LOG_SQRT_2PI = 0.5 * Math.log(2.0 * Math.PI);

function distribution(values: unknown, size: number | null, name: string): number[] {
  if (!Array.isArray(values) || !values.every((v) => typeof v === "number")) invalid(`${name} must be a list of numbers`);
  const p = [...(values as number[])];
  if (size !== null && p.length !== size) invalid(`${name} must have ${size} entries`);
  if (!p.length || p.some((v) => !(Number.isFinite(v) && v >= 0)) || Math.abs(fsum(p) - 1.0) > 1e-12)
    invalid(`${name} must be a probability distribution (non-negative, summing to 1)`);
  return p;
}

/** An observation: a number, or a vector for MultivariateGaussianEmission. */
export type EmissionInput = number | number[];

export interface Emission {
  readonly states: number;
  spec(): Config;
  contract(name: string): InputContract;
  logDensity(k: number, y: EmissionInput): number;
  /** Complete-data sufficient statistic g_k(y), for EM (the same length for every state). */
  statistics(k: number, y: EmissionInput): number[];
  /** Maximum-likelihood (M-step) update from totals[k] = [sum of weights, weighted sums of g...]; empty states keep theirs. */
  fromStatistics(totals: number[][], minScale?: number): Emission;
}

export class CategoricalEmission implements Emission {
  readonly p: number[][];
  readonly states: number;

  constructor(probabilities: unknown) {
    if (!Array.isArray(probabilities)) invalid("probabilities must be a list of rows");
    this.p = probabilities.map((row) => distribution(row, null, "emission row"));
    if (new Set(this.p.map((r) => r.length)).size !== 1) invalid("every state needs a distribution over the same symbols");
    this.states = this.p.length;
  }

  spec(): Config {
    return { model: "categorical", probabilities: this.p.map((r) => [...r]) };
  }

  contract(name: string): InputContract {
    return new InputContract(name, "integer", [], 0, this.p[0].length - 1);
  }

  logDensity(k: number, y: number): number {
    const p = this.p[k][y];
    return p > 0 ? Math.log(p) : -Infinity;
  }

  /** g(y): the indicator of each symbol. */
  statistics(_k: number, y: number): number[] {
    const out = new Array<number>(this.p[0].length).fill(0.0);
    out[y] = 1.0;
    return out;
  }

  /** `minScale` is a pseudo-count added to every symbol (0: plain maximum likelihood). */
  fromStatistics(totals: number[][], minScale = 0.0): CategoricalEmission {
    const rows = totals.map(([w, ...counts], k) => {
      if (!(w > 1e-300)) return this.p[k];
      const total = w + minScale * counts.length;
      const row = counts.map((c) => (c + minScale) / total);
      const s = fsum(row);
      return row.map((v) => v / s);
    });
    return new CategoricalEmission(rows);
  }
}

export class GaussianEmission implements Emission {
  readonly states: number;

  constructor(readonly means: number[], readonly sds: number[]) {
    if (means.length !== sds.length || !means.length) invalid("one mean and one standard deviation per state");
    if (!means.every(Number.isFinite) || !sds.every((s) => 0 < s && s < Infinity))
      invalid("means must be finite and standard deviations positive");
    this.states = means.length;
  }

  spec(): Config {
    return { model: "gaussian", means: [...this.means], sds: [...this.sds] };
  }

  contract(name: string): InputContract {
    return new InputContract(name);
  }

  logDensity(k: number, y: number): number {
    const z = (y - this.means[k]) / this.sds[k];
    return -0.5 * z * z - Math.log(this.sds[k]) - LOG_SQRT_2PI;
  }

  /** g(y) = (y, y^2). */
  statistics(_k: number, y: number): number[] {
    return [y, y * y];
  }

  fromStatistics(totals: number[][], minScale = 1e-6): GaussianEmission {
    const means: number[] = [], sds: number[] = [];
    totals.forEach(([w, s1, s2], k) => {
      if (!(w > 1e-300)) {
        means.push(this.means[k]);
        sds.push(this.sds[k]);
        return;
      }
      const m = s1 / w;
      means.push(m);
      sds.push(pyMax(Math.sqrt(pyMax(s2 / w - m * m, 0.0)), minScale));
    });
    return new GaussianEmission(means, sds);
  }
}

export class PoissonEmission implements Emission {
  readonly states: number;

  constructor(readonly rates: number[]) {
    if (!rates.length || !rates.every((r) => 0 < r && r < Infinity)) invalid("rates must be positive and finite");
    this.states = rates.length;
  }

  spec(): Config {
    return { model: "poisson", rates: [...this.rates] };
  }

  contract(name: string): InputContract {
    return new InputContract(name, "integer", [], 0);
  }

  logDensity(k: number, y: number): number {
    return y * Math.log(this.rates[k]) - this.rates[k] - lgamma(y + 1.0);
  }

  /** g(y) = (y,). */
  statistics(_k: number, y: number): number[] {
    return [y];
  }

  fromStatistics(totals: number[][], minScale = 1e-6): PoissonEmission {
    return new PoissonEmission(totals.map(([w, s1], k) => (w > 1e-300 ? pyMax(s1 / w, minScale) : this.rates[k])));
  }
}

/** Covariances whose Cholesky pivot falls below this fraction of the diagonal entry are numerically singular. */
const MIN_PIVOT = 1e-12;

function covarianceFactor(cov: la.Matrix, name: string): la.Matrix {
  if (!la.isSymmetric(cov)) invalid(`${name} must be symmetric`);
  let low: la.Matrix;
  try {
    low = la.cholesky(cov);
  } catch {
    return invalid(`${name} must be positive definite`);
  }
  for (let i = 0; i < cov.length; i++)
    if (!(low[i][i] * low[i][i] > MIN_PIVOT * cov[i][i])) invalid(`${name} is numerically singular (a correlation within about 1e-12 of +-1)`);
  return low;
}

function rows(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value) || !value.length || !value.every(Array.isArray)) invalid(`${name} must be a non-empty list of lists`);
  return value;
}

function observation(y: EmissionInput, d: number): number[] {
  if (!Array.isArray(y) || y.length !== d) invalid(`expected an observation of length ${d}, got ${Array.isArray(y) ? y.length : 1}`);
  return y;
}

/** y | z = k ~ N_d(means[k], covariances[k]); see MultivariateGaussianEmission in hmm.py. */
export class MultivariateGaussianEmission implements Emission {
  readonly means: number[][];
  readonly covariances: number[][][];
  readonly dimension: number;
  readonly states: number;
  private readonly low: la.Matrix[];
  private readonly halfLogDet: number[];
  private readonly logNorm: number;

  constructor(means: unknown, covariances: unknown) {
    this.means = rows(means, "means").map((m) => la.vector(m, null, "mean"));
    const d = (this.dimension = this.means[0].length);
    if (this.means.some((m) => m.length !== d)) invalid("every mean must have the same length");
    const covs = rows(covariances, "covariances");
    if (covs.length !== this.means.length) invalid("one mean and one covariance per state");
    this.covariances = covs.map((c) => la.matrix(rows(c, "covariance"), d, d, "covariance"));
    this.low = this.covariances.map((c, k) => covarianceFactor(c, `covariance ${k}`));
    this.halfLogDet = this.low.map((low) => {
      let total = 0.0;
      for (let i = 0; i < d; i++) total += Math.log(low[i][i]);
      return total;
    });
    this.logNorm = d * LOG_SQRT_2PI;
    this.states = this.means.length;
  }

  spec(): Config {
    return { model: "multivariate_gaussian", means: this.means.map((m) => [...m]), covariances: this.covariances.map((c) => c.map((r) => [...r])) };
  }

  contract(name: string): InputContract {
    return new InputContract(name, "real", [this.dimension]);
  }

  logDensity(k: number, y: EmissionInput): number {
    const d = this.dimension, mean = this.means[k], low = this.low[k];
    const x = observation(y, d);
    const v = new Array<number>(d).fill(0.0);
    let q = 0.0;
    for (let i = 0; i < d; i++) {
      let s = x[i] - mean[i];
      for (let j = 0; j < i; j++) s -= low[i][j] * v[j];
      v[i] = s / low[i][i];
      q += v[i] * v[i];
    }
    return -0.5 * q - this.halfLogDet[k] - this.logNorm;
  }

  /** g(y) = (y_1..y_d, then y_i y_j for i <= j, row by row). */
  statistics(_k: number, y: EmissionInput): number[] {
    const d = this.dimension;
    const x = observation(y, d);
    const out = [...x];
    for (let i = 0; i < d; i++) for (let j = i; j < d; j++) out.push(x[i] * x[j]);
    return out;
  }

  fromStatistics(totals: number[][], minScale = 1e-6): MultivariateGaussianEmission {
    const d = this.dimension;
    const size = d + (d * (d + 1)) / 2;
    const means: number[][] = [], covs: number[][][] = [];
    totals.forEach((row, k) => {
      if (row.length !== size + 1) invalid(`totals of state ${k} must have ${size + 1} entries`);
      const w = row[0];
      if (!(w > 1e-300)) {
        means.push(this.means[k]);
        covs.push(this.covariances[k]);
        return;
      }
      const m = Array.from({ length: d }, (_, i) => row[1 + i] / w);
      const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0.0));
      let index = 1 + d;
      for (let i = 0; i < d; i++) {
        for (let j = i; j < d; j++) {
          let c = row[index] / w - m[i] * m[j];
          if (i === j) c += minScale * minScale;
          cov[i][j] = cov[j][i] = c;
          index++;
        }
      }
      try {
        if (!m.every(Number.isFinite)) invalid("non-finite mean");
        covarianceFactor(la.matrix(cov, d, d, "covariance"), "covariance");
      } catch (e) {
        if (!(e instanceof RangeError)) throw e;
        means.push(this.means[k]);
        covs.push(this.covariances[k]);
        return;
      }
      means.push(m);
      covs.push(cov);
    });
    return new MultivariateGaussianEmission(means, covs);
  }
}

function perState(values: unknown, states: number, name: string): number[] {
  if (typeof values === "number") return new Array<number>(states).fill(values);
  if (!Array.isArray(values) || !values.every((v) => typeof v === "number")) invalid(`${name} must be a number or a list of numbers`);
  if (values.length !== states) invalid(`${name} must have one entry per state (or be one number)`);
  return [...(values as number[])];
}

/** log |a - b| for finite a != b, also when a - b overflows. */
function logAbsDifference(a: number, b: number): number {
  const d = a - b;
  if (Number.isFinite(d)) return Math.log(Math.abs(d));
  return Math.log(Math.abs(0.5 * a - 0.5 * b)) + Math.log(2.0);
}

/** y | z = k ~ means[k] + scales[k] T(df[k]); see StudentTEmission in hmm.py. */
export class StudentTEmission implements Emission {
  readonly means: number[];
  readonly scales: number[];
  readonly df: number[];
  readonly states: number;
  private readonly constant: number[];
  private readonly rootDf: number[];
  private readonly logWidth: number[];

  constructor(means: number[], scales: number | number[], df: number | number[]) {
    this.means = [...means];
    const k = this.means.length;
    this.scales = perState(scales, k, "scales");
    this.df = perState(df, k, "df");
    if (!k) invalid("one mean, scale and df per state");
    if (!this.means.every(Number.isFinite) || !this.scales.every((s) => 0 < s && s < Infinity)) invalid("means must be finite and scales positive");
    if (!this.df.every((v) => 0 < v && v < Infinity)) invalid("degrees of freedom must be positive and finite");
    this.constant = this.df.map((v, i) => lgammaRatio(0.5 * v, 0.5) - 0.5 * (Math.log(v) + Math.log(Math.PI)) - Math.log(this.scales[i]));
    this.rootDf = this.df.map((v) => Math.sqrt(v));
    this.logWidth = this.df.map((v, i) => Math.log(this.scales[i]) + 0.5 * Math.log(v));
    this.states = k;
  }

  spec(): Config {
    return { model: "student_t", means: [...this.means], scales: [...this.scales], df: [...this.df] };
  }

  contract(name: string): InputContract {
    return new InputContract(name);
  }

  logDensity(k: number, y: number): number {
    const r = Math.abs((y - this.means[k]) / this.scales[k]) / this.rootDf[k];
    const log1pR2 = r < 1e150 ? Math.log1p(r * r) : 2.0 * (logAbsDifference(y, this.means[k]) - this.logWidth[k]);
    return this.constant[k] - 0.5 * (this.df[k] + 1.0) * log1pR2;
  }

  /** E[precision | y, z = k] = (df + 1) / (df + z^2). */
  weight(k: number, y: number): number {
    const z = (y - this.means[k]) / this.scales[k];
    return (this.df[k] + 1.0) / (this.df[k] + z * z);
  }

  /** g_k(y) = (u, u y, u y^2) at the current parameters of state k. */
  statistics(k: number, y: number): number[] {
    const u = this.weight(k, y);
    return [u, u * y, u * y * y];
  }

  /** The ECM M-step with df fixed; see hmm.py. */
  fromStatistics(totals: number[][], minScale = 1e-6): StudentTEmission {
    const means: number[] = [], scales: number[] = [];
    totals.forEach(([w, su, suy, suy2], k) => {
      if (!(w > 1e-300 && su > 1e-300)) {
        means.push(this.means[k]);
        scales.push(this.scales[k]);
        return;
      }
      const m = suy / su;
      const v = (suy2 - 2.0 * m * suy + m * m * su) / w;
      means.push(m);
      scales.push(pyMax(Math.sqrt(pyMax(v, 0.0)), minScale));
    });
    return new StudentTEmission(means, scales, this.df);
  }
}

export function emissionFromSpec(spec: Config): Emission {
  switch (str(spec, "model")) {
    case "categorical":
      if (!("probabilities" in spec)) invalid("config is missing 'probabilities'");
      return new CategoricalEmission(spec.probabilities);
    case "gaussian":
      return new GaussianEmission(nums(spec, "means"), nums(spec, "sds"));
    case "poisson":
      return new PoissonEmission(nums(spec, "rates"));
    case "multivariate_gaussian":
      for (const key of ["means", "covariances"]) if (!(key in spec)) invalid(`config is missing '${key}'`);
      return new MultivariateGaussianEmission(spec.means, spec.covariances);
    case "student_t":
      if (!("df" in spec)) invalid("config is missing 'df'");
      return new StudentTEmission(nums(spec, "means"), nums(spec, "scales"), spec.df as number | number[]);
    default:
      return invalid(`unknown emission model ${JSON.stringify(spec.model)}`);
  }
}

type HMMState = { n: number; probabilities: number[]; log_likelihood: number; comp: number };

export class HiddenMarkovFilter extends BaseProcedure {
  readonly name = "hidden_markov_filter";
  readonly version = "1";
  readonly initial: number[];
  readonly transition: number[][];
  readonly states: number;

  constructor(initial: unknown, transition: unknown, readonly emission: Emission, readonly input = "y") {
    super();
    this.initial = distribution(initial, null, "initial");
    const k = this.initial.length;
    if (!Array.isArray(transition)) invalid("transition must be a list of rows");
    this.transition = transition.map((row) => distribution(row, k, "transition row"));
    if (this.transition.length !== k) invalid(`transition must be ${k} x ${k}`);
    if (emission.states !== k) invalid(`the emission model must describe ${k} states`);
    this.states = k;
  }

  static fromConfig(c: Config): HiddenMarkovFilter {
    for (const key of ["initial", "transition"]) if (!(key in c)) invalid(`config is missing '${key}'`);
    return new HiddenMarkovFilter(c.initial, c.transition, emissionFromSpec(obj(c, "emission")), str(c, "input"));
  }

  config(): Config {
    return { initial: [...this.initial], transition: this.transition.map((r) => [...r]), emission: this.emission.spec(), input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.emission.contract(this.input) };
  }

  initialState(): HMMState {
    return { n: 0, probabilities: [], log_likelihood: 0.0, comp: 0.0 };
  }

  /** P(z_{n+1} = k | y_1..y_n). */
  predicted(s: HMMState): number[] {
    if (s.n === 0) return this.initial;
    const out: number[] = [];
    for (let k = 0; k < this.states; k++) {
      let total = 0.0;
      for (let j = 0; j < this.states; j++) total += s.probabilities[j] * this.transition[j][k];
      out.push(total);
    }
    return out;
  }

  step(state: State, x: StatInput): [HMMState, State] {
    const s = state as HMMState;
    const y = x[this.input] as EmissionInput;
    const prior = this.predicted(s);
    const logs = prior.map((p, k) => (p > 0 ? this.emission.logDensity(k, y) : -Infinity));
    const top = Math.max(...logs);
    if (top === -Infinity) throw new NumericalError(`${this.name}: observation ${Array.isArray(y) ? JSON.stringify(y) : y} is impossible in every reachable state`);
    const joint = prior.map((p, k) => p * Math.exp(logs[k] - top));
    let total = 0.0;
    for (const v of joint) total += v;
    const posterior = joint.map((v) => v / total);
    const [logLik, comp] = neumaierAdd(s.log_likelihood, s.comp, top + Math.log(total));
    const n = s.n + 1;
    let best = 0;
    for (let k = 1; k < this.states; k++) if (posterior[k] > posterior[best]) best = k;
    return [
      { n, probabilities: posterior, log_likelihood: logLik, comp },
      { n, probabilities: [...posterior], most_likely: best, log_likelihood: requireFinite(logLik + comp, "log likelihood") },
    ];
  }

  decodeState(data: Json): HMMState {
    const s = decodeFields({ n: "count", probabilities: "numbers", log_likelihood: "number", comp: "number" }, data) as HMMState;
    requireState(s.probabilities.length === (s.n ? this.states : 0), "one probability per state");
    requireState(s.probabilities.every((p) => p >= 0), "negative probability");
    requireState(s.n === 0 || Math.abs(fsum(s.probabilities) - 1.0) < 1e-9, "probabilities must sum to 1");
    return s;
  }
}

register("hidden_markov_filter", "1", (c) => HiddenMarkovFilter.fromConfig(c));
