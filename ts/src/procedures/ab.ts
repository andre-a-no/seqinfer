// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Continuously monitored A/B tests; mirrors procedures/ab.py (SampleRatioMismatch, TwoProportionTest, ABTest).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, int, invalid, num, nums, pyMax, pyMin, REJECT_H0, str } from "./util.js";

const ALTERNATIVES = ["two-sided", "greater", "less"];

function checkAlpha(alpha: number): number {
  if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
  return alpha;
}

function checkArms(arms: number[]): [number, number] {
  if (arms.length !== 2 || arms.some((a) => !Number.isInteger(a) || a < 0) || arms[0] === arms[1])
    invalid("arms must be two different non-negative integer labels");
  return [arms[0], arms[1]];
}

/** k log(k / n), with 0 log 0 = 0. */
const xlogxRatio = (k: number, n: number): number => (k > 0 ? k * Math.log(k / n) : 0.0);

/** Python's math.fsum: the correctly rounded sum of finite values (Shewchuk's partials). */
export function fsum(values: number[]): number {
  const partials: number[] = [];
  for (let x of values) {
    let i = 0;
    for (let y of partials) {
      if (Math.abs(x) < Math.abs(y)) [x, y] = [y, x];
      const hi = x + y;
      const lo = y - (hi - x);
      if (lo) partials[i++] = lo;
      x = hi;
    }
    partials.length = i;
    partials.push(x);
  }
  let n = partials.length, hi = 0.0;
  if (n > 0) {
    hi = partials[--n];
    let lo = 0.0;
    while (n > 0) {
      const x = hi;
      const y = partials[--n];
      hi = x + y;
      lo = y - (hi - x);
      if (lo !== 0.0) break;
    }
    if (n > 0 && ((lo < 0 && partials[n - 1] < 0) || (lo > 0 && partials[n - 1] > 0))) {
      const y = lo * 2;
      const x = hi + y;
      if (y === x - hi) hi = x;
    }
  }
  return hi;
}

// ---------------------------------------------------------------- sample ratio mismatch

type SRMState = { n: number; counts: number[]; log_bf: number; comp: number; max_log_bf: number; decision: Decision };

export class SampleRatioMismatch extends BaseProcedure {
  readonly name = "sample_ratio_mismatch";
  readonly version = "1";
  readonly eValueField = "log_e_value";
  readonly p: number[];
  private readonly logLevel: number;

  constructor(weights: number[] = [0.5, 0.5], readonly alpha = 0.001, readonly concentration = 100.0, readonly stopOnReject = false, readonly input = "arm") {
    super();
    if (weights.length < 2 || !weights.every((v) => Number.isFinite(v) && v > 0)) invalid("weights must be at least two positive numbers");
    const total = fsum(weights);
    this.p = weights.map((v) => v / total);
    checkAlpha(alpha);
    if (!(Number.isFinite(concentration) && concentration > 0)) invalid("concentration must be positive and finite");
    this.logLevel = -Math.log(alpha);
  }

  static fromConfig(c: Config): SampleRatioMismatch {
    return new SampleRatioMismatch(nums(c, "weights"), num(c, "alpha"), num(c, "concentration"), bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return { weights: [...this.p], alpha: this.alpha, concentration: this.concentration, stop_on_reject: this.stopOnReject, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input, "integer", [], 0, this.p.length - 1) };
  }

  initialState(): SRMState {
    return { n: 0, counts: this.p.map(() => 0), log_bf: 0.0, comp: 0.0, max_log_bf: 0.0, decision: null };
  }

  step(state: State, x: StatInput): [SRMState, State] {
    const s = state as SRMState;
    const k = x[this.input] as number;
    const p = this.p[k];
    const term = Math.log((this.concentration * p + s.counts[k]) / ((this.concentration + s.n) * p));
    const [logBf, comp] = neumaierAdd(s.log_bf, s.comp, term);
    const n = s.n + 1;
    const counts = s.counts.map((c, i) => (i === k ? c + 1 : c));
    const value = requireFinite(logBf + comp, "log Bayes factor");
    const best = pyMax(s.max_log_bf, value);
    const decision = s.decision || (value >= this.logLevel ? REJECT_H0 : null);
    return [
      { n, counts, log_bf: logBf, comp, max_log_bf: best, decision },
      { n, counts: [...counts], proportions: counts.map((c) => c / n), log_e_value: value, p_value: pyMin(1.0, Math.exp(-best)), decision },
    ];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): SRMState {
    const s = decodeFields({ n: "count", counts: "counts", log_bf: "number", comp: "number", max_log_bf: "number", decision: "decision" }, data) as SRMState;
    requireState(s.counts.length === this.p.length, "one count per arm");
    requireState(fsum(s.counts) === s.n, "counts must add up to n");
    return s;
  }
}

// ---------------------------------------------------------------- two proportions

type TPState = { successes: number[]; trials: number[]; log_marginal: number; comp: number; max_log_e: number; decision: Decision };

export class TwoProportionTest extends BaseProcedure {
  readonly name = "two_proportion_test";
  readonly version = "1";
  readonly eValueField = "log_e_value";
  readonly arms: [number, number];
  readonly prior: [number, number];
  private readonly logLevel: number;

  constructor(
    readonly alpha = 0.05,
    readonly alternative = "two-sided",
    arms: number[] = [1, 0],
    prior: number[] = [0.5, 0.5],
    readonly stopOnReject = false,
    readonly armInput = "arm",
    readonly input = "y",
  ) {
    super();
    checkAlpha(alpha);
    if (!ALTERNATIVES.includes(alternative)) invalid(`alternative must be one of ${ALTERNATIVES}`);
    this.arms = checkArms(arms);
    if (prior.length !== 2) invalid("prior must be two positive numbers");
    const [a, b] = prior;
    if (!(Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0)) invalid("prior must be two positive numbers");
    if (armInput === input) invalid("the arm and the outcome need different names");
    this.prior = [a, b];
    this.logLevel = -Math.log(alpha);
  }

  static fromConfig(c: Config): TwoProportionTest {
    return new TwoProportionTest(num(c, "alpha"), str(c, "alternative"), nums(c, "arms"), nums(c, "prior"), bool(c, "stop_on_reject"),
      str(c, "arm_input"), str(c, "input"));
  }

  config(): Config {
    return {
      alpha: this.alpha, alternative: this.alternative, arms: [...this.arms], prior: [...this.prior],
      stop_on_reject: this.stopOnReject, arm_input: this.armInput, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.armInput]: new InputContract(this.armInput, "integer", [], 0), [this.input]: new InputContract(this.input, "binary") };
  }

  initialState(): TPState {
    return { successes: [0, 0], trials: [0, 0], log_marginal: 0.0, comp: 0.0, max_log_e: 0.0, decision: null };
  }

  /** log sup of the likelihood over H0. */
  logNullLikelihood(successes: number[], trials: number[]): number {
    const [st, sc] = successes, [nt, nc] = trials;
    const arm = (s: number, n: number): number => (n ? xlogxRatio(s, n) + xlogxRatio(n - s, n) : 0.0);
    if (this.alternative !== "two-sided" && (nt === 0 || nc === 0)) return arm(st, nt) + arm(sc, nc);
    if (this.alternative !== "two-sided") {
      const free = this.alternative === "greater" ? st * nc <= sc * nt : st * nc >= sc * nt;
      if (free) return arm(st, nt) + arm(sc, nc);
    }
    return arm(st + sc, nt + nc);
  }

  logEValue(s: TPState): number {
    return s.log_marginal + s.comp - this.logNullLikelihood(s.successes, s.trials);
  }

  private output(s: TPState, value: number): State {
    const [st, sc] = s.successes, [nt, nc] = s.trials;
    const rt = nt ? st / nt : null;
    const rc = nc ? sc / nc : null;
    const difference = rt === null || rc === null ? null : rt - rc;
    return { n: nt + nc, rates: [rt, rc], difference, log_e_value: value, p_value: pyMin(1.0, Math.exp(-s.max_log_e)), decision: s.decision };
  }

  step(state: State, x: StatInput): [TPState, State] {
    const s = state as TPState;
    const label = x[this.armInput] as number;
    if (!this.arms.includes(label)) return [s, this.output(s, this.logEValue(s))];
    const k = this.arms.indexOf(label);
    const y = x[this.input] as number;
    const [a, b] = this.prior;
    const sk = s.successes[k], nk = s.trials[k];
    const term = Math.log((y ? a + sk : b + nk - sk) / (a + b + nk));
    const [logMarginal, comp] = neumaierAdd(s.log_marginal, s.comp, term);
    const successes = [...s.successes], trials = [...s.trials];
    successes[k] += y ? 1 : 0;
    trials[k] += 1;
    const partial: TPState = { successes, trials, log_marginal: logMarginal, comp, max_log_e: 0.0, decision: null };
    const value = requireFinite(this.logEValue(partial), "log e-value");
    const decision = s.decision || (value >= this.logLevel ? REJECT_H0 : null);
    const next: TPState = { successes, trials, log_marginal: logMarginal, comp, max_log_e: pyMax(s.max_log_e, value), decision };
    return [next, this.output(next, value)];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): TPState {
    const s = decodeFields({
      successes: "counts", trials: "counts", log_marginal: "number", comp: "number", max_log_e: "number", decision: "decision",
    }, data) as TPState;
    requireState(s.successes.length === 2 && s.trials.length === 2, "two arms");
    requireState(s.successes.every((v, i) => 0 <= v && v <= s.trials[i]), "successes out of range");
    return s;
  }
}

// ---------------------------------------------------------------- difference of means

type ABState = {
  n: number; treated: number; sum_treated: number; sum_control: number; mean_x: number; mean_y: number; sxx: number; sxy: number;
  psi_mean: number; psi_m2: number; plain_mean: number; plain_m2: number; decision: Decision;
};

export class ABTest extends BaseProcedure {
  readonly name = "ab_test";
  readonly version = "1";
  readonly arms: [number, number];
  private readonly rho2: number;

  constructor(
    readonly allocation = 0.5,
    readonly alpha = 0.05,
    readonly covariate = false,
    arms: number[] = [1, 0],
    readonly start = 1000,
    readonly tunedFor = 1000.0,
    readonly stopOnReject = false,
    readonly armInput = "arm",
    readonly input = "y",
    readonly covariateInput = "x",
  ) {
    super();
    if (!(0 < allocation && allocation < 1)) invalid("allocation must lie in (0, 1)");
    checkAlpha(alpha);
    this.arms = checkArms(arms);
    if (!Number.isInteger(start) || start < 2) invalid("start must be an int, at least 2");
    if (!(Number.isFinite(tunedFor) && tunedFor >= 1)) invalid("tuned_for must be a number of observations, at least 1");
    if (new Set([armInput, input, covariateInput]).size !== 3) invalid("the arm, outcome and covariate need different names");
    const c = -2.0 * Math.log(alpha);
    this.rho2 = (c + Math.log1p(c)) / tunedFor;
  }

  static fromConfig(c: Config): ABTest {
    return new ABTest(num(c, "allocation"), num(c, "alpha"), bool(c, "covariate"), nums(c, "arms"), int(c, "start"), num(c, "tuned_for"),
      bool(c, "stop_on_reject"), str(c, "arm_input"), str(c, "input"), str(c, "covariate_input"));
  }

  config(): Config {
    return {
      allocation: this.allocation, alpha: this.alpha, covariate: this.covariate, arms: [...this.arms], start: this.start,
      tuned_for: this.tunedFor, stop_on_reject: this.stopOnReject, arm_input: this.armInput, input: this.input,
      covariate_input: this.covariateInput,
    };
  }

  inputs(): Record<string, InputContract> {
    const out: Record<string, InputContract> = {
      [this.armInput]: new InputContract(this.armInput, "integer", [], 0),
      [this.input]: new InputContract(this.input),
    };
    if (this.covariate) out[this.covariateInput] = new InputContract(this.covariateInput);
    return out;
  }

  initialState(): ABState {
    return {
      n: 0, treated: 0, sum_treated: 0.0, sum_control: 0.0, mean_x: 0.0, mean_y: 0.0, sxx: 0.0, sxy: 0.0,
      psi_mean: 0.0, psi_m2: 0.0, plain_mean: 0.0, plain_m2: 0.0, decision: null,
    };
  }

  /** The asymptotic confidence sequence's half-width. */
  halfWidth(n: number, sd: number): number {
    const v = n * this.rho2 + 1.0;
    return sd * Math.sqrt(((2.0 * v) / (n * n * this.rho2)) * (0.5 * Math.log(v) - Math.log(this.alpha)));
  }

  /** f_{t-1}(x) with and without the covariate, from earlier units only. */
  prediction(s: ABState, xValue: number | null): [number, number] {
    if (s.n === 0) return [0.0, 0.0];
    const plain = s.mean_y;
    if (xValue === null || s.sxx <= 0.0) return [plain, plain];
    return [plain + (s.sxy / s.sxx) * (xValue - s.mean_x), plain];
  }

  private output(s: ABState): State {
    const n = s.n;
    const control = n - s.treated;
    const means = [s.treated ? s.sum_treated / s.treated : null, control ? s.sum_control / control : null];
    let lower: number | null = null, upper: number | null = null;
    if (n >= this.start && s.psi_m2 > 0.0) {
      const h = this.halfWidth(n, Math.sqrt(s.psi_m2 / (n - 1)));
      [lower, upper] = [s.psi_mean - h, s.psi_mean + h];
    }
    let ratio: number | null = null;
    if (this.covariate && n >= 2 && s.plain_m2 > 0.0) ratio = s.psi_m2 / s.plain_m2;
    return { n, means, estimate: s.psi_mean, lower, upper, variance_ratio: ratio, decision: s.decision };
  }

  step(state: State, x: StatInput): [ABState, State] {
    const s = state as ABState;
    const label = x[this.armInput] as number;
    if (!this.arms.includes(label)) return [s, this.output(s)];
    const treated = label === this.arms[0];
    const y = x[this.input] as number;
    const xv = this.covariate ? (x[this.covariateInput] as number) : null;
    const weight = treated ? 1.0 / this.allocation : -1.0 / (1.0 - this.allocation);
    const [fitted, plainFit] = this.prediction(s, xv);
    const psi = weight * (y - fitted), plain = weight * (y - plainFit);
    const n = s.n + 1;
    const d = psi - s.psi_mean;
    const psiMean = s.psi_mean + d / n;
    const psiM2 = s.psi_m2 + d * (psi - psiMean);
    const d0 = plain - s.plain_mean;
    const plainMean = s.plain_mean + d0 / n;
    const plainM2 = s.plain_m2 + d0 * (plain - plainMean);
    const dy = y - s.mean_y;
    const meanY = s.mean_y + dy / n;
    let { mean_x: meanX, sxx, sxy } = s;
    if (xv !== null) {
      const dx = xv - s.mean_x;
      meanX = s.mean_x + dx / n;
      sxx = s.sxx + dx * (xv - meanX);
      sxy = s.sxy + dx * (y - meanY);
    }
    let next: ABState = {
      n, treated: s.treated + (treated ? 1 : 0), sum_treated: s.sum_treated + (treated ? y : 0.0),
      sum_control: s.sum_control + (treated ? 0.0 : y), mean_x: meanX, mean_y: meanY, sxx, sxy,
      psi_mean: requireFinite(psiMean, "effect estimate"), psi_m2: psiM2, plain_mean: plainMean, plain_m2: plainM2, decision: s.decision,
    };
    let out = this.output(next);
    const lo = out.lower as number | null, hi = out.upper as number | null;
    const excluded = lo !== null && hi !== null && !(lo <= 0.0 && 0.0 <= hi);
    if (next.decision === null && excluded) {
      next = { ...next, decision: REJECT_H0 };
      out = this.output(next);
    }
    return [next, out];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): ABState {
    const s = decodeFields({
      n: "count", treated: "count", sum_treated: "number", sum_control: "number", mean_x: "number", mean_y: "number",
      sxx: "number", sxy: "number", psi_mean: "number", psi_m2: "number", plain_mean: "number", plain_m2: "number", decision: "decision",
    }, data) as ABState;
    requireState(0 <= s.treated && s.treated <= s.n, "treated count out of range");
    requireState(s.sxx >= 0 && s.psi_m2 >= 0 && s.plain_m2 >= 0, "negative sum of squares");
    return s;
  }
}

register("sample_ratio_mismatch", "1", (c) => SampleRatioMismatch.fromConfig(c));
register("two_proportion_test", "1", (c) => TwoProportionTest.fromConfig(c));
register("ab_test", "1", (c) => ABTest.fromConfig(c));
