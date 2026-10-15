// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Continuously monitored A/B tests; mirrors procedures/ab.py (SampleRatioMismatch, TwoProportionTest, ABTest,
// e_bh and MultipleTests).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, fromIdentity, type Identity, identity, type Procedure, register, requireState, type State } from "../procedure.js";
import type { SplitMix64 } from "../rng.js";
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

const METHODS = ["grow", "universal"];

type TPState = {
  successes: number[]; trials: number[]; log_sum: number; comp: number;
  pending_treatment: number[]; pending_control: number[]; block_successes: number[]; block_trials: number[];
  max_log_e: number; decision: Decision;
};

export class TwoProportionTest extends BaseProcedure {
  readonly name = "two_proportion_test";
  readonly version = "2";
  readonly eValueField = "log_e_value";
  readonly arms: [number, number];
  readonly prior: [number, number];
  readonly block: [number, number];
  private readonly logLevel: number;

  constructor(
    readonly alpha = 0.05,
    readonly alternative = "two-sided",
    readonly method = "grow",
    block: number[] = [1, 1],
    arms: number[] = [1, 0],
    prior: number[] = [0.5, 0.5],
    readonly stopOnReject = false,
    readonly armInput = "arm",
    readonly input = "y",
  ) {
    super();
    checkAlpha(alpha);
    if (!ALTERNATIVES.includes(alternative)) invalid(`alternative must be one of ${ALTERNATIVES}`);
    if (!METHODS.includes(method)) invalid(`method must be one of ${METHODS}`);
    if (block.length !== 2 || block.some((k) => !Number.isInteger(k) || k < 1 || k > 100))
      invalid("block must be two ints in [1, 100]: treatment and control units per block");
    this.block = [block[0], block[1]];
    this.arms = checkArms(arms);
    if (prior.length !== 2) invalid("prior must be two positive numbers");
    const [a, b] = prior;
    if (!(Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0)) invalid("prior must be two positive numbers");
    if (armInput === input) invalid("the arm and the outcome need different names");
    this.prior = [a, b];
    this.logLevel = -Math.log(alpha);
  }

  static fromConfig(c: Config): TwoProportionTest {
    return new TwoProportionTest(num(c, "alpha"), str(c, "alternative"), str(c, "method"), nums(c, "block"), nums(c, "arms"),
      nums(c, "prior"), bool(c, "stop_on_reject"), str(c, "arm_input"), str(c, "input"));
  }

  config(): Config {
    return {
      alpha: this.alpha, alternative: this.alternative, method: this.method, block: [...this.block], arms: [...this.arms],
      prior: [...this.prior], stop_on_reject: this.stopOnReject, arm_input: this.armInput, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.armInput]: new InputContract(this.armInput, "integer", [], 0), [this.input]: new InputContract(this.input, "binary") };
  }

  initialState(): TPState {
    return {
      successes: [0, 0], trials: [0, 0], log_sum: 0.0, comp: 0.0, pending_treatment: [], pending_control: [],
      block_successes: [0, 0], block_trials: [0, 0], max_log_e: 0.0, decision: null,
    };
  }

  /** log sup of the likelihood over H0 (method "universal"). */
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

  /** (a, b, p): the alternative fitted on completed blocks and its projection on H0. */
  plugIn(successes: number[], trials: number[]): [number, number, number] {
    const [pa, pb] = this.prior;
    const a = (successes[0] + pa) / (trials[0] + pa + pb);
    const b = (successes[1] + pa) / (trials[1] + pa + pb);
    const [kt, kc] = this.block;
    if ((this.alternative === "greater" && a <= b) || (this.alternative === "less" && a >= b)) {
      const p = (kt * a + kc * b) / (kt + kc);
      return [p, p, p];
    }
    return [a, b, (kt * a + kc * b) / (kt + kc)];
  }

  blockLogE(a: number, b: number, p: number, treated: number[], control: number[]): number {
    if (a === b && b === p) return 0.0;
    let out = 0.0;
    for (const y of treated) out += y ? Math.log(a / p) : Math.log1p(-a) - Math.log1p(-p);
    for (const y of control) out += y ? Math.log(b / p) : Math.log1p(-b) - Math.log1p(-p);
    return out;
  }

  logEValue(s: TPState): number {
    if (this.method === "grow") return s.log_sum + s.comp;
    return s.log_sum + s.comp - this.logNullLikelihood(s.successes, s.trials);
  }

  private output(s: TPState, value: number): State {
    const [st, sc] = s.successes, [nt, nc] = s.trials;
    const rt = nt ? st / nt : null;
    const rc = nc ? sc / nc : null;
    const difference = rt === null || rc === null ? null : rt - rc;
    const blocks = this.method === "grow" ? Math.floor(s.block_trials[0] / this.block[0]) : 0;
    return {
      n: nt + nc, rates: [rt, rc], difference, log_e_value: value, p_value: pyMin(1.0, Math.exp(-s.max_log_e)),
      decision: s.decision, blocks,
    };
  }

  step(state: State, x: StatInput): [TPState, State] {
    const s = state as TPState;
    const label = x[this.armInput] as number;
    if (!this.arms.includes(label)) return [s, this.output(s, this.logEValue(s))];
    const k = this.arms.indexOf(label);
    const y = (x[this.input] as number) ? 1 : 0;
    const successes = [...s.successes], trials = [...s.trials];
    successes[k] += y;
    trials[k] += 1;
    let logSum = s.log_sum, comp = s.comp;
    let pendingT = s.pending_treatment, pendingC = s.pending_control;
    let blockS = s.block_successes, blockN = s.block_trials;
    if (this.method === "universal") {
      const [pa, pb] = this.prior;
      const sk = s.successes[k], nk = s.trials[k];
      [logSum, comp] = neumaierAdd(logSum, comp, Math.log((y ? pa + sk : pb + nk - sk) / (pa + pb + nk)));
    } else {
      if (k === 0) pendingT = [...pendingT, y];
      else pendingC = [...pendingC, y];
      const [kt, kc] = this.block;
      while (pendingT.length >= kt && pendingC.length >= kc) {
        const treated = pendingT.slice(0, kt), control = pendingC.slice(0, kc);
        pendingT = pendingT.slice(kt);
        pendingC = pendingC.slice(kc);
        const [a, b, p] = this.plugIn(blockS, blockN);
        [logSum, comp] = neumaierAdd(logSum, comp, this.blockLogE(a, b, p, treated, control));
        const ones = (v: number[]) => v.filter((u) => u === 1).length;
        blockS = [blockS[0] + ones(treated), blockS[1] + ones(control)];
        blockN = [blockN[0] + kt, blockN[1] + kc];
      }
    }
    const partial: TPState = {
      successes, trials, log_sum: logSum, comp, pending_treatment: pendingT, pending_control: pendingC,
      block_successes: blockS, block_trials: blockN, max_log_e: 0.0, decision: null,
    };
    const value = requireFinite(this.logEValue(partial), "log e-value");
    const decision = s.decision || (value >= this.logLevel ? REJECT_H0 : null);
    const next: TPState = { ...partial, max_log_e: pyMax(s.max_log_e, value), decision };
    return [next, this.output(next, value)];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): TPState {
    const s = decodeFields({
      successes: "counts", trials: "counts", log_sum: "number", comp: "number", pending_treatment: "counts",
      pending_control: "counts", block_successes: "counts", block_trials: "counts", max_log_e: "number", decision: "decision",
    }, data) as TPState;
    const pairs = [s.successes, s.trials, s.block_successes, s.block_trials];
    requireState(pairs.every((v) => v.length === 2), "two arms");
    requireState([0, 1].every((i) => s.successes[i] <= s.trials[i] && s.block_successes[i] <= s.block_trials[i]),
      "successes out of range");
    requireState([...s.pending_treatment, ...s.pending_control].every((y) => y === 0 || y === 1), "pending outcomes are 0 or 1");
    if (this.method === "grow") {
      const [kt, kc] = this.block;
      requireState(s.block_trials[0] * kc === s.block_trials[1] * kt && s.block_trials[0] % kt === 0, "completed blocks");
      requireState(s.block_trials[0] + s.pending_treatment.length === s.trials[0]
        && s.block_trials[1] + s.pending_control.length === s.trials[1], "every unit is in a block or waiting for one");
    } else {
      requireState(!s.pending_treatment.length && !s.pending_control.length && s.block_trials[0] === 0 && s.block_trials[1] === 0,
        "universal inference has no blocks");
    }
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
register("two_proportion_test", "2", (c) => TwoProportionTest.fromConfig(c));
register("ab_test", "1", (c) => ABTest.fromConfig(c));

// ---------------------------------------------------------------- many tests

/** One contract satisfying two: the same kind and shape, the narrower bounds. */
function mergeContracts(a: InputContract | undefined, b: InputContract): InputContract {
  if (a === undefined) return b;
  if (a.kind !== b.kind || a.shape.join(",") !== b.shape.join(",") || a.shape.length !== b.shape.length)
    invalid(`tests disagree on the kind or shape of input '${a.name}'`);
  const lowers = [a.lower, b.lower].filter((v): v is number => v !== null);
  const uppers = [a.upper, b.upper].filter((v): v is number => v !== null);
  return new InputContract(a.name, a.kind, a.shape, lowers.length ? lowers.reduce(pyMax) : null,
    uppers.length ? uppers.reduce(pyMin) : null, a.unit);
}

/** Indices rejected by the e-BH procedure at level alpha, from log e-values (Wang and Ramdas 2022). */
export function eBH(logEValues: number[], alpha: number): number[] {
  checkAlpha(alpha);
  const count = logEValues.length;
  if (logEValues.some(Number.isNaN)) invalid("log e-values must not be NaN");
  const order = logEValues.map((_, i) => i).sort((i, j) => {
    const a = logEValues[i], b = logEValues[j];
    return a > b ? -1 : a < b ? 1 : i - j;
  });
  let best = 0;
  for (let k = 1; k <= count; k++)
    if (logEValues[order[k - 1]] >= Math.log(count) - Math.log(k) - Math.log(alpha)) best = k;
  return order.slice(0, best).sort((i, j) => i - j);
}

type MultipleTestsState = { n: number; states: State[]; log_e_values: number[] };

/** Several anytime-valid tests on the same observations, with e-BH false discovery rate control. */
export class MultipleTests extends BaseProcedure {
  readonly name = "multiple_tests";
  readonly version = "1";
  override readonly randomized: boolean;
  readonly names: string[];
  readonly tests: Procedure[];
  private readonly contracts: Record<string, InputContract>;
  private readonly fields: string[];

  constructor(tests: [string, Procedure][], readonly alpha = 0.05) {
    super();
    if (!tests.length) invalid("at least one test is needed");
    checkAlpha(alpha);
    // Python takes a mapping: a repeated name keeps its first position and its last test.
    const byName = new Map<string, Procedure>();
    for (const [name, test] of tests) byName.set(name, test);
    this.names = [...byName.keys()];
    this.tests = [...byName.values()];
    const contracts: Record<string, InputContract> = {};
    this.names.forEach((name, i) => {
      const test = this.tests[i];
      if (typeof (test as { eValueField?: unknown }).eValueField !== "string")
        invalid(`test '${name}' (${test.name}) does not report an e-process`);
      if (!test.jointInputs) invalid(`test '${name}' takes its inputs one at a time; run it separately`);
      for (const [key, contract] of Object.entries(test.inputs())) contracts[key] = mergeContracts(contracts[key], contract);
    });
    this.contracts = contracts;
    this.fields = this.tests.map((t) => (t as unknown as { eValueField: string }).eValueField);
    this.randomized = this.tests.some((t) => t.randomized);
  }

  static fromConfig(c: Config): MultipleTests {
    const raw = c.tests;
    if (!Array.isArray(raw)) invalid("config 'tests' must be a list of [name, identity] pairs");
    const tests = raw.map((pair): [string, Procedure] => {
      if (!Array.isArray(pair) || pair.length !== 2 || typeof pair[0] !== "string" || pair[1] === null || typeof pair[1] !== "object"
        || Array.isArray(pair[1]))
        return invalid("config 'tests' must be a list of [name, identity] pairs");
      return [pair[0], fromIdentity(pair[1] as unknown as Identity)];
    });
    return new MultipleTests(tests, num(c, "alpha"));
  }

  config(): Config {
    return { alpha: this.alpha, tests: this.names.map((k, i) => [k, identity(this.tests[i]) as unknown as Json]) };
  }

  inputs(): Record<string, InputContract> {
    return { ...this.contracts };
  }

  initialState(): MultipleTestsState {
    return { n: 0, states: this.tests.map((t) => t.initialState()), log_e_values: this.tests.map(() => 0.0) };
  }

  step(state: State, x: StatInput, rng: SplitMix64 | null): [MultipleTestsState, State] {
    const s = state as MultipleTestsState;
    const states: State[] = [], logs: number[] = [];
    const results: { [name: string]: Json } = {};
    this.tests.forEach((test, i) => {
      const name = this.names[i];
      if (test.isTerminal(s.states[i])) {
        states.push(s.states[i]);
        logs.push(s.log_e_values[i]);
        results[name] = null;
        return;
      }
      const sub: StatInput = {};
      for (const k of Object.keys(test.inputs())) sub[k] = x[k];
      const [next, out] = test.step(s.states[i], sub, rng);
      states.push(next);
      logs.push(out[this.fields[i]] as number);
      results[name] = out;
    });
    const rejected = eBH(logs, this.alpha).map((i) => this.names[i]);
    const n = s.n + 1;
    const logEValues: { [name: string]: Json } = {};
    this.names.forEach((name, i) => (logEValues[name] = logs[i]));
    return [{ n, states, log_e_values: logs }, { n, log_e_values: logEValues, rejected, results }];
  }

  override isTerminal(state: State): boolean {
    const s = state as MultipleTestsState;
    return this.tests.every((t, i) => t.isTerminal(s.states[i]));
  }

  decodeState(data: Json): MultipleTestsState {
    requireState(data !== null && typeof data === "object" && !Array.isArray(data)
      && Object.keys(data).sort().join(",") === "log_e_values,n,states", "fields");
    const d = data as { [key: string]: Json };
    const { n, states, log_e_values: logs } = d;
    requireState(typeof n === "number" && Number.isInteger(n) && n >= 0, "n");
    requireState(Array.isArray(states) && states.length === this.tests.length, "one state per test");
    requireState(Array.isArray(logs) && logs.length === this.tests.length && logs.every((v) => typeof v === "number" && !Number.isNaN(v)),
      "log e-values");
    return {
      n: n as number,
      states: this.tests.map((t, i) => t.decodeState((states as Json[])[i])),
      log_e_values: logs as number[],
    };
  }
}

register("multiple_tests", "1", (c) => MultipleTests.fromConfig(c));
