// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Group sequential test of a normal mean on a stream; mirrors procedures/group_sequential.py (version 4).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { ContractViolation, NumericalError } from "../errors.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, refuseTerminal, register, requireState, type State } from "../procedure.js";
import { erfc, logNormalUpper, tIsfLog } from "./distributions.js";
import { ACCEPT_H0, BaseProcedure, bool, type Config, type Decision, int, invalid, num, nums, optNum, pyMax, pyMin, REJECT_H0, str } from "./util.js";

const UNDERFLOW =
  "the sample variance underflows to zero although the observations differ; " +
  "rescale the data in the adapter (for example to units where they are of order one)";

type GSState = { n: number; mean: number; m2: number; analysis: number; decision: Decision; low: number | null; high: number | null };

const copysign = (x: number, sign: number): number => (sign < 0 || Object.is(sign, -0) ? -Math.abs(x) : Math.abs(x));

export class GroupSequentialTest extends BaseProcedure {
  readonly name = "group_sequential_test";
  readonly version = "4";
  readonly analyses: number[];
  readonly bounds: number[];
  readonly futility: number[] | null;
  private readonly convertedBounds: number[];
  private readonly convertedFutility: number[] | null;

  constructor(
    readonly sigma: number | null,
    analyses: number[],
    bounds: number[],
    readonly theta0 = 0.0,
    readonly twoSided = false,
    futility: number[] | null = null,
    readonly label = "",
    readonly input = "x",
  ) {
    super();
    if (sigma !== null && !(sigma > 0 && Number.isFinite(sigma))) invalid("sigma must be positive and finite, or None to estimate it");
    const ns = analyses.map((n) => Math.trunc(n));
    if (!ns.length || ns[0] < (sigma !== null ? 1 : 2) || ns.some((b, i) => i > 0 && b <= ns[i - 1]))
      invalid("analyses must be increasing positive sample sizes");
    if (bounds.length !== ns.length || bounds.some((b) => !(b > 0))) invalid("need one positive boundary per analysis");
    if (futility !== null) {
      if (twoSided) invalid("futility boundaries are for one-sided tests");
      if (futility.length !== ns.length || futility.some((f, i) => !(f <= bounds[i])))
        invalid("need one futility boundary per analysis, none above its efficacy boundary");
    }
    this.analyses = ns;
    this.bounds = [...bounds];
    this.futility = futility === null ? null : [...futility];
    this.convertedBounds = this.bounds.map((b, i) => this.converted(b, ns[i]));
    this.convertedFutility = this.futility === null ? null : this.futility.map((f, i) => this.converted(f, ns[i]));
  }

  static fromConfig(c: Config): GroupSequentialTest {
    const fut = c.futility === null ? null : nums(c, "futility");
    return new GroupSequentialTest(optNum(c, "sigma"), nums(c, "analyses"), nums(c, "bounds"), num(c, "theta0"), bool(c, "two_sided"),
      fut, str(c, "label"), str(c, "input"));
  }

  config(): Config {
    return {
      sigma: this.sigma, theta0: this.theta0, analyses: [...this.analyses], bounds: [...this.bounds], two_sided: this.twoSided,
      futility: this.futility === null ? null : [...this.futility], label: this.label, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): GSState {
    return { n: 0, mean: 0.0, m2: 0.0, analysis: 0, decision: null, low: null, high: null };
  }

  /** A boundary for Z, as a boundary for the statistic actually used at sample size n. */
  private converted(z: number, n: number): number {
    if (this.sigma !== null || !Number.isFinite(z)) return z;
    return copysign(tIsfLog(logNormalUpper(Math.abs(z)), n - 1), z);
  }

  boundAt(k: number): number {
    return this.convertedBounds[k];
  }

  step(state: State, x: StatInput): [GSState, State] {
    refuseTerminal(this, state);
    const s = state as GSState;
    const v = x[this.input] as number;
    const n = s.n + 1;
    const delta = v - s.mean;
    const mean = requireFinite(s.mean + delta / n, "running mean");
    const m2 = requireFinite(s.m2 + delta * (v - mean), "sum of squared deviations");
    const low = s.low === null ? v : pyMin(s.low, v);
    const high = s.high === null ? v : pyMax(s.high, v);
    const k = s.analysis;
    if (n !== this.analyses[k])
      return [{ n, mean, m2, analysis: k, decision: null, low, high }, { n, mean, analysis: null, z: null, bound: null, decision: null }];
    const scale = this.sigma !== null ? this.sigma : Math.sqrt(m2 / (n - 1));
    let z: number;
    if (scale > 0.0) z = ((mean - this.theta0) * Math.sqrt(n)) / scale;
    else if (low !== high) throw new NumericalError(UNDERFLOW);
    else z = mean === this.theta0 ? 0.0 : copysign(Infinity, mean - this.theta0);
    const bound = this.boundAt(k);
    const crossed = this.twoSided ? Math.abs(z) >= bound : z >= bound;
    const futile = this.convertedFutility !== null && z <= this.convertedFutility[k];
    const last = k + 1 === this.analyses.length;
    const decision = crossed ? REJECT_H0 : futile || last ? ACCEPT_H0 : null;
    return [
      { n, mean, m2, analysis: k + 1, decision, low, high },
      { n, mean, analysis: k + 1, z: Number.isFinite(z) ? z : null, bound: Number.isFinite(bound) ? bound : null, decision },
    ];
  }

  override isTerminal(state: State): boolean {
    return state.decision !== null;
  }

  decodeState(data: Json): GSState {
    const s = decodeFields({
      n: "count", mean: "number", m2: "number", analysis: "count", decision: "decision", low: "number?", high: "number?",
    }, data) as GSState;
    requireState(s.m2 >= 0, "negative sum of squares");
    return s;
  }
}

register("group_sequential_test", "4", (c) => GroupSequentialTest.fromConfig(c));

// ---------------------------------------------------------------- alpha spending (group_sequential.py)

const SQRT2 = Math.SQRT2;
const NEVER = 40.0; // a boundary no path reaches
const LOWER = -10.0;

/** 1 - Phi(x), through erfc. */
const upperTail = (x: number): number => 0.5 * erfc(x / SQRT2);

/** CPython's statistics.NormalDist().inv_cdf (Wichura's AS241), operation for operation. */
export function normalInvCdf(p: number): number {
  const q = p - 0.5;
  if (Math.abs(q) <= 0.425) {
    const r = 0.180625 - q * q;
    const num = (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r + 6.7265770927008700853e+4) * r +
      4.5921953931549871457e+4) * r + 1.3731693765509461125e+4) * r + 1.9715909503065514427e+3) * r +
      1.3314166789178437745e+2) * r + 3.3871328727963666080e+0) * q;
    const den = (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r + 3.9307895800092710610e+4) * r +
      2.1213794301586595867e+4) * r + 5.3941960214247511077e+3) * r + 6.8718700749205790830e+2) * r +
      4.2313330701600911252e+1) * r + 1.0);
    return 0.0 + (num / den) * 1.0;
  }
  let r = q <= 0.0 ? p : 1.0 - p;
  r = Math.sqrt(-Math.log(r));
  let num: number, den: number;
  if (r <= 5.0) {
    r = r - 1.6;
    num = (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r + 2.41780725177450611770e-1) * r +
      1.27045825245236838258e+0) * r + 3.64784832476320460504e+0) * r + 5.76949722146069140550e+0) * r +
      4.63033784615654529590e+0) * r + 1.42343711074968357734e+0);
    den = (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r + 1.51986665636164571966e-2) * r +
      1.48103976427480074590e-1) * r + 6.89767334985100004550e-1) * r + 1.67638483018380384940e+0) * r +
      2.05319162663775882187e+0) * r + 1.0);
  } else {
    r = r - 5.0;
    num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 1.24266094738807843860e-3) * r +
      2.65321895265761230930e-2) * r + 2.96560571828504891230e-1) * r + 1.78482653991729133580e+0) * r +
      5.46378491116411436990e+0) * r + 6.65790464350110377720e+0);
    den = (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r + 1.84631831751005468180e-5) * r +
      7.86869131145613259100e-4) * r + 1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r +
      5.99832206555887937690e-1) * r + 1.0);
  }
  let x = num / den;
  if (q < 0.0) x = -x;
  return 0.0 + x * 1.0;
}

/** A spending specification: "obrien-fleming", "pocock" or ["power", rho]. */
export function spendingFunction(spec: Json, alpha: number): (t: number) => number {
  if (spec === "obrien-fleming") {
    const z = -normalInvCdf(alpha / 2.0);
    return (t) => (t <= 0 ? 0.0 : 2.0 * upperTail(z / Math.sqrt(t)));
  }
  if (spec === "pocock") return (t) => alpha * Math.log(1.0 + (Math.E - 1.0) * t);
  if (Array.isArray(spec) && spec.length === 2 && spec[0] === "power" && typeof spec[1] === "number" && spec[1] > 0) {
    const rho = spec[1];
    return (t) => alpha * t ** rho;
  }
  return invalid(`unknown spending function ${JSON.stringify(spec)}`);
}

function simpson(count: number): number[] {
  const w: number[] = Array.from({ length: count }, (_, i) => (i % 2 === 0 ? 2.0 : 4.0));
  w[0] = w[count - 1] = 1.0;
  return w;
}

/** Sub-density of B(t_k) on the continuation region (group_sequential.py, _Recursion). */
class Recursion {
  t = 0.0;
  xs: number[] = [0.0];
  mass: number[] = [1.0];
  constructor(readonly drift: number, readonly twoSided: boolean, readonly points: number) {}

  private grid(lo: number, hi: number, sd: number): number[] {
    let m = pyMax(2, Math.ceil(((hi - lo) / sd) * this.points));
    m += m % 2;
    const h = (hi - lo) / m;
    return Array.from({ length: m + 1 }, (_, i) => lo + i * h);
  }

  cross(t: number, b: number): number {
    const sd = Math.sqrt(t - this.t);
    const mu = this.drift * (t - this.t);
    const upper = b * Math.sqrt(t);
    let p = 0.0;
    for (let i = 0; i < this.xs.length; i++) {
      const x = this.xs[i], m = this.mass[i];
      p += m * upperTail((upper - x - mu) / sd);
      if (this.twoSided) p += m * upperTail((upper + x + mu) / sd);
    }
    return p;
  }

  crossBelow(t: number, f: number): number {
    const sd = Math.sqrt(t - this.t);
    const mu = this.drift * (t - this.t);
    const lower = f * Math.sqrt(t);
    let p = 0.0;
    for (let i = 0; i < this.xs.length; i++) p += this.mass[i] * upperTail((this.xs[i] + mu - lower) / sd);
    return p;
  }

  advance(t: number, b: number, f: number | null = null): void {
    const sd = Math.sqrt(t - this.t);
    const mu = this.drift * (t - this.t);
    const upper = b * Math.sqrt(t);
    let lower: number;
    if (f !== null) lower = f * Math.sqrt(t);
    else if (this.twoSided) lower = -upper;
    else lower = pyMin(LOWER * Math.sqrt(t) + this.drift * t, upper - 1.0);
    const ys = this.grid(lower, upper, sd);
    const h = (ys[1] - ys[0]) / 3.0;
    const weights = simpson(ys.length);
    const norm = sd * Math.sqrt(2.0 * Math.PI);
    const mass: number[] = [];
    for (let j = 0; j < ys.length; j++) {
      let g = 0.0;
      for (let i = 0; i < this.xs.length; i++) {
        const z = (ys[j] - this.xs[i] - mu) / sd;
        g += this.mass[i] * Math.exp(-0.5 * z * z);
      }
      mass.push(h * weights[j] * (g / norm));
    }
    this.t = t;
    this.xs = ys;
    this.mass = mass;
  }
}

export function efficacyBounds(ts: number[], cumulative: number[], twoSided: boolean, points: number): number[] {
  const rec = new Recursion(0.0, twoSided, points);
  const bounds: number[] = [];
  let previous = 0.0;
  for (let k = 0; k < ts.length; k++) {
    const t = ts[k], c = cumulative[k];
    const target = (c - previous) * (twoSided ? 2.0 : 1.0);
    previous = c;
    let b: number;
    if (target <= 0) b = Infinity;
    else {
      let lo = 0.0, hi = 40.0;
      for (let i = 0; i < 80; i++) {
        const mid = 0.5 * (lo + hi);
        if (rec.cross(t, mid) > target) lo = mid;
        else hi = mid;
      }
      b = 0.5 * (lo + hi);
    }
    bounds.push(b);
    if (k + 1 < ts.length) rec.advance(t, Number.isFinite(b) ? b : NEVER);
  }
  return bounds;
}

export function futilityBounds(ts: number[], finite: number[], betaCumulative: number[], drift: number, points: number): [number[], number] {
  const rec = new Recursion(drift, false, points);
  const fs: number[] = [];
  let previous = 0.0;
  for (let k = 0; k < ts.length - 1; k++) {
    const t = ts[k], b = finite[k], c = betaCumulative[k];
    const target = c - previous;
    previous = c;
    let f: number;
    if (rec.crossBelow(t, b) <= target) f = b;
    else {
      let lo = -40.0, hi = b;
      for (let i = 0; i < 60; i++) {
        const mid = 0.5 * (lo + hi);
        if (rec.crossBelow(t, mid) > target) hi = mid;
        else lo = mid;
      }
      f = 0.5 * (lo + hi);
    }
    fs.push(f);
    rec.advance(t, b, f);
  }
  return [fs, previous + rec.crossBelow(ts[ts.length - 1], finite[finite.length - 1])];
}

type MonitorState = { fractions: number[]; bounds: number[]; futility: number[]; decision: Decision };

/** Alpha spending at the information actually reached; mirrors GroupSequentialMonitor (version 1). */
export class GroupSequentialMonitor extends BaseProcedure {
  readonly name = "group_sequential_monitor";
  readonly version = "1";
  private readonly side: number;
  private readonly alphaSpend: (t: number) => number;
  private readonly betaSpend: ((t: number) => number) | null;

  constructor(
    readonly alpha: number, readonly maxInformation: number, readonly spending: Json, readonly twoSided: boolean,
    readonly futility: Json, readonly power: number | null, readonly drift: number | null, readonly maxAnalyses: number,
    readonly pointsPerSd: number, readonly input: string, readonly informationInput: string, readonly finalInput: string | null,
  ) {
    super();
    if (!(alpha > 0 && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (!(Number.isFinite(maxInformation) && maxInformation > 0)) invalid("max_information must be positive and finite");
    this.side = twoSided ? alpha / 2.0 : alpha;
    this.alphaSpend = spendingFunction(spending, this.side);
    if (futility !== null && (power === null || drift === null)) invalid("futility boundaries need the power and the drift");
    this.betaSpend = futility === null ? null : spendingFunction(futility, 1.0 - (power as number));
  }

  static fromConfig(c: Config): GroupSequentialMonitor {
    const finalInput = c.final_input === null ? null : str(c, "final_input");
    return new GroupSequentialMonitor(num(c, "alpha"), num(c, "max_information"), c.spending, bool(c, "two_sided"), c.futility,
      optNum(c, "power"), optNum(c, "drift"), int(c, "max_analyses"), int(c, "points_per_sd"), str(c, "input"),
      str(c, "information_input"), finalInput);
  }

  config(): Config {
    return {
      alpha: this.alpha, max_information: this.maxInformation, spending: this.spending, two_sided: this.twoSided,
      futility: this.futility, power: this.power, drift: this.drift, max_analyses: this.maxAnalyses,
      points_per_sd: this.pointsPerSd, input: this.input, information_input: this.informationInput, final_input: this.finalInput,
    };
  }

  inputs(): Record<string, InputContract> {
    const contracts: Record<string, InputContract> = {
      [this.input]: new InputContract(this.input),
      [this.informationInput]: new InputContract(this.informationInput, "real", [], 0.0),
    };
    if (this.finalInput !== null) contracts[this.finalInput] = new InputContract(this.finalInput, "binary");
    return contracts;
  }

  initialState(): MonitorState {
    return { fractions: [], bounds: [], futility: [], decision: null };
  }

  boundaries(ts: number[], final: boolean): [number[], number[]] {
    const cumulative = ts.map((u) => this.alphaSpend(pyMin(u, 1.0)));
    if (final) cumulative[cumulative.length - 1] = this.side;
    const bounds = efficacyBounds(ts, cumulative, this.twoSided, this.pointsPerSd);
    if (this.betaSpend === null) return [bounds, []];
    const finite = bounds.map((b) => (Number.isFinite(b) ? b : NEVER));
    const betaCumulative = ts.map((u) => (this.betaSpend as (t: number) => number)(pyMin(u, 1.0)));
    const drift = this.drift as number;
    if (final) {
      betaCumulative[betaCumulative.length - 1] = 1.0 - (this.power as number);
      const [fs] = futilityBounds(ts, finite, betaCumulative, drift, this.pointsPerSd);
      return [bounds, [...fs, finite[finite.length - 1]]];
    }
    const [fs] = futilityBounds([...ts, ts[ts.length - 1] + 1.0], [...finite, NEVER],
      [...betaCumulative, betaCumulative[betaCumulative.length - 1]], drift, this.pointsPerSd);
    return [bounds, fs];
  }

  step(state: State, x: StatInput): [MonitorState, State] {
    refuseTerminal(this, state);
    const s = state as MonitorState;
    const z = x[this.input] as number, information = x[this.informationInput] as number;
    const t = information / this.maxInformation;
    const previous = s.fractions.length ? s.fractions[s.fractions.length - 1] : 0.0;
    if (!(t > previous)) throw new ContractViolation(`${this.name}: information must grow from one analysis to the next`);
    const k = s.fractions.length;
    const final = t >= 1.0 || k + 1 === this.maxAnalyses || (this.finalInput !== null && x[this.finalInput] === 1);
    const fractions = [...s.fractions, t];
    const [bounds, fs] = this.boundaries(fractions, final);
    const b = bounds[bounds.length - 1];
    const f = fs.length ? fs[fs.length - 1] : null;
    const spent = final ? this.side : this.alphaSpend(t);
    const crossed = this.twoSided ? Math.abs(z) >= b : z >= b;
    const futile = f !== null && z <= f;
    const decision: Decision = crossed ? REJECT_H0 : futile || final ? ACCEPT_H0 : null;
    const next: MonitorState = {
      fractions, bounds: [...s.bounds, Number.isFinite(b) ? b : NEVER], futility: f !== null ? [...s.futility, f] : s.futility, decision,
    };
    return [next, { analysis: k + 1, fraction: t, z, bound: Number.isFinite(b) ? b : null, futility: f, spent, final, decision }];
  }

  override isTerminal(state: State): boolean {
    return state.decision !== null;
  }

  decodeState(data: Json): MonitorState {
    const s = decodeFields({ fractions: "numbers", bounds: "numbers", futility: "numbers", decision: "decision" }, data) as MonitorState;
    const k = s.fractions.length;
    requireState(s.bounds.length === k && (s.futility.length === 0 || s.futility.length === k), "one boundary per analysis");
    requireState(s.fractions.every((v, i) => v > (i === 0 ? 0.0 : s.fractions[i - 1])), "fractions must grow");
    requireState(k <= this.maxAnalyses, "more analyses than max_analyses");
    return s;
  }
}

register("group_sequential_monitor", "1", (c) => GroupSequentialMonitor.fromConfig(c));

// ---------------------------------------------------------------- two arms: proportions and log-rank

abstract class TwoArm extends BaseProcedure {
  readonly analyses: number[];
  readonly bounds: number[];
  readonly futility: number[] | null;
  readonly arms: [number, number];

  constructor(c: Config) {
    super();
    const ns = nums(c, "analyses");
    if (!ns.length || ns[0] < 1 || ns.some((b, i) => i > 0 && b <= ns[i - 1])) invalid("analyses must be increasing positive counts");
    const bounds = nums(c, "bounds");
    if (bounds.length !== ns.length || bounds.some((b) => !(b > 0))) invalid("need one positive boundary per analysis");
    this.analyses = ns;
    this.bounds = bounds.map((b) => pyMin(b, NEVER));
    this.futility = c.futility === null ? null : nums(c, "futility");
    const arms = nums(c, "arms");
    this.arms = [arms[0], arms[1]];
  }

  protected common(c: Config): Config {
    return {
      analyses: [...this.analyses], bounds: [...this.bounds], two_sided: c.two_sided, futility: this.futility === null ? null : [...this.futility],
      alternative: c.alternative, arms: [...this.arms], arm_input: c.arm_input, input: c.input, label: c.label,
    };
  }

  protected decide(k: number, z: number, twoSided: boolean): Decision {
    const b = this.bounds[k];
    const crossed = twoSided ? Math.abs(z) >= b : z >= b;
    const futile = this.futility !== null && z <= this.futility[k];
    return crossed ? REJECT_H0 : futile || k + 1 === this.analyses.length ? ACCEPT_H0 : null;
  }

  override isTerminal(state: State): boolean {
    return state.decision !== null;
  }
}

type PropState = {
  n_treatment: number; successes_treatment: number; n_control: number; successes_control: number; analysis: number; decision: Decision;
};

export class GroupSequentialProportions extends TwoArm {
  readonly name = "group_sequential_proportions";
  readonly version = "1";
  private readonly c: Config;

  constructor(c: Config) {
    super(c);
    this.c = c;
  }

  config(): Config {
    return this.common(this.c);
  }

  inputs(): Record<string, InputContract> {
    const arm = str(this.c, "arm_input"), y = str(this.c, "input");
    return { [arm]: new InputContract(arm, "integer", [], 0), [y]: new InputContract(y, "binary") };
  }

  initialState(): PropState {
    return { n_treatment: 0, successes_treatment: 0, n_control: 0, successes_control: 0, analysis: 0, decision: null };
  }

  statistic(s: PropState): number {
    const nT = s.n_treatment, nC = s.n_control;
    if (nT === 0 || nC === 0) return 0.0;
    const pooled = (s.successes_treatment + s.successes_control) / (nT + nC);
    if (pooled <= 0.0 || pooled >= 1.0) return 0.0;
    const diff = s.successes_treatment / nT - s.successes_control / nC;
    const z = diff / Math.sqrt(pooled * (1.0 - pooled) * (1.0 / nT + 1.0 / nC));
    return this.c.alternative === "less" ? -z : z;
  }

  private rates(s: PropState): [number | null, number | null] {
    return [s.n_treatment ? s.successes_treatment / s.n_treatment : null, s.n_control ? s.successes_control / s.n_control : null];
  }

  step(state: State, x: StatInput): [PropState, State] {
    refuseTerminal(this, state);
    let s = state as PropState;
    const label = x[str(this.c, "arm_input")] as number;
    const out = (st: PropState, analysis: number | null, z: number | null, bound: number | null, decision: Decision) => {
      const [rt, rc] = this.rates(st);
      return { n: st.n_treatment + st.n_control, rate_treatment: rt, rate_control: rc, analysis, z, bound, decision };
    };
    if (label !== this.arms[0] && label !== this.arms[1]) return [s, out(s, null, null, null, null)];
    const y = x[str(this.c, "input")] as number;
    s = label === this.arms[0]
      ? { ...s, n_treatment: s.n_treatment + 1, successes_treatment: s.successes_treatment + y, decision: null }
      : { ...s, n_control: s.n_control + 1, successes_control: s.successes_control + y, decision: null };
    const n = s.n_treatment + s.n_control, k = s.analysis;
    if (n !== this.analyses[k]) return [s, out(s, null, null, null, null)];
    const z = this.statistic(s);
    const decision = this.decide(k, z, bool(this.c, "two_sided"));
    const next = { ...s, analysis: k + 1, decision };
    return [next, out(next, k + 1, z, this.bounds[k], decision)];
  }

  decodeState(data: Json): PropState {
    const s = decodeFields({
      n_treatment: "count", successes_treatment: "count", n_control: "count", successes_control: "count", analysis: "count", decision: "decision",
    }, data) as PropState;
    requireState(s.successes_treatment <= s.n_treatment && s.successes_control <= s.n_control, "more successes than units");
    requireState(s.analysis <= this.analyses.length, "more analyses than planned");
    return s;
  }
}

register("group_sequential_proportions", "1", (c) => new GroupSequentialProportions(c));

type LRState = { at_risk: number[]; events: number; observed: number; expected: number; variance: number; analysis: number; decision: Decision };

export class GroupSequentialLogRank extends TwoArm {
  readonly name = "group_sequential_logrank";
  readonly version = "1";
  private readonly c: Config;
  readonly strata: number;

  constructor(c: Config) {
    super(c);
    this.c = c;
    this.strata = int(c, "strata");
  }

  config(): Config {
    return { ...this.common(this.c), strata: this.strata, stratum_input: this.c.stratum_input };
  }

  inputs(): Record<string, InputContract> {
    const arm = str(this.c, "arm_input"), kind = str(this.c, "input");
    const contracts: Record<string, InputContract> = {
      [arm]: new InputContract(arm, "integer", [], 0), [kind]: new InputContract(kind, "integer", [], 0, 2),
    };
    if (this.strata > 1) {
      const name = str(this.c, "stratum_input");
      contracts[name] = new InputContract(name, "integer", [], 0, this.strata - 1);
    }
    return contracts;
  }

  initialState(): LRState {
    return { at_risk: Array.from({ length: 2 * this.strata }, () => 0), events: 0, observed: 0, expected: 0.0, variance: 0.0, analysis: 0, decision: null };
  }

  statistic(s: LRState): number {
    if (s.variance <= 0.0) return 0.0;
    const z = (s.expected - s.observed) / Math.sqrt(s.variance);
    return this.c.alternative === "greater" ? -z : z;
  }

  private output(s: LRState, analysis: number | null, z: number | null, bound: number | null, decision: Decision): State {
    let t = 0, c = 0;
    for (let i = 0; i < s.at_risk.length; i += 2) {
      t += s.at_risk[i];
      c += s.at_risk[i + 1];
    }
    const estimate = s.variance > 0 ? Math.exp((s.observed - s.expected) / s.variance) : null;
    return {
      events: s.events, at_risk: [t, c], analysis, z, information: analysis !== null ? s.variance : null, hazard_ratio_estimate: estimate,
      bound, decision,
    };
  }

  step(state: State, x: StatInput): [LRState, State] {
    refuseTerminal(this, state);
    const s = state as LRState;
    const label = x[str(this.c, "arm_input")] as number;
    if (label !== this.arms[0] && label !== this.arms[1]) return [s, this.output(s, null, null, null, null)];
    const arm = label === this.arms[0] ? 0 : 1;
    const stratum = this.strata > 1 ? Math.trunc(x[str(this.c, "stratum_input")] as number) : 0;
    const i = 2 * stratum + arm;
    const kind = x[str(this.c, "input")] as number;
    const atRisk = [...s.at_risk];
    if (kind === 0) {
      atRisk[i] += 1;
      const next = { ...s, at_risk: atRisk, decision: null };
      return [next, this.output(next, null, null, null, null)];
    }
    if (atRisk[i] === 0) {
      const what = kind === 1 ? "an event" : "a censoring";
      const where = this.strata > 1 ? ` of stratum ${stratum}` : "";
      throw new ContractViolation(`${this.name}: ${what} in arm ${label}${where} with nobody at risk`);
    }
    const nT = atRisk[2 * stratum], nC = atRisk[2 * stratum + 1];
    atRisk[i] -= 1;
    if (kind === 2) {
      const next = { ...s, at_risk: atRisk, decision: null };
      return [next, this.output(next, null, null, null, null)];
    }
    const p = nT / (nT + nC);
    const next: LRState = {
      at_risk: atRisk, events: s.events + 1, observed: s.observed + (arm === 0 ? 1 : 0), expected: s.expected + p,
      variance: s.variance + p * (1.0 - p), analysis: s.analysis, decision: null,
    };
    const k = next.analysis;
    if (next.events !== this.analyses[k]) return [next, this.output(next, null, null, null, null)];
    const z = this.statistic(next);
    const decision = this.decide(k, z, bool(this.c, "two_sided"));
    const done = { ...next, analysis: k + 1, decision };
    return [done, this.output(done, k + 1, z, this.bounds[k], decision)];
  }

  decodeState(data: Json): LRState {
    const s = decodeFields({
      at_risk: "counts", events: "count", observed: "count", expected: "number", variance: "number", analysis: "count", decision: "decision",
    }, data) as LRState;
    requireState(s.at_risk.length === 2 * this.strata, "one pair of numbers at risk per stratum");
    requireState(s.observed <= s.events && s.expected >= 0.0 && s.expected <= s.events, "counts");
    requireState(s.variance >= 0.0 && s.variance <= 0.25 * s.events + 1e-9, "variance");
    requireState(s.analysis <= this.analyses.length, "more analyses than planned");
    return s;
  }
}

register("group_sequential_logrank", "1", (c) => new GroupSequentialLogRank(c));

type MeansState = {
  n_treatment: number; mean_treatment: number; m2_treatment: number; n_control: number; mean_control: number; m2_control: number;
  analysis: number; decision: Decision;
};

/** Two means, known or estimated variance; mirrors GroupSequentialTwoMeans (version 1). */
export class GroupSequentialTwoMeans extends TwoArm {
  readonly name = "group_sequential_two_means";
  readonly version = "1";
  private readonly c: Config;
  readonly sigma: number | null;
  private readonly convertedBounds: number[];
  private readonly convertedFutility: number[] | null;

  constructor(c: Config) {
    super(c);
    this.c = c;
    this.sigma = optNum(c, "sigma");
    if (this.sigma !== null && !(Number.isFinite(this.sigma) && this.sigma > 0)) invalid("sigma must be positive and finite, or None");
    if (this.sigma === null && this.analyses[0] < 3) invalid("with an estimated variance the first analysis needs at least three units");
    this.convertedBounds = this.bounds.map((b, i) => this.converted(b, this.analyses[i]));
    this.convertedFutility = this.futility === null ? null : this.futility.map((f, i) => this.converted(f, this.analyses[i]));
  }

  private converted(z: number, n: number): number {
    if (this.sigma !== null || z >= NEVER) return z;
    return copysign(tIsfLog(logNormalUpper(Math.abs(z)), n - 2), z);
  }

  boundAt(k: number): number {
    return this.convertedBounds[k];
  }

  config(): Config {
    return { ...this.common(this.c), sigma: this.sigma };
  }

  inputs(): Record<string, InputContract> {
    const arm = str(this.c, "arm_input"), y = str(this.c, "input");
    return { [arm]: new InputContract(arm, "integer", [], 0), [y]: new InputContract(y) };
  }

  initialState(): MeansState {
    return { n_treatment: 0, mean_treatment: 0.0, m2_treatment: 0.0, n_control: 0, mean_control: 0.0, m2_control: 0.0, analysis: 0, decision: null };
  }

  statistic(s: MeansState): number {
    const nT = s.n_treatment, nC = s.n_control;
    if (nT === 0 || nC === 0) return 0.0;
    const diff = s.mean_treatment - s.mean_control;
    let scale: number;
    if (this.sigma !== null) scale = this.sigma;
    else {
      if (nT + nC < 3) return 0.0;
      scale = Math.sqrt((s.m2_treatment + s.m2_control) / (nT + nC - 2));
      if (scale === 0.0) return diff === 0.0 ? 0.0 : copysign(Infinity, this.c.alternative === "greater" ? diff : -diff);
    }
    const z = diff / (scale * Math.sqrt(1.0 / nT + 1.0 / nC));
    return this.c.alternative === "less" ? -z : z;
  }

  step(state: State, x: StatInput): [MeansState, State] {
    refuseTerminal(this, state);
    let s = state as MeansState;
    const label = x[str(this.c, "arm_input")] as number;
    const known = label === this.arms[0] || label === this.arms[1];
    if (known) {
      const y = x[str(this.c, "input")] as number;
      if (label === this.arms[0]) {
        const n = s.n_treatment + 1;
        const delta = y - s.mean_treatment;
        const mean = requireFinite(s.mean_treatment + delta / n, "running mean");
        const m2 = requireFinite(s.m2_treatment + delta * (y - mean), "sum of squared deviations");
        s = { ...s, n_treatment: n, mean_treatment: mean, m2_treatment: m2, decision: null };
      } else {
        const n = s.n_control + 1;
        const delta = y - s.mean_control;
        const mean = requireFinite(s.mean_control + delta / n, "running mean");
        const m2 = requireFinite(s.m2_control + delta * (y - mean), "sum of squared deviations");
        s = { ...s, n_control: n, mean_control: mean, m2_control: m2, decision: null };
      }
    }
    const total = s.n_treatment + s.n_control;
    const meanT = s.n_treatment ? s.mean_treatment : null, meanC = s.n_control ? s.mean_control : null;
    const k = s.analysis;
    if (!known || total !== this.analyses[k])
      return [s, { n: total, mean_treatment: meanT, mean_control: meanC, analysis: null, z: null, bound: null, decision: null }];
    const z = this.statistic(s);
    const b = this.convertedBounds[k];
    const crossed = bool(this.c, "two_sided") ? Math.abs(z) >= b : z >= b;
    const futile = this.convertedFutility !== null && z <= this.convertedFutility[k];
    const decision: Decision = crossed ? REJECT_H0 : futile || k + 1 === this.analyses.length ? ACCEPT_H0 : null;
    const next = { ...s, analysis: k + 1, decision };
    return [next, {
      n: total, mean_treatment: meanT, mean_control: meanC, analysis: k + 1, z: Number.isFinite(z) ? z : null, bound: b, decision,
    }];
  }

  decodeState(data: Json): MeansState {
    const s = decodeFields({
      n_treatment: "count", mean_treatment: "number", m2_treatment: "number", n_control: "count", mean_control: "number",
      m2_control: "number", analysis: "count", decision: "decision",
    }, data) as MeansState;
    requireState(s.m2_treatment >= 0 && s.m2_control >= 0, "negative sum of squares");
    requireState(s.analysis <= this.analyses.length, "more analyses than planned");
    return s;
  }
}

register("group_sequential_two_means", "1", (c) => new GroupSequentialTwoMeans(c));
