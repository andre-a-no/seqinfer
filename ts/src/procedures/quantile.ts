// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Anytime-valid inference for a quantile (Howard and Ramdas 2022); mirrors procedures/quantile.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { lgamma, regularizedBeta } from "./distributions.js";
import { BaseProcedure, bool, type Config, type Decision, invalid, num, pyMax, pyMin, REJECT_H0, str } from "./util.js";

const ALTERNATIVES = ["two-sided", "greater", "less"];

const logBeta = (a: number, b: number): number => lgamma(a) + lgamma(b) - lgamma(a + b);

/** The one-sided beta-binomial mixtures for a hypothesised probability p (Python's _Mixture). */
class Mixture {
  readonly r: number;
  readonly a: number;
  readonly b: number;
  readonly logLevel: number;
  private readonly logMassBelow: number;
  private readonly logMassAbove: number;
  private readonly norm: number;

  constructor(readonly p: number, alpha: number, tunedFor: number, twoSided: boolean) {
    const level = twoSided ? alpha / 2.0 : alpha;
    const c = 2.0 * Math.log(1.0 / level);
    this.r = pyMax(1.0, tunedFor / (c + Math.log1p(c)) - 1.0);
    this.a = this.r * p;
    this.b = this.r * (1.0 - p);
    this.logLevel = Math.log(1.0 / level);
    this.logMassBelow = Math.log(regularizedBeta(p, this.a, this.b));
    this.logMassAbove = Math.log(regularizedBeta(1.0 - p, this.b, this.a));
    this.norm = logBeta(this.a, this.b);
  }

  /** log M_n(k), mixing over theta < p: evidence that F(q) < p, from k = #{X_i <= q}. */
  logBelow(k: number, n: number): number {
    const mass = regularizedBeta(this.p, this.a + k, this.b + n - k);
    if (mass <= 0.0) return -Infinity;
    return logBeta(this.a + k, this.b + n - k) + Math.log(mass) - this.norm - this.logMassBelow
      - k * Math.log(this.p) - (n - k) * Math.log1p(-this.p);
  }

  /** The mirror image, mixing over theta > p: evidence that F(q-) > p, from k = #{X_i < q}. */
  logAbove(k: number, n: number): number {
    const mass = regularizedBeta(1.0 - this.p, this.b + n - k, this.a + k);
    if (mass <= 0.0) return -Infinity;
    return logBeta(this.a + k, this.b + n - k) + Math.log(mass) - this.norm - this.logMassAbove
      - k * Math.log(this.p) - (n - k) * Math.log1p(-this.p);
  }

  smallestKept(n: number): number {
    let lo = 0, hi = n;
    if (this.logBelow(0, n) < this.logLevel) return 0;
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (this.logBelow(mid, n) >= this.logLevel) lo = mid;
      else hi = mid;
    }
    return this.logBelow(hi, n) < this.logLevel ? hi : n + 1;
  }

  largestKept(n: number): number {
    let lo = 0, hi = n;
    if (this.logAbove(n, n) < this.logLevel) return n;
    if (this.logAbove(0, n) >= this.logLevel) return -1;
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (this.logAbove(mid, n) < this.logLevel) lo = mid;
      else hi = mid;
    }
    return lo;
  }
}

function check(p: number, alpha: number, tunedFor: number): void {
  if (!(0 < p && p < 1)) invalid("p must lie strictly between 0 and 1");
  if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
  if (!(Number.isFinite(tunedFor) && tunedFor >= 1)) invalid("tuned_for must be a number of observations, at least 1");
}

// ---------------------------------------------------------------- test

type QuantileTestState = { n: number; below: number; at_or_below: number; max_log_evidence: number; decision: Decision };

export class QuantileTest extends BaseProcedure {
  readonly name = "quantile_test";
  readonly version = "1";
  readonly eValueField = "log_evidence";
  private readonly mix: Mixture;

  constructor(
    readonly q0: number,
    readonly p = 0.5,
    readonly alpha = 0.05,
    readonly alternative = "two-sided",
    readonly tunedFor = 1000.0,
    readonly stopOnReject = true,
    readonly input = "x",
  ) {
    super();
    check(p, alpha, tunedFor);
    if (!Number.isFinite(q0)) invalid("q0 must be finite");
    if (!ALTERNATIVES.includes(alternative)) invalid(`alternative must be one of ${ALTERNATIVES}`);
    this.mix = new Mixture(p, alpha, tunedFor, alternative === "two-sided");
  }

  static fromConfig(c: Config): QuantileTest {
    return new QuantileTest(num(c, "q0"), num(c, "p"), num(c, "alpha"), str(c, "alternative"), num(c, "tuned_for"),
      bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return {
      q0: this.q0, p: this.p, alpha: this.alpha, alternative: this.alternative, tuned_for: this.tunedFor,
      stop_on_reject: this.stopOnReject, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): QuantileTestState {
    return { n: 0, below: 0, at_or_below: 0, max_log_evidence: 0.0, decision: null };
  }

  logEvidence(n: number, below: number, atOrBelow: number): number {
    if (n === 0) return 0.0;
    const sides: number[] = [];
    if (this.alternative !== "less") sides.push(this.mix.logBelow(atOrBelow, n));
    if (this.alternative !== "greater") sides.push(this.mix.logAbove(below, n));
    const shift = this.alternative === "two-sided" ? Math.log(2.0) : 0.0;
    return sides.reduce(pyMax) - shift;
  }

  step(state: State, x: StatInput): [QuantileTestState, State] {
    const s = state as QuantileTestState;
    const v = x[this.input] as number;
    const n = s.n + 1;
    const below = s.below + (v < this.q0 ? 1 : 0);
    const atOrBelow = s.at_or_below + (v <= this.q0 ? 1 : 0);
    const evidence = this.logEvidence(n, below, atOrBelow);
    const best = pyMax(s.max_log_evidence, evidence);
    const decision = s.decision || (evidence >= Math.log(1.0 / this.alpha) ? REJECT_H0 : null);
    return [
      { n, below, at_or_below: atOrBelow, max_log_evidence: best, decision },
      { n, fraction_at_or_below: atOrBelow / n, log_evidence: evidence, p_value: pyMin(1.0, Math.exp(-best)), decision },
    ];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): QuantileTestState {
    const s = decodeFields({ n: "count", below: "count", at_or_below: "count", max_log_evidence: "number", decision: "decision" },
      data) as QuantileTestState;
    requireState(s.below <= s.at_or_below && s.at_or_below <= s.n, "counts out of order");
    return s;
  }
}

// ---------------------------------------------------------------- confidence sequence

type QuantileCSState = { n: number; sample: number[]; lower: number | null; upper: number | null };

/** Python's bisect.insort: insert after any equal elements. */
function insort(sample: number[], v: number): void {
  let lo = 0, hi = sample.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (v < sample[mid]) hi = mid;
    else lo = mid + 1;
  }
  sample.splice(lo, 0, v);
}

export class QuantileConfidenceSequence extends BaseProcedure {
  readonly name = "quantile_confidence_sequence";
  readonly version = "1";
  private readonly mix: Mixture;

  constructor(readonly p = 0.5, readonly alpha = 0.05, readonly tunedFor = 1000.0, readonly input = "x") {
    super();
    check(p, alpha, tunedFor);
    this.mix = new Mixture(p, alpha, tunedFor, true);
  }

  static fromConfig(c: Config): QuantileConfidenceSequence {
    return new QuantileConfidenceSequence(num(c, "p"), num(c, "alpha"), num(c, "tuned_for"), str(c, "input"));
  }

  config(): Config {
    return { p: this.p, alpha: this.alpha, tuned_for: this.tunedFor, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): QuantileCSState {
    return { n: 0, sample: [], lower: null, upper: null };
  }

  bounds(sample: number[]): [number, number] {
    const n = sample.length;
    const kLo = this.mix.smallestKept(n);
    const kHi = this.mix.largestKept(n);
    const lower = kLo <= 0 ? -Infinity : kLo <= n ? sample[kLo - 1] : Infinity;
    const upper = kHi + 1 > n ? Infinity : kHi >= 0 ? sample[kHi] : -Infinity;
    return [lower, upper];
  }

  step(state: State, x: StatInput): [QuantileCSState, State] {
    const s = state as QuantileCSState;
    const v = x[this.input] as number;
    const sample = [...s.sample];
    insort(sample, v);
    const n = s.n + 1;
    let [lo, hi] = this.bounds(sample);
    const oldLo = s.lower === null ? -Infinity : s.lower;
    const oldHi = s.upper === null ? Infinity : s.upper;
    lo = pyMax(lo, oldLo);
    hi = pyMin(hi, oldHi);
    const lower = Number.isFinite(lo) ? lo : null;
    const upper = Number.isFinite(hi) ? hi : null;
    const k = pyMax(1, Math.ceil(this.p * n));
    return [{ n, sample, lower, upper }, { n, estimate: sample[k - 1], lower, upper }];
  }

  decodeState(data: Json): QuantileCSState {
    const s = decodeFields({ n: "count", sample: "numbers", lower: "number?", upper: "number?" }, data) as QuantileCSState;
    requireState(s.sample.length === s.n, "sample size differs from n");
    requireState(s.sample.every((v, i) => i === 0 || s.sample[i - 1] <= v), "sample not sorted");
    return s;
  }
}

register("quantile_test", "1", (c) => QuantileTest.fromConfig(c));
register("quantile_confidence_sequence", "1", (c) => QuantileConfidenceSequence.fromConfig(c));
