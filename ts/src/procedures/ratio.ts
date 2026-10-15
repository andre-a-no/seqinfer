// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Anytime-valid A/B test of a ratio metric and of relative lift; mirrors procedures/ratio.py (RatioABTest).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { ContractViolation } from "../errors.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { AsymptoticConfidenceSequence } from "./robust_mean.js";
import { BaseProcedure, bool, type Config, type Decision, int, invalid, num, nums, pyMax, REJECT_H0, str } from "./util.js";

const SCALES = ["difference", "relative"];

type RatioABState = {
  units: number[]; sum_y: number[]; sum_d: number[]; cyy: number[]; cdd: number[]; cyd: number[]; decision: Decision;
};

/** The pair with its k-th element replaced. */
const put = (pair: number[], k: number, value: number): number[] => (k === 0 ? [value, pair[1]] : [pair[0], value]);

/** Anytime-valid confidence sequences for a ratio metric's difference and relative lift (Python's RatioABTest). */
export class RatioABTest extends BaseProcedure {
  readonly name = "ratio_ab_test";
  readonly version = "1";
  readonly arms: [number, number];
  private readonly cs: AsymptoticConfidenceSequence;

  constructor(
    readonly alpha = 0.05,
    readonly denominator = true,
    readonly scale = "difference",
    readonly effect0 = 0.0,
    arms: number[] = [1, 0],
    readonly start = 1000,
    readonly tunedFor = 1000.0,
    readonly stopOnReject = false,
    readonly armInput = "arm",
    readonly input = "y",
    readonly denominatorInput = "d",
  ) {
    super();
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (!SCALES.includes(scale)) invalid(`scale must be one of ${SCALES}`);
    if (!Number.isFinite(effect0) || (scale === "relative" && effect0 <= -1.0)) invalid("effect0 must be finite, and above -1 for a relative lift");
    if (arms.length !== 2 || arms.some((a) => !Number.isInteger(a) || a < 0) || arms[0] === arms[1])
      invalid("arms must be two different non-negative integer labels");
    this.arms = [arms[0], arms[1]];
    const names = new Set(denominator ? [armInput, input, denominatorInput] : [armInput, input]);
    if (names.size !== (denominator ? 3 : 2)) invalid("the arm, numerator and denominator need different names");
    // the boundary is AsymptoticConfidenceSequence's (it checks start and tuned_for)
    this.cs = new AsymptoticConfidenceSequence(alpha, null, start, tunedFor);
  }

  static fromConfig(c: Config): RatioABTest {
    return new RatioABTest(num(c, "alpha"), bool(c, "denominator"), str(c, "scale"), num(c, "effect0"), nums(c, "arms"), int(c, "start"),
      num(c, "tuned_for"), bool(c, "stop_on_reject"), str(c, "arm_input"), str(c, "input"), str(c, "denominator_input"));
  }

  config(): Config {
    return {
      alpha: this.alpha, denominator: this.denominator, scale: this.scale, effect0: this.effect0, arms: [...this.arms], start: this.start,
      tuned_for: this.tunedFor, stop_on_reject: this.stopOnReject, arm_input: this.armInput, input: this.input,
      denominator_input: this.denominatorInput,
    };
  }

  inputs(): Record<string, InputContract> {
    const out: Record<string, InputContract> = {
      [this.armInput]: new InputContract(this.armInput, "integer", [], 0),
      [this.input]: new InputContract(this.input),
    };
    if (this.denominator) out[this.denominatorInput] = new InputContract(this.denominatorInput, "real", [], 0.0);
    return out;
  }

  initialState(): RatioABState {
    return { units: [0, 0], sum_y: [0.0, 0.0], sum_d: [0.0, 0.0], cyy: [0.0, 0.0], cdd: [0.0, 0.0], cyd: [0.0, 0.0], decision: null };
  }

  /** The asymptotic confidence sequence's half-width for scores of standard deviation sd. */
  halfWidth(n: number, sd: number): number {
    return this.cs.halfWidth(n, sd);
  }

  /** The ratio of sums of arm k (0: treatment, 1: control). */
  static ratio(s: RatioABState, k: number): number | null {
    return s.sum_d[k] > 0.0 ? s.sum_y[k] / s.sum_d[k] : null;
  }

  /** V_k, the delta-method variance of arm k's ratio of sums; null below two units or while sum d = 0. */
  static variance(s: RatioABState, k: number): number | null {
    const n = s.units[k], sd = s.sum_d[k];
    if (n < 2 || !(sd > 0.0)) return null;
    const r = s.sum_y[k] / sd;
    const q = pyMax(0.0, s.cyy[k] + r * (r * s.cdd[k] - 2.0 * s.cyd[k])); // sum (y - r d)^2
    return (q * n) / ((n - 1) * sd * sd);
  }

  private output(s: RatioABState): State {
    const n = s.units[0] + s.units[1];
    const rt = RatioABTest.ratio(s, 0), rc = RatioABTest.ratio(s, 1);
    const vt = RatioABTest.variance(s, 0), vc = RatioABTest.variance(s, 1);
    let difference: number | null = null, stdError: number | null = null, lower: number | null = null, upper: number | null = null;
    let lift: number | null = null, liftLower: number | null = null, liftUpper: number | null = null;
    if (rt !== null && rc !== null) {
      difference = rt - rc;
      if (rt > 0.0 && rc > 0.0) lift = rt / rc - 1.0;
    }
    if (vt !== null && vc !== null && rt !== null && rc !== null) {
      const v = vt + vc;
      stdError = Math.sqrt(v);
      if (n >= this.start && v > 0.0) {
        const h = this.halfWidth(n, Math.sqrt(n * v));
        [lower, upper] = [rt - rc - h, rt - rc + h];
      }
      if (n >= this.start && lift !== null) {
        const vLog = vt / (rt * rt) + vc / (rc * rc);
        if (vLog > 0.0) {
          const logRatio = Math.log(rt / rc);
          const h = this.halfWidth(n, Math.sqrt(n * vLog));
          [liftLower, liftUpper] = [Math.expm1(logRatio - h), Math.expm1(logRatio + h)];
        }
      }
    }
    return {
      n, units: [...s.units], ratios: [rt, rc], difference, std_error: stdError, lower, upper, lift, lift_lower: liftLower,
      lift_upper: liftUpper, decision: s.decision,
    };
  }

  step(state: State, x: StatInput): [RatioABState, State] {
    const s = state as RatioABState;
    const label = x[this.armInput] as number;
    if (!this.arms.includes(label)) return [s, this.output(s)];
    const k = this.arms.indexOf(label);
    const y = x[this.input] as number;
    const d = this.denominator ? (x[this.denominatorInput] as number) : 1.0;
    if (!Number.isFinite(y)) throw new ContractViolation(`input '${this.input}': the numerator must be finite, got ${y}`);
    if (!(Number.isFinite(d) && d >= 0.0))
      throw new ContractViolation(`input '${this.denominatorInput}': the denominator must be finite and at least 0, got ${d}`);
    const nOld = s.units[k];
    const n = nOld + 1;
    const sy = requireFinite(s.sum_y[k] + y, "numerator sum");
    const sd = requireFinite(s.sum_d[k] + d, "denominator sum");
    let cyy = s.cyy[k], cdd = s.cdd[k], cyd = s.cyd[k];
    if (nOld) {
      // Welford: (v - mean before) (w - mean after); a square's increment is >= 0 but for rounding
      const dy = y - s.sum_y[k] / nOld, dd = d - s.sum_d[k] / nOld;
      const ey = y - sy / n, ed = d - sd / n;
      cyy = requireFinite(cyy + pyMax(0.0, dy * ey), "numerator sum of squares");
      cdd = requireFinite(cdd + pyMax(0.0, dd * ed), "denominator sum of squares");
      cyd = requireFinite(cyd + dy * ed, "cross product");
    }
    let next: RatioABState = {
      units: put(s.units, k, n), sum_y: put(s.sum_y, k, sy), sum_d: put(s.sum_d, k, sd), cyy: put(s.cyy, k, cyy),
      cdd: put(s.cdd, k, cdd), cyd: put(s.cyd, k, cyd), decision: s.decision,
    };
    let out = this.output(next);
    if (next.decision === null) {
      const [lo, hi] = (this.scale === "difference" ? [out.lower, out.upper] : [out.lift_lower, out.lift_upper]) as (number | null)[];
      if (lo !== null && hi !== null && !(lo <= this.effect0 && this.effect0 <= hi)) {
        next = { ...next, decision: REJECT_H0 };
        out = { ...out, decision: REJECT_H0 };
      }
    }
    return [next, out];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): RatioABState {
    const s = decodeFields({
      units: "counts", sum_y: "numbers", sum_d: "numbers", cyy: "numbers", cdd: "numbers", cyd: "numbers", decision: "decision",
    }, data) as RatioABState;
    const pairs = [s.units, s.sum_y, s.sum_d, s.cyy, s.cdd, s.cyd];
    requireState(pairs.every((v) => v.length === 2), "two arms");
    requireState([...s.sum_d, ...s.cyy, ...s.cdd].every((v) => v >= 0.0), "negative denominator sum or sum of squares");
    requireState([0, 1].every((k) => s.units[k] > 0 || pairs.slice(1).every((v) => v[k] === 0.0)), "sums without units");
    if (!this.denominator)
      requireState(s.sum_d[0] === s.units[0] && s.sum_d[1] === s.units[1] && [...s.cdd, ...s.cyd].every((v) => v === 0.0),
        "without a denominator every d is 1");
    return s;
  }
}

register("ratio_ab_test", "1", (c) => RatioABTest.fromConfig(c));
