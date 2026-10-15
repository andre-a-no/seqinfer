// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Exact anytime-valid comparison of two event rates; mirrors procedures/rates.py (RateRatioTest).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { ContractViolation } from "../errors.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, int, invalid, num, nums, pyMax, pyMin, REJECT_H0, str } from "./util.js";

const ALTERNATIVES = ["two-sided", "greater", "less"];

type RateRatioState = {
  counts: number[]; exposures: number[]; units: number[]; pending: number[]; pairs: number;
  pair_counts: number[]; pair_exposures: number[]; log_e: number; comp: number; max_log_e: number;
  grid_log_e: number[]; excluded: number[]; decision: Decision;
};

/** log Bin(k; n, q1) / Bin(k; n, q0): the binomial coefficients cancel. */
function logBinomialRatio(k: number, n: number, q1: number, q0: number): number {
  let out = 0.0;
  if (k) out += k * (Math.log(q1) - Math.log(q0));
  if (n - k) out += (n - k) * (Math.log1p(-q1) - Math.log1p(-q0));
  return out;
}

/** Exact anytime-valid test and confidence sequence for a ratio of Poisson rates. */
export class RateRatioTest extends BaseProcedure {
  readonly name = "rate_ratio_test";
  readonly version = "1";
  readonly eValueField = "log_e_value";
  readonly arms: [number, number];
  readonly ratioRange: [number, number];
  private readonly logLevel: number;
  private readonly points: number[];

  constructor(
    readonly alpha = 0.05,
    readonly rho0 = 1.0,
    readonly alternative = "two-sided",
    arms: number[] = [1, 0],
    readonly exposure = true,
    readonly prior = 0.5,
    readonly grid = 200,
    ratioRange: number[] = [0.1, 10.0],
    readonly stopOnReject = false,
    readonly armInput = "arm",
    readonly input = "count",
    readonly exposureInput = "exposure",
  ) {
    super();
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (!(Number.isFinite(rho0) && rho0 > 0)) invalid("rho0 must be positive and finite");
    if (!ALTERNATIVES.includes(alternative)) invalid(`alternative must be one of ${ALTERNATIVES}`);
    if (arms.length !== 2 || arms.some((a) => !Number.isInteger(a) || a < 0) || arms[0] === arms[1])
      invalid("arms must be two different non-negative integer labels");
    this.arms = [arms[0], arms[1]];
    if (!(Number.isFinite(prior) && prior > 0)) invalid("prior must be a positive number of pseudo-events");
    if (!Number.isInteger(grid) || !(grid === 0 || (10 <= grid && grid <= 100_000)))
      invalid("grid must be 0 (no confidence sequence) or an int in [10, 100000]");
    if (ratioRange.length !== 2) invalid("ratio_range must be 0 < low < high");
    const [lo, hi] = ratioRange;
    if (!(0 < lo && lo < hi && hi < Infinity)) invalid("ratio_range must be 0 < low < high");
    const names = new Set([armInput, input, ...(exposure ? [exposureInput] : [])]);
    if (names.size !== (exposure ? 3 : 2)) invalid("the arm, count and exposure need different names");
    this.ratioRange = [lo, hi];
    this.logLevel = -Math.log(alpha);
    const span = Math.log(hi) - Math.log(lo);
    this.points = [];
    for (let j = 0; j < grid; j++) this.points.push(Math.exp(Math.log(lo) + (span * j) / (grid - 1)));
  }

  static fromConfig(c: Config): RateRatioTest {
    return new RateRatioTest(num(c, "alpha"), num(c, "rho0"), str(c, "alternative"), nums(c, "arms"), bool(c, "exposure"),
      num(c, "prior"), int(c, "grid"), nums(c, "ratio_range"), bool(c, "stop_on_reject"), str(c, "arm_input"), str(c, "input"),
      str(c, "exposure_input"));
  }

  config(): Config {
    return {
      alpha: this.alpha, rho0: this.rho0, alternative: this.alternative, arms: [...this.arms], exposure: this.exposure,
      prior: this.prior, grid: this.grid, ratio_range: [...this.ratioRange], stop_on_reject: this.stopOnReject,
      arm_input: this.armInput, input: this.input, exposure_input: this.exposureInput,
    };
  }

  inputs(): Record<string, InputContract> {
    const out: Record<string, InputContract> = {
      [this.armInput]: new InputContract(this.armInput, "integer", [], 0),
      [this.input]: new InputContract(this.input, "integer", [], 0),
    };
    if (this.exposure) out[this.exposureInput] = new InputContract(this.exposureInput);
    return out;
  }

  initialState(): RateRatioState {
    return {
      counts: [0, 0], exposures: [0.0, 0.0], units: [0, 0], pending: [], pairs: 0, pair_counts: [0, 0], pair_exposures: [0.0, 0.0],
      log_e: 0.0, comp: 0.0, max_log_e: 0.0, grid_log_e: this.points.map(() => 0.0), excluded: this.points.map(() => 0),
      decision: null,
    };
  }

  /** rho_hat from the completed pairs, shrunk towards 1 by `prior` pseudo-events in each arm. */
  plugIn(counts: number[], exposures: number[]): number {
    if (exposures[0] <= 0 || exposures[1] <= 0) return 1.0;
    return (counts[0] + this.prior) / exposures[0] / ((counts[1] + this.prior) / exposures[1]);
  }

  /** P(an event of the pair is the treatment unit's) under rate ratio rho. */
  static share(rho: number, eT: number, eC: number): number {
    return (rho * eT) / (rho * eT + eC);
  }

  /** log of one pair's e-value against rho0 (0 when not betting). */
  pairLogE(rhoHat: number, rho0: number, alternative: string, kT: number, k: number, eT: number, eC: number): number {
    if ((alternative === "greater" && rhoHat <= rho0) || (alternative === "less" && rhoHat >= rho0)) return 0.0;
    if (k === 0 || rhoHat === rho0) return 0.0;
    return logBinomialRatio(kT, k, RateRatioTest.share(rhoHat, eT, eC), RateRatioTest.share(rho0, eT, eC));
  }

  private interval(s: RateRatioState): [number | null, number | null] {
    const kept: number[] = [];
    s.excluded.forEach((x, j) => {
      if (!x) kept.push(j);
    });
    if (!kept.length) {
      // every point excluded: the least-excluded point (the first smallest e-value) says where the ratio is (rates.py)
      let least = 0;
      for (let j = 1; j < this.grid; j++) if (s.grid_log_e[j] < s.grid_log_e[least]) least = j;
      if (least === this.grid - 1) return [this.points[this.grid - 1], null];
      if (least === 0) return [null, this.points[0]];
      return [this.points[least - 1], this.points[least + 1]];
    }
    const lo = kept[0] === 0 ? null : this.points[kept[0] - 1];
    const hi = kept[kept.length - 1] === this.grid - 1 ? null : this.points[kept[kept.length - 1] + 1];
    return [lo, hi];
  }

  private output(s: RateRatioState, value: number): State {
    const [kt, kc] = s.counts, [et, ec] = s.exposures;
    const rt = et > 0 ? kt / et : null;
    const rc = ec > 0 ? kc / ec : null;
    const ratio = rt !== null && rc ? rt / rc : null;
    const [lower, upper] = this.grid ? this.interval(s) : [null, null];
    return {
      units: s.units[0] + s.units[1], rates: [rt, rc], rate_ratio: ratio, log_e_value: value,
      p_value: pyMin(1.0, Math.exp(-s.max_log_e)), decision: s.decision, lower, upper, pairs: s.pairs,
    };
  }

  step(state: State, x: StatInput): [RateRatioState, State] {
    const s = state as RateRatioState;
    const label = x[this.armInput] as number;
    if (!this.arms.includes(label)) return [s, this.output(s, s.log_e + s.comp)];
    const k = this.arms.indexOf(label);
    const count = x[this.input] as number;
    const e = this.exposure ? (x[this.exposureInput] as number) : 1.0;
    if (!(e > 0 && Number.isFinite(e)))
      throw new ContractViolation(`input '${this.exposureInput}': exposure must be positive, got ${e}`);
    const counts = [...s.counts], exposures = [...s.exposures], units = [...s.units];
    counts[k] += count;
    exposures[k] += e;
    units[k] += 1;
    let logE = s.log_e, comp = s.comp, pairs = s.pairs;
    let pairK = s.pair_counts, pairE = s.pair_exposures;
    let gridLogE = s.grid_log_e, excluded = s.excluded;
    let pending: number[];
    if (!s.pending.length || s.pending[0] === k) {
      pending = [k, count, e]; // wait (replacing a unit of the same arm)
    } else {
      pending = [];
      const [kT, eT, kC, eC] = k === 0 ? [count, e, s.pending[1], s.pending[2]] : [s.pending[1], s.pending[2], count, e];
      const total = kT + kC;
      const rhoHat = this.plugIn(pairK, pairE);
      [logE, comp] = neumaierAdd(logE, comp, this.pairLogE(rhoHat, this.rho0, this.alternative, kT, total, eT, eC));
      if (this.grid) {
        const newGrid: number[] = [], newExcluded: number[] = [];
        this.points.forEach((rho, j) => {
          const g = gridLogE[j] + this.pairLogE(rhoHat, rho, "two-sided", kT, total, eT, eC);
          newGrid.push(g);
          newExcluded.push(excluded[j] || g >= this.logLevel ? 1 : 0);
        });
        gridLogE = newGrid;
        excluded = newExcluded;
      }
      pairs += 1;
      pairK = [pairK[0] + kT, pairK[1] + kC];
      pairE = [pairE[0] + eT, pairE[1] + eC];
    }
    const value = requireFinite(logE + comp, "log e-value");
    const decision = s.decision || (value >= this.logLevel ? REJECT_H0 : null);
    const next: RateRatioState = {
      counts, exposures, units, pending, pairs, pair_counts: pairK, pair_exposures: pairE, log_e: logE, comp,
      max_log_e: pyMax(s.max_log_e, value), grid_log_e: gridLogE, excluded, decision,
    };
    return [next, this.output(next, value)];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): RateRatioState {
    const s = decodeFields({
      counts: "counts", exposures: "numbers", units: "counts", pending: "numbers", pairs: "count", pair_counts: "counts",
      pair_exposures: "numbers", log_e: "number", comp: "number", max_log_e: "number", grid_log_e: "numbers", excluded: "counts",
      decision: "decision",
    }, data) as RateRatioState;
    requireState(s.grid_log_e.length === this.grid && s.excluded.length === this.grid, "one value per grid point");
    requireState(s.excluded.every((x) => x === 0 || x === 1), "exclusion flags are 0 or 1");
    const p = s.pending;
    requireState(p.length === 0 || (p.length === 3 && (p[0] === 0 || p[0] === 1) && Number.isInteger(p[1]) && p[2] > 0),
      "pending (arm, count, exposure)");
    requireState([s.counts, s.exposures, s.units, s.pair_counts, s.pair_exposures].every((v) => v.length === 2), "two arms");
    requireState(2 * s.pairs <= s.units[0] + s.units[1], "more pairs than units");
    return s;
  }
}

register("rate_ratio_test", "1", (c) => RateRatioTest.fromConfig(c));
