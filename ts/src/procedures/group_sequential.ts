// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Group sequential test of a normal mean on a stream; mirrors procedures/group_sequential.py (version 4).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { NumericalError } from "../errors.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, refuseTerminal, register, requireState, type State } from "../procedure.js";
import { logNormalUpper, tIsfLog } from "./distributions.js";
import { ACCEPT_H0, BaseProcedure, bool, type Config, type Decision, invalid, num, nums, optNum, pyMax, pyMin, REJECT_H0, str } from "./util.js";

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
