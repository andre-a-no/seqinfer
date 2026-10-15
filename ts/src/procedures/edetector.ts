// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Nonparametric change detection with e-detectors; mirrors procedures/edetector.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, refuseTerminal, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, type Config, int, invalid, num, pyMax, str } from "./util.js";

const DIRECTIONS = ["increase", "decrease", "both"];
const STATISTICS = ["shiryaev-roberts", "cusum"];
const FRACTION = 0.5;

type EDetectorState = { n: number; values: number[]; starts: number[]; alarm_at: number | null };

/** The shared recursion (Python's _EDetector); subclasses define the centred score. */
abstract class EDetector extends BaseProcedure {
  readonly bets: number[];
  readonly weight: number;

  /** `low` < 0 < `high` bound the centred score Y. */
  constructor(readonly arl: number, readonly direction: string, readonly statistic: string, readonly grid: number, low: number, high: number) {
    super();
    if (!(Number.isFinite(arl) && 1 <= arl && arl <= 1e15)) invalid("arl must be a number of observations in [1, 1e15]");
    if (!DIRECTIONS.includes(direction)) invalid(`direction must be one of ${DIRECTIONS}`);
    if (!STATISTICS.includes(statistic)) invalid(`statistic must be one of ${STATISTICS}`);
    if (!Number.isInteger(grid) || !(1 <= grid && grid <= 60)) invalid("grid must be an int in [1, 60]");
    const bets: number[] = [];
    if (direction !== "decrease") for (let k = 0; k < grid; k++) bets.push((FRACTION / -low) * 2.0 ** -k);
    if (direction !== "increase") for (let k = 0; k < grid; k++) bets.push((-FRACTION / high) * 2.0 ** -k);
    this.bets = bets;
    this.weight = 1.0 / bets.length;
  }

  protected abstract score(x: StatInput): number;

  initialState(): EDetectorState {
    return { n: 0, values: [], starts: [], alarm_at: null };
  }

  step(state: State, x: StatInput): [EDetectorState, State] {
    refuseTerminal(this, state);
    const s = state as EDetectorState;
    const y = this.score(x);
    const n = s.n + 1;
    const values: number[] = [], starts: number[] = [];
    this.bets.forEach((bet, k) => {
      const m = s.values.length ? s.values[k] : 0.0;
      const start = s.starts.length ? s.starts[k] : 1;
      const factor = 1.0 + bet * y;
      const carried = this.statistic === "shiryaev-roberts" ? m + 1.0 : pyMax(m, 1.0);
      values.push(requireFinite(factor * carried, "e-detector"));
      starts.push(m < 1.0 ? n : start);
    });
    let total = 0.0;
    for (const v of values) total += this.weight * v;
    const alarm = total >= this.arl;
    let best = 0;
    for (let k = 1; k < values.length; k++) if (values[k] > values[best]) best = k;
    const change = values[best] > 1.0 ? starts[best] : null;
    return [{ n, values, starts, alarm_at: alarm ? n : null }, { n, statistic: total, alarm, change_estimate: change }];
  }

  override isTerminal(state: State): boolean {
    return state.alarm_at !== null;
  }

  decodeState(data: Json): EDetectorState {
    const s = decodeFields({ n: "count", values: "numbers", starts: "counts", alarm_at: "count?" }, data) as EDetectorState;
    const size = s.n ? this.bets.length : 0;
    requireState(s.values.length === size && s.starts.length === size, "one value per bet");
    requireState(s.values.every((v) => v >= 0), "negative statistic");
    requireState(s.starts.every((v) => 1 <= v && v <= s.n), "start out of range");
    requireState(s.alarm_at === null || s.alarm_at === s.n, "alarm before the last observation");
    return s;
  }
}

/** Detects a change of the mean of values in [lo, hi] whose mean is m before the change. */
export class MeanEDetector extends EDetector {
  readonly name = "mean_e_detector";
  readonly version = "1";

  constructor(readonly m: number, readonly lo = 0.0, readonly hi = 1.0, arl = 1000.0, direction = "both", statistic = "shiryaev-roberts",
    grid = 10, readonly input = "x") {
    if (!(Number.isFinite(lo) && Number.isFinite(hi) && lo < m && m < hi)) invalid("need finite lo < m < hi");
    super(arl, direction, statistic, grid, lo - m, hi - m);
  }

  static fromConfig(c: Config): MeanEDetector {
    return new MeanEDetector(num(c, "m"), num(c, "lo"), num(c, "hi"), num(c, "arl"), str(c, "direction"), str(c, "statistic"),
      int(c, "grid"), str(c, "input"));
  }

  config(): Config {
    return {
      m: this.m, lo: this.lo, hi: this.hi, arl: this.arl, direction: this.direction, statistic: this.statistic, grid: this.grid,
      input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input, "real", [], this.lo, this.hi) };
  }

  protected score(x: StatInput): number {
    return (x[this.input] as number) - this.m;
  }
}

/** Detects a change of the p-quantile (q0 before the change), for data of any distribution. */
export class QuantileEDetector extends EDetector {
  readonly name = "quantile_e_detector";
  readonly version = "1";

  constructor(readonly q0: number, readonly p = 0.5, arl = 1000.0, direction = "both", statistic = "shiryaev-roberts", grid = 10,
    readonly input = "x") {
    if (!Number.isFinite(q0)) invalid("q0 must be finite");
    if (!(0 < p && p < 1)) invalid("p must lie in (0, 1)");
    super(arl, direction, statistic, grid, p - 1.0, p);
  }

  static fromConfig(c: Config): QuantileEDetector {
    return new QuantileEDetector(num(c, "q0"), num(c, "p"), num(c, "arl"), str(c, "direction"), str(c, "statistic"), int(c, "grid"),
      str(c, "input"));
  }

  config(): Config {
    return { q0: this.q0, p: this.p, arl: this.arl, direction: this.direction, statistic: this.statistic, grid: this.grid, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  protected score(x: StatInput): number {
    return this.p - ((x[this.input] as number) <= this.q0 ? 1.0 : 0.0);
  }
}

register("mean_e_detector", "1", (c) => MeanEDetector.fromConfig(c));
register("quantile_e_detector", "1", (c) => QuantileEDetector.fromConfig(c));
