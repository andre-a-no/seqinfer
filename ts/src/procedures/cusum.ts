// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Page's CUSUM change-point detector; mirrors procedures/cusum.py.
import type { Json } from "../canonical.js";
import type { InputContract, StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, refuseTerminal, register, requireState, type State } from "../procedure.js";
import { type Family, familyFromSpec } from "./families.js";
import { BaseProcedure, type Config, invalid, num, obj, pyMax, sameState, str } from "./util.js";

type CusumState = { n: number; w: number; alarm_at: number | null };

export class CUSUM extends BaseProcedure {
  readonly name = "cusum";
  readonly version = "1";

  constructor(readonly family: Family, readonly theta0: number, readonly theta1: number, readonly threshold: number, readonly input = "x") {
    super();
    family.check(theta0);
    family.check(theta1);
    if (theta0 === theta1) invalid("need theta0 != theta1: with equal parameters the statistic never moves");
    if (!(threshold > 0 && Number.isFinite(threshold))) invalid("threshold must be positive");
  }

  static fromConfig(c: Config): CUSUM {
    return new CUSUM(familyFromSpec(obj(c, "family")), num(c, "theta0"), num(c, "theta1"), num(c, "threshold"), str(c, "input"));
  }

  config(): Config {
    return { family: this.family.spec(), theta0: this.theta0, theta1: this.theta1, threshold: this.threshold, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.family.contract(this.input) };
  }

  initialState(): CusumState {
    return { n: 0, w: 0.0, alarm_at: null };
  }

  step(state: State, x: StatInput): [CusumState, State] {
    refuseTerminal(this, state);
    const s = state as CusumState;
    const inc = this.family.llr(x[this.input] as number, this.theta1, this.theta0);
    const w = pyMax(0.0, requireFinite(s.w + inc, "CUSUM statistic"));
    const n = s.n + 1;
    const alarm = w >= this.threshold;
    return [{ n, w, alarm_at: alarm ? n : null }, { n, statistic: w, alarm }];
  }

  override isTerminal(state: State): boolean {
    return state.alarm_at !== null;
  }

  decodeState(data: Json): CusumState {
    const s = decodeFields({ n: "count", w: "number", alarm_at: "count?" }, data) as CusumState;
    requireState(s.w >= 0, "negative statistic");
    requireState(s.alarm_at === null || s.alarm_at === s.n, "alarm before the last observation");
    if (s.n) requireState((s.alarm_at !== null) === s.w >= this.threshold, "alarm does not match statistic");
    else requireState(sameState(s, this.initialState()), "no observations, but a statistic or alarm");
    return s;
  }
}

register("cusum", "1", (c) => CUSUM.fromConfig(c));
