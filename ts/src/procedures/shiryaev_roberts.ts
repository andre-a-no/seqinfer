// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The Shiryaev-Roberts change-point detector, on the log scale; mirrors procedures/shiryaev_roberts.py.
import type { Json } from "../canonical.js";
import type { InputContract, StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, refuseTerminal, register, requireState, type State } from "../procedure.js";
import { type Family, familyFromSpec } from "./families.js";
import { BaseProcedure, type Config, invalid, num, obj, pyMax, sameState, str } from "./util.js";

type SRState = { n: number; log_r: number | null; alarm_at: number | null };

/** log(1 + e^x) without overflow. */
const log1pExp = (x: number): number => pyMax(x, 0.0) + Math.log1p(Math.exp(-Math.abs(x)));

export class ShiryaevRoberts extends BaseProcedure {
  readonly name = "shiryaev_roberts";
  readonly version = "1";
  private readonly logThreshold: number;

  constructor(readonly family: Family, readonly theta0: number, readonly theta1: number, readonly threshold: number, readonly input = "x") {
    super();
    family.check(theta0);
    family.check(theta1);
    if (theta0 === theta1) invalid("need theta0 != theta1");
    if (!(1 < threshold && threshold < Infinity)) invalid("threshold A must exceed 1");
    this.logThreshold = Math.log(threshold);
  }

  static fromConfig(c: Config): ShiryaevRoberts {
    return new ShiryaevRoberts(familyFromSpec(obj(c, "family")), num(c, "theta0"), num(c, "theta1"), num(c, "threshold"), str(c, "input"));
  }

  config(): Config {
    return { family: this.family.spec(), theta0: this.theta0, theta1: this.theta1, threshold: this.threshold, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.family.contract(this.input) };
  }

  initialState(): SRState {
    return { n: 0, log_r: null, alarm_at: null };
  }

  step(state: State, x: StatInput): [SRState, State] {
    refuseTerminal(this, state);
    const s = state as SRState;
    const llr = this.family.llr(x[this.input] as number, this.theta1, this.theta0);
    const base = s.log_r === null ? 0.0 : log1pExp(s.log_r);
    const logR = requireFinite(base + llr, "Shiryaev-Roberts statistic");
    const n = s.n + 1;
    const alarm = logR >= this.logThreshold;
    return [{ n, log_r: logR, alarm_at: alarm ? n : null }, { n, log_statistic: logR, alarm }];
  }

  override isTerminal(state: State): boolean {
    return state.alarm_at !== null;
  }

  decodeState(data: Json): SRState {
    const s = decodeFields({ n: "count", log_r: "number?", alarm_at: "count?" }, data) as SRState;
    requireState(s.alarm_at === null || s.alarm_at === s.n, "alarm before the last observation");
    if (s.log_r === null) requireState(sameState(s, this.initialState()), "observations, but no statistic");
    else {
      requireState(s.n > 0, "a statistic without observations");
      requireState((s.alarm_at !== null) === s.log_r >= this.logThreshold, "alarm does not match statistic");
    }
    return s;
  }
}

register("shiryaev_roberts", "1", (c) => ShiryaevRoberts.fromConfig(c));
