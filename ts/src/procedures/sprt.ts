// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Wald's sequential probability ratio test; mirrors procedures/sprt.py.
import type { Json } from "../canonical.js";
import type { InputContract, StatInput } from "../contracts.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, refuseTerminal, register, requireState, type State } from "../procedure.js";
import { type Family, familyFromSpec } from "./families.js";
import { ACCEPT_H0, BaseProcedure, type Config, type Decision, invalid, num, obj, REJECT_H0, sameState, str } from "./util.js";

type SPRTState = { n: number; llr: number; comp: number; decision: Decision };

export class SPRT extends BaseProcedure {
  readonly name = "sprt";
  readonly version = "1";
  readonly upper: number;
  readonly lower: number;

  constructor(
    readonly family: Family,
    readonly theta0: number,
    readonly theta1: number,
    readonly alpha = 0.05,
    readonly beta = 0.05,
    readonly input = "x",
  ) {
    super();
    family.check(theta0);
    family.check(theta1);
    if (theta0 === theta1 || !(0 < alpha && alpha < 1 && 0 < beta && beta < 1) || alpha + beta >= 1)
      invalid("need theta0 != theta1, error rates in (0, 1) and alpha + beta < 1");
    this.upper = Math.log((1.0 - beta) / alpha);
    this.lower = Math.log(beta / (1.0 - alpha));
  }

  static fromConfig(c: Config): SPRT {
    return new SPRT(familyFromSpec(obj(c, "family")), num(c, "theta0"), num(c, "theta1"), num(c, "alpha"), num(c, "beta"), str(c, "input"));
  }

  config(): Config {
    return { family: this.family.spec(), theta0: this.theta0, theta1: this.theta1, alpha: this.alpha, beta: this.beta, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.family.contract(this.input) };
  }

  initialState(): SPRTState {
    return { n: 0, llr: 0.0, comp: 0.0, decision: null };
  }

  step(state: State, x: StatInput): [SPRTState, State] {
    refuseTerminal(this, state);
    const s = state as SPRTState;
    const inc = this.family.llr(x[this.input] as number, this.theta1, this.theta0);
    const [total, comp] = neumaierAdd(s.llr, s.comp, inc);
    const value = requireFinite(total + comp, "log likelihood ratio");
    const decision = this.decide(value);
    const n = s.n + 1;
    return [{ n, llr: total, comp, decision }, { n, llr: value, decision }];
  }

  private decide(value: number): Decision {
    return value >= this.upper ? REJECT_H0 : value <= this.lower ? ACCEPT_H0 : null;
  }

  override isTerminal(state: State): boolean {
    return state.decision !== null;
  }

  decodeState(data: Json): SPRTState {
    const s = decodeFields({ n: "count", llr: "number", comp: "number", decision: "decision" }, data) as SPRTState;
    if (s.n === 0) requireState(sameState(s, this.initialState()), "no observations, but a statistic or decision");
    else requireState(s.decision === this.decide(s.llr + s.comp), "decision does not match the statistic");
    return s;
  }
}

register("sprt", "1", (c) => SPRT.fromConfig(c));
