// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Lorden's 2-SPRT; mirrors procedures/two_sprt.py (the transition, not the design code).
import type { Json } from "../canonical.js";
import type { InputContract, StatInput } from "../contracts.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, refuseTerminal, register, requireState, type State } from "../procedure.js";
import { type Family, familyFromSpec } from "./families.js";
import { ACCEPT_H0, BaseProcedure, type Config, type Decision, invalid, num, obj, REJECT_H0, sameState, str } from "./util.js";

/** Intermediate point at which both boundaries are reached at the same rate (bisection, as in Python). */
export function kieferWeissPoint(family: Family, theta0: number, theta1: number, a0: number, a1: number): number {
  let [lo, hi] = theta0 < theta1 ? [theta0, theta1] : [theta1, theta0];
  const h = (theta: number) => a0 * family.kl(theta, theta1) - a1 * family.kl(theta, theta0);
  const signLo = h(lo + (hi - lo) * 1e-12) > 0;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (h(mid) > 0 === signLo) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

type TwoSPRTState = { n: number; l0: number; c0: number; l1: number; c1: number; decision: Decision };

export class TwoSPRT extends BaseProcedure {
  readonly name = "two_sprt";
  readonly version = "1";
  readonly a0: number;
  readonly a1: number;
  readonly thetaStar: number;

  constructor(
    readonly family: Family,
    readonly theta0: number,
    readonly theta1: number,
    readonly alpha0 = 0.05,
    readonly alpha1 = 0.05,
    thetaStar: number | null = null,
    thresholds: [number, number] | null = null,
    readonly input = "x",
  ) {
    super();
    family.check(theta0);
    family.check(theta1);
    if (theta0 === theta1 || !(0 < alpha0 && alpha0 < 1 && 0 < alpha1 && alpha1 < 1))
      invalid("need theta0 != theta1 and error rates in (0, 1)");
    [this.a0, this.a1] = thresholds ?? [Math.log(1.0 / alpha0), Math.log(1.0 / alpha1)];
    if (!(0 < this.a0 && this.a0 < Infinity && 0 < this.a1 && this.a1 < Infinity)) invalid("thresholds must be positive");
    thetaStar ??= kieferWeissPoint(family, theta0, theta1, this.a0, this.a1);
    if (!(Math.min(theta0, theta1) < thetaStar && thetaStar < Math.max(theta0, theta1)))
      invalid("theta_star must lie strictly between theta0 and theta1");
    this.thetaStar = thetaStar;
  }

  static fromConfig(c: Config): TwoSPRT {
    return new TwoSPRT(familyFromSpec(obj(c, "family")), num(c, "theta0"), num(c, "theta1"), num(c, "alpha0"), num(c, "alpha1"),
      num(c, "theta_star"), [num(c, "a0"), num(c, "a1")], str(c, "input"));
  }

  config(): Config {
    return {
      family: this.family.spec(), theta0: this.theta0, theta1: this.theta1, theta_star: this.thetaStar,
      alpha0: this.alpha0, alpha1: this.alpha1, a0: this.a0, a1: this.a1, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.family.contract(this.input) };
  }

  initialState(): TwoSPRTState {
    return { n: 0, l0: 0.0, c0: 0.0, l1: 0.0, c1: 0.0, decision: null };
  }

  step(state: State, x: StatInput): [TwoSPRTState, State] {
    refuseTerminal(this, state);
    const s = state as TwoSPRTState;
    const v = x[this.input] as number;
    const [l0, c0] = neumaierAdd(s.l0, s.c0, this.family.llr(v, this.thetaStar, this.theta0));
    const [l1, c1] = neumaierAdd(s.l1, s.c1, this.family.llr(v, this.thetaStar, this.theta1));
    const s0 = requireFinite(l0 + c0, "log likelihood ratio against theta0");
    const s1 = requireFinite(l1 + c1, "log likelihood ratio against theta1");
    const decision = this.decide(s0, s1);
    const n = s.n + 1;
    return [{ n, l0, c0, l1, c1, decision }, { n, l0: s0, l1: s1, decision }];
  }

  private decide(s0: number, s1: number): Decision {
    const excess0 = s0 - this.a0, excess1 = s1 - this.a1;
    if (excess0 >= 0 || excess1 >= 0) return excess0 > excess1 ? REJECT_H0 : ACCEPT_H0;
    return null;
  }

  override isTerminal(state: State): boolean {
    return state.decision !== null;
  }

  decodeState(data: Json): TwoSPRTState {
    const s = decodeFields({ n: "count", l0: "number", c0: "number", l1: "number", c1: "number", decision: "decision" }, data) as TwoSPRTState;
    if (s.n === 0) requireState(sameState(s, this.initialState()), "no observations, but a statistic or decision");
    else requireState(s.decision === this.decide(s.l0 + s.c0, s.l1 + s.c1), "decision does not match the statistics");
    return s;
  }
}

register("two_sprt", "1", (c) => TwoSPRT.fromConfig(c));
