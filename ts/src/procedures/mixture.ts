// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Normal-mixture SPRT (mSPRT) and its confidence sequence; mirrors procedures/mixture.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, invalid, num, pyMax, pyMin, REJECT_H0, str } from "./util.js";

type MixtureState = { n: number; total: number; comp: number; max_log_lambda: number; decision: Decision };

export class NormalMixtureSPRT extends BaseProcedure {
  readonly name = "normal_mixture_sprt";
  readonly version = "1";
  readonly eValueField = "log_lambda";
  private readonly logLevel: number;

  constructor(
    readonly sigma: number,
    readonly tau: number,
    readonly theta0 = 0.0,
    readonly alpha = 0.05,
    readonly stopOnReject = true,
    readonly input = "x",
  ) {
    super();
    if (!(sigma > 0 && tau > 0 && Number.isFinite(sigma) && Number.isFinite(tau))) invalid("sigma and tau must be positive and finite");
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (!Number.isFinite(theta0)) invalid("theta0 must be finite");
    this.logLevel = Math.log(1.0 / alpha);
  }

  static fromConfig(c: Config): NormalMixtureSPRT {
    return new NormalMixtureSPRT(num(c, "sigma"), num(c, "tau"), num(c, "theta0"), num(c, "alpha"), bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return { sigma: this.sigma, tau: this.tau, theta0: this.theta0, alpha: this.alpha, stop_on_reject: this.stopOnReject, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): MixtureState {
    return { n: 0, total: 0.0, comp: 0.0, max_log_lambda: 0.0, decision: null };
  }

  logLambda(n: number, mean: number): number {
    const s2 = this.sigma * this.sigma, t2 = this.tau * this.tau;
    const v = s2 + n * t2;
    const d = mean - this.theta0;
    return 0.5 * Math.log(s2 / v) + (n * n * t2 * (d * d)) / (2.0 * s2 * v);
  }

  halfWidth(n: number): number {
    const s2 = this.sigma * this.sigma, t2 = this.tau * this.tau;
    const v = s2 + n * t2;
    return Math.sqrt(((2.0 * s2 * v) / (n * n * t2)) * (this.logLevel + 0.5 * Math.log(v / s2)));
  }

  step(state: State, x: StatInput): [MixtureState, State] {
    const s = state as MixtureState;
    const [total, comp] = neumaierAdd(s.total, s.comp, x[this.input] as number);
    const n = s.n + 1;
    const mean = requireFinite((total + comp) / n, "running mean");
    const logLambda = this.logLambda(n, mean);
    const maxLogLambda = pyMax(s.max_log_lambda, logLambda);
    const decision = s.decision || (logLambda >= this.logLevel ? REJECT_H0 : null);
    const h = this.halfWidth(n);
    const pValue = pyMin(1.0, Math.exp(-maxLogLambda));
    return [
      { n, total, comp, max_log_lambda: maxLogLambda, decision },
      { n, mean, log_lambda: logLambda, p_value: pValue, lower: mean - h, upper: mean + h, decision },
    ];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): MixtureState {
    return decodeFields({ n: "count", total: "number", comp: "number", max_log_lambda: "number", decision: "decision" }, data) as MixtureState;
  }
}

register("normal_mixture_sprt", "1", (c) => NormalMixtureSPRT.fromConfig(c));
