// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Anytime-valid t-test and confidence sequence; mirrors procedures/t_mixture.py (version 3).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { NumericalError } from "../errors.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, invalid, num, pyMax, pyMin, REJECT_H0, str } from "./util.js";

const UNDERFLOW =
  "the sample variance underflows to zero although the observations differ; " +
  "rescale the data in the adapter (for example to units where they are of order one)";

type TMixtureState = { n: number; mean: number; m2: number; max_log_bf: number; decision: Decision; low: number | null; high: number | null };

export class TMixtureSPRT extends BaseProcedure {
  readonly name = "t_mixture_sprt";
  readonly version = "3";
  readonly eValueField = "log_bf";
  private readonly logLevel: number;

  constructor(readonly theta0 = 0.0, readonly effect = 0.5, readonly alpha = 0.05, readonly stopOnReject = true, readonly input = "x") {
    super();
    if (!Number.isFinite(theta0)) invalid("theta0 must be finite");
    if (!(0 < effect && effect <= 1e100)) invalid("effect (prior sd of the standardised effect) must lie in (0, 1e100]");
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    this.logLevel = Math.log(1.0 / alpha);
  }

  static fromConfig(c: Config): TMixtureSPRT {
    return new TMixtureSPRT(num(c, "theta0"), num(c, "effect"), num(c, "alpha"), bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return { theta0: this.theta0, effect: this.effect, alpha: this.alpha, stop_on_reject: this.stopOnReject, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): TMixtureState {
    return { n: 0, mean: 0.0, m2: 0.0, max_log_bf: 0.0, decision: null, low: null, high: null };
  }

  logBf(n: number, t2: number): number {
    const nu = n - 1, r = 1.0 + n * (this.effect * this.effect);
    return -0.5 * Math.log(r) + 0.5 * (nu + 1) * (Math.log1p(t2 / nu) - Math.log1p(t2 / (nu * r)));
  }

  /** |t| at which B_n reaches 1/alpha, or null if it cannot at this n. */
  criticalT(n: number): number | null {
    const nu = n - 1, logR = Math.log1p(n * (this.effect * this.effect));
    const logK = (2.0 * (this.logLevel + 0.5 * logR)) / (nu + 1);
    if (logK >= logR) return null;
    return Math.sqrt((nu * Math.expm1(logK)) / -Math.expm1(logK - logR));
  }

  step(state: State, x: StatInput): [TMixtureState, State] {
    const s = state as TMixtureState;
    const v = x[this.input] as number;
    const n = s.n + 1;
    const delta = v - s.mean;
    let mean = s.mean + delta / n;
    const m2 = requireFinite(s.m2 + delta * (v - mean), "sum of squared deviations");
    mean = requireFinite(mean, "running mean");
    const low = s.low === null ? v : pyMin(s.low, v);
    const high = s.high === null ? v : pyMax(s.high, v);
    const r = 1.0 + n * (this.effect * this.effect);
    if (n >= 2 && m2 <= 0.0 && low !== high) throw new NumericalError(UNDERFLOW);
    let sd: number | null, t: number | null, logBf: number, lower: number | null, upper: number | null;
    if (n < 2) {
      sd = t = null;
      logBf = 0.0;
      lower = upper = null;
    } else if (m2 <= 0.0) {
      sd = 0.0;
      t = null;
      logBf = mean === this.theta0 ? -0.5 * Math.log(r) : 0.5 * (n - 1) * Math.log(r);
      lower = upper = this.criticalT(n) !== null ? mean : null;
    } else {
      sd = Math.sqrt(m2 / (n - 1));
      t = ((mean - this.theta0) * Math.sqrt(n)) / sd;
      const t2 = t * t;
      logBf = t2 === Infinity ? 0.5 * (n - 1) * Math.log(r) : this.logBf(n, t2);
      logBf = requireFinite(logBf, "log Bayes factor");
      const crit = this.criticalT(n);
      lower = upper = null;
      if (crit !== null) {
        const half = (crit * sd) / Math.sqrt(n);
        [lower, upper] = [mean - half, mean + half];
      }
    }
    const maxLogBf = pyMax(s.max_log_bf, logBf);
    const decision = s.decision || (logBf >= this.logLevel ? REJECT_H0 : null);
    const pValue = pyMin(1.0, Math.exp(-maxLogBf));
    return [
      { n, mean, m2, max_log_bf: maxLogBf, decision, low, high },
      { n, mean, sd, t, log_bf: logBf, p_value: pValue, lower, upper, decision },
    ];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): TMixtureState {
    const s = decodeFields({
      n: "count", mean: "number", m2: "number", max_log_bf: "number", decision: "decision", low: "number?", high: "number?",
    }, data) as TMixtureState;
    requireState(s.m2 >= 0, "negative sum of squares");
    return s;
  }
}

register("t_mixture_sprt", "3", (c) => TMixtureSPRT.fromConfig(c));
