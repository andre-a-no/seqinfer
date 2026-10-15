// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Online false discovery rate control for a stream of experiments; mirrors procedures/online_fdr.py (e-LOND).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { LifecycleError } from "../errors.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, type Config, invalid, num, pyMax, str } from "./util.js";

/** Ramdas, Yang, Wainwright and Jordan (2017); with it gamma sums to 0.976 <= 1. */
const GAMMA_NORMALISER = 0.07720838;

/** gamma_j of Javanmard and Montanari (2018), j = 1, 2, ...; the sequence sums to at most 1. */
export function defaultGamma(j: number): number {
  return (GAMMA_NORMALISER * Math.log(pyMax(j, 2))) / (j * Math.exp(Math.sqrt(Math.log(j))));
}

type OnlineFDRState = { t: number; discoveries: number };

/** e-LOND: decide a stream of hypotheses, each by its e-value, with FDR <= alpha. */
export class OnlineFDR extends BaseProcedure {
  readonly name = "online_fdr_elond";
  readonly version = "1";

  constructor(readonly alpha = 0.05, readonly horizon: number | null = null, readonly input = "log_e") {
    super();
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (horizon !== null && (!Number.isInteger(horizon) || horizon < 1)) invalid("horizon must be a positive int or None");
  }

  static fromConfig(c: Config): OnlineFDR {
    if (!("horizon" in c)) invalid("config is missing 'horizon'");
    const horizon = c.horizon === null ? null : num(c, "horizon");
    return new OnlineFDR(num(c, "alpha"), horizon, str(c, "input"));
  }

  config(): Config {
    return { alpha: this.alpha, horizon: this.horizon, input: this.input };
  }

  gamma(t: number): number {
    if (this.horizon === null) return defaultGamma(t);
    return t <= this.horizon ? 1.0 / this.horizon : 0.0;
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): OnlineFDRState {
    return { t: 0, discoveries: 0 };
  }

  /** alpha_t for the t-th hypothesis (t = 1, 2, ...) after `discoveries` discoveries. */
  level(t: number, discoveries: number): number {
    return this.alpha * this.gamma(t) * (discoveries + 1);
  }

  step(state: State, x: StatInput): [OnlineFDRState, State] {
    const s = state as OnlineFDRState;
    const t = s.t + 1;
    if (this.horizon !== null && t > this.horizon)
      throw new LifecycleError(`${this.name}: the series was planned for ${this.horizon} hypotheses`);
    const level = this.level(t, s.discoveries);
    const rejected = (x[this.input] as number) >= -Math.log(level);
    const discoveries = s.discoveries + (rejected ? 1 : 0);
    return [{ t, discoveries }, { t, level, rejected, discoveries }];
  }

  override isTerminal(state: State): boolean {
    return this.horizon !== null && (state as OnlineFDRState).t >= this.horizon;
  }

  decodeState(data: Json): OnlineFDRState {
    const s = decodeFields({ t: "count", discoveries: "count" }, data) as OnlineFDRState;
    requireState(s.discoveries <= s.t, "more discoveries than hypotheses");
    return s;
  }
}

register("online_fdr_elond", "1", (c) => OnlineFDR.fromConfig(c));
