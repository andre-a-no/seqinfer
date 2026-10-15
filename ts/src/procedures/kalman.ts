// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Kalman filter for the local-level model, optionally with a control input; mirrors procedures/kalman.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, invalid, num } from "./util.js";

type KalmanState = { n: number; mean: number; var: number };

export class LocalLevelKalman extends BaseProcedure {
  readonly name = "local_level_kalman";
  readonly version = "2";

  constructor(readonly q: number, readonly r: number, readonly m0 = 0.0, readonly p0 = 1.0, readonly control = false) {
    super();
    if (!(0 <= q && q < Infinity && 0 < r && r < Infinity && 0 <= p0 && p0 < Infinity && Number.isFinite(m0)))
      invalid("need q >= 0, r > 0 and p0 >= 0");
  }

  static fromConfig(c: Config): LocalLevelKalman {
    return new LocalLevelKalman(num(c, "q"), num(c, "r"), num(c, "m0"), num(c, "p0"), bool(c, "control"));
  }

  config(): Config {
    return { q: this.q, r: this.r, m0: this.m0, p0: this.p0, control: this.control };
  }

  inputs(): Record<string, InputContract> {
    const contracts: Record<string, InputContract> = { y: new InputContract("y") };
    if (this.control) contracts.u = new InputContract("u");
    return contracts;
  }

  initialState(): KalmanState {
    return { n: 0, mean: this.m0, var: this.p0 };
  }

  step(state: State, x: StatInput): [KalmanState, State] {
    const s = state as KalmanState;
    const mPred = s.mean + (this.control ? (x.u as number) : 0.0);
    const pPred = s.var + this.q;
    const innovation = (x.y as number) - mPred;
    const sv = pPred + this.r;
    const gain = pPred / sv;
    const mean = requireFinite(mPred + gain * innovation, "filtered mean");
    const v = (pPred * this.r) / sv;
    const n = s.n + 1;
    return [{ n, mean, var: v }, { n, mean, var: v, innovation, innovation_var: sv }];
  }

  decodeState(data: Json): KalmanState {
    const s = decodeFields({ n: "count", mean: "number", var: "number" }, data) as KalmanState;
    requireState(s.var >= 0, "negative variance");
    return s;
  }
}

register("local_level_kalman", "2", (c) => LocalLevelKalman.fromConfig(c));
