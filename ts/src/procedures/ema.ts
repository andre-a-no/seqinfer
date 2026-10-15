// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Exponential moving average; mirrors procedures/ema.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, type Config, invalid, num, str } from "./util.js";

type EMAState = { n: number; value: number };

export class EMA extends BaseProcedure {
  readonly name = "ema";
  readonly version = "1";

  constructor(readonly alpha: number, readonly input = "x") {
    super();
    if (!(0 < alpha && alpha <= 1)) invalid("alpha must lie in (0, 1]");
  }

  static fromConfig(c: Config): EMA {
    return new EMA(num(c, "alpha"), str(c, "input"));
  }

  config(): Config {
    return { alpha: this.alpha, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): EMAState {
    return { n: 0, value: 0.0 };
  }

  step(state: State, x: StatInput): [EMAState, State] {
    const s = state as EMAState;
    const v = x[this.input] as number;
    if (s.n === 0) return [{ n: 1, value: v }, { n: 1, value: v, residual: 0.0 }];
    const value = requireFinite(s.value + this.alpha * (v - s.value), "moving average");
    return [{ n: s.n + 1, value }, { n: s.n + 1, value, residual: v - s.value }];
  }

  decodeState(data: Json): EMAState {
    const s = decodeFields({ n: "count", value: "number" }, data) as EMAState;
    requireState(s.n > 0 || s.value === 0, "an average without observations");
    return s;
  }
}

register("ema", "1", (c) => EMA.fromConfig(c));
