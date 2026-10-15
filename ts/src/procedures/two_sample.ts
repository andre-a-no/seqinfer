// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Running difference of two independent sample means; mirrors procedures/two_sample.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, type Config, str } from "./util.js";

type MDState = { n_a: number; mean_a: number; ss_a: number; n_b: number; mean_b: number; ss_b: number };

function welford(n: number, mean: number, ss: number, x: number): [number, number, number] {
  n += 1;
  const delta = x - mean;
  mean += delta / n;
  ss += delta * (x - mean);
  return [n, mean, ss];
}

export class MeanDifference extends BaseProcedure {
  readonly name = "mean_difference";
  readonly version = "1";
  override readonly jointInputs = false;

  constructor(readonly a = "a", readonly b = "b") {
    super();
  }

  static fromConfig(c: Config): MeanDifference {
    return new MeanDifference(str(c, "a"), str(c, "b"));
  }

  config(): Config {
    return { a: this.a, b: this.b };
  }

  inputs(): Record<string, InputContract> {
    return { [this.a]: new InputContract(this.a), [this.b]: new InputContract(this.b) };
  }

  initialState(): MDState {
    return { n_a: 0, mean_a: 0.0, ss_a: 0.0, n_b: 0, mean_b: 0.0, ss_b: 0.0 };
  }

  step(state: State, x: StatInput): [MDState, State] {
    const s = state as MDState;
    let { n_a, mean_a, ss_a, n_b, mean_b, ss_b } = s;
    if (this.a in x) [n_a, mean_a, ss_a] = welford(n_a, mean_a, ss_a, x[this.a] as number);
    if (this.b in x) [n_b, mean_b, ss_b] = welford(n_b, mean_b, ss_b, x[this.b] as number);
    const difference = n_a && n_b ? mean_a - mean_b : null;
    const std_error = n_a > 1 && n_b > 1 ? Math.sqrt(ss_a / (n_a - 1) / n_a + ss_b / (n_b - 1) / n_b) : null;
    return [{ n_a, mean_a, ss_a, n_b, mean_b, ss_b }, { n_a, n_b, difference, std_error }];
  }

  decodeState(data: Json): MDState {
    const s = decodeFields({ n_a: "count", mean_a: "number", ss_a: "number", n_b: "count", mean_b: "number", ss_b: "number" }, data) as MDState;
    requireState(s.ss_a >= 0 && s.ss_b >= 0, "negative sum of squares");
    for (const [n, mean, ss] of [[s.n_a, s.mean_a, s.ss_a], [s.n_b, s.mean_b, s.ss_b]])
      requireState(n > 0 || (mean === 0 && ss === 0), "statistics without observations");
    return s;
  }
}

register("mean_difference", "1", (c) => MeanDifference.fromConfig(c));
