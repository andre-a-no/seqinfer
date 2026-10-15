// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Execute a precomputed sequential sampling plan; mirrors procedures/plan.py.
import type { Json } from "../canonical.js";
import { InputContract, type Kind, type StatInput } from "../contracts.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, refuseTerminal, register, requireState, type State } from "../procedure.js";
import { ACCEPT_H0, BaseProcedure, bool, type Config, type Decision, invalid, obj, REJECT_H0, sameState, str } from "./util.js";

type PlanState = { n: number; total: number; comp: number; decision: Decision };
type Bound = number | null;

/** An InputContract from its spec() form. */
export function contractFromSpec(spec: Config): InputContract {
  const kind = str(spec, "kind");
  if (!["real", "integer", "binary"].includes(kind)) invalid(`unknown kind '${kind}'`);
  const shape = spec.shape;
  if (!Array.isArray(shape) || shape.length > 1 || !shape.every((d) => typeof d === "number" && Number.isInteger(d) && d >= 1))
    invalid("shape must be [] or [d]");
  const bound = (k: string): number | null => {
    const v = spec[k] ?? null;
    if (v !== null && (typeof v !== "number" || !Number.isFinite(v))) invalid(`bounds must be finite numbers or null`);
    return v as number | null;
  };
  const unit = spec.unit ?? null;
  if (unit !== null && typeof unit !== "string") invalid("unit must be a string or null");
  return new InputContract(str(spec, "name"), kind as Kind, shape as number[], bound("lower"), bound("upper"), unit as string | null);
}

export class PlanTest extends BaseProcedure {
  readonly name = "plan_test";
  readonly version = "1";
  readonly plan: [Bound, Bound][];
  readonly contract: InputContract;

  constructor(
    continuation: Bound[][],
    contract: InputContract | null = null,
    readonly rejectHigh = true,
    readonly label = "",
    readonly input = "x",
  ) {
    super();
    this.plan = continuation.map((stage, i) => {
      if (stage.length !== 2) invalid(`step ${i + 1}: expected (lower, upper)`);
      for (const b of stage)
        if (b !== null && !Number.isFinite(b)) invalid(`step ${i + 1}: use None for a missing boundary, not ${b}`);
      return [stage[0], stage[1]];
    });
    if (!this.plan.length) invalid("a plan needs at least one step");
    const [lower, upper] = this.plan[this.plan.length - 1];
    if (lower === null || upper === null || lower < upper) invalid("the last step must force a decision: it needs lower >= upper");
    this.contract = contract ?? new InputContract(input);
  }

  static fromConfig(c: Config): PlanTest {
    const cont = c.continuation;
    if (!Array.isArray(cont) || !cont.every((s) => Array.isArray(s) && s.every((b) => b === null || typeof b === "number")))
      invalid("continuation must be a list of [lower, upper] pairs");
    return new PlanTest(cont as Bound[][], contractFromSpec(obj(c, "contract")), bool(c, "reject_high"), str(c, "label"), str(c, "input"));
  }

  get horizon(): number {
    return this.plan.length;
  }

  config(): Config {
    return {
      continuation: this.plan.map((s) => [s[0], s[1]]), reject_high: this.rejectHigh, label: this.label,
      input: this.input, contract: this.contract.spec(),
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.contract };
  }

  initialState(): PlanState {
    return { n: 0, total: 0.0, comp: 0.0, decision: null };
  }

  step(state: State, x: StatInput): [PlanState, State] {
    refuseTerminal(this, state);
    const s = state as PlanState;
    const [total, comp] = neumaierAdd(s.total, s.comp, x[this.input] as number);
    const sum = requireFinite(total + comp, "running sum");
    const n = s.n + 1;
    const decision = this.decide(n, sum);
    return [{ n, total, comp, decision }, { n, statistic: sum, decision }];
  }

  private decide(n: number, s: number): Decision {
    const [lower, upper] = this.plan[n - 1];
    const [low, high] = this.rejectHigh ? [ACCEPT_H0, REJECT_H0] as const : [REJECT_H0, ACCEPT_H0] as const;
    if (lower !== null && s <= lower) return low;
    if (upper !== null && s >= upper) return high;
    return null;
  }

  override isTerminal(state: State): boolean {
    return state.decision !== null;
  }

  decodeState(data: Json): PlanState {
    const s = decodeFields({ n: "count", total: "number", comp: "number", decision: "decision" }, data) as PlanState;
    requireState(s.n <= this.horizon, `${s.n} steps in a plan of ${this.horizon}`);
    if (s.n === 0) requireState(sameState(s, this.initialState()), "no observations, but a statistic or decision");
    else requireState(s.decision === this.decide(s.n, s.total + s.comp), "decision does not match the running sum");
    return s;
  }
}

register("plan_test", "1", (c) => PlanTest.fromConfig(c));
