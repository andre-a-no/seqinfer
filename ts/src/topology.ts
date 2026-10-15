// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Data topology: how observations become statistical inputs (SPEC.md, "Topologies").
import type { Json } from "./canonical.js";
import type { StatInput } from "./contracts.js";
import { ContractViolation, OrderingError } from "./errors.js";
import { complete, type Observation } from "./observation.js";

/** Stands for an input a transformation could not map; the run reports and skips it. */
export class InvalidInput {
  constructor(readonly violation: ContractViolation) {}
}

export interface Topology {
  spec(): Json;
  push(obs: Observation): (StatInput | InvalidInput)[];
  state(): Json;
  restore(state: Json): void;
}

export class Independent implements Topology {
  spec(): Json {
    return { type: "independent" };
  }
  push(obs: Observation): StatInput[] {
    return [{ [obs.input]: obs.value }];
  }
  state(): Json {
    return null;
  }
  restore(): void {}
}

abstract class Buffered {
  readonly inputs: string[];
  constructor(inputs: string[]) {
    if (inputs.length < 2 || new Set(inputs).size !== inputs.length) throw new Error("need at least two distinct logical inputs");
    this.inputs = [...inputs];
  }
  protected check(obs: Observation): void {
    if (!this.inputs.includes(obs.input)) throw new ContractViolation(`observation for input '${obs.input}' is not part of this topology`);
  }
}

/** Pair the i-th observation of every input. */
export class PositionalPair extends Buffered implements Topology {
  private queues: Record<string, Json[]> = {};
  constructor(inputs: string[]) {
    super(inputs);
    for (const n of this.inputs) this.queues[n] = [];
  }
  spec(): Json {
    return { type: "positional_pair", inputs: this.inputs };
  }
  push(obs: Observation): StatInput[] {
    this.check(obs);
    this.queues[obs.input].push(obs.value);
    const out: StatInput[] = [];
    while (this.inputs.every((n) => this.queues[n].length)) {
      const x: StatInput = {};
      for (const n of this.inputs) x[n] = this.queues[n].shift()!;
      out.push(x);
    }
    return out;
  }
  state(): Json {
    return Object.fromEntries(this.inputs.map((n) => [n, [...this.queues[n]]]));
  }
  restore(state: Json): void {
    const s = state as Record<string, Json[]>;
    this.queues = Object.fromEntries(this.inputs.map((n) => {
      if (!Array.isArray(s[n])) throw new Error(`no queue for input '${n}'`);
      return [n, [...s[n]]];
    }));
  }
}

/** Join observations that carry the same key; emits in completion order. */
export class KeyJoin extends Buffered implements Topology {
  private partial = new Map<string | number, Record<string, Json>>();
  spec(): Json {
    return { type: "key_join", inputs: this.inputs };
  }
  push(obs: Observation): StatInput[] {
    this.check(obs);
    const key = complete(obs).key;
    if (typeof key !== "string" && typeof key !== "number")
      throw new ContractViolation(`key join needs a str or int key on every observation, got ${JSON.stringify(key)}`);
    let row = this.partial.get(key);
    if (!row) this.partial.set(key, (row = {}));
    if (obs.input in row) throw new OrderingError(`key ${JSON.stringify(key)}: input '${obs.input}' was observed twice`);
    row[obs.input] = obs.value;
    if (Object.keys(row).length < this.inputs.length) return [];
    this.partial.delete(key);
    return [Object.fromEntries(this.inputs.map((n) => [n, row![n]]))];
  }
  state(): Json {
    return [...this.partial].map(([k, row]) => [k, { ...row }]);
  }
  restore(state: Json): void {
    this.partial = new Map((state as [string | number, Record<string, Json>][]).map(([k, row]) => [k, { ...row }]));
  }
}

/** Align two time-ordered inputs within a tolerance. */
export class TimeAlign extends Buffered implements Topology {
  unmatched = 0;
  private queues: Record<string, [number, Json][]> = {};
  private last: Record<string, number | null> = {};
  constructor(inputs: string[], readonly tolerance: number) {
    super(inputs);
    if (this.inputs.length !== 2) throw new Error("time alignment is defined for exactly two inputs");
    if (!(Number.isFinite(tolerance) && tolerance >= 0)) throw new Error("tolerance must be a finite non-negative number");
    for (const n of this.inputs) {
      this.queues[n] = [];
      this.last[n] = null;
    }
  }
  spec(): Json {
    return { type: "time_align", inputs: this.inputs, tolerance: this.tolerance };
  }
  push(obs: Observation): StatInput[] {
    this.check(obs);
    const time = complete(obs).time;
    if (time === null) throw new ContractViolation("time alignment needs an event time on every observation");
    const last = this.last[obs.input];
    if (last !== null && time < last) throw new OrderingError(`input '${obs.input}': event time went backwards (${time} < ${last})`);
    this.last[obs.input] = time;
    this.queues[obs.input].push([time, obs.value]);
    const [a, b] = this.inputs;
    const qa = this.queues[a], qb = this.queues[b];
    const out: StatInput[] = [];
    while (qa.length && qb.length) {
      const [ta, va] = qa[0], [tb, vb] = qb[0];
      if (Math.abs(ta - tb) <= this.tolerance) {
        qa.shift();
        qb.shift();
        out.push({ [a]: va, [b]: vb });
      } else if (ta < tb) {
        qa.shift();
        this.unmatched += 1;
      } else {
        qb.shift();
        this.unmatched += 1;
      }
    }
    return out;
  }
  state(): Json {
    return {
      queues: Object.fromEntries(this.inputs.map((n) => [n, this.queues[n].map(([t, v]) => [t, v])])),
      last: { ...this.last },
      unmatched: this.unmatched,
    };
  }
  restore(state: Json): void {
    const s = state as { queues: Record<string, [number, Json][]>; last: Record<string, number | null>; unmatched: number };
    this.queues = Object.fromEntries(this.inputs.map((n) => [n, s.queues[n].map(([t, v]) => [t, v] as [number, Json])]));
    this.last = Object.fromEntries(this.inputs.map((n) => [n, s.last[n]]));
    this.unmatched = s.unmatched;
  }
}

/** A topology followed by the difference transform {a, b} -> {out: a - b} (Python: topology.map(difference(a, b, out))). */
export class Difference implements Topology {
  constructor(readonly inner: Topology, readonly a: string, readonly b: string, readonly out = "x") {}
  spec(): Json {
    return { type: "mapped", map: { name: "difference", version: "1", config: { a: this.a, b: this.b, out: this.out } }, inner: this.inner.spec() };
  }
  push(obs: Observation): (StatInput | InvalidInput)[] {
    return this.inner.push(obs).map((x) => {
      if (x instanceof InvalidInput) return x;
      const u = x[this.a], v = x[this.b];
      if (typeof u !== "number" || typeof v !== "number")
        return new InvalidInput(new ContractViolation(`difference failed on ${JSON.stringify(x)}: not numbers`));
      return { [this.out]: u - v };
    });
  }
  state(): Json {
    return this.inner.state();
  }
  restore(state: Json): void {
    this.inner.restore(state);
  }
}

/** The topology a spec describes (the built-in ones; a mapped topology only with the difference transform). */
export function topologyFromSpec(spec: Json): Topology {
  const s = spec as Record<string, Json>;
  switch (s.type) {
    case "independent":
      return new Independent();
    case "positional_pair":
      return new PositionalPair(s.inputs as string[]);
    case "key_join":
      return new KeyJoin(s.inputs as string[]);
    case "time_align":
      return new TimeAlign(s.inputs as string[], s.tolerance as number);
    case "mapped": {
      const map = s.map as { name: string; version: string; config: { a: string; b: string; out: string } };
      if (map.name !== "difference" || map.version !== "1") throw new Error(`transform ${map.name} has no TypeScript implementation`);
      return new Difference(topologyFromSpec(s.inner), map.config.a, map.config.b, map.config.out);
    }
    default:
      throw new Error(`unknown topology ${JSON.stringify(s.type)}`);
  }
}
