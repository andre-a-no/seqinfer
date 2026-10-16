// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Delivery: per-source ordering and deduplication by sequence number (SPEC.md, "Delivery").
import type { Json } from "./canonical.js";
import { OrderingError } from "./errors.js";
import { complete, type Observation } from "./observation.js";

export type Policy = "arrival" | "sequence" | "strict";
const POLICIES: Policy[] = ["arrival", "sequence", "strict"];

export class Delivery {
  duplicates = 0;
  private next = new Map<string, number>();
  private ahead = new Map<string, Set<number>>();
  private pendingBySource = new Map<string, Map<number, Observation>>();
  private gaps = new Map<string, Map<number, number>>();

  constructor(
    readonly policy: Policy = "arrival",
    readonly dedup = true,
    readonly start = 0,
  ) {
    if (!POLICIES.includes(policy)) throw new Error(`unknown delivery policy ${policy}`);
    if (!Number.isInteger(start) || Math.abs(start) > 2 ** 53) throw new Error("start must be an integer within 2**53");
  }

  spec(): Json {
    return { policy: this.policy, dedup: this.dedup, start: this.start };
  }

  private position(source: string): number {
    return this.next.get(source) ?? this.start;
  }

  /** True if this observation was already delivered. */
  delivered(obs: Observation): boolean {
    const o = complete(obs);
    if (o.seq === null) return false;
    return o.seq < this.position(o.source) || (this.ahead.get(o.source)?.has(o.seq) ?? false);
  }

  positions(): Record<string, number> {
    return Object.fromEntries(this.next);
  }

  pending(): number {
    let n = 0;
    for (const p of this.pendingBySource.values()) n += p.size;
    return n;
  }

  private settle(source: string, nxt: number): number {
    const ahead = this.ahead.get(source) ?? new Set<number>();
    const gaps = this.gaps.get(source) ?? new Map<number, number>();
    for (;;) {
      if (ahead.has(nxt)) {
        ahead.delete(nxt);
        nxt += 1;
      } else if ([...gaps.keys()].some((f) => f <= nxt)) {
        for (const f of [...gaps.keys()].filter((f) => f <= nxt)) {
          nxt = Math.max(nxt, gaps.get(f)!);
          gaps.delete(f);
        }
        for (const n of [...ahead]) if (n < nxt) ahead.delete(n);
      } else return nxt;
    }
  }

  /** Declare that sequence numbers first..stop-1 of a source do not exist (arrival policy only). */
  skipRange(source: string, first: number, stop: number): void {
    if (this.policy !== "arrival") throw new OrderingError(`policy '${this.policy}' cannot pass over sequence numbers`);
    if (stop <= first) return;
    const nxt = this.position(source);
    if (stop <= nxt) return;
    if (first <= nxt) {
      const ahead = this.ahead.get(source);
      if (ahead) for (const n of [...ahead]) if (n < stop) ahead.delete(n); // passed over with the range (delivery.py)
      this.next.set(source, this.settle(source, stop));
    } else {
      const gaps = this.gaps.get(source) ?? new Map<number, number>();
      gaps.set(first, Math.max(stop, gaps.get(first) ?? stop));
      this.gaps.set(source, gaps);
    }
  }

  /** Pass over an observation that will never be delivered (arrival policy only). */
  skip(obs: Observation): void {
    if (this.policy !== "arrival") throw new OrderingError(`policy '${this.policy}' cannot pass over sequence numbers`);
    const o = complete(obs);
    if (o.seq === null || this.delivered(o)) return;
    const nxt = this.position(o.source);
    if (o.seq === nxt) this.next.set(o.source, this.settle(o.source, nxt + 1));
    else this.aheadOf(o.source).add(o.seq);
  }

  private aheadOf(source: string): Set<number> {
    let a = this.ahead.get(source);
    if (!a) this.ahead.set(source, (a = new Set()));
    return a;
  }

  private duplicate(o: Required<Observation>): Observation[] {
    if (!this.dedup) throw new OrderingError(`source '${o.source}': sequence number ${o.seq} was already delivered`);
    this.duplicates += 1;
    return [];
  }

  /** Accept one arriving observation; return those now deliverable, in order. */
  push(obs: Observation): Observation[] {
    const o = complete(obs);
    if (o.seq === null) {
      if (this.policy !== "arrival") throw new OrderingError(`policy '${this.policy}' needs sequence numbers (source '${o.source}')`);
      return [obs];
    }
    const src = o.source, seq = o.seq;
    let nxt = this.position(src);
    if (this.policy === "arrival") {
      const ahead = this.aheadOf(src);
      if (seq < nxt || ahead.has(seq)) {
        if (this.dedup) {
          this.duplicates += 1;
          return [];
        }
        return [obs];
      }
      if (seq === nxt) nxt = this.settle(src, nxt + 1);
      else ahead.add(seq);
      this.next.set(src, nxt);
      return [obs];
    }
    if (seq < nxt) return this.duplicate(o);
    if (this.policy === "strict") {
      if (seq !== nxt) throw new OrderingError(`source '${src}': expected sequence number ${nxt}, got ${seq}`);
      this.next.set(src, nxt + 1);
      return [obs];
    }
    let pending = this.pendingBySource.get(src);
    if (!pending) this.pendingBySource.set(src, (pending = new Map()));
    if (pending.has(seq)) return this.duplicate(o);
    pending.set(seq, obs);
    const out: Observation[] = [];
    while (pending.has(nxt)) {
      out.push(pending.get(nxt)!);
      pending.delete(nxt);
      nxt += 1;
    }
    this.next.set(src, nxt);
    return out;
  }

  state(): Json {
    const ahead: Record<string, number[]> = {};
    for (const [s, a] of this.ahead) if (a.size) ahead[s] = [...a].sort((x, y) => x - y);
    return { next: Object.fromEntries(this.next), ahead, duplicates: this.duplicates };
  }

  restore(state: Json): void {
    const s = state as { next: Record<string, unknown>; ahead?: Record<string, unknown[]>; duplicates?: unknown };
    const position = (n: unknown): number => {
      if (typeof n !== "number" || !Number.isInteger(n)) throw new Error(`sequence position ${JSON.stringify(n)} is not an integer`);
      return n;
    };
    this.next = new Map(Object.entries(s.next).map(([k, v]) => [k, position(v)]));
    this.ahead = new Map(Object.entries(s.ahead ?? {}).map(([k, v]) => [k, new Set(v.map(position))]));
    this.pendingBySource = new Map();
    this.gaps = new Map();
    this.duplicates = position(s.duplicates ?? 0);
    if (this.duplicates < 0) throw new Error("negative duplicate count");
  }
}
