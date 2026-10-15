// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The TypeScript procedures against the conformance vectors of the Python library.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { type Json, sameJson } from "../src/canonical.js";
import { fromIdentity, identity, type Identity, type State } from "../src/procedure.js";
import "../src/procedures/index.js";
import { SplitMix64 } from "../src/rng.js";

interface Vector {
  case: string;
  procedure: Identity;
  seed: string | null;
  inputs: Record<string, Json>[];
  outputs: Json[];
  state: Json;
  rng: string | null;
  terminal: boolean;
}

const vectors = JSON.parse(readFileSync(new URL("../../../conformance/procedures.json", import.meta.url), "utf8")) as Vector[];

/** Compensation fields of a state and the running sums they belong to. */
const PARTNERS: Record<string, string[]> = {
  comp: ["llr", "total", "log_bf", "log_marginal", "log_sum", "log_likelihood", "log_evidence", "log_e"], c0: ["l0"], c1: ["l1"],
};

/** Largest relative difference of two JSON values; throws if they differ in anything but numbers. */
function difference(a: Json, b: Json, path: string): number {
  if (typeof a === "number" && typeof b === "number") {
    if (a === b) return 0;
    const scale = Math.max(Math.abs(a), Math.abs(b));
    return scale < 1e-300 ? 0 : Math.abs(a - b) / scale;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    assert.equal(a.length, b.length, `${path}: length`);
    return a.reduce<number>((m, v, i) => Math.max(m, difference(v, b[i], `${path}[${i}]`)), 0);
  }
  if (a !== null && b !== null && typeof a === "object" && typeof b === "object" && !Array.isArray(a) && !Array.isArray(b)) {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), `${path}: fields`);
    return Object.keys(a).reduce((m, k) => {
      const partner = PARTNERS[k]?.find((p) => p in b);
      const ak = a[k], bk = b[k], pk = partner === undefined ? null : b[partner];
      if (typeof ak === "number" && typeof bk === "number" && typeof pk === "number") {
        // A Neumaier compensation is the low part of its running sum: it is compared on the scale of that sum.
        const scale = Math.max(Math.abs(ak), Math.abs(bk), Math.abs(pk));
        return Math.max(m, scale < 1e-300 ? 0 : Math.abs(ak - bk) / scale);
      }
      return Math.max(m, difference(ak, bk, `${path}.${k}`));
    }, 0);
  }
  assert.equal(a, b, path);
  return 0;
}

const TOLERANCE = 1e-12;

for (const v of vectors) {
  test(`conformance: ${v.case}`, () => {
    const p = fromIdentity(v.procedure);
    assert.ok(sameJson(identity(p), v.procedure), "identity round-trips");
    const rng = p.randomized ? SplitMix64.forRole(BigInt(v.seed as string), "statistical") : null;
    let state: State = p.initialState();
    let exact = 0, worst = 0;
    const check = (got: Json, want: Json, what: string) => {
      if (sameJson(got, want)) exact++;
      else {
        const d = difference(got, want, what);
        assert.ok(d <= TOLERANCE, `${what}: relative difference ${d}`);
        worst = Math.max(worst, d);
      }
    };
    v.inputs.forEach((x, i) => {
      assert.ok(!p.isTerminal(state), `terminal before input ${i}`);
      const before = JSON.stringify(state);
      const [next, out] = p.step(state, x, rng);
      assert.equal(JSON.stringify(state), before, "step must not modify its state argument");
      check(out, v.outputs[i], `output ${i}`);
      state = next;
    });
    check(state, v.state, "state");
    assert.equal(rng === null ? null : rng.hex(), v.rng, "rng");
    assert.equal(p.isTerminal(state), v.terminal, "terminal");
    assert.ok(sameJson(p.decodeState(JSON.parse(JSON.stringify(v.state)) as Json), v.state), "decodeState round-trips");
    const total = v.outputs.length + 1;
    console.log(`${v.case.padEnd(30)} ${exact === total ? "exact" : `${exact}/${total} exact, max relative difference ${worst.toExponential(2)}`}`);
  });
}

test("decodeState rejects inconsistent states", () => {
  const sprt = fromIdentity(vectors.find((v) => v.procedure.name === "sprt")!.procedure);
  assert.throws(() => sprt.decodeState({ n: 0, llr: 1, comp: 0, decision: null }), /inconsistent state/);
  assert.throws(() => sprt.decodeState({ n: 1, llr: 0, comp: 0 }), /fields/);
  assert.throws(() => sprt.decodeState({ n: -1, llr: 0, comp: 0, decision: null }), /count/);
});
