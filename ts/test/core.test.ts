// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The executor's core against values computed by the Python library.
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  Delivery,
  IncompatibleCheckpoint,
  KeyJoin,
  ObservationLog,
  OrderingError,
  Run,
  SplitMix64,
  TimeAlign,
  canonicalJson,
  checkpointId,
  fromIdentity,
  restoreFromCheckpoint,
} from "../src/index.js";

const mixture = (tau = 0.5) =>
  fromIdentity({ name: "normal_mixture_sprt", version: "1", config: { sigma: 1, tau, theta0: 0, alpha: 0.05, stop_on_reject: false, input: "x" } });
const make = () => mixture();

const root = new URL("../../../", import.meta.url);
const numbers = JSON.parse(readFileSync(new URL("conformance/numbers.json", root), "utf8")) as { bits: string; json: string }[];

test("canonical JSON: number formatting of 500 binary64 values, keys, escapes", () => {
  const view = new DataView(new ArrayBuffer(8));
  for (const { bits, json } of numbers) {
    view.setBigUint64(0, BigInt("0x" + bits));
    assert.equal(canonicalJson(view.getFloat64(0)), json, bits);
  }
  // the Python library's output for the same value
  assert.equal(
    canonicalJson({ b: [1.0, 2.5, 1e21, 1e-7, -0], a: "é \"", "\u{1F600}": null, z: true }),
    '{"a":"é \\"","b":[1,2.5,1e+21,1e-7,0],"z":true,"\u{1F600}":null}',
  );
  assert.throws(() => canonicalJson({ x: NaN }), RangeError);
  assert.throws(() => canonicalJson("\uD800"), RangeError);
});

test("SplitMix64 reproduces the Python generator", () => {
  const r = SplitMix64.forRole(12345, "statistical");
  assert.equal(r.hex(), "729cf85899dccae9");
  assert.deepEqual([r.nextU64(), r.nextU64(), r.nextU64()], [6339891518893877478n, 3130633060930783416n, 9839606651724196377n]);
  assert.deepEqual([r.random(), r.random(), r.random()], [0.8365473952036482, 0.30554751342492814, 0.00017197501380572522]);
  assert.ok(Math.abs(r.normal() - -2.002692249038914) <= 4e-16 * 2.0027);
  assert.equal(r.hex(), "6458c6249430ab91");
});

test("delivery policies", () => {
  const d = new Delivery("sequence");
  assert.deepEqual(d.push({ input: "x", value: 1, seq: 1 }), []);
  assert.deepEqual(d.push({ input: "x", value: 0, seq: 0 }).map((o) => o.seq), [0, 1]);
  assert.deepEqual(d.push({ input: "x", value: 0, seq: 0 }), []);
  assert.equal(d.duplicates, 1);
  assert.throws(() => new Delivery("strict").push({ input: "x", value: 0, seq: 3 }), OrderingError);
  const a = new Delivery();
  a.push({ input: "x", value: 0, seq: 0 });
  a.skipRange("x", 3, 10);
  a.push({ input: "x", value: 0, seq: 2 });
  a.push({ input: "x", value: 0, seq: 1 });
  assert.deepEqual(a.state(), { next: { x: 10 }, ahead: {}, duplicates: 0 });
});

test("topologies", () => {
  const k = new KeyJoin(["arm", "y"]);
  assert.deepEqual(k.push({ input: "y", value: 1, key: "u1" }), []);
  assert.deepEqual(k.state(), [["u1", { y: 1 }]]);
  assert.deepEqual(k.push({ input: "arm", value: 0, key: "u1" }), [{ arm: 0, y: 1 }]);
  const t = new TimeAlign(["a", "b"], 0.5);
  t.push({ input: "a", value: 1, time: 0 });
  assert.deepEqual(t.push({ input: "b", value: 2, time: 0.3 }), [{ a: 1, b: 2 }]);
  t.push({ input: "a", value: 1, time: 1 });
  t.push({ input: "b", value: 2, time: 3 });
  assert.equal(t.unmatched, 1);
});

test("a run checkpoints, refuses a mismatched restore, and continues exactly", () => {
  const obs = Array.from({ length: 60 }, (_, i) => ({ input: "x", value: Math.sin(i) + 0.3, seq: i }));
  const full = new Run(make(), { runId: "r", clock: () => 0 });
  full.offerBatch(obs);
  const first = new Run(make(), { runId: "r", clock: () => 0 });
  first.offerBatch(obs.slice(0, 25));
  const cp = JSON.parse(JSON.stringify(first.checkpoint()));
  const second = restoreFromCheckpoint(cp, { clock: () => 0 });
  second.start();
  second.offerBatch(obs.slice(20)); // a source that rewinds too far: duplicates are dropped
  assert.equal(second.historyDigest, full.historyDigest);
  assert.deepEqual(second.state, full.state);
  assert.equal(second.delivery.duplicates, 5);
  assert.throws(() => Run.restore(mixture(0.6), cp), IncompatibleCheckpoint);
  const edited = { ...cp, t: 3 };
  assert.notEqual(checkpointId(edited), checkpointId(cp));
  assert.throws(() => restoreFromCheckpoint({ ...cp, format: "x" }), IncompatibleCheckpoint);
});

test("observation log: hash chain, verification and cut-back on restore", () => {
  const dir = mkdtempSync(join(tmpdir(), "sa-"));
  const path = join(dir, "log.jsonl");
  const run = new Run(make(), { runId: "r", log: new ObservationLog(path), clock: () => 0 });
  for (let i = 0; i < 10; i++) run.offer({ input: "x", value: i / 7, seq: i });
  const cp = run.checkpoint();
  for (let i = 10; i < 15; i++) run.offer({ input: "x", value: i / 7, seq: i });
  run.close();
  const restored = restoreFromCheckpoint(cp, { clock: () => 0 });
  assert.equal(restored.log!.state() && (restored.log!.state() as { count: number }).count, 10);
  const events = restored.provenance.events as { event: string; log_discarded?: number }[];
  assert.equal(events.at(-1)!.log_discarded, 5);
  writeFileSync(path, readFileSync(path, "utf8").replace('"value": 0', '"value": 9'));
  assert.throws(() => restoreFromCheckpoint(cp), IncompatibleCheckpoint);
});
