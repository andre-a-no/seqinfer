// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The ratio-metric A/B test (procedures/ratio.ts) against the Python library and against its definition.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Json } from "../src/canonical.js";
import { ContractViolation } from "../src/errors.js";
import { fromIdentity, type State } from "../src/procedure.js";
import { RatioABTest } from "../src/procedures/ratio.js";

const ROWS = [
  { arm: 1, y: 3.5, d: 2.0 }, { arm: 0, y: 0.0, d: 0.0 }, { arm: 0, y: 1.25, d: 1.0 }, { arm: 2, y: 9.0, d: 1.0 },
  { arm: 1, y: 7.0, d: 3.0 }, { arm: 0, y: 4.5, d: 2.0 }, { arm: 1, y: 0.5, d: 1.0 }, { arm: 0, y: 2.0, d: 3.0 },
  { arm: 1, y: 12.0, d: 4.0 }, { arm: 0, y: 0.75, d: 1.0 }, { arm: 1, y: 6.0, d: 2.0 }, { arm: 0, y: 1.0, d: 2.0 },
];

// RatioABTest(start=6, tuned_for=10, scale="relative") of the Python library on ROWS: the last output and state
const CONFIG = {
  alpha: 0.05, denominator: true, scale: "relative", effect0: 0.0, arms: [1, 0], start: 6, tuned_for: 10.0, stop_on_reject: false,
  arm_input: "arm", input: "y", denominator_input: "d",
};
const LAST_OUTPUT = {
  n: 11, units: [5, 6], ratios: [2.4166666666666665, 1.0555555555555556], difference: 1.361111111111111, std_error: 0.4815417593441163,
  lower: -0.10055690149716345, upper: 2.8227791237193856, lift: 1.289473684210526, lift_lower: -0.2350825920238283,
  lift_upper: 5.852621859608411, decision: null,
};
const LAST_STATE = {
  units: [5, 6], sum_y: [29.0, 9.5], sum_d: [12.0, 9.0], cyy: [73.3, 12.333333333333334], cdd: [5.2, 5.5], cyd: [18.9, 4.75], decision: null,
};

function fold(proc: RatioABTest, rows: Record<string, number>[]): [State, State[]] {
  let state: State = proc.initialState();
  const outs: State[] = [];
  for (const x of rows) {
    const [next, out] = proc.step(state, x);
    state = next;
    outs.push(out);
  }
  return [state, outs];
}

function close(a: Json, b: Json, path = ""): void {
  if (typeof a === "number" && typeof b === "number") {
    assert.ok(Math.abs(a - b) <= 1e-12 * Math.max(Math.abs(a), Math.abs(b)), `${path}: ${a} != ${b}`);
  } else if (Array.isArray(a) && Array.isArray(b)) {
    assert.equal(a.length, b.length, path);
    a.forEach((v, i) => close(v, b[i], `${path}[${i}]`));
  } else if (a !== null && b !== null && typeof a === "object" && typeof b === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), path);
    for (const k of Object.keys(a)) close((a as Record<string, Json>)[k], (b as Record<string, Json>)[k], `${path}.${k}`);
  } else {
    assert.equal(a, b, path);
  }
}

test("ratio_ab_test reproduces the Python library", () => {
  const proc = fromIdentity({ name: "ratio_ab_test", version: "1", config: CONFIG }) as RatioABTest;
  assert.deepEqual(proc.config(), CONFIG);
  const [state, outs] = fold(proc, ROWS);
  assert.deepEqual(state, LAST_STATE); // +, -, *, /: bit for bit
  close(outs[outs.length - 1] as Json, LAST_OUTPUT as Json);
  assert.deepEqual(proc.decodeState(JSON.parse(JSON.stringify(state))), state);
});

test("ratio_ab_test: the estimates are the ratios of sums, users of other arms are ignored", () => {
  const proc = new RatioABTest(0.05, true, "difference", 0.0, [1, 0], 2);
  const [, outs] = fold(proc, ROWS);
  const sums = { 1: [0, 0], 0: [0, 0] } as Record<number, number[]>;
  ROWS.forEach((x, i) => {
    if (x.arm in sums) {
      sums[x.arm][0] += x.y;
      sums[x.arm][1] += x.d;
    }
    const ratios = [1, 0].map((a) => (sums[a][1] > 0 ? sums[a][0] / sums[a][1] : null));
    assert.deepEqual(outs[i].ratios, ratios);
  });
  assert.deepEqual(outs[3], outs[2]); // arm 2
});

test("ratio_ab_test refuses bad denominators, bad states and bad arguments", () => {
  const proc = new RatioABTest();
  for (const d of [-1.0, Infinity, NaN]) assert.throws(() => proc.step(proc.initialState(), { arm: 1, y: 1.0, d }), ContractViolation);
  for (const broken of [{ ...LAST_STATE, units: [5] }, { ...LAST_STATE, cyy: [-1, 0] }, { ...LAST_STATE, units: [0, 6] }, { ...LAST_STATE, extra: 1 }])
    assert.throws(() => proc.decodeState(broken as Json));
  assert.throws(() => new RatioABTest(0.05, false).decodeState(LAST_STATE as Json));
  assert.throws(() => new RatioABTest(0.05, true, "log"), RangeError);
  assert.throws(() => new RatioABTest(0.05, true, "relative", -1.0), RangeError);
  assert.throws(() => new RatioABTest(0.05, true, "difference", 0.0, [1, 1]), RangeError);
  assert.throws(() => new RatioABTest(0.05, true, "difference", 0.0, [1, 0], 1), RangeError);
});
