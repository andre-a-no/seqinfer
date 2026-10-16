// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Thompson allocation (procedures/allocation.ts) and the propensity-weighted BoundedABTest: refusals, checks of
// decoded states, and the quadrature against a closed form.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Json } from "../src/canonical.js";
import { ContractViolation } from "../src/errors.js";
import { fromIdentity, identity, type State } from "../src/procedure.js";
import { BoundedABTest } from "../src/procedures/ab.js";
import { bestArmProbabilities, Posterior, ThompsonAllocation } from "../src/procedures/allocation.js";
import { lgamma } from "../src/procedures/distributions.js";

const logBeta = (x: number, y: number): number => lgamma(x) + lgamma(y) - lgamma(x + y);

/** Evan Miller's closed form of P(p_B > p_A) for p_A ~ Beta(aA, bA), p_B ~ Beta(aB, bB), aB an integer. */
function millerProbability(aA: number, bA: number, aB: number, bB: number): number {
  let total = 0.0;
  for (let i = 0; i < aB; i++)
    total += Math.exp(logBeta(aA + i, bA + bB) - Math.log(bB + i) - logBeta(1 + i, bB) - logBeta(aA, bA));
  return total;
}

function fold(proc: ThompsonAllocation, rows: { arm: number; y: number }[]): State {
  let state: State = proc.initialState();
  for (const x of rows) state = proc.step(state, x)[0];
  return state;
}

const ROWS = Array.from({ length: 60 }, (_, i) => ({ arm: i % 3, y: (i * 7919) % 100 < 20 + 10 * (i % 3) ? 1 : 0 }));

test("bestArmProbabilities matches Evan Miller's closed form for two Beta posteriors", () => {
  for (const [sA, nA, sB, nB] of [[40, 200, 55, 210], [3, 20, 5, 18], [120, 1000, 118, 1000], [0, 30, 2, 25]]) {
    // with the uniform prior, arm A is Beta(1 + sA, 1 + nA - sA) and arm B Beta(1 + sB, 1 + nB - sB)
    const a = new Posterior("binary", nA, sA / nA, 0.0, [1.0, 1.0]);
    const b = new Posterior("binary", nB, sB / nB, 0.0, [1.0, 1.0]);
    assert.equal(a.a, 1 + sA);
    assert.equal(b.b, 1 + nB - sB);
    const [best, loss] = bestArmProbabilities([a, b], "binary");
    const want = millerProbability(a.a, a.b, b.a, b.b);
    assert.ok(Math.abs(best[1] - want) <= 1e-10, `P(B best) ${best[1]} against ${want}`);
    assert.ok(Math.abs(best[0] + best[1] - 1.0) <= 1e-15);
    assert.ok(loss.every((v) => v >= 0.0));
  }
});

test("thompson_allocation round-trips its identity and keeps an equal split until every arm has min_units", () => {
  const proc = new ThompsonAllocation(3, "binary", 0.05, 0.05, 25, 7);
  const again = fromIdentity(identity(proc)) as ThompsonAllocation;
  assert.deepEqual(identity(again), identity(proc));
  let state: State = proc.initialState();
  for (const x of ROWS.slice(0, 74)) {
    const [next, out] = proc.step(state, x);
    if ((next.counts as number[]).some((c) => c < 25)) {
      assert.deepEqual(out.shares, [1 / 3, 1 / 3, 1 / 3]);
      assert.equal(out.leader, null);
    }
    state = next;
  }
});

test("thompson_allocation refuses invalid arguments", () => {
  const cases: [() => unknown, RegExp][] = [
    [() => new ThompsonAllocation(1), /arms must be an int >= 2/],
    [() => new ThompsonAllocation(2.5), /arms must be an int >= 2/],
    [() => new ThompsonAllocation(2, "poisson"), /metric must be one of/],
    [() => new ThompsonAllocation(2, "binary", 0.6), /floor must lie in \[0, 1\/arms\]/],
    [() => new ThompsonAllocation(2, "binary", -0.1), /floor must lie in \[0, 1\/arms\]/],
    [() => new ThompsonAllocation(2, "binary", 0.05, -1.0), /threshold must be a non-negative number/],
    [() => new ThompsonAllocation(2, "binary", 0.05, Infinity), /threshold must be a non-negative number/],
    [() => new ThompsonAllocation(2, "normal", 0.05, 0.0, 1), /min_units must be an int/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, -1), /min_units must be an int/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 0), /update_every must be a positive int/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [0.0, 1.0]), /the Beta prior needs positive parameters/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, Infinity]), /the Beta prior needs positive parameters/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, 1.0], null, 0.0, "y", "y"),
      /the arm and the outcome need different names/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, 1.0], null, 0.1), /a reserve needs a control arm/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, 1.0], 2), /control must be one of the arms/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, 1.0], -1), /control must be one of the arms/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, 1.0], 0.5), /control must be one of the arms/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, 1.0], 0, 0.95), /reserve must lie in \[0, 1 - arms \* floor\]/],
    [() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, 1.0], 0, -0.1), /reserve must lie in \[0, 1 - arms \* floor\]/],
  ];
  for (const [make, message] of cases) assert.throws(make, message);
  assert.doesNotThrow(() => new ThompsonAllocation(2, "binary", 0.5)); // floor = 1/arms is allowed
  assert.doesNotThrow(() => new ThompsonAllocation(2, "binary", 0.05, 0.0, 100, 100, [1.0, 1.0], 1, 0.9)); // reserve = 1 - K floor
  const config = new ThompsonAllocation().config();
  assert.equal(config.control, null);
  assert.equal(config.reserve, 0.0);
});

test("thompson_allocation gives the control arm its reserve on top of the Thompson shares", () => {
  const plain = new ThompsonAllocation(3, "binary", 0.05, 0.0, 5, 7);
  const reserved = new ThompsonAllocation(3, "binary", 0.05, 0.0, 5, 7, [1.0, 1.0], 0, 0.25);
  const sp = fold(plain, ROWS), sr = fold(reserved, ROWS);
  assert.deepEqual(sr.best, sp.best);
  const best = sr.best as number[], shares = sr.shares as number[];
  best.forEach((p, i) => assert.equal(shares[i], 0.05 + (1.0 - 3 * 0.05 - 0.25) * p + (i === 0 ? 0.25 : 0.0)));
  assert.ok(Math.abs(shares.reduce((a, b) => a + b, 0) - 1.0) < 1e-12);
  assert.deepEqual(reserved.decodeState(JSON.parse(JSON.stringify(sr)) as Json), sr);
});

test("thompson_allocation keeps the last split when the posteriors are not available", () => {
  // a normal metric whose arm 1 has no spread yet: the split is kept, and so are best, loss and deploy
  const proc = new ThompsonAllocation(2, "normal", 0.1, 0.0, 2, 3);
  const state: State = {
    n: 6, counts: [3, 3], means: [1.0, 2.0], m2: [2.0, 0.0], shares: [0.3, 0.7], since: 2, best: [0.25, 0.75], loss: [1.0, 0.1],
    deploy: 1,
  };
  proc.decodeState(state as Json);
  const [next, out] = proc.step(state, { arm: 1, y: 2.0 });
  assert.deepEqual(next, {
    n: 7, counts: [3, 4], means: [1.0, 2.0], m2: [2.0, 0.0], shares: [0.3, 0.7], since: 0, best: [0.25, 0.75], loss: [1.0, 0.1],
    deploy: 1,
  });
  assert.equal(out.leader, 1);
});

test("thompson_allocation decodeState refuses inconsistent states", () => {
  const proc = new ThompsonAllocation(3, "binary", 0.05, 0.05, 5, 7);
  const good = fold(proc, ROWS);
  assert.ok((good.best as number[]).length === 3, "the split has been computed");
  assert.deepEqual(proc.decodeState(JSON.parse(JSON.stringify(good)) as Json), good);
  const edit = (change: Partial<Record<string, Json>>): Json => ({ ...(JSON.parse(JSON.stringify(good)) as State), ...change } as Json);
  const counts = good.counts as number[], shares = good.shares as number[];
  const refusals: [Json, RegExp][] = [
    [edit({ counts: counts.slice(0, 2) }), /inconsistent state: one entry per arm/],
    [edit({ shares: [0.5, 0.5] }), /inconsistent state: one entry per arm/],
    [edit({ n: (good.n as number) + 1 }), /inconsistent state: counts/],
    [edit({ m2: [1.0, -0.5, 2.0] }), /inconsistent state: sums of squares/],
    [edit({ shares: [shares[0] + 0.01, shares[1], shares[2]] }), /inconsistent state: shares must sum to 1/],
    [edit({ shares: [1.2, -0.1, -0.1] }), /inconsistent state: shares must sum to 1/],
    [edit({ best: [0.5, 0.5] }), /inconsistent state: probabilities/],
    [edit({ loss: [] }), /inconsistent state: probabilities/],
    [edit({ since: (good.n as number) + 8 }), /inconsistent state: units since the last split/],
    [edit({ since: 8 }), /inconsistent state: units since the last split/],
    [edit({ deploy: 3 }), /inconsistent state: deployed arm/],
    [edit({ deploy: 0, best: [], loss: [] }), /inconsistent state: deployed arm/],
    [edit({ means: [0.5, 1.5, 0.2] }), /inconsistent state: binary means must lie in \[0, 1\]/],
    [edit({ means: [-0.1, 0.5, 0.2] }), /inconsistent state: binary means must lie in \[0, 1\]/],
    [edit({ counts: [1, -1, 0] }), /state field 'counts'/],
    [edit({ deploy: -1 }), /state field 'deploy'/],
    [edit({ means: [0.1, null, 0.2] }), /state field 'means'/],
    [{ ...(good as Record<string, Json>), extra: 1 }, /fields/],
    [[] as Json, /JSON object/],
  ];
  for (const [data, message] of refusals) assert.throws(() => proc.decodeState(data), message);
  // no split computed yet: best and loss both empty
  const fresh = proc.initialState();
  assert.deepEqual(proc.decodeState(fresh as Json), fresh);
  const one = { ...fresh, n: 1, counts: [1, 0, 0], means: [1.0, 0.0, 0.0], since: 1 };
  assert.deepEqual(proc.decodeState(one as Json), one);
  assert.throws(() => proc.decodeState({ ...one, m2: [0.25, 0.0, 0.0] } as Json), /inconsistent state: a spread from one unit/);
  assert.throws(() => proc.decodeState({ ...fresh, m2: [0.0, 0.5, 0.0] } as Json), /inconsistent state: a spread from one unit/);
  // means outside [0, 1] are fine for a normal metric
  const normal = new ThompsonAllocation(3, "normal", 0.05, 0.0, 2, 7);
  const big = { ...one, means: [12.5, 0.0, 0.0] };
  assert.deepEqual(normal.decodeState(big as Json), big);
  assert.throws(() => proc.decodeState(big as Json), /binary means must lie in \[0, 1\]/);
});

test("bounded_ab_test with propensities: refusals, the others count, and unchanged identities without them", () => {
  const make = (floor: number | null, armCount = 2, arms = [1, 0], allocation = 0.5, propensityInput: string | null = "p") =>
    new BoundedABTest(0.0, 1.0, allocation, 0.05, 0.0, "two-sided", arms, 50, false, "arm", "y", propensityInput, armCount, floor);
  const t = make(0.2);
  assert.throws(() => t.step(t.initialState(), { arm: 0, y: 0.5, p: [0.9, 0.1] }), ContractViolation);
  assert.throws(() => t.step(t.initialState(), { arm: 0, y: 0.5, p: [0.5, 0.4] }),
    (e: unknown) => e instanceof ContractViolation && /the propensities must sum to 1, got 0\.9/.test(e.message));
  assert.equal(t.inputs().arm.upper, 1);
  assert.equal(new BoundedABTest().inputs().arm.upper, null);
  assert.throws(() => make(null), /propensities need a floor in \(0, 1\/2\]/);
  assert.throws(() => make(0.6), /propensities need a floor in \(0, 1\/2\]/);
  assert.throws(() => make(0.1, 2, [1, 0], 0.5, null), /floor and arm_count come with propensity_input/);
  assert.throws(() => make(null, 3, [1, 0], 0.5, null), /floor and arm_count come with propensity_input/);
  assert.throws(() => make(0.2, 2, [1, 0], 0.3), /leave allocation alone/);
  assert.throws(() => make(0.2, 2, [3, 0]), /arm_count must be an int >= 2 covering the compared arms/);
  assert.throws(() => new BoundedABTest(0, 1, 0.5, 0.05, 0, "two-sided", [1, 0], 50, false, "arm", "y", "y", 2, 0.2),
    /the propensities need their own input name/);
  assert.ok(!("floor" in new BoundedABTest().config()));
  assert.equal(make(0.1).bound, 5.0);

  const three = make(0.1, 3, [2, 0]);
  const again = fromIdentity(identity(three)) as BoundedABTest;
  assert.deepEqual(identity(again), identity(three));
  assert.deepEqual(Object.keys(three.inputs()).sort(), ["arm", "p", "y"]);
  const [state, out] = three.step(three.initialState(), { arm: 1, y: 0.7, p: [0.3, 0.4, 0.3] });
  assert.equal(state.n, 1);
  assert.equal(state.others, 1);
  assert.deepEqual(out.means, [null, null]);
  assert.deepEqual(three.decodeState(JSON.parse(JSON.stringify(state)) as Json), state);
  const { others: _others, ...withoutOthers } = state;
  assert.throws(() => three.decodeState(withoutOthers as Json), /inconsistent state: fields/);
  assert.throws(() => three.decodeState({ ...state, others: 2 } as Json), /inconsistent state: counts/);
  assert.throws(() => new BoundedABTest(0, 1, 0.5, 0.05, 0, "two-sided", [1, 0], 50).decodeState({ ...state } as Json),
    /inconsistent state: fields/);
});
