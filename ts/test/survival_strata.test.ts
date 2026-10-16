// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The stratified anytime-valid log-rank test (AnytimeLogRankTest with strata > 1) against the Python library.
import assert from "node:assert/strict";
import { test } from "node:test";
import { type Json, sameJson } from "../src/canonical.js";
import { ContractViolation } from "../src/errors.js";
import { fromIdentity, identity, type State } from "../src/procedure.js";
import { AnytimeLogRankTest } from "../src/procedures/survival.js";

const ENTER = 0, EVENT = 1, CENSOR = 2;

/** [arm, kind, stratum] records of a three-centre trial (tests/test_survival.py centre_trial, seed 31), after one
 * record of an arm that is not tested. */
const INPUTS: number[][] =
  [[2,1,2],[0,0,1],[0,0,0],[0,0,1],[0,0,0],[0,0,1],[0,0,1],[0,0,0],[1,0,2],[1,0,2],[1,0,0],[1,0,1],[0,0,1],[1,0,0],[1,0,
  0],[1,0,1],[1,0,0],[1,1,1],[0,1,1],[1,0,2],[0,0,1],[0,0,2],[0,0,0],[0,0,0],[0,1,1],[1,0,1],[0,1,2],[1,0,1],[0,0,2],[1,
  0,0],[1,0,1],[1,1,2],[1,0,2],[0,2,0],[0,0,0],[0,0,1],[0,0,0],[0,0,1],[1,0,0],[0,2,0],[0,2,1],[0,2,0],[0,0,0],[0,2,0],
  [1,0,1],[0,0,0],[0,1,2],[1,1,2],[0,0,2],[1,1,2],[1,1,1],[0,1,2],[0,0,1],[0,0,1],[1,0,2],[0,2,1],[1,0,0],[0,0,0],[1,0,
  1],[0,0,2],[0,1,0],[0,2,1],[0,0,0],[0,1,1],[0,2,0],[0,2,0],[0,0,1],[0,1,1],[0,1,0],[0,0,2],[1,0,0],[1,0,1],[0,1,2],[0,
  2,0],[1,1,1],[1,0,2],[1,1,2],[0,0,1],[0,0,1],[1,0,2],[1,0,0],[1,1,1],[1,0,0],[0,0,0],[1,2,0],[0,0,2],[1,2,2],[0,1,2],
  [0,0,1],[1,0,2],[0,0,1],[1,1,2],[1,0,0],[0,1,1],[1,1,1],[1,1,2],[0,1,1],[0,1,1],[0,1,1],[1,2,1],[1,1,0],[1,1,1],[0,2,
  0],[0,2,0],[0,1,1],[0,1,1],[0,1,2],[1,1,0],[1,2,0],[1,1,0],[1,1,0],[1,2,0],[1,2,2],[0,1,1],[0,1,0],[1,1,0],[0,1,1],[1,
  2,0],[1,2,0],[1,2,0],[1,2,1]];

/** Python's outputs (at some steps) and final state for two configurations, run over INPUTS. */
const CASES: { config: Record<string, Json>; checked: Record<string, Json>; state: Json }[] =
  [{"config":{"alpha":0.05,"theta0":1.0,"alternative":"two-sided","method":"mixture","prior_sd":0.5,
  "confidence_sequence":true,"cs_range":[0.1,10.0],"cs_grid":30,"arms":[1,0],"stop_on_reject":false,"arm_input":"arm",
  "input":"kind","strata":3,"stratum_input":"stratum"},"checked":{"0":{"n_events":0,"at_risk":[0,0],"log_e_value":0.0,
  "p_value":1.0,"decision":null,"hazard_ratio_estimate":1.0,"lower":null,"upper":null},"15":{"n_events":0,"at_risk":[7,
  8],"log_e_value":0.0,"p_value":1.0,"decision":null,"hazard_ratio_estimate":1.0,"lower":null,"upper":null},
  "30":{"n_events":4,"at_risk":[12,10],"log_e_value":-0.0643170748029469,"p_value":0.9643552205152849,"decision":null,
  "hazard_ratio_estimate":0.9238770868019738,"lower":null,"upper":null},"45":{"n_events":5,"at_risk":[14,11],
  "log_e_value":-0.1002557348424089,"p_value":0.9643552205152849,"decision":null,
  "hazard_ratio_estimate":0.9757755150725314,"lower":null,"upper":null},"60":{"n_events":11,"at_risk":[14,12],
  "log_e_value":-0.11296708926972387,"p_value":0.9643552205152849,"decision":null,
  "hazard_ratio_estimate":0.8366134762566464,"lower":null,"upper":null},"75":{"n_events":16,"at_risk":[16,7],
  "log_e_value":0.26286592718614665,"p_value":0.6185646220201666,"decision":null,
  "hazard_ratio_estimate":0.6786199614942504,"lower":null,"upper":3.2903445623126695},"90":{"n_events":19,"at_risk":[16,
  12],"log_e_value":0.06121550787550234,"p_value":0.6185646220201666,"decision":null,
  "hazard_ratio_estimate":0.7286029557980641,"lower":null,"upper":3.2903445623126695},"105":{"n_events":30,
  "at_risk":[11,4],"log_e_value":-0.24083016189390216,"p_value":0.6185646220201666,"decision":null,
  "hazard_ratio_estimate":0.8064011327538553,"lower":0.16102620275609394,"upper":2.807216203941179},
  "120":{"n_events":38,"at_risk":[0,0],"log_e_value":0.17483449405635598,"p_value":0.6185646220201666,"decision":null,
  "hazard_ratio_estimate":0.7067059522109341,"lower":0.16102620275609394,"upper":2.043359717856943}},
  "state":{"at_risk":[0,0,0,0,0,0],"n_events":38,"log_lik":[-5.868737508214998,-5.327755617345833,-4.804879130734756,
  -4.300669445093929,-3.8156865935672686,-3.350487947375477,-2.905626885427598,-2.4816514368128546,-2.079102901246416,
  -1.6985144526632907,-1.3404097312373229,-1.0053014291506535,-0.6936898754585984,-0.4060616253914743,
  -0.14288805941712673,0.0953760026366115,0.3082936671221517,0.4954473065090937,0.6564398684457642,0.7908961486443798,
  0.8984640222350339,0.9788156281553255,1.0316485010438217,1.056686644993859,1.0536815434104063,1.0224130991104259,
  0.9626904987317024,0.8743529954835288,0.7572706043016574,0.6113447035754127,0.4365085378107937,0.2327276158907874,0.0,
  -0.2616435192013882,-0.5521393649410804,-0.871391237288823,-1.2192702989261741,-1.5956154850458832,
  -2.0002339377184044,-2.432901564038661,-2.893363716313686,-3.3813359914913232,-3.896505145981676,-4.438530121005613,
  -5.007043172636254,-5.601651099795987,-6.221936562647074,-6.86745948308023,-7.537758518372184,-8.232352598556274,
  -8.950742517635817,-9.69241256846879,-10.45683221096593,-11.243457763170493,-12.051734104823073,-12.881096383154494,
  -13.730971710887847,-14.600780846760005,-15.489939849286156,-16.397861694979227,-17.3239578527923,-18.2676398071659,
  -19.228320522726598,-20.205415844388433,-21.198345827346635],"log_e":0.0,"comp":0.0,"cs_log_lik":[-32.58781565119564,
  -30.366257046135352,-28.170492813439278,-26.00435800527501,-23.87216043234896,-21.778710681686224,-19.729346199132713,
  -17.72994720730624,-15.786942152207729,-13.907300440585288,-12.098510465826264,-10.36854132507029,-8.725787186788489,
  -7.178993941015749,-5.737168506782787,-4.409471931232728,-3.205098140780102,-2.1331408476767004,-1.2024516322334762,
  -0.4214925806576281,0.20181294612381562,0.6602277777840222,0.9473430108304117,1.0577110729991268,0.9869660798980578,
  0.7319276392223623,0.29068411316131226,-0.3373486448125006,-1.1513967437238457,-2.149328815921824,-3.327689401901227,
  -4.681765311666082,-6.2056833555377855,-7.892536089669408,-9.734530683597898,-11.723154842840053,-13.849353001958601,
  -16.10370577321585,-18.47660586964745,-20.958424357087356,-23.539662042771536,-26.21108198444222,-28.963820406818243,
  -31.78947464589281,-34.680168012687005,-37.62859259182761,-40.62803189804717,-43.672365961968524,-46.75606179233126,
  -49.87415228207398,-53.022206531026924,-56.19629430297079,-59.39294697770982,-62.60911695304904],"cs_kept":[4,18],
  "max_log_e":0.48035361076400385,"decision":null}},{"config":{"alpha":0.05,"theta0":1.0,"alternative":"less",
  "method":"grow","prior_sd":0.5,"confidence_sequence":false,"cs_range":[0.1,10.0],"cs_grid":101,"arms":[1,0],
  "stop_on_reject":false,"arm_input":"arm","input":"kind","strata":3,"stratum_input":"stratum"},
  "checked":{"0":{"n_events":0,"at_risk":[0,0],"log_e_value":0.0,"p_value":1.0,"decision":null,
  "hazard_ratio_estimate":1.0,"lower":null,"upper":null},"15":{"n_events":0,"at_risk":[7,8],"log_e_value":0.0,
  "p_value":1.0,"decision":null,"hazard_ratio_estimate":1.0,"lower":null,"upper":null},"30":{"n_events":4,"at_risk":[12,
  10],"log_e_value":0.0,"p_value":1.0,"decision":null,"hazard_ratio_estimate":0.9238770868019738,"lower":null,
  "upper":null},"45":{"n_events":5,"at_risk":[14,11],"log_e_value":-0.020389480952772487,"p_value":1.0,"decision":null,
  "hazard_ratio_estimate":0.9757755150725314,"lower":null,"upper":null},"60":{"n_events":11,"at_risk":[14,12],
  "log_e_value":-0.06279738484781583,"p_value":1.0,"decision":null,"hazard_ratio_estimate":0.8366134762566464,
  "lower":null,"upper":null},"75":{"n_events":16,"at_risk":[16,7],"log_e_value":0.3018915507318074,
  "p_value":0.5997723878550798,"decision":null,"hazard_ratio_estimate":0.6786199614942504,"lower":null,"upper":null},
  "90":{"n_events":19,"at_risk":[16,12],"log_e_value":0.11061069487717345,"p_value":0.5997723878550798,"decision":null,
  "hazard_ratio_estimate":0.7286029557980641,"lower":null,"upper":null},"105":{"n_events":30,"at_risk":[11,4],
  "log_e_value":-0.14846453110622493,"p_value":0.5997723878550798,"decision":null,
  "hazard_ratio_estimate":0.8064011327538553,"lower":null,"upper":null},"120":{"n_events":38,"at_risk":[0,0],
  "log_e_value":0.25170371089700894,"p_value":0.5997723878550798,"decision":null,
  "hazard_ratio_estimate":0.7067059522109341,"lower":null,"upper":null}},"state":{"at_risk":[0,0,0,0,0,0],"n_events":38,
  "log_lik":[-5.868737508214998,-5.327755617345833,-4.804879130734756,-4.300669445093929,-3.8156865935672686,
  -3.350487947375477,-2.905626885427598,-2.4816514368128546,-2.079102901246416,-1.6985144526632907,-1.3404097312373229,
  -1.0053014291506535,-0.6936898754585984,-0.4060616253914743,-0.14288805941712673,0.0953760026366115,
  0.3082936671221517,0.4954473065090937,0.6564398684457642,0.7908961486443798,0.8984640222350339,0.9788156281553255,
  1.0316485010438217,1.056686644993859,1.0536815434104063,1.0224130991104259,0.9626904987317024,0.8743529954835288,
  0.7572706043016574,0.6113447035754127,0.4365085378107937,0.2327276158907874,0.0,-0.2616435192013882,
  -0.5521393649410804,-0.871391237288823,-1.2192702989261741,-1.5956154850458832,-2.0002339377184044,-2.432901564038661,
  -2.893363716313686,-3.3813359914913232,-3.896505145981676,-4.438530121005613,-5.007043172636254,-5.601651099795987,
  -6.221936562647074,-6.86745948308023,-7.537758518372184,-8.232352598556274,-8.950742517635817,-9.69241256846879,
  -10.45683221096593,-11.243457763170493,-12.051734104823073,-12.881096383154494,-13.730971710887847,
  -14.600780846760005,-15.489939849286156,-16.397861694979227,-17.3239578527923,-18.2676398071659,-19.228320522726598,
  -20.205415844388433,-21.198345827346635],"log_e":0.25170371089700894,"comp":0.0,"cs_log_lik":[],"cs_kept":[0,0],
  "max_log_e":0.5112050493136278,"decision":null}}];

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
    // a Neumaier compensation is compared on the scale of its running sum
    return Object.keys(a).reduce((m, k) => {
      const ak = a[k], bk = b[k];
      if (k === "comp" && typeof ak === "number" && typeof bk === "number" && typeof b.log_e === "number") {
        const scale = Math.max(Math.abs(ak), Math.abs(bk), Math.abs(b.log_e));
        return Math.max(m, scale < 1e-300 ? 0 : Math.abs(ak - bk) / scale);
      }
      return Math.max(m, difference(ak, bk, `${path}.${k}`));
    }, 0);
  }
  assert.equal(a, b, path);
  return 0;
}

const record = ([arm, kind, stratum]: number[]) => ({ arm, kind, stratum });

function run(p: AnytimeLogRankTest, xs: Record<string, number>[], state?: State): [State, State][] {
  let s = state ?? p.initialState();
  const out: [State, State][] = [];
  for (const x of xs) {
    const [next, o] = p.step(s, x);
    out.push([next, o]);
    s = next;
  }
  return out;
}

for (const [n, c] of CASES.entries()) {
  test(`stratified log-rank test reproduces Python (case ${n})`, () => {
    const p = fromIdentity({ name: "anytime_logrank_test", version: "1", config: c.config }) as AnytimeLogRankTest;
    assert.ok(sameJson(p.config(), c.config), "the configuration round-trips");
    const steps = run(p, INPUTS.map(record));
    let worst = 0;
    for (const [i, want] of Object.entries(c.checked)) worst = Math.max(worst, difference(steps[Number(i)][1], want, `output ${i}`));
    const final = steps[steps.length - 1][0];
    worst = Math.max(worst, difference(final, c.state, "state"));
    assert.ok(worst <= 1e-12, `relative difference ${worst}`); // the last bits of exp and log may differ
    assert.ok(sameJson(p.decodeState(JSON.parse(JSON.stringify(c.state)) as Json), c.state), "decodeState round-trips");
  });
}

test("one stratum is the unstratified test: same identity, inputs and state", () => {
  const plain = new AnytimeLogRankTest(), one = new AnytimeLogRankTest(0.05, 1, "two-sided", "mixture", 0.5, true,
    [0.1, 10], 101, [1, 0], false, "arm", "kind", 1, "centre");
  assert.ok(sameJson(identity(one), identity(plain)));
  assert.ok(!("strata" in one.config()) && !("stratum_input" in one.config()));
  assert.deepEqual(Object.keys(one.inputs()).sort(), ["arm", "kind"]);
  assert.deepEqual((one.initialState() as { at_risk: number[] }).at_risk, [0, 0]);
  assert.throws(() => AnytimeLogRankTest.fromConfig({ ...plain.config(), strata: 1, stratum_input: "stratum" }), /only when/);
  assert.throws(() => AnytimeLogRankTest.fromConfig({ ...plain.config(), stratum_input: "stratum" }), /missing 'strata'/);
  const four = AnytimeLogRankTest.fromConfig({ ...plain.config(), strata: 4, stratum_input: "centre" });
  assert.equal(four.inputs().centre.upper, 3);
  assert.equal(identity(four).version, "1");
});

test("records all in one stratum give the unstratified test", () => {
  const xs = INPUTS.map(([arm, kind]) => ({ arm, kind }));
  const plain = run(new AnytimeLogRankTest(), xs).map(([, o]) => o);
  for (const stratum of [0, 2]) {
    const p = new AnytimeLogRankTest(0.05, 1, "two-sided", "mixture", 0.5, true, [0.1, 10], 101, [1, 0], false, "arm", "kind", 3);
    const got = run(p, xs.map((x) => ({ ...x, stratum }))).map(([, o]) => o);
    assert.ok(sameJson(got, plain), `stratum ${stratum}`);
  }
});

test("each event's factor has mean one under the null, in every stratum", () => {
  const counts = [[3, 1], [1, 4], [6, 2]];
  for (const theta0 of [1.0, 0.4, 2.5]) {
    for (const alternative of ["two-sided", "less", "greater"]) {
      const p = new AnytimeLogRankTest(0.05, theta0, alternative, "mixture", 0.5, true, [0.1, 10], 101, [1, 0], false, "arm",
        "kind", 3);
      let s = p.initialState();
      counts.forEach(([nT, nC], stratum) => {
        for (let i = 0; i < nT; i++) s = p.step(s, { arm: 1, kind: ENTER, stratum })[0];
        for (let i = 0; i < nC; i++) s = p.step(s, { arm: 0, kind: ENTER, stratum })[0];
      });
      counts.forEach(([nT, nC], stratum) => {
        const q = (nT * theta0) / (nT * theta0 + nC);
        const treated = p.step(s, { arm: 1, kind: EVENT, stratum })[1].log_e_value as number;
        const control = p.step(s, { arm: 0, kind: EVENT, stratum })[1].log_e_value as number;
        assert.ok(Math.abs(treated) > 1e-3);
        assert.ok(Math.abs(q * Math.exp(treated) + (1 - q) * Math.exp(control) - 1) < 1e-14, `${theta0} ${alternative} ${stratum}`);
      });
    }
  }
});

test("strata are validated in the arguments, the records and the state", () => {
  assert.throws(() => new AnytimeLogRankTest(0.05, 1, "two-sided", "mixture", 0.5, true, [0.1, 10], 101, [1, 0], false, "arm",
    "kind", 0), RangeError);
  assert.throws(() => new AnytimeLogRankTest(0.05, 1, "two-sided", "mixture", 0.5, true, [0.1, 10], 101, [1, 0], false, "arm",
    "kind", 2, "arm"), RangeError);
  const p = new AnytimeLogRankTest(0.05, 1, "two-sided", "mixture", 0.5, true, [0.1, 10], 101, [1, 0], false, "arm", "kind", 2);
  let s = p.initialState();
  for (const stratum of [2, -1, 0.5, Number.NaN]) assert.throws(() => p.step(s, { arm: 1, kind: ENTER, stratum }), ContractViolation);
  assert.throws(() => p.step(s, { arm: 1, kind: ENTER }), ContractViolation);
  s = p.step(s, { arm: 1, kind: ENTER, stratum: 1 })[0];
  s = p.step(s, { arm: 0, kind: ENTER, stratum: 0 })[0];
  assert.deepEqual((s as { at_risk: number[] }).at_risk, [0, 1, 1, 0]);
  assert.throws(() => p.step(s, { arm: 1, kind: EVENT, stratum: 0 }), /arm 1 of stratum 0 with nobody at risk/);
  assert.throws(() => p.step(s, { arm: 0, kind: CENSOR, stratum: 1 }), /arm 0 of stratum 1 with nobody at risk/);
  const [after, out] = p.step(s, { arm: 1, kind: EVENT, stratum: 1 });
  assert.deepEqual([out.n_events, out.at_risk, out.log_e_value], [1, [0, 1], 0]);
  const encoded = JSON.parse(JSON.stringify(after)) as Record<string, Json>;
  assert.ok(sameJson(p.decodeState(encoded), after));
  assert.throws(() => p.decodeState({ ...encoded, at_risk: [0, 1] }), /inconsistent state/);
  assert.throws(() => new AnytimeLogRankTest().decodeState(encoded), /inconsistent state/);
});
