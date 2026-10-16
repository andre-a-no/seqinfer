// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The forecasters (procedures/forecast.ts): construction, checkpoint refusals and the mixture quantile, beyond the conformance vectors.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { type Json, sameJson } from "../src/canonical.js";
import { fromIdentity, identity, type Identity, type State } from "../src/procedure.js";
import { ContractViolation } from "../src/errors.js";
import {
  AutoRegressive, averagedCounts, averagedSmoothing, ConformalForecast, countQuantile, ExponentialSmoothing, ForecastComparison,
  ModelAveraging, negativeBinomialCdf, negativeBinomialLogPmf, normalInvCdf, PoissonSmoothing, Predictive,
} from "../src/procedures/index.js";

interface Vector {
  case: string;
  procedure: Identity;
  state: Json;
}

const vectors = JSON.parse(readFileSync(new URL("../../../conformance/procedures.json", import.meta.url), "utf8")) as Vector[];
const vector = (name: string): Vector => vectors.find((v) => v.case === name)!;

/** The final state of a conformance case, changed by `edit`, must be refused with `message`. */
function refuses(name: string, edit: (s: State) => void, message: RegExp): void {
  const v = vector(name);
  const p = fromIdentity(v.procedure);
  const s = JSON.parse(JSON.stringify(v.state)) as State;
  edit(s);
  assert.throws(() => p.decodeState(s), message, `${name}: ${message}`);
}

test("forecasters refuse inconsistent states, as Python does", () => {
  refuses("ets_fixed_seasonal", (s) => void (s.seasonal = (s.seasonal as number[]).slice(0, 5)), /one seasonal effect per season/);
  refuses("ets_fixed_seasonal", (s) => void (s.errors = 61), /error statistics/);
  refuses("ets_fixed_seasonal", (s) => Object.assign(s, { n: 0, errors: 0, log_likelihood: 0 }), /statistics before data/);
  refuses("conformal_unnormalized", (s) => void ((s.inner as State).trend = 0.5), /a trend without a trend model/);
  refuses("ar_learnt_forgetting", (s) => void ((s.information as number[][])[0][1] = 1.0), /symmetric/);
  refuses("ar_learnt_forgetting", (s) => void (s.lags = (s.lags as number[]).slice(0, 2)), /one lag per observation/);
  refuses("ar_learnt_forgetting", (s) => void (s.information = null), /information/);
  refuses("ar_learnt_forgetting", (s) => void (s.shape = -1.0), /noise statistics/);
  refuses("ar_fixed", (s) => void (s.moment = [1.0]), /learnt statistics with fixed coefficients/);
  refuses("model_averaging", (s) => void (s.log_weights = [-1.0, -1.0, -1.0]), /weights must sum to 1/);
  refuses("model_averaging", (s) => void (s.log_weights = [0.5, -1.0, -1.0]), /log weights/);
  refuses("model_averaging", (s) => void delete s.comp, /fields/);
  refuses("model_averaging", (s) => void ((s.states as State[])[2].lags = [1.0]), /one lag per observation/);
  refuses("conformal_forecast", (s) => void (s.misses = [69, 16, 17]), /counts/);
  refuses("conformal_forecast", (s) => void ((s.pending as number[][][])[0][0].pop()), /pending interval/);
  refuses("conformal_forecast", (s) => void (s.quantiles = [1.0, 1.0]), /one entry per horizon/);
  refuses("forecast_comparison", (s) => void (s.variance = -1.0), /negative variance/);
  refuses("forecast_comparison", (s) => void (s.n = 0), /statistics before data/);
});

test("forecasters with regressors, and counts, refuse inconsistent states, as Python does", () => {
  refuses("ets_learnt_effects", (s) => void (s.effects = [[0, 0]]), /effects' sizes/);
  refuses("ets_learnt_effects", (s) => void (s.moment = []), /statistics' sizes/);
  refuses("ets_learnt_effects", (s) => void (s.information = [[-1.0]]), /symmetric with a non-negative diagonal/);
  refuses("ets_learnt_effects", (s) => void (s.sse = 1.0), /noise statistics/);
  refuses("ets_learnt_effects", (s) => void (s.plan = []), /one plan per regressor/);
  refuses("ets_learnt_effects", (s) => void (s.plan = [[0, 0]]), /one plan per regressor/);
  refuses("ets_learnt_effects", (s) => void (s.effects = null), /state field 'effects'/);
  refuses("ets_fixed_effects", (s) => void (s.shape = 1.0), /learnt statistics without learnt effects/);
  refuses("ets_fixed_effects", (s) => void (s.effects = [[1.0, 2.0]]), /learnt statistics without learnt effects/);
  refuses("ar_learnt_regressor", (s) => void (s.plan = [[1]]), /one plan per regressor/);
  refuses("ar_learnt_regressor", (s) => void (s.information = (s.information as number[][]).slice(0, 3)), /statistics' sizes/);
  refuses("ar_fixed_log", (s) => void (s.plan = [[1, 2, 3]]), /one plan per regressor/);
  refuses("poisson_smoothing", (s) => void (s.shape = -1.0), /the gamma distribution's parameters/);
  refuses("poisson_smoothing", (s) => void (s.plan = (s.plan as number[]).slice(0, 2)), /one planned exposure per horizon/);
  refuses("poisson_smoothing", (s) => void delete s.plan, /fields/);
  refuses("averaged_counts", (s) => void ((s.states as State[])[0].n = 0), /statistics before data/);
  refuses("averaged_counts", (s) => void ((s.states as State[])[1].plan = [1.0]), /one planned exposure per horizon/);
});

test("forecasters check their arguments", () => {
  assert.throws(() => new ExponentialSmoothing(0.5, 0.0, 0.0, 0.9), /phi < 1 needs trend='damped'/);
  assert.throws(() => new ExponentialSmoothing(0.5, 0.1), /beta and initial_trend need a trend/);
  assert.throws(() => new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, 4, null, 0.0, [0, 0, 0]), /one value per season/);
  assert.throws(() => new AutoRegressive(2, [0.5, 0.1], 0.0, 1.0, [], null, 0.9), /forgetting and ridge are for learnt coefficients/);
  assert.throws(() => new AutoRegressive(2, null, 0.0, 1.0), /sigma, intercept and effects come with fixed coefficients/);
  const es = new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, null, null, 0.0, null, 1.0);
  assert.throws(() => new ConformalForecast(es, 0.1, 0.05, false), /initial half width/);
  const twoSteps = new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, null, null, 0.0, null, null, [], null, 1.0, 1e-6, null, 2);
  assert.throws(() => new ModelAveraging([es, twoSteps]), /same horizon/);
  const conformal = identity(new ConformalForecast(es));
  const id = { name: "model_averaging", version: "1", config: { models: [conformal], weights: [1], forgetting: 1, coverage: 0.9 } };
  assert.throws(() => fromIdentity(id as unknown as Identity), /ExponentialSmoothing or AutoRegressive/);
  assert.throws(() => new ForecastComparison([0, 1], 0.05, null, "x", "x"), /different names/);
});

test("regressors, transforms and counts check their arguments", () => {
  const ets = (regressors: string[], coefficients: number[] | null, sigma: number | null = null, transform: string | null = null) =>
    new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, null, 10.0, 0.0, null, sigma, regressors, coefficients, 1.0, 1e-6, transform);
  assert.throws(() => ets(["promo"], null, 1.0), /learnt effects come with a learnt sigma/);
  assert.throws(() => ets(["promo"], [1.0, 2.0]), /one finite coefficient per regressor/);
  assert.throws(() => ets(["y"], [1.0]), /distinct names, different from the input/);
  assert.throws(() => ets(["a", "a"], null), /distinct names/);
  assert.throws(() => ets([], null, null, "sqrt"), /transform must be one of/);
  assert.throws(() => new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, null, null, 0.0, null, null, [], null, 0.9),
    /forgetting and ridge are for learnt effects/);
  assert.throws(() => new AutoRegressive(1, [0.5], 0.0, 1.0, ["promo"]), /one finite effect per regressor/);
  assert.throws(() => new AutoRegressive(1, null, 0.0, null, ["promo"], [1.0]), /sigma, intercept and effects/);
  assert.throws(() => new PoissonSmoothing(0.0), /discount must lie in \(0, 1\]/);
  assert.throws(() => new PoissonSmoothing(0.9, "y"), /exposure must name an input different from the counts'/);
  assert.throws(() => new PoissonSmoothing(0.9, null, [-1.0, 0.0]), /finite and non-negative/);
  const counts = new PoissonSmoothing(0.9);
  assert.throws(() => new ModelAveraging([counts, ets([], null)]), /or all PoissonSmoothing ones/);
  assert.throws(() => new ModelAveraging([ets(["promo"], [1.0], 1.0), ets([], null)]), /the same inputs/);
  assert.throws(() => new ModelAveraging([counts, new PoissonSmoothing(0.8, "hours")]), /the same inputs/);
  // the inputs: regressors and exposures are windows of horizon + 1 numbers, counts non-negative integers
  const ar = new AutoRegressive(1, null, 0.0, null, ["promo"], null, 1.0, 1e-6, null, 3);
  assert.deepEqual(ar.inputs().promo.spec(), { name: "promo", kind: "real", shape: [4], lower: null, upper: null, unit: null });
  const exposed = new PoissonSmoothing(0.9, "hours", [0.0, 0.0], 2).inputs();
  assert.deepEqual(exposed.y.spec(), { name: "y", kind: "integer", shape: [], lower: 0, upper: null, unit: null });
  assert.deepEqual(exposed.hours.spec(), { name: "hours", kind: "real", shape: [3], lower: 0, upper: null, unit: null });
  // values the model cannot take are contract violations, as in Python
  const logged = ets([], null, 1.0, "log");
  assert.throws(() => logged.step(logged.initialState(), { y: 0.0 }), ContractViolation);
  const hours = new PoissonSmoothing(0.9, "hours", [0.0, 0.0], 1);
  assert.throws(() => hours.step(hours.initialState(), { y: 3, hours: [1.0, 0.0] }), ContractViolation);
});

test("averagedCounts builds the grid of Python's averaged_counts", () => {
  const ma = averagedCounts({ discounts: [0.7, 0.9, 1.0], horizon: 2, coverage: 0.8 });
  assert.ok(sameJson(identity(ma), vector("averaged_counts").procedure));
  assert.ok(ma.counts);
});

test("negative binomial forecasts: the CDF sums the probabilities, the quantile is the smallest count reaching p", () => {
  const [size, rate, exposure] = [3.7, 1.9, 1.3];
  let total = 0.0;
  for (let k = 0; k < 40; k++) {
    total += Math.exp(negativeBinomialLogPmf(k, size, rate, exposure));
    assert.ok(Math.abs(negativeBinomialCdf(k, size, rate, exposure) - total) < 1e-13, `k = ${k}`);
  }
  assert.equal(negativeBinomialCdf(-1, size, rate, exposure), 0.0);
  const pred = new Predictive([exposure * size / rate], [1.0], null, null, [size], [rate], [exposure]);
  for (const p of [0.01, 0.05, 0.5, 0.95, 0.999]) {
    const k = countQuantile(pred, 0, p);
    assert.ok(pred.cdf(0, k) >= p && (k === 0 || pred.cdf(0, k - 1) < p), `p = ${p}: ${k}`);
  }
});

test("a model without a season rebuilds from its identity", () => {
  // its config gives initial_seasonal null, as Python's does
  const es = new ExponentialSmoothing(0.4, 0.1, 0.0, 0.9, "damped", null, null, 0.0, null, null, [], null, 1.0, 1e-6, null, 3, 0.8);
  const again = fromIdentity(identity(es));
  assert.ok(sameJson(identity(again), identity(es)));
  assert.equal(es.config().initial_seasonal, null);
});

test("averagedSmoothing builds the grid of Python's averaged_smoothing", () => {
  const ma = averagedSmoothing({ trend: "additive", period: 4, alphas: [0.1, 0.4], betas: [0.1], gammas: [0.05, 0.3], horizon: 2 });
  assert.ok(sameJson(identity(ma), vector("averaged_smoothing").procedure));
});

test("the mixture quantile of one normal component is the normal quantile", () => {
  const es = new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, null, null, 0.0, null, 1.0);
  const x = ModelAveraging.solve([[1.0, es, new Predictive([3.0], [2.0], null)]], 0, 0.95, 0.0, 8.0);
  assert.ok(Math.abs(x - (3.0 + 2.0 * normalInvCdf(0.95))) < 1e-12, `${x}`);
});
