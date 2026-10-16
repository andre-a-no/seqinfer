// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The forecasters (procedures/forecast.ts): construction, checkpoint refusals and the mixture quantile, beyond the conformance vectors.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { type Json, sameJson } from "../src/canonical.js";
import { fromIdentity, identity, type Identity, type State } from "../src/procedure.js";
import {
  AutoRegressive, averagedSmoothing, ConformalForecast, ExponentialSmoothing, ForecastComparison, ModelAveraging, normalInvCdf,
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

test("forecasters check their arguments", () => {
  assert.throws(() => new ExponentialSmoothing(0.5, 0.0, 0.0, 0.9), /phi < 1 needs trend='damped'/);
  assert.throws(() => new ExponentialSmoothing(0.5, 0.1), /beta and initial_trend need a trend/);
  assert.throws(() => new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, 4, null, 0.0, [0, 0, 0]), /one value per season/);
  assert.throws(() => new AutoRegressive(2, [0.5, 0.1], 0.0, 1.0, 0.9), /forgetting and ridge are for learnt coefficients/);
  assert.throws(() => new AutoRegressive(2, null, 0.0, 1.0), /sigma and intercept come with fixed coefficients/);
  const es = new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, null, null, 0.0, null, 1.0);
  assert.throws(() => new ConformalForecast(es, 0.1, 0.05, false), /initial half width/);
  assert.throws(() => new ModelAveraging([es, new ExponentialSmoothing(0.5, 0.0, 0.0, 1.0, null, null, null, 0.0, null, null, 2)]),
    /same horizon/);
  const conformal = identity(new ConformalForecast(es));
  const id = { name: "model_averaging", version: "1", config: { models: [conformal], weights: [1], forgetting: 1, coverage: 0.9 } };
  assert.throws(() => fromIdentity(id as unknown as Identity), /ExponentialSmoothing or AutoRegressive/);
  assert.throws(() => new ForecastComparison([0, 1], 0.05, null, "x", "x"), /different names/);
});

test("a model without a season rebuilds from its identity", () => {
  // its config gives initial_seasonal null, as Python's does
  const es = new ExponentialSmoothing(0.4, 0.1, 0.0, 0.9, "damped", null, null, 0.0, null, null, 3, 0.8);
  const again = fromIdentity(identity(es));
  assert.ok(sameJson(identity(again), identity(es)));
  assert.equal(es.config().initial_seasonal, null);
});

test("averagedSmoothing builds the grid of Python's averaged_smoothing", () => {
  const ma = averagedSmoothing({ trend: "additive", period: 4, alphas: [0.1, 0.4], betas: [0.1], gammas: [0.05, 0.3], horizon: 2 });
  assert.ok(sameJson(identity(ma), vector("averaged_smoothing").procedure));
});

test("the mixture quantile of one normal component is the normal quantile", () => {
  const live: [number, { mean: number[]; scale: number[]; df: null }][] = [[1.0, { mean: [3.0], scale: [2.0], df: null }]];
  const x = ModelAveraging.solve(live, 0, 0.95, 0.0, 8.0);
  assert.ok(Math.abs(x - (3.0 + 2.0 * normalInvCdf(0.95))) < 1e-12, `${x}`);
});
