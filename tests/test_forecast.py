# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Forecasting: exponential smoothing and autoregression, fixed and learnt, averaging, conformal, comparison.

* Exponential smoothing with fixed parameters reproduces statsmodels'
  ETSModel: one-step errors, forecasts and intervals at every horizon.
  fit_ets reaches at least statsmodels' maximum likelihood.
* fit_ar is statsmodels' AutoReg (conditional least squares); the fixed AR
  forecaster gives AutoReg's forecasts and intervals.
* Learning is exact Bayesian updating: the AR coefficients are the
  (weighted) ridge regression on the lags, the noise statistics its
  residual; exponential smoothing with sigma unknown has as its
  prequential likelihood the closed-form marginal likelihood; model
  averaging's weights are the posterior model probabilities and its
  likelihood the marginal of the mixture.
* Conformal intervals satisfy the telescoping identity for any data, and
  reach their coverage where the model's own intervals do not.
* The comparison's confidence sequence covers the average difference at
  every time in simulations, and finds a better forecaster.
"""
import math
import random
import unittest

from sequential_analysis.forecasting import fit_ar, fit_ets, nelder_mead
from sequential_analysis.procedures.forecast import (
    AutoRegressive,
    ConformalForecast,
    ExponentialSmoothing,
    ForecastComparison,
    ModelAveraging,
    averaged_smoothing,
    log_density,
    standard_cdf,
    stitched_boundary,
)

try:
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
except ImportError:  # optional: the comparisons are skipped
    ETSModel = None

CONFIGS = [(None, None), ("additive", None), ("damped", None), (None, 4), ("additive", 12), ("damped", 7)]


def series(n, seed, trend=None, period=None):
    rnd = random.Random(seed)
    walk, out = 0.0, []
    for i in range(n):
        walk += rnd.gauss(0.0, 0.5)
        season = 3.0 * math.sin(2.0 * math.pi * i / period) if period else 0.0
        out.append(50.0 + (0.2 * i if trend else 0.0) + season + walk + rnd.gauss(0.0, 1.0))
    return out


def ar_series(n, seed, phi=(0.5, -0.3, 0.2), c=1.0, sd=1.0):
    rnd = random.Random(seed)
    ys = [0.0] * len(phi)
    for _ in range(n):
        v = c
        for i, p in enumerate(phi):
            v += p * ys[-1 - i]
        ys.append(v + rnd.gauss(0.0, sd))
    return ys[len(phi):]


def drive(procedure, ys, key="y"):
    state, outs = procedure.initial_state(), []
    for y in ys:
        state, out = procedure.step(state, {key: y})
        outs.append(out)
    return state, outs


@unittest.skipIf(ETSModel is None, "statsmodels is not installed")
class AgainstStatsmodels(unittest.TestCase):
    def sm_fit(self, ys, trend, period):
        return ETSModel(pd.Series(ys), error="add", trend="add" if trend else None, damped_trend=trend == "damped",
                        seasonal="add" if period else None, seasonal_periods=period).fit(disp=False)

    def test_exponential_smoothing_is_statsmodels_ets(self):
        for seed, (trend, period) in enumerate(CONFIGS):
            ys = series(120, seed, trend, period)
            res = self.sm_fit(ys, trend, period)
            p = dict(zip(res.param_names, res.params))
            seasonal = None
            if period:
                seasonal = [float(p[f"initial_seasonal.{i}"]) for i in range(period)][::-1]  # theirs: newest first
            horizon = 2 * (period or 3) + 1
            model = ExponentialSmoothing(
                float(p["smoothing_level"]), beta=float(p.get("smoothing_trend", 0.0)),
                gamma=float(p.get("smoothing_seasonal", 0.0)), phi=float(p.get("damping_trend", 1.0)), trend=trend,
                period=period, initial_level=float(p["initial_level"]),
                initial_trend=float(p.get("initial_trend", 0.0)), initial_seasonal=seasonal, sigma=math.sqrt(res.mse),
                horizon=horizon, coverage=0.9)
            _, outs = drive(model, ys)
            for out, resid in zip(outs, res.resid):
                self.assertAlmostEqual(out.error, float(resid), delta=1e-9)
            frame = res.get_prediction(start=len(ys), end=len(ys) + horizon - 1).summary_frame(alpha=0.1)
            last = outs[-1]
            for h in range(horizon):
                row = frame.iloc[h]
                self.assertAlmostEqual(last.mean[h], float(row["mean"]), delta=1e-9 * abs(row["mean"]))
                self.assertAlmostEqual(last.lower[h], float(row["pi_lower"]), delta=1e-9 * abs(row["mean"]))
                self.assertAlmostEqual(last.upper[h], float(row["pi_upper"]), delta=1e-9 * abs(row["mean"]))
            self.assertAlmostEqual(last.log_likelihood, float(res.llf), delta=1e-8 * abs(res.llf))

    def test_fit_ets_reaches_statsmodels_likelihood(self):
        for seed, (trend, period) in enumerate(CONFIGS):
            ys = series(150, 100 + seed, trend, period)
            ours = fit_ets(ys, trend=trend, period=period)
            res = self.sm_fit(ys, trend, period)
            gamma = dict(zip(res.param_names, res.params)).get("smoothing_seasonal", 0.0)
            if gamma >= 0:  # statsmodels also allows a slightly negative gamma, outside our region
                self.assertGreaterEqual(ours.log_likelihood, res.llf - 1e-6, (trend, period))
            self.assertAlmostEqual(ours.state.log_likelihood + ours.state.comp, ours.log_likelihood,
                                   delta=1e-9 * abs(ours.log_likelihood))

    def test_fit_ar_and_forecasts_are_autoreg(self):
        ys = ar_series(400, 3)
        for order in (1, 3):
            fit = fit_ar(ys, order, horizon=6)
            res = AutoReg(np.array(ys), order, trend="c").fit()
            theta = [fit.parameters["intercept"]] + fit.parameters["coefficients"]
            for a, b in zip(theta, res.params):
                self.assertAlmostEqual(a, float(b), delta=1e-10)
            self.assertAlmostEqual(fit.parameters["sigma"] ** 2, float(res.sigma2), delta=1e-10)
            frame = res.get_prediction(start=len(ys), end=len(ys) + 5).summary_frame(alpha=0.1)
            pred = fit.model.predictive(fit.state)  # the forecast from the end of the history
            lower, upper = fit.model.interval(pred)
            for h in range(6):
                self.assertAlmostEqual(pred.mean[h], float(frame["mean"].iloc[h]), delta=1e-9)
                self.assertAlmostEqual(lower[h], float(frame["mean_ci_lower"].iloc[h]), delta=1e-9)
                self.assertAlmostEqual(upper[h], float(frame["mean_ci_upper"].iloc[h]), delta=1e-9)
            self.assertEqual(len(pred.mean), 6)
            self.assertAlmostEqual(fit.state.log_likelihood + fit.state.comp, fit.log_likelihood,
                                   delta=1e-9 * abs(fit.log_likelihood))


def solve(a, b):
    """Gaussian elimination with partial pivoting, for the reference computations."""
    n = len(a)
    m = [[*list(row), v] for row, v in zip(a, b)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        for r in range(i + 1, n):
            f = m[r][i] / m[i][i]
            for c in range(i, n + 1):
                m[r][c] -= f * m[i][c]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (m[i][n] - sum(m[i][c] * x[c] for c in range(i + 1, n))) / m[i][i]
    return x


class LearntAutoregression(unittest.TestCase):
    def reference(self, ys, order, lam, ridge):
        """Weighted ridge regression on the lags: weights lam^(N + 1 - t), penalty ridge (not discounted)."""
        rows = [([1.0] + [ys[t - i] for i in range(1, order + 1)], ys[t]) for t in range(order, len(ys))]
        n = len(rows)
        k = order + 1
        gram = [[ridge if i == j else 0.0 for j in range(k)] for i in range(k)]
        rhs = [0.0] * k
        for t, (x, y) in enumerate(rows):
            w = lam ** (n - t)
            for i in range(k):
                rhs[i] += w * x[i] * y
                for j in range(k):
                    gram[i][j] += w * x[i] * x[j]
        return solve(gram, rhs), gram

    def test_coefficients_are_the_weighted_ridge_regression(self):
        ys = ar_series(300, 5)
        for lam in (1.0, 0.97):
            model = AutoRegressive(3, forgetting=lam, ridge=0.5)
            state, outs = drive(model, ys)
            theta, _ = self.reference(ys, 3, lam, 0.5)
            params = outs[-1].parameters
            for a, b in zip([params["intercept"], *params["coefficients"]], theta):
                self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)))
            self.assertAlmostEqual(state.shape, 0.5 * sum(lam ** i for i in range(len(ys) - 3)), delta=1e-9)

    def test_one_step_score_is_the_exact_student_t_predictive(self):
        ys = ar_series(60, 6)
        model = AutoRegressive(2, ridge=1.0)
        state = model.initial_state()
        rate = 0.0
        for t, y in enumerate(ys):
            new, _, score = model.update(state, {"y": y})
            if t >= 2:
                theta, gram = self.reference(ys[:t], 2, 1.0, 1.0)
                x = [1.0, ys[t - 1], ys[t - 2]]
                s = 1.0 + sum(a * b for a, b in zip(x, solve(gram, x)))
                error = y - sum(a * b for a, b in zip(theta, x))
                shape = 0.5 * (t - 2)
                if score is not None:
                    expected = log_density(error, math.sqrt(rate / shape * s), 2.0 * shape)
                    self.assertAlmostEqual(score, expected, delta=1e-9)
                rate += error * error / (2.0 * s)
                self.assertAlmostEqual(new.rate, rate, delta=1e-9 * rate)
            state = new

    def test_learnt_approach_the_fitted_coefficients(self):
        ys = ar_series(4000, 7)
        _state, outs = drive(AutoRegressive(3), ys)
        fit = fit_ar(ys, 3)
        for a, b in zip(outs[-1].parameters["coefficients"], fit.parameters["coefficients"]):
            self.assertAlmostEqual(a, b, delta=1e-4)
        for a, b in zip(outs[-1].parameters["coefficients"], (0.5, -0.3, 0.2)):
            self.assertAlmostEqual(a, b, delta=0.05)

    def test_forgetting_follows_a_change(self):
        ys = ar_series(1500, 8, phi=(0.8,)) + ar_series(1500, 9, phi=(-0.5,))
        fixed = drive(AutoRegressive(1), ys)[1][-1].parameters["coefficients"][0]
        forgetting = drive(AutoRegressive(1, forgetting=0.98), ys)[1][-1].parameters["coefficients"][0]
        self.assertLess(abs(forgetting + 0.5), 0.2)
        self.assertGreater(abs(fixed + 0.5), 0.4)

    def test_fixed_forecast_variance_uses_psi_weights(self):
        model = AutoRegressive(2, coefficients=(0.6, 0.2), intercept=0.0, sigma=2.0, horizon=4)
        state, _ = drive(model, [1.0, 2.0])
        pred = model.predictive(state)
        psi = [1.0, 0.6, 0.6 * 0.6 + 0.2, 0.6 * (0.6 * 0.6 + 0.2) + 0.2 * 0.6]
        for h in range(4):
            self.assertAlmostEqual(pred.scale[h], 2.0 * math.sqrt(sum(p * p for p in psi[: h + 1])), delta=1e-12)
        self.assertEqual(pred.mean[0], 0.6 * 2.0 + 0.2 * 1.0)


class LearntSmoothing(unittest.TestCase):
    def test_prequential_likelihood_is_the_marginal_likelihood(self):
        # prior 1/sigma^2: p(e_2..e_n | e_1) = Gamma(n/2) (S_n/2)^(-n/2) / (Gamma(1/2) (S_1/2)^(-1/2)) (2 pi)^(-(n-1)/2)
        ys = series(80, 11, "additive", 4)
        model = ExponentialSmoothing(0.3, beta=0.05, gamma=0.1, trend="additive", period=4)
        _state, outs = drive(model, ys)
        errors = [o.error for o in outs if o.error is not None]
        sse, first = sum(e * e for e in errors), errors[0] ** 2
        k = len(errors)
        expected = (math.lgamma(0.5 * k) - 0.5 * k * math.log(0.5 * sse) - math.lgamma(0.5)
                    + 0.5 * math.log(0.5 * first) - 0.5 * (k - 1) * math.log(2.0 * math.pi))
        self.assertAlmostEqual(outs[-1].log_likelihood, expected, delta=1e-9 * abs(expected))
        self.assertEqual(outs[-1].df, float(k))

    def test_first_value_sets_the_level(self):
        model = ExponentialSmoothing(0.5, sigma=1.0, horizon=2)
        _state, out = model.step(model.initial_state(), {"y": 7.0})
        self.assertEqual(out.mean, (7.0, 7.0))
        self.assertIsNone(out.error)
        self.assertIsNone(out.log_score)

    def test_config_rebuilds_the_model(self):
        for model in (ExponentialSmoothing(0.4, beta=0.1, phi=0.9, trend="damped"),
                      ExponentialSmoothing(0.3, gamma=0.1, period=3, sigma=1.0)):
            self.assertEqual(ExponentialSmoothing(**model.config()).identity(), model.identity())

    def test_damped_trend_forecast(self):
        model = ExponentialSmoothing(0.5, beta=0.1, phi=0.8, trend="damped", initial_level=10.0, initial_trend=2.0,
                                     sigma=1.0, horizon=3)
        pred = model.predictive(model.initial_state())
        self.assertAlmostEqual(pred.mean[0], 10.0 + 0.8 * 2.0, delta=1e-12)
        self.assertAlmostEqual(pred.mean[2], 10.0 + (0.8 + 0.64 + 0.512) * 2.0, delta=1e-12)


class Averaging(unittest.TestCase):
    def test_weights_are_posterior_model_probabilities(self):
        ys = series(150, 12, "additive")
        models = [ExponentialSmoothing(a, beta=0.1 * a, trend="additive") for a in (0.1, 0.4, 0.8)]
        avg = ModelAveraging(models, weights=(1.0, 2.0, 1.0))
        _state, outs = drive(avg, ys)
        logs = [drive(m, ys)[1][-1].log_likelihood + math.log(w) for m, w in zip(models, (0.25, 0.5, 0.25))]
        top = max(logs)
        total = sum(math.exp(v - top) for v in logs)
        for w, v in zip(outs[-1].parameters["weights"], logs):
            self.assertAlmostEqual(w, math.exp(v - top) / total, delta=1e-9)
        self.assertAlmostEqual(outs[-1].log_likelihood, top + math.log(total), delta=1e-9 * abs(top))

    def test_forgetting_recursion(self):
        ys = series(100, 13)
        models = [ExponentialSmoothing(a) for a in (0.2, 0.9)]
        lam = 0.9
        _, outs = drive(ModelAveraging(models, forgetting=lam), ys)
        singles = [drive(m, ys)[1] for m in models]
        w = [0.5, 0.5]
        for t in range(len(ys)):
            scores = [s[t].log_score for s in singles]
            if None in scores:
                continue
            flat = [v ** lam for v in w]
            joint = [f / sum(flat) * math.exp(sc) for f, sc in zip(flat, scores)]
            w = [j / sum(joint) for j in joint]
        for a, b in zip(outs[-1].parameters["weights"], w):
            self.assertAlmostEqual(a, b, delta=1e-9)

    def test_interval_holds_the_coverage_of_the_mixture(self):
        ys = series(40, 14)
        avg = ModelAveraging([ExponentialSmoothing(0.2, horizon=2), AutoRegressive(1, horizon=2)], coverage=0.8)
        state, outs = drive(avg, ys)
        comps = avg._components(state)
        for h in range(2):
            for bound, target in ((outs[-1].lower[h], 0.1), (outs[-1].upper[h], 0.9)):
                cdf = sum(w * standard_cdf((bound - p.mean[h]) / p.scale[h], p.df) for w, _, p in comps)
                self.assertAlmostEqual(cdf, target, delta=1e-9)

    def test_grid_learns_the_smoothing_parameter(self):
        rnd = random.Random(15)
        level, ys = 0.0, []
        for _ in range(1500):  # local level model: the optimal alpha is about 0.62 for q/r = 1
            level += rnd.gauss(0.0, 1.0)
            ys.append(level + rnd.gauss(0.0, 1.0))
        avg = averaged_smoothing()
        _, outs = drive(avg, ys)
        weights = outs[-1].parameters["weights"]
        best = max(range(len(weights)), key=lambda i: weights[i])
        self.assertIn(avg.models[best].alpha, (0.5, 0.7))

    def test_refuses_mismatched_models(self):
        with self.assertRaises(ValueError):
            ModelAveraging([ExponentialSmoothing(0.2, horizon=1), ExponentialSmoothing(0.3, horizon=2)])
        with self.assertRaises(ValueError):
            ModelAveraging([ModelAveraging([ExponentialSmoothing(0.2)])])  # type: ignore[list-item]


class Conformal(unittest.TestCase):
    def test_telescoping_identity_holds_for_any_data(self):
        rnd = random.Random(16)
        ys, v = [], 0.0
        for t in range(2000):  # heavy tails, jumps and a change of scale
            v += rnd.gauss(0.0, 1.0) / max(0.05, abs(rnd.gauss(0.0, 1.0)))
            ys.append(v + (500.0 if t % 333 == 0 else 0.0) * (1 + (t > 1000)))
        procedure = ConformalForecast(ExponentialSmoothing(0.5, sigma=1.0, horizon=3), alpha=0.2, step_size=0.1)
        _state, outs = drive(procedure, ys)
        for h in range(3):
            count = outs[-1].counts[h]
            identity = (outs[-1].quantiles[h] - procedure.initial) / (procedure.step_size * count)
            self.assertAlmostEqual(outs[-1].miscoverage[h] - 0.2, identity, delta=1e-12)

    def test_reaches_coverage_where_gaussian_intervals_do_not(self):
        rnd = random.Random(17)
        level, ys = 0.0, []
        for _ in range(4000):
            level += rnd.gauss(0.0, 0.3)
            ys.append(level + rnd.gauss(0.0, 1.0) / max(1e-3, abs(rnd.gauss(0.0, 1.0))) * 0.3)  # Cauchy-like
        model = ExponentialSmoothing(0.3, horizon=1, coverage=0.9)
        _, own = drive(model, ys)
        hits = sum(o.lower[0] <= y <= o.upper[0] for o, y in zip(own[1:-1], ys[2:]) if o.mean)
        own_rate = 1.0 - hits / len(ys[2:])
        _, outs = drive(ConformalForecast(model, alpha=0.1), ys)
        self.assertGreater(abs(own_rate - 0.1), 0.03)
        self.assertLess(abs(outs[-1].miscoverage[0] - 0.1), 0.01)

    def test_unnormalized_needs_an_initial_width(self):
        with self.assertRaises(ValueError):
            ConformalForecast(ExponentialSmoothing(0.5), normalize=False)
        ConformalForecast(ExponentialSmoothing(0.5), normalize=False, initial=2.0)


class Comparison(unittest.TestCase):
    def test_covers_the_average_difference_at_every_time(self):
        misses = 0
        runs = 200
        for r in range(runs):
            rnd = random.Random(1000 + r)
            procedure = ForecastComparison((0.0, 1.0), alpha=0.1)
            state = procedure.initial_state()
            total = 0.0
            for t in range(1, 1001):
                drift = 0.3 * math.sin(t / 50.0)  # E[d_t] drifts; the target is its running average
                total += drift
                a = rnd.random() < 0.5 + drift / 2
                state, out = procedure.step(state, {"a": float(a), "b": 0.5})
                if not out.lower <= total / t <= out.upper:
                    misses += 1
                    break
        self.assertLessEqual(misses / runs, 0.1)

    def test_finds_the_better_forecaster(self):
        ys = ar_series(3000, 18)
        good, naive = AutoRegressive(3), ExponentialSmoothing(1.0, sigma=1.0)
        sg, sn = good.initial_state(), naive.initial_state()
        comparison = ForecastComparison((0.0, 5.0))
        sc = comparison.initial_state()
        decided = None
        for y in ys:
            pg, pn = good.predictive(sg), naive.predictive(sn)
            if pg and pn:
                sc, out = comparison.step(sc, {"a": min(5.0, abs(y - pg.mean[0])), "b": min(5.0, abs(y - pn.mean[0]))})
                if out.better and decided is None:
                    decided = out.n, out.better
            sg, _, _ = good.update(sg, {"y": y})
            sn, _, _ = naive.update(sn, {"y": y})
        self.assertIsNotNone(decided)
        self.assertEqual(decided[1], "a")

    def test_stitched_boundary_formula(self):
        v, c, alpha, m = 50.0, 2.0, 0.05, 1.0
        ell = 1.4 * math.log(math.log(2.0 * v / m)) + math.log(3.105547277977581 / (alpha * math.log(2.0) ** 1.4))
        k1, k2 = (2 ** 0.25 + 2 ** -0.25) / math.sqrt(2.0), (math.sqrt(2.0) + 1.0) / 2.0
        expected = math.sqrt(k1 ** 2 * v * ell + k2 ** 2 * c ** 2 * ell ** 2) + c * k2 * ell
        self.assertAlmostEqual(stitched_boundary(v, c, alpha, m), expected, delta=1e-12)
        self.assertEqual(stitched_boundary(0.1, c, alpha, m), stitched_boundary(m, c, alpha, m))


class Optimizer(unittest.TestCase):
    def test_nelder_mead_finds_a_minimum(self):
        x, value, _ = nelder_mead(lambda p: (p[0] - 1.0) ** 2 + 10.0 * (p[1] + 2.0) ** 2, [0.0, 0.0])
        self.assertAlmostEqual(x[0], 1.0, delta=1e-4)
        self.assertAlmostEqual(x[1], -2.0, delta=1e-4)
        self.assertLess(value, 1e-8)


class Refusals(unittest.TestCase):
    def test_parameters(self):
        bad = [
            lambda: ExponentialSmoothing(0.0),
            lambda: ExponentialSmoothing(0.5, beta=0.1),
            lambda: ExponentialSmoothing(0.5, phi=0.9, trend="additive"),
            lambda: ExponentialSmoothing(0.5, gamma=0.1),
            lambda: ExponentialSmoothing(0.5, period=4, initial_seasonal=(1.0, 2.0)),
            lambda: AutoRegressive(0),
            lambda: AutoRegressive(2, coefficients=(0.1,), sigma=1.0),
            lambda: AutoRegressive(1, coefficients=(0.1,)),
            lambda: AutoRegressive(1, forgetting=0.0),
            lambda: ForecastComparison((1.0, 1.0)),
            lambda: fit_ets([1.0, 2.0, 3.0], trend="additive", period=12),
            lambda: fit_ar([1.0, 2.0, 3.0], 2),
        ]
        for make in bad:
            with self.assertRaises(ValueError):
                make()


if __name__ == "__main__":
    unittest.main()
