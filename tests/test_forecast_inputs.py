# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Forecasting with regressors, of counts, and on the log scale.

* Regressors: with fixed effects, exponential smoothing is the same
  smoothing of y - b'z, shifted by the planned b'z; the learnt effects are
  exactly the ridge regression of the errors of the run without
  regressors on u (the decomposition of the module docstring), and the
  reported states are those of the model with these effects.  fit_ar with
  regressors is statsmodels' AutoReg with exog, forecasts and intervals
  included.  fit_ets and the learnt models recover a simulated effect.
* Counts: the predictive probabilities are those of a Poisson count with
  a gamma rate, checked by numerical integration; the recursion is the
  exact conjugate update; the integer intervals hold at least their
  coverage; fit_counts finds a discount below 1 when the rate moves.
* Log transform: the forecasts are those of the model of log y mapped
  back, the log scores carry the Jacobian, and mixtures of transformed
  and untransformed models have the right coverage.
"""
import math
import random
import unittest

from sequential_analysis import Run
from sequential_analysis.errors import ContractViolation
from sequential_analysis.forecasting import fit_ar, fit_counts, fit_ets, forecast_records
from sequential_analysis.procedures.forecast import (
    AutoRegressive,
    ConformalForecast,
    ExponentialSmoothing,
    ModelAveraging,
    PoissonSmoothing,
    averaged_counts,
    averaged_smoothing,
    negative_binomial_cdf,
    negative_binomial_log_pmf,
)

try:
    import numpy as np
    from statsmodels.tsa.ar_model import AutoReg
except ImportError:  # optional: the comparison is skipped
    AutoReg = None


def drive(procedure, records):
    state, outs = procedure.initial_state(), []
    for x in records:
        state, out = procedure.step(state, x)
        outs.append(out)
    return state, outs


def promo_series(n, seed, effect=20.0, period=7, horizon=3):
    """A wandering level, a weekly season and a promotion every 10 days or so: (records, promo column)."""
    rnd = random.Random(seed)
    promo = [1.0 if rnd.random() < 0.1 else 0.0 for _ in range(n + horizon)]
    level, ys = 100.0, []
    for t in range(n):
        level += rnd.gauss(0.0, 1.0)
        ys.append(level + 5.0 * math.sin(2.0 * math.pi * t / period) + effect * promo[t] + rnd.gauss(0.0, 2.0))
    return forecast_records(ys, {"promo": promo}, horizon=horizon), promo


def solve(a, b):
    n = len(a)
    m = [[*row, v] for row, v in zip(a, b)]
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


class Records(unittest.TestCase):
    def test_windows(self):
        records = forecast_records([1.0, 2.0], {"z": [10, 11, 12, 13]}, horizon=2)
        self.assertEqual(records, [{"y": 1.0, "z": [10.0, 11.0, 12.0]}, {"y": 2.0, "z": [11.0, 12.0, 13.0]}])
        with self.assertRaises(ValueError):
            forecast_records([1.0, 2.0], {"z": [10, 11, 12]}, horizon=2)


class SmoothingWithRegressors(unittest.TestCase):
    def test_fixed_effects_smooth_y_minus_the_effects(self):
        records, _promo = promo_series(120, 1)
        common = {"beta": 0.05, "gamma": 0.1, "trend": "additive", "period": 7, "sigma": 2.0, "horizon": 3}
        with_z = ExponentialSmoothing(0.3, regressors=("promo",), coefficients=(20.0,), **common)
        plain = ExponentialSmoothing(0.3, **common)
        _, a = drive(with_z, records)
        _, b = drive(plain, [{"y": x["y"] - 20.0 * x["promo"][0]} for x in records])
        for oa, ob, x in zip(a, b, records):
            self.assertAlmostEqual(oa.log_score or 0.0, ob.log_score or 0.0, delta=1e-12)
            for h in range(3):
                self.assertAlmostEqual(oa.mean[h], ob.mean[h] + 20.0 * x["promo"][h + 1], delta=1e-9)

    def test_learnt_effects_are_the_regression_on_u(self):
        records, promo = promo_series(150, 2)
        alpha, beta, gamma, ridge = 0.3, 0.02, 0.1, 0.5
        model = ExponentialSmoothing(alpha, beta=beta, gamma=gamma, trend="additive", period=7,
                                     regressors=("promo",), ridge=ridge, horizon=3)
        _state, outs = drive(model, records)
        # brute force: the errors of the run without regressors, and u from the run driven by the promotion
        ys = [x["y"] for x in records]
        level, trend, season = ys[0], 0.0, [0.0] * 7
        a_level, a_trend, a_season = promo[0], 0.0, [0.0] * 7
        season, a_season = [*season[1:], 0.0], [*a_season[1:], 0.0]
        errors, us = [], []
        for t in range(1, len(ys)):
            e = ys[t] - (level + trend + season[0])
            u = promo[t] - (a_level + a_trend + a_season[0])
            errors.append(e)
            us.append(u)
            level, trend, season = level + trend + alpha * e, trend + beta * e, [*season[1:], season[0] + gamma * e]
            a_level, a_trend, a_season = (a_level + a_trend + alpha * u, a_trend + beta * u,
                                          [*a_season[1:], a_season[0] + gamma * u])
        b = sum(u * e for u, e in zip(us, errors)) / (sum(u * u for u in us) + ridge)
        self.assertAlmostEqual(outs[-1].parameters["coefficients"][0], b, delta=1e-9)
        # the reported states are those of the model with this effect
        fixed = ExponentialSmoothing(alpha, beta=beta, gamma=gamma, trend="additive", period=7, sigma=1.0,
                                     regressors=("promo",), coefficients=(b,), horizon=3)
        fixed_state, _ = drive(fixed, records)
        self.assertAlmostEqual(outs[-1].parameters["level"], fixed_state.level, delta=1e-9)
        self.assertAlmostEqual(outs[-1].parameters["trend"], fixed_state.trend, delta=1e-9)
        for v, w in zip(outs[-1].parameters["seasonal"], fixed_state.seasonal):
            self.assertAlmostEqual(v, w, delta=1e-9)
        # and the forecasts are that model's, with the posterior mean effect
        pred = fixed.predictive(fixed_state)
        for h in range(3):
            self.assertAlmostEqual(outs[-1].mean[h], pred.mean[h], delta=1e-8)

    def test_one_step_score_is_the_exact_student_t(self):
        records, promo = promo_series(40, 3)
        model = ExponentialSmoothing(0.4, regressors=("promo",), ridge=1.0)
        state = model.initial_state()
        ys = [x["y"] for x in records]
        level, a_level, errors, us = ys[0], promo[0], [], []
        for t, x in enumerate(records):
            new, _, score = model.update(state, x)
            if t >= 1:
                e, u = ys[t] - level, promo[t] - a_level
                m = 1.0 + sum(v * v for v in us)
                b = sum(v * w for v, w in zip(us, errors)) / m
                rate = 0.5 * sum(w * w for w in errors) - 0.5 * b * b * m
                shape = 0.5 * len(errors)
                if score is not None:
                    s = 1.0 + u * u / m
                    scale = math.sqrt(rate / shape * s)
                    z = (e - b * u) / scale
                    df = 2.0 * shape
                    expected = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
                                - math.log(scale) - (df + 1) / 2 * math.log1p(z * z / df))
                    self.assertAlmostEqual(score, expected, delta=1e-9)
                errors.append(e)
                us.append(u)
                level, a_level = level + 0.4 * e, a_level + 0.4 * u
            state = new

    def test_recovers_the_effect(self):
        records, _ = promo_series(730, 4)
        fit = fit_ets(records[:-3], trend="additive", period=7, regressors=("promo",), horizon=3)
        self.assertAlmostEqual(fit.parameters["coefficients"][0], 20.0, delta=1.0)
        self.assertAlmostEqual(fit.state.log_likelihood + fit.state.comp, fit.log_likelihood,
                               delta=1e-9 * abs(fit.log_likelihood))
        plain = fit_ets([x["y"] for x in records[:-3]], trend="additive", period=7)
        self.assertGreater(fit.log_likelihood, plain.log_likelihood)
        _, outs = drive(averaged_smoothing(trend="additive", period=7, regressors=("promo",),
                                           alphas=(0.1, 0.3), betas=(0.01,), gammas=(0.05,), horizon=3), records)
        self.assertEqual(len(outs[-1].mean), 3)
        _, learnt = drive(ExponentialSmoothing(0.3, beta=0.01, gamma=0.05, trend="additive", period=7,
                                               regressors=("promo",), horizon=3), records)
        self.assertAlmostEqual(learnt[-1].parameters["coefficients"][0], 20.0, delta=1.0)

    def test_needs_the_regressors(self):
        model = ExponentialSmoothing(0.3, regressors=("promo",))
        run = Run(model, on_invalid="raise")
        with self.assertRaises(ContractViolation):
            run.step({"y": 1.0})
        with self.assertRaises(ValueError):
            ExponentialSmoothing(0.3, regressors=("promo",), sigma=1.0)  # learnt effects learn sigma too
        with self.assertRaises(ValueError):
            ExponentialSmoothing(0.3, regressors=("y",))


class AutoregressionWithRegressors(unittest.TestCase):
    @unittest.skipIf(AutoReg is None, "statsmodels is not installed")
    def test_fit_ar_is_autoreg_with_exog(self):
        rnd = random.Random(5)
        n, horizon = 300, 4
        z = [rnd.gauss(0.0, 1.0) for _ in range(n + horizon)]
        ys = [0.0, 0.0]
        for t in range(2, n):
            ys.append(1.0 + 0.5 * ys[-1] - 0.2 * ys[-2] + 3.0 * z[t] + rnd.gauss(0.0, 1.0))
        records = forecast_records(ys, {"z": z}, horizon=horizon)
        fit = fit_ar(records, 2, regressors=("z",), horizon=horizon)
        res = AutoReg(np.array(ys), 2, trend="c", exog=np.array(z[:n])[:, None]).fit()
        ours = [fit.parameters["intercept"], *fit.parameters["coefficients"], *fit.parameters["effects"]]
        for a, b in zip(ours, res.params):
            self.assertAlmostEqual(a, float(b), delta=1e-9)
        frame = res.get_prediction(start=n, end=n + horizon - 1,
                                   exog_oos=np.array(z[n:])[:, None]).summary_frame(alpha=0.1)
        pred = fit.model.predictive(fit.state)
        lower, upper = fit.model.interval(pred)
        for h in range(horizon):
            self.assertAlmostEqual(pred.mean[h], float(frame["mean"].iloc[h]), delta=1e-9)
            self.assertAlmostEqual(lower[h], float(frame["mean_ci_lower"].iloc[h]), delta=1e-9)
            self.assertAlmostEqual(upper[h], float(frame["mean_ci_upper"].iloc[h]), delta=1e-9)

    def test_learnt_with_regressors_is_the_ridge_regression(self):
        rnd = random.Random(6)
        n = 200
        z = [float(rnd.random() < 0.2) for _ in range(n + 1)]
        ys = [0.0]
        for t in range(1, n):
            ys.append(2.0 + 0.6 * ys[-1] + 5.0 * z[t] + rnd.gauss(0.0, 1.0))
        records = forecast_records(ys, {"z": z})
        _, outs = drive(AutoRegressive(1, regressors=("z",), ridge=0.3), records)
        gram = [[0.3 if i == j else 0.0 for j in range(3)] for i in range(3)]
        rhs = [0.0] * 3
        for t in range(1, n):
            x = [1.0, ys[t - 1], z[t]]
            for i in range(3):
                rhs[i] += x[i] * ys[t]
                for j in range(3):
                    gram[i][j] += x[i] * x[j]
        theta = solve(gram, rhs)
        p = outs[-1].parameters
        for a, b in zip([p["intercept"], *p["coefficients"], *p["effects"]], theta):
            self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)))
        self.assertAlmostEqual(p["effects"][0], 5.0, delta=0.5)


def poisson(rnd, mean):
    k, p, limit = 0, rnd.random(), math.exp(-mean)
    while p > limit:
        k += 1
        p *= rnd.random()
    return k


def mixture_probability(k, size, rate, exposure):
    """P(Y = k) by integrating Poisson(k; e mu) Gamma(mu; size, rate) over mu, on a fine grid in log mu.

    The grid goes down to mu = e^-40: with size < 1 the density is large near 0.
    """
    def f(mu):
        return math.exp(k * math.log(exposure * mu) - exposure * mu - math.lgamma(k + 1)
                        + size * math.log(rate) + (size - 1) * math.log(mu) - rate * mu
                        - math.lgamma(size)) * mu  # d mu = mu d log mu

    step = 52.0 / 40000
    grid = [math.exp(-40.0 + step * i) for i in range(40001)]
    return sum(f(m) for m in grid) * step - 0.5 * step * (f(grid[0]) + f(grid[-1]))


def mixture_cdf(comps, h, k):
    return sum(w * p.cdf(h, k) for w, _, p in comps)


class Counts(unittest.TestCase):
    def test_probabilities_are_the_poisson_gamma_mixture(self):
        for size, rate, exposure in ((2.5, 0.8, 1.0), (40.0, 3.0, 2.0), (0.7, 0.2, 0.5)):
            total = 0.0
            for k in range(0, 400):
                p = math.exp(negative_binomial_log_pmf(k, size, rate, exposure))
                total += p
                if k in (0, 3, 17):
                    self.assertAlmostEqual(p, mixture_probability(k, size, rate, exposure), delta=1e-8)
                if total > 1.0 - 1e-15:
                    break
                self.assertAlmostEqual(negative_binomial_cdf(k, size, rate, exposure), total, delta=1e-12)
            self.assertAlmostEqual(total, 1.0, delta=1e-12)

    def test_conjugate_update(self):
        model = PoissonSmoothing(0.8, prior=(1.0, 0.5))
        state = model.initial_state()
        a, b = 1.0, 0.5
        for y in (3, 0, 5, 2, 7):
            state, error, score = model.update(state, {"y": y})
            a, b = 0.8 * a, 0.8 * b
            self.assertAlmostEqual(score, negative_binomial_log_pmf(y, a, b, 1.0), delta=1e-14)
            self.assertAlmostEqual(error, y - a / b, delta=1e-12)
            a, b = a + y, b + 1.0
            self.assertAlmostEqual(state.shape, a, delta=1e-12)
            self.assertAlmostEqual(state.rate, b, delta=1e-12)

    def test_constant_rate_forecast_is_the_average(self):
        ys = [3, 5, 2, 4, 6]
        _, outs = drive(PoissonSmoothing(1.0, horizon=2), [{"y": y} for y in ys])
        self.assertAlmostEqual(outs[-1].mean[0], sum(ys) / len(ys), delta=1e-12)
        self.assertEqual(outs[-1].mean[0], outs[-1].mean[1])

    def test_intervals_are_counts_with_their_coverage(self):
        rnd = random.Random(7)
        model = PoissonSmoothing(0.95, coverage=0.8)
        state, inside, total = model.initial_state(), 0, 0
        for t in range(3000):
            y = poisson(rnd, 4.0)
            pred = model.predictive(state)
            if pred is not None and t > 100:
                lower, upper = model.interval(pred)
                self.assertEqual(lower[0], int(lower[0]))
                self.assertLess(pred.cdf(0, lower[0] - 1), 0.1)
                self.assertGreaterEqual(pred.cdf(0, upper[0]), 0.9)
                inside += lower[0] <= y <= upper[0]
                total += 1
            state, _, _ = model.update(state, {"y": y})
        self.assertGreaterEqual(inside / total, 0.8)

    def test_exposure_scales_the_forecast(self):
        records = forecast_records([4, 8, 6, 12], {"hours": [1.0, 2.0, 1.5, 3.0, 2.0, 0.5]}, horizon=2)
        _, outs = drive(PoissonSmoothing(1.0, exposure="hours", horizon=2), records)
        rate = 30.0 / 7.5
        self.assertAlmostEqual(outs[-1].mean[0], 2.0 * rate, delta=1e-12)
        self.assertAlmostEqual(outs[-1].mean[1], 0.5 * rate, delta=1e-12)

    def test_fit_counts_follows_a_moving_rate(self):
        rnd = random.Random(8)
        log_rate, moving, still = math.log(5.0), [], []
        for _ in range(800):
            log_rate += rnd.gauss(0.0, 0.1)
            moving.append(poisson(rnd, math.exp(log_rate)))
            still.append(poisson(rnd, 5.0))
        fit_moving, fit_still = fit_counts(moving), fit_counts(still)
        self.assertLess(fit_moving.parameters["discount"], 0.97)
        self.assertGreater(fit_still.parameters["discount"], 0.98)
        self.assertAlmostEqual(fit_moving.state.log_likelihood + fit_moving.state.comp, fit_moving.log_likelihood,
                               delta=1e-9)

    def test_averaged_counts_and_conformal(self):
        rnd = random.Random(9)
        records = [{"y": poisson(rnd, 3.0 + 2.0 * (t > 150))} for t in range(300)]
        avg = averaged_counts(horizon=2, coverage=0.8)
        state, outs = drive(avg, records)
        comps = avg._components(state)
        for h in range(2):
            lower, upper = outs[-1].lower[h], outs[-1].upper[h]
            self.assertEqual(lower, int(lower))
            self.assertLess(mixture_cdf(comps, h, lower - 1), 0.1)
            self.assertGreaterEqual(mixture_cdf(comps, h, lower), 0.1 - 1e-9)
            self.assertGreaterEqual(mixture_cdf(comps, h, upper), 0.9 - 1e-9)
            self.assertLess(mixture_cdf(comps, h, upper - 1), 0.9)
        _, conformal = drive(ConformalForecast(PoissonSmoothing(0.9)), records)
        self.assertEqual(conformal[-1].counts[0], 299)  # forecasts from the first non-zero count on

    def test_refuses_negative_counts(self):
        run = Run(PoissonSmoothing(0.9), on_invalid="raise")
        with self.assertRaises(ContractViolation):
            run.step({"y": -1})
        with self.assertRaises(ValueError):
            ModelAveraging([PoissonSmoothing(0.9), ExponentialSmoothing(0.3)])


class LogTransform(unittest.TestCase):
    def test_forecasts_are_the_log_model_mapped_back(self):
        rnd = random.Random(10)
        level, ys = 3.0, []
        for _ in range(100):
            level += rnd.gauss(0.0, 0.05)
            ys.append(math.exp(level + rnd.gauss(0.0, 0.1)))
        on_log = ExponentialSmoothing(0.3, horizon=2, coverage=0.8)
        mapped = ExponentialSmoothing(0.3, transform="log", horizon=2, coverage=0.8)
        _, a = drive(on_log, [{"y": math.log(y)} for y in ys])
        _, b = drive(mapped, [{"y": y} for y in ys])
        for oa, ob, y in zip(a, b, ys):
            if oa.log_score is not None:
                self.assertAlmostEqual(ob.log_score, oa.log_score - math.log(y), delta=1e-12)
            for h in range(len(oa.mean)):
                self.assertAlmostEqual(ob.mean[h], math.exp(oa.mean[h]), delta=1e-12 * ob.mean[h])
                self.assertAlmostEqual(ob.lower[h], math.exp(oa.lower[h]), delta=1e-12 * ob.mean[h])
                self.assertAlmostEqual(ob.upper[h], math.exp(oa.upper[h]), delta=1e-12 * ob.mean[h])

    def test_fit_reports_the_likelihood_of_y(self):
        rnd = random.Random(11)
        ys = [math.exp(2.0 + 0.3 * math.sin(t / 3.0) + rnd.gauss(0.0, 0.1)) for t in range(80)]
        on_log = fit_ets([math.log(y) for y in ys])
        mapped = fit_ets(ys, transform="log")
        self.assertAlmostEqual(mapped.log_likelihood, on_log.log_likelihood - sum(math.log(y) for y in ys),
                               delta=1e-6)
        self.assertAlmostEqual(mapped.state.log_likelihood + mapped.state.comp, mapped.log_likelihood, delta=1e-9)
        ar = fit_ar(ys, 1, transform="log")
        self.assertAlmostEqual(ar.state.log_likelihood + ar.state.comp, ar.log_likelihood, delta=1e-9)

    def test_needs_positive_values(self):
        model = ExponentialSmoothing(0.3, transform="log")
        with self.assertRaises(ContractViolation):
            model.update(model.initial_state(), {"y": 0.0})
        with self.assertRaises(ValueError):
            fit_ets([1.0, 2.0, 0.0, 3.0, 4.0, 5.0], transform="log")

    def test_mixture_of_log_and_raw_models(self):
        rnd = random.Random(12)
        ys = [math.exp(1.0 + rnd.gauss(0.0, 0.3)) for _ in range(60)]
        avg = ModelAveraging([ExponentialSmoothing(0.2, transform="log", horizon=2),
                              ExponentialSmoothing(0.2, horizon=2)], coverage=0.8)
        state, outs = drive(avg, [{"y": y} for y in ys])
        comps = avg._components(state)
        for h in range(2):
            for bound, target in ((outs[-1].lower[h], 0.1), (outs[-1].upper[h], 0.9)):
                self.assertAlmostEqual(sum(w * p.cdf(h, bound) for w, _, p in comps), target, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
