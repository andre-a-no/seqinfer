# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Mathematical correctness of the smoothers, against independent references (tests/oracles.py).

* Rauch-Tung-Striebel: E[z_t | y_1..T] and Cov[z_t | y_1..T] from the joint
  Gaussian distribution of all states and observations, conditioned on all
  observations at once, in exact rational arithmetic.
* Forward-backward: P(z_t = k | y_1..T) by summing over all K^T paths.
* Viterbi: the most probable path by enumeration of all paths.
"""
import math
import random
import unittest

from sequential_analysis import Recorder, Run, run_sync
from sequential_analysis.errors import NumericalError
from sequential_analysis.procedures import (
    CategoricalEmission,
    GaussianEmission,
    HiddenMarkovFilter,
    KalmanFilter,
    PoissonEmission,
    RobustKalmanFilter,
)
from sequential_analysis.smoothing import forward_backward, kalman_smoother, viterbi
from sequential_analysis.sources import from_values

from . import test_filters
from .oracles import gaussian_smoother, hmm_most_probable_path, hmm_smoother
from .test_filters import _m, normal_density


class RauchTungStriebel(unittest.TestCase):
    def test_against_exact_conditioning_on_all_observations(self):
        rnd = random.Random(1)
        for label, spec in test_filters.KalmanAgainstExactConditioning.MODELS.items():
            kf = KalmanFilter(**spec)
            m = kf.n_obs
            ys = [[rnd.gauss(0, 2) for _ in range(m)] if m > 1 else rnd.gauss(0, 2) for _ in range(6)]
            us = [rnd.gauss(0, 1) for _ in range(6)]
            inputs = [{"y": y, "u": u} if spec.get("control") else {"y": y} for y, u in zip(ys, us, strict=True)]
            result = kalman_smoother(kf, inputs)
            ref = gaussian_smoother(_m(spec["transition"]), _m(spec["observation"]), _m(spec["process_noise"]),
                                    _m(spec["observation_noise"]), spec["initial_mean"],
                                    _m(spec["initial_covariance"]), ys, B=spec.get("control"), us=us)
            for t, (mean, cov, (rmean, rcov)) in enumerate(zip(result.means, result.covariances, ref, strict=True)):
                with self.subTest(label, t=t):
                    for a, b in zip(mean, rmean, strict=True):
                        self.assertAlmostEqual(a, b, delta=1e-11 * max(1.0, abs(b)))
                    for row, rrow in zip(cov, rcov, strict=True):
                        for a, b in zip(row, rrow, strict=True):
                            self.assertAlmostEqual(a, b, delta=1e-11 * max(1.0, abs(b)))

    def test_last_point_equals_the_filter_and_smoothing_never_adds_variance(self):
        rnd = random.Random(2)
        kf = KalmanFilter.local_linear_trend(0.1, 0.01, 1.0)
        inputs = [{"y": 0.3 * t + rnd.gauss(0, 1)} for t in range(200)]
        result = kalman_smoother(kf, inputs)
        self.assertEqual(result.means[-1], result.filtered_means[-1])
        for s, f in zip(result.covariances, result.filtered_covariances, strict=True):
            self.assertLessEqual(s[0][0], f[0][0] + 1e-12)
            self.assertLessEqual(s[1][1], f[1][1] + 1e-12)

    def test_smoothed_errors_are_calibrated(self):
        # simulate the model; (z_t - m_t|T) / sqrt(P_t|T) is standard normal
        rnd = random.Random(3)
        q, r = 0.2, 1.0
        z, zs, inputs = 0.0, [], []
        for _ in range(4000):
            z += rnd.gauss(0, math.sqrt(q))
            zs.append(z)
            inputs.append({"y": z + rnd.gauss(0, math.sqrt(r))})
        result = kalman_smoother(KalmanFilter(1.0, 1.0, q, r, (0.0,), 1.0), inputs)
        u = [(zt - m[0]) / math.sqrt(p[0][0]) for zt, m, p in zip(zs, result.means, result.covariances, strict=True)]
        mean = math.fsum(u) / len(u)
        var = math.fsum(v * v for v in u) / len(u)
        self.assertLess(abs(mean), 0.1)  # autocorrelated errors: a loose bound
        self.assertAlmostEqual(var, 1.0, delta=0.1)

    def test_likelihood_matches_the_run(self):
        kf = KalmanFilter.autoregressive((0.6, 0.2), 0.5, 0.3)
        values = [math.sin(t / 3) for t in range(40)]
        rec = Recorder()
        run_sync(Run(kf, consumers=[rec]), from_values("y", values))
        result = kalman_smoother(kf, [{"y": v} for v in values])
        self.assertEqual(result.log_likelihood, rec.outputs[-1].log_likelihood)
        self.assertEqual(result.filtered_means[-1], rec.outputs[-1].mean)

    def test_robust_filter_is_smoothed_without_a_likelihood(self):
        kf = RobustKalmanFilter.local_linear_trend(0.01, 0.001, 1.0)
        result = kalman_smoother(kf, [{"y": 0.1 * t + (40.0 if t == 20 else 0.0)} for t in range(50)])
        self.assertIsNone(result.log_likelihood)
        self.assertLess(abs(result.means[20][0] - 2.0), 1.0)  # the outlier is discounted

    def test_singular_prediction_is_reported(self):
        kf = KalmanFilter(((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0),), ((0.0, 0.0), (0.0, 0.0)), 1.0, (0.0, 0.0),
                          ((1.0, 0.0), (0.0, 0.0)))
        with self.assertRaises(NumericalError):
            kalman_smoother(kf, [{"y": 1.0}, {"y": 2.0}])
        with self.assertRaises(ValueError):
            kalman_smoother(kf, [])


class ForwardBackward(unittest.TestCase):
    A = ((0.9, 0.07, 0.03), (0.1, 0.8, 0.1), (0.2, 0.3, 0.5))
    PI = (0.5, 0.3, 0.2)

    def emissions(self):
        rnd = random.Random(7)
        means, sds = (0.0, 2.0, 5.0), (1.0, 0.5, 2.0)
        rates = (1.0, 4.0, 10.0)
        table = ((0.7, 0.2, 0.1), (0.1, 0.8, 0.1), (0.3, 0.3, 0.4))
        yield (GaussianEmission(means, sds), lambda k, y: normal_density(y, means[k], sds[k]),
               [rnd.gauss(2, 3) for _ in range(7)])
        yield (PoissonEmission(rates), lambda k, y: rates[k] ** y * math.exp(-rates[k]) / math.factorial(y),
               [rnd.randrange(12) for _ in range(7)])
        yield CategoricalEmission(table), lambda k, y: table[k][y], [rnd.randrange(3) for _ in range(7)]

    def test_against_enumeration_of_paths(self):
        for emission, density, ys in self.emissions():
            hmm = HiddenMarkovFilter(self.PI, self.A, emission)
            result = forward_backward(hmm, [{"y": y} for y in ys])
            ref = hmm_smoother(self.PI, self.A, density, ys)
            for t, (row, rrow) in enumerate(zip(result.probabilities, ref, strict=True)):
                for a, b in zip(row, rrow, strict=True):
                    self.assertAlmostEqual(a, b, delta=1e-12, msg=f"{type(emission).__name__} t={t}")
            self.assertEqual(result.probabilities[-1], result.filtered[-1])

    def test_viterbi_against_enumeration(self):
        for emission, density, ys in self.emissions():
            hmm = HiddenMarkovFilter(self.PI, self.A, emission)
            path = viterbi(hmm, [{"y": y} for y in ys])
            best, log_p = hmm_most_probable_path(self.PI, self.A, density, ys)
            self.assertAlmostEqual(path.log_probability, log_p, delta=1e-12 * abs(log_p))
            if not isinstance(emission, CategoricalEmission):  # no ties with continuous densities
                self.assertEqual(path.states, best)

    def test_long_series_far_in_the_tails(self):
        hmm = HiddenMarkovFilter((0.5, 0.5), ((0.99, 0.01), (0.02, 0.98)), GaussianEmission((0.0, 50.0), (1.0, 1.0)))
        ys = [0.0] * 300 + [50.0] * 300 + [0.0] * 300
        result = forward_backward(hmm, [{"y": y} for y in ys])
        self.assertTrue(all(abs(math.fsum(p) - 1.0) < 1e-12 for p in result.probabilities))
        self.assertEqual(viterbi(hmm, [{"y": y} for y in ys]).states, tuple([0] * 300 + [1] * 300 + [0] * 300))

    def test_impossible_observations(self):
        hmm = HiddenMarkovFilter((1.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), CategoricalEmission(((1.0, 0.0), (0.0, 1.0))))
        with self.assertRaises(NumericalError):
            viterbi(hmm, [{"y": 1}])
        with self.assertRaises(ValueError):
            forward_backward(hmm, [])


if __name__ == "__main__":
    unittest.main()
