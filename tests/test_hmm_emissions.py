# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The multivariate Gaussian and Student t emissions of the hidden Markov model.

* Densities against independent formulas (exact rational linear algebra
  for the multivariate Gaussian, Gamma functions for the t).
* The filter, forward-backward and Viterbi with these emissions against
  enumeration of every path of the hidden chain (tests/oracles.py).
* EM: the M-steps (`from_statistics`) against direct weighted estimates
  and, for the t, against the maximum of the likelihood found by a direct
  golden-section search; the likelihood never decreases over EM iterations.
* Edge cases: vectors of the wrong length, (nearly) singular covariances,
  very large degrees of freedom (the Gaussian limit), far outliers.
* `emission_from_spec` for every emission family.
"""
from __future__ import annotations

import math
import random
import unittest
from fractions import Fraction

from sequential_analysis._compat import zip_strict
from sequential_analysis.errors import ContractViolation
from sequential_analysis.procedures.hmm import (
    CategoricalEmission,
    GaussianEmission,
    HiddenMarkovFilter,
    MultivariateGaussianEmission,
    PoissonEmission,
    StudentTEmission,
    emission_from_spec,
)
from sequential_analysis.run import Run
from sequential_analysis.smoothing import forward_backward, viterbi

from .oracles import hmm_filter, hmm_most_probable_path, hmm_smoother

A = ((0.9, 0.07, 0.03), (0.1, 0.8, 0.1), (0.2, 0.3, 0.5))
PI = (0.5, 0.3, 0.2)

MEANS2 = ((0.0, 0.0), (2.0, -1.0), (-1.0, 3.0))
COVS2 = (((1.0, 0.3), (0.3, 0.5)), ((2.0, -0.8), (-0.8, 1.0)), ((0.4, 0.0), (0.0, 3.0)))


def fold(proc, ys):
    state, outs = proc.initial_state(), []
    for y in ys:
        state, out = proc.step(state, {"y": y})
        outs.append(out)
    return outs


def exact_log_mvn(y, mean, cov):
    """log N(y; mean, cov) with the quadratic form and determinant in exact rational arithmetic."""
    d = len(y)
    a = [[Fraction(cov[i][j]) for j in range(d)] + [Fraction(y[i]) - Fraction(mean[i])] for i in range(d)]
    det = Fraction(1)
    for c in range(d):  # Gaussian elimination without pivoting (positive definite)
        det *= a[c][c]
        for r in range(c + 1, d):
            f = a[r][c] / a[c][c]
            for j in range(c, d + 1):
                a[r][j] -= f * a[c][j]
    # back substitution for x = cov^-1 r, then q = r . x
    x = [Fraction(0)] * d
    for i in reversed(range(d)):
        s = a[i][d]
        for j in range(i + 1, d):
            s -= a[i][j] * x[j]
        x[i] = s / a[i][i]
    q = sum((Fraction(y[i]) - Fraction(mean[i])) * x[i] for i in range(d))
    return -0.5 * float(q) - 0.5 * math.log(det) - 0.5 * d * math.log(2.0 * math.pi)


def t_density(y, mean, scale, df):
    z = (y - mean) / scale
    return (math.gamma((df + 1) / 2) / (math.gamma(df / 2) * math.sqrt(df * math.pi) * scale)
            * (1.0 + z * z / df) ** (-(df + 1) / 2))


def t_sample(rnd, mean, scale, df):
    return mean + scale * rnd.gauss(0.0, 1.0) / math.sqrt(rnd.gammavariate(df / 2, 2.0 / df))


def golden_max(f, a, b, tol=1e-12):
    """The maximiser of a unimodal f on [a, b], by golden-section search."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol * (1.0 + abs(a) + abs(b)):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def totals_for(emission, ys, weights):
    """totals[k] = (sum_t w_tk, sum_t w_tk g_k(y_t)...), by math.fsum."""
    out = []
    for k in range(emission.states):
        rows = [emission.statistics(k, y) for y in ys]
        w = [weights[t][k] for t in range(len(ys))]
        out.append((math.fsum(w), *(math.fsum(wt * g[i] for wt, g in zip_strict(w, rows))
                                   for i in range(len(rows[0])))))
    return out


class MultivariateGaussian(unittest.TestCase):
    def test_log_density_against_exact_linear_algebra(self):
        rnd = random.Random(1)
        mean = (0.5, -1.0, 2.0)
        cov = ((2.0, 0.6, -0.3), (0.6, 1.5, 0.4), (-0.3, 0.4, 0.8))
        e = MultivariateGaussianEmission([mean], [cov])
        for _ in range(20):
            y = [rnd.gauss(0, 2) for _ in range(3)]
            ref = exact_log_mvn(y, mean, cov)
            self.assertAlmostEqual(e.log_density(0, y), ref, delta=1e-13 * max(1.0, abs(ref)))

    def test_one_dimension_is_the_gaussian_emission(self):
        g = GaussianEmission((1.0, -2.0), (2.0, 0.5))
        m = MultivariateGaussianEmission([[1.0], [-2.0]], [[[4.0]], [[0.25]]])
        for y in (-7.0, 0.0, 1.3, 25.0):
            for k in range(2):
                a, b = g.log_density(k, y), m.log_density(k, [y])
                self.assertAlmostEqual(a, b, delta=4e-16 * abs(a))
        self.assertEqual(m.statistics(0, [3.0]), g.statistics(0, 3.0))

    def test_filter_smoother_and_viterbi_against_enumeration(self):
        rnd = random.Random(2)
        e = MultivariateGaussianEmission(MEANS2, COVS2)
        dens = [lambda y, k=k: math.exp(exact_log_mvn(y, MEANS2[k], COVS2[k])) for k in range(3)]

        def density(k, y):
            return dens[k](y)

        ys = [[rnd.gauss(0.5, 2.0), rnd.gauss(0.5, 2.0)] for _ in range(6)]
        hmm = HiddenMarkovFilter(PI, A, e)
        for t, (out, (probs, loglik)) in enumerate(zip_strict(fold(hmm, ys), hmm_filter(PI, A, density, ys))):
            with self.subTest(t=t):
                for a, b in zip_strict(out.probabilities, probs):
                    self.assertAlmostEqual(a, b, delta=1e-12)
                self.assertAlmostEqual(out.log_likelihood, loglik, delta=1e-12 * abs(loglik))
        inputs = [{"y": y} for y in ys]
        smooth = forward_backward(hmm, inputs)
        for row, ref in zip_strict(smooth.probabilities, hmm_smoother(PI, A, density, ys)):
            for a, b in zip_strict(row, ref):
                self.assertAlmostEqual(a, b, delta=1e-12)
        path, log_p = hmm_most_probable_path(PI, A, density, ys)
        best = viterbi(hmm, inputs)
        self.assertEqual(best.states, tuple(path))
        self.assertAlmostEqual(best.log_probability, log_p, delta=1e-12 * abs(log_p))

    def test_vector_contract(self):
        hmm = HiddenMarkovFilter((0.5, 0.5), ((0.9, 0.1), (0.1, 0.9)),
                                 MultivariateGaussianEmission(MEANS2[:2], COVS2[:2]))
        contract = hmm.inputs()["y"]
        self.assertEqual((contract.kind, contract.shape), ("real", (2,)))
        contract.validate([1.0, 2])
        for bad in ([1.0], [1.0, 2.0, 3.0], 1.0, [1.0, math.nan], [1.0, "2"]):
            with self.subTest(bad=bad), self.assertRaises(ContractViolation):
                contract.validate(bad)
        with self.assertRaises(ValueError):
            hmm.emission.log_density(0, [1.0])
        with self.assertRaises(ValueError):
            hmm.emission.statistics(0, [1.0, 2.0, 3.0])
        run = Run(hmm)
        run.start()
        self.assertIsNotNone(run.step({"y": [0.1, 0.2]}))
        with self.assertLogs("sequential_analysis", "WARNING"):
            self.assertIsNone(run.step({"y": [0.1]}))
        self.assertEqual(run.counters["invalid_skipped"], 1)
        self.assertEqual(hmm.identity()["config"]["emission"]["model"], "multivariate_gaussian")

    def test_covariances_are_checked(self):
        bad = {
            "asymmetric": [[1.0, 0.2], [0.3, 1.0]],
            "indefinite": [[1.0, 2.0], [2.0, 1.0]],
            "singular": [[1.0, 1.0], [1.0, 1.0]],
            "numerically singular": [[1.0, 1.0 - 1e-14], [1.0 - 1e-14, 1.0]],
            "negative variance": [[-1.0, 0.0], [0.0, 1.0]],
            "wrong size": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "not finite": [[math.inf, 0.0], [0.0, 1.0]],
        }
        for name, cov in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                MultivariateGaussianEmission([[0.0, 0.0]], [cov])
        for means, covs in (([0.0, 1.0], [[[1.0]], [[1.0]]]), ([[0.0], [1.0, 2.0]], [[[1.0]], [[1.0]]]),
                            ([[0.0]], [[[1.0]], [[1.0]]]), ([], [])):
            with self.subTest(means=means), self.assertRaises(ValueError):
                MultivariateGaussianEmission(means, covs)
        # close to singular but resolvable: a correlation of 1 - 1e-9, also at very different scales
        rho = 1.0 - 1e-9
        e = MultivariateGaussianEmission([[0.0, 0.0]], [[[1e-6, rho * 1e-3 * 1e3], [rho * 1e-3 * 1e3, 1e6]]])
        on, off = e.log_density(0, [1e-3, 1e3]), e.log_density(0, [1e-3, -1e3])
        self.assertTrue(math.isfinite(on) and math.isfinite(off))
        self.assertGreater(on - off, 1e8)  # along the ridge versus across it
        self.assertAlmostEqual(on, exact_log_mvn([1e-3, 1e3], [0.0, 0.0], e.covariances[0]), delta=1e-6)

    def test_m_step_against_weighted_estimates(self):
        rnd = random.Random(3)
        e = MultivariateGaussianEmission(MEANS2[:2], COVS2[:2])
        ys = [[rnd.gauss(1, 2), rnd.gauss(-1, 1)] for _ in range(50)]
        weights = []
        for _ in ys:
            p = rnd.random()
            weights.append((p, 1.0 - p))
        new = e.from_statistics(totals_for(e, ys, weights), min_scale=1e-3)
        for k in range(2):
            w = [row[k] for row in weights]
            total = math.fsum(w)
            mean = [math.fsum(wt * y[i] for wt, y in zip_strict(w, ys)) / total for i in range(2)]
            for i in range(2):
                self.assertAlmostEqual(new.means[k][i], mean[i], delta=1e-13)
                for j in range(2):
                    c = math.fsum(wt * (y[i] - mean[i]) * (y[j] - mean[j]) for wt, y in zip_strict(w, ys)) / total
                    self.assertAlmostEqual(new.covariances[k][i][j], c + (1e-6 if i == j else 0.0), delta=1e-12)
            self.assertEqual(new.covariances[k][0][1], new.covariances[k][1][0])

    def test_m_step_keeps_empty_and_degenerate_states(self):
        e = MultivariateGaussianEmission(MEANS2[:2], COVS2[:2])
        y = [1.0, 2.0]
        g = e.statistics(0, y)
        self.assertEqual(g, (1.0, 2.0, 1.0, 2.0, 4.0))
        new = e.from_statistics([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, *g)], min_scale=0.0)
        self.assertEqual(new.spec()["means"][0], list(MEANS2[0]))  # no weight: kept
        self.assertEqual(new.spec()["covariances"][1], [list(r) for r in COVS2[1]])  # one point, no ridge: kept
        new = e.from_statistics([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, *g)], min_scale=1e-3)
        self.assertEqual(new.means[1], (1.0, 2.0))
        self.assertAlmostEqual(new.covariances[1][0][0], 1e-6, delta=1e-18)
        with self.assertRaises(ValueError):
            e.from_statistics([(1.0, 0.0), (1.0, 0.0)])

    def test_em_never_decreases_the_likelihood(self):
        rnd = random.Random(4)
        ys, z = [], 0
        for _ in range(150):
            z = z if rnd.random() < 0.9 else 1 - z
            ys.append([rnd.gauss(3.0 * z, 1.0), rnd.gauss(-z, 0.5 + z)])
        start = MultivariateGaussianEmission([[0.5, 0.0], [1.0, -0.5]], [[[4.0, 0.0], [0.0, 4.0]]] * 2)
        hmm = HiddenMarkovFilter((0.5, 0.5), ((0.9, 0.1), (0.1, 0.9)), start)
        self.check_monotone(hmm, ys)

    def check_monotone(self, hmm, ys, iterations=30):
        inputs = [{"y": y} for y in ys]
        previous = -math.inf
        history = []
        for _ in range(iterations):
            smooth = forward_backward(hmm, inputs)
            self.assertGreaterEqual(smooth.log_likelihood, previous - 1e-10 * abs(smooth.log_likelihood))
            previous = smooth.log_likelihood
            history.append(previous)
            emission = hmm.emission.from_statistics(totals_for(hmm.emission, ys, smooth.probabilities))
            hmm = HiddenMarkovFilter(hmm.initial, hmm.transition, emission)
        self.assertGreater(history[-1], history[0] + 1.0)
        return hmm


class StudentT(unittest.TestCase):
    MEANS, SCALES, DF = (0.0, 2.0, -3.0), (1.0, 0.5, 2.0), (1.5, 4.0, 30.0)

    def test_log_density_against_the_gamma_function(self):
        e = StudentTEmission(self.MEANS, self.SCALES, self.DF)
        for k in range(3):
            for y in (-40.0, -3.0, -0.1, 0.0, 0.7, 2.0, 15.0):
                ref = math.log(t_density(y, self.MEANS[k], self.SCALES[k], self.DF[k]))
                self.assertAlmostEqual(e.log_density(k, y), ref, delta=1e-13 * max(1.0, abs(ref)))

    def test_filter_smoother_and_viterbi_against_enumeration(self):
        rnd = random.Random(5)
        e = StudentTEmission(self.MEANS, self.SCALES, self.DF)

        def density(k, y):
            return t_density(y, self.MEANS[k], self.SCALES[k], self.DF[k])

        ys = [t_sample(rnd, 0.0, 2.0, 2.0) for _ in range(6)] + [1e6]
        hmm = HiddenMarkovFilter(PI, A, e)
        for t, (out, (probs, loglik)) in enumerate(zip_strict(fold(hmm, ys), hmm_filter(PI, A, density, ys))):
            with self.subTest(t=t):
                for a, b in zip_strict(out.probabilities, probs):
                    self.assertAlmostEqual(a, b, delta=1e-12)
                self.assertAlmostEqual(out.log_likelihood, loglik, delta=1e-12 * abs(loglik))
        inputs = [{"y": y} for y in ys]
        for row, ref in zip_strict(forward_backward(hmm, inputs).probabilities, hmm_smoother(PI, A, density, ys)):
            for a, b in zip_strict(row, ref):
                self.assertAlmostEqual(a, b, delta=1e-12)
        path, log_p = hmm_most_probable_path(PI, A, density, ys)
        best = viterbi(hmm, inputs)
        self.assertEqual(best.states, tuple(path))
        self.assertAlmostEqual(best.log_probability, log_p, delta=1e-12 * abs(log_p))

    def test_gaussian_limit(self):
        g = GaussianEmission((1.0,), (2.0,))
        for df in (1e8, 1e12, 1e15, 1e300):
            e = StudentTEmission((1.0,), (2.0,), df)
            for y in (-9.0, 0.0, 1.0, 4.5, 11.0):
                z2 = ((y - 1.0) / 2.0) ** 2
                # log t - log N = (z^4 - 2 z^2 - 1) / (4 df) + O(df^-2)
                expected = (z2 * z2 - 2.0 * z2 - 1.0) / (4.0 * df)
                with self.subTest(df=df, y=y):
                    difference = e.log_density(0, y) - g.log_density(0, y)
                    self.assertAlmostEqual(difference, expected, delta=1e-13 + 1e-4 * abs(expected))
                    self.assertAlmostEqual(e.weight(0, y), 1.0, delta=2.0 * (z2 + 1.0) / df)

    def test_far_outliers_have_finite_log_density_and_no_weight(self):
        e = StudentTEmission((0.0,), (1.0,), 3.0)
        for y in (1e100, -1e200, 1e300, 1.7e308):
            self.assertTrue(math.isfinite(e.log_density(0, y)), y)
            self.assertEqual(e.statistics(0, y)[0] * y * y < 1e-100 * y * y, True)
        self.assertEqual(e.statistics(0, 1e300), (0.0, 0.0, 0.0))
        tiny = StudentTEmission((-1e308,), (1e-300,), 3.0)
        self.assertTrue(math.isfinite(tiny.log_density(0, 1e308)))  # y - mean and z overflow
        # the two branches (r below and above 1e150) meet
        root = math.sqrt(3.0)
        below, above = e.log_density(0, 1e150 * root * (1 - 1e-12)), e.log_density(0, 1e150 * root * (1 + 1e-12))
        self.assertAlmostEqual(below, above, delta=1e-14 * abs(below) + 4 * 2e-12)
        # an outlier does not move the filter into a state it does not belong to
        hmm = HiddenMarkovFilter((0.5, 0.5), ((0.9, 0.1), (0.1, 0.9)), StudentTEmission((0.0, 5.0), (1.0, 1.0), 2.0))
        state = hmm.initial_state()
        for y in (0.1, -0.3, 0.2, 1e250):
            state, out = hmm.step(state, {"y": y})
        self.assertTrue(math.isfinite(out.log_likelihood))
        # equally implausible in both states: the filter keeps what the earlier data said (a Gaussian would jump)
        self.assertGreater(out.probabilities[0], 0.85)
        gaussian = HiddenMarkovFilter((0.5, 0.5), ((0.9, 0.1), (0.1, 0.9)), GaussianEmission((0.0, 5.0), (1.0, 1.0)))
        jumped = fold(gaussian, [0.1, -0.3, 0.2, 20.0])[-1]
        self.assertGreater(jumped.probabilities[1], 0.999)

    def test_arguments(self):
        e = StudentTEmission((0.0, 1.0), 2.0, 5.0)
        self.assertEqual(e.spec(), {"model": "student_t", "means": [0.0, 1.0], "scales": [2.0, 2.0],
                                    "df": [5.0, 5.0]})
        self.assertEqual(e.contract("y").shape, ())
        for args in (((0.0,), (0.0,), (1.0,)), ((0.0,), (1.0,), (0.0,)), ((0.0,), (1.0,), (math.inf,)),
                     ((0.0, 1.0), (1.0,), (2.0, 2.0)), ((math.nan,), (1.0,), (1.0,)), ((), (), ())):
            with self.subTest(args=args), self.assertRaises(ValueError):
                StudentTEmission(*args)

    def test_m_step_fixed_point_is_the_maximum_likelihood_estimate(self):
        rnd = random.Random(6)
        df = 4.0
        ys = [t_sample(rnd, 1.0, 2.0, df) for _ in range(120)] + [60.0, -45.0]

        def loglik(mean, scale):
            return math.fsum(math.log(t_density(y, mean, scale, df)) for y in ys)

        def profile_scale(mean):
            return math.exp(golden_max(lambda s: loglik(mean, math.exp(s)), math.log(0.05), math.log(50.0)))

        median = sorted(ys)[len(ys) // 2]
        mean_ref = golden_max(lambda m: loglik(m, profile_scale(m)), median - 2.0, median + 2.0)
        scale_ref = profile_scale(mean_ref)

        e = StudentTEmission((median,), (1.0,), df)
        ones = [(1.0,)] * len(ys)
        history = []
        for _ in range(2000):
            history.append(loglik(e.means[0], e.scales[0]))
            new = e.from_statistics(totals_for(e, ys, ones))
            done = abs(new.means[0] - e.means[0]) + abs(new.scales[0] - e.scales[0]) < 1e-15
            e = new
            if done:
                break
        for a, b in zip(history, history[1:]):
            self.assertGreaterEqual(b, a - 1e-9)
        self.assertAlmostEqual(e.means[0], mean_ref, delta=1e-6)
        self.assertAlmostEqual(e.scales[0], scale_ref, delta=1e-6)
        # and it is a maximum: every nearby point is lower
        best = loglik(e.means[0], e.scales[0])
        for dm, ds in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
            self.assertLess(loglik(e.means[0] + dm, e.scales[0] + ds), best)

    def test_em_never_decreases_the_likelihood(self):
        rnd = random.Random(7)
        ys, z = [], 0
        for _ in range(200):
            z = z if rnd.random() < 0.92 else 1 - z
            ys.append(t_sample(rnd, 4.0 * z, 1.0, 3.0))
        hmm = HiddenMarkovFilter((0.5, 0.5), ((0.9, 0.1), (0.1, 0.9)), StudentTEmission((0.5, 1.5), (3.0, 3.0), 3.0))
        fitted = MultivariateGaussian.check_monotone(self, hmm, ys, iterations=40)
        self.assertAlmostEqual(fitted.emission.means[0], 0.0, delta=0.4)
        self.assertAlmostEqual(fitted.emission.means[1], 4.0, delta=0.4)

    def test_m_step_keeps_empty_states(self):
        e = StudentTEmission((0.0, 1.0), (1.0, 2.0), (3.0, 5.0))
        new = e.from_statistics([(0.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0)])
        self.assertEqual(new.spec(), e.spec())
        new = e.from_statistics([(2.0, 2.0, 2.0, 2.0), (1.0, 1.0, 3.0, 9.0)], min_scale=0.5)
        self.assertEqual((new.means, new.scales, new.df), ((1.0, 3.0), (0.5, 0.5), (3.0, 5.0)))


class FromSpec(unittest.TestCase):
    EMISSIONS = (
        CategoricalEmission(((0.7, 0.2, 0.1), (0.1, 0.8, 0.1))),
        GaussianEmission((0.0, 2.0), (1.0, 0.5)),
        PoissonEmission((1.0, 4.0)),
        MultivariateGaussianEmission(MEANS2[:2], COVS2[:2]),
        StudentTEmission((0.0, 2.0), (1.0, 0.5), (3.0, 7.5)),
    )
    POINTS = (1, 0.7, 3, [0.3, -0.4], 0.7)

    def test_round_trip(self):
        for e, y in zip_strict(self.EMISSIONS, self.POINTS):
            with self.subTest(e.spec()["model"]):
                again = emission_from_spec(e.spec())
                self.assertIs(type(again), type(e))
                self.assertEqual(again.spec(), e.spec())
                for k in range(2):
                    self.assertEqual(again.log_density(k, y), e.log_density(k, y))

    def test_permuting_the_states(self):
        for e, y in zip_strict(self.EMISSIONS, self.POINTS):
            spec = e.spec()
            swapped = emission_from_spec({key: value[::-1] if isinstance(value, list) else value
                                          for key, value in spec.items()})
            for k in range(2):
                self.assertEqual(swapped.log_density(1 - k, y), e.log_density(k, y))

    def test_refusals(self):
        for spec in ({"model": "laplace", "means": [0.0]}, {"means": [0.0]}, {"model": "gaussian", "means": [0.0]},
                     {"model": "student_t", "means": [0.0], "scales": [1.0]}, [("model", "poisson")]):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                emission_from_spec(spec)


if __name__ == "__main__":
    unittest.main()
