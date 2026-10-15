# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Mathematical correctness of the filters, against independent references (tests/oracles.py).

* Kalman filter: the exact conditional distribution of the state and the
  likelihood, from the joint Gaussian distribution of all states and
  observations, in exact rational arithmetic.
* Hidden Markov filter: enumeration of every path of the hidden chain.
* Bayesian change points: enumeration of every partition into segments,
  with segment marginal likelihoods from their closed forms.
* Particle filter: the Kalman filter (linear Gaussian model), a filter by
  numerical integration on a grid (stochastic volatility), and the
  unbiasedness of its likelihood estimate.
"""
import math
import random
import unittest

from sequential_analysis.procedures import (
    BayesianChangePoint,
    BetaBernoulliModel,
    CategoricalEmission,
    GaussianEmission,
    HiddenMarkovFilter,
    KalmanFilter,
    LinearGaussianModel,
    LocalLevelKalman,
    NormalGammaModel,
    ParticleFilter,
    PoissonEmission,
    PoissonGammaModel,
    RobustKalmanFilter,
    StochasticVolatilityModel,
    StudentTLocalLevel,
)
from sequential_analysis.rng import SplitMix64

from .oracles import changepoint_filter, gaussian_filter, grid_filter, hmm_filter


def normal_density(x, mean, sd):
    return math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))


def fold(proc, inputs, rng=None):
    state, outs = proc.initial_state(), []
    for x in inputs:
        state, out = proc.step(state, x, rng)
        outs.append(out)
    return state, outs


class KalmanAgainstExactConditioning(unittest.TestCase):
    MODELS = {
        "tracking with control": dict(
            transition=((1.0, 1.0), (0.0, 1.0)), observation=((1.0, 0.0), (0.5, 1.0)),
            process_noise=((0.3, 0.1), (0.1, 0.2)), observation_noise=((1.0, 0.2), (0.2, 0.5)),
            initial_mean=(0.5, -1.0), initial_covariance=((2.0, 0.3), (0.3, 1.0)), control=((1.0,), (0.25,))),
        "scalar, unstable": dict(transition=1.3, observation=0.7, process_noise=0.4, observation_noise=2.0,
                                 initial_mean=(1.0,), initial_covariance=0.5),
        "three states, one observation": dict(
            transition=((0.5, 0.2, 0.0), (0.0, 0.9, 0.1), (0.1, 0.0, 0.8)), observation=((1.0, -1.0, 0.5),),
            process_noise=((0.2, 0.0, 0.05), (0.0, 0.1, 0.0), (0.05, 0.0, 0.3)), observation_noise=0.7,
            initial_mean=(0.0, 1.0, -1.0), initial_covariance=((1.0, 0.0, 0.0), (0.0, 2.0, 0.5), (0.0, 0.5, 1.0))),
    }

    def test_mean_covariance_and_likelihood_are_exact(self):
        rnd = random.Random(1)
        for label, spec in self.MODELS.items():
            kf = KalmanFilter(**spec)
            m = kf.n_obs
            ys = [[rnd.gauss(0, 2) for _ in range(m)] if m > 1 else rnd.gauss(0, 2) for _ in range(6)]
            us = [rnd.gauss(0, 1) for _ in range(6)]
            inputs = [{"y": y, "u": u} if spec.get("control") else {"y": y} for y, u in zip(ys, us, strict=True)]
            _, outs = fold(kf, inputs)
            ref = gaussian_filter(_m(spec["transition"]), _m(spec["observation"]), _m(spec["process_noise"]),
                                  _m(spec["observation_noise"]),
                                  spec["initial_mean"], _m(spec["initial_covariance"]), ys,
                                  B=spec.get("control"), us=us)
            for t, (out, (mean, cov, loglik)) in enumerate(zip(outs, ref, strict=True)):
                with self.subTest(label, t=t):
                    for a, b in zip(out.mean, mean, strict=True):
                        self.assertAlmostEqual(a, b, delta=1e-12 * max(1.0, abs(b)))
                    for row, ref_row in zip(out.covariance, cov, strict=True):
                        for a, b in zip(row, ref_row, strict=True):
                            self.assertAlmostEqual(a, b, delta=1e-12 * max(1.0, abs(b)))
                    self.assertAlmostEqual(out.log_likelihood, loglik, delta=1e-11 * abs(loglik))

    def test_local_level_special_case(self):
        rnd = random.Random(2)
        ys = [rnd.gauss(0, 1) for _ in range(50)]
        _, general = fold(KalmanFilter(1.0, 1.0, 0.3, 2.0, (0.5,), 4.0), [{"y": y} for y in ys])
        _, special = fold(LocalLevelKalman(q=0.3, r=2.0, m0=0.5, p0=4.0), [{"y": y} for y in ys])
        for g, s in zip(general, special, strict=True):
            self.assertAlmostEqual(g.mean[0], s.mean, delta=1e-12)
            self.assertAlmostEqual(g.covariance[0][0], s.var, delta=1e-12)

    def test_variance_reaches_the_riccati_fixed_point(self):
        q, r = 0.3, 2.0  # local level: P = (P + q) r / (P + q + r), so P = (-q + sqrt(q^2 + 4 q r)) / 2
        _, outs = fold(KalmanFilter(1.0, 1.0, q, r, (0.0,), 100.0), [{"y": 0.0}] * 200)
        self.assertAlmostEqual(outs[-1].covariance[0][0], (-q + math.sqrt(q * q + 4 * q * r)) / 2, delta=1e-12)

    def test_innovations_are_calibrated(self):
        # under the model, v' S^-1 v is chi-square with m degrees of freedom: mean m, independent over time
        spec = self.MODELS["tracking with control"]
        kf = KalmanFilter(**spec)
        rnd = random.Random(3)
        z = list(spec["initial_mean"])
        state, total, count = kf.initial_state(), 0.0, 0
        chol_q, chol_r = _chol(spec["process_noise"]), _chol(spec["observation_noise"])
        start = _mv(_chol(spec["initial_covariance"]), [rnd.gauss(0, 1) for _ in z])
        z = [a + b for a, b in zip(z, start, strict=True)]
        for _ in range(3000):
            u = rnd.gauss(0, 1)
            z = [a + b + c for a, b, c in zip(_mv(spec["transition"], z), [r[0] * u for r in spec["control"]],
                                              _mv(chol_q, [rnd.gauss(0, 1), rnd.gauss(0, 1)]), strict=True)]
            y = [a + b for a, b in zip(_mv(spec["observation"], z), _mv(chol_r, [rnd.gauss(0, 1), rnd.gauss(0, 1)]),
                                        strict=True)]
            state, out = kf.step(state, {"y": y, "u": u}, None)
            s, v = out.innovation_covariance, out.innovation
            det = s[0][0] * s[1][1] - s[0][1] * s[1][0]
            total += (s[1][1] * v[0] ** 2 - 2 * s[0][1] * v[0] * v[1] + s[0][0] * v[1] ** 2) / det
            count += 1
        self.assertAlmostEqual(total / count, 2.0, delta=4 * 2.0 / math.sqrt(count))  # sd of chi2_2 is 2

    def test_constructors(self):
        trend = KalmanFilter.local_linear_trend(0.01, 0.001, 1.0)
        _, outs = fold(trend, [{"y": 2.0 * t + 1.0} for t in range(60)])
        self.assertAlmostEqual(outs[-1].mean[1], 2.0, delta=1e-3)  # the slope of a noiseless line
        ar = KalmanFilter.autoregressive((0.5, -0.3), 1.0, 0.1)
        self.assertEqual(ar.F, ((0.5, -0.3), (1.0, 0.0)))

    def test_arguments(self):
        with self.assertRaises(ValueError):
            KalmanFilter(1.0, 1.0, 1.0, 0.0, (0.0,), 1.0)  # singular observation noise
        with self.assertRaises(ValueError):
            KalmanFilter(((1.0, 0.0),), 1.0, 1.0, 1.0, (0.0,), 1.0)  # transition not square
        with self.assertRaises(ValueError):
            KalmanFilter(1.0, 1.0, ((1.0, 0.5), (0.0, 1.0)), 1.0, (0.0,), 1.0)  # wrong size


class RobustKalman(unittest.TestCase):
    def make(self, df):
        return RobustKalmanFilter(1.0, 1.0, 0.1, 1.0, (0.0,), 10.0, df=df)

    def test_becomes_the_kalman_filter_as_df_grows(self):
        rnd = random.Random(4)
        ys = [{"y": rnd.gauss(0, 1)} for _ in range(30)]
        _, robust = fold(self.make(1e12), ys)
        _, plain = fold(KalmanFilter(1.0, 1.0, 0.1, 1.0, (0.0,), 10.0), ys)
        for a, b in zip(robust, plain, strict=True):
            self.assertAlmostEqual(a.mean[0], b.mean[0], delta=1e-9)

    def test_weight_solves_the_variational_fixed_point(self):
        f = self.make(4.0)
        state, _ = fold(f, [{"y": 0.1}] * 10)
        _, out = f.step(state, {"y": 7.0}, None)
        # lambda = (df + 1) / (df + (y - m)^2 / r + p / r) at the returned posterior (H = 1, R = 1)
        expected = (4.0 + 1.0) / (4.0 + (7.0 - out.mean[0]) ** 2 + out.covariance[0][0])
        self.assertAlmostEqual(out.weight, expected, delta=1e-10)

    def test_outliers_are_discounted_and_their_influence_redescends(self):
        f = self.make(3.0)
        state, _ = fold(f, [{"y": 0.0}] * 20)
        moves = [abs(f.step(state, {"y": y}, None)[1].mean[0]) for y in (2.0, 10.0, 100.0, 1e4)]
        self.assertLess(moves[1], moves[0] * 2)  # the plain filter would move 5 times further
        self.assertLess(moves[3], moves[2])  # a larger outlier moves the state less
        self.assertLess(moves[3], 1e-3)

    def rmse(self, noise, seed):
        rnd = random.Random(seed)
        robust, plain = self.make(3.0), KalmanFilter(1.0, 1.0, 0.1, 1.0, (0.0,), 10.0)
        rs, ps, z = robust.initial_state(), plain.initial_state(), 0.0
        err_r = err_p = 0.0
        for _ in range(1500):
            z += rnd.gauss(0, math.sqrt(0.1))
            y = z + noise(rnd)
            rs, ro = robust.step(rs, {"y": y}, None)
            ps, po = plain.step(ps, {"y": y}, None)
            err_r += (ro.mean[0] - z) ** 2
            err_p += (po.mean[0] - z) ** 2
        return math.sqrt(err_r / 1500), math.sqrt(err_p / 1500)

    def test_smaller_error_under_heavy_tails_little_loss_under_normal_noise(self):
        def t3(rnd):  # Student t with 3 degrees of freedom, unit scale
            return rnd.gauss(0, 1) / math.sqrt(sum(rnd.gauss(0, 1) ** 2 for _ in range(3)) / 3)

        robust, plain = self.rmse(t3, 5)
        self.assertLess(robust, 0.9 * plain)
        robust, plain = self.rmse(lambda r: r.gauss(0, 1), 6)
        self.assertLess(robust, 1.1 * plain)


class HiddenMarkov(unittest.TestCase):
    A = ((0.9, 0.07, 0.03), (0.1, 0.8, 0.1), (0.2, 0.3, 0.5))
    PI = (0.5, 0.3, 0.2)

    def check(self, emission, density, ys):
        _, outs = fold(HiddenMarkovFilter(self.PI, self.A, emission), [{"y": y} for y in ys])
        ref = hmm_filter(self.PI, self.A, density, ys)
        for t, (out, (probs, loglik)) in enumerate(zip(outs, ref, strict=True)):
            with self.subTest(type(emission).__name__, t=t):
                for a, b in zip(out.probabilities, probs, strict=True):
                    self.assertAlmostEqual(a, b, delta=1e-12)
                self.assertAlmostEqual(out.log_likelihood, loglik, delta=1e-12 * abs(loglik))
                self.assertEqual(out.most_likely, max(range(3), key=lambda k: (probs[k], -k)))

    def test_against_enumeration_of_paths(self):
        rnd = random.Random(7)
        means, sds = (0.0, 2.0, 5.0), (1.0, 0.5, 2.0)
        self.check(GaussianEmission(means, sds), lambda k, y: normal_density(y, means[k], sds[k]),
                   [rnd.gauss(2, 3) for _ in range(7)])
        rates = (1.0, 4.0, 10.0)
        self.check(PoissonEmission(rates), lambda k, y: rates[k] ** y * math.exp(-rates[k]) / math.factorial(y),
                   [rnd.randrange(12) for _ in range(7)])
        table = ((0.7, 0.2, 0.1), (0.1, 0.8, 0.1), (0.3, 0.3, 0.4))
        self.check(CategoricalEmission(table), lambda k, y: table[k][y], [rnd.randrange(3) for _ in range(7)])

    def test_far_tails_do_not_underflow(self):
        f = HiddenMarkovFilter((0.5, 0.5), ((0.9, 0.1), (0.1, 0.9)), GaussianEmission((0.0, 1.0), (1.0, 1.0)))
        _, out = f.step(f.initial_state(), {"y": 1e3}, None)  # densities near exp(-5e5)
        self.assertAlmostEqual(out.probabilities[1], 1.0, delta=1e-12)
        self.assertTrue(math.isfinite(out.log_likelihood))

    def test_recovers_simulated_regimes(self):
        rnd = random.Random(8)
        f = HiddenMarkovFilter((0.5, 0.5), ((0.95, 0.05), (0.05, 0.95)), GaussianEmission((0.0, 3.0), (1.0, 1.0)))
        state, hits, z = f.initial_state(), 0, 0
        for _ in range(2000):
            z = z if rnd.random() < 0.95 else 1 - z
            state, out = f.step(state, {"y": rnd.gauss(3.0 * z, 1.0)}, None)
            hits += out.most_likely == z
        self.assertGreater(hits / 2000, 0.93)

    def test_arguments(self):
        with self.assertRaises(ValueError):
            HiddenMarkovFilter((0.5, 0.6), ((1, 0), (0, 1)), PoissonEmission((1, 2)))
        with self.assertRaises(ValueError):
            HiddenMarkovFilter((0.5, 0.5), ((1, 0), (0.5, 0.4)), PoissonEmission((1, 2)))
        with self.assertRaises(ValueError):
            HiddenMarkovFilter((0.5, 0.5), ((1, 0), (0, 1)), PoissonEmission((1, 2, 3)))


def normal_gamma_marginal(mu0, k0, a0, b0):
    def log_marginal(seg):
        n = len(seg)
        xbar = sum(seg) / n
        ss = sum((v - xbar) ** 2 for v in seg)
        kn, an = k0 + n, a0 + n / 2
        bn = b0 + ss / 2 + k0 * n * (xbar - mu0) ** 2 / (2 * kn)
        return (math.lgamma(an) - math.lgamma(a0) + a0 * math.log(b0) - an * math.log(bn)
                + 0.5 * math.log(k0 / kn) - n / 2 * math.log(2 * math.pi))
    return log_marginal


def poisson_gamma_marginal(a, b):
    def log_marginal(seg):
        n, s = len(seg), sum(seg)
        return (a * math.log(b) - math.lgamma(a) + math.lgamma(a + s) - (a + s) * math.log(b + n)
                - sum(math.lgamma(v + 1) for v in seg))
    return log_marginal


def beta_bernoulli_marginal(a, b):
    def log_beta(x, y):
        return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)

    def log_marginal(seg):
        n, s = len(seg), sum(seg)
        return log_beta(a + s, b + n - s) - log_beta(a, b)
    return log_marginal


class BayesianChangePoints(unittest.TestCase):
    def test_against_enumeration_of_partitions(self):
        rnd = random.Random(5)
        cases = [
            (NormalGammaModel(0.5, 2.0, 3.0, 1.5), normal_gamma_marginal(0.5, 2.0, 3.0, 1.5),
             [rnd.gauss(0, 1) for _ in range(5)] + [rnd.gauss(4, 2) for _ in range(4)]),
            (PoissonGammaModel(2.0, 0.5), poisson_gamma_marginal(2.0, 0.5),
             [rnd.randrange(4) for _ in range(5)] + [rnd.randrange(8, 15) for _ in range(4)]),
            (BetaBernoulliModel(0.7, 1.3), beta_bernoulli_marginal(0.7, 1.3),
             [int(rnd.random() < 0.2) for _ in range(5)] + [int(rnd.random() < 0.9) for _ in range(5)]),
        ]
        for model, marginal, ys in cases:
            proc = BayesianChangePoint(model, hazard=0.15, prune=0.0, max_hypotheses=None)
            ref = changepoint_filter(0.15, marginal, ys)
            state = proc.initial_state()
            for t, y in enumerate(ys):
                state, out = proc.step(state, {"x": y}, None)
                dist = proc.run_length_distribution(state)
                probs, log_evidence = ref[t]
                with self.subTest(type(model).__name__, t=t):
                    for length in range(1, t + 2):
                        self.assertAlmostEqual(dist.get(length, 0.0), probs[length], delta=1e-12)
                    self.assertAlmostEqual(out.log_evidence, log_evidence, delta=1e-12 * abs(log_evidence))
                    if t:
                        self.assertAlmostEqual(out.change_probability, probs[1], delta=1e-12)

    def test_detects_a_change_of_variance_with_unknown_parameters(self):
        rnd = random.Random(9)
        proc = BayesianChangePoint(NormalGammaModel(), hazard=0.01, alarm_probability=0.9, alarm_window=20)
        ys = [rnd.gauss(0, 1) for _ in range(150)] + [rnd.gauss(0, 4) for _ in range(150)]
        state, alarm = proc.initial_state(), None
        for t, y in enumerate(ys):
            state, out = proc.step(state, {"x": y}, None)
            if out.decision and alarm is None:
                alarm = t
        self.assertIsNotNone(alarm)
        self.assertTrue(150 <= alarm < 175, alarm)

    def test_false_alarms_are_rare_without_a_change(self):
        alarms = 0
        for seed in range(30):
            rnd = random.Random(seed)
            proc = BayesianChangePoint(PoissonGammaModel(2.0, 0.5), hazard=0.002, alarm_probability=0.95,
                                       alarm_window=10)
            state = proc.initial_state()
            for _ in range(300):
                state, out = proc.step(state, {"x": sum(rnd.random() < 0.4 for _ in range(10))}, None)
                if out.decision:
                    alarms += 1
                    break
        self.assertLessEqual(alarms, 2)

    def test_pruning_changes_little(self):
        rnd = random.Random(10)
        ys = [{"x": rnd.gauss(0, 1)} for _ in range(100)] + [{"x": rnd.gauss(2, 1)} for _ in range(100)]
        _, exact = fold(BayesianChangePoint(NormalGammaModel(), 0.01, prune=0.0, max_hypotheses=None), ys)
        for prune, cap, tolerance, evidence_tolerance in ((1e-12, None, 1e-10, 1e-8), (1e-10, 100, 2e-5, 2e-3),
                                                          (1e-10, 50, 1e-3, 0.05)):
            _, pruned = fold(BayesianChangePoint(NormalGammaModel(), 0.01, prune=prune, max_hypotheses=cap), ys)
            for a, b in zip(exact, pruned, strict=True):
                with self.subTest(prune=prune, cap=cap, n=a.n):
                    self.assertAlmostEqual(a.change_probability, b.change_probability, delta=tolerance)
                    self.assertAlmostEqual(a.log_evidence, b.log_evidence, delta=evidence_tolerance)


class ParticleFilters(unittest.TestCase):
    def test_likelihood_estimate_is_unbiased_and_the_mean_converges(self):
        model = LinearGaussianModel(0.9, 0.5, 1.0, 0.0, 2.0)
        rnd = random.Random(3)
        ys = [{"y": rnd.gauss(0, 1.5)} for _ in range(20)]
        _, kalman = fold(KalmanFilter(0.9, 1.0, 0.5, 1.0, (0.0,), 2.0), ys)
        ratios, means = [], []
        for seed in range(150):
            _, outs = fold(ParticleFilter(model, 200), ys, SplitMix64.for_role(seed, "test"))
            ratios.append(math.exp(outs[-1].log_likelihood - kalman[-1].log_likelihood))
            means.append(outs[-1].mean)
        mean_ratio = sum(ratios) / len(ratios)
        se = math.sqrt(sum((r - mean_ratio) ** 2 for r in ratios) / (len(ratios) - 1) / len(ratios))
        self.assertLess(abs(mean_ratio - 1.0), 4 * se)  # E[p_hat] = p
        _, big = fold(ParticleFilter(model, 20000), ys, SplitMix64.for_role(1, "test"))
        self.assertAlmostEqual(big[-1].mean, kalman[-1].mean[0], delta=0.03)
        self.assertAlmostEqual(big[-1].variance, kalman[-1].covariance[0][0], delta=0.03)

    def test_stochastic_volatility_against_a_grid_filter(self):
        mu, phi, sigma = -1.0, 0.95, 0.3
        rnd = random.Random(4)
        h, ys = mu, []
        for _ in range(8):
            h = mu + phi * (h - mu) + sigma * rnd.gauss(0, 1)
            ys.append(math.exp(h / 2) * rnd.gauss(0, 1))
        sd0 = sigma / math.sqrt(1 - phi**2)
        ref = grid_filter(mu - 8 * sd0, mu + 8 * sd0, 500, lambda z: normal_density(z, mu, sd0),
                          lambda a, b: normal_density(b, mu + phi * (a - mu), sigma),
                          lambda z, y: normal_density(y, 0.0, math.exp(z / 2)), ys)
        _, outs = fold(ParticleFilter(StochasticVolatilityModel(mu, phi, sigma), 20000), [{"y": y} for y in ys],
                       SplitMix64.for_role(2, "test"))
        for out, (mean, loglik) in zip(outs, ref, strict=True):
            self.assertAlmostEqual(out.mean, mean, delta=0.02)
            self.assertAlmostEqual(out.log_likelihood, loglik, delta=0.05)

    def test_heavy_tailed_level_against_a_grid_filter(self):
        model = StudentTLocalLevel(q=0.05, scale=0.5, df=2.0, m0=0.0, p0=1.0)
        ys = [0.1, -0.2, 0.3, 8.0, 0.2, 0.0]  # an outlier at the fourth
        const = math.exp(math.lgamma(1.5) - math.lgamma(1.0)) / math.sqrt(2.0 * math.pi) / 0.5
        ref = grid_filter(-6.0, 6.0, 500, lambda z: normal_density(z, 0.0, 1.0),
                          lambda a, b: normal_density(b, a, math.sqrt(0.05)),
                          lambda z, y: const * (1 + ((y - z) / 0.5) ** 2 / 2.0) ** -1.5, ys)
        _, outs = fold(ParticleFilter(model, 20000), [{"y": y} for y in ys], SplitMix64.for_role(3, "test"))
        for out, (mean, loglik) in zip(outs, ref, strict=True):
            self.assertAlmostEqual(out.mean, mean, delta=0.02)
            self.assertAlmostEqual(out.log_likelihood, loglik, delta=0.05)
        self.assertLess(abs(outs[3].mean), 0.5)  # the outlier at 8 hardly moves the level

    def test_resampling_keeps_the_weights_normalised_and_the_run_reproducible(self):
        model = LinearGaussianModel(0.9, 0.5, 1.0)
        ys = [{"y": v} for v in (0.3, -1.0, 2.5, 0.1)]
        a, outs_a = fold(ParticleFilter(model, 300, resample_threshold=1.0), ys, SplitMix64.for_role(5, "t"))
        b, _ = fold(ParticleFilter(model, 300, resample_threshold=1.0), ys, SplitMix64.for_role(5, "t"))
        self.assertEqual(a, b)
        self.assertTrue(all(o.resampled for o in outs_a))
        self.assertAlmostEqual(math.fsum(math.exp(w) for w in a.log_weights), 1.0, delta=1e-12)


def _m(a):
    if isinstance(a, (int, float)):
        return ((float(a),),)
    return tuple(tuple(r) if isinstance(r, (tuple, list)) else (r,) for r in a)


def _chol(a):
    a = _m(a)
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            low[i][j] = math.sqrt(s) if i == j else s / low[j][j]
    return low


def _mv(a, v):
    return [sum(x * y for x, y in zip(row, v, strict=True)) for row in _m(a)]
