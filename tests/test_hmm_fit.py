# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Learning the parameters of a hidden Markov model: Baum-Welch (fit_hmm) and online EM (OnlineHMM).

* fit_hmm: one EM step agrees with hmmlearn (Gaussian, Poisson and
  categorical emissions) when hmmlearn is installed; the log-likelihood
  never decreases; the fitted model is a fixed point of EM; on long
  simulated series the estimates approach the true parameters; restarts
  return the best run, reproducibly.
* OnlineHMM: before its burn-in it filters exactly as HiddenMarkovFilter;
  with step sizes 1/t and the parameters held fixed its statistics are
  exactly the batch E-step's averages (Cappé's identity); on long series
  its estimates approach the batch maximum-likelihood estimates.
"""
import math
import random
import unittest

from sequential_analysis import Run
from sequential_analysis.errors import ContractViolation, NumericalError
from sequential_analysis.hmm_fit import e_step, fit_hmm
from sequential_analysis.procedures.hmm import (
    CategoricalEmission,
    GaussianEmission,
    HiddenMarkovFilter,
    PoissonEmission,
)
from sequential_analysis.procedures.online_hmm import OnlineHMM

try:
    import numpy as np
    from hmmlearn import hmm as hmmlearn
except ImportError:  # optional: the comparison is skipped
    hmmlearn = None


def simulate(transition, draw, n, seed, initial=0):
    rnd = random.Random(seed)
    z, ys = initial, []
    for _ in range(n):
        u, acc = rnd.random(), 0.0
        for k, p in enumerate(transition[z]):
            acc += p
            if u < acc:
                z = k
                break
        ys.append(draw(z, rnd))
    return ys


A2 = [[0.9, 0.1], [0.2, 0.8]]


def gaussian_series(n, seed):
    means, sds = (0.0, 3.0), (1.0, 0.5)
    return simulate(A2, lambda z, r: r.gauss(means[z], sds[z]), n, seed)


def poisson_series(n, seed):
    rates = (1.0, 6.0)

    def draw(z, r):
        k, p, threshold = 0, 1.0, math.exp(-rates[z])
        while True:
            p *= r.random()
            if p <= threshold:
                return k
            k += 1

    return simulate(A2, draw, n, seed)


def categorical_series(n, seed):
    probs = ((0.7, 0.2, 0.1), (0.1, 0.3, 0.6))

    def draw(z, r):
        u, acc = r.random(), 0.0
        for s, p in enumerate(probs[z]):
            acc += p
            if u < acc:
                return s
        return len(probs[z]) - 1

    return simulate(A2, draw, n, seed)


def records(ys):
    return [{"y": y} for y in ys]


START_A = [[0.7, 0.3], [0.4, 0.6]]


class FitHMM(unittest.TestCase):
    @unittest.skipIf(hmmlearn is None, "hmmlearn is not installed")
    def test_one_step_agrees_with_hmmlearn(self):
        ys = gaussian_series(400, 1)
        start = HiddenMarkovFilter([0.6, 0.4], START_A, GaussianEmission([-0.5, 2.0], [1.5, 1.2]))
        ours = fit_hmm(records(ys), start, iterations=1).model
        ref = hmmlearn.GaussianHMM(2, covariance_type="diag", init_params="", params="stmc", n_iter=1,
                                   tol=-math.inf, min_covar=1e-300, means_prior=0, means_weight=0, covars_prior=0,
                                   covars_weight=1, implementation="scaling")
        ref.startprob_, ref.transmat_ = np.array([0.6, 0.4]), np.array(START_A)
        ref.means_, ref.covars_ = np.array([[-0.5], [2.0]]), np.array([[1.5**2], [1.2**2]])
        ref.fit(np.array(ys).reshape(-1, 1))
        self.compare(ours, ref.startprob_, ref.transmat_, ours.emission.means, ref.means_.ravel())
        self.compare_values(ours.emission.sds, np.sqrt(ref.covars_.ravel()))

        ys = poisson_series(400, 2)
        start = HiddenMarkovFilter([0.5, 0.5], START_A, PoissonEmission([0.5, 3.0]))
        ours = fit_hmm(records(ys), start, iterations=1).model
        ref = hmmlearn.PoissonHMM(2, init_params="", params="stl", n_iter=1, tol=-math.inf, implementation="scaling")
        ref.startprob_, ref.transmat_, ref.lambdas_ = np.array([0.5, 0.5]), np.array(START_A), np.array([[0.5], [3.0]])
        ref.fit(np.array(ys).reshape(-1, 1))
        self.compare(ours, ref.startprob_, ref.transmat_, ours.emission.rates, ref.lambdas_.ravel())

        ys = categorical_series(400, 3)
        start = HiddenMarkovFilter([0.5, 0.5], START_A, CategoricalEmission([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]]))
        ours = fit_hmm(records(ys), start, iterations=1).model
        ref = hmmlearn.CategoricalHMM(2, init_params="", params="ste", n_iter=1, tol=-math.inf, n_features=3,
                                      implementation="scaling")
        ref.startprob_, ref.transmat_ = np.array([0.5, 0.5]), np.array(START_A)
        ref.emissionprob_ = np.array([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]])
        ref.fit(np.array(ys).reshape(-1, 1))
        # hmmlearn's categorical M-step differs from the exact ratio in the ninth digit (its transitions,
        # from the same expectations, agree to 1e-16); ours is the plain ratio of expected counts
        self.compare(ours, ref.startprob_, ref.transmat_, [v for row in ours.emission.p for v in row],
                     ref.emissionprob_.ravel(), emission_delta=1e-8)

    def compare(self, ours, startprob, transmat, mine, theirs, emission_delta=1e-10):
        self.compare_values(ours.initial, startprob)
        self.compare_values([v for row in ours.transition for v in row], np.asarray(transmat).ravel())
        self.compare_values(mine, theirs, emission_delta)

    def compare_values(self, a, b, delta=1e-10):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, float(y), delta=delta)

    def test_likelihood_never_decreases_and_the_fit_is_a_fixed_point(self):
        for ys, start in (
            (gaussian_series(800, 4),
             HiddenMarkovFilter([0.5, 0.5], START_A, GaussianEmission([1.0, 2.0], [2.0, 2.0]))),
            (poisson_series(800, 5), HiddenMarkovFilter([0.5, 0.5], START_A, PoissonEmission([2.0, 4.0]))),
            (categorical_series(800, 6),
             HiddenMarkovFilter([0.5, 0.5], START_A, CategoricalEmission([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]))),
        ):
            fit = fit_hmm(records(ys), start, tolerance=1e-12, iterations=2000)
            self.assertTrue(fit.converged)
            h = fit.history
            self.assertTrue(all(b >= a - 1e-9 * abs(a) for a, b in zip(h, h[1:])), h[:5])
            # the reported likelihood is the filter's
            state, out = fit.model.initial_state(), None
            for y in ys:
                state, out = fit.model.step(state, {"y": y})
            self.assertAlmostEqual(out.log_likelihood, fit.log_likelihood, delta=1e-8 * abs(fit.log_likelihood))
            # one more EM step changes nothing (to the tolerance)
            again = fit_hmm(records(ys), fit.model, iterations=1)
            self.assertLess(again.log_likelihood - fit.log_likelihood, 1e-6)

    def test_estimates_approach_the_truth(self):
        ys = gaussian_series(6000, 7)
        start = HiddenMarkovFilter([0.5, 0.5], START_A, GaussianEmission([-1.0, 1.0], [2.0, 2.0]))
        fit = fit_hmm(records(ys), start)
        a, e = fit.model.transition, fit.model.emission
        self.assertAlmostEqual(a[0][1], 0.1, delta=0.02)
        self.assertAlmostEqual(a[1][0], 0.2, delta=0.04)
        for got, want, tol in zip((*e.means, *e.sds), (0.0, 3.0, 1.0, 0.5), (0.06, 0.06, 0.05, 0.04)):
            self.assertAlmostEqual(got, want, delta=tol)

    def test_restarts_find_the_better_maximum(self):
        ys = gaussian_series(600, 8)
        # a poor start: both states on the same side
        start = HiddenMarkovFilter([0.5, 0.5], START_A, GaussianEmission([5.0, 5.1], [0.3, 0.3]))
        alone = fit_hmm(records(ys), start)
        best = fit_hmm(records(ys), start, restarts=4, seed=3)
        self.assertEqual(len(best.restarts), 5)
        self.assertEqual(best.log_likelihood, max(best.restarts))
        self.assertGreaterEqual(best.log_likelihood, alone.log_likelihood - 1e-9)
        self.assertLess(best.model.emission.means[0], best.model.emission.means[1])  # labelled like `start`
        again = fit_hmm(records(ys), start, restarts=4, seed=3)  # reproducible
        self.assertEqual((again.model.identity(), again.history, again.restarts),
                         (best.model.identity(), best.history, best.restarts))

    def test_refusals(self):
        start = HiddenMarkovFilter([1.0], [[1.0]], PoissonEmission([1.0]))
        with self.assertRaises(ValueError):
            fit_hmm(records([1]), start)
        with self.assertRaises(ContractViolation):
            fit_hmm(records([1, -1]), start)  # outside the contract
        with self.assertRaises(NumericalError):  # impossible observation
            fit_hmm(records([0, 3]), HiddenMarkovFilter([1.0], [[1.0]], CategoricalEmission([[1.0, 0.0, 0.0, 0.0]])))


class OnlineEM(unittest.TestCase):
    def test_before_burn_in_it_is_the_filter(self):
        start = HiddenMarkovFilter([0.5, 0.5], START_A, GaussianEmission([0.0, 2.0], [1.0, 1.0]))
        online = OnlineHMM(start, burn_in=10_000)
        ys = gaussian_series(300, 9)
        a, b = start.initial_state(), online.initial_state()
        for y in ys:
            a, oa = start.step(a, {"y": y})
            b, ob = online.step(b, {"y": y})
            self.assertEqual(oa.probabilities, ob.probabilities)
            self.assertEqual(oa.log_likelihood, ob.log_likelihood)

    def test_statistics_are_the_batch_expectations_with_steps_one_over_t(self):
        """Cappé (2011): with gamma_t = 1/t and fixed parameters, S_t = E[sum_s s_s | y_1..y_t] / t exactly."""
        for ys, start in (
            (gaussian_series(200, 10),
             HiddenMarkovFilter([0.3, 0.7], START_A, GaussianEmission([0.0, 2.0], [1.0, 1.5]))),
            (categorical_series(150, 11),
             HiddenMarkovFilter([0.5, 0.5], START_A, CategoricalEmission([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]))),
        ):
            online = OnlineHMM(start, kappa=1.0, burn_in=10_000)
            state = online.initial_state()
            for t, y in enumerate(ys, start=1):
                state, _ = online.step(state, {"y": y})
                if t in (1, 2, 17, len(ys)):
                    trans, emis = online.statistics(state)
                    batch = e_step(start, ys[:t])
                    for j in range(2):
                        for k in range(2):
                            self.assertAlmostEqual(trans[j][k], batch.transitions[j][k] / t, delta=1e-12)
                        for got, want in zip(emis[j], batch.emissions[j]):
                            self.assertAlmostEqual(got, want / t, delta=1e-11 * max(1.0, abs(want / t)))

    def test_estimates_approach_the_batch_maximum_likelihood(self):
        ys = gaussian_series(15_000, 12)
        start = HiddenMarkovFilter([0.5, 0.5], START_A, GaussianEmission([-0.5, 1.5], [1.5, 1.5]))
        batch = fit_hmm(records(ys), start).model
        online = OnlineHMM(start, burn_in=100)
        state = online.initial_state()
        for y in ys:
            state, out = online.step(state, {"y": y})
        current = online.model(state)
        self.assertAlmostEqual(current.transition[0][1], batch.transition[0][1], delta=0.03)
        self.assertAlmostEqual(current.transition[1][0], batch.transition[1][0], delta=0.05)
        for got, want in zip((*current.emission.means, *current.emission.sds),
                             (*batch.emission.means, *batch.emission.sds)):
            self.assertAlmostEqual(got, want, delta=0.1)
        self.assertEqual(out.transition, current.transition)

    def test_poisson_and_checkpoints(self):
        start = HiddenMarkovFilter([0.5, 0.5], START_A, PoissonEmission([0.5, 3.0]))
        ys = poisson_series(3000, 13)
        whole = Run(OnlineHMM(start, burn_in=50), run_id="hmm", clock=lambda: 0.0)
        for y in ys:
            whole.step({"y": y})
        rates = whole.state and OnlineHMM(start, burn_in=50).model(whole.state).emission.rates
        self.assertAlmostEqual(rates[0], 1.0, delta=0.15)
        self.assertAlmostEqual(rates[1], 6.0, delta=0.4)
        first = Run(OnlineHMM(start, burn_in=50), run_id="hmm", clock=lambda: 0.0)
        for y in ys[:1234]:
            first.step({"y": y})
        resumed = Run.restore(OnlineHMM(start, burn_in=50), first.checkpoint(), clock=lambda: 0.0)
        resumed.start()
        for y in ys[1234:]:
            resumed.step({"y": y})
        self.assertEqual(resumed.history_digest, whole.history_digest)
        self.assertEqual(OnlineHMM(start, burn_in=50).encode_state(resumed.state),
                         OnlineHMM(start, burn_in=50).encode_state(whole.state))

    def test_refusals(self):
        start = HiddenMarkovFilter([0.5, 0.5], START_A, PoissonEmission([0.5, 3.0]))
        for bad in (dict(kappa=0.5), dict(kappa=1.2), dict(burn_in=0), dict(min_scale=-1.0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                OnlineHMM(start, **bad)
        with self.assertRaises(TypeError):
            OnlineHMM(PoissonEmission([1.0]))
        proc = OnlineHMM(start)
        with self.assertRaises(ValueError):
            proc.decode_state({"n": 1, "probabilities": [0.5], "transitions": [], "emissions": [],
                               "log_likelihood": 0.0, "comp": 0.0, "averaged": []})


if __name__ == "__main__":
    unittest.main()
