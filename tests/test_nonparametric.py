# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Methods that do not assume normality: validity on heavy-tailed, skewed and discrete data.

The Monte Carlo checks are small (they run in seconds) and their limits are
loose: they catch a method that is invalid, not one that is slightly
conservative.  The larger simulations behind the numbers in the
docstrings are in the commit history of each module.
"""
import math
import random
import unittest

from sequential_analysis import Run
from sequential_analysis.procedures import (
    AsymptoticConfidenceSequence,
    BettingConfidenceSequence,
    CatoniConfidenceSequence,
    KernelTwoSampleTest,
    QuantileConfidenceSequence,
    QuantileTest,
    SequentialSignTest,
)


def cauchy(rnd: random.Random) -> float:
    return math.tan(math.pi * (rnd.random() - 0.5))


def ever_misses(make, gen, n, reps, truth, seed=1) -> float:
    """Fraction of runs in which the interval excludes the truth at some time."""
    rnd, bad = random.Random(seed), 0
    for _ in range(reps):
        proc = make()
        state = proc.initial_state()
        for _ in range(n):
            state, out = proc.step(state, {"x": gen(rnd)}, None)
            if (out.lower is not None and out.lower > truth) or (out.upper is not None and out.upper < truth):
                bad += 1
                break
    return bad / reps


def rejects(make, gen, n, reps, seed=1, key="x") -> float:
    rnd, hits = random.Random(seed), 0
    for _ in range(reps):
        proc = make()
        state = proc.initial_state()
        for _ in range(n):
            state, out = proc.step(state, gen(rnd) if key is None else {key: gen(rnd)}, None)
            if out.decision is not None:
                hits += 1
                break
    return hits / reps


class Quantiles(unittest.TestCase):
    def test_median_sequence_covers_for_cauchy_data(self):
        self.assertLessEqual(ever_misses(QuantileConfidenceSequence, cauchy, 300, 150, 0.0), 0.08)

    def test_upper_quantile_for_skewed_data(self):
        q90 = math.exp(1.2815515655446004)  # lognormal(0, 1)
        make = lambda: QuantileConfidenceSequence(0.9)
        self.assertLessEqual(ever_misses(make, lambda r: math.exp(r.gauss(0, 1)), 300, 150, q90), 0.08)

    def test_discrete_data_with_an_atom_at_the_median(self):
        gen = lambda r: r.choices([-1.0, 0.0, 1.0], [0.4, 0.45, 0.15])[0]  # median 0, asymmetric
        self.assertLessEqual(rejects(lambda: QuantileTest(0.0), gen, 500, 150), 0.08)
        self.assertLessEqual(ever_misses(QuantileConfidenceSequence, gen, 300, 100, 0.0), 0.08)
        # the sign test asks whether P(X > 0) = P(X < 0), which is false here: it rejects, rightly
        self.assertGreater(rejects(lambda: SequentialSignTest(0.0), gen, 500, 40), 0.5)

    def test_power_and_shrinking_interval(self):
        self.assertGreater(rejects(lambda: QuantileTest(0.0), lambda r: cauchy(r) + 0.5, 600, 30), 0.9)
        proc = QuantileConfidenceSequence()
        state, rnd = proc.initial_state(), random.Random(2)
        widths = []
        for n in range(1, 1001):
            state, out = proc.step(state, {"x": cauchy(rnd)}, None)
            if n in (100, 1000):
                widths.append(out.upper - out.lower)
        self.assertLess(widths[1], widths[0] / 2)

    def test_the_sequence_is_the_set_the_test_keeps(self):
        rnd = random.Random(3)
        test, cs = QuantileTest(0.0, stop_on_reject=False), QuantileConfidenceSequence()
        state, sample = test.initial_state(), []
        for _ in range(400):
            v = cauchy(rnd) + 0.2
            sample.append(v)
            state, out = test.step(state, {"x": v}, None)
            lower, upper = cs.bounds(tuple(sorted(sample)))
            self.assertEqual(out.log_evidence >= math.log(1 / 0.05), not lower <= 0.0 <= upper)

    def test_arguments(self):
        for kwargs in ({"p": 0.0}, {"p": 1.0}, {"alpha": 1.0}, {"tuned_for": 0.5}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                QuantileConfidenceSequence(**kwargs)
        with self.assertRaises(ValueError):
            QuantileTest(math.inf)


class TwoSample(unittest.TestCase):
    @staticmethod
    def pairs(ga, gb):
        return lambda r: {"a": ga(r), "b": gb(r)}

    def test_level_under_heavy_tails_and_zero_inflation(self):
        revenue = lambda r: 0.0 if r.random() < 0.7 else math.exp(r.gauss(3, 1))
        for gen in (cauchy, revenue):
            with self.subTest(gen=gen):
                self.assertLessEqual(rejects(KernelTwoSampleTest, self.pairs(gen, gen), 200, 60, key=None), 0.08)

    def test_detects_a_difference_of_scale_with_equal_means(self):
        gen = self.pairs(lambda r: r.gauss(0, 1), lambda r: r.gauss(0, 2))
        self.assertGreater(rejects(KernelTwoSampleTest, gen, 300, 20, key=None), 0.9)

    def test_payoff_is_antisymmetric(self):
        proc = KernelTwoSampleTest(bandwidth=1.0, burn_in=0)
        state, rnd = proc.initial_state(), random.Random(4)
        for _ in range(30):
            state, _ = proc.step(state, {"a": rnd.gauss(0, 1), "b": rnd.gauss(0.5, 1)}, None)
        _, one = proc.step(state, {"a": 0.3, "b": -1.2}, None)
        _, swapped = proc.step(state, {"a": -1.2, "b": 0.3}, None)
        self.assertEqual(one.payoff, -swapped.payoff)

    def test_runs_on_paired_streams(self):
        from sequential_analysis import PositionalPair, run_sync
        from sequential_analysis.sources import gaussian, interleave

        run = Run(KernelTwoSampleTest(), topology=PositionalPair(("a", "b")))
        run_sync(run, interleave(gaussian("a", 0.0, 1.0, seed=1, n=400), gaussian("b", 0.0, 3.0, seed=2, n=400)))
        self.assertIsNotNone(run.state.decision)


class Means(unittest.TestCase):
    def test_catoni_covers_heavy_tailed_data(self):
        a = 2.5  # Pareto: finite variance, heavy tail
        mean, sd = a / (a - 1), math.sqrt(a / ((a - 1) ** 2 * (a - 2)))
        make = lambda: CatoniConfidenceSequence(sd)
        self.assertLessEqual(ever_misses(make, lambda r: (1 - r.random()) ** (-1 / a), 150, 60, mean), 0.08)

    def test_catoni_resists_outliers(self):
        proc = CatoniConfidenceSequence(1.0)
        state, rnd = proc.initial_state(), random.Random(5)
        for i in range(300):
            state, out = proc.step(state, {"x": 1e6 if i == 150 else rnd.gauss(0, 1)}, None)
        # one value no distribution with sd 1 produces moves the interval by a tenth; the sample mean moved by 3300
        self.assertLess(out.lower, 0.2)
        self.assertLess(out.upper, 0.6)

    def test_catoni_test_of_the_mean(self):
        make = lambda: CatoniConfidenceSequence(1.0, mean0=0.0, stop_on_reject=True)
        self.assertGreater(rejects(make, lambda r: r.gauss(0.5, 1), 300, 10), 0.9)

    def test_asymptotic_sequence_starts_late_and_covers(self):
        proc = AsymptoticConfidenceSequence(start=50)
        state, rnd = proc.initial_state(), random.Random(6)
        for n in range(1, 61):
            state, out = proc.step(state, {"x": rnd.gauss(0, 1)}, None)
            self.assertEqual(out.lower is None, n < 50)
        make = lambda: AsymptoticConfidenceSequence(start=100)
        self.assertLessEqual(ever_misses(make, lambda r: r.gauss(0, 1), 600, 100, 0.0), 0.1)

    def test_betting_sequence_for_a_conversion_rate(self):
        make = lambda: BettingConfidenceSequence(grid=200)
        self.assertLessEqual(ever_misses(make, lambda r: float(r.random() < 0.05), 400, 60, 0.05), 0.08)
        proc = BettingConfidenceSequence(0.0, 10.0, grid=200)
        state, rnd = proc.initial_state(), random.Random(7)
        for _ in range(500):
            state, out = proc.step(state, {"x": 10.0 * rnd.random()}, None)
        self.assertLess(out.upper - out.lower, 2.0)
        self.assertTrue(out.lower <= 5.0 <= out.upper)


if __name__ == "__main__":
    unittest.main()
