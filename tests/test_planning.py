# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The planner against exact operating characteristics.

For a sampling plan on Bernoulli data the design module computes the
rejection probability and the distribution of the stopping time exactly;
the simulated plan must agree within Monte Carlo error.
"""
import math
import unittest

from sequential_analysis.design import operating_characteristic
from sequential_analysis.planning import conversion_units, metric_units, simulate
from sequential_analysis.procedures import Bernoulli, TwoProportionTest


class AgainstExactOperatingCharacteristics(unittest.TestCase):
    def test_power_and_median_of_an_sprt_plan(self):
        from sequential_analysis.design import kiefer_weiss_plan

        test = kiefer_weiss_plan(0.3, 0.5, 0.05, 0.05, 100, family=Bernoulli()).plan
        for theta in (0.3, 0.5):
            oc = operating_characteristic(test, Bernoulli(), theta)
            summary = simulate(lambda: test, lambda rnd, theta=theta: {"x": int(rnd.random() < theta)},
                               runs=4000, max_n=100, seed=11,
                               stop=lambda out: out.decision == "reject_h0")
            se = math.sqrt(oc.reject * (1 - oc.reject) / 4000)
            self.assertLess(abs(summary.power - oc.reject), 4 * se + 1e-9, theta)
            # stopping-time distribution among rejecting runs: compare the median with the exact one
            lo, hi = summary.power_interval
            self.assertLessEqual(lo, oc.reject + 4 * se)
            self.assertGreaterEqual(hi, oc.reject - 4 * se)

    def test_stopping_time_distribution(self):
        from sequential_analysis.design import kiefer_weiss_plan

        test = kiefer_weiss_plan(0.3, 0.5, 0.05, 0.05, 100, family=Bernoulli()).plan
        oc = operating_characteristic(test, Bernoulli(), 0.4)
        summary = simulate(lambda: test, lambda rnd: {"x": int(rnd.random() < 0.4)}, runs=4000, max_n=100, seed=3)
        # every run stops (the plan forces a decision by n = 100): compare the median
        self.assertEqual(summary.stopped, 4000)
        cumulative, exact_median = 0.0, None
        for n, p in enumerate(oc.stop, start=1):
            cumulative += p
            if cumulative >= 0.5:
                exact_median = n
                break
        self.assertLessEqual(abs(summary.median_n - exact_median), 2)
        mean = summary.mean_n_stopped
        self.assertLess(abs(mean - oc.expected_n), 0.05 * oc.expected_n)


class Scenarios(unittest.TestCase):
    def test_conversion_plan_and_censoring(self):
        summary = simulate(TwoProportionTest, conversion_units(0.15, 0.10), runs=30, max_n=20_000, seed=1)
        self.assertGreater(summary.power, 0.8)
        self.assertIsNotNone(summary.median_n)
        short = simulate(TwoProportionTest, conversion_units(0.15, 0.10), runs=30, max_n=200, seed=1)
        self.assertLess(short.power, 0.5)
        self.assertIsNone(short.median_n)  # the budget is too small to tell
        null = simulate(lambda: TwoProportionTest(0.05), conversion_units(0.1, 0.1), runs=100, max_n=2000, seed=5)
        self.assertLessEqual(null.power, 0.05 + 3 * math.sqrt(0.05 * 0.95 / 100))

    def test_reproducible_and_arguments(self):
        draw = metric_units(lambda r: r.random(), lambda r: r.random() * 0.9)
        a = simulate(TwoProportionTest, conversion_units(0.2, 0.1), runs=5, max_n=3000, seed=9)
        b = simulate(TwoProportionTest, conversion_units(0.2, 0.1), runs=5, max_n=3000, seed=9)
        self.assertEqual(a, b)
        self.assertIn("y", draw(__import__("random").Random(1)))
        with self.assertRaises(ValueError):
            simulate(TwoProportionTest, conversion_units(0.2, 0.1), runs=0)
        with self.assertRaises(ValueError):
            conversion_units(1.5, 0.1)
        with self.assertRaises(ValueError):
            a.quantile(1.0)


if __name__ == "__main__":
    unittest.main()
