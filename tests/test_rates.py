# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Mathematical correctness of the rate-ratio test.

* A block's e-value has conditional mean exactly 1 under H0, by summing
  over every split of the block's events (for any plug-in rate ratio).
* Type I error by simulation with unequal exposures and a common rate that
  drifts over time; coverage of the confidence sequence uniformly over time.
* Power, and the effect of overdispersion stated in the docstring.
"""
import math
import random
import unittest

from sequential_analysis.procedures import RateRatioTest


def fold(proc, inputs):
    state, outs = proc.initial_state(), []
    for x in inputs:
        state, out = proc.step(state, x)
        outs.append(out)
    return state, outs


def poisson(rnd, mean):
    k, p, limit = 0, rnd.random(), math.exp(-mean)
    while p > limit:
        k += 1
        p *= rnd.random()
    return k


def units(n, seed, ratio=1.0, base=lambda i: 2.0, allocation=0.5):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        arm = int(rnd.random() < allocation)
        e = rnd.uniform(0.2, 3.0)
        rate = base(i) * (ratio if arm else 1.0)
        out.append({"arm": arm, "count": poisson(rnd, rate * e), "exposure": e})
    return out


def upper_bound(alpha, runs):
    return alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / runs)


class RateRatio(unittest.TestCase):
    def test_block_e_value_has_mean_one_under_the_null(self):
        proc = RateRatioTest(grid=0)
        for e_t, e_c in ((1.0, 1.0), (0.3, 2.5), (4.0, 0.7)):
            for rho0 in (1.0, 0.5, 2.0):
                q0 = proc.share(rho0, e_t, e_c)
                for rho_hat in (0.2, 0.9, 1.7, 6.0):
                    for total in (0, 1, 5, 30):
                        mean = 0.0
                        for k_t in range(total + 1):
                            p = math.comb(total, k_t) * q0**k_t * (1 - q0) ** (total - k_t)
                            mean += p * math.exp(proc.pair_log_e(rho_hat, rho0, "two-sided", k_t, total, e_t, e_c))
                        self.assertAlmostEqual(mean, 1.0, delta=1e-12)

    def test_level_with_unequal_exposures_and_a_drifting_rate(self):
        runs, alpha = 300, 0.05
        rejected = 0
        for r in range(runs):
            data = units(1500, 100 + r, base=lambda i: 1.0 + 0.8 * math.sin(i / 50.0))
            state, _ = fold(RateRatioTest(alpha, grid=0), data)
            rejected += state.decision is not None
        self.assertLessEqual(rejected / runs, upper_bound(alpha, runs))

    def test_coverage_of_the_rate_ratio_uniformly_over_time(self):
        runs, alpha = 150, 0.1
        missed = 0
        for r in range(runs):
            _, outs = fold(RateRatioTest(alpha, grid=60), units(800, 500 + r, ratio=1.3, base=lambda i: 1.5))
            missed += any((o.lower is not None and o.lower > 1.3) or (o.upper is not None and o.upper < 1.3)
                          for o in outs)
        self.assertLessEqual(missed / runs, upper_bound(alpha, runs))

    def test_power_and_shrinking_interval(self):
        hits = 0
        for r in range(20):
            state, outs = fold(RateRatioTest(0.05, alternative="greater", grid=100), units(1500, 900 + r, ratio=1.25))
            hits += state.decision is not None
        self.assertGreaterEqual(hits, 18)
        self.assertLess(outs[-1].upper / outs[-1].lower, outs[100].upper / outs[100].lower)
        self.assertAlmostEqual(outs[-1].rate_ratio, 1.25, delta=0.1)

    def test_one_sided_level_inside_the_null(self):
        runs, alpha = 200, 0.05
        rejected = 0
        for r in range(runs):
            state, _ = fold(RateRatioTest(alpha, alternative="greater", grid=0), units(1000, 2000 + r, ratio=0.8))
            rejected += state.decision is not None
        self.assertLessEqual(rejected / runs, upper_bound(alpha, runs))

    def test_overdispersion_breaks_exactness_as_documented(self):
        # gamma-mixed Poisson counts (negative binomial) with equal means: the test over-rejects
        runs, alpha = 200, 0.05
        rejected = 0
        for r in range(runs):
            rnd = random.Random(3000 + r)
            data = []
            for _ in range(1500):
                arm = int(rnd.random() < 0.5)
                data.append({"arm": arm, "count": poisson(rnd, 3.0 * rnd.gammavariate(0.3, 1 / 0.3)), "exposure": 1.0})
            state, _ = fold(RateRatioTest(alpha, grid=0), data)
            rejected += state.decision is not None
        self.assertGreater(rejected / runs, 2 * alpha)

    def test_blocks_state_and_arguments(self):
        proc = RateRatioTest(grid=20)
        state, outs = fold(proc, [{"arm": 1, "count": 3, "exposure": 1.0}, {"arm": 1, "count": 0, "exposure": 2.0},
                                  {"arm": 0, "count": 1, "exposure": 1.0}, {"arm": 1, "count": 2, "exposure": 0.5},
                                  {"arm": 2, "count": 9, "exposure": 1.0}])
        # the first unit is replaced by the second, which pairs with the third; the fourth waits
        self.assertEqual((outs[-1].pairs, state.pending, state.pair_counts), (1, (0.0, 2.0, 0.5), (0, 1)))
        self.assertEqual(outs[-1], outs[-2])
        self.assertEqual(proc.decode_state(proc.encode_state(state)), state)
        from sequential_analysis.errors import ContractViolation

        with self.assertRaises(ContractViolation):
            proc.step(state, {"arm": 1, "count": 1, "exposure": 0.0})
        for kwargs in ({"rho0": 0.0}, {"arms": (1, 1)}, {"grid": 5}, {"ratio_range": (2.0, 1.0)},
                       {"input": "arm"}, {"alternative": "both"}):
            with self.assertRaises(ValueError):
                RateRatioTest(**kwargs)
        self.assertEqual(RateRatioTest(exposure=False).inputs().keys(), {"arm", "count"})


if __name__ == "__main__":
    unittest.main()
