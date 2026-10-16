# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Redirecting traffic: Thompson allocation, the deployment rule, and tests that stay valid.

* The probability that each arm is best and its expected loss agree with
  the closed form for two Beta posteriors (Miller's sum) and with mpmath
  quadrature for three arms and for Student t posteriors.
* The shares are floor + (1 - K floor) P(best), equal during the burn-in,
  recomputed every update_every units; they send most traffic to the
  better arm.
* BoundedABTest with propensities: with constant 50/50 propensities it is
  the fixed-allocation test exactly; under traffic redirected by Thompson
  allocation with no true effect it rejects at most alpha of the time.
"""
import math
import random
import unittest

from sequential_analysis.errors import ContractViolation
from sequential_analysis.procedures import BoundedABTest
from sequential_analysis.procedures.allocation import ThompsonAllocation, _Posterior, best_arm_probabilities

try:
    import mpmath
except ImportError:  # optional: the high-precision references are skipped
    mpmath = None


def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def miller(aa, ba, ab, bb):
    """P(p_B > p_A) for p_A ~ Beta(aa, ba), p_B ~ Beta(ab, bb), ab an integer (Evan Miller's formula)."""
    total = 0.0
    for i in range(int(ab)):
        total += math.exp(log_beta(aa + i, ba + bb) - math.log(bb + i) - log_beta(1 + i, bb) - log_beta(aa, ba))
    return total


def binary(successes, n, prior=(1.0, 1.0)):
    return _Posterior("binary", n, successes / n, 0.0, prior)


class BestArm(unittest.TestCase):
    def test_two_beta_posteriors_match_millers_formula(self):
        for (sa, na), (sb, nb) in (((12, 100), (18, 100)), ((300, 3000), (330, 3000)), ((3, 10), (40, 400)),
                                   ((0, 50), (1, 50))):
            best, _loss = best_arm_probabilities([binary(sa, na), binary(sb, nb)], "binary")
            expected = miller(1 + sa, 1 + na - sa, 1 + sb, 1 + nb - sb)
            self.assertAlmostEqual(best[1], expected, delta=1e-10)
            self.assertAlmostEqual(best[0] + best[1], 1.0, delta=1e-15)

    @unittest.skipIf(mpmath is None, "mpmath is not installed")
    def test_three_arms_and_expected_loss_match_quadrature(self):
        mpmath.mp.dps = 30
        posts = [binary(40, 400), binary(52, 410), binary(9, 60)]
        best, loss = best_arm_probabilities(posts, "binary")

        def pdf(p, x):
            log = (p.a - 1) * mpmath.log(x) + (p.b - 1) * mpmath.log(1 - x)
            return mpmath.exp(log - mpmath.log(mpmath.beta(p.a, p.b)))

        def cdf(p, x):
            return mpmath.betainc(p.a, p.b, 0, x, regularized=True)

        def best_density(k, x):
            return pdf(posts[k], x) * mpmath.fprod(cdf(posts[j], x) for j in range(3) if j != k)

        pieces = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 1]
        top = mpmath.quad(lambda x: x * sum(best_density(k, x) for k in range(3)), pieces)
        for k in range(3):
            ref = mpmath.quad(lambda x, k=k: best_density(k, x), pieces)
            self.assertAlmostEqual(best[k], float(ref), delta=1e-9)
            self.assertAlmostEqual(loss[k], float(top) - posts[k].location, delta=1e-9)

    @unittest.skipIf(mpmath is None, "mpmath is not installed")
    def test_student_t_posteriors_match_quadrature(self):
        mpmath.mp.dps = 30
        posts = [_Posterior("normal", 50, 10.0, 49 * 4.0, (1.0, 1.0)),
                 _Posterior("normal", 80, 10.6, 79 * 9.0, (1.0, 1.0))]
        best, _loss = best_arm_probabilities(posts, "normal")

        def pdf(p, x):
            z = (x - p.location) / p.scale
            df = p.df
            return mpmath.gamma((df + 1) / 2) / (mpmath.gamma(df / 2) * mpmath.sqrt(df * mpmath.pi) * p.scale) * (
                1 + z * z / df) ** (-(df + 1) / 2)

        def cdf(p, x):  # Student t through the regularized incomplete beta function
            z = (x - p.location) / p.scale
            tail = mpmath.betainc(p.df / 2, mpmath.mpf(1) / 2, 0, p.df / (p.df + z * z), regularized=True) / 2
            return 1 - tail if z > 0 else tail

        # P(arm 1 best) = P(mu_1 > mu_0) = int f_1 F_0
        ref = mpmath.quad(lambda x: pdf(posts[1], x) * cdf(posts[0], x), [-mpmath.inf, 9, 10, 11, 12, mpmath.inf])
        self.assertAlmostEqual(best[1], float(ref), delta=1e-7)


def simulate(allocator, rates, n, seed, record=None):
    """Assign n units with the allocator's shares; outcomes 0/1 with the arms' rates. (outputs, records)."""
    rnd = random.Random(seed)
    state, outs, records = allocator.initial_state(), [], []
    shares = state.shares
    for _ in range(n):
        u, arm, acc = rnd.random(), 0, 0.0
        for k, s in enumerate(shares):
            acc += s
            if u < acc:
                arm = k
                break
        else:
            arm = len(shares) - 1
        y = 1 if rnd.random() < rates[arm] else 0
        records.append({"arm": arm, "y": y, "p": list(shares)})
        state, out = allocator.step(state, {"arm": arm, "y": y})
        outs.append(out)
        shares = out.shares
    return outs, records


class Allocation(unittest.TestCase):
    def test_shares(self):
        allocator = ThompsonAllocation(3, floor=0.05, min_units=20, update_every=50)
        outs, _ = simulate(allocator, (0.1, 0.1, 0.3), 2000, 1)
        early = [o for o in outs if min(o.counts) < 20]
        self.assertTrue(all(o.shares == (1 / 3,) * 3 for o in early))
        last = outs[-1]
        for s, p in zip(last.shares, last.best):
            self.assertAlmostEqual(s, 0.05 + 0.85 * p, delta=1e-15)
        self.assertAlmostEqual(sum(last.shares), 1.0, delta=1e-12)
        self.assertEqual(last.leader, 2)
        changes = [i for i in range(1, len(outs)) if outs[i].shares != outs[i - 1].shares]
        self.assertTrue(all((b - a) % 50 == 0 for a, b in zip(changes, changes[1:])))

    def test_sends_most_traffic_to_the_better_arm(self):
        outs, _ = simulate(ThompsonAllocation(2, floor=0.05), (0.10, 0.12), 20000, 2)
        self.assertGreater(outs[-1].counts[1] / 20000, 0.7)
        self.assertGreater(outs[-1].best[1], 0.99)

    def test_deployment_rule(self):
        chosen = []
        for seed in range(10):
            outs, _ = simulate(ThompsonAllocation(2, threshold=0.002), (0.10, 0.13), 8000, 100 + seed)
            chosen.append(outs[-1].deploy)
        self.assertGreaterEqual(chosen.count(1), 9)

    def test_normal_metric(self):
        rnd = random.Random(3)
        allocator = ThompsonAllocation(2, metric="normal", min_units=10, update_every=10)
        state = allocator.initial_state()
        for _t in range(3000):
            arm = 0 if rnd.random() < state.shares[0] else 1
            state, out = allocator.step(state, {"arm": arm, "y": rnd.gauss(10.0 + 0.5 * arm, 2.0)})
        self.assertEqual(out.leader, 1)

    def test_refusals(self):
        for make in (lambda: ThompsonAllocation(1), lambda: ThompsonAllocation(2, floor=0.6),
                     lambda: ThompsonAllocation(2, metric="poisson"), lambda: ThompsonAllocation(2, update_every=0)):
            with self.assertRaises(ValueError):
                make()


class PropensityWeightedTest(unittest.TestCase):
    def test_constant_half_propensities_are_the_fixed_allocation_test(self):
        rnd = random.Random(4)
        fixed = BoundedABTest(grid=100, backend="python")
        weighted = BoundedABTest(grid=100, backend="python", propensity_input="p", floor=0.5)
        sf, sw = fixed.initial_state(), weighted.initial_state()
        for _ in range(500):
            arm = rnd.randrange(2)
            y = rnd.random() * (0.6 if arm else 0.5)
            sf, of = fixed.step(sf, {"arm": arm, "y": y})
            sw, ow = weighted.step(sw, {"arm": arm, "y": y, "p": [0.5, 0.5]})
            self.assertEqual((of.lower, of.upper, of.log_e_value), (ow.lower, ow.upper, ow.log_e_value))

    def test_valid_under_redirected_traffic(self):
        rejections, runs, alpha = 0, 40, 0.1
        for r in range(runs):
            _, records = simulate(ThompsonAllocation(2, floor=0.1, min_units=50, update_every=50), (0.3, 0.3),
                                  1000, 1000 + r)
            test = BoundedABTest(alpha=alpha, grid=20, backend="python", propensity_input="p", floor=0.1)
            state = test.initial_state()
            for x in records:
                state, out = test.step(state, x)
                if out.decision is not None:
                    rejections += 1
                    break
        self.assertLessEqual(rejections / runs, alpha)

    def test_finds_an_effect_under_redirected_traffic(self):
        outs, records = simulate(ThompsonAllocation(2, floor=0.1), (0.2, 0.3), 20000, 5)
        test = BoundedABTest(arms=(1, 0), grid=200, backend="python", propensity_input="p", floor=0.1)
        state = test.initial_state()
        for x in records:
            state, out = test.step(state, x)
        self.assertLess(outs[-1].counts[0] / 20000, 0.12)  # the worse arm got little more than its floor
        self.assertGreater(out.lower, 0.0)
        self.assertLess(out.lower, 0.1)
        self.assertGreater(out.upper, 0.1)

    def test_other_arms_count_as_zero(self):
        test = BoundedABTest(grid=50, backend="python", propensity_input="p", arm_count=3, floor=0.1, arms=(2, 0))
        state = test.initial_state()
        state, out = test.step(state, {"arm": 1, "y": 0.7, "p": [0.3, 0.4, 0.3]})
        self.assertEqual(state.n, 1)
        self.assertEqual(state.others, 1)
        self.assertEqual(out.means, (None, None))

    def test_refusals(self):
        test = BoundedABTest(propensity_input="p", floor=0.2)
        with self.assertRaises(ContractViolation):
            test.step(test.initial_state(), {"arm": 0, "y": 0.5, "p": [0.9, 0.1]})
        for make in (lambda: BoundedABTest(propensity_input="p"), lambda: BoundedABTest(floor=0.1),
                     lambda: BoundedABTest(propensity_input="p", floor=0.2, allocation=0.3),
                     lambda: BoundedABTest(propensity_input="p", floor=0.2, arms=(3, 0))):
            with self.assertRaises(ValueError):
                make()
        self.assertNotIn("floor", BoundedABTest().config())


if __name__ == "__main__":
    unittest.main()
