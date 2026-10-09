# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Statistical properties of the procedures and of the design module.

Every claim a docstring makes about error rates, coverage or run lengths
is checked here, by simulation with fixed seeds or by exact computation.
Monte Carlo bounds allow three standard errors.
"""
import math
import random
import unittest

from seqinfer import ContractViolation, PositionalPair, Run, difference
from seqinfer.design import kiefer_weiss_plan, lattice, operating_characteristic
from seqinfer.procedures import (
    REJECT_H0,
    SPRT,
    Bernoulli,
    BettingMeanTest,
    Exponential,
    Gaussian,
    NormalMixtureSPRT,
    PlanTest,
    Poisson,
    SequentialSignTest,
    ShiryaevRoberts,
)
from seqinfer.sources import from_values


def run_length(proc, draw, horizon):
    """Steps until the procedure stops, or None; and the last output."""
    state, out = proc.initial_state(), None
    for n in range(1, horizon + 1):
        state, out = proc.step(state, {proc.input: draw()}, None)
        if proc.is_terminal(state):
            return n, out
    return None, out


def rejection_rate(proc, make_draw, reps, horizon, seed):
    """Fraction of runs that reject H0 within the horizon."""
    rnd = random.Random(seed)
    rejected = 0
    for _ in range(reps):
        n, out = run_length(proc, make_draw(rnd), horizon)
        rejected += n is not None and out.decision == REJECT_H0
    return rejected / reps


def upper_bound(p, reps):
    return p + 3.0 * math.sqrt(p * (1.0 - p) / reps)


def poisson_draw(rnd, lam):
    def draw():
        limit, k, prod = math.exp(-lam), 0, rnd.random()
        while prod > limit:
            k += 1
            prod *= rnd.random()
        return k

    return draw


class Families(unittest.TestCase):
    def test_llr_is_the_difference_of_log_densities(self):
        def log_poisson(x, lam):
            return x * math.log(lam) - lam - math.lgamma(x + 1)

        def log_exponential(x, rate):
            return math.log(rate) - rate * x

        for family, logpdf, xs, (a, b) in (
            (Poisson(), log_poisson, [0, 1, 4, 17], (2.0, 3.5)),
            (Exponential(), log_exponential, [0.0, 0.3, 2.5], (0.5, 2.0)),
        ):
            for x in xs:
                self.assertAlmostEqual(family.llr(x, a, b), logpdf(x, a) - logpdf(x, b), places=12)

    def test_kl_is_the_expected_llr(self):
        a, b = 2.0, 3.5
        pmf = (math.exp(k * math.log(a) - a - math.lgamma(k + 1)) for k in range(200))
        exact = sum(p * Poisson().llr(k, a, b) for k, p in enumerate(pmf))
        self.assertAlmostEqual(Poisson().kl(a, b), exact, places=12)
        h = 1e-3  # midpoint rule on [0, 40]
        midpoints = ((i + 0.5) * h for i in range(40_000))
        integral = sum(a * math.exp(-a * x) * Exponential().llr(x, a, b) * h for x in midpoints)
        self.assertAlmostEqual(Exponential().kl(a, b), integral, places=6)

    def test_contracts(self):
        run = Run(SPRT(Poisson(), 1.0, 2.0), on_invalid="raise")
        for bad in (-1, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ContractViolation), self.assertLogs("seqinfer", "WARNING"):
                Run(SPRT(Poisson(), 1.0, 2.0), on_invalid="raise").step({"x": bad})
        run.step({"x": 3})
        with self.assertRaises(ValueError):
            Exponential().check(0.0)

    def test_sprt_error_rates_on_poisson_data(self):
        proc = SPRT(Poisson(), 2.0, 3.0, 0.05, 0.05)
        reps = 600
        type1 = rejection_rate(proc, lambda r: poisson_draw(r, 2.0), reps, 10_000, 1)
        self.assertLessEqual(type1, upper_bound(0.05, reps))
        type2 = 1.0 - rejection_rate(proc, lambda r: poisson_draw(r, 3.0), reps, 10_000, 2)
        self.assertLessEqual(type2, upper_bound(0.05, reps))


class ShiryaevRobertsDetector(unittest.TestCase):
    def test_average_run_length_to_false_alarm_is_at_least_the_threshold(self):
        threshold, reps = 50.0, 600
        proc = ShiryaevRoberts(Gaussian(1.0), 0.0, 1.0, threshold)
        rnd = random.Random(2)
        lengths = [run_length(proc, lambda: rnd.gauss(0.0, 1.0), 100_000)[0] for _ in range(reps)]
        mean = sum(lengths) / reps
        se = math.sqrt(sum((x - mean) ** 2 for x in lengths) / reps / reps)
        self.assertGreaterEqual(mean + 3 * se, threshold)

    def test_detects_a_change(self):
        proc = ShiryaevRoberts(Gaussian(1.0), 0.0, 1.0, 1000.0)
        rnd = random.Random(3)
        delays = []
        for _ in range(200):
            draws = iter([rnd.gauss(0.0, 1.0) for _ in range(100)] + [rnd.gauss(1.0, 1.0) for _ in range(1000)])
            n, _ = run_length(proc, draws.__next__, 1100)
            if n is not None and n > 100:
                delays.append(n - 100)
        self.assertGreater(len(delays), 150)
        self.assertLess(sum(delays) / len(delays), 25)  # log(1000) / KL = 13.8, plus overshoot

    def test_first_step_from_zero(self):
        proc = ShiryaevRoberts(Gaussian(1.0), 0.0, 1.0, 10.0)
        state, out = proc.step(proc.initial_state(), {"x": 0.5}, None)
        self.assertAlmostEqual(out.log_statistic, Gaussian(1.0).llr(0.5, 1.0, 0.0))
        self.assertEqual(proc.decode_state(proc.encode_state(state)), state)


class MixtureSPRT(unittest.TestCase):
    def test_type_one_error_under_continuous_monitoring(self):
        reps = 600
        rate = rejection_rate(NormalMixtureSPRT(1.0, 0.5), lambda r: (lambda: r.gauss(0.0, 1.0)), reps, 1000, 4)
        self.assertLessEqual(rate, upper_bound(0.05, reps))

    def test_power(self):
        rate = rejection_rate(NormalMixtureSPRT(1.0, 0.5), lambda r: (lambda: r.gauss(0.4, 1.0)), 100, 2000, 5)
        self.assertEqual(rate, 1.0)

    def test_confidence_sequence_covers_uniformly_over_time(self):
        proc = NormalMixtureSPRT(1.0, 0.5, alpha=0.1, stop_on_reject=False)
        rnd, reps, misses = random.Random(6), 600, 0
        for _ in range(reps):
            state = proc.initial_state()
            for _ in range(400):
                state, out = proc.step(state, {"x": rnd.gauss(2.0, 1.0)}, None)
                if not out.lower <= 2.0 <= out.upper:
                    misses += 1
                    break
        self.assertLessEqual(misses / reps, upper_bound(0.1, reps))

    def test_p_value_and_interval_agree_with_the_test(self):
        proc = NormalMixtureSPRT(1.0, 0.5, theta0=0.0, alpha=0.05, stop_on_reject=False)
        rnd, state, last_p = random.Random(7), proc.initial_state(), 1.0
        for _ in range(300):
            state, out = proc.step(state, {"x": rnd.gauss(0.15, 1.0)}, None)
            self.assertLessEqual(out.p_value, last_p)  # always-valid p-values only decrease
            last_p = out.p_value
            # the interval is the inverted test: it excludes theta0 exactly when the test rejects
            if abs(out.log_lambda - math.log(20)) > 1e-9:
                self.assertEqual(not out.lower <= 0.0 <= out.upper, out.log_lambda >= math.log(20))


class BettingTests(unittest.TestCase):
    def test_mean_test_is_valid_for_skewed_and_discrete_data(self):
        reps = 600
        for label, make in (
            ("beta(0.5, 2)", lambda r: (lambda: r.betavariate(0.5, 2.0))),
            ("two-point", lambda r: (lambda: float(r.random() < 0.2))),
        ):
            for alternative in ("two-sided", "greater", "less"):
                with self.subTest(data=label, alternative=alternative):
                    proc = BettingMeanTest(0.2, alternative=alternative)
                    self.assertLessEqual(rejection_rate(proc, make, reps, 400, 8), upper_bound(0.05, reps))

    def test_mean_test_power_and_direction(self):
        greater = rejection_rate(BettingMeanTest(0.2), lambda r: (lambda: r.betavariate(0.75, 1.75)), 100, 2000, 9)
        self.assertEqual(greater, 1.0)
        wrong_side = rejection_rate(
            BettingMeanTest(0.2, alternative="less"), lambda r: (lambda: r.betavariate(0.75, 1.75)), 300, 400, 10
        )
        self.assertLessEqual(wrong_side, upper_bound(0.05, 300))

    def test_mean_test_does_not_depend_on_the_units(self):
        rnd = random.Random(11)
        data = [rnd.random() for _ in range(200)]
        unit, scaled = BettingMeanTest(0.4), BettingMeanTest(40.0, 0.0, 100.0)
        su, ss = unit.initial_state(), scaled.initial_state()
        for v in data:
            su, ou = unit.step(su, {"x": v}, None)
            ss, os_ = scaled.step(ss, {"x": 100.0 * v}, None)
            self.assertAlmostEqual(ou.log_wealth, os_.log_wealth, places=9)
            self.assertAlmostEqual(100.0 * ou.estimate, os_.estimate, places=9)

    def test_bounds_are_part_of_the_contract(self):
        with self.assertLogs("seqinfer", "WARNING"):
            run = Run(BettingMeanTest(0.5))
            run.step({"x": 1.5})
        self.assertEqual(run.counters["invalid_skipped"], 1)

    def test_sign_test_is_valid_without_moments(self):
        reps = 600

        def cauchy(r, loc=0.0):
            return lambda: loc + math.tan(math.pi * (r.random() - 0.5))

        self.assertLessEqual(rejection_rate(SequentialSignTest(), cauchy, reps, 400, 12), upper_bound(0.05, reps))
        ties = lambda r: (lambda: float(r.choice([-1, 0, 0, 1])))
        self.assertLessEqual(rejection_rate(SequentialSignTest(), ties, reps, 400, 13), upper_bound(0.05, reps))
        self.assertEqual(rejection_rate(SequentialSignTest(), lambda r: cauchy(r, 0.7), 100, 3000, 14), 1.0)

    def test_sign_test_counts_ties(self):
        proc = SequentialSignTest(1.0)
        state = proc.initial_state()
        for v in (1.0, 2.0, 1.0, 0.0):
            state, out = proc.step(state, {"x": v}, None)
        self.assertEqual((state.ties, state.n, out.estimate), (2, 2, 0.5))

    def test_paired_sign_test_through_a_topology(self):
        rnd = random.Random(15)
        treated = [rnd.expovariate(1.0) + 0.8 for _ in range(400)]
        control = [rnd.expovariate(1.0) for _ in range(400)]
        run = Run(SequentialSignTest(0.0), topology=PositionalPair(("t", "c")).map(difference("t", "c"), "t-c"))
        for a, b in zip(from_values("t", treated), from_values("c", control), strict=True):
            run.offer(a)
            run.offer(b)
            if run.terminal:
                break
        self.assertTrue(run.terminal)
        self.assertEqual(run.state.decision, REJECT_H0)


class Design(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = kiefer_weiss_plan(0.2, 0.5, 0.05, 0.1, horizon=60)

    def test_error_rates_are_met_exactly(self):
        d = self.design
        self.assertLessEqual(d.at_theta0.reject, 0.05)
        self.assertLessEqual(d.at_theta1.accept, 0.1)
        self.assertLessEqual(d.plan.horizon, 60)

    def test_operating_characteristic_is_a_distribution(self):
        for theta in (0.2, 0.35, 0.5):
            oc = operating_characteristic(self.design.plan, Bernoulli(), theta)
            self.assertAlmostEqual(oc.reject + oc.accept, 1.0, places=12)
            self.assertAlmostEqual(sum(oc.stop), 1.0, places=12)
            self.assertAlmostEqual(oc.expected_n, sum(n * p for n, p in enumerate(oc.stop, start=1)), places=9)

    def test_backward_induction_agrees_with_the_forward_recursion(self):
        d = self.design
        forward = d.at_theta_star.expected_n + d.lambda0 * d.at_theta0.reject + d.lambda1 * d.at_theta1.accept
        self.assertAlmostEqual(d.lagrangian, forward, places=9)

    def test_no_plan_with_the_same_horizon_has_a_smaller_lagrangian(self):
        d = self.design
        rnd = random.Random(16)
        base = [list(stage) for stage in d.plan.plan]
        for _ in range(30):
            stages = [list(s) for s in base]
            n = rnd.randrange(len(stages) - 1)
            lower, upper = stages[n]
            if lower is not None and rnd.random() < 0.5:
                stages[n][0] = lower + rnd.choice((-1, 1))
            elif upper is not None:
                stages[n][1] = upper + rnd.choice((-1, 1))
            if stages[n][0] is not None and stages[n][1] is not None and stages[n][0] >= stages[n][1]:
                continue
            other = PlanTest(stages, contract=Bernoulli().contract("x"))
            value = (
                operating_characteristic(other, Bernoulli(), d.theta_star).expected_n
                + d.lambda0 * operating_characteristic(other, Bernoulli(), 0.2).reject
                + d.lambda1 * operating_characteristic(other, Bernoulli(), 0.5).accept
            )
            self.assertGreaterEqual(value, d.lagrangian - 1e-9)

    def test_exact_values_match_simulation_through_a_run(self):
        rnd, reps, rejected, total = random.Random(17), 4000, 0, 0
        for _ in range(reps):
            run = Run(self.design.plan)
            while not run.terminal:
                event = run.step({"x": int(rnd.random() < 0.35)})
            rejected += event.output.decision == REJECT_H0
            total += event.output.n
        exact = operating_characteristic(self.design.plan, Bernoulli(), 0.35)
        self.assertLess(abs(rejected / reps - exact.reject), 3 * math.sqrt(0.25 / reps))
        self.assertLess(abs(total / reps - exact.expected_n), 1.0)

    def test_smaller_maximum_sample_size_than_the_sprt(self):
        """The point of the Kiefer-Weiss problem, on the case from the docs."""
        grid = [0.2 + 0.015 * i for i in range(21)]

        def max_asn(plan):
            return max(operating_characteristic(plan, Bernoulli(), t).expected_n for t in grid)

        a, b = math.log(0.5 / 0.8), math.log(0.5 / 0.2) - math.log(0.5 / 0.8)
        for scale in (x / 50 for x in range(30, 100)):  # Wald thresholds scaled until errors are met
            hi_t, lo_t = math.log(18) * scale, math.log(0.1 / 0.95) * scale
            stages = []
            for n in range(1, 400):
                lo, hi = math.floor((lo_t - n * a) / b), math.ceil((hi_t - n * a) / b)
                stages.append((lo if lo >= 0 else None, hi if hi <= n else None))
            cut = (lo + hi) / 2  # truncation far beyond any realistic stopping time
            stages.append((cut, cut))
            sprt = PlanTest(stages, contract=Bernoulli().contract("x"))
            if (
                operating_characteristic(sprt, Bernoulli(), 0.2).reject <= 0.05
                and operating_characteristic(sprt, Bernoulli(), 0.5).accept <= 0.1
            ):
                break
        self.assertLess(max_asn(self.design.plan), max_asn(sprt))

    def test_poisson_operating_characteristic(self):
        plan = PlanTest([(None, 6), (None, 9), (7.5, 7.5)], contract=Poisson().contract("x"))
        oc = operating_characteristic(plan, Poisson(), 2.0)
        self.assertAlmostEqual(oc.reject + oc.accept, 1.0, places=12)
        p_first = 1.0 - sum(math.exp(-2.0) * 2.0**k / math.factorial(k) for k in range(6))
        self.assertAlmostEqual(oc.stop[0], p_first, places=12)

    def test_impossible_requirements_are_refused(self):
        with self.assertRaises(ValueError):
            kiefer_weiss_plan(0.2, 0.5, 0.001, 0.001, horizon=5)
        with self.assertRaises(NotImplementedError):
            operating_characteristic(self.design.plan, Exponential(), 1.0)


def simulate_plan(plan, draw, reps):
    """Rejection rate and average sample number of a plan run on real data."""
    rejected = total = 0
    for _ in range(reps):
        state = plan.initial_state()
        while not plan.is_terminal(state):
            state, out = plan.step(state, {"x": draw()}, None)
        rejected += out.decision == REJECT_H0
        total += out.n
    return rejected / reps, total / reps


class Lattices(unittest.TestCase):
    def test_poisson_lattice_is_the_distribution(self):
        lat = lattice(Poisson(), 3.0)
        self.assertTrue(lat.exact)
        self.assertAlmostEqual(sum(lat.weights), 1.0, places=15)
        self.assertAlmostEqual(sum(k * w for k, w in enumerate(lat.weights)), 3.0, places=13)

    def test_normal_lattice_keeps_mean_and_variance(self):
        lat = lattice(Gaussian(2.0), 0.3, step=0.25)
        self.assertFalse(lat.exact)
        values = [(lat.offset + i) * lat.step for i in range(len(lat.weights))]
        mean = sum(v * w for v, w in zip(values, lat.weights, strict=True))
        var = sum((v - mean) ** 2 * w for v, w in zip(values, lat.weights, strict=True))
        self.assertAlmostEqual(mean, 0.3, places=12)
        self.assertAlmostEqual(var, 4.0, places=10)


class PoissonDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = kiefer_weiss_plan(2.0, 3.0, 0.05, 0.05, horizon=60, family=Poisson())

    def test_error_rates_and_consistency(self):
        d = self.design
        self.assertTrue(d.exact)
        self.assertLessEqual(d.at_theta0.reject, 0.05)
        self.assertLessEqual(d.at_theta1.accept, 0.05)
        self.assertGreater(d.at_theta0.reject, 0.045)  # the slack is spent
        forward = d.at_theta_star.expected_n + d.lambda0 * d.at_theta0.reject + d.lambda1 * d.at_theta1.accept
        self.assertAlmostEqual(d.lagrangian, forward, places=8)

    def test_exact_values_match_simulation(self):
        rnd, reps = random.Random(18), 3000
        rate, asn = simulate_plan(self.design.plan, poisson_draw(rnd, 2.5), reps)
        exact = operating_characteristic(self.design.plan, Poisson(), 2.5)
        self.assertLess(abs(rate - exact.reject), 3 * math.sqrt(0.25 / reps))
        self.assertLess(abs(asn - exact.expected_n), 1.0)

    def test_no_perturbed_plan_has_a_smaller_lagrangian(self):
        d = self.design
        rnd = random.Random(19)
        for _ in range(20):
            stages = [list(stage) for stage in d.plan.plan]
            n = rnd.randrange(len(stages) - 1)
            side = 0 if stages[n][0] is not None and rnd.random() < 0.5 else 1
            if stages[n][side] is None:
                continue
            stages[n][side] += rnd.choice((-1, 1))
            if stages[n][0] is not None and stages[n][1] is not None and stages[n][0] >= stages[n][1]:
                continue
            other = PlanTest(stages, contract=Poisson().contract("x"))
            value = (
                operating_characteristic(other, Poisson(), d.theta_star).expected_n
                + d.lambda0 * operating_characteristic(other, Poisson(), 2.0).reject
                + d.lambda1 * operating_characteristic(other, Poisson(), 3.0).accept
            )
            self.assertGreaterEqual(value, d.lagrangian - 1e-9)


class NormalDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = kiefer_weiss_plan(0.0, 1.0, 0.05, 0.05, horizon=40, family=Gaussian(1.0), step=0.125)

    def test_error_rates_on_the_lattice_and_consistency(self):
        d = self.design
        self.assertFalse(d.exact)
        self.assertLessEqual(d.at_theta0.reject, 0.05)
        self.assertLessEqual(d.at_theta1.accept, 0.05)
        forward = d.at_theta_star.expected_n + d.lambda0 * d.at_theta0.reject + d.lambda1 * d.at_theta1.accept
        self.assertAlmostEqual(d.lagrangian, forward, places=6)

    def test_error_rates_on_real_normal_data(self):
        rnd, reps = random.Random(20), 4000
        for theta, target in ((0.0, 0.05), (1.0, 0.95)):
            rate, _ = simulate_plan(self.design.plan, lambda theta=theta: rnd.gauss(theta, 1.0), reps)
            self.assertLess(abs(rate - target), 3 * math.sqrt(target * (1 - target) / reps), msg=f"theta={theta}")

    def test_smaller_expected_sample_size_than_the_2sprt(self):
        """Against Lorden's 2-SPRT with thresholds raised until its errors are the same."""
        from seqinfer.procedures import TwoSPRT

        d, gauss = self.design, Gaussian(1.0)
        for a in (x / 20 for x in range(40, 120)):
            plan = PlanTest(TwoSPRT(gauss, 0.0, 1.0, thresholds=(a, a)).as_plan(), contract=gauss.contract("x"))
            if (
                operating_characteristic(plan, gauss, 0.0, step=0.125).reject <= 0.05
                and operating_characteristic(plan, gauss, 1.0, step=0.125).accept <= 0.05
            ):
                break
        star = operating_characteristic(plan, gauss, d.theta_star, step=0.125).expected_n
        self.assertLess(d.at_theta_star.expected_n, star)


class GroupSequential(unittest.TestCase):
    FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)

    def test_obrien_fleming_boundaries_match_published_values(self):
        from seqinfer.group_sequential import group_sequential_design

        d = group_sequential_design(0.025, self.FRACTIONS, "obrien-fleming")
        for got, published in zip(d.bounds, (4.877, 3.357, 2.680, 2.290, 2.031), strict=True):
            self.assertAlmostEqual(got, published, places=3)
        two = group_sequential_design(0.05, self.FRACTIONS, "obrien-fleming", two_sided=True)
        for a, b in zip(two.bounds, d.bounds, strict=True):
            self.assertAlmostEqual(a, b, places=4)

    def test_alpha_is_spent_as_planned(self):
        from seqinfer.group_sequential import group_sequential_design

        for spending in ("obrien-fleming", "pocock", ("power", 2.0)):
            d = group_sequential_design(0.025, self.FRACTIONS, spending)
            crossing = d.crossing(0.0)
            cumulative = 0.0
            for p, spent in zip(crossing, d.spent, strict=True):
                cumulative += p
                self.assertAlmostEqual(cumulative, spent, places=7, msg=str(spending))

    def test_crossing_probabilities_match_simulation(self):
        from seqinfer.group_sequential import group_sequential_design

        d = group_sequential_design(0.025, self.FRACTIONS, "pocock")
        rnd, reps, drift = random.Random(21), 40_000, 2.5
        hits = [0] * 5
        for _ in range(reps):
            b, previous = 0.0, 0.0
            for k, (t, c) in enumerate(zip(d.fractions, d.bounds, strict=True)):
                b += rnd.gauss(drift * (t - previous), math.sqrt(t - previous))
                previous = t
                if b / math.sqrt(t) >= c:
                    hits[k] += 1
                    break
        for got, exact in zip(hits, d.crossing(drift), strict=True):
            self.assertLess(abs(got / reps - exact), 4 * math.sqrt(exact * (1 - exact) / reps) + 1e-4)

    def test_executed_design_has_its_error_rates(self):
        from seqinfer.group_sequential import group_sequential_design
        from seqinfer.procedures import GroupSequentialTest

        d = group_sequential_design(0.025, self.FRACTIONS)
        n_max = d.max_sample_size(effect=0.5, sigma=1.0, power=0.9)
        self.assertEqual(n_max, math.ceil((d.drift_for_power(0.9) / 0.5) ** 2))
        proc = GroupSequentialTest.from_design(d, n_max, sigma=1.0)
        reps = 4000
        for theta, target in ((0.0, 0.025), (0.5, 0.9)):
            rate = rejection_rate(proc, lambda r, theta=theta: (lambda: r.gauss(theta, 1.0)), reps, n_max, 23)
            self.assertLess(abs(rate - target), 3 * math.sqrt(target * (1 - target) / reps), msg=f"theta={theta}")

    def test_looks_only_at_the_planned_analyses(self):
        from seqinfer.procedures import GroupSequentialTest

        proc = GroupSequentialTest(1.0, analyses=[2, 4], bounds=[3.0, 1.0])
        state, outs = proc.initial_state(), []
        for v in (5.0, 5.0, -5.0, -5.0):
            state, out = proc.step(state, {"x": v}, None)
            outs.append(out)
        self.assertEqual([o.analysis for o in outs], [None, 1, None, 2])
        self.assertEqual(outs[1].decision, REJECT_H0)  # z = 5 sqrt(2) >= 3

    def test_invalid_designs_are_refused(self):
        from seqinfer.group_sequential import group_sequential_design
        from seqinfer.procedures import GroupSequentialTest

        for fractions in ((0.5, 0.4, 1.0), (0.5, 0.9)):
            with self.assertRaises(ValueError):
                group_sequential_design(0.025, fractions)
        with self.assertRaises(ValueError):
            group_sequential_design(0.025, (0.5, 1.0), spending=lambda t: 0.5 * t)  # does not spend alpha
        with self.assertRaises(ValueError):
            GroupSequentialTest(1.0, analyses=[10, 5], bounds=[2.0, 2.0])


class StudentT(unittest.TestCase):
    def test_distribution_function_against_known_quantiles(self):
        from seqinfer.distributions import t_cdf, t_ppf

        known = ((0.975, 10, 2.228138851986274), (0.975, 1, 12.706204736174698), (0.9, 2, 1.885618083164127))
        for p, df, quantile in known:
            self.assertAlmostEqual(t_ppf(p, df), quantile, places=10)
            self.assertAlmostEqual(t_cdf(quantile, df), p, places=13)
        self.assertEqual(t_cdf(0.0, 7), 0.5)
        self.assertAlmostEqual(t_cdf(-1.3, 4) + t_cdf(1.3, 4), 1.0, places=14)

    def test_large_degrees_of_freedom_approach_the_normal(self):
        from statistics import NormalDist

        from seqinfer.distributions import t_cdf

        self.assertAlmostEqual(t_cdf(1.96, 1e7), NormalDist().cdf(1.96), places=6)


class TMixture(unittest.TestCase):
    def test_bayes_factor_matches_direct_integration(self):
        from seqinfer.procedures import TMixtureSPRT

        proc = TMixtureSPRT(0.0, effect=0.7)
        xs = [0.9, -0.4, 2.1, 1.3, 0.2, 1.7]
        state = proc.initial_state()
        for v in xs:
            state, out = proc.step(state, {"x": v}, None)

        def lik(mu, sig):
            return math.exp(sum(-0.5 * ((x - mu) / sig) ** 2 for x in xs)) / sig ** len(xs)

        ls = [math.log(1e-3) + i * math.log(1e6) / 800 for i in range(801)]
        ds = [-5.6 + i * 11.2 / 400 for i in range(401)]
        dl, dd = ls[1] - ls[0], ds[1] - ds[0]
        h0 = sum(lik(0.0, math.exp(ls_)) for ls_ in ls) * dl
        h1 = sum(
            lik(d * math.exp(ls_), math.exp(ls_)) * math.exp(-0.5 * (d / 0.7) ** 2) / (0.7 * math.sqrt(2 * math.pi))
            for ls_ in ls for d in ds
        ) * dd * dl
        self.assertAlmostEqual(out.log_bf, math.log(h1 / h0), places=4)

    def test_valid_whatever_the_variance(self):
        from seqinfer.procedures import TMixtureSPRT

        reps = 800
        rates = [
            rejection_rate(TMixtureSPRT(3.0), lambda r, s=sigma: (lambda: r.gauss(3.0, s)), reps, 300, 25)
            for sigma in (0.01, 1.0, 40.0)
        ]
        self.assertEqual(len(set(rates)), 1)  # scale invariance: identical decisions on rescaled data
        self.assertLessEqual(rates[0], upper_bound(0.05, reps))

    def test_power_and_confidence_sequence(self):
        from seqinfer.procedures import TMixtureSPRT

        self.assertEqual(rejection_rate(TMixtureSPRT(0.0), lambda r: (lambda: r.gauss(2.1, 7.0)), 100, 2000, 26), 1.0)
        proc = TMixtureSPRT(0.0, alpha=0.1, stop_on_reject=False)
        rnd, reps, misses = random.Random(27), 800, 0
        for _ in range(reps):
            state = proc.initial_state()
            for _ in range(300):
                state, out = proc.step(state, {"x": rnd.gauss(2.0, 3.0)}, None)
                if out.lower is not None and not out.lower <= 2.0 <= out.upper:
                    misses += 1
                    break
        self.assertLessEqual(misses / reps, upper_bound(0.1, reps))

    def test_interval_is_the_inverted_test(self):
        from seqinfer.procedures import TMixtureSPRT

        proc = TMixtureSPRT(0.0, alpha=0.05, stop_on_reject=False)
        rnd, state = random.Random(28), proc.initial_state()
        for _ in range(200):
            state, out = proc.step(state, {"x": rnd.gauss(0.4, 2.0)}, None)
            if out.lower is None or abs(out.log_bf - math.log(20)) < 1e-9:
                continue
            self.assertEqual(not out.lower <= 0.0 <= out.upper, out.log_bf >= math.log(20))


class GroupSequentialExtensions(unittest.TestCase):
    def test_estimated_variance_keeps_the_type_one_error_approximately(self):
        from seqinfer.group_sequential import group_sequential_design
        from seqinfer.procedures import GroupSequentialTest

        d = group_sequential_design(0.025, (0.2, 0.4, 0.6, 0.8, 1.0))
        proc = GroupSequentialTest.from_design(d, 100, sigma=None)
        self.assertGreater(proc.bound_at(0), d.bounds[0])  # t boundaries are wider
        reps = 3000
        rate = rejection_rate(proc, lambda r: (lambda: r.gauss(5.0, 3.0) - 5.0), reps, 100, 29)
        self.assertLess(abs(rate - 0.025), 3 * math.sqrt(0.025 * 0.975 / reps) + 0.003)

    def test_futility_boundaries(self):
        from seqinfer.group_sequential import group_sequential_design

        d = group_sequential_design(0.025, (1 / 3, 2 / 3, 1.0), futility="obrien-fleming", power=0.9)
        for got, published in zip(d.bounds, (3.7103, 2.5114, 1.9930), strict=True):
            self.assertAlmostEqual(got, published, places=3)
        self.assertEqual(d.futility[-1], d.bounds[-1])  # the boundaries meet at the end
        self.assertAlmostEqual(d.power(d.drift), 0.9, places=6)
        cumulative = 0.0
        for q, spent in zip(d.futility_crossing(d.drift)[:-1], d.beta_spent[:-1], strict=True):
            cumulative += q
            self.assertAlmostEqual(cumulative, spent, places=7)
        self.assertAlmostEqual(sum(d.crossing(0.0, binding=False)), 0.025, places=9)  # non-binding
        self.assertLess(sum(d.crossing(0.0)), 0.025)
        fixed = (1.959964 + 1.281552) ** 2 / 0.25
        self.assertGreater(d.max_sample_size(0.5, 1.0), fixed)  # the price of looking early

    def test_futility_executed_on_a_stream(self):
        from seqinfer.group_sequential import group_sequential_design
        from seqinfer.procedures import ACCEPT_H0, GroupSequentialTest

        d = group_sequential_design(0.025, (1 / 3, 2 / 3, 1.0), futility="obrien-fleming", power=0.9)
        n_max = d.max_sample_size(0.5, 1.0)
        proc = GroupSequentialTest.from_design(d, n_max, sigma=1.0)
        reps = 3000
        power = rejection_rate(proc, lambda r: (lambda: r.gauss(0.5, 1.0)), reps, n_max, 30)
        self.assertLess(abs(power - 0.9), 3 * math.sqrt(0.09 / reps))
        state, out = proc.initial_state(), None
        for _ in range(proc.analyses[0]):
            state, out = proc.step(state, {"x": -1.0}, None)
        self.assertEqual((out.analysis, out.decision), (1, ACCEPT_H0))  # stopped for futility
        with self.assertRaises(ValueError):
            group_sequential_design(0.025, (0.5, 1.0), futility="pocock")  # needs the power


if __name__ == "__main__":
    unittest.main()
