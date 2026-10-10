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
            operating_characteristic(self.design.plan, _Unsupported(), 1.0)


class _Unsupported:
    """A family the design module does not know."""

    def spec(self):
        return {"family": "unsupported"}

    def check(self, theta):
        pass


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


class ProcedureReviewRegressions(unittest.TestCase):
    """Defects found by review of the procedures; each test reproduces one."""

    def test_sprt_refuses_error_rates_that_invert_its_boundaries(self):
        with self.assertRaises(ValueError):
            SPRT(Gaussian(1.0), 0.0, 1.0, alpha=0.7, beta=0.7)

    def test_parameters_must_be_finite(self):
        from seqinfer.procedures import CUSUM, LocalLevelKalman

        for make in (
            lambda: Gaussian(float("nan")),
            lambda: Gaussian(float("inf")),
            lambda: CUSUM(Gaussian(1.0), 0.0, 1.0, threshold=float("nan")),
            lambda: ShiryaevRoberts(Gaussian(1.0), 0.0, 1.0, threshold=float("inf")),
            lambda: LocalLevelKalman(float("nan"), 1.0),
        ):
            with self.assertRaises(ValueError):
                make()

    def test_t_boundaries_for_large_z(self):
        from seqinfer.procedures import GroupSequentialTest

        proc = GroupSequentialTest(None, [5, 10], [8.0, 2.0])
        self.assertAlmostEqual(proc.bound_at(0) / 8333.2793582, 1.0, places=9)  # mpmath
        from seqinfer.distributions import _log_normal_upper, _log_upper_tail

        far = GroupSequentialTest(None, [5, 10], [40.0, 2.0]).bound_at(0)  # its tail underflows: done in logs
        self.assertAlmostEqual(_log_upper_tail(far, 4) / _log_normal_upper(40.0), 1.0, places=12)
        GroupSequentialTest(None, [5, 10], [8.5, 2.0])  # used to raise

    def test_identical_observations_away_from_theta0_are_evidence_against_it(self):
        from seqinfer.procedures import GroupSequentialTest, TMixtureSPRT

        gs = GroupSequentialTest(None, [5, 10], [2.5, 2.0])
        state = gs.initial_state()
        for _ in range(5):
            state, out = gs.step(state, {"x": 1.0}, None)
        self.assertEqual(out.decision, REJECT_H0)
        tm = TMixtureSPRT(0.0)
        state = tm.initial_state()
        for _ in range(30):
            state, out = tm.step(state, {"x": 1.0}, None)
        self.assertEqual(out.decision, REJECT_H0)
        self.assertEqual((out.lower, out.upper), (1.0, 1.0))

    def test_student_t_extremes(self):
        from seqinfer.distributions import _upper_tail, t_cdf, t_ppf

        # references from mpmath at 50 digits
        self.assertAlmostEqual(t_cdf(1e-8, 5) - 0.5, 3.7960669e-9, delta=1e-15)
        self.assertEqual(t_ppf(0.5, 1), 0.0)
        self.assertAlmostEqual(t_cdf(0.5, 1e9), 0.6914624612190029, delta=1e-15)
        self.assertAlmostEqual(t_cdf(-1e155, 1) / 3.1830988618379067e-156, 1.0, places=12)
        self.assertAlmostEqual(t_ppf(1e-300, 1) / -3.1830988618379067e299, 1.0, places=11)
        self.assertAlmostEqual(_upper_tail(1e10, 5) / 9.490167245562362e-50, 1.0, places=12)
        self.assertAlmostEqual(_upper_tail(8.0, 1e5) / 6.286959938128186e-16, 1.0, places=10)


class NumpyBackwardPass(unittest.TestCase):
    """The array version of the backward pass computes the same recursion as the pure-Python one."""

    def setUp(self):
        from seqinfer import design

        if design._np is None:
            self.skipTest("numpy is not installed")
        self.design = design

    def test_same_regions_and_values(self):
        import random

        rnd = random.Random(3)
        cases = [
            (Bernoulli(), 0.3, 0.5, 0.4, 60, None),
            (Poisson(), 2.0, 3.0, 2.45, 30, None),
            (Gaussian(1.0), 0.0, 0.5, 0.25, 40, None),
            (Gaussian(2.0), 1.0, 0.0, 0.5, 30, 0.25),
            (Exponential(), 1.0, 0.5, 0.72, 30, None),
        ]
        for family, t0, t1, ts, horizon, step in cases:
            problem = self.design._Problem(family, t0, t1, ts, horizon, step)
            for _ in range(4):
                u0, u1 = rnd.uniform(0, 8), rnd.uniform(0, 8)
                a, b = problem._solve_python(u0, u1), problem._solve_numpy(u0, u1)
                with self.subTest(family=family.spec(), u=(u0, u1)):
                    self.assertEqual(a.stages, b.stages)
                    for name in ("value", "alpha0", "alpha1", "asn"):
                        x, y = getattr(a, name), getattr(b, name)
                        self.assertLessEqual(abs(x - y), 1e-12 * max(abs(x), abs(y)))

    def test_same_plans(self):
        np = self.design._np
        for args, family in (((0.0, 0.5, 0.05, 0.1, 60), Gaussian(1.0)), ((1.0, 0.5, 0.05, 0.1, 40), Exponential())):
            fast = kiefer_weiss_plan(*args, family=family, cache=False)
            try:
                self.design._np = None
                slow = kiefer_weiss_plan(*args, family=family, cache=False)
            finally:
                self.design._np = np
            self.assertEqual(fast.plan.plan, slow.plan.plan)


class ThirdReviewNumerics(unittest.TestCase):
    """Defects found by the third review; each test reproduces one."""

    def test_far_tails_with_huge_df(self):
        from seqinfer.distributions import _log_upper_tail, t_cdf

        # the continued fraction did not converge here; references from mpmath's 2F1 at 50 digits
        for t, df, ref in (
            (1e10, 1e15, -5756467732460133.0),
            (10**10.5, 1e13, -92103403769777.72),
            (1e11, 1e14, -921034037697635.2),
            (10**11.5, 1e15, -9210340376976200.0),
        ):
            self.assertAlmostEqual(_log_upper_tail(t, df) / ref, 1.0, places=14, msg=f"t={t}, df={df}")
        self.assertEqual(t_cdf(-1e10, 1e15), 0.0)

    def test_fisher_terms_do_not_overflow(self):
        from seqinfer.distributions import _log_normal_upper, _log_upper_tail

        t, df = 1.333521432163324e28, 1.7782794100389227e115  # t^11 overflowed: the log tail was +inf
        self.assertAlmostEqual(_log_upper_tail(t, df) / _log_normal_upper(t), 1.0, places=14)

    def test_sprt_baseline_quoted_in_the_readme(self):
        from seqinfer.design import maximum_expected_n
        from seqinfer.procedures import PlanTest

        # Bernoulli 0.3 vs 0.5: Wald's SPRT with thresholds calibrated to errors 0.05, as a plan on the sum
        up, dn = math.log(5 / 3), math.log(5 / 7)

        def sprt(a, b, horizon=1500):
            rows = [((b - n * dn) / (up - dn), (a - n * dn) / (up - dn)) for n in range(1, horizon)]
            end = -horizon * dn / (up - dn)
            return PlanTest([*rows, (end, end)])

        def errors(a, b):
            plan = sprt(a, b)
            return operating_characteristic(plan, Bernoulli(), 0.3).reject, operating_characteristic(
                plan, Bernoulli(), 0.5
            ).accept

        a, b = 2.7407160438597202, -2.8049585931003094  # found by alternating bisection
        e0, e1 = errors(a, b)
        self.assertLessEqual(max(e0, e1), 0.05)
        self.assertGreater(errors(a - 0.01, b)[0], 0.05)  # neither threshold can come closer to 0
        self.assertGreater(errors(a, b + 0.01)[1], 0.05)
        self.assertAlmostEqual(maximum_expected_n(sprt(a, b), Bernoulli(), 0.3, 0.5)[0], 50.79, places=2)

    def test_continued_fraction_converges(self):
        from seqinfer.distributions import t_cdf

        # |delta - 1| stuck at 2**-53 for tiny df; the iteration limit was too low for large df
        self.assertAlmostEqual((t_cdf(0.01, 1e-10) - 0.5) / 3.800451e-10, 1.0, places=4)
        self.assertEqual(t_cdf(8.912509381e9, 1e13), 1.0)
        for t, df in ((1e74, 1e300), (1e60, 1e250)):
            self.assertEqual(t_cdf(t, df), 1.0)

    def test_quantiles_at_extreme_df(self):
        from seqinfer.distributions import t_isf_log

        # log tails far below -1e100 at df up to the largest float: t is about sqrt(-2 log tail)
        for log_tail, df in ((-1e148, 1e300), (-1e120, 1e250), (-1e150, 1.7e308)):
            self.assertAlmostEqual(t_isf_log(log_tail, df) / math.sqrt(-2.0 * log_tail), 1.0, places=12)

    def test_huge_df_asymptotic_has_its_second_term(self):
        from seqinfer.distributions import _log_upper_tail

        # 120-digit quadrature; without the 1/t^2 term the log was off by 6e-7 and stepped up at the switch
        for t, df, ref in ((1536.0027, 2e15, -1179660.4023854834), (2500.0, 1e17, -3125008.742887048)):
            self.assertLess(abs(_log_upper_tail(t, df) - ref), 1e-8)

    def test_neyman_pearson_bound_keeps_tiny_errors(self):
        from seqinfer.design import _check_attainable, _neyman_pearson_error

        # exact values from mpmath; 1 - power used to leave rounding of 1e-13 and refuse feasible targets
        for args, ref in (
            ((0.5, 0.1, 0.05, 200), 4.80956242070674e-36),
            ((0.3, 0.6, 0.05, 300), 2.1246751364944591e-19),
        ):
            self.assertAlmostEqual(_neyman_pearson_error(Bernoulli(), *args) / ref, 1.0, places=9)
        plan = kiefer_weiss_plan(0.5, 0.1, 0.05, 1e-14, 200)
        self.assertLessEqual(plan.at_theta1.accept, 1e-14)
        # the normal bound with alpha0 below the spacing of floats near 1
        self.assertAlmostEqual(_neyman_pearson_error(Gaussian(), 0.0, 1.0, 1.6e-16, 100), 0.03329288349842, places=10)
        _check_attainable(Gaussian(), 0.0, 1.0, 1.6e-16, 0.035, 100)
        self.assertGreater(_neyman_pearson_error(Gaussian(), 0.0, 1.0, 5e-17, 200), 0.0)

    def test_parameters_that_cannot_work_are_refused(self):
        from seqinfer.group_sequential import group_sequential_design
        from seqinfer.procedures import TMixtureSPRT

        with self.assertRaises(ValueError):
            TMixtureSPRT(effect=1e160)  # every step overflowed
        with self.assertRaises(ValueError):
            group_sequential_design(0.9, [1.0])  # silently spent 0.5
        group_sequential_design(0.9, [0.5, 1.0], two_sided=True)

    def test_zero_argument(self):
        from seqinfer.distributions import _log_upper_tail, _upper_tail

        for df in (1e-3, 0.5, 30.0, 1e20):
            self.assertEqual(_upper_tail(0.0, df), 0.5)
            self.assertEqual(_log_upper_tail(0.0, df), math.log(0.5))


class SecondReviewNumerics(unittest.TestCase):
    """Defects found by the second review and by fuzzing; each test reproduces one."""

    def test_student_t_across_the_range(self):
        from seqinfer.distributions import _upper_tail, t_cdf, t_ppf

        # (t, df, tail) with tails from mpmath at 60 digits
        for t, df, ref in (
            (3.0, 1e10, 0.0013498980349539808),  # large df: Fisher's expansion
            (6.0, 1e10, 9.8658767875884845e-10),
            (3.0, 1e8, 0.0013498983640187472),
            (1.0, 1e4, 0.15866735216521456),  # near the old Stirling switch
            (1.7e308, 0.1, 6.2731701753179726e-32),  # t / sqrt(df) overflows
        ):
            self.assertAlmostEqual(_upper_tail(t, df) / ref, 1.0, places=12, msg=f"t={t}, df={df}")
        self.assertAlmostEqual(t_cdf(1.0, math.inf), 0.8413447460685429, places=15)
        self.assertAlmostEqual(t_ppf(0.05, 1e200), -1.6448536269514724, places=13)
        self.assertEqual(t_ppf(1e-300, 0.1), -math.inf)  # the quantile is about -1e2990

    def test_fisher_expansion_at_large_df(self):
        from seqinfer.distributions import _fisher_correction, _log_normal_upper, _log_upper_tail

        # log P(T > t), 120-digit references; the expansion once had wrong second-order coefficients
        for t, df, ref in (
            (3.0, 1e6, -6.607701598412306),
            (2.0, 1e5, -3.7831250047718967),
            (37.0, 1e10, -689.0305386544447),
            (39.0, 1e10, -765.0830986523333),
            (60.0, 1e9, -1805.0103188885957),  # log x once lost digits that df/2 = 5e8 then multiplied
            (37.0, 1e16, -689.0305855768437),  # the normal limit was once taken for every t beyond df = 1e15
            (3000.0, 1e16, -4500008.923281211),
            (1e155, 1e16, -3.384800086701247e18),  # was -inf
            (1e50, 1e200, -5.000000000000001e99),
            (1e300, 1e16, -6.723548471542614e18),  # t / sqrt(df) squared once overflowed
        ):
            self.assertAlmostEqual(_log_upper_tail(t, df) / ref, 1.0, places=13, msg=f"t={t}, df={df}")
        for t, df, ref in ((37.0, 1e15, -689.0305855764212), (37.0, 2e15, -689.030585576656)):
            self.assertAlmostEqual(_log_upper_tail(t, df) / ref, 1.0, places=14, msg=f"t={t}, df={df}")
        self.assertAlmostEqual(_fisher_correction(3.0, 1e12) * 1e12, 7.5 + 49.125e-12, places=12)
        self.assertEqual(_log_normal_upper(1e100), -5e199)  # no overflow in the Mills series
        self.assertEqual(_log_normal_upper(1e200), -math.inf)

    def test_rescaled_data_fail_loudly_not_wrongly(self):
        from seqinfer import NumericalError
        from seqinfer.procedures import GroupSequentialTest, TMixtureSPRT

        tiny = [v * 1e-200 for v in (1, -1, 2, -2, 1, -1, 2, -2, 1, -1.1)]
        huge = [v * 1e160 for v in range(10, 20)]
        for proc in (TMixtureSPRT(0.0, stop_on_reject=False), GroupSequentialTest(None, [10], [2.32], two_sided=True)):
            for data in (tiny, huge):
                with self.subTest(proc=proc.name, scale=data[0]), self.assertRaises(NumericalError):
                    state = proc.initial_state()
                    for v in data:
                        state, _ = proc.step(state, {"x": v}, None)

    def test_overflow_is_a_numerical_error(self):
        from seqinfer import NumericalError, Run
        from seqinfer.procedures import BootstrapParticleFilter, MeanDifference

        for make, x, seed in (
            (lambda: NormalMixtureSPRT(1.0, 0.5), {"x": 1e200}, None),
            (lambda: BootstrapParticleFilter(0.1, 1.0, particles=8), {"y": 1e200}, 1),
            (MeanDifference, {"a": 0.0, "b": 1.7e308}, None),
        ):
            run = Run(make(), seed=seed)
            if isinstance(run.procedure, MeanDifference):
                run.step({"a": 0.0, "b": -1e308})
            with self.subTest(procedure=run.procedure.name), self.assertRaises(NumericalError):
                run.step(x)

    def test_invalid_values_never_reach_a_checkpoint(self):
        import json

        from seqinfer import Difference, Observation, PositionalPair, Run
        from seqinfer.procedures import EMA

        def topo():
            return PositionalPair(["a", "b"]).map(Difference("a", "b"))

        for bad in (float("nan"), float("inf"), 2**60):
            run = Run(EMA(0.5), topology=topo())
            with self.assertLogs("seqinfer", "WARNING"):
                run.offer(Observation("a", bad, seq=0))
            Run.restore(EMA(0.5), json.loads(json.dumps(run.checkpoint())), topology=topo())


class ExponentialDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = kiefer_weiss_plan(1.0, 2.0, 0.05, 0.05, horizon=40, family=Exponential(), step=0.125)

    def test_lattice_keeps_the_mean(self):
        lat = lattice(Exponential(), 2.0, step=0.05)
        mean = sum(((i + lat.offset) * lat.step + lat.shift) * w for i, w in enumerate(lat.weights))
        self.assertAlmostEqual(mean, 0.5, delta=0.5 * 0.05**2)  # midpoint rule: O(h^2)

    def test_error_rates_consistency_and_orientation(self):
        d = self.design
        self.assertFalse(d.plan.reject_high)  # a larger rate means shorter durations
        self.assertLessEqual(d.at_theta0.reject, 0.05)
        self.assertLessEqual(d.at_theta1.accept, 0.05)
        forward = d.at_theta_star.expected_n + d.lambda0 * d.at_theta0.reject + d.lambda1 * d.at_theta1.accept
        self.assertAlmostEqual(d.lagrangian, forward, places=6)

    def test_error_rates_on_real_exponential_data(self):
        rnd, reps = random.Random(31), 4000
        for rate_, target in ((1.0, 0.05), (2.0, 0.95)):
            got, _ = simulate_plan(self.design.plan, lambda r=rate_: rnd.expovariate(r), reps)
            self.assertLess(abs(got - target), 3 * math.sqrt(target * (1 - target) / reps), msg=f"rate={rate_}")


class LeastFavourable(unittest.TestCase):
    def test_the_search_lowers_the_maximum_expected_sample_size(self):
        first = kiefer_weiss_plan(0.2, 0.5, 0.05, 0.1, horizon=60)
        best = kiefer_weiss_plan(0.2, 0.5, 0.05, 0.1, horizon=60, theta_star="least-favourable")
        value, _ = best.maximum_expected_n()
        self.assertLessEqual(value, first.maximum_expected_n()[0] + 1e-9)  # the objective of Kiefer-Weiss
        self.assertLessEqual(best.at_theta0.reject, 0.05)
        self.assertLessEqual(best.at_theta1.accept, 0.1)

    def test_maximum_is_found(self):
        from seqinfer.design import maximum_expected_n

        d = kiefer_weiss_plan(0.2, 0.5, 0.05, 0.1, horizon=60)
        value, where = maximum_expected_n(d.plan, Bernoulli(), 0.2, 0.5)
        grid = max(operating_characteristic(d.plan, Bernoulli(), 0.2 + 0.3 * i / 200).expected_n for i in range(201))
        self.assertGreaterEqual(value, grid - 1e-9)
        self.assertTrue(0.2 < where < 0.5)


class DesignReviewRegressions(unittest.TestCase):
    """Defects found by review of the design modules; each test reproduces one."""

    def test_least_favourable_reports_bad_arguments(self):
        for args in ((0.3, 0.5, 1.5, 0.1, 30), (0.3, 0.5, 0.05, 0.1, 0), (0.3, 0.3, 0.05, 0.1, 30)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "need theta0 != theta1"):
                kiefer_weiss_plan(*args, theta_star="least-favourable")

    def test_impossible_targets_fail_at_once(self):
        import time

        cases = [
            ((0.4905, 0.5701, 0.168, 0.0274, 78), Bernoulli()),  # took 332 s
            ((0.4, 0.5, 0.05, 0.1, 10), Bernoulli()),
            ((2.0, 2.5, 0.05, 0.1, 10), Poisson()),  # 97 s
            ((50.0, 55.0, 0.05, 0.1, 5), Poisson()),  # 172 s
            ((0.0, 0.5, 0.05, 0.1, 5), Gaussian(1.0)),  # over 550 s
        ]
        for args, family in cases:
            for theta_star in (None, "least-favourable"):
                start = time.perf_counter()
                with self.subTest(args=args, ts=theta_star), self.assertRaisesRegex(ValueError, "Neyman-Pearson"):
                    kiefer_weiss_plan(*args, family=family, theta_star=theta_star)
                self.assertLess(time.perf_counter() - start, 2.0)

    def test_neyman_pearson_bound(self):
        from seqinfer.design import _neyman_pearson_error

        # one Bernoulli observation, 0.5 vs 0.7 at level 0.05: reject x = 1 with probability 0.1
        self.assertAlmostEqual(_neyman_pearson_error(Bernoulli(), 0.5, 0.7, 0.05, 1), 0.93, places=14)
        self.assertAlmostEqual(_neyman_pearson_error(Bernoulli(), 0.5, 0.3, 0.05, 1), 0.93, places=14)
        self.assertIsNone(_neyman_pearson_error(Exponential(), 1.0, 0.5, 0.05, 10))
        # every plan the module returns respects the bound
        for family, args in ((Bernoulli(), (0.3, 0.5, 0.05, 0.1, 60)), (Poisson(), (3.0, 2.0, 0.05, 0.1, 30))):
            d = kiefer_weiss_plan(*args, family=family)
            bound = _neyman_pearson_error(family, args[0], args[1], d.at_theta0.reject, d.plan.horizon)
            self.assertGreaterEqual(d.at_theta1.accept, bound * (1 - 1e-9))

    def test_feasible_problems_near_the_smallest_horizon_get_a_plan(self):
        for args, family in (
            ((0.05, 0.5, 0.2, 0.2, 3), Bernoulli()),
            ((0.2, 0.6, 0.2, 0.2, 12), Bernoulli()),
            ((0.3, 0.5, 0.01, 0.05, 94), Bernoulli()),
            ((1.0, 1.5, 0.05, 0.05, 56), Poisson()),
            ((3.0, 5.0, 0.02, 0.1, 12), Poisson()),
        ):
            with self.subTest(args=args, family=family.spec()["family"]):
                d = kiefer_weiss_plan(*args, family=family)
                self.assertLessEqual(d.plan.horizon, args[4])
                self.assertLessEqual(d.at_theta0.reject, args[2])
                self.assertLessEqual(d.at_theta1.accept, args[3])
        with self.assertRaises(ValueError):
            kiefer_weiss_plan(0.2, 0.5, 0.001, 0.001, 5)

    def test_least_favourable_is_never_worse_than_the_first_order_point(self):
        first = kiefer_weiss_plan(0.3, 0.5, 0.05, 0.05, 100)
        best = kiefer_weiss_plan(0.3, 0.5, 0.05, 0.05, 100, theta_star="least-favourable")
        self.assertLessEqual(best.maximum_expected_n(48)[0], first.maximum_expected_n(48)[0] + 1e-9)

    def test_poisson_lattice_for_large_rates(self):
        for rate in (730.0, 745.0, 800.0, 2000.0):
            self.assertAlmostEqual(sum(lattice(Poisson(), rate).weights), 1.0, places=10)
        plan = PlanTest([(1000, 1000)], contract=Poisson().contract("x"))
        self.assertAlmostEqual(operating_characteristic(plan, Poisson(), 800.0).accept, 1.0, places=10)

    def test_group_designs_with_an_infinite_boundary(self):
        from seqinfer.group_sequential import group_sequential_design

        d = group_sequential_design(0.025, (0.5, 1.0), spending=lambda t: 0.0 if t < 1 else 0.025)
        self.assertEqual(d.bounds[0], math.inf)
        self.assertAlmostEqual(sum(d.crossing(0.0)), 0.025, places=12)
        self.assertGreater(d.power(2.0), 0.025)

    def test_power_must_lie_between_alpha_and_one(self):
        from seqinfer.group_sequential import group_sequential_design

        d = group_sequential_design(0.025, (0.5, 1.0))
        for power in (1.5, 1.0, 0.01):
            with self.subTest(power=power), self.assertRaises(ValueError):
                d.drift_for_power(power)
        with self.assertRaises(ValueError):
            group_sequential_design(0.025, (0.5, 1.0), futility="pocock", power=0.02)

    def test_crossing_probabilities_are_never_negative(self):
        from seqinfer.group_sequential import group_sequential_design

        d = group_sequential_design(0.025, (0.5, 1.0))
        for drift in (20.0, 40.0, 60.0):
            self.assertTrue(all(p >= 0.0 for p in d.crossing(drift)))


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
