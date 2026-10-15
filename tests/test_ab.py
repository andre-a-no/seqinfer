# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Mathematical correctness of the A/B tests.

* The e-values are checked against exact rational computations of the
  same quantities from their closed forms (products of Gamma ratios), and
  the null suprema against brute-force grid maximisation.
* Type I error: Monte Carlo under the null, monitored after every unit;
  the rejection rate must not exceed alpha by more than Monte Carlo error.
* Power, the covariate's variance reduction (1 - rho^2) and the
  confidence sequence's coverage, by simulation.
* e-BH: against its definition by enumeration, and its false discovery
  rate by simulation.
"""
import math
import random
import unittest
from fractions import Fraction

from sequential_analysis import KeyJoin, Run, run_sync
from sequential_analysis._compat import zip_strict
from sequential_analysis.procedures import (
    ABTest,
    BoundedABTest,
    MultipleTests,
    NormalMixtureSPRT,
    SampleRatioMismatch,
    TwoProportionTest,
    e_bh,
)

from .helpers import ab_units


def fold(proc, inputs):
    state, outs = proc.initial_state(), []
    for x in inputs:
        state, out = proc.step(state, x)
        outs.append(out)
    return state, outs


def rising(a, k):
    """Gamma(a + k) / Gamma(a), exactly, for rational a."""
    out = Fraction(1)
    for j in range(k):
        out *= a + j
    return out


def binomial_upper_bound(alpha, runs):
    """alpha plus three standard errors of a Monte Carlo proportion."""
    return alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / runs)


class SampleRatioMismatchTests(unittest.TestCase):
    def test_bayes_factor_is_the_exact_dirichlet_multinomial_ratio(self):
        weights = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
        c = Fraction(12)
        proc = SampleRatioMismatch([float(w) for w in weights], concentration=float(c))
        rnd = random.Random(1)
        arms = [rnd.choice((0, 0, 1, 2, 2)) for _ in range(60)]
        _, outs = fold(proc, [{"arm": k} for k in arms])
        for t in (1, 5, 17, 60):
            counts = [arms[:t].count(k) for k in range(3)]
            exact = Fraction(1)
            for w, k in zip_strict(weights, counts):
                exact *= rising(c * w, k) / w**k
            exact /= rising(c, t)
            tolerance = 1e-12 * max(1.0, abs(math.log(exact)))
            self.assertAlmostEqual(outs[t - 1].log_e_value, math.log(exact), delta=tolerance)
            self.assertAlmostEqual(proc.log_e_value(counts), math.log(exact), delta=1e-9)

    def test_level_under_the_null(self):
        runs, alpha = 400, 0.05
        rejected = 0
        for r in range(runs):
            rnd = random.Random(100 + r)
            proc = SampleRatioMismatch((0.3, 0.7), alpha, concentration=50.0)
            state = proc.initial_state()
            for _ in range(1500):
                state, _ = proc.step(state, {"arm": int(rnd.random() >= 0.3)})
            rejected += state.decision is not None
        self.assertLessEqual(rejected / runs, binomial_upper_bound(alpha, runs))

    def test_detects_a_two_percent_imbalance(self):
        rnd = random.Random(7)
        proc = SampleRatioMismatch((0.5, 0.5), 0.001, concentration=1000.0)
        state = proc.initial_state()
        for t in range(200_000):  # noqa: B007 -- the stopping time is checked below
            state, out = proc.step(state, {"arm": int(rnd.random() < 0.51)})
            if out.decision:
                break
        self.assertEqual(out.decision, "reject_h0")
        self.assertLess(t, 150_000)

    def test_arguments(self):
        for bad in ((1.0,), (0.5, 0.0), (0.5, math.nan)):
            with self.assertRaises(ValueError):
                SampleRatioMismatch(bad)
        with self.assertRaises(ValueError):
            SampleRatioMismatch(concentration=0.0)
        self.assertEqual(SampleRatioMismatch((1, 3)).p, (0.25, 0.75))


class TwoProportionTests(unittest.TestCase):
    @staticmethod
    def data(n, rates, seed, arms=(0, 1)):
        rnd = random.Random(seed)
        out = []
        for _ in range(n):
            a = rnd.choice(arms)
            out.append({"arm": a, "y": int(rnd.random() < rates[a])})
        return out

    @staticmethod
    def log_lik(s, n, p):
        out = 0.0
        if s:
            out += s * math.log(p) if p > 0 else -math.inf
        if n - s:
            out += (n - s) * math.log1p(-p) if p < 1 else -math.inf
        return out

    def test_numerator_is_the_exact_beta_binomial_marginal(self):
        data = self.data(80, {0: 0.3, 1: 0.6}, 1)
        proc = TwoProportionTest(method="universal")
        state, _ = fold(proc, data)
        half = Fraction(1, 2)
        exact = Fraction(1)
        for k in (1, 0):
            ys = [x["y"] for x in data if x["arm"] == k]
            s, n = sum(ys), len(ys)
            exact *= rising(half, s) * rising(half, n - s) / rising(Fraction(1), n)
        self.assertAlmostEqual(state.log_sum + state.comp, math.log(exact), delta=1e-12 * abs(math.log(exact)))

    def test_denominator_is_the_supremum_over_the_null(self):
        grid = [i / 2000 for i in range(2001)]
        for seed, rates in ((2, {0: 0.3, 1: 0.6}), (3, {0: 0.6, 1: 0.3}), (4, {0: 0.0, 1: 0.2})):
            data = self.data(60, rates, seed)
            for alternative in ("two-sided", "greater", "less"):
                proc = TwoProportionTest(alternative=alternative, method="universal")
                state, _ = fold(proc, data)
                (st, sc), (nt, nc) = state.successes, state.trials
                sup = proc.log_null_likelihood(state.successes, state.trials)
                best = -math.inf
                for pt in grid:
                    for pc in (grid if alternative != "two-sided" else [pt]):
                        if (alternative == "greater" and pt > pc) or (alternative == "less" and pt < pc):
                            continue
                        best = max(best, self.log_lik(st, nt, pt) + self.log_lik(sc, nc, pc))
                self.assertGreaterEqual(sup + 1e-9, best, (seed, alternative))
                self.assertLess(sup - best, 0.05, (seed, alternative))  # the grid's resolution

    def test_level_under_the_null_for_unbalanced_and_adaptive_allocation(self):
        runs, alpha = 300, 0.05
        rejected = 0
        for r in range(runs):
            rnd = random.Random(1000 + r)
            proc = TwoProportionTest(alpha, method="universal")
            state = proc.initial_state()
            for _ in range(1000):
                # the allocation depends on the past: still exact
                share = 0.8 if state.successes[0] > state.successes[1] else 0.3
                arm = int(rnd.random() < share)
                state, _ = proc.step(state, {"arm": arm, "y": int(rnd.random() < 0.1)})
            rejected += state.decision is not None
        self.assertLessEqual(rejected / runs, binomial_upper_bound(alpha, runs))

    def test_block_e_value_is_valid_for_every_common_rate(self):
        # E_p[E_block] <= 1 for every rate p, by enumerating every outcome of a block, for any plug-in (a, b)
        pts = [i / 23 for i in range(1, 23)]
        for block in ((1, 1), (2, 1), (1, 3), (3, 2)):
            kt, kc = block
            proc = TwoProportionTest(block=block)
            worst = 0.0
            for a in pts:
                for b in pts:
                    pbar = (kt * a + kc * b) / (kt + kc)
                    for r in [i / 200 for i in range(1, 200)] + [pbar]:
                        expected = 0.0
                        for st in range(kt + 1):
                            for sc in range(kc + 1):
                                ways = math.comb(kt, st) * math.comb(kc, sc)
                                prob = ways * r ** (st + sc) * (1 - r) ** (kt + kc - st - sc)
                                treated, control = [1] * st + [0] * (kt - st), [1] * sc + [0] * (kc - sc)
                                expected += prob * math.exp(proc.block_log_e(a, b, pbar, treated, control))
                        worst = max(worst, expected)
            self.assertLessEqual(worst, 1 + 1e-12, block)
            self.assertGreater(worst, 1 - 1e-12, block)  # attained at r = pbar: no slack left

    def test_level_under_the_null_with_blocks(self):
        runs, alpha = 300, 0.05
        for block, share in (((1, 1), 0.5), ((2, 1), 2 / 3)):
            rejected = 0
            for r in range(runs):
                rnd = random.Random(3000 + r)
                proc = TwoProportionTest(alpha, block=block)
                state = proc.initial_state()
                for _ in range(1500):
                    state, _ = proc.step(state, {"arm": int(rnd.random() < share), "y": int(rnd.random() < 0.1)})
                rejected += state.decision is not None
            self.assertLessEqual(rejected / runs, binomial_upper_bound(alpha, runs), block)

    def test_one_sided_level_at_the_boundary_and_inside_the_null(self):
        runs, alpha = 200, 0.05
        for method in ("grow", "universal"):
            for rates in ({1: 0.2, 0: 0.2}, {1: 0.15, 0: 0.25}):
                rejected = 0
                for r in range(runs):
                    data = self.data(800, rates, 5000 + r)
                    state, _ = fold(TwoProportionTest(alpha, alternative="greater", method=method), data)
                    rejected += state.decision is not None
                self.assertLessEqual(rejected / runs, binomial_upper_bound(alpha, runs), (method, rates))

    def test_grow_needs_fewer_units_than_universal_inference(self):
        def units_to_decide(method, seed):
            rnd = random.Random(seed)
            proc = TwoProportionTest(0.05, method=method)
            state = proc.initial_state()
            for n in range(1, 30_001):
                arm = int(rnd.random() < 0.5)
                state, out = proc.step(state, {"arm": arm, "y": int(rnd.random() < (0.15 if arm else 0.10))})
                if out.decision:
                    return n
            return 30_000

        grow = sorted(units_to_decide("grow", 9000 + r) for r in range(15))
        universal = sorted(units_to_decide("universal", 9000 + r) for r in range(15))
        self.assertLess(grow[7], 0.6 * universal[7])  # medians: about 2,000 against 4,800

    def test_power(self):
        hits = 0
        for r in range(20):
            data = self.data(4000, {1: 0.15, 0: 0.10}, 9000 + r)
            state, _ = fold(TwoProportionTest(0.05, alternative="greater"), data)
            hits += state.decision is not None
        self.assertGreaterEqual(hits, 17)

    def test_blocks_wait_for_their_units(self):
        proc = TwoProportionTest(block=(2, 1))
        state, outs = fold(proc, [{"arm": 1, "y": 1}, {"arm": 1, "y": 0}, {"arm": 1, "y": 1}])
        self.assertEqual((state.pending_treatment, outs[-1].blocks), ((1, 0, 1), 0))
        state, out = proc.step(state, {"arm": 0, "y": 0})
        self.assertEqual((state.pending_treatment, state.pending_control, out.blocks), ((1,), (), 1))
        self.assertEqual(proc.decode_state(proc.encode_state(state)), state)
        broken = {**proc.encode_state(state), "pending_treatment": []}
        with self.assertRaises(ValueError):
            proc.decode_state(broken)

    def test_other_arms_are_ignored(self):
        data = self.data(300, {0: 0.2, 1: 0.3, 2: 0.9}, 11, arms=(0, 1, 2))
        a, _ = fold(TwoProportionTest(arms=(1, 0)), data)
        b, _ = fold(TwoProportionTest(arms=(1, 0)), [x for x in data if x["arm"] != 2])
        self.assertEqual(a, b)

    def test_arguments(self):
        for kwargs in ({"arms": (1, 1)}, {"arms": (1, -1)}, {"prior": (0.0, 1.0)}, {"alternative": "both"},
                       {"arm_input": "y"}, {"method": "lrt"}, {"block": (0, 1)}, {"block": (1, 1.5)}):
            with self.assertRaises(ValueError):
                TwoProportionTest(**kwargs)


class ABTestTests(unittest.TestCase):
    @staticmethod
    def units(n, effect, seed, rho=0.9, allocation=0.5):
        rnd = random.Random(seed)
        out = []
        for _ in range(n):
            x = rnd.gauss(0.0, 1.0)
            t = rnd.random() < allocation
            y = 2.0 + rho * x + math.sqrt(1 - rho * rho) * rnd.gauss(0.0, 1.0) + (effect if t else 0.0)
            out.append({"arm": int(t), "y": y, "x": x})
        return out

    def test_scores_against_a_direct_refit_on_the_past(self):
        units = self.units(120, 0.3, 1, allocation=0.4)
        _, outs = fold(ABTest(0.4, covariate=True, start=2), units)
        scores = []
        for t, u in enumerate(units):
            past = units[:t]
            if not past:
                f = 0.0
            else:
                xs = [Fraction(p["x"]) for p in past]
                ys = [Fraction(p["y"]) for p in past]
                mx, my = sum(xs) / t, sum(ys) / t
                sxx = sum((a - mx) ** 2 for a in xs)
                sxy = sum((a - mx) * (b - my) for a, b in zip_strict(xs, ys))
                f = float(my + (sxy / sxx) * (Fraction(u["x"]) - mx)) if sxx else float(my)
            w = 1 / 0.4 if u["arm"] else -1 / 0.6
            scores.append(w * (u["y"] - f))
            self.assertAlmostEqual(outs[t].estimate, math.fsum(scores) / (t + 1), delta=1e-9)

    def test_estimate_is_unbiased_with_and_without_covariate(self):
        for covariate in (False, True):
            estimates = []
            for r in range(300):
                _state, outs = fold(ABTest(covariate=covariate, start=10), self.units(200, 0.5, 300 + r))
                estimates.append(outs[-1].estimate)
            mean = math.fsum(estimates) / len(estimates)
            sd = math.sqrt(math.fsum((e - mean) ** 2 for e in estimates) / (len(estimates) - 1))
            self.assertLess(abs(mean - 0.5), 4 * sd / math.sqrt(len(estimates)), covariate)

    def test_covariate_removes_rho_squared_of_the_variance(self):
        for rho in (0.5, 0.9):
            _, outs = fold(ABTest(covariate=True), self.units(20_000, 0.2, 21, rho=rho))
            # psi without the covariate has variance 4 (var y + E[...]) ; the
            # covariate removes the share rho^2 of var(y)
            self.assertAlmostEqual(outs[-1].variance_ratio, 1 - rho * rho, delta=0.03)

    def test_coverage_uniformly_over_time(self):
        runs, alpha = 200, 0.05
        missed = 0
        for r in range(runs):
            proc = ABTest(alpha=alpha, covariate=True, start=500, tuned_for=500)
            _state, outs = fold(proc, self.units(2500, 0.3, 7000 + r))
            missed += any(o.lower is not None and not o.lower <= 0.3 <= o.upper for o in outs)
        self.assertLessEqual(missed / runs, binomial_upper_bound(alpha, runs))

    def test_covariate_shrinks_the_interval(self):
        units = self.units(5000, 0.1, 31)
        _, plain = fold(ABTest(), units)
        _, adjusted = fold(ABTest(covariate=True), units)
        ratio = (adjusted[-1].upper - adjusted[-1].lower) / (plain[-1].upper - plain[-1].lower)
        self.assertLess(ratio, 0.55)  # sqrt(1 - 0.81) = 0.44
        self.assertEqual(adjusted[-1].decision, "reject_h0")

    def test_runs_on_joined_event_streams(self):
        proc = ABTest(covariate=True, start=10)
        run = Run(proc, topology=KeyJoin(("arm", "y", "x")), run_id="ab")
        run_sync(run, ab_units(60, seed=3))
        self.assertEqual(run.state.n, 60)

    def test_arguments(self):
        for kwargs in ({"allocation": 1.0}, {"start": 1}, {"arms": (0, 0)}, {"input": "x", "covariate": True},
                       {"tuned_for": 0.5}):
            with self.assertRaises(ValueError):
                ABTest(**kwargs)


class BoundedABTests(unittest.TestCase):
    @staticmethod
    def units(n, seed, allocation=0.5, treat=(2.0, 5.0), control=(2.0, 5.0)):
        """Outcomes in [0, 1] from Beta distributions: skewed, not normal."""
        rnd = random.Random(seed)
        out = []
        for _ in range(n):
            t = rnd.random() < allocation
            a, b = treat if t else control
            out.append({"arm": int(t), "y": rnd.betavariate(a, b)})
        return out

    def test_scores_are_unbiased_for_the_effect(self):
        # E[psi] = E[y(T)] - E[y(C)] exactly, by enumerating a discrete outcome distribution
        for allocation in (0.5, 0.3, 0.8):
            proc = BoundedABTest(0.0, 1.0, allocation)
            values, pt, pc = (0.0, 0.25, 1.0), (0.2, 0.3, 0.5), (0.5, 0.4, 0.1)
            expected = 0.0
            for v, qt, qc in zip(values, pt, pc):
                expected += allocation * qt * proc.score(True, v) + (1 - allocation) * qc * proc.score(False, v)
            effect = sum(v * q for v, q in zip(values, pt)) - sum(v * q for v, q in zip(values, pc))
            self.assertAlmostEqual(expected, effect, delta=1e-12)
            self.assertLessEqual(max(abs(proc.score(t, v)) for t in (True, False) for v in values), proc.bound)

    def test_coverage_uniformly_over_time_on_skewed_data(self):
        runs, alpha = 100, 0.1
        for allocation in (0.5, 0.3):
            truth = 2.0 / 7.0 - 2.0 / 9.0  # Beta(2, 5) against Beta(2, 7)
            missed = 0
            for r in range(runs):
                proc = BoundedABTest(0.0, 1.0, allocation, alpha, grid=200, backend="python")
                _, outs = fold(proc, self.units(600, 100 + r, allocation, treat=(2.0, 5.0), control=(2.0, 7.0)))
                missed += any(not o.lower <= truth <= o.upper for o in outs)
            self.assertLessEqual(missed / runs, binomial_upper_bound(alpha, runs), allocation)

    def test_level_with_equal_means_but_different_distributions(self):
        runs, alpha = 200, 0.1
        rejected = 0
        for r in range(runs):
            rnd = random.Random(700 + r)
            proc = BoundedABTest(0.0, 1.0, 0.5, alpha, grid=50, backend="python")
            state = proc.initial_state()
            for _ in range(800):
                t = rnd.random() < 0.5
                y = float(rnd.random() < 0.5) if t else rnd.random()  # both have mean 1/2
                state, _ = proc.step(state, {"arm": int(t), "y": y})
            rejected += state.test.decision is not None
        self.assertLessEqual(rejected / runs, binomial_upper_bound(alpha, runs))

    def test_detects_a_lift_and_the_interval_shrinks(self):
        proc = BoundedABTest(0.0, 1.0, 0.5, 0.05, grid=400, backend="python")
        _, outs = fold(proc, self.units(4000, 3, treat=(3.0, 5.0), control=(2.0, 5.0)))
        last = outs[-1]
        truth = 3.0 / 8.0 - 2.0 / 7.0
        self.assertEqual(last.decision, "reject_h0")
        self.assertLess(last.lower, truth)
        self.assertGreater(last.upper, truth)
        self.assertLess(last.upper - last.lower, outs[200].upper - outs[200].lower)

    def test_parts_equal_the_procedures_they_are_made_of(self):
        from sequential_analysis.procedures import BettingMeanTest

        units = self.units(300, 4, allocation=0.4)
        proc = BoundedABTest(0.0, 1.0, 0.4, grid=100, backend="python")
        state, outs = fold(proc, units)
        test = BettingMeanTest(0.0, -proc.bound, proc.bound, proc.alpha, stop_on_reject=False, input="psi")
        _, alone = fold(test, [{"psi": proc.score(u["arm"] == 1, u["y"])} for u in units])
        self.assertEqual([o.log_e_value for o in outs], [o.log_wealth for o in alone])
        self.assertEqual(proc.decode_state(proc.encode_state(state)), state)

    def test_other_arms_and_arguments(self):
        rows = [{"arm": 2, "y": 0.5}, *self.units(50, 5), {"arm": 2, "y": 0.1}]
        proc = BoundedABTest(arms=(1, 0), grid=50, backend="python")
        _, outs = fold(proc, rows)
        self.assertEqual(outs[-1], outs[-2])
        self.assertEqual((outs[0].n, outs[0].lower, outs[0].upper), (0, -proc.bound, proc.bound))
        for kwargs in ({"lower": 1.0, "upper": 0.0}, {"allocation": 1.0}, {"effect0": 5.0}, {"arms": (1, 1)}):
            with self.assertRaises(ValueError):
                BoundedABTest(**kwargs)


class EBHTests(unittest.TestCase):
    @staticmethod
    def by_definition(e, alpha):
        k_count = len(e)
        best = set()
        for k in range(1, k_count + 1):
            top = sorted(range(k_count), key=lambda i: (-e[i], i))[:k]
            if all(e[i] >= k_count / (k * alpha) for i in top) and min(e[i] for i in top) >= k_count / (k * alpha):
                best = set(top)
        return sorted(best)

    def test_against_the_definition(self):
        rnd = random.Random(1)
        for _ in range(2000):
            e = [math.exp(rnd.uniform(-2, 6)) for _ in range(rnd.randint(1, 8))]
            self.assertEqual(e_bh([math.log(v) for v in e], 0.1), self.by_definition(e, 0.1))

    def test_false_discovery_rate(self):
        rnd = random.Random(2)
        alpha, mu, runs = 0.2, 3.0, 4000
        fdp = 0.0
        for _ in range(runs):
            truth = [False] * 6 + [True] * 4
            # log of the likelihood ratio N(mu,1)/N(0,1): an e-value under the null
            logs = [mu * (rnd.gauss(mu if h else 0.0, 1.0)) - mu * mu / 2 for h in truth]
            rej = e_bh(logs, alpha)
            if rej:
                fdp += sum(not truth[i] for i in rej) / len(rej)
        self.assertLessEqual(fdp / runs, alpha)

    def test_arguments(self):
        with self.assertRaises(ValueError):
            e_bh([math.nan], 0.1)
        self.assertEqual(e_bh([], 0.1), [])


class MultipleTestsTests(unittest.TestCase):
    def test_equals_the_tests_run_separately(self):
        data = TwoProportionTests.data(1500, {0: 0.1, 1: 0.1, 2: 0.25}, 4, arms=(0, 1, 2))
        multi = MultipleTests.versus_control([1, 2], 0, alpha=0.05)
        _state, outs = fold(multi, data)
        for name, test in zip_strict(multi.names, multi.tests):
            _alone, alone_outs = fold(test, data)
            self.assertEqual(outs[-1].log_e_values[name], alone_outs[-1].log_e_value)
        self.assertEqual(outs[-1].rejected, ("2-vs-0",))

    def test_family_wise_behaviour_under_the_global_null(self):
        runs, alpha = 150, 0.1
        any_rejection = 0
        for r in range(runs):
            data = TwoProportionTests.data(600, {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2}, 300 + r, arms=(0, 1, 2, 3))
            _, outs = fold(MultipleTests.versus_control([1, 2, 3], 0, alpha=alpha), data)
            any_rejection += bool(outs[-1].rejected)
        # under the global null FDR = P(any rejection)
        self.assertLessEqual(any_rejection / runs, binomial_upper_bound(alpha, runs))

    def test_checkpoint_round_trip(self):
        multi = MultipleTests({"conversion": TwoProportionTest(), "srm": SampleRatioMismatch(input="arm")})
        data = TwoProportionTests.data(50, {0: 0.3, 1: 0.5}, 5)
        state, _ = fold(multi, data)
        encoded = multi.encode_state(state)
        self.assertEqual(multi.decode_state(encoded), state)

    def test_contracts_are_intersected(self):
        multi = MultipleTests({"conversion": TwoProportionTest(), "srm": SampleRatioMismatch((1, 1, 1))})
        self.assertEqual((multi.inputs()["arm"].lower, multi.inputs()["arm"].upper), (0, 2))
        with self.assertRaises(ValueError):
            MultipleTests({"a": TwoProportionTest(input="x"), "b": NormalMixtureSPRT(1.0, 1.0)})

    def test_refuses_tests_without_an_e_process(self):
        with self.assertRaises(ValueError):
            MultipleTests({"ab": ABTest()})
        with self.assertRaises(ValueError):
            MultipleTests({"a": NormalMixtureSPRT(1.0, 1.0, input="arm"), "b": TwoProportionTest()})
        with self.assertRaises(ValueError):
            MultipleTests({})


if __name__ == "__main__":
    unittest.main()
