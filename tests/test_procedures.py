# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Statistical and numerical behavior of the reference procedures."""
import json
import math
import random
import statistics
import unittest

from seqinfer import Run, SplitMix64, equivalent, run_sync, trajectory
from seqinfer.numerics import logsumexp, neumaier_add
from seqinfer.procedures import (
    ACCEPT_H0,
    CUSUM,
    REJECT_H0,
    SPRT,
    Bernoulli,
    BootstrapParticleFilter,
    Gaussian,
    LocalLevelKalman,
    MeanDifference,
    PlanTest,
    TwoSPRT,
    kiefer_weiss_point,
)
from seqinfer.sources import gaussian

from .helpers import cases, chain_ema_cusum


def decide(proc, draw, limit=100_000):
    state = proc.initial_state()
    for _ in range(limit):
        state, _ = proc.step(state, {"x": draw()}, None)
        if proc.is_terminal(state):
            return state
    raise AssertionError("no decision")


class RandomState(unittest.TestCase):
    def test_reference_vector(self):
        rng = SplitMix64(0)
        self.assertEqual(rng.next_u64(), 0xE220A8397B1DCDAF)

    def test_state_is_one_integer_and_roles_are_separate(self):
        a, b = SplitMix64.for_role(1, "statistical"), SplitMix64.for_role(1, "simulation/x")
        self.assertNotEqual(a.state, b.state)
        c = SplitMix64(a.state)
        self.assertEqual([a.normal() for _ in range(5)], [c.normal() for _ in range(5)])

    def test_moments(self):
        rng = SplitMix64(42)
        u = [rng.random() for _ in range(20_000)]
        z = [rng.normal(1.0, 2.0) for _ in range(20_000)]
        self.assertTrue(all(0.0 <= v < 1.0 for v in u))
        self.assertAlmostEqual(statistics.fmean(u), 0.5, delta=0.01)
        self.assertAlmostEqual(statistics.fmean(z), 1.0, delta=0.05)
        self.assertAlmostEqual(statistics.pstdev(z), 2.0, delta=0.05)


class Numerics(unittest.TestCase):
    def test_compensated_sum(self):
        total = comp = 0.0
        for x in (1e16, 1.0, -1e16):
            total, comp = neumaier_add(total, comp, x)
        self.assertEqual(total + comp, 1.0)
        self.assertEqual((1e16 + 1.0) - 1e16, 0.0)  # what plain accumulation gives

    def test_logsumexp(self):
        self.assertAlmostEqual(logsumexp([-1000.0, -1000.0]), -1000.0 + math.log(2.0))
        self.assertEqual(logsumexp([-math.inf, -math.inf]), -math.inf)

    def test_equivalence_distinguishes_numbers_from_decisions(self):
        a = {"llr": 1.0, "n": 3, "decision": None}
        self.assertTrue(equivalent(a, {"llr": 1.0 + 1e-12, "n": 3, "decision": None}, rel_tol=1e-9))
        self.assertFalse(equivalent(a, {"llr": 1.0 + 1e-12, "n": 3, "decision": None}))
        self.assertFalse(equivalent(a, {"llr": 1.0, "n": 3, "decision": REJECT_H0}, rel_tol=1.0))
        self.assertFalse(equivalent(a, {"llr": 1.0, "n": 4, "decision": None}, rel_tol=1.0))


class StateEncoding(unittest.TestCase):
    def test_every_state_survives_json_exactly(self):
        for label, make_proc, make_topo, make_delivery, observations, seed in cases():
            with self.subTest(case=label):
                proc = make_proc()
                run = Run(proc, topology=make_topo(), delivery=make_delivery(), seed=seed)
                run_sync(run, observations[:30], stop_on_terminal=False)
                wire = json.dumps(proc.encode_state(run.state))
                self.assertEqual(proc.decode_state(json.loads(wire)), run.state)


class WaldSPRT(unittest.TestCase):
    def test_thresholds_and_decisions(self):
        proc = SPRT(Gaussian(1.0), 0.0, 1.0, alpha=0.05, beta=0.1)
        self.assertAlmostEqual(proc.upper, math.log(0.9 / 0.05))
        self.assertAlmostEqual(proc.lower, math.log(0.1 / 0.95))
        up = trajectory(proc, [{"x": 1.0}] * 20)
        self.assertEqual(up[-1][0].decision, REJECT_H0)
        self.assertEqual(len(up), math.ceil(proc.upper / 0.5))
        down = trajectory(proc, [{"x": 0.0}] * 20)
        self.assertEqual(down[-1][0].decision, ACCEPT_H0)

    def test_bernoulli(self):
        proc = SPRT(Bernoulli(), 0.2, 0.5)
        self.assertEqual(trajectory(proc, [{"x": 1}] * 20)[-1][0].decision, REJECT_H0)
        self.assertEqual(trajectory(proc, [{"x": 0}] * 20)[-1][0].decision, ACCEPT_H0)


class KieferWeiss(unittest.TestCase):
    def test_intermediate_point(self):
        g = Gaussian(2.0)
        self.assertAlmostEqual(kiefer_weiss_point(g, 0.0, 1.0, 3.0, 3.0), 0.5)
        a0, a1 = math.log(100), math.log(10)
        closed_form = math.sqrt(a0) / (math.sqrt(a0) + math.sqrt(a1))
        self.assertAlmostEqual(kiefer_weiss_point(g, 0.0, 1.0, a0, a1), closed_form)
        self.assertAlmostEqual(kiefer_weiss_point(g, 1.0, 0.0, a0, a1), 1.0 - closed_form)
        b = Bernoulli()
        p = kiefer_weiss_point(b, 0.2, 0.5, a0, a1)
        self.assertAlmostEqual(a0 / b.kl(p, 0.2), a1 / b.kl(p, 0.5))

    def test_the_gaussian_test_is_closed(self):
        proc = TwoSPRT(Gaussian(1.0), 0.0, 0.5)
        n_max = proc.max_sample_size()
        self.assertEqual(n_max, 96)
        # data sitting exactly at theta* is the slowest case and attains the bound
        self.assertEqual(decide(proc, lambda: proc.theta_star).n, n_max)
        rnd = random.Random(0)
        for theta in (-0.5, 0.0, 0.25, 0.5, 1.0):
            for _ in range(300):
                self.assertLessEqual(decide(proc, lambda theta=theta: rnd.gauss(theta, 1.0)).n, n_max)
        self.assertIsNone(TwoSPRT(Bernoulli(), 0.2, 0.5).max_sample_size())

    def test_error_probabilities_respect_the_bound(self):
        proc = TwoSPRT(Gaussian(1.0), 0.0, 0.5, alpha0=0.05, alpha1=0.05)
        rnd = random.Random(1)
        reps = 3000
        false_reject = sum(decide(proc, lambda: rnd.gauss(0.0, 1.0)).decision == REJECT_H0 for _ in range(reps))
        false_accept = sum(decide(proc, lambda: rnd.gauss(0.5, 1.0)).decision == ACCEPT_H0 for _ in range(reps))
        self.assertLessEqual(false_reject / reps, 0.05)
        self.assertLessEqual(false_accept / reps, 0.05)

    def test_calibrated_thresholds_are_part_of_the_identity(self):
        default = TwoSPRT(Gaussian(1.0), 0.0, 0.5)
        tuned = TwoSPRT(Gaussian(1.0), 0.0, 0.5, thresholds=(2.4562, 2.4562))
        self.assertAlmostEqual(default.config()["a0"], math.log(20))
        self.assertEqual(tuned.max_sample_size(), 79)
        self.assertNotEqual(default.identity(), tuned.identity())

    def test_bernoulli_decisions(self):
        proc = TwoSPRT(Bernoulli(), 0.2, 0.5)
        self.assertEqual(decide(proc, lambda: 1).decision, REJECT_H0)
        self.assertEqual(decide(proc, lambda: 0).decision, ACCEPT_H0)


class SamplingPlans(unittest.TestCase):
    def test_a_plan_reproduces_the_test_it_was_derived_from(self):
        for alphas in ((0.05, 0.05), (0.01, 0.1)):
            test = TwoSPRT(Gaussian(1.3), 0.0, 0.5, *alphas)
            plan = PlanTest(test.as_plan())
            self.assertEqual(plan.horizon, test.max_sample_size())
            rnd = random.Random(5)
            for theta in (-0.2, 0.0, test.theta_star, 0.5, 0.8):
                for _ in range(400):
                    xs = [rnd.gauss(theta, 1.3) for _ in range(plan.horizon)]
                    a = trajectory(test, [{"x": x} for x in xs])[-1][0]
                    b = trajectory(plan, [{"x": x} for x in xs])[-1][0]
                    self.assertEqual((a.n, a.decision), (b.n, b.decision))

    def test_a_curtailed_binomial_plan_is_exact(self):
        """Reject H0 iff at least 3 successes in 5 trials, stopping as soon as that is settled."""
        proc = PlanTest(
            [(None, 3), (None, 3), (0, 3), (1, 3), (2, 2)],
            contract=Bernoulli().contract("x"),
            label="curtailed 3-of-5",
        )
        self.assertEqual(proc.horizon, 5)
        stopped_early = 0
        for bits in range(32):
            xs = [(bits >> i) & 1 for i in range(5)]
            final = trajectory(proc, [{"x": x} for x in xs])[-1][0]
            self.assertEqual(final.decision, REJECT_H0 if sum(xs) >= 3 else ACCEPT_H0)
            stopped_early += final.n < 5
        self.assertEqual(stopped_early, 20)

    def test_direction_and_validation(self):
        flipped = PlanTest([(0, 0)], reject_high=False)
        self.assertEqual(trajectory(flipped, [{"x": 1.0}])[-1][0].decision, ACCEPT_H0)
        self.assertEqual(trajectory(flipped, [{"x": -1.0}])[-1][0].decision, REJECT_H0)
        for bad in ([], [(0, 1)], [(None, 1)], [(0, 1, 2)], [(float("-inf"), 0), (0, 0)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PlanTest(bad)
        with self.assertRaises(NotImplementedError):
            TwoSPRT(Bernoulli(), 0.2, 0.5).as_plan()


class ChangeDetection(unittest.TestCase):
    def test_cusum(self):
        proc = CUSUM(Gaussian(1.0), 0.0, 1.0, threshold=4.0)
        quiet = trajectory(proc, [{"x": 0.0}] * 100)
        self.assertEqual(len(quiet), 100)
        self.assertEqual(quiet[-1][0].w, 0.0)
        shifted = trajectory(proc, [{"x": 0.0}] * 50 + [{"x": 1.0}] * 50)
        self.assertEqual(shifted[-1][0].alarm_at, 58)  # eight steps of +0.5 reach 4.0

    def test_chain(self):
        proc = chain_ema_cusum()
        values = [o.value for o in gaussian("x", lambda i: 0.0 if i < 60 else 3.0, 0.5, seed=1, n=120)]
        traj = trajectory(proc, [{"x": v} for v in values])
        self.assertTrue(proc.is_terminal(traj[-1][0]))
        self.assertGreater(len(traj), 60)
        self.assertLess(len(traj), 70)
        self.assertTrue(traj[-1][1].second.alarm)


class Filtering(unittest.TestCase):
    def test_kalman_closed_form(self):
        proc = LocalLevelKalman(q=0.0, r=1.0, m0=0.0, p0=1.0)
        (s1, o1), (s2, _) = trajectory(proc, [{"y": 1.0}, {"y": 1.0}])
        self.assertEqual((s1.mean, s1.var, o1.innovation), (0.5, 0.5, 1.0))
        self.assertAlmostEqual(s2.mean, 2 / 3)
        self.assertAlmostEqual(s2.var, 1 / 3)

    def test_particle_filter_tracks_the_exact_filter(self):
        ys = [{"y": o.value} for o in gaussian("y", lambda i: math.sin(i / 10), 0.5, seed=4, n=60)]
        exact = trajectory(LocalLevelKalman(q=0.05, r=0.25), ys)
        approx = trajectory(BootstrapParticleFilter(q=0.05, r=0.25, particles=2000), ys, seed=1)
        err = [abs(a[1].mean - e[1].mean) for a, e in zip(approx, exact, strict=True)]
        self.assertLess(max(err), 0.1)
        self.assertLess(statistics.fmean(err), 0.03)

    def test_particle_filter_is_reproducible_from_its_random_state(self):
        ys = [{"y": 0.1 * i} for i in range(20)]
        proc = BootstrapParticleFilter(q=0.05, r=0.25, particles=50)
        self.assertEqual(trajectory(proc, ys, seed=3), trajectory(proc, ys, seed=3))
        self.assertNotEqual(trajectory(proc, ys, seed=3), trajectory(proc, ys, seed=4))


class TwoSample(unittest.TestCase):
    def test_matches_batch_statistics_for_any_interleaving(self):
        rnd = random.Random(2)
        a = [rnd.gauss(1.0, 1.0) for _ in range(40)]
        b = [rnd.gauss(0.0, 2.0) for _ in range(25)]
        proc = MeanDifference()

        def final(history):
            return trajectory(proc, history)[-1]

        state, out = final([{"a": v} for v in a] + [{"b": v} for v in b])
        self.assertAlmostEqual(out.difference, statistics.fmean(a) - statistics.fmean(b))
        se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
        self.assertAlmostEqual(out.std_error, se)

        for _ in range(10):
            ia = ib = 0
            history = []
            while ia < len(a) or ib < len(b):
                if ib >= len(b) or (ia < len(a) and rnd.random() < 0.6):
                    history.append({"a": a[ia]})
                    ia += 1
                else:
                    history.append({"b": b[ib]})
                    ib += 1
            self.assertEqual(final(history)[0], state)  # bitwise: the samples update disjoint state


if __name__ == "__main__":
    unittest.main()


class PortableArithmetic(unittest.TestCase):
    def test_the_inference_path_does_not_use_the_builtin_float_sum(self):
        """sum() of floats changed in Python 3.12; trajectories must not depend on the Python version."""
        import ast
        from pathlib import Path

        import seqinfer

        root = Path(seqinfer.__file__).parent
        files = [*sorted((root / "procedures").glob("*.py")), *(root / n for n in ("numerics.py", "core.py", "run.py"))]
        offenders = []
        for path in files:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum":
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], "use numerics.ordered_sum (or neumaier_add) for floats")

    def test_ordered_sum_is_left_to_right(self):
        from seqinfer.numerics import ordered_sum

        values = [1e16, 1.0, -1e16, 1.0]
        self.assertEqual(ordered_sum(values), ((1e16 + 1.0) - 1e16) + 1.0)
