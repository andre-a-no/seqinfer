# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Group sequential designs for any endpoint.

* Boundaries at realised information: the boundaries of analyses already
  done never change; type I error by simulation of the score process
  with analysis times that differ from the plan, with under- and
  over-running; the monitor reproduces the design at planned times.
* GroupSequentialProportions and GroupSequentialLogRank: type I error and
  power by simulation of trials of the sizes the design asks for;
  analyses at realised counts; stratification; contracts and states.
(Agreement with rpact and R's survival package: tests/test_r_reference.py.)
"""
import math
import random
import unittest

from sequential_analysis.errors import ContractViolation, LifecycleError
from sequential_analysis.group_sequential import group_sequential_design
from sequential_analysis.procedures.group_sequential import (
    GroupSequentialLogRank,
    GroupSequentialMonitor,
    GroupSequentialProportions,
    GroupSequentialTwoMeans,
)
from sequential_analysis.procedures.sprt import ACCEPT_H0, REJECT_H0

ENTER, EVENT, CENSOR = 0, 1, 2


def within(test, rate, target, reps, msg=""):
    test.assertLess(abs(rate - target), 3.5 * math.sqrt(target * (1 - target) / reps) + 1e-3, msg)


def simulate_crossing(fractions, bounds, drift, reps, seed):
    """Monte Carlo P(Z_k >= b_k for some k) for the score process with this drift (per unit of fraction)."""
    rnd = random.Random(seed)
    hits = 0
    for _ in range(reps):
        b, previous = 0.0, 0.0
        for t, c in zip(fractions, bounds):
            b += rnd.gauss(drift * (t - previous), math.sqrt(t - previous))
            previous = t
            if b / math.sqrt(t) >= c:
                hits += 1
                break
    return hits / reps


class RealisedInformation(unittest.TestCase):
    PLAN = (0.25, 0.5, 0.75, 1.0)

    def test_earlier_boundaries_do_not_move(self):
        design = group_sequential_design(0.025, self.PLAN, "obrien-fleming")
        first = design.at([0.3, 0.55, 1.0]).bounds
        for later in ([0.3, 0.55, 0.8, 1.0], [0.3, 0.55, 0.9], [0.3, 0.55, 1.2]):
            self.assertEqual(design.at(later).bounds[:2], first[:2])

    def test_type_one_error_with_moved_analyses(self):
        design = group_sequential_design(0.025, self.PLAN, "pocock")
        for fractions in ([0.2, 0.45, 0.85, 1.0], [0.3, 0.6, 0.9], [0.35, 0.7, 1.15], [0.5, 0.6, 0.7, 1.0]):
            moved = design.at(fractions)
            self.assertAlmostEqual(sum(moved.crossing(0.0)), 0.025, delta=1e-8)  # exactly alpha, also off plan
            reps = 20_000
            rate = simulate_crossing(moved.fractions, moved.bounds, 0.0, reps, seed=len(fractions))
            within(self, rate, 0.025, reps, str(fractions))

    def test_monitor_reproduces_the_planned_design(self):
        design = group_sequential_design(0.025, self.PLAN, "obrien-fleming", futility="pocock", power=0.85)
        monitor = GroupSequentialMonitor.from_design(design, 120.0)
        state = monitor.initial_state()
        for k, t in enumerate(self.PLAN):
            state, out = monitor.step(state, {"z": 0.5, "information": 120.0 * t})
            self.assertAlmostEqual(out.bound, design.bounds[k], delta=1e-12)
            self.assertAlmostEqual(out.futility, design.futility[k], delta=1e-9)
            if out.decision:
                break
        self.assertEqual(out.decision, ACCEPT_H0)  # 0.5 is below the second futility boundary
        self.assertEqual(out.analysis, 2)

    def test_monitor_final_analysis(self):
        monitor = GroupSequentialMonitor(0.025, 100.0, max_analyses=3, final_input="final")
        state = monitor.initial_state()
        state, out = monitor.step(state, {"z": 1.0, "information": 40.0, "final": 0})
        self.assertFalse(out.final)
        self.assertLess(out.spent, 0.025)
        state, out = monitor.step(state, {"z": 1.0, "information": 70.0, "final": 1})  # under-running
        self.assertTrue(out.final)
        self.assertEqual(out.spent, 0.025)
        self.assertEqual(out.decision, ACCEPT_H0)
        with self.assertRaises(LifecycleError):
            monitor.step(state, {"z": 1.0, "information": 90.0, "final": 0})
        # over-running: information beyond the maximum makes the analysis final
        monitor = GroupSequentialMonitor(0.025, 100.0, two_sided=True)
        state, out = monitor.step(monitor.initial_state(), {"z": -2.3, "information": 50.0})
        self.assertIsNone(out.decision)
        state, out = monitor.step(state, {"z": -2.3, "information": 130.0})
        self.assertTrue(out.final)
        self.assertEqual(out.decision, REJECT_H0)  # |Z| above the final two-sided boundary
        # the third analysis is final when max_analyses is 3
        monitor = GroupSequentialMonitor(0.025, 100.0, max_analyses=2)
        state, _ = monitor.step(monitor.initial_state(), {"z": 0.0, "information": 30.0})
        _, out = monitor.step(state, {"z": 0.0, "information": 60.0})
        self.assertTrue(out.final)

    def test_monitor_refuses(self):
        monitor = GroupSequentialMonitor(0.025, 100.0)
        state, _ = monitor.step(monitor.initial_state(), {"z": 0.0, "information": 30.0})
        with self.assertRaises(ContractViolation):
            monitor.step(state, {"z": 0.0, "information": 30.0})  # information must grow
        for bad in (dict(alpha=0.0), dict(max_information=0.0), dict(spending=lambda t: t),
                    dict(spending=("power", -1.0)), dict(futility="pocock"), dict(max_analyses=0)):
            args = {"alpha": 0.025, "max_information": 100.0, **bad}
            with self.assertRaises(ValueError, msg=str(bad)):
                GroupSequentialMonitor(**args)
        with self.assertRaises(ValueError):
            monitor.decode_state({"fractions": [0.5, 0.4], "bounds": [3.0, 2.0], "futility": [], "decision": None})


def binary_trial(proc, p_treatment, p_control, rnd):
    state = proc.initial_state()
    n = 0
    while True:
        arm = n % 2  # 1:1, alternating
        p = p_treatment if arm == 1 else p_control
        state, out = proc.step(state, {"arm": arm, "y": int(rnd.random() < p)})
        n += 1
        if out.decision:
            return out


class Proportions(unittest.TestCase):
    def test_level_and_power(self):
        design = group_sequential_design(0.025, [0.25, 0.5, 0.75, 1.0], "obrien-fleming")
        n_max = design.max_sample_size_two_proportions(0.3, 0.4, power=0.8)
        n_max += n_max % 2
        proc = GroupSequentialProportions.from_design(design, n_max)
        rnd, reps = random.Random(5), 1500
        rejected = sum(binary_trial(proc, 0.3, 0.3, rnd).decision == REJECT_H0 for _ in range(reps))
        within(self, rejected / reps, 0.025, reps, "level")
        rejected = sum(binary_trial(proc, 0.4, 0.3, rnd).decision == REJECT_H0 for _ in range(reps))
        within(self, rejected / reps, 0.8, reps, "power")

    def test_statistic_and_directions(self):
        proc = GroupSequentialProportions([4, 8], [40.0, 40.0])
        state = proc.initial_state()
        for arm, y in ((1, 1), (0, 0), (1, 1), (0, 1)):
            state, out = proc.step(state, {"arm": arm, "y": y})
        z = (1.0 - 0.5) / math.sqrt(0.75 * 0.25 * (1 / 2 + 1 / 2))
        self.assertAlmostEqual(out.z, z, places=14)
        self.assertEqual((out.rate_treatment, out.rate_control, out.analysis), (1.0, 0.5, 1))
        less = GroupSequentialProportions([4, 8], [40.0, 40.0], alternative="less")
        self.assertAlmostEqual(less.statistic(state), -z, places=14)
        _, out = proc.step(state, {"arm": 7, "y": 1})  # another arm: ignored
        self.assertEqual((out.n, out.analysis), (4, None))
        empty = GroupSequentialProportions([1], [1.0])
        self.assertEqual(empty.step(empty.initial_state(), {"arm": 1, "y": 1})[1].z, 0.0)

    def test_analyses_at_other_counts(self):
        design = group_sequential_design(0.025, [0.5, 1.0], "pocock")
        planned = GroupSequentialProportions.from_design(design, 200)
        self.assertEqual(planned.analyses, (100, 200))
        moved = GroupSequentialProportions.from_design(design, 200, analyses=[80, 210])
        self.assertEqual(moved.analyses, (80, 210))
        self.assertEqual(moved.bounds, design.at([0.4, 1.05]).bounds)
        with self.assertRaises(ValueError):
            GroupSequentialProportions([10, 5], [2.0, 2.0])
        with self.assertRaises(ValueError):
            GroupSequentialProportions([10], [2.0], alternative="two-sided")


def means_trial(proc, effect, rnd, sd=2.0):
    state, n = proc.initial_state(), 0
    while True:
        arm = n % 2
        state, out = proc.step(state, {"arm": arm, "y": rnd.gauss(effect if arm == 1 else 0.0, sd)})
        n += 1
        if out.decision:
            return out


class TwoMeans(unittest.TestCase):
    def test_level_and_power(self):
        design = group_sequential_design(0.025, [0.25, 0.5, 0.75, 1.0], "obrien-fleming", futility="pocock",
                                         power=0.9)
        n_max = design.max_sample_size_two_means(effect=1.0, sigma=2.0)
        n_max += n_max % 2
        rnd, reps = random.Random(17), 1500
        for sigma in (2.0, None):  # known, and estimated (t boundaries)
            proc = GroupSequentialTwoMeans.from_design(design, n_max, sigma)
            level = sum(means_trial(proc, 0.0, rnd).decision == REJECT_H0 for _ in range(reps)) / reps
            within(self, level, 0.025, reps, f"level, sigma={sigma}")
            power = sum(means_trial(proc, 1.0, rnd).decision == REJECT_H0 for _ in range(reps)) / reps
            within(self, power, 0.9, reps, f"power, sigma={sigma}")

    def test_statistics(self):
        known = GroupSequentialTwoMeans(2.0, [4], [40.0])
        estimated = GroupSequentialTwoMeans(None, [4], [40.0], alternative="less")
        data = ((1, 3.0), (0, 1.0), (1, 5.0), (0, 2.0))
        for proc in (known, estimated):
            state = proc.initial_state()
            for arm, y in data:
                state, out = proc.step(state, {"arm": arm, "y": y})
            if proc is known:
                self.assertAlmostEqual(out.z, 2.5 / (2.0 * math.sqrt(1.0)), places=14)
            else:  # pooled sd sqrt((2 + 0.5) / 2); "less" reverses the sign
                self.assertAlmostEqual(out.z, -2.5 / (math.sqrt(1.25) * 1.0), places=14)
        # t boundaries: the same one-sided level with n - 2 degrees of freedom
        proc = GroupSequentialTwoMeans(None, [12, 24], [2.5, 2.0])
        from sequential_analysis.distributions import t_cdf

        self.assertAlmostEqual(1 - t_cdf(proc.bound_at(0), 10), 0.5 * math.erfc(2.5 / math.sqrt(2)), delta=1e-14)
        with self.assertRaises(ValueError):
            GroupSequentialTwoMeans(None, [2, 4], [2.5, 2.0])  # no variance at two units
        constant = GroupSequentialTwoMeans(None, [4], [3.0])
        state = constant.initial_state()
        for arm, y in ((1, 1.0), (0, 0.0), (1, 1.0), (0, 0.0)):
            state, out = constant.step(state, {"arm": arm, "y": y})
        self.assertEqual((out.z, out.decision), (None, REJECT_H0))  # an infinite t: a certain difference


def survival_trial(proc, hazard_ratio, rnd, n, *, strata_hazards=(0.05,), entry_span=12.0):
    happenings = []
    for i in range(n):
        arm, stratum = i % 2, (i // 2) % len(strata_hazards)
        entry = rnd.uniform(0.0, entry_span)
        rate = strata_hazards[stratum] * (hazard_ratio if arm == 1 else 1.0)
        event = entry + rnd.expovariate(rate)
        dropout = entry + rnd.expovariate(0.01)
        end = min(event, dropout, 48.0)
        happenings += [(entry, arm, ENTER, stratum), (end, arm, EVENT if end == event else CENSOR, stratum)]
    happenings.sort()
    state, out = proc.initial_state(), None
    for _, arm, kind, stratum in happenings:
        state, out = proc.step(state, {"arm": arm, "kind": kind, "stratum": stratum})
        if out.decision:
            return out
    return out


class LogRank(unittest.TestCase):
    def test_level_and_power(self):
        design = group_sequential_design(0.025, [1 / 3, 2 / 3, 1.0], "obrien-fleming")
        d_max = design.max_events(0.7, power=0.8)
        proc = GroupSequentialLogRank.from_design(design, d_max)
        rnd, reps = random.Random(9), 700
        n = 2 * math.ceil(1.6 * d_max / 2)  # enough patients to reach d_max events
        outs = [survival_trial(proc, 1.0, rnd, n) for _ in range(reps)]
        self.assertTrue(all(o.decision for o in outs))
        within(self, sum(o.decision == REJECT_H0 for o in outs) / reps, 0.025, reps, "level")
        outs = [survival_trial(proc, 0.7, rnd, n) for _ in range(reps)]
        within(self, sum(o.decision == REJECT_H0 for o in outs) / reps, 0.8, reps, "power")

    def test_stratified(self):
        rnd, reps = random.Random(13), 600
        hazards = (0.01, 0.2)  # very different baseline hazards: the unstratified test loses power
        stratified = GroupSequentialLogRank([60, 120], [2.8, 1.98], strata=2)
        plain = GroupSequentialLogRank([60, 120], [2.8, 1.98])
        level = sum(survival_trial(stratified, 1.0, rnd, 400, strata_hazards=hazards).decision == REJECT_H0
                    for _ in range(reps))
        within(self, level / reps, 0.025, reps, "stratified level")
        # one stratum: the stratified statistic is the plain one
        one = GroupSequentialLogRank([60, 120], [2.8, 1.98], strata=3)
        rnd_a, rnd_b = random.Random(1), random.Random(1)
        a = survival_trial(one, 0.8, rnd_a, 300)
        b = survival_trial(plain, 0.8, rnd_b, 300)
        self.assertEqual((a.z, a.information, a.events), (b.z, b.information, b.events))

    def test_records_are_checked(self):
        proc = GroupSequentialLogRank([5], [2.0])
        state = proc.initial_state()
        with self.assertRaises(ContractViolation):
            proc.step(state, {"arm": 1, "kind": EVENT})  # nobody at risk
        state, _ = proc.step(state, {"arm": 1, "kind": ENTER})
        state, out = proc.step(state, {"arm": 1, "kind": EVENT})  # nobody at risk in the other arm
        self.assertEqual((out.events, state.variance), (1, 0.0))
        self.assertIsNone(out.hazard_ratio_estimate)
        with self.assertRaises(ValueError):
            proc.decode_state({"at_risk": [0, 0, 0, 0], "events": 0, "observed": 0, "expected": 0.0,
                               "variance": 0.0, "analysis": 0, "decision": None})
        with self.assertRaises(ValueError):
            proc.decode_state({"at_risk": [0, 0], "events": 1, "observed": 2, "expected": 0.0,
                               "variance": 0.0, "analysis": 0, "decision": None})


if __name__ == "__main__":
    unittest.main()
