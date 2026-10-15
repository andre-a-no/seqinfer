# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""e-LOND: the gamma sequence sums to at most one, the levels follow the definition, and the FDR is
at most alpha by simulation -- with independent and with strongly dependent e-values."""
import math
import random
import unittest

from sequential_analysis.procedures import OnlineFDR
from sequential_analysis.procedures.online_fdr import default_gamma


def stream(proc, logs):
    state, outs = proc.initial_state(), []
    for v in logs:
        state, out = proc.step(state, {"log_e": v})
        outs.append(out)
    return outs


class ELOND(unittest.TestCase):
    def test_gamma_sums_to_at_most_one(self):
        n = 10**6
        partial = math.fsum(default_gamma(j) for j in range(1, n + 1))
        # the terms decrease beyond j = 3, so the tail is below the integral from n, which is
        # c * 2 exp(-u) (u^3 + 3u^2 + 6u + 6) with u = sqrt(log n) (substitute x = exp(u^2))
        u = math.sqrt(math.log(n))
        tail = 0.07720838 * 2.0 * math.exp(-u) * (u**3 + 3 * u**2 + 6 * u + 6)
        self.assertLessEqual(partial + tail, 1.0)
        self.assertGreater(partial + tail, 0.95)

    def test_levels_follow_the_definition(self):
        proc = OnlineFDR(0.1)
        outs = stream(proc, [10.0, -1.0, 10.0, 0.0, 10.0])
        discoveries = 0
        for t, out in enumerate(outs, start=1):
            self.assertAlmostEqual(out.level, 0.1 * default_gamma(t) * (discoveries + 1), delta=1e-15)
            discoveries += out.rejected
            self.assertEqual(out.discoveries, discoveries)
        self.assertEqual([o.rejected for o in outs], [True, False, True, False, True])

    def check_fdr(self, dependent):
        rnd = random.Random(1 if dependent else 2)
        alpha, runs, total = 0.2, 2000, 0.0
        for _ in range(runs):
            truth = [rnd.random() < 0.3 for _ in range(60)]  # 30% real effects
            shared = rnd.gauss(0, 1)
            logs = []
            for real in truth:
                z = (0.8 * shared + 0.6 * rnd.gauss(0, 1)) if dependent else rnd.gauss(0, 1)
                mu = 3.5
                logs.append(mu * (z + (mu if real else 0.0)) - mu * mu / 2)  # a likelihood ratio: an e-value
            outs = stream(OnlineFDR(alpha), logs)
            found = [i for i, o in enumerate(outs) if o.rejected]
            if found:
                total += sum(not truth[i] for i in found) / len(found)
        return total / runs

    def test_false_discovery_rate_independent(self):
        self.assertLessEqual(self.check_fdr(False), 0.2)

    def test_false_discovery_rate_dependent(self):
        self.assertLessEqual(self.check_fdr(True), 0.2)

    def test_planned_horizon(self):
        proc = OnlineFDR(0.1, horizon=4)
        outs = stream(proc, [3.0, 4.0, 4.0, 2.0])
        for out, expected in zip(outs, [0.025, 0.025, 0.05, 0.075]):  # 0.1/4 times (D+1)
            self.assertAlmostEqual(out.level, expected, delta=1e-15)
        self.assertEqual([o.rejected for o in outs], [False, True, True, False])
        state = proc.initial_state()
        for v in (0.0,) * 4:
            state, _ = proc.step(state, {"log_e": v})
        self.assertTrue(proc.is_terminal(state))
        from sequential_analysis.errors import LifecycleError

        with self.assertRaises(LifecycleError):
            proc.step(state, {"log_e": 9.0})

    def test_false_discovery_rate_with_a_horizon(self):
        rnd = random.Random(4)
        alpha, runs, total = 0.2, 2000, 0.0
        for _ in range(runs):
            truth = [rnd.random() < 0.3 for _ in range(40)]
            logs = [3.0 * (rnd.gauss(0, 1) + (3.0 if real else 0.0)) - 4.5 for real in truth]
            outs = stream(OnlineFDR(alpha, horizon=40), logs)
            found = [i for i, o in enumerate(outs) if o.rejected]
            if found:
                total += sum(not truth[i] for i in found) / len(found)
        self.assertLessEqual(total / runs, alpha)

    def test_state_and_arguments(self):
        proc = OnlineFDR()
        state = proc.initial_state()
        state, _ = proc.step(state, {"log_e": 50.0})
        self.assertEqual(proc.decode_state(proc.encode_state(state)), state)
        with self.assertRaises(ValueError):
            proc.decode_state({"t": 1, "discoveries": 2})
        with self.assertRaises(ValueError):
            OnlineFDR(1.0)
        with self.assertRaises(ValueError):
            OnlineFDR(horizon=0)


if __name__ == "__main__":
    unittest.main()
