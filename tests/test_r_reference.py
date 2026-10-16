# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Agreement with independent implementations in R (tools/r_reference.R).

* rpact: efficacy and non-binding futility boundaries of 56 designs
  (O'Brien-Fleming, Pocock and power spending, one- and two-sided, two to
  five analyses), the drift for the planned power, crossing and futility
  probabilities, expected sample sizes under H0 and H1, inflation factors;
  boundaries at realised information fractions, including under- and
  over-running; maximum sample sizes for one and two means, two
  proportions and a log-rank test.
* survival: the log-rank statistic at four interim analyses of a trial
  with staggered entry, as the score test of a Cox model on (entry, exit]
  data, and Peto's estimate against the Cox estimate.
* stats: the two-proportion chi-square test and the pooled two-sample
  t-test.

The reference file is tests/r_reference.json.  CI regenerates it with R
and runs these tests against the fresh file (job "r-reference"); set
R_REFERENCE to a path to use another file.
"""
import json
import math
import os
import unittest

from sequential_analysis.group_sequential import group_sequential_design
from sequential_analysis.procedures.group_sequential import (
    GroupSequentialLogRank,
    GroupSequentialMonitor,
    GroupSequentialProportions,
    GroupSequentialTwoMeans,
)

_PATH = os.environ.get("R_REFERENCE") or os.path.join(os.path.dirname(__file__), "r_reference.json")
with open(_PATH, encoding="utf-8") as f:
    REF = json.load(f)

# rpact computes boundaries to about 1e-8 and two-sided crossing probabilities to a few 1e-7
# (ours do not move by 1e-9 with a grid three times finer); ours are bisected to 1e-12 on a Simpson
# grid exact to about 1e-9.  The futility drift is bisected to 1e-7.
BOUND = 2e-7
PROBABILITY = 1e-6


def spec(value):
    return tuple(value) if isinstance(value, list) else value


def design_of(case):
    futility = case.get("futility_spending")
    return group_sequential_design(
        case["alpha"], case["fractions"], spec(case["spending"]), two_sided=case["sided"] == 2,
        futility=spec(futility) if futility is not None else None,
        power=case["power"] if futility is not None else None,
    )


def close(test, got, want, tol, msg):
    test.assertEqual(len(got), len(want), msg)
    for g, w in zip(got, want):
        test.assertLess(abs(g - w), tol, f"{msg}: {list(got)} vs {list(want)}")


class RpactDesigns(unittest.TestCase):
    def test_designs(self):
        self.assertEqual(len(REF["designs"]), 56)
        for case in REF["designs"]:
            msg = json.dumps({k: case[k] for k in ("alpha", "sided", "fractions", "spending", "futility_spending")})
            d = design_of(case)
            close(self, d.bounds, case["critical"], BOUND, msg + " critical")
            futility = case["futility_bounds"]
            power = None if futility is not None else case["power"]
            drift = d.drift if futility is not None else d.drift_for_power(case["power"])
            self.assertLess(abs(drift**2 / case["shift"] - 1.0), 1e-6, msg + " shift")
            self.assertLess(abs(d.inflation_factor(power) / case["inflation"] - 1.0), 1e-6, msg + " inflation")
            # rpact's rejection probabilities of a two-sided design are those in the direction of the drift
            upward = d._walk(drift, 24, True, upper_only=True)[0]
            close(self, upward, case["rejection"], PROBABILITY, msg + " rejection")
            inflation = d.inflation_factor(power)
            self.assertLess(abs(d.expected_fraction(drift) * inflation - case["asn1"]), 1e-6, msg + " ASN H1")
            self.assertLess(abs(d.expected_fraction(0.0) * inflation - case["asn0"]), 1e-6, msg + " ASN H0")
            if futility is not None:
                close(self, d.futility[:-1], futility, 1e-6, msg + " futility")
                close(self, d.futility_crossing(drift)[:-1], case["futility_probabilities"], 1e-6,
                      msg + " futility probabilities")

    def test_boundaries_at_realised_information(self):
        planned = group_sequential_design(0.025, [0.25, 0.5, 0.75, 1.0], "obrien-fleming")
        for case in REF["realised"]:
            msg = str(case["fractions"])
            close(self, planned.at(case["fractions"]).bounds, case["critical"], BOUND, msg)
            monitor = GroupSequentialMonitor(0.025, 200.0, final_input="final")
            state, bounds = monitor.initial_state(), []
            last = len(case["fractions"]) - 1
            for k, t in enumerate(case["fractions"]):
                state, out = monitor.step(state, {"z": 0.0, "information": 200.0 * t, "final": int(k == last)})
                bounds.append(out.bound)
            close(self, bounds, case["critical"], BOUND, msg + " monitor")

    def test_sample_sizes(self):
        designs = {}
        for case in REF["sample_sizes"]:
            key = json.dumps(case["design"])
            if key not in designs:
                c = case["design"]
                designs[key] = group_sequential_design(c["alpha"], c["fractions"], spec(c["spending"]),
                                                       two_sided=c["sided"] == 2)
            d, power = designs[key], case["design"]["power"]
            kind = case["kind"]
            if kind == "rates":
                got = d.max_sample_size_two_proportions(case["p_control"], case["p_treatment"], case["allocation"],
                                                        power=power)
            elif kind == "events":
                got = d.max_events(case["hazard_ratio"], case["allocation"], power=power)
            elif kind == "two_means":
                got = d.max_sample_size_two_means(case["effect"], case["sigma"], case["allocation"], power=power)
            else:
                got = d.max_sample_size(case["effect"], case["sigma"], power=power)
            self.assertEqual(got, math.ceil(case["n"] - 1e-6), json.dumps(case))


class SurvivalLogRank(unittest.TestCase):
    def test_score_test_with_staggered_entry(self):
        ref = REF["logrank"]
        records = []
        for arm, entry, exit_, status in zip(ref["arm"], ref["entry"], ref["exit"], ref["status"]):
            records.append((entry, 0, arm))
            records.append((exit_, 1 if status == 1 else 2, arm))
        records.sort()
        analyses = [a["events"] for a in ref["analyses"]]
        for alternative in ("greater", "less"):
            proc = GroupSequentialLogRank(analyses, [40.0] * len(analyses), alternative=alternative)
            state, seen = proc.initial_state(), []
            for _, kind, arm in records:
                state, out = proc.step(state, {"arm": int(arm), "kind": kind})
                if out.analysis:
                    seen.append(out)
                if proc.is_terminal(state):
                    break
            self.assertEqual([o.events for o in seen], analyses)
            for out, want in zip(seen, ref["analyses"]):
                u = want["score"]  # O - E of the treatment arm
                self.assertAlmostEqual(out.z**2, want["score_chisq"], delta=1e-12 * want["score_chisq"])
                self.assertEqual(out.z > 0, (u > 0) == (alternative == "greater"))
                self.assertAlmostEqual(out.information, u * u / want["score_chisq"], delta=1e-10 * out.information)
                # Peto's one-step estimate approximates the Cox estimate
                self.assertLess(abs(math.log(out.hazard_ratio_estimate) - want["log_hazard_ratio"]), 0.06)


class ChiSquare(unittest.TestCase):
    def test_pooled_z_is_the_chi_square_test(self):
        for case in REF["proportions"]:
            proc = GroupSequentialProportions([case["n_treatment"] + case["n_control"]], [40.0])
            state = proc.initial_state()
            for arm, successes, n in ((1, case["successes_treatment"], case["n_treatment"]),
                                      (0, case["successes_control"], case["n_control"])):
                for i in range(n):
                    state, out = proc.step(state, {"arm": arm, "y": int(i < successes)})
            self.assertAlmostEqual(out.z**2, case["chisq"], delta=1e-12 * case["chisq"])

    def test_pooled_t_is_students(self):
        for case in REF["two_means"]:
            treatment, control = case["treatment"], case["control"]
            n = len(treatment) + len(control)
            proc = GroupSequentialTwoMeans(None, [n], [40.0])
            state = proc.initial_state()
            for arm, values in ((1, treatment), (0, control)):
                for y in values:
                    state, out = proc.step(state, {"arm": arm, "y": y})
            self.assertAlmostEqual(out.z, case["t"], delta=1e-12 * abs(case["t"]) + 1e-14)
            self.assertEqual(n - 2, case["df"])


if __name__ == "__main__":
    unittest.main()
