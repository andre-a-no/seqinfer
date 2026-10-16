# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Mathematical correctness of the ratio-metric A/B test (procedures/ratio.py).

* The estimates against exact rational ratios of sums, after every unit.
* The delta-method variance against Deng et al.'s formula computed
  directly, and against the Monte Carlo variance of the ratio estimator.
* Coverage of both confidence sequences under continuous monitoring, at
  50/50 and 30/70, on users with Poisson sessions and lognormal spend
  correlated within the user.  The same simulations show that the naive
  per-session test over-rejects.
* Checkpoints, state validation, refused inputs, and the TypeScript port
  (when node and the built executor are present).
"""
import json
import math
import os
import random
import shutil
import subprocess
import unittest
from fractions import Fraction
from pathlib import Path

from sequential_analysis import ContractViolation, KeyJoin, Recorder, Run, run_sync
from sequential_analysis.procedures.ratio import RatioABTest

from .helpers import record_stream

ROOT = Path(__file__).resolve().parent.parent
NODE = os.environ.get("NODE") or shutil.which("node")
TS_MODULE = ROOT / "ts" / "dist" / "src" / "procedures" / "ratio.js"

# sessions ~ Poisson(LAM); spend per session ~ exp(sigma_u Z + B sessions + SIGMA_Z Z')
LAM, B, SIGMA_Z = 2.0, -0.1, 0.5


def fold(proc, inputs):
    state, outs = proc.initial_state(), []
    for x in inputs:
        state, out = proc.step(state, x)
        outs.append(out)
    return state, outs


def binomial_upper_bound(alpha, runs):
    """alpha plus three standard errors of a Monte Carlo proportion."""
    return alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / runs)


def user(rnd, allocation=0.5, lift=0.0, sigma_u=1.0):
    """(arm, spend of each session): a per-user multiplier, shared by the sessions, that falls with their number."""
    treated = rnd.random() < allocation
    sessions, p, limit = 0, rnd.random(), math.exp(-LAM)
    while p > limit:
        sessions += 1
        p *= rnd.random()
    u = math.exp(sigma_u * rnd.gauss(0.0, 1.0) + B * sessions) * ((1.0 + lift) if treated else 1.0)
    return int(treated), [u * math.exp(SIGMA_Z * rnd.gauss(0.0, 1.0)) for _ in range(sessions)]


def true_ratio(sigma_u=1.0):
    """E[spend] / E[sessions]: E[S exp(B S)] / E[S] = exp(B + LAM (e^B - 1)) for S ~ Poisson(LAM)."""
    return math.exp(sigma_u**2 / 2 + SIGMA_Z**2 / 2 + B + LAM * (math.exp(B) - 1.0))


def rows(n, seed, **kwargs):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        arm, sessions = user(rnd, **kwargs)
        out.append({"arm": arm, "y": math.fsum(sessions), "d": float(len(sessions))})
    return out


class Estimates(unittest.TestCase):
    def test_estimates_are_the_ratios_of_sums_after_every_unit(self):
        data = [{"arm": 2, "y": 5.0, "d": 1.0}, *rows(300, 1, allocation=0.4)]
        _, outs = fold(RatioABTest(start=2), data)
        sums = {k: [Fraction(0), Fraction(0)] for k in (0, 1)}
        for x, out in zip(data, outs):
            if x["arm"] in (0, 1):
                sums[x["arm"]][0] += Fraction(x["y"])
                sums[x["arm"]][1] += Fraction(x["d"])
            exact = [sums[k][0] / sums[k][1] if sums[k][1] else None for k in (1, 0)]
            for got, want in zip(out.ratios, exact):
                if want is None:
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(got, float(want), delta=1e-14 * abs(float(want)))
            if None not in exact:
                self.assertAlmostEqual(out.difference, float(exact[0] - exact[1]), delta=1e-13)
                self.assertAlmostEqual(out.lift, float(exact[0] / exact[1] - 1), delta=1e-13)

    def test_without_a_denominator_it_compares_means_per_user(self):
        data = rows(200, 2)
        _, plain = fold(RatioABTest(denominator=False, start=10), [{"arm": x["arm"], "y": x["y"]} for x in data])
        _, ones = fold(RatioABTest(start=10), [{**x, "d": 1.0} for x in data])
        self.assertEqual(plain, ones)
        last = plain[-1]
        for k, arm in ((0, 1), (1, 0)):
            ys = [x["y"] for x in data if x["arm"] == arm]
            self.assertAlmostEqual(last.ratios[k], math.fsum(ys) / len(ys), delta=1e-12)
        # the variance is the unpooled s_T^2 / n_T + s_C^2 / n_C
        var = 0.0
        for arm in (0, 1):
            ys = [x["y"] for x in data if x["arm"] == arm]
            mean = math.fsum(ys) / len(ys)
            var += math.fsum((y - mean) ** 2 for y in ys) / (len(ys) - 1) / len(ys)
        self.assertAlmostEqual(last.std_error, math.sqrt(var), delta=1e-12)

    def test_zero_denominators_and_other_arms(self):
        proc = RatioABTest(start=2)
        state, out = proc.step(proc.initial_state(), {"arm": 1, "y": 0.0, "d": 0.0})
        self.assertEqual((out.n, out.ratios, out.difference), (1, (None, None), None))
        state, out = proc.step(state, {"arm": 0, "y": 3.0, "d": 2.0})
        self.assertEqual((out.ratios, out.std_error), ((None, 1.5), None))
        same, out2 = proc.step(state, {"arm": 7, "y": 9.0, "d": 1.0})
        self.assertEqual((same, out2), (state, out))
        # a user without sessions counts as a unit (in n and the variance) but leaves the ratios as they are
        _, outs = fold(proc, [{"arm": 1, "y": 0.0, "d": 0.0}, *rows(50, 3)])
        _, without = fold(proc, rows(50, 3))
        self.assertEqual(outs[-1].ratios, without[-1].ratios)
        self.assertEqual(outs[-1].n, without[-1].n + 1)

    def test_no_lift_unless_both_ratios_are_positive(self):
        data = [{"arm": 1, "y": -1.0, "d": 1.0}, {"arm": 1, "y": -2.0, "d": 1.0},
                {"arm": 0, "y": 1.0, "d": 1.0}, {"arm": 0, "y": 3.0, "d": 1.0}]
        _, outs = fold(RatioABTest(start=2), data)
        last = outs[-1]
        self.assertEqual((last.lift, last.lift_lower, last.lift_upper), (None, None, None))
        self.assertEqual(last.difference, -1.5 - 2.0)
        self.assertIsNotNone(last.lower)


class Variance(unittest.TestCase):
    def test_delta_method_variance_is_deng_et_al_formula(self):
        data = rows(400, 4, allocation=0.3)
        proc = RatioABTest(start=10**6)
        state = proc.initial_state()
        for t, x in enumerate(data, 1):
            state, out = proc.step(state, x)
            if t not in (10, 37, 150, 400):
                continue
            total = Fraction(0)
            for k, arm in ((0, 1), (1, 0)):
                ys = [Fraction(u["y"]) for u in data[:t] if u["arm"] == arm]
                ds = [Fraction(u["d"]) for u in data[:t] if u["arm"] == arm]
                n = len(ys)
                my, md = sum(ys) / n, sum(ds) / n
                syy = sum((y - my) ** 2 for y in ys) / (n - 1)
                sdd = sum((d - md) ** 2 for d in ds) / (n - 1)
                syd = sum((y - my) * (d - md) for y, d in zip(ys, ds)) / (n - 1)
                deng = (syy / md**2 - 2 * my * syd / md**3 + my**2 * sdd / md**4) / n
                self.assertAlmostEqual(proc.variance(state, k), float(deng), delta=1e-11 * float(deng))
                total += deng
            self.assertAlmostEqual(out.std_error, math.sqrt(float(total)), delta=1e-11 * math.sqrt(float(total)))

    def test_delta_method_variance_matches_the_monte_carlo_variance(self):
        proc = RatioABTest(start=10**6)
        estimates, variances, lifts, log_variances = [], [], [], []
        for r in range(600):
            state, outs = fold(proc, rows(300, 100 + r, sigma_u=0.5))
            estimates.append(outs[-1].difference)
            variances.append(outs[-1].std_error ** 2)
            (rt, rc), vt, vc = outs[-1].ratios, proc.variance(state, 0), proc.variance(state, 1)
            lifts.append(math.log(rt / rc))
            log_variances.append(vt / rt**2 + vc / rc**2)

        def var(v):
            m = math.fsum(v) / len(v)
            return math.fsum((e - m) ** 2 for e in v) / (len(v) - 1)

        # the Monte Carlo variance has a relative standard error of about 6% here
        self.assertAlmostEqual(var(estimates) / (math.fsum(variances) / 600), 1.0, delta=0.15)
        self.assertAlmostEqual(var(lifts) / (math.fsum(log_variances) / 600), 1.0, delta=0.15)
        self.assertLess(abs(math.fsum(estimates) / 600), 4 * math.sqrt(var(estimates) / 600))  # no effect


class Monitoring(unittest.TestCase):
    """Continuous monitoring from `start` = 300 to 1,500 users, 200 runs per setting."""

    runs, n, alpha, start = 200, 1500, 0.05, 300
    results: dict = {}

    @classmethod
    def simulate(cls, allocation, lift):
        key = (allocation, lift)
        if key in cls.results:
            return cls.results[key]
        truth = true_ratio()
        difference = truth * lift
        proc = RatioABTest(cls.alpha, start=cls.start, tuned_for=cls.start)
        missed_difference = missed_lift = naive_missed = rejected = 0
        for r in range(cls.runs):
            rnd = random.Random(10_000 * int(allocation * 10) + r)
            state = proc.initial_state()
            # the naive per-session analysis: sessions as independent observations, per arm
            count, total, squares = [0, 0], [0.0, 0.0], [0.0, 0.0]
            miss_d = miss_l = naive = False
            for t in range(1, cls.n + 1):
                arm, sessions = user(rnd, allocation, lift)
                state, out = proc.step(state, {"arm": arm, "y": math.fsum(sessions), "d": float(len(sessions))})
                count[arm] += len(sessions)
                total[arm] += math.fsum(sessions)
                squares[arm] += math.fsum(v * v for v in sessions)
                if out.lower is None:
                    continue
                miss_d = miss_d or not out.lower <= difference <= out.upper
                miss_l = miss_l or not out.lift_lower <= lift <= out.lift_upper
                if t >= cls.start and min(count) >= 2:
                    var = 0.0
                    for k in (0, 1):
                        mean = total[k] / count[k]
                        var += max(0.0, squares[k] - count[k] * mean * mean) / (count[k] - 1) / count[k]
                    sessions_seen = count[0] + count[1]
                    h = proc.half_width(sessions_seen, math.sqrt(sessions_seen * var))
                    naive_estimate = total[1] / count[1] - total[0] / count[0]
                    naive = naive or abs(naive_estimate - difference) > h
            missed_difference += miss_d
            missed_lift += miss_l
            naive_missed += naive
            rejected += state.decision is not None
        out = {"difference": missed_difference / cls.runs, "lift": missed_lift / cls.runs,
               "naive": naive_missed / cls.runs, "rejected": rejected / cls.runs}
        cls.results[key] = out
        return out

    def test_coverage_at_50_50_with_no_effect_and_the_naive_test_over_rejects(self):
        res = self.simulate(0.5, 0.0)
        bound = binomial_upper_bound(self.alpha, self.runs)
        self.assertLessEqual(res["difference"], bound, res)
        self.assertLessEqual(res["lift"], bound, res)
        self.assertEqual(res["rejected"], res["difference"])  # no effect: a rejection is a miss
        # sessions of a user share their spend: the per-session variance is far too small
        self.assertGreater(res["naive"], 2 * self.alpha, res)
        self.assertGreater(res["naive"], res["difference"] + 0.1, res)

    def test_coverage_at_30_70_with_a_ten_percent_lift(self):
        res = self.simulate(0.3, 0.1)
        bound = binomial_upper_bound(self.alpha, self.runs)
        self.assertLessEqual(res["difference"], bound, res)
        self.assertLessEqual(res["lift"], bound, res)
        self.assertGreater(res["naive"], 2 * self.alpha, res)

    def test_power(self):
        data = rows(20_000, 5, allocation=0.5, lift=0.15)
        for scale in ("difference", "relative"):
            _, outs = fold(RatioABTest(scale=scale, stop_on_reject=True), data)
            self.assertEqual(outs[-1].decision, "reject_h0", scale)
            self.assertAlmostEqual(outs[-1].lift, 0.15, delta=0.1)

    def test_the_decision_is_the_first_exclusion_of_the_null_value(self):
        data = rows(5000, 6, lift=0.1)
        for scale, effect0, low, high in (("relative", 0.1, "lift_lower", "lift_upper"),
                                          ("relative", -0.2, "lift_lower", "lift_upper"),
                                          ("difference", 0.3, "lower", "upper")):
            _, outs = fold(RatioABTest(scale=scale, effect0=effect0), data)
            excluded = [getattr(o, low) is not None and not getattr(o, low) <= effect0 <= getattr(o, high)
                        for o in outs]
            first = excluded.index(True) if True in excluded else len(outs)
            self.assertEqual([o.decision for o in outs], [None] * first + ["reject_h0"] * (len(outs) - first))


class Checkpoints(unittest.TestCase):
    @staticmethod
    def stream(n=240, seed=7):
        return record_stream(rows(n, seed, allocation=0.4), ("arm", "y", "d"))

    def test_a_restored_run_continues_bit_for_bit(self):
        make = lambda: RatioABTest(start=40, tuned_for=50, scale="relative")
        topology = lambda: KeyJoin(("arm", "y", "d"))
        stream = self.stream()
        full_rec = Recorder()
        full = Run(make(), topology=topology(), consumers=[full_rec], run_id="r", clock=lambda: 0.0)
        run_sync(full, stream, stop_on_terminal=False)
        rec = Recorder()
        first = Run(make(), topology=topology(), consumers=[rec], run_id="r", clock=lambda: 0.0)
        cut = 3 * 100
        run_sync(first, stream[:cut], stop_on_terminal=False)
        checkpoint = json.loads(json.dumps(first.checkpoint()))
        second = Run.restore(make(), checkpoint, topology=topology(), consumers=[rec], clock=lambda: 0.0)
        run_sync(second, stream[cut:], stop_on_terminal=False)
        self.assertEqual(second.state, full.state)
        self.assertEqual(rec.outputs, full_rec.outputs)
        self.assertEqual(second.checkpoint()["history_digest"], full.checkpoint()["history_digest"])

    def test_decode_state(self):
        proc = RatioABTest()
        state, _ = fold(proc, rows(30, 8))
        encoded = json.loads(json.dumps(proc.encode_state(state)))
        self.assertEqual(proc.decode_state(encoded), state)
        broken = [
            {**encoded, "units": [1]},
            {**encoded, "units": [-1, 3]},
            {**encoded, "cyy": [-1.0, 0.0]},
            {**encoded, "sum_d": [-2.0, 1.0]},
            {**encoded, "decision": "maybe"},
            {**encoded, "units": [0, encoded["units"][1]]},  # sums without units
            {k: v for k, v in encoded.items() if k != "cyd"},
            {**encoded, "extra": 1},
        ]
        for data in broken:
            with self.assertRaises((ValueError, TypeError), msg=str(data)):
                proc.decode_state(data)
        plain = RatioABTest(denominator=False)
        with self.assertRaises(ValueError):  # a denominator other than 1
            plain.decode_state(encoded)
        state, _ = fold(plain, [{"arm": x["arm"], "y": x["y"]} for x in rows(30, 8)])
        self.assertEqual(plain.decode_state(json.loads(json.dumps(plain.encode_state(state)))), state)

    def test_bad_inputs_are_refused(self):
        proc = RatioABTest()
        for bad in ({"arm": 1, "y": 1.0, "d": -1.0}, {"arm": 0, "y": math.nan, "d": 1.0},
                    {"arm": 1, "y": 1.0, "d": math.inf}):
            with self.assertRaises(ContractViolation):
                proc.step(proc.initial_state(), bad)
        run = Run(RatioABTest(), on_invalid="skip")
        run.step({"arm": 1, "y": 2.0, "d": -1.0})
        run.step({"arm": 1, "y": 2.0, "d": 1.0})
        run.step({"arm": 1.5, "y": 2.0, "d": 1.0})
        self.assertEqual((run.counters["invalid_skipped"], run.state.units), (2, (1, 0)))
        self.assertEqual(RatioABTest(denominator=False).inputs().keys(), {"arm", "y"})

    def test_arguments(self):
        for kwargs in ({"alpha": 1.0}, {"scale": "log"}, {"scale": "relative", "effect0": -1.0},
                       {"effect0": math.inf}, {"arms": (1, 1)}, {"arms": (1, -1)}, {"start": 1}, {"start": 10.0},
                       {"tuned_for": 0.5}, {"denominator_input": "y"}, {"arm_input": "d"}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                RatioABTest(**kwargs)
        RatioABTest(denominator=False, denominator_input="y")  # no denominator: the name is unused
        self.assertEqual(RatioABTest(effect0=-3.0).effect0, -3.0)  # a difference may be below -1


NODE_SCRIPT = """
import { readFileSync } from "node:fs";
const { RatioABTest } = await import(process.env.RATIO_MODULE);
const data = JSON.parse(readFileSync(0, "utf8"));
const proc = RatioABTest.fromConfig(data.config);
const run = (state, inputs) => {
  const outs = [];
  for (const x of inputs) { const [next, out] = proc.step(state, x); state = next; outs.push(out); }
  return [state, outs];
};
const [state, outs] = run(proc.initialState(), data.inputs);
const [resumed, rest] = run(proc.decodeState(data.mid_state), data.inputs.slice(data.cut));
process.stdout.write(JSON.stringify({ state, outs, resumed, rest }));
"""


def largest_difference(a, b):
    """Largest relative difference of two JSON values that agree in everything but the last bits of numbers."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        scale = max(abs(a), abs(b))
        return 0.0 if a == b or scale < 1e-300 else abs(a - b) / scale
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return max((largest_difference(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return max((largest_difference(a[k], b[k]) for k in a), default=0.0)
    if a != b:
        raise AssertionError(f"{a!r} != {b!r}")
    return 0.0


@unittest.skipUnless((NODE and TS_MODULE.exists()) or os.environ.get("REQUIRE_TYPESCRIPT"),
                     "node or the built TypeScript port (ts/dist) is missing")
class TypeScriptPort(unittest.TestCase):
    def test_the_port_reproduces_python(self):
        data = [{"arm": 2, "y": 1.0, "d": 1.0}, *rows(400, 9, allocation=0.3, lift=0.4)]
        for proc in (RatioABTest(start=30, tuned_for=100), RatioABTest(start=30, scale="relative", effect0=0.2),
                     RatioABTest(denominator=False, start=20)):
            with self.subTest(config=proc.config()):
                inputs = data if proc.denominator else [{"arm": x["arm"], "y": x["y"]} for x in data]
                state, outs = fold(proc, inputs)
                cut = 150
                mid, _ = fold(proc, inputs[:cut])
                payload = {"config": proc.config(), "inputs": inputs, "cut": cut, "mid_state": proc.encode_state(mid)}
                result = subprocess.run([NODE, "--input-type=module", "-e", NODE_SCRIPT], input=json.dumps(payload),
                                        capture_output=True, text=True, check=True, timeout=60,
                                        env={**os.environ, "RATIO_MODULE": TS_MODULE.as_uri()})
                got = json.loads(result.stdout)
                want_outs = json.loads(json.dumps([proc.encode_output(o) for o in outs]))
                want_state = json.loads(json.dumps(proc.encode_state(state)))
                self.assertEqual(got["state"], want_state)  # +, -, *, /: bit for bit
                self.assertEqual(got["resumed"], want_state)  # continued from a Python state
                self.assertLessEqual(largest_difference(got["outs"], want_outs), 1e-13)  # log, expm1: last bits
                self.assertLessEqual(largest_difference(got["rest"], want_outs[cut:]), 1e-13)


if __name__ == "__main__":
    unittest.main()
