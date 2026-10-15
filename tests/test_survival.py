# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Mathematical correctness of the anytime-valid log-rank test.

* Exact martingale property: every sequence of event arms of a small
  trial is enumerated, with risk sets shrinking with events and a
  censoring and an entry that depend on the past; E[e-value] is 1 under
  H0 at every number of events and at a stopping time, and at most 1 on
  the null side of a one-sided test.
* Type I error by Monte Carlo (exponential and Weibull baseline hazards,
  staggered entry, random censoring, monitored after every event), power
  at a hazard ratio of 0.6, coverage of the confidence sequence uniformly
  over time.
* Agreement with independent computations: the classical log-rank z,
  the penalised Cox estimate by Newton's method, the mixtures from their
  definition.
* Invariance to the time scale, state round trip, contract violations.
* Stratification: strata=1 is the unstratified test bit for bit (its
  conformance vectors); the exact martingale property by enumeration in
  two strata; the stratified partial likelihood from its definition;
  entry cohorts against hazards in time since entry, and centres with
  different baseline hazards, by Monte Carlo; checkpoints, contracts, and
  a run moved to the TypeScript executor and back.
"""
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
import unittest
from pathlib import Path

from sequential_analysis import KeyJoin, Observation, Recorder, Run, equivalent, run_sync
from sequential_analysis.canonical import canonical_json
from sequential_analysis.core import trajectory
from sequential_analysis.errors import ContractViolation, IncompatibleCheckpoint
from sequential_analysis.numerics import logsumexp
from sequential_analysis.procedures.survival import AnytimeLogRankTest

from .test_typescript import CLI, NODE

TREATMENT, CONTROL = 1, 0
ENTER, EVENT, CENSOR = 0, 1, 2


def binomial_upper_bound(alpha, runs):
    """alpha plus three standard errors of a Monte Carlo proportion."""
    return alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / runs)


def trial(rnd, patients, hazard_ratio, *, shape=1.0, scale=1.0, accrual=2.0, censoring=0.3, end=None,
          time_scale="calendar", stretch=1.0):
    """(time, kind, arm) records of a randomised 1:1 trial, sorted by calendar time.

    The baseline cumulative hazard is (t / scale)^shape (Weibull; exponential
    for shape 1), multiplied by hazard_ratio in the treatment arm.  On the
    calendar time scale a patient entering at e is left-truncated there:
    H(T) - H(e) ~ Exp(1).  On the patient time scale T - e is Weibull.
    Censoring is exponential after entry, and the study ends at `end`.
    `stretch` multiplies every time by a constant.
    """
    records = []
    for _ in range(patients):
        e = rnd.uniform(0.0, accrual)
        arm = TREATMENT if rnd.random() < 0.5 else CONTROL
        g = rnd.expovariate(1.0) / (hazard_ratio if arm == TREATMENT else 1.0)
        if time_scale == "calendar":
            t = scale * ((e / scale) ** shape + g) ** (1.0 / shape)
        else:
            t = e + scale * g ** (1.0 / shape)
        c = e + rnd.expovariate(censoring)
        records.append((e, ENTER, arm))
        records.append((t, EVENT, arm) if t <= c else (c, CENSOR, arm))
    records.sort()
    if end is not None:
        records = [r for r in records if r[0] <= end]
    return [(stretch * t, k, a) for t, k, a in records]


def inputs(records):
    return [{"arm": a, "kind": k} for _, k, a in records]


def first_rejection(proc, xs):
    """Number of events at the first rejection, None if the test never rejects."""
    state = proc.initial_state()
    for x in xs:
        state, out = proc.step(state, x)
        if out.decision is not None:
            return out.n_events
    return None


def informative_events(xs):
    """(x, n_T, n_C) for every event with both arms at risk, by an independent count."""
    at_risk = {TREATMENT: 0, CONTROL: 0}
    out = []
    for x in xs:
        arm, kind = x["arm"], x["kind"]
        if kind == ENTER:
            at_risk[arm] += 1
            continue
        if kind == EVENT and at_risk[TREATMENT] and at_risk[CONTROL]:
            out.append((1 if arm == TREATMENT else 0, at_risk[TREATMENT], at_risk[CONTROL]))
        at_risk[arm] -= 1
    return out


# ---------------------------------------------------------------- exact martingale property

#: A small trial: three treated and four control patients enter, then seven
#: events; after the second event the larger arm loses a patient to
#: censoring, after the third the smaller arm gains one (both depend on the
#: past only, which is allowed).
SCHEDULE = (["enter-t"] * 3 + ["enter-c"] * 4
            + ["event", "event", "censor-larger", "event", "enter-smaller", "event", "event", "event", "event"])
EVENTS = SCHEDULE.count("event")


def expectations(proc, theta, stop_at=None):
    """E[e-value after n events] for n = 1..EVENTS, and E[e-value at a stopping time], exactly by enumeration.

    The arm of each event is Bernoulli(n_T theta / (n_T theta + n_C)); every
    sequence of arms is followed with its probability.  The stopping time is
    the first n with log e-value >= stop_at, or the end.
    """
    by_n = [0.0] * (EVENTS + 1)
    stopped = [0.0]

    def walk(state, i, prob, frozen):
        # frozen: the e-value at the stopping time once it has passed, else None
        if i == len(SCHEDULE):
            stopped[0] += prob * (math.exp(proc.log_e_value(state)) if frozen is None else frozen)
            return
        op = SCHEDULE[i]
        n_t, n_c = state.at_risk
        if op == "event":
            p = n_t * theta / (n_t * theta + n_c)
            for arm, q in ((TREATMENT, p), (CONTROL, 1.0 - p)):
                if q == 0.0:
                    continue
                new, out = proc.step(state, {"arm": arm, "kind": EVENT})
                e = math.exp(out.log_e_value)
                by_n[out.n_events] += prob * q * e
                stop = frozen is None and stop_at is not None and out.log_e_value >= stop_at
                walk(new, i + 1, prob * q, e if stop else frozen)
            return
        if op == "enter-t":
            x = {"arm": TREATMENT, "kind": ENTER}
        elif op == "enter-c":
            x = {"arm": CONTROL, "kind": ENTER}
        elif op == "censor-larger":
            x = {"arm": TREATMENT if n_t >= n_c else CONTROL, "kind": CENSOR}
        else:
            x = {"arm": TREATMENT if n_t <= n_c else CONTROL, "kind": ENTER}
        new, _ = proc.step(state, x)
        walk(new, i + 1, prob, frozen)

    walk(proc.initial_state(), 0, 1.0, None)
    return by_n[1:], stopped[0]


class ExactMartingaleTests(unittest.TestCase):
    def procedures(self):
        for method in ("mixture", "grow"):
            for alternative in ("two-sided", "less", "greater"):
                for theta0 in (1.0, 1.7):
                    yield AnytimeLogRankTest(theta0=theta0, alternative=alternative, method=method,
                                             prior_sd=1.0, confidence_sequence=False)

    def test_expected_e_value_is_one_under_the_null(self):
        for proc in self.procedures():
            with self.subTest(**proc.config()):
                by_n, at_stop = expectations(proc, proc.theta0, stop_at=math.log(2.0))
                for n, mean in enumerate(by_n, 1):
                    self.assertAlmostEqual(mean, 1.0, delta=1e-12, msg=f"after {n} events")
                self.assertAlmostEqual(at_stop, 1.0, delta=1e-12, msg="at the stopping time")

    def test_the_e_values_are_not_trivial(self):
        """The enumeration really bets: the e-values vary along paths and grow under the alternative."""
        for proc in self.procedures():
            if proc.alternative == "greater":
                theta = proc.theta0 * 3.0
            else:
                theta = proc.theta0 / 3.0
            with self.subTest(**proc.config()):
                by_n, _ = expectations(proc, theta)
                self.assertGreater(by_n[-1], 1.05)

    def test_expected_e_value_is_at_most_one_inside_a_one_sided_null(self):
        for proc in self.procedures():
            if proc.alternative == "two-sided":
                continue
            factors = (1.2, 2.0, 5.0) if proc.alternative == "less" else (1 / 1.2, 1 / 2.0, 1 / 5.0)
            for f in factors:
                with self.subTest(theta=proc.theta0 * f, **proc.config()):
                    by_n, at_stop = expectations(proc, proc.theta0 * f, stop_at=math.log(2.0))
                    for n, mean in enumerate(by_n, 1):
                        self.assertLessEqual(mean, 1.0 + 1e-12, msg=f"after {n} events")
                    self.assertLessEqual(at_stop, 1.0 + 1e-12)
                    self.assertLess(by_n[-1], 1.0 - 1e-6)  # strictly inside: a supermartingale

    def test_mixture_is_the_weighted_average_of_likelihood_ratio_products(self):
        """The mixture from its definition, prod_t LR_t(delta_j) averaged with the prior weights."""
        tau, theta0 = 0.7, 1.3
        for alternative in ("two-sided", "less", "greater"):
            proc = AnytimeLogRankTest(theta0=theta0, alternative=alternative, prior_sd=tau, confidence_sequence=False)
            xs = inputs(trial(random.Random(3), 80, 0.7))
            _, out = trajectory(proc, xs)[-1]
            deltas = [j * tau / 8 for j in range(-32, 33) if j != 0]
            if alternative == "less":
                deltas = [d for d in deltas if d < 0]
            elif alternative == "greater":
                deltas = [d for d in deltas if d > 0]
            weights = [math.exp(-d * d / (2 * tau * tau)) for d in deltas]
            logs = []
            for d, w in zip(deltas, weights):
                theta1 = theta0 * math.exp(d)
                log_lr = 0.0
                for x, n_t, n_c in informative_events(xs):
                    p1, p0 = n_t * theta1 / (n_t * theta1 + n_c), n_t * theta0 / (n_t * theta0 + n_c)
                    log_lr += math.log(p1 / p0) if x else math.log((1 - p1) / (1 - p0))
                logs.append(math.log(w / math.fsum(weights)) + log_lr)
            with self.subTest(alternative=alternative):
                self.assertAlmostEqual(out.log_e_value, logsumexp(logs), delta=1e-9)

    def test_grow_bets_the_penalised_estimate_of_the_past(self):
        """method="grow" multiplies LR_t(d_t), d_t the estimate before event t (predictable)."""
        proc = AnytimeLogRankTest(method="grow", alternative="less", confidence_sequence=False)
        xs = inputs(trial(random.Random(4), 120, 0.6))
        state, log_e = proc.initial_state(), 0.0
        for x in xs:
            n_t, n_c = state.at_risk
            if x["kind"] == EVENT and n_t and n_c:
                d = min(proc.estimate(state.log_lik), 0.0)
                theta1 = math.exp(d)
                p1, p0 = n_t * theta1 / (n_t * theta1 + n_c), n_t / (n_t + n_c)
                log_e += math.log(p1 / p0) if x["arm"] == TREATMENT else math.log((1 - p1) / (1 - p0))
            state, out = proc.step(state, x)
            self.assertAlmostEqual(out.log_e_value, log_e, delta=1e-9)
        self.assertGreater(log_e, 1.0)


# ---------------------------------------------------------------- Monte Carlo

class MonteCarloTests(unittest.TestCase):
    def test_type_one_error_under_staggered_entry_and_censoring(self):
        """Rejection rate under H0, monitored after every event, for several baseline hazards.

        Weibull hazards are on the calendar time scale (late entry is left
        truncation); exponential hazards are the same on both scales.  See
        the module docstring for Weibull hazards in time since entry.
        """
        alpha, runs = 0.1, 250
        scenarios = [
            ("exponential", "mixture", dict(shape=1.0, accrual=2.0)),
            ("exponential, patient time", "grow", dict(shape=1.0, accrual=2.0, time_scale="patient")),
            ("weibull 0.5", "mixture", dict(shape=0.5, accrual=3.0)),
            ("weibull 2", "mixture", dict(shape=2.0, accrual=3.0, end=2.5)),
            ("weibull 2", "grow", dict(shape=2.0, accrual=3.0, end=2.5)),
        ]
        bound = binomial_upper_bound(alpha, runs)
        for label, method, kw in scenarios:
            proc = AnytimeLogRankTest(alpha, method=method, confidence_sequence=False)
            rejected = sum(first_rejection(proc, inputs(trial(random.Random(1000 + r), 200, 1.0, **kw))) is not None
                           for r in range(runs))
            with self.subTest(label, method=method):
                self.assertLessEqual(rejected / runs, bound)

    def test_power_and_events_needed_at_hazard_ratio_0_6(self):
        runs = 60
        needed = {}
        for method in ("mixture", "grow"):
            proc = AnytimeLogRankTest(method=method, confidence_sequence=False)
            needed[method] = [first_rejection(proc, inputs(trial(random.Random(r), 600, 0.6))) for r in range(runs)]
        for method, events in needed.items():
            self.assertGreaterEqual(sum(e is not None for e in events) / runs, 0.9, method)
        medians = {m: statistics.median(e if e is not None else 10**9 for e in v) for m, v in needed.items()}
        # by simulation (100 runs): about 110 events for both; a fixed-sample test needs about 120
        self.assertLessEqual(medians["grow"], 1.2 * medians["mixture"], medians)
        self.assertLess(medians["mixture"], 160, medians)

    def test_agrees_with_the_classical_logrank_statistic(self):
        """Sign and size of the evidence against z = (O - E) / sqrt(V), computed independently.

        For a normal prior N(0, tau^2) on the log hazard ratio and the normal
        approximation of the score, log E is about
        tau^2 V z^2 / (2 (1 + tau^2 V)) - log(1 + tau^2 V) / 2.  The plug-in
        estimate is checked against the penalised Cox estimate by Newton.
        """
        tau = 0.5
        proc = AnytimeLogRankTest(prior_sd=tau, confidence_sequence=False)
        for r in range(30):
            hr = (0.6, 1.0, 1.5)[r % 3]
            xs = inputs(trial(random.Random(200 + r), 300, hr))
            _, out = trajectory(proc, xs)[-1]
            events = informative_events(xs)
            u = math.fsum(x - n_t / (n_t + n_c) for x, n_t, n_c in events)
            v = math.fsum(n_t * n_c / (n_t + n_c) ** 2 for _, n_t, n_c in events)
            z = u / math.sqrt(v)
            beta = 0.0
            for _ in range(30):
                fitted = [n_t * math.exp(beta) / (n_t * math.exp(beta) + n_c) for _, n_t, n_c in events]
                score = math.fsum(x - f for (x, _, _), f in zip(events, fitted)) - beta / tau**2
                info = math.fsum(f * (1 - f) for f in fitted) + 1 / tau**2
                beta += score / info
            approx = tau**2 * v * z * z / (2 * (1 + tau**2 * v)) - 0.5 * math.log1p(tau**2 * v)
            with self.subTest(run=r, hazard_ratio=hr, z=z):
                estimate = math.log(out.hazard_ratio_estimate)
                self.assertAlmostEqual(estimate, beta, delta=5e-4)
                if abs(z) > 0.5:
                    self.assertEqual(estimate > 0, z > 0)
                self.assertLess(abs(out.log_e_value - approx), 0.3 + 0.1 * abs(approx))

    def test_confidence_sequence_covers_uniformly_over_time(self):
        alpha, runs = 0.2, 120
        bound = binomial_upper_bound(alpha, runs)
        for theta in (0.7, 2.5):
            proc = AnytimeLogRankTest(alpha)
            missed, widths = 0, []
            for r in range(runs):
                state, ever = proc.initial_state(), False
                for x in inputs(trial(random.Random(5000 + r), 200, theta)):
                    state, out = proc.step(state, x)
                    if (out.lower is not None and theta < out.lower) or (out.upper is not None and theta > out.upper):
                        ever = True
                missed += ever
                self.assertIsNotNone(out.lower)
                self.assertIsNotNone(out.upper)
                widths.append(math.log(out.upper / out.lower))
            with self.subTest(theta=theta):
                self.assertLessEqual(missed / runs, bound)
                self.assertLess(statistics.median(widths), 1.5)  # about 140 events: a useful interval


# ---------------------------------------------------------------- invariance and the confidence sequence

class InvarianceTests(unittest.TestCase):
    def test_only_the_order_of_the_records_matters(self):
        """A monotone change of the time axis leaves the order, and so every output, unchanged.

        Multiplying all times by a constant is one such change.  A non-linear
        one, t -> g(t), turns a trial with cumulative baseline hazard H(t)
        into one with H(g^-1(t)): another baseline hazard (t -> t^3 makes an
        exponential hazard Weibull with shape 1/3), and another entry and
        censoring pattern, with the same order of records.
        """
        records = trial(random.Random(10), 150, 0.6, accrual=1.0)
        reference = {m: trajectory(AnytimeLogRankTest(method=m), inputs(records)) for m in ("mixture", "grow")}
        self.assertTrue(reference["mixture"][-1][1].lower is not None)
        for transform in (lambda t: 7.3 * t, lambda t: 1e-3 * t, lambda t: t**3, lambda t: math.exp(t) - 1.0):
            changed = sorted((transform(t), k, a) for t, k, a in records)
            self.assertNotEqual([t for t, _, _ in changed], [t for t, _, _ in records])
            self.assertEqual(inputs(changed), inputs(records))
            for method, expected in reference.items():
                proc = AnytimeLogRankTest(method=method)
                got = trajectory(proc, inputs(changed))
                self.assertEqual(got, expected)
                self.assertEqual(json.dumps(proc.encode_output(got[-1][1])),
                                 json.dumps(proc.encode_output(expected[-1][1])))

    def test_confidence_sequence_e_values_against_the_definition(self):
        proc = AnytimeLogRankTest(prior_sd=0.4, cs_range=(0.2, 5.0), cs_grid=41)
        xs = inputs(trial(random.Random(11), 150, 0.6))
        events = informative_events(xs)
        state, _ = trajectory(proc, xs)[-1]

        def log_lik(beta):
            return math.fsum(x * beta - math.log(n_t * math.exp(beta) + n_c) + math.log(n_t + n_c)
                             for x, n_t, n_c in events)

        h = math.log(25.0) / 40
        c = max(1, round(0.4 / 8 / h))
        half = int(4 * 0.4 / (c * h))
        offsets = [j * c * h for j in range(-half, half + 1) if j != 0]
        logw = [-(m * m) / (2 * 0.4**2) for m in offsets]
        norm = logsumexp(logw)
        values = proc.cs_log_e_values(state.cs_log_lik, 0, 40)
        for i in range(41):
            b = math.log(0.2) + i * h
            direct = logsumexp(w - norm + log_lik(b + m) - log_lik(b) for w, m in zip(logw, offsets))
            self.assertAlmostEqual(values[i], direct, delta=1e-8 * max(1.0, abs(direct)))
        kept = [i for i, v in enumerate(values) if v < math.log(1 / proc.alpha)]
        # the running interval is never wider than today's (it is intersected over time)
        self.assertLessEqual(kept[0], state.cs_kept[0])
        self.assertGreaterEqual(kept[-1], state.cs_kept[1])

    def test_confidence_sequence_far_from_the_maximum(self):
        """Points whose whole prior window lies far below the maximum use the log-domain fallback."""
        proc = AnytimeLogRankTest(cs_grid=21)
        width = len(proc.initial_state().cs_log_lik)
        slope = -40.0  # log-likelihood falling by 40 nats per grid step: exp underflows away from the top
        lik = tuple(slope * u for u in range(width))
        values = proc.cs_log_e_values(lik, 0, 20)
        for i, v in enumerate(values):
            centre = i + proc._pad
            direct = logsumexp(w + lik[centre + m] - lik[centre] for w, m in zip(proc._cs_logw, proc._cs_offsets))
            self.assertAlmostEqual(v, direct, delta=1e-9 * abs(direct))

    def test_records_of_other_arms_and_events_with_one_arm_empty(self):
        proc = AnytimeLogRankTest(arms=(3, 2))
        state = proc.initial_state()
        state, out = proc.step(state, {"arm": 7, "kind": EVENT})  # ignored
        self.assertEqual(state, proc.initial_state())
        state, _ = proc.step(state, {"arm": 3, "kind": ENTER})
        state, out = proc.step(state, {"arm": 3, "kind": EVENT})  # nobody in control: certain, no evidence
        self.assertEqual((out.n_events, out.at_risk, out.log_e_value), (1, (0, 0), 0.0))
        self.assertEqual((out.lower, out.upper, out.hazard_ratio_estimate), (None, None, 1.0))


# ---------------------------------------------------------------- persistence and contracts

class StateAndContractTests(unittest.TestCase):
    def test_state_round_trip(self):
        xs = inputs(trial(random.Random(12), 100, 0.5))
        for kw in ({}, {"method": "grow"}, {"confidence_sequence": False, "alternative": "less"}):
            proc = AnytimeLogRankTest(**kw)
            states = trajectory(proc, xs)
            for state, _ in states[::25] + states[-1:]:
                decoded = proc.decode_state(json.loads(json.dumps(proc.encode_state(state))))
                self.assertEqual(decoded, state)
            # continuing from a decoded state is the same run
            middle = len(xs) // 2
            restored = proc.decode_state(json.loads(json.dumps(proc.encode_state(states[middle - 1][0]))))
            self.assertEqual(trajectory(proc, xs[middle:], state=restored), states[middle:])

    def test_malformed_states_are_rejected(self):
        proc = AnytimeLogRankTest(alpha=0.05)
        grow = AnytimeLogRankTest(method="grow", confidence_sequence=False)
        xs = inputs(trial(random.Random(13), 200, 0.3))
        good = proc.encode_state(trajectory(proc, xs)[-1][0])
        self.assertEqual(good["decision"], "reject_h0")
        grow_good = grow.encode_state(trajectory(grow, xs)[-1][0])

        def broken(base, **changes):
            data = json.loads(json.dumps(base))
            data.update(changes)
            return data

        cases = [
            (proc, broken(good, at_risk=[1, 2, 3])),
            (proc, broken(good, at_risk=[-1, 2])),
            (proc, broken(good, n_events=1.5)),
            (proc, broken(good, log_lik=good["log_lik"][:-1])),
            (proc, broken(good, log_lik=[1.0] * len(good["log_lik"]))),  # the null point must be 0
            (proc, broken(good, log_e=0.3)),  # a mixture keeps no plug-in product
            (proc, broken(good, cs_log_lik=good["cs_log_lik"][1:])),
            (proc, broken(good, cs_kept=[5, 4])),
            (proc, broken(good, cs_kept=[0, proc.cs_grid])),
            (proc, broken(good, decision=None)),  # rejected, by the running maximum
            (proc, broken(good, decision="accept_h0")),
            (proc, broken(good, max_log_e=0.1)),
            (proc, broken(good, max_log_e=float("nan"))),
            (proc, broken(good, n_events=0)),
            (proc, broken(good, extra=1)),
            (grow, broken(grow_good, cs_kept=[0, 1])),  # no confidence sequence
            (grow, broken(grow_good, cs_log_lik=[0.0])),
            (proc, [1, 2]),
        ]
        for i, (p, data) in enumerate(cases):
            with self.subTest(case=i):
                with self.assertRaises((ValueError, TypeError)):
                    p.decode_state(data)

    def test_events_and_censorings_need_someone_at_risk(self):
        proc = AnytimeLogRankTest()
        state = proc.initial_state()
        state, _ = proc.step(state, {"arm": CONTROL, "kind": ENTER})
        for x in ({"arm": TREATMENT, "kind": EVENT}, {"arm": TREATMENT, "kind": CENSOR}):
            with self.assertRaises(ContractViolation):
                proc.step(state, x)
        state, out = proc.step(state, {"arm": CONTROL, "kind": CENSOR})
        self.assertEqual(out.at_risk, (0, 0))
        with self.assertRaises(ContractViolation):
            proc.step(state, {"arm": CONTROL, "kind": EVENT})

    def test_runs_on_joined_streams_and_skips_impossible_records(self):
        """Through Run: arm and kind joined by key; an impossible record is skipped as invalid input."""
        xs = inputs(trial(random.Random(16), 100, 0.6))
        observations = []
        for i, x in enumerate([{"arm": TREATMENT, "kind": EVENT}, *xs]):  # an event before anyone entered
            observations += [Observation(name, value, key=f"r{i}") for name, value in x.items()]
        proc = AnytimeLogRankTest()
        run = Run(proc, topology=KeyJoin(("arm", "kind")), run_id="logrank")
        with self.assertLogs("sequential_analysis", level="WARNING") as logs:
            run_sync(run, observations)
        self.assertIn("nobody at risk", "\n".join(logs.output))
        self.assertEqual(run.state, trajectory(proc, xs)[-1][0])

    def test_input_contracts(self):
        proc = AnytimeLogRankTest()
        for bad in ({"arm": 1, "kind": 3}, {"arm": 1, "kind": -1}, {"arm": 1, "kind": 0.5}, {"arm": -1, "kind": 0},
                    {"arm": 1}, {"arm": True, "kind": 0}):
            with self.subTest(x=bad), self.assertRaises(ContractViolation):
                trajectory(proc, [bad])

    def test_argument_validation(self):
        bad = [
            dict(alpha=0.0), dict(alpha=1.0), dict(theta0=0.0), dict(theta0=float("inf")), dict(theta0=True),
            dict(alternative="lower"), dict(method="universal"), dict(prior_sd=0.0), dict(prior_sd=float("nan")),
            dict(prior_sd=10.0), dict(confidence_sequence=1), dict(cs_range=(1.0, 1.0)), dict(cs_range=(0.0, 2.0)),
            dict(cs_range=(0.5,)), dict(cs_grid=2), dict(cs_grid=10.0), dict(arms=(1, 1)), dict(arms=(1, -1)),
            dict(arms=(1, 0, 2)), dict(arm_input="x", input="x"),
        ]
        for kw in bad:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}), self.assertRaises(ValueError):
                AnytimeLogRankTest(**kw)

    def test_extreme_prior_widths(self):
        """The smallest and largest prior_sd still give finite e-values and an interval."""
        xs = inputs(trial(random.Random(15), 150, 0.5))
        for tau in (0.01, 5.0):
            for method in ("mixture", "grow"):
                proc = AnytimeLogRankTest(prior_sd=tau, method=method, cs_grid=21)
                state, out = trajectory(proc, xs)[-1]
                self.assertTrue(math.isfinite(out.log_e_value))
                self.assertEqual(proc.decode_state(json.loads(json.dumps(proc.encode_state(state)))), state)
                if out.lower is not None and out.upper is not None:
                    self.assertLess(out.lower, out.upper)

    def test_identity_and_terminal_state(self):
        proc = AnytimeLogRankTest(stop_on_reject=True)
        self.assertEqual(proc.identity()["name"], "anytime_logrank_test")
        self.assertEqual(proc.identity()["version"], "1")
        json.dumps(proc.config())
        xs = inputs(trial(random.Random(14), 300, 0.3))
        steps = trajectory(proc, xs)
        self.assertLess(len(steps), len(xs))  # stopped at the rejection
        self.assertTrue(proc.is_terminal(steps[-1][0]))
        self.assertEqual(steps[-1][1].p_value, math.exp(-steps[-1][0].max_log_e))
        self.assertLessEqual(steps[-1][1].p_value, proc.alpha)


# ---------------------------------------------------------------- stratification

def cohort_trial(rnd, patients, hazard_ratio, *, shape=0.3, accrual=25.0, censoring=0.3):
    """(time, kind, arm, entry) records with Weibull hazards in time since entry, sorted by calendar time.

    The setup of the module docstring's time-scale paragraph: entry uniform
    over `accrual` scale units, censoring exponential after entry.
    """
    records = []
    for _ in range(patients):
        e = rnd.uniform(0.0, accrual)
        arm = TREATMENT if rnd.random() < 0.5 else CONTROL
        g = rnd.expovariate(1.0) / (hazard_ratio if arm == TREATMENT else 1.0)
        t, c = e + g ** (1.0 / shape), e + rnd.expovariate(censoring)
        records.append((e, ENTER, arm, e))
        records.append((t, EVENT, arm, e) if t <= c else (c, CENSOR, arm, e))
    records.sort()
    return records


def cohort_inputs(records, cohorts, accrual=25.0):
    """Stratum = entry cohort: the accrual period cut into `cohorts` equal periods."""
    return [{"arm": a, "kind": k, "stratum": min(cohorts - 1, int(e / accrual * cohorts))} for _, k, a, e in records]


def centre_trial(rnd, patients, hazard_ratio, rates, allocation, *, accrual=2.0, censoring=0.3):
    """Records {arm, kind, stratum} of a trial in several centres, in calendar order.

    Centre s (chosen uniformly) has exponential baseline hazard rates[s] and
    randomises to treatment with probability allocation[s].
    """
    records = []
    for _ in range(patients):
        s = rnd.randrange(len(rates))
        e = rnd.uniform(0.0, accrual)
        arm = TREATMENT if rnd.random() < allocation[s] else CONTROL
        t = e + rnd.expovariate(rates[s] * (hazard_ratio if arm == TREATMENT else 1.0))
        c = e + rnd.expovariate(censoring)
        records.append((e, ENTER, arm, s))
        records.append((t, EVENT, arm, s) if t <= c else (c, CENSOR, arm, s))
    records.sort()
    return [{"arm": a, "kind": k, "stratum": s} for _, k, a, s in records]


def unstratified(xs):
    return [{"arm": x["arm"], "kind": x["kind"]} for x in xs]


def stratified_informative_events(xs):
    """(x, n_T, n_C) of every event with both arms of its stratum at risk, by an independent count."""
    at_risk = {}
    out = []
    for x in xs:
        counts = at_risk.setdefault(x["stratum"], {TREATMENT: 0, CONTROL: 0})
        if x["kind"] == ENTER:
            counts[x["arm"]] += 1
            continue
        if x["kind"] == EVENT and counts[TREATMENT] and counts[CONTROL]:
            out.append((1 if x["arm"] == TREATMENT else 0, counts[TREATMENT], counts[CONTROL]))
        counts[x["arm"]] -= 1
    return out


#: Two strata: entries in both, then events, a censoring and an entry; "busiest"
#: is the stratum with more patients at risk, a choice that depends on the past.
STRATIFIED_SCHEDULE = ([("enter-t", 0)] * 2 + [("enter-c", 0)] * 3 + [("enter-t", 1)] * 3 + [("enter-c", 1)]
                       + [("event", 0), ("event", 1), ("censor-larger", 0), ("event", "busiest"),
                          ("enter-smaller", 1), ("event", 1), ("event", 0), ("event", "busiest"),
                          ("event", "busiest")])
STRATIFIED_EVENTS = sum(op == "event" for op, _ in STRATIFIED_SCHEDULE)


def stratified_expectations(proc, theta, stop_at=None, at_end=None):
    """As `expectations`, for a two-stratum procedure: the arm of an event in stratum s is
    Bernoulli(n_T^s theta / (n_T^s theta + n_C^s)).  `at_end(state)`, if given, is averaged
    over the final states too (third value returned)."""
    by_n = [0.0] * (STRATIFIED_EVENTS + 1)
    totals = [0.0, 0.0]

    def walk(state, i, prob, frozen):
        if i == len(STRATIFIED_SCHEDULE):
            totals[0] += prob * (math.exp(proc.log_e_value(state)) if frozen is None else frozen)
            if at_end is not None:
                totals[1] += prob * at_end(state)
            return
        op, s = STRATIFIED_SCHEDULE[i]
        if s == "busiest":
            s = 0 if sum(state.at_risk[0:2]) >= sum(state.at_risk[2:4]) else 1
        n_t, n_c = state.at_risk[2 * s], state.at_risk[2 * s + 1]
        if op == "event":
            p = n_t * theta / (n_t * theta + n_c)
            for arm, q in ((TREATMENT, p), (CONTROL, 1.0 - p)):
                if q == 0.0:
                    continue
                new, out = proc.step(state, {"arm": arm, "kind": EVENT, "stratum": s})
                e = math.exp(out.log_e_value)
                by_n[out.n_events] += prob * q * e
                stop = frozen is None and stop_at is not None and out.log_e_value >= stop_at
                walk(new, i + 1, prob * q, e if stop else frozen)
            return
        if op == "enter-t":
            arm, kind = TREATMENT, ENTER
        elif op == "enter-c":
            arm, kind = CONTROL, ENTER
        elif op == "censor-larger":
            arm, kind = (TREATMENT if n_t >= n_c else CONTROL), CENSOR
        else:
            arm, kind = (TREATMENT if n_t <= n_c else CONTROL), ENTER
        new, _ = proc.step(state, {"arm": arm, "kind": kind, "stratum": s})
        walk(new, i + 1, prob, frozen)

    walk(proc.initial_state(), 0, 1.0, None)
    return by_n[1:], totals[0], totals[1]


class StratifiedExactTests(unittest.TestCase):
    def procedures(self, **kw):
        for method in ("mixture", "grow"):
            for alternative in ("two-sided", "less", "greater"):
                for theta0 in (1.0, 1.7):
                    yield AnytimeLogRankTest(theta0=theta0, alternative=alternative, method=method, prior_sd=1.0,
                                             confidence_sequence=False, strata=2, **kw)

    def test_each_factor_has_mean_one_under_the_null_in_every_stratum(self):
        """One event from a state with different risk sets in three strata: E[e-value] = 1 for an event in any."""
        counts = ((3, 1), (1, 4), (6, 2))
        for theta0 in (1.0, 0.4, 2.5):
            for alternative in ("two-sided", "less", "greater"):
                proc = AnytimeLogRankTest(theta0=theta0, alternative=alternative, strata=3)
                state = proc.initial_state()
                for s, (n_t, n_c) in enumerate(counts):
                    for arm, n in ((TREATMENT, n_t), (CONTROL, n_c)):
                        for _ in range(n):
                            state, _ = proc.step(state, {"arm": arm, "kind": ENTER, "stratum": s})
                for s, (n_t, n_c) in enumerate(counts):
                    p = n_t * theta0 / (n_t * theta0 + n_c)
                    _, treated = proc.step(state, {"arm": TREATMENT, "kind": EVENT, "stratum": s})
                    _, control = proc.step(state, {"arm": CONTROL, "kind": EVENT, "stratum": s})
                    with self.subTest(theta0=theta0, alternative=alternative, stratum=s):
                        self.assertNotAlmostEqual(treated.log_e_value, 0.0, delta=1e-3)
                        mean = p * math.exp(treated.log_e_value) + (1 - p) * math.exp(control.log_e_value)
                        self.assertAlmostEqual(mean, 1.0, delta=1e-14)

    def test_expected_e_value_is_one_under_the_null(self):
        for proc in self.procedures():
            with self.subTest(**proc.config()):
                by_n, at_stop, _ = stratified_expectations(proc, proc.theta0, stop_at=math.log(2.0))
                for n, mean in enumerate(by_n, 1):
                    self.assertAlmostEqual(mean, 1.0, delta=1e-12, msg=f"after {n} events")
                self.assertAlmostEqual(at_stop, 1.0, delta=1e-12, msg="at the stopping time")

    def test_one_sided_nulls_and_alternatives(self):
        for proc in self.procedures():
            if proc.alternative == "two-sided":
                continue
            inside = proc.theta0 * (2.0 if proc.alternative == "less" else 0.5)
            outside = proc.theta0 * (0.3 if proc.alternative == "less" else 3.0)
            with self.subTest(**proc.config()):
                by_n, at_stop, _ = stratified_expectations(proc, inside, stop_at=math.log(2.0))
                self.assertLessEqual(max(by_n), 1.0 + 1e-12)
                self.assertLessEqual(at_stop, 1.0 + 1e-12)
                self.assertLess(by_n[-1], 1.0 - 1e-6)
                self.assertGreater(stratified_expectations(proc, outside)[0][-1], 1.05)

    def test_confidence_sequence_e_values_are_martingales_at_every_grid_point(self):
        """E[E_n(theta_i)] = 1 when theta = theta_i, for the mixture behind the confidence sequence."""
        proc = AnytimeLogRankTest(prior_sd=1.0, cs_range=(0.25, 4.0), cs_grid=9, strata=2)
        for i in (0, 3, 8):
            theta = proc._cs_points[i]
            _, _, mean = stratified_expectations(
                proc, theta, at_end=lambda state, i=i: math.exp(proc.cs_log_e_values(state.cs_log_lik, i, i)[0]))
            with self.subTest(theta=theta):
                self.assertAlmostEqual(mean, 1.0, delta=1e-12)


class StratifiedTests(unittest.TestCase):
    def test_one_stratum_is_the_unstratified_test_as_released(self):
        """strata=1: the identity, inputs and state of the unstratified test; its conformance vectors bit for bit."""
        default, explicit = AnytimeLogRankTest(), AnytimeLogRankTest(strata=1, stratum_input="centre")
        self.assertEqual(explicit.identity(), default.identity())
        self.assertNotIn("strata", explicit.config())
        self.assertEqual(set(explicit.inputs()), {"arm", "kind"})
        self.assertEqual(explicit.encode_state(explicit.initial_state()), default.encode_state(default.initial_state()))
        root = Path(__file__).resolve().parent.parent
        vectors = [v for v in json.loads((root / "conformance" / "procedures.json").read_text())
                   if v["procedure"]["name"] == "anytime_logrank_test"]
        self.assertGreaterEqual(len(vectors), 2)
        for v in vectors:
            proc = AnytimeLogRankTest(**v["procedure"]["config"], strata=1)
            with self.subTest(case=v["case"]):
                self.assertEqual(proc.identity(), v["procedure"])
                steps = trajectory(proc, v["inputs"])
                outputs = [proc.encode_output(o) for _, o in steps]
                self.assertEqual(canonical_json(outputs), canonical_json(v["outputs"]))
                self.assertEqual(canonical_json(proc.encode_state(steps[-1][0])), canonical_json(v["state"]))

    def test_records_all_in_one_stratum_give_the_unstratified_test(self):
        xs = inputs(trial(random.Random(20), 150, 0.6))
        for method in ("mixture", "grow"):
            plain = trajectory(AnytimeLogRankTest(method=method), xs)
            for stratum in (0, 3):
                proc = AnytimeLogRankTest(method=method, strata=4)
                got = trajectory(proc, [{**x, "stratum": stratum} for x in xs])
                with self.subTest(method=method, stratum=stratum):
                    self.assertEqual([o for _, o in got], [o for _, o in plain])
                    for (state, _), (reference, _) in zip(got, plain):
                        self.assertEqual(state.at_risk[2 * stratum:2 * stratum + 2], reference.at_risk)
                        self.assertEqual(sum(state.at_risk), sum(reference.at_risk))
                        self.assertEqual(state.log_lik, reference.log_lik)
                        self.assertEqual(state.cs_log_lik, reference.cs_log_lik)

    def test_stratified_partial_likelihood_from_its_definition(self):
        """The mixture and the estimate against the stratified Cox partial likelihood, computed independently."""
        tau = 0.5
        xs = centre_trial(random.Random(21), 300, 0.6, (0.2, 1.0, 5.0), (0.3, 0.5, 0.7))
        events = stratified_informative_events(xs)
        self.assertNotEqual(events, informative_events(unstratified(xs)))
        proc = AnytimeLogRankTest(prior_sd=tau, confidence_sequence=False, strata=3)
        _, out = trajectory(proc, xs)[-1]
        self.assertEqual(out.n_events, sum(x["kind"] == EVENT for x in xs))
        deltas = [j * tau / 8 for j in range(-32, 33) if j != 0]
        weights = [math.exp(-d * d / (2 * tau * tau)) for d in deltas]
        logs = []
        for d, w in zip(deltas, weights):
            log_lr = math.fsum(x * d - math.log(n_t * math.exp(d) + n_c) + math.log(n_t + n_c)
                               for x, n_t, n_c in events)
            logs.append(math.log(w / math.fsum(weights)) + log_lr)
        self.assertAlmostEqual(out.log_e_value, logsumexp(logs), delta=1e-9)
        beta = 0.0
        for _ in range(30):
            fitted = [n_t * math.exp(beta) / (n_t * math.exp(beta) + n_c) for _, n_t, n_c in events]
            score = math.fsum(x - f for (x, _, _), f in zip(events, fitted)) - beta / tau**2
            beta += score / (math.fsum(f * (1 - f) for f in fitted) + 1 / tau**2)
        self.assertAlmostEqual(math.log(out.hazard_ratio_estimate), beta, delta=5e-4)
        self.assertLess(beta, 0.0)

    def test_relabelling_the_strata_changes_nothing(self):
        xs = centre_trial(random.Random(22), 120, 0.7, (0.5, 1.0, 2.0), (0.5, 0.5, 0.5))
        proc = AnytimeLogRankTest(strata=3)
        reference = [o for _, o in trajectory(proc, xs)]
        relabelled = [{**x, "stratum": (2, 0, 1)[x["stratum"]]} for x in xs]
        self.assertEqual([o for _, o in trajectory(proc, relabelled)], reference)

    def test_entry_cohorts_against_hazards_in_time_since_entry(self):
        """The module docstring's case: Weibull shape 0.3 in time since entry, 200 patients entering over 25 units.

        Rejection rates of a true H0 at alpha = 0.05 measured with 4000 trials
        of cohort_trial (random.Random(seed), seeds 0..3999): unstratified 21%,
        8 entry cohorts 11%, 25 cohorts 3.8%; an exact analysis (the records
        sorted by time since entry, seeds 0..1999) 1.7%.  Stratifying by fine enough cohorts brings the test back
        under its level, but not to exactness: within a cohort entry still varies.
        """
        runs = 150
        rejected = {1: 0, 8: 0, 25: 0}
        for r in range(runs):
            records = cohort_trial(random.Random(7000 + r), 200, 1.0)
            for cohorts in rejected:
                proc = AnytimeLogRankTest(strata=cohorts, confidence_sequence=False)
                xs = cohort_inputs(records, cohorts)
                rejected[cohorts] += first_rejection(proc, unstratified(xs) if cohorts == 1 else xs) is not None
        rates = {k: v / runs for k, v in rejected.items()}
        self.assertGreater(rates[1], 0.12, rates)  # badly anti-conservative
        self.assertLess(rates[25], rates[8], rates)
        self.assertLess(rates[8], rates[1], rates)
        self.assertLessEqual(rates[25], binomial_upper_bound(0.05, runs), rates)

    def test_centres_with_different_baseline_hazards(self):
        """Stratified by centre, the test keeps its level where the unstratified test is confounded, and has more power.

        Measured with 600 trials of centre_trial (seeds 0..599): with allocation 1:3, 1:1, 3:1 in centres with
        hazards 0.2, 1 and 5, a true H0 is rejected in 96% of trials
        unstratified and 2.5% stratified; with 1:1 allocation and hazards 0.1
        and 10, power at a hazard ratio of 0.6 (400 patients) is 18%
        unstratified and 82% stratified.
        """
        runs = 120
        confounded = [centre_trial(random.Random(8000 + r), 200, 1.0, (0.2, 1.0, 5.0), (0.25, 0.5, 0.75))
                      for r in range(runs)]
        stratified = AnytimeLogRankTest(strata=3, confidence_sequence=False)
        plain = AnytimeLogRankTest(confidence_sequence=False)
        level = sum(first_rejection(stratified, xs) is not None for xs in confounded) / runs
        self.assertLessEqual(level, binomial_upper_bound(0.05, runs))
        self.assertGreater(sum(first_rejection(plain, unstratified(xs)) is not None for xs in confounded[:20]), 10)

        runs = 30
        effect = [centre_trial(random.Random(9000 + r), 400, 0.6, (0.1, 10.0), (0.5, 0.5)) for r in range(runs)]
        stratified = AnytimeLogRankTest(strata=2, confidence_sequence=False)
        power = {"stratified": sum(first_rejection(stratified, xs) is not None for xs in effect) / runs,
                 "unstratified": sum(first_rejection(plain, unstratified(xs)) is not None for xs in effect) / runs}
        self.assertGreaterEqual(power["stratified"], 0.6, power)
        self.assertGreaterEqual(power["stratified"], power["unstratified"] + 0.3, power)

    def test_stratified_state_round_trip(self):
        xs = centre_trial(random.Random(23), 150, 0.6, (0.3, 1.0, 3.0), (0.5, 0.4, 0.6))
        for kw in ({}, {"method": "grow"}, {"confidence_sequence": False, "alternative": "less"}):
            proc = AnytimeLogRankTest(strata=3, **kw)
            states = trajectory(proc, xs)
            for state, _ in states[::30] + states[-1:]:
                encoded = json.loads(json.dumps(proc.encode_state(state)))
                self.assertEqual(len(encoded["at_risk"]), 6)
                self.assertEqual(proc.decode_state(encoded), state)
            middle = len(xs) // 2
            restored = proc.decode_state(json.loads(json.dumps(proc.encode_state(states[middle - 1][0]))))
            self.assertEqual(trajectory(proc, xs[middle:], state=restored), states[middle:])

    def test_checkpoint_and_restore_through_run(self):
        xs = centre_trial(random.Random(24), 120, 0.6, (0.3, 3.0), (0.5, 0.5))
        observations = []
        for i, x in enumerate(xs):
            observations += [Observation(name, value, key=f"r{i}") for name, value in x.items()]

        def make():
            return Run(AnytimeLogRankTest(strata=2, stratum_input="stratum"),
                       topology=KeyJoin(("arm", "kind", "stratum")), run_id="strata", clock=lambda: 0.0)

        full = make()
        run_sync(full, observations)
        self.assertEqual(full.state, trajectory(full.procedure, xs)[-1][0])
        first = make()
        run_sync(first, observations[:len(observations) // 2])
        checkpoint = json.loads(json.dumps(first.checkpoint()))
        self.assertEqual(checkpoint["procedure"]["config"]["strata"], 2)
        second = Run.restore(AnytimeLogRankTest(strata=2), checkpoint, topology=KeyJoin(("arm", "kind", "stratum")),
                             clock=lambda: 0.0)
        run_sync(second, observations[len(observations) // 2:])
        self.assertEqual(second.state, full.state)
        self.assertEqual(second.history_digest, full.history_digest)
        with self.assertRaises(IncompatibleCheckpoint):  # another number of strata is another procedure
            Run.restore(AnytimeLogRankTest(strata=3), checkpoint, topology=KeyJoin(("arm", "kind", "stratum")))

    def test_malformed_stratified_states_are_rejected(self):
        proc = AnytimeLogRankTest(strata=3)
        xs = centre_trial(random.Random(25), 100, 0.6, (0.3, 1.0, 3.0), (0.5, 0.5, 0.5))
        good = proc.encode_state(trajectory(proc, xs)[-1][0])
        unstratified_state = AnytimeLogRankTest().encode_state(AnytimeLogRankTest().initial_state())
        for i, bad in enumerate([good["at_risk"][:2], good["at_risk"][:5], good["at_risk"] + [0, 0],
                                 [-1, *good["at_risk"][1:]], [0.5, *good["at_risk"][1:]]]):
            with self.subTest(case=i), self.assertRaises((ValueError, TypeError)):
                proc.decode_state({**good, "at_risk": bad})
        with self.assertRaises(ValueError):
            proc.decode_state(unstratified_state)
        with self.assertRaises(ValueError):
            AnytimeLogRankTest().decode_state(good)

    def test_stratum_contracts(self):
        proc = AnytimeLogRankTest(strata=2)
        self.assertEqual(set(proc.inputs()), {"arm", "kind", "stratum"})
        for bad in ({"arm": 1, "kind": 0}, {"arm": 1, "kind": 0, "stratum": 2}, {"arm": 1, "kind": 0, "stratum": -1},
                    {"arm": 1, "kind": 0, "stratum": 0.5}, {"arm": 1, "kind": 0, "stratum": True}):
            with self.subTest(x=bad), self.assertRaises(ContractViolation):
                trajectory(proc, [bad])
        state = proc.initial_state()
        for bad in (2, -1, 0.5, True, float("nan"), "0"):  # step itself refuses, without the contracts of Run
            with self.subTest(stratum=bad), self.assertRaises(ContractViolation):
                proc.step(state, {"arm": 1, "kind": 0, "stratum": bad})
        state, out = proc.step(state, {"arm": TREATMENT, "kind": ENTER, "stratum": 1.0})  # an integral float is fine
        self.assertEqual(state.at_risk, (0, 0, 1, 0))
        state, _ = proc.step(state, {"arm": CONTROL, "kind": ENTER, "stratum": 0})
        self.assertEqual(out.at_risk, (1, 0))
        for x in ({"arm": TREATMENT, "kind": EVENT, "stratum": 0}, {"arm": CONTROL, "kind": CENSOR, "stratum": 1}):
            with self.subTest(x=x), self.assertRaisesRegex(ContractViolation, f"stratum {x['stratum']} with nobody"):
                proc.step(state, x)
        state, out = proc.step(state, {"arm": TREATMENT, "kind": EVENT, "stratum": 1})  # alone in its stratum
        self.assertEqual((out.n_events, out.at_risk, out.log_e_value), (1, (0, 1), 0.0))
        state, out = proc.step(state, {"arm": 7, "kind": EVENT, "stratum": 1})  # another arm: ignored
        self.assertEqual(out.n_events, 1)

    def test_strata_argument_validation(self):
        for kw in (dict(strata=0), dict(strata=1.0), dict(strata=True), dict(strata=10_001),
                   dict(strata=2, stratum_input="arm"), dict(strata=2, stratum_input="kind")):
            with self.subTest(**{k: repr(v) for k, v in kw.items()}), self.assertRaises(ValueError):
                AnytimeLogRankTest(**kw)
        proc = AnytimeLogRankTest(strata=5, stratum_input="centre")
        self.assertEqual((proc.config()["strata"], proc.config()["stratum_input"]), (5, "centre"))
        self.assertEqual(proc.identity()["version"], "1")
        self.assertEqual(proc.inputs()["centre"].upper, 4)
        self.assertEqual(AnytimeLogRankTest(strata=1, stratum_input="arm").config(), AnytimeLogRankTest().config())


@unittest.skipUnless((NODE and CLI.exists()) or os.environ.get("REQUIRE_TYPESCRIPT"),
                     "node or the built TypeScript executor (ts/dist) is missing")
class StratifiedTypeScriptTests(unittest.TestCase):
    def test_a_stratified_run_moves_between_executors(self):
        """Python, then the TypeScript executor from a checkpoint, then Python again (as tests/test_typescript.py)."""
        xs = centre_trial(random.Random(26), 90, 0.5, (0.3, 1.0, 3.0), (0.5, 0.4, 0.6))
        observations = []
        for i, x in enumerate(xs):
            observations += [Observation(name, value, key=i) for name, value in x.items()]
        for kw in ({}, {"method": "grow", "alternative": "less", "confidence_sequence": False}):
            def make(kw=kw):
                return AnytimeLogRankTest(strata=3, stratum_input="stratum", cs_grid=41, **kw)

            def topology():
                return KeyJoin(("arm", "kind", "stratum"))

            with self.subTest(**kw), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                full_rec, rec = Recorder(), Recorder()
                full = Run(make(), topology=topology(), consumers=[full_rec], run_id="r", clock=lambda: 0.0)
                run_sync(full, observations, stop_on_terminal=False)
                a, b = len(observations) // 3, 2 * len(observations) // 3
                first = Run(make(), topology=topology(), consumers=[rec], run_id="r", clock=lambda: 0.0)
                run_sync(first, observations[:a], stop_on_terminal=False)
                (tmp / "cp.json").write_text(json.dumps(first.checkpoint()))
                (tmp / "obs.jsonl").write_text("".join(json.dumps(o.to_json()) + "\n" for o in observations[a:b]))
                done = subprocess.run(
                    [NODE, str(CLI), "continue", str(tmp / "cp.json"), str(tmp / "obs.jsonl"),
                     "--checkpoint", str(tmp / "cp2.json"), "--outputs", str(tmp / "out.jsonl")],
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(done.returncode, 0, done.stderr)
                ts_outputs = [json.loads(line)["output"] for line in (tmp / "out.jsonl").read_text().splitlines()]
                second = Run.restore(make(), json.loads((tmp / "cp2.json").read_text()), topology=topology(),
                                     consumers=[rec], clock=lambda: 0.0)
                run_sync(second, observations[b:], stop_on_terminal=False)
                proc = full.procedure
                self.assertEqual(second.history_digest, full.history_digest)
                self.assertTrue(equivalent(json.loads(json.dumps(proc.encode_state(second.state))),
                                           json.loads(json.dumps(proc.encode_state(full.state))),
                                           abs_tol=1e-12, rel_tol=1e-12))
                expected = json.loads(json.dumps([proc.encode_output(o) for o in full_rec.outputs]))
                got = json.loads(json.dumps([proc.encode_output(o) for o in rec.outputs[:first.t]])) + ts_outputs \
                    + json.loads(json.dumps([proc.encode_output(o) for o in rec.outputs[first.t:]]))
                self.assertEqual(len(got), len(expected))
                self.assertGreater(len(ts_outputs), 20)
                self.assertTrue(equivalent(got, expected, abs_tol=1e-12, rel_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
