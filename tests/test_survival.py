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
"""
import json
import math
import random
import statistics
import unittest

from sequential_analysis import KeyJoin, Observation, Run, run_sync
from sequential_analysis.core import trajectory
from sequential_analysis.errors import ContractViolation
from sequential_analysis.numerics import logsumexp
from sequential_analysis.procedures.survival import AnytimeLogRankTest

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


if __name__ == "__main__":
    unittest.main()
