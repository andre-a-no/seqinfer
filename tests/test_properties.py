# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Property-based tests: the invariants of the formats and procedures, on generated inputs.

The other tests check the invariants on fixed histories; here hypothesis
generates the histories, the split points and the values, extreme ones
included, and shrinks a failure to a minimal example.

* checkpoint equivalence at any split, for every procedure of
  tests/helpers.py, with values anywhere in the input contracts; and, at
  the level of the transition, a decoded state continues exactly as the
  original at every step;
* properties of the outputs: always-valid p-values never increase and are
  1 / the running maximum of the e-process, running intersections are
  nested, decisions are kept, a terminal run takes nothing more;
* canonical JSON against a reference implementation of RFC 8785 and of
  ECMAScript's Number.prototype.toString;
* delivery and topologies: any arrival order and duplication the policy
  allows gives the in-order result;
* SplitMix64 against the specification, the t distribution, Neumaier
  summation, logsumexp and e-BH.

Bugs found this way and not fixed yet are expected failures with a minimal
reproduction (KnownBugs, at the end); the generators keep away from them
(`_AVOID`, `_HUGE_INTEGER`) until they are fixed.

hypothesis is a test dependency only (``pip install hypothesis``); without
it this module is skipped.  The examples are chosen by a profile:

* ``ci`` (the default): few examples, derandomized -- the same examples on
  every run, so that CI is reproducible and a failure can be re-run;
* ``explore``: ten times as many examples, from a random seed, to look for
  new failures (``HYPOTHESIS_PROFILE=explore python -m unittest
  tests.test_properties``).  A failure prints the example and the
  ``@reproduce_failure`` blob.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import struct
import unittest
from decimal import Decimal
from fractions import Fraction

from sequential_analysis import (
    Delivery,
    InputContract,
    KeyJoin,
    LifecycleError,
    NumericalError,
    Observation,
    OrderingError,
    PositionalPair,
    Recorder,
    Run,
    SplitMix64,
    TimeAlign,
    difference,
    trajectory,
)
from sequential_analysis.canonical import canonical_json
from sequential_analysis.distributions import _log_normal_upper, _log_upper_tail, regularized_beta, t_cdf, t_isf_log
from sequential_analysis.numerics import equivalent, logsumexp, neumaier_add
from sequential_analysis.procedures import SPRT, Gaussian, e_bh
from sequential_analysis.sources import skip_to

from .helpers import cases, fixed_clock

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError:
    raise unittest.SkipTest("hypothesis is not installed (pip install hypothesis)") from None

# --------------------------------------------------------------------------------------------------- profiles

_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "ci")
if _PROFILE not in ("ci", "explore"):
    raise ValueError(f"HYPOTHESIS_PROFILE must be 'ci' or 'explore', got {_PROFILE!r}")
_SCALE = 10 if _PROFILE == "explore" else 1


def _settings(examples: int):
    """The profile's settings for a test that runs `examples` examples in the ci profile.

    No deadline: the time of a step grows with the run for some procedures, and a shared CI runner is
    slow at times.  No example database: in CI it starts empty on every run, and the source tree stays clean.
    """
    return settings(
        max_examples=examples * _SCALE,
        deadline=None,
        derandomize=_PROFILE == "ci",
        database=None,
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much],
    )


@contextlib.contextmanager
def _quiet():
    """Generated histories hold invalid inputs; their warnings are expected, not news."""
    log = logging.getLogger("sequential_analysis")
    disabled, log.disabled = log.disabled, True
    try:
        yield
    finally:
        log.disabled = disabled


def _floats(value):
    """Numbers as floats, as canonical JSON has them (1 is 1.0); booleans stay booleans."""
    if isinstance(value, (list, tuple)):
        return [_floats(v) for v in value]
    if isinstance(value, dict):
        return {k: _floats(v) for k, v in value.items()}
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _text(value) -> str:
    """JSON text for the comparison of 'bit for bit': every bit of a float, 0.0 and -0.0 different.

    An integer and the float of the same value are one number, as in canonical JSON.
    """
    return json.dumps(_floats(value), sort_keys=True)


# ------------------------------------------------------------------------- generated values within a contract

_MAX = 1.7976931348623157e308
_EXTREME = (0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1e-300, -1e-300, 1e-150, -1e-150, 1e150, -1e150,
            1e300, -1e300, _MAX, -_MAX, 2.0**53, -(2.0**53), 1e16 + 2.0)


def _reals(lower, upper):
    if lower is None and upper is None:
        return st.one_of(
            st.floats(-1e3, 1e3),
            st.floats(allow_nan=False, allow_infinity=False),
            st.sampled_from(_EXTREME),
        )
    lo = -_MAX if lower is None else float(lower)
    hi = _MAX if upper is None else float(upper)
    width = hi - lo if math.isfinite(hi - lo) else _MAX
    near = st.one_of(st.floats(lo, min(hi, lo + 1e-9 * width)), st.floats(max(lo, hi - 1e-9 * width), hi))
    return st.one_of(st.sampled_from([b for b in (lower, upper) if b is not None]), st.floats(lo, hi), near)


#: The largest integer input generated.  2**53 is valid, but a procedure that sums such counts commits a
#: state no checkpoint can hold (KnownBugs); raise it to 2**53
#: when that is fixed.
_HUGE_INTEGER = 2**40


def _integers(lower, upper):
    lo = -20 if lower is None else math.ceil(lower)
    hi = lo + 40 if upper is None else math.floor(upper)
    typical = st.integers(lo, hi)
    edges = st.sampled_from([v for v in (lo, hi, _HUGE_INTEGER, -_HUGE_INTEGER) if (lower is None or v >= lower)
                             and (upper is None or v <= upper)])
    numbers = st.one_of(typical, typical, edges)
    return st.one_of(numbers, numbers.map(float))  # 3.0 is an integer value too


def _scalar(contract: InputContract):
    if contract.kind == "binary":
        return st.sampled_from([0, 1, 0.0, 1.0])
    if contract.kind == "integer":
        return _integers(contract.lower, contract.upper)
    return _reals(contract.lower, contract.upper)


def _values(contract: InputContract):
    """Any value the contract accepts: in its kind and bounds, boundaries and extreme magnitudes included."""
    if not contract.shape:
        return _scalar(contract)
    d = contract.shape[0]
    return st.lists(_scalar(contract), min_size=d, max_size=d)


def _contract(procedure, name: str) -> InputContract:
    """The contract of an observed input; inputs that a topology maps (a, b -> a - b) take any real."""
    return procedure.inputs().get(name) or InputContract(name)


#: Valid values left out because they hit a known bug -- (procedure, input): keep the value if true.
#: Information below about 1e-4 of max_information makes GroupSequentialMonitor's integration grid
#: explode (KnownBugs): a hang, or all memory.  Remove when fixed.
#: Covariates below about 1e-154 make AnytimeLinearModel write a state its decode_state refuses
#: (KnownBugs).
_AVOID = {
    ("group_sequential_monitor", "information"): lambda v: v == 0 or v >= 0.01,
    ("anytime_linear_model", "x"): lambda v: v == 0 or abs(v) >= 1e-100,
    ("anytime_linear_model", "z"): lambda v: v == 0 or abs(v) >= 1e-100,
}


def _generated_history(data, case, *, moderate: bool = False):
    """A history shaped like the case's own (inputs, sequence numbers, keys, times), with generated values.

    Some values are replaced (`sparse`), every value of an input is one repeated value (`constant`), or
    every value is generated (`dense`).  With `moderate`, values stay within a few orders of magnitude of
    the case's, so that runs last.
    """
    _, make_procedure, _, _, observations, _ = case
    procedure = make_procedure()
    n = data.draw(st.integers(0, len(observations)), label="length")
    observations = observations[:n]
    if moderate:
        def strategy(name, template):
            contract = _contract(procedure, name)
            if contract.kind != "real" or contract.shape or contract.lower is not None:
                return st.sampled_from(template)
            near = st.sampled_from(template)
            return near.flatmap(lambda v: st.sampled_from([v, -v, 0.0, 2.0 * v, v + 1e-9, 1e3 * v]))
    else:
        def strategy(name, template):
            values = _values(_contract(procedure, name))
            avoid = _AVOID.get((procedure.name, name))
            return st.one_of(st.sampled_from(template), values if avoid is None else values.filter(avoid))
    names = sorted({o.input for o in observations})
    templates = {name: [o.value for o in observations if o.input == name] for name in names}
    values = [o.value for o in observations]
    mode = data.draw(st.sampled_from(["sparse", "constant", "dense"]), label="mode") if n else "sparse"
    if mode == "sparse" and n:
        for i in data.draw(st.lists(st.integers(0, n - 1), max_size=8, unique=True), label="replaced"):
            values[i] = data.draw(strategy(observations[i].input, templates[observations[i].input]))
    elif mode == "constant":
        constant = {name: data.draw(strategy(name, templates[name]), label=f"constant {name}") for name in names}
        values = [constant[o.input] for o in observations]
    elif mode == "dense":
        values = [data.draw(strategy(o.input, templates[o.input])) for o in observations]
    return [Observation(o.input, v, o.source, o.seq, o.key, o.time) for o, v in zip(observations, values)]


# ------------------------------------------------------------------------------ checkpoint at any split point


class _Spy:
    """A procedure that records every transition it computes: (state, input, random state, result)."""

    def __init__(self, inner):
        self.inner = inner
        self.steps: list = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, state, x, rng):
        before = None if rng is None else rng.state
        result = self.inner.step(state, x, rng)
        self.steps.append((state, dict(x), before, result, None if rng is None else rng.state))
        return result


def _drive(run: Run, observations, start: int):
    """Offer observations[start:] until the run fails; return (index, exception type) of a failure, or None."""
    for i in range(start, len(observations)):
        try:
            run.offer(observations[i])
        except Exception as error:
            return i, type(error).__name__
    return None


def _trajectory(run: Run, recorder: Recorder) -> dict:
    p = run.procedure
    return json.loads(json.dumps({
        "t": run.t,
        "state": p.encode_state(run.state),
        "rng": run.rng_state,
        "terminal": run.terminal,
        "digest": run.history_digest,
        "counters": run.counters,
        "outputs": [p.encode_output(o) for o in recorder.outputs],
    }))


def _check_any_split(test: unittest.TestCase, case, data) -> None:
    _, make_procedure, make_topology, make_delivery, _, seed = case
    observations = _generated_history(data, case)
    for i in data.draw(st.lists(st.integers(0, max(len(observations) - 1, 0)), max_size=3), label="redelivered"):
        if observations:  # at-least-once delivery: an observation comes again, later; delivery drops it
            later = data.draw(st.integers(i + 1, len(observations)), label="again at")
            observations.insert(later, observations[i])
    n = len(observations)
    splits = set(data.draw(st.lists(st.integers(0, n), min_size=1, max_size=3), label="splits"))
    if seed is not None:
        seed = data.draw(st.one_of(st.just(seed), st.integers(-(2**70), 2**70)), label="seed")

    spy = _Spy(make_procedure())
    recorder = Recorder()
    with _quiet():
        run = Run(spy, topology=make_topology(), delivery=make_delivery(), consumers=[recorder], seed=seed,
                  run_id="p", clock=fixed_clock)
        checkpoints = {}
        failure = None
        for i in range(n + 1):
            if i in splits:  # through text, as save_checkpoint and load_checkpoint move it
                checkpoints[i] = json.loads(json.dumps(run.checkpoint(), sort_keys=True)), len(recorder.events)
            if i < n:
                failure = _drive(run, observations[: i + 1], i)
                if failure is not None:
                    break
        expected = _trajectory(run, recorder)

        for k, (checkpoint, emitted) in sorted(checkpoints.items()):
            resumed_recorder = Recorder()
            resumed = Run.restore(make_procedure(), checkpoint, topology=make_topology(), delivery=make_delivery(),
                                  consumers=[resumed_recorder], clock=fixed_clock)
            resumed.start()
            end = n if failure is None else failure[0] + 1
            again = _drive(resumed, observations[:end], k)
            test.assertEqual(again, failure, f"restored at {k}")
            got = _trajectory(resumed, resumed_recorder)
            want = dict(expected, outputs=expected["outputs"][emitted:])
            test.assertEqual(_text(got), _text(want), f"restored at {k}")

    # The transition itself: from the decoded state, with the same input and random state, the same result.
    procedure = spy.inner
    for state, x, rng_before, (new_state, output), rng_after in spy.steps:
        encoded = json.loads(json.dumps(procedure.encode_state(state)))
        decoded = procedure.decode_state(encoded)
        test.assertEqual(_text(procedure.encode_state(decoded)), _text(encoded))
        rng = None if rng_before is None else SplitMix64(rng_before)
        state2, output2 = procedure.step(decoded, x, rng)
        test.assertEqual(_text(procedure.encode_state(state2)), _text(procedure.encode_state(new_state)))
        test.assertEqual(_text(procedure.encode_output(output2)), _text(procedure.encode_output(output)))
        test.assertEqual(None if rng is None else rng.state, rng_after)


class CheckpointAtAnySplit(unittest.TestCase):
    """I2 of test_invariants, with generated values (anywhere in the contracts), split points and seeds.

    A run interrupted at any point, checkpointed through JSON text and
    restored continues exactly as the uninterrupted run -- outputs, state,
    random state, digest and counters, to the last bit, -0.0 and 0.0
    different (1 and 1.0 are one number, as in canonical JSON) -- and
    fails, if it fails, at the same observation with the same error.
    Observations come again at times (at-least-once delivery).
    """


def _split_test(index: int):
    @_settings(examples=10)
    @given(data=st.data())
    def test(self, data):
        _check_any_split(self, _CASES[index], data)

    return test


_CASES = cases()
for _index, _case in enumerate(_CASES):
    setattr(CheckpointAtAnySplit, f"test_{_case[0]}", _split_test(_index))


# ----------------------------------------------------------------------- properties of procedures and outputs

#: Procedures whose confidence sequence is a running intersection (their docstrings say so).
_INTERSECTED = {"quantile_confidence_sequence", "catoni_confidence_sequence", "rate_ratio_test",
                "anytime_logrank_test", "betting_confidence_sequence", "bounded_ab_test"}


def _number_or(value, default: float) -> float:
    return default if value is None else value


def _check_outputs(test: unittest.TestCase, case, data) -> None:
    _, make_procedure, make_topology, make_delivery, _, seed = case
    observations = _generated_history(data, case, moderate=True)
    spy = _Spy(make_procedure())
    procedure = spy.inner
    recorder = Recorder()
    with _quiet():
        run = Run(spy, topology=make_topology(), delivery=make_delivery(), consumers=[recorder], seed=seed,
                  clock=fixed_clock)
        digests = []
        for obs in observations:
            try:
                run.offer(obs)
            except NumericalError:
                break
            digests.append((run.t, run.history_digest, run.terminal))
    events = recorder.events
    outputs = [json.loads(json.dumps(procedure.encode_output(e.output))) for e in events]

    # a terminal run takes nothing more: no output after the terminal one, t and digest frozen
    for e in events[:-1]:
        test.assertFalse(e.terminal)
    if run.terminal:
        frozen = [(t, d) for t, d, terminal in digests if terminal]
        test.assertEqual(set(frozen), {(run.t, run.history_digest)})
        test.assertTrue(procedure.is_terminal(run.state))
        # and the state is absorbing for loops over step: refused, or terminal again with the same decision
        _, x, rng_state, _, _ = spy.steps[-1]
        try:
            after, _ = procedure.step(run.state, x, None if rng_state is None else SplitMix64(rng_state))
        except LifecycleError:
            pass
        else:
            test.assertTrue(procedure.is_terminal(after))
            test.assertEqual(getattr(after, "decision", None), getattr(run.state, "decision", None))

    e_field = getattr(procedure, "e_value_field", None)
    running_max = 0.0
    previous = None
    for out in outputs:
        if "p_value" in out:
            test.assertTrue(0.0 <= out["p_value"] <= 1.0, out)
            if previous is not None:
                test.assertLessEqual(out["p_value"], previous["p_value"])  # always valid: never increases
            if e_field in out:
                running_max = max(running_max, out[e_field])
                test.assertAlmostEqual(out["p_value"], min(1.0, math.exp(-running_max)), delta=1e-15)
        if "lower" in out and "upper" in out and not isinstance(out["lower"], list):
            lower, upper = _number_or(out["lower"], -math.inf), _number_or(out["upper"], math.inf)
            if procedure.name not in _INTERSECTED:  # an intersection may become empty: data against the model
                test.assertLessEqual(lower, upper, out)
            elif previous is not None and not (procedure.name == "rate_ratio_test" and out["lower"] is None
                                               and out["upper"] is None):  # KnownBugs: an exhausted grid
                test.assertGreaterEqual(lower, _number_or(previous["lower"], -math.inf))
                test.assertLessEqual(upper, _number_or(previous["upper"], math.inf))
        if previous is not None and previous.get("decision") is not None:
            test.assertEqual(out.get("decision"), previous["decision"])  # a decision is never taken back
        previous = out


class OutputProperties(unittest.TestCase):
    """Promises of the procedures, on generated histories near the cases' own."""

def _outputs_test(index: int):
    @_settings(examples=8)
    @given(data=st.data())
    def test(self, data):
        _check_outputs(self, _CASES[index], data)

    return test


for _index, _case in enumerate(_CASES):
    setattr(OutputProperties, f"test_{_case[0]}", _outputs_test(_index))


# ---------------------------------------------------------------------------------- canonical JSON (RFC 8785)


def _shortest_digits(x: float):
    """(s, n): the fewest decimal digits s, and the point position n, with s * 10**(n - len(s)) == x.

    Found by search, independently of float.__repr__: for k = 1, 2, ... digits, the correctly rounded
    k-digit decimal and its two neighbours are candidates; of those that read back as x, the closest
    to x, then the even one (ECMAScript Number::toString, step 5).
    """
    exact = Decimal(x)
    for k in range(1, 18):
        mantissa, _, exponent = f"{exact:.{k - 1}e}".partition("e")
        s, n = int(mantissa.replace(".", "")), int(exponent) + 1
        found = []
        for candidate in (s - 1, s, s + 1):
            c, m = candidate, n
            if c >= 10**k:
                c, m = c // 10, m + 1
            elif c < 10 ** (k - 1):
                c, m = c * 10 + 9, m - 1
            if c > 0 and float(f"{c}e{m - k}") == x:
                found.append((abs(Decimal(c).scaleb(m - k) - exact), c % 2, c, m))
        if found:
            _, _, s, n = min(found)
            return str(s), n
    raise AssertionError(f"no decimal of 17 digits reads back as {x!r}")


def _es_number(x: float) -> str:
    """ECMAScript Number::toString(x), written from the specification."""
    if x == 0:
        return "0"
    if x < 0:
        return "-" + _es_number(-x)
    s, n = _shortest_digits(x)
    k = len(s)
    if k <= n <= 21:
        return s + "0" * (n - k)
    if 0 < n <= 21:
        return s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return "0." + "0" * -n + s
    e = n - 1
    exponent = ("+" if e >= 0 else "-") + str(abs(e))
    return s + "e" + exponent if k == 1 else s[0] + "." + s[1:] + "e" + exponent


_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _es_string(text: str) -> str:
    """JSON.stringify of a well-formed string."""
    return '"' + "".join(_ESCAPES.get(c, f"\\u{ord(c):04x}" if ord(c) < 0x20 else c) for c in text) + '"'


def _utf16_units(text: str) -> list:
    units = []
    for c in map(ord, text):
        if c >= 0x10000:
            units += [0xD800 + ((c - 0x10000) >> 10), 0xDC00 + ((c - 0x10000) & 0x3FF)]
        else:
            units.append(c)
    return units


def _reference_canonical(value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return _es_number(float(value))
    if isinstance(value, str):
        return _es_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_reference_canonical(v) for v in value) + "]"
    keys = sorted(value, key=_utf16_units)
    return "{" + ",".join(_es_string(k) + ":" + _reference_canonical(value[k]) for k in keys) + "}"


def _bits(u: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", u))[0]


def _neighbours(x: float):
    """x and the floats one and two units in the last place either side of it (finite ones)."""
    (u,) = struct.unpack(">Q", struct.pack(">d", abs(x)))
    sign = -1.0 if x < 0 else 1.0
    return [sign * _bits(v) for v in range(max(u - 2, 0), u + 3) if math.isfinite(_bits(v))]


_FINITE = st.floats(allow_nan=False, allow_infinity=False)
_BIT_PATTERNS = st.integers(0, 2**64 - 1).map(_bits).filter(math.isfinite)
#: Where the notation changes (1e21, 1e-6, 1e-7), powers of ten, subnormals, integers near 2**53.
_SPECIAL = st.one_of(
    st.integers(-330, 310).map(lambda e: float(f"1e{e}")).flatmap(lambda x: st.sampled_from(_neighbours(x))),
    st.sampled_from([1e21, 1e-6, 1e-7, 9.999999999999999e20, 5e-324, 2.2250738585072014e-308, 2.0**53, 2.0**-1074])
    .flatmap(lambda x: st.sampled_from(_neighbours(x))),
    st.integers(1, 2**52 - 1).map(_bits),  # subnormal
    st.integers(-(2**53), 2**53).map(float),
)
_NUMBERS = st.one_of(_FINITE, _BIT_PATTERNS, _SPECIAL).flatmap(lambda x: st.sampled_from([x, -x]))

_CHARACTERS = st.one_of(
    st.characters(min_codepoint=0, max_codepoint=0x7F),
    st.characters(min_codepoint=0xD7F0, max_codepoint=0x10FFFF, exclude_categories=("Cs",)),
    st.characters(exclude_categories=("Cs",)),
)
_TEXT = st.text(_CHARACTERS, max_size=8)
_JSON = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(-(2**53), 2**53), _NUMBERS, _TEXT),
    lambda children: st.one_of(st.lists(children, max_size=4), st.dictionaries(_TEXT, children, max_size=4)),
    max_leaves=16,
)


def _shuffled(value, rnd):
    """The same JSON value with every object's keys inserted in another order."""
    if isinstance(value, list):
        return [_shuffled(v, rnd) for v in value]
    if isinstance(value, dict):
        keys = list(value)
        rnd.shuffle(keys)
        return {k: _shuffled(value[k], rnd) for k in keys}
    return value


def _same_json(a, b) -> bool:
    """Equal as JSON values: numbers by value (1 is 1.0, -0 is 0), true is not 1."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_json(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_json(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


class CanonicalJson(unittest.TestCase):
    @_settings(examples=300)
    @given(_NUMBERS)
    def test_numbers_are_written_as_ecmascript_writes_them(self, x):
        self.assertEqual(canonical_json(x), _es_number(x))
        self.assertEqual(float(canonical_json(x)), x)  # and read back exactly (-0 as 0)

    @_settings(examples=100)
    @given(st.integers(-(2**53), 2**53))
    def test_integers_within_2_53_are_their_decimal_digits(self, n):
        self.assertEqual(canonical_json(n), str(n))
        self.assertEqual(canonical_json(n), canonical_json(float(n)))

    @_settings(examples=50)
    @given(st.one_of(st.integers(2**53 + 1), st.integers(max_value=-(2**53) - 1)))
    def test_integers_beyond_2_53_are_refused(self, n):
        with self.assertRaises(ValueError):
            canonical_json({"n": [n]})

    @_settings(examples=150)
    @given(_JSON)
    def test_matches_the_reference_and_reads_back(self, value):
        text = canonical_json(value)
        self.assertEqual(text, _reference_canonical(value))
        again = json.loads(text, parse_int=float)  # as JSON.parse reads it: every number a binary64
        self.assertTrue(_same_json(again, value), (again, value))
        self.assertEqual(canonical_json(again), text)  # a fixed point

    @_settings(examples=100)
    @given(_JSON, st.randoms(use_true_random=False))
    def test_invariant_to_key_order(self, value, rnd):
        self.assertEqual(canonical_json(_shuffled(value, rnd)), canonical_json(value))

    @_settings(examples=100)
    @given(st.lists(_TEXT, max_size=6, unique=True))
    def test_keys_are_sorted_by_utf16_code_units(self, keys):
        text = canonical_json({k: 0 for k in keys})
        self.assertEqual(text, "{" + ",".join(_es_string(k) + ":0" for k in sorted(keys, key=_utf16_units)) + "}")

    @_settings(examples=50)
    @given(_TEXT, st.integers(0xD800, 0xDFFF), _TEXT)
    def test_lone_surrogates_are_refused(self, before, surrogate, after):
        with self.assertRaises(ValueError):
            canonical_json([before + chr(surrogate) + after])
        with self.assertRaises(ValueError):
            canonical_json({before + chr(surrogate) + after: 1})

    @_settings(examples=20)
    @given(st.sampled_from([math.nan, math.inf, -math.inf]), _JSON)
    def test_non_finite_numbers_are_refused(self, bad, value):
        with self.assertRaises(ValueError):
            canonical_json([value, {"x": bad}])


# ------------------------------------------------------------------------------------ delivery and topologies


@st.composite
def _arrivals(draw, sources=("s",), max_n=12):
    """Observations 0..n-1 of each source (from `start`), arriving in any order, some of them repeatedly."""
    start = draw(st.sampled_from([0, 1, 2**53 - 30, -(2**53)]))
    observations = []
    for source in sources:
        n = draw(st.integers(0, max_n))
        observations += [Observation(source, f"{source}{i}", seq=start + i) for i in range(n)]
    repeats = draw(st.lists(st.sampled_from(observations), max_size=6)) if observations else []
    arrival = draw(st.permutations(observations + repeats))
    return start, observations, arrival


def _per_source(delivered):
    out: dict = {}
    for obs in delivered:
        out.setdefault(obs.source, []).append(obs.seq)
    return out


class DeliveryProperties(unittest.TestCase):
    @_settings(examples=100)
    @given(_arrivals(sources=("s", "t", "u")))
    def test_sequence_policy_delivers_each_source_in_order_once(self, arrivals):
        start, observations, arrival = arrivals
        delivery = Delivery("sequence", start=start)
        delivered = [d for obs in arrival for d in delivery.push(obs)]
        expected = _per_source(observations)
        self.assertEqual(_per_source(delivered), expected)
        self.assertEqual(delivery.duplicates, len(arrival) - len(observations))
        self.assertEqual(delivery.pending(), 0)
        self.assertEqual(delivery.positions(), {s: start + len(seqs) for s, seqs in expected.items()})

    @_settings(examples=100)
    @given(_arrivals(sources=("s", "t")), st.booleans())
    def test_strict_policy_takes_only_the_next_number(self, arrivals, dedup):
        """The next number is delivered; a late repeat is dropped with dedup (Delivery's docstring; SPEC 4
        says only "anything else is an ordering error"), and anything else is an ordering error."""
        start, _, arrival = arrivals
        delivery, position = Delivery("strict", dedup=dedup, start=start), {}
        for obs in arrival:
            expected = position.get(obs.source, start)
            if obs.seq == expected:
                self.assertEqual(delivery.push(obs), [obs])
                position[obs.source] = expected + 1
            elif obs.seq < expected and dedup:
                self.assertEqual(delivery.push(obs), [])
            else:
                with self.assertRaises(OrderingError):
                    delivery.push(obs)
                break

    @_settings(examples=100)
    @given(_arrivals(sources=("s", "t")))
    def test_without_dedup_a_repeat_is_an_ordering_error(self, arrivals):
        start, _, arrival = arrivals
        for policy in ("sequence", "arrival"):
            delivery, seen = Delivery(policy, dedup=False, start=start), set()
            for obs in arrival:
                repeated = (obs.source, obs.seq) in seen
                seen.add((obs.source, obs.seq))
                if repeated and policy == "sequence":
                    with self.assertRaises(OrderingError):
                        delivery.push(obs)
                    break
                delivered = delivery.push(obs)
                if policy == "arrival":
                    self.assertEqual(delivered, [obs])  # arrival without dedup delivers again

    @_settings(examples=100)
    @given(_arrivals(sources=("s", "t")), st.data())
    def test_arrival_policy_keeps_first_copies_and_a_minimal_state(self, arrivals, data):
        start, _, arrival = arrivals
        split = data.draw(st.integers(0, len(arrival)), label="split")
        delivery = Delivery("arrival", start=start)
        delivered, seen = [], set()
        for i, obs in enumerate(arrival):
            if i == split:  # checkpoint and restore: the arrival policy holds nothing back
                state = json.loads(json.dumps(delivery.state()))
                delivery = Delivery("arrival", start=start)
                delivery.restore(state)
            delivered += delivery.push(obs)
            seen.add((obs.source, obs.seq))
            # the state: per source, the first number not delivered, and the numbers delivered beyond it
            state = delivery.state()
            for source in {o.source for o in arrival[: i + 1]}:
                numbers = {q for s, q in seen if s == source}
                position = start
                while position in numbers:
                    position += 1
                self.assertEqual(state["next"][source], position)
                self.assertEqual(state["ahead"].get(source, []), sorted(q for q in numbers if q > position))
        first = []
        for obs in arrival:
            if obs not in first:
                first.append(obs)
        self.assertEqual(delivered, first)
        self.assertEqual(delivery.duplicates, len(arrival) - len(first))  # the count survives the restore

    @_settings(examples=100)
    @given(st.data())
    def test_skip_range_keeps_the_state_minimal(self, data):
        """Observations and ranges declared not to exist, in any order, against a model of SPEC 4.

        The model: the position is the first number neither delivered nor inside a declared range that
        the position has reached; an observation is delivered if it is at or beyond the position and not
        delivered before; `ahead` is what was delivered beyond the position.  A range declared from the
        position never covers a number delivered ahead here (that is in KnownBugs).
        """
        delivery = Delivery()
        delivered: set = set()
        gaps: list = []

        def position() -> int:
            p = 0
            while True:
                if p in delivered:
                    p += 1
                    continue
                reached = [stop for first, stop in gaps if first <= p < stop]
                if not reached:
                    return p
                p = max(reached)

        for _ in range(data.draw(st.integers(1, 12), label="operations")):
            p = position()
            if data.draw(st.booleans(), label="observation"):
                seq = data.draw(st.integers(0, 30), label="seq")
                got = delivery.push(Observation("x", 0.0, seq=seq))
                fresh = seq >= p and seq not in delivered
                self.assertEqual(len(got), 1 if fresh else 0)
                if fresh:
                    delivered.add(seq)
            else:
                first = data.draw(st.integers(0, 30), label="first")
                stop = data.draw(st.integers(first, 32), label="stop")
                if first <= p and any(p < d < stop for d in delivered):
                    continue
                delivery.skip_range("x", first, stop)
                gaps.append((first, stop))
            p = position()
            state = delivery.state()
            self.assertEqual(state["next"].get("x", 0), p)
            self.assertEqual(state["ahead"].get("x", []), sorted(d for d in delivered if d > p))

    @_settings(examples=60)
    @given(_arrivals(sources=("a", "b"), max_n=10), st.data())
    def test_a_run_with_redelivery_after_a_restore_gives_the_in_order_result(self, arrivals, data):
        """Sequence policy, paired sources: any arrival order, duplicates, and a restore at any point.

        After the restore the sources send everything again from their positions, in yet another order
        (what was held back is not in the checkpoint); the outputs are those of the pairs in order.
        """
        start, observations, arrival = arrivals
        values = {(o.source, o.seq): (0.3 if o.source == "a" else -0.2) + 0.5 * (o.seq % 7) for o in observations}
        arrival = [Observation(o.source, values[(o.source, o.seq)], seq=o.seq) for o in arrival]
        replayed = data.draw(st.permutations(arrival), label="replayed")
        split = data.draw(st.integers(0, len(arrival)), label="split")

        def make():
            return SPRT(Gaussian(1.5), 0.0, 0.5, 1e-4, 1e-4)

        def topology():
            return PositionalPair(("a", "b")).map(difference("a", "b"), "a-b")

        a = [values[("a", q)] for q in sorted(q for s, q in values if s == "a")]
        b = [values[("b", q)] for q in sorted(q for s, q in values if s == "b")]
        procedure = make()
        expected = [procedure.encode_output(o) for _, o in trajectory(procedure, [{"x": x - y} for x, y in zip(a, b)])]

        recorder = Recorder()
        first = Run(make(), topology=topology(), delivery=Delivery("sequence", start=start), consumers=[recorder])
        for obs in arrival[:split]:
            first.offer(obs)
        checkpoint = json.loads(json.dumps(first.checkpoint()))
        delivery = Delivery("sequence", start=start)
        second = Run.restore(make(), checkpoint, topology=topology(), delivery=delivery, consumers=[recorder])
        second.start()
        for obs in list(arrival[split:]) + list(skip_to(replayed, second.delivery)):
            second.offer(obs)
        got = [second.procedure.encode_output(o) for o in recorder.outputs]
        self.assertEqual(_text(got), _text(expected))
        self.assertEqual(delivery.pending(), 0)


class TopologyProperties(unittest.TestCase):
    @_settings(examples=100)
    @given(
        st.lists(st.one_of(st.integers(-(2**53), 2**53), _TEXT), max_size=8, unique=True),
        st.lists(st.sampled_from(["a", "b", "c"]), min_size=2, max_size=3, unique=True),
        st.data(),
    )
    def test_key_join_pairs_fields_of_one_key_whatever_the_interleaving(self, keys, inputs, data):
        """Every complete key is emitted once, when its last field arrives, with its own fields only.

        Keys 1 and "1" are different keys, also after the state has been through JSON.
        """
        missing = data.draw(st.lists(st.sampled_from(keys), unique=True), label="incomplete") if keys else []
        observations = [Observation(name, [key, name], key=key) for key in keys for name in inputs
                        if not (key in missing and name == inputs[-1])]
        arrival = data.draw(st.permutations(observations), label="arrival")
        split = data.draw(st.integers(0, len(arrival)), label="split")
        join, emitted, completion = KeyJoin(inputs), [], []
        counts: dict = {}
        for i, obs in enumerate(arrival):
            if i == split:
                state = json.loads(json.dumps(join.state()))
                join = KeyJoin(inputs)
                join.restore(state)
            emitted += join.push(obs)
            k = (type(obs.key), obs.key)
            counts[k] = counts.get(k, 0) + 1
            if counts[k] == len(inputs):
                completion.append(obs.key)
        self.assertEqual(len(emitted), len(completion))
        for row, key in zip(emitted, completion):
            self.assertEqual(list(row), list(inputs))
            for name, value in row.items():
                self.assertEqual(value, [key, name])
                self.assertIs(type(value[0]), type(key))
        pending = join.state()
        self.assertEqual(sorted(map(repr, (k for k, _ in pending))), sorted(map(repr, missing)))

    @_settings(examples=100)
    @given(
        st.lists(st.one_of(st.integers(-5, 5), st.floats(-5.0, 5.0)), max_size=12),
        st.lists(st.one_of(st.integers(-5, 5), st.floats(-5.0, 5.0)), max_size=12),
        st.sampled_from([0, 0.0, 0.25, 0.5, 1, 100.0]),
        st.data(),
    )
    def test_time_alignment_is_confluent(self, ta, tb, tolerance, data):
        """The pairs depend on the event times of each input, not on how the inputs interleave.

        Reference: the greedy pairing of the two whole sequences.  Ties and tolerance 0 included, and a
        checkpoint of the buffers at any point.
        """
        ta, tb = sorted(ta), sorted(tb)
        a = [Observation("a", i, seq=i, time=t) for i, t in enumerate(ta)]
        b = [Observation("b", 100 + i, seq=i, time=t) for i, t in enumerate(tb)]
        expected, unmatched, i, j = [], 0, 0, 0
        while i < len(ta) and j < len(tb):
            if abs(ta[i] - tb[j]) <= tolerance:
                expected.append({"a": i, "b": 100 + j})
                i, j = i + 1, j + 1
            elif ta[i] < tb[j]:
                i, unmatched = i + 1, unmatched + 1
            else:
                j, unmatched = j + 1, unmatched + 1
        choices = data.draw(st.lists(st.booleans(), min_size=len(a) + len(b), max_size=len(a) + len(b)))
        arrival, ia, ib = [], 0, 0
        for take_a in choices:
            if ib >= len(b) or (take_a and ia < len(a)):
                arrival.append(a[ia])
                ia += 1
            else:
                arrival.append(b[ib])
                ib += 1
        split = data.draw(st.integers(0, len(arrival)), label="split")
        topology, emitted = TimeAlign(("a", "b"), tolerance), []
        for k, obs in enumerate(arrival):
            if k == split:
                state = json.loads(json.dumps(topology.state()))
                topology = TimeAlign(("a", "b"), tolerance)
                topology.restore(state)
            emitted += topology.push(obs)
        self.assertEqual(emitted, expected)
        self.assertEqual(topology.unmatched, unmatched)


# ------------------------------------------------------------------------------------------------- SplitMix64

_MASK = 2**64 - 1


def _reference_splitmix(state: int, count: int) -> tuple:
    """SPEC section 7, written out."""
    out = []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) % 2**64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % 2**64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % 2**64
        out.append(z ^ (z >> 31))
    return out, state


class RandomState(unittest.TestCase):
    @_settings(examples=100)
    @given(st.integers(0, _MASK), st.integers(0, 20), st.integers(0, 20))
    def test_splitmix64_is_the_specified_one_and_resumes_from_its_state(self, state, k, more):
        expected, _ = _reference_splitmix(state, k + more)
        rng = SplitMix64(state)
        got = [rng.next_u64() for _ in range(k)]
        resumed = SplitMix64(int(f"{rng.state:016x}", 16))  # the random state as a checkpoint holds it
        got += [resumed.next_u64() for _ in range(more)]
        self.assertEqual(got, expected)

    @_settings(examples=100)
    @given(st.integers(-(2**80), 2**80), st.sampled_from(["statistical", "simulation/x", "test"]))
    def test_seeding_is_the_specified_one(self, seed, role):
        expected = int.from_bytes(hashlib.sha256(f"{seed}/{role}".encode()).digest()[:8], "big")
        self.assertEqual(SplitMix64.for_role(seed, role).state, expected)

    @_settings(examples=100)
    @given(st.integers(0, _MASK), _FINITE.filter(lambda m: abs(m) < 1e300), st.floats(0.0, 1e300))
    def test_uniform_and_normal_draws_are_the_specified_ones(self, state, mean, sd):
        (z1, z2), _ = _reference_splitmix(state, 2)
        u1, u2 = (z1 >> 11) * 2.0**-53, (z2 >> 11) * 2.0**-53
        self.assertTrue(0.0 <= u1 < 1.0)
        self.assertEqual(SplitMix64(state).random(), u1)
        normal = mean + sd * math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)
        self.assertEqual(SplitMix64(state).normal(mean, sd), normal)


# --------------------------------------------------------------------------------------------------- numerics

_DF = st.one_of(
    st.floats(0.1, 100.0),
    st.floats(100.0, 1e200),
    st.sampled_from([0.1, 0.5, 1.0, 2.0, 3.0, 1e5, 1e13, 1e15, 1e16, 1e200, math.inf]),
)
_T = st.one_of(st.floats(-50.0, 50.0), st.floats(-1e300, 1e300), st.sampled_from([0.0, 1e-300, 1e-8, 1.0, 37.5, 1e150]))


class TDistribution(unittest.TestCase):
    @_settings(examples=200)
    @given(_T, _T, _DF)
    def test_cdf_is_monotone_in_t(self, s, t, df):
        if s > t:
            s, t = t, s
        self.assertLessEqual(t_cdf(s, df), t_cdf(t, df))

    @_settings(examples=200)
    @given(_T, _DF)
    def test_cdf_is_symmetric(self, t, df):
        p, q = t_cdf(t, df), t_cdf(-t, df)
        self.assertTrue(0.0 <= p <= 1.0)
        self.assertAlmostEqual(p + q, 1.0, delta=2.3e-16)

    @_settings(examples=200)
    @given(st.floats(0.0, 1e300), _DF, _DF)
    def test_tail_is_monotone_in_the_degrees_of_freedom(self, t, df1, df2):
        """For t > 0, P(T > t) decreases as the degrees of freedom grow (Ghosh 1973), to rounding."""
        if df1 > df2:
            df1, df2 = df2, df1
        upper1, upper2 = _log_upper_tail(t, df1), _log_upper_tail(t, df2)
        if math.isinf(upper1):  # the tail's logarithm itself overflows (t^2 beyond the floats)
            self.assertEqual(upper2, -math.inf)
        else:
            self.assertLessEqual(upper2, upper1 + 1e-9 * abs(upper1) + 1e-15)

    @_settings(examples=200)
    @given(st.floats(1e-6, 1e300), _DF)
    def test_isf_inverts_the_tail(self, t, df):
        log_tail = _log_upper_tail(t, df)
        if not (math.isfinite(log_tail) and log_tail < math.log(0.5) - 1e-6):
            return
        self.assertAlmostEqual(t_isf_log(log_tail, df) / t, 1.0, delta=1e-8)

    @_settings(examples=200)
    @given(st.floats(0.0, 1e300))
    def test_normal_tail_in_logarithms(self, z):
        q = 0.5 * math.erfc(z / math.sqrt(2.0))
        log_q = _log_normal_upper(z)
        if q > 1e-300:
            self.assertAlmostEqual(log_q, math.log(q), delta=1e-12 * max(1.0, abs(log_q)))
        self.assertLessEqual(log_q, _log_normal_upper(z * 0.5))  # decreasing in z
        self.assertEqual(t_cdf(-z, math.inf), q)

    @_settings(examples=150)
    @given(st.floats(0.0, 1.0), st.floats(0.05, 50.0), st.floats(0.05, 50.0))
    def test_regularized_beta_is_a_distribution_function(self, x, a, b):
        value = regularized_beta(x, a, b)
        self.assertTrue(0.0 <= value <= 1.0)
        self.assertAlmostEqual(value + regularized_beta(1.0 - x, b, a, x), 1.0, delta=1e-12)
        self.assertLessEqual(regularized_beta(x * 0.5, a, b), value + 1e-15)


class Summation(unittest.TestCase):
    @_settings(examples=150)
    @given(st.lists(st.one_of(st.floats(-1e290, 1e290), st.floats(-1.0, 1.0)), max_size=40))
    def test_neumaier_sum_is_as_accurate_as_promised(self, values):
        """total + comp is the sum to about 2 units in the last place, also with heavy cancellation."""
        total, comp = 0.0, 0.0
        for v in values:
            total, comp = neumaier_add(total, comp, v)
        exact = sum(map(Fraction, values), Fraction(0))
        magnitude = sum(map(abs, values))
        error = abs(Fraction(total) + Fraction(comp) - exact)
        bound = Fraction(2.3e-16) * abs(exact) + Fraction((len(values) ** 2 + 1) * 1e-31) * Fraction(magnitude)
        self.assertLessEqual(error, bound)

    @_settings(examples=150)
    @given(st.lists(st.floats(-700.0, 700.0), min_size=1, max_size=30), st.floats(-100.0, 100.0))
    def test_logsumexp_bounds_and_shift(self, values, shift):
        result = logsumexp(values)
        top = max(values)
        self.assertGreaterEqual(result, top - 1e-12 * abs(top))
        self.assertLessEqual(result, top + math.log(len(values)) + 1e-12 * max(1.0, abs(top)))
        moved = logsumexp([v + shift for v in values])
        self.assertAlmostEqual(moved, result + shift, delta=1e-9 * max(1.0, abs(moved)))

    @_settings(examples=100)
    @given(_JSON)
    def test_equivalent_is_reflexive_and_sees_through_json(self, value):
        self.assertTrue(equivalent(value, value))
        self.assertTrue(equivalent(value, json.loads(canonical_json(value))))


class EBH(unittest.TestCase):
    @_settings(examples=150)
    @given(st.lists(st.floats(-20.0, 20.0), max_size=12), st.floats(0.001, 0.5), st.floats(1.0, 4.0),
           st.randoms(use_true_random=False))
    def test_e_bh_is_the_brute_force_rule(self, log_e, alpha, factor, rnd):
        """The largest k whose k-th largest e-value is at least K / (k alpha); equivariant; monotone in alpha."""
        rejected = e_bh(log_e, alpha)
        count = len(log_e)
        best = max([k for k in range(1, count + 1)
                    if sorted(log_e, reverse=True)[k - 1] >= math.log(count) - math.log(k) - math.log(alpha)],
                   default=0)
        self.assertEqual(len(rejected), best)
        if best:
            threshold = sorted(log_e, reverse=True)[best - 1]
            self.assertEqual(rejected, [i for i, v in enumerate(log_e) if v >= threshold])
        order = list(range(count))
        rnd.shuffle(order)
        permuted = e_bh([log_e[i] for i in order], alpha)
        self.assertEqual(sorted(order[i] for i in permuted), rejected)
        if alpha * factor < 1.0:
            self.assertLessEqual(set(rejected), set(e_bh(log_e, alpha * factor)))


# ------------------------------------------------------------------------------------------------- known bugs


class KnownBugs(unittest.TestCase):
    """Bugs these properties found, each reduced to a minimal reproduction; expected to fail until fixed.

    When one is fixed, remove its expectedFailure (an unexpected success fails the run as a reminder)
    and the matching restriction of the generators (`_AVOID`, `_HUGE_INTEGER`).
    """

    @unittest.expectedFailure
    def test_a_count_beyond_2_53_fails_the_step_not_the_checkpoint(self):
        """Bug: a step commits a state that no checkpoint can hold; the failure comes later, at checkpoint().

        Found by CheckpointAtAnySplit.test_rate_ratio_test (explore profile).  Counts are integer
        inputs, valid up to 2**53 (SPEC 3), and RateRatioTest adds them up in its state: two units with
        2**53 events give a count of 2**54.  checked_step only refuses non-finite numbers (SPEC 6, step
        5), so the step is committed and the run goes on -- but canonical JSON refuses integers beyond
        2**53, so run.checkpoint() raises NumericalError from then on, and the state that was committed
        cannot be decoded exactly either (decode_fields reads integers through float).  Any procedure
        that sums integer inputs into an int field can do this.  Suggested fix: in checked_step, treat
        an integer beyond 2**53 in the new state or output like a non-finite number (a numerical
        failure, nothing committed), and say so in SPEC 6, step 5.
        """
        from sequential_analysis.procedures import RateRatioTest

        run = Run(RateRatioTest(grid=10), run_id="r", clock=fixed_clock)
        with _quiet(), self.assertRaises(NumericalError):  # the second step should fail, nothing committed
            for _ in range(2):
                run.step({"arm": 1, "count": 2**53, "exposure": 1.0})
        run.checkpoint()

    @unittest.expectedFailure
    def test_a_run_with_a_tiny_covariate_restores_from_its_checkpoint(self):
        """Bug: AnytimeLinearModel refuses to decode a state it reached itself.

        Found by CheckpointAtAnySplit.test_anytime_linear_model (explore profile).  decode_state checks
        the Cauchy-Schwarz inequality of the stored cross products, |a_ij| <= sqrt(a_ii a_jj) (1 + 1e-9),
        a relative tolerance only.  With a covariate deviation d below about 1e-154, d * d underflows
        (to a subnormal with a few bits, or to 0) while d * dy does not: here x = 1e-170 gives
        a_xx = 0.0 and a_xy = 5e-171, and x = 1e-160 a subnormal a_xx; Run.restore then refuses the
        run's own checkpoint ("cannot decode its state").  Suggested fix: an absolute slack for
        underflow in that check, e.g. sqrt((a_ii + n 2**-1022) (a_jj + n 2**-1022)), or skip it when a_ii
        or a_jj is below the normal range.
        """
        from sequential_analysis.procedures import AnytimeLinearModel

        for x in (1e-150, 1e-160, 1e-170):
            run = Run(AnytimeLinearModel(["x"], ["z"]), run_id="r", clock=fixed_clock)
            run.step({"x": 0.0, "z": 0.0, "y": 0.0})
            run.step({"x": x, "z": 1.0, "y": 1.0})
            Run.restore(AnytimeLinearModel(["x"], ["z"]), run.checkpoint(), clock=fixed_clock)

    @unittest.expectedFailure
    def test_the_integration_grid_does_not_depend_on_the_scale_of_information(self):
        """Bug: an analysis at a small information fraction makes the integration grid huge.

        Found by CheckpointAtAnySplit.test_group_sequential_monitor (explore profile), which was killed
        for lack of memory.  _Recursion.advance integrates the continuation region of B(t) = Z sqrt(t)
        over [lower, upper] on a grid of `points_per_sd` points per standard deviation sqrt(t - t_prev).
        For a one-sided design without futility, lower = min(-10 sqrt(t) + drift t, upper - 1.0): when
        sqrt(t) < 1 / (b + 10), about t < 0.005, the second term wins, and its 1.0 does not scale with
        sqrt(t), so the grid has 24 / sqrt(t) points: 289 at t = 0.5 and 0.01, 2 401 at 1e-4, 240 001 at
        1e-8, and about 1e152 at 1e-302 (information 1e-300 with max_information 100; valid input, the
        contract is lower=0).  The next analysis costs the product of two such grids: a
        GroupSequentialMonitor fed information 1e-6, 2e-6, 3e-6 (max 100) does not finish in a minute,
        and information 1e-300 exhausts the memory.  Suggested fix: ``upper - math.sqrt(t)`` (the same
        width in standard deviations at any scale), and a cap on the number of grid points in
        _Recursion._grid that raises NumericalError instead of allocating without bound.
        """
        from sequential_analysis.group_sequential import _Recursion

        for t in (0.5, 0.01, 1e-4, 1e-8):
            recursion = _Recursion(0.0, False, 24)
            recursion.advance(t, 2.0)
            self.assertLessEqual(len(recursion.xs), 1000, f"t = {t}")

    @unittest.expectedFailure
    def test_a_checkpoint_written_by_json_stringify_is_restored(self):
        """Bug: a float of magnitude 2**53 to 1e21, written by an ECMAScript executor, is refused on restore.

        JSON.stringify -- like canonical JSON, both use Number::toString -- writes 1e17 as
        100000000000000000.  Python's json reads that as an int beyond 2**53, and validate_checkpoint
        refuses it ("not canonical JSON"), so a run continued by the TypeScript executor (ts/, whose
        command line writes its checkpoint with JSON.stringify) cannot come back to Python once any
        number in its state, buffers or provenance is that large.  An observation log written by the
        executor has the same problem (log.ts writes values with JSON.stringify): on replay such a value
        becomes an int beyond 2**53, invalid input, and the replay departs from the run.  Reproduced with
        the executor itself: a NormalMixtureSPRT whose running total is 1e17.

        SPEC 1 says that numbers are binary64 and that 1 and 1.0 are the same number; a reader should then
        take an integer literal beyond 2**53 as the binary64 number it denotes (as JSON.parse does) --
        e.g. json.load(..., parse_int=...) returning float(text) beyond 2**53 in load_checkpoint and in the
        log reader, and Run.restore converting such ints when int(float(n)) == n -- and SPEC 1 should say
        so; the refusal of integers beyond 2**53 then applies to integers the writer meant as integers.
        """
        from sequential_analysis.procedures import NormalMixtureSPRT

        run = Run(NormalMixtureSPRT(1.0, 0.5, stop_on_reject=False), run_id="r", clock=fixed_clock)
        run.offer(Observation("x", 1e17, seq=0))
        written = canonical_json(run.checkpoint())  # the number formatting of JSON.stringify
        self.assertIn('"total":100000000000000000', written)
        Run.restore(NormalMixtureSPRT(1.0, 0.5, stop_on_reject=False), json.loads(written), clock=fixed_clock)

    @unittest.expectedFailure
    def test_rate_ratio_interval_stays_nested_when_the_grid_is_exhausted(self):
        """Bug: RateRatioTest reports (None, None), "unbounded", once every grid point is excluded.

        Found by test_rate_ratio_test (explore profile).  The confidence sequence is a running
        intersection (module docstring), and lower = None means "below the grid".  When the ratio lies
        beyond ratio_range (here about 120 against (0.1, 10)), the points are excluded from the bottom
        up: [0.79, inf), [2.04, inf), ..., [8.53, inf) -- and when the last one goes, _interval finds
        nothing kept and returns (None, None): the interval jumps back to (0, inf), the least informative
        answer, right when the evidence is strongest.  AnytimeLogRankTest handles the same situation
        (it collapses onto the least-excluded point, see its docstring).  Suggested fix in
        RateRatioTest._interval: with nothing kept, report (points[-1], None) when the point excluded
        last is the top one (the ratio is above the grid), (None, points[0]) when it is the bottom one,
        or collapse onto the least-excluded point as AnytimeLogRankTest does; and document it.
        """
        from sequential_analysis.procedures import RateRatioTest

        rows = [{"arm": arm, "count": 60 if arm else 0, "exposure": 1.0} for _ in range(12) for arm in (1, 0)]
        lowers = [o.lower for _, o in trajectory(RateRatioTest(grid=30), rows)[1::2]]
        self.assertEqual(lowers[0:8], sorted(lowers[0:8]))  # [0.79, 2.04, ..., 8.53, 8.53]: nested so far
        for previous, lower in zip(lowers, lowers[1:]):
            if previous is not None:  # once bounded below, bounded below at least as high
                self.assertIsNotNone(lower)
                self.assertGreaterEqual(lower, previous)

    @unittest.expectedFailure
    def test_skip_range_leaves_no_number_below_the_position(self):
        """Bug: skip_range from the position keeps the numbers delivered inside the range in `ahead`.

        SPEC 4: `ahead` lists the numbers delivered beyond a gap.  Here 5 was delivered ahead of the gap
        1..4; skip_range(1, 10) moves the position to 10 but leaves 5 in the state, for ever -- the
        growth of the checkpoint that skip_range exists to prevent, and a state that differs from that of
        a delivery which never saw 5.  (The other branch, a range declared ahead of the position, prunes
        `ahead` when it is reached.)  Delivery.skip_range should, after moving the position, drop the
        numbers below it: ``self._ahead.get(source, set()).difference_update(range(nxt, stop))`` or
        prune in `_settle`.
        """
        delivery = Delivery()
        for seq in (0, 5):
            delivery.push(Observation("x", 1.0, seq=seq))
        delivery.skip_range("x", 1, 10)
        self.assertEqual(delivery.state(), {"next": {"x": 10}, "ahead": {}, "duplicates": 0})


if __name__ == "__main__":
    unittest.main()
