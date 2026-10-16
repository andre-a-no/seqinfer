# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The statistical boundary: observations, procedures, reference trajectory.

A Sequential Procedure is the transition

    (S_t, O_t, R_t) = F(S_{t-1}, X_t, R_{t-1})

and nothing else.  It does not know where X_t came from, how it was
scheduled, where S_t is stored or what happens to O_t.
"""
from __future__ import annotations

import math
import re
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Generic, TypeVar

from .contracts import InputContract, validate_input
from .errors import LifecycleError, NumericalError
from .rng import SplitMix64

S = TypeVar("S")
O = TypeVar("O")  # noqa: E741 -- O_t in the paper

#: One statistical input X_t: logical input name -> value.
StatInput = typing.Mapping[str, Any]  # typing: evaluated at import, also on Python 3.8


@dataclass(frozen=True)
class Observation:
    """One value as delivered by an adapter, before topology.

    input  -- logical input the value is bound to
    value  -- the observed value (must be JSON-serializable to be logged)
    source -- identity of the physical source; defaults to the input name
    seq    -- per-source sequence number, used for ordering and deduplication
    key    -- join key, used by key-based topologies
    time   -- event time, used by time-aligned topologies
    """

    input: str
    value: Any
    source: str = ""
    seq: int | None = None
    key: Any = None
    time: float | None = None

    def __post_init__(self):
        if isinstance(self.source, str) and not self.source:  # anything else is reported by envelope_problem
            object.__setattr__(self, "source", self.input)

    def to_json(self) -> dict:
        return {
            "input": self.input,
            "value": self.value,
            "source": self.source,
            "seq": self.seq,
            "key": self.key,
            "time": self.time,
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Observation:
        return cls(d["input"], d["value"], d.get("source", ""), d.get("seq"), d.get("key"), d.get("time"))


def _encode(value: Any) -> Any:
    """Dataclass instances become dicts; anything else is taken to be JSON already."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


#: Value types an observation may carry: exactly these (no subclasses), so that
#: an observation log records them without loss and a replay sees what the run saw.
_PLAIN = (int, float, str, bool, type(None))
_MAX_EXACT = 2**53
#: Marker a log writes in place of an observation it could not record.
UNRECORDABLE = "$unrecordable"


def _text(v: Any) -> bool:
    """A str that UTF-8 can encode (no lone surrogates): JSON files and hashes need its bytes."""
    if type(v) is not str:
        return False
    try:
        v.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _exact_int(v: Any) -> bool:
    return type(v) is int and abs(v) <= _MAX_EXACT


def _plain_value(v: Any) -> bool:
    if type(v) is float:
        return math.isfinite(v)
    if type(v) is int:
        return abs(v) <= _MAX_EXACT
    if type(v) is str:
        return _text(v)
    if type(v) in _PLAIN:
        return True
    if type(v) is list:
        return all(_plain_value(x) for x in v)
    if type(v) is dict:
        return UNRECORDABLE not in v and all(_text(k) and _plain_value(x) for k, x in v.items())
    return False


def envelope_problem(obs: Any) -> str | None:
    """Why delivery cannot handle an arriving observation, or None if it can.

    Delivery orders and deduplicates by source and sequence number, and
    checkpoints its position per source, so these must be exact: the input
    and source names strings, the sequence number None or an int within
    2**53.  Such an observation is invalid input; it is not delivered, so
    it is not in the observation log either.
    """
    if not isinstance(obs, Observation):
        return f"expected an Observation, got {type(obs).__qualname__}; adapters wrap values in Observation(...)"
    if not _text(obs.input):
        return f"input name {obs.input!r} must be a str"
    if not _text(obs.source):
        return f"source {obs.source!r} must be a str (it names the source in checkpoints)"
    if obs.seq is not None and not _exact_int(obs.seq):
        return f"sequence number {obs.seq!r} must be None or an int within 2**53"
    return None


def recording_problem(obs: Observation) -> str | None:
    """Why an observation cannot be recorded exactly, or None if it can.

    Values must be int, float, str, bool, None, or lists and dicts (with
    string keys) of these, with floats finite, integers within 2**53 and
    strings encodable as UTF-8; keys str or int; event times finite numbers.
    Anything else -- numpy scalars, Fraction, tuples, UUIDs, subclasses of
    int or float -- would come back from an observation log as something
    else, so a run treats such an observation as invalid input, whether or
    not it keeps a log, and the adapter is the place to convert it.
    """
    if not _plain_value(obs.value):
        return (
            f"value {obs.value!r} of type {type(obs.value).__qualname__} cannot be recorded exactly; "
            f"convert it in the adapter to int, float (finite), str, bool, None, or a list or dict of these"
        )
    if obs.key is not None and not (_text(obs.key) or _exact_int(obs.key)):
        return (
            f"key {obs.key!r} must be a str or an int within 2**53; build composite keys as strings in the adapter"
        )
    if obs.time is not None and not (_exact_int(obs.time) or (type(obs.time) is float and math.isfinite(obs.time))):
        return f"event time {obs.time!r} must be a finite float or an int within 2**53"
    return None


#: Decisions a test procedure can reach.
DECISIONS = (None, "reject_h0", "accept_h0")


def _number(name: str, value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"state field {name!r} must be a finite number, got {value!r}")
    return float(value)


class _Annotation:
    """A field annotation of a state dataclass, read from its text.

    The annotations are strings (``from __future__ import annotations``);
    typing.get_type_hints would evaluate ``float | None`` and
    ``tuple[float, ...]``, which Python 3.8 and 3.9 cannot.  They are read
    here with names bound to these objects instead, the same way on every
    Python version.
    """

    def __init__(self, base: Any, args: tuple[Any, ...] = (), optional: bool = False):
        self.base, self.args, self.optional = base, args, optional

    def __or__(self, other: Any) -> _Annotation:
        if other is None:
            return _Annotation(self.base, self.args, True)
        raise TypeError("only X | None unions are supported in states")

    __ror__ = __or__

    def __getitem__(self, item: Any) -> _Annotation:
        return _Annotation(self.base, item if isinstance(item, tuple) else (item,))


_NAMES = {"int": _Annotation(int), "float": _Annotation(float), "str": _Annotation(str),
          "tuple": _Annotation(tuple), "Any": _Annotation(Any)}


def _annotation(text: Any) -> _Annotation:
    if isinstance(text, _Annotation):
        return text
    if not isinstance(text, str):
        raise TypeError(f"unsupported state annotation {text!r}")
    result = eval(text, {"__builtins__": {}}, dict(_NAMES))  # the package's own annotations, not data
    if not isinstance(result, _Annotation):
        raise TypeError(f"unsupported state annotation {text!r}")
    return result


def _field(name: str, kind: Any, value: Any) -> Any:
    kind = _annotation(kind)
    if kind.optional and value is None:
        return None
    if kind.base is int:  # every integer in a state is a count
        number = _number(name, value)
        if number != int(number) or number < 0:
            raise ValueError(f"state field {name!r} must be a non-negative integer, got {value!r}")
        return int(number)
    if kind.base is float:
        return _number(name, value)
    if kind.base is str:
        if type(value) is not str:
            raise ValueError(f"state field {name!r} must be a string, got {value!r}")
        return value
    if kind.base is tuple:
        if type(value) not in (list, tuple):
            raise ValueError(f"state field {name!r} must be a list, got {value!r}")
        return tuple(_field(name, kind.args[0], v) for v in value)
    raise TypeError(f"state field {name!r} has an unsupported type {kind!r}")


def decode_fields(cls: type[S], data: Any) -> S:
    """A state dataclass from its JSON form, with the type of every field checked.

    Integers are counts (non-negative), floats are finite, a ``decision``
    is one of DECISIONS, and no field may be missing or extra: a checkpoint
    is data that may have been edited, and the next step must not be the
    first to notice.  Raises ValueError (TypeError for a wrong shape).
    """
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} expects a JSON object, got {type(data).__name__}")
    hints = {f.name: f.type for f in fields(cls)}  # type: ignore[arg-type]
    names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
    if set(data) != set(names):
        raise TypeError(f"{cls.__name__} has fields {names}, got {sorted(data)}")
    values = {name: _field(name, hints[name], data[name]) for name in names}
    if "decision" in values and values["decision"] not in DECISIONS:
        raise ValueError(f"unknown decision {values['decision']!r}")
    return cls(**values)


def require_state(condition: bool, message: str) -> None:
    """A decoded state must be one the procedure can reach; ValueError otherwise."""
    if not condition:
        raise ValueError(f"inconsistent state: {message}")


def refuse_terminal(procedure: Any, state: Any) -> None:
    """A stopped procedure takes no further observations.

    `Run` never offers them; this guards loops that call `step` directly.
    """
    if procedure.is_terminal(state):
        raise LifecycleError(f"{procedure.name} has stopped; a terminal state takes no further observations")


#: What a procedure's outputs promise, from the strongest to the weakest.
GUARANTEES = {
    "exact": "error rates, coverage or false-alarm rates hold at every sample size, under the stated assumptions",
    "asymptotic": "they hold approximately, the better the larger the sample",
    "model": "exact probabilities under the stated model and prior; no error rate beyond the model",
    "approximate": "an approximation even under the model (Monte Carlo, variational, stochastic approximation)",
    "none": "a transformation of the data; no inferential claim",
}


class Procedure(ABC, Generic[S, O]):
    """Semantic contract of a Sequential Procedure.

    The base class is a convenience, not a requirement: `Run` only relies on
    the attributes and methods below, so any object providing them conforms.

    States must be immutable values.  `step` returns a new state and must
    not modify its arguments; this is what makes state updates transactional
    and checkpoints trivially consistent.
    """

    name: str = "procedure"
    version: str = "1"
    #: True if `step` draws from the statistical random state.
    randomized: bool = False
    #: True if every declared input is required in every step.
    joint_inputs: bool = True
    #: What the outputs promise; one of GUARANTEES.
    guarantee: str = "none"

    @abstractmethod
    def config(self) -> dict:
        """JSON-serializable parameters that determine F."""

    @abstractmethod
    def inputs(self) -> Mapping[str, InputContract]:
        """Contracts of the logical inputs."""

    @abstractmethod
    def initial_state(self) -> S:
        """The default initial state S_0."""

    @abstractmethod
    def step(self, state: S, x: StatInput, rng: SplitMix64 | None) -> tuple[S, O]:
        """Return (S_t, O_t) from S_{t-1} and X_t.

        `rng` carries R_{t-1} on entry and R_t on return; it is None for
        deterministic procedures.  The caller commits both together.

        Raise ContractViolation for an input that turns out to be invalid
        only during the step; the run treats it like a contract violation.
        Any other exception is a failure of the procedure.
        """

    def is_terminal(self, state: S) -> bool:
        """Terminal states are absorbing: no further transitions are applied."""
        return False

    def encode_state(self, state: S) -> Any:
        return _encode(state)

    def decode_state(self, data: Any) -> S:
        raise NotImplementedError(f"{type(self).__name__} does not define decode_state")

    def encode_output(self, output: O) -> Any:
        return _encode(output)

    def identity(self) -> dict:
        return {"name": self.name, "version": self.version, "config": self.config()}


def _finite(obj: Any) -> bool:
    """No NaN or infinity, and no integer beyond 2**53, anywhere in a JSON-like value.

    Either could be neither checkpointed (canonical JSON refuses them) nor
    decoded exactly.  A count that outgrows 2**53 -- two units each with
    2**53 events -- is a numerical failure of the step, not of the next
    checkpoint.
    """
    if isinstance(obj, float):
        return math.isfinite(obj)
    if type(obj) is int:
        return abs(obj) <= _MAX_EXACT
    if isinstance(obj, (list, tuple)):
        return all(_finite(v) for v in obj)
    if isinstance(obj, dict):
        return all(_finite(v) for v in obj.values())
    return True


#: A domain error of the math module: "math domain error" before Python 3.14,
#: then messages such as "expected a positive input, got 0.0".
_MATH_DOMAIN = re.compile(r"^math domain error$|^expected .*, got ")


def checked_step(procedure: Procedure, state: Any, x: StatInput, rng: SplitMix64 | None) -> tuple[Any, Any]:
    """procedure.step, with every numerical failure reported as NumericalError.

    Overflow or a math domain error inside the transition, and a new state
    or output holding NaN, infinity or an integer beyond 2**53 -- which
    could be neither reported nor checkpointed -- all become
    NumericalError, and nothing is returned to commit.
    """
    try:
        new_state, output = procedure.step(state, x, rng)
    except ArithmeticError as error:  # overflow, division by zero
        raise NumericalError(f"{procedure.name}: {type(error).__name__}: {error}") from error
    except ValueError as error:
        if not _MATH_DOMAIN.search(str(error)):
            raise
        raise NumericalError(f"{procedure.name}: {error}") from error
    if not _finite(procedure.encode_state(new_state)):
        raise NumericalError(f"{procedure.name}: the new state is not finite (or holds an integer beyond 2**53); "
                             f"nothing was committed")
    if not _finite(procedure.encode_output(output)):
        raise NumericalError(f"{procedure.name}: the output is not finite (or holds an integer beyond 2**53); "
                             f"nothing was committed")
    return new_state, output


def statistical_rng(procedure: Procedure, seed: int | None) -> int | None:
    """Initial statistical random state R_0, or None for deterministic procedures."""
    if not procedure.randomized:
        return None
    if seed is None:
        raise ValueError(f"{procedure.name} is randomized: an explicit seed is required")
    return SplitMix64.for_role(seed, "statistical").state


def trajectory(
    procedure: Procedure[S, O],
    inputs: Iterable[StatInput],
    *,
    seed: int | None = None,
    state: S | None = None,
) -> list[tuple[S, O]]:
    """Reference semantics: fold F over an ordered input history.

    This is the definition every runtime is tested against.  It has no
    scheduling, no persistence and no consumers.
    """
    if state is None:
        state = procedure.initial_state()
    rng_state = statistical_rng(procedure, seed)
    out: list[tuple[S, O]] = []
    for x in inputs:
        if procedure.is_terminal(state):
            break
        validate_input(procedure, x)
        rng = SplitMix64(rng_state) if rng_state is not None else None
        state, output = checked_step(procedure, state, x, rng)
        if rng is not None:
            rng_state = rng.state
        out.append((state, output))
    return out
