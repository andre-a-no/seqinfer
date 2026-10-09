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
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Generic, TypeVar

from .contracts import InputContract, validate_input
from .errors import NumericalError
from .rng import SplitMix64

S = TypeVar("S")
O = TypeVar("O")  # noqa: E741 -- O_t in the paper

#: One statistical input X_t: logical input name -> value.
StatInput = Mapping[str, Any]


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
        if not self.source:
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
    """No NaN or infinity anywhere in a JSON-like value."""
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, (list, tuple)):
        return all(_finite(v) for v in obj)
    if isinstance(obj, dict):
        return all(_finite(v) for v in obj.values())
    return True


def checked_step(procedure: Procedure, state: Any, x: StatInput, rng: SplitMix64 | None) -> tuple[Any, Any]:
    """procedure.step, with every numerical failure reported as NumericalError.

    Overflow or a math domain error inside the transition, and a new state
    or output holding NaN or infinity -- which could be neither reported
    nor checkpointed -- all become NumericalError, and nothing is returned
    to commit.
    """
    try:
        new_state, output = procedure.step(state, x, rng)
    except ArithmeticError as error:  # overflow, division by zero
        raise NumericalError(f"{procedure.name}: {type(error).__name__}: {error}") from error
    except ValueError as error:
        if "math domain error" not in str(error):
            raise
        raise NumericalError(f"{procedure.name}: {error}") from error
    if not _finite(procedure.encode_state(new_state)):
        raise NumericalError(f"{procedure.name}: the new state is not finite; nothing was committed")
    if not _finite(procedure.encode_output(output)):
        raise NumericalError(f"{procedure.name}: the output is not finite; nothing was committed")
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
