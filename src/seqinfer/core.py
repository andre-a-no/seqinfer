"""The statistical boundary: observations, procedures, reference trajectory.

A Sequential Procedure is the transition

    (S_t, O_t, R_t) = F(S_{t-1}, X_t, R_{t-1})

and nothing else.  It does not know where X_t came from, how it was
scheduled, where S_t is stored or what happens to O_t.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Generic, Iterable, Mapping, TypeVar

from .contracts import InputContract, validate_input
from .rng import SplitMix64

S = TypeVar("S")
O = TypeVar("O")

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
    def from_json(cls, d: Mapping[str, Any]) -> "Observation":
        return cls(d["input"], d["value"], d.get("source", ""), d.get("seq"), d.get("key"), d.get("time"))


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
        """

    def is_terminal(self, state: S) -> bool:
        """Terminal states are absorbing: no further transitions are applied."""
        return False

    def encode_state(self, state: S) -> Any:
        return asdict(state) if is_dataclass(state) else state

    def decode_state(self, data: Any) -> S:
        raise NotImplementedError(f"{type(self).__name__} does not define decode_state")

    def encode_output(self, output: O) -> Any:
        return asdict(output) if is_dataclass(output) else output

    def identity(self) -> dict:
        return {"name": self.name, "version": self.version, "config": self.config()}


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
        state, output = procedure.step(state, x, rng)
        if rng is not None:
            rng_state = rng.state
        out.append((state, output))
    return out
