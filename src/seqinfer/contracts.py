"""Input contracts: what a valid statistical input is."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ContractViolation

_KINDS = ("real", "integer", "binary")


@dataclass(frozen=True)
class InputContract:
    """Contract of one logical input.

    kind   -- "real", "integer" or "binary"
    shape  -- () for a scalar, (d,) for a length-d vector
    lower / upper -- inclusive domain bounds, applied element-wise
    unit   -- recorded in provenance; adapters are responsible for conversion
    """

    name: str
    kind: str = "real"
    shape: tuple[int, ...] = ()
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    description: str = ""

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if len(self.shape) > 1:
            raise ValueError("only scalar and vector inputs are supported")

    def spec(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": list(self.shape),
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
        }

    def _check_scalar(self, v: Any) -> None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ContractViolation(f"input {self.name!r}: expected a number, got {type(v).__name__}")
        if not math.isfinite(v):
            raise ContractViolation(f"input {self.name!r}: non-finite value {v!r}")
        if self.kind == "integer" and v != int(v):
            raise ContractViolation(f"input {self.name!r}: expected an integer, got {v!r}")
        if self.kind == "binary" and v not in (0, 1):
            raise ContractViolation(f"input {self.name!r}: expected 0 or 1, got {v!r}")
        if self.lower is not None and v < self.lower:
            raise ContractViolation(f"input {self.name!r}: {v!r} is below {self.lower!r}")
        if self.upper is not None and v > self.upper:
            raise ContractViolation(f"input {self.name!r}: {v!r} is above {self.upper!r}")

    def validate(self, value: Any) -> None:
        if not self.shape:
            self._check_scalar(value)
            return
        if not isinstance(value, (list, tuple)) or len(value) != self.shape[0]:
            raise ContractViolation(f"input {self.name!r}: expected a vector of length {self.shape[0]}")
        for v in value:
            self._check_scalar(v)


def validate_input(procedure, x: Any) -> None:
    """Check one statistical input against a procedure's contracts.

    A statistical input is a non-empty mapping from logical input names to
    values.  Procedures with ``joint_inputs = True`` need every declared
    input in every step (paired or one-sample analyses); the others accept
    any non-empty subset (independent samples arriving one at a time).
    """
    contracts = procedure.inputs()
    if not isinstance(x, Mapping) or not x:
        raise ContractViolation("a statistical input must be a non-empty mapping")
    unknown = set(x) - set(contracts)
    if unknown:
        raise ContractViolation(f"undeclared logical inputs: {sorted(unknown)}")
    if procedure.joint_inputs and set(x) != set(contracts):
        missing = sorted(set(contracts) - set(x))
        raise ContractViolation(f"joint procedure is missing inputs: {missing}")
    for name, value in x.items():
        contracts[name].validate(value)
