"""Input contracts: what a valid statistical input is."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ContractViolation

_KINDS = ("real", "integer", "binary")
#: Exactly these types are numbers.  Subclasses (bool, numpy.float64, IntEnum)
#: are rejected so that every value has one unambiguous meaning in the history.
_NUMBER_TYPES = (int, float)


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
        if type(v) not in _NUMBER_TYPES:
            raise ContractViolation(f"input {self.name!r}: {_explain_type(v, self.kind)}")
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
        if type(value) not in (list, tuple):
            hint = " (convert it in the adapter with .tolist())" if hasattr(value, "tolist") else ""
            raise ContractViolation(
                f"input {self.name!r}: expected a list or tuple of length {self.shape[0]}, "
                f"got {_type_name(value)}{hint}"
            )
        if len(value) != self.shape[0]:
            raise ContractViolation(f"input {self.name!r}: expected a vector of length {self.shape[0]}, got {len(value)}")
        for v in value:
            self._check_scalar(v)


def _type_name(v: Any) -> str:
    t = type(v)
    return t.__qualname__ if t.__module__ == "builtins" else f"{t.__module__}.{t.__qualname__}"


def _explain_type(v: Any, kind: str) -> str:
    """Why a value is not accepted as a number, and what the adapter should do about it."""
    got = f"got {_type_name(v)} {v!r}"
    target = "float" if kind == "real" else "int"
    if isinstance(v, bool):
        return (
            f"expected int or float, {got}; booleans are not numbers here. "
            f"Convert explicitly in the adapter, e.g. int(value)"
        )
    if hasattr(v, "item") and hasattr(v, "dtype"):
        return (
            f"expected a built-in int or float, {got}. "
            f"Convert in the adapter with value.item() or {target}(value), "
            f"so that the type recorded in the history is unambiguous"
        )
    if isinstance(v, str):
        return f"expected int or float, {got}; parse strings in the adapter, not in the procedure"
    return f"expected a built-in int or float, {got}; convert it in the adapter with {target}(value)"


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
