"""Numerical building blocks and the numerical contract.

Long runs accumulate many small terms, so log-domain statistics are summed
with Neumaier's compensated scheme.  The running sum and its compensation
are both stored in the inferential state: the compensation is part of what
is needed to continue the run exactly.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from .errors import NumericalError


def neumaier_add(total: float, comp: float, x: float) -> tuple[float, float]:
    """One step of Neumaier summation.  The represented value is total + comp."""
    t = total + x
    if abs(total) >= abs(x):
        comp += (total - t) + x
    else:
        comp += (x - t) + total
    return t, comp


def logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return -math.inf
    m = max(vals)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def require_finite(value: float, what: str) -> float:
    if not math.isfinite(value):
        raise NumericalError(f"{what} is not finite: {value!r}")
    return value


def equivalent(a: Any, b: Any, abs_tol: float = 0.0, rel_tol: float = 0.0) -> bool:
    """Structural equivalence under a numerical contract.

    Floats are compared within the given tolerances; everything else
    (counts, decisions, labels, structure) must match exactly.  With both
    tolerances at zero this is bitwise equality of the encoded values,
    except that NaN is considered equal to NaN.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if a == b:
            return True
        return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k], abs_tol, rel_tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equivalent(x, y, abs_tol, rel_tol) for x, y in zip(a, b))
    return a == b
