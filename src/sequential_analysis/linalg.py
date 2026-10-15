# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Small dense matrices from the standard library, for state-space filters.

Matrices are tuples of rows (tuples of floats); vectors are tuples.  Every
sum runs left to right in a fixed order, so that results are reproducible
bit for bit across Python versions and can be repeated in any language.
The dimensions of a filter are small (a few states), so nothing here is
optimised beyond that.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]


def vector(value: Any, size: int | None, name: str) -> Vector:
    """A tuple of finite floats of the given size (any size if None)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{name} must be a sequence of numbers")
    out = tuple(float(v) for v in value)
    if size is not None and len(out) != size:
        raise ValueError(f"{name} must have {size} entries, got {len(out)}")
    if not out or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be non-empty and finite")
    return out


def matrix(value: Any, rows: int | None, cols: int | None, name: str) -> Matrix:
    """A tuple of rows of finite floats; a number is a 1x1 matrix."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = ((value,),)
    if not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty sequence of rows")
    out = tuple(vector(row, cols, f"{name} row") for row in value)
    if rows is not None and len(out) != rows:
        raise ValueError(f"{name} must have {rows} rows, got {len(out)}")
    width = len(out[0])
    if any(len(row) != width for row in out):
        raise ValueError(f"{name} rows differ in length")
    return out


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += x * y
    return total


def transpose(a: Matrix) -> Matrix:
    return tuple(tuple(row[j] for row in a) for j in range(len(a[0])))


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = transpose(b)
    return tuple(tuple(dot(row, col) for col in bt) for row in a)


def matvec(a: Matrix, v: Sequence[float]) -> Vector:
    return tuple(dot(row, v) for row in a)


def add(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(x + y for x, y in zip(r, s, strict=True)) for r, s in zip(a, b, strict=True))


def sub(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(x - y for x, y in zip(r, s, strict=True)) for r, s in zip(a, b, strict=True))


def vadd(a: Sequence[float], b: Sequence[float]) -> Vector:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def scale(a: Matrix, c: float) -> Matrix:
    return tuple(tuple(c * x for x in row) for row in a)


def identity(n: int) -> Matrix:
    return tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))


def symmetrize(a: Matrix) -> Matrix:
    """(A + A') / 2: removes the asymmetry that rounding leaves in a covariance."""
    n = len(a)
    return tuple(tuple(0.5 * (a[i][j] + a[j][i]) for j in range(n)) for i in range(n))


def is_symmetric(a: Matrix) -> bool:
    n = len(a)
    return len(a[0]) == n and all(a[i][j] == a[j][i] for i in range(n) for j in range(i))


def quadratic(a: Matrix, b: Matrix) -> Matrix:
    """A B A', symmetrized."""
    return symmetrize(matmul(matmul(a, b), transpose(a)))


def cholesky(a: Matrix) -> Matrix:
    """Lower-triangular L with L L' = A; ValueError if A is not positive definite."""
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j]
            for k in range(j):
                s -= low[i][k] * low[j][k]
            if i == j:
                if not s > 0.0:
                    raise ValueError("matrix is not positive definite")
                low[i][i] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return tuple(tuple(row) for row in low)


def solve_cholesky(low: Matrix, b: Sequence[float]) -> Vector:
    """x with L L' x = b."""
    n = len(low)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= low[i][k] * y[k]
        y[i] = s / low[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        s = y[i]
        for k in range(i + 1, n):
            s -= low[k][i] * x[k]
        x[i] = s / low[i][i]
    return tuple(x)


def inverse_spd(a: Matrix) -> Matrix:
    """Inverse of a symmetric positive definite matrix, symmetrized."""
    low = cholesky(a)
    n = len(a)
    columns = [solve_cholesky(low, tuple(1.0 if i == j else 0.0 for i in range(n))) for j in range(n)]
    return symmetrize(transpose(tuple(columns)))


def log_det_spd(a: Matrix) -> float:
    low = cholesky(a)
    total = 0.0
    for i in range(len(a)):
        total += math.log(low[i][i])
    return 2.0 * total


def trace(a: Matrix) -> float:
    total = 0.0
    for i in range(len(a)):
        total += a[i][i]
    return total
