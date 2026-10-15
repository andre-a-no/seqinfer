# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Arithmetic backends: pure Python, or numpy for procedures that work on large arrays.

Procedures whose step touches thousands of numbers -- particles, a stored
sample, a grid of candidate means -- take `backend="auto" | "python" |
"numpy"`.  "auto" means numpy when it is installed.  The backend actually
used is part of the procedure's configuration, so it is recorded in every
checkpoint: a run computed with numpy is continued with numpy, and
restoring it where numpy is missing is refused (IncompatibleCheckpoint)
instead of continuing with different arithmetic.

Within one backend a run is reproducible bit for bit, and a restored run
continues exactly as the uninterrupted one would have -- on the same
machine and library versions.  numpy evaluates exp, log and sums with its
own routines, whose last bits may change with the numpy version and the
processor; the Python backend uses the math module and left-to-right sums
and gives the same bits everywhere (it is what the conformance vectors
pin down).  Use backend="python" where results must match bit for bit
across machines or another language; the numbers differ from numpy's only
in the last bits, the statistics not at all.

Random numbers come from the run's SplitMix64 state in both backends: the
vectorised generator below produces exactly the uniforms that successive
calls of SplitMix64.random would, and leaves the generator in the same
state.
"""
from __future__ import annotations

from typing import Any

from .rng import SplitMix64

BACKENDS = ("auto", "python", "numpy")

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on interpreters without numpy
    np = None


def resolve(backend: str) -> str:
    """The backend to use: "python" or "numpy"."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend == "auto":
        return "numpy" if np is not None else "python"
    if backend == "numpy" and np is None:
        raise ValueError('backend="numpy" needs numpy: pip install "sequential-analysis[fast]"')
    return backend


def numpy_version() -> str | None:
    return None if np is None else str(np.__version__)


_GOLDEN = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1


class VectorRandom:
    """numpy draws from a SplitMix64 generator, advancing its state as the scalar draws would."""

    def __init__(self, rng: SplitMix64):
        self.rng = rng

    def _u64(self, count: int) -> Any:
        start = self.rng.state
        steps = np.arange(1, count + 1, dtype=np.uint64)
        z = np.uint64(start) + steps * np.uint64(_GOLDEN)  # wraps modulo 2**64
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        self.rng.state = (start + count * _GOLDEN) & _MASK
        return z

    def random(self, count: int) -> Any:
        """`count` uniforms on [0, 1): the values of `count` calls of SplitMix64.random."""
        return (self._u64(count) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, count: int, mean: Any = 0.0, sd: Any = 1.0) -> Any:
        """`count` normal draws by Box-Muller on the same uniforms as SplitMix64.normal (two per draw)."""
        u = self.random(2 * count)
        u1, u2 = 1.0 - u[0::2], u[1::2]
        return mean + sd * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
