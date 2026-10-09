# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Execute a group sequential test of a normal mean on a stream.

Observations arrive one at a time; the test looks at them only at the
planned analyses n_1 < ... < n_K.  At analysis k it computes

    Z_k = (mean_k - theta0) sqrt(n_k) / sigma

and stops for efficacy (rejects H0) if Z_k >= c_k, or |Z_k| >= c_k for
a two-sided design.  If no boundary is crossed by n_K, H0 is accepted
(not rejected).  The boundaries come from `seqinfer.group_sequential`.

The schedule is fixed in advance.  If analyses happen at other sample
sizes than planned, the boundaries must be recomputed for the actual
information fractions, which is what alpha spending allows: build a new
design with the realised fractions before the next analysis.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from ..contracts import InputContract
from ..core import Procedure
from ..numerics import neumaier_add, require_finite
from .sprt import ACCEPT_H0, REJECT_H0


@dataclass(frozen=True)
class GroupSequentialState:
    n: int = 0
    total: float = 0.0
    comp: float = 0.0
    analysis: int = 0  # analyses done so far
    decision: str | None = None


@dataclass(frozen=True)
class GroupSequentialOutput:
    n: int
    mean: float
    analysis: int | None  # 1-based index if this observation completed an analysis
    z: float | None
    bound: float | None
    decision: str | None


class GroupSequentialTest(Procedure):
    name = "group_sequential_test"
    version = "1"

    def __init__(
        self,
        sigma: float,
        analyses: Sequence[int],
        bounds: Sequence[float],
        theta0: float = 0.0,
        *,
        two_sided: bool = False,
        label: str = "",
        input: str = "x",
    ):
        if not (sigma > 0 and math.isfinite(sigma)):
            raise ValueError("sigma must be positive and finite")
        ns = [int(n) for n in analyses]
        if not ns or ns[0] < 1 or any(b <= a for a, b in pairwise(ns)):
            raise ValueError("analyses must be increasing positive sample sizes")
        if len(bounds) != len(ns) or any(not b > 0 for b in bounds):
            raise ValueError("need one positive boundary per analysis")
        self.sigma, self.theta0, self.two_sided = float(sigma), float(theta0), two_sided
        self.analyses, self.bounds = tuple(ns), tuple(float(b) for b in bounds)
        self.label, self.input = label, input

    @classmethod
    def from_design(cls, design, n_max: int, sigma: float, theta0: float = 0.0, **kwargs) -> GroupSequentialTest:
        """Analyses at the design's information fractions of n_max (rounded up)."""
        analyses = [math.ceil(t * n_max - 1e-9) for t in design.fractions]
        return cls(sigma, analyses, design.bounds, theta0, two_sided=design.two_sided, **kwargs)

    def config(self) -> dict:
        return {
            "sigma": self.sigma,
            "theta0": self.theta0,
            "analyses": list(self.analyses),
            "bounds": list(self.bounds),
            "two_sided": self.two_sided,
            "label": self.label,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> GroupSequentialState:
        return GroupSequentialState()

    def step(self, state: GroupSequentialState, x, rng=None):
        total, comp = neumaier_add(state.total, state.comp, x[self.input])
        n = state.n + 1
        mean = require_finite((total + comp) / n, "running mean")
        k = state.analysis
        if n != self.analyses[k]:
            new = GroupSequentialState(n, total, comp, k, None)
            return new, GroupSequentialOutput(n, mean, None, None, None, None)
        z = (mean - self.theta0) * math.sqrt(n) / self.sigma
        bound = self.bounds[k]
        crossed = abs(z) >= bound if self.two_sided else z >= bound
        decision = REJECT_H0 if crossed else ACCEPT_H0 if k + 1 == len(self.analyses) else None
        new = GroupSequentialState(n, total, comp, k + 1, decision)
        return new, GroupSequentialOutput(n, mean, k + 1, z, bound, decision)

    def is_terminal(self, state: GroupSequentialState) -> bool:
        return state.decision is not None

    def decode_state(self, data) -> GroupSequentialState:
        return GroupSequentialState(**data)
