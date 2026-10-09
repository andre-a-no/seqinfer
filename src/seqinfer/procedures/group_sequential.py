# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Execute a group sequential test of a normal mean on a stream.

Observations arrive one at a time; the test looks at them only at the
planned analyses n_1 < ... < n_K.  At analysis k it computes

    Z_k = (mean_k - theta0) sqrt(n_k) / sigma

and stops for efficacy (rejects H0) if Z_k >= c_k, or |Z_k| >= c_k for
a two-sided design.  With futility boundaries f_k it stops and accepts H0
if Z_k <= f_k.  If no boundary is crossed by n_K, H0 is accepted (not
rejected).  The boundaries come from `seqinfer.group_sequential`.

With `sigma=None` the variance is estimated at each analysis and Z_k is
replaced by the t statistic, compared with the boundary of the same
one-sided significance level, t_{n_k - 1}^{-1}(Phi(c_k)) -- the
significance-level approach of Jennison and Turnbull (2000, sec. 11.4).
The type I error is then approximately, not exactly, the design's.

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
from statistics import NormalDist

from ..contracts import InputContract
from ..core import Procedure
from ..distributions import t_ppf
from ..numerics import require_finite
from .sprt import ACCEPT_H0, REJECT_H0


@dataclass(frozen=True)
class GroupSequentialState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations (Welford)
    analysis: int = 0  # analyses done so far
    decision: str | None = None


@dataclass(frozen=True)
class GroupSequentialOutput:
    n: int
    mean: float
    analysis: int | None  # 1-based index if this observation completed an analysis
    z: float | None  # Z_k, or the t statistic when the variance is estimated
    bound: float | None
    decision: str | None


class GroupSequentialTest(Procedure):
    name = "group_sequential_test"
    version = "2"

    def __init__(
        self,
        sigma: float | None,
        analyses: Sequence[int],
        bounds: Sequence[float],
        theta0: float = 0.0,
        *,
        two_sided: bool = False,
        futility: Sequence[float] | None = None,
        label: str = "",
        input: str = "x",
    ):
        if sigma is not None and not (sigma > 0 and math.isfinite(sigma)):
            raise ValueError("sigma must be positive and finite, or None to estimate it")
        ns = [int(n) for n in analyses]
        if not ns or ns[0] < (1 if sigma is not None else 2) or any(b <= a for a, b in pairwise(ns)):
            raise ValueError("analyses must be increasing positive sample sizes")
        if len(bounds) != len(ns) or any(not b > 0 for b in bounds):
            raise ValueError("need one positive boundary per analysis")
        self.sigma = None if sigma is None else float(sigma)
        self.theta0, self.two_sided = float(theta0), two_sided
        if futility is not None:
            if two_sided:
                raise ValueError("futility boundaries are for one-sided tests")
            if len(futility) != len(ns) or any(f > b for f, b in zip(futility, bounds, strict=True)):
                raise ValueError("need one futility boundary per analysis, none above its efficacy boundary")
        self.analyses, self.bounds = tuple(ns), tuple(float(b) for b in bounds)
        self.futility = None if futility is None else tuple(float(f) for f in futility)
        self.label, self.input = label, input
        self._bounds = tuple(self._converted(b, n) for b, n in zip(self.bounds, ns, strict=True))
        self._futility = (
            None if self.futility is None
            else tuple(self._converted(f, n) for f, n in zip(self.futility, ns, strict=True))
        )

    @classmethod
    def from_design(
        cls, design, n_max: int, sigma: float | None, theta0: float = 0.0, **kwargs
    ) -> GroupSequentialTest:
        """Analyses at the design's information fractions of n_max (rounded up)."""
        analyses = [math.ceil(t * n_max - 1e-9) for t in design.fractions]
        return cls(
            sigma, analyses, design.bounds, theta0, two_sided=design.two_sided, futility=design.futility, **kwargs
        )

    def config(self) -> dict:
        return {
            "sigma": self.sigma,
            "theta0": self.theta0,
            "analyses": list(self.analyses),
            "bounds": list(self.bounds),
            "two_sided": self.two_sided,
            "futility": None if self.futility is None else list(self.futility),
            "label": self.label,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> GroupSequentialState:
        return GroupSequentialState()

    def _converted(self, z: float, n: int) -> float:
        """A boundary for Z, as a boundary for the statistic actually used at sample size n."""
        if self.sigma is not None or not math.isfinite(z):
            return z
        return t_ppf(NormalDist().cdf(z), n - 1)

    def bound_at(self, k: int) -> float:
        """Efficacy boundary for the statistic at analysis k (0-based)."""
        return self._bounds[k]

    def step(self, state: GroupSequentialState, x, rng=None):
        v = x[self.input]
        n = state.n + 1
        delta = v - state.mean
        mean = require_finite(state.mean + delta / n, "running mean")
        m2 = state.m2 + delta * (v - mean)
        k = state.analysis
        if n != self.analyses[k]:
            new = GroupSequentialState(n, mean, m2, k, None)
            return new, GroupSequentialOutput(n, mean, None, None, None, None)
        scale = self.sigma if self.sigma is not None else math.sqrt(m2 / (n - 1))
        if scale == 0.0:  # estimated variance of identical observations: no evidence either way
            z = 0.0
        else:
            z = (mean - self.theta0) * math.sqrt(n) / scale
        bound = self.bound_at(k)
        crossed = abs(z) >= bound if self.two_sided else z >= bound
        futile = self._futility is not None and z <= self._futility[k]
        last = k + 1 == len(self.analyses)
        decision = REJECT_H0 if crossed else ACCEPT_H0 if futile or last else None
        new = GroupSequentialState(n, mean, m2, k + 1, decision)
        return new, GroupSequentialOutput(n, mean, k + 1, z, bound, decision)

    def is_terminal(self, state: GroupSequentialState) -> bool:
        return state.decision is not None

    def decode_state(self, data) -> GroupSequentialState:
        return GroupSequentialState(**data)
