# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Running difference of two independent sample means.

A two-input procedure that does *not* need its inputs paired: every step
may carry an observation for either sample, or for both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure


@dataclass(frozen=True)
class MeanDifferenceState:
    n_a: int = 0
    mean_a: float = 0.0
    ss_a: float = 0.0
    n_b: int = 0
    mean_b: float = 0.0
    ss_b: float = 0.0


@dataclass(frozen=True)
class MeanDifferenceOutput:
    n_a: int
    n_b: int
    difference: float | None
    std_error: float | None


def _welford(n: int, mean: float, ss: float, x: float) -> tuple[int, float, float]:
    n += 1
    delta = x - mean
    mean += delta / n
    ss += delta * (x - mean)
    return n, mean, ss


class MeanDifference(Procedure):
    name = "mean_difference"
    version = "1"
    joint_inputs = False

    def __init__(self, a: str = "a", b: str = "b"):
        self.a, self.b = a, b

    def config(self) -> dict:
        return {"a": self.a, "b": self.b}

    def inputs(self):
        return {self.a: InputContract(self.a), self.b: InputContract(self.b)}

    def initial_state(self) -> MeanDifferenceState:
        return MeanDifferenceState()

    def step(self, state: MeanDifferenceState, x, rng=None):
        n_a, mean_a, ss_a = state.n_a, state.mean_a, state.ss_a
        n_b, mean_b, ss_b = state.n_b, state.mean_b, state.ss_b
        if self.a in x:
            n_a, mean_a, ss_a = _welford(n_a, mean_a, ss_a, x[self.a])
        if self.b in x:
            n_b, mean_b, ss_b = _welford(n_b, mean_b, ss_b, x[self.b])
        diff = mean_a - mean_b if n_a and n_b else None
        se = None
        if n_a > 1 and n_b > 1:
            se = math.sqrt(ss_a / (n_a - 1) / n_a + ss_b / (n_b - 1) / n_b)
        new = MeanDifferenceState(n_a, mean_a, ss_a, n_b, mean_b, ss_b)
        return new, MeanDifferenceOutput(n_a, n_b, diff, se)

    def decode_state(self, data) -> MeanDifferenceState:
        return MeanDifferenceState(**data)
