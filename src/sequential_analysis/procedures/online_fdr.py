# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Online false discovery rate control for a stream of experiments (e-LOND).

A team runs experiments one after another, each ending with an e-value
for its own hypothesis (the e-process of any test here, read at the time
the experiment stopped).  Deciding each one on its own at level alpha
lets false discoveries pile up; waiting to apply e-BH to all of them is
impossible when decisions are needed now.  e-LOND (Xu and Ramdas 2024)
decides each hypothesis when its e-value arrives:

    reject H_t  when  e_t >= 1 / alpha_t,   alpha_t = alpha gamma_t (D_{t-1} + 1),

D_{t-1} being the number of discoveries so far and gamma a fixed
non-negative sequence with sum <= 1.  The false discovery rate,
E[false discoveries / max(1, discoveries)], is at most alpha at every
time, under arbitrary dependence between the experiments.  Every
discovery raises the level of the following ones; the default gamma is
that of Javanmard and Montanari (2018), gamma_j proportional to
log(max(j, 2)) / (j exp(sqrt(log j))), with the constant 0.07720838 of
Ramdas et al. (2017), which makes it sum to 0.976 (checked in the tests).

The input is the experiment's log e-value (`input`, "log_e" by default),
a finite number; pass a large negative one for an e-value of 0.

Cost per step (PERFORMANCE.md): constant, about 1 µs; a constant state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state

#: Ramdas, Yang, Wainwright and Jordan (2017); with it gamma sums to 0.976 <= 1 (tests/test_online_fdr.py).
_GAMMA_NORMALISER = 0.07720838


def default_gamma(j: int) -> float:
    """gamma_j of Javanmard and Montanari (2018), j = 1, 2, ...; the sequence sums to at most 1."""
    return _GAMMA_NORMALISER * math.log(max(j, 2)) / (j * math.exp(math.sqrt(math.log(j))))


@dataclass(frozen=True)
class OnlineFDRState:
    t: int = 0  # hypotheses decided
    discoveries: int = 0


@dataclass(frozen=True)
class OnlineFDROutput:
    t: int
    level: float  # alpha_t
    rejected: bool
    discoveries: int


class OnlineFDR(Procedure):
    """e-LOND: decide a stream of hypotheses, each by its e-value, with FDR <= alpha; see the module docstring."""

    name = "online_fdr_elond"
    version = "1"

    def __init__(self, alpha: float = 0.05, *, input: str = "log_e"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha, self.input = float(alpha), input

    def config(self) -> dict:
        return {"alpha": self.alpha, "gamma": "javanmard-montanari", "input": self.input}

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> OnlineFDRState:
        return OnlineFDRState()

    def level(self, t: int, discoveries: int) -> float:
        """alpha_t for the t-th hypothesis (t = 1, 2, ...) after `discoveries` discoveries."""
        return self.alpha * default_gamma(t) * (discoveries + 1)

    def step(self, state: OnlineFDRState, x, rng=None):
        t = state.t + 1
        level = self.level(t, state.discoveries)
        rejected = x[self.input] >= -math.log(level)
        discoveries = state.discoveries + rejected
        return OnlineFDRState(t, discoveries), OnlineFDROutput(t, level, bool(rejected), discoveries)

    def decode_state(self, data) -> OnlineFDRState:
        state = decode_fields(OnlineFDRState, data)
        require_state(state.discoveries <= state.t, "more discoveries than hypotheses")
        return state
