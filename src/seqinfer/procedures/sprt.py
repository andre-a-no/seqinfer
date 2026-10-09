# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Wald's sequential probability ratio test."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core import Procedure
from ..numerics import neumaier_add, require_finite
from .families import Family

REJECT_H0 = "reject_h0"
ACCEPT_H0 = "accept_h0"


@dataclass(frozen=True)
class SPRTState:
    n: int = 0
    llr: float = 0.0
    comp: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class SPRTOutput:
    n: int
    llr: float
    decision: str | None


class SPRT(Procedure):
    """Test theta0 against theta1; stop when the log likelihood ratio leaves (lower, upper).

    Thresholds are Wald's approximations, upper = log((1-beta)/alpha) and
    lower = log(beta/(1-alpha)).
    """

    name = "sprt"
    version = "1"

    def __init__(
        self, family: Family, theta0: float, theta1: float, alpha: float = 0.05, beta: float = 0.05, input: str = "x"
    ):
        family.check(theta0)
        family.check(theta1)
        if theta0 == theta1 or not (0 < alpha < 1 and 0 < beta < 1):
            raise ValueError("need theta0 != theta1 and error rates in (0, 1)")
        self.family, self.theta0, self.theta1 = family, float(theta0), float(theta1)
        self.alpha, self.beta, self.input = alpha, beta, input
        self.upper = math.log((1.0 - beta) / alpha)
        self.lower = math.log(beta / (1.0 - alpha))

    def config(self) -> dict:
        return {
            "family": self.family.spec(),
            "theta0": self.theta0,
            "theta1": self.theta1,
            "alpha": self.alpha,
            "beta": self.beta,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: self.family.contract(self.input)}

    def initial_state(self) -> SPRTState:
        return SPRTState()

    def step(self, state: SPRTState, x, rng=None):
        inc = self.family.llr(x[self.input], self.theta1, self.theta0)
        total, comp = neumaier_add(state.llr, state.comp, inc)
        value = require_finite(total + comp, "log likelihood ratio")
        decision = REJECT_H0 if value >= self.upper else ACCEPT_H0 if value <= self.lower else None
        n = state.n + 1
        return SPRTState(n, total, comp, decision), SPRTOutput(n, value, decision)

    def is_terminal(self, state: SPRTState) -> bool:
        return state.decision is not None

    def decode_state(self, data) -> SPRTState:
        return SPRTState(**data)
