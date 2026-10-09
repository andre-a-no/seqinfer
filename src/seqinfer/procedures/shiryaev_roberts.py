# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The Shiryaev-Roberts change-point detector.

    R_n = (1 + R_{n-1}) * f_theta1(X_n) / f_theta0(X_n),   R_0 = 0,

alarm at the first n with R_n >= A.  Before the change R_n - n is a
martingale, so the average run length to a false alarm is at least A
(Pollak 1985).  Unlike CUSUM, the statistic is a sum over all possible
change points rather than a maximum; it is optimal for detecting a change
that happens after a long stationary stretch.

The statistic is kept on the log scale.  R_0 = 0 has no logarithm, so the
state stores `log_r = None` until the first observation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core import Procedure
from ..numerics import require_finite
from .families import Family


@dataclass(frozen=True)
class ShiryaevRobertsState:
    n: int = 0
    log_r: float | None = None
    alarm_at: int | None = None


@dataclass(frozen=True)
class ShiryaevRobertsOutput:
    n: int
    log_statistic: float
    alarm: bool


def _log1p_exp(x: float) -> float:
    """log(1 + e^x) without overflow."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


class ShiryaevRoberts(Procedure):
    name = "shiryaev_roberts"
    version = "1"

    def __init__(self, family: Family, theta0: float, theta1: float, threshold: float, input: str = "x"):
        family.check(theta0)
        family.check(theta1)
        if theta0 == theta1:
            raise ValueError("need theta0 != theta1")
        if not 1 < threshold < math.inf:
            raise ValueError("threshold A must exceed 1")
        self.family, self.theta0, self.theta1 = family, float(theta0), float(theta1)
        self.threshold, self.input = float(threshold), input
        self._log_threshold = math.log(self.threshold)

    def config(self) -> dict:
        return {
            "family": self.family.spec(),
            "theta0": self.theta0,
            "theta1": self.theta1,
            "threshold": self.threshold,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: self.family.contract(self.input)}

    def initial_state(self) -> ShiryaevRobertsState:
        return ShiryaevRobertsState()

    def step(self, state: ShiryaevRobertsState, x, rng=None):
        llr = self.family.llr(x[self.input], self.theta1, self.theta0)
        base = 0.0 if state.log_r is None else _log1p_exp(state.log_r)
        log_r = require_finite(base + llr, "Shiryaev-Roberts statistic")
        n = state.n + 1
        alarm = log_r >= self._log_threshold
        return ShiryaevRobertsState(n, log_r, n if alarm else None), ShiryaevRobertsOutput(n, log_r, alarm)

    def is_terminal(self, state: ShiryaevRobertsState) -> bool:
        return state.alarm_at is not None

    def decode_state(self, data) -> ShiryaevRobertsState:
        return ShiryaevRobertsState(**data)
