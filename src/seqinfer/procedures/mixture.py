# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Anytime-valid test and confidence sequence for a normal mean (mixture SPRT).

Observations are N(theta, sigma^2) with sigma known.  Mixing the
likelihood ratio over theta ~ N(theta0, tau^2) (Robbins 1970) gives

    Lambda_n = sqrt(sigma^2 / (sigma^2 + n tau^2))
               * exp( n^2 tau^2 (xbar_n - theta0)^2 / (2 sigma^2 (sigma^2 + n tau^2)) ),

a non-negative martingale with mean one under theta = theta0.  By Ville's
inequality P(sup_n Lambda_n >= 1/alpha) <= alpha, so

* rejecting when Lambda_n >= 1/alpha is a level-alpha test that may be
  monitored after every observation and stopped at any time;
* p_n = min(1, 1 / max_{k<=n} Lambda_k) is an always-valid p-value;
* inverting the test gives a confidence sequence: with probability at
  least 1 - alpha the true mean lies in every interval simultaneously,

    xbar_n +/- sqrt( 2 sigma^2 (sigma^2 + n tau^2) / (n^2 tau^2)
                     * (log(1/alpha) + log(1 + n tau^2 / sigma^2) / 2) ).

tau sets where the test is most sensitive: roughly to effects of size tau.
This is the mSPRT used for continuously monitored A/B tests (Johari et al.
2017, 2022).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure
from ..numerics import neumaier_add, require_finite
from .sprt import REJECT_H0


@dataclass(frozen=True)
class MixtureState:
    n: int = 0
    total: float = 0.0
    comp: float = 0.0
    max_log_lambda: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class MixtureOutput:
    n: int
    mean: float
    log_lambda: float
    p_value: float
    lower: float
    upper: float
    decision: str | None


class NormalMixtureSPRT(Procedure):
    name = "normal_mixture_sprt"
    version = "1"

    def __init__(
        self,
        sigma: float,
        tau: float,
        theta0: float = 0.0,
        alpha: float = 0.05,
        *,
        stop_on_reject: bool = True,
        input: str = "x",
    ):
        if not (sigma > 0 and tau > 0 and math.isfinite(sigma) and math.isfinite(tau)):
            raise ValueError("sigma and tau must be positive and finite")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if not math.isfinite(theta0):
            raise ValueError("theta0 must be finite")
        self.sigma, self.tau, self.theta0, self.alpha = float(sigma), float(tau), float(theta0), float(alpha)
        self.stop_on_reject, self.input = stop_on_reject, input
        self._log_level = math.log(1.0 / self.alpha)

    def config(self) -> dict:
        return {
            "sigma": self.sigma,
            "tau": self.tau,
            "theta0": self.theta0,
            "alpha": self.alpha,
            "stop_on_reject": self.stop_on_reject,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> MixtureState:
        return MixtureState()

    def log_lambda(self, n: int, mean: float) -> float:
        s2, t2 = self.sigma**2, self.tau**2
        v = s2 + n * t2
        return 0.5 * math.log(s2 / v) + n * n * t2 * (mean - self.theta0) ** 2 / (2.0 * s2 * v)

    def half_width(self, n: int) -> float:
        s2, t2 = self.sigma**2, self.tau**2
        v = s2 + n * t2
        return math.sqrt(2.0 * s2 * v / (n * n * t2) * (self._log_level + 0.5 * math.log(v / s2)))

    def step(self, state: MixtureState, x, rng=None):
        total, comp = neumaier_add(state.total, state.comp, x[self.input])
        n = state.n + 1
        mean = require_finite((total + comp) / n, "running mean")
        log_lambda = self.log_lambda(n, mean)
        max_log_lambda = max(state.max_log_lambda, log_lambda)
        decision = state.decision or (REJECT_H0 if log_lambda >= self._log_level else None)
        h = self.half_width(n)
        p_value = min(1.0, math.exp(-max_log_lambda))
        new = MixtureState(n, total, comp, max_log_lambda, decision)
        return new, MixtureOutput(n, mean, log_lambda, p_value, mean - h, mean + h, decision)

    def is_terminal(self, state: MixtureState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> MixtureState:
        return MixtureState(**data)
