# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Anytime-valid t-test and confidence sequence: normal mean, unknown variance.

Observations are N(theta, sigma^2) with sigma unknown.  The standardised
effect delta = (theta - theta0) / sigma gets the prior N(0, c^2) and sigma
the scale-invariant prior d sigma / sigma.  The resulting Bayes factor
(Goenen et al. 2005)

    B_n = (1 + n c^2)^(-1/2)
          * [ (1 + t_n^2 / nu) / (1 + t_n^2 / (nu (1 + n c^2))) ]^((nu + 1) / 2),

with nu = n - 1 and t_n the one-sample t statistic, is the likelihood
ratio of the scale-invariant reduction of the data.  Under H0 it is
therefore a non-negative martingale with mean one whatever sigma is (Lai
1976; Perez-Ortiz et al. 2024), and Ville's inequality makes

* rejecting when B_n >= 1/alpha a level-alpha test under continuous
  monitoring and optional stopping,
* 1 / max_k B_k an always-valid p-value, and
* the set of theta0 not rejected a (1 - alpha) confidence sequence.  It is
  xbar_n +/- t* s_n / sqrt(n) with t* in closed form, and the whole line
  while the data cannot yet exclude anything.

c is the effect size, in units of sigma, at which the test is most
sensitive.  The first observation carries no information about the
standardised effect, so B_1 = 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure
from ..errors import NumericalError
from ..numerics import require_finite
from .sprt import REJECT_H0

_UNDERFLOW = (
    "the sample variance underflows to zero although the observations differ; "
    "rescale the data in the adapter (for example to units where they are of order one)"
)


@dataclass(frozen=True)
class TMixtureState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    max_log_bf: float = 0.0
    decision: str | None = None
    low: float | None = None  # smallest and largest observation: tell identical data
    high: float | None = None  # from a variance that underflowed


@dataclass(frozen=True)
class TMixtureOutput:
    n: int
    mean: float
    sd: float | None
    t: float | None
    log_bf: float
    p_value: float
    lower: float | None  # None: the confidence sequence is still unbounded
    upper: float | None
    decision: str | None


class TMixtureSPRT(Procedure):
    name = "t_mixture_sprt"
    version = "2"

    def __init__(
        self,
        theta0: float = 0.0,
        effect: float = 0.5,
        alpha: float = 0.05,
        *,
        stop_on_reject: bool = True,
        input: str = "x",
    ):
        if not math.isfinite(theta0):
            raise ValueError("theta0 must be finite")
        if not (effect > 0 and math.isfinite(effect)):
            raise ValueError("effect (prior sd of the standardised effect) must be positive")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.theta0, self.effect, self.alpha = float(theta0), float(effect), float(alpha)
        self.stop_on_reject, self.input = stop_on_reject, input
        self._log_level = math.log(1.0 / self.alpha)

    def config(self) -> dict:
        return {
            "theta0": self.theta0,
            "effect": self.effect,
            "alpha": self.alpha,
            "stop_on_reject": self.stop_on_reject,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> TMixtureState:
        return TMixtureState()

    def log_bf(self, n: int, t2: float) -> float:
        nu, r = n - 1, 1.0 + n * self.effect**2
        return -0.5 * math.log(r) + 0.5 * (nu + 1) * (math.log1p(t2 / nu) - math.log1p(t2 / (nu * r)))

    def critical_t(self, n: int) -> float | None:
        """|t| at which B_n reaches 1/alpha, or None if it cannot at this n."""
        nu, r = n - 1, 1.0 + n * self.effect**2
        k = math.exp(2.0 * (self._log_level + 0.5 * math.log(r)) / (nu + 1))
        if k >= r:
            return None
        return math.sqrt(nu * (k - 1.0) / (1.0 - k / r))

    def step(self, state: TMixtureState, x, rng=None):
        v = x[self.input]
        n = state.n + 1
        delta = v - state.mean
        mean = state.mean + delta / n
        m2 = require_finite(state.m2 + delta * (v - mean), "sum of squared deviations")
        mean = require_finite(mean, "running mean")
        low = v if state.low is None else min(state.low, v)
        high = v if state.high is None else max(state.high, v)
        r = 1.0 + n * self.effect**2
        if n >= 2 and m2 <= 0.0 and low != high:
            raise NumericalError(_UNDERFLOW)
        if n < 2:
            sd = t = None
            log_bf = 0.0
            lower = upper = None
        elif m2 <= 0.0:
            # Identical observations: the Bayes factor at its limit t^2 -> infinity
            # (or t = 0 if they sit exactly at theta0), and a zero-width interval.
            sd, t = 0.0, None
            log_bf = -0.5 * math.log(r) if mean == self.theta0 else 0.5 * (n - 1) * math.log(r)
            lower = upper = mean if self.critical_t(n) is not None else None
        else:
            sd = math.sqrt(m2 / (n - 1))
            t = (mean - self.theta0) * math.sqrt(n) / sd
            t2 = t * t
            # beyond float range t^2 has reached the limit the Bayes factor tends to
            log_bf = 0.5 * (n - 1) * math.log(r) if math.isinf(t2) else self.log_bf(n, t2)
            log_bf = require_finite(log_bf, "log Bayes factor")
            crit = self.critical_t(n)
            lower = upper = None
            if crit is not None:
                half = crit * sd / math.sqrt(n)
                lower, upper = mean - half, mean + half
        max_log_bf = max(state.max_log_bf, log_bf)
        decision = state.decision or (REJECT_H0 if log_bf >= self._log_level else None)
        new = TMixtureState(n, mean, m2, max_log_bf, decision, low, high)
        p_value = min(1.0, math.exp(-max_log_bf))
        return new, TMixtureOutput(n, mean, sd, t, log_bf, p_value, lower, upper, decision)

    def is_terminal(self, state: TMixtureState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> TMixtureState:
        return TMixtureState(**data)
