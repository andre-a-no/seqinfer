# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Nonparametric anytime-valid tests by betting.

Testing by betting (Shafer 2021; Waudby-Smith and Ramdas 2024) needs no
model for the data.  For observations Y_i in [0, 1] and the null
hypothesis E[Y_i | past] = m, a gambler who starts with wealth 1 and
stakes a fraction lambda_i, chosen before Y_i is seen, ends with

    K_n = prod_{i<=n} (1 + lambda_i (Y_i - m)).

Under the null K_n is a non-negative martingale with mean one, whatever
the distribution of the Y_i, so by Ville's inequality
P(sup_n K_n >= 1/alpha) <= alpha.  Rejecting when the wealth reaches
1/alpha is a level-alpha test that may be monitored after every
observation and stopped at any time; 1 / max_k K_k is an always-valid
p-value.  The only assumption is boundedness (for the mean test) or a
median (for the sign test); no distribution family is involved.

Bets follow the aGRAPA rule: lambda = (mu - m) / (s2 + (mu - m)^2) from
predictable estimates mu, s2 of the mean and variance, truncated to
[0, c/m] for bets on a larger mean and [0, c/(1 - m)] for bets on a
smaller one (c = 1/2), so that the wealth stays positive.  A two-sided
test averages the two one-sided wealth processes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure
from ..numerics import require_finite
from .sprt import REJECT_H0

ALTERNATIVES = ("two-sided", "greater", "less")
_TRUNCATION = 0.5
_LOG2 = math.log(2.0)


@dataclass(frozen=True)
class BettingState:
    n: int = 0
    mean: float = 0.0  # running mean of the scaled observations
    m2: float = 0.0  # running sum of squared deviations
    log_up: float = 0.0  # log wealth betting on a larger mean
    log_down: float = 0.0  # log wealth betting on a smaller mean
    max_log_wealth: float = 0.0
    ties: int = 0  # sign test: observations equal to the median, not bet on
    decision: str | None = None


@dataclass(frozen=True)
class BettingOutput:
    n: int
    estimate: float
    log_wealth: float
    p_value: float
    decision: str | None


def _log_add(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class _Betting(Procedure):
    """Shared wealth process; subclasses map an input to a value in [0, 1]."""

    def _setup(self, m: float, alpha: float, alternative: str, stop_on_reject: bool) -> None:
        if not 0 < m < 1:
            raise ValueError("the null value must lie strictly inside the range of the data")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if alternative not in ALTERNATIVES:
            raise ValueError(f"alternative must be one of {ALTERNATIVES}")
        self.m, self.alpha, self.alternative, self.stop_on_reject = m, float(alpha), alternative, stop_on_reject
        self._log_level = math.log(1.0 / self.alpha)

    def initial_state(self) -> BettingState:
        return BettingState()

    def _bets(self, state: BettingState) -> tuple[float, float]:
        t = state.n
        mu = (0.5 + t * state.mean) / (t + 1)  # shrunk towards 1/2: predictable, defined at t = 0
        s2 = (0.25 + state.m2) / (t + 1)
        d = mu - self.m
        lam = d / (s2 + d * d)
        up = min(max(lam, 0.0), _TRUNCATION / self.m) if self.alternative != "less" else 0.0
        down = min(max(-lam, 0.0), _TRUNCATION / (1.0 - self.m)) if self.alternative != "greater" else 0.0
        return up, down

    def _log_wealth(self, log_up: float, log_down: float) -> float:
        if self.alternative == "greater":
            return log_up
        if self.alternative == "less":
            return log_down
        return _log_add(log_up, log_down) - _LOG2

    def _bet(self, state: BettingState, y: float, ties: int) -> tuple[BettingState, BettingOutput]:
        up, down = self._bets(state)
        log_up = state.log_up + math.log1p(up * (y - self.m))
        log_down = state.log_down + math.log1p(-down * (y - self.m))
        n = state.n + 1
        delta = y - state.mean
        mean = state.mean + delta / n
        m2 = state.m2 + delta * (y - mean)
        log_wealth = require_finite(self._log_wealth(log_up, log_down), "log wealth")
        max_log_wealth = max(state.max_log_wealth, log_wealth)
        decision = state.decision or (REJECT_H0 if log_wealth >= self._log_level else None)
        new = BettingState(n, mean, m2, log_up, log_down, max_log_wealth, ties, decision)
        return new, self._output(new, log_wealth)

    def _output(self, state: BettingState, log_wealth: float) -> BettingOutput:
        return BettingOutput(state.n, state.mean, log_wealth, min(1.0, math.exp(-state.max_log_wealth)), state.decision)

    def is_terminal(self, state: BettingState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> BettingState:
        return BettingState(**data)


class BettingMeanTest(_Betting):
    """Anytime-valid test of E[X] = mean0 for observations bounded in [lower, upper].

    Valid for every distribution on [lower, upper] -- including skewed,
    discrete and heavy-tailed-but-bounded ones -- and for dependent data
    whose conditional mean given the past equals mean0.
    """

    name = "betting_mean_test"
    version = "1"

    def __init__(
        self,
        mean0: float,
        lower: float = 0.0,
        upper: float = 1.0,
        alpha: float = 0.05,
        *,
        alternative: str = "two-sided",
        stop_on_reject: bool = True,
        input: str = "x",
    ):
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("need finite bounds lower < upper")
        self.mean0, self.lower, self.upper, self.input = float(mean0), float(lower), float(upper), input
        self._setup((self.mean0 - self.lower) / (self.upper - self.lower), alpha, alternative, stop_on_reject)

    def config(self) -> dict:
        return {
            "mean0": self.mean0,
            "lower": self.lower,
            "upper": self.upper,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "stop_on_reject": self.stop_on_reject,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input, lower=self.lower, upper=self.upper)}

    def step(self, state: BettingState, x, rng=None):
        y = (x[self.input] - self.lower) / (self.upper - self.lower)
        new, out = self._bet(state, y, state.ties)
        estimate = self.lower + new.mean * (self.upper - self.lower)
        return new, BettingOutput(out.n, estimate, out.log_wealth, out.p_value, out.decision)


class SequentialSignTest(_Betting):
    """Anytime-valid sign test of the median: H0 says P(X > median0) = P(X < median0).

    Only the signs of X - median0 are used, so the test is valid for every
    distribution with that median, with no moment assumptions at all
    (Cauchy data are fine).  Observations equal to median0 carry no sign;
    they are counted as ties and skipped.  For paired data, test the
    differences against 0: pair the inputs with PositionalPair and map
    them with `difference`.

    `estimate` is the fraction of positive signs among the non-tied
    observations.
    """

    name = "sequential_sign_test"
    version = "1"

    def __init__(
        self,
        median0: float = 0.0,
        alpha: float = 0.05,
        *,
        alternative: str = "two-sided",
        stop_on_reject: bool = True,
        input: str = "x",
    ):
        if not math.isfinite(median0):
            raise ValueError("median0 must be finite")
        self.median0, self.input = float(median0), input
        self._setup(0.5, alpha, alternative, stop_on_reject)

    def config(self) -> dict:
        return {
            "median0": self.median0,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "stop_on_reject": self.stop_on_reject,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def step(self, state: BettingState, x, rng=None):
        v = x[self.input]
        if v == self.median0:
            new = BettingState(
                state.n, state.mean, state.m2, state.log_up, state.log_down,
                state.max_log_wealth, state.ties + 1, state.decision,
            )
            return new, self._output(new, self._log_wealth(state.log_up, state.log_down))
        return self._bet(state, 1.0 if v > self.median0 else 0.0, state.ties)
