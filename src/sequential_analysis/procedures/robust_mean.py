# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Confidence sequences for a mean without normality.

CatoniConfidenceSequence (Wang and Ramdas 2023) assumes only a known upper
bound sigma^2 on the variance -- heavy tails, skewness and outliers are
allowed.  With Catoni's influence function

    psi(y) = log(1 + y + y^2/2) for y >= 0,   psi(y) = -log(1 - y + y^2/2) for y < 0,

which satisfies exp(psi(y)) <= 1 + y + y^2/2 and exp(-psi(y)) <= 1 - y + y^2/2,
and predictable lambda_i > 0, both

    exp( +- sum_i psi(lambda_i (X_i - m)) - sigma^2 sum_i lambda_i^2 / 2 )

are non-negative supermartingales when m is the mean, so by Ville's
inequality the set of m with |sum psi(lambda_i (X_i - m))| <= sigma^2
sum lambda_i^2 / 2 + log(2/alpha) contains the mean at every n
simultaneously with probability at least 1 - alpha.  psi grows only
logarithmically, so a few wild observations move the interval little.
The interval is found by bisection over the stored sample.

AsymptoticConfidenceSequence (Waudby-Smith, Arbour, Sinha, Kennedy and
Ramdas 2024) assumes only a finite variance (and slightly more than two
moments); the variance is estimated.  It is Robbins' normal-mixture
boundary with the running standard deviation in place of sigma,

    xbar_n +- s_n sqrt( 2 (n rho^2 + 1) / (n^2 rho^2) log( sqrt(n rho^2 + 1) / alpha ) ),

and its coverage is 1 - alpha asymptotically (time-uniformly, as the
start of monitoring grows), not exactly.  In simulations at alpha = 0.05
the probability that the interval ever misses the mean was 0.06 for
normal data from the first observations, but for skewed data (lognormal,
Pareto with tail index 2.5) 0.21 when monitored from the start, 0.08 to
0.09 from n = 300 and 0.04 to 0.05 from n = 1000.  It therefore reports
nothing before `start` observations (default 1000).  Use it for long
runs; when a variance bound is known, or n is small, the Catoni sequence
has exact coverage.  rho is tuned so that the sequence is tightest after
`tuned_for` observations.  Its state is a handful of numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..numerics import ordered_sum
from .sprt import REJECT_H0


def _check(alpha: float, mean0: float | None) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if mean0 is not None and not math.isfinite(mean0):
        raise ValueError("mean0 must be finite or None")


@dataclass(frozen=True)
class CatoniState:
    n: int = 0
    sample: tuple[float, ...] = ()
    lower: float | None = None  # running intersection; None while unbounded
    upper: float | None = None
    decision: str | None = None


@dataclass(frozen=True)
class MeanCSOutput:
    n: int
    mean: float
    lower: float | None  # None: unbounded
    upper: float | None
    decision: str | None  # mean0 rejected (if a mean0 was given)


class CatoniConfidenceSequence(Procedure):
    """Confidence sequence for a mean, given only an upper bound `sigma` on the standard deviation.

    Valid for every distribution with that variance bound (and, more
    generally, for dependent data whose conditional mean is the constant m
    and conditional variance at most sigma^2).  With `mean0`, the procedure
    also tests H0: E[X] = mean0, rejecting when mean0 leaves the interval.
    The state holds the sample; each step costs O(n log(1/precision)).
    """

    name = "catoni_confidence_sequence"
    version = "1"

    def __init__(
        self,
        sigma: float,
        alpha: float = 0.05,
        *,
        mean0: float | None = None,
        stop_on_reject: bool = False,
        input: str = "x",
    ):
        if not (math.isfinite(sigma) and sigma > 0):
            raise ValueError("sigma must be a positive, finite bound on the standard deviation")
        _check(alpha, mean0)
        self.sigma, self.alpha = float(sigma), float(alpha)
        self.mean0 = None if mean0 is None else float(mean0)
        self.stop_on_reject, self.input = stop_on_reject, input
        self._log_level = math.log(2.0 / self.alpha)

    def config(self) -> dict:
        return {
            "sigma": self.sigma,
            "alpha": self.alpha,
            "mean0": self.mean0,
            "stop_on_reject": self.stop_on_reject,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> CatoniState:
        return CatoniState()

    def rate(self, i: int) -> float:
        """lambda_i, the predictable stake on the i-th observation (Wang and Ramdas 2023)."""
        return math.sqrt(2.0 * self._log_level / (self.sigma**2 * i * math.log(i + 1.0)))

    def interval(self, sample: tuple[float, ...], guess: tuple[float, float] | None = None) -> tuple[float, float]:
        """The means m kept after the sample (in arrival order), before intersection over time.

        Each bound is a root of a decreasing function of m, found by Newton steps kept inside a
        bracket (bisection when a step leaves it), from `guess` -- the previous bounds -- if given.
        """
        rates = [self.rate(i) for i in range(1, len(sample) + 1)]
        budget = 0.5 * self.sigma**2 * ordered_sum(r * r for r in rates) + self._log_level
        scale = max(max(sample) - min(sample), self.sigma)

        def score(m: float) -> tuple[float, float]:  # sum psi(r (x - m)), decreasing in m, and its derivative
            value, slope = 0.0, 0.0
            for r, x in zip(rates, sample, strict=True):
                y = r * (x - m)
                if y >= 0.0:
                    value += math.log1p(y + 0.5 * y * y)
                    slope -= r * (1.0 + y) / (1.0 + y + 0.5 * y * y)
                else:
                    value -= math.log1p(-y + 0.5 * y * y)
                    slope -= r * (1.0 - y) / (1.0 - y + 0.5 * y * y)
            return value, slope

        def root(target: float, start: float) -> float:
            lo = hi = start
            width = scale
            while score(lo)[0] < target:  # the root lies below
                lo -= width
                width *= 2.0
            width = scale
            while score(hi)[0] > target:  # the root lies above
                hi += width
                width *= 2.0
            m = start if lo <= start <= hi else 0.5 * (lo + hi)
            for _ in range(200):
                value, slope = score(m)
                value -= target
                if value > 0.0:
                    lo = m
                elif value < 0.0:
                    hi = m
                else:
                    return m
                if hi - lo <= 1e-12 * (abs(lo) + abs(hi) + scale):
                    break
                nxt = m - value / slope if slope < 0.0 else 0.5 * (lo + hi)
                m = nxt if lo < nxt < hi else 0.5 * (lo + hi)
            return 0.5 * (lo + hi)

        centre = ordered_sum(sample) / len(sample)
        lo_guess, hi_guess = guess if guess is not None else (centre, centre)
        return root(budget, lo_guess), root(-budget, hi_guess)

    def step(self, state: CatoniState, x, rng=None):
        sample = (*state.sample, float(x[self.input]))
        n = state.n + 1
        guess = None if state.lower is None or state.upper is None else (state.lower, state.upper)
        lo, hi = self.interval(sample, guess)
        if state.lower is not None:
            lo = max(lo, state.lower)
        if state.upper is not None:
            hi = min(hi, state.upper)
        decision = state.decision
        if decision is None and self.mean0 is not None and not lo <= self.mean0 <= hi:
            decision = REJECT_H0
        new = CatoniState(n, sample, lo, hi, decision)
        return new, MeanCSOutput(n, ordered_sum(sample) / n, lo, hi, decision)

    def is_terminal(self, state: CatoniState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def encode_state(self, state: CatoniState):
        return {"n": state.n, "sample": list(state.sample), "lower": state.lower, "upper": state.upper,
                "decision": state.decision}

    def decode_state(self, data) -> CatoniState:
        state = decode_fields(CatoniState, data)
        require_state(len(state.sample) == state.n, "sample size differs from n")
        require_state((state.lower is None) == (state.n == 0) == (state.upper is None), "bounds without data")
        return state


@dataclass(frozen=True)
class AsymptoticState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    lower: float | None = None
    upper: float | None = None
    decision: str | None = None


class AsymptoticConfidenceSequence(Procedure):
    """Asymptotic confidence sequence for a mean: needs only a finite variance, which it estimates.

    The anytime-valid counterpart of the t interval for data far from
    normal (Waudby-Smith et al. 2024): coverage 1 - alpha holds
    asymptotically and uniformly over time, not exactly at small n -- see
    the module docstring for how large n has to be.  No interval (None)
    is reported before `start` observations, nor while all of them are
    equal; it is not intersected over time, as asymptotic coverage does
    not license that.
    With `mean0`, also tests H0: E[X] = mean0 (rejecting when mean0
    leaves the interval).
    """

    name = "asymptotic_confidence_sequence"
    version = "1"

    def __init__(
        self,
        alpha: float = 0.05,
        *,
        mean0: float | None = None,
        start: int = 1000,
        tuned_for: float = 1000.0,
        stop_on_reject: bool = False,
        input: str = "x",
    ):
        _check(alpha, mean0)
        if not (math.isfinite(tuned_for) and tuned_for >= 1):
            raise ValueError("tuned_for must be a number of observations, at least 1")
        if type(start) is not int or start < 2:
            raise ValueError("start must be an int, at least 2")
        self.alpha, self.tuned_for, self.start = float(alpha), float(tuned_for), start
        self.mean0 = None if mean0 is None else float(mean0)
        self.stop_on_reject, self.input = stop_on_reject, input
        c = -2.0 * math.log(self.alpha)
        self._rho2 = (c + math.log1p(c)) / self.tuned_for

    def config(self) -> dict:
        return {
            "alpha": self.alpha,
            "mean0": self.mean0,
            "start": self.start,
            "tuned_for": self.tuned_for,
            "stop_on_reject": self.stop_on_reject,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> AsymptoticState:
        return AsymptoticState()

    def half_width(self, n: int, sd: float) -> float:
        v = n * self._rho2 + 1.0
        return sd * math.sqrt(2.0 * v / (n * n * self._rho2) * (0.5 * math.log(v) - math.log(self.alpha)))

    def step(self, state: AsymptoticState, x, rng=None):
        v = x[self.input]
        n = state.n + 1
        delta = v - state.mean
        mean = state.mean + delta / n
        m2 = state.m2 + delta * (v - mean)
        lower = upper = None
        if n >= self.start and m2 > 0.0:
            h = self.half_width(n, math.sqrt(m2 / (n - 1)))
            lower, upper = mean - h, mean + h
        decision = state.decision
        if decision is None and self.mean0 is not None and lower is not None and upper is not None:
            if not lower <= self.mean0 <= upper:
                decision = REJECT_H0
        new = AsymptoticState(n, mean, m2, lower, upper, decision)
        return new, MeanCSOutput(n, mean, lower, upper, decision)

    def is_terminal(self, state: AsymptoticState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> AsymptoticState:
        state = decode_fields(AsymptoticState, data)
        require_state(state.m2 >= 0, "negative sum of squares")
        return state
