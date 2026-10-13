# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Anytime-valid inference for a quantile, for every distribution (Howard and Ramdas 2022).

Whatever the distribution of the X_i -- heavy-tailed, skewed, discrete --
the indicators 1{X_i <= q} are Bernoulli with mean F(q).  For a p-quantile
q of the distribution, F(q) >= p and F(q-) <= p: the counts of
observations at or below q cannot fall far below p n, nor the counts
strictly below q rise far above it.  How far is measured by a mixture of
likelihood ratios over the Bernoulli mean theta, with a beta prior
restricted to one side of p,

    M_n(k) = integral over theta < p of (theta/p)^k ((1-theta)/(1-p))^(n-k) dBeta(theta; a, b) / P(theta < p),

for k = #{X_i <= q} (and symmetrically, over theta > p, for #{X_i < q}).
Under F(q) >= p it is a non-negative supermartingale with mean at most
one, so by Ville's inequality it exceeds 2/alpha with probability at most
alpha/2, at any time.  This gives

* `QuantileTest`: an anytime-valid test of "q0 is a p-quantile" (for
  p = 1/2, of the median) from two counts.  Unlike the sign test it is
  valid when the distribution has an atom at q0 -- discrete data, ties --
  as long as q0 is a p-quantile;
* `QuantileConfidenceSequence`: the values q that the test keeps, an
  interval between two order statistics that contains every p-quantile
  at every n simultaneously with probability at least 1 - alpha.

No moments, no density and no continuity are assumed; only independence
of the observations given the past and a common distribution (or, more
generally, F_i(q) >= p and F_i(q-) <= p for each i given the past).

The prior Beta(r p, r (1 - p)) is tuned by `tuned_for`, the number of
observations at which the sequence is to be tightest; any value is valid.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..distributions import regularized_beta
from .sprt import REJECT_H0

ALTERNATIVES = ("two-sided", "greater", "less")


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class _Mixture:
    """The one-sided beta-binomial mixtures for a hypothesised probability p."""

    def __init__(self, p: float, alpha: float, tuned_for: float, two_sided: bool):
        # a normal mixture with variance ratio rho^2 is tightest near n = (2 log(1/a) + log(1 + 2 log(1/a))) / rho^2;
        # the beta prior Beta(r p, r (1 - p)) has variance p (1 - p) / (r + 1), i.e. rho^2 = 1 / (r + 1)
        level = alpha / 2.0 if two_sided else alpha
        c = 2.0 * math.log(1.0 / level)
        self.r = max(1.0, tuned_for / (c + math.log1p(c)) - 1.0)
        self.p, self.a, self.b = p, self.r * p, self.r * (1.0 - p)
        self.log_level = math.log(1.0 / level)
        self._log_mass_below = math.log(regularized_beta(p, self.a, self.b))
        self._log_mass_above = math.log(regularized_beta(1.0 - p, self.b, self.a))
        self._norm = _log_beta(self.a, self.b)

    def log_below(self, k: int, n: int) -> float:
        """log M_n(k), mixing over theta < p: evidence that F(q) < p, from k = #{X_i <= q}."""
        mass = regularized_beta(self.p, self.a + k, self.b + n - k)
        if mass <= 0.0:
            return -math.inf
        return (_log_beta(self.a + k, self.b + n - k) + math.log(mass) - self._norm - self._log_mass_below
                - k * math.log(self.p) - (n - k) * math.log1p(-self.p))

    def log_above(self, k: int, n: int) -> float:
        """The mirror image, mixing over theta > p: evidence that F(q-) > p, from k = #{X_i < q}."""
        mass = regularized_beta(1.0 - self.p, self.b + n - k, self.a + k)
        if mass <= 0.0:
            return -math.inf
        return (_log_beta(self.a + k, self.b + n - k) + math.log(mass) - self._norm - self._log_mass_above
                - k * math.log(self.p) - (n - k) * math.log1p(-self.p))

    def smallest_kept(self, n: int) -> int:
        """The least k = #{X_i <= q} that log_below does not reject (it decreases in k)."""
        lo, hi = 0, n
        if self.log_below(0, n) < self.log_level:
            return 0
        while hi - lo > 1:  # log_below(lo) rejects; find the first k that does not
            mid = (lo + hi) // 2
            if self.log_below(mid, n) >= self.log_level:
                lo = mid
            else:
                hi = mid
        return hi if self.log_below(hi, n) < self.log_level else n + 1

    def largest_kept(self, n: int) -> int:
        """The greatest k = #{X_i < q} that log_above does not reject (it increases in k)."""
        lo, hi = 0, n
        if self.log_above(n, n) < self.log_level:
            return n
        if self.log_above(0, n) >= self.log_level:
            return -1
        while hi - lo > 1:  # log_above(lo) keeps, log_above(hi) rejects
            mid = (lo + hi) // 2
            if self.log_above(mid, n) < self.log_level:
                lo = mid
            else:
                hi = mid
        return lo


def _check(p: float, alpha: float, tuned_for: float) -> None:
    if not 0 < p < 1:
        raise ValueError("p must lie strictly between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not (math.isfinite(tuned_for) and tuned_for >= 1):
        raise ValueError("tuned_for must be a number of observations, at least 1")


@dataclass(frozen=True)
class QuantileTestState:
    n: int = 0
    below: int = 0  # observations < q0
    at_or_below: int = 0  # observations <= q0
    max_log_evidence: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class QuantileTestOutput:
    n: int
    fraction_at_or_below: float | None
    log_evidence: float
    p_value: float
    decision: str | None


class QuantileTest(Procedure):
    """Anytime-valid test of H0: q0 is a p-quantile, i.e. P(X < q0) <= p <= P(X <= q0).

    `alternative="greater"` rejects when the p-quantile is above q0 (too
    few observations at or below q0), `"less"` when it is below q0; the
    two-sided test spends alpha/2 on each side.  `p_value` is always valid
    and never increases.  The state is three counts, whatever the length
    of the run.
    """

    name = "quantile_test"
    version = "1"

    def __init__(
        self,
        q0: float,
        p: float = 0.5,
        alpha: float = 0.05,
        *,
        alternative: str = "two-sided",
        tuned_for: float = 1000.0,
        stop_on_reject: bool = True,
        input: str = "x",
    ):
        _check(p, alpha, tuned_for)
        if not math.isfinite(q0):
            raise ValueError("q0 must be finite")
        if alternative not in ALTERNATIVES:
            raise ValueError(f"alternative must be one of {ALTERNATIVES}")
        self.q0, self.p, self.alpha, self.alternative = float(q0), float(p), float(alpha), alternative
        self.tuned_for, self.stop_on_reject, self.input = float(tuned_for), stop_on_reject, input
        self._mix = _Mixture(self.p, self.alpha, self.tuned_for, alternative == "two-sided")

    def config(self) -> dict:
        return {
            "q0": self.q0,
            "p": self.p,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "tuned_for": self.tuned_for,
            "stop_on_reject": self.stop_on_reject,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> QuantileTestState:
        return QuantileTestState()

    def log_evidence(self, n: int, below: int, at_or_below: int) -> float:
        """log of the larger mixture, minus log 2 for a two-sided test: rejects at log(1/alpha)."""
        if n == 0:
            return 0.0
        sides = []
        if self.alternative != "less":
            sides.append(self._mix.log_below(at_or_below, n))
        if self.alternative != "greater":
            sides.append(self._mix.log_above(below, n))
        shift = math.log(2.0) if self.alternative == "two-sided" else 0.0
        return max(sides) - shift

    def step(self, state: QuantileTestState, x, rng=None):
        v = x[self.input]
        n = state.n + 1
        below = state.below + (v < self.q0)
        at_or_below = state.at_or_below + (v <= self.q0)
        evidence = self.log_evidence(n, below, at_or_below)
        best = max(state.max_log_evidence, evidence)
        decision = state.decision or (REJECT_H0 if evidence >= math.log(1.0 / self.alpha) else None)
        new = QuantileTestState(n, below, at_or_below, best, decision)
        return new, QuantileTestOutput(n, at_or_below / n, evidence, min(1.0, math.exp(-best)), decision)

    def is_terminal(self, state: QuantileTestState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> QuantileTestState:
        state = decode_fields(QuantileTestState, data)
        require_state(state.below <= state.at_or_below <= state.n, "counts out of order")
        return state


@dataclass(frozen=True)
class QuantileCSState:
    n: int = 0
    sample: tuple[float, ...] = ()  # sorted
    lower: float | None = None  # running intersection; None for -infinity
    upper: float | None = None  # None for +infinity


@dataclass(frozen=True)
class QuantileCSOutput:
    n: int
    estimate: float | None  # the sample p-quantile (lower one)
    lower: float | None  # None: unbounded
    upper: float | None


class QuantileConfidenceSequence(Procedure):
    """Confidence sequence for the p-quantile of any distribution.

    With probability at least 1 - alpha, every p-quantile of the
    distribution lies in [lower, upper] at every n simultaneously, so the
    interval may be watched continuously and the experiment stopped
    whenever it is narrow enough.  The bounds are order statistics of the
    sample, intersected over time; None stands for an infinite bound
    (too few observations so far).

    The state holds the sorted sample: memory grows with the run, and each
    step costs O(n) to insert and O(log n) evaluations of the incomplete
    beta function.  To test one value q0, `QuantileTest` needs three counts.
    """

    name = "quantile_confidence_sequence"
    version = "1"

    def __init__(self, p: float = 0.5, alpha: float = 0.05, *, tuned_for: float = 1000.0, input: str = "x"):
        _check(p, alpha, tuned_for)
        self.p, self.alpha, self.tuned_for, self.input = float(p), float(alpha), float(tuned_for), input
        self._mix = _Mixture(self.p, self.alpha, self.tuned_for, True)

    def config(self) -> dict:
        return {"p": self.p, "alpha": self.alpha, "tuned_for": self.tuned_for, "input": self.input}

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> QuantileCSState:
        return QuantileCSState()

    def bounds(self, sample: tuple[float, ...]) -> tuple[float, float]:
        """The interval the mixtures keep at n = len(sample), before intersection over time."""
        n = len(sample)
        k_lo = self._mix.smallest_kept(n)  # need #{X <= q} >= k_lo
        k_hi = self._mix.largest_kept(n)  # need #{X < q} <= k_hi
        lower = -math.inf if k_lo <= 0 else (sample[k_lo - 1] if k_lo <= n else math.inf)
        upper = math.inf if k_hi + 1 > n else (sample[k_hi] if k_hi >= 0 else -math.inf)
        return lower, upper

    def step(self, state: QuantileCSState, x, rng=None):
        v = float(x[self.input])
        sample = list(state.sample)
        bisect.insort(sample, v)
        n = state.n + 1
        lo, hi = self.bounds(tuple(sample))
        old_lo = -math.inf if state.lower is None else state.lower
        old_hi = math.inf if state.upper is None else state.upper
        lo, hi = max(lo, old_lo), min(hi, old_hi)
        lower = lo if math.isfinite(lo) else None
        upper = hi if math.isfinite(hi) else None
        k = max(1, math.ceil(self.p * n))  # the lower sample p-quantile: the k-th order statistic
        new = QuantileCSState(n, tuple(sample), lower, upper)
        return new, QuantileCSOutput(n, sample[k - 1], lower, upper)

    def encode_state(self, state: QuantileCSState):
        return {"n": state.n, "sample": list(state.sample), "lower": state.lower, "upper": state.upper}

    def decode_state(self, data) -> QuantileCSState:
        state = decode_fields(QuantileCSState, data)
        require_state(len(state.sample) == state.n, "sample size differs from n")
        require_state(all(a <= b for a, b in zip(state.sample, state.sample[1:], strict=False)), "sample not sorted")
        return state
