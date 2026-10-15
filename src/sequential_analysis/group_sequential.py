# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Group sequential designs with alpha spending (Lan and DeMets 1983).

A trial is analysed K times, at information fractions 0 < t_1 < ... < t_K = 1
(for a normal mean with known variance, t_k = n_k / n_K).  At analysis k
the standardised statistic Z_k is compared with an efficacy boundary c_k;
the trial stops for efficacy the first time Z_k >= c_k (or |Z_k| >= c_k
for a two-sided test).  A spending function alpha(t) says how much of the
type I error may be used up by information t; the boundaries are chosen
so that, under H0,

    P(first crossing at analysis k) = alpha(t_k) - alpha(t_{k-1}).

Under H0 the score process B(t) = Z(t) sqrt(t) is a Brownian motion, so
these probabilities are computed by recursive numerical integration of
its sub-density on the continuation region (Armitage, McPherson and Rowe
1969; Jennison and Turnbull 2000, ch. 19), here with Simpson's rule.
The same recursion with a drift gives power and expected sample size.

Spending functions (one-sided level a; a two-sided test spends a/2 on
each side):

    "obrien-fleming"  2 - 2 Phi(Phi^-1(1 - a/2) / sqrt(t))    (Lan-DeMets)
    "pocock"          a log(1 + (e - 1) t)                    (Lan-DeMets)
    ("power", rho)    a t^rho                                 (Kim-DeMets)
    a callable        t -> cumulative alpha, with f(1) = a
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Any

from ._compat import pairwise, zip_strict

_N = NormalDist()
_SQRT2 = math.sqrt(2.0)


def _upper(x: float) -> float:
    """1 - Phi(x), through erfc so that small tails keep their relative accuracy."""
    return 0.5 * math.erfc(x / _SQRT2)
_LOWER = -10.0  # B(t) / sqrt(t) below this is never reached in practice
_NEVER = 40.0  # a boundary no path reaches: stands in for an infinite one in the recursion


def spending_function(kind, alpha: float) -> Callable[[float], float]:
    """Cumulative alpha spent by information fraction t, for one side."""
    if callable(kind):
        custom: Callable[[float], float] = kind
        return custom
    if kind == "obrien-fleming":
        z = -_N.inv_cdf(alpha / 2.0)  # not inv_cdf(1 - alpha/2): 1 - alpha/2 rounds away a small alpha
        return lambda t: 0.0 if t <= 0 else 2.0 * _upper(z / math.sqrt(t))
    if kind == "pocock":
        return lambda t: alpha * math.log(1.0 + (math.e - 1.0) * t)
    if isinstance(kind, tuple) and len(kind) == 2 and kind[0] == "power" and kind[1] > 0:
        rho = float(kind[1])
        return lambda t: alpha * t**rho
    raise ValueError("spending must be 'obrien-fleming', 'pocock', ('power', rho) or a callable")


def _simpson(xs_count: int) -> list[float]:
    w = [2.0 if i % 2 == 0 else 4.0 for i in range(xs_count)]
    w[0] = w[-1] = 1.0
    return w


class _Recursion:
    """Sub-density of B(t_k) on the continuation region, propagated analysis by analysis."""

    def __init__(self, drift: float, two_sided: bool, points_per_sd: int):
        self.drift, self.two_sided, self.points = drift, two_sided, points_per_sd
        self.t = 0.0
        self.xs: list[float] = [0.0]  # degenerate start at B(0) = 0
        self.mass: list[float] = [1.0]  # quadrature weight times density; a point mass at the start

    def _grid(self, lo: float, hi: float, sd: float) -> list[float]:
        m = max(2, math.ceil((hi - lo) / sd * self.points))
        m += m % 2  # Simpson needs an even number of intervals
        h = (hi - lo) / m
        return [lo + i * h for i in range(m + 1)]

    def cross(self, t: float, b: float, upper_only: bool = False) -> float:
        """P(continue to t_{k-1} and cross the boundary b sqrt(t) at t); of a two-sided design either one,
        unless `upper_only`."""
        sd = math.sqrt(t - self.t)
        mu = self.drift * (t - self.t)
        upper = b * math.sqrt(t)
        both = self.two_sided and not upper_only
        p = 0.0
        for x, m in zip_strict(self.xs, self.mass):
            p += m * _upper((upper - x - mu) / sd)
            if both:
                p += m * _upper((upper + x + mu) / sd)
        return p

    def cross_below(self, t: float, f: float) -> float:
        """P(continue to t_{k-1} and end at or below the futility boundary f sqrt(t) at t)."""
        sd = math.sqrt(t - self.t)
        mu = self.drift * (t - self.t)
        lower = f * math.sqrt(t)
        return float(sum(m * _upper((x + mu - lower) / sd) for x, m in zip_strict(self.xs, self.mass)))

    def advance(self, t: float, b: float, f: float | None = None) -> None:
        """Move to t, keeping only paths that crossed neither boundary."""
        sd = math.sqrt(t - self.t)
        mu = self.drift * (t - self.t)
        upper = b * math.sqrt(t)
        if f is not None:
            lower = f * math.sqrt(t)
        elif self.two_sided:
            lower = -upper
        else:
            lower = min(_LOWER * math.sqrt(t) + self.drift * t, upper - 1.0)
        ys = self._grid(lower, upper, sd)
        h = (ys[1] - ys[0]) / 3.0
        weights = _simpson(len(ys))
        density = []
        for y in ys:
            f = 0.0
            for x, m in zip_strict(self.xs, self.mass):
                z = (y - x - mu) / sd
                f += m * math.exp(-0.5 * z * z)
            density.append(f / (sd * math.sqrt(2.0 * math.pi)))
        self.t, self.xs = t, ys
        self.mass = [h * w * f for w, f in zip_strict(weights, density)]


def _efficacy_bounds(ts: Sequence[float], cumulative: Sequence[float], two_sided: bool,
                     points_per_sd: int) -> list[float]:
    """Boundaries for Z_k spending cumulative[k] (one side) by analysis k, under H0."""
    rec = _Recursion(0.0, two_sided, points_per_sd)
    bounds, previous = [], 0.0
    for k, (t, c) in enumerate(zip_strict(ts, cumulative)):
        target = (c - previous) * (2.0 if two_sided else 1.0)
        previous = c
        if target <= 0:
            b = math.inf
        else:
            lo, hi = 0.0, 40.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if rec.cross(t, mid) > target:
                    lo = mid
                else:
                    hi = mid
            b = 0.5 * (lo + hi)
        bounds.append(b)
        if k + 1 < len(ts):
            rec.advance(t, b if math.isfinite(b) else _NEVER)
    return bounds


def _futility_bounds(ts: Sequence[float], finite: Sequence[float], beta_cumulative: Sequence[float], drift: float,
                     points_per_sd: int) -> tuple[list[float], float]:
    """Futility boundaries before the last analysis spending beta at this drift, and the beta left at the end.

    `finite` are the efficacy boundaries with infinite ones replaced by _NEVER.
    """
    rec = _Recursion(drift, False, points_per_sd)
    fs, previous = [], 0.0
    for t, b, c in zip_strict(ts[:-1], finite[:-1], beta_cumulative[:-1]):
        target = c - previous
        previous = c
        if rec.cross_below(t, b) <= target:  # the boundaries meet before the end
            f = b
        else:
            lo, hi = -40.0, b
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if rec.cross_below(t, mid) > target:
                    hi = mid
                else:
                    lo = mid
            f = 0.5 * (lo + hi)
        fs.append(f)
        rec.advance(t, b, f)
    return fs, previous + rec.cross_below(ts[-1], finite[-1])


def _cumulative(spend: Callable[[float], float], ts: Sequence[float], total: float) -> list[float]:
    """Cumulative spending at analyses ts; the last analysis is final and spends everything left."""
    return [spend(min(t, 1.0)) for t in ts[:-1]] + [total]


@dataclass(frozen=True)
class GroupSequentialDesign:
    fractions: tuple[float, ...]
    bounds: tuple[float, ...]  # efficacy boundaries for Z_k
    spent: tuple[float, ...]  # cumulative alpha by analysis (one side)
    alpha: float
    two_sided: bool
    futility: tuple[float, ...] | None = None  # non-binding futility boundaries for Z_k
    beta_spent: tuple[float, ...] | None = None  # cumulative beta by analysis
    drift: float | None = None  # theta sqrt(I_max) the futility boundaries were built for
    spending: Any = None  # the specifications the design was built from, for `at`
    futility_spending: Any = None
    points_per_sd: int = 24

    def _walk(self, drift: float, points_per_sd: int, binding: bool,
              upper_only: bool = False) -> tuple[list[float], list[float]]:
        """First-crossing probabilities of the efficacy and of the futility boundary at each analysis."""
        rec = _Recursion(drift, self.two_sided, points_per_sd)
        futility = self.futility if binding else None
        up, down = [], []
        last = len(self.fractions) - 1
        for k, (t, b) in enumerate(zip_strict(self.fractions, self.bounds)):
            up.append(rec.cross(t, b, upper_only))
            f = futility[k] if futility is not None else None
            down.append(rec.cross_below(t, f) if f is not None else 0.0)
            if k < last:
                rec.advance(t, b if math.isfinite(b) else _NEVER, f)
        return up, down

    def crossing(self, drift: float = 0.0, points_per_sd: int = 24, *, binding: bool = True) -> tuple[float, ...]:
        """P(stop for efficacy at analysis k), with Z(t) sqrt(t) drifting at `drift` per unit information.

        With futility boundaries, `binding=True` assumes the trial stops when
        it crosses one; `binding=False` ignores them, which is how the type I
        error of a non-binding design is controlled.
        """
        return tuple(self._walk(drift, points_per_sd, binding)[0])

    def futility_crossing(self, drift: float = 0.0, points_per_sd: int = 24) -> tuple[float, ...]:
        """P(stop for futility at analysis k); zeros without futility boundaries."""
        return tuple(self._walk(drift, points_per_sd, True)[1])

    def power(self, drift: float) -> float:
        """P(reject H0) at this drift; for a two-sided design, P(reject in the direction of the drift).

        A two-sided trial that stops for the wrong sign has not shown the
        effect it was powered for; rpact and gsDesign count power the same
        way.  `sum(crossing(drift))` counts both sides.
        """
        return sum(self._walk(drift, 24, True, upper_only=True)[0])

    def expected_fraction(self, drift: float) -> float:
        """Expected information at stopping, as a fraction of the maximum (futility obeyed)."""
        up, down = self._walk(drift, 24, True)
        stopped_early = 0.0
        expected = 0.0
        for t, p, q in zip_strict(self.fractions[:-1], up[:-1], down[:-1]):
            expected += t * (p + q)
            stopped_early += p + q
        return expected + self.fractions[-1] * (1.0 - stopped_early)

    def drift_for_power(self, power: float) -> float:
        """Drift theta * sqrt(I_max) at which the design has the given power.

        Solved for probit(power(drift)) = probit(power), which is nearly
        linear in the drift (exactly, for a single analysis), by the
        Illinois method: a few evaluations of the power instead of fifty.
        """
        if not self.alpha < power < 1.0:
            raise ValueError(f"power must lie between alpha ({self.alpha}) and 1")
        target = _N.inv_cdf(power)

        def gap(drift: float) -> float:
            p = min(max(self.power(drift), 1e-300), 1.0 - 1e-16)
            return _N.inv_cdf(p) - target

        side = self.alpha / 2.0 if self.two_sided else self.alpha
        lo, f_lo = 0.0, gap(0.0)
        hi = -_N.inv_cdf(side) + target  # the drift of a single analysis: a lower bound for most designs
        f_hi = gap(hi)
        while f_hi < 0.0:
            lo, f_lo, hi = hi, f_hi, 1.5 * hi
            f_hi = gap(hi)
        moved = 0
        for _ in range(100):
            if hi - lo <= 1e-12 * hi:
                break
            mid = hi - f_hi * (hi - lo) / (f_hi - f_lo)
            if not lo < mid < hi:
                mid = 0.5 * (lo + hi)
            f_mid = gap(mid)
            if f_mid < 0.0:
                lo, f_lo = mid, f_mid
                if moved == -1:
                    f_hi *= 0.5
                moved = -1
            else:
                hi, f_hi = mid, f_mid
                if moved == 1:
                    f_lo *= 0.5
                moved = 1
                if f_mid == 0.0:
                    break
        return hi

    def _drift_and_power(self, power: float | None) -> tuple[float, float]:
        """The drift for `power`, or the design's own drift and power when it was built for one."""
        if power is None:
            if self.drift is None or self.beta_spent is None:
                raise ValueError("give the power: this design was not built for one")
            return self.drift, 1.0 - self.beta_spent[-1]
        return self.drift_for_power(power), power

    def inflation_factor(self, power: float | None = None) -> float:
        """Maximum information over the information of a single-analysis test with the same alpha and power.

        For a fixed-sample design the factor is 1; the sample sizes below
        are the fixed-sample ones multiplied by it (as rpact and gsDesign do).
        """
        drift, power = self._drift_and_power(power)
        side = self.alpha / 2.0 if self.two_sided else self.alpha
        fixed = -_N.inv_cdf(side) + _N.inv_cdf(power)
        return (drift / fixed) ** 2

    def max_sample_size(self, effect: float, sigma: float, power: float | None = None) -> int:
        """Smallest maximum sample size for a normal mean: I_max = n / sigma^2, drift = effect sqrt(I_max).

        A design with futility boundaries was built for one power; leave
        `power` out to use it.
        """
        drift, _ = self._drift_and_power(power)
        return math.ceil((drift * sigma / abs(effect)) ** 2)

    def max_sample_size_two_means(self, effect: float, sigma: float, allocation: float = 0.5,
                                  power: float | None = None) -> int:
        """Total maximum sample size to compare two normal means (known sigma, normal approximation).

        `allocation` is the share of the treatment arm.  The information on
        the difference is n a (1 - a) / sigma^2.
        """
        _check_allocation(allocation)
        drift, _ = self._drift_and_power(power)
        return math.ceil(_ceil_guard((drift * sigma / abs(effect)) ** 2 / (allocation * (1.0 - allocation))))

    def max_sample_size_two_proportions(self, p_control: float, p_treatment: float, allocation: float = 0.5,
                                        power: float | None = None) -> int:
        """Total maximum sample size to compare two proportions (normal approximation).

        The fixed-sample size uses the pooled variance under H0 and the
        unpooled one under the alternative,

            n = (z_a sqrt(p (1 - p) (1/a + 1/(1 - a))) + z_b sqrt(p_T q_T / a + p_C q_C / (1 - a)))^2 / (p_T - p_C)^2,

        p = a p_T + (1 - a) p_C, and is multiplied by the inflation factor:
        the method of rpact's getSampleSizeRates (checked against it in the
        tests).
        """
        _check_allocation(allocation)
        for p in (p_control, p_treatment):
            if not 0.0 < p < 1.0:
                raise ValueError("the proportions must lie strictly between 0 and 1")
        if p_control == p_treatment:
            raise ValueError("the proportions must differ")
        _, power = self._drift_and_power(power)
        a = allocation
        side = self.alpha / 2.0 if self.two_sided else self.alpha
        z_a, z_b = -_N.inv_cdf(side), _N.inv_cdf(power)
        pooled = a * p_treatment + (1.0 - a) * p_control
        h0 = math.sqrt(pooled * (1.0 - pooled) * (1.0 / a + 1.0 / (1.0 - a)))
        h1 = math.sqrt(p_treatment * (1.0 - p_treatment) / a + p_control * (1.0 - p_control) / (1.0 - a))
        fixed = ((z_a * h0 + z_b * h1) / (p_treatment - p_control)) ** 2
        return math.ceil(_ceil_guard(fixed * self.inflation_factor(power)))

    def max_events(self, hazard_ratio: float, allocation: float = 0.5, power: float | None = None) -> int:
        """Maximum number of events for a log-rank test (Schoenfeld 1981).

        The information of the log-rank statistic is about D a (1 - a)
        after D events, so D_max = drift^2 / (a (1 - a) log(HR)^2).  The
        number of patients and the follow-up needed to observe them depend
        on accrual and on the baseline hazard, which this does not model.
        """
        _check_allocation(allocation)
        if not (math.isfinite(hazard_ratio) and hazard_ratio > 0.0) or hazard_ratio == 1.0:
            raise ValueError("the hazard ratio must be positive, finite and different from 1")
        drift, _ = self._drift_and_power(power)
        return math.ceil(_ceil_guard(drift**2 / (allocation * (1.0 - allocation) * math.log(hazard_ratio) ** 2)))

    def at(self, fractions: Sequence[float]) -> GroupSequentialDesign:
        """The design recomputed for the analyses actually done, at these information fractions.

        This is what alpha spending is for (Lan and DeMets 1983): the
        analyses need not happen when planned.  The fractions are realised
        information over the planned maximum; the last one is the final
        analysis, which spends all the alpha left, also when it comes
        before (under-running) or after (over-running) the planned maximum.
        A boundary depends only on the analyses up to its own, so the
        boundaries of analyses already done do not change when later ones
        move.  Futility boundaries keep the drift the design was built for,
        and the last one equals the last efficacy boundary.
        """
        if self.spending is None:
            raise ValueError("this design does not carry its spending function; build it with group_sequential_design")
        ts = [float(t) for t in fractions]
        if not ts or not ts[0] > 0 or any(b <= a for a, b in pairwise(ts)) or not all(map(math.isfinite, ts)):
            raise ValueError("fractions must be positive, finite and increasing")
        if any(t >= 1.0 for t in ts[:-1]):
            raise ValueError("an analysis at or beyond the planned maximum information is the final one")
        side = self.alpha / 2.0 if self.two_sided else self.alpha
        cumulative = _cumulative(spending_function(self.spending, side), ts, side)
        bounds = _efficacy_bounds(ts, cumulative, self.two_sided, self.points_per_sd)
        if self.futility is None:
            return replace(self, fractions=tuple(ts), bounds=tuple(bounds), spent=tuple(cumulative))
        assert self.drift is not None and self.beta_spent is not None
        beta = self.beta_spent[-1]
        beta_cumulative = _cumulative(spending_function(self.futility_spending, beta), ts, beta)
        finite = [b if math.isfinite(b) else _NEVER for b in bounds]
        fs, _ = _futility_bounds(ts, finite, beta_cumulative, self.drift, self.points_per_sd)
        fs.append(finite[-1])
        return replace(self, fractions=tuple(ts), bounds=tuple(bounds), spent=tuple(cumulative),
                       futility=tuple(fs), beta_spent=tuple(beta_cumulative))


def _check_allocation(a: float) -> None:
    if not 0.0 < a < 1.0:
        raise ValueError("allocation (the share of the treatment arm) must lie strictly between 0 and 1")


def _ceil_guard(x: float) -> float:
    """x less a relative 1e-12, so that ceil does not add a unit for rounding error in an exact integer."""
    return x * (1.0 - 1e-12)


def group_sequential_design(
    alpha: float,
    fractions: Sequence[float],
    spending="obrien-fleming",
    *,
    two_sided: bool = False,
    futility=None,
    power: float | None = None,
    points_per_sd: int = 24,
) -> GroupSequentialDesign:
    """Efficacy boundaries, and optionally futility boundaries, for K analyses.

    `alpha` is the overall type I error: one-sided, or the total of both
    sides when `two_sided` (then each side spends alpha/2).

    `futility` is a spending function for beta = 1 - power (same forms as
    `spending`); it requires `power` and a one-sided design.  Futility
    boundaries are non-binding: the efficacy boundaries are computed as if
    they did not exist, so the type I error stays at most alpha whether or
    not a trial stops when it crosses one.  The design's drift is chosen
    so that, with futility obeyed, the power is exactly `power` and the
    two boundaries meet at the last analysis (Pampallona and Tsiatis 1994;
    the "non-binding" designs of gsDesign and rpact, against which the
    boundaries are checked in the tests).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    ts = [float(t) for t in fractions]
    if not ts or any(b <= a for a, b in zip([0.0, *ts], ts)) or abs(ts[-1] - 1.0) > 1e-12:  # novermin (not PEP 798)
        raise ValueError("fractions must increase strictly and end at 1")
    side = alpha / 2.0 if two_sided else alpha
    if side >= 0.5:
        raise ValueError("a one-sided alpha must be below 0.5: the efficacy boundary has to be positive")
    spend = spending_function(spending, side)
    cumulative = [spend(t) for t in ts]
    increasing = all(b >= a for a, b in zip([0.0, *cumulative], cumulative))  # novermin (not PEP 798)
    if abs(cumulative[-1] - side) > 1e-9 * side or not increasing:
        raise ValueError("the spending function must increase and spend exactly alpha by t = 1")
    bounds = _efficacy_bounds(ts, cumulative, two_sided, points_per_sd)
    if futility is None:
        return GroupSequentialDesign(tuple(ts), tuple(bounds), tuple(cumulative), alpha, two_sided,
                                     spending=spending, points_per_sd=points_per_sd)
    if two_sided:
        raise ValueError("futility boundaries are implemented for one-sided designs")
    if power is None or not alpha < power < 1:
        raise ValueError("futility boundaries need the power they are designed for, between alpha and 1")
    beta = 1.0 - power
    beta_spend = spending_function(futility, beta)
    beta_cumulative = [beta_spend(t) for t in ts]
    if abs(beta_cumulative[-1] - beta) > 1e-9:
        raise ValueError("the futility spending function must spend exactly beta by t = 1")
    finite = [b if math.isfinite(b) else _NEVER for b in bounds]

    # Type II error (futility obeyed) falls as the drift grows; find the drift giving exactly beta.
    lo, hi = 0.0, 1.0
    while _futility_bounds(ts, finite, beta_cumulative, hi, points_per_sd)[1] > beta:
        lo, hi = hi, 2.0 * hi
    while hi - lo > 1e-7:
        mid = 0.5 * (lo + hi)
        if _futility_bounds(ts, finite, beta_cumulative, mid, points_per_sd)[1] > beta:
            lo = mid
        else:
            hi = mid
    fs, _ = _futility_bounds(ts, finite, beta_cumulative, hi, points_per_sd)
    fs.append(finite[-1])
    return GroupSequentialDesign(
        tuple(ts), tuple(bounds), tuple(cumulative), alpha, two_sided, tuple(fs), tuple(beta_cumulative), hi,
        spending=spending, futility_spending=futility, points_per_sd=points_per_sd,
    )
