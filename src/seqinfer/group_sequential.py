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
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()
_SQRT2 = math.sqrt(2.0)


def _upper(x: float) -> float:
    """1 - Phi(x), through erfc so that small tails keep their relative accuracy."""
    return 0.5 * math.erfc(x / _SQRT2)
_LOWER = -10.0  # B(t) / sqrt(t) below this is never reached in practice


def spending_function(kind, alpha: float) -> Callable[[float], float]:
    """Cumulative alpha spent by information fraction t, for one side."""
    if callable(kind):
        custom: Callable[[float], float] = kind
        return custom
    if kind == "obrien-fleming":
        z = _N.inv_cdf(1.0 - alpha / 2.0)
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

    def cross(self, t: float, b: float) -> float:
        """P(continue to t_{k-1} and cross the boundary b sqrt(t) at t)."""
        sd = math.sqrt(t - self.t)
        mu = self.drift * (t - self.t)
        upper = b * math.sqrt(t)
        p = 0.0
        for x, m in zip(self.xs, self.mass, strict=True):
            p += m * _upper((upper - x - mu) / sd)
            if self.two_sided:
                p += m * _upper((upper + x + mu) / sd)
        return p

    def cross_below(self, t: float, f: float) -> float:
        """P(continue to t_{k-1} and end at or below the futility boundary f sqrt(t) at t)."""
        sd = math.sqrt(t - self.t)
        mu = self.drift * (t - self.t)
        lower = f * math.sqrt(t)
        return sum(m * _upper((x + mu - lower) / sd) for x, m in zip(self.xs, self.mass, strict=True))

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
            lower = min(_LOWER * math.sqrt(t), upper - 1.0) + self.drift * t
        ys = self._grid(lower, upper, sd)
        h = (ys[1] - ys[0]) / 3.0
        weights = _simpson(len(ys))
        density = []
        for y in ys:
            f = 0.0
            for x, m in zip(self.xs, self.mass, strict=True):
                z = (y - x - mu) / sd
                f += m * math.exp(-0.5 * z * z)
            density.append(f / (sd * math.sqrt(2.0 * math.pi)))
        self.t, self.xs = t, ys
        self.mass = [h * w * f for w, f in zip(weights, density, strict=True)]


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

    def _walk(self, drift: float, points_per_sd: int, binding: bool) -> tuple[list[float], list[float]]:
        """First-crossing probabilities of the efficacy and of the futility boundary at each analysis."""
        rec = _Recursion(drift, self.two_sided, points_per_sd)
        futility = self.futility if binding else None
        up, down = [], []
        last = len(self.fractions) - 1
        for k, (t, b) in enumerate(zip(self.fractions, self.bounds, strict=True)):
            up.append(rec.cross(t, b))
            f = futility[k] if futility is not None else None
            down.append(rec.cross_below(t, f) if f is not None else 0.0)
            if k < last:
                rec.advance(t, b, f)
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
        return sum(self.crossing(drift))

    def expected_fraction(self, drift: float) -> float:
        """Expected information at stopping, as a fraction of the maximum (futility obeyed)."""
        up, down = self._walk(drift, 24, True)
        stopped_early = 0.0
        expected = 0.0
        for t, p, q in zip(self.fractions[:-1], up[:-1], down[:-1], strict=True):
            expected += t * (p + q)
            stopped_early += p + q
        return expected + 1.0 * (1.0 - stopped_early)

    def drift_for_power(self, power: float) -> float:
        """Drift theta * sqrt(I_max) at which the design has the given power (one-sided)."""
        lo, hi = 0.0, 1.0
        while self.power(hi) < power:
            lo, hi = hi, 2 * hi
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self.power(mid) < power:
                lo = mid
            else:
                hi = mid
        return hi

    def max_sample_size(self, effect: float, sigma: float, power: float | None = None) -> int:
        """Smallest maximum sample size for a normal mean: I_max = n / sigma^2, drift = effect sqrt(I_max).

        A design with futility boundaries was built for one power; leave
        `power` out to use it.
        """
        if power is None:
            if self.drift is None:
                raise ValueError("give the power: this design was not built for one")
            drift = self.drift
        else:
            drift = self.drift_for_power(power)
        return math.ceil((drift * sigma / abs(effect)) ** 2)


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
    the "non-binding" designs of gsDesign).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    ts = [float(t) for t in fractions]
    if not ts or any(b <= a for a, b in zip([0.0, *ts], ts, strict=False)) or abs(ts[-1] - 1.0) > 1e-12:
        raise ValueError("fractions must increase strictly and end at 1")
    side = alpha / 2.0 if two_sided else alpha
    spend = spending_function(spending, side)
    cumulative = [spend(t) for t in ts]
    if abs(cumulative[-1] - side) > 1e-9 or any(b < a for a, b in zip([0.0, *cumulative], cumulative, strict=False)):
        raise ValueError("the spending function must increase and spend exactly alpha by t = 1")
    rec = _Recursion(0.0, two_sided, points_per_sd)
    bounds, previous = [], 0.0
    for k, (t, c) in enumerate(zip(ts, cumulative, strict=True)):
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
            rec.advance(t, b if math.isfinite(b) else 40.0)
    if futility is None:
        return GroupSequentialDesign(tuple(ts), tuple(bounds), tuple(cumulative), alpha, two_sided)
    if two_sided:
        raise ValueError("futility boundaries are implemented for one-sided designs")
    if power is None or not 0 < power < 1:
        raise ValueError("futility boundaries need the power they are designed for")
    beta = 1.0 - power
    beta_spend = spending_function(futility, beta)
    beta_cumulative = [beta_spend(t) for t in ts]
    if abs(beta_cumulative[-1] - beta) > 1e-9:
        raise ValueError("the futility spending function must spend exactly beta by t = 1")
    finite = [b if math.isfinite(b) else 40.0 for b in bounds]

    def futility_for(drift: float) -> tuple[list[float], float]:
        """Futility boundaries spending beta at this drift, and the beta actually left at the end."""
        rec = _Recursion(drift, False, points_per_sd)
        fs, previous = [], 0.0
        for t, b, c in zip(ts[:-1], finite[:-1], beta_cumulative[:-1], strict=True):
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

    # Type II error (futility obeyed) falls as the drift grows; find the drift giving exactly beta.
    lo, hi = 0.0, 1.0
    while futility_for(hi)[1] > beta:
        lo, hi = hi, 2.0 * hi
    while hi - lo > 1e-7:
        mid = 0.5 * (lo + hi)
        if futility_for(mid)[1] > beta:
            lo = mid
        else:
            hi = mid
    fs, _ = futility_for(hi)
    fs.append(finite[-1])
    return GroupSequentialDesign(
        tuple(ts), tuple(bounds), tuple(cumulative), alpha, two_sided, tuple(fs), tuple(beta_cumulative), hi
    )
