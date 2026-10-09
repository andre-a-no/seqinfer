# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Student's t distribution, from the standard library only.

The normal distribution comes from `statistics.NormalDist`.  The t
distribution function is computed through the regularized incomplete beta
function, evaluated by its continued fraction (Lentz's method), which is
accurate to about 1e-14; the quantile function inverts it by bisection.
"""
from __future__ import annotations

import math


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-300, 1e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _lgamma_ratio(z: float, a: float) -> float:
    """log Gamma(z + a) - log Gamma(z), accurate when z is large (no cancellation of two huge lgammas)."""
    if z < 1e4:
        return math.lgamma(z + a) - math.lgamma(z)
    w = z + a
    # Stirling: lgamma(x) = (x - 1/2) log x - x + log(2 pi)/2 + 1/(12 x) - 1/(360 x^3) + ...
    main = (z - 0.5) * math.log1p(a / z) + a * math.log(w) - a
    return main + (1.0 / w - 1.0 / z) / 12.0 - (1.0 / w**3 - 1.0 / z**3) / 360.0


def _log_inverse_beta(a: float, b: float) -> float:
    """log Gamma(a + b) - log Gamma(a) - log Gamma(b), i.e. -log B(a, b)."""
    big, small = (a, b) if a >= b else (b, a)
    return _lgamma_ratio(big, small) - math.lgamma(small)


def regularized_beta(
    x: float, a: float, b: float, y: float | None = None, logs: tuple[float, float] | None = None
) -> float:
    """I_x(a, b).

    Pass y = 1 - x, and the pair (log x, log y), when they are known more
    accurately than they can be computed from x: with a or b large, the
    rounding of x is multiplied by them in the prefactor.
    """
    if y is None:
        y = 1.0 - x
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    log_x, log_y = logs if logs is not None else (math.log(x), math.log(y))
    log_front = _log_inverse_beta(a, b) + a * log_x + b * log_y
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_continued_fraction(b, a, y) / b


def _upper_tail(t: float, df: float) -> float:
    """P(T > t) for t >= 0, without forming t*t (which overflows) or 1 - (something near 1).

    P(T > t) = I_x(df/2, 1/2) / 2 with x = df / (df + t^2); x and 1 - x are
    both formed from s = t / sqrt(df) so that each keeps its precision.
    """
    s = t / math.sqrt(df)
    a = 0.5 * df
    if s > 1e100:
        # x underflows; use the leading term of I_x(a, 1/2) as x -> 0, relative error of order x.
        log_x = -2.0 * math.log(s) - math.log1p(1.0 / (s * s))
        return 0.5 * math.exp(_log_inverse_beta(a, 0.5) + a * log_x - math.log(a))
    if s < 1.0:
        s2 = s * s
        x, y = 1.0 / (1.0 + s2), s2 / (1.0 + s2)
        logs = (-math.log1p(s2), 2.0 * math.log(s) - math.log1p(s2)) if s > 0.0 else (0.0, -math.inf)
    else:
        r2 = 1.0 / (s * s)
        x, y = r2 / (1.0 + r2), 1.0 / (1.0 + r2)
        logs = (-2.0 * math.log(s) - math.log1p(r2), -math.log1p(r2))
    return 0.5 * regularized_beta(x, a, 0.5, y, logs)


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with `df` degrees of freedom."""
    if not df > 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isnan(t):
        return math.nan
    if t == 0.0:
        return 0.5
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    tail = _upper_tail(abs(t), df)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p: float, df: float) -> float:
    """The p-quantile of Student's t.

    Solved on the smaller tail, so quantiles far out (p = 1e-300) keep
    their relative accuracy; p close to 1 is limited by the precision of
    1 - p itself.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    if not df > 0:
        raise ValueError("degrees of freedom must be positive")
    if p == 0.5:
        return 0.0
    tail, sign = (p, -1.0) if p < 0.5 else (1.0 - p, 1.0)
    lo, hi = 0.0, 1.0
    while _upper_tail(hi, df) > tail:
        lo, hi = hi, 2.0 * hi
        if math.isinf(hi):
            return sign * math.inf
    for _ in range(400):
        mid = math.sqrt(lo * hi) if lo > 0.0 and hi > 4.0 * lo else 0.5 * (lo + hi)
        if _upper_tail(mid, df) > tail:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 2e-16 * hi:
            break
    return sign * 0.5 * (lo + hi)
