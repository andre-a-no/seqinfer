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


_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_NORMAL_DF = 1e15  # beyond this, t and normal tails agree to machine precision for every |t| that matters
_TINY = 1e-280  # tails below this are recomputed in logarithms


def _lgamma_ratio(z: float, a: float) -> float:
    """log Gamma(z + a) - log Gamma(z), without the cancellation of two large lgammas."""
    if z < 100.0:
        return math.lgamma(z + a) - math.lgamma(z)
    w = z + a
    # Stirling: lgamma(x) = (x - 1/2) log x - x + log(2 pi)/2 + 1/(12x) - 1/(360x^3) + 1/(1260x^5) - ...
    iw, iz = 1.0 / w, 1.0 / z
    main = (z - 0.5) * math.log1p(a / z) + a * math.log(w) - a
    return main + (iw - iz) / 12.0 - (iw**3 - iz**3) / 360.0 + (iw**5 - iz**5) / 1260.0


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


def _log_normal_upper(z: float) -> float:
    """log(1 - Phi(z)) for z >= 0, also where 1 - Phi(z) underflows."""
    q = 0.5 * math.erfc(z / _SQRT2)
    if q > _TINY:
        return math.log(q)
    z2 = z * z
    series = 1.0 - 1.0 / z2 + 3.0 / z2**2 - 15.0 / z2**3 + 105.0 / z2**4  # Mills ratio, z > 37
    return -0.5 * z2 - math.log(z) - _LOG_SQRT_2PI + math.log(series)


def _fisher_applies(t: float, df: float) -> bool:
    """Fisher's two-term expansion is the more accurate method here (error about t^12 / df^3)."""
    return df >= 1e6 and (t == 0.0 or 12.0 * math.log(t) <= math.log(1e-12) + 3.0 * math.log(df))


def _fisher_correction(t: float, df: float) -> float:
    """(P(T > t) - (1 - Phi(t))) / phi(t), to second order in 1/df (Fisher 1925)."""
    t3 = t**3
    return (t3 + t) / (4.0 * df) + (3.0 * t3 * t**4 + 19.0 * t3 * t * t + 17.0 * t3 - 15.0 * t) / (96.0 * df * df)


def _upper_tail(t: float, df: float) -> float:
    """P(T > t) for t >= 0."""
    if df > _NORMAL_DF:
        return 0.5 * math.erfc(t / _SQRT2)
    if _fisher_applies(t, df):
        phi = math.exp(-0.5 * t * t - _LOG_SQRT_2PI)
        return 0.5 * math.erfc(t / _SQRT2) + phi * _fisher_correction(t, df)
    return math.exp(_log_upper_tail(t, df)) if _beta_tail_is_tiny(t, df) else _beta_upper_tail(t, df)


def _log_s(t: float, df: float) -> float:
    return math.log(t) - 0.5 * math.log(df)


def _beta_tail_is_tiny(t: float, df: float) -> bool:
    return t > 0.0 and _log_s(t, df) > math.log(1e100)


def _beta_upper_tail(t: float, df: float) -> float:
    """P(T > t) = I_x(df/2, 1/2) / 2 with x = df / (df + t^2), x and 1 - x formed from s = t / sqrt(df)."""
    s = t / math.sqrt(df)
    a = 0.5 * df
    if s < 1.0:
        s2 = s * s
        x, y = 1.0 / (1.0 + s2), s2 / (1.0 + s2)
        logs = (-math.log1p(s2), 2.0 * math.log(s) - math.log1p(s2)) if s > 0.0 else (0.0, -math.inf)
    else:
        r2 = 1.0 / (s * s)
        x, y = r2 / (1.0 + r2), 1.0 / (1.0 + r2)
        logs = (-2.0 * math.log(s) - math.log1p(r2), -math.log1p(r2))
    return 0.5 * regularized_beta(x, a, 0.5, y, logs)


def _log_upper_tail(t: float, df: float) -> float:
    """log P(T > t) for t >= 0, also where the tail underflows."""
    if df > _NORMAL_DF:
        return _log_normal_upper(t)
    if _fisher_applies(t, df):
        log_phi = -0.5 * t * t - _LOG_SQRT_2PI
        log_q = _log_normal_upper(t)
        return log_q + math.log1p(math.exp(log_phi - log_q) * _fisher_correction(t, df))
    a = 0.5 * df
    log_s = _log_s(t, df)
    if log_s > math.log(1e100):
        # x = df / (df + t^2) underflows: leading term of I_x(a, 1/2) as x -> 0, relative error of order x
        log_x = -2.0 * log_s
        return math.log(0.5) + _log_inverse_beta(a, 0.5) + a * log_x - math.log(a)
    tail = _beta_upper_tail(t, df)
    if tail > _TINY:
        return math.log(tail)
    # deep in the tail x is small, so I_x comes from the direct continued fraction: take its logarithm
    s = math.exp(log_s)
    r2 = 1.0 / (s * s)
    x = r2 / (1.0 + r2)
    log_x, log_y = -2.0 * log_s - math.log1p(r2), -math.log1p(r2)
    log_front = _log_inverse_beta(a, 0.5) + a * log_x + 0.5 * log_y
    return math.log(0.5) + log_front + math.log(_beta_continued_fraction(a, 0.5, x)) - math.log(a)


def _check_df(df: float) -> None:
    if not df > 0:  # also NaN
        raise ValueError("degrees of freedom must be positive")


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with `df` degrees of freedom (df = inf gives the normal distribution)."""
    _check_df(df)
    if math.isnan(t):
        return math.nan
    if t == 0.0:
        return 0.5
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    tail = _upper_tail(abs(t), df)
    return 1.0 - tail if t > 0 else tail


def t_isf_log(log_tail: float, df: float) -> float:
    """The t >= 0 with log P(T > t) = log_tail (log_tail <= log 1/2); inf if beyond every float."""
    _check_df(df)
    if log_tail >= math.log(0.5):
        return 0.0
    lo, hi = 0.0, 1.0
    while _log_upper_tail(hi, df) > log_tail:
        lo, hi = hi, 2.0 * hi
        if math.isinf(hi):
            return math.inf
    for _ in range(400):
        mid = math.sqrt(lo * hi) if lo > 0.0 and hi > 4.0 * lo else 0.5 * (lo + hi)
        if _log_upper_tail(mid, df) > log_tail:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 2e-16 * hi:
            break
    return 0.5 * (lo + hi)


def t_ppf(p: float, df: float) -> float:
    """The p-quantile of Student's t.

    Solved on the smaller tail, so quantiles far out (p = 1e-300) keep
    their relative accuracy; p close to 1 is limited by the precision of
    1 - p itself.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    _check_df(df)
    if p == 0.5:
        return 0.0
    tail, sign = (p, -1.0) if p < 0.5 else (1.0 - p, 1.0)
    return sign * t_isf_log(math.log(tail), df)
