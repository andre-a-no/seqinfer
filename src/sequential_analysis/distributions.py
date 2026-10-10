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
    tiny, eps = 1e-300, 2.3e-16  # two units in the last place: |delta - 1| can stick at 2**-53
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 20000):
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


def _beta_factor(a: float, b: float, x: float) -> float:
    """F with I_x(a, b) = x^a (1 - x)^b F / (a B(a, b)); F = 2F1(a + b, 1; a + 1; x).

    The continued fraction converges slowly, or not at all, when a is huge and
    x small (a t tail far out with df beyond 1e13).  For b <= 1 the terms of the
    hypergeometric series shrink at least by the factor x, so for x <= 1/2 the
    series converges in about 55 terms whatever a is.
    """
    if b <= 1.0 and x <= 0.5:
        total, term, n = 1.0, 1.0, 0.0
        while True:
            term *= (a + b + n) / (a + 1.0 + n) * x
            total += term
            n += 1.0
            if term <= 1e-17 * total:
                return total
    return _beta_continued_fraction(a, b, x)


_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_HUGE_DF = 1e15  # beyond this, the incomplete beta function is not used
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
        return math.exp(log_front) * _beta_factor(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_factor(b, a, y) / b


def _log_normal_upper(z: float) -> float:
    """log(1 - Phi(z)) for z >= 0, also where 1 - Phi(z) underflows."""
    q = 0.5 * math.erfc(z / _SQRT2)
    if q > _TINY:
        return math.log(q)
    w = 1.0 / (z * z)  # may underflow to 0; z * z itself may overflow, and the log tail is then -inf
    series = 1.0 - w * (1.0 - w * (3.0 - w * (15.0 - 105.0 * w)))  # Mills ratio, z > 37
    return -0.5 * z * z - math.log(z) - _LOG_SQRT_2PI + math.log(series)


def _fisher_applies(t: float, df: float) -> bool:
    """Fisher's expansion to 1/df^3 is the more accurate method here (relative error about t^16 / (6144 df^4))."""
    return df >= 1e5 and (t == 0.0 or 16.0 * math.log(t) <= math.log(6e-11) + 4.0 * math.log(df))


def _fisher_correction(t: float, df: float) -> float:
    """(P(T > t) - (1 - Phi(t))) / phi(t), to third order in 1/df (Fisher 1925).

    The coefficients were checked against 120-digit evaluations of the incomplete beta function.
    """
    # g1/df + g2/df^2 + g3/df^3 with g1 = t(t^2 + 1)/4, g2 = t(3t^6 - 7t^4 - 5t^2 - 3)/96,
    # g3 = t(t^10 - 11t^8 + 14t^6 + 6t^4 - 3t^2 - 15)/384, in u = t^2/df, w = t^4/df and
    # v = 1/df: where the expansion applies w is below 1, so nothing overflows or meets 0 * inf
    root = math.sqrt(df)
    r2 = t * t / root if t < 1e150 else (t / math.sqrt(root)) ** 2  # t^2 / sqrt(df)
    u, w, v = r2 / root, r2 * r2, 1.0 / df
    first = (u + v) / 4.0
    second = (3.0 * w * u - 7.0 * u * u - 5.0 * u * v - 3.0 * v * v) / 96.0
    third = (w * w * u - 11.0 * w * u * u + 14.0 * u**3 + 6.0 * u * u * v - 3.0 * u * v * v - 15.0 * v**3) / 384.0
    return t * (first + second + third)


def _normal_applies(t: float, df: float) -> bool:
    """Fisher's first correction is below 1e-17 of the tail and of its distance from 1/2."""
    return math.isinf(df) or (t * t + 1.0) * max(1.0, t * t) < 4e-17 * df


def _log_huge_df_tail(t: float, df: float) -> float:
    """log P(T > t) for df > 1e15 beyond the reach of Fisher's expansion (t > 2000).

    P(T > t) = f(t) (1 + t^2/df) / t * df/(df + 1) * (1 - (1/t^2 - 1/df)/(1 + 1/df) + O(1/t^4)),
    f the density: the error of the logarithm is below 1e-12, against a logarithm below -2e6.
    """
    s = t / math.sqrt(df)
    log1p_s2 = math.log1p(s * s) if s < 1e150 else 2.0 * math.log(s)
    log_density = _lgamma_ratio(0.5 * df, 0.5) - 0.5 * (math.log(df) + math.log(math.pi)) - 0.5 * (df + 1.0) * log1p_s2
    # Laplace: integral of e^-g from t is e^-g(t) / g'(t) (1 - g''(t) / g'(t)^2 + O(t^-4)),
    # g' = (df + 1) t / (df + t^2), g'' / g'^2 = (df - t^2) / ((df + 1) t^2)
    correction = math.log1p(-(1.0 / (t * t) - 1.0 / df) / (1.0 + 1.0 / df))
    return log_density - math.log(t) + log1p_s2 - math.log1p(1.0 / df) + correction


def _upper_tail(t: float, df: float) -> float:
    """P(T > t) for t >= 0."""
    if t == 0.0:
        return 0.5
    if _normal_applies(t, df):
        return 0.5 * math.erfc(t / _SQRT2)
    if df > _HUGE_DF and not _fisher_applies(t, df):
        return math.exp(_log_huge_df_tail(t, df))
    if _fisher_applies(t, df):
        phi = math.exp(-0.5 * t * t - _LOG_SQRT_2PI)
        return 0.5 * math.erfc(t / _SQRT2) + phi * _fisher_correction(t, df)
    return math.exp(_log_upper_tail(t, df)) if _beta_tail_is_tiny(t, df) else _beta_upper_tail(t, df)


def _log_s(t: float, df: float) -> float:
    return math.log(t) - 0.5 * math.log(df)


def _beta_tail_is_tiny(t: float, df: float) -> bool:
    return t > 0.0 and _log_s(t, df) > math.log(1e100)


def _beta_arguments(t: float, df: float) -> tuple[float, float, tuple[float, float]]:
    """x = df / (df + t^2), y = 1 - x and their logarithms, formed from s = t / sqrt(df) without cancellation.

    With df large, log x is multiplied by df / 2, so it must be accurate relative to itself.
    """
    s = t / math.sqrt(df)
    if s < 1.0:
        s2 = s * s
        logs = (-math.log1p(s2), 2.0 * math.log(s) - math.log1p(s2)) if s > 0.0 else (0.0, -math.inf)
        return 1.0 / (1.0 + s2), s2 / (1.0 + s2), logs
    r2 = 1.0 / (s * s)
    return r2 / (1.0 + r2), 1.0 / (1.0 + r2), (-2.0 * math.log(s) - math.log1p(r2), -math.log1p(r2))


def _beta_upper_tail(t: float, df: float) -> float:
    """P(T > t) = I_x(df/2, 1/2) / 2."""
    x, y, logs = _beta_arguments(t, df)
    return 0.5 * regularized_beta(x, 0.5 * df, 0.5, y, logs)


def _log_upper_tail(t: float, df: float) -> float:
    """log P(T > t) for t >= 0, also where the tail underflows."""
    if t == 0.0:
        return math.log(0.5)
    if _normal_applies(t, df):
        return _log_normal_upper(t)
    if df > _HUGE_DF and not _fisher_applies(t, df):
        return _log_huge_df_tail(t, df)
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
    x, _, (log_x, log_y) = _beta_arguments(t, df)
    log_front = _log_inverse_beta(a, 0.5) + a * log_x + 0.5 * log_y
    return math.log(0.5) + log_front + math.log(_beta_factor(a, 0.5, x)) - math.log(a)


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
