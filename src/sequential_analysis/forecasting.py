# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Fitting forecasters to a history: the "fixed training" half of forecasting.

`fit_ets` estimates an exponential smoothing model by maximum likelihood.
The initial states (level, trend, seasonal effects) enter the one-step
errors linearly, e_t = e_t(0) - w' D^(t-1) x0 with D = F - g w' (Hyndman et
al. 2008), so for given smoothing parameters the best initial
states are a least-squares regression and the likelihood is profiled over
them exactly.  The smoothing parameters are then found by Nelder-Mead over
the usual admissible region, as statsmodels bounds it: alpha in [1e-4, 0.9999],
beta / alpha in [1e-4, 0.9999], gamma in [0, 0.9999 (1 - alpha)] (statsmodels
also allows a slightly negative gamma) and phi in [0.8, 0.98].
The seasonal effects are normalized to sum to zero (level and season are
otherwise confounded).  sigma^2 is the mean squared one-step error.

`fit_ar` estimates an AR(p) with intercept by conditional least squares
(the first p values are conditioned on), as statsmodels' AutoReg; sigma^2 is
the residual sum of squares over n - p.

Both return the fitted forecaster, whose forecasts follow the model with
the estimated parameters held fixed, and the state after the history, from
which a run can continue (`Run(fit.model, initial_state=fit.state)`).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from . import linalg as la
from .procedures.forecast import TRENDS, AutoRegressive, ExponentialSmoothing

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class ForecastFit:
    model: Any  # the fitted forecaster, parameters fixed
    state: Any  # its state after the history
    log_likelihood: float  # Gaussian log-likelihood of the history at the estimates
    aic: float
    parameters: dict  # the estimates, by name
    evaluations: int  # likelihood evaluations used


def _values(inputs: Iterable[Any], key: str) -> list[float]:
    out = []
    for x in inputs:
        v = x[key] if isinstance(x, dict) else x
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("the history must be finite numbers")
        out.append(float(v))
    return out


def nelder_mead(f: Callable[[Sequence[float]], float], start: Sequence[float], *, step: float = 0.5,
                tolerance: float = 1e-10, max_evaluations: int = 4000) -> tuple[list[float], float, int]:
    """Minimize f from `start` by the Nelder-Mead simplex (standard coefficients); (x, f(x), evaluations)."""
    n = len(start)
    simplex = [list(start)]
    for i in range(n):
        point = list(start)
        point[i] += step
        simplex.append(point)
    values = [f(p) for p in simplex]
    evaluations = n + 1
    while evaluations < max_evaluations:
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) <= tolerance * (abs(values[0]) + tolerance):
            break
        centroid = [0.0] * n
        for p in simplex[:-1]:
            for j in range(n):
                centroid[j] += p[j] / n
        worst = simplex[-1]
        reflected = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = f(reflected)
        evaluations += 1
        if fr < values[0]:
            expanded = [c + 2.0 * (c - w) for c, w in zip(centroid, worst)]
            fe = f(expanded)
            evaluations += 1
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            if fr < values[-1]:
                contracted = [c + 0.5 * (r - c) for c, r in zip(centroid, reflected)]
            else:
                contracted = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
            fc = f(contracted)
            evaluations += 1
            if fc < min(fr, values[-1]):
                simplex[-1], values[-1] = contracted, fc
            else:
                best = simplex[0]
                for i in range(1, n + 1):
                    simplex[i] = [b + 0.5 * (p - b) for b, p in zip(best, simplex[i])]
                    values[i] = f(simplex[i])
                evaluations += n
    i = min(range(n + 1), key=lambda k: values[k])
    return simplex[i], values[i], evaluations


def _sigmoid(u: float) -> float:
    if u >= 0:
        return 1.0 / (1.0 + math.exp(-u))
    e = math.exp(u)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class _ETSProblem:
    """Profile likelihood of an ETS model over its initial states, as a function of the smoothing parameters."""

    def __init__(self, ys: list[float], trend: str | None, period: int | None):
        self.ys, self.trend, self.period = ys, trend, period
        self.m = period or 0
        # x0 = (level, [trend], s_1 .. s_{m-1}); s_m = -(s_1 + ... + s_{m-1})
        self.k = 1 + (trend is not None) + (self.m - 1 if period else 0)

    def unpack(self, u: Sequence[float]) -> tuple[float, float, float, float]:
        lo, hi = 1e-4, 0.9999
        alpha = lo + (hi - lo) * _sigmoid(u[0])
        i = 1
        beta, gamma, phi = 0.0, 0.0, 1.0
        if self.trend is not None:
            beta = alpha * (lo + (hi - lo) * _sigmoid(u[i]))
            i += 1
        if self.period:
            gamma = (1.0 - alpha) * hi * _sigmoid(u[i])
            i += 1
        if self.trend == "damped":
            phi = 0.8 + 0.18 * _sigmoid(u[i])
        return alpha, beta, gamma, phi

    def pack(self, alpha: float, beta: float, gamma: float, phi: float) -> list[float]:
        lo, hi = 1e-4, 0.9999
        u = [_logit((alpha - lo) / (hi - lo))]
        if self.trend is not None:
            u.append(_logit((beta / alpha - lo) / (hi - lo)))
        if self.period:
            u.append(_logit(gamma / ((1.0 - alpha) * hi)))
        if self.trend == "damped":
            u.append(_logit((phi - 0.8) / 0.18))
        return u

    def _run(self, params, x0_level, x0_trend, seasonal):
        """One pass of the recursions from the given initial states; the one-step errors."""
        alpha, beta, gamma, phi = params
        level, trend, season = x0_level, x0_trend, list(seasonal)
        errors = []
        for y in self.ys:
            s = season[0] if season else 0.0
            e = y - (level + phi * trend + s)
            errors.append(e)
            level = level + phi * trend + alpha * e
            trend = phi * trend + beta * e
            if season:
                season = [*season[1:], s + gamma * e]
        return errors

    def initial_states(self, params) -> tuple[list[float], list[float]]:
        """(best x0, its errors): least squares of e(0) on the columns of each unit initial state."""
        m = self.m
        base = self._run(params, 0.0, 0.0, [0.0] * m)
        columns = []
        # each initial state's effect on the errors is linear: run with the data zeroed, unit state
        zero = _ETSProblem([0.0] * len(self.ys), self.trend, self.period)
        columns.append(zero._run(params, 1.0, 0.0, [0.0] * m))
        if self.trend is not None:
            columns.append(zero._run(params, 0.0, 1.0, [0.0] * m))
        for j in range(m - 1):
            unit = [0.0] * m
            unit[j], unit[m - 1] = 1.0, -1.0
            columns.append(zero._run(params, 0.0, 0.0, unit))
        k = len(columns)
        # base = errors at x0 = 0; errors(x0) = base + Z x0 with Z the columns; minimize |base + Z x0|^2
        gram = [[0.0] * k for _ in range(k)]
        rhs = [0.0] * k
        for i in range(k):
            for j in range(i, k):
                total = 0.0
                for a, b in zip(columns[i], columns[j]):
                    total += a * b
                gram[i][j] = gram[j][i] = total
            total = 0.0
            for a, b in zip(columns[i], base):
                total += a * b
            rhs[i] = -total
        ridge = 1e-12 * max(1.0, max(gram[i][i] for i in range(k)))
        for i in range(k):
            gram[i][i] += ridge
        x0 = la.solve_cholesky(la.cholesky(tuple(tuple(r) for r in gram)), rhs)
        errors = []
        for t in range(len(base)):
            v = base[t]
            for i in range(k):
                v += columns[i][t] * x0[i]
            errors.append(v)
        return list(x0), errors

    def objective(self, u: Sequence[float]) -> float:
        _, errors = self.initial_states(self.unpack(u))
        sse = 0.0
        for e in errors:
            sse += e * e
        return sse


def fit_ets(inputs: Iterable[Any], *, trend: str | None = None, period: int | None = None, input: str = "y",
            horizon: int = 1, coverage: float = 0.9, max_evaluations: int = 4000) -> ForecastFit:
    """Maximum likelihood exponential smoothing (additive errors); see the module docstring.

    `inputs` are records ({input: value}) or plain numbers.  `trend` is
    None, "additive" or "damped"; `period` the length of the season.
    """
    if trend not in TRENDS:
        raise ValueError(f"trend must be one of {TRENDS}")
    ys = _values(inputs, input)
    problem = _ETSProblem(ys, trend, period)
    n_params = len(problem.pack(0.5, 0.1, 0.1, 0.9))
    if len(ys) < problem.k + n_params + 2:
        raise ValueError("the history is too short for this model")
    # start from the best point of a small grid, then Nelder-Mead
    best_u, best_f = None, math.inf
    for a in (0.1, 0.3, 0.6):
        for fb in ((0.05, 0.3) if trend is not None else (None,)):
            for fg in ((0.05, 0.3) if period else (None,)):
                b = a * fb if fb is not None else 0.0
                g = (1.0 - a) * fg if fg is not None else 0.0
                u = problem.pack(a, b, g, 0.95)
                value = problem.objective(u)
                if value < best_f:
                    best_u, best_f = u, value
    assert best_u is not None
    u, sse, evaluations = nelder_mead(problem.objective, best_u, max_evaluations=max_evaluations)
    u, sse, more = nelder_mead(problem.objective, u, step=0.1, max_evaluations=max_evaluations)  # restart
    evaluations += more
    alpha, beta, gamma, phi = problem.unpack(u)
    x0, _errors = problem.initial_states((alpha, beta, gamma, phi))
    n = len(ys)
    sigma2 = sse / n
    log_lik = -0.5 * n * (_LOG_2PI + math.log(sigma2) + 1.0)
    level0 = x0[0]
    trend0 = x0[1] if trend is not None else 0.0
    seasonal: tuple[float, ...] | None = None
    if period:
        free = x0[1 + (trend is not None):]
        last = 0.0
        for v in free:
            last -= v
        seasonal = (*tuple(free), last)
    model = ExponentialSmoothing(alpha, beta=beta, gamma=gamma, phi=phi, trend=trend, period=period,
                                 initial_level=level0, initial_trend=trend0, initial_seasonal=seasonal,
                                 sigma=math.sqrt(sigma2), horizon=horizon, coverage=coverage, input=input)
    state = model.initial_state()
    for y in ys:
        state, _, _ = model.update(state, y)
    parameters = {"alpha": alpha, "beta": beta, "gamma": gamma, "phi": phi, "initial_level": level0,
                  "initial_trend": trend0, "initial_seasonal": None if seasonal is None else list(seasonal),
                  "sigma": math.sqrt(sigma2)}
    k = n_params + problem.k + 1
    return ForecastFit(model, state, log_lik, 2.0 * k - 2.0 * log_lik, parameters, evaluations)


def fit_ar(inputs: Iterable[Any], order: int, *, input: str = "y", horizon: int = 1,
           coverage: float = 0.9) -> ForecastFit:
    """Conditional least squares AR(p) with intercept; see the module docstring."""
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive int")
    ys = _values(inputs, input)
    n = len(ys) - order
    k = order + 1
    if n < k + 1:
        raise ValueError("the history is too short for this order")
    gram = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for t in range(order, len(ys)):
        x = [1.0] + [ys[t - i] for i in range(1, order + 1)]
        for i in range(k):
            rhs[i] += x[i] * ys[t]
            for j in range(k):
                gram[i][j] += x[i] * x[j]
    try:
        theta = la.solve_cholesky(la.cholesky(tuple(tuple(r) for r in gram)), rhs)
    except ValueError:
        raise ValueError("the lags are collinear (a constant series?)") from None
    sse = 0.0
    for t in range(order, len(ys)):
        e = ys[t] - theta[0]
        for i in range(1, order + 1):
            e -= theta[i] * ys[t - i]
        sse += e * e
    sigma2 = sse / n
    if not sigma2 > 0:
        raise ValueError("the history is fitted exactly; sigma would be 0")
    log_lik = -0.5 * n * (_LOG_2PI + math.log(sigma2) + 1.0)
    model = AutoRegressive(order, coefficients=theta[1:], intercept=theta[0], sigma=math.sqrt(sigma2),
                           horizon=horizon, coverage=coverage, input=input)
    state = model.initial_state()
    for y in ys:
        state, _, _ = model.update(state, y)
    parameters = {"intercept": theta[0], "coefficients": list(theta[1:]), "sigma": math.sqrt(sigma2)}
    return ForecastFit(model, state, log_lik, 2.0 * (k + 1) - 2.0 * log_lik, parameters, 1)
