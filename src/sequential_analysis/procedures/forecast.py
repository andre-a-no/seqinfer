# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Forecasting: predictive distributions for the next `horizon` values of a series.

Every forecaster gives, after each observation, the forecast of the next
1..horizon values: a location, a scale, a distribution (normal, or
Student t with `df` degrees of freedom) and the central interval holding
`coverage` of it.  Each observation is first scored against the one-step
forecast made for it: `error` is y minus that forecast and `log_score` the
log predictive density of y, so `log_likelihood` is the prequential
log-likelihood, a fair score for comparing forecasters on the same data.

Two kinds of training:

* Fixed: the parameters are given, usually estimated from a history with
  `forecasting.fit_ets` or `forecasting.fit_ar`.  The state (level,
  trend, season, recent values) moves with the data, the parameters do
  not.  The forecasts are then exactly those of the fitted model.
* Continuous: the parameters are learnt from the stream itself.
  AutoRegressive without coefficients is Bayesian linear regression on
  the lags, with optional exponential forgetting for drift.
  ExponentialSmoothing without sigma learns the noise level.
  ModelAveraging weights several forecasters (a grid of smoothing
  parameters, say) by their predictive densities, optionally forgetting,
  so the weight moves to whichever currently forecasts best.

ConformalForecast puts distribution-free intervals around any of them, and
ForecastComparison tells, anytime-valid, whether one forecaster is better
than another.

ExponentialSmoothing is the additive-error ETS model of Hyndman, Koehler,
Ord and Snyder (2008), with no, additive or damped trend and an optional
additive season of period m:

    yhat_t = l_{t-1} + phi b_{t-1} + s_{t-m},     e_t = y_t - yhat_t,
    l_t = l_{t-1} + phi b_{t-1} + alpha e_t,
    b_t = phi b_{t-1} + beta e_t,
    s_t = s_{t-m} + gamma e_t.

The forecast h steps ahead is l + (phi + ... + phi^h) b + s (the season h
steps ahead), with variance sigma^2 (1 + sum_{j<h} c_j^2), c_j = alpha +
beta (phi + ... + phi^j) + gamma [j divisible by m].
These are the recursions and intervals of statsmodels' ETSModel, which the
tests check to 1e-12.  With sigma unknown, it gets the prior p(sigma^2) ~
1/sigma^2; after k errors the predictive distribution is then Student t
with k degrees of freedom and scale^2 = (sum of squared errors / k) times
the same factor, exact for the one-step forecast given the smoothing
parameters.

AutoRegressive is the AR(p) model y_t = c + phi_1 y_{t-1} + ... + phi_p
y_{t-p} + e_t.  With coefficients given, forecasts are the usual
recursions with variance sigma^2 sum_{j<h} psi_j^2.  Without, the
coefficients get the prior N(0, sigma^2 I / ridge) and sigma^2 the prior
1/sigma^2, and each observation updates the posterior exactly (Bayesian
linear regression on the lags).  With forgetting lambda < 1 the data's
information decays by lambda per step (the prior's does not), so the
coefficients are an exponentially weighted ridge regression and follow
drift.  The state holds the weighted sums x x' and x y, never a
covariance obtained by subtraction, so it stays valid for any data.  The
one-step predictive distribution is exactly Student t; for h > 1 the
scale adds the parameter uncertainty of the one-step regressors to the
plug-in variance, an approximation.

Cost per step (PERFORMANCE.md): constant time and state.  10 µs for
exponential smoothing with fixed parameters (trend, season of 12, 12
steps ahead); about 0.3 ms when sigma is learnt, most of it the Student t
quantile of the interval; 0.4 ms for a learnt AR(5); 7.5 ms and a 26 kB
state for averaged_smoothing over 63 models; 3 µs for the comparison.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from .._compat import zip_strict
from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..distributions import t_cdf, t_isf_log
from ..numerics import logsumexp, neumaier_add, require_finite

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


def _check_coverage(coverage: float) -> float:
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    return float(coverage)


def _check_horizon(horizon: int) -> int:
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive int")
    return horizon


def log_density(error: float, scale: float, df: float | None) -> float:
    """Log density of a normal (df None) or Student t location-scale law at `error` from its location."""
    z = error / scale
    if df is None:
        return -0.5 * _LOG_2PI - math.log(scale) - 0.5 * z * z
    return (math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df) - 0.5 * math.log(df * math.pi)
            - math.log(scale) - 0.5 * (df + 1.0) * math.log1p(z * z / df))


def standard_cdf(z: float, df: float | None) -> float:
    return 0.5 * math.erfc(-z / _SQRT2) if df is None else t_cdf(z, df)


@dataclass(frozen=True)
class Predictive:
    """The forecast of the next 1..horizon values: location + scale * (standard normal or t_df)."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    df: float | None


@dataclass(frozen=True)
class ForecastOutput:
    n: int
    mean: tuple[float, ...]  # forecasts of the next 1..horizon values; () until the model can forecast
    scale: tuple[float, ...]
    df: float | None  # Student t degrees of freedom of the forecasts; None for normal ones
    lower: tuple[float, ...]  # central intervals holding `coverage` of each forecast
    upper: tuple[float, ...]
    error: float | None  # this observation minus its one-step forecast; None if there was none
    log_score: float | None  # log predictive density of this observation
    log_likelihood: float  # sum of the log scores so far
    parameters: dict  # what the model currently uses: level and trend, or coefficients, or weights


class Forecaster(Procedure):
    """Common protocol: `update` scores and absorbs one value, `predictive` forecasts from a state."""

    horizon: int
    coverage: float
    input: str

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def update(self, state: Any, y: float) -> tuple[Any, float | None, float | None]:
        """(new state, error, log score) after observing y."""
        raise NotImplementedError

    def predictive(self, state: Any) -> Predictive | None:
        raise NotImplementedError

    def parameters(self, state: Any) -> dict:
        return {}

    def log_likelihood(self, state: Any) -> float:
        return float(state.log_likelihood + state.comp)

    def interval(self, pred: Predictive) -> tuple[tuple[float, ...], tuple[float, ...]]:
        q = self._quantile(pred.df)
        lower = tuple(m - q * s for m, s in zip_strict(pred.mean, pred.scale))
        upper = tuple(m + q * s for m, s in zip_strict(pred.mean, pred.scale))
        return lower, upper

    def _quantile(self, df: float | None) -> float:
        tail = 0.5 * (1.0 - self.coverage)
        if df is None:
            return -NormalDist().inv_cdf(tail)
        return t_isf_log(math.log(tail), df)

    def step(self, state, x, rng=None):
        y = x[self.input]
        state, error, score = self.update(state, y)
        pred = self.predictive(state)
        n = state.n
        log_lik = require_finite(self.log_likelihood(state), "log likelihood")
        if pred is None:
            return state, ForecastOutput(n, (), (), None, (), (), error, score, log_lik, self.parameters(state))
        lower, upper = self.interval(pred)
        return state, ForecastOutput(n, pred.mean, pred.scale, pred.df, lower, upper, error, score, log_lik,
                                     self.parameters(state))


# ---------------------------------------------------------------------------------------- ETS


@dataclass(frozen=True)
class ExponentialSmoothingState:
    n: int
    level: float
    trend: float
    seasonal: tuple[float, ...]  # seasonal[0] applies to the next observation
    errors: int  # one-step errors so far (they estimate sigma when it is not given)
    sse: float
    sse_comp: float
    log_likelihood: float = 0.0
    comp: float = 0.0


TRENDS = (None, "additive", "damped")


class ExponentialSmoothing(Forecaster):
    """Additive-error exponential smoothing (ETS) with optional trend and season; see the module docstring.

    `initial_level` None lets the first observation set the level (it is
    not scored); `initial_seasonal` lists the seasonal effects of the
    first `period` observations, in order (zeros by default).  `sigma`
    None learns the noise level from the one-step errors.
    """

    name = "exponential_smoothing"
    version = "1"

    def __init__(self, alpha: float, *, beta: float = 0.0, gamma: float = 0.0, phi: float = 1.0,
                 trend: str | None = None, period: int | None = None, initial_level: float | None = None,
                 initial_trend: float = 0.0, initial_seasonal: Sequence[float] | None = None,
                 sigma: float | None = None, horizon: int = 1, coverage: float = 0.9, input: str = "y"):
        if trend not in TRENDS:
            raise ValueError(f"trend must be one of {TRENDS}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0.0 <= beta <= 1.0 or not 0.0 <= gamma <= 1.0:
            raise ValueError("beta and gamma must lie in [0, 1]")
        if not 0.0 < phi <= 1.0:
            raise ValueError("phi must lie in (0, 1]")
        if trend is None and (beta != 0.0 or initial_trend != 0.0):
            raise ValueError("beta and initial_trend need a trend")
        if trend != "damped" and phi != 1.0:
            raise ValueError("phi < 1 needs trend='damped'")
        if period is None:
            if gamma != 0.0 or initial_seasonal is not None:
                raise ValueError("gamma and initial_seasonal need a period")
        elif type(period) is not int or period < 2:
            raise ValueError("period must be an int >= 2")
        if sigma is not None and not 0.0 < sigma < math.inf:
            raise ValueError("sigma must be positive")
        for v in (initial_level, initial_trend):
            if v is not None and not math.isfinite(v):
                raise ValueError("initial values must be finite")
        seasonal = tuple(float(v) for v in initial_seasonal) if initial_seasonal is not None else (
            (0.0,) * period if period is not None else ())
        if period is not None and len(seasonal) != period:
            raise ValueError("initial_seasonal needs one value per season")
        if not all(math.isfinite(v) for v in seasonal):
            raise ValueError("initial values must be finite")
        self.alpha, self.beta, self.gamma, self.phi = float(alpha), float(beta), float(gamma), float(phi)
        self.trend, self.period = trend, period
        self.initial_level = None if initial_level is None else float(initial_level)
        self.initial_trend, self.initial_seasonal = float(initial_trend), seasonal
        self.sigma = None if sigma is None else float(sigma)
        self.horizon, self.coverage, self.input = _check_horizon(horizon), _check_coverage(coverage), input
        # phi_h = phi + ... + phi^h multiplies the trend; v_h is the variance factor of the h-step forecast
        damping, factors = [], []
        acc, power, v = 0.0, 1.0, 1.0
        for h in range(1, self.horizon + 1):
            if h > 1:
                c = self.alpha + self.beta * acc + (self.gamma if period is not None and (h - 1) % period == 0
                                                    else 0.0)
                v += c * c
            power *= self.phi
            acc += power
            damping.append(acc)
            factors.append(math.sqrt(v))
        self._damping, self._factors = tuple(damping), tuple(factors)

    def config(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "phi": self.phi, "trend": self.trend,
                "period": self.period, "initial_level": self.initial_level, "initial_trend": self.initial_trend,
                "initial_seasonal": list(self.initial_seasonal) if self.period is not None else None,
                "sigma": self.sigma, "horizon": self.horizon,
                "coverage": self.coverage, "input": self.input}

    def initial_state(self) -> ExponentialSmoothingState:
        level = 0.0 if self.initial_level is None else self.initial_level
        return ExponentialSmoothingState(0, level, self.initial_trend, self.initial_seasonal, 0, 0.0, 0.0)

    def _started(self, state: ExponentialSmoothingState) -> bool:
        return state.n > 0 or self.initial_level is not None

    def _rotate(self, seasonal: tuple[float, ...], new: float) -> tuple[float, ...]:
        return (*seasonal[1:], new) if seasonal else ()

    def _noise(self, state: ExponentialSmoothingState) -> tuple[float, float | None] | None:
        """(sigma, df) of the one-step forecast, or None while sigma cannot be estimated."""
        if self.sigma is not None:
            return self.sigma, None
        sse = state.sse + state.sse_comp
        if state.errors < 1 or not sse > 0.0:
            return None
        return math.sqrt(sse / state.errors), float(state.errors)

    def update(self, state: ExponentialSmoothingState, y: float):
        season = state.seasonal[0] if state.seasonal else 0.0
        if not self._started(state):
            new = ExponentialSmoothingState(1, y - season, state.trend, self._rotate(state.seasonal, season), 0,
                                            0.0, 0.0, state.log_likelihood, state.comp)
            return new, None, None
        error = y - (state.level + self.phi * state.trend + season)
        noise = self._noise(state)
        score = None if noise is None else log_density(error, noise[0], noise[1])
        level = require_finite(state.level + self.phi * state.trend + self.alpha * error, "level")
        trend = require_finite(self.phi * state.trend + self.beta * error, "trend")
        seasonal = self._rotate(state.seasonal, season + self.gamma * error)
        sse, sse_comp = neumaier_add(state.sse, state.sse_comp, error * error)
        log_lik, comp = state.log_likelihood, state.comp
        if score is not None:
            log_lik, comp = neumaier_add(log_lik, comp, score)
        return (ExponentialSmoothingState(state.n + 1, level, trend, seasonal, state.errors + 1, sse, sse_comp,
                                          log_lik, comp), error, score)

    def predictive(self, state: ExponentialSmoothingState) -> Predictive | None:
        noise = self._noise(state) if self._started(state) else None
        if noise is None:
            return None
        sigma, df = noise
        mean = []
        for h in range(1, self.horizon + 1):
            season = state.seasonal[(h - 1) % len(state.seasonal)] if state.seasonal else 0.0
            mean.append(state.level + self._damping[h - 1] * state.trend + season)
        return Predictive(tuple(mean), tuple(sigma * f for f in self._factors), df)

    def parameters(self, state: ExponentialSmoothingState) -> dict:
        noise = self._noise(state) if self._started(state) else None
        return {"level": state.level, "trend": state.trend, "seasonal": list(state.seasonal),
                "sigma": None if noise is None else noise[0]}

    def decode_state(self, data) -> ExponentialSmoothingState:
        state = decode_fields(ExponentialSmoothingState, data)
        require_state(len(state.seasonal) == len(self.initial_seasonal), "one seasonal effect per season")
        require_state(0 <= state.errors <= state.n and state.sse >= 0, "error statistics")
        require_state(state.n > 0 or (state.errors == 0 and state.sse == 0 and state.log_likelihood == 0),
                      "statistics before data")
        require_state(self.trend is not None or state.trend == 0, "a trend without a trend model")
        return state


# ------------------------------------------------------------------------------------------ AR


def _cholesky_floor(a: Sequence[Sequence[float]]) -> list[list[float]]:
    """Cholesky factor of a symmetric positive definite matrix, pivots floored at 1e-13 of the diagonal.

    In exact arithmetic the matrices here are positive definite; the floor
    only keeps a numerically singular direction (collinear lags) from
    breaking the factorization -- it is regularized instead.
    """
    k = len(a)
    low = [[0.0] * k for _ in range(k)]
    for j in range(k):
        d = a[j][j]
        for i in range(j):
            d -= low[j][i] * low[j][i]
        d = max(d, 1e-13 * a[j][j])
        low[j][j] = math.sqrt(d)
        for r in range(j + 1, k):
            v = a[r][j]
            for i in range(j):
                v -= low[r][i] * low[j][i]
            low[r][j] = v / low[j][j]
    return low


def _forward(low: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    out: list[float] = []
    for i in range(len(b)):
        v = b[i]
        for j in range(i):
            v -= low[i][j] * out[j]
        out.append(v / low[i][i])
    return out


def _backward(low: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    k = len(b)
    out = [0.0] * k
    for i in reversed(range(k)):
        v = b[i]
        for j in range(i + 1, k):
            v -= low[j][i] * out[j]
        out[i] = v / low[i][i]
    return out


@dataclass(frozen=True)
class AutoRegressiveState:
    n: int
    lags: tuple[float, ...]  # the most recent values, newest first (fewer than p at the start)
    information: tuple[tuple[float, ...], ...]  # sum of lambda^(n-t) x_t x_t'; () with fixed coefficients
    moment: tuple[float, ...]  # sum of lambda^(n-t) x_t y_t
    shape: float  # inverse-gamma posterior of sigma^2: shape a and rate b
    rate: float
    log_likelihood: float = 0.0
    comp: float = 0.0


class AutoRegressive(Forecaster):
    """AR(p) forecasts, with fixed coefficients or learnt by Bayesian regression on the lags.

    Give `coefficients` (phi_1..phi_p), `intercept` and `sigma` for fixed
    training.  Leave them out to learn them: `forgetting` lambda in (0, 1]
    discounts the past by lambda per step (1 keeps everything), and `ridge`
    is the prior precision of the coefficients relative to the noise, which
    is not discounted.  The coefficients used for each forecast are then
    the exponentially weighted ridge regression on the lags,

        (sum_t lambda^(n+1-t) x_t x_t' + ridge I)^-1 sum_t lambda^(n+1-t) x_t y_t,

    x_t = (1, y_{t-1}, ..., y_{t-p}): the exact posterior mean when the
    data's information decays by lambda per step and the prior's does not.
    """

    name = "autoregressive"
    version = "1"

    def __init__(self, order: int, *, coefficients: Sequence[float] | None = None, intercept: float = 0.0,
                 sigma: float | None = None, forgetting: float = 1.0, ridge: float = 1e-6, horizon: int = 1,
                 coverage: float = 0.9, input: str = "y"):
        if type(order) is not int or order < 1:
            raise ValueError("order must be a positive int")
        self.order = order
        if coefficients is not None:
            phi = tuple(float(c) for c in coefficients)
            if len(phi) != order or not all(math.isfinite(c) for c in phi) or not math.isfinite(intercept):
                raise ValueError("need `order` finite coefficients and a finite intercept")
            if sigma is None or not 0.0 < sigma < math.inf:
                raise ValueError("fixed coefficients need a positive sigma")
            if forgetting != 1.0 or ridge != 1e-6:
                raise ValueError("forgetting and ridge are for learnt coefficients")
            self.phi: tuple[float, ...] | None = phi
        else:
            if sigma is not None or intercept != 0.0:
                raise ValueError("sigma and intercept come with fixed coefficients")
            if not 0.0 < forgetting <= 1.0:
                raise ValueError("forgetting must lie in (0, 1]")
            if not 0.0 < ridge < math.inf:
                raise ValueError("ridge must be positive")
            self.phi = None
        self.intercept = float(intercept)
        self.sigma = None if sigma is None else float(sigma)
        self.forgetting, self.ridge = float(forgetting), float(ridge)
        self.horizon, self.coverage, self.input = _check_horizon(horizon), _check_coverage(coverage), input

    def config(self) -> dict:
        return {"order": self.order, "coefficients": None if self.phi is None else list(self.phi),
                "intercept": self.intercept, "sigma": self.sigma, "forgetting": self.forgetting,
                "ridge": self.ridge, "horizon": self.horizon, "coverage": self.coverage, "input": self.input}

    def initial_state(self) -> AutoRegressiveState:
        if self.phi is not None:
            return AutoRegressiveState(0, (), (), (), 0.0, 0.0)
        k = self.order + 1
        return AutoRegressiveState(0, (), tuple((0.0,) * k for _ in range(k)), (0.0,) * k, 0.0, 0.0)

    def _point(self, theta: Sequence[float], lags: Sequence[float]) -> float:
        total = theta[0]
        for i in range(self.order):
            total += theta[i + 1] * lags[i]
        return total

    def _prior(self, state: AutoRegressiveState, x: Sequence[float]) -> tuple[list[float], float]:
        """Coefficients for the next value, and x' M^-1 x with M = lambda (information) + ridge I."""
        lam, k = self.forgetting, self.order + 1
        precision = [[lam * state.information[i][j] + (self.ridge if i == j else 0.0) for j in range(k)]
                     for i in range(k)]
        low = _cholesky_floor(precision)
        theta = _backward(low, _forward(low, [lam * v for v in state.moment]))
        w = _forward(low, x)
        quad = 0.0
        for v in w:
            quad += v * v
        return theta, quad

    def update(self, state: AutoRegressiveState, y: float):
        lags = (y, *state.lags)[: self.order]
        if len(state.lags) < self.order:
            return (AutoRegressiveState(state.n + 1, lags, state.information, state.moment, state.shape,
                                        state.rate, state.log_likelihood, state.comp), None, None)
        log_lik, comp = state.log_likelihood, state.comp
        if self.phi is not None:
            assert self.sigma is not None
            error = y - self._point((self.intercept, *self.phi), state.lags)
            fixed_score = log_density(error, self.sigma, None)
            log_lik, comp = neumaier_add(log_lik, comp, fixed_score)
            return AutoRegressiveState(state.n + 1, lags, (), (), 0.0, 0.0, log_lik, comp), error, fixed_score
        x = (1.0, *state.lags)
        k = len(x)
        theta, quad = self._prior(state, x)
        error = y - self._point(theta, state.lags)
        s = 1.0 + quad
        score = None
        if state.shape > 0.0 and state.rate > 0.0:
            score = log_density(error, math.sqrt(state.rate / state.shape * s), 2.0 * self.forgetting * state.shape)
            log_lik, comp = neumaier_add(log_lik, comp, score)
        lam = self.forgetting
        information = tuple(tuple(require_finite(lam * state.information[i][j] + x[i] * x[j], "information")
                                  for j in range(k)) for i in range(k))
        moment = tuple(require_finite(lam * state.moment[i] + x[i] * y, "moment") for i in range(k))
        shape = lam * state.shape + 0.5
        rate = require_finite(lam * state.rate + error * error / (2.0 * s), "noise statistic")
        return (AutoRegressiveState(state.n + 1, lags, information, moment, shape, rate, log_lik, comp),
                error, score)

    def _coefficients(self, state: AutoRegressiveState) -> tuple[list[float], float]:
        if self.phi is not None:
            return [self.intercept, *self.phi], 0.0
        return self._prior(state, (1.0, *state.lags))

    def predictive(self, state: AutoRegressiveState) -> Predictive | None:
        if len(state.lags) < self.order:
            return None
        if self.phi is not None:
            assert self.sigma is not None
            sigma2, df = self.sigma * self.sigma, None
        elif state.shape > 0.0 and state.rate > 0.0:
            sigma2, df = state.rate / state.shape, 2.0 * self.forgetting * state.shape
        else:
            return None
        theta, extra = self._coefficients(state)
        history = list(state.lags)
        psi = [1.0]
        mean, scale = [], []
        total = 0.0
        for h in range(1, self.horizon + 1):
            value = self._point(theta, history)
            mean.append(value)
            history = [value, *history[:-1]]
            total += psi[h - 1] * psi[h - 1]
            scale.append(math.sqrt(sigma2 * (extra + total)))
            nxt = 0.0
            for i in range(1, min(h, self.order) + 1):
                nxt += theta[i] * psi[h - i]
            psi.append(nxt)
        return Predictive(tuple(mean), tuple(scale), df)

    def parameters(self, state: AutoRegressiveState) -> dict:
        """The coefficients and noise level the next forecast uses."""
        if self.phi is not None:
            return {"intercept": self.intercept, "coefficients": list(self.phi), "sigma": self.sigma}
        if len(state.lags) < self.order:
            return {"intercept": None, "coefficients": None, "sigma": None}
        theta, _ = self._coefficients(state)
        sigma = math.sqrt(state.rate / state.shape) if state.shape > 0.0 and state.rate > 0.0 else None
        return {"intercept": theta[0], "coefficients": theta[1:], "sigma": sigma}

    def decode_state(self, data) -> AutoRegressiveState:
        state = decode_fields(AutoRegressiveState, data)
        require_state(len(state.lags) == min(state.n, self.order), "one lag per observation, up to the order")
        if self.phi is not None:
            require_state(state.information == () and state.moment == () and state.shape == state.rate == 0,
                          "learnt statistics with fixed coefficients")
        else:
            k = self.order + 1
            info = state.information
            require_state(len(info) == k and len(state.moment) == k and all(len(row) == k for row in info),
                          "statistics' sizes")
            require_state(all(info[i][i] >= 0 and info[i][j] == info[j][i] for i in range(k) for j in range(k)),
                          "information must be symmetric with a non-negative diagonal")
            require_state(state.shape >= 0 and state.rate >= 0, "noise statistics")
        return state


# ----------------------------------------------------------------------------- model averaging


@dataclass(frozen=True)
class ModelAveragingState:
    n: int
    states: tuple[Any, ...]
    log_weights: tuple[float, ...]  # posterior weights after the last observation, normalized
    log_likelihood: float = 0.0
    comp: float = 0.0


class ModelAveraging(Forecaster):
    """Bayesian model averaging of forecasters, with optional forgetting (dynamic model averaging).

    The weight of model k is its prior weight times its predictive density
    of every observation so far, normalized: the exact posterior
    probability of each model.  With `forgetting` lambda < 1 the weights
    are flattened before each observation (raised to the power lambda,
    Raftery, Karny and Ettler 2010), so an old record fades and the weight
    can move to a model that has become better.  The forecast is the
    mixture of the models' forecasts with these weights: `mean` is the
    weighted mean of their locations, the interval holds `coverage` of the
    mixture, and `scale` is the normal-equivalent one (the interval's half
    width over the normal quantile).

    Weights move only on observations that every model forecast.
    `averaged_smoothing` builds one over a grid of smoothing parameters.
    The models must be ExponentialSmoothing or AutoRegressive forecasters
    with the same horizon and input.
    """

    name = "model_averaging"
    version = "1"

    def __init__(self, models: Sequence[Forecaster], *, weights: Sequence[float] | None = None,
                 forgetting: float = 1.0, coverage: float = 0.9):
        models = tuple(models)
        if not models:
            raise ValueError("at least one model is needed")
        for m in models:
            if not isinstance(m, (ExponentialSmoothing, AutoRegressive)):
                raise ValueError("models must be ExponentialSmoothing or AutoRegressive forecasters")
        if len({m.horizon for m in models}) != 1 or len({m.input for m in models}) != 1:
            raise ValueError("the models need the same horizon and the same input")
        prior = tuple(float(w) for w in weights) if weights is not None else (1.0,) * len(models)
        if len(prior) != len(models) or not all(0.0 < w < math.inf for w in prior):
            raise ValueError("need one positive weight per model")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0, 1]")
        self.models, self.prior, self.forgetting = models, prior, float(forgetting)
        self.horizon, self.input = models[0].horizon, models[0].input
        self.coverage = _check_coverage(coverage)
        logs = [math.log(w) for w in prior]
        top = logsumexp(logs)
        self._log_prior = tuple(v - top for v in logs)

    def config(self) -> dict:
        return {"models": [m.identity() for m in self.models], "weights": list(self.prior),
                "forgetting": self.forgetting, "coverage": self.coverage}

    def initial_state(self) -> ModelAveragingState:
        return ModelAveragingState(0, tuple(m.initial_state() for m in self.models), self._log_prior)

    def _flattened(self, log_weights: Sequence[float]) -> list[float]:
        logs = [self.forgetting * v for v in log_weights]
        top = logsumexp(logs)
        return [v - top for v in logs]

    def update(self, state: ModelAveragingState, y: float):
        states, scores = [], []
        for model, s in zip_strict(self.models, state.states):
            s, _, score = model.update(s, y)
            states.append(s)
            scores.append(score)
        log_weights, log_lik, comp = state.log_weights, state.log_likelihood, state.comp
        score = None
        known = [v for v in scores if v is not None]
        if len(known) == len(scores):
            prior = self._flattened(state.log_weights)
            joint = [w + v for w, v in zip_strict(prior, known)]
            score = require_finite(logsumexp(joint), "log score")
            log_weights = tuple(v - score for v in joint)
            log_lik, comp = neumaier_add(log_lik, comp, score)
        new = ModelAveragingState(state.n + 1, tuple(states), log_weights, log_lik, comp)
        return new, None, score

    def _components(self, state: ModelAveragingState) -> list[tuple[float, Predictive]] | None:
        weights = self._flattened(state.log_weights)
        out = []
        for model, s, w in zip_strict(self.models, state.states, weights):
            pred = model.predictive(s)
            if pred is None:
                return None
            out.append((math.exp(w), pred))
        return out

    def _mixture(self, state: ModelAveragingState):
        comps = self._components(state)
        if comps is None:
            return None
        # the components that carry all but 1e-10 of the weight; the rest cannot move a quantile visibly
        ranked = sorted(range(len(comps)), key=lambda i: -comps[i][0])
        live, mass = [], 0.0
        for i in ranked:
            if mass >= 1.0 - 1e-10:
                break
            live.append(comps[i])
            mass += comps[i][0]
        tail = 0.5 * (1.0 - self.coverage)
        quantiles: dict[float | None, float] = {}
        for _, p in live:
            if p.df not in quantiles:
                quantiles[p.df] = self._quantile(p.df)
        mean, lower, upper = [], [], []
        for h in range(self.horizon):
            total = 0.0
            for w, p in comps:
                total += w * p.mean[h]
            mean.append(total)
            lo = min(p.mean[h] - quantiles[p.df] * p.scale[h] for _, p in live)
            hi = max(p.mean[h] + quantiles[p.df] * p.scale[h] for _, p in live)
            lower.append(self._solve(live, h, tail, lo, hi))
            upper.append(self._solve(live, h, 1.0 - tail, lo, hi))
        return mean, lower, upper

    @staticmethod
    def _solve(live, h: int, target: float, lo: float, hi: float) -> float:
        """The point where the mixture CDF reaches `target` in [lo, hi]: Newton steps, bisecting when they leave."""
        x = 0.5 * (lo + hi)
        for _ in range(100):
            cdf, pdf = 0.0, 0.0
            for w, p in live:
                z = (x - p.mean[h]) / p.scale[h]
                cdf += w * standard_cdf(z, p.df)
                pdf += w * math.exp(log_density(z, 1.0, p.df)) / p.scale[h]
            if cdf == target:
                return x
            if cdf < target:
                lo = x
            else:
                hi = x
            step = (cdf - target) / pdf if pdf > 0.0 else math.inf
            nxt = x - step
            if not lo < nxt < hi:
                nxt = 0.5 * (lo + hi)
            if abs(nxt - x) <= 1e-13 * (abs(x) + hi - lo) or nxt in (lo, hi):
                return nxt
            x = nxt
        return x

    def predictive(self, state: ModelAveragingState) -> Predictive | None:
        mix = self._mixture(state)
        if mix is None:
            return None
        mean, lower, upper = mix
        z = self._quantile(None)
        return Predictive(tuple(mean), tuple((u - lo) / (2.0 * z) for lo, u in zip_strict(lower, upper)), None)

    def step(self, state, x, rng=None):
        state, _, score = self.update(state, x[self.input])
        mix = self._mixture(state)
        log_lik = require_finite(state.log_likelihood + state.comp, "log likelihood")
        params = self.parameters(state)
        if mix is None:
            return state, ForecastOutput(state.n, (), (), None, (), (), None, score, log_lik, params)
        mean, lower, upper = mix
        z = self._quantile(None)
        scale = tuple((u - lo) / (2.0 * z) for lo, u in zip_strict(lower, upper))
        return state, ForecastOutput(state.n, tuple(mean), scale, None, tuple(lower), tuple(upper), None, score,
                                     log_lik, params)

    def parameters(self, state: ModelAveragingState) -> dict:
        return {"weights": [math.exp(v) for v in state.log_weights]}

    def encode_state(self, state: ModelAveragingState):
        return {"n": state.n, "states": [m.encode_state(s) for m, s in zip_strict(self.models, state.states)],
                "log_weights": list(state.log_weights), "log_likelihood": state.log_likelihood, "comp": state.comp}

    def decode_state(self, data) -> ModelAveragingState:
        require_state(isinstance(data, dict) and set(data) == {"n", "states", "log_weights", "log_likelihood",
                                                               "comp"}, "fields")
        n, states, logs = data["n"], data["states"], data["log_weights"]
        require_state(type(n) is int and n >= 0, "n")
        require_state(isinstance(states, list) and len(states) == len(self.models), "one state per model")
        require_state(isinstance(logs, list) and len(logs) == len(self.models)
                      and all(type(v) in (int, float) and v <= 1e-9 for v in logs), "log weights")
        require_state(abs(logsumexp([float(v) for v in logs])) < 1e-9, "weights must sum to 1")
        for key in ("log_likelihood", "comp"):
            require_state(type(data[key]) in (int, float) and math.isfinite(data[key]), key)
        decoded = tuple(m.decode_state(s) for m, s in zip_strict(self.models, states))
        require_state(all(s.n == n for s in decoded), "every model has seen every observation")
        return ModelAveragingState(n, decoded, tuple(float(v) for v in logs), float(data["log_likelihood"]),
                                   float(data["comp"]))


def averaged_smoothing(*, trend: str | None = None, period: int | None = None,
                       alphas: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
                       betas: Sequence[float] = (0.01, 0.1, 0.3), gammas: Sequence[float] = (0.01, 0.1, 0.3),
                       phis: Sequence[float] = (0.9, 0.98), forgetting: float = 1.0, horizon: int = 1,
                       coverage: float = 0.9, input: str = "y") -> ModelAveraging:
    """Exponential smoothing that learns its smoothing parameters: model averaging over a grid.

    beta is given as a fraction of alpha and gamma as a fraction of
    1 - alpha (the usual admissible region beta < alpha, gamma < 1 - alpha);
    `phis` are used only with a damped trend.  Each model learns its own
    noise level, so the averaging is over the smoothing parameters given the
    data.
    """
    models = []
    for a in alphas:
        for b in (betas if trend is not None else (0.0,)):
            for g in (gammas if period is not None else (0.0,)):
                for f in (phis if trend == "damped" else (1.0,)):
                    models.append(ExponentialSmoothing(a, beta=a * b, gamma=(1.0 - a) * g, phi=f, trend=trend,
                                                       period=period, horizon=horizon, coverage=coverage,
                                                       input=input))
    return ModelAveraging(models, forgetting=forgetting, coverage=coverage)


# ----------------------------------------------------------------------------------- conformal


@dataclass(frozen=True)
class ConformalForecastState:
    n: int
    inner: Any
    pending: tuple[Any, ...]  # intervals issued 1..horizon steps ago: (lower, upper) tuples, or None
    quantiles: tuple[float, ...]  # one per horizon
    counts: tuple[int, ...]  # intervals judged, per horizon
    misses: tuple[int, ...]


@dataclass(frozen=True)
class ConformalForecastOutput:
    n: int
    mean: tuple[float, ...]
    lower: tuple[float, ...]  # distribution-free intervals for the next 1..horizon values
    upper: tuple[float, ...]
    quantiles: tuple[float, ...]  # their half widths, in units of the forecaster's scale when normalized
    miscoverage: tuple[float | None, ...]  # fraction of judged intervals that missed, per horizon
    counts: tuple[int, ...]
    forecaster: Any  # the wrapped forecaster's own output


class ConformalForecast(Procedure):
    """Distribution-free forecast intervals around any forecaster: online conformal quantile tracking.

    For each horizon h it keeps a half width q_h (in units of the
    forecaster's scale with `normalize`, else in the data's units) and
    issues the interval mean +- q_h scale.  When the value it was issued
    for arrives, q_h moves by step_size (miss - alpha): up after a miss,
    down a little after a hit (Angelopoulos, Candes and Tibshirani 2023,
    the quantile-tracking form of adaptive conformal inference, Gibbs and
    Candes 2021).  Telescoping the updates gives, for every sequence of
    data whatsoever,

        miscoverage - alpha = (q_now - q_start) / (step_size * count),

    and q stays within step_size of the range of the scores, so the
    fraction of intervals that missed tends to alpha at rate 1/count with
    no assumption on the data or the forecaster -- a long-run guarantee,
    not a per-interval probability.  An interval with q < 0 is empty
    (lower above upper) and counts as a miss.
    """

    name = "conformal_forecast"
    version = "1"

    def __init__(self, forecaster: Forecaster, *, alpha: float = 0.1, step_size: float = 0.05,
                 normalize: bool = True, initial: float | None = None):
        if not isinstance(forecaster, Forecaster):
            raise ValueError("forecaster must be a forecaster (ExponentialSmoothing, AutoRegressive, ...)")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < step_size < math.inf:
            raise ValueError("step_size must be positive")
        if initial is None:
            if not normalize:
                raise ValueError("without normalize, give the initial half width in the data's units")
            initial = -NormalDist().inv_cdf(0.5 * alpha)
        if not math.isfinite(initial):
            raise ValueError("initial must be finite")
        self.forecaster, self.alpha, self.step_size = forecaster, float(alpha), float(step_size)
        self.normalize, self.initial = bool(normalize), float(initial)
        self.horizon, self.input = forecaster.horizon, forecaster.input

    def config(self) -> dict:
        return {"forecaster": self.forecaster.identity(), "alpha": self.alpha, "step_size": self.step_size,
                "normalize": self.normalize, "initial": self.initial}

    def inputs(self):
        return self.forecaster.inputs()

    def initial_state(self) -> ConformalForecastState:
        h = self.horizon
        return ConformalForecastState(0, self.forecaster.initial_state(), (None,) * h, (self.initial,) * h,
                                      (0,) * h, (0,) * h)

    def step(self, state: ConformalForecastState, x, rng=None):
        y = x[self.input]
        inner, out = self.forecaster.step(state.inner, x, rng)
        quantiles, counts, misses = list(state.quantiles), list(state.counts), list(state.misses)
        for h in range(self.horizon):
            issued = state.pending[h]
            if issued is None:
                continue
            miss = not issued[0][h] <= y <= issued[1][h]
            quantiles[h] = require_finite(quantiles[h] + self.step_size * ((1.0 if miss else 0.0) - self.alpha),
                                          "conformal quantile")
            counts[h] += 1
            misses[h] += 1 if miss else 0
        if out.mean:
            unit = out.scale if self.normalize else (1.0,) * self.horizon
            lower = tuple(m - q * s for m, q, s in zip_strict(out.mean, quantiles, unit))
            upper = tuple(m + q * s for m, q, s in zip_strict(out.mean, quantiles, unit))
            issued_now: Any = (lower, upper)
        else:
            lower = upper = ()
            issued_now = None
        pending = (issued_now, *state.pending[:-1])
        new = ConformalForecastState(state.n + 1, inner, pending, tuple(quantiles), tuple(counts), tuple(misses))
        rates = tuple(m / c if c else None for m, c in zip_strict(misses, counts))
        return new, ConformalForecastOutput(new.n, out.mean, lower, upper, tuple(quantiles), rates, tuple(counts),
                                            self.forecaster.encode_output(out))

    def encode_state(self, state: ConformalForecastState):
        pending = [None if p is None else [list(p[0]), list(p[1])] for p in state.pending]
        return {"n": state.n, "inner": self.forecaster.encode_state(state.inner), "pending": pending,
                "quantiles": list(state.quantiles), "counts": list(state.counts), "misses": list(state.misses)}

    def decode_state(self, data) -> ConformalForecastState:
        require_state(isinstance(data, dict) and set(data) == {"n", "inner", "pending", "quantiles", "counts",
                                                               "misses"}, "fields")
        h = self.horizon
        n = data["n"]
        require_state(type(n) is int and n >= 0, "n")
        lists = [data[k] for k in ("pending", "quantiles", "counts", "misses")]
        require_state(all(isinstance(v, list) and len(v) == h for v in lists), "one entry per horizon")
        require_state(all(type(v) in (int, float) and math.isfinite(v) for v in data["quantiles"]), "quantiles")
        require_state(all(type(c) is int and type(m) is int and 0 <= m <= c <= n
                          for c, m in zip_strict(data["counts"], data["misses"])), "counts")
        pending: list[tuple[tuple[float, ...], tuple[float, ...]] | None] = []
        for p in data["pending"]:
            if p is None:
                pending.append(None)
                continue
            require_state(isinstance(p, list) and len(p) == 2 and all(
                isinstance(side, list) and len(side) == h and all(type(v) in (int, float) and math.isfinite(v)
                                                                  for v in side) for side in p), "pending interval")
            pending.append((tuple(float(v) for v in p[0]), tuple(float(v) for v in p[1])))
        inner = self.forecaster.decode_state(data["inner"])
        require_state(getattr(inner, "n", n) == n, "the forecaster has seen every observation")
        return ConformalForecastState(n, inner, tuple(pending),
                                      tuple(float(v) for v in data["quantiles"]), tuple(data["counts"]),
                                      tuple(data["misses"]))


# ---------------------------------------------------------------------------------- comparison

_ZETA_1_4 = 3.105547277977581  # Riemann zeta(1.4)
_STITCH_S, _STITCH_ETA = 1.4, 2.0


def stitched_boundary(v: float, c: float, alpha: float, m: float) -> float:
    """Polynomial stitched boundary of Howard, Ramdas, McAuliffe and Sekhon (2021), with s = 1.4 and eta = 2.

    For a process S_t that is sub-exponential with variance process V_t
    and scale c, P(exists t: S_t >= boundary(max(V_t, m))) <= alpha.
    """
    v = max(v, m)
    eta, s = _STITCH_ETA, _STITCH_S
    ell = s * math.log(math.log(eta * v / m)) + math.log(_ZETA_1_4 / (alpha * math.log(eta) ** s))
    k1 = (eta ** 0.25 + eta ** -0.25) / _SQRT2
    k2 = (math.sqrt(eta) + 1.0) / 2.0
    return math.sqrt(k1 * k1 * v * ell + k2 * k2 * c * c * ell * ell) + c * k2 * ell


@dataclass(frozen=True)
class ForecastComparisonState:
    n: int
    total: float
    total_comp: float
    variance: float  # sum of (d_t - dhat_{t-1})^2, dhat the running mean before d_t
    variance_comp: float


@dataclass(frozen=True)
class ForecastComparisonOutput:
    n: int
    mean: float  # average loss of a minus that of b
    lower: float  # confidence sequence for the average of the conditional expected differences
    upper: float
    better: str | None  # "a" once the upper bound is below 0, "b" once the lower bound is above 0


class ForecastComparison(Procedure):
    """Anytime-valid comparison of two forecasters by their losses (Choe and Ramdas 2024).

    Each step takes the two forecasters' losses on the same value, both
    within `bounds` (an absolute error clipped at a cap, a Brier or
    interval score...).  With d_t = loss_a - loss_b, the target is
    Delta_n = (1/n) sum_t E[d_t | past]: the difference in expected loss
    averaged over the steps so far, which may drift -- no stationarity is
    assumed.  The sequence is the empirical Bernstein one of Howard et al.
    (2021) with their polynomial stitched boundary: with probability at least
    1 - alpha, Delta_n lies in [lower, upper] at every n at once, so it can
    be monitored after every value and stopped at will.
    """

    name = "forecast_comparison"
    version = "1"

    def __init__(self, bounds: Sequence[float], *, alpha: float = 0.05, min_variance: float | None = None,
                 a: str = "a", b: str = "b"):
        lo, hi = (float(v) for v in bounds)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError("bounds must be finite, lower < upper")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        width = hi - lo
        m = width * width if min_variance is None else float(min_variance)
        if not 0.0 < m < math.inf:
            raise ValueError("min_variance must be positive")
        if a == b:
            raise ValueError("the two losses need different names")
        self.bounds, self.alpha, self.min_variance, self.a, self.b = (lo, hi), float(alpha), m, a, b
        self._width = width

    def config(self) -> dict:
        return {"bounds": list(self.bounds), "alpha": self.alpha, "min_variance": self.min_variance,
                "a": self.a, "b": self.b}

    def inputs(self):
        lo, hi = self.bounds
        return {self.a: InputContract(self.a, lower=lo, upper=hi), self.b: InputContract(self.b, lower=lo, upper=hi)}

    def initial_state(self) -> ForecastComparisonState:
        return ForecastComparisonState(0, 0.0, 0.0, 0.0, 0.0)

    def step(self, state: ForecastComparisonState, x, rng=None):
        d = x[self.a] - x[self.b]
        guess = (state.total + state.total_comp) / state.n if state.n else 0.0
        total, total_comp = neumaier_add(state.total, state.total_comp, d)
        variance, variance_comp = neumaier_add(state.variance, state.variance_comp, (d - guess) ** 2)
        n = state.n + 1
        new = ForecastComparisonState(n, total, total_comp, variance, variance_comp)
        mean = (total + total_comp) / n
        radius = stitched_boundary(variance + variance_comp, 2.0 * self._width, 0.5 * self.alpha,
                                   self.min_variance) / n
        lower, upper = mean - radius, mean + radius
        better = "a" if upper < 0.0 else "b" if lower > 0.0 else None
        return new, ForecastComparisonOutput(n, mean, lower, upper, better)

    def decode_state(self, data) -> ForecastComparisonState:
        state = decode_fields(ForecastComparisonState, data)
        require_state(state.variance >= 0, "negative variance")
        require_state(state.n > 0 or (state.total == 0 and state.variance == 0), "statistics before data")
        return state
