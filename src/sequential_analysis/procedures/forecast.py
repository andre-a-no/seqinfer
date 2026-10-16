# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Forecasting: predictive distributions for the next `horizon` values of a series.

Every forecaster gives, after each observation, the forecast of the next
1..horizon values: a point forecast, a scale, a distribution (normal,
Student t with `df` degrees of freedom, or negative binomial for counts)
and the central interval holding `coverage` of it.  Each observation is
first scored against the one-step forecast made for it: `error` is y
minus that forecast and `log_score` the log predictive density (for
counts, probability) of y, so `log_likelihood` is the prequential
log-likelihood, a fair score for comparing forecasters on the same data.

Two kinds of training:

* Fixed: the parameters are given, usually estimated from a history with
  `forecasting.fit_ets`, `fit_ar` or `fit_counts`.  The state (level,
  trend, season, recent values) moves with the data, the parameters do
  not.  The forecasts are then exactly those of the fitted model.
* Continuous: the parameters are learnt from the stream itself.
  AutoRegressive without coefficients is Bayesian linear regression on
  the lags, with optional exponential forgetting for drift.
  ExponentialSmoothing without sigma learns the noise level, and without
  coefficients the effects of its regressors.  ModelAveraging weights
  several forecasters (a grid of smoothing parameters, say) by their
  predictive densities, optionally forgetting, so the weight moves to
  whichever currently forecasts best.

Regressors (promotions, holidays, prices, weather) are inputs named in
`regressors`.  Each is a list of horizon + 1 numbers: its value at the
current step, which explains the current observation, followed by its
planned values for the next 1..horizon steps, which the forecasts use.
A regressor must therefore be known ahead (a calendar, a planned
campaign); `forecasting.forecast_records` cuts these windows from
aligned columns.  `transform="log"` models log y: multiplicative effects
and a season that grows with the level become additive, the intervals
are those of the model mapped back exactly, and the point forecast is
then the median.

ConformalForecast puts distribution-free intervals around any of them, and
ForecastComparison tells, anytime-valid, whether one forecaster is better
than another.

ExponentialSmoothing is the additive-error ETS model of Hyndman, Koehler,
Ord and Snyder (2008), with no, additive or damped trend, an optional
additive season of period m and optional regressors z with effects b:

    yhat_t = l_{t-1} + phi b_{t-1} + s_{t-m} + b'z_t,     e_t = y_t - yhat_t,
    l_t = l_{t-1} + phi b_{t-1} + alpha e_t,
    b_t = phi b_{t-1} + beta e_t,
    s_t = s_{t-m} + gamma e_t.

The forecast h steps ahead is l + (phi + ... + phi^h) b + s (the season h
steps ahead) + b'z_{t+h}, with variance sigma^2 (1 + sum_{j<h} c_j^2),
c_j = alpha + beta (phi + ... + phi^j) + gamma [j divisible by m].
These are the recursions and intervals of statsmodels' ETSModel, which the
tests check to 1e-9.  With sigma unknown, it gets the prior p(sigma^2) ~
1/sigma^2; after k errors the predictive distribution is then Student t
with k degrees of freedom and scale^2 = (sum of squared errors / k) times
the same factor, exact given the smoothing parameters.

The effects of the regressors can be learnt too.  The state and errors
of the model are affine in the effects: with x~ the states of the run
without regressors and A_j those of a run driven by regressor j alone
(from zero), the states are x~ - sum_j b_j A_j and the errors
e~_t - sum_j b_j u_jt, with u_jt = z_jt - (what A_j predicts).  So the
effects are a linear regression of e~ on u, updated exactly (normal
prior with precision `ridge`, sigma^2 ~ 1/sigma^2, optional forgetting
of the data's information): the forecasts are Student t, exact at every
horizon given the smoothing parameters when nothing is forgotten.

AutoRegressive is the AR(p) model y_t = c + phi_1 y_{t-1} + ... + phi_p
y_{t-p} + b'z_t + e_t.  With coefficients given, forecasts are the usual
recursions with variance sigma^2 sum_{j<h} psi_j^2.  Without, the
coefficients get the prior N(0, sigma^2 I / ridge) and sigma^2 the prior
1/sigma^2, and each observation updates the posterior exactly (Bayesian
linear regression on the lags and the regressors).  With forgetting
lambda < 1 the data's information decays by lambda per step (the
prior's does not), so the coefficients are an exponentially weighted
ridge regression and follow drift.  The state holds the weighted sums
x x' and x y, never a covariance obtained by subtraction, so it stays
valid for any data.  The one-step predictive distribution is exactly
Student t; for h > 1 the scale adds the parameter uncertainty of the
one-step regressors to the plug-in variance, an approximation.

PoissonSmoothing forecasts counts (orders, calls, incidents) with the
Poisson local level model of Harvey and Fernandes (1989).  The rate mu_t
has a gamma distribution given the past, Gamma(a, b); before each
observation it is discounted, Gamma(omega a, omega b): same mean, less
certainty, so the rate can move.  Then y_t ~ Poisson(e_t mu_t), with an
optional exposure e_t (opening hours, a seasonal factor, a population),
and the update a + y_t, b + e_t is exact.  The forecast h steps ahead is
negative binomial with mean e a / b, whose spread comes from the
uncertainty in the rate; the intervals are counts, holding at least
`coverage`.  omega plays the part of the smoothing parameter: 1 is a
constant rate, smaller values follow changes faster.

Cost per step (PERFORMANCE.md): constant time and state.  20 µs for
exponential smoothing with fixed parameters (trend, season of 12, 12
steps ahead); about 0.3 ms when sigma is learnt, most of it the Student t
quantile of the interval, and 0.4 ms with two learnt regressors; 0.45 ms
for a learnt AR(5); 0.25 ms for counts 7 steps ahead (the count
quantiles); 6 ms and a 31 kB state for averaged_smoothing over 63 models;
2 µs for the comparison.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from .._compat import zip_strict
from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..distributions import regularized_beta, t_cdf, t_isf_log
from ..errors import ContractViolation
from ..numerics import logsumexp, neumaier_add, require_finite

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
TRANSFORMS = (None, "log")


def _check_coverage(coverage: float) -> float:
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    return float(coverage)


def _check_horizon(horizon: int) -> int:
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive int")
    return horizon


def _check_names(input: str, regressors: Sequence[str]) -> tuple[str, ...]:
    names = tuple(regressors)
    if not all(isinstance(r, str) for r in names) or len(set(names)) != len(names) or input in names:
        raise ValueError("regressors need distinct names, different from the input")
    return names


def log_density(error: float, scale: float, df: float | None) -> float:
    """Log density of a normal (df None) or Student t location-scale law at `error` from its location."""
    z = error / scale
    if df is None:
        return -0.5 * _LOG_2PI - math.log(scale) - 0.5 * z * z
    return (math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df) - 0.5 * math.log(df * math.pi)
            - math.log(scale) - 0.5 * (df + 1.0) * math.log1p(z * z / df))


def standard_cdf(z: float, df: float | None) -> float:
    return 0.5 * math.erfc(-z / _SQRT2) if df is None else t_cdf(z, df)


def negative_binomial_log_pmf(k: int, size: float, rate: float, exposure: float) -> float:
    """log P(Y = k) for Poisson(exposure mu) counts with mu ~ Gamma(size, rate)."""
    log_total = math.log(rate + exposure)
    return (math.lgamma(size + k) - math.lgamma(size) - math.lgamma(k + 1.0)
            + size * (math.log(rate) - log_total) + k * (math.log(exposure) - log_total))


def negative_binomial_cdf(k: int, size: float, rate: float, exposure: float) -> float:
    """P(Y <= k) for the same law: I_q(size, k + 1) with q = rate / (rate + exposure)."""
    if k < 0:
        return 0.0
    total = rate + exposure
    log_total = math.log(total)
    return regularized_beta(rate / total, size, k + 1.0, exposure / total,
                            (math.log(rate) - log_total, math.log(exposure) - log_total))


@dataclass(frozen=True)
class Predictive:
    """The forecast of the next 1..horizon values.

    Continuous (`size` empty): `mean` + `scale` * (standard normal, or t
    with `df`), on the log scale with `transform` "log".  Counts: negative
    binomial with `size`, `rate` and `exposure` per horizon; `mean` and
    `scale` are its mean and standard deviation.
    """

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    df: float | None
    transform: str | None = None
    size: tuple[float, ...] = ()
    rate: tuple[float, ...] = ()
    exposure: tuple[float, ...] = ()

    @property
    def counts(self) -> bool:
        return bool(self.size)

    def cdf(self, h: int, x: float) -> float:
        """P(Y <= x) for the forecast h + 1 steps ahead, in the data's units."""
        if self.counts:
            return negative_binomial_cdf(math.floor(x), self.size[h], self.rate[h], self.exposure[h])
        if self.transform == "log":
            if x <= 0.0:
                return 0.0
            x = math.log(x)
        return standard_cdf((x - self.mean[h]) / self.scale[h], self.df)

    def density(self, h: int, x: float) -> float:
        """The density at x in the data's units (continuous forecasts only)."""
        if self.transform == "log":
            if x <= 0.0:
                return 0.0
            return math.exp(log_density(math.log(x) - self.mean[h], self.scale[h], self.df)) / x
        return math.exp(log_density(x - self.mean[h], self.scale[h], self.df))


@dataclass(frozen=True)
class ForecastOutput:
    n: int
    mean: tuple[float, ...]  # point forecasts of the next 1..horizon values; () until the model can forecast
    scale: tuple[float, ...]
    df: float | None  # Student t degrees of freedom of the forecasts; None for normal ones and counts
    lower: tuple[float, ...]  # central intervals holding `coverage` of each forecast
    upper: tuple[float, ...]
    error: float | None  # this observation minus its one-step point forecast; None if there was none
    log_score: float | None  # log predictive density (probability, for counts) of this observation
    log_likelihood: float  # sum of the log scores so far
    parameters: dict  # what the model currently uses: level and trend, coefficients, effects, weights


class Forecaster(Procedure):
    """Common protocol: `update` scores and absorbs one record, `predictive` forecasts from a state."""

    horizon: int
    coverage: float
    input: str
    regressors: tuple[str, ...] = ()
    transform: str | None = None

    def inputs(self):
        contracts = {self.input: InputContract(self.input)}
        for r in self.regressors:
            contracts[r] = InputContract(r, shape=(self.horizon + 1,))
        return contracts

    def update(self, state: Any, x: Mapping[str, Any]) -> tuple[Any, float | None, float | None]:
        """(new state, error, log score) after observing the record x."""
        raise NotImplementedError

    def predictive(self, state: Any) -> Predictive | None:
        raise NotImplementedError

    def parameters(self, state: Any) -> dict:
        return {}

    def log_likelihood(self, state: Any) -> float:
        return float(state.log_likelihood + state.comp)

    def _value(self, x: Mapping[str, Any]) -> tuple[float, float]:
        """(the observation on the model's scale, the log Jacobian to subtract from its log density)."""
        y = x[self.input]
        if self.transform == "log":
            if not y > 0.0:
                raise ContractViolation(f"{self.name}: the log transform needs positive values, got {y!r}")
            log_y = math.log(y)
            return log_y, log_y
        return y, 0.0

    def _windows(self, x: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
        """(the regressors' current values, their planned values for 1..horizon steps ahead)."""
        now = tuple(float(x[r][0]) for r in self.regressors)
        plan = tuple(tuple(float(v) for v in x[r][1:]) for r in self.regressors)
        return now, plan

    def _point(self, location: float) -> float:
        return math.exp(location) if self.transform == "log" else location

    def interval(self, pred: Predictive) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if pred.counts:
            tail = 0.5 * (1.0 - self.coverage)
            lower = tuple(float(count_quantile(pred, h, tail)) for h in range(len(pred.mean)))
            upper = tuple(float(count_quantile(pred, h, 1.0 - tail)) for h in range(len(pred.mean)))
            return lower, upper
        q = self._quantile(pred.df)
        lower = tuple(self._point(m - q * s) for m, s in zip_strict(pred.mean, pred.scale))
        upper = tuple(self._point(m + q * s) for m, s in zip_strict(pred.mean, pred.scale))
        return lower, upper

    def _quantile(self, df: float | None) -> float:
        tail = 0.5 * (1.0 - self.coverage)
        if df is None:
            return -NormalDist().inv_cdf(tail)
        return t_isf_log(math.log(tail), df)

    def present(self, pred: Predictive) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """(point forecasts, scales) in the data's units: on the log scale, the median and m * s."""
        if self.transform == "log" and not pred.counts:
            mean = tuple(math.exp(m) for m in pred.mean)
            return mean, tuple(m * s for m, s in zip_strict(mean, pred.scale))
        return pred.mean, pred.scale

    def step(self, state, x, rng=None):
        state, error, score = self.update(state, x)
        pred = self.predictive(state)
        n = state.n
        log_lik = require_finite(self.log_likelihood(state), "log likelihood")
        if pred is None:
            return state, ForecastOutput(n, (), (), None, (), (), error, score, log_lik, self.parameters(state))
        lower, upper = self.interval(pred)
        mean, scale = self.present(pred)
        return state, ForecastOutput(n, mean, scale, pred.df, lower, upper, error, score, log_lik,
                                     self.parameters(state))


def count_quantile(pred: Predictive, h: int, p: float) -> int:
    """The smallest count k with P(Y <= k) >= p."""
    if pred.cdf(h, 0.0) >= p:
        return 0
    lo, hi = 0, 1
    while pred.cdf(h, float(hi)) < p:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:  # P(Y <= lo) < p <= P(Y <= hi)
        mid = (lo + hi) // 2
        if pred.cdf(h, float(mid)) < p:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------ Bayesian regression helpers


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


def _posterior(information, moment, forgetting: float, ridge: float):
    """(coefficients for the next value, Cholesky factor of M) with M = lambda information + ridge I."""
    k = len(moment)
    precision = [[forgetting * information[i][j] + (ridge if i == j else 0.0) for j in range(k)] for i in range(k)]
    low = _cholesky_floor(precision)
    return _backward(low, _forward(low, [forgetting * v for v in moment])), low


def _quad(low, u: Sequence[float]) -> float:
    """u' M^-1 u from the Cholesky factor of M."""
    total = 0.0
    for v in _forward(low, u):
        total += v * v
    return total


def _accumulate(information, moment, u: Sequence[float], target: float, forgetting: float):
    k = len(u)
    info = tuple(tuple(require_finite(forgetting * information[i][j] + u[i] * u[j], "information")
                       for j in range(k)) for i in range(k))
    mom = tuple(require_finite(forgetting * moment[i] + u[i] * target, "moment") for i in range(k))
    return info, mom


def _check_information(information, moment, k: int) -> None:
    require_state(len(information) == k and len(moment) == k and all(len(row) == k for row in information),
                  "statistics' sizes")
    require_state(all(information[i][i] >= 0 and information[i][j] == information[j][i]
                      for i in range(k) for j in range(k)),
                  "information must be symmetric with a non-negative diagonal")


def _check_learning(forgetting: float, ridge: float) -> None:
    if not 0.0 < forgetting <= 1.0:
        raise ValueError("forgetting must lie in (0, 1]")
    if not 0.0 < ridge < math.inf:
        raise ValueError("ridge must be positive")


# ---------------------------------------------------------------------------------------- ETS


@dataclass(frozen=True)
class ExponentialSmoothingState:
    n: int
    level: float  # with learnt effects: the states of the run without regressors
    trend: float
    seasonal: tuple[float, ...]  # seasonal[0] applies to the next observation
    errors: int  # one-step errors so far (they estimate sigma when it is not given)
    sse: float
    sse_comp: float
    log_likelihood: float = 0.0
    comp: float = 0.0
    plan: tuple[tuple[float, ...], ...] = ()  # the regressors' planned values for 1..horizon steps ahead
    effects: tuple[tuple[float, ...], ...] = ()  # learnt effects: the states A_j driven by each regressor
    information: tuple[tuple[float, ...], ...] = ()  # learnt effects: weighted sums u u' and u e~
    moment: tuple[float, ...] = ()
    shape: float = 0.0  # learnt effects: inverse-gamma posterior of sigma^2
    rate: float = 0.0


TRENDS = (None, "additive", "damped")


class ExponentialSmoothing(Forecaster):
    """Additive-error exponential smoothing (ETS) with optional trend, season and regressors.

    See the module docstring.  `initial_level` None lets the first
    observation set the level (it is not scored); `initial_seasonal` lists
    the seasonal effects of the first `period` observations, in order
    (zeros by default).  `sigma` None learns the noise level from the
    one-step errors.  With `regressors`, `coefficients` gives their
    effects; None learns them (then sigma must be None), with `forgetting`
    and `ridge` as in AutoRegressive.
    """

    name = "exponential_smoothing"
    version = "1"

    def __init__(self, alpha: float, *, beta: float = 0.0, gamma: float = 0.0, phi: float = 1.0,
                 trend: str | None = None, period: int | None = None, initial_level: float | None = None,
                 initial_trend: float = 0.0, initial_seasonal: Sequence[float] | None = None,
                 sigma: float | None = None, regressors: Sequence[str] = (),
                 coefficients: Sequence[float] | None = None, forgetting: float = 1.0, ridge: float = 1e-6,
                 transform: str | None = None, horizon: int = 1, coverage: float = 0.9, input: str = "y"):
        if trend not in TRENDS:
            raise ValueError(f"trend must be one of {TRENDS}")
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
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
        self.regressors = _check_names(input, regressors)
        self.effects: tuple[float, ...] | None = None
        if coefficients is not None:
            self.effects = tuple(float(c) for c in coefficients)
            if len(self.effects) != len(self.regressors) or not all(math.isfinite(c) for c in self.effects):
                raise ValueError("need one finite coefficient per regressor")
            if forgetting != 1.0 or ridge != 1e-6:
                raise ValueError("forgetting and ridge are for learnt effects")
        elif self.regressors:
            if sigma is not None:
                raise ValueError("learnt effects come with a learnt sigma (sigma=None)")
            _check_learning(forgetting, ridge)
        elif forgetting != 1.0 or ridge != 1e-6:
            raise ValueError("forgetting and ridge are for learnt effects")
        self.alpha, self.beta, self.gamma, self.phi = float(alpha), float(beta), float(gamma), float(phi)
        self.trend, self.period, self.transform = trend, period, transform
        self.initial_level = None if initial_level is None else float(initial_level)
        self.initial_trend, self.initial_seasonal = float(initial_trend), seasonal
        self.sigma = None if sigma is None else float(sigma)
        self.forgetting, self.ridge = float(forgetting), float(ridge)
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

    @property
    def _learning(self) -> bool:
        return bool(self.regressors) and self.effects is None

    def config(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "phi": self.phi, "trend": self.trend,
                "period": self.period, "initial_level": self.initial_level, "initial_trend": self.initial_trend,
                "initial_seasonal": list(self.initial_seasonal) if self.period is not None else None,
                "sigma": self.sigma, "regressors": list(self.regressors),
                "coefficients": None if self.effects is None else list(self.effects), "forgetting": self.forgetting,
                "ridge": self.ridge, "transform": self.transform, "horizon": self.horizon,
                "coverage": self.coverage, "input": self.input}

    def initial_state(self) -> ExponentialSmoothingState:
        level = 0.0 if self.initial_level is None else self.initial_level
        if not self._learning:
            return ExponentialSmoothingState(0, level, self.initial_trend, self.initial_seasonal, 0, 0.0, 0.0)
        r, width = len(self.regressors), 2 + len(self.initial_seasonal)
        return ExponentialSmoothingState(
            0, level, self.initial_trend, self.initial_seasonal, 0, 0.0, 0.0,
            effects=tuple((0.0,) * width for _ in range(r)), information=tuple((0.0,) * r for _ in range(r)),
            moment=(0.0,) * r)

    def _started(self, state: ExponentialSmoothingState) -> bool:
        return state.n > 0 or self.initial_level is not None

    def _rotate(self, seasonal: tuple[float, ...], new: float) -> tuple[float, ...]:
        return (*seasonal[1:], new) if seasonal else ()

    def _noise(self, state: ExponentialSmoothingState) -> tuple[float, float | None] | None:
        """(sigma, df) of the one-step forecast, or None while sigma cannot be estimated."""
        if self.sigma is not None:
            return self.sigma, None
        if self._learning:
            if not (state.shape > 0.0 and state.rate > 0.0):
                return None
            return math.sqrt(state.rate / state.shape), 2.0 * self.forgetting * state.shape
        sse = state.sse + state.sse_comp
        if state.errors < 1 or not sse > 0.0:
            return None
        return math.sqrt(sse / state.errors), float(state.errors)

    def _effect(self, values: Sequence[float]) -> float:
        total = 0.0
        if self.effects is not None:
            for b, z in zip_strict(self.effects, values):
                total += b * z
        return total

    def _advance(self, a: Sequence[float], innovation: float) -> tuple[float, ...]:
        """One step of a state vector (level, trend, season...) given its innovation."""
        level, trend, season = a[0], a[1], a[2:]
        first = season[0] if season else 0.0
        out = (level + self.phi * trend + self.alpha * innovation, self.phi * trend + self.beta * innovation)
        return out + self._rotate(tuple(season), first + self.gamma * innovation)

    def _predicted(self, a: Sequence[float], h: int) -> float:
        """What a state vector (level, trend, season...) predicts h steps ahead."""
        season = a[2 + (h - 1) % (len(a) - 2)] if len(a) > 2 else 0.0
        return a[0] + self._damping[h - 1] * a[1] + season

    def update(self, state: ExponentialSmoothingState, x):
        y, jacobian = self._value(x)
        now, plan = self._windows(x)
        season = state.seasonal[0] if state.seasonal else 0.0
        if not self._started(state):
            # the first value sets the level: l = y - s - b'z, so A_j starts with level z_j
            effects = tuple((z, 0.0) + (0.0,) * len(state.seasonal) for z in now) if self._learning else ()
            new = ExponentialSmoothingState(1, y - season - self._effect(now), state.trend,
                                            self._rotate(state.seasonal, season), 0, 0.0, 0.0, state.log_likelihood,
                                            state.comp, plan, effects, state.information, state.moment, state.shape,
                                            state.rate)
            return new, None, None
        base = state.level + self.phi * state.trend + season
        log_lik, comp = state.log_likelihood, state.comp
        if not self._learning:
            error = y - (base + self._effect(now))
            noise = self._noise(state)
            score = None if noise is None else log_density(error, noise[0], noise[1]) - jacobian
            level = require_finite(state.level + self.phi * state.trend + self.alpha * error, "level")
            trend = require_finite(self.phi * state.trend + self.beta * error, "trend")
            seasonal = self._rotate(state.seasonal, season + self.gamma * error)
            sse, sse_comp = neumaier_add(state.sse, state.sse_comp, error * error)
            if score is not None:
                log_lik, comp = neumaier_add(log_lik, comp, score)
            new = ExponentialSmoothingState(state.n + 1, level, trend, seasonal, state.errors + 1, sse, sse_comp,
                                            log_lik, comp, plan)
            return new, self._point(y) - self._point(base + self._effect(now)), score
        # learnt effects: regress the error of the run without regressors on u
        u = [z - self._predicted(a, 1) for z, a in zip_strict(now, state.effects)]
        theta, low = _posterior(state.information, state.moment, self.forgetting, self.ridge)
        raw = y - base
        fitted = 0.0
        for b, v in zip_strict(theta, u):
            fitted += b * v
        error = raw - fitted
        s = 1.0 + _quad(low, u)
        score = None
        if state.shape > 0.0 and state.rate > 0.0:
            score = log_density(error, math.sqrt(state.rate / state.shape * s),
                                2.0 * self.forgetting * state.shape) - jacobian
            log_lik, comp = neumaier_add(log_lik, comp, score)
        lam = self.forgetting
        information, moment = _accumulate(state.information, state.moment, u, raw, lam)
        level = require_finite(state.level + self.phi * state.trend + self.alpha * raw, "level")
        trend = require_finite(self.phi * state.trend + self.beta * raw, "trend")
        seasonal = self._rotate(state.seasonal, season + self.gamma * raw)
        effects = tuple(self._advance(a, v) for a, v in zip_strict(state.effects, u))
        new = ExponentialSmoothingState(
            state.n + 1, level, trend, seasonal, state.errors + 1, 0.0, 0.0, log_lik, comp, plan, effects,
            information, moment, lam * state.shape + 0.5,
            require_finite(lam * state.rate + error * error / (2.0 * s), "noise statistic"))
        return new, self._point(y) - self._point(base + fitted), score

    def predictive(self, state: ExponentialSmoothingState) -> Predictive | None:
        noise = self._noise(state) if self._started(state) else None
        if noise is None or (self.regressors and not state.plan):
            return None
        sigma, df = noise
        x0 = (state.level, state.trend, *state.seasonal)
        mean, scale = [], []
        if not self._learning:
            for h in range(1, self.horizon + 1):
                mean.append(self._predicted(x0, h) + self._effect([p[h - 1] for p in state.plan]))
            return Predictive(tuple(mean), tuple(sigma * f for f in self._factors), df, self.transform)
        theta, low = _posterior(state.information, state.moment, self.forgetting, self.ridge)
        for h in range(1, self.horizon + 1):
            u = [p[h - 1] - self._predicted(a, h) for p, a in zip_strict(state.plan, state.effects)]
            value = self._predicted(x0, h)
            for b, v in zip_strict(theta, u):
                value += b * v
            mean.append(value)
            f = self._factors[h - 1]
            scale.append(sigma * math.sqrt(f * f + _quad(low, u)))
        return Predictive(tuple(mean), tuple(scale), df, self.transform)

    def parameters(self, state: ExponentialSmoothingState) -> dict:
        noise = self._noise(state) if self._started(state) else None
        out: dict[str, Any] = {"level": state.level, "trend": state.trend, "seasonal": list(state.seasonal),
                               "sigma": None if noise is None else noise[0]}
        if self._learning:
            theta, _ = _posterior(state.information, state.moment, self.forgetting, self.ridge)
            # the states of the model itself are x~ - sum_j b_j A_j
            x = [state.level, state.trend, *state.seasonal]
            for b, a in zip_strict(theta, state.effects):
                x = [v - b * w for v, w in zip_strict(x, a)]
            out = {"level": x[0], "trend": x[1], "seasonal": x[2:], "sigma": out["sigma"], "coefficients": theta}
        elif self.effects is not None:
            out["coefficients"] = list(self.effects)
        return out

    def decode_state(self, data) -> ExponentialSmoothingState:
        state = decode_fields(ExponentialSmoothingState, data)
        require_state(len(state.seasonal) == len(self.initial_seasonal), "one seasonal effect per season")
        require_state(0 <= state.errors <= state.n and state.sse >= 0, "error statistics")
        require_state(state.n > 0 or (state.errors == 0 and state.sse == 0 and state.log_likelihood == 0),
                      "statistics before data")
        require_state(self.trend is not None or state.trend == 0, "a trend without a trend model")
        r = len(self.regressors)
        require_state(len(state.plan) == (r if state.n > 0 else 0)
                      and all(len(p) == self.horizon for p in state.plan), "one plan per regressor")
        if self._learning:
            width = 2 + len(self.initial_seasonal)
            require_state(len(state.effects) == r and all(len(a) == width for a in state.effects), "effects' sizes")
            _check_information(state.information, state.moment, r)
            require_state(state.shape >= 0 and state.rate >= 0 and state.sse == 0, "noise statistics")
        else:
            require_state(state.effects == () and state.information == () and state.moment == ()
                          and state.shape == state.rate == 0, "learnt statistics without learnt effects")
        return state


# ------------------------------------------------------------------------------------------ AR


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
    plan: tuple[tuple[float, ...], ...] = ()  # the regressors' planned values for 1..horizon steps ahead


class AutoRegressive(Forecaster):
    """AR(p) forecasts, with fixed coefficients or learnt by Bayesian regression on the lags.

    Give `coefficients` (phi_1..phi_p), `intercept`, `sigma` and, with
    `regressors`, their `effects` for fixed training.  Leave them out to
    learn them: `forgetting` lambda in (0, 1] discounts the past by lambda
    per step (1 keeps everything), and `ridge` is the prior precision of
    the coefficients relative to the noise, which is not discounted.  The
    coefficients used for each forecast are then the exponentially weighted
    ridge regression

        (sum_t lambda^(n+1-t) x_t x_t' + ridge I)^-1 sum_t lambda^(n+1-t) x_t y_t,

    x_t = (1, y_{t-1}, ..., y_{t-p}, z_t): the exact posterior mean when the
    data's information decays by lambda per step and the prior's does not.
    """

    name = "autoregressive"
    version = "1"

    def __init__(self, order: int, *, coefficients: Sequence[float] | None = None, intercept: float = 0.0,
                 sigma: float | None = None, regressors: Sequence[str] = (), effects: Sequence[float] | None = None,
                 forgetting: float = 1.0, ridge: float = 1e-6, transform: str | None = None, horizon: int = 1,
                 coverage: float = 0.9, input: str = "y"):
        if type(order) is not int or order < 1:
            raise ValueError("order must be a positive int")
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
        self.order = order
        self.regressors = _check_names(input, regressors)
        if coefficients is not None:
            phi = tuple(float(c) for c in coefficients)
            if len(phi) != order or not all(math.isfinite(c) for c in phi) or not math.isfinite(intercept):
                raise ValueError("need `order` finite coefficients and a finite intercept")
            if sigma is None or not 0.0 < sigma < math.inf:
                raise ValueError("fixed coefficients need a positive sigma")
            if forgetting != 1.0 or ridge != 1e-6:
                raise ValueError("forgetting and ridge are for learnt coefficients")
            beta = tuple(float(b) for b in (effects if effects is not None else ()))
            if len(beta) != len(self.regressors) or not all(math.isfinite(b) for b in beta):
                raise ValueError("fixed coefficients need one finite effect per regressor")
            self.phi: tuple[float, ...] | None = phi
            self.beta: tuple[float, ...] = beta
        else:
            if sigma is not None or intercept != 0.0 or effects is not None:
                raise ValueError("sigma, intercept and effects come with fixed coefficients")
            _check_learning(forgetting, ridge)
            self.phi, self.beta = None, ()
        self.intercept = float(intercept)
        self.sigma = None if sigma is None else float(sigma)
        self.forgetting, self.ridge, self.transform = float(forgetting), float(ridge), transform
        self.horizon, self.coverage, self.input = _check_horizon(horizon), _check_coverage(coverage), input

    def config(self) -> dict:
        return {"order": self.order, "coefficients": None if self.phi is None else list(self.phi),
                "intercept": self.intercept, "sigma": self.sigma, "regressors": list(self.regressors),
                "effects": None if self.phi is None else list(self.beta), "forgetting": self.forgetting,
                "ridge": self.ridge, "transform": self.transform, "horizon": self.horizon,
                "coverage": self.coverage, "input": self.input}

    def initial_state(self) -> AutoRegressiveState:
        if self.phi is not None:
            return AutoRegressiveState(0, (), (), (), 0.0, 0.0)
        k = self.order + 1 + len(self.regressors)
        return AutoRegressiveState(0, (), tuple((0.0,) * k for _ in range(k)), (0.0,) * k, 0.0, 0.0)

    def _point_ar(self, theta: Sequence[float], lags: Sequence[float], z: Sequence[float]) -> float:
        total = theta[0]
        for i in range(self.order):
            total += theta[i + 1] * lags[i]
        for j, v in enumerate(z):
            total += theta[self.order + 1 + j] * v
        return total

    def _theta(self) -> list[float]:
        assert self.phi is not None
        return [self.intercept, *self.phi, *self.beta]

    def update(self, state: AutoRegressiveState, x):
        y, jacobian = self._value(x)
        now, plan = self._windows(x)
        lags = (y, *state.lags)[: self.order]
        if len(state.lags) < self.order:
            return (AutoRegressiveState(state.n + 1, lags, state.information, state.moment, state.shape,
                                        state.rate, state.log_likelihood, state.comp, plan), None, None)
        log_lik, comp = state.log_likelihood, state.comp
        if self.phi is not None:
            assert self.sigma is not None
            point = self._point_ar(self._theta(), state.lags, now)
            fixed_score = log_density(y - point, self.sigma, None) - jacobian
            log_lik, comp = neumaier_add(log_lik, comp, fixed_score)
            new = AutoRegressiveState(state.n + 1, lags, (), (), 0.0, 0.0, log_lik, comp, plan)
            return new, self._point(y) - self._point(point), fixed_score
        design = (1.0, *state.lags, *now)
        theta, low = _posterior(state.information, state.moment, self.forgetting, self.ridge)
        point = self._point_ar(theta, state.lags, now)
        error = y - point
        s = 1.0 + _quad(low, design)
        score = None
        if state.shape > 0.0 and state.rate > 0.0:
            score = log_density(error, math.sqrt(state.rate / state.shape * s),
                                2.0 * self.forgetting * state.shape) - jacobian
            log_lik, comp = neumaier_add(log_lik, comp, score)
        lam = self.forgetting
        information, moment = _accumulate(state.information, state.moment, design, y, lam)
        rate = require_finite(lam * state.rate + error * error / (2.0 * s), "noise statistic")
        new = AutoRegressiveState(state.n + 1, lags, information, moment, lam * state.shape + 0.5, rate, log_lik,
                                  comp, plan)
        return new, self._point(y) - self._point(point), score

    def predictive(self, state: AutoRegressiveState) -> Predictive | None:
        if len(state.lags) < self.order or (self.regressors and not state.plan):
            return None
        if self.phi is not None:
            assert self.sigma is not None
            sigma2, df, theta, extra = self.sigma * self.sigma, None, self._theta(), 0.0
        elif state.shape > 0.0 and state.rate > 0.0:
            sigma2, df = state.rate / state.shape, 2.0 * self.forgetting * state.shape
            theta, low = _posterior(state.information, state.moment, self.forgetting, self.ridge)
            extra = _quad(low, (1.0, *state.lags, *(p[0] for p in state.plan)))
        else:
            return None
        history = list(state.lags)
        psi = [1.0]
        mean, scale = [], []
        total = 0.0
        for h in range(1, self.horizon + 1):
            value = self._point_ar(theta, history, [p[h - 1] for p in state.plan])
            mean.append(value)
            history = [value, *history[:-1]]
            total += psi[h - 1] * psi[h - 1]
            scale.append(math.sqrt(sigma2 * (extra + total)))
            nxt = 0.0
            for i in range(1, min(h, self.order) + 1):
                nxt += theta[i] * psi[h - i]
            psi.append(nxt)
        return Predictive(tuple(mean), tuple(scale), df, self.transform)

    def parameters(self, state: AutoRegressiveState) -> dict:
        """The coefficients and noise level the next forecast uses."""
        p = self.order
        if self.phi is not None:
            out: dict[str, Any] = {"intercept": self.intercept, "coefficients": list(self.phi), "sigma": self.sigma}
        elif len(state.lags) < self.order:
            out = {"intercept": None, "coefficients": None, "sigma": None}
        else:
            theta, _ = _posterior(state.information, state.moment, self.forgetting, self.ridge)
            sigma = math.sqrt(state.rate / state.shape) if state.shape > 0.0 and state.rate > 0.0 else None
            out = {"intercept": theta[0], "coefficients": theta[1:p + 1], "sigma": sigma}
            if self.regressors:
                out["effects"] = theta[p + 1:]
            return out
        if self.regressors:
            out["effects"] = list(self.beta) if self.phi is not None else None
        return out

    def decode_state(self, data) -> AutoRegressiveState:
        state = decode_fields(AutoRegressiveState, data)
        require_state(len(state.lags) == min(state.n, self.order), "one lag per observation, up to the order")
        r = len(self.regressors)
        require_state(len(state.plan) == (r if state.n > 0 else 0)
                      and all(len(p) == self.horizon for p in state.plan), "one plan per regressor")
        if self.phi is not None:
            require_state(state.information == () and state.moment == () and state.shape == state.rate == 0,
                          "learnt statistics with fixed coefficients")
        else:
            _check_information(state.information, state.moment, self.order + 1 + r)
            require_state(state.shape >= 0 and state.rate >= 0, "noise statistics")
        return state


# ------------------------------------------------------------------------------------- counts


@dataclass(frozen=True)
class PoissonSmoothingState:
    n: int
    shape: float  # the rate's gamma distribution after the last observation: Gamma(shape, rate)
    rate: float
    plan: tuple[float, ...] = ()  # planned exposures for 1..horizon steps ahead
    log_likelihood: float = 0.0
    comp: float = 0.0


class PoissonSmoothing(Forecaster):
    """Forecasts of counts: the Poisson local level model with a discounted gamma rate; see the module docstring.

    `discount` omega in (0, 1]; `exposure` names an optional input of
    horizon + 1 positive numbers (now, then planned), like a regressor.
    `prior` is the rate's Gamma(shape, rate) before any data; the default
    (0, 0) lets the data alone decide, so the first forecast comes after
    the first non-zero count.
    """

    name = "poisson_smoothing"
    version = "1"

    def __init__(self, discount: float, *, exposure: str | None = None, prior: Sequence[float] = (0.0, 0.0),
                 horizon: int = 1, coverage: float = 0.9, input: str = "y"):
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")
        a0, b0 = (float(v) for v in prior)
        if not (0.0 <= a0 < math.inf and 0.0 <= b0 < math.inf):
            raise ValueError("the prior's shape and rate must be finite and non-negative")
        if exposure is not None and (not isinstance(exposure, str) or exposure == input):
            raise ValueError("exposure must name an input different from the counts'")
        self.discount, self.exposure, self.prior = float(discount), exposure, (a0, b0)
        self.horizon, self.coverage, self.input = _check_horizon(horizon), _check_coverage(coverage), input

    def config(self) -> dict:
        return {"discount": self.discount, "exposure": self.exposure, "prior": list(self.prior),
                "horizon": self.horizon, "coverage": self.coverage, "input": self.input}

    def inputs(self):
        contracts = {self.input: InputContract(self.input, kind="integer", lower=0)}
        if self.exposure is not None:
            contracts[self.exposure] = InputContract(self.exposure, shape=(self.horizon + 1,), lower=0.0)
        return contracts

    def initial_state(self) -> PoissonSmoothingState:
        return PoissonSmoothingState(0, self.prior[0], self.prior[1])

    def _exposures(self, x) -> tuple[float, tuple[float, ...]]:
        if self.exposure is None:
            return 1.0, ()
        window = [float(v) for v in x[self.exposure]]
        if not all(v > 0.0 for v in window):
            raise ContractViolation(f"{self.name}: exposures must be positive, got {window!r}")
        return window[0], tuple(window[1:])

    def update(self, state: PoissonSmoothingState, x):
        y = x[self.input]
        e, plan = self._exposures(x)
        a, b = self.discount * state.shape, self.discount * state.rate
        log_lik, comp = state.log_likelihood, state.comp
        score = error = None
        if a > 0.0 and b > 0.0:
            score = negative_binomial_log_pmf(y, a, b, e)
            log_lik, comp = neumaier_add(log_lik, comp, score)
            error = y - e * a / b
        new = PoissonSmoothingState(state.n + 1, require_finite(a + y, "shape"), require_finite(b + e, "rate"),
                                    plan, log_lik, comp)
        return new, error, score

    def predictive(self, state: PoissonSmoothingState) -> Predictive | None:
        if not (state.shape > 0.0 and state.rate > 0.0) or (self.exposure is not None and not state.plan):
            return None
        mean, scale, size, rate, exposure = [], [], [], [], []
        factor = 1.0
        for h in range(1, self.horizon + 1):
            factor *= self.discount
            e = state.plan[h - 1] if self.exposure is not None else 1.0
            a, b = factor * state.shape, factor * state.rate
            m = e * state.shape / state.rate
            mean.append(m)
            scale.append(math.sqrt(m * (1.0 + e / b)))
            size.append(a)
            rate.append(b)
            exposure.append(e)
        return Predictive(tuple(mean), tuple(scale), None, None, tuple(size), tuple(rate), tuple(exposure))

    def parameters(self, state: PoissonSmoothingState) -> dict:
        level = state.shape / state.rate if state.shape > 0.0 and state.rate > 0.0 else None
        return {"level": level, "shape": state.shape, "rate": state.rate}

    def decode_state(self, data) -> PoissonSmoothingState:
        state = decode_fields(PoissonSmoothingState, data)
        require_state(state.shape >= 0 and state.rate >= 0, "the gamma distribution's parameters")
        require_state(len(state.plan) == (self.horizon if self.exposure is not None and state.n > 0 else 0),
                      "one planned exposure per horizon")
        require_state(state.n > 0 or state.log_likelihood == 0, "statistics before data")
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
    weighted mean of their point forecasts, the interval holds `coverage`
    of the mixture, and `scale` is the normal-equivalent one (the
    interval's half width over the normal quantile).

    Weights move only on observations that every model forecast.
    `averaged_smoothing` builds one over a grid of smoothing parameters,
    `averaged_counts` over a grid of discounts.  The models must be
    ExponentialSmoothing and AutoRegressive forecasters (with or without
    the log transform), or PoissonSmoothing ones, with the same horizon and
    inputs.
    """

    name = "model_averaging"
    version = "1"

    def __init__(self, models: Sequence[Forecaster], *, weights: Sequence[float] | None = None,
                 forgetting: float = 1.0, coverage: float = 0.9):
        models = tuple(models)
        if not models:
            raise ValueError("at least one model is needed")
        counts = all(isinstance(m, PoissonSmoothing) for m in models)
        for m in models:
            if not (counts or isinstance(m, (ExponentialSmoothing, AutoRegressive))):
                raise ValueError("models must be ExponentialSmoothing or AutoRegressive forecasters, "
                                 "or all PoissonSmoothing ones")
        first = models[0]
        if len({m.horizon for m in models}) != 1 or any(m.inputs() != first.inputs() for m in models):
            raise ValueError("the models need the same horizon and the same inputs")
        prior = tuple(float(w) for w in weights) if weights is not None else (1.0,) * len(models)
        if len(prior) != len(models) or not all(0.0 < w < math.inf for w in prior):
            raise ValueError("need one positive weight per model")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0, 1]")
        self.models, self.prior, self.forgetting = models, prior, float(forgetting)
        self.horizon, self.input, self.counts = first.horizon, first.input, counts
        self._inputs = first.inputs()
        self.coverage = _check_coverage(coverage)
        logs = [math.log(w) for w in prior]
        top = logsumexp(logs)
        self._log_prior = tuple(v - top for v in logs)

    def config(self) -> dict:
        return {"models": [m.identity() for m in self.models], "weights": list(self.prior),
                "forgetting": self.forgetting, "coverage": self.coverage}

    def inputs(self):
        return dict(self._inputs)

    def initial_state(self) -> ModelAveragingState:
        return ModelAveragingState(0, tuple(m.initial_state() for m in self.models), self._log_prior)

    def _flattened(self, log_weights: Sequence[float]) -> list[float]:
        logs = [self.forgetting * v for v in log_weights]
        top = logsumexp(logs)
        return [v - top for v in logs]

    def update(self, state: ModelAveragingState, x):
        states, scores = [], []
        for model, s in zip_strict(self.models, state.states):
            s, _, score = model.update(s, x)
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

    def _components(self, state: ModelAveragingState) -> list[tuple[float, Forecaster, Predictive]] | None:
        weights = self._flattened(state.log_weights)
        out = []
        for model, s, w in zip_strict(self.models, state.states, weights):
            pred = model.predictive(s)
            if pred is None:
                return None
            out.append((math.exp(w), model, pred))
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
        points = [model.present(p)[0] for _, model, p in comps]
        quantiles: dict[float | None, float] = {}
        for _, _, p in live:
            if not p.counts and p.df not in quantiles:
                quantiles[p.df] = self._quantile(p.df)
        mean, lower, upper = [], [], []
        for h in range(self.horizon):
            total = 0.0
            for (w, _, _), point in zip_strict(comps, points):
                total += w * point[h]
            mean.append(total)
            # the mixture's quantiles lie between the components' quantiles at the same levels
            ends = [self._ends(p, h, tail, quantiles) for _, _, p in live]
            lo = min(e[0] for e in ends)
            hi = max(e[1] for e in ends)
            if self.counts:
                lower.append(float(self._count_solve(live, h, tail, lo, hi)))
                upper.append(float(self._count_solve(live, h, 1.0 - tail, lo, hi)))
            else:
                lower.append(self._solve(live, h, tail, lo, hi))
                upper.append(self._solve(live, h, 1.0 - tail, lo, hi))
        return mean, lower, upper

    @staticmethod
    def _ends(p: Predictive, h: int, tail: float, quantiles: dict) -> tuple[float, float]:
        """A component's quantiles at `tail` and 1 - `tail`, in the data's units."""
        if p.counts:
            return float(count_quantile(p, h, tail)), float(count_quantile(p, h, 1.0 - tail))
        q = quantiles[p.df]
        lo, hi = p.mean[h] - q * p.scale[h], p.mean[h] + q * p.scale[h]
        if p.transform == "log":
            return math.exp(lo), math.exp(hi)
        return lo, hi

    @staticmethod
    def _count_solve(live, h: int, target: float, lo: float, hi: float) -> int:
        """The smallest count where the mixture CDF reaches `target`, between the components' quantiles."""
        def cdf(k: int) -> float:
            total = 0.0
            for w, _, p in live:
                total += w * p.cdf(h, float(k))
            return total

        a, b = int(lo) - 1, int(hi)  # cdf(a) < target <= cdf(b), up to the weight left out
        if cdf(b) < target:
            return b
        while b - a > 1:
            mid = (a + b) // 2
            if cdf(mid) < target:
                a = mid
            else:
                b = mid
        return b

    @staticmethod
    def _solve(live, h: int, target: float, lo: float, hi: float) -> float:
        """The point where the mixture CDF reaches `target` in [lo, hi]: Newton steps, bisecting when they leave."""
        x = 0.5 * (lo + hi)
        for _ in range(100):
            cdf, pdf = 0.0, 0.0
            for w, _, p in live:
                cdf += w * p.cdf(h, x)
                pdf += w * p.density(h, x)
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
        state, _, score = self.update(state, x)
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
                       phis: Sequence[float] = (0.9, 0.98), regressors: Sequence[str] = (),
                       transform: str | None = None, forgetting: float = 1.0, horizon: int = 1,
                       coverage: float = 0.9, input: str = "y") -> ModelAveraging:
    """Exponential smoothing that learns its smoothing parameters: model averaging over a grid.

    beta is given as a fraction of alpha and gamma as a fraction of
    1 - alpha (the usual admissible region beta < alpha, gamma < 1 - alpha);
    `phis` are used only with a damped trend.  Each model learns its own
    noise level and the effects of the regressors, so the averaging is over
    the smoothing parameters given the data.
    """
    models = []
    for a in alphas:
        for b in (betas if trend is not None else (0.0,)):
            for g in (gammas if period is not None else (0.0,)):
                for f in (phis if trend == "damped" else (1.0,)):
                    models.append(ExponentialSmoothing(a, beta=a * b, gamma=(1.0 - a) * g, phi=f, trend=trend,
                                                       period=period, regressors=regressors, transform=transform,
                                                       horizon=horizon, coverage=coverage, input=input))
    return ModelAveraging(models, forgetting=forgetting, coverage=coverage)


def averaged_counts(*, discounts: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995, 1.0),
                    exposure: str | None = None, forgetting: float = 1.0, horizon: int = 1, coverage: float = 0.9,
                    input: str = "y") -> ModelAveraging:
    """Count forecasts that learn how fast the rate moves: model averaging over a grid of discounts."""
    models = [PoissonSmoothing(d, exposure=exposure, horizon=horizon, coverage=coverage, input=input)
              for d in discounts]
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
