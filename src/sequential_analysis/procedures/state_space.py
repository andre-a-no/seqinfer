# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Linear state-space filters.

KalmanFilter is the exact filter of the linear Gaussian model

    z_t = F z_{t-1} + B u_t + w_t,   w_t ~ N(0, Q)       (state, n entries)
    y_t = H z_t + v_t,               v_t ~ N(0, R)       (observation, m entries)

with z_0 ~ N(m0, P0): after each observation the state is N(mean, covariance),
the exact conditional distribution of z_t given y_1..y_t, and
`log_likelihood` is log p(y_1..y_t).  The covariance is updated in Joseph
form, (I - K H) P (I - K H)' + K R K', which keeps it symmetric and positive
semi-definite in floating point.  Local linear trends, autoregressions,
constant-velocity tracking and regression with drifting coefficients are
all special cases; constructors for the first two are provided.

RobustKalmanFilter replaces the Gaussian observation noise by a
multivariate Student t with `df` degrees of freedom and scale R -- a
Gaussian whose precision is scaled by a Gamma(df/2, df/2) weight lambda.
The exact filter is then intractable; the variational Bayes update of
Piche, Sarkka and Hartikainen (2012) alternates a Kalman update with
noise covariance R / lambda and

    lambda = (df + m) / (df + E[(y - H z)' R^-1 (y - H z)]),

the expectation under the current state posterior, until lambda settles.
An observation far from its prediction gets a small weight and moves the
state little: outliers are discounted instead of dragging the filter.  As
df grows the filter becomes the Kalman filter.

Cost per step (PERFORMANCE.md): constant, O(n^3) in the state dimension; 0.1 ms
for two states, 0.16 ms for four, in Python (numpy is slower at these
sizes); the robust filter iterates the update, about 1 ms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import linalg as la
from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..errors import NumericalError
from ..numerics import neumaier_add, require_finite

_LOG_2PI = math.log(2.0 * math.pi)


def _check_covariance(a: la.Matrix, name: str, definite: bool) -> None:
    if not la.is_symmetric(a):
        raise ValueError(f"{name} must be symmetric")
    if definite:
        try:
            la.cholesky(a)
        except ValueError:
            raise ValueError(f"{name} must be positive definite") from None
    elif any(a[i][i] < 0 for i in range(len(a))):
        raise ValueError(f"{name} must have a non-negative diagonal")


@dataclass(frozen=True)
class KalmanFilterState:
    n: int
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    log_likelihood: float = 0.0
    comp: float = 0.0


@dataclass(frozen=True)
class KalmanFilterOutput:
    n: int
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    innovation: tuple[float, ...]  # y - H (predicted state)
    innovation_covariance: tuple[tuple[float, ...], ...]
    log_likelihood: float | None  # log p(y_1..y_n); None for the robust filter
    weight: float  # 1 for the Kalman filter; the robust filter's lambda


class KalmanFilter(Procedure):
    """Exact filter of a linear Gaussian state-space model; see the module docstring.

    Matrices are given as nested sequences (a number for 1x1).  With one
    observed quantity the input is a number, otherwise a list of m numbers;
    the same holds for the control input u (declared when `control` is given).
    """

    name = "kalman_filter"
    version = "1"
    guarantee = "model"

    def __init__(
        self,
        transition,
        observation,
        process_noise,
        observation_noise,
        initial_mean,
        initial_covariance,
        *,
        control=None,
        input: str = "y",
        control_input: str = "u",
    ):
        self.F = la.matrix(transition, None, None, "transition")
        n = len(self.F)
        if len(self.F[0]) != n:
            raise ValueError("transition must be square")
        self.H = la.matrix(observation, None, n, "observation")
        m = len(self.H)
        self.Q = la.matrix(process_noise, n, n, "process_noise")
        self.R = la.matrix(observation_noise, m, m, "observation_noise")
        self.m0 = la.vector(initial_mean, n, "initial_mean")
        self.P0 = la.matrix(initial_covariance, n, n, "initial_covariance")
        self.B = None if control is None else la.matrix(control, n, None, "control")
        _check_covariance(self.Q, "process_noise", definite=False)
        _check_covariance(self.P0, "initial_covariance", definite=False)
        _check_covariance(self.R, "observation_noise", definite=True)
        if input == control_input:
            raise ValueError("the observation and the control need different names")
        self.input, self.control_input = input, control_input
        self.n_state, self.n_obs = n, m
        self._r_inverse = la.inverse_spd(self.R)

    @classmethod
    def local_linear_trend(cls, level_noise: float, slope_noise: float, observation_noise: float, **kwargs):
        """Level and slope: level_t = level_{t-1} + slope_{t-1} + noise, slope a random walk, y = level + noise."""
        kwargs.setdefault("initial_mean", (0.0, 0.0))
        kwargs.setdefault("initial_covariance", ((1e6, 0.0), (0.0, 1e6)))
        return cls(((1.0, 1.0), (0.0, 1.0)), ((1.0, 0.0),), ((level_noise, 0.0), (0.0, slope_noise)),
                   observation_noise, **kwargs)

    @classmethod
    def autoregressive(cls, coefficients, process_noise: float, observation_noise: float, **kwargs):
        """AR(p) observed with noise: z_t = sum_i phi_i z_{t-i} + w_t, y_t = z_t + v_t (companion form)."""
        phi = la.vector(coefficients, None, "coefficients")
        p = len(phi)
        rows = [phi] + [tuple(1.0 if j == i else 0.0 for j in range(p)) for i in range(p - 1)]
        q = tuple(tuple(process_noise if i == j == 0 else 0.0 for j in range(p)) for i in range(p))
        kwargs.setdefault("initial_mean", (0.0,) * p)
        vague = tuple(tuple(1e6 if i == j else 0.0 for j in range(p)) for i in range(p))
        kwargs.setdefault("initial_covariance", vague)
        h = (tuple(1.0 if j == 0 else 0.0 for j in range(p)),)
        return cls(tuple(rows), h, q, observation_noise, **kwargs)

    def config(self) -> dict:
        return {
            "transition": [list(r) for r in self.F],
            "observation": [list(r) for r in self.H],
            "process_noise": [list(r) for r in self.Q],
            "observation_noise": [list(r) for r in self.R],
            "initial_mean": list(self.m0),
            "initial_covariance": [list(r) for r in self.P0],
            "control": None if self.B is None else [list(r) for r in self.B],
            "input": self.input,
            "control_input": self.control_input,
        }

    def inputs(self):
        def contract(name: str, size: int) -> InputContract:
            return InputContract(name) if size == 1 else InputContract(name, shape=(size,))

        contracts = {self.input: contract(self.input, self.n_obs)}
        if self.B is not None:
            contracts[self.control_input] = contract(self.control_input, len(self.B[0]))
        return contracts

    def initial_state(self) -> KalmanFilterState:
        return KalmanFilterState(0, self.m0, self.P0)

    def _predict(self, state: KalmanFilterState, x) -> tuple[la.Vector, la.Matrix]:
        mean = la.matvec(self.F, state.mean)
        if self.B is not None:
            mean = la.vadd(mean, la.matvec(self.B, la.vector(x[self.control_input], len(self.B[0]), "control")))
        return mean, la.add(la.quadratic(self.F, state.covariance), self.Q)

    def _update(self, mean: la.Vector, cov: la.Matrix, y: la.Vector, noise: la.Matrix):
        """Kalman update with observation noise covariance `noise`: (mean, cov, innovation, S, S^-1)."""
        innovation = la.vsub(y, la.matvec(self.H, mean))
        s = la.add(la.quadratic(self.H, cov), noise)
        try:
            s_inverse = la.inverse_spd(s)
        except ValueError:
            raise NumericalError(f"{self.name}: the innovation covariance is not positive definite") from None
        gain = la.matmul(la.matmul(cov, la.transpose(self.H)), s_inverse)
        new_mean = la.vadd(mean, la.matvec(gain, innovation))
        reduction = la.sub(la.identity(self.n_state), la.matmul(gain, self.H))
        new_cov = la.add(la.quadratic(reduction, cov), la.quadratic(gain, noise))  # Joseph form
        return new_mean, new_cov, innovation, s, s_inverse

    def step(self, state: KalmanFilterState, x, rng=None):
        y = la.vector(x[self.input], self.n_obs, "observation")
        mean, cov = self._predict(state, x)
        new_mean, new_cov, innovation, s, s_inverse = self._update(mean, cov, y, self.R)
        term = -0.5 * (self.n_obs * _LOG_2PI + la.log_det_spd(s) + la.dot(innovation, la.matvec(s_inverse, innovation)))
        total, comp = neumaier_add(state.log_likelihood, state.comp, term)
        n = state.n + 1
        log_lik = require_finite(total + comp, "log likelihood")
        new = KalmanFilterState(n, new_mean, new_cov, total, comp)
        return new, KalmanFilterOutput(n, new_mean, new_cov, innovation, s, log_lik, 1.0)

    def encode_state(self, state: KalmanFilterState):
        return {"n": state.n, "mean": list(state.mean), "covariance": [list(r) for r in state.covariance],
                "log_likelihood": state.log_likelihood, "comp": state.comp}

    def encode_output(self, output: KalmanFilterOutput):
        return {
            "n": output.n,
            "mean": list(output.mean),
            "covariance": [list(r) for r in output.covariance],
            "innovation": list(output.innovation),
            "innovation_covariance": [list(r) for r in output.innovation_covariance],
            "log_likelihood": output.log_likelihood,
            "weight": output.weight,
        }

    def decode_state(self, data) -> KalmanFilterState:
        state = decode_fields(KalmanFilterState, data)
        require_state(len(state.mean) == self.n_state, "state dimension")
        require_state(len(state.covariance) == self.n_state
                      and all(len(r) == self.n_state for r in state.covariance), "covariance dimension")
        require_state(la.is_symmetric(state.covariance), "covariance not symmetric")
        require_state(all(state.covariance[i][i] >= 0 for i in range(self.n_state)), "negative variance")
        return state


class RobustKalmanFilter(KalmanFilter):
    """Linear state-space filter with Student-t observation noise (variational Bayes); see the module docstring.

    `df` sets how heavy the tails are: 3 to 5 for data with occasional gross
    outliers.  `weight` in the output is lambda: near 1 for an ordinary
    observation, small for one the filter treats as an outlier.  The
    posterior is an approximation, so no likelihood is reported.
    """

    name = "robust_kalman_filter"
    version = "1"
    guarantee = "approximate"

    def __init__(self, *args, df: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not (math.isfinite(df) and df > 0):
            raise ValueError("df must be positive and finite")
        self.df = float(df)

    def config(self) -> dict:
        return {**super().config(), "df": self.df}

    def step(self, state: KalmanFilterState, x, rng=None):
        y = la.vector(x[self.input], self.n_obs, "observation")
        mean, cov = self._predict(state, x)
        weight = 1.0
        for _ in range(100):
            new_mean, new_cov, innovation, s, _ = self._update(mean, cov, y, la.scale(self.R, 1.0 / weight))
            residual = la.vsub(y, la.matvec(self.H, new_mean))
            expected = la.dot(residual, la.matvec(self._r_inverse, residual)) + la.trace(
                la.matmul(la.quadratic(self.H, new_cov), self._r_inverse))
            updated = (self.df + self.n_obs) / (self.df + expected)
            if abs(updated - weight) <= 1e-12 * weight:
                weight = updated
                break
            weight = updated
        new_mean, new_cov, innovation, s, _ = self._update(mean, cov, y, la.scale(self.R, 1.0 / weight))
        n = state.n + 1
        new = KalmanFilterState(n, new_mean, new_cov, 0.0, 0.0)
        return new, KalmanFilterOutput(n, new_mean, new_cov, innovation, s, None, weight)
