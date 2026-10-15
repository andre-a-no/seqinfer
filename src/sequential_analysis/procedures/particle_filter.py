# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Particle filter for any state-space model you can simulate and evaluate.

A model says how the hidden state starts (`initial`, the distribution of
z_0), how it moves (`transition`, a draw of z_t given z_{t-1}) and how
likely an observation is (`log_likelihood`, log p(y_t | z_t)).  Nothing
needs to be linear or Gaussian: stochastic volatility, heavy-tailed noise,
counts with a latent rate, saturation and thresholds.

ParticleFilter is the bootstrap filter (Gordon, Salmond and Smith 1993)
with adaptive systematic resampling: N particles are moved by the model,
weighted by the likelihood of y_t, and resampled when the effective sample
size N_eff = 1 / sum W_i^2 falls below `resample_threshold` N.  Weighted
averages estimate posterior expectations given y_1..y_t, with error of
order 1/sqrt(N); and the product of the averaged likelihoods,

    p_hat(y_1..t) = prod_s sum_i W_{s-1,i} p(y_s | z_s,i),

is an unbiased estimate of the likelihood p(y_1..y_t) (Del Moral 2004).
All randomness comes from the run's statistical random state, so a run is
reproducible and resumes bit for bit from a checkpoint.

Models provided: LinearGaussianModel (scalar, the Kalman filter's model:
use it to check a filter), StochasticVolatilityModel, and
StudentTLocalLevel (a random walk observed with heavy-tailed noise).
Write your own by subclassing StateSpaceModel.

Cost per step (PERFORMANCE.md): O(N); 20 000 particles take 2.7 ms with numpy and
35 ms in Python; the checkpoint holds every particle and its weight,
about 40 bytes each.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .. import backend as _backend
from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..errors import NumericalError
from ..numerics import logsumexp, neumaier_add, require_finite

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class StateSpaceModel(ABC):
    """What ParticleFilter needs from a model.  States are floats (dimension 1) or tuples of floats."""

    name: str = "model"
    version: str = "1"
    dimension: int = 1
    #: True if the model also implements initial_many, transition_many and log_likelihood_many
    #: (numpy arrays of particles), which the numpy backend of ParticleFilter needs.
    vectorized: bool = False

    @abstractmethod
    def config(self) -> dict: ...

    @abstractmethod
    def initial(self, rng) -> Any:
        """A draw of z_0."""

    @abstractmethod
    def transition(self, state, rng) -> Any:
        """A draw of z_t given z_{t-1} = state."""

    @abstractmethod
    def log_likelihood(self, state, y) -> float:
        """log p(y_t | z_t = state)."""

    def initial_many(self, rng, count: int) -> Any:
        """`count` draws of z_0 as a numpy array; `rng` is a backend.VectorRandom."""
        raise NotImplementedError

    def transition_many(self, states, rng) -> Any:
        """One draw of z_t for each z_{t-1} in the numpy array `states`."""
        raise NotImplementedError

    def log_likelihood_many(self, states, y) -> Any:
        """log p(y | z) for each z in the numpy array `states`."""
        raise NotImplementedError

    def contract(self, name: str) -> InputContract:
        return InputContract(name)

    def spec(self) -> dict:
        return {"name": self.name, "version": self.version, "config": self.config()}


def _normal_log_density(y: float, mean: float, sd: float) -> float:
    z = (y - mean) / sd
    return -0.5 * z * z - math.log(sd) - _LOG_SQRT_2PI


class LinearGaussianModel(StateSpaceModel):
    """z_t = phi z_{t-1} + N(0, q),  y_t = z_t + N(0, r),  z_0 ~ N(m0, p0)."""

    name = "linear_gaussian"

    def __init__(self, phi: float, q: float, r: float, m0: float = 0.0, p0: float = 1.0):
        if not (math.isfinite(phi) and math.isfinite(m0) and 0 <= q < math.inf and 0 < r < math.inf
                and 0 <= p0 < math.inf):
            raise ValueError("need finite phi and m0, q >= 0, r > 0, p0 >= 0")
        self.phi, self.q, self.r, self.m0, self.p0 = float(phi), float(q), float(r), float(m0), float(p0)

    def config(self) -> dict:
        return {"phi": self.phi, "q": self.q, "r": self.r, "m0": self.m0, "p0": self.p0}

    def initial(self, rng) -> float:
        return float(rng.normal(self.m0, math.sqrt(self.p0)))

    def transition(self, state, rng) -> float:
        return float(rng.normal(self.phi * state, math.sqrt(self.q)))

    def log_likelihood(self, state, y) -> float:
        return _normal_log_density(y, state, math.sqrt(self.r))

    vectorized = True

    def initial_many(self, rng, count):
        return rng.normal(count, self.m0, math.sqrt(self.p0))

    def transition_many(self, states, rng):
        return rng.normal(len(states), self.phi * states, math.sqrt(self.q))

    def log_likelihood_many(self, states, y):
        sd = math.sqrt(self.r)
        z = (y - states) / sd
        return -0.5 * z * z - math.log(sd) - _LOG_SQRT_2PI


class StochasticVolatilityModel(StateSpaceModel):
    """Log-variance h_t = mu + phi (h_{t-1} - mu) + sigma eta_t; return y_t = exp(h_t / 2) eps_t.

    h_0 is drawn from the stationary distribution N(mu, sigma^2 / (1 - phi^2)).
    """

    name = "stochastic_volatility"

    def __init__(self, mu: float, phi: float, sigma: float):
        if not (math.isfinite(mu) and -1 < phi < 1 and 0 < sigma < math.inf):
            raise ValueError("need a finite mu, |phi| < 1 and sigma > 0")
        self.mu, self.phi, self.sigma = float(mu), float(phi), float(sigma)

    def config(self) -> dict:
        return {"mu": self.mu, "phi": self.phi, "sigma": self.sigma}

    def initial(self, rng) -> float:
        return float(rng.normal(self.mu, self.sigma / math.sqrt(1.0 - self.phi**2)))

    def transition(self, state, rng) -> float:
        return float(rng.normal(self.mu + self.phi * (state - self.mu), self.sigma))

    def log_likelihood(self, state, y) -> float:
        return _normal_log_density(y, 0.0, math.exp(0.5 * state))

    vectorized = True

    def initial_many(self, rng, count):
        return rng.normal(count, self.mu, self.sigma / math.sqrt(1.0 - self.phi**2))

    def transition_many(self, states, rng):
        return rng.normal(len(states), self.mu + self.phi * (states - self.mu), self.sigma)

    def log_likelihood_many(self, states, y):
        np = _backend.np
        sd = np.exp(0.5 * states)
        z = y / sd
        return -0.5 * z * z - np.log(sd) - _LOG_SQRT_2PI


class StudentTLocalLevel(StateSpaceModel):
    """z_t = z_{t-1} + N(0, q),  y_t = z_t + scale * t_df,  z_0 ~ N(m0, p0): a level observed with outliers."""

    name = "student_t_local_level"

    def __init__(self, q: float, scale: float, df: float, m0: float = 0.0, p0: float = 1.0):
        if not (0 <= q < math.inf and 0 < scale < math.inf and 0 < df < math.inf and math.isfinite(m0)
                and 0 <= p0 < math.inf):
            raise ValueError("need q >= 0, positive scale and df, finite m0, p0 >= 0")
        self.q, self.scale, self.df, self.m0, self.p0 = float(q), float(scale), float(df), float(m0), float(p0)
        self._const = (math.lgamma(0.5 * (self.df + 1.0)) - math.lgamma(0.5 * self.df)
                       - 0.5 * math.log(self.df * math.pi) - math.log(self.scale))

    def config(self) -> dict:
        return {"q": self.q, "scale": self.scale, "df": self.df, "m0": self.m0, "p0": self.p0}

    def initial(self, rng) -> float:
        return float(rng.normal(self.m0, math.sqrt(self.p0)))

    def transition(self, state, rng) -> float:
        return float(rng.normal(state, math.sqrt(self.q)))

    def log_likelihood(self, state, y) -> float:
        z = (y - state) / self.scale
        return self._const - 0.5 * (self.df + 1.0) * math.log1p(z * z / self.df)

    vectorized = True

    def initial_many(self, rng, count):
        return rng.normal(count, self.m0, math.sqrt(self.p0))

    def transition_many(self, states, rng):
        return rng.normal(len(states), states, math.sqrt(self.q))

    def log_likelihood_many(self, states, y):
        z = (y - states) / self.scale
        return self._const - 0.5 * (self.df + 1.0) * _backend.np.log1p(z * z / self.df)


@dataclass(frozen=True)
class ParticleFilterState:
    n: int = 0
    particles: tuple[Any, ...] = ()
    log_weights: tuple[float, ...] = ()  # normalized: logsumexp = 0
    log_likelihood: float = 0.0
    comp: float = 0.0


@dataclass(frozen=True)
class ParticleFilterOutput:
    n: int
    mean: Any  # float, or a tuple for a vector state
    variance: Any
    ess: float  # effective sample size after weighting, before any resampling
    log_likelihood: float  # log of the unbiased likelihood estimate
    resampled: bool


class ParticleFilter(Procedure):
    """Bootstrap particle filter with adaptive systematic resampling; see the module docstring."""

    name = "particle_filter"
    version = "1"
    randomized = True

    def __init__(self, model: StateSpaceModel, particles: int = 500, *, resample_threshold: float = 0.5,
                 backend: str = "auto", input: str = "y"):
        if not isinstance(model, StateSpaceModel):
            raise TypeError("model must be a StateSpaceModel")
        if type(particles) is not int or particles < 2:
            raise ValueError("particles must be an int, at least 2")
        if not 0 <= resample_threshold <= 1:
            raise ValueError("resample_threshold must lie in [0, 1] (1 resamples at every step)")
        self.model, self.size, self.threshold, self.input = model, particles, float(resample_threshold), input
        self.vector = model.dimension > 1
        vectorizable = model.vectorized and not self.vector
        if backend == "numpy" and not vectorizable:
            raise ValueError(f'backend="numpy" needs a scalar model with vectorized methods; {model.name} has none')
        resolved = _backend.resolve(backend)
        self.backend = resolved if vectorizable else "python"

    def config(self) -> dict:
        return {"model": self.model.spec(), "particles": self.size, "resample_threshold": self.threshold,
                "backend": self.backend, "input": self.input}

    def inputs(self):
        return {self.input: self.model.contract(self.input)}

    def initial_state(self) -> ParticleFilterState:
        return ParticleFilterState()

    def _as_state(self, value) -> Any:
        if self.vector:
            state = tuple(float(v) for v in value)
            if len(state) != self.model.dimension:
                raise NumericalError(f"{self.model.name}: a state must have {self.model.dimension} entries")
            return state
        return float(value)

    def _moments(self, particles, weights) -> tuple[Any, Any]:
        def moments(values) -> tuple[float, float]:
            mean = 0.0
            for w, v in zip(weights, values, strict=True):
                mean += w * v
            var = 0.0
            for w, v in zip(weights, values, strict=True):
                var += w * (v - mean) ** 2
            return mean, var

        if not self.vector:
            return moments(particles)
        pairs = [moments([p[d] for p in particles]) for d in range(self.model.dimension)]
        return tuple(m for m, _ in pairs), tuple(v for _, v in pairs)

    def step(self, state: ParticleFilterState, x, rng):
        if self.backend == "numpy":
            return self._step_numpy(state, x[self.input], rng)
        y = x[self.input]
        n_part = self.size
        if state.n == 0:
            previous = [self._as_state(self.model.initial(rng)) for _ in range(n_part)]
            log_w = [-math.log(n_part)] * n_part
        else:
            previous, log_w = list(state.particles), list(state.log_weights)
        moved = [self._as_state(self.model.transition(p, rng)) for p in previous]
        log_w = [w + self.model.log_likelihood(p, y) for w, p in zip(log_w, moved, strict=True)]
        increment = logsumexp(log_w)  # log sum_i W_i p(y | z_i): the weights were normalized
        if not math.isfinite(increment):
            raise NumericalError(f"{self.name}: every particle gives the observation zero likelihood")
        log_w = [w - increment for w in log_w]
        weights = [math.exp(w) for w in log_w]
        square = 0.0
        for w in weights:
            square += w * w
        ess = 1.0 / square
        mean, variance = self._moments(moved, weights)
        resampled = ess < self.threshold * n_part
        if resampled:  # systematic: one uniform draw
            u = rng.random() / n_part
            chosen, i, cum = [], 0, weights[0]
            for j in range(n_part):
                target = u + j / n_part
                while cum < target and i < n_part - 1:
                    i += 1
                    cum += weights[i]
                chosen.append(moved[i])
            moved, log_w = chosen, [-math.log(n_part)] * n_part
        total, comp = neumaier_add(state.log_likelihood, state.comp, increment)
        n = state.n + 1
        new = ParticleFilterState(n, tuple(moved), tuple(log_w), total, comp)
        out = ParticleFilterOutput(n, mean, variance, ess, require_finite(total + comp, "log likelihood"), resampled)
        return new, out

    def _step_numpy(self, state: ParticleFilterState, y, rng):
        np, draws, n_part = _backend.np, _backend.VectorRandom(rng), self.size
        if state.n == 0:
            previous = np.asarray(self.model.initial_many(draws, n_part), dtype=float)
            log_w = np.full(n_part, -math.log(n_part))
        else:
            previous, log_w = np.array(state.particles), np.array(state.log_weights)
        moved = np.asarray(self.model.transition_many(previous, draws), dtype=float)
        log_w = log_w + self.model.log_likelihood_many(moved, y)
        top = log_w.max()
        if not np.isfinite(top):
            raise NumericalError(f"{self.name}: every particle gives the observation zero likelihood")
        increment = float(top + math.log(np.exp(log_w - top).sum()))
        log_w = log_w - increment
        weights = np.exp(log_w)
        ess = float(1.0 / np.dot(weights, weights))
        mean = float(np.dot(weights, moved))
        variance = float(np.dot(weights, (moved - mean) ** 2))
        resampled = ess < self.threshold * n_part
        if resampled:  # systematic, from the same single uniform as the Python backend
            u = draws.random(1)[0] / n_part
            index = np.searchsorted(np.cumsum(weights), u + np.arange(n_part) / n_part, side="left")
            moved, log_w = moved[np.minimum(index, n_part - 1)], np.full(n_part, -math.log(n_part))
        if not (np.all(np.isfinite(moved)) and math.isfinite(increment)):
            raise NumericalError(f"{self.name}: particles or weights are not finite")
        total, comp = neumaier_add(state.log_likelihood, state.comp, increment)
        n = state.n + 1
        new = ParticleFilterState(n, tuple(moved.tolist()), tuple(log_w.tolist()), total, comp)
        out = ParticleFilterOutput(n, mean, variance, ess, require_finite(total + comp, "log likelihood"), resampled)
        return new, out

    def encode_state(self, state: ParticleFilterState):
        particles = [list(p) for p in state.particles] if self.vector else list(state.particles)
        return {"n": state.n, "particles": particles, "log_weights": list(state.log_weights),
                "log_likelihood": state.log_likelihood, "comp": state.comp}

    def encode_output(self, output: ParticleFilterOutput):
        def plain(v):
            return list(v) if isinstance(v, tuple) else v

        return {"n": output.n, "mean": plain(output.mean), "variance": plain(output.variance), "ess": output.ess,
                "log_likelihood": output.log_likelihood, "resampled": output.resampled}

    def decode_state(self, data) -> ParticleFilterState:
        if not isinstance(data, dict) or not isinstance(data.get("particles"), list):
            raise TypeError("particle filter state needs a list of particles")
        raw = data["particles"]
        particles: tuple[Any, ...]
        if self.vector:
            particles = tuple(tuple(float(v) for v in p) for p in raw)
            require_state(all(len(p) == self.model.dimension and all(math.isfinite(v) for v in p)
                              for p in particles), "particles of the wrong dimension")
        else:
            require_state(all(type(p) in (int, float) and math.isfinite(p) for p in raw), "particles must be numbers")
            particles = tuple(float(p) for p in raw)
        state = decode_fields(ParticleFilterState, {**data, "particles": []})
        state = ParticleFilterState(state.n, particles, state.log_weights, state.log_likelihood, state.comp)
        expected = self.size if state.n else 0
        require_state(len(particles) == len(state.log_weights) == expected, f"{expected} particles and weights")
        require_state(state.n == 0 or abs(logsumexp(state.log_weights)) < 1e-9, "weights must be normalized")
        return state
