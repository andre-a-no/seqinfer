# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Filtering in a hidden Markov model: which regime is the process in now?

A hidden state z_t in {0, ..., K-1} moves by a Markov chain,
`P(z_t = k | z_{t-1} = j) = A[j][k]`, starting from `P(z_1 = k) = initial[k]`;
the observation y_t depends only on z_t.  The forward algorithm gives the
exact filtering distribution

    P(z_t = k | y_1..y_t)  proportional to  e_k(y_t) sum_j P(z_{t-1} = j | y_1..y_{t-1}) A[j][k],

and the normalising constants multiply to the likelihood p(y_1..y_t).  The
emission densities are combined in logarithms, so observations far in the
tails do not underflow.

Emission models: CategoricalEmission (a symbol 0..M-1), GaussianEmission
(a mean and standard deviation per state), PoissonEmission (a rate per
state), MultivariateGaussianEmission (a mean vector and covariance matrix
per state, for a vector observation), StudentTEmission (a location, scale
and degrees of freedom per state: Gaussian-like, robust to outliers).
`emission_from_spec` rebuilds any of them from its `spec()`.  Typical
uses: regime detection (calm/volatile), a machine's hidden condition, a
patient's disease stage.

Cost per step (PERFORMANCE.md): O(K^2) for K states, 6 µs for three; a constant
state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import linalg as la
from .._compat import zip_strict
from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..distributions import _lgamma_ratio
from ..errors import NumericalError
from ..numerics import neumaier_add, require_finite

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _distribution(values, size: int | None, name: str) -> tuple[float, ...]:
    p = tuple(float(v) for v in values)
    if size is not None and len(p) != size:
        raise ValueError(f"{name} must have {size} entries")
    if not p or any(not (math.isfinite(v) and v >= 0) for v in p) or abs(math.fsum(p) - 1.0) > 1e-12:
        raise ValueError(f"{name} must be a probability distribution (non-negative, summing to 1)")
    return p


class CategoricalEmission:
    """y in {0, ..., M-1} with `P(y = s | z = k) = probabilities[k][s]`."""

    def __init__(self, probabilities):
        self.p = tuple(_distribution(row, None, "emission row") for row in probabilities)
        if len({len(r) for r in self.p}) != 1:
            raise ValueError("every state needs a distribution over the same symbols")
        self.states = len(self.p)

    def spec(self) -> dict:
        return {"model": "categorical", "probabilities": [list(r) for r in self.p]}

    def contract(self, name: str) -> InputContract:
        return InputContract(name, kind="integer", lower=0, upper=len(self.p[0]) - 1)

    def log_density(self, k: int, y) -> float:
        p = self.p[k][int(y)]
        return math.log(p) if p > 0 else -math.inf

    # complete-data sufficient statistics, for EM (sequential_analysis.hmm_fit, OnlineHMM)
    def statistics(self, k: int, y) -> tuple[float, ...]:
        """g(y): the indicator of each symbol."""
        out = [0.0] * len(self.p[0])
        out[int(y)] = 1.0
        return tuple(out)

    def from_statistics(self, totals, min_scale: float = 0.0) -> CategoricalEmission:
        """Maximum-likelihood update from totals[k] = (sum of weights, weighted sums of g); empty states keep theirs.

        `min_scale` is a pseudo-count added to every symbol (0: plain maximum likelihood).
        """
        rows = []
        for k, (w, *counts) in enumerate(totals):
            if not w > 1e-300:
                rows.append(self.p[k])
                continue
            total = w + min_scale * len(counts)
            row = [(c + min_scale) / total for c in counts]
            s = math.fsum(row)
            rows.append(tuple(v / s for v in row))
        return CategoricalEmission(rows)


class GaussianEmission:
    """y | z = k ~ N(means[k], sds[k]^2)."""

    def __init__(self, means, sds):
        self.means = tuple(float(m) for m in means)
        self.sds = tuple(float(s) for s in sds)
        if len(self.means) != len(self.sds) or not self.means:
            raise ValueError("one mean and one standard deviation per state")
        if not all(math.isfinite(m) for m in self.means) or not all(0 < s < math.inf for s in self.sds):
            raise ValueError("means must be finite and standard deviations positive")
        self.states = len(self.means)

    def spec(self) -> dict:
        return {"model": "gaussian", "means": list(self.means), "sds": list(self.sds)}

    def contract(self, name: str) -> InputContract:
        return InputContract(name)

    def log_density(self, k: int, y) -> float:
        z = (y - self.means[k]) / self.sds[k]
        return float(-0.5 * z * z - math.log(self.sds[k]) - _LOG_SQRT_2PI)

    def statistics(self, k: int, y) -> tuple[float, ...]:
        """g(y) = (y, y^2)."""
        return (float(y), float(y) * float(y))

    def from_statistics(self, totals, min_scale: float = 1e-6) -> GaussianEmission:
        """Maximum-likelihood means and standard deviations (at least `min_scale`); empty states keep theirs."""
        means, sds = [], []
        for k, (w, s1, s2) in enumerate(totals):
            if not w > 1e-300:
                means.append(self.means[k])
                sds.append(self.sds[k])
                continue
            m = s1 / w
            means.append(m)
            sds.append(max(math.sqrt(max(s2 / w - m * m, 0.0)), min_scale))
        return GaussianEmission(means, sds)


class PoissonEmission:
    """y | z = k ~ Poisson(rates[k])."""

    def __init__(self, rates):
        self.rates = tuple(float(r) for r in rates)
        if not self.rates or not all(0 < r < math.inf for r in self.rates):
            raise ValueError("rates must be positive and finite")
        self.states = len(self.rates)

    def spec(self) -> dict:
        return {"model": "poisson", "rates": list(self.rates)}

    def contract(self, name: str) -> InputContract:
        return InputContract(name, kind="integer", lower=0)

    def log_density(self, k: int, y) -> float:
        return float(y * math.log(self.rates[k]) - self.rates[k] - math.lgamma(y + 1.0))

    def statistics(self, k: int, y) -> tuple[float, ...]:
        """g(y) = (y,)."""
        return (float(y),)

    def from_statistics(self, totals, min_scale: float = 1e-6) -> PoissonEmission:
        """Maximum-likelihood rates (at least `min_scale`); empty states keep theirs."""
        return PoissonEmission([max(s1 / w, min_scale) if w > 1e-300 else self.rates[k]
                                for k, (w, s1) in enumerate(totals)])


#: A covariance whose Cholesky pivot falls below this fraction of its diagonal entry is refused as numerically
#: singular.  The ratio is 1 / (Sigma_ii (Sigma^-1)_ii), unchanged by rescaling the coordinates, and at least
#: 1 / cond(Sigma): the bound refuses correlations within about 5e-13 of +-1.
_MIN_PIVOT = 1e-12


def _covariance_factor(cov: la.Matrix, name: str) -> la.Matrix:
    """The Cholesky factor of a symmetric positive definite covariance; ValueError otherwise."""
    if not la.is_symmetric(cov):
        raise ValueError(f"{name} must be symmetric")
    try:
        low = la.cholesky(cov)
    except ValueError:
        raise ValueError(f"{name} must be positive definite") from None
    for i in range(len(cov)):
        if not low[i][i] * low[i][i] > _MIN_PIVOT * cov[i][i]:
            raise ValueError(f"{name} is numerically singular (a correlation within about 1e-12 of +-1)")
    return low


def _rows(value, name: str) -> list:
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(r, (list, tuple)) for r in value):
        raise ValueError(f"{name} must be a non-empty list of lists")
    return list(value)


class MultivariateGaussianEmission:
    """y | z = k ~ N_d(means[k], covariances[k]), for a vector observation y of length d.

    Each covariance must be symmetric (exactly) and positive definite; its
    Cholesky factor L (L L' = covariance) is computed once, and

        log e_k(y) = -|L^-1 (y - mean)|^2 / 2 - sum_i log L_ii - d log(2 pi) / 2,

    by forward substitution, every sum left to right.  With d = 1 this is
    GaussianEmission with sd = sqrt(variance).
    """

    def __init__(self, means, covariances):
        self.means = tuple(la.vector(m, None, "mean") for m in _rows(means, "means"))
        self.dimension = d = len(self.means[0])
        if any(len(m) != d for m in self.means):
            raise ValueError("every mean must have the same length")
        covs = _rows(covariances, "covariances")
        if len(covs) != len(self.means):
            raise ValueError("one mean and one covariance per state")
        self.covariances = tuple(la.matrix(_rows(c, "covariance"), d, d, "covariance") for c in covs)
        self._low = tuple(_covariance_factor(c, f"covariance {k}") for k, c in enumerate(self.covariances))
        half_log_dets = []
        for low in self._low:
            total = 0.0
            for i in range(d):
                total += math.log(low[i][i])
            half_log_dets.append(total)
        self._half_log_det = tuple(half_log_dets)
        self._log_norm = d * _LOG_SQRT_2PI
        self.states = len(self.means)

    def spec(self) -> dict:
        return {"model": "multivariate_gaussian", "means": [list(m) for m in self.means],
                "covariances": [[list(r) for r in c] for c in self.covariances]}

    def contract(self, name: str) -> InputContract:
        return InputContract(name, shape=(self.dimension,))

    def log_density(self, k: int, y) -> float:
        d, mean, low = self.dimension, self.means[k], self._low[k]
        if len(y) != d:
            raise ValueError(f"expected an observation of length {d}, got {len(y)}")
        v = [0.0] * d
        q = 0.0
        for i in range(d):
            s = y[i] - mean[i]
            for j in range(i):
                s -= low[i][j] * v[j]
            v[i] = s / low[i][i]
            q += v[i] * v[i]
        return float(-0.5 * q - self._half_log_det[k] - self._log_norm)

    def statistics(self, k: int, y) -> tuple[float, ...]:
        """g(y) = (y_1, ..., y_d, then y_i y_j for i <= j, row by row: y_1 y_1, y_1 y_2, ..., y_1 y_d, y_2 y_2, ...)."""
        d = self.dimension
        if len(y) != d:
            raise ValueError(f"expected an observation of length {d}, got {len(y)}")
        out = [float(v) for v in y]
        for i in range(d):
            for j in range(i, d):
                out.append(float(y[i]) * float(y[j]))
        return tuple(out)

    def from_statistics(self, totals, min_scale: float = 1e-6) -> MultivariateGaussianEmission:
        """Maximum-likelihood means and covariances; empty states keep theirs.

        The covariance is the weighted sum of squares about the mean over
        the weight, plus `min_scale`^2 on the diagonal (a ridge: every
        variance at least min_scale^2, and a covariance of collinear data
        positive definite).  A state whose updated covariance is still not
        positive definite (by the test of the constructor) keeps its old
        parameters too.
        """
        d = self.dimension
        size = d + d * (d + 1) // 2
        means, covs = [], []
        for k, row in enumerate(totals):
            if len(row) != size + 1:
                raise ValueError(f"totals of state {k} must have {size + 1} entries")
            w = row[0]
            if not w > 1e-300:
                means.append(self.means[k])
                covs.append(self.covariances[k])
                continue
            m = [row[1 + i] / w for i in range(d)]
            cov = [[0.0] * d for _ in range(d)]
            index = 1 + d
            for i in range(d):
                for j in range(i, d):
                    c = row[index] / w - m[i] * m[j]
                    if i == j:
                        c += min_scale * min_scale
                    cov[i][j] = cov[j][i] = c
                    index += 1
            matrix = tuple(tuple(r) for r in cov)
            try:
                if not all(math.isfinite(v) for v in m):
                    raise ValueError("non-finite mean")
                _covariance_factor(la.matrix(matrix, d, d, "covariance"), "covariance")
            except ValueError:
                means.append(self.means[k])
                covs.append(self.covariances[k])
                continue
            means.append(tuple(m))
            covs.append(matrix)
        return MultivariateGaussianEmission(means, covs)


def _per_state(values, states: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return (float(values),) * states
    out = tuple(float(v) for v in values)
    if len(out) != states:
        raise ValueError(f"{name} must have one entry per state (or be one number)")
    return out


def _log_abs_difference(a: float, b: float) -> float:
    """log |a - b| for finite a != b, also when a - b overflows."""
    d = a - b
    if math.isfinite(d):
        return math.log(abs(d))
    return math.log(abs(0.5 * a - 0.5 * b)) + math.log(2.0)


class StudentTEmission:
    """y | z = k ~ means[k] + scales[k] T(df[k]): location-scale Student t, robust to outliers.

        log e_k(y) = log Gamma((df+1)/2) - log Gamma(df/2) - log(df pi)/2 - log scale
                     - (df+1)/2 log1p(z^2 / df),       z = (y - mean) / scale,

    with the ratio of Gamma functions evaluated without cancellation for
    large df, and log1p(z^2/df) as 2 (log|y - mean| - log scale - log(df)/2)
    when z^2/df exceeds 1e300, so any finite outlier has a finite
    log-density.  As df grows the
    density tends to GaussianEmission(means, scales).  `df` is one number
    for every state or one per state; it is held fixed by `from_statistics`
    (not estimated).
    """

    def __init__(self, means, scales, df):
        self.means = tuple(float(m) for m in means)
        k = len(self.means)
        self.scales = _per_state(scales, k, "scales")
        self.df = _per_state(df, k, "df")
        if not self.means:
            raise ValueError("one mean, scale and df per state")
        if not all(math.isfinite(m) for m in self.means) or not all(0 < s < math.inf for s in self.scales):
            raise ValueError("means must be finite and scales positive")
        if not all(0 < v < math.inf for v in self.df):
            raise ValueError("degrees of freedom must be positive and finite")
        self._const = tuple(_lgamma_ratio(0.5 * v, 0.5) - 0.5 * (math.log(v) + math.log(math.pi)) - math.log(s)
                            for v, s in zip_strict(self.df, self.scales))
        self._root_df = tuple(math.sqrt(v) for v in self.df)
        self._log_width = tuple(math.log(s) + 0.5 * math.log(v) for v, s in zip_strict(self.df, self.scales))
        self.states = k

    def spec(self) -> dict:
        return {"model": "student_t", "means": list(self.means), "scales": list(self.scales), "df": list(self.df)}

    def contract(self, name: str) -> InputContract:
        return InputContract(name)

    def log_density(self, k: int, y) -> float:
        r = abs((y - self.means[k]) / self.scales[k]) / self._root_df[k]
        if r < 1e150:
            log1p_r2 = math.log1p(r * r)
        else:  # log1p(r^2) = 2 log r to double precision; in logarithms, as r itself may overflow
            log1p_r2 = 2.0 * (_log_abs_difference(y, self.means[k]) - self._log_width[k])
        return float(self._const[k] - 0.5 * (self.df[k] + 1.0) * log1p_r2)

    def weight(self, k: int, y) -> float:
        """E[precision | y, z = k] = (df + 1) / (df + z^2), the weight of y in the M-step (0 for a far outlier)."""
        z = (y - self.means[k]) / self.scales[k]
        return float((self.df[k] + 1.0) / (self.df[k] + z * z))

    def statistics(self, k: int, y) -> tuple[float, ...]:
        """g_k(y) = (u, u y, u y^2) with u = weight(k, y), at the current parameters of state k (the ECM E-step).

        y | z = k is N(mean, scale^2 / tau) with tau ~ Gamma(df/2, rate df/2);
        u = E[tau | y].  Given u, the complete-data likelihood is Gaussian
        with weights u, which `from_statistics` maximises.
        """
        u = self.weight(k, y)
        y = float(y)
        return (u, u * y, u * y * y)

    def from_statistics(self, totals, min_scale: float = 1e-6) -> StudentTEmission:
        """The ECM M-step with df fixed (Liu & Rubin 1995; McLachlan & Peel 2000, section 7.5).

        totals[k] = (w, S_u, S_uy, S_uy2), w the sum of the posterior
        weights of state k and S_* the posterior-weighted sums of
        `statistics(k, y)`:

            mean = S_uy / S_u,  scale^2 = (S_uy2 - 2 mean S_uy + mean^2 S_u) / w,

        the scale at least `min_scale`.  With the u of the current
        parameters this is an exact EM step for (mean, scale): the
        likelihood never decreases.  Empty states (or states whose
        observations all have weight u = 0) keep their parameters.
        """
        means, scales = [], []
        for k, (w, su, suy, suy2) in enumerate(totals):
            if not (w > 1e-300 and su > 1e-300):
                means.append(self.means[k])
                scales.append(self.scales[k])
                continue
            m = suy / su
            v = (suy2 - 2.0 * m * suy + m * m * su) / w
            means.append(m)
            scales.append(max(math.sqrt(max(v, 0.0)), min_scale))
        return StudentTEmission(means, scales, self.df)


def emission_from_spec(spec) -> CategoricalEmission | GaussianEmission | PoissonEmission | \
        MultivariateGaussianEmission | StudentTEmission:
    """The emission model whose `spec()` is `spec` (mirrors emissionFromSpec of the TypeScript executor)."""
    if not isinstance(spec, dict):
        raise ValueError("an emission spec must be a dict")
    model = spec.get("model")
    try:
        if model == "categorical":
            return CategoricalEmission(spec["probabilities"])
        if model == "gaussian":
            return GaussianEmission(spec["means"], spec["sds"])
        if model == "poisson":
            return PoissonEmission(spec["rates"])
        if model == "multivariate_gaussian":
            return MultivariateGaussianEmission(spec["means"], spec["covariances"])
        if model == "student_t":
            return StudentTEmission(spec["means"], spec["scales"], spec["df"])
    except KeyError as e:
        raise ValueError(f"emission spec {model!r} is missing {e.args[0]!r}") from None
    raise ValueError(f"unknown emission model {model!r}")


@dataclass(frozen=True)
class HMMState:
    n: int = 0
    probabilities: tuple[float, ...] = ()  # P(z_n = k | y_1..y_n); empty before the first observation
    log_likelihood: float = 0.0
    comp: float = 0.0


@dataclass(frozen=True)
class HMMOutput:
    n: int
    probabilities: tuple[float, ...]
    most_likely: int
    log_likelihood: float


class HiddenMarkovFilter(Procedure):
    """Exact filtering distribution of the hidden state of an HMM; see the module docstring."""

    name = "hidden_markov_filter"
    version = "1"
    guarantee = "model"

    def __init__(self, initial, transition, emission, *, input: str = "y"):
        self.initial = _distribution(initial, None, "initial")
        k = len(self.initial)
        self.transition = tuple(_distribution(row, k, "transition row") for row in transition)
        if len(self.transition) != k:
            raise ValueError(f"transition must be {k} x {k}")
        if getattr(emission, "states", None) != k:
            raise ValueError(f"the emission model must describe {k} states")
        self.emission, self.input, self.states = emission, input, k

    def config(self) -> dict:
        return {
            "initial": list(self.initial),
            "transition": [list(r) for r in self.transition],
            "emission": self.emission.spec(),
            "input": self.input,
        }

    def inputs(self):
        return {self.input: self.emission.contract(self.input)}

    def initial_state(self) -> HMMState:
        return HMMState()

    def predicted(self, state: HMMState) -> tuple[float, ...]:
        """P(z_{n+1} = k | y_1..y_n)."""
        if state.n == 0:
            return self.initial
        out = []
        for k in range(self.states):
            total = 0.0
            for j in range(self.states):
                total += state.probabilities[j] * self.transition[j][k]
            out.append(total)
        return tuple(out)

    def step(self, state: HMMState, x, rng=None):
        y = x[self.input]
        prior = self.predicted(state)
        logs = [self.emission.log_density(k, y) if prior[k] > 0 else -math.inf for k in range(self.states)]
        top = max(logs)
        if top == -math.inf:
            raise NumericalError(f"{self.name}: observation {y!r} is impossible in every reachable state")
        joint = [p * math.exp(e - top) for p, e in zip_strict(prior, logs)]
        total = 0.0
        for v in joint:
            total += v
        posterior = tuple(v / total for v in joint)
        log_lik, comp = neumaier_add(state.log_likelihood, state.comp, top + math.log(total))
        n = state.n + 1
        best = max(range(self.states), key=lambda k: (posterior[k], -k))
        new = HMMState(n, posterior, log_lik, comp)
        return new, HMMOutput(n, posterior, best, require_finite(log_lik + comp, "log likelihood"))

    def encode_state(self, state: HMMState):
        return {"n": state.n, "probabilities": list(state.probabilities), "log_likelihood": state.log_likelihood,
                "comp": state.comp}

    def encode_output(self, output: HMMOutput):
        return {"n": output.n, "probabilities": list(output.probabilities), "most_likely": output.most_likely,
                "log_likelihood": output.log_likelihood}

    def decode_state(self, data) -> HMMState:
        state = decode_fields(HMMState, data)
        require_state(len(state.probabilities) == (self.states if state.n else 0), "one probability per state")
        require_state(all(p >= 0 for p in state.probabilities), "negative probability")
        require_state(state.n == 0 or abs(math.fsum(state.probabilities) - 1.0) < 1e-9, "probabilities must sum to 1")
        return state
