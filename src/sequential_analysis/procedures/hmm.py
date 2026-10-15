# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Filtering in a hidden Markov model: which regime is the process in now?

A hidden state z_t in {0, ..., K-1} moves by a Markov chain,
P(z_t = k | z_{t-1} = j) = A[j][k], starting from P(z_1 = k) = initial[k];
the observation y_t depends only on z_t.  The forward algorithm gives the
exact filtering distribution

    P(z_t = k | y_1..y_t)  proportional to  e_k(y_t) sum_j P(z_{t-1} = j | y_1..y_{t-1}) A[j][k],

and the normalising constants multiply to the likelihood p(y_1..y_t).  The
emission densities are combined in logarithms, so observations far in the
tails do not underflow.

Emission models: CategoricalEmission (a symbol 0..M-1), GaussianEmission
(a mean and standard deviation per state), PoissonEmission (a rate per
state).  Typical uses: regime detection (calm/volatile), a machine's
hidden condition, a patient's disease stage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
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
    """y in {0, ..., M-1} with P(y = s | z = k) = probabilities[k][s]."""

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
        joint = [p * math.exp(e - top) for p, e in zip(prior, logs, strict=True)]
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
