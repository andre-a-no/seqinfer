# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""A hidden Markov model that learns its parameters on the stream: online EM (Cappé 2011).

`HiddenMarkovFilter` needs the transition matrix and the emission
parameters; `fit_hmm` estimates them from a recorded series.  OnlineHMM
filters and estimates at once, one observation at a time, in constant
memory -- for a process that cannot be recorded first, or whose
parameters drift slowly.

It keeps, besides the filter phi_t(k) = P(z_t = k | y_1..y_t), the
expected complete-data sufficient statistics conditionally on the current
state,

    rho_t(k) = E[ (1/t-weighted) sum_s s(z_{s-1}, z_s, y_s) | z_t = k, y_1..y_t ],

for the transitions (counts of j -> l) and the emissions (the
statistics `emission.statistics` of each state), updated by the backward
kernel r_t(j | k) = phi_{t-1}(j) A[j][k] / sum_j' phi_{t-1}(j') A[j'][k]:

    rho_t(k) = gamma_t s(., k, y_t) + (1 - gamma_t) sum_j rho_{t-1}(j) r_t(j | k),

with step sizes gamma_t = t^-kappa (1/2 < kappa <= 1).  The statistics
S_t = sum_k rho_t(k) phi_t(k) approximate the normalised expected
statistics of the batch E-step, and after `burn_in` observations the
parameters are their maximiser (the M-step of fit_hmm): A[j][l] =
S(j, l) / sum_l S(j, l), the emissions `emission.from_statistics`.
Before that the starting model is used.  Cappé (2011) shows that the
estimates converge to a stationary point of the limiting likelihood under
regularity conditions; the tests check that they approach the batch
maximum-likelihood estimates (fit_hmm) and the true parameters on long
simulated series.

The starting model fixes the number of states, the emission family and
the labels: like any EM, online EM finds the maximum nearest to where it
starts, and states keep their labels only if they start apart.

Outputs: the filtered probabilities and most likely state, the current
transition matrix and emission parameters, and the prequential
log-likelihood sum_t log p(y_t | y_1..y_{t-1}), each term under the
parameters of the step before -- a fair score of the learner.

Parameters (`average`, default True): the M-step is applied to the
statistics averaged since the burn-in (Polyak-Ruppert averaging), which
removes most of the noise of the step sizes.  On series of 15 000 from a
two-state Gaussian HMM, the standard deviation over seeds of an estimated
transition probability was 0.019 for the last iterate with kappa = 0.6,
and 0.005 with averaging -- that of the batch maximum-likelihood
estimate.

Cost per step (PERFORMANCE.md): O(K^4 + K^2 m) for K states and m emission
statistics, 56 µs for three Gaussian states (CPython); the state holds
K + K^3 + K^2 (m + 1) numbers and as many averages, 1.7 kB of JSON for
three states, constant in time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..errors import NumericalError
from ..numerics import neumaier_add, require_finite
from .hmm import HiddenMarkovFilter


@dataclass(frozen=True)
class OnlineHMMState:
    n: int = 0
    probabilities: tuple[float, ...] = ()  # phi_n(k)
    transitions: tuple[float, ...] = ()  # rho_n for the transition counts: index (k, j, l) -> (k K + j) K + l
    emissions: tuple[float, ...] = ()  # rho_n for the emission statistics: index (k, j, i) -> (k K + j)(m + 1) + i
    log_likelihood: float = 0.0
    comp: float = 0.0
    averaged: tuple[float, ...] = ()  # running mean of S_n since the burn-in: K^2 transition, K (m + 1) emission


@dataclass(frozen=True)
class OnlineHMMOutput:
    n: int
    probabilities: tuple[float, ...]
    most_likely: int
    transition: tuple[tuple[float, ...], ...]  # the current estimate
    emission: Any  # the current emission parameters, as emission.spec()
    log_likelihood: float  # prequential


class OnlineHMM(Procedure):
    """Filtering and online-EM estimation of an HMM's parameters; see the module docstring."""

    name = "online_hmm"
    version = "1"

    def __init__(self, start: HiddenMarkovFilter, *, kappa: float = 0.6, burn_in: int = 100, min_scale: float = 1e-6,
                 average: bool = True):
        if not isinstance(start, HiddenMarkovFilter):
            raise TypeError("start must be a HiddenMarkovFilter: the starting model")
        if type(kappa) not in (int, float) or not 0.5 < kappa <= 1.0:
            raise ValueError("kappa must lie in (0.5, 1]")
        if type(burn_in) is not int or burn_in < 1:
            raise ValueError("burn_in must be a positive int")
        if type(min_scale) not in (int, float) or not (math.isfinite(min_scale) and min_scale >= 0):
            raise ValueError("min_scale must be a non-negative number")
        if type(average) is not bool:
            raise ValueError("average must be True or False")
        self.average = average
        self.start, self.kappa, self.burn_in, self.min_scale = start, float(kappa), burn_in, float(min_scale)
        self.states, self.input = start.states, start.input
        probe = start.emission.statistics(0, _probe_value(start))
        self._m = len(probe)

    def config(self) -> dict:
        return {"start": self.start.config(), "kappa": self.kappa, "burn_in": self.burn_in,
                "min_scale": self.min_scale, "average": self.average}

    def inputs(self):
        return {self.input: self.start.inputs()[self.input]}

    def initial_state(self) -> OnlineHMMState:
        return OnlineHMMState()

    # ------------------------------------------------------------ parameters from the statistics

    def statistics(self, state: OnlineHMMState) -> tuple[list[list[float]], list[list[float]]]:
        """S_n = sum_k rho_n(k) phi_n(k): transition counts (K x K) and emission totals (K x (m + 1))."""
        k_states, m1 = self.states, self._m + 1
        trans = [[0.0] * k_states for _ in range(k_states)]
        emis = [[0.0] * m1 for _ in range(k_states)]
        for k, pk in enumerate(state.probabilities):
            if pk == 0.0:
                continue
            base = k * k_states * k_states
            for j in range(k_states):
                row = trans[j]
                for ell in range(k_states):
                    row[ell] += pk * state.transitions[base + j * k_states + ell]
            base = k * k_states * m1
            for j in range(k_states):
                row = emis[j]
                for i in range(m1):
                    row[i] += pk * state.emissions[base + j * m1 + i]
        return trans, emis

    def model(self, state: OnlineHMMState) -> HiddenMarkovFilter:
        """The parameters used for the next observation: the starting model until `burn_in`, then the M-step
        (of the statistics averaged since the burn-in, with `average`)."""
        if state.n < self.burn_in:
            return self.start
        if self.average:
            trans, emis = self._unflatten(state.averaged)
        else:
            trans, emis = self.statistics(state)
        transition = []
        for j, row in enumerate(trans):
            total = 0.0
            for v in row:
                total += v
            transition.append(tuple(v / total for v in row) if total > 0 else self.start.transition[j])
        emission = self.start.emission.from_statistics(emis, min_scale=self.min_scale)
        return HiddenMarkovFilter(self.start.initial, transition, emission, input=self.input)

    def _unflatten(self, flat: tuple[float, ...]) -> tuple[list[list[float]], list[list[float]]]:
        k, m1 = self.states, self._m + 1
        trans = [list(flat[j * k:(j + 1) * k]) for j in range(k)]
        emis = [list(flat[k * k + j * m1:k * k + (j + 1) * m1]) for j in range(k)]
        return trans, emis

    # ------------------------------------------------------------ the step

    def step(self, state: OnlineHMMState, x, rng=None):
        y = x[self.input]
        k_states, m1 = self.states, self._m + 1
        model = self.model(state)
        a, emission = model.transition, model.emission
        n = state.n + 1
        if state.n == 0:
            prior = list(model.initial)
        else:
            prior = []
            for k in range(k_states):
                total = 0.0
                for j in range(k_states):
                    total += state.probabilities[j] * a[j][k]
                prior.append(total)
        logs = [emission.log_density(k, y) if prior[k] > 0 else -math.inf for k in range(k_states)]
        top = max(logs)
        if top == -math.inf:
            raise NumericalError(f"{self.name}: observation {y!r} is impossible in every reachable state")
        joint = [p * math.exp(e - top) for p, e in zip(prior, logs)]
        c = 0.0
        for v in joint:
            c += v
        phi = tuple(v / c for v in joint)
        log_lik, comp = neumaier_add(state.log_likelihood, state.comp, top + math.log(c))
        stats = [emission.statistics(k, y) for k in range(k_states)]
        if state.n == 0:  # rho_1(k): the first observation's own statistics; no transition yet
            transitions = [0.0] * (k_states ** 3)
            emissions = [0.0] * (k_states * k_states * m1)
            for k in range(k_states):
                base = (k * k_states + k) * m1
                emissions[base] = 1.0
                for i, v in enumerate(stats[k]):
                    emissions[base + 1 + i] = v
        else:
            gamma = n ** -self.kappa
            keep = 1.0 - gamma
            transitions = [0.0] * (k_states ** 3)
            emissions = [0.0] * (k_states * k_states * m1)
            for k in range(k_states):
                if prior[k] == 0.0:
                    continue
                # backward kernel r(j | k)
                r = [state.probabilities[j] * a[j][k] / prior[k] for j in range(k_states)]
                tk = k * k_states * k_states
                for j in range(k_states):
                    rj = r[j]
                    if rj == 0.0:
                        continue
                    old = j * k_states * k_states
                    for p in range(k_states * k_states):
                        transitions[tk + p] += keep * rj * state.transitions[old + p]
                    transitions[tk + j * k_states + k] += gamma * rj
                ek = k * k_states * m1
                for j in range(k_states):
                    rj = r[j]
                    if rj == 0.0:
                        continue
                    old = j * k_states * m1
                    for p in range(k_states * m1):
                        emissions[ek + p] += keep * rj * state.emissions[old + p]
                base = ek + k * m1
                emissions[base] += gamma
                for i, v in enumerate(stats[k]):
                    emissions[base + 1 + i] += gamma * v
        new = OnlineHMMState(n, phi, tuple(transitions), tuple(emissions), log_lik, comp)
        averaged: tuple[float, ...] = ()
        if self.average and n >= self.burn_in:
            trans_now, emis_now = self.statistics(new)
            flat = [v for row in trans_now for v in row] + [v for row in emis_now for v in row]
            if n == self.burn_in:
                averaged = tuple(flat)
            else:
                weight = 1.0 / (n - self.burn_in + 1)
                averaged = tuple(a + weight * (v - a) for a, v in zip(state.averaged, flat))
            new = OnlineHMMState(n, phi, tuple(transitions), tuple(emissions), log_lik, comp, averaged)
        current = self.model(new)
        best = max(range(k_states), key=lambda k: (phi[k], -k))
        out = OnlineHMMOutput(n, phi, best, current.transition, current.emission.spec(),
                              require_finite(log_lik + comp, "log likelihood"))
        return new, out

    def encode_output(self, output: OnlineHMMOutput):
        return {"n": output.n, "probabilities": list(output.probabilities), "most_likely": output.most_likely,
                "transition": [list(r) for r in output.transition], "emission": output.emission,
                "log_likelihood": output.log_likelihood}

    def decode_state(self, data) -> OnlineHMMState:
        state = decode_fields(OnlineHMMState, data)
        k, m1 = self.states, self._m + 1
        if state.n == 0:
            require_state(not (state.probabilities or state.transitions or state.emissions or state.averaged),
                          "statistics before data")
            return state
        require_state(len(state.probabilities) == k, "one probability per state")
        require_state(len(state.transitions) == k ** 3 and len(state.emissions) == k * k * m1, "statistics' sizes")
        require_state(all(p >= 0 for p in state.probabilities), "negative probability")
        require_state(abs(math.fsum(state.probabilities) - 1.0) < 1e-9, "probabilities must sum to 1")
        require_state(all(v >= 0 for v in state.transitions), "negative transition statistics")
        averaging = self.average and state.n >= self.burn_in
        require_state(len(state.averaged) == (k * k + k * m1 if averaging else 0), "averaged statistics' size")
        return state


def _probe_value(model: HiddenMarkovFilter) -> Any:
    """A valid observation, to learn the number of emission statistics."""
    contract: InputContract = model.inputs()[model.input]
    if contract.shape:
        return [0.0] * contract.shape[0]
    if contract.kind in ("integer", "binary"):
        return 0
    return 0.0
