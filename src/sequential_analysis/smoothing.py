# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Smoothing: the hidden state at every time, given all the observations.

A filter answers "where is the state now, given what has been seen so
far?" -- the question of a run that acts while data arrive.  Looking back
over a finished series (a recorded run, a day of sensor data), the better
question is "where was the state at time t, given everything, including
what came after t?".  These functions answer it for a stretch of inputs,
offline; they reuse the filter's own model and step, so a smoothing pass
over the inputs of a run agrees with that run's filtering outputs.

kalman_smoother -- Rauch-Tung-Striebel: the exact conditional distribution
    N(m_{t|T}, P_{t|T}) of z_t given y_1..y_T in a linear Gaussian model,
    from a forward Kalman pass and the backward recursion

        G_t     = P_{t|t} F' P_{t+1|t}^-1
        m_{t|T} = m_{t|t} + G_t (m_{t+1|T} - m_{t+1|t})
        P_{t|T} = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t'.

    With a RobustKalmanFilter the same recursion runs on its variational
    posteriors: an approximation, as the filter is.

forward_backward -- P(z_t = k | y_1..y_T) in a hidden Markov model, from
    the scaled forward and backward recursions.
viterbi -- the single most probable path of hidden states and its log
    joint probability log p(z_1..z_T, y_1..y_T).

Cost (PERFORMANCE.md): linear in T; per time point O(n^3) for the Kalman
smoother (0.17 ms for two states) and O(K^2) for the HMM (17 µs for
forward-backward, 3 µs for Viterbi, three states).  Memory is linear in T: the forward pass
is kept for the backward one.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from . import linalg as la
from ._compat import zip_strict
from .errors import NumericalError
from .procedures.hmm import HiddenMarkovFilter
from .procedures.state_space import KalmanFilter


@dataclass(frozen=True)
class KalmanSmoothing:
    means: tuple[tuple[float, ...], ...]  # E[z_t | y_1..y_T], t = 1..T
    covariances: tuple[la.Matrix, ...]
    filtered_means: tuple[tuple[float, ...], ...]  # E[z_t | y_1..y_t]
    filtered_covariances: tuple[la.Matrix, ...]
    log_likelihood: float | None  # log p(y_1..y_T); None for the robust filter


def kalman_smoother(filter: KalmanFilter, inputs: Iterable[Mapping[str, Any]]) -> KalmanSmoothing:
    """Rauch-Tung-Striebel smoothing of `inputs` (the statistical inputs, as given to the filter's step)."""
    rows = list(inputs)
    if not rows:
        raise ValueError("nothing to smooth")
    state = filter.initial_state()
    predicted: list[tuple[la.Vector, la.Matrix]] = []
    filtered: list[tuple[la.Vector, la.Matrix]] = []
    output = None
    for x in rows:
        predicted.append(filter._predict(state, x))
        state, output = filter.step(state, x)
        filtered.append((state.mean, state.covariance))
    means = [filtered[-1][0]]
    covs = [filtered[-1][1]]
    for t in range(len(rows) - 2, -1, -1):
        m_f, p_f = filtered[t]
        m_p, p_p = predicted[t + 1]
        try:
            p_p_inverse = la.inverse_spd(la.symmetrize(p_p))
        except ValueError:
            raise NumericalError("kalman_smoother: a predicted covariance is singular "
                                 "(a state without process noise that the data never reach?)") from None
        gain = la.matmul(la.matmul(p_f, la.transpose(filter.F)), p_p_inverse)
        mean = la.vadd(m_f, la.matvec(gain, la.vsub(means[-1], m_p)))
        cov = la.symmetrize(la.add(p_f, la.quadratic(gain, la.sub(covs[-1], p_p))))
        means.append(mean)
        covs.append(cov)
    means.reverse()
    covs.reverse()
    assert output is not None
    return KalmanSmoothing(tuple(means), tuple(covs), tuple(m for m, _ in filtered), tuple(p for _, p in filtered),
                           output.log_likelihood)


@dataclass(frozen=True)
class HMMSmoothing:
    probabilities: tuple[tuple[float, ...], ...]  # P(z_t = k | y_1..y_T)
    filtered: tuple[tuple[float, ...], ...]  # P(z_t = k | y_1..y_t)
    log_likelihood: float


def _scaled_emissions(hmm: HiddenMarkovFilter, prior: tuple[float, ...], y: Any) -> tuple[list[float], float]:
    """e_k(y) / exp(top) for the states with prior mass, and top: densities scaled to avoid underflow."""
    logs = [hmm.emission.log_density(k, y) if prior[k] > 0 else -math.inf for k in range(hmm.states)]
    top = max(logs)
    if top == -math.inf:
        raise NumericalError(f"forward_backward: observation {y!r} is impossible in every reachable state")
    return [math.exp(v - top) for v in logs], top


def forward_backward(hmm: HiddenMarkovFilter, inputs: Iterable[Mapping[str, Any]]) -> HMMSmoothing:
    """Smoothed state probabilities of a hidden Markov model over `inputs`."""
    ys = [x[hmm.input] for x in inputs]
    if not ys:
        raise ValueError("nothing to smooth")
    k_states, a = hmm.states, hmm.transition
    alphas: list[tuple[float, ...]] = []
    emissions: list[list[float]] = []
    scales: list[float] = []
    state = hmm.initial_state()
    log_lik = 0.0
    for y in ys:
        prior = hmm.predicted(state)
        e, _ = _scaled_emissions(hmm, prior, y)
        state, out = hmm.step(state, {hmm.input: y})
        joint = [p * v for p, v in zip_strict(prior, e)]
        scales.append(math.fsum(joint))
        emissions.append(e)
        alphas.append(state.probabilities)
        log_lik = out.log_likelihood
    beta = [1.0] * k_states
    smoothed = [alphas[-1]]
    for t in range(len(ys) - 2, -1, -1):
        e, c = emissions[t + 1], scales[t + 1]
        beta = [math.fsum(a[j][k] * e[k] * beta[k] for k in range(k_states)) / c for j in range(k_states)]
        gamma = [p * b for p, b in zip_strict(alphas[t], beta)]
        total = math.fsum(gamma)
        smoothed.append(tuple(g / total for g in gamma))
    smoothed.reverse()
    return HMMSmoothing(tuple(smoothed), tuple(alphas), log_lik)


@dataclass(frozen=True)
class ViterbiPath:
    states: tuple[int, ...]
    log_probability: float  # log p(z_1..z_T, y_1..y_T) of that path


def viterbi(hmm: HiddenMarkovFilter, inputs: Iterable[Mapping[str, Any]]) -> ViterbiPath:
    """The most probable sequence of hidden states (ties go to the smaller state index)."""
    ys = [x[hmm.input] for x in inputs]
    if not ys:
        raise ValueError("nothing to decode")

    def log(p: float) -> float:
        return math.log(p) if p > 0 else -math.inf

    k_states = hmm.states
    log_a = [[log(p) for p in row] for row in hmm.transition]
    delta = [log(hmm.initial[k]) + hmm.emission.log_density(k, ys[0]) if hmm.initial[k] > 0 else -math.inf
             for k in range(k_states)]
    back: list[list[int]] = []
    for y in ys[1:]:
        pointers, nxt = [], []
        for k in range(k_states):
            best = max(range(k_states), key=lambda j: (delta[j] + log_a[j][k], -j))
            score = delta[best] + log_a[best][k]
            pointers.append(best)
            nxt.append(score + hmm.emission.log_density(k, y) if score > -math.inf else -math.inf)
        back.append(pointers)
        delta = nxt
    last = max(range(k_states), key=lambda k: (delta[k], -k))
    if delta[last] == -math.inf:
        raise NumericalError("viterbi: the observations are impossible under the model")
    path = [last]
    for pointers in reversed(back):
        path.append(pointers[path[-1]])
    path.reverse()
    return ViterbiPath(tuple(path), delta[last])
