# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Fitting a hidden Markov model to a recorded series: the Baum-Welch (EM) algorithm.

`HiddenMarkovFilter` needs the model: the initial distribution, the
transition matrix and the emission parameters.  When they are not known,
`fit_hmm` estimates them by maximum likelihood from a series observed
before (Baum, Petrie, Soules and Weiss 1970; Rabiner 1989).  Each
iteration runs the scaled forward-backward recursions of
`sequential_analysis.smoothing`, accumulates the expected transitions
and the posterior-weighted sufficient statistics of the emissions
(`emission.statistics`), and maximises (`emission.from_statistics`).
The log-likelihood never decreases from one iteration to the next (the
EM property; checked in the tests), but EM finds a local maximum: give
`restarts` to try further starting points drawn from the data, and the
fit with the highest likelihood is returned.

The likelihood does not change when the states are relabelled ("label
switching"): a run keeps the labels of its starting model, and the best
of several random restarts is relabelled to match `start` as closely as
possible.

The result is a `HiddenMarkovFilter`, ready to filter new data as they
arrive (or `OnlineHMM` to keep learning on the stream).  Checked against
hmmlearn: one EM step from the same starting point agrees to 1e-10
(Gaussian, Poisson and categorical emissions).

Cost: O(T K^2) per iteration for T observations and K states, in Python;
a few seconds for T = 10 000 and K = 3 per hundred iterations.
"""
from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ._compat import zip_strict
from .errors import NumericalError
from .procedures.hmm import HiddenMarkovFilter

__all__ = ["HMMFit", "e_step", "fit_hmm"]


@dataclass(frozen=True)
class HMMFit:
    model: HiddenMarkovFilter  # the fitted model
    log_likelihood: float  # log p(y_1..y_T) under it
    history: tuple[float, ...]  # log-likelihood before each iteration of the best run, then after the last
    iterations: int
    converged: bool  # the log-likelihood rose by less than `tolerance` in the last iteration
    restarts: tuple[float, ...]  # the final log-likelihood of every run (the starting model first)


@dataclass(frozen=True)
class _Expectations:
    log_likelihood: float
    first: tuple[float, ...]  # P(z_1 = k | y)
    transitions: tuple[tuple[float, ...], ...]  # sum_t P(z_{t-1} = j, z_t = k | y)
    emissions: tuple[tuple[float, ...], ...]  # per state: (sum_t gamma_t(k), sum_t gamma_t(k) g_k(y_t)...)


def e_step(model: HiddenMarkovFilter, ys) -> _Expectations:
    """Expected complete-data sufficient statistics of a series under a model (scaled forward-backward)."""
    k_states, a, emission = model.states, model.transition, model.emission
    alphas: list[tuple[float, ...]] = []
    scaled: list[list[float]] = []  # e_k(y_t) / exp(top_t)
    norms: list[float] = []  # c_t = sum_k predicted_t(k) e_k(y_t) / exp(top_t)
    log_lik = 0.0
    previous: tuple[float, ...] | None = None
    for y in ys:
        if previous is None:
            prior = model.initial
        else:
            prior = tuple(_dot_column(previous, a, k) for k in range(k_states))
        logs = [emission.log_density(k, y) if prior[k] > 0 else -math.inf for k in range(k_states)]
        top = max(logs)
        if top == -math.inf:
            raise NumericalError(f"fit_hmm: observation {y!r} is impossible in every reachable state")
        e = [math.exp(v - top) for v in logs]
        joint = [p * v for p, v in zip_strict(prior, e)]
        c = 0.0
        for v in joint:
            c += v
        alpha = tuple(v / c for v in joint)
        log_lik += top + math.log(c)
        alphas.append(alpha)
        scaled.append(e)
        norms.append(c)
        previous = alpha
    t_len = len(alphas)
    m = len(emission.statistics(0, ys[0]))
    trans = [[0.0] * k_states for _ in range(k_states)]
    stats = [[0.0] * (m + 1) for _ in range(k_states)]
    beta = [1.0] * k_states
    first: tuple[float, ...] = alphas[0]
    for t in range(t_len - 1, -1, -1):
        gamma = [alphas[t][k] * beta[k] for k in range(k_states)]
        total = 0.0
        for v in gamma:
            total += v
        gamma = [v / total for v in gamma]
        y = ys[t]
        for k in range(k_states):
            g = gamma[k]
            if g == 0.0:
                continue
            row = stats[k]
            row[0] += g
            for i, v in enumerate(emission.statistics(k, y)):
                row[i + 1] += g * v
        if t == 0:
            first = tuple(gamma)
            break
        # xi_t(j, k) = alpha_{t-1}(j) A[j][k] e_t(k) beta_t(k) / c_t, and beta_{t-1}
        e, c, alpha_prev = scaled[t], norms[t], alphas[t - 1]
        eb = [e[k] * beta[k] for k in range(k_states)]
        new_beta = []
        for j in range(k_states):
            acc = 0.0
            for k in range(k_states):
                term = a[j][k] * eb[k] / c
                trans[j][k] += alpha_prev[j] * term
                acc += term
            new_beta.append(acc)
        beta = new_beta
    return _Expectations(log_lik, first, tuple(tuple(r) for r in trans), tuple(tuple(r) for r in stats))


def _dot_column(p: tuple[float, ...], a, k: int) -> float:
    total = 0.0
    for j, pj in enumerate(p):
        total += pj * a[j][k]
    return total


def _normalised(row, floor: float = 0.0) -> tuple[float, ...]:
    values = [v + floor for v in row]
    total = math.fsum(values)
    if not total > 0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(v / total for v in values)


def _m_step(model: HiddenMarkovFilter, ex: _Expectations, min_scale: float, fit_initial: bool,
            fit_transition: bool) -> HiddenMarkovFilter:
    initial = _normalised(ex.first) if fit_initial else model.initial
    if fit_transition:
        transition = tuple(_normalised(row) if math.fsum(row) > 0 else model.transition[j]
                           for j, row in enumerate(ex.transitions))
    else:
        transition = model.transition
    emission = model.emission.from_statistics(ex.emissions, min_scale=min_scale)
    return HiddenMarkovFilter(initial, transition, emission, input=model.input)


def _run(model: HiddenMarkovFilter, ys, iterations: int, tolerance: float, min_scale: float, fit_initial: bool,
         fit_transition: bool) -> tuple[HiddenMarkovFilter, float, list[float], int, bool]:
    history: list[float] = []
    ex = e_step(model, ys)
    for it in range(iterations):
        history.append(ex.log_likelihood)
        candidate = _m_step(model, ex, min_scale, fit_initial, fit_transition)
        new = e_step(candidate, ys)
        model, previous, ex = candidate, ex.log_likelihood, new
        if new.log_likelihood - previous < tolerance:
            history.append(new.log_likelihood)
            return model, new.log_likelihood, history, it + 1, True
    history.append(ex.log_likelihood)
    return model, ex.log_likelihood, history, iterations, False


def _random_start(model: HiddenMarkovFilter, ys, rnd: random.Random, min_scale: float) -> HiddenMarkovFilter:
    """A starting model from a random soft assignment of the observations to the states."""
    k_states = model.states
    m = len(model.emission.statistics(0, ys[0]))
    totals = [[0.0] * (m + 1) for _ in range(k_states)]
    for y in ys:
        weights = [rnd.expovariate(1.0) for _ in range(k_states)]
        s = math.fsum(weights)
        for k in range(k_states):
            w = weights[k] / s
            totals[k][0] += w
            for i, v in enumerate(model.emission.statistics(k, y)):
                totals[k][i + 1] += w * v
    emission = model.emission.from_statistics(totals, min_scale=min_scale)
    stay = rnd.uniform(0.5, 0.95)
    transition = tuple(tuple(stay if j == k else (1.0 - stay) / (k_states - 1) for k in range(k_states))
                       for j in range(k_states)) if k_states > 1 else ((1.0,),)
    return HiddenMarkovFilter(tuple(1.0 / k_states for _ in range(k_states)), transition, emission,
                              input=model.input)


def _per_state(model: HiddenMarkovFilter) -> list[list[float]]:
    """Each state's emission parameters as one flat vector (from the spec's per-state lists)."""
    spec = model.emission.spec()
    vectors: list[list[float]] = [[] for _ in range(model.states)]
    for value in spec.values():
        if isinstance(value, list) and len(value) == model.states:
            for k, item in enumerate(value):
                vectors[k].extend(_flat(item))
    return vectors


def _flat(item: Any) -> list[float]:
    if isinstance(item, (list, tuple)):
        return [v for x in item for v in _flat(x)]
    return [float(item)] if isinstance(item, (int, float)) and not isinstance(item, bool) else []


def _aligned(model: HiddenMarkovFilter, start: HiddenMarkovFilter) -> HiddenMarkovFilter:
    """The fitted model with its states relabelled to match the starting model's as closely as possible.

    The labels of a run from a random start are arbitrary; state k of the
    result is the fitted state whose emission parameters are nearest
    (squared distance, the best of all relabellings) to state k of
    `start`.  For more than seven states the labels are left as fitted.
    """
    k_states = model.states
    if k_states > 7:
        return model
    fitted, target = _per_state(model), _per_state(start)

    def cost(order: tuple[int, ...]) -> float:
        total = 0.0
        for k, j in enumerate(order):
            for a, b in zip(fitted[j], target[k]):
                total += (a - b) ** 2
        return total

    order = min(itertools.permutations(range(k_states)), key=lambda o: (cost(o), o))
    if list(order) == list(range(k_states)):
        return model
    from .procedures.hmm import emission_from_spec

    spec = model.emission.spec()
    permuted = {key: ([value[k] for k in order] if isinstance(value, list) and len(value) == k_states else value)
                for key, value in spec.items()}
    initial = tuple(model.initial[k] for k in order)
    transition = tuple(tuple(model.transition[j][k] for k in order) for j in order)
    return HiddenMarkovFilter(initial, transition, emission_from_spec(permuted), input=model.input)


def fit_hmm(inputs: Iterable[Mapping[str, Any]], start: HiddenMarkovFilter, *, iterations: int = 500,
            tolerance: float = 1e-9, min_scale: float = 1e-6, restarts: int = 0, seed: int = 0,
            fit_initial: bool = True, fit_transition: bool = True) -> HMMFit:
    """Maximum-likelihood HMM parameters for a recorded series, by EM from `start` (and `restarts` random starts).

    `start` gives the number of states, the emission family and the
    starting parameters; its input name selects the field of the
    records.  EM stops when the log-likelihood rises by less than
    `tolerance` in an iteration, or after `iterations`.  `min_scale`
    bounds standard deviations, scales and rates from below, and is the
    ridge on covariances (a state that collapses onto one value otherwise
    drives the likelihood to infinity).  With `restarts`, further runs start from random soft
    assignments of the observations to the states (seeded by `seed`), and
    the best run is returned with its states sorted by their first
    emission parameter; `fit_initial` / `fit_transition` False keep those
    parameters as given.
    """
    ys = [x[start.input] for x in inputs]
    if len(ys) < 2:
        raise ValueError("fit_hmm needs at least two observations")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive int")
    if type(restarts) is not int or restarts < 0:
        raise ValueError("restarts must be a non-negative int")
    contract = start.inputs()[start.input]
    for y in ys:
        contract.validate(y)
    rnd = random.Random(seed)
    starts = [start] + [_random_start(start, ys, rnd, min_scale) for _ in range(restarts)]
    best: tuple[HiddenMarkovFilter, float, list[float], int, bool] | None = None
    finals = []
    for model in starts:
        result = _run(model, ys, iterations, tolerance, min_scale, fit_initial, fit_transition)
        finals.append(result[1])
        if best is None or result[1] > best[1]:
            best = result
    assert best is not None
    model, log_lik, history, its, converged = best
    if restarts:
        model = _aligned(model, start)
    return HMMFit(model, log_lik, tuple(history), its, converged, tuple(finals))
