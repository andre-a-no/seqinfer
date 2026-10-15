# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Bayesian online change-point detection (Adams and MacKay 2007).

The data come in segments; before each observation after the first, a new
segment starts with probability `hazard`, independently, and the
parameters of each segment are drawn afresh from a conjugate prior.  The
run length L_t -- how many observations the current segment holds,
including y_t -- then has an exactly computable posterior: either the
segment grows,

    P(L_t = l + 1, y_1..t)  =  P(L_{t-1} = l, y_1..t-1) (1 - hazard) p(y_t | the l observations of the segment),

or a new one starts with y_t,

    P(L_t = 1, y_1..t)  =  hazard p(y_1..t-1) p(y_t | prior).

Unlike CUSUM, nothing about the parameters before or after the change is
assumed known: unknown means and variances, unknown rates.  The output
says how probable it is that a new segment began with this observation
(`change_probability`), the most probable run length, and the evidence
log p(y_1..y_t).  With `alarm_probability` set, the procedure stops once
the probability that a change happened within the last `alarm_window`
observations reaches it.

Conjugate models: NormalGammaModel (normal data, unknown mean and
variance; Student t predictive), PoissonGammaModel (counts; negative
binomial predictive), BetaBernoulliModel (0/1 data).

The exact posterior has one hypothesis per possible run length, so it
grows with the run.  Hypotheses below `prune` probability are dropped
and at most `max_hypotheses` are kept, the rest renormalised: the
posterior and the evidence are then approximations (in the tests,
dropping below 1e-12 changes the change probabilities by less than 1e-10;
keeping 100 hypotheses for 200 observations, by 1e-5 and the log evidence
by 1e-3; keeping 50, by 4e-4 and 0.02).
prune = 0 and max_hypotheses = None keep the posterior exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..numerics import logsumexp, neumaier_add, require_finite
from .sprt import REJECT_H0


def _positive(*values: float) -> bool:
    return all(math.isfinite(v) and v > 0 for v in values)


class NormalGammaModel:
    """Normal data with unknown mean and precision: mean | tau ~ N(mu0, 1/(kappa0 tau)), tau ~ Gamma(alpha0, beta0)."""

    def __init__(self, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0):
        if not (math.isfinite(mu0) and _positive(kappa0, alpha0, beta0)):
            raise ValueError("need a finite mu0 and positive kappa0, alpha0, beta0")
        self.mu0, self.kappa0, self.alpha0, self.beta0 = float(mu0), float(kappa0), float(alpha0), float(beta0)

    def spec(self) -> dict:
        return {"model": "normal_gamma", "mu0": self.mu0, "kappa0": self.kappa0, "alpha0": self.alpha0,
                "beta0": self.beta0}

    def contract(self, name: str) -> InputContract:
        return InputContract(name)

    empty = (0.0, 0.0, 0.0)  # n, mean, sum of squared deviations

    def update(self, stats, y: float):
        n, mean, ss = stats
        n += 1.0
        delta = y - mean
        mean += delta / n
        return (n, mean, ss + delta * (y - mean))

    def posterior(self, stats):
        n, mean, ss = stats
        kappa = self.kappa0 + n
        mu = (self.kappa0 * self.mu0 + n * mean) / kappa
        alpha = self.alpha0 + 0.5 * n
        beta = self.beta0 + 0.5 * ss + self.kappa0 * n * (mean - self.mu0) ** 2 / (2.0 * kappa)
        return mu, kappa, alpha, beta

    def log_predictive(self, stats, y: float) -> float:
        """Student t with 2 alpha degrees of freedom, location mu, scale^2 = beta (kappa + 1) / (alpha kappa)."""
        mu, kappa, alpha, beta = self.posterior(stats)
        nu, s2 = 2.0 * alpha, beta * (kappa + 1.0) / (alpha * kappa)
        z2 = (y - mu) ** 2 / s2
        return float((math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi * s2))
                - 0.5 * (nu + 1.0) * math.log1p(z2 / nu))

    def mean(self, stats) -> float:
        return float(self.posterior(stats)[0])


class PoissonGammaModel:
    """Counts with unknown rate: rate ~ Gamma(shape, rate)."""

    def __init__(self, shape: float = 1.0, rate: float = 1.0):
        if not _positive(shape, rate):
            raise ValueError("shape and rate must be positive")
        self.shape, self.rate = float(shape), float(rate)

    def spec(self) -> dict:
        return {"model": "poisson_gamma", "shape": self.shape, "rate": self.rate}

    def contract(self, name: str) -> InputContract:
        return InputContract(name, kind="integer", lower=0)

    empty = (0.0, 0.0)  # n, sum

    def update(self, stats, y: float):
        return (stats[0] + 1.0, stats[1] + y)

    def log_predictive(self, stats, y: float) -> float:
        """Negative binomial: Gamma(r + y) / (Gamma(r) y!) (b / (b + 1))^r (1 / (b + 1))^y."""
        r, b = self.shape + stats[1], self.rate + stats[0]
        return float((math.lgamma(r + y) - math.lgamma(r) - math.lgamma(y + 1.0) + r * math.log(b / (b + 1.0)))
                - y * math.log1p(b))

    def mean(self, stats) -> float:
        return float((self.shape + stats[1]) / (self.rate + stats[0]))


class BetaBernoulliModel:
    """0/1 data with unknown success probability ~ Beta(a, b)."""

    def __init__(self, a: float = 1.0, b: float = 1.0):
        if not _positive(a, b):
            raise ValueError("a and b must be positive")
        self.a, self.b = float(a), float(b)

    def spec(self) -> dict:
        return {"model": "beta_bernoulli", "a": self.a, "b": self.b}

    def contract(self, name: str) -> InputContract:
        return InputContract(name, kind="binary")

    empty = (0.0, 0.0)  # n, successes

    def update(self, stats, y: float):
        return (stats[0] + 1.0, stats[1] + y)

    def log_predictive(self, stats, y: float) -> float:
        p = (self.a + stats[1]) / (self.a + self.b + stats[0])
        return math.log(p) if y else math.log1p(-p)

    def mean(self, stats) -> float:
        return float((self.a + stats[1]) / (self.a + self.b + stats[0]))


@dataclass(frozen=True)
class BOCPDState:
    n: int = 0
    lengths: tuple[int, ...] = ()  # run lengths with positive posterior probability
    log_probs: tuple[float, ...] = ()  # their log posterior probabilities
    stats: tuple[tuple[float, ...], ...] = ()  # the segment's sufficient statistics under each
    log_evidence: float = 0.0
    comp: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class BOCPDOutput:
    n: int
    change_probability: float  # P(a new segment began with this observation | y_1..y_n)
    recent_change_probability: float  # P(a change within the last alarm_window observations)
    most_likely_run_length: int
    expected_run_length: float
    segment_mean: float  # posterior mean of the current segment's parameter, averaged over run lengths
    log_evidence: float  # log p(y_1..y_n)
    decision: str | None  # "reject_h0" (a change) once the alarm probability is reached


class BayesianChangePoint(Procedure):
    """Bayesian online change-point detection with a conjugate model; see the module docstring."""

    name = "bayesian_change_point"
    version = "1"

    def __init__(
        self,
        model,
        hazard: float = 0.01,
        *,
        alarm_probability: float | None = None,
        alarm_window: int = 5,
        prune: float = 1e-12,
        max_hypotheses: int | None = 2000,
        input: str = "x",
    ):
        if not 0 < hazard < 1:
            raise ValueError("hazard must lie in (0, 1): the probability that a segment ends at each step")
        if alarm_probability is not None and not 0 < alarm_probability < 1:
            raise ValueError("alarm_probability must lie in (0, 1), or be None")
        if type(alarm_window) is not int or alarm_window < 1:
            raise ValueError("alarm_window must be a positive int")
        if not (0 <= prune < 1):
            raise ValueError("prune must lie in [0, 1)")
        if max_hypotheses is not None and (type(max_hypotheses) is not int or max_hypotheses < 2):
            raise ValueError("max_hypotheses must be an int of at least 2, or None")
        self.model, self.hazard = model, float(hazard)
        self.alarm_probability = None if alarm_probability is None else float(alarm_probability)
        self.alarm_window, self.prune = alarm_window, float(prune)
        self.max_hypotheses, self.input = max_hypotheses, input
        self._log_h, self._log_1mh = math.log(self.hazard), math.log1p(-self.hazard)

    def config(self) -> dict:
        return {
            "model": self.model.spec(),
            "hazard": self.hazard,
            "alarm_probability": self.alarm_probability,
            "alarm_window": self.alarm_window,
            "prune": self.prune,
            "max_hypotheses": self.max_hypotheses,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: self.model.contract(self.input)}

    def initial_state(self) -> BOCPDState:
        return BOCPDState()

    def step(self, state: BOCPDState, x, rng=None):
        y = x[self.input]
        model = self.model
        if state.n == 0:
            lengths, logs, stats = [1], [model.log_predictive(model.empty, y)], [model.update(model.empty, y)]
        else:
            lengths = [1]
            logs = [self._log_h + model.log_predictive(model.empty, y)]
            stats = [model.update(model.empty, y)]
            for length, lp, st in zip(state.lengths, state.log_probs, state.stats, strict=True):
                lengths.append(length + 1)
                logs.append(lp + self._log_1mh + model.log_predictive(st, y))
                stats.append(model.update(st, y))
        increment = require_finite(logsumexp(logs), "evidence")
        logs = [v - increment for v in logs]
        log_evidence, comp = neumaier_add(state.log_evidence, state.comp, increment)
        lengths, logs, stats = self._prune(lengths, logs, stats)
        probs = [math.exp(v) for v in logs]
        n = state.n + 1
        change = probs[0] if n > 1 and lengths[0] == 1 else 0.0
        recent = 0.0
        if n > self.alarm_window:
            for length, p in zip(lengths, probs, strict=True):
                if length <= self.alarm_window:
                    recent += p
        best = max(range(len(lengths)), key=lambda i: (probs[i], -lengths[i]))
        expected, seg_mean = 0.0, 0.0
        for length, p, st in zip(lengths, probs, stats, strict=True):
            expected += p * length
            seg_mean += p * model.mean(st)
        decision = state.decision
        if decision is None and self.alarm_probability is not None and recent >= self.alarm_probability:
            decision = REJECT_H0
        new = BOCPDState(n, tuple(lengths), tuple(logs), tuple(tuple(s) for s in stats), log_evidence, comp, decision)
        out = BOCPDOutput(n, change, recent, lengths[best], expected, seg_mean,
                          require_finite(log_evidence + comp, "evidence"), decision)
        return new, out

    def _prune(self, lengths, logs, stats):
        keep = list(range(len(lengths)))
        if self.prune > 0:
            cut = math.log(self.prune)
            keep = [i for i in keep if logs[i] >= cut] or [max(keep, key=lambda i: logs[i])]
        if self.max_hypotheses is not None and len(keep) > self.max_hypotheses:
            keep = sorted(sorted(keep, key=lambda i: (-logs[i], i))[: self.max_hypotheses])
        if len(keep) == len(lengths):
            return lengths, logs, stats
        norm = logsumexp(logs[i] for i in keep)
        return [lengths[i] for i in keep], [logs[i] - norm for i in keep], [stats[i] for i in keep]

    def run_length_distribution(self, state: BOCPDState) -> dict[int, float]:
        """P(L_n = l | y_1..y_n) for the run lengths kept."""
        return {length: math.exp(lp) for length, lp in zip(state.lengths, state.log_probs, strict=True)}

    def is_terminal(self, state: BOCPDState) -> bool:
        return state.decision is not None

    def encode_state(self, state: BOCPDState):
        return {"n": state.n, "lengths": list(state.lengths), "log_probs": list(state.log_probs),
                "stats": [list(s) for s in state.stats], "log_evidence": state.log_evidence, "comp": state.comp,
                "decision": state.decision}

    def decode_state(self, data) -> BOCPDState:
        state = decode_fields(BOCPDState, data)
        k = len(state.lengths)
        require_state(len(state.log_probs) == len(state.stats) == k, "one probability and statistic per run length")
        require_state((k == 0) == (state.n == 0), "hypotheses without observations")
        require_state(all(1 <= length <= state.n for length in state.lengths), "run length out of range")
        require_state(k == 0 or abs(math.fsum(math.exp(v) for v in state.log_probs) - 1.0) < 1e-9,
                      "probabilities must sum to 1")
        width = len(self.model.empty)
        require_state(all(len(s) == width for s in state.stats), "sufficient statistics of the wrong size")
        return state
