# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Anytime-valid log-rank test for two-arm survival data (ter Schure, Pérez-Ortiz, Ly and Grünwald 2024).

The input is the stream of what happens to the patients of a trial, in the
order it happens: a patient enters the risk set (kind 0), has the event
(kind 1: death, failure) or leaves it without one (kind 2: censored, lost
to follow-up).  The procedure keeps n_T and n_C, the numbers at risk in
the treatment and control arms.  Under proportional hazards with hazard
ratio theta (treatment over control), an event that happens when n_T and
n_C patients are at risk is in the treatment arm with probability

    p_theta = n_T theta / (n_T theta + n_C),

whatever the baseline hazard (Cox's partial likelihood).  So under
H0: theta = theta0 each event's arm is a Bernoulli(p_theta0) variable
given the past, and for any alternative theta1 = theta0 exp(delta)
chosen before the event the likelihood ratio

    LR_t(delta) = exp(x_t delta) (n_T theta0 + n_C) / (n_T theta0 exp(delta) + n_C),

x_t = 1 for an event in the treatment arm and 0 in the control arm, has
conditional mean exactly 1 under H0.  Products of such factors are test
martingales, and Ville's inequality makes rejecting at E >= 1/alpha a
level-alpha test at any stopping time: the trial may be monitored after
every event, stopped or extended at will, with no maximum number of
events.  No assumption is made on the baseline hazard, on censoring or on
entry beyond their being non-informative: given the past, who enters and
who is censored does not depend on the arms of future events.

Time scale.  Risk sets are those of the order the records arrive in.
Validity needs proportional hazards on that time scale -- the calendar
time of the trial when records are streamed as they happen; a patient
who enters late is then left-truncated, which the partial likelihood
handles exactly.  If instead the hazards are a function of time since
entry and they are not constant, staggered entry breaks the martingale
property (the paper says so too), and not by a little.  By simulation
(200 patients entering uniformly over 25 scale units, 1:1, a Weibull
hazard of shape 0.3 in time since entry with scale 1, exponential
censoring at rate 0.3 after entry, the default mixture, alpha = 0.05),
the test rejects a true H0 in 21% of 4000 trials (25% of 800 with
censoring at rate 0.05; shape 0.5: 14% of 2000; shape 2: none of 2000),
where an exact analysis of the same kind of trial, records in time since
entry, rejects 1.7% of 2000.  The events of patients who entered recently then tell
which arm they are in more often than n_T / (n_T + n_C) says.
Exponential (constant) hazards are the same on both time scales and are
fine.  With hazards in time since entry, stream the records in that time
scale (possible only after the fact) or stratify by narrow entry cohorts
(next paragraph), which brings the same trials back under the level.

Stratification (strata > 1).  Each record also names its stratum s -- a
centre, a prognostic group, an entry cohort -- and the procedure keeps
n_T^s and n_C^s, the numbers at risk of each arm in each stratum.  Under
proportional hazards within strata (a baseline hazard of any shape in
each stratum, one hazard ratio theta common to all), an event in stratum
s is in the treatment arm with probability
n_T^s theta / (n_T^s theta + n_C^s): the stratified partial likelihood.
LR_t is the factor above with n_T^s and n_C^s in place of n_T and n_C,
its conditional mean under H0 is still exactly 1, and the products over
the events of all strata, in arrival order, are test martingales: the
methods, the one-sided composite nulls and the confidence sequence carry
over unchanged, and the estimate is the (penalised) stratified Cox
estimate.  An event in a stratum with one arm empty is certain: factor 1.
By simulation (exponential hazards, 200 patients, 600 trials, alpha =
0.05): in three centres with hazards 0.2, 1 and 5 randomising 1:3, 1:1
and 3:1, the unstratified test rejects a true H0 in 96% of trials and
the stratified one in 2.5%; in two centres with hazards 0.1 and 10
randomising 1:1 (400 patients), power at a hazard ratio of 0.6 is 18%
unstratified and 82% stratified.

Entry cohorts are an approximation: within a cohort, entry still varies,
so the cohorts must be narrow compared with the time over which the
hazard changes after entry.  In the shape-0.3 simulation above (4000
trials; cohorts are equal periods of entry), the rejection rate of a true
H0 is 21% unstratified, 20% with 2 cohorts, 16% with 4, 11% with 8, 5.8%
with 16, 3.8% with 25 (one scale unit each, 8 patients on average) and
1.5% with 50.  Narrow cohorts cost power, since an event in a cohort with
one arm empty carries no information: at a hazard ratio of 0.6, 39% with
25 cohorts and 21% with 50 (1000 trials), against 54% for the exact
analysis in time since entry (2000 trials).

With strata = 1, the default, the inputs, the configuration and the state
are exactly those of the unstratified test: the configuration names
`strata` and `stratum_input` only when strata > 1, so the identity,
checkpoints and conformance vectors of earlier releases are unchanged.
The state's `at_risk` holds n_T^0, n_C^0, n_T^1, n_C^1, ... (two numbers
per stratum); the output's `at_risk` is the total of each arm.

Ties.  Events are processed one at a time in arrival order, each removed
from the risk set before the next.  Tied events should be streamed in an
order that does not depend on their arms (random, say); under
H0: theta = 1 the test is then exact.

Alternatives.  "less" means theta < theta0 (with events as deaths: the
treatment is better), "greater" theta > theta0, "two-sided" either.

method="mixture" (default): E_n = sum_j w_j prod_t LR_t(delta_j), a
    discrete normal prior on delta = log(theta / theta0) with standard
    deviation `prior_sd` (tau): delta_j = j tau / 8 for 0 < |j| <= 32
    (so up to 4 tau), w_j proportional to exp(-delta_j^2 / (2 tau^2)),
    restricted to delta_j < 0 for "less" and delta_j > 0 for "greater"
    and normalised.  A finite mixture of test martingales is one.  The
    normal prior is Robbins' method of mixtures; in the normal
    approximation of the log-rank statistic it is tightest at about
    32 / tau^2 events for alpha = 0.05 with 1:1 allocation (128 for the
    default tau = 0.5, about the number a fixed-sample test needs for a
    hazard ratio of 0.6).  It is not the paper's own prior (the paper
    mainly uses a point alternative or virtual data points).

method="grow": E_n = prod_t LR_t(d_t), with d_t the plug-in estimate of
    delta from the events before t (predictable): the maximiser of the
    partial log-likelihood plus the log-density of the N(0, tau^2) prior,
    found on the grid delta_j above and refined by a parabola through
    the best grid point and its neighbours.  For a one-sided test an
    estimate on the null side is not bet on (d_t = 0, factor 1).

For one-sided tests every factor has conditional mean at most 1 for every
theta on the null side (for "less": E_theta LR(delta) decreases in theta
when delta < 0), so the e-process is valid for the composite null.

`hazard_ratio_estimate` is theta0 exp(d), d the (unrestricted) plug-in
estimate after the latest event: the partial-likelihood estimate, shrunk
towards theta0 by the prior.

Confidence sequence (confidence_sequence=True).  The two-sided mixture is
inverted over a grid of `cs_grid` hazard ratios theta_i log-spaced over
`cs_range` (default 101 points over [1/10, 10], step h = log(100)/100):
theta_i is excluded once

    E_n(theta_i) = sum_j w'_j exp(l_n(b_i + m_j) - l_n(b_i)) >= 1/alpha,

with l_n(b) the partial log-likelihood at log hazard ratio b and m_j the
offsets of the same normal prior, on multiples of h so that every
b_i + m_j is a point of one grid.  The state holds l_n on that grid (the
CS grid widened by 4 tau on each side), which is all the mixtures need:
constant memory, and one log per grid point per event.  Excluded points
are never evaluated again (the sequence is intersected over time); the
interval spans the points not excluded, widened by one grid step on each
side.  lower (upper) is None while the end of the grid is not excluded:
the data do not yet bound the ratio within cs_range.  For every grid
point theta_i, P(theta_i is ever excluded | theta = theta_i) <= alpha;
between grid points the one-step widening is the same approximation as
in BettingConfidenceSequence.  If every point is excluded (a <= alpha
event, or the true ratio lies outside cs_range) the interval collapses to
the least-excluded point and its neighbours.  The interval is two-sided
at level alpha, whatever `alternative`.

Cost per event (CPython, default grids): 65 logarithms for the test, 40 to
55 µs; with the confidence sequence also 187 logarithms and exponentials
and a convolution of the points still in the interval with 86 prior
weights (8 700 multiply-adds at the start, falling as the interval
narrows): 0.6 ms for the first events, 0.2 ms once the interval is
narrow.  Entries and censorings cost a few µs.  The state is constant:
252 floats with the default grids (5 kB of JSON; 1.5 kB without the
confidence sequence), and two integers per stratum.  25 strata do not
change the cost measurably; 1000 strata about double the cost of an
entry or censoring (the output sums the numbers at risk over the strata).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..errors import ContractViolation
from ..numerics import logsumexp, neumaier_add, require_finite
from .sprt import REJECT_H0

ALTERNATIVES = ("two-sided", "less", "greater")
METHODS = ("mixture", "grow")
ENTER, EVENT, CENSOR = 0, 1, 2
_STEPS = 8  # grid points per prior standard deviation
_SPAN = 4  # prior standard deviations on each side
_HALF = _STEPS * _SPAN  # 32: the alternative grid has 2 * 32 + 1 points
_TINY = 1e-280  # below this a convolution is recomputed in the log domain
_MAX_STRATA = 10_000


@dataclass(frozen=True)
class AnytimeLogRankState:
    at_risk: tuple[int, ...] = (0, 0)  # treatment, control; per stratum: n_T^0, n_C^0, n_T^1, n_C^1, ...
    n_events: int = 0
    log_lik: tuple[float, ...] = ()  # sum_t log LR_t(delta_j), j = -32..32
    log_e: float = 0.0  # grow: log of the product of the plug-in factors
    comp: float = 0.0
    cs_log_lik: tuple[float, ...] = ()  # partial log-likelihood ratio against theta = 1 on the CS grid
    cs_kept: tuple[int, int] = (0, 0)  # first and last CS grid index not yet excluded
    max_log_e: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class AnytimeLogRankOutput:
    n_events: int
    at_risk: tuple[int, int]  # treatment, control: totals over the strata
    log_e_value: float
    p_value: float
    decision: str | None
    hazard_ratio_estimate: float
    lower: float | None  # None: unbounded (or no confidence sequence)
    upper: float | None


def _add(counts: tuple[int, ...], i: int, change: int) -> tuple[int, ...]:
    """counts with counts[i] changed by `change`."""
    return (*counts[:i], counts[i] + change, *counts[i + 1:])


def _log_factor(x: int, n_t: int, n_c: int, delta: float, theta0: float, theta1: float) -> float:
    """log LR_t: x delta + log(n_T theta0 + n_C) - log(n_T theta1 + n_C)."""
    return x * delta + math.log(n_t * theta0 + n_c) - math.log(n_t * theta1 + n_c)


class AnytimeLogRankTest(Procedure):
    """Anytime-valid log-rank test and confidence sequence for a hazard ratio; see the module docstring.

    Inputs: the arm of the record (`arm_input`, an integer label) and its
    kind (`input`): 0 = the patient enters the risk set, 1 = event,
    2 = censored.  `arms` = (treatment label, control label); records of
    other arms are ignored.  An event or censoring in an arm with nobody
    at risk is a contract violation.  `theta0` is the null hazard ratio
    (treatment over control); `alternative` "less" tests H0: theta >= theta0
    against theta < theta0.  `prior_sd` is the standard deviation of the
    normal prior on log(theta / theta0) (method="mixture" and the
    confidence sequence) and of the shrinkage of the plug-in estimate.

    `strata` > 1 makes the test stratified (see "Stratification" in the
    module docstring): every record then also carries its stratum
    (`stratum_input`, an integer in 0..strata-1), the state keeps the
    numbers at risk of each arm in each stratum, and an event or censoring
    in an arm of a stratum with nobody at risk is a contract violation.
    The output's `at_risk` is the total over the strata.  With the default
    strata=1 the procedure, its inputs, its configuration and its state
    are exactly those of the unstratified test (the configuration names
    `strata` and `stratum_input` only when strata > 1), so earlier
    checkpoints restore unchanged.
    """

    name = "anytime_logrank_test"
    version = "1"
    e_value_field = "log_e_value"

    def __init__(self, alpha: float = 0.05, *, theta0: float = 1.0, alternative: str = "two-sided",
                 method: str = "mixture", prior_sd: float = 0.5, confidence_sequence: bool = True,
                 cs_range: tuple[float, float] = (0.1, 10.0), cs_grid: int = 101,
                 arms: tuple[int, int] = (1, 0), stop_on_reject: bool = False,
                 arm_input: str = "arm", input: str = "kind", strata: int = 1, stratum_input: str = "stratum"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if type(theta0) not in (int, float) or not (math.isfinite(theta0) and theta0 > 0):
            raise ValueError("theta0 must be a positive finite hazard ratio")
        if alternative not in ALTERNATIVES:
            raise ValueError(f"alternative must be one of {ALTERNATIVES}")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if type(prior_sd) not in (int, float) or not (math.isfinite(prior_sd) and 0.01 <= prior_sd <= 5):
            raise ValueError("prior_sd must be a number in [0.01, 5] (a standard deviation of the log hazard ratio)")
        if type(confidence_sequence) is not bool:
            raise ValueError("confidence_sequence must be True or False")
        if len(cs_range) != 2 or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in cs_range) \
                or not cs_range[0] < cs_range[1]:
            raise ValueError("cs_range must be two positive finite hazard ratios, lower < upper")
        if type(cs_grid) is not int or not 3 <= cs_grid <= 2000:
            raise ValueError("cs_grid must be an int in [3, 2000]")
        if len(arms) != 2 or any(type(a) is not int or a < 0 for a in arms) or arms[0] == arms[1]:
            raise ValueError("arms must be two different non-negative integer labels")
        if arm_input == input:
            raise ValueError("the arm and the kind need different names")
        if type(strata) is not int or not 1 <= strata <= _MAX_STRATA:
            raise ValueError(f"strata must be an int in [1, {_MAX_STRATA}]")
        if strata > 1 and stratum_input in (arm_input, input):
            raise ValueError("the stratum needs a name different from the arm and the kind")
        self.alpha, self.theta0, self.alternative, self.method = float(alpha), float(theta0), alternative, method
        self.prior_sd, self.confidence_sequence = float(prior_sd), confidence_sequence
        self.cs_range, self.cs_grid = (float(cs_range[0]), float(cs_range[1])), cs_grid
        self.arms, self.stop_on_reject = (arms[0], arms[1]), stop_on_reject
        self.arm_input, self.input = arm_input, input
        self.strata, self.stratum_input = strata, stratum_input
        self._log_level = math.log(1.0 / self.alpha)

        tau = self.prior_sd
        step = tau / _STEPS
        self._step = step
        self._deltas = tuple(j * step for j in range(-_HALF, _HALF + 1))
        self._thetas = tuple(self.theta0 * math.exp(d) for d in self._deltas)  # exactly theta0 at j = 0
        self._penalty = tuple(-d * d / (2.0 * tau * tau) for d in self._deltas)
        if alternative == "less":
            side = [j for j in range(-_HALF, 0)]
        elif alternative == "greater":
            side = [j for j in range(1, _HALF + 1)]
        else:
            side = [j for j in range(-_HALF, _HALF + 1) if j != 0]
        norm = logsumexp(self._penalty[j + _HALF] for j in side)
        self._mix_index = tuple(j + _HALF for j in side)
        self._mix_logw = tuple(self._penalty[j + _HALF] - norm for j in side)

        # confidence sequence: points b_i = log(lo) + i h, offsets m_j = j c h, one grid for all b_i + m_j
        lo, hi = self.cs_range
        self._h = math.log(hi / lo) / (cs_grid - 1)
        self._c = max(1, round(step / self._h))
        self._pad = max(1, int(_SPAN * tau / (self._c * self._h))) * self._c  # grid points beyond each end
        width = cs_grid + 2 * self._pad
        self._cs_betas = tuple(math.log(lo) + (u - self._pad) * self._h for u in range(width))
        self._cs_exp = tuple(math.exp(b) for b in self._cs_betas)
        self._cs_points = tuple(lo * math.exp(i * self._h) for i in range(cs_grid))
        offsets = [j * self._c for j in range(-self._pad // self._c, self._pad // self._c + 1) if j != 0]
        logs = [-(m * self._h) ** 2 / (2.0 * tau * tau) for m in offsets]
        cs_norm = logsumexp(logs)
        self._cs_offsets = tuple(offsets)
        self._cs_logw = tuple(v - cs_norm for v in logs)
        self._cs_w = tuple(math.exp(v) for v in self._cs_logw)

    def config(self) -> dict:
        config = {"alpha": self.alpha, "theta0": self.theta0, "alternative": self.alternative,
                  "method": self.method, "prior_sd": self.prior_sd,
                  "confidence_sequence": self.confidence_sequence, "cs_range": list(self.cs_range),
                  "cs_grid": self.cs_grid, "arms": list(self.arms), "stop_on_reject": self.stop_on_reject,
                  "arm_input": self.arm_input, "input": self.input}
        if self.strata != 1:  # only then: the unstratified test keeps the identity it had before strata existed
            config["strata"] = self.strata
            config["stratum_input"] = self.stratum_input
        return config

    def inputs(self):
        contracts = {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0),
                     self.input: InputContract(self.input, kind="integer", lower=ENTER, upper=CENSOR)}
        if self.strata != 1:
            contracts[self.stratum_input] = InputContract(self.stratum_input, kind="integer", lower=0,
                                                          upper=self.strata - 1)
        return contracts

    def initial_state(self) -> AnytimeLogRankState:
        cs: tuple[float, ...] = ()
        kept = (0, 0)
        if self.confidence_sequence:
            cs = (0.0,) * len(self._cs_betas)
            kept = (0, self.cs_grid - 1)
        return AnytimeLogRankState(at_risk=(0, 0) * self.strata, log_lik=(0.0,) * len(self._deltas), cs_log_lik=cs,
                                   cs_kept=kept)

    # ------------------------------------------------------------ e-values and estimates

    def log_e_value(self, state: AnytimeLogRankState) -> float:
        if self.method == "grow":
            return state.log_e + state.comp
        return logsumexp(w + state.log_lik[i] for w, i in zip(self._mix_logw, self._mix_index))

    def estimate(self, log_lik: tuple[float, ...]) -> float:
        """d: the penalised partial-likelihood estimate of log(theta / theta0), from the grid."""
        f = [v + p for v, p in zip(log_lik, self._penalty)]
        best = max(range(len(f)), key=f.__getitem__)  # the first maximum
        if 0 < best < len(f) - 1:
            left, mid, right = f[best - 1], f[best], f[best + 1]
            curvature = left - 2.0 * mid + right
            shift = 0.5 * (left - right) / curvature if curvature < 0.0 else 0.0
            return (best - _HALF + shift) * self._step
        return (best - _HALF) * self._step

    def _bet(self, log_lik: tuple[float, ...]) -> float:
        d = self.estimate(log_lik)
        if self.alternative == "less":
            return min(d, 0.0)
        if self.alternative == "greater":
            return max(d, 0.0)
        return d

    def cs_log_e_values(self, cs_log_lik: tuple[float, ...], first: int, last: int) -> list[float]:
        """log E_n(theta_i) of the two-sided mixture for the CS grid points first..last."""
        pad, offsets = self._pad, self._cs_offsets
        top = max(cs_log_lik[first:last + 2 * pad + 1])
        scaled = [math.exp(v - top) for v in cs_log_lik[first:last + 2 * pad + 1]]
        out = []
        for i in range(first, last + 1):
            centre = i + pad
            total = 0.0
            for w, m in zip(self._cs_w, offsets):
                total += w * scaled[centre + m - first]
            if total > _TINY:
                out.append(math.log(total) + top - cs_log_lik[centre])
            else:
                base = cs_log_lik[centre]
                out.append(logsumexp(lw + cs_log_lik[centre + m] - base
                                     for lw, m in zip(self._cs_logw, offsets)))
        return out

    def _bounds(self, kept: tuple[int, int]) -> tuple[float | None, float | None]:
        if not self.confidence_sequence:
            return None, None
        first, last = kept
        lower = self._cs_points[first - 1] if first > 0 else None
        upper = self._cs_points[last + 1] if last < self.cs_grid - 1 else None
        return lower, upper

    def _output(self, state: AnytimeLogRankState, value: float) -> AnytimeLogRankOutput:
        lower, upper = self._bounds(state.cs_kept)
        estimate = self.theta0 * math.exp(self.estimate(state.log_lik))
        at_risk = state.at_risk
        n_t, n_c = at_risk[0], at_risk[1]
        for i in range(2, len(at_risk), 2):  # totals over the strata (integers)
            n_t += at_risk[i]
            n_c += at_risk[i + 1]
        return AnytimeLogRankOutput(state.n_events, (n_t, n_c), value, min(1.0, math.exp(-state.max_log_e)),
                                    state.decision, estimate, lower, upper)

    # ------------------------------------------------------------ the step

    def step(self, state: AnytimeLogRankState, x, rng=None):
        label = x[self.arm_input]
        if label not in self.arms:
            return state, self._output(state, self.log_e_value(state))
        k = self.arms.index(label)
        kind = x[self.input]
        stratum, where = 0, ""
        if self.strata != 1:
            given = x[self.stratum_input]
            if type(given) not in (int, float) or not 0 <= given < self.strata or given != int(given):
                raise ContractViolation(f"{self.name}: stratum {given!r} is not an integer in 0..{self.strata - 1}")
            stratum = int(given)
            where = f" of stratum {stratum}"
        i = 2 * stratum + k  # the position of this arm of this stratum in the state
        n_t, n_c = state.at_risk[2 * stratum], state.at_risk[2 * stratum + 1]
        if kind == ENTER:
            at_risk = _add(state.at_risk, i, 1)
            new = AnytimeLogRankState(at_risk, state.n_events, state.log_lik, state.log_e, state.comp,
                                      state.cs_log_lik, state.cs_kept, state.max_log_e, state.decision)
            return new, self._output(new, self.log_e_value(new))
        if state.at_risk[i] == 0:
            what = "an event" if kind == EVENT else "a censoring"
            raise ContractViolation(f"{self.name}: {what} in arm {label}{where} with nobody at risk")
        at_risk = _add(state.at_risk, i, -1)
        if kind == CENSOR:
            new = AnytimeLogRankState(at_risk, state.n_events, state.log_lik, state.log_e, state.comp,
                                      state.cs_log_lik, state.cs_kept, state.max_log_e, state.decision)
            return new, self._output(new, self.log_e_value(new))
        log_lik, log_e, comp = state.log_lik, state.log_e, state.comp
        cs_log_lik, kept = state.cs_log_lik, state.cs_kept
        if n_t > 0 and n_c > 0:  # otherwise the arm of the event is certain: every factor is 1
            xt = 1 if k == 0 else 0
            theta0 = self.theta0
            if self.method == "grow":
                d = self._bet(log_lik)
                if d != 0.0:
                    log_e, comp = neumaier_add(log_e, comp, _log_factor(xt, n_t, n_c, d, theta0,
                                                                        theta0 * math.exp(d)))
            log_lik = tuple(v + _log_factor(xt, n_t, n_c, d, theta0, t)
                            for v, d, t in zip(log_lik, self._deltas, self._thetas))
            if self.confidence_sequence:
                cs_log_lik = tuple(v + _log_factor(xt, n_t, n_c, b, 1.0, e)
                                   for v, b, e in zip(cs_log_lik, self._cs_betas, self._cs_exp))
                first, last = kept
                values = self.cs_log_e_values(cs_log_lik, first, last)
                inside = [first + i for i, v in enumerate(values) if v < self._log_level]
                if inside:
                    kept = (inside[0], inside[-1])
                else:  # every point excluded: collapse onto the least-excluded one
                    least = first + min(range(len(values)), key=values.__getitem__)
                    kept = (least, least)
        new = AnytimeLogRankState(at_risk, state.n_events + 1, log_lik, log_e, comp, cs_log_lik, kept,
                                  state.max_log_e, state.decision)
        value = require_finite(self.log_e_value(new), "log e-value")
        decision = state.decision or (REJECT_H0 if value >= self._log_level else None)
        new = AnytimeLogRankState(new.at_risk, new.n_events, new.log_lik, new.log_e, new.comp, new.cs_log_lik,
                                  new.cs_kept, max(state.max_log_e, value), decision)
        return new, self._output(new, value)

    def is_terminal(self, state: AnytimeLogRankState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def encode_state(self, state: AnytimeLogRankState) -> Any:
        return {"at_risk": list(state.at_risk), "n_events": state.n_events, "log_lik": list(state.log_lik),
                "log_e": state.log_e, "comp": state.comp, "cs_log_lik": list(state.cs_log_lik),
                "cs_kept": list(state.cs_kept), "max_log_e": state.max_log_e, "decision": state.decision}

    def decode_state(self, data) -> AnytimeLogRankState:
        state = decode_fields(AnytimeLogRankState, data)
        require_state(len(state.at_risk) == 2 * self.strata and len(state.cs_kept) == 2,
                      "two arms (in every stratum), two CS indices")
        require_state(len(state.log_lik) == len(self._deltas), "one log-likelihood per alternative")
        require_state(state.log_lik[_HALF] == 0.0, "the null point has likelihood ratio one")
        if self.method == "mixture":
            require_state(state.log_e == 0.0 and state.comp == 0.0, "a mixture keeps no plug-in product")
        if self.confidence_sequence:
            require_state(len(state.cs_log_lik) == len(self._cs_betas), "one log-likelihood per CS grid point")
            first, last = state.cs_kept
            require_state(first <= last <= self.cs_grid - 1, "CS indices out of range")
        else:
            require_state(state.cs_log_lik == () and state.cs_kept == (0, 0), "no confidence sequence")
        if state.n_events == 0:
            require_state(all(v == 0.0 for v in state.log_lik + state.cs_log_lik) and state.log_e == 0.0
                          and state.comp == 0.0 and state.max_log_e == 0.0, "no events, no evidence")
            if self.confidence_sequence:
                require_state(state.cs_kept == (0, self.cs_grid - 1), "no events, nothing excluded")
        value = self.log_e_value(state)
        require_state(state.max_log_e >= 0.0 and state.max_log_e >= value, "the running maximum")
        require_state(state.decision in (None, REJECT_H0), "the test only rejects")
        require_state((state.decision is not None) == (state.max_log_e >= self._log_level),
                      "the decision follows the running maximum")
        return state
