# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Continuously monitored A/B tests.

Each observation is one unit (a user, a session) with the arm it was
assigned to and its outcome.  The tests may be looked at after every unit
and stopped whenever one likes; none needs a sample size fixed in advance.

SampleRatioMismatch -- is traffic split in the intended proportions?
    Units fall into arm k with probability p_k under H0.  The Bayes factor
    of a Dirichlet(c p) mixture of multinomials against H0 (Lindon and
    Malek 2022),

        BF_n = prod_t  (c p_{k_t} + n_{k_t, t-1}) / ((c + t - 1) p_{k_t}),

    is a test martingale under H0, so rejecting at BF_n >= 1/alpha is an
    exact level-alpha test at any stopping time.  A mismatch means the
    assignment or the logging is broken and no other result of the
    experiment can be trusted; run it alongside every A/B test.

TwoProportionTest -- do two conversion rates differ?  Exact, any n.
    H0: p_T = p_C (or p_T <= p_C, p_T >= p_C), the common rate unknown.
    By default (method="grow") units are grouped, in arrival order, into
    blocks of k_T treatment and k_C control units, and each block
    contributes the e-value of Turner, Ly and Grunwald (2024),

        E_block = prod q_a(y_T) prod q_b(y_C) / prod q_pbar(y),
        pbar = (k_T a + k_C b) / (k_T + k_C),

    q_r being the Bernoulli(r) likelihood and (a, b) the rates estimated
    from the earlier blocks.  pbar is the projection of (a, b) onto H0
    that makes E_block an e-value for every common rate at once (checked
    in the tests by enumerating every outcome of a block), so the product
    over blocks is a test supermartingale under H0 and Ville's inequality
    gives P(sup E >= 1/alpha) <= alpha.  This needs an allocation that does
    not look at outcomes -- randomised, or a fixed schedule, as in an A/B
    test: a later block's estimates use units that arrived after some of
    its own, which is harmless only when the assignment ignores outcomes.
    Estimation errors in the common rate cancel between numerator and
    denominator; by simulation it needs 1.3 to 1.6 times the units of the
    asymptotic test, against 2.3 to 3 times for method="universal".

    method="universal" is sequential universal inference (Wasserman,
    Ramdas and Balakrishnan 2020), E_n = m_T(y_T) m_C(y_C) / sup_H0 L, with
    Beta marginal likelihoods m_k: no blocks, valid for any allocation
    rule, adaptive ones included, at the price of more units.

ABTest -- difference of means, optionally adjusted by a covariate.
    With units assigned to treatment with known probability pi (the
    allocation), the inverse-propensity score

        psi_t = (T_t / pi - C_t / (1 - pi)) (y_t - f_{t-1}(x_t))

    has mean E[y(T)] - E[y(C)] for every f fitted on earlier units only:
    assignment is independent of x and of f.  Choosing f as the regression
    of y on the pre-experiment covariate x fitted on the past (pooled over
    arms) removes the part of the variance that x explains, like CUPED
    (Deng et al. 2013), without the bias that fitting on the current unit
    would bring.  The confidence sequence for the effect is the asymptotic
    one of Waudby-Smith et al. (2024) applied to psi, valid uniformly over
    time once n is large (start=1000 by default; see robust_mean).  Without
    a covariate f is the running mean of y, which already removes the
    variance an uncentred inverse-propensity estimate would carry.
    `variance_ratio` reports var(psi with covariate) / var(psi without):
    the fraction of the sample size the covariate saves is 1 minus it.

e_bh and MultipleTests -- many metrics or many arms, false discovery rate.
    Every exact test here produces an e-process: at any stopping time its
    value is an e-value.  The e-BH procedure (Wang and Ramdas 2022) rejects
    the k hypotheses with the largest e-values, k the largest such that the
    k-th largest is at least K / (k alpha); the false discovery rate is at
    most alpha under any dependence between the tests.  MultipleTests runs
    several procedures on the same observations and applies e-BH after
    every step: the set it reports when one stops has FDR <= alpha (the
    guarantee is at the stopping time chosen, not simultaneously for all
    times).  `MultipleTests.versus_control` builds the multi-arm
    comparison: each treatment arm against the control.

Cost per step (PERFORMANCE.md): constant time and state for the three
tests, 3 to 4 µs; MultipleTests costs the sum of its tests plus e-BH over
K e-values (25 µs for two two-proportion tests).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .._compat import zip_strict
from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..numerics import neumaier_add, require_finite
from .sprt import REJECT_H0

ALTERNATIVES = ("two-sided", "greater", "less")


def _check_alpha(alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    return float(alpha)


def _xlogx_ratio(k: int, n: int) -> float:
    """k log(k / n), with 0 log 0 = 0."""
    return k * math.log(k / n) if k > 0 else 0.0


# ---------------------------------------------------------------- sample ratio mismatch


@dataclass(frozen=True)
class SRMState:
    n: int = 0
    counts: tuple[int, ...] = ()
    log_bf: float = 0.0
    comp: float = 0.0
    max_log_bf: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class SRMOutput:
    n: int
    counts: tuple[int, ...]
    proportions: tuple[float, ...]
    log_e_value: float
    p_value: float
    decision: str | None


class SampleRatioMismatch(Procedure):
    """Anytime-valid test that units arrive in each arm in the intended proportions.

    `weights` are the intended proportions (normalised to sum 1); the input
    is the index of the arm a unit was assigned to.  `concentration` c sets
    the Dirichlet(c p) mixture: larger values look for smaller imbalances
    and need more units to find a large one.
    """

    name = "sample_ratio_mismatch"
    version = "1"
    e_value_field = "log_e_value"

    def __init__(self, weights: Sequence[float] = (0.5, 0.5), alpha: float = 0.001, *,
                 concentration: float = 100.0, stop_on_reject: bool = False, input: str = "arm"):
        w = [float(v) for v in weights]
        if len(w) < 2 or not all(math.isfinite(v) and v > 0 for v in w):
            raise ValueError("weights must be at least two positive numbers")
        total = math.fsum(w)
        self.p = tuple(v / total for v in w)
        self.alpha = _check_alpha(alpha)
        if not (math.isfinite(concentration) and concentration > 0):
            raise ValueError("concentration must be positive and finite")
        self.concentration = float(concentration)
        self.stop_on_reject, self.input = stop_on_reject, input
        self._log_level = -math.log(self.alpha)

    def config(self) -> dict:
        return {"weights": list(self.p), "alpha": self.alpha, "concentration": self.concentration,
                "stop_on_reject": self.stop_on_reject, "input": self.input}

    def inputs(self):
        return {self.input: InputContract(self.input, kind="integer", lower=0, upper=len(self.p) - 1)}

    def initial_state(self) -> SRMState:
        return SRMState(0, (0,) * len(self.p))

    def log_e_value(self, counts: Sequence[int]) -> float:
        """Closed form of log BF_n (the step accumulates it term by term)."""
        c, n = self.concentration, math.fsum(counts)  # integers: exact
        out = math.lgamma(c) - math.lgamma(c + n)
        for p, k in zip_strict(self.p, counts):
            out += math.lgamma(c * p + k) - math.lgamma(c * p) - k * math.log(p)
        return out

    def step(self, state: SRMState, x, rng=None):
        k = int(x[self.input])
        p = self.p[k]
        term = math.log((self.concentration * p + state.counts[k]) / ((self.concentration + state.n) * p))
        log_bf, comp = neumaier_add(state.log_bf, state.comp, term)
        n = state.n + 1
        counts = (*state.counts[:k], state.counts[k] + 1, *state.counts[k + 1:])
        value = require_finite(log_bf + comp, "log Bayes factor")
        best = max(state.max_log_bf, value)
        decision = state.decision or (REJECT_H0 if value >= self._log_level else None)
        new = SRMState(n, counts, log_bf, comp, best, decision)
        return new, SRMOutput(n, counts, tuple(c / n for c in counts), value, min(1.0, math.exp(-best)), decision)

    def is_terminal(self, state: SRMState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> SRMState:
        state = decode_fields(SRMState, data)
        require_state(len(state.counts) == len(self.p), "one count per arm")
        require_state(math.fsum(state.counts) == state.n, "counts must add up to n")
        return state


# ---------------------------------------------------------------- two proportions

METHODS = ("grow", "universal")


@dataclass(frozen=True)
class TwoProportionState:
    successes: tuple[int, int] = (0, 0)  # every unit seen: treatment, control
    trials: tuple[int, int] = (0, 0)
    log_sum: float = 0.0  # grow: log E over the completed blocks; universal: log m_T + log m_C
    comp: float = 0.0
    pending_treatment: tuple[int, ...] = ()  # grow: outcomes waiting for their block, oldest first
    pending_control: tuple[int, ...] = ()
    block_successes: tuple[int, int] = (0, 0)  # grow: units in completed blocks (the plug-in estimates)
    block_trials: tuple[int, int] = (0, 0)
    max_log_e: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class TwoProportionOutput:
    n: int
    rates: tuple[float | None, float | None]  # treatment, control
    difference: float | None  # treatment minus control
    log_e_value: float
    p_value: float
    decision: str | None
    blocks: int  # grow: completed blocks; universal: 0


class TwoProportionTest(Procedure):
    """Exact anytime-valid comparison of two conversion rates; see the module docstring.

    Inputs: the arm of the unit (`arm_input`, an integer label) and its
    outcome (`input`, 0 or 1).  `arms` = (treatment label, control label);
    units of other arms are ignored, so several tests can share one stream
    of a multi-arm experiment.  `alternative` "greater" tests H0: p_T <= p_C
    against p_T > p_C.

    method="grow" (the default) groups units into blocks of `block` =
    (treatment units, control units) in arrival order and multiplies the
    blocks' e-values; choose `block` in the proportion of the allocation,
    (1, 1) for 50/50, (2, 1) for 2:1.  Units wait for their block, so the
    state holds the unmatched outcomes: with random allocation in that
    proportion their number grows like the square root of n.
    method="universal" needs no blocks and is valid for any allocation
    rule, outcome-adaptive ones (bandits) included, but needs about twice
    as many units.
    `prior` = (a, b) is the Beta prior of the plug-in estimates (grow) or of
    the mixture (universal); (1/2, 1/2) by default.
    """

    name = "two_proportion_test"
    version = "2"
    e_value_field = "log_e_value"

    def __init__(self, alpha: float = 0.05, *, alternative: str = "two-sided", method: str = "grow",
                 block: tuple[int, int] = (1, 1), arms: tuple[int, int] = (1, 0),
                 prior: tuple[float, float] = (0.5, 0.5), stop_on_reject: bool = False,
                 arm_input: str = "arm", input: str = "y"):
        self.alpha = _check_alpha(alpha)
        if alternative not in ALTERNATIVES:
            raise ValueError(f"alternative must be one of {ALTERNATIVES}")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if len(block) != 2 or any(type(k) is not int or not 1 <= k <= 100 for k in block):
            raise ValueError("block must be two ints in [1, 100]: treatment and control units per block")
        if len(arms) != 2 or any(type(a) is not int or a < 0 for a in arms) or arms[0] == arms[1]:
            raise ValueError("arms must be two different non-negative integer labels")
        a, b = (float(v) for v in prior)
        if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
            raise ValueError("prior must be two positive numbers")
        if arm_input == input:
            raise ValueError("the arm and the outcome need different names")
        self.alternative, self.method, self.block = alternative, method, (block[0], block[1])
        self.arms, self.prior = (arms[0], arms[1]), (a, b)
        self.stop_on_reject, self.arm_input, self.input = stop_on_reject, arm_input, input
        self._log_level = -math.log(self.alpha)

    def config(self) -> dict:
        return {"alpha": self.alpha, "alternative": self.alternative, "method": self.method,
                "block": list(self.block), "arms": list(self.arms), "prior": list(self.prior),
                "stop_on_reject": self.stop_on_reject, "arm_input": self.arm_input, "input": self.input}

    def inputs(self):
        return {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0),
                self.input: InputContract(self.input, kind="binary")}

    def initial_state(self) -> TwoProportionState:
        return TwoProportionState()

    # ------------------------------------------------------------ universal inference

    def log_null_likelihood(self, successes: tuple[int, int], trials: tuple[int, int]) -> float:
        """log sup of the likelihood over H0 (method="universal")."""
        (st, sc), (nt, nc) = successes, trials

        def arm(s: int, n: int) -> float:
            return _xlogx_ratio(s, n) + _xlogx_ratio(n - s, n) if n else 0.0

        if self.alternative != "two-sided" and (nt == 0 or nc == 0):
            return arm(st, nt) + arm(sc, nc)
        if self.alternative != "two-sided":
            free = st * nc <= sc * nt if self.alternative == "greater" else st * nc >= sc * nt
            if free:  # the unconstrained maximum already lies in H0
                return arm(st, nt) + arm(sc, nc)
        return arm(st + sc, nt + nc)

    # ------------------------------------------------------------ GROW blocks

    def plug_in(self, successes: tuple[int, int], trials: tuple[int, int]) -> tuple[float, float, float]:
        """(a, b, p): the alternative fitted on completed blocks, and its projection on H0.

        For a one-sided test an estimate on the null side is not bet on:
        a = b = p, and the block's e-value is 1.
        """
        pa, pb = self.prior
        a = (successes[0] + pa) / (trials[0] + pa + pb)
        b = (successes[1] + pa) / (trials[1] + pa + pb)
        kt, kc = self.block
        if (self.alternative == "greater" and a <= b) or (self.alternative == "less" and a >= b):
            p = (kt * a + kc * b) / (kt + kc)
            return p, p, p
        return a, b, (kt * a + kc * b) / (kt + kc)

    def block_log_e(self, a: float, b: float, p: float, treated: Sequence[int], control: Sequence[int]) -> float:
        """log of one block's e-value: prod q_a(y_T) prod q_b(y_C) / prod q_p(y)."""
        if a == b == p:
            return 0.0
        out = 0.0
        for y in treated:
            out += math.log(a / p) if y else math.log1p(-a) - math.log1p(-p)
        for y in control:
            out += math.log(b / p) if y else math.log1p(-b) - math.log1p(-p)
        return out

    # ------------------------------------------------------------ the step

    def log_e_value(self, state: TwoProportionState) -> float:
        if self.method == "grow":
            return state.log_sum + state.comp
        return state.log_sum + state.comp - self.log_null_likelihood(state.successes, state.trials)

    def _output(self, state: TwoProportionState, value: float) -> TwoProportionOutput:
        (st, sc), (nt, nc) = state.successes, state.trials
        rt = st / nt if nt else None
        rc = sc / nc if nc else None
        diff = None if rt is None or rc is None else rt - rc
        blocks = state.block_trials[0] // self.block[0] if self.method == "grow" else 0
        return TwoProportionOutput(nt + nc, (rt, rc), diff, value, min(1.0, math.exp(-state.max_log_e)),
                                   state.decision, blocks)

    def step(self, state: TwoProportionState, x, rng=None):
        label = x[self.arm_input]
        if label not in self.arms:
            return state, self._output(state, self.log_e_value(state))
        k = self.arms.index(label)
        y = int(x[self.input])
        successes = list(state.successes)
        trials = list(state.trials)
        successes[k] += y
        trials[k] += 1
        log_sum, comp = state.log_sum, state.comp
        pending_t, pending_c = state.pending_treatment, state.pending_control
        block_s, block_n = state.block_successes, state.block_trials
        if self.method == "universal":
            pa, pb = self.prior
            s, n = state.successes[k], state.trials[k]
            log_sum, comp = neumaier_add(log_sum, comp, math.log((pa + s if y else pb + n - s) / (pa + pb + n)))
        else:
            if k == 0:
                pending_t = (*pending_t, y)
            else:
                pending_c = (*pending_c, y)
            kt, kc = self.block
            while len(pending_t) >= kt and len(pending_c) >= kc:
                treated, control = pending_t[:kt], pending_c[:kc]
                pending_t, pending_c = pending_t[kt:], pending_c[kc:]
                a, b, p = self.plug_in(block_s, block_n)
                log_sum, comp = neumaier_add(log_sum, comp, self.block_log_e(a, b, p, treated, control))
                block_s = (block_s[0] + treated.count(1), block_s[1] + control.count(1))
                block_n = (block_n[0] + kt, block_n[1] + kc)
        new = TwoProportionState((successes[0], successes[1]), (trials[0], trials[1]), log_sum, comp,
                                 pending_t, pending_c, block_s, block_n)
        value = require_finite(self.log_e_value(new), "log e-value")
        decision = state.decision or (REJECT_H0 if value >= self._log_level else None)
        new = replace(new, max_log_e=max(state.max_log_e, value), decision=decision)
        return new, self._output(new, value)

    def is_terminal(self, state: TwoProportionState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> TwoProportionState:
        state = decode_fields(TwoProportionState, data)
        pairs = (state.successes, state.trials, state.block_successes, state.block_trials)
        require_state(all(len(v) == 2 for v in pairs), "two arms")
        require_state(all(0 <= s <= n for s, n in zip_strict(state.successes + state.block_successes,
                                                              state.trials + state.block_trials)),
                      "successes out of range")
        require_state(all(y in (0, 1) for y in state.pending_treatment + state.pending_control),
                      "pending outcomes are 0 or 1")
        if self.method == "grow":
            kt, kc = self.block
            require_state(state.block_trials[0] * kc == state.block_trials[1] * kt
                          and state.block_trials[0] % kt == 0, "completed blocks")
            require_state(state.block_trials[0] + len(state.pending_treatment) == state.trials[0]
                          and state.block_trials[1] + len(state.pending_control) == state.trials[1],
                          "every unit is in a block or waiting for one")
        else:
            require_state(not state.pending_treatment and not state.pending_control
                          and state.block_trials == (0, 0), "universal inference has no blocks")
        return state


# ---------------------------------------------------------------- difference of means


@dataclass(frozen=True)
class ABState:
    n: int = 0
    treated: int = 0
    sum_treated: float = 0.0
    sum_control: float = 0.0
    mean_x: float = 0.0  # pooled running moments, for the regression f
    mean_y: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0
    psi_mean: float = 0.0  # the scores, with the covariate when there is one
    psi_m2: float = 0.0
    plain_mean: float = 0.0  # the scores without the covariate (for variance_ratio)
    plain_m2: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class ABOutput:
    n: int
    means: tuple[float | None, float | None]  # treatment, control (plain averages)
    estimate: float  # mean of the scores: the effect estimate
    lower: float | None  # None before `start` units
    upper: float | None
    variance_ratio: float | None  # with covariate / without; None without a covariate
    decision: str | None


class ABTest(Procedure):
    """Anytime-valid confidence sequence for a difference of means, with covariate adjustment.

    Inputs: the arm (`arm_input`), the outcome (`input`) and, with
    `covariate=True`, a pre-experiment covariate (`covariate_input`) such as
    the same metric before the experiment.  `allocation` is the probability
    that a unit of the two compared arms goes to treatment, as set by the
    randomisation.  Rejects H0: no effect when 0 leaves the interval.  The
    coverage is asymptotic (see the module docstring and robust_mean).
    """

    name = "ab_test"
    version = "1"

    def __init__(self, allocation: float = 0.5, alpha: float = 0.05, *, covariate: bool = False,
                 arms: tuple[int, int] = (1, 0), start: int = 1000, tuned_for: float = 1000.0,
                 stop_on_reject: bool = False, arm_input: str = "arm", input: str = "y",
                 covariate_input: str = "x"):
        if not (0 < allocation < 1):
            raise ValueError("allocation must lie in (0, 1)")
        self.alpha = _check_alpha(alpha)
        if len(arms) != 2 or any(type(a) is not int or a < 0 for a in arms) or arms[0] == arms[1]:
            raise ValueError("arms must be two different non-negative integer labels")
        if type(start) is not int or start < 2:
            raise ValueError("start must be an int, at least 2")
        if not (math.isfinite(tuned_for) and tuned_for >= 1):
            raise ValueError("tuned_for must be a number of observations, at least 1")
        if len({arm_input, input, covariate_input}) != 3:
            raise ValueError("the arm, outcome and covariate need different names")
        self.allocation, self.covariate, self.arms = float(allocation), bool(covariate), (arms[0], arms[1])
        self.start, self.tuned_for = start, float(tuned_for)
        self.stop_on_reject = stop_on_reject
        self.arm_input, self.input, self.covariate_input = arm_input, input, covariate_input
        c = -2.0 * math.log(self.alpha)
        self._rho2 = (c + math.log1p(c)) / self.tuned_for

    def config(self) -> dict:
        return {"allocation": self.allocation, "alpha": self.alpha, "covariate": self.covariate,
                "arms": list(self.arms), "start": self.start, "tuned_for": self.tuned_for,
                "stop_on_reject": self.stop_on_reject, "arm_input": self.arm_input, "input": self.input,
                "covariate_input": self.covariate_input}

    def inputs(self):
        out = {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0),
               self.input: InputContract(self.input)}
        if self.covariate:
            out[self.covariate_input] = InputContract(self.covariate_input)
        return out

    def initial_state(self) -> ABState:
        return ABState()

    def half_width(self, n: int, sd: float) -> float:
        """The asymptotic confidence sequence's half-width (as AsymptoticConfidenceSequence)."""
        v = n * self._rho2 + 1.0
        return sd * math.sqrt(2.0 * v / (n * n * self._rho2) * (0.5 * math.log(v) - math.log(self.alpha)))

    def prediction(self, state: ABState, x_value: float | None) -> tuple[float, float]:
        """f_{t-1}(x) with and without the covariate, from earlier units only."""
        if state.n == 0:
            return 0.0, 0.0
        plain = state.mean_y
        if x_value is None or state.sxx <= 0.0:
            return plain, plain
        return plain + state.sxy / state.sxx * (x_value - state.mean_x), plain

    def _output(self, state: ABState) -> ABOutput:
        n = state.n
        control = n - state.treated
        means = (state.sum_treated / state.treated if state.treated else None,
                 state.sum_control / control if control else None)
        lower = upper = None
        if n >= self.start and state.psi_m2 > 0.0:
            h = self.half_width(n, math.sqrt(state.psi_m2 / (n - 1)))
            lower, upper = state.psi_mean - h, state.psi_mean + h
        ratio = None
        if self.covariate and n >= 2 and state.plain_m2 > 0.0:
            ratio = state.psi_m2 / state.plain_m2
        return ABOutput(n, means, state.psi_mean, lower, upper, ratio, state.decision)

    def step(self, state: ABState, x, rng=None):
        label = x[self.arm_input]
        if label not in self.arms:
            return state, self._output(state)
        treated = label == self.arms[0]
        y = x[self.input]
        xv = x[self.covariate_input] if self.covariate else None
        weight = 1.0 / self.allocation if treated else -1.0 / (1.0 - self.allocation)
        fitted, plain_fit = self.prediction(state, xv)
        psi, plain = weight * (y - fitted), weight * (y - plain_fit)
        n = state.n + 1
        d = psi - state.psi_mean
        psi_mean = state.psi_mean + d / n
        psi_m2 = state.psi_m2 + d * (psi - psi_mean)
        d0 = plain - state.plain_mean
        plain_mean = state.plain_mean + d0 / n
        plain_m2 = state.plain_m2 + d0 * (plain - plain_mean)
        dy = y - state.mean_y
        mean_y = state.mean_y + dy / n
        mean_x, sxx, sxy = state.mean_x, state.sxx, state.sxy
        if xv is not None:
            dx = xv - state.mean_x
            mean_x = state.mean_x + dx / n
            sxx = state.sxx + dx * (xv - mean_x)
            sxy = state.sxy + dx * (y - mean_y)
        new = ABState(n, state.treated + treated, state.sum_treated + (y if treated else 0.0),
                      state.sum_control + (0.0 if treated else y), mean_x, mean_y, sxx, sxy,
                      require_finite(psi_mean, "effect estimate"), psi_m2, plain_mean, plain_m2, state.decision)
        out = self._output(new)
        excluded = out.lower is not None and out.upper is not None and not out.lower <= 0.0 <= out.upper
        if new.decision is None and excluded:
            new = replace(new, decision=REJECT_H0)
            out = self._output(new)
        return new, out

    def is_terminal(self, state: ABState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> ABState:
        state = decode_fields(ABState, data)
        require_state(0 <= state.treated <= state.n, "treated count out of range")
        require_state(state.sxx >= 0 and state.psi_m2 >= 0 and state.plain_m2 >= 0, "negative sum of squares")
        return state


# ---------------------------------------------------------------- many tests


def _merge(a: InputContract | None, b: InputContract) -> InputContract:
    """One contract satisfying two: the same kind and shape, the narrower bounds."""
    if a is None:
        return b
    if (a.kind, a.shape) != (b.kind, b.shape):
        raise ValueError(f"tests disagree on the kind or shape of input {a.name!r}")
    lowers = [v for v in (a.lower, b.lower) if v is not None]
    uppers = [v for v in (a.upper, b.upper) if v is not None]
    return replace(a, lower=max(lowers) if lowers else None, upper=min(uppers) if uppers else None)


def e_bh(log_e_values: Sequence[float], alpha: float) -> list[int]:
    """Indices rejected by the e-BH procedure at level alpha, from log e-values.

    With K e-values, the k largest are rejected for the largest k such
    that the k-th largest e-value is at least K / (k alpha); none if there
    is no such k.  FDR <= alpha under arbitrary dependence (Wang and Ramdas 2022).
    """
    alpha = _check_alpha(alpha)
    count = len(log_e_values)
    if any(math.isnan(v) for v in log_e_values):
        raise ValueError("log e-values must not be NaN")
    order = sorted(range(count), key=lambda i: (-log_e_values[i], i))
    best = 0
    for k in range(1, count + 1):
        if log_e_values[order[k - 1]] >= math.log(count) - math.log(k) - math.log(alpha):
            best = k
    return sorted(order[:best])


@dataclass(frozen=True)
class MultipleTestsState:
    n: int
    states: tuple[Any, ...]
    log_e_values: tuple[float, ...]


@dataclass(frozen=True)
class MultipleTestsOutput:
    n: int
    log_e_values: dict
    rejected: tuple[str, ...]  # e-BH at level alpha, in the order the tests were given
    results: dict  # each test's own output


class MultipleTests(Procedure):
    """Several anytime-valid tests on the same observations, with e-BH false discovery rate control.

    `tests` maps a name to a procedure with an e-process (one that declares
    `e_value_field`: TwoProportionTest, SampleRatioMismatch,
    NormalMixtureSPRT, TMixtureSPRT, BettingMeanTest, SequentialSignTest,
    QuantileTest, KernelTwoSampleTest).  A test that has stopped keeps its
    last e-value.
    """

    name = "multiple_tests"
    version = "1"

    def __init__(self, tests: Mapping[str, Procedure], alpha: float = 0.05):
        if not tests:
            raise ValueError("at least one test is needed")
        self.alpha = _check_alpha(alpha)
        self.names = tuple(tests)
        self.tests = tuple(tests[k] for k in self.names)
        contracts: dict[str, InputContract] = {}
        for name, test in zip_strict(self.names, self.tests):
            if not isinstance(getattr(test, "e_value_field", None), str):
                raise ValueError(f"test {name!r} ({type(test).__name__}) does not report an e-process")
            if not getattr(test, "joint_inputs", True):
                raise ValueError(f"test {name!r} takes its inputs one at a time; run it separately")
            for key, contract in test.inputs().items():
                contracts[key] = _merge(contracts.get(key), contract)
        self._contracts = contracts
        self._fields = tuple(str(getattr(t, "e_value_field")) for t in self.tests)  # noqa: B009
        self.randomized = any(getattr(t, "randomized", False) for t in self.tests)

    @classmethod
    def versus_control(cls, treatments: Sequence[int], control: int = 0, *, alpha: float = 0.05,
                       test: Callable[[int, int], Procedure] | None = None) -> MultipleTests:
        """Each treatment arm against the control: `test(treatment, control)`, two proportions by default."""
        def default(t: int, c: int) -> Procedure:
            return TwoProportionTest(arms=(t, c))

        make = test or default
        return cls({f"{t}-vs-{control}": make(t, control) for t in treatments}, alpha)

    def tests_by_name(self) -> dict[str, Procedure]:
        return dict(zip_strict(self.names, self.tests))

    def config(self) -> dict:
        return {"alpha": self.alpha, "tests": [[k, t.identity()] for k, t in zip_strict(self.names, self.tests)]}

    def inputs(self):
        return dict(self._contracts)

    def initial_state(self) -> MultipleTestsState:
        return MultipleTestsState(0, tuple(t.initial_state() for t in self.tests), (0.0,) * len(self.tests))

    def step(self, state: MultipleTestsState, x, rng=None):
        states: list[Any] = []
        logs: list[float] = []
        results: dict[str, Any] = {}
        rows = zip_strict(self.names, self.tests, self._fields, state.states, state.log_e_values)
        for name, test, field, s, e in rows:
            if test.is_terminal(s):
                states.append(s)
                logs.append(e)
                results[name] = None
                continue
            s, out = test.step(s, {k: x[k] for k in test.inputs()}, rng)
            encoded = test.encode_output(out)
            states.append(s)
            logs.append(float(encoded[field]))
            results[name] = encoded
        rejected = tuple(self.names[i] for i in e_bh(logs, self.alpha))
        new = MultipleTestsState(state.n + 1, tuple(states), tuple(logs))
        return new, MultipleTestsOutput(new.n, dict(zip_strict(self.names, logs)), rejected, results)

    def is_terminal(self, state: MultipleTestsState) -> bool:
        return all(t.is_terminal(s) for t, s in zip_strict(self.tests, state.states))

    def encode_state(self, state: MultipleTestsState):
        return {"n": state.n, "states": [t.encode_state(s) for t, s in zip_strict(self.tests, state.states)],
                "log_e_values": list(state.log_e_values)}

    def encode_output(self, output: MultipleTestsOutput):
        return {"n": output.n, "log_e_values": dict(output.log_e_values), "rejected": list(output.rejected),
                "results": dict(output.results)}

    def decode_state(self, data) -> MultipleTestsState:
        require_state(isinstance(data, dict) and set(data) == {"n", "states", "log_e_values"}, "fields")
        n, states, logs = data["n"], data["states"], data["log_e_values"]
        require_state(type(n) is int and n >= 0, "n")
        require_state(isinstance(states, list) and len(states) == len(self.tests), "one state per test")
        require_state(isinstance(logs, list) and len(logs) == len(self.tests)
                      and all(type(v) in (int, float) and not math.isnan(v) for v in logs), "log e-values")
        return MultipleTestsState(n, tuple(t.decode_state(s) for t, s in zip_strict(self.tests, states)),
                                  tuple(float(v) for v in logs))
