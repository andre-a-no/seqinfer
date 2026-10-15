# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Execute a group sequential test of a normal mean on a stream.

Observations arrive one at a time; the test looks at them only at the
planned analyses n_1 < ... < n_K.  At analysis k it computes

    Z_k = (mean_k - theta0) sqrt(n_k) / sigma

and stops for efficacy (rejects H0) if Z_k >= c_k, or |Z_k| >= c_k for
a two-sided design.  With futility boundaries f_k it stops and accepts H0
if Z_k <= f_k.  If no boundary is crossed by n_K, H0 is accepted (not
rejected).  The boundaries come from `sequential_analysis.group_sequential`.

With `sigma=None` the variance is estimated at each analysis and Z_k is
replaced by the t statistic, compared with the boundary of the same
one-sided significance level, t_{n_k - 1}^{-1}(Phi(c_k)) -- the
significance-level approach of Jennison and Turnbull (2000, sec. 11.4).
The type I error is then approximately, not exactly, the design's.

The schedule is fixed in advance.  If analyses happen at other sample
sizes than planned, the boundaries must be recomputed for the actual
information fractions, which is what alpha spending allows: build a new
design with the realised fractions before the next analysis.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._compat import pairwise, zip_strict
from ..contracts import InputContract
from ..core import Procedure, decode_fields, refuse_terminal, require_state
from ..distributions import _log_normal_upper, t_isf_log
from ..errors import ContractViolation, NumericalError
from ..group_sequential import (
    _NEVER,
    GroupSequentialDesign,
    _efficacy_bounds,
    _futility_bounds,
    spending_function,
)
from ..numerics import require_finite
from .sprt import ACCEPT_H0, REJECT_H0

_UNDERFLOW = (
    "the sample variance underflows to zero although the observations differ; "
    "rescale the data in the adapter (for example to units where they are of order one)"
)


@dataclass(frozen=True)
class GroupSequentialState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations (Welford)
    analysis: int = 0  # analyses done so far
    decision: str | None = None
    low: float | None = None  # smallest and largest observation: tell identical data
    high: float | None = None  # from a variance that underflowed


@dataclass(frozen=True)
class GroupSequentialOutput:
    n: int
    mean: float
    analysis: int | None  # 1-based index if this observation completed an analysis
    z: float | None  # Z_k, or the t statistic when the variance is estimated
    bound: float | None
    decision: str | None


class GroupSequentialTest(Procedure):
    name = "group_sequential_test"
    version = "4"  # 4: t boundaries beyond 2e4 degrees of freedom from the more accurate t tail

    def __init__(
        self,
        sigma: float | None,
        analyses: Sequence[int],
        bounds: Sequence[float],
        theta0: float = 0.0,
        *,
        two_sided: bool = False,
        futility: Sequence[float] | None = None,
        label: str = "",
        input: str = "x",
    ):
        if sigma is not None and not (sigma > 0 and math.isfinite(sigma)):
            raise ValueError("sigma must be positive and finite, or None to estimate it")
        ns = [int(n) for n in analyses]
        if not ns or ns[0] < (1 if sigma is not None else 2) or any(b <= a for a, b in pairwise(ns)):
            raise ValueError("analyses must be increasing positive sample sizes")
        if len(bounds) != len(ns) or any(not b > 0 for b in bounds):
            raise ValueError("need one positive boundary per analysis")
        self.sigma = None if sigma is None else float(sigma)
        self.theta0, self.two_sided = float(theta0), two_sided
        if futility is not None:
            if two_sided:
                raise ValueError("futility boundaries are for one-sided tests")
            if len(futility) != len(ns) or any(not f <= b for f, b in zip_strict(futility, bounds)):
                raise ValueError("need one futility boundary per analysis, none above its efficacy boundary")
        self.analyses, self.bounds = tuple(ns), tuple(float(b) for b in bounds)
        self.futility = None if futility is None else tuple(float(f) for f in futility)
        self.label, self.input = label, input
        self._bounds = tuple(self._converted(b, n) for b, n in zip_strict(self.bounds, ns))
        self._futility = (
            None if self.futility is None
            else tuple(self._converted(f, n) for f, n in zip_strict(self.futility, ns))
        )

    @classmethod
    def from_design(
        cls, design, n_max: int, sigma: float | None, theta0: float = 0.0, **kwargs
    ) -> GroupSequentialTest:
        """Analyses at the design's information fractions of n_max (rounded up)."""
        analyses = [math.ceil(t * n_max - 1e-9) for t in design.fractions]
        return cls(
            sigma, analyses, design.bounds, theta0, two_sided=design.two_sided, futility=design.futility, **kwargs
        )

    def config(self) -> dict:
        return {
            "sigma": self.sigma,
            "theta0": self.theta0,
            "analyses": list(self.analyses),
            "bounds": list(self.bounds),
            "two_sided": self.two_sided,
            "futility": None if self.futility is None else list(self.futility),
            "label": self.label,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> GroupSequentialState:
        return GroupSequentialState()

    def _converted(self, z: float, n: int) -> float:
        """A boundary for Z, as a boundary for the statistic actually used at sample size n."""
        if self.sigma is not None or not math.isfinite(z):
            return z
        # The boundary of the same one-sided significance level, found on the smaller tail and in
        # logarithms: Phi(z) rounds to 1 above z = 8.3, and the tail itself underflows above z = 38.5.
        t = t_isf_log(_log_normal_upper(abs(z)), n - 1)
        return math.copysign(t, z)

    def bound_at(self, k: int) -> float:
        """Efficacy boundary for the statistic at analysis k (0-based)."""
        return self._bounds[k]

    def step(self, state: GroupSequentialState, x, rng=None):
        refuse_terminal(self, state)
        v = x[self.input]
        n = state.n + 1
        delta = v - state.mean
        mean = require_finite(state.mean + delta / n, "running mean")
        m2 = require_finite(state.m2 + delta * (v - mean), "sum of squared deviations")
        low = v if state.low is None else min(state.low, v)
        high = v if state.high is None else max(state.high, v)
        k = state.analysis
        if n != self.analyses[k]:
            new = GroupSequentialState(n, mean, m2, k, None, low, high)
            return new, GroupSequentialOutput(n, mean, None, None, None, None)
        scale = self.sigma if self.sigma is not None else math.sqrt(m2 / (n - 1))
        if scale > 0.0:
            z = (mean - self.theta0) * math.sqrt(n) / scale
        elif low != high:
            raise NumericalError(_UNDERFLOW)
        else:  # identical observations: t is infinite unless they sit exactly at theta0
            z = 0.0 if mean == self.theta0 else math.copysign(math.inf, mean - self.theta0)
        bound = self.bound_at(k)
        crossed = abs(z) >= bound if self.two_sided else z >= bound
        futile = self._futility is not None and z <= self._futility[k]
        last = k + 1 == len(self.analyses)
        decision = REJECT_H0 if crossed else ACCEPT_H0 if futile or last else None
        new = GroupSequentialState(n, mean, m2, k + 1, decision, low, high)
        reported = z if math.isfinite(z) else None  # JSON has no infinity; the decision says it all
        return new, GroupSequentialOutput(n, mean, k + 1, reported, bound if math.isfinite(bound) else None, decision)

    def is_terminal(self, state: GroupSequentialState) -> bool:
        return state.decision is not None

    def decode_state(self, data) -> GroupSequentialState:
        state = decode_fields(GroupSequentialState, data)
        require_state(state.m2 >= 0, "negative sum of squares")
        return state


# ---------------------------------------------------------------- any endpoint: Z and information


def _spending_spec(spending) -> Any:
    """A spending function as JSON: a name or ["power", rho]; a callable has no identity to record."""
    if isinstance(spending, str):
        spending_function(spending, 0.025)  # refuses unknown names
        return spending
    if isinstance(spending, (tuple, list)) and len(spending) == 2 and spending[0] == "power":
        rho = spending[1]
        if type(rho) not in (int, float) or not (math.isfinite(rho) and rho > 0):
            raise ValueError("the exponent of a power spending function must be positive and finite")
        return ["power", float(rho)]
    raise ValueError("spending must be 'obrien-fleming', 'pocock' or ('power', rho) (a callable has no identity)")


def _function(spec) -> Any:
    return tuple(spec) if isinstance(spec, list) else spec


@dataclass(frozen=True)
class GroupSequentialMonitorState:
    fractions: tuple[float, ...] = ()  # realised information over the planned maximum, by analysis
    bounds: tuple[float, ...] = ()  # efficacy boundary of each analysis; 40.0 stands for "none"
    futility: tuple[float, ...] = ()  # futility boundary of each analysis (empty without futility)
    decision: str | None = None


@dataclass(frozen=True)
class GroupSequentialMonitorOutput:
    analysis: int
    fraction: float
    z: float
    bound: float | None  # None: no alpha spent at this analysis
    futility: float | None
    spent: float  # cumulative alpha (one side) by this analysis
    final: bool
    decision: str | None


class GroupSequentialMonitor(Procedure):
    """Alpha spending at the information actually reached, for any endpoint (Lan and DeMets 1983).

    Each input is one analysis: the standardised statistic Z_k and the
    information I_k it is based on (for a difference of means
    1 / Var(estimate); for a log-rank statistic its variance, about
    events / 4 with 1:1 allocation; any statistic whose increments are
    approximately independent with Var Z_k sqrt(I_k) = I_k).  The
    boundary of analysis k is computed when it happens, from the spending
    function at t_k = I_k / max_information and the boundaries already
    used, so analyses need not be planned: a data monitoring committee can
    meet when it meets.  The analysis is final when I_k reaches
    `max_information`, when `max_analyses` is reached, or when the record
    says so (`final_input`, if given); the final analysis spends all the
    alpha left (under- and over-running).  Boundaries are those of
    `GroupSequentialDesign.at`, which the tests check against rpact.

    Futility boundaries (one-sided only) need the beta-spending function,
    the design's drift theta sqrt(I_max) and its power: give them, or use
    `from_design`.  They are non-binding, as in the design.

    Cost per analysis: a recursive numerical integration over the
    analyses so far, about 10 ms per analysis at the third and 0.1 s at
    the tenth (CPython); the state grows by three numbers per analysis,
    at most `max_analyses`.
    """

    name = "group_sequential_monitor"
    version = "1"

    def __init__(self, alpha: float, max_information: float, spending="obrien-fleming", *, two_sided: bool = False,
                 futility=None, power: float | None = None, drift: float | None = None, max_analyses: int = 20,
                 points_per_sd: int = 24, input: str = "z", information_input: str = "information",
                 final_input: str | None = None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if type(max_information) not in (int, float) or not (math.isfinite(max_information) and max_information > 0):
            raise ValueError("max_information must be positive and finite")
        if type(max_analyses) is not int or not 1 <= max_analyses <= 100:
            raise ValueError("max_analyses must be an int in [1, 100]")
        if type(points_per_sd) is not int or not 8 <= points_per_sd <= 200:
            raise ValueError("points_per_sd must be an int in [8, 200]")
        side = alpha / 2.0 if two_sided else alpha
        if side >= 0.5:
            raise ValueError("a one-sided alpha must be below 0.5")
        self.alpha, self.max_information, self.two_sided = float(alpha), float(max_information), bool(two_sided)
        self.spending = _spending_spec(spending)
        self.futility_spending = None if futility is None else _spending_spec(futility)
        if futility is not None:
            if two_sided:
                raise ValueError("futility boundaries are for one-sided tests")
            if power is None or not alpha < power < 1 or drift is None or not (math.isfinite(drift) and drift > 0):
                raise ValueError("futility boundaries need the power and the drift of the design (see from_design)")
        self.power = None if futility is None else float(power)  # type: ignore[arg-type]
        self.drift = None if futility is None else float(drift)  # type: ignore[arg-type]
        self.max_analyses, self.points_per_sd = max_analyses, points_per_sd
        names = [input, information_input] + ([final_input] if final_input is not None else [])
        if len(set(names)) != len(names):
            raise ValueError("the inputs need different names")
        self.input, self.information_input, self.final_input = input, information_input, final_input
        self._side = side
        self._alpha_spend = spending_function(_function(self.spending), side)
        self._beta_spend = None if self.futility_spending is None else spending_function(
            _function(self.futility_spending), 1.0 - self.power)  # type: ignore[operator]

    @classmethod
    def from_design(cls, design: GroupSequentialDesign, max_information: float, **kwargs) -> GroupSequentialMonitor:
        """The monitor of a design: its alpha, spending functions, sidedness and, with futility, power and drift."""
        if design.spending is None:
            raise ValueError("this design does not carry its spending function; build it with group_sequential_design")
        extra: dict = {}
        if design.futility is not None:
            assert design.beta_spent is not None
            extra = {"futility": design.futility_spending, "power": 1.0 - design.beta_spent[-1], "drift": design.drift}
        return cls(design.alpha, max_information, design.spending, two_sided=design.two_sided,
                   points_per_sd=design.points_per_sd, **extra, **kwargs)

    def config(self) -> dict:
        return {"alpha": self.alpha, "max_information": self.max_information, "spending": self.spending,
                "two_sided": self.two_sided, "futility": self.futility_spending, "power": self.power,
                "drift": self.drift, "max_analyses": self.max_analyses, "points_per_sd": self.points_per_sd,
                "input": self.input, "information_input": self.information_input, "final_input": self.final_input}

    def inputs(self):
        contracts = {self.input: InputContract(self.input),
                     self.information_input: InputContract(self.information_input, lower=0.0)}
        if self.final_input is not None:
            contracts[self.final_input] = InputContract(self.final_input, kind="binary")
        return contracts

    def initial_state(self) -> GroupSequentialMonitorState:
        return GroupSequentialMonitorState()

    def boundaries(self, fractions: Sequence[float], final: bool = True) -> tuple[list[float], list[float]]:
        """Efficacy and futility boundaries for analyses at these fractions; the last one final or not.

        Futility boundaries are empty without a beta-spending function.
        """
        ts = list(fractions)
        cumulative = [self._alpha_spend(min(u, 1.0)) for u in ts]
        if final:
            cumulative[-1] = self._side
        bounds = _efficacy_bounds(ts, cumulative, self.two_sided, self.points_per_sd)
        if self._beta_spend is None:
            return bounds, []
        assert self.power is not None and self.drift is not None
        finite = [b if math.isfinite(b) else _NEVER for b in bounds]
        beta_cumulative = [self._beta_spend(min(u, 1.0)) for u in ts]
        if final:
            beta_cumulative[-1] = 1.0 - self.power
            fs, _ = _futility_bounds(ts, finite, beta_cumulative, self.drift, self.points_per_sd)
            return bounds, [*fs, finite[-1]]
        # an interim analysis: a futility boundary for every analysis given, so close with a dummy final one
        fs, _ = _futility_bounds([*ts, ts[-1] + 1.0], [*finite, _NEVER], [*beta_cumulative, beta_cumulative[-1]],
                                 self.drift, self.points_per_sd)
        return bounds, fs

    def step(self, state: GroupSequentialMonitorState, x, rng=None):
        refuse_terminal(self, state)
        z, information = x[self.input], x[self.information_input]
        t = information / self.max_information
        previous = state.fractions[-1] if state.fractions else 0.0
        if not t > previous:
            raise ContractViolation(f"{self.name}: information must grow from one analysis to the next")
        k = len(state.fractions)
        final = t >= 1.0 or k + 1 == self.max_analyses or (
            self.final_input is not None and x[self.final_input] == 1)
        fractions = (*state.fractions, t)
        bounds, fs = self.boundaries(fractions, final)
        b = bounds[-1]
        f = fs[-1] if fs else None
        spent = self._side if final else self._alpha_spend(t)
        crossed = (abs(z) >= b) if self.two_sided else z >= b
        futile = f is not None and z <= f
        decision = REJECT_H0 if crossed else ACCEPT_H0 if futile or final else None
        new = GroupSequentialMonitorState(fractions, (*state.bounds, b if math.isfinite(b) else _NEVER),
                                          (*state.futility, f) if f is not None else state.futility, decision)
        return new, GroupSequentialMonitorOutput(k + 1, t, z, b if math.isfinite(b) else None, f, spent, final,
                                                 decision)

    def is_terminal(self, state: GroupSequentialMonitorState) -> bool:
        return state.decision is not None

    def decode_state(self, data) -> GroupSequentialMonitorState:
        state = decode_fields(GroupSequentialMonitorState, data)
        k = len(state.fractions)
        require_state(len(state.bounds) == k and len(state.futility) in (0, k), "one boundary per analysis")
        steps = (0.0, *state.fractions)
        require_state(all(b > a for a, b in pairwise(steps)), "fractions must grow")
        require_state(k <= self.max_analyses, "more analyses than max_analyses")
        return state


# ---------------------------------------------------------------- two arms: proportions and log-rank

_DIRECTIONS = ("greater", "less")


def _analyses_and_bounds(design: GroupSequentialDesign, maximum: int, analyses: Sequence[int] | None):
    """Analyses (counts) and boundaries of a design for a maximum count; at realised counts if given."""
    if analyses is None:
        counts = [math.ceil(t * maximum - 1e-9) for t in design.fractions]
        return counts, design.bounds, design.futility
    counts = [int(n) for n in analyses]
    realised = design.at([n / maximum for n in counts])
    return counts, realised.bounds, realised.futility


class _TwoArmGroupSequential(Procedure):
    """Analyses at counts n_1 < ... < n_K of units (or events) and the boundaries for Z there."""

    def _setup(self, analyses, bounds, futility, two_sided, alternative, arms, arm_input, input, label):
        ns = [int(n) for n in analyses]
        if not ns or ns[0] < 1 or any(b <= a for a, b in pairwise(ns)):
            raise ValueError("analyses must be increasing positive counts")
        if len(bounds) != len(ns) or any(not b > 0 for b in bounds):
            raise ValueError("need one positive boundary per analysis")
        if futility is not None:
            if two_sided:
                raise ValueError("futility boundaries are for one-sided tests")
            if len(futility) != len(ns) or any(not f <= b for f, b in zip_strict(futility, bounds)):
                raise ValueError("need one futility boundary per analysis, none above its efficacy boundary")
        if alternative not in _DIRECTIONS:
            raise ValueError(f"alternative must be one of {_DIRECTIONS} (the direction of a one-sided test, "
                             f"and the sign of Z of a two-sided one)")
        if len(arms) != 2 or any(type(a) is not int or a < 0 for a in arms) or arms[0] == arms[1]:
            raise ValueError("arms must be two different non-negative integer labels (treatment, control)")
        if arm_input == input:
            raise ValueError("the arm and the outcome need different names")
        self.analyses, self.bounds = tuple(ns), tuple(min(float(b), _NEVER) for b in bounds)
        self.futility = None if futility is None else tuple(float(f) for f in futility)
        self.two_sided, self.alternative = bool(two_sided), alternative
        self.arms = (arms[0], arms[1])
        self.arm_input, self.input, self.label = arm_input, input, label

    def _common_config(self) -> dict:
        return {"analyses": list(self.analyses), "bounds": list(self.bounds), "two_sided": self.two_sided,
                "futility": None if self.futility is None else list(self.futility), "alternative": self.alternative,
                "arms": list(self.arms), "arm_input": self.arm_input, "input": self.input, "label": self.label}

    def _decide(self, k: int, z: float) -> str | None:
        b = self.bounds[k]
        crossed = abs(z) >= b if self.two_sided else z >= b
        futile = self.futility is not None and z <= self.futility[k]
        return REJECT_H0 if crossed else ACCEPT_H0 if futile or k + 1 == len(self.analyses) else None

    def is_terminal(self, state) -> bool:
        return state.decision is not None


@dataclass(frozen=True)
class GroupSequentialProportionsState:
    n_treatment: int = 0
    successes_treatment: int = 0
    n_control: int = 0
    successes_control: int = 0
    analysis: int = 0
    decision: str | None = None


@dataclass(frozen=True)
class GroupSequentialProportionsOutput:
    n: int
    rate_treatment: float | None
    rate_control: float | None
    analysis: int | None  # 1-based index if this unit completed an analysis
    z: float | None
    bound: float | None
    decision: str | None


class GroupSequentialProportions(_TwoArmGroupSequential):
    """Group sequential comparison of two proportions (conversion, response, adverse event rates).

    Each input is one unit: its arm and its binary outcome.  At analysis k
    (after n_k units of both arms together) the pooled two-proportion
    statistic

        Z_k = (p_T - p_C) / sqrt(p (1 - p) (1/n_T + 1/n_C)),   p the pooled rate,

    (whose square is the chi-square test without continuity correction)
    is compared with the boundary: alternative="greater" rejects for large
    p_T - p_C, "less" for small (Z is then computed with the sign
    reversed); a two-sided design compares |Z|.  While an arm is empty or
    every outcome so far is the same, Z = 0.  Units of other arms are
    ignored.  Boundaries come from a design (`from_design`, which also
    recomputes them for analyses at other counts than planned) or are
    given; `GroupSequentialDesign.max_sample_size_two_proportions` gives
    the maximum sample size.  The type I error is that of the normal
    approximation: close to the design's for the sample sizes such
    designs give (simulated in the tests).

    Cost per step: a few additions; at an analysis a square root.  The
    state is six numbers.
    """

    name = "group_sequential_proportions"
    version = "1"

    def __init__(self, analyses: Sequence[int], bounds: Sequence[float], *, futility: Sequence[float] | None = None,
                 two_sided: bool = False, alternative: str = "greater", arms: tuple[int, int] = (1, 0),
                 arm_input: str = "arm", input: str = "y", label: str = ""):
        self._setup(analyses, bounds, futility, two_sided, alternative, arms, arm_input, input, label)

    @classmethod
    def from_design(cls, design: GroupSequentialDesign, n_max: int, *, analyses: Sequence[int] | None = None,
                    **kwargs) -> GroupSequentialProportions:
        """Analyses at the design's fractions of n_max units, or at the counts `analyses` (boundaries recomputed)."""
        counts, bounds, futility = _analyses_and_bounds(design, n_max, analyses)
        return cls(counts, bounds, futility=futility, two_sided=design.two_sided, **kwargs)

    def config(self) -> dict:
        return self._common_config()

    def inputs(self):
        return {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0),
                self.input: InputContract(self.input, kind="binary")}

    def initial_state(self) -> GroupSequentialProportionsState:
        return GroupSequentialProportionsState()

    def statistic(self, state: GroupSequentialProportionsState) -> float:
        n_t, n_c = state.n_treatment, state.n_control
        if n_t == 0 or n_c == 0:
            return 0.0
        pooled = (state.successes_treatment + state.successes_control) / (n_t + n_c)
        if pooled <= 0.0 or pooled >= 1.0:
            return 0.0
        diff = state.successes_treatment / n_t - state.successes_control / n_c
        z = diff / math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_t + 1.0 / n_c))
        return -z if self.alternative == "less" else z

    def step(self, state: GroupSequentialProportionsState, x, rng=None):
        refuse_terminal(self, state)
        label = x[self.arm_input]
        rates = self._rates(state)
        if label not in self.arms:
            return state, GroupSequentialProportionsOutput(state.n_treatment + state.n_control, *rates,
                                                           None, None, None, None)
        y = x[self.input]
        if label == self.arms[0]:
            state = GroupSequentialProportionsState(state.n_treatment + 1, state.successes_treatment + y,
                                                    state.n_control, state.successes_control, state.analysis)
        else:
            state = GroupSequentialProportionsState(state.n_treatment, state.successes_treatment,
                                                    state.n_control + 1, state.successes_control + y, state.analysis)
        n = state.n_treatment + state.n_control
        k = state.analysis
        rates = self._rates(state)
        if n != self.analyses[k]:
            return state, GroupSequentialProportionsOutput(n, *rates, None, None, None, None)
        z = self.statistic(state)
        decision = self._decide(k, z)
        new = GroupSequentialProportionsState(state.n_treatment, state.successes_treatment, state.n_control,
                                              state.successes_control, k + 1, decision)
        return new, GroupSequentialProportionsOutput(n, *rates, k + 1, z, self.bounds[k], decision)

    @staticmethod
    def _rates(state: GroupSequentialProportionsState) -> tuple[float | None, float | None]:
        return (state.successes_treatment / state.n_treatment if state.n_treatment else None,
                state.successes_control / state.n_control if state.n_control else None)

    def decode_state(self, data) -> GroupSequentialProportionsState:
        state = decode_fields(GroupSequentialProportionsState, data)
        require_state(state.successes_treatment <= state.n_treatment and state.successes_control <= state.n_control,
                      "more successes than units")
        require_state(state.analysis <= len(self.analyses), "more analyses than planned")
        return state


@dataclass(frozen=True)
class GroupSequentialTwoMeansState:
    n_treatment: int = 0
    mean_treatment: float = 0.0
    m2_treatment: float = 0.0  # sum of squared deviations (Welford)
    n_control: int = 0
    mean_control: float = 0.0
    m2_control: float = 0.0
    analysis: int = 0
    decision: str | None = None


@dataclass(frozen=True)
class GroupSequentialTwoMeansOutput:
    n: int
    mean_treatment: float | None
    mean_control: float | None
    analysis: int | None  # 1-based index if this unit completed an analysis
    z: float | None  # Z, or the pooled t statistic when the variance is estimated
    bound: float | None  # the boundary for that statistic
    decision: str | None


class GroupSequentialTwoMeans(_TwoArmGroupSequential):
    """Group sequential comparison of two means: the most common trial, a continuous endpoint in two arms.

    Each input is one unit: its arm and its outcome.  At analysis k (after
    n_k units of both arms together)

        Z_k = (mean_T - mean_C) / (sigma sqrt(1/n_T + 1/n_C))

    with a known `sigma`; with sigma=None the pooled two-sample t
    statistic (Student's, n - 2 degrees of freedom; checked against R's
    t.test in the tests) is compared with the boundary of the same
    one-sided significance level, as GroupSequentialTest does for one
    mean (Jennison and Turnbull 2000, sec. 11.4: approximately the
    design's type I error).  alternative="greater" rejects for a large
    mean_T - mean_C, "less" for a small one; a two-sided design compares
    |Z|.  While an arm is empty (or, with an estimated variance, fewer
    than three units are in), Z = 0.  Units of other arms are ignored.
    `GroupSequentialDesign.max_sample_size_two_means` gives the maximum
    sample size.

    Cost per step: a Welford update; the state is eight numbers.
    """

    name = "group_sequential_two_means"
    version = "1"

    def __init__(self, sigma: float | None, analyses: Sequence[int], bounds: Sequence[float], *,
                 futility: Sequence[float] | None = None, two_sided: bool = False, alternative: str = "greater",
                 arms: tuple[int, int] = (1, 0), arm_input: str = "arm", input: str = "y", label: str = ""):
        if sigma is not None and (type(sigma) not in (int, float) or not (math.isfinite(sigma) and sigma > 0)):
            raise ValueError("sigma must be positive and finite, or None to estimate it")
        self._setup(analyses, bounds, futility, two_sided, alternative, arms, arm_input, input, label)
        if sigma is None and self.analyses[0] < 3:
            raise ValueError("with an estimated variance the first analysis needs at least three units")
        self.sigma = None if sigma is None else float(sigma)
        self._bounds = tuple(self._converted(b, n) for b, n in zip_strict(self.bounds, self.analyses))
        self._futility = (None if self.futility is None
                          else tuple(self._converted(f, n) for f, n in zip_strict(self.futility, self.analyses)))

    @classmethod
    def from_design(cls, design: GroupSequentialDesign, n_max: int, sigma: float | None, *,
                    analyses: Sequence[int] | None = None, **kwargs) -> GroupSequentialTwoMeans:
        """Analyses at the design's fractions of n_max units, or at the counts `analyses` (boundaries recomputed)."""
        counts, bounds, futility = _analyses_and_bounds(design, n_max, analyses)
        return cls(sigma, counts, bounds, futility=futility, two_sided=design.two_sided, **kwargs)

    def config(self) -> dict:
        return {**self._common_config(), "sigma": self.sigma}

    def inputs(self):
        return {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0),
                self.input: InputContract(self.input)}

    def initial_state(self) -> GroupSequentialTwoMeansState:
        return GroupSequentialTwoMeansState()

    def _converted(self, z: float, n: int) -> float:
        """A boundary for Z as a boundary for the statistic used at n units: t with n - 2 degrees of freedom."""
        if self.sigma is not None or z >= _NEVER:
            return z
        return math.copysign(t_isf_log(_log_normal_upper(abs(z)), n - 2), z)

    def bound_at(self, k: int) -> float:
        """Efficacy boundary for the statistic at analysis k (0-based)."""
        return self._bounds[k]

    def statistic(self, state: GroupSequentialTwoMeansState) -> float:
        n_t, n_c = state.n_treatment, state.n_control
        if n_t == 0 or n_c == 0:
            return 0.0
        diff = state.mean_treatment - state.mean_control
        if self.sigma is not None:
            scale = self.sigma
        else:
            if n_t + n_c < 3:
                return 0.0
            scale = math.sqrt((state.m2_treatment + state.m2_control) / (n_t + n_c - 2))
            if scale == 0.0:  # every outcome equal within the arms: the difference is certain, or none
                return 0.0 if diff == 0.0 else math.copysign(math.inf, diff if self.alternative == "greater" else -diff)
        z = diff / (scale * math.sqrt(1.0 / n_t + 1.0 / n_c))
        return -z if self.alternative == "less" else z

    def step(self, state: GroupSequentialTwoMeansState, x, rng=None):
        refuse_terminal(self, state)
        label = x[self.arm_input]
        if label in self.arms:
            y = x[self.input]
            if label == self.arms[0]:
                n = state.n_treatment + 1
                delta = y - state.mean_treatment
                mean = require_finite(state.mean_treatment + delta / n, "running mean")
                m2 = require_finite(state.m2_treatment + delta * (y - mean), "sum of squared deviations")
                state = GroupSequentialTwoMeansState(n, mean, m2, state.n_control, state.mean_control,
                                                     state.m2_control, state.analysis)
            else:
                n = state.n_control + 1
                delta = y - state.mean_control
                mean = require_finite(state.mean_control + delta / n, "running mean")
                m2 = require_finite(state.m2_control + delta * (y - mean), "sum of squared deviations")
                state = GroupSequentialTwoMeansState(state.n_treatment, state.mean_treatment, state.m2_treatment,
                                                     n, mean, m2, state.analysis)
        total = state.n_treatment + state.n_control
        means = (state.mean_treatment if state.n_treatment else None,
                 state.mean_control if state.n_control else None)
        k = state.analysis
        if label not in self.arms or total != self.analyses[k]:
            return state, GroupSequentialTwoMeansOutput(total, *means, None, None, None, None)
        z = self.statistic(state)
        b = self._bounds[k]
        crossed = abs(z) >= b if self.two_sided else z >= b
        futile = self._futility is not None and z <= self._futility[k]
        decision = REJECT_H0 if crossed else ACCEPT_H0 if futile or k + 1 == len(self.analyses) else None
        new = GroupSequentialTwoMeansState(state.n_treatment, state.mean_treatment, state.m2_treatment,
                                           state.n_control, state.mean_control, state.m2_control, k + 1, decision)
        reported = z if math.isfinite(z) else None  # JSON has no infinity; the decision says it all
        return new, GroupSequentialTwoMeansOutput(total, *means, k + 1, reported, b, decision)

    def decode_state(self, data) -> GroupSequentialTwoMeansState:
        state = decode_fields(GroupSequentialTwoMeansState, data)
        require_state(state.m2_treatment >= 0 and state.m2_control >= 0, "negative sum of squares")
        require_state(state.analysis <= len(self.analyses), "more analyses than planned")
        return state


ENTER, EVENT, CENSOR = 0, 1, 2
_MAX_STRATA = 10_000


@dataclass(frozen=True)
class GroupSequentialLogRankState:
    at_risk: tuple[int, ...] = (0, 0)  # treatment, control; per stratum: n_T^0, n_C^0, n_T^1, n_C^1, ...
    events: int = 0
    observed: int = 0  # events in the treatment arm
    expected: float = 0.0  # sum over events of n_T / (n_T + n_C), within the stratum
    variance: float = 0.0  # sum over events of p (1 - p): the information
    analysis: int = 0
    decision: str | None = None


@dataclass(frozen=True)
class GroupSequentialLogRankOutput:
    events: int
    at_risk: tuple[int, int]
    analysis: int | None  # 1-based index if this event completed an analysis
    z: float | None
    information: float | None
    hazard_ratio_estimate: float | None  # exp((O - E) / V), Peto's one-step estimate
    bound: float | None
    decision: str | None


class GroupSequentialLogRank(_TwoArmGroupSequential):
    """Group sequential log-rank test, with analyses after given numbers of events.

    The input is the stream of AnytimeLogRankTest: each record a patient's
    arm and what happened (0 enters the risk set, 1 has the event,
    2 leaves without one), in the order it happened.  At each event, with
    n_T and n_C at risk (in the event's stratum), p = n_T / (n_T + n_C)
    is added to E and p (1 - p) to V; O counts the events in the
    treatment arm.  At analysis k (after d_k events)

        Z_k = (E - O) / sqrt(V)

    for alternative="less" (the treatment lowers the hazard; Z positive
    when it has fewer events than expected), (O - E) / sqrt(V) for
    "greater"; a two-sided design compares |Z|.  This is the log-rank
    statistic (the score test of a Cox model with the arm as covariate,
    late entry handled as left truncation; checked against R's survival
    package in the tests), analysed at event counts as is usual
    (information fraction d_k / d_max).  With `strata` > 1 each record
    carries its stratum and E, V are summed within strata: the stratified
    log-rank test.  Events with nobody at risk in the other arm add
    nothing.  Records of other arms are ignored.  Tied events are taken
    one at a time in arrival order (a slightly larger V than Breslow's
    tie correction).

    `GroupSequentialDesign.max_events` gives the maximum number of events
    (Schoenfeld).  Cost per step: a few additions; the state is 2 per
    stratum plus five numbers.
    """

    name = "group_sequential_logrank"
    version = "1"

    def __init__(self, analyses: Sequence[int], bounds: Sequence[float], *, futility: Sequence[float] | None = None,
                 two_sided: bool = False, alternative: str = "less", arms: tuple[int, int] = (1, 0),
                 arm_input: str = "arm", input: str = "kind", strata: int = 1, stratum_input: str = "stratum",
                 label: str = ""):
        self._setup(analyses, bounds, futility, two_sided, alternative, arms, arm_input, input, label)
        if type(strata) is not int or not 1 <= strata <= _MAX_STRATA:
            raise ValueError(f"strata must be an int in [1, {_MAX_STRATA}]")
        if strata > 1 and stratum_input in (arm_input, input):
            raise ValueError("the stratum needs a name different from the arm and the kind")
        self.strata, self.stratum_input = strata, stratum_input

    @classmethod
    def from_design(cls, design: GroupSequentialDesign, max_events: int, *, analyses: Sequence[int] | None = None,
                    **kwargs) -> GroupSequentialLogRank:
        """Analyses at the design's fractions of max_events, or at the event counts `analyses`."""
        counts, bounds, futility = _analyses_and_bounds(design, max_events, analyses)
        return cls(counts, bounds, futility=futility, two_sided=design.two_sided, **kwargs)

    def config(self) -> dict:
        return {**self._common_config(), "strata": self.strata, "stratum_input": self.stratum_input}

    def inputs(self):
        contracts = {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0),
                     self.input: InputContract(self.input, kind="integer", lower=ENTER, upper=CENSOR)}
        if self.strata > 1:
            contracts[self.stratum_input] = InputContract(self.stratum_input, kind="integer", lower=0,
                                                          upper=self.strata - 1)
        return contracts

    def initial_state(self) -> GroupSequentialLogRankState:
        return GroupSequentialLogRankState(at_risk=(0, 0) * self.strata)

    def statistic(self, state: GroupSequentialLogRankState) -> float:
        if state.variance <= 0.0:
            return 0.0
        z = (state.expected - state.observed) / math.sqrt(state.variance)
        return -z if self.alternative == "greater" else z

    def _output(self, state: GroupSequentialLogRankState, analysis, z, bound, decision):
        treatment = control = 0  # counts: a loop, as everywhere on the inference path (no builtin sum)
        for i in range(0, len(state.at_risk), 2):
            treatment += state.at_risk[i]
            control += state.at_risk[i + 1]
        totals = (treatment, control)
        estimate = math.exp((state.observed - state.expected) / state.variance) if state.variance > 0 else None
        info = state.variance if analysis is not None else None
        return GroupSequentialLogRankOutput(state.events, totals, analysis, z, info, estimate, bound, decision)

    def step(self, state: GroupSequentialLogRankState, x, rng=None):
        refuse_terminal(self, state)
        label = x[self.arm_input]
        if label not in self.arms:
            return state, self._output(state, None, None, None, None)
        arm = self.arms.index(label)
        stratum = int(x[self.stratum_input]) if self.strata > 1 else 0
        i = 2 * stratum + arm
        kind = x[self.input]
        at_risk = list(state.at_risk)
        if kind == ENTER:
            at_risk[i] += 1
            new = GroupSequentialLogRankState(tuple(at_risk), state.events, state.observed, state.expected,
                                              state.variance, state.analysis)
            return new, self._output(new, None, None, None, None)
        if at_risk[i] == 0:
            what = "an event" if kind == EVENT else "a censoring"
            where = f" of stratum {stratum}" if self.strata > 1 else ""
            raise ContractViolation(f"{self.name}: {what} in arm {label}{where} with nobody at risk")
        n_t, n_c = at_risk[2 * stratum], at_risk[2 * stratum + 1]
        at_risk[i] -= 1
        if kind == CENSOR:
            new = GroupSequentialLogRankState(tuple(at_risk), state.events, state.observed, state.expected,
                                              state.variance, state.analysis)
            return new, self._output(new, None, None, None, None)
        p = n_t / (n_t + n_c)
        new = GroupSequentialLogRankState(tuple(at_risk), state.events + 1, state.observed + (arm == 0),
                                          state.expected + p, state.variance + p * (1.0 - p), state.analysis)
        k = new.analysis
        if new.events != self.analyses[k]:
            return new, self._output(new, None, None, None, None)
        z = self.statistic(new)
        decision = self._decide(k, z)
        new = GroupSequentialLogRankState(new.at_risk, new.events, new.observed, new.expected, new.variance, k + 1,
                                          decision)
        return new, self._output(new, k + 1, z, self.bounds[k], decision)

    def decode_state(self, data) -> GroupSequentialLogRankState:
        state = decode_fields(GroupSequentialLogRankState, data)
        require_state(len(state.at_risk) == 2 * self.strata, "one pair of numbers at risk per stratum")
        require_state(state.observed <= state.events and 0.0 <= state.expected <= state.events, "counts")
        require_state(0.0 <= state.variance <= 0.25 * state.events + 1e-9, "variance")
        require_state(state.analysis <= len(self.analyses), "more analyses than planned")
        return state
