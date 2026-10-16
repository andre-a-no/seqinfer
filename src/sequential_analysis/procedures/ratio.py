# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Continuously monitored A/B tests of ratio metrics and of relative lift.

Many product metrics are ratios whose analysis unit is not the unit of
randomisation.  Revenue per session, click-through rate (clicks per
impression) and average order value (revenue per order) are examples.
Users are randomised, and each user contributes a numerator y_i and a
denominator d_i: their revenue and their number of sessions.  The metric
of an arm is

    R = E[y] / E[d],   estimated by the ratio of sums  R_hat = sum y / sum d

over the arm's users.  Treating the sessions as independent observations
is wrong: one user's sessions are correlated, and the per-session
standard error is too small.  The delta method on the user totals is
correct (Deng, Knoblich and Lu 2018, "Applying the delta method in metric
analytics").  To first order, the error of the ratio of sums in arm k is
the average of the per-user influence scores

    phi_i = (y_i - R d_i) / E[d],

so its variance is estimated by

    V_k = sum_k (y_i - R_hat d_i)^2 / (sum_k d_i)^2 * n_k / (n_k - 1),

with the sum over the arm's n_k users.  This is Deng et al.'s formula
(s_y^2 - 2 R s_yd + R^2 s_d^2) / (n_k dbar^2).  The state holds each arm's
sums and centred second moments of y and d, updated as in Welford's
algorithm, so it has a constant size.

RatioABTest -- confidence sequences for R_T - R_C and for R_T / R_C - 1.
    Over all n users of the two arms, the treatment indicator T_i and
    (y_i, d_i) are independent and identically distributed.  The
    difference of the ratios of sums is then

        R_hat_T - R_hat_C = R_T - R_C + (1/n) sum_i psi_i + r_n,
        psi_i = T_i (y_i - R_T d_i) / E[T d] - C_i (y_i - R_C d_i) / E[C d].

    The psi_i are independent, with mean zero.  The remainder r_n is a
    product of two estimation errors, so by the law of the iterated
    logarithm it is O(log log n / n) almost surely.  The asymptotic
    confidence sequence of Waudby-Smith, Arbour, Sinha, Kennedy and Ramdas
    (2024, "Time-uniform central limit theory and asymptotic confidence
    sequences") covers the mean of such scores uniformly over time.  It is
    the one that AsymptoticConfidenceSequence and ABTest use, and its
    half-width is a function of n and the scores' standard deviation:

        (R_hat_T - R_hat_C) +- half_width(n, sqrt(n (V_T + V_C))),

    since n (V_T + V_C) estimates var(psi).  The remainder shifts the sum
    of the scores by O(log log n), which is less than the strong
    approximation error the asymptotic sequence already allows.  Coverage
    is therefore 1 - alpha asymptotically and uniformly over time, as for
    ABTest.  It is not exact at small n.

    The relative lift L = R_T / R_C - 1 has its sequence on the log scale:

        log(R_hat_T / R_hat_C) +- half_width(n, sqrt(n (V_T / R_hat_T^2 + V_C / R_hat_C^2))),

    mapped back by exp(.) - 1.  The bounds stay above -1 and are not
    symmetric about the estimate.  This needs both ratios positive, and
    no lift is reported otherwise.  The direct delta-method interval
    L_hat +- ... missed far more often in the simulations below, so it is
    not offered.  Monitored from 30 users it missed in 0.11 to 0.35 of the
    runs, against 0.048 to 0.098 on the log scale.

    The test rejects H0 when `effect0` leaves the chosen sequence.  With
    scale="difference" it tests R_T - R_C = effect0; with
    scale="relative" it tests R_T / R_C - 1 = effect0.  The decision then
    stays.  The intervals are not intersected over time, because
    asymptotic coverage does not license that.

    With denominator=False every d_i is 1.  R is then the mean per user,
    and the same procedure gives the relative lift of a mean: of revenue
    per user, for instance, which is how product teams report results.
    The difference is then the usual difference of means with the
    unpooled variance s_T^2 / n_T + s_C^2 / n_C.

    Why not a predictable plug-in?  ABTest centres each unit with a
    regression fitted on earlier units only, so that its scores have
    conditional mean exactly the effect.  No such score exists for a
    ratio: R_T - R_C is not linear in the means.  The one-step scores

        R_T,t-1 - R_C,t-1 + T_t (y_t - R_T,t-1 d_t) / (pi dbar_T,t-1) - ...,

    with plug-ins from the past, have a conditional bias that is a product
    of two estimation errors, of the same order as r_n above.  So the
    predictable form gains no exactness.  It also centres the interval
    away from the ratio of sums that is reported.  In the simulations
    below its time-uniform miscoverage was within 0.013 of the ratio of
    sums' in every cell.  The ratio of sums is used.

    The allocation probability cancels from the ratio of sums, so it is
    not a parameter.  The variance uses the observed arm sizes, which
    allows for chance imbalance.  It must still be fixed for the whole
    run and independent of the outcomes.

    Assumptions: users, the units of randomisation, are independent, and
    each one is counted once, with its totals over the window of the
    analysis.  They are randomised to an arm with a fixed probability
    that does not look at outcomes.  y and d have finite variances, a
    little more than two moments in fact.  d >= 0.  A user with d = 0,
    who had no session, must be kept: dropping them conditions on an
    outcome of the treatment.  Coverage is asymptotic, as for ABTest.

    Measured coverage.  The table gives the probability that the interval
    misses the truth at some n from `start` to 20,000 users, at
    alpha = 0.05, over 1,000 runs per cell.  The Monte Carlo standard
    error is about 0.007.  Sessions are Poisson(2) per user, so 14% of
    users have none.  Revenue per session is lognormal, with a per-user
    multiplier exp(sigma_u Z - 0.1 sessions) shared by the user's
    sessions.  This makes sessions within a user strongly correlated, and
    spend per session depends on the number of sessions.  The treatment
    multiplies spend by 1 (no effect) or 1.1:

        allocation, sigma_u   effect   difference, from 30 / 300 / 1000   lift, from 30 / 300 / 1000
        50/50, 1.0            none     0.028 / 0.028 / 0.020              0.048 / 0.041 / 0.025
                              +10%     0.035 / 0.033 / 0.027              0.060 / 0.052 / 0.033
        50/50, 1.5 (heavy)    none     0.019 / 0.018 / 0.014              0.070 / 0.050 / 0.031
                              +10%     0.020 / 0.019 / 0.016              0.056 / 0.044 / 0.029
        30/70, 1.0            none     0.038 / 0.032 / 0.022              0.058 / 0.041 / 0.027
                              +10%     0.071 / 0.059 / 0.033              0.078 / 0.058 / 0.034
        30/70, 1.5 (heavy)    none     0.046 / 0.044 / 0.035              0.098 / 0.069 / 0.043
                              +10%     0.065 / 0.058 / 0.047              0.096 / 0.065 / 0.045

    From 1,000 users, the default `start`, the miscoverage was at most
    0.047.  Monitored from 30 users, the difference's exceeded alpha only
    at 30/70 (up to 0.071), and the lift's in 7 of the 8 cells (up to
    0.098).  This is the small-sample behaviour of the asymptotic sequence
    (see robust_mean).

    The naive per-session test treats every session as an observation.
    It has the same estimate and the same boundary, but the session-level
    variance.  In the same simulations, with no effect, it rejected in
    0.24 to 0.26 of the runs when monitored from 1,000 users, and in 0.35
    to 0.43 from 30 users.  This test rejected in 0.014 to 0.035 from
    1,000 users, and in 0.019 to 0.046 from 30.

    Not covered:
    - Covariate adjustment of a ratio metric (CUPED for ratios).
    - Randomisation by clusters above the unit, such as households or
      cities, which needs the cluster as the unit.
    - Interference between units.
    - An allocation that changes during the run.
    - Small samples: below a few hundred users, use an exact test on a
      bounded metric (BoundedABTest) or a Poisson rate (RateRatioTest).

Cost per step (PERFORMANCE.md): constant time and state, 7 µs (ABTest with a
covariate: 5 µs on the same machine).  The state is each arm's count, two
sums and three centred second moments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..errors import ContractViolation
from ..numerics import require_finite
from .robust_mean import AsymptoticConfidenceSequence
from .sprt import REJECT_H0

SCALES = ("difference", "relative")


def _put(pair: tuple, k: int, value):
    """The pair with its k-th element replaced."""
    return (value, pair[1]) if k == 0 else (pair[0], value)


@dataclass(frozen=True)
class RatioABState:
    units: tuple[int, int] = (0, 0)  # treatment, control
    sum_y: tuple[float, float] = (0.0, 0.0)
    sum_d: tuple[float, float] = (0.0, 0.0)
    cyy: tuple[float, float] = (0.0, 0.0)  # centred second moments: sum (y - ybar)^2, ...
    cdd: tuple[float, float] = (0.0, 0.0)
    cyd: tuple[float, float] = (0.0, 0.0)
    decision: str | None = None


@dataclass(frozen=True)
class RatioABOutput:
    n: int  # users of the two arms
    units: tuple[int, int]  # treatment, control
    ratios: tuple[float | None, float | None]  # sum y / sum d per arm; None while sum d = 0
    difference: float | None  # R_T - R_C
    std_error: float | None  # its delta-method standard error, sqrt(V_T + V_C)
    lower: float | None  # confidence sequence for the difference; None before `start` users
    upper: float | None
    lift: float | None  # R_T / R_C - 1; None unless both ratios are positive
    lift_lower: float | None  # confidence sequence for the lift
    lift_upper: float | None
    decision: str | None


class RatioABTest(Procedure):
    """Anytime-valid confidence sequences for a ratio metric's difference and relative lift; see the module docstring.

    Inputs: the user's arm (`arm_input`), numerator (`input`) and
    denominator (`denominator_input`, at least 0; leave it out with
    denominator=False to compare means per user).  `arms` = (treatment
    label, control label).  Units of other arms are ignored.  Rejects H0
    when `effect0` leaves the sequence of the chosen `scale`:
    "difference" (R_T - R_C) or "relative" (R_T / R_C - 1).  No interval
    is reported before `start` users.  The coverage is asymptotic (see
    robust_mean).
    """

    name = "ratio_ab_test"
    version = "1"

    def __init__(self, alpha: float = 0.05, *, denominator: bool = True, scale: str = "difference",
                 effect0: float = 0.0, arms: tuple[int, int] = (1, 0), start: int = 1000, tuned_for: float = 1000.0,
                 stop_on_reject: bool = False, arm_input: str = "arm", input: str = "y",
                 denominator_input: str = "d"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        if not math.isfinite(effect0) or (scale == "relative" and effect0 <= -1.0):
            raise ValueError("effect0 must be finite, and above -1 for a relative lift")
        if len(arms) != 2 or any(type(a) is not int or a < 0 for a in arms) or arms[0] == arms[1]:
            raise ValueError("arms must be two different non-negative integer labels")
        names = {arm_input, input} | ({denominator_input} if denominator else set())
        if len(names) != (3 if denominator else 2):
            raise ValueError("the arm, numerator and denominator need different names")
        # the boundary is AsymptoticConfidenceSequence's (it checks start and tuned_for)
        self._cs = AsymptoticConfidenceSequence(alpha, start=start, tuned_for=tuned_for)
        self.alpha, self.start, self.tuned_for = self._cs.alpha, start, self._cs.tuned_for
        self.denominator, self.scale, self.effect0 = bool(denominator), scale, float(effect0)
        self.arms = (arms[0], arms[1])
        self.stop_on_reject, self.arm_input, self.input, self.denominator_input = (
            stop_on_reject, arm_input, input, denominator_input)

    def config(self) -> dict:
        return {"alpha": self.alpha, "denominator": self.denominator, "scale": self.scale, "effect0": self.effect0,
                "arms": list(self.arms), "start": self.start, "tuned_for": self.tuned_for,
                "stop_on_reject": self.stop_on_reject, "arm_input": self.arm_input, "input": self.input,
                "denominator_input": self.denominator_input}

    def inputs(self):
        out = {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0),
               self.input: InputContract(self.input)}
        if self.denominator:
            out[self.denominator_input] = InputContract(self.denominator_input, lower=0.0)
        return out

    def initial_state(self) -> RatioABState:
        return RatioABState()

    def half_width(self, n: int, sd: float) -> float:
        """The asymptotic confidence sequence's half-width for scores of standard deviation sd."""
        return self._cs.half_width(n, sd)

    @staticmethod
    def ratio(state: RatioABState, k: int) -> float | None:
        """The ratio of sums of arm k (0: treatment, 1: control)."""
        return state.sum_y[k] / state.sum_d[k] if state.sum_d[k] > 0.0 else None

    @staticmethod
    def variance(state: RatioABState, k: int) -> float | None:
        """V_k, the delta-method variance of arm k's ratio of sums; None below two units or while sum d = 0."""
        n, sd = state.units[k], state.sum_d[k]
        if n < 2 or not sd > 0.0:
            return None
        r = state.sum_y[k] / sd
        q = max(0.0, state.cyy[k] + r * (r * state.cdd[k] - 2.0 * state.cyd[k]))  # sum (y - r d)^2
        return q * n / ((n - 1) * sd * sd)

    def _output(self, state: RatioABState) -> RatioABOutput:
        n = state.units[0] + state.units[1]
        rt, rc = self.ratio(state, 0), self.ratio(state, 1)
        vt, vc = self.variance(state, 0), self.variance(state, 1)
        difference = std_error = lower = upper = lift = lift_lower = lift_upper = None
        if rt is not None and rc is not None:
            difference = rt - rc
            if rt > 0.0 and rc > 0.0:
                lift = rt / rc - 1.0
        if vt is not None and vc is not None and rt is not None and rc is not None:
            v = vt + vc
            std_error = math.sqrt(v)
            if n >= self.start and v > 0.0:
                h = self.half_width(n, math.sqrt(n * v))
                lower, upper = rt - rc - h, rt - rc + h
            if n >= self.start and lift is not None:
                v_log = vt / (rt * rt) + vc / (rc * rc)
                if v_log > 0.0:
                    log_ratio = math.log(rt / rc)
                    h = self.half_width(n, math.sqrt(n * v_log))
                    lift_lower, lift_upper = math.expm1(log_ratio - h), math.expm1(log_ratio + h)
        return RatioABOutput(n, state.units, (rt, rc), difference, std_error, lower, upper, lift, lift_lower,
                             lift_upper, state.decision)

    def step(self, state: RatioABState, x, rng=None):
        label = x[self.arm_input]
        if label not in self.arms:
            return state, self._output(state)
        k = self.arms.index(label)
        y = float(x[self.input])
        d = float(x[self.denominator_input]) if self.denominator else 1.0
        if not math.isfinite(y):
            raise ContractViolation(f"input {self.input!r}: the numerator must be finite, got {y!r}")
        if not (math.isfinite(d) and d >= 0.0):
            raise ContractViolation(f"input {self.denominator_input!r}: the denominator must be finite and at least 0, "
                                    f"got {d!r}")
        n_old = state.units[k]
        n = n_old + 1
        sy = require_finite(state.sum_y[k] + y, "numerator sum")
        sd = require_finite(state.sum_d[k] + d, "denominator sum")
        cyy, cdd, cyd = state.cyy[k], state.cdd[k], state.cyd[k]
        if n_old:
            # Welford: (v - mean before) (w - mean after); a square's increment is >= 0 but for rounding
            dy, dd = y - state.sum_y[k] / n_old, d - state.sum_d[k] / n_old
            ey, ed = y - sy / n, d - sd / n
            cyy = require_finite(cyy + max(0.0, dy * ey), "numerator sum of squares")
            cdd = require_finite(cdd + max(0.0, dd * ed), "denominator sum of squares")
            cyd = require_finite(cyd + dy * ed, "cross product")
        new = RatioABState(_put(state.units, k, n), _put(state.sum_y, k, sy), _put(state.sum_d, k, sd),
                           _put(state.cyy, k, cyy), _put(state.cdd, k, cdd), _put(state.cyd, k, cyd), state.decision)
        out = self._output(new)
        if new.decision is None:
            lo, hi = (out.lower, out.upper) if self.scale == "difference" else (out.lift_lower, out.lift_upper)
            if lo is not None and hi is not None and not lo <= self.effect0 <= hi:
                new = replace(new, decision=REJECT_H0)
                out = replace(out, decision=REJECT_H0)
        return new, out

    def is_terminal(self, state: RatioABState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> RatioABState:
        state = decode_fields(RatioABState, data)
        pairs = (state.units, state.sum_y, state.sum_d, state.cyy, state.cdd, state.cyd)
        require_state(all(len(v) == 2 for v in pairs), "two arms")
        require_state(all(n >= 0 for n in state.units), "negative count")
        require_state(all(v >= 0.0 for v in state.sum_d + state.cyy + state.cdd),
                      "negative denominator sum or sum of squares")
        require_state(all(state.units[k] > 0 or all(v[k] == 0.0 for v in pairs[1:]) for k in (0, 1)),
                      "sums without units")
        if not self.denominator:
            require_state(state.sum_d == (float(state.units[0]), float(state.units[1]))
                          and state.cdd == state.cyd == (0.0, 0.0), "without a denominator every d is 1")
        return state
