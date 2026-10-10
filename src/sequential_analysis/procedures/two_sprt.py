# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Lorden's 2-SPRT for the Kiefer-Weiss problem.

The Kiefer-Weiss problem asks for the test of theta0 against theta1 that,
subject to error probabilities alpha0 and alpha1, minimizes the *maximum*
expected sample size over theta.  The SPRT is optimal at theta0 and theta1
but can be slow in between.  The 2-SPRT runs two one-sided SPRTs against
an intermediate point theta*:

    L0_n = sum_i log f_{theta*}(X_i) / f_{theta0}(X_i)    reject H0 when L0_n >= a0
    L1_n = sum_i log f_{theta*}(X_i) / f_{theta1}(X_i)    accept H0 when L1_n >= a1

and stops at the first n at which either boundary is reached.  Under
theta_i, exp(Li_n) is a non-negative martingale with mean one, so by
Ville's inequality the wrong decision has probability at most exp(-a_i).
The default thresholds a_i = log(1/alpha_i) therefore guarantee the error
rates, conservatively.  Explicit `thresholds=(a0, a1)` can be passed when
they have been calibrated to attain the error rates more tightly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core import Procedure
from ..numerics import neumaier_add, require_finite
from .families import Family, Gaussian
from .sprt import ACCEPT_H0, REJECT_H0


def kiefer_weiss_point(family: Family, theta0: float, theta1: float, a0: float, a1: float) -> float:
    """Intermediate point at which both boundaries are reached at the same rate.

    Solves a0 / KL(theta, theta0) = a1 / KL(theta, theta1) on (theta0, theta1)
    by bisection.  This is the first-order choice of theta*; see Lorden
    (1976) and Huffman (1983) for refinements.
    """
    lo, hi = (theta0, theta1) if theta0 < theta1 else (theta1, theta0)

    def h(theta: float) -> float:
        return a0 * family.kl(theta, theta1) - a1 * family.kl(theta, theta0)

    sign_lo = h(lo + (hi - lo) * 1e-12) > 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (h(mid) > 0) == sign_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class TwoSPRTState:
    n: int = 0
    l0: float = 0.0
    c0: float = 0.0
    l1: float = 0.0
    c1: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class TwoSPRTOutput:
    n: int
    l0: float
    l1: float
    decision: str | None


class TwoSPRT(Procedure):
    name = "two_sprt"
    version = "1"

    def __init__(
        self,
        family: Family,
        theta0: float,
        theta1: float,
        alpha0: float = 0.05,
        alpha1: float = 0.05,
        theta_star: float | None = None,
        thresholds: tuple[float, float] | None = None,
        input: str = "x",
    ):
        family.check(theta0)
        family.check(theta1)
        if theta0 == theta1 or not (0 < alpha0 < 1 and 0 < alpha1 < 1):
            raise ValueError("need theta0 != theta1 and error rates in (0, 1)")
        self.family, self.theta0, self.theta1 = family, float(theta0), float(theta1)
        self.alpha0, self.alpha1, self.input = alpha0, alpha1, input
        self.a0, self.a1 = thresholds if thresholds is not None else (math.log(1.0 / alpha0), math.log(1.0 / alpha1))
        if not (0 < self.a0 < math.inf and 0 < self.a1 < math.inf):
            raise ValueError("thresholds must be positive")
        if theta_star is None:
            theta_star = kiefer_weiss_point(family, self.theta0, self.theta1, self.a0, self.a1)
        if not min(theta0, theta1) < theta_star < max(theta0, theta1):
            raise ValueError("theta_star must lie strictly between theta0 and theta1")
        self.theta_star = float(theta_star)

    def config(self) -> dict:
        return {
            "family": self.family.spec(),
            "theta0": self.theta0,
            "theta1": self.theta1,
            "theta_star": self.theta_star,
            "alpha0": self.alpha0,
            "alpha1": self.alpha1,
            "a0": self.a0,
            "a1": self.a1,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: self.family.contract(self.input)}

    def initial_state(self) -> TwoSPRTState:
        return TwoSPRTState()

    def step(self, state: TwoSPRTState, x, rng=None):
        v = x[self.input]
        l0, c0 = neumaier_add(state.l0, state.c0, self.family.llr(v, self.theta_star, self.theta0))
        l1, c1 = neumaier_add(state.l1, state.c1, self.family.llr(v, self.theta_star, self.theta1))
        s0 = require_finite(l0 + c0, "log likelihood ratio against theta0")
        s1 = require_finite(l1 + c1, "log likelihood ratio against theta1")
        excess0, excess1 = s0 - self.a0, s1 - self.a1
        decision = None
        if excess0 >= 0 or excess1 >= 0:
            # If both boundaries are reached in the same step, decide by the larger
            # excess; an exact tie keeps H0.  Either way the error bounds hold.
            decision = REJECT_H0 if excess0 > excess1 else ACCEPT_H0
        n = state.n + 1
        return TwoSPRTState(n, l0, c0, l1, c1, decision), TwoSPRTOutput(n, s0, s1, decision)

    def is_terminal(self, state: TwoSPRTState) -> bool:
        return state.decision is not None

    def decode_state(self, data) -> TwoSPRTState:
        return TwoSPRTState(**data)

    def max_sample_size(self) -> int | None:
        """Largest possible stopping time, where one is known in closed form.

        For Gaussian observations the continuation region is a triangle in
        the (n, sum) plane, so the test is closed: it always stops by

            n_max = ceil( 2 sigma^2 (a0/d0 + a1/d1) / (d0 + d1) ),

        with d_i = |theta* - theta_i|.
        """
        if not isinstance(self.family, Gaussian):
            return None
        d0 = abs(self.theta_star - self.theta0)
        d1 = abs(self.theta_star - self.theta1)
        return math.ceil(2.0 * self.family.sigma**2 * (self.a0 / d0 + self.a1 / d1) / (d0 + d1))

    def as_plan(self) -> list[tuple[float, float]]:
        """The same test written as a sampling plan on the running sum S_n.

        For Gaussian observations and theta0 < theta1 the two boundaries are
        straight lines in the (n, S_n) plane,

            reject H0  when  S_n >= a0 sigma^2 / d0 + n (theta* + theta0) / 2
            accept H0  when  S_n <= n (theta* + theta1) / 2 - a1 sigma^2 / d1,

        and the plan ends at the first step at which they meet.  The result
        can be executed by `PlanTest` and is used to validate it.
        """
        if not isinstance(self.family, Gaussian) or not self.theta0 < self.theta1:
            raise NotImplementedError("plans are available for Gaussian observations with theta0 < theta1")
        var = self.family.sigma**2
        d0 = self.theta_star - self.theta0
        d1 = self.theta1 - self.theta_star
        plan, n = [], 0
        while True:
            n += 1
            upper = self.a0 * var / d0 + n * (self.theta_star + self.theta0) / 2.0
            lower = n * (self.theta_star + self.theta1) / 2.0 - self.a1 * var / d1
            if lower < upper:
                plan.append((lower, upper))
                continue
            # both boundaries are reached: the decision goes to the larger excess
            tie = (d0 * upper + d1 * lower) / (d0 + d1)
            plan.append((tie, tie))
            return plan
