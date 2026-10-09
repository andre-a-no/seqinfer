# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Design of sequential sampling plans: exact operating characteristics and optimal plans.

Executing a plan is a Sequential Procedure (`PlanTest`).  Computing one is
a design problem, solved here offline, before the experiment.  Both
functions work on the running sum S_n, which is sufficient for the
one-parameter exponential families in `seqinfer.procedures`.

operating_characteristic
    Exact probabilities of each decision, the distribution of the
    stopping time and its expectation (ASN), by forward recursion over
    the distribution of S_n.  Discrete families (Bernoulli, Poisson) only:
    there it is exact up to floating point, with no simulation error.

kiefer_weiss_plan
    The plan with at most `horizon` observations that minimises the
    expected sample size at theta*, subject to error probabilities alpha0
    at theta0 and alpha1 at theta1 -- the modified Kiefer-Weiss problem
    (Lorden 1980).  With theta* at the least favourable point it is the
    Kiefer-Weiss test, which minimises the *maximum* expected sample size.
    Solved by backward induction on the Lagrangian

        E_theta*[N] + lambda0 P_theta0(reject H0) + lambda1 P_theta1(accept H0)

    (Lorden 1980; Novikov 2009), with the multipliers adjusted until the
    exact error probabilities meet their targets.  Bernoulli data only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .procedures.families import Bernoulli, Poisson
from .procedures.plan import PlanTest
from .procedures.sprt import ACCEPT_H0, REJECT_H0
from .procedures.two_sprt import kiefer_weiss_point

_TAIL = 1e-16


@dataclass(frozen=True)
class OperatingCharacteristic:
    theta: float
    reject: float  # P(reject H0)
    accept: float  # P(accept H0)
    expected_n: float  # E[N], the average sample number
    stop: tuple[float, ...]  # stop[n-1] = P(N = n)


def _pmf(family, theta: float) -> list[tuple[int, float]]:
    """Distribution of one observation, as (value, probability) pairs."""
    family.check(theta)
    if isinstance(family, Bernoulli):
        return [(0, 1.0 - theta), (1, theta)]
    if isinstance(family, Poisson):
        out, k, p = [], 0, math.exp(-theta)
        total = 0.0
        while True:
            out.append((k, p))
            total += p
            if k > theta and 1.0 - total < _TAIL:
                return out
            k += 1
            p *= theta / k
    raise NotImplementedError(
        f"exact operating characteristics need a discrete family, not {family.spec()['family']}; "
        f"estimate them by simulation"
    )


def operating_characteristic(plan: PlanTest, family, theta: float) -> OperatingCharacteristic:
    """Exact operating characteristic of `plan` for observations from `family` at `theta`."""
    pmf = _pmf(family, theta)
    low, high = (ACCEPT_H0, REJECT_H0) if plan.reject_high else (REJECT_H0, ACCEPT_H0)
    alive: dict[int, float] = {0: 1.0}  # P(S_n = s, still sampling)
    reject = accept = expected = 0.0
    stop = []
    for n, (lower, upper) in enumerate(plan.plan, start=1):
        nxt: dict[int, float] = {}
        for s, p in alive.items():
            for v, q in pmf:
                nxt[s + v] = nxt.get(s + v, 0.0) + p * q
        alive, stopped = {}, 0.0
        for s, p in nxt.items():
            if lower is not None and s <= lower:
                decision = low
            elif upper is not None and s >= upper:
                decision = high
            else:
                alive[s] = p
                continue
            stopped += p
            if decision == REJECT_H0:
                reject += p
            else:
                accept += p
        stop.append(stopped)
        expected += n * stopped
        if not alive:
            break
    return OperatingCharacteristic(theta, reject, accept, expected, tuple(stop))


# --------------------------------------------------------------------- design


@dataclass(frozen=True)
class KieferWeissDesign:
    plan: PlanTest
    theta_star: float
    lambda0: float
    lambda1: float
    at_theta0: OperatingCharacteristic
    at_theta1: OperatingCharacteristic
    at_theta_star: OperatingCharacteristic
    lagrangian: float  # optimal value of the Lagrangian, from the backward induction


def _affine_llr(family, num: float, den: float) -> tuple[float, float]:
    """llr(x) = a + b x for an exponential family."""
    a = family.llr(0, num, den)
    return a, family.llr(1, num, den) - a


def _backward(family, theta0, theta1, theta_star, lambda0, lambda1, horizon):
    """Optimal continuation regions for given multipliers.

    Returns, for every n = 1..horizon, a dict s -> "continue" | ACCEPT_H0 |
    REJECT_H0, and the optimal value of the Lagrangian.  Values are in
    units of the theta* path probability, so the likelihood ratios
    Z_i(n, s) = prod f_theta_i / f_theta* turn error probabilities into
    expectations under theta*.
    """
    pmf = _pmf(family, theta_star)
    a0, b0 = _affine_llr(family, theta0, theta_star)
    a1, b1 = _affine_llr(family, theta1, theta_star)
    log_l0, log_l1 = math.log(lambda0), math.log(lambda1)

    def stop(n: int, s: int) -> tuple[float, str]:
        reject_cost = math.exp(log_l0 + n * a0 + b0 * s)
        accept_cost = math.exp(log_l1 + n * a1 + b1 * s)
        return (reject_cost, REJECT_H0) if reject_cost < accept_cost else (accept_cost, ACCEPT_H0)

    regions: list[dict[int, str]] = [{} for _ in range(horizon)]
    value = {}
    for s in range(horizon + 1):
        value[s], regions[horizon - 1][s] = stop(horizon, s)
    for n in range(horizon - 1, 0, -1):
        new = {}
        for s in range(n + 1):
            cont = 1.0 + sum(q * value[s + v] for v, q in pmf)
            cost, decision = stop(n, s)
            if cont < cost:
                new[s], regions[n - 1][s] = cont, "continue"
            else:
                new[s], regions[n - 1][s] = cost, decision
        value = new
    lagrangian = 1.0 + sum(q * value[v] for v, q in pmf)  # the first observation is always taken
    return regions, lagrangian


def _as_plan(regions, reject_high: bool, label: str) -> PlanTest:
    """Write optimal regions as a PlanTest; refuses regions that are not intervals in S_n."""
    low_decision, high_decision = (ACCEPT_H0, REJECT_H0) if reject_high else (REJECT_H0, ACCEPT_H0)
    plan = []
    for n, region in enumerate(regions, start=1):
        labels = [region[s] for s in range(n + 1)]
        # expected pattern: low_decision* continue* high_decision*
        i = 0
        while i <= n and labels[i] == low_decision:
            i += 1
        j = i
        while j <= n and labels[j] == "continue":
            j += 1
        if any(lab != high_decision for lab in labels[j:]):
            raise ValueError(f"step {n}: the optimal region is not an interval in S_n; it cannot be written as a plan")
        if j == i:  # nothing continues: the plan ends here with a cut between the two decisions
            cut = i - 0.5
            plan.append((cut, cut))
            break
        plan.append((i - 1 if i > 0 else None, j if j <= n else None))
    return PlanTest(plan, contract=Bernoulli().contract("x"), reject_high=reject_high, label=label)


def kiefer_weiss_plan(
    theta0: float,
    theta1: float,
    alpha0: float,
    alpha1: float,
    horizon: int,
    *,
    theta_star: float | None = None,
    family=None,
) -> KieferWeissDesign:
    """Optimal plan for Bernoulli data; see the module docstring.

    `horizon` is the largest sample size the experiment can afford.  The
    returned plan never exceeds it and has error probabilities at most
    alpha0 and alpha1 -- exactly, not by simulation.  Raises ValueError if
    no plan within the horizon can meet them.
    """
    family = family if family is not None else Bernoulli()
    if not isinstance(family, Bernoulli):
        raise NotImplementedError("optimal plans are implemented for Bernoulli data")
    family.check(theta0)
    family.check(theta1)
    if theta0 == theta1 or not (0 < alpha0 < 1 and 0 < alpha1 < 1) or horizon < 1:
        raise ValueError("need theta0 != theta1, error rates in (0, 1) and horizon >= 1")
    if theta_star is None:
        theta_star = kiefer_weiss_point(family, theta0, theta1, math.log(1 / alpha0), math.log(1 / alpha1))
    if not min(theta0, theta1) < theta_star < max(theta0, theta1):
        raise ValueError("theta_star must lie strictly between theta0 and theta1")
    reject_high = theta1 > theta0
    label = f"kiefer-weiss bernoulli theta0={theta0} theta1={theta1} theta*={theta_star:.6g}"

    cache: dict = {}

    def solve(l0: float, l1: float):
        if (l0, l1) not in cache:
            regions, value = _backward(family, theta0, theta1, theta_star, l0, l1, horizon)
            plan = _as_plan(regions, reject_high, label)
            cache[(l0, l1)] = (
                plan, value, operating_characteristic(plan, family, theta0), operating_characteristic(plan, family, theta1)
            )
        return cache[(l0, l1)]

    def smallest(target: float, error, l_other: float, which: int) -> float:
        """Smallest multiplier (on a log scale) whose plan meets the target."""
        lo, hi = 0.0, 1.0
        while True:  # grow until feasible
            args = (math.exp(hi), l_other) if which == 0 else (l_other, math.exp(hi))
            if error(*args) <= target:
                break
            lo, hi = hi, hi * 2 + 1
            if hi > 200:
                raise ValueError(f"no plan with at most {horizon} observations meets the error rates; increase the horizon")
        while hi - lo > 1e-6:  # the error rates are step functions of the multipliers
            mid = 0.5 * (lo + hi)
            args = (math.exp(mid), l_other) if which == 0 else (l_other, math.exp(mid))
            if error(*args) <= target:
                hi = mid
            else:
                lo = mid
        return math.exp(hi)

    def err0(l0, l1):
        return solve(l0, l1)[2].reject

    def err1(l0, l1):
        return solve(l0, l1)[3].accept

    l0 = l1 = 1.0
    for _ in range(30):
        new_l0 = smallest(alpha0, err0, l1, 0)
        new_l1 = smallest(alpha1, err1, new_l0, 1)
        done = abs(math.log(new_l0 / l0)) < 1e-5 and abs(math.log(new_l1 / l1)) < 1e-5
        l0, l1 = new_l0, new_l1
        if done:
            break
    plan, value, oc0, oc1 = solve(l0, l1)
    if oc0.reject > alpha0 or oc1.accept > alpha1:  # the last adjustment of lambda1 can move alpha0
        l0 = smallest(alpha0, err0, l1, 0)
        plan, value, oc0, oc1 = solve(l0, l1)
        if oc0.reject > alpha0 or oc1.accept > alpha1:
            raise ValueError("could not meet both error rates; increase the horizon")
    return KieferWeissDesign(
        plan, theta_star, l0, l1, oc0, oc1, operating_characteristic(plan, family, theta_star), value
    )
