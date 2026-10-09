# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Design of sequential sampling plans: operating characteristics and optimal plans.

Executing a plan is a Sequential Procedure (`PlanTest`).  Computing one is
a design problem, solved here offline, before the experiment.  Both
functions work on the running sum S_n, which is sufficient for the
one-parameter exponential families in `seqinfer.procedures`.

The sum is carried on a lattice.  For Bernoulli and Poisson data the
lattice is the integers and every result is exact up to floating point
(the Poisson distribution is cut where its tail is below 1e-16).  For
normal data the observation is replaced by its discretisation on a
lattice of step h (default sigma/16); the error of the operating
characteristic is of order h^2, and `exact` is False in the results.
Plans are executed on the real data, so their actual error rates can be
checked by simulation, as the tests do.

operating_characteristic
    Probabilities of each decision, the distribution of the stopping time
    and its expectation (ASN), by forward recursion over the distribution
    of S_n.

kiefer_weiss_plan
    The plan with at most `horizon` observations that minimises the
    expected sample size at theta*, subject to error probabilities alpha0
    at theta0 and alpha1 at theta1 -- the modified Kiefer-Weiss problem
    (Lorden 1980).  With theta* at the least favourable point it is the
    Kiefer-Weiss test, which minimises the *maximum* expected sample size.
    Solved by backward induction on the Lagrangian

        E_theta*[N] + lambda0 P_theta0(reject H0) + lambda1 P_theta1(accept H0)

    following Novikov, Novikov and Farkhshatov (2022, 2023).  The same
    backward pass also yields the error probabilities, written as
    expectations under theta* of the likelihood ratios, so adjusting the
    multipliers to the targets costs one pass per trial.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from operator import mul

from .procedures.families import Bernoulli, Family, Gaussian, Poisson
from .procedures.plan import PlanTest
from .procedures.sprt import ACCEPT_H0, REJECT_H0
from .procedures.two_sprt import kiefer_weiss_point

_TAIL = 1e-17
_NORMAL_WIDTH = 8.0  # the normal lattice covers theta +/- 8 sigma
_CONTINUE = "continue"


# ------------------------------------------------------------------ lattices


@dataclass(frozen=True)
class Lattice:
    """X takes the values (offset + i) * step with probabilities weights[i]."""

    step: float
    offset: int
    weights: tuple[float, ...]
    exact: bool


def lattice(family: Family, theta: float, step: float | None = None) -> Lattice:
    """The distribution of one observation on the lattice used by the design."""
    family.check(theta)
    if isinstance(family, Bernoulli):
        return Lattice(1.0, 0, (1.0 - theta, theta), True)
    if isinstance(family, Poisson):
        # Stop at the first term below _TAIL past 2 theta: the remaining tail is then
        # below 2 _TAIL, since later terms shrink by a factor theta / k < 1/2.
        weights, k, p = [], 0, math.exp(-theta)
        while True:
            weights.append(p)
            if k > 2.0 * theta and p < _TAIL:
                return Lattice(1.0, 0, tuple(weights), True)
            k += 1
            p *= theta / k
    if isinstance(family, Gaussian):
        sigma = family.sigma
        h = step if step is not None else sigma / 16.0
        if not h > 0:
            raise ValueError("the lattice step must be positive")
        lo = math.floor((theta - _NORMAL_WIDTH * sigma) / h)
        hi = math.ceil((theta + _NORMAL_WIDTH * sigma) / h)
        raw = [math.exp(-0.5 * ((j * h - theta) / sigma) ** 2) for j in range(lo, hi + 1)]
        total = sum(raw)
        return Lattice(h, lo, tuple(w / total for w in raw), False)
    raise NotImplementedError(
        f"designs are available for Bernoulli, Poisson and normal data, not {family.spec()['family']}"
    )


def _support(family: Family, n: int) -> tuple[int | None, int | None]:
    """Reachable lattice indices of S_n (None: unbounded)."""
    if isinstance(family, Bernoulli):
        return 0, n
    if isinstance(family, Poisson):
        return 0, None
    return None, None


# ------------------------------------------------- operating characteristic


@dataclass(frozen=True)
class OperatingCharacteristic:
    theta: float
    reject: float  # P(reject H0)
    accept: float  # P(accept H0)
    expected_n: float  # E[N], the average sample number
    stop: tuple[float, ...]  # stop[n-1] = P(N = n)
    exact: bool  # False when computed on a lattice approximation of continuous data


def operating_characteristic(
    plan: PlanTest, family: Family, theta: float, *, step: float | None = None
) -> OperatingCharacteristic:
    """Operating characteristic of `plan` for observations from `family` at `theta`."""
    lat = lattice(family, theta, step)
    h, weights, width = lat.step, lat.weights, len(lat.weights)
    low, high = (ACCEPT_H0, REJECT_H0) if plan.reject_high else (REJECT_H0, ACCEPT_H0)
    base, alive = 0, [1.0]  # alive[i] = P(S_n = (base + i) h, still sampling)
    reject = accept = expected = 0.0
    stop = []
    for n, (lower, upper) in enumerate(plan.plan, start=1):
        nxt = [0.0] * (len(alive) + width - 1)
        for i, w in enumerate(weights):
            for j, p in enumerate(alive):
                nxt[i + j] += p * w
        base += lat.offset
        kept: list[float] = []
        first: int | None = None
        closed = False  # a stop has followed the continuation points
        stopped = 0.0
        for i, p in enumerate(nxt):
            s = (base + i) * h
            if lower is not None and s <= lower:
                decision = low
            elif upper is not None and s >= upper:
                decision = high
            else:
                if closed:
                    raise ValueError(f"step {n}: the continuation region is not an interval")
                if first is None:
                    first = i
                kept.append(p)
                continue
            closed = first is not None
            stopped += p
            if decision == REJECT_H0:
                reject += p
            else:
                accept += p
        stop.append(stopped)
        expected += n * stopped
        if not kept:
            break
        base += first if first is not None else 0
        alive = kept
    return OperatingCharacteristic(theta, reject, accept, expected, tuple(stop), lat.exact)


# ------------------------------------------------------------ optimal design


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
    exact: bool
    family: Family | None = None
    step: float | None = None

    def maximum_expected_n(self, grid: int = 24) -> tuple[float, float]:
        """max over theta in [theta0, theta1] of E_theta[N], and where it is attained."""
        assert self.family is not None
        return maximum_expected_n(
            self.plan, self.family, self.at_theta0.theta, self.at_theta1.theta, step=self.step, grid=grid
        )


_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def maximum_expected_n(
    plan: PlanTest, family: Family, lower: float, upper: float, *, step: float | None = None, grid: int = 24
) -> tuple[float, float]:
    """The largest expected sample size of `plan` for theta between `lower` and `upper`, and its argmax.

    A grid locates the maximum; golden-section search refines it between
    the neighbouring grid points.
    """
    lo, hi = min(lower, upper), max(lower, upper)

    def asn(theta: float) -> float:
        return operating_characteristic(plan, family, theta, step=step).expected_n

    thetas = [lo + (hi - lo) * i / grid for i in range(grid + 1)]
    values = [asn(t) for t in thetas]
    i = max(range(len(values)), key=values.__getitem__)
    a, b = thetas[max(i - 1, 0)], thetas[min(i + 1, grid)]
    best_theta, best = thetas[i], values[i]
    c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
    fc, fd = asn(c), asn(d)
    for _ in range(40):
        if b - a < 1e-4 * (hi - lo):
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = asn(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = asn(d)
    for t, v in ((c, fc), (d, fd)):
        if v > best:
            best_theta, best = t, v
    return best, best_theta


def _affine_llr(family: Family, num: float, den: float) -> tuple[float, float]:
    """llr(x) = a + b x for an exponential family."""
    a = family.llr(0, num, den)
    return a, family.llr(1, num, den) - a


@dataclass
class _Pass:
    """Result of one backward pass for given multipliers."""

    value: float  # Lagrangian
    alpha0: float  # P_theta0(reject H0), as E_theta*[Z0 1{reject}]
    alpha1: float  # P_theta1(accept H0)
    asn: float  # E_theta*[N]
    stages: list[tuple[int, list[str]]]  # per n: first index of the computed region and its labels


class _Problem:
    def __init__(self, family: Family, theta0: float, theta1: float, theta_star: float, horizon: int, step):
        self.family, self.horizon = family, horizon
        self.lat = lattice(family, theta_star, step)
        self.a0, self.b0 = _affine_llr(family, theta0, theta_star)
        self.a1, self.b1 = _affine_llr(family, theta1, theta_star)

    def _logs(self, u0: float, u1: float, n: int, s: float) -> tuple[float, float]:
        """Log costs of stopping to reject and to accept at (n, s)."""
        return u0 + n * self.a0 + self.b0 * s, u1 + n * self.a1 + self.b1 * s

    def _region(self, u0: float, u1: float, n: int) -> tuple[int, int]:
        """Lattice indices where both stopping costs exceed 1, i.e. where continuing can pay."""
        h = self.lat.step
        lo, hi = -math.inf, math.inf
        for u, a, b in ((u0, self.a0, self.b0), (u1, self.a1, self.b1)):
            # u + n a + b s > 0
            edge = -(u + n * a) / b
            if b > 0:
                lo = max(lo, edge)
            else:
                hi = min(hi, edge)
        kmin, kmax = _support(self.family, n)
        k_lo = math.floor(lo / h) if math.isfinite(lo) else (kmin if kmin is not None else 0)
        k_hi = math.ceil(hi / h) if math.isfinite(hi) else (kmax if kmax is not None else 0)
        if kmin is not None:
            k_lo = max(k_lo, kmin)
        if kmax is not None:
            k_hi = min(k_hi, kmax)
        return k_lo, k_hi

    def _stop(self, u0: float, u1: float, n: int, k: int) -> tuple[float, float, float, str]:
        """(cost, Z0 if reject, Z1 if accept, decision) for stopping at (n, k)."""
        r, c = self._logs(u0, u1, n, k * self.lat.step)
        if r < c:
            return math.exp(r), math.exp(r - u0), 0.0, REJECT_H0
        return math.exp(c), 0.0, math.exp(c - u1), ACCEPT_H0

    def solve(self, u0: float, u1: float) -> _Pass:
        lat, H = self.lat, self.horizon
        weights, off, width = lat.weights, lat.offset, len(lat.weights)
        # at stage H everything stops: an empty computed region, values from the stopping costs
        stages: list[tuple[int, list[str]]] = [(0, [])] * H
        nxt_lo = 0
        nxt_v: list[float] = []
        nxt_q0: list[float] = []
        nxt_q1: list[float] = []
        nxt_en: list[float] = []
        nxt_n = H
        for n in range(H - 1, -1, -1):
            if n > 0:
                k_lo, k_hi = self._region(u0, u1, n)
            else:
                k_lo, k_hi = 0, 0  # the start; the first observation is always taken
            # values at stage n+1 for every index reachable from [k_lo, k_hi]
            e_lo, e_hi = k_lo + off, k_hi + off + width - 1
            ev, eq0, eq1, een = [], [], [], []
            for k in range(e_lo, e_hi + 1):
                i = k - nxt_lo
                if 0 <= i < len(nxt_v):
                    ev.append(nxt_v[i])
                    eq0.append(nxt_q0[i])
                    eq1.append(nxt_q1[i])
                    een.append(nxt_en[i])
                else:
                    cost, z0, z1, _ = self._stop(u0, u1, nxt_n, k)
                    ev.append(cost)
                    eq0.append(z0)
                    eq1.append(z1)
                    een.append(0.0)
            cur_v, cur_q0, cur_q1, cur_en, labels = [], [], [], [], []
            for k in range(k_lo, k_hi + 1):
                j = k - k_lo
                cont = 1.0 + sum(map(mul, weights, ev[j : j + width]))
                if n == 0:
                    stop_cost = math.inf
                else:
                    stop_cost, z0, z1, decision = self._stop(u0, u1, n, k)
                if cont < stop_cost:
                    cur_v.append(cont)
                    cur_q0.append(sum(map(mul, weights, eq0[j : j + width])))
                    cur_q1.append(sum(map(mul, weights, eq1[j : j + width])))
                    cur_en.append(1.0 + sum(map(mul, weights, een[j : j + width])))
                    labels.append(_CONTINUE)
                else:
                    cur_v.append(stop_cost)
                    cur_q0.append(z0)
                    cur_q1.append(z1)
                    cur_en.append(0.0)
                    labels.append(decision)
            if n > 0:
                stages[n - 1] = (k_lo, labels)
            nxt_lo, nxt_v, nxt_q0, nxt_q1, nxt_en, nxt_n = k_lo, cur_v, cur_q0, cur_q1, cur_en, n
        return _Pass(nxt_v[0], nxt_q0[0], nxt_q1[0], nxt_en[0], stages)

    def plan(self, u0: float, u1: float, stages, reject_high: bool, label: str) -> PlanTest:
        """Write the optimal regions as a PlanTest; refuses regions that are not intervals in S_n."""
        h = self.lat.step
        low_decision, high_decision = (ACCEPT_H0, REJECT_H0) if reject_high else (REJECT_H0, ACCEPT_H0)
        plan: list[tuple[float | None, float | None]] = []
        for n, (k_lo, region) in enumerate(stages, start=1):
            r0, c0 = self._logs(u0, u1, n, 0.0)
            k_tie = math.floor((c0 - r0) / ((self.b0 - self.b1) * h))  # r == c
            kmin, kmax = _support(self.family, n)
            k_hi = k_lo + len(region) - 1
            w_lo = min(k_lo, k_tie) - 1 if region else k_tie - 1
            w_hi = max(k_hi, k_tie + 1) + 1 if region else k_tie + 2
            if kmin is not None:
                w_lo = max(w_lo, kmin)
            if kmax is not None:
                w_hi = min(w_hi, kmax)
            ks = list(range(w_lo, w_hi + 1))
            labels = [
                region[k - k_lo] if region and k_lo <= k <= k_hi else self._stop(u0, u1, n, k)[3] for k in ks
            ]
            i = 0
            while i < len(ks) and labels[i] == low_decision:
                i += 1
            j = i
            while j < len(ks) and labels[j] == _CONTINUE:
                j += 1
            if any(lab != high_decision for lab in labels[j:]):
                raise ValueError(f"step {n}: the optimal region is not an interval in S_n")
            lower = (ks[i - 1] + 0.5) * h if i > 0 else None
            upper = (ks[j] - 0.5) * h if j < len(ks) else None
            if j == i:  # nothing continues: the plan ends here
                cut = lower if lower is not None else upper
                if cut is None:
                    raise ValueError(f"step {n}: no reachable state")
                plan.append((cut, cut))
                break
            plan.append((lower, upper))
        contract = self.family.contract("x")
        return PlanTest(plan, contract=contract, reject_high=reject_high, label=label)


def kiefer_weiss_plan(
    theta0: float,
    theta1: float,
    alpha0: float,
    alpha1: float,
    horizon: int,
    *,
    family: Family | None = None,
    theta_star: float | str | None = None,
    step: float | None = None,
) -> KieferWeissDesign:
    """Optimal plan with at most `horizon` observations; see the module docstring.

    `family` is Bernoulli (default), Poisson or Gaussian.  The returned plan
    never exceeds the horizon and has error probabilities at most alpha0
    and alpha1 -- exactly for discrete data, on the lattice for normal data.
    Raises ValueError if no plan within the horizon can meet them.

    `theta_star` is where the expected sample size is minimised.  By
    default it is the first-order point of Lorden (1976).  With
    ``theta_star="least-favourable"`` it solves the Kiefer-Weiss problem
    itself: by Lorden's characterisation the optimal test is the solution
    of the modified problem whose expected sample size is largest at
    theta* itself, so theta* is found by bisection on argmax_theta
    E_theta[N] - theta*, each candidate designed in full.  Near the
    solution the maximum expected sample size is flat in theta*; for
    discrete data it also moves in small steps, because the attainable
    error rates do, so the gain over the first-order theta* is often a
    fraction of a percent.
    """
    family = family if family is not None else Bernoulli()
    if theta_star == "least-favourable":
        return _least_favourable(theta0, theta1, alpha0, alpha1, horizon, family, step)
    if isinstance(theta_star, str):
        raise ValueError("theta_star must be a number, None or 'least-favourable'")
    family.check(theta0)
    family.check(theta1)
    if theta0 == theta1 or not (0 < alpha0 < 1 and 0 < alpha1 < 1) or horizon < 1:
        raise ValueError("need theta0 != theta1, error rates in (0, 1) and horizon >= 1")
    if theta_star is None:
        theta_star = kiefer_weiss_point(family, theta0, theta1, math.log(1 / alpha0), math.log(1 / alpha1))
    if not min(theta0, theta1) < theta_star < max(theta0, theta1):
        raise ValueError("theta_star must lie strictly between theta0 and theta1")
    problem = _Problem(family, theta0, theta1, theta_star, horizon, step)
    cache: dict[tuple[float, float], _Pass] = {}

    def solve(u0: float, u1: float) -> _Pass:
        if (u0, u1) not in cache:
            cache[(u0, u1)] = problem.solve(u0, u1)
        return cache[(u0, u1)]

    targets = (math.log(alpha0), math.log(alpha1))

    def residual(u0: float, u1: float) -> tuple[float, float]:
        """log error rates minus log targets; both decrease in their multiplier."""
        p = solve(u0, u1)
        return (
            math.log(max(p.alpha0, 1e-300)) - targets[0],
            math.log(max(p.alpha1, 1e-300)) - targets[1],
        )

    def feasible(u: tuple[float, float]) -> bool:
        r = residual(*u)
        return r[0] <= 0 and r[1] <= 0

    # Broyden's method on the residual, starting from the Wald-like Jacobian -I:
    # log alpha_i falls roughly one-for-one with log lambda_i.
    u = (math.log(horizon), math.log(horizon))
    f = residual(*u)
    jac = [[-1.0, 0.0], [0.0, -1.0]]
    best, stalled = max(abs(f[0]), abs(f[1])), 0
    for _ in range(40):
        if best < 1e-5 or stalled >= 4:  # converged, or stuck on the steps of a discrete problem
            break
        det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
        if abs(det) < 1e-12:
            jac, det = [[-1.0, 0.0], [0.0, -1.0]], 1.0
        d0 = -(jac[1][1] * f[0] - jac[0][1] * f[1]) / det
        d1 = -(-jac[1][0] * f[0] + jac[0][0] * f[1]) / det
        scale = max(1.0, abs(d0) / 2.0, abs(d1) / 2.0)  # damp long steps
        d0, d1 = d0 / scale, d1 / scale
        new_u = (u[0] + d0, u[1] + d1)
        if max(abs(x) for x in new_u) > 200:
            raise ValueError(f"no plan with at most {horizon} observations meets the error rates; increase the horizon")
        new_f = residual(*new_u)
        df = (new_f[0] - f[0], new_f[1] - f[1])
        norm = d0 * d0 + d1 * d1
        if norm > 0:
            jd = (jac[0][0] * d0 + jac[0][1] * d1, jac[1][0] * d0 + jac[1][1] * d1)
            for r in range(2):
                for c, dc in enumerate((d0, d1)):
                    jac[r][c] += (df[r] - jd[r]) * dc / norm
        u, f = new_u, new_f
        size = max(abs(f[0]), abs(f[1]))
        if size < 0.9 * best:
            best, stalled = size, 0
        else:
            stalled += 1

    # Near the root the error rates move in steps (always for discrete data, and on
    # the lattice for normal data), so an exact root need not exist.  Raising both
    # multipliers lowers both error rates: take the smallest raise that meets both.
    def raised(t: float) -> tuple[float, float]:
        return (u[0] + t, u[1] + t)

    if not feasible(u):
        lo, hi = 0.0, 1e-4
        while not feasible(raised(hi)):
            lo, hi = hi, 2 * hi
            if hi > 200:
                raise ValueError(
                    f"no plan with at most {horizon} observations meets the error rates; increase the horizon"
                )
        while hi - lo > 1e-9:
            mid = 0.5 * (lo + hi)
            if feasible(raised(mid)):
                hi = mid
            else:
                lo = mid
        u = raised(hi)

    # Spend any slack: lower each multiplier while both error rates stay within target.
    for _ in range(3):
        moved = False
        for which in (0, 1):
            if residual(*u)[which] > -1e-4:  # this error rate already sits at its target
                continue

            def at(x: float, which: int = which, base: tuple[float, float] = u) -> tuple[float, float]:
                return (x, base[1]) if which == 0 else (base[0], x)

            lo, hi = u[which] - 1.0, u[which]
            if feasible(at(lo)):
                hi = lo
            else:
                while hi - lo > 1e-6:
                    mid = 0.5 * (lo + hi)
                    if feasible(at(mid)):
                        hi = mid
                    else:
                        lo = mid
            if hi < u[which]:
                u, moved = at(hi), True
        if not moved:
            break
    u0, u1 = u
    p = solve(u0, u1)
    if p.alpha0 > alpha0 or p.alpha1 > alpha1:
        raise ValueError("could not meet both error rates; increase the horizon")
    reject_high = theta1 > theta0
    label = f"kiefer-weiss {family.spec()['family']} theta0={theta0} theta1={theta1} theta*={theta_star:.6g}"
    plan = problem.plan(u0, u1, p.stages, reject_high, label)

    def oc(theta: float) -> OperatingCharacteristic:
        return operating_characteristic(plan, family, theta, step=problem.lat.step if step is not None else None)

    return KieferWeissDesign(
        plan, theta_star, math.exp(u0), math.exp(u1), oc(theta0), oc(theta1), oc(theta_star), p.value,
        problem.lat.exact, family, step,
    )


def _least_favourable(theta0, theta1, alpha0, alpha1, horizon, family, step) -> KieferWeissDesign:
    lo, hi = min(theta0, theta1), max(theta0, theta1)
    margin = 0.02 * (hi - lo)

    def design(ts: float) -> tuple[float, KieferWeissDesign]:
        d = kiefer_weiss_plan(theta0, theta1, alpha0, alpha1, horizon, family=family, theta_star=ts, step=step)
        return d.maximum_expected_n()[1] - ts, d

    a, b = lo + margin, hi - margin
    ga, da = design(a)
    gb, db = design(b)
    if ga <= 0:  # the maximum sits at or below every candidate: take the lowest
        return da
    if gb >= 0:
        return db
    best = da if abs(ga) < abs(gb) else db
    best_gap = min(abs(ga), abs(gb))
    while b - a > 2e-3 * (hi - lo):
        mid = 0.5 * (a + b)
        g, d = design(mid)
        if abs(g) < best_gap:
            best, best_gap = d, abs(g)
        if g > 0:
            a = mid
        else:
            b = mid
    return best
