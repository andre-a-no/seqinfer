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
lattice of step h (default sigma/16), and exponential data by the
midpoints of cells of width h (default a eighth of the mean at theta*);
the error of the operating characteristic is of order h^2, and `exact`
is False in the results.
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

import json
import math
import time
from dataclasses import dataclass
from operator import mul
from typing import Any

from .cache import design_cache_dir, entry_path, read_entry, report, write_entry
from .canonical import canonical_json
from .procedures.families import Bernoulli, Exponential, Family, Gaussian, Poisson
from .procedures.plan import PlanTest
from .procedures.sprt import ACCEPT_H0, REJECT_H0
from .procedures.two_sprt import kiefer_weiss_point
from .version import __version__

_TAIL = 1e-17
_NORMAL_WIDTH = 8.0  # the normal lattice covers theta +/- 8 sigma
_EXPONENTIAL_TAIL = 30.0  # the exponential lattice stops where the tail is e^-30
_CONTINUE = "continue"
_LABELS = (_CONTINUE, REJECT_H0, ACCEPT_H0)
_NUMPY_WIDTH = 8  # lattices with at least this many weights use the numpy pass when numpy is installed

try:  # optional: the backward pass in array operations, several times faster on wide lattices
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on interpreters without numpy
    _np = None
_QUICK_RAISE = True  # try a small joint raise before the coordinate search
_HORIZON_RETRIES = 25  # smaller horizons tried when the requested one cannot be met


# ------------------------------------------------------------------ lattices


@dataclass(frozen=True)
class Lattice:
    """X takes the values (offset + i) * step + shift with probabilities weights[i].

    After n observations S_n = k * step + n * shift for an integer k.
    """

    step: float
    offset: int
    weights: tuple[float, ...]
    exact: bool
    shift: float = 0.0


def lattice(family: Family, theta: float, step: float | None = None) -> Lattice:
    """The distribution of one observation on the lattice used by the design."""
    family.check(theta)
    if isinstance(family, Bernoulli):
        return Lattice(1.0, 0, (1.0 - theta, theta), True)
    if isinstance(family, Poisson):
        # Work outwards from the mode in logs (exp(-theta) underflows beyond theta = 745),
        # keeping every term above _TAIL; the tails left out are of that order.
        def log_pmf(k: int) -> float:
            return k * math.log(theta) - theta - math.lgamma(k + 1)

        mode = math.floor(theta)
        lo = mode
        while lo > 0 and log_pmf(lo - 1) > math.log(_TAIL):
            lo -= 1
        hi = mode
        while log_pmf(hi + 1) > math.log(_TAIL):
            hi += 1
        return Lattice(1.0, lo, tuple(math.exp(log_pmf(k)) for k in range(lo, hi + 1)), True)
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
    if isinstance(family, Exponential):
        # Midpoints of cells of width h: the lattice mean is 1/theta up to O(h^2).
        h = step if step is not None else 1.0 / (8.0 * theta)
        if not h > 0:
            raise ValueError("the lattice step must be positive")
        count = math.ceil(_EXPONENTIAL_TAIL / (theta * h)) + 1
        raw = [math.exp(-theta * (j + 0.5) * h) for j in range(count)]
        total = sum(raw)
        return Lattice(h, 0, tuple(w / total for w in raw), False, 0.5 * h)
    raise NotImplementedError(
        f"designs are available for Bernoulli, Poisson, normal and exponential data, not {family.spec()['family']}"
    )


def _support(family: Family, n: int) -> tuple[int | None, int | None]:
    """Reachable lattice indices of S_n (None: unbounded)."""
    if isinstance(family, Bernoulli):
        return 0, n
    if isinstance(family, (Poisson, Exponential)):
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
            s = (base + i) * h + n * lat.shift
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


def _affine_llr(family: Family, num: float, den: float, step: float | None = None) -> tuple[float, float]:
    """llr(x) = a + b x for an exponential family, as seen on the design's lattice.

    For the exponential family the lattice's own normalisation differs
    from the density's by O(h^2), so the ratio is taken from the lattice:
    then the error probabilities computed backwards under theta* agree
    exactly with those computed forwards under theta0 and theta1.
    """
    if isinstance(family, Exponential):
        ln, ld = lattice(family, num, step), lattice(family, den, step)
        if ln.step != ld.step:
            raise ValueError("give an explicit lattice step for exponential designs")
        x0, x1 = ld.shift, ld.step + ld.shift
        r0, r1 = math.log(ln.weights[0] / ld.weights[0]), math.log(ln.weights[1] / ld.weights[1])
        b = (r1 - r0) / (x1 - x0)
        return r0 - b * x0, b
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
        self.a0, self.b0 = _affine_llr(family, theta0, theta_star, self.lat.step)
        self.a1, self.b1 = _affine_llr(family, theta1, theta_star, self.lat.step)

    def _s(self, n: int, k: float) -> float:
        return k * self.lat.step + n * self.lat.shift

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
        base = n * self.lat.shift
        k_lo = math.floor((lo - base) / h) if math.isfinite(lo) else (kmin if kmin is not None else 0)
        k_hi = math.ceil((hi - base) / h) if math.isfinite(hi) else (kmax if kmax is not None else 0)
        if kmin is not None:
            k_lo = max(k_lo, kmin)
        if kmax is not None:
            k_hi = min(k_hi, kmax)
        return k_lo, k_hi

    def _stop(self, u0: float, u1: float, n: int, k: int) -> tuple[float, float, float, str]:
        """(cost, Z0 if reject, Z1 if accept, decision) for stopping at (n, k)."""
        r, c = self._logs(u0, u1, n, self._s(n, k))
        if r < c:
            return math.exp(r), math.exp(r - u0), 0.0, REJECT_H0
        return math.exp(c), 0.0, math.exp(c - u1), ACCEPT_H0

    def solve(self, u0: float, u1: float) -> _Pass:
        # array operations pay off only on wide lattices (normal, exponential, Poisson with larger rates)
        if _np is not None and len(self.lat.weights) >= _NUMPY_WIDTH:
            return self._solve_numpy(u0, u1)
        return self._solve_python(u0, u1)

    def _stop_arrays(self, u0: float, u1: float, n: int, k_first: int, k_last: int):
        """`_stop` for every k in [k_first, k_last]: cost, Z0, Z1 and whether to reject."""
        assert _np is not None
        s = _np.arange(k_first, k_last + 1, dtype=float) * self.lat.step + n * self.lat.shift
        r = u0 + n * self.a0 + self.b0 * s
        c = u1 + n * self.a1 + self.b1 * s
        reject = r < c
        cost = _np.exp(_np.where(reject, r, c))
        z0 = _np.where(reject, _np.exp(r - u0), 0.0)
        z1 = _np.where(reject, 0.0, _np.exp(c - u1))
        return cost, z0, z1, reject

    def _solve_numpy(self, u0: float, u1: float) -> _Pass:
        """The backward pass of `_solve_python`, one stage at a time as array operations.

        The same recursion; sums over the lattice weights are dot products in
        numpy, so results agree with the pure-Python pass up to rounding.
        """
        assert _np is not None
        np = _np
        lat, H = self.lat, self.horizon
        weights = np.asarray(lat.weights, dtype=float)
        off, width = lat.offset, len(lat.weights)
        stages: list[tuple[int, list[str]]] = [(0, [])] * H
        nxt_lo, nxt_n = 0, H
        nxt = np.zeros((4, 0))  # rows: value, Z0, Z1, E[N] at stage n+1
        for n in range(H - 1, -1, -1):
            k_lo, k_hi = self._region(u0, u1, n) if n > 0 else (0, 0)
            if k_hi < k_lo:  # nothing can continue at this stage
                stages[n - 1] = (k_lo, [])
                nxt_lo, nxt, nxt_n = k_lo, np.zeros((4, 0)), n
                continue
            e_lo, e_hi = k_lo + off, k_hi + off + width - 1
            # stage n+1 on [e_lo, e_hi]: computed values where known, stopping values elsewhere
            cost, z0, z1, _ = self._stop_arrays(u0, u1, nxt_n, e_lo, e_hi)
            ext = np.vstack((cost, z0, z1, np.zeros_like(cost)))
            a, b = max(e_lo, nxt_lo), min(e_hi, nxt_lo + nxt.shape[1] - 1)
            if a <= b:
                ext[:, a - e_lo : b - e_lo + 1] = nxt[:, a - nxt_lo : b - nxt_lo + 1]
            # sum_i w_i ext[:, j + i] for every j: a sliding window against the weights
            windows = np.lib.stride_tricks.sliding_window_view(ext, width, axis=1)
            sums = windows @ weights  # shape (4, k_hi - k_lo + 1)
            cont = 1.0 + sums[0]
            if n == 0:
                cur = np.vstack((cont, sums[1], sums[2], 1.0 + sums[3]))
            else:
                s_cost, s_z0, s_z1, reject = self._stop_arrays(u0, u1, n, k_lo, k_hi)
                go = cont < s_cost
                cur = np.vstack((
                    np.where(go, cont, s_cost),
                    np.where(go, sums[1], s_z0),
                    np.where(go, sums[2], s_z1),
                    np.where(go, 1.0 + sums[3], 0.0),
                ))
                codes = np.where(go, 0, np.where(reject, 1, 2)).tolist()
                stages[n - 1] = (k_lo, [_LABELS[code] for code in codes])
            nxt_lo, nxt, nxt_n = k_lo, cur, n
        return _Pass(float(nxt[0, 0]), float(nxt[1, 0]), float(nxt[2, 0]), float(nxt[3, 0]), stages)

    def _solve_python(self, u0: float, u1: float) -> _Pass:
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
            s_tie = (c0 - r0) / (self.b0 - self.b1)  # r == c
            k_tie = math.floor((s_tie - n * self.lat.shift) / h)
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
            lower = self._s(n, ks[i - 1] + 0.5) if i > 0 else None
            upper = self._s(n, ks[j] - 0.5) if j < len(ks) else None
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
    cache: Any = None,
) -> KieferWeissDesign:
    """Optimal plan with at most `horizon` observations; see the module docstring.

    `family` is Bernoulli (default), Poisson, Gaussian or Exponential.  The returned plan
    never exceeds the horizon and has error probabilities at most alpha0
    and alpha1 -- exactly for discrete data, on the lattice for normal data.
    Raises ValueError if no plan within the horizon can meet them.

    `theta_star` is where the expected sample size is minimised.  By
    default it is the first-order point of Lorden (1976).  With
    ``theta_star="least-favourable"`` it solves the Kiefer-Weiss problem
    itself.  By Lorden's characterisation the optimal test is the solution
    of the modified problem whose expected sample size is largest at
    theta* itself; candidates are generated by bisection on argmax_theta
    E_theta[N] - theta*, each designed in full, and the candidate (the
    first-order point included) with the smallest maximum expected sample
    size is returned.  For discrete data the attainable error rates move
    in steps, so the best candidate need not satisfy the characterisation
    exactly, and the gain over the first-order theta* is often a fraction
    of a percent.

    Every plan computed is stored in the design cache and returned from it
    when the same question is asked again, after its error rates have been
    verified; `cache` selects the directory, or False turns the cache off
    (see `seqinfer.cache`).  Saving and loading are reported on the logger
    "seqinfer.cache".
    """
    family = family if family is not None else Bernoulli()
    directory = design_cache_dir(cache)
    key = _cache_key(theta0, theta1, alpha0, alpha1, horizon, family, theta_star, step) if directory else None
    if directory is None or key is None:
        return _kiefer_weiss(theta0, theta1, alpha0, alpha1, horizon, family, theta_star, step)
    path = entry_path(directory, "kiefer_weiss", key)
    entry = read_entry(path, key)
    if entry is not None:
        design = _verified(entry["design"], family, alpha0, alpha1, horizon)
        if design is not None:
            report(
                "loaded",
                f"design cache: loaded {_describe(key)} from {path} (computed in {entry.get('seconds', 0):.1f} s); "
                f"error rates verified",
                path, key,
            )
            return design
        report("ignored", f"design cache: ignored {path} (its plan does not reproduce); computing again", path, key)
    started = time.perf_counter()
    design = _kiefer_weiss(theta0, theta1, alpha0, alpha1, horizon, family, theta_star, step)
    seconds = time.perf_counter() - started
    if write_entry(path, key, _design_json(design), seconds, time.time()):
        report("saved", f"design cache: computed {_describe(key)} in {seconds:.1f} s and saved it to {path}", path, key)
    return design


_ALGORITHM = "3"  # bump when a change of the design algorithm may change its results


def _cache_key(theta0, theta1, alpha0, alpha1, horizon, family, theta_star, step) -> dict | None:
    """The question, as canonical JSON can hold it; None (no caching) for arguments it cannot."""
    key = {
        "function": "kiefer_weiss_plan",
        "seqinfer": __version__,
        "algorithm": _ALGORITHM,
        "theta0": theta0, "theta1": theta1, "alpha0": alpha0, "alpha1": alpha1, "horizon": horizon,
        "family": family.spec(), "theta_star": theta_star, "step": step,
    }
    try:
        canonical_json(key)
    except (TypeError, ValueError):
        return None
    normalized: dict = json.loads(json.dumps(key))
    return normalized


def _describe(key: dict) -> str:
    return (
        f"kiefer_weiss_plan({key['theta0']}, {key['theta1']}, {key['alpha0']}, {key['alpha1']}, "
        f"horizon={key['horizon']}, family={key['family']['family']}, theta_star={key['theta_star']!r})"
    )


def _oc_json(oc: OperatingCharacteristic) -> dict:
    return {"theta": oc.theta, "reject": oc.reject, "accept": oc.accept, "expected_n": oc.expected_n}


def _design_json(d: KieferWeissDesign) -> dict:
    return {
        "plan": d.plan.config(),
        "theta_star": d.theta_star, "lambda0": d.lambda0, "lambda1": d.lambda1, "lagrangian": d.lagrangian,
        "at_theta0": _oc_json(d.at_theta0), "at_theta1": _oc_json(d.at_theta1),
        "at_theta_star": _oc_json(d.at_theta_star), "exact": d.exact, "step": d.step,
    }


def _verified(data: dict, family: Family, alpha0: float, alpha1: float, horizon: int) -> KieferWeissDesign | None:
    """The stored design, if its plan reproduces the stored operating characteristic and meets the targets."""
    try:
        config = data["plan"]
        plan = PlanTest(
            config["continuation"], contract=family.contract(config["input"]),
            reject_high=config["reject_high"], label=config["label"], input=config["input"],
        )
        if json.loads(json.dumps(plan.config())) != config or plan.horizon > horizon:
            return None
        step = data["step"]
        ocs = [operating_characteristic(plan, family, data[k]["theta"], step=step)
               for k in ("at_theta0", "at_theta1", "at_theta_star")]
        for oc, k in zip(ocs, ("at_theta0", "at_theta1", "at_theta_star"), strict=True):
            stored = data[k]
            for field in ("reject", "accept", "expected_n"):
                if abs(getattr(oc, field) - stored[field]) > 1e-9 * max(1.0, abs(stored[field])):
                    return None
        if ocs[0].reject > alpha0 * (1 + 1e-12) or ocs[1].accept > alpha1 * (1 + 1e-12):
            return None
        return KieferWeissDesign(
            plan, float(data["theta_star"]), float(data["lambda0"]), float(data["lambda1"]), ocs[0], ocs[1], ocs[2],
            float(data["lagrangian"]), bool(data["exact"]), family, step,
        )
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None


def _kiefer_weiss(theta0, theta1, alpha0, alpha1, horizon, family, theta_star, step) -> KieferWeissDesign:
    if isinstance(theta_star, str) and theta_star != "least-favourable":
        raise ValueError("theta_star must be a number, None or 'least-favourable'")
    family.check(theta0)
    family.check(theta1)
    if theta0 == theta1 or not (0 < alpha0 < 1 and 0 < alpha1 < 1) or horizon < 1:
        raise ValueError("need theta0 != theta1, error rates in (0, 1) and horizon >= 1")
    _check_attainable(family, theta0, theta1, alpha0, alpha1, horizon)
    if isinstance(theta_star, str):  # "least-favourable"
        return _least_favourable(theta0, theta1, alpha0, alpha1, horizon, family, step)
    if theta_star is None:
        theta_star = kiefer_weiss_point(family, theta0, theta1, math.log(1 / alpha0), math.log(1 / alpha1))
    if not min(theta0, theta1) < theta_star < max(theta0, theta1):
        raise ValueError("theta_star must lie strictly between theta0 and theta1")
    # A plan with fewer observations also respects the horizon.  The non-randomised
    # plans that are optimal for some multipliers can miss the targets at one horizon
    # and meet them at a slightly smaller one, so smaller horizons are tried in turn.
    error: ValueError | None = None
    retries = _HORIZON_RETRIES if isinstance(family, (Bernoulli, Poisson)) else 1  # only discrete data has gaps
    for h in range(horizon, max(0, horizon - retries), -1):
        if h < horizon and not _attainable(family, theta0, theta1, alpha0, alpha1, h):
            break  # nor at any smaller horizon
        try:
            return _calibrated(family, theta0, theta1, alpha0, alpha1, h, theta_star, step)
        except ValueError as e:
            error = error or e
    assert error is not None
    raise error


def _neyman_pearson_error(family: Family, theta0: float, theta1: float, alpha0: float, n: int) -> float | None:
    """P_theta1(accept H0) of the most powerful randomised level-alpha0 test with n observations.

    Every plan stopping by n observations is a test based on n observations, so
    no plan with smaller alpha1 exists (Neyman-Pearson lemma).  The plans of this
    module see the data only through the lattice, a function of the observations,
    so the bound holds for them too.  None where it is not computed (exponential
    data).  The likelihood ratio is monotone in the sum for these families.
    """
    if isinstance(family, Gaussian):
        from statistics import NormalDist

        shift = math.sqrt(n) * abs(theta1 - theta0) / family.sigma
        return 0.5 * math.erfc((shift + NormalDist().inv_cdf(alpha0)) / math.sqrt(2.0))  # z_alpha0 = -inv_cdf(alpha0)
    if isinstance(family, Bernoulli):
        support = range(n + 1)

        def log_pmf(k: int, theta: float) -> float:
            binomial = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            return binomial + k * math.log(theta) + (n - k) * math.log1p(-theta)
    elif isinstance(family, Poisson):
        lo_mean, hi_mean = n * min(theta0, theta1), n * max(theta0, theta1)
        lo = max(0, math.floor(lo_mean - 40.0 * math.sqrt(lo_mean) - 40.0))
        support = range(lo, math.ceil(hi_mean + 40.0 * math.sqrt(hi_mean) + 40.0) + 1)

        def log_pmf(k: int, theta: float) -> float:
            return k * math.log(n * theta) - n * theta - math.lgamma(k + 1)
    else:
        return None
    # reject H0 first where the likelihood ratio of theta1 to theta0 is largest
    order = reversed(support) if theta1 > theta0 else iter(support)
    size = 0.0
    for k in order:
        p0, p1 = math.exp(log_pmf(k, theta0)), math.exp(log_pmf(k, theta1))
        if size + p0 <= alpha0:
            size += p0
            continue
        # randomise on the boundary; the error is summed over the acceptance region itself,
        # not as 1 - power, which would leave rounding of order 1e-16 in place of a tiny error
        beta = (1.0 - (alpha0 - size) / p0) * p1
        return beta + math.fsum(math.exp(log_pmf(j, theta1)) for j in order)
    return 0.0


def _attainable(family, theta0, theta1, alpha0, alpha1, n) -> bool:
    beta = _neyman_pearson_error(family, theta0, theta1, alpha0, n)
    return beta is None or beta <= alpha1 * (1.0 + 1e-9) + 1e-15


def _check_attainable(family, theta0, theta1, alpha0, alpha1, horizon) -> None:
    """Fail at once, rather than after a long search, when no plan can exist."""
    if not _attainable(family, theta0, theta1, alpha0, alpha1, horizon):
        beta = _neyman_pearson_error(family, theta0, theta1, alpha0, horizon)
        raise ValueError(
            f"no plan with at most {horizon} observations meets the error rates; increase the horizon.  "
            f"Even the most powerful test with {horizon} observations at level {alpha0:g} accepts H0 "
            f"at theta1 with probability {beta:.4g} > {alpha1:g} (Neyman-Pearson)."
        )


def _calibrated(family, theta0, theta1, alpha0, alpha1, horizon, theta_star, step) -> KieferWeissDesign:
    """The optimal plan for one horizon, with multipliers adjusted to the error rates."""
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
    # the lattice for normal data), so an exact root need not exist, and the set of
    # multipliers meeting both targets can be a narrow band.  Fall back, in order, on
    # a coordinate search, a joint raise and a coarse grid; the slack is spent below.
    if not feasible(u) and _QUICK_RAISE and best < 1e-3:
        # Broyden ended next to the root on the wrong side of a target: a small joint raise
        # is enough, and the slack is spent below exactly as after the other fallbacks
        u = _joint_raise(u, feasible, limit=1e-2) or u
    if not feasible(u):
        u = _coordinate_search(u, residual, feasible) or _joint_raise(u, feasible) or _grid_search(feasible)
    if u is None:
        raise ValueError(
            f"no plan with at most {horizon} observations meets the error rates; increase the horizon.  "
            f"(For discrete data near the smallest feasible horizon a fixed-sample test may still meet them: "
            f"the plans optimal for some multipliers are non-randomised and do not reach every pair of "
            f"error rates, which would take randomised boundaries.)"
        )

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
    # reject where its cost falls below the cost of accepting: at large S_n when b0 < b1
    reject_high = problem.b0 < problem.b1
    label = f"kiefer-weiss {family.spec()['family']} theta0={theta0} theta1={theta1} theta*={theta_star:.6g}"
    plan = problem.plan(u0, u1, p.stages, reject_high, label)

    def oc(theta: float) -> OperatingCharacteristic:
        return operating_characteristic(plan, family, theta, step=None if problem.lat.exact else problem.lat.step)

    return KieferWeissDesign(
        plan, theta_star, math.exp(u0), math.exp(u1), oc(theta0), oc(theta1), oc(theta_star), p.value,
        problem.lat.exact, family, None if problem.lat.exact else problem.lat.step,
    )


def _coordinate_search(u, residual, feasible, rounds: int = 30):
    """Alternately take the smallest multiplier meeting each target, the other held fixed."""

    def smallest(which: int, other: float, start: float) -> float | None:
        def ok(x: float) -> bool:
            return bool(residual(*((x, other) if which == 0 else (other, x)))[which] <= 0)

        lo = hi = start
        if ok(hi):
            step = 1e-3
            while ok(lo - step) and lo - step > -200:
                lo, step = lo - step, 2 * step
            lo -= step
        else:
            step = 1e-3
            while not ok(hi):
                lo, hi, step = hi, hi + step, 2 * step
                if hi > 200:
                    return None
        while hi - lo > 1e-9:
            mid = 0.5 * (lo + hi)
            if ok(mid):
                hi = mid
            else:
                lo = mid
        return hi

    u0, u1 = u
    for _ in range(rounds):
        new0 = smallest(0, u1, u0)
        if new0 is None:
            return None
        new1 = smallest(1, new0, u1)
        if new1 is None:
            return None
        if feasible((new0, new1)):
            return (new0, new1)
        if abs(new0 - u0) < 1e-9 and abs(new1 - u1) < 1e-9:
            return None
        u0, u1 = new0, new1
    return None


def _joint_raise(u, feasible, limit: float = 200.0):
    """The smallest equal raise of both multipliers (at most `limit`) that meets both targets."""
    lo, hi = 0.0, 1e-4
    while not feasible((u[0] + hi, u[1] + hi)):
        lo, hi = hi, 2 * hi
        if hi > limit:
            return None
    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        if feasible((u[0] + mid, u[1] + mid)):
            hi = mid
        else:
            lo = mid
    return (u[0] + hi, u[1] + hi)


def _grid_search(feasible):
    """Last resort: any feasible point on a coarse grid of log multipliers."""
    points = [(a, b) for a in range(-10, 61, 4) for b in range(-10, 61, 4)]
    for a, b in sorted(points, key=lambda p: p[0] + p[1]):  # small multipliers first: shorter tests
        if feasible((float(a), float(b))):
            return (float(a), float(b))
    return None


_TABLE: dict | bool | None = False  # not loaded yet


def _least_favourable_table() -> dict | None:
    """The precomputed table of least favourable theta* for normal data (tools/least_favourable_table.py)."""
    global _TABLE
    if _TABLE is False:
        try:
            from importlib.resources import files

            _TABLE = json.loads(files("seqinfer").joinpath("data/least_favourable_normal.json").read_text())
        except (OSError, ValueError, ModuleNotFoundError):
            _TABLE = None
    return _TABLE if isinstance(_TABLE, dict) else None


def _legendre(n: int, x: float) -> float:
    p0, p1 = 1.0, x
    if n == 0:
        return 1.0
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1


def tabulated_least_favourable(alpha0: float, alpha1: float, rho: float) -> float | None:
    """lambda = (theta* - theta0) / (theta1 - theta0) of the least favourable theta* for normal data, from the table.

    `rho` = delta sqrt(H) / (z_alpha0 + z_alpha1), delta = |theta1 - theta0| / sigma: how much longer the
    horizon is than the shortest fixed-sample test.  The value is the root of Lorden's characterisation
    (largest expected sample size at theta* itself), fitted to computed roots within about 3e-4 for
    alpha0, alpha1 in [0.005, 0.2] and rho in [1.5, 2.5]; it hardly depends on rho there, so larger rho
    use the value at 2.5.  Below 1.5, where the horizon nearly binds, the value at 1.5 is returned and
    may be off by 0.015.  None for error rates outside the table.  The search for the least favourable
    theta* uses it only to bracket its bisection.
    """
    table = _least_favourable_table()
    if table is None:
        return None
    (la, lb), (ra, rb) = table["log_alpha"], table["rho"]
    u = [
        (2.0 * (math.log(alpha0) - la) / (lb - la)) - 1.0,
        (2.0 * (math.log(alpha1) - la) / (lb - la)) - 1.0,
        (2.0 * (min(max(rho, ra), rb) - ra) / (rb - ra)) - 1.0,
    ]
    if any(not -1.0 - 1e-9 <= x <= 1.0 + 1e-9 for x in u[:2]):
        return None
    total = 0.0
    for (i, j, k), c in zip(table["terms"], table["coefficients"], strict=True):
        # antisymmetric in (alpha0, alpha1): exchanging the hypotheses maps lambda to 1 - lambda
        total += c * (_legendre(i, u[0]) * _legendre(j, u[1]) - _legendre(j, u[0]) * _legendre(i, u[1])) * _legendre(
            k, u[2]
        )
    a0, a1 = math.log(1.0 / alpha0), math.log(1.0 / alpha1)
    return math.sqrt(a0) / (math.sqrt(a0) + math.sqrt(a1)) + total


def _least_favourable_hint(family, theta0, theta1, alpha0, alpha1, horizon) -> tuple[float, float] | None:
    """(theta*, half-width of a bracket) from the table, for normal data within its range; else None."""
    if not isinstance(family, Gaussian):
        return None
    table = _least_favourable_table()
    if table is None:
        return None
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1.0 - alpha0) + NormalDist().inv_cdf(1.0 - alpha1)
    rho = abs(theta1 - theta0) / family.sigma * math.sqrt(horizon) / z
    lam = tabulated_least_favourable(alpha0, alpha1, rho)
    if lam is None:
        return None
    width = table["bracket"] if rho >= table["rho"][0] else table["bracket_below"]
    return theta0 + lam * (theta1 - theta0), width * abs(theta1 - theta0)


def _least_favourable(theta0, theta1, alpha0, alpha1, horizon, family, step) -> KieferWeissDesign:
    lo, hi = min(theta0, theta1), max(theta0, theta1)
    margin = 0.02 * (hi - lo)
    candidates: list[tuple[float, KieferWeissDesign]] = []

    def design(ts: float | None) -> float | None:
        """argmax E[N] - theta* for this candidate (None if it cannot be designed); keeps the candidate."""
        try:
            d = _kiefer_weiss(theta0, theta1, alpha0, alpha1, horizon, family, ts, step)
        except ValueError:
            return None
        value, where = d.maximum_expected_n(grid=12)
        candidates.append((value, d))
        return where - d.theta_star

    # the first-order theta*: the search never returns anything worse; its errors propagate
    first = _kiefer_weiss(theta0, theta1, alpha0, alpha1, horizon, family, None, step)
    value, _ = first.maximum_expected_n(grid=12)
    candidates.append((value, first))
    a, b = lo + margin, hi - margin
    ga = gb = None
    hint = _least_favourable_hint(family, theta0, theta1, alpha0, alpha1, horizon)
    if hint is not None:
        # a bracket around the tabulated theta*; the full interval if it does not bracket the solution
        centre, width = hint
        ha, hb = max(a, centre - width), min(b, centre + width)
        if ha < hb:
            ga, gb = design(ha), design(hb)
            if ga is not None and gb is not None and ga > 0 > gb:
                a, b = ha, hb
            else:
                ga = gb = None
    if ga is None or gb is None:
        ga, gb = design(a), design(b)
    if ga is not None and gb is not None and ga > 0 > gb:
        # Lorden: the solution has its largest expected sample size at theta* itself
        while b - a > 2e-3 * (hi - lo):
            mid = 0.5 * (a + b)
            g = design(mid)
            if g is None:
                break
            if g > 0:
                a = mid
            else:
                b = mid
    # the steps of discrete error rates can break the characterisation; the objective decides
    return min(candidates, key=lambda c: c[0])[1]
