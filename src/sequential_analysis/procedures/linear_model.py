# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Anytime-valid tests and confidence sequences for regression coefficients.

After Lindon, Ham, Tingley and Bojinov (2022), "Anytime-Valid Linear
Models and Regression Adjusted Causal Inference in Randomized
Experiments" (arXiv:2210.08589).  The model is

    y_t = x_t' beta + z_t' delta + eps_t,    eps_t ~ N(0, sigma^2) i.i.d.,

with p nuisance coefficients beta (an intercept and the `covariates`),
d tested coefficients delta (the `tested` inputs, e.g. a treatment
indicator: d = 1 gives a covariate-adjusted A/B effect) and sigma
unknown.  H0: delta = delta0.

The Bayes factor.  Write X_n (n x p), Z_n (n x d) for the designs,
P the projection onto the columns of X_n, Zr = (I - P) Z_n and
yr = (I - P)(y - Z_n delta0) the parts the nuisance covariates do not
explain, S = Zr'Zr, and

    RSS = the residual sum of squares of the full regression,
    nu  = n - p - d (its degrees of freedom),
    Q0  = yr'yr = RSS + (dhat - delta0)' S (dhat - delta0),
    F   = (Q0 - RSS) / d / (RSS / nu)   (the usual F statistic; t^2 if d = 1).

Under the alternative, (beta, sigma) gets the right-Haar prior
d beta d sigma / sigma and delta | sigma ~ N(delta0, g sigma^2 I_d): the
standardised coefficients (delta - delta0) / sigma have the fixed prior
N(0, g I_d), Phi = I_d / g in the paper's notation.  Integrating beta
(Gaussian), delta (Gaussian) and then sigma (a Gamma integral) gives,
under the alternative and under H0 alike, a constant times
Q^(-(n - p) / 2), with the same constant except for a determinant, so

    B_n = det(I + g S)^(-1/2) * (Q0 / Q1)^((nu + d) / 2),
    Q1  = Q0 - h' (S + I/g)^(-1) h,   h = Zr' yr = S (dhat - delta0),
        = RSS + (dhat - delta0)' (S^-1 + g I)^(-1) (dhat - delta0),

the Bayes factor of the paper with Phi = I/g, written with the quantities
of a regression table (its sequential t-test is the case d = 1).
For d = 1 with s = S (a number), r = 1 / (1 + g s) and t the usual t
statistic of the tested coefficient,

    B_n = r^(1/2) * [ (1 + t^2 / nu) / (1 + r t^2 / nu) ]^((nu + 1) / 2),

the one-sample t-mixture of t_mixture.py with n c^2 replaced by g s
(and nu = n - p - 1).  The tests check both forms against a direct
numerical integration of the mixture.

Validity.  The model is invariant under y -> a y + X b (a > 0), the prior
on (beta, sigma) is that group's right-Haar measure and the prior on the
standardised effect does not depend on n or on the data, so B_n is the
likelihood ratio of the maximal invariant of y_1..y_n (Wijsman's
representation).  The
maximal invariants are nested in n, so under H0 B_n is a non-negative
martingale with mean one in their filtration, whatever beta and sigma
are, and Ville's inequality gives P(sup_n B_n >= 1/alpha) <= alpha.
Hence rejecting when B_n >= 1/alpha is a level-alpha test under
continuous monitoring and any stopping rule, 1/max_k B_k is an
anytime-valid p-value, and {delta0 : B_n(delta0) < 1/alpha} is a
(1 - alpha) confidence sequence.  The conditions: Gaussian errors with a
common variance, independent of the covariates; the covariates may be
random, but the argument is conditional on their whole sequence, so they
must not depend on earlier outcomes (randomised assignment and
pre-treatment covariates qualify; an adaptive design that looks at the
outcomes does not).  The guarantee is exact for Gaussian errors and only
approximate otherwise.

B_n is reported as 1 (log_e_value 0) while [X, Z] does not yet have full
rank or nu < 1.  Which steps those are depends on the covariates alone, so
this changes nothing above: the process is B_n from the first full-rank
step with nu >= 1 on, and E[B_n] = 1 there.

For d = 1 the set is an interval, dhat +/- t* se(dhat), t* in closed
form (as in t_mixture.py), and None (the whole line) while the data
cannot yet exclude anything.  g is the prior variance of a tested
coefficient in units of sigma^2 per unit of its input squared: for a 0/1
treatment indicator, of the standardised effect (Cohen's d).  The test is
most sensitive to effects of about that size; it is valid for every g,
but g is not invariant to rescaling the tested inputs (the nuisance
covariates may be rescaled and shifted freely).  This is a fixed prior,
not Zellner's g-prior N(0, g sigma^2 S^-1): that one depends on the
design at time n, and the Bayes factors would no longer form a martingale.

State: n, the means of (covariates, tested, y) and their centred
cross-product matrix (packed lower triangle), updated by Welford's
rank-one step; without an intercept, the raw cross products.  The
normal equations are solved by Cholesky; the cost of a step is
O((p + d)^3), independent of n.
"""
from __future__ import annotations

import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..numerics import require_finite
from .sprt import REJECT_H0

#: A pivot of the Cholesky factorisation smaller than this fraction of its
#: diagonal entry (1 - R^2 of a column on the earlier ones) counts as zero.
_RANK_TOLERANCE = 1e-10
#: A residual sum of squares below this fraction of the total is rounding.
_EXACT_FIT = 1e-13

Rows = typing.List[typing.List[float]]


def _cholesky(a: Rows, size: int) -> Rows | None:
    """Lower-triangular L with L L' = A[:size, :size], or None if A is (numerically) singular."""
    low = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            s = a[i][j]
            for k in range(j):
                s -= low[i][k] * low[j][k]
            if i == j:
                if not (a[i][i] > 0.0 and s > _RANK_TOLERANCE * a[i][i]):
                    return None
                low[i][i] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return low


_SMALLEST_NORMAL = 2.0**-1022

@dataclass(frozen=True)
class LinearModelState:
    n: int = 0
    mean: tuple[float, ...] = ()  # of (covariates, tested, y); empty without an intercept
    comoment: tuple[float, ...] = ()  # packed lower triangle, row by row
    max_log_e_value: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class LinearModelOutput:
    n: int
    estimates: tuple[float, ...] | None  # of delta; None until the design has full rank
    f: float | None  # the F statistic of H0 (t^2 if d = 1); None while undefined
    log_e_value: float
    p_value: float
    lower: float | None  # d = 1 only; None: the confidence sequence is still unbounded
    upper: float | None
    decision: str | None


@dataclass(frozen=True)
class _Fit:
    estimates: tuple[float, ...] | None
    f: float | None
    log_e_value: float
    lower: float | None
    upper: float | None


_SINGULAR = _Fit(None, None, 0.0, None, None)


def _names(value: Sequence[str], what: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{what} must be a list of input names")
    out = tuple(value)
    if not all(isinstance(v, str) and v for v in out):
        raise ValueError(f"{what} must be non-empty strings")
    return out


class AnytimeLinearModel(Procedure):
    """Anytime-valid F-test (t-test for d = 1) and confidence sequence for regression coefficients.

    Inputs: the outcome `input` and every name in `covariates` (nuisance,
    with an intercept unless intercept=False) and `tested` (the d tested
    coefficients), all real numbers, in every step.
    """

    name = "anytime_linear_model"
    version = "1"
    e_value_field = "log_e_value"  # an e-process: usable by MultipleTests

    def __init__(
        self,
        covariates: Sequence[str],
        tested: Sequence[str],
        delta0: Sequence[float] | None = None,
        g: float = 1.0,
        alpha: float = 0.05,
        *,
        intercept: bool = True,
        stop_on_reject: bool = False,
        input: str = "y",
    ):
        self.covariates = _names(covariates, "covariates")
        self.tested = _names(tested, "tested")
        if not self.tested:
            raise ValueError("at least one coefficient must be tested")
        if not (isinstance(input, str) and input):
            raise ValueError("input must be a non-empty string")
        everything = (*self.covariates, *self.tested, input)
        if len(set(everything)) != len(everything):
            raise ValueError("the outcome, covariates and tested inputs need different names")
        d = len(self.tested)
        if delta0 is None:
            delta0 = (0.0,) * d
        if isinstance(delta0, (str, bytes)) or not isinstance(delta0, Sequence) or len(delta0) != d:
            raise ValueError(f"delta0 must be a list of {d} numbers, one per tested input")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in delta0):
            raise ValueError("delta0 must be finite numbers")
        if type(g) not in (int, float) or not (0 < g <= 1e100):
            raise ValueError("g (prior variance of a standardised coefficient) must lie in (0, 1e100]")
        if type(alpha) not in (int, float) or not (0 < alpha < 1):
            raise ValueError("alpha must lie in (0, 1)")
        if type(intercept) is not bool or type(stop_on_reject) is not bool:
            raise ValueError("intercept and stop_on_reject must be True or False")
        self.delta0 = tuple(float(v) for v in delta0)
        self.g, self.alpha = float(g), float(alpha)
        self.intercept, self.stop_on_reject, self.input = intercept, stop_on_reject, input
        self._columns = (*self.covariates, *self.tested)
        self._k, self._d = len(self.covariates), d
        self._m = len(self._columns) + 1  # with y
        self._p = self._k + (1 if intercept else 0)
        self._log_level = math.log(1.0 / self.alpha)

    def config(self) -> dict:
        return {
            "covariates": list(self.covariates),
            "tested": list(self.tested),
            "delta0": list(self.delta0),
            "g": self.g,
            "alpha": self.alpha,
            "intercept": self.intercept,
            "stop_on_reject": self.stop_on_reject,
            "input": self.input,
        }

    def inputs(self):
        names = (*self._columns, self.input)
        return {name: InputContract(name) for name in names}

    def initial_state(self) -> LinearModelState:
        m = self._m
        return LinearModelState(0, (0.0,) * m if self.intercept else (), (0.0,) * (m * (m + 1) // 2))

    # ------------------------------------------------------------ statistics

    def _unpack(self, packed: Sequence[float]) -> Rows:
        m = self._m
        a = [[0.0] * m for _ in range(m)]
        at = 0
        for i in range(m):
            for j in range(i + 1):
                a[i][j] = a[j][i] = packed[at]
                at += 1
        return a

    def critical_log_ratio(self, nu: int, log_r_inv: float) -> float | None:
        """t*^2 / nu at which B_n reaches 1/alpha (d = 1, log_r_inv = log(1 + g s)), or None."""
        log_k = 2.0 * (self._log_level + 0.5 * log_r_inv) / (nu + 1)  # in logs: k itself may overflow
        if log_k >= log_r_inv:
            return None
        return math.expm1(log_k) / -math.expm1(log_k - log_r_inv)

    def fit(self, n: int, comoment: Sequence[float]) -> _Fit:
        """Estimates, F, log B_n and (d = 1) the interval from the sufficient statistics."""
        a = self._unpack(comoment)
        q, k, d = self._m - 1, self._k, self._d
        low = _cholesky(a, q)
        if low is None:
            return _SINGULAR
        c = [0.0] * q  # L c = X'y (centred): c = L^-1 b
        for i in range(q):
            s = a[q][i]
            for j in range(i):
                s -= low[i][j] * c[j]
            c[i] = s / low[i][i]
        # M = L[k:, k:] is the Cholesky factor of S = Z'(I - P)Z, and M' dhat = c[k:].
        est = [0.0] * d
        for i in reversed(range(d)):
            s = c[k + i]
            for j in range(i + 1, d):
                s -= low[k + j][k + i] * est[j]
            est[i] = s / low[k + i][k + i]
        estimates = tuple(require_finite(v, "coefficient estimate") for v in est)
        nu = n - self._p - d
        if nu < 1:
            return _Fit(estimates, None, 0.0, None, None)
        total = a[q][q]
        rss = total
        for value in c:
            rss -= value * value
        if not rss > _EXACT_FIT * total:
            rss = 0.0  # an exact fit; what is left is rounding
        # c2 = M' (dhat - delta0); B = I + g M'M, det B = det(I + g S).
        c2 = [0.0] * d
        for i in range(d):
            s = c[k + i]
            for j in range(i, d):
                s -= low[k + j][k + i] * self.delta0[j]
            c2[i] = s
        b = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i + 1):
                s = 0.0
                for t in range(max(i, j), d):
                    s += low[k + t][k + i] * low[k + t][k + j]
                b[i][j] = b[j][i] = (1.0 if i == j else 0.0) + self.g * s
        lb = _cholesky(b, d)
        if lb is None:  # cannot happen: B >= I
            raise ArithmeticError("I + g S is not positive definite")
        log_det = 0.0
        for i in range(d):
            log_det += 2.0 * math.log(lb[i][i])
        v = [0.0] * d
        q0 = q1 = 0.0
        for i in range(d):
            s = c2[i]
            for j in range(i):
                s -= lb[i][j] * v[j]
            v[i] = s / lb[i][i]
            q0 += c2[i] * c2[i]
            q1 += v[i] * v[i]
        half = 0.5 * (nu + d)
        if rss > 0.0:
            log_ratio = math.log1p(q0 / rss) - math.log1p(q1 / rss)
            f: float | None = q0 / d / (rss / nu)
        else:  # the limit RSS -> 0
            log_ratio = math.log(q0) - math.log(q1) if q0 > 0.0 else 0.0
            f = None
        log_e = require_finite(-0.5 * log_det + half * log_ratio, "log Bayes factor")
        lower = upper = None
        if d == 1:
            ratio = self.critical_log_ratio(nu, log_det)
            if ratio is not None:
                width = math.sqrt(ratio * rss) / low[k][k]
                lower, upper = estimates[0] - width, estimates[0] + width
        return _Fit(estimates, f, log_e, lower, upper)

    # ------------------------------------------------------------ transition

    def step(self, state: LinearModelState, x, rng=None):
        w = [float(x[name]) for name in self._columns]
        w.append(float(x[self.input]))
        m, n = self._m, state.n + 1
        if self.intercept:
            dev = [w[i] - state.mean[i] for i in range(m)]
            mean = tuple(require_finite(state.mean[i] + dev[i] / n, "running mean") for i in range(m))
            factor = state.n / n
            scaled = [v * factor for v in dev]
        else:
            mean, dev, scaled = (), w, w
        packed = list(state.comoment)
        at = 0
        for i in range(m):
            for j in range(i + 1):
                packed[at] = require_finite(packed[at] + scaled[i] * dev[j], "cross product")
                at += 1
        comoment = tuple(packed)
        fit = self.fit(n, comoment)
        max_log_e = max(state.max_log_e_value, fit.log_e_value)
        decision = state.decision or (REJECT_H0 if fit.log_e_value >= self._log_level else None)
        new = LinearModelState(n, mean, comoment, max_log_e, decision)
        p_value = min(1.0, math.exp(-max_log_e))
        out = LinearModelOutput(n, fit.estimates, fit.f, fit.log_e_value, p_value, fit.lower, fit.upper, decision)
        return new, out

    def is_terminal(self, state: LinearModelState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> LinearModelState:
        state = decode_fields(LinearModelState, data)
        m = self._m
        require_state(len(state.mean) == (m if self.intercept else 0), "wrong number of means")
        require_state(len(state.comoment) == m * (m + 1) // 2, "wrong size of the cross-product matrix")
        a = self._unpack(state.comoment)
        require_state(all(a[i][i] >= 0.0 for i in range(m)), "negative sum of squares")
        # a relative slack for rounding, and an absolute one for underflow: with a deviation d below about
        # 1e-154, d * d underflows while d * dy does not (n smallest normals cover every product lost)
        tiny = state.n * _SMALLEST_NORMAL
        require_state(
            all(abs(a[i][j]) <= math.sqrt((a[i][i] + tiny) * (a[j][j] + tiny)) * (1.0 + 1e-9)
                for i in range(m) for j in range(i)),
            "cross products violate the Cauchy-Schwarz inequality",
        )
        if state.n == 0:
            require_state(all(v == 0.0 for v in state.mean + state.comoment), "statistics without observations")
        require_state(state.decision in (None, REJECT_H0), "unknown decision")
        require_state(
            (state.decision is not None) == (state.max_log_e_value >= self._log_level),
            "the decision does not match the largest e-value",
        )
        require_state(state.max_log_e_value >= 0.0, "the largest log e-value is below its initial 0")
        return state
