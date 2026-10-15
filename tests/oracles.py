# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Independent reference computations for the mathematical tests.

None of this code is shared with the package.  The Gaussian references work
in exact rational arithmetic (every float is a dyadic rational, so the
inputs are represented exactly), the discrete ones by brute-force
enumeration of every path or segmentation; they are slow and only meant
for short series.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

# ---------------------------------------------------------------- exact linear algebra over the rationals

def _q(x):
    return Fraction(x)


def mat(a):
    return [[_q(x) for x in row] for row in a]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def tr(a):
    return [list(col) for col in zip(*a, strict=True)]


def madd(a, b):
    return [[x + y for x, y in zip(r, s, strict=True)] for r, s in zip(a, b, strict=True)]


def solve(a, b):
    """x with A x = b (A square, b a matrix), and det A, by Gaussian elimination."""
    n = len(a)
    m = [row[:] + brow[:] for row, brow in zip(a, b, strict=True)]
    det = Fraction(1)
    for col in range(n):
        pivot = next(r for r in range(col, n) if m[r][col] != 0)
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            det = -det
        det *= m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col], strict=True)]
    x = [[m[i][j] / m[i][i] for j in range(n, len(m[0]))] for i in range(n)]
    return x, det


def log_fraction(x: Fraction) -> float:
    return math.log(x.numerator) - math.log(x.denominator)


def gaussian_filter(F, H, Q, R, m0, P0, ys, B=None, us=None):
    """For each prefix y_1..y_t of a linear Gaussian model: (E[z_t | y_1..t], Cov[z_t | y_1..t], log p(y_1..t)).

    Built from the joint distribution of all states and observations and
    exact conditioning -- a computation that shares nothing with the filter.
    """
    F, H, Q, R, P0 = mat(F), mat(H), mat(Q), mat(R), mat(P0)
    n, m = len(F), len(H)
    T = len(ys)
    ys = [[_q(v) for v in (y if isinstance(y, (list, tuple)) else [y])] for y in ys]
    mean = [[_q(v)] for v in m0]
    means, covs = [mean], [P0]  # marginal mean and covariance of z_0..z_T
    for t in range(1, T + 1):
        mt = mul(F, means[-1])
        if B is not None:
            u = us[t - 1]
            u = [[_q(v)] for v in (u if isinstance(u, (list, tuple)) else [u])]
            mt = madd(mt, mul(mat(B), u))
        means.append(mt)
        covs.append(madd(mul(mul(F, covs[-1]), tr(F)), Q))
    power = {0: [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]}
    for k in range(1, T + 1):
        power[k] = mul(F, power[k - 1])

    def cross(t, s):  # Cov(z_t, z_s)
        return mul(power[t - s], covs[s]) if t >= s else tr(mul(power[s - t], covs[t]))

    results = []
    for t in range(1, T + 1):
        # observations 1..t stacked: mean, covariance; and their covariance with z_t
        my = [row for s in range(1, t + 1) for row in mul(H, means[s])]
        syy = [[None] * (t * m) for _ in range(t * m)]
        for a in range(1, t + 1):
            for b in range(1, t + 1):
                block = mul(mul(H, cross(a, b)), tr(H))
                if a == b:
                    block = madd(block, R)
                for i in range(m):
                    for j in range(m):
                        syy[(a - 1) * m + i][(b - 1) * m + j] = block[i][j]
        szy = [[None] * (t * m) for _ in range(n)]
        for b in range(1, t + 1):
            block = mul(cross(t, b), tr(H))
            for i in range(n):
                for j in range(m):
                    szy[i][(b - 1) * m + j] = block[i][j]
        resid = [[ys[s][i] - my[s * m + i][0]] for s in range(t) for i in range(m)]
        sol, det = solve(syy, [r[:] for r in resid])
        cmean = madd(means[t], mul(szy, sol))
        k, _ = solve(syy, tr(szy))  # Syy^-1 Syz
        ccov = [[covs[t][i][j] - sum(szy[i][a] * k[a][j] for a in range(t * m)) for j in range(n)] for i in range(n)]
        quad = sum(resid[i][0] * sol[i][0] for i in range(t * m))
        loglik = -0.5 * (t * m * math.log(2 * math.pi) + log_fraction(det) + float(quad))
        results.append(([float(v[0]) for v in cmean], [[float(v) for v in row] for row in ccov], loglik))
    return results


# ---------------------------------------------------------------- hidden Markov models by enumeration

def hmm_filter(initial, transition, emission, ys):
    """For each prefix: P(z_t = k | y_1..t) for every k, and log p(y_1..t), summing over all K^t paths.

    `emission(k, y)` is the density or probability of y in state k.
    """
    K = len(initial)
    out = []
    for t in range(1, len(ys) + 1):
        weights = [0.0] * K
        total = 0.0
        for path in itertools.product(range(K), repeat=t):
            p = initial[path[0]] * emission(path[0], ys[0])
            for s in range(1, t):
                p *= transition[path[s - 1]][path[s]] * emission(path[s], ys[s])
            weights[path[-1]] += p
            total += p
        out.append(([w / total for w in weights], math.log(total)))
    return out


# ---------------------------------------------------------------- product partitions by enumeration

def changepoint_filter(hazard, segment_log_marginal, ys):
    """For each prefix: P(current segment has length L | y_1..t) for L = 1..t, and log p(y_1..t).

    Change points occur independently before each observation after the
    first, with probability `hazard`; segments are independent given the
    partition, each with marginal likelihood exp(segment_log_marginal(seg)).
    Sums over all 2^(t-1) partitions.
    """
    out = []
    for t in range(1, len(ys) + 1):
        by_length = [0.0] * (t + 1)
        total = 0.0
        for flags in itertools.product((0, 1), repeat=t - 1):
            starts = [0] + [i + 1 for i, f in enumerate(flags) if f]
            log_p = sum(math.log(hazard) if f else math.log1p(-hazard) for f in flags)
            bounds = [*starts, t]
            for a, b in itertools.pairwise(bounds):
                log_p += segment_log_marginal(ys[a:b])
            p = math.exp(log_p)
            by_length[t - starts[-1]] += p
            total += p
        out.append(([w / total for w in by_length], math.log(total)))
    return out


# ---------------------------------------------------------------- one-dimensional filtering on a grid

def grid_filter(lo, hi, points, initial_density, transition_density, likelihood, ys):
    """For each prefix: (E[z_t | y_1..t], log p(y_1..t)) by numerical integration on a uniform grid.

    initial_density(z0), transition_density(z_prev, z), likelihood(z, y) are densities; the
    trapezoidal rule on `points` nodes over [lo, hi] approximates every integral (z_0 included).
    """
    h = (hi - lo) / (points - 1)
    grid = [lo + i * h for i in range(points)]
    w = [h * (0.5 if i in (0, points - 1) else 1.0) for i in range(points)]
    belief = [initial_density(z) for z in grid]  # density of z_{t-1} given the past
    log_total, out = 0.0, []
    for y in ys:
        predicted = [sum(w[j] * belief[j] * transition_density(grid[j], z) for j in range(points)) for z in grid]
        joint = [p * likelihood(z, y) for p, z in zip(predicted, grid, strict=True)]
        c = sum(wi * v for wi, v in zip(w, joint, strict=True))
        belief = [v / c for v in joint]
        log_total += math.log(c)
        out.append((sum(wi * b * z for wi, b, z in zip(w, belief, grid, strict=True)), log_total))
    return out
