# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Compute the table of least favourable theta* for normal data.

    python tools/least_favourable_table.py pilot   results/pilot.jsonl
    python tools/least_favourable_table.py design  results/design.jsonl --degrees 4 4 3
    python tools/least_favourable_table.py check   results/check.jsonl --points 16
    python tools/least_favourable_table.py fit     results/design.jsonl results/check.jsonl

For normal data with known sigma the Kiefer-Weiss problem is invariant
under shifts, scaling and reflection, so its solution is described by

    lambda = (theta* - theta0) / (theta1 - theta0)

as a function of alpha0, alpha1, the standardised difference
delta = |theta1 - theta0| / sigma and the horizon H.  The table holds the
correction y = lambda - lambda1, lambda1 = sqrt(a0) / (sqrt(a0) + sqrt(a1))
with a = log(1/alpha) being Lorden's first-order point, as a function of

    x1 = log alpha0,  x2 = log alpha1,  x3 = rho = delta sqrt(H) / (z_alpha0 + z_alpha1)

Exchanging the hypotheses (theta -> theta0 + theta1 - theta) maps lambda to
1 - lambda, so y is antisymmetric in (alpha0, alpha1) and zero when they
are equal.  Only nodes with alpha0 < alpha1 are computed, and the model is
the antisymmetric part of a tensor Legendre series:

    y = sum c_ijk (P_i(u1) P_j(u2) - P_j(u1) P_i(u2)) P_k(u3),  i > j

rho says how much longer than the shortest feasible fixed-sample test the
horizon is.  The table's points have H = 100 (delta follows from rho);
the pilot checks how much the horizon matters beyond rho.

Each evaluation finds the root of Lorden's characterisation (see
`characterisation`), a handful of designs, run in parallel; results are appended
to a JSON-lines file, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seqinfer.design import _kiefer_weiss
from seqinfer.procedures import Gaussian

LOG_ALPHA = (math.log(0.005), math.log(0.2))
RHO = (1.15, 2.5)
HORIZON = 100  # the horizon of the table's points; delta follows from rho (the pilot checks other horizons)


def first_order(alpha0: float, alpha1: float) -> float:
    a0, a1 = math.log(1 / alpha0), math.log(1 / alpha1)
    return math.sqrt(a0) / (math.sqrt(a0) + math.sqrt(a1))


def delta_for(alpha0: float, alpha1: float, rho: float, h: int) -> float:
    """The standardised difference with delta sqrt(h) = rho (z_alpha0 + z_alpha1)."""
    z = NormalDist().inv_cdf(1 - alpha0) + NormalDist().inv_cdf(1 - alpha1)
    return rho * z / math.sqrt(h)


def characterisation(alpha0: float, alpha1: float, delta: float, h: int, lam: float) -> float:
    """g(lambda) = argmax_theta E_theta[N] - theta* (in units of delta) for the plan designed at theta* = lambda delta.

    Lorden: the Kiefer-Weiss test is the modified test whose expected sample size is largest at theta*
    itself, g = 0.  g is smooth and decreasing in lambda; the maximum of E[N] itself is so flat in theta*
    that its minimiser is lost in the calibration noise of the error rates, so the root of g is tabulated.
    """
    d = _kiefer_weiss(0.0, delta, alpha0, alpha1, h, Gaussian(1.0), lam * delta, None)
    _, where = d.maximum_expected_n(grid=24)
    return where / delta - lam


def root(alpha0: float, alpha1: float, delta: float, h: int) -> tuple[float, int]:
    """The root of g by the Illinois method, from a bracket grown around the first-order point."""
    calls = 0

    def g(lam: float) -> float:
        nonlocal calls
        calls += 1
        return characterisation(alpha0, alpha1, delta, h, lam)

    a = first_order(alpha0, alpha1)
    ga = g(a)
    if ga == 0.0:
        return a, calls
    step = 0.04 if ga > 0 else -0.04
    b = min(0.98, max(0.02, a + step))
    gb = g(b)
    while ga * gb > 0:
        a, ga = b, gb
        step *= 2.0
        b = min(0.98, max(0.02, a + step))
        if b == a:
            raise ValueError("no sign change of Lorden's characterisation in (0.02, 0.98)")
        gb = g(b)
    side = 0
    while abs(b - a) > 2e-4:
        c = b - gb * (b - a) / (gb - ga)
        gc = g(c)
        if gc == 0.0:
            return c, calls
        if gc * gb < 0:
            a, ga = b, gb
            side = 0
        else:
            ga = ga / 2.0 if side == -1 else ga  # Illinois: halve the stale end
            side = -1
        b, gb = c, gc
        if abs(gc) < 1e-6:
            break
    return b - gb * (b - a) / (gb - ga) if gb != ga else b, calls


def evaluate(point: dict) -> dict:
    alpha0, alpha1 = math.exp(point["x1"]), math.exp(point["x2"])
    h = point.get("horizon", HORIZON)
    delta = delta_for(alpha0, alpha1, point["x3"], h)
    started = time.perf_counter()
    try:
        lam, calls = root(alpha0, alpha1, delta, h)
    except ValueError as error:
        return {**point, "horizon": h, "delta": delta, "error": str(error), "seconds": time.perf_counter() - started}
    return {
        **point, "delta": delta, "horizon": h, "lambda": lam, "lambda1": first_order(alpha0, alpha1),
        "y": lam - first_order(alpha0, alpha1), "designs": calls, "seconds": time.perf_counter() - started,
    }


def run(points: list[dict], out: Path, workers: int) -> None:
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            done.add(json.loads(line)["id"])
    todo = [p for p in points if p["id"] not in done]
    print(f"{len(points)} points, {len(done)} done, {len(todo)} to compute", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool, open(out, "a") as f:
        for result in pool.imap_unordered(evaluate, todo):
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(json.dumps({k: result[k] for k in ("id", "horizon", "seconds") if k in result}
                             | {"y": result.get("y"), "error": result.get("error")}), flush=True)


# ------------------------------------------------------------------ designs
def scaled(u: float, lo_hi: tuple[float, float]) -> float:
    """[-1, 1] -> the variable's range."""
    lo, hi = lo_hi
    return lo + (u + 1.0) * (hi - lo) / 2.0


def lobatto(degree: int) -> list[float]:
    """Gauss-Lobatto-Legendre points: the D-optimal design for polynomial regression of this degree on [-1, 1].

    They are -1, 1 and the roots of P'_degree; found by Newton's method from Chebyshev-Lobatto points.
    """
    if degree == 1:
        return [-1.0, 1.0]
    points = []
    for j in range(1, degree):
        x = -math.cos(math.pi * j / degree)
        for _ in range(100):
            # P_n, P_n' and P_n'' by the recurrence
            p0, p1 = 1.0, x
            for k in range(2, degree + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = degree * (x * p1 - p0) / (x * x - 1.0)
            d2p = (2.0 * x * dp - degree * (degree + 1) * p1) / (1.0 - x * x)
            step = dp / d2p
            x -= step
            if abs(step) < 1e-15:
                break
        points.append(x)
    return [-1.0, *sorted(points), 1.0]


def pilot_points() -> list[dict]:
    centre = (0.0, 0.0, 0.0)
    line = (-1.0, -0.5, 0.0, 0.5, 1.0)
    pts = []
    for axis in range(3):
        for u in line:
            c = list(centre)
            c[axis] = u
            pts.append(tuple(c))
    pts = sorted(set(pts))
    out = [{"id": f"pilot-{i}", "u": list(p), "x1": scaled(p[0], LOG_ALPHA), "x2": scaled(p[1], LOG_ALPHA),
            "x3": scaled(p[2], RHO)} for i, p in enumerate(pts)]
    # rho lines off the diagonal (on it y = 0) and a finer alpha line, for the degrees of the model
    for i, (a, b) in enumerate([(-1.0, 0.0), (-0.5, 0.5)]):
        for c in (-1.0, -0.5, 0.0, 0.5, 1.0):
            if (a, b, c) != (-1.0, 0.0, 0.0):
                out.append({"id": f"rho-{i}-{c}", "u": [a, b, c], "x1": scaled(a, LOG_ALPHA),
                            "x2": scaled(b, LOG_ALPHA), "x3": scaled(c, RHO)})
    for b in (-0.75, -0.25, 0.25, 0.75, 0.9):
        out.append({"id": f"fine-{b}", "u": [0.0, b, 0.0], "x1": scaled(0.0, LOG_ALPHA),
                    "x2": scaled(b, LOG_ALPHA), "x3": scaled(0.0, RHO)})
    # does the horizon matter beyond rho?  the same (alpha0, alpha1, rho) with other horizons
    for i, p in enumerate([(0.0, 0.0, 0.0), (-0.7, 0.7, -0.6), (0.7, -0.7, 0.6)]):
        for h in (30, 100, 250):
            if h == HORIZON and p == (0.0, 0.0, 0.0):
                continue  # already on the lines
            out.append({"id": f"horizon-{i}-{h}", "u": list(p), "x1": scaled(p[0], LOG_ALPHA),
                        "x2": scaled(p[1], LOG_ALPHA), "x3": scaled(p[2], RHO), "horizon": h})
    return out


def design_points(degrees: list[int]) -> list[dict]:
    """Tensor Gauss-Lobatto-Legendre nodes with u1 < u2 (the rest follows by antisymmetry)."""
    axes = [lobatto(d) for d in degrees]
    out = []
    for a in axes[0]:
        for b in axes[1]:
            if not a < b - 1e-12:
                continue
            for c in axes[2]:
                out.append({"id": f"node-{a:.6f}-{b:.6f}-{c:.6f}", "u": [a, b, c],
                            "x1": scaled(a, LOG_ALPHA), "x2": scaled(b, LOG_ALPHA), "x3": scaled(c, RHO)})
    return out


def halton_points(count: int) -> list[dict]:
    """Check points: the Halton sequence in bases 2, 3 and 5, from its second point."""

    def radical_inverse(n: int, base: int) -> float:
        x, f = 0.0, 1.0 / base
        while n:
            n, digit = divmod(n, base)
            x += digit * f
            f /= base
        return x

    out = []
    for n in range(1, count + 1):
        u = [2.0 * radical_inverse(n, b) - 1.0 for b in (2, 3, 5)]
        out.append({"id": f"check-{n}", "u": u, "x1": scaled(u[0], LOG_ALPHA), "x2": scaled(u[1], LOG_ALPHA),
                    "x3": scaled(u[2], RHO)})
    return out


# ------------------------------------------------------------------ fit
def legendre(n: int, x: float) -> float:
    p0, p1 = 1.0, x
    if n == 0:
        return 1.0
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1


def terms(degrees: list[int]) -> list[tuple[int, int, int]]:
    return [(i, j, k) for i in range(degrees[0] + 1) for j in range(i) for k in range(degrees[2] + 1)]


def basis(u: list[float], degrees: list[int]) -> list[float]:
    return [(legendre(i, u[0]) * legendre(j, u[1]) - legendre(j, u[0]) * legendre(i, u[1])) * legendre(k, u[2])
            for i, j, k in terms(degrees)]


def least_squares(rows: list[list[float]], ys: list[float]) -> list[float]:
    """Normal equations solved by Cholesky; the design keeps them well conditioned."""
    m = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(m)] for i in range(m)]
    b = [sum(r[i] * y for r, y in zip(rows, ys, strict=True)) for i in range(m)]
    lower = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1):
            s = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(s) if i == j else s / lower[j][j]
    z = [0.0] * m
    for i in range(m):
        z[i] = (b[i] - sum(lower[i][k] * z[k] for k in range(i))) / lower[i][i]
    c = [0.0] * m
    for i in reversed(range(m)):
        c[i] = (z[i] - sum(lower[k][i] * c[k] for k in range(i + 1, m))) / lower[i][i]
    return c


def load(path: Path) -> list[dict]:
    return [r for r in (json.loads(line) for line in path.read_text().splitlines()) if "y" in r]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "design", "check", "fit"])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--degrees", nargs=3, type=int, default=[4, 4, 3])
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--rho-min", type=float, default=RHO[0])
    args = parser.parse_args()
    if args.stage == "pilot":
        run(pilot_points(), args.files[0], args.workers)
    elif args.stage == "design":
        run(design_points(args.degrees), args.files[0], args.workers)
    elif args.stage == "check":
        run(halton_points(args.points), args.files[0], args.workers)
    else:
        fit(args.files[0], args.files[1] if len(args.files) > 1 else None, args.degrees, args.table, args.rho_min)


def restricted(rows: list[dict], rho_min: float) -> list[dict]:
    """Points with rho >= rho_min, their third coordinate rescaled to [rho_min, RHO[1]]."""
    out = []
    for r in rows:
        if r["x3"] >= rho_min - 1e-9:
            u3 = 2.0 * (r["x3"] - rho_min) / (RHO[1] - rho_min) - 1.0
            out.append({**r, "u": [r["u"][0], r["u"][1], u3]})
    return out


def fit(design_file: Path, check_file: Path | None, degrees: list[int], table: Path | None, rho_min: float) -> None:
    """Least-squares fit on rho >= rho_min.

    Below about rho = 1.5 (a horizon barely above that of the shortest fixed-sample test) lambda moves by up
    to 0.012 in a boundary layer that a low-degree polynomial cannot follow; there the table is used with a
    wider bracket instead (see seqinfer.design).
    """
    data = restricted(load(design_file), rho_min)
    rows = [basis(r["u"], degrees) for r in data]
    coef = least_squares(rows, [r["y"] for r in data])
    res = [r["y"] - sum(c * b for c, b in zip(coef, row, strict=True)) for r, row in zip(data, rows, strict=True)]
    dof = max(1, len(data) - len(coef))
    print(f"{len(data)} nodes, {len(coef)} coefficients; residual sd {math.sqrt(sum(e * e for e in res) / dof):.2e}, "
          f"max |residual| {max(abs(e) for e in res):.2e}")
    worst = max(abs(e) for e in res)
    if check_file is not None:
        check = restricted(load(check_file), rho_min)
        errs = [r["y"] - sum(c * b for c, b in zip(coef, basis(r["u"], degrees), strict=True)) for r in check]
        worst = max(worst, max(abs(e) for e in errs))
        print(f"{len(check)} check points: rms error {math.sqrt(sum(e * e for e in errs) / len(errs)):.2e}, "
              f"max {worst:.2e}")
    # below rho_min the table is used with rho clamped to rho_min: measure that error too
    below = [r for r in load(design_file) + (load(check_file) if check_file else []) if r["x3"] < rho_min - 1e-9]
    worst_below = max((abs(r["y"] - sum(c * b for c, b in zip(coef, basis([r["u"][0], r["u"][1], -1.0], degrees),
                                                                 strict=True))) for r in below), default=0.05)
    print(f"{len(below)} points below rho {rho_min}: largest error with rho clamped {worst_below:.2e}")
    if table is not None:
        table.write_text(json.dumps({
            "format": "seqinfer.least-favourable-normal/1",
            "description": "lambda = (theta* - theta0) / (theta1 - theta0) of the least favourable theta* for "
                           "normal data, minus Lorden's first-order point, as a tensor Legendre series in "
                           "u = (log alpha0, log alpha1, rho) scaled to [-1, 1]^3; see "
                           "tools/least_favourable_table.py",
            "log_alpha": list(LOG_ALPHA), "rho": [rho_min, RHO[1]], "horizon": HORIZON, "degrees": degrees,
            "terms": terms(degrees), "coefficients": coef, "nodes": len(data),
            # half-width of the search bracket, in units of |theta1 - theta0|: twice the largest error seen
            "bracket": max(2.0 * worst, 0.004),
            "bracket_below": max(2.0 * worst_below, 0.004),  # for rho below the table's range
        }, indent=1) + "\n")
        print(f"table written to {table}")


if __name__ == "__main__":
    main()
