# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Mathematical correctness of the anytime-valid linear model.

* The Bayes factor against two independent numerical integrations: of the
  defining mixture of the raw-data likelihood over sigma and the
  standardised effect (the nuisance coefficients projected out in exact
  rational arithmetic), and, for one coefficient, of the noncentral t
  likelihood of the t statistic over the prior (Wijsman's representation).
* The closed form against the pooled two-sample t statistic, simple
  regression and the one-sample t-mixture.
* Type I error and uniform coverage of the confidence sequence by Monte
  Carlo, monitored after every observation (three standard errors).
* The precision gained by adjusting for covariates, invariance under the
  group the test is built on, states and arguments.
"""
import json
import math
import random
import unittest
from fractions import Fraction

from sequential_analysis import ContractViolation
from sequential_analysis.core import trajectory
from sequential_analysis.procedures.linear_model import AnytimeLinearModel
from sequential_analysis.procedures.t_mixture import TMixtureSPRT


def fold(proc, rows):
    state, outs = proc.initial_state(), []
    for x in rows:
        state, out = proc.step(state, x)
        outs.append(out)
    return state, outs


def binomial_upper_bound(alpha, runs):
    """alpha plus three standard errors of a Monte Carlo proportion."""
    return alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / runs)


# ------------------------------------------------------------- quadrature


def gauss_legendre(m):
    """Nodes and weights of the m-point Gauss-Legendre rule on [-1, 1] (Newton on P_m)."""
    nodes, weights = [], []
    for i in range(1, m + 1):
        x = math.cos(math.pi * (i - 0.25) / (m + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, m + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = m * (x * p1 - p0) / (x * x - 1.0)
            step = p1 / dp
            x -= step
            if abs(step) < 1e-16:
                break
        p0, p1 = 1.0, x
        for k in range(2, m + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = m * (x * p1 - p0) / (x * x - 1.0)
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    return nodes, weights


GL = gauss_legendre(20)


def grid(a, b, panels):
    """(node, log weight) pairs of a composite 20-point Gauss-Legendre rule on [a, b]."""
    out, h = [], (b - a) / panels
    for j in range(panels):
        lo = a + j * h
        for x, w in zip(*GL):
            out.append((lo + 0.5 * h * (x + 1.0), math.log(0.5 * h * w)))
    return out


def log_integral(terms):
    """log of sum exp(t) for a list of log terms."""
    top = max(terms)
    return top + math.log(math.fsum(math.exp(t - top) for t in terms))


# ---------------------------------------------- exact projections (rational)


def solve(a, b):
    """x with A x = b, exactly (Fractions, Gaussian elimination)."""
    n = len(a)
    m = [list(row) for row in a]
    for row, rhs in zip(m, b):
        row.append(rhs)
    for i in range(n):
        pivot = next(r for r in range(i, n) if m[r][i] != 0)
        m[i], m[pivot] = m[pivot], m[i]
        for r in range(n):
            if r != i and m[r][i] != 0:
                f = m[r][i] / m[i][i]
                m[r] = [u - f * v for u, v in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def residual(xs, v):
    """v minus its projection onto the columns of xs (rows), exactly."""
    p = len(xs[0])
    gram = [[sum(r[i] * r[j] for r in xs) for j in range(p)] for i in range(p)]
    coef = solve(gram, [sum(r[i] * vi for r, vi in zip(xs, v)) for i in range(p)])
    return [vi - sum(c * ri for c, ri in zip(coef, r)) for r, vi in zip(xs, v)]


def reduced(rows, covariates, tested, delta0, intercept=True):
    """Q = yr'yr, h = Zr'yr, S = Zr'Zr and RSS, with yr = (I - P)(y - Z delta0), exactly, as floats."""
    xs = [[Fraction(1)] * intercept + [Fraction(r[c]) for c in covariates] for r in rows]
    zs = [[Fraction(r[c]) for c in tested] for r in rows]
    y = [Fraction(r["y"]) - sum(Fraction(d0) * z for d0, z in zip(delta0, zr)) for r, zr in zip(rows, zs)]
    if xs[0]:
        yr = residual(xs, y)
        zr = [residual(xs, [z[j] for z in zs]) for j in range(len(tested))]
    else:
        yr, zr = y, [[z[j] for z in zs] for j in range(len(tested))]
    d = len(tested)
    q = sum(v * v for v in yr)
    h = [sum(a * b for a, b in zip(zr[j], yr)) for j in range(d)]
    s = [[sum(a * b for a, b in zip(zr[i], zr[j])) for j in range(d)] for i in range(d)]
    rss = q - sum(a * b for a, b in zip(h, solve(s, h)))
    return float(q), [float(v) for v in h], [[float(v) for v in row] for row in s], float(rss), len(xs[0])


def chol2(a):
    """Cholesky factor of a 1x1 or 2x2 symmetric positive definite matrix."""
    if len(a) == 1:
        return [[math.sqrt(a[0][0])]]
    l11 = math.sqrt(a[0][0])
    l21 = a[1][0] / l11
    return [[l11, 0.0], [l21, math.sqrt(a[1][1] - l21 * l21)]]


def inv(a):
    if len(a) == 1:
        return [[1.0 / a[0][0]]]
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]]


def mixture_log_bf(rows, covariates, tested, delta0, g, intercept=True):
    """log B_n by integrating the defining mixture numerically.

    With beta integrated out (a Gaussian integral), the likelihood of the
    data given sigma = 1/u and the standardised effect theta = (delta - delta0)/sigma is
    proportional to u^(n-p) exp(-(u^2 Q - 2 u h'theta + theta'S theta)/2);
    the right-Haar prior contributes du/u and theta ~ N(0, g I).  Both the
    alternative (over u and theta) and H0 (theta = 0, over u) are
    integrated by Gauss-Legendre quadrature.
    """
    q, h, s, rss, p = reduced(rows, covariates, tested, delta0, intercept)
    n, d = len(rows), len(tested)
    power = n - p - 1
    a = [[s[i][j] + (1.0 / g if i == j else 0.0) for j in range(d)] for i in range(d)]
    a_inv = inv(a)
    root = chol2(a_inv)  # theta = mu(u) + root eta: at fixed u the theta integrand is a Gaussian
    log_jac = math.log(root[0][0]) + (math.log(root[1][1]) if d == 2 else 0.0)
    eta = grid(-11.0, 11.0, 4 if d == 1 else 2)
    etas = [(e,) for e in eta] if d == 1 else [(e1, e2) for e1 in eta for e2 in eta]
    # u: from below the H0 peak sqrt(power / Q) to well beyond the largest possible one, sqrt(power / RSS)
    lo = max(0.0, math.sqrt(power / q) - 14.0 / math.sqrt(q))
    hi = math.sqrt(power / rss) + 14.0 / math.sqrt(rss)
    us = grid(lo, hi, 40 if d == 1 else 16)

    def log_alt(u):
        mu = [u * sum(a_inv[i][j] * h[j] for j in range(d)) for i in range(d)]
        terms = []
        for nodes in etas:
            z = [x for x, _ in nodes]
            theta = [mu[i] + sum(root[i][j] * z[j] for j in range(d)) for i in range(d)]
            quad = sum(theta[i] * s[i][j] * theta[j] for i in range(d) for j in range(d))
            prior = -0.5 * sum(t * t for t in theta) / g - 0.5 * d * math.log(2 * math.pi * g)
            log_w = sum(w for _, w in nodes)
            terms.append(log_w + prior + u * sum(hi_ * t for hi_, t in zip(h, theta)) - 0.5 * quad)
        return log_jac + log_integral(terms)

    alt = log_integral([lw + power * math.log(u) - 0.5 * u * u * q + log_alt(u) for u, lw in us if u > 0])
    null = log_integral([lw + power * math.log(u) - 0.5 * u * u * q for u, lw in grid(lo, hi, 40) if u > 0])
    return alt - null


def noncentral_t_log_bf(t, nu, s, g):
    """log B_n from the t statistic: the N(0, g) mixture of noncentral t densities over the central one.

    t = (Z + lambda) / sqrt(V / nu), V ~ chi^2_nu, lambda = theta sqrt(s), theta ~ N(0, g); V = w^2.
    """
    log_chi = lambda w: (nu - 1) * math.log(w) - 0.5 * w * w - (nu / 2 - 1) * math.log(2) - math.lgamma(nu / 2)
    precision = 1.0 / g + s
    terms = []
    for w, lw in grid(0.0, math.sqrt(nu) + 16.0, 30):
        if w <= 0:
            continue
        a = t * w / math.sqrt(nu)  # Z + lambda = a
        mean = a * math.sqrt(s) / precision
        inner = []
        for e, le in grid(mean - 12 / math.sqrt(precision), mean + 12 / math.sqrt(precision), 3):
            log_prior = -0.5 * e * e / g - 0.5 * math.log(2 * math.pi * g)
            inner.append(le + log_prior - 0.5 * (a - e * math.sqrt(s)) ** 2 - 0.5 * math.log(2 * math.pi))
        # density of t given V: f(a) * da/dt, integrated against that of V (log_chi includes dV = 2 w dw)
        terms.append(lw + log_chi(w) + math.log(w / math.sqrt(nu)) + log_integral(inner))
    log_central = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
                   - (nu + 1) / 2 * math.log1p(t * t / nu))
    return log_integral(terms) - log_central


def closed_form_t(t, nu, s, g):
    r = 1.0 / (1.0 + g * s)
    return 0.5 * math.log(r) + 0.5 * (nu + 1) * (math.log1p(t * t / nu) - math.log1p(r * t * t / nu))


# ------------------------------------------------------------------- data


def regression_rows(n, seed, *, effect=0.0, sigma=1.0, beta=(1.0, 2.0, -1.0), tested=1):
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        x1, x2 = rnd.gauss(0, 1), rnd.uniform(-1, 3)
        z = float(rnd.random() < 0.5)
        row = {"x1": x1, "x2": x2, "z": z}
        mean = beta[0] + beta[1] * x1 + beta[2] * x2 + effect * z
        if tested == 2:
            row["w"] = rnd.gauss(0.5 * x1, 1)
            mean += 0.5 * effect * row["w"]
        row["y"] = mean + sigma * rnd.gauss(0, 1)
        rows.append(row)
    return rows


class BayesFactor(unittest.TestCase):
    def test_equals_the_numerically_integrated_mixture_one_coefficient(self):
        rows = regression_rows(60, 1, effect=0.3, sigma=2.0)
        proc = AnytimeLinearModel(["x1"], ["z"], delta0=[0.1], g=0.7)
        _, outs = fold(proc, rows)
        for n in (5, 9, 20, 60):
            want = mixture_log_bf(rows[:n], ["x1"], ["z"], [0.1], 0.7)
            self.assertAlmostEqual(outs[n - 1].log_e_value, want, delta=1e-8 * max(1.0, abs(want)), msg=f"n={n}")

    def test_equals_the_numerically_integrated_mixture_two_coefficients(self):
        rows = regression_rows(40, 2, effect=0.4, tested=2)
        proc = AnytimeLinearModel(["x1", "x2"], ["z", "w"], delta0=[0.2, -0.1], g=1.5)
        _, outs = fold(proc, rows)
        for n in (8, 40):
            want = mixture_log_bf(rows[:n], ["x1", "x2"], ["z", "w"], [0.2, -0.1], 1.5)
            self.assertAlmostEqual(outs[n - 1].log_e_value, want, delta=1e-8 * max(1.0, abs(want)), msg=f"n={n}")

    def test_without_intercept(self):
        rows = regression_rows(30, 3, effect=0.5)
        proc = AnytimeLinearModel(["x2"], ["z"], g=2.0, intercept=False)
        _, outs = fold(proc, rows)
        for n in (4, 30):
            want = mixture_log_bf(rows[:n], ["x2"], ["z"], [0.0], 2.0, intercept=False)
            self.assertAlmostEqual(outs[n - 1].log_e_value, want, delta=1e-8 * max(1.0, abs(want)))

    def test_equals_the_mixture_of_noncentral_t_likelihoods(self):
        """Wijsman: the Bayes factor of the data is the Bayes factor of the t statistic alone."""
        rows = regression_rows(50, 4, effect=0.6)
        proc = AnytimeLinearModel(["x1", "x2"], ["z"], g=0.5)
        _, outs = fold(proc, rows)
        for n in (6, 15, 50):
            _, h, s, rss, p = reduced(rows[:n], ["x1", "x2"], ["z"], [0.0])
            nu = n - p - 1
            t = h[0] / math.sqrt(s[0][0] * rss / nu)
            want = noncentral_t_log_bf(t, nu, s[0][0], 0.5)
            self.assertAlmostEqual(closed_form_t(t, nu, s[0][0], 0.5), want, delta=1e-8 * max(1.0, abs(want)))
            self.assertAlmostEqual(outs[n - 1].log_e_value, want, delta=1e-8 * max(1.0, abs(want)), msg=f"n={n}")
            self.assertAlmostEqual(outs[n - 1].f, t * t, delta=1e-9 * t * t)


class ClosedForms(unittest.TestCase):
    def test_two_sample_t_without_covariates(self):
        rnd = random.Random(5)
        rows = [{"z": float(rnd.random() < 0.3), "y": 0.0} for _ in range(80)]
        for r in rows:
            r["y"] = 10.0 + 0.4 * r["z"] + rnd.gauss(0, 1.5)
        g, delta0 = 0.8, 0.25
        _, outs = fold(AnytimeLinearModel([], ["z"], delta0=[delta0], g=g), rows)
        checked = 0
        for n in range(1, 81):
            treated = [r["y"] for r in rows[:n] if r["z"] == 1.0]
            control = [r["y"] for r in rows[:n] if r["z"] == 0.0]
            n1, n0 = len(treated), len(control)
            out = outs[n - 1]
            if n1 == 0 or n0 == 0:
                self.assertIsNone(out.estimates)
                self.assertEqual(out.log_e_value, 0.0)
                continue
            m1, m0 = math.fsum(treated) / n1, math.fsum(control) / n0
            self.assertAlmostEqual(out.estimates[0], m1 - m0, delta=1e-12 * (1 + abs(m1 - m0)))
            nu = n - 2
            if nu < 1:
                continue
            pooled = (math.fsum((v - m1) ** 2 for v in treated) + math.fsum((v - m0) ** 2 for v in control)) / nu
            t = (m1 - m0 - delta0) / math.sqrt(pooled * (1 / n1 + 1 / n0))
            want = closed_form_t(t, nu, n1 * n0 / n, g)
            self.assertAlmostEqual(out.log_e_value, want, delta=1e-10 * max(1.0, abs(want)))
            checked += 1
        self.assertGreater(checked, 70)

    def test_simple_regression(self):
        rnd = random.Random(6)
        rows = [{"z": rnd.uniform(0, 5)} for _ in range(40)]
        for r in rows:
            r["y"] = 1.0 + 0.3 * r["z"] + rnd.gauss(0, 1)
        _, outs = fold(AnytimeLinearModel([], ["z"], g=3.0), rows)
        for n in (3, 10, 40):
            zs, ys = [r["z"] for r in rows[:n]], [r["y"] for r in rows[:n]]
            mz, my = math.fsum(zs) / n, math.fsum(ys) / n
            sxx = math.fsum((z - mz) ** 2 for z in zs)
            slope = math.fsum((z - mz) * (y - my) for z, y in zip(zs, ys)) / sxx
            rss = math.fsum((y - my - slope * (z - mz)) ** 2 for z, y in zip(zs, ys))
            t = slope / math.sqrt(rss / (n - 2) / sxx)
            self.assertAlmostEqual(outs[n - 1].log_e_value, closed_form_t(t, n - 2, sxx, 3.0), delta=1e-10)
            self.assertAlmostEqual(outs[n - 1].estimates[0], slope, delta=1e-12)

    def test_reduces_to_the_one_sample_t_mixture(self):
        """No intercept, the constant 1 tested: N(theta, sigma^2) with theta / sigma ~ N(0, g)."""
        rnd = random.Random(7)
        lm = AnytimeLinearModel([], ["one"], delta0=[0.3], g=0.25, intercept=False)
        tm = TMixtureSPRT(theta0=0.3, effect=0.5, stop_on_reject=False)
        a, b = lm.initial_state(), tm.initial_state()
        for _ in range(300):
            v = rnd.gauss(0.6, 2.0)
            a, oa = lm.step(a, {"one": 1.0, "y": v})
            b, ob = tm.step(b, {"x": v})
            self.assertAlmostEqual(oa.log_e_value, ob.log_bf, delta=1e-10 * max(1.0, abs(ob.log_bf)))
            self.assertEqual(oa.lower is None, ob.lower is None)
            if ob.lower is not None:
                self.assertAlmostEqual(oa.lower, ob.lower, delta=1e-10)
                self.assertAlmostEqual(oa.upper, ob.upper, delta=1e-10)
        self.assertAlmostEqual(oa.p_value, ob.p_value, delta=1e-10)

    def test_interval_endpoints_are_where_the_e_value_reaches_one_over_alpha(self):
        rows = regression_rows(200, 8, effect=0.5)
        proc = AnytimeLinearModel(["x1", "x2"], ["z"], g=0.3, alpha=0.1)
        _, outs = fold(proc, rows)
        for n in (60, 200):
            out = outs[n - 1]
            for end in (out.lower, out.upper):
                _, at = fold(AnytimeLinearModel(["x1", "x2"], ["z"], delta0=[end], g=0.3, alpha=0.1), rows[:n])
                self.assertAlmostEqual(at[-1].log_e_value, math.log(10.0), delta=1e-8)
            _, mid = fold(AnytimeLinearModel(["x1", "x2"], ["z"], delta0=[out.estimates[0]], g=0.3), rows[:n])
            self.assertLess(mid[-1].log_e_value, 0.0)

    def test_estimates_agree_with_exact_least_squares(self):
        rows = regression_rows(25, 9, effect=1.0, tested=2)
        _, outs = fold(AnytimeLinearModel(["x1", "x2"], ["z", "w"]), rows)
        xs = [[Fraction(1), Fraction(r["x1"]), Fraction(r["x2"]), Fraction(r["z"]), Fraction(r["w"])] for r in rows]
        gram = [[sum(r[i] * r[j] for r in xs) for j in range(5)] for i in range(5)]
        coef = solve(gram, [sum(r[i] * Fraction(row["y"]) for r, row in zip(xs, rows)) for i in range(5)])
        for got, want in zip(outs[-1].estimates, coef[3:]):
            self.assertAlmostEqual(got, float(want), delta=1e-12)

    def test_undefined_until_the_design_has_full_rank(self):
        proc = AnytimeLinearModel(["x1"], ["z"])
        rows = [{"x1": float(i), "z": 0.0, "y": float(i * i)} for i in range(5)]  # z constant: no information
        rows += [{"x1": 2.0 * i, "z": 1.0, "y": 1.0} for i in range(3)]
        _, outs = fold(proc, rows)
        self.assertTrue(all(o.estimates is None and o.log_e_value == 0.0 for o in outs[:5]))
        self.assertIsNotNone(outs[-1].estimates)
        # a covariate that duplicates another: never identified
        dup = AnytimeLinearModel(["x1", "x2"], ["z"])
        rows = [{"x1": float(i), "x2": 3.0 * i + 1.0, "z": float(i % 2), "y": float(i)} for i in range(20)]
        _, outs = fold(dup, rows)
        self.assertTrue(all(o.estimates is None and o.log_e_value == 0.0 for o in outs))

    def test_exact_fits_are_finite(self):
        proc = AnytimeLinearModel([], ["z"])
        _, outs = fold(proc, [{"z": float(i % 2), "y": 3.0} for i in range(30)])  # constant outcome
        self.assertEqual(outs[-1].estimates, (0.0,))
        self.assertLess(outs[-1].log_e_value, 0.0)
        _, outs = fold(proc, [{"z": float(i % 2), "y": 1.0 + 2.0 * (i % 2)} for i in range(30)])
        self.assertGreater(outs[-1].log_e_value, 10.0)
        self.assertAlmostEqual(outs[-1].lower, 2.0, delta=1e-12)
        self.assertEqual(outs[-1].lower, outs[-1].upper)


class MonteCarlo(unittest.TestCase):
    def simulate(self, runs, steps, seed, *, effect, delta0, covariates=("x1", "x2"), tested=("z",)):
        """Fraction of runs whose e-value ever reached 1/alpha, and of runs whose interval ever missed effect."""
        proc = AnytimeLinearModel(list(covariates), list(tested), delta0=list(delta0), g=0.5)
        rnd = random.Random(seed)
        rejected = missed = 0
        for _ in range(runs):
            state = proc.initial_state()
            ever_rejected = ever_missed = False
            for _ in range(steps):
                x1, x2 = rnd.gauss(0, 1), rnd.expovariate(1.0)
                z = 1.0 if rnd.random() < 0.5 else 0.0
                row = {"x1": x1, "x2": x2, "z": z, "y": 5.0 + 1.5 * x1 - 2.0 * x2 + effect * z + 3.0 * rnd.gauss(0, 1)}
                if len(tested) == 2:
                    row["w"] = rnd.gauss(0, 1)
                    row["y"] += effect * row["w"]
                state, out = proc.step(state, row)
                if out.log_e_value >= math.log(1 / proc.alpha):
                    ever_rejected = True
                if out.lower is not None and not out.lower < effect < out.upper:
                    ever_missed = True
            rejected += ever_rejected
            missed += ever_missed
            self.assertEqual(state.decision is not None, ever_rejected)
        return rejected / runs, missed / runs

    def test_type_one_error_under_continuous_monitoring(self):
        runs = 400
        rate, _ = self.simulate(runs, 500, 11, effect=0.0, delta0=(0.0,))
        print(f"\n  linear model: type I error {rate:.4f} (400 runs x 500 observations, alpha 0.05)")
        self.assertLessEqual(rate, binomial_upper_bound(0.05, runs))

    def test_type_one_error_two_coefficients(self):
        runs = 200
        rate, _ = self.simulate(runs, 300, 12, effect=0.0, delta0=(0.0, 0.0), tested=("z", "w"))
        print(f"\n  linear model, d = 2: type I error {rate:.4f} (200 runs x 300 observations)")
        self.assertLessEqual(rate, binomial_upper_bound(0.05, runs))

    def test_confidence_sequence_covers_uniformly_in_time(self):
        runs = 300
        rate, missed = self.simulate(runs, 500, 13, effect=0.7, delta0=(0.7,))
        print(f"\n  linear model: confidence sequence missed the effect at some time in {missed:.4f} of runs")
        self.assertLessEqual(missed, binomial_upper_bound(0.05, runs))
        self.assertEqual(rate, missed)  # the interval is exactly the set of delta0 the test keeps

    def test_power_against_a_true_effect(self):
        rate, _ = self.simulate(100, 500, 14, effect=1.5, delta0=(0.0,))
        self.assertGreater(rate, 0.8)


class Precision(unittest.TestCase):
    def test_covariate_adjustment_shortens_the_interval(self):
        """With x explaining 9/10 of the variance, the half-width shrinks by about sqrt(1/10)."""
        adjusted = AnytimeLinearModel(["x"], ["z"], g=0.5)
        plain = AnytimeLinearModel([], ["z"], g=0.5)
        ratios, wins = [], 0
        for seed in range(10):
            rnd = random.Random(100 + seed)
            a, b = adjusted.initial_state(), plain.initial_state()
            for _ in range(1000):
                x, z = rnd.gauss(0, 1), float(rnd.random() < 0.5)
                row = {"x": x, "z": z, "y": 2.0 + 3.0 * x + 0.2 * z + rnd.gauss(0, 1)}
                a, oa = adjusted.step(a, row)
                b, ob = plain.step(b, {"z": z, "y": row["y"]})
            ratios.append((oa.upper - oa.lower) / (ob.upper - ob.lower))
            wins += oa.log_e_value > ob.log_e_value
        mean = math.fsum(ratios) / len(ratios)
        print(f"\n  linear model: adjusted / unadjusted interval width {mean:.4f} (theory about {math.sqrt(0.1):.4f})")
        self.assertTrue(all(r < 0.5 for r in ratios))
        self.assertAlmostEqual(mean, math.sqrt(0.1), delta=0.03)
        self.assertGreaterEqual(wins, 9)


class Invariance(unittest.TestCase):
    def run_rows(self, proc, rows):
        return [(o.log_e_value, o.lower, o.upper) for o in fold(proc, rows)[1]]

    def assert_same(self, first, second, scale=1.0, shift=0.0):
        for (e1, l1, u1), (e2, l2, u2) in zip(first, second):
            self.assertAlmostEqual(e1, e2, delta=1e-8 * max(1.0, abs(e1)))
            if scale < 0:
                l2, u2 = u2, l2
            self.assertEqual(l1 is None, l2 is None)
            if l1 is not None:
                for v1, v2 in ((l1, l2), (u1, u2)):
                    self.assertAlmostEqual(scale * v1 + shift, v2, delta=1e-8 * (abs(scale) + abs(v2)))

    def test_outcome_rescaled_shifted_and_moved_along_the_covariates(self):
        rows = regression_rows(150, 21, effect=0.4)
        base = self.run_rows(AnytimeLinearModel(["x1", "x2"], ["z"], delta0=[0.3]), rows)
        for a, b in ((1000.0, -5e4), (1e-3, 7.0), (-2.5, 1.0)):
            moved = [dict(r, y=a * r["y"] + b + 4.0 * r["x1"] - r["x2"]) for r in rows]
            got = self.run_rows(AnytimeLinearModel(["x1", "x2"], ["z"], delta0=[a * 0.3]), moved)
            self.assert_same(base, got, scale=a)

    def test_covariates_transformed_affinely(self):
        rows = regression_rows(150, 22, effect=0.4)
        base = self.run_rows(AnytimeLinearModel(["x1", "x2"], ["z"]), rows)
        moved = [{"u": 3.0 * r["x1"] - r["x2"] + 100.0, "v": 0.5 * r["x1"] + 2.0 * r["x2"] - 7.0,
                  "z": r["z"] + 0.1 * r["x1"] - 2.0, "y": r["y"]} for r in rows]
        got = self.run_rows(AnytimeLinearModel(["u", "v"], ["z"]), moved)
        self.assert_same(base, got)

    def test_order_of_the_covariates(self):
        rows = regression_rows(80, 23, tested=2)
        base = self.run_rows(AnytimeLinearModel(["x1", "x2"], ["z", "w"], delta0=[0.1, 0.2]), rows)
        got = self.run_rows(AnytimeLinearModel(["x2", "x1"], ["w", "z"], delta0=[0.2, 0.1]), rows)
        self.assert_same(base, got)


class StatesAndArguments(unittest.TestCase):
    def test_round_trip_and_continuation(self):
        proc = AnytimeLinearModel(["x1", "x2"], ["z"], delta0=[0.1], g=0.4)
        rows = regression_rows(120, 31, effect=1.0)
        whole, outs = fold(proc, rows)
        state, _ = fold(proc, rows[:60])
        data = json.loads(json.dumps(proc.encode_state(state)))
        decoded = proc.decode_state(data)
        self.assertEqual(decoded, state)
        for x in rows[60:]:
            decoded, out = proc.step(decoded, x)
        self.assertEqual(decoded, whole)
        self.assertEqual(out, outs[-1])
        self.assertEqual(len(json.dumps(proc.encode_state(whole))), len(json.dumps(proc.encode_state(decoded))))

    def test_state_size_does_not_grow(self):
        proc = AnytimeLinearModel(["x1"], ["z"])
        rows = regression_rows(400, 32)
        sizes = {len(json.loads(json.dumps(proc.encode_state(fold(proc, rows[:n])[0])))["comoment"]) for n in (5, 400)}
        self.assertEqual(sizes, {6})

    def test_malformed_states_are_rejected(self):
        proc = AnytimeLinearModel(["x1"], ["z"])
        good = proc.encode_state(fold(proc, regression_rows(50, 33))[0])
        good = json.loads(json.dumps(good))
        bad = [
            {**good, "mean": good["mean"][:2]},
            {**good, "comoment": good["comoment"][:5]},
            {**good, "comoment": [-1.0, *good["comoment"][1:]]},
            {**good, "comoment": [good["comoment"][0], 1e6, *good["comoment"][2:]]},  # Cauchy-Schwarz
            {**good, "n": -1},
            {**good, "n": 1.5},
            {**good, "decision": "reject_h0"},  # with a small e-value
            {**good, "decision": "accept_h0"},
            {**good, "max_log_e_value": 100.0},  # large, without a decision
            {**good, "max_log_e_value": -1.0},
            {**good, "mean": [float("nan"), 0.0, 0.0]},
            {**good, "extra": 1},
            {k: v for k, v in good.items() if k != "n"},
            {**proc.encode_state(proc.initial_state()), "mean": [1.0, 0.0, 0.0]},
            [],
        ]
        for data in bad:
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                proc.decode_state(data)
        without = AnytimeLinearModel(["x1"], ["z"], intercept=False)
        with self.assertRaises(ValueError):
            without.decode_state(good)  # means where there is no intercept

    def test_arguments(self):
        bad = [
            dict(covariates=["x"], tested=[]),
            dict(covariates="x", tested=["z"]),
            dict(covariates=["x"], tested="z"),
            dict(covariates=["x", "x"], tested=["z"]),
            dict(covariates=["x"], tested=["x"]),
            dict(covariates=["y"], tested=["z"]),
            dict(covariates=[""], tested=["z"]),
            dict(covariates=[], tested=["z"], delta0=[0.0, 1.0]),
            dict(covariates=[], tested=["z"], delta0=[math.inf]),
            dict(covariates=[], tested=["z"], delta0=0.0),
            dict(covariates=[], tested=["z"], g=0.0),
            dict(covariates=[], tested=["z"], g=-1.0),
            dict(covariates=[], tested=["z"], g=math.nan),
            dict(covariates=[], tested=["z"], alpha=0.0),
            dict(covariates=[], tested=["z"], alpha=1.0),
            dict(covariates=[], tested=["z"], intercept=1),
            dict(covariates=[], tested=["z"], input=""),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AnytimeLinearModel(**kwargs)
        proc = AnytimeLinearModel(["x"], ["z", "w"], delta0=[1, 2], g=2)
        self.assertEqual(proc.config(), {"covariates": ["x"], "tested": ["z", "w"], "delta0": [1.0, 2.0], "g": 2.0,
                                         "alpha": 0.05, "intercept": True, "stop_on_reject": False, "input": "y"})
        self.assertEqual(set(proc.inputs()), {"x", "z", "w", "y"})
        self.assertEqual(proc.e_value_field, "log_e_value")

    def test_inputs_are_joint_and_checked(self):
        proc = AnytimeLinearModel(["x1"], ["z"])
        with self.assertRaises(ContractViolation):
            trajectory(proc, [{"x1": 1.0, "y": 2.0}])
        with self.assertRaises(ContractViolation):
            trajectory(proc, [{"x1": 1.0, "z": True, "y": 2.0}])
        rows = [{k: r[k] for k in ("x1", "z", "y")} for r in regression_rows(10, 34)]
        self.assertEqual(len(trajectory(proc, rows)), 10)

    def test_stop_on_reject_and_the_p_value(self):
        proc = AnytimeLinearModel(["x1"], ["z"], stop_on_reject=True)
        rows = regression_rows(400, 35, effect=3.0)
        history = trajectory(proc, [{k: r[k] for k in ("x1", "z", "y")} for r in rows])
        self.assertLess(len(history), 400)
        state, out = history[-1]
        self.assertTrue(proc.is_terminal(state))
        self.assertEqual(out.decision, "reject_h0")
        self.assertLessEqual(out.p_value, 0.05)
        p_values = [o.p_value for _, o in history]
        self.assertEqual(p_values, sorted(p_values, reverse=True))


if __name__ == "__main__":
    unittest.main()
