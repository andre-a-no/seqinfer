"""Operating characteristics of the SPRT and the 2-SPRT (Table 5 of the paper).

Gaussian observations with unit variance, theta0 = 0 against theta1 = 0.5,
nominal error rates 0.05.  Three tests are compared:

  SPRT               Wald's thresholds
  2-SPRT, bound      thresholds log(1/alpha): error rates guaranteed, conservative
  2-SPRT, matched    thresholds calibrated by simulation so that the 2-SPRT
                     attains the same error rate as the SPRT

The procedures are driven by a bare loop: a Monte Carlo study needs the
transition function and nothing else.  Calibration and evaluation use
separate random streams.

Usage: python examples/04_kiefer_weiss_oc.py [replications]

The default 20000 replications take several minutes; pass e.g. 2000 for a
quick look (with correspondingly wider Monte Carlo error).
"""
import sys

from seqinfer import SplitMix64
from seqinfer.procedures import ACCEPT_H0, REJECT_H0, SPRT, Gaussian, TwoSPRT

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
SEED = 20261003
THETA0, THETA1 = 0.0, 0.5
THETAS = (0.0, 0.25, 0.5)
family = Gaussian(1.0)


def simulate(proc, theta, rng):
    state = proc.initial_state()
    while not proc.is_terminal(state):
        state, _ = proc.step(state, {"x": rng.normal(theta, 1.0)}, None)
    return state.n, state.decision


def evaluate(proc, theta, role):
    rng = SplitMix64.for_role(SEED, role)
    ns, rejects = [], 0
    for _ in range(REPS):
        n, decision = simulate(proc, theta, rng)
        ns.append(n)
        rejects += decision == REJECT_H0
    mean = sum(ns) / REPS
    se = (sum((n - mean) ** 2 for n in ns) / (REPS - 1) / REPS) ** 0.5
    return mean, se, max(ns), rejects / REPS


def matched_threshold(target):
    """Bisect the common threshold a0 = a1 until the 2-SPRT's error rate meets `target`.

    Every evaluation reuses the same calibration stream (common random
    numbers), so the estimated error rate is a monotone step function of
    the threshold.  By symmetry both error rates are equal.
    """
    def error(a):
        proc = TwoSPRT(family, THETA0, THETA1, thresholds=(a, a))
        rng = SplitMix64.for_role(SEED, "simulation/calibration")
        return sum(simulate(proc, THETA0, rng)[1] == REJECT_H0 for _ in range(REPS)) / REPS

    lo, hi = 1.0, 3.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if error(mid) > target else (lo, mid)
    return hi


sprt = SPRT(family, THETA0, THETA1, alpha=0.05, beta=0.05)
bound = TwoSPRT(family, THETA0, THETA1, alpha0=0.05, alpha1=0.05)

results = {"SPRT": {th: evaluate(sprt, th, f"simulation/theta={th}") for th in THETAS}}
sprt_error = 0.5 * (results["SPRT"][THETA0][3] + 1.0 - results["SPRT"][THETA1][3])
a = matched_threshold(sprt_error)
matched = TwoSPRT(family, THETA0, THETA1, thresholds=(a, a))
# the same evaluation data for all three tests at a given theta
results["2-SPRT, bound"] = {th: evaluate(bound, th, f"simulation/theta={th}") for th in THETAS}
results["2-SPRT, matched"] = {th: evaluate(matched, th, f"simulation/theta={th}") for th in THETAS}

print(f"replications per cell: {REPS}, seed {SEED}")
print(f"SPRT thresholds: {sprt.lower:.4f}, {sprt.upper:.4f}; attained error rate {sprt_error:.4f}")
print(f"2-SPRT, bound:   a = {bound.a0:.4f}, theta* = {bound.theta_star}, n_max = {bound.max_sample_size()}")
print(f"2-SPRT, matched: a = {matched.a0:.4f}, theta* = {matched.theta_star}, n_max = {matched.max_sample_size()}")
print(f"\n{'test':16s} {'theta':>6s} {'E[N]':>7s} {'s.e.':>5s} {'max N':>6s} {'P(reject H0)':>13s}")
for name, rows in results.items():
    for theta, (mean, se, longest, reject) in rows.items():
        print(f"{name:16s} {theta:6.2f} {mean:7.2f} {se:5.2f} {longest:6d} {reject:13.4f}")
