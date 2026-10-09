# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Design an optimal plan, then run it.

A response rate of 30% (H0) is tested against 50% (H1) with error rates
0.05 and at most 150 patients.  `kiefer_weiss_plan` computes the plan by
backward induction; `operating_characteristic` gives its exact error
rates and average sample numbers, with no simulation.  The plan is then
executed on a simulated stream like any other procedure.  At the end the
same design problem is solved for counts and for normal measurements.
"""
from seqinfer import Recorder, Run, run_sync
from seqinfer.design import kiefer_weiss_plan, operating_characteristic
from seqinfer.procedures import Bernoulli, Gaussian, Poisson
from seqinfer.rng import SplitMix64
from seqinfer.sources import from_values

design = kiefer_weiss_plan(theta0=0.3, theta1=0.5, alpha0=0.05, alpha1=0.05, horizon=150)
plan = design.plan
print(f"theta* = {design.theta_star:.4f}, at most {plan.horizon} observations")
print(f"exact errors: alpha0 = {design.at_theta0.reject:.4f}, alpha1 = {design.at_theta1.accept:.4f}")

print("\n theta   P(reject H0)   E[N]")
for theta in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55):
    oc = operating_characteristic(plan, Bernoulli(), theta)
    print(f"  {theta:.2f}      {oc.reject:.4f}     {oc.expected_n:6.2f}")

rng = SplitMix64.for_role(1, "simulation/patients")
outcomes = from_values("x", (int(rng.random() < 0.45) for _ in range(plan.horizon)))
recorder = Recorder()
run = Run(plan, consumers=[recorder])
run_sync(run, outcomes)
last = recorder.outputs[-1]
print(f"\nsimulated trial at 45%: {last.decision} after {last.n} patients ({int(last.statistic)} responses)")

print("\nthe same problem for other data:")
for family, theta0, theta1 in ((Poisson(), 2.0, 3.0), (Gaussian(sigma=1.0), 0.0, 0.5)):
    d = kiefer_weiss_plan(theta0, theta1, 0.05, 0.05, horizon=120, family=family)
    kind = "exact" if d.exact else "on a lattice"
    print(f"  {family.spec()['family']:8s} {theta0} vs {theta1}: E[N] at theta* = {d.at_theta_star.expected_n:.1f}, "
          f"at most {d.plan.horizon} observations, errors {d.at_theta0.reject:.4f} / {d.at_theta1.accept:.4f} ({kind})")
