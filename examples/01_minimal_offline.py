# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The smallest instantiation: a procedure and an iterator.

No adapters, no topology, no runtime, no persistence.  `trajectory` is the
reference fold of the transition function over an ordered history.
"""
from seqinfer import trajectory
from seqinfer.procedures import Gaussian, TwoSPRT
from seqinfer.sources import gaussian

procedure = TwoSPRT(Gaussian(sigma=1.0), theta0=0.0, theta1=0.5, alpha0=0.05, alpha1=0.05)
history = [{"x": obs.value} for obs in gaussian("x", mean=0.5, sd=1.0, seed=7, n=200)]

for state, output in trajectory(procedure, history):
    print(f"n={output.n:3d}  L0={output.l0:7.3f}  L1={output.l1:7.3f}  {output.decision or ''}")

print(f"\ntheta* = {procedure.theta_star}, guaranteed to stop by n = {procedure.max_sample_size()}")
