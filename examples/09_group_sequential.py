# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""A group sequential trial with O'Brien-Fleming alpha spending.

Five equally spaced analyses, one-sided alpha 0.025, 90% power for an
effect of half a standard deviation.  The design gives the boundaries,
the maximum sample size and the expected sample size; the trial then
runs on a simulated stream and looks at the data only at the analyses.
"""
from seqinfer import Recorder, Run, run_sync
from seqinfer.group_sequential import group_sequential_design
from seqinfer.procedures import GroupSequentialTest
from seqinfer.sources import gaussian

effect, sigma = 0.5, 1.0
design = group_sequential_design(alpha=0.025, fractions=[0.2, 0.4, 0.6, 0.8, 1.0], spending="obrien-fleming")
n_max = design.max_sample_size(effect=effect, sigma=sigma, power=0.9)
drift = effect * n_max**0.5 / sigma

print("analysis  fraction  boundary  alpha spent  P(stop here | effect)")
rows = zip(design.fractions, design.bounds, design.spent, design.crossing(drift), strict=True)
for k, (t, c, a, p) in enumerate(rows, start=1):
    print(f"   {k}        {t:.1f}      {c:6.3f}     {a:.5f}       {p:.3f}")
print(f"\nmaximum sample size {n_max}; expected {design.expected_fraction(drift) * n_max:.1f} under the effect, "
      f"{design.expected_fraction(0.0) * n_max:.1f} under H0")

trial = GroupSequentialTest.from_design(design, n_max, sigma=sigma, label="OBF, 5 looks")
recorder = Recorder()
run = Run(trial, consumers=[recorder])
run_sync(run, gaussian("x", mean=0.4, sd=sigma, seed=11, n=n_max))
for out in recorder.outputs:
    if out.analysis is not None:
        print(f"analysis {out.analysis} at n={out.n}: Z = {out.z:+.3f} vs {out.bound:.3f}  {out.decision or ''}")
