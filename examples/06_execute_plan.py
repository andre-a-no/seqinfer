# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Design offline, execute on a stream.

A sampling plan is a table of continuation intervals for the running sum.
It can come from any design code, for example from a numerical solution of
the Kiefer-Weiss problem.  Here the plan is derived from the 2-SPRT so that
the example is self-contained; it is written to JSON, read back, and
executed by `PlanTest` inside a run with checkpointing and provenance like
any other procedure.
"""
import json
import tempfile
from pathlib import Path

from seqinfer import Recorder, Run, run_sync
from seqinfer.procedures import Gaussian, PlanTest, TwoSPRT
from seqinfer.sources import gaussian

design = TwoSPRT(Gaussian(1.0), theta0=0.0, theta1=0.5, alpha0=0.05, alpha1=0.05)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "plan.json"
    path.write_text(json.dumps({"label": "2-SPRT, theta*=0.25", "continuation": design.as_plan()}))
    stored = json.loads(path.read_text())

procedure = PlanTest(stored["continuation"], label=stored["label"])
print(f"plan with horizon {procedure.horizon}; first steps: "
      + ", ".join(f"({lo:.2f}, {hi:.2f})" for lo, hi in procedure.plan[:3]) + ", ...")

recorder = Recorder()
run = Run(procedure, consumers=[recorder])
run_sync(run, gaussian("x", mean=0.5, sd=1.0, seed=7))
last = recorder.outputs[-1]
print(f"decision after {last.n} observations: {last.decision} (sum {last.statistic:.3f})")

reference = Run(design)
run_sync(reference, gaussian("x", mean=0.5, sd=1.0, seed=7))
print("same stopping time and decision as the 2-SPRT:",
      (reference.state.n, reference.state.decision) == (last.n, last.decision))
print("plan recorded in provenance:", run.provenance["procedure"]["config"]["label"])
