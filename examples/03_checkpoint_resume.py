# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Interrupt a randomized run, restore it in a "new process", continue.

The checkpoint holds the inferential state, the statistical random state,
the topology buffers, the source positions and the provenance.  It holds
nothing of the runtime.  After the restore the source is rewound too far on
purpose; delivery drops what the run has already seen.
"""
import tempfile
from pathlib import Path

from seqinfer import Delivery, Recorder, Run, load_checkpoint, run_sync, save_checkpoint, trajectory
from seqinfer.procedures import BootstrapParticleFilter
from seqinfer.sources import gaussian


def procedure():
    return BootstrapParticleFilter(q=0.02, r=0.5, particles=200)


observations = list(gaussian("y", lambda i: 0.02 * i, 0.7, seed=3, n=300, source="sensor"))
recorder = Recorder()

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "run.checkpoint.json"

    run = Run(procedure(), delivery=Delivery("sequence"), consumers=[recorder], seed=2024)
    run_sync(run, observations[:120])
    save_checkpoint(path, run.checkpoint())
    print(f"checkpoint at t={run.t}, {path.stat().st_size} bytes, source positions {run.delivery.positions()}")
    del run  # the runtime is disposable

    run = Run.restore(procedure(), load_checkpoint(path), delivery=Delivery("sequence"), consumers=[recorder])
    run_sync(run, observations[100:])
    print(f"resumed to t={run.t}; redelivered duplicates dropped: {run.delivery.duplicates}")

uninterrupted = [o for _, o in trajectory(procedure(), [{"y": o.value} for o in observations], seed=2024)]
print("identical to the uninterrupted trajectory:", recorder.outputs == uninterrupted)
print("lifecycle:", [e["event"] for e in run.provenance["events"]])
