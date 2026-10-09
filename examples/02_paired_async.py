# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Two instruments, asynchronous arrival, paired analysis.

The procedure is a plain synchronous SPRT on differences.  Pairing is done
by the topology, concurrency by the runtime.  Because positional pairing is
confluent, the result does not depend on how the two streams interleave:
the asynchronous run reproduces the synchronous one exactly.
"""
import asyncio
import random

from seqinfer import PositionalPair, Recorder, Run, difference, run_async, run_sync
from seqinfer.procedures import SPRT, Gaussian
from seqinfer.sources import as_async, gaussian


def make_run():
    recorder = Recorder()
    topology = PositionalPair(("treated", "control")).map(difference("treated", "control"), "treated-control")
    procedure = SPRT(Gaussian(sigma=2**0.5), theta0=0.0, theta1=0.8, alpha=0.01, beta=0.01)
    return Run(procedure, topology=topology, consumers=[recorder]), recorder


treated = list(gaussian("treated", 10.8, 1.0, seed=1, n=500, source="instrument-A"))
control = list(gaussian("control", 10.0, 1.0, seed=1, n=500, source="instrument-B"))

sync_run, sync_rec = make_run()
run_sync(sync_run, treated + control)  # all of A, then all of B

jitter = random.Random(0)
async_run, async_rec = make_run()
asyncio.run(
    run_async(
        async_run,
        [
            as_async(treated, lambda: jitter.uniform(0, 0.002)),
            as_async(control, lambda: jitter.uniform(0, 0.004)),
        ],
        maxsize=8,
    )
)

last = async_rec.outputs[-1]
print(f"decision after {last.n} pairs: {last.decision} (log LR {last.llr:.3f})")
print("identical to the synchronous run:", async_rec.outputs == sync_rec.outputs)
print("history digest:", async_run.history_digest[:16], "==", sync_run.history_digest[:16])
