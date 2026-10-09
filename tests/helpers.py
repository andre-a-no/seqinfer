# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Shared fixtures: procedures, observation histories and trace helpers."""
import json
import random

from seqinfer import (
    Chain,
    Delivery,
    Independent,
    PositionalPair,
    Recorder,
    Run,
    TimeAlign,
    difference,
)
from seqinfer.procedures import (
    CUSUM,
    EMA,
    SPRT,
    BootstrapParticleFilter,
    Gaussian,
    LocalLevelKalman,
    MeanDifference,
    PlanTest,
    TwoSPRT,
)
from seqinfer.sources import from_values, gaussian, interleave


def fixed_clock():
    return 0.0


def summary(run: Run, recorder: Recorder) -> dict:
    """Everything that defines the inferential trajectory of a run, JSON-normalized."""
    p = run.procedure
    return json.loads(
        json.dumps(
            {
                "t": run.t,
                "state": p.encode_state(run.state),
                "rng": run.rng_state,
                "terminal": run.terminal,
                "digest": run.history_digest,
                "outputs": [p.encode_output(o) for o in recorder.outputs],
            }
        )
    )


def reference(procedure, pairs) -> list:
    return json.loads(json.dumps([procedure.encode_output(o) for _, o in pairs]))


def unbalanced_pairs(n: int, lead: int, seed: int):
    """Two sources where `a` runs `lead` observations ahead of `b`."""
    a = list(gaussian("a", 0.4, 1.0, seed=seed, n=n))
    b = list(gaussian("b", 0.0, 1.0, seed=seed, n=n))
    out = a[:lead]
    for i in range(n - lead):
        out += [a[lead + i], b[i]]
    return out + b[n - lead:]


def jittered_times(n: int, seed: int):
    rnd = random.Random(seed)
    ta = [float(i) for i in range(n)]
    tb = [i + rnd.uniform(-0.6, 0.6) for i in range(n)]
    tb.sort()
    return ta, tb


def chain_ema_cusum():
    return Chain(
        EMA(0.2),
        CUSUM(Gaussian(1.0), 0.0, 1.0, threshold=6.0),
        link=lambda o: {"x": o.residual},
        link_name="residual",
    )


def cases():
    """(label, procedure factory, topology factory, delivery factory, observations, seed)."""
    n = 80
    ta, tb = jittered_times(n, 5)
    level = list(gaussian("y", lambda i: 0.05 * i, 1.0, seed=3, n=n))
    shift = list(gaussian("x", lambda i: 0.0 if i < 40 else 1.5, 1.0, seed=4, n=n))
    return [
        (
            "two_sprt",
            lambda: TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.001, 0.001),
            Independent,
            Delivery,
            list(gaussian("x", 0.25, 1.0, seed=1, n=n)),
            None,
        ),
        ("kalman", lambda: LocalLevelKalman(q=0.01, r=1.0), Independent, Delivery, level, None),
        (
            "particle_filter",
            lambda: BootstrapParticleFilter(q=0.01, r=1.0, particles=64),
            Independent,
            Delivery,
            level,
            11,
        ),
        (
            "independent_two_sample",
            MeanDifference,
            Independent,
            Delivery,
            list(interleave(gaussian("a", 1.0, 1.0, seed=2, n=n), gaussian("b", 0.0, 2.0, seed=2, n=n // 2))),
            None,
        ),
        (
            "paired_difference_sprt",
            lambda: SPRT(Gaussian(1.5), 0.0, 0.5, 1e-4, 1e-4),
            lambda: PositionalPair(("a", "b")).map(difference("a", "b"), "a-b"),
            Delivery,
            unbalanced_pairs(n, lead=5, seed=6),
            None,
        ),
        (
            "time_aligned_two_sample",
            MeanDifference,
            lambda: TimeAlign(("a", "b"), tolerance=0.3),
            Delivery,
            list(
                interleave(
                    from_values("a", [0.1 * i for i in range(n)], times=ta),
                    from_values("b", [0.2 * i for i in range(n)], times=tb),
                )
            ),
            None,
        ),
        ("chain_ema_cusum", chain_ema_cusum, Independent, Delivery, shift, None),
        (
            "plan_test",
            lambda: PlanTest(TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.001, 0.001).as_plan(), label="2-SPRT"),
            Independent,
            Delivery,
            list(gaussian("x", 0.25, 1.0, seed=12, n=n)),
            None,
        ),
    ]
