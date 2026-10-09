# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Write the conformance vectors in conformance/.

An implementation of seqinfer in any language conforms when, for every
case, folding its procedure over `inputs` from the default initial state
(with `seed` for randomized procedures) reproduces `outputs`, `state`,
`rng` and `history_digest` exactly.  Floats are compared through their
RFC 8785 encoding, i.e. bit for bit.

    python tools/make_conformance.py          # rewrite the vectors
    python -m unittest tests.test_conformance # check this implementation

The vectors change only when a procedure's version does.
"""
from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from seqinfer import Run  # noqa: E402
from seqinfer.canonical import _number, canonical_json  # noqa: E402
from seqinfer.procedures import (  # noqa: E402
    CUSUM,
    EMA,
    SPRT,
    Bernoulli,
    BettingMeanTest,
    BootstrapParticleFilter,
    Exponential,
    Gaussian,
    GroupSequentialTest,
    LocalLevelKalman,
    MeanDifference,
    NormalMixtureSPRT,
    PlanTest,
    Poisson,
    SequentialSignTest,
    ShiryaevRoberts,
    TwoSPRT,
)

OUT = ROOT / "conformance"


def cases():
    rnd = random.Random(20261009)

    def normal(mu, n, sd=1.0):
        return [{"x": rnd.gauss(mu, sd)} for _ in range(n)]

    def poisson(lam, n):
        out = []
        for _ in range(n):
            k, prod, limit = 0, rnd.random(), 2.718281828459045 ** -lam
            while prod > limit:
                k += 1
                prod *= rnd.random()
            out.append({"x": k})
        return out

    def bernoulli(p, n):
        return [{"x": int(rnd.random() < p)} for _ in range(n)]

    two_sprt = TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.05, 0.05)
    level = [{"y": 0.05 * i + rnd.gauss(0.0, 1.0)} for i in range(40)]
    return [
        ("sprt_gaussian", SPRT(Gaussian(1.0), 0.0, 0.5, 0.05, 0.05), None, normal(0.25, 60)),
        ("sprt_bernoulli", SPRT(Bernoulli(), 0.3, 0.5, 0.05, 0.05), None, bernoulli(0.4, 120)),
        ("sprt_poisson", SPRT(Poisson(), 2.0, 3.0, 0.05, 0.05), None, poisson(2.5, 80)),
        ("two_sprt", TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.05, 0.05), None, normal(0.25, 60)),
        ("plan_test", PlanTest(two_sprt.as_plan(), label="2-SPRT"), None, normal(0.25, 60)),
        ("cusum", CUSUM(Gaussian(1.0), 0.0, 1.0, 8.0), None, normal(0.0, 30) + normal(1.0, 40)),
        ("shiryaev_roberts_exponential", ShiryaevRoberts(Exponential(), 1.0, 0.5, 200.0), None,
         [{"x": rnd.expovariate(1.0)} for _ in range(30)] + [{"x": rnd.expovariate(0.5)} for _ in range(60)]),
        ("ema", EMA(0.2), None, normal(1.0, 30)),
        ("kalman_control", LocalLevelKalman(0.01, 1.0, control=True), None,
         [{"y": rnd.gauss(0.1 * i, 1.0), "u": 0.1} for i in range(40)]),
        ("particle_filter", BootstrapParticleFilter(0.01, 1.0, particles=32), 7, level),
        ("mean_difference", MeanDifference(), None,
         [{"a": rnd.gauss(1.0, 1.0)} if i % 3 else {"b": rnd.gauss(0.0, 2.0)} for i in range(45)]),
        ("mixture_sprt", NormalMixtureSPRT(1.0, 0.5, stop_on_reject=False), None, normal(0.3, 80)),
        ("betting_mean", BettingMeanTest(0.3, 0.0, 2.0, alpha=1e-4, stop_on_reject=False), None,
         [{"x": 2.0 * rnd.betavariate(1.0, 2.0)} for _ in range(80)]),
        ("sign_test", SequentialSignTest(0.0, alpha=1e-4, stop_on_reject=False), None,
         [{"x": float(rnd.choice([-2, -1, 0, 1, 2, 3]))} for _ in range(80)]),
        ("group_sequential", GroupSequentialTest(1.0, [20, 40, 60], [3.71, 2.51, 1.99]), None, normal(0.2, 60)),
    ]


def vector(label, procedure, seed, inputs):
    run = Run(procedure, seed=seed, run_id=label, clock=lambda: 0.0)
    outputs = []
    for x in inputs:
        event = run.step(x)
        if event is None:
            break
        outputs.append(procedure.encode_output(event.output))
    used = inputs[: run.t]
    return {
        "case": label,
        "procedure": procedure.identity(),
        "seed": None if seed is None else str(seed),
        "inputs": used,
        "outputs": outputs,
        "state": procedure.encode_state(run.state),
        "rng": None if run.rng_state is None else f"{run.rng_state:016x}",
        "terminal": run.terminal,
        "history_digest": run.history_digest,
    }


def number_vectors(count=500):
    rnd = random.Random(8785)
    values = [0.0, -0.0, 5e-324, 1.7976931348623157e308, 1e21, 1e-7, 1e-6, 0.1, 1 / 3, 2.0**53, 123.0, -1.5]
    while len(values) < count:
        x = struct.unpack("<d", struct.pack("<Q", rnd.getrandbits(64)))[0]
        if x == x and abs(x) != float("inf"):
            values.append(x)
    return [{"bits": f"{struct.unpack('<Q', struct.pack('<d', v))[0]:016x}", "json": _number(v)} for v in values]


def main():
    OUT.mkdir(exist_ok=True)
    vectors = [vector(*case) for case in cases()]
    (OUT / "procedures.json").write_text(canonical_json(vectors) + "\n", encoding="utf-8")
    (OUT / "numbers.json").write_text(canonical_json(number_vectors()) + "\n", encoding="utf-8")
    print(f"wrote {len(vectors)} procedure vectors and the number vectors to {OUT}")


if __name__ == "__main__":
    main()
