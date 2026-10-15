# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""One run, three executors in turn: Python, then the TypeScript executor, then Python again.

The run is checkpointed by Python, continued by the TypeScript executor
(ts/, through its command line) from that checkpoint, and its checkpoint
restored by Python, which finishes the run.  The history digest must be
the one of the uninterrupted Python run, bit for bit, and the state and
outputs equal (to the last bit where the arithmetic is +, -, *, /, sqrt;
within 1e-12 where exp and log are involved, whose last bits the two
platforms' libraries may round differently; a Neumaier compensation term,
the low-order part of a sum, is compared on the scale of that sum).

Skipped unless node is installed and the executor is built
(cd ts && npm ci && npm run build); with REQUIRE_TYPESCRIPT set it fails instead.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sequential_analysis import Delivery, Independent, KeyJoin, Recorder, Run, equivalent, run_sync
from sequential_analysis.procedures import (
    SPRT,
    ABTest,
    Bernoulli,
    MeanEDetector,
    NormalMixtureSPRT,
    TwoProportionTest,
)
from sequential_analysis.sources import from_values

from .helpers import ab_units

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "ts" / "dist" / "src" / "cli.js"
NODE = os.environ.get("NODE") or shutil.which("node")


def cases():
    values = [((i * 7919) % 101) / 101 for i in range(120)]
    return [
        ("mixture SPRT", lambda: NormalMixtureSPRT(1.0, 0.5, stop_on_reject=False), Independent, Delivery,
         list(from_values("x", [v - 0.3 for v in values]))),
        ("Bernoulli SPRT, sequence policy", lambda: SPRT(Bernoulli(), 0.3, 0.6, 1e-6, 1e-6), Independent,
         lambda: Delivery("sequence"), list(from_values("x", [int(v < 0.45) for v in values]))),
        ("two proportions, joined events", lambda: TwoProportionTest(alpha=1e-9), lambda: KeyJoin(("arm", "y")),
         Delivery, [o for o in ab_units(120, seed=5) if o.input != "x"]),
        ("A/B test with covariate", lambda: ABTest(covariate=True, start=20), lambda: KeyJoin(("arm", "y", "x")),
         Delivery, ab_units(120, seed=6)),
        ("e-detector", lambda: MeanEDetector(0.5, arl=1e9), Independent, Delivery, list(from_values("x", values))),
    ]


def normalized(value):
    return json.loads(json.dumps(value))


@unittest.skipUnless((NODE and CLI.exists()) or os.environ.get("REQUIRE_TYPESCRIPT"),
                     "node or the built TypeScript executor (ts/dist) is missing")
class PythonTypeScriptPython(unittest.TestCase):
    def test_a_run_moves_between_executors(self):
        for label, make, topology, delivery, observations in cases():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                full_rec = Recorder()
                full = Run(make(), topology=topology(), delivery=delivery(), consumers=[full_rec], run_id="r",
                           clock=lambda: 0.0)
                run_sync(full, observations, stop_on_terminal=False)

                a, b = len(observations) // 3, 2 * len(observations) // 3
                rec = Recorder()
                first = Run(make(), topology=topology(), delivery=delivery(), consumers=[rec], run_id="r",
                            clock=lambda: 0.0)
                run_sync(first, observations[:a], stop_on_terminal=False)
                (tmp / "cp.json").write_text(json.dumps(first.checkpoint()))
                (tmp / "obs.jsonl").write_text("".join(json.dumps(o.to_json()) + "\n" for o in observations[a:b]))
                done = subprocess.run(
                    [NODE, str(CLI), "continue", str(tmp / "cp.json"), str(tmp / "obs.jsonl"),
                     "--checkpoint", str(tmp / "cp2.json"), "--outputs", str(tmp / "out.jsonl")],
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(done.returncode, 0, done.stderr)
                ts_outputs = [json.loads(line)["output"] for line in (tmp / "out.jsonl").read_text().splitlines()]

                second = Run.restore(make(), json.loads((tmp / "cp2.json").read_text()), topology=topology(),
                                     delivery=delivery(), consumers=[rec], clock=lambda: 0.0)
                run_sync(second, observations[b:], stop_on_terminal=False)

                proc = full.procedure
                self.assertEqual(second.history_digest, full.history_digest)
                self.assertEqual(second.t, full.t)
                self.assertTrue(equivalent(normalized(proc.encode_state(second.state)),
                                           normalized(proc.encode_state(full.state)), abs_tol=1e-12, rel_tol=1e-12))
                expected = normalized([proc.encode_output(o) for o in full_rec.outputs])
                got = normalized([proc.encode_output(o) for o in rec.outputs[:first.t]]) + ts_outputs + normalized(
                    [proc.encode_output(o) for o in rec.outputs[first.t:]])
                self.assertEqual(len(got), len(expected))
                self.assertTrue(equivalent(got, expected, abs_tol=1e-12, rel_tol=1e-12))
                events = [e["event"] for e in second.provenance["events"]]
                self.assertEqual(events.count("restore"), 2)  # once in TypeScript, once in Python


if __name__ == "__main__":
    unittest.main()
