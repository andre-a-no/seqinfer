# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""This implementation reproduces the language-independent conformance vectors bit for bit."""
import json
import struct
import sys
import unittest
from pathlib import Path

from seqinfer import Run
from seqinfer.canonical import _number, canonical_json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_conformance  # noqa: E402


class Conformance(unittest.TestCase):
    def test_procedures(self):
        stored = {v["case"]: v for v in json.loads((ROOT / "conformance" / "procedures.json").read_text())}
        cases = make_conformance.cases()
        self.assertEqual(set(stored), {c[0] for c in cases})
        for label, procedure, seed, _ in cases:
            with self.subTest(case=label):
                expected = stored[label]
                self.assertEqual(procedure.identity(), expected["procedure"])
                run = Run(procedure, seed=None if expected["seed"] is None else int(expected["seed"]))
                outputs = [procedure.encode_output(run.step(x).output) for x in expected["inputs"]]
                self.assertEqual(canonical_json(outputs), canonical_json(expected["outputs"]))
                self.assertEqual(canonical_json(procedure.encode_state(run.state)), canonical_json(expected["state"]))
                self.assertEqual(None if run.rng_state is None else f"{run.rng_state:016x}", expected["rng"])
                self.assertEqual(run.terminal, expected["terminal"])
                self.assertEqual(run.history_digest, expected["history_digest"])

    def test_numbers(self):
        for item in json.loads((ROOT / "conformance" / "numbers.json").read_text()):
            value = struct.unpack("<d", struct.pack("<Q", int(item["bits"], 16)))[0]
            self.assertEqual(_number(value), item["json"])


if __name__ == "__main__":
    unittest.main()
