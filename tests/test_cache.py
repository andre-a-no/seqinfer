# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The design cache: plans are stored, reported, verified on loading, and never trusted blindly."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from seqinfer.cache import clear_design_cache, design_cache_dir
from seqinfer.design import kiefer_weiss_plan
from seqinfer.procedures import Gaussian, Poisson

ARGS = (0.3, 0.5, 0.05, 0.05, 80)


class DesignCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def design(self, *args, **kwargs):
        with self.assertLogs("seqinfer.cache", "WARNING") as logs:
            d = kiefer_weiss_plan(*(args or ARGS), cache=self.dir, **kwargs)
        return d, [r.incident for r in logs.records]

    def test_saved_then_loaded_and_identical(self):
        with self.assertLogs("seqinfer.cache", "WARNING") as logs:
            first = kiefer_weiss_plan(*ARGS, cache=self.dir)
        events = [r.incident for r in logs.records]
        self.assertEqual([e["action"] for e in events], ["saved"])
        self.assertIn("saved it to", logs.output[0])
        files = list(self.dir.glob("kiefer_weiss-*.json"))
        self.assertEqual(len(files), 1)
        self.assertEqual(events[0]["path"], str(files[0]))
        again, events = self.design()
        self.assertEqual([e["action"] for e in events], ["loaded"])
        self.assertEqual(again.plan.config(), first.plan.config())
        for name in ("at_theta0", "at_theta1", "at_theta_star"):
            self.assertEqual(getattr(again, name), getattr(first, name))
        self.assertEqual((again.theta_star, again.lagrangian), (first.theta_star, first.lagrangian))
        self.assertEqual(again.maximum_expected_n(), first.maximum_expected_n())

    def test_every_argument_is_part_of_the_question(self):
        self.design()
        _, events = self.design(0.3, 0.5, 0.05, 0.04, 80)
        self.assertEqual([e["action"] for e in events], ["saved"])
        _, events = self.design(*ARGS, theta_star=0.4)
        self.assertEqual([e["action"] for e in events], ["saved"])
        _, events = self.design(2.0, 3.0, 0.05, 0.05, 30, family=Poisson())
        self.assertEqual([e["action"] for e in events], ["saved"])
        self.assertEqual(len(list(self.dir.glob("*.json"))), 4)

    def test_a_tampered_plan_is_not_trusted(self):
        self.design()
        (path,) = self.dir.glob("*.json")
        entry = json.loads(path.read_text())
        steps = entry["design"]["plan"]["continuation"]
        entry["design"]["plan"]["continuation"] = [*steps[:10], [5.0, 5.0]]  # stops after 11 observations
        path.write_text(json.dumps(entry))
        _, events = self.design()
        self.assertEqual([e["action"] for e in events], ["ignored", "saved"])

    def test_damaged_and_foreign_files_are_ignored(self):
        self.design()
        (path,) = self.dir.glob("*.json")
        for content in ("{not json", json.dumps({"format": "other"}), ""):
            path.write_text(content)
            _, events = self.design()
            self.assertEqual([e["action"] for e in events], ["ignored", "saved"])
        entry = json.loads(path.read_text())
        entry["key"]["alpha0"] = 0.01  # a file answering another question under this name
        path.write_text(json.dumps(entry))
        _, events = self.design()
        self.assertEqual([e["action"] for e in events], ["ignored", "saved"])

    def test_a_cache_that_cannot_be_written_does_not_fail_the_design(self):
        self.dir.write_text("a file, not a directory")
        with self.assertLogs("seqinfer.cache", "WARNING") as logs:
            d = kiefer_weiss_plan(*ARGS, cache=self.dir)
        self.assertEqual([r.incident["action"] for r in logs.records], ["not_saved"])
        self.assertLessEqual(d.at_theta0.reject, 0.05)

    def test_off_and_directory_selection(self):
        with self.assertNoLogs("seqinfer.cache", "WARNING"):
            kiefer_weiss_plan(*ARGS, cache=False)
            kiefer_weiss_plan(*ARGS)  # the test suite runs with SEQINFER_CACHE=off
        self.assertIsNone(design_cache_dir())
        self.assertEqual(design_cache_dir(self.dir), self.dir.resolve())
        saved = os.environ.get("SEQINFER_CACHE")
        try:
            os.environ["SEQINFER_CACHE"] = str(self.dir)
            self.assertEqual(design_cache_dir(), self.dir.resolve())
            del os.environ["SEQINFER_CACHE"]
            os.environ["XDG_CACHE_HOME"] = self.tmp.name
            self.assertEqual(design_cache_dir(), Path(self.tmp.name).resolve() / "seqinfer")
        finally:
            os.environ.pop("XDG_CACHE_HOME", None)
            os.environ["SEQINFER_CACHE"] = saved if saved is not None else "off"

    def test_invalid_arguments_are_not_cached(self):
        with self.assertRaises(ValueError), self.assertNoLogs("seqinfer.cache", "WARNING"):
            kiefer_weiss_plan(0.3, 0.3, 0.05, 0.05, 80, cache=self.dir)
        with self.assertRaises(ValueError):
            kiefer_weiss_plan(float("nan"), 0.5, 0.05, 0.05, 80, cache=self.dir)
        self.assertFalse(list(self.dir.glob("*.json")) if self.dir.exists() else [])

    def test_lattice_designs_round_trip(self):
        first, _ = self.design(0.0, 1.0, 0.05, 0.1, 25, family=Gaussian(1.0), step=0.25)
        again, events = self.design(0.0, 1.0, 0.05, 0.1, 25, family=Gaussian(1.0), step=0.25)
        self.assertEqual([e["action"] for e in events], ["loaded"])
        self.assertEqual(again.at_theta_star, first.at_theta_star)
        self.assertFalse(again.exact)
        self.assertEqual(again.step, 0.25)

    def test_clear(self):
        self.design()
        with self.assertLogs("seqinfer.cache", "WARNING"):
            self.assertEqual(clear_design_cache(self.dir), 1)
        self.assertEqual(list(self.dir.glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
