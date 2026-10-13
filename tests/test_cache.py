# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The design cache: plans are stored, reported, verified on loading, and never trusted blindly."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from sequential_analysis.cache import clear_design_cache, design_cache_dir
from sequential_analysis.design import kiefer_weiss_plan
from sequential_analysis.procedures import Gaussian, Poisson

ARGS = (0.3, 0.5, 0.05, 0.05, 80)


class DesignCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def design(self, *args, **kwargs):
        with self.assertLogs("sequential_analysis.cache", "WARNING") as logs:
            d = kiefer_weiss_plan(*(args or ARGS), cache=self.dir, **kwargs)
        return d, [r.incident for r in logs.records]

    def test_saved_then_loaded_and_identical(self):
        with self.assertLogs("sequential_analysis.cache", "WARNING") as logs:
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
        with self.assertLogs("sequential_analysis.cache", "WARNING") as logs:
            d = kiefer_weiss_plan(*ARGS, cache=self.dir)
        self.assertEqual([r.incident["action"] for r in logs.records], ["not_saved"])
        self.assertLessEqual(d.at_theta0.reject, 0.05)

    def test_off_and_directory_selection(self):
        with self.assertNoLogs("sequential_analysis.cache", "WARNING"):
            kiefer_weiss_plan(*ARGS, cache=False)
            kiefer_weiss_plan(*ARGS)  # the test suite runs with SEQUENTIAL_ANALYSIS_CACHE=off
        self.assertIsNone(design_cache_dir())
        self.assertEqual(design_cache_dir(self.dir), self.dir.resolve())
        saved = os.environ.get("SEQUENTIAL_ANALYSIS_CACHE")
        try:
            os.environ["SEQUENTIAL_ANALYSIS_CACHE"] = str(self.dir)
            self.assertEqual(design_cache_dir(), self.dir.resolve())
            del os.environ["SEQUENTIAL_ANALYSIS_CACHE"]
            os.environ["XDG_CACHE_HOME"] = self.tmp.name
            self.assertEqual(design_cache_dir(), Path(self.tmp.name).resolve() / "sequential-analysis")
        finally:
            os.environ.pop("XDG_CACHE_HOME", None)
            os.environ["SEQUENTIAL_ANALYSIS_CACHE"] = saved if saved is not None else "off"

    def test_invalid_arguments_are_not_cached(self):
        with self.assertRaises(ValueError), self.assertNoLogs("sequential_analysis.cache", "WARNING"):
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

    def test_a_family_given_by_name_is_a_clear_error(self):
        with self.assertRaisesRegex(TypeError, "family must be Bernoulli"):
            kiefer_weiss_plan(*ARGS, family="bernoulli", cache=self.dir)

    def test_a_plan_for_another_question_is_not_trusted(self):
        # a file with the right key but the design of another problem passed as "verified"
        self.design(0.2, 0.5, 0.05, 0.05, 60)
        self.design()
        files = {json.loads(f.read_text())["key"]["theta0"]: f for f in self.dir.glob("*.json")}
        mine = json.loads(files[0.3].read_text())
        mine["design"] = json.loads(files[0.2].read_text())["design"]
        files[0.3].write_text(json.dumps(mine))
        d, events = self.design()
        self.assertEqual([e["action"] for e in events], ["ignored", "saved"])
        self.assertEqual(d.at_theta0.theta, 0.3)
        self.assertLessEqual(d.at_theta0.reject, 0.05)

    def test_corrupt_files_never_fail_the_design(self):
        self.design()
        (path,) = self.dir.glob("*.json")
        good = json.loads(path.read_text())
        for seconds in ("1.0", None):
            path.write_text(json.dumps({**good, "seconds": seconds}))
            _, events = self.design()
            self.assertEqual([e["action"] for e in events], ["loaded"])
        path.write_text("[" * 100000 + "]" * 100000)
        _, events = self.design()
        self.assertEqual([e["action"] for e in events], ["ignored", "saved"])

    def test_clear_deletes_only_its_own_entries(self):
        self.design()
        (self.dir / "package.json").write_text("{}")
        (self.dir / "kiefer_weiss-x.json").mkdir()
        with self.assertLogs("sequential_analysis.cache", "WARNING"):
            self.assertEqual(clear_design_cache(self.dir), 1)
        self.assertTrue((self.dir / "package.json").exists())

    def test_arguments_are_checked_and_normalised_before_the_cache(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("numpy is not installed")
        self.design()
        with self.assertRaises(TypeError):  # a cache hit must not skip the checks
            kiefer_weiss_plan(0.3, 0.5, 0.05, 0.05, 80.0, cache=self.dir)
        with self.assertLogs("sequential_analysis.cache", "WARNING") as logs:
            kiefer_weiss_plan(numpy.float64(0.3), 0.5, 0.05, 0.05, numpy.int64(80), cache=self.dir)
        self.assertEqual([r.incident["action"] for r in logs.records], ["loaded"])
        self.assertIsNone(design_cache_dir("off"))
        with self.assertRaises(TypeError):
            design_cache_dir(0)
        with self.assertRaises(ValueError):
            design_cache_dir("")

    def test_clear(self):
        self.design()
        with self.assertLogs("sequential_analysis.cache", "WARNING"):
            self.assertEqual(clear_design_cache(self.dir), 1)
        self.assertEqual(list(self.dir.glob("*.json")), [])



class FourthReviewCache(DesignCache):
    """Line-by-line review after 0.1.0: the cache."""

    def entry(self):
        self.design()
        (path,) = self.dir.glob("*.json")
        return path, json.loads(path.read_text())

    def test_stored_multipliers_must_belong_to_the_plan(self):
        edits = [
            lambda d: d.__setitem__("lagrangian", -1.0),
            lambda d: d.__setitem__("lambda0", 1e9),
            lambda d: d["plan"].__setitem__("reject_high", "yes"),
            lambda d: d.__setitem__("exact", "no"),
            lambda d: d["plan"].__setitem__("label", 5),
        ]
        for edit in edits:
            path, entry = self.entry()
            edit(entry["design"])
            path.write_text(json.dumps(entry))
            _, events = self.design()
            with self.subTest(edit=edit):
                self.assertEqual([e["action"] for e in events], ["ignored", "saved"])

    def test_entries_are_readable_by_others(self):
        path, _ = self.entry()
        self.assertEqual(path.stat().st_mode & 0o777, 0o644)

    def test_clear_reports_files_it_cannot_remove(self):
        self.entry()
        original = Path.unlink

        def refuse(path, *args, **kwargs):
            raise PermissionError(1, "Operation not permitted")

        Path.unlink = refuse  # type: ignore[method-assign]
        try:
            with self.assertLogs("sequential_analysis.cache", "WARNING") as logs:
                self.assertEqual(clear_design_cache(self.dir), 0)
        finally:
            Path.unlink = original  # type: ignore[method-assign]
        self.assertEqual([r.incident["action"] for r in logs.records], ["cleared", "not_removed"])

    def test_a_path_named_off_is_a_directory(self):
        self.assertEqual(design_cache_dir(Path("off")), Path("off").resolve())
        self.assertIsNone(design_cache_dir("off"))

    def test_a_relative_xdg_cache_home_is_ignored(self):
        saved = {k: os.environ.get(k) for k in ("XDG_CACHE_HOME", "SEQUENTIAL_ANALYSIS_CACHE")}
        os.environ.pop("SEQUENTIAL_ANALYSIS_CACHE", None)
        os.environ["XDG_CACHE_HOME"] = "relative/dir"
        try:
            self.assertEqual(design_cache_dir(), Path("~/.cache/sequential-analysis").expanduser().resolve())
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class FourthReviewSteps(unittest.TestCase):
    def test_lattice_steps_are_checked(self):
        for step in (float("inf"), float("nan"), 0.0, -1.0, 1e-9, 1e300):
            with self.subTest(step=step), self.assertRaises(ValueError):
                kiefer_weiss_plan(0.0, 1.0, 0.05, 0.05, 20, family=Gaussian(), step=step, cache=False)

    def test_discrete_data_take_no_step(self):
        with self.assertRaises(ValueError):
            kiefer_weiss_plan(*ARGS, step=0.5, cache=False)

    def test_a_coarse_lattice_asks_for_a_finer_step(self):
        with self.assertRaises(ValueError) as caught:
            kiefer_weiss_plan(0.0, 1.0, 0.05, 0.05, 20, family=Gaussian(), step=3.0, cache=False)
        self.assertIn("finer lattice step", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
