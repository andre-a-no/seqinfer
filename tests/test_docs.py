# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The Python code of the documentation runs: each page's blocks in order, in one namespace."""
import contextlib
import importlib.util
import io
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
OPTIONAL = {"sequential_analysis.plotting": "matplotlib"}  # a page using the module needs the optional dependency


def pages():
    return sorted((ROOT / "docs").rglob("*.md"))


@unittest.skipUnless((ROOT / "docs").is_dir() and (ROOT / "mkdocs.yml").exists(), "the documentation is not here")
class Documentation(unittest.TestCase):
    def test_every_python_block_runs(self):
        for page in pages():
            blocks = BLOCK.findall(page.read_text(encoding="utf-8"))
            if not blocks:
                continue
            with self.subTest(page=str(page.relative_to(ROOT))):
                needed = [dep for module, dep in OPTIONAL.items() if any(module in block for block in blocks)]
                missing = [dep for dep in needed if importlib.util.find_spec(dep) is None]
                if missing:
                    self.skipTest(f"needs {', '.join(missing)}")
                if "matplotlib" in needed:
                    import matplotlib

                    matplotlib.use("Agg")  # no windows
                namespace: dict = {"__name__": "__docs__"}
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    for block in blocks:
                        exec(compile(block, str(page), "exec"), namespace)
                if "matplotlib.pyplot" in sys.modules:
                    sys.modules["matplotlib.pyplot"].close("all")
                self.assertTrue(output.getvalue().strip(), "a tutorial should show its results")

    def test_navigation_lists_every_page(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        for page in pages():
            self.assertIn(str(page.relative_to(ROOT / "docs")), nav)


if __name__ == "__main__":
    unittest.main()
