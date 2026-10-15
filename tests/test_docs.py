# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The Python code of the documentation runs: each page's blocks in order, in one namespace."""
import contextlib
import io
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


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
                namespace: dict = {"__name__": "__docs__"}
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    for block in blocks:
                        exec(compile(block, str(page), "exec"), namespace)
                self.assertTrue(output.getvalue().strip(), "a tutorial should show its results")

    def test_navigation_lists_every_page(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        for page in pages():
            self.assertIn(str(page.relative_to(ROOT / "docs")), nav)


if __name__ == "__main__":
    unittest.main()
