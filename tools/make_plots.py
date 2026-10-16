# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Make the pictures of docs/plotting.md by running the page's own code.

    python tools/make_plots.py            # needs matplotlib: pip install "sequential-analysis[plot]"

The page's python blocks run in order, in one namespace (as in
tests/test_docs.py).  Each figure a block creates is saved under the
name of the next image reference after the block, `![...](img/plots/NAME.png)`,
in order; a block must be followed by exactly as many image references
as it creates figures.  The pictures are therefore always what the code
on the page draws.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "plotting.md"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)  # the same pattern as tests/test_docs.py
IMAGE = re.compile(r"!\[[^\]]*\]\((img/plots/[^)\s]+\.png)\)")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.figsize": (7.0, 3.8), "font.size": 10, "axes.titlesize": 11})
    pieces = BLOCK.split(PAGE.read_text(encoding="utf-8"))  # text, code, text, code, ..., text
    namespace: dict = {"__name__": "__docs__"}
    saved = []
    for i in range(1, len(pieces), 2):
        code, after = pieces[i], pieces[i + 1]
        before = set(plt.get_fignums())
        exec(compile(code, str(PAGE), "exec"), namespace)
        figures = [n for n in plt.get_fignums() if n not in before]
        targets = IMAGE.findall(after)
        if len(figures) != len(targets):
            sys.exit(f"{PAGE.name}: block {i // 2 + 1} creates {len(figures)} figure(s) "
                     f"but is followed by {len(targets)} image reference(s)")
        for number, target in zip(figures, targets):
            path = PAGE.parent / target
            path.parent.mkdir(parents=True, exist_ok=True)
            figure = plt.figure(number)
            figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
            plt.close(figure)
            saved.append(path)
    for path in saved:
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
