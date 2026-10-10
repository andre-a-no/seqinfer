# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Make the tests runnable from a source checkout without installing the package."""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Designs are computed afresh in tests; the design cache is tested with explicit directories.
import os  # noqa: E402

os.environ["SEQINFER_CACHE"] = "off"
