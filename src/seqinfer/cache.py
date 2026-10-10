# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Design cache: optimal plans are computed once and kept on disk.

Computing a Kiefer-Weiss plan takes from a second to hours; executing it
takes microseconds per observation.  `kiefer_weiss_plan` therefore stores
every plan it computes and returns the stored one when it is asked the
same question again.

Where
    `cache=` names a directory; otherwise the environment variable
    SEQINFER_CACHE does; otherwise $XDG_CACHE_HOME/seqinfer, or
    ~/.cache/seqinfer.  ``cache=False``, or SEQINFER_CACHE set to "off",
    turns the cache off.  Deleting the directory, or `clear_design_cache()`,
    empties it.

What
    One JSON file per question.  Its name is a hash of the question: every
    argument, the family, the library version and the version of the
    design algorithm, so a new version of either computes afresh.  The file
    repeats the question in full, with the time the computation took.

Trust
    A stored plan is not taken on faith.  On loading, its operating
    characteristic is recomputed (a forward pass, cheap next to the design)
    and must reproduce the stored one and meet the requested error rates
    within the horizon; otherwise the file is ignored and the plan computed
    again.  An edited, truncated or foreign file can cost time, never the
    guarantee.

Messages
    Every save, load, ignored file and failed write is reported on the
    logger "seqinfer.cache" at WARNING level, so it is visible by default,
    with a structured ``incident`` record like the run's incidents.
    ``logging.getLogger("seqinfer.cache").setLevel(logging.ERROR)`` silences
    it.  A cache that cannot be written never fails the design.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .canonical import canonical_json

logger = logging.getLogger("seqinfer.cache")

FORMAT = "seqinfer.design-cache/1"
_OFF = ("off", "0", "false", "no", "none")


def design_cache_dir(cache: Any = None) -> Path | None:
    """The cache directory selected by `cache` (see the module docstring), or None if the cache is off."""
    if cache is False:
        return None
    if cache is not None and cache is not True:
        return Path(os.fspath(cache)).expanduser().resolve()
    env = os.environ.get("SEQINFER_CACHE")
    if env is not None:
        if env.strip().lower() in _OFF or not env.strip():
            return None
        return Path(env).expanduser().resolve()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base).resolve() / "seqinfer"


def clear_design_cache(cache: Any = None) -> int:
    """Delete the stored designs; returns how many files were removed."""
    directory = design_cache_dir(cache)
    if directory is None or not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob("*.json"):
        path.unlink()
        removed += 1
    report("cleared", f"design cache: removed {removed} stored design(s) from {directory}", directory)
    return removed


def entry_path(directory: Path, kind: str, key: dict) -> Path:
    digest = hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()[:32]
    return directory / f"{kind}-{digest}.json"


def report(action: str, message: str, path: Path, key: dict | None = None, **extra: Any) -> None:
    logger.warning(
        message,
        extra={"incident": {"kind": "design_cache", "action": action, "path": str(path), "key": key, **extra}},
    )


def read_entry(path: Path, key: dict) -> dict | None:
    """The stored record for `key`, or None (missing, unreadable or for another question: reported)."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
        if not isinstance(entry, dict) or entry.get("format") != FORMAT:
            raise ValueError("not a design cache entry of this format")
        if entry.get("key") != json.loads(json.dumps(key)):
            raise ValueError("it answers a different question")
        if not isinstance(entry.get("design"), dict):
            raise ValueError("it holds no design")
    except (OSError, ValueError) as error:  # json.JSONDecodeError is a ValueError
        report("ignored", f"design cache: ignored {path} ({error}); computing the design again", path, key)
        return None
    return entry


def write_entry(path: Path, key: dict, design: dict, seconds: float, created_at: float) -> bool:
    """Store atomically; a failure is reported and swallowed (the caller has its design anyway)."""
    entry = {"format": FORMAT, "key": key, "seconds": seconds, "created_at": created_at, "design": design}
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entry, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as error:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
        report("not_saved", f"design cache: could not save to {path} ({error}); the design is not cached", path, key)
        return False
    return True
