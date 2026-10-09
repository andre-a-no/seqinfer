"""Checkpoint storage.  Checkpoints are plain JSON values; this module only moves them to disk."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .errors import IncompatibleCheckpoint
from .run import CHECKPOINT_FORMAT


def save_checkpoint(path: str | os.PathLike, checkpoint: dict) -> None:
    """Write atomically: a crash leaves either the old file or the new one, never a torn one."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Make the rename durable.  POSIX only: Windows cannot open a directory."""
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_checkpoint(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        checkpoint = json.load(f)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise IncompatibleCheckpoint(f"{path}: unsupported checkpoint format {checkpoint.get('format')!r}")
    return checkpoint
