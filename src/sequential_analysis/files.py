# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Append-only files that stay open between writes.

Opening a file for every record costs a system call pair per record and
is the slow part of a long stream.  The file is opened on the first write
and kept open, like a logging handler.  What is configurable is
durability, the `sync` mode:

flush   every record is handed to the operating system: it survives a
        crash of the process, not a power failure (default)
fsync   every record is forced to the disk: it survives a power failure,
        at the cost of one disk write per record
"""
from __future__ import annotations

import os
from typing import IO, Any

SYNC_MODES = ("flush", "fsync")


class AppendFile:
    def __init__(self, path: str | os.PathLike[str], sync: str = "flush"):
        if sync not in SYNC_MODES:
            raise ValueError(f"sync must be one of {SYNC_MODES}, got {sync!r}")
        self.path = os.fspath(path)
        self.sync = sync
        self._file: IO[bytes] | None = None

    def write(self, data: bytes) -> None:
        if self._file is None:
            self._file = open(self.path, "ab")
        self._file.write(data)
        self._file.flush()
        if self.sync == "fsync":
            os.fsync(self._file.fileno())

    def fsync(self) -> None:
        """Force everything written so far to the disk."""
        if self._file is not None:
            os.fsync(self._file.fileno())
        elif os.path.exists(self.path):
            with open(self.path, "ab") as f:
                os.fsync(f.fileno())

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def closed(self) -> bool:
        return self._file is None

    def __enter__(self) -> AppendFile:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
