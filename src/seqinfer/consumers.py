# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Scientific consumers: they see outputs, never state.

A consumer is any callable taking an OutputEvent.  Exceptions raised by a
consumer are caught by the run and recorded; they never affect inference.
A consumer that wants to influence the experiment acts on a source or an
actuator, not on the run.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .files import AppendFile


@dataclass(frozen=True)
class OutputEvent:
    run_id: str
    t: int
    output: Any
    terminal: bool


Consumer = Callable[[OutputEvent], None]


class Recorder:
    """Keeps every event in memory."""

    def __init__(self):
        self.events: list[OutputEvent] = []

    def __call__(self, event: OutputEvent) -> None:
        self.events.append(event)

    @property
    def outputs(self) -> list[Any]:
        return [e.output for e in self.events]


class JsonlSink:
    """Appends one JSON line per output to a file.

    The file stays open between events; `sync` is "flush" or "fsync", see
    `seqinfer.files`.  Close the sink, or use it as a context manager, when
    the run is over.
    """

    def __init__(self, path: str | os.PathLike[str], encode: Callable[[Any], Any], *, sync: str = "flush"):
        self.encode = encode
        self._file = AppendFile(path, sync)
        self.path = self._file.path

    def __call__(self, event: OutputEvent) -> None:
        line = {"run_id": event.run_id, "t": event.t, "terminal": event.terminal, "output": self.encode(event.output)}
        self._file.write((json.dumps(line) + "\n").encode("utf-8"))

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
