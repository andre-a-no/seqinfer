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
from dataclasses import dataclass
from typing import Any, Callable


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
    """Appends one JSON line per output to a file."""

    def __init__(self, path, encode: Callable[[Any], Any]):
        self.path, self.encode = path, encode

    def __call__(self, event: OutputEvent) -> None:
        line = {"run_id": event.run_id, "t": event.t, "terminal": event.terminal, "output": self.encode(event.output)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
