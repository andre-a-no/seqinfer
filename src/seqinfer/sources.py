# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Sources and adapters: everything that produces Observations.

A live instrument, a file of historical data, a replay log and a simulator
all look the same from the run's side.  A real adapter (DICOM, OPC UA,
MQTT, a database cursor) is a function or generator that yields
`Observation` values bound to a logical input.
"""
from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import os
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from itertools import zip_longest
from typing import Any, cast

from .core import Observation
from .delivery import Delivery
from .errors import IncompatibleCheckpoint
from .files import AppendFile
from .rng import SplitMix64

_LOG_GENESIS = hashlib.sha256(b"seqinfer.log/2").hexdigest()


def from_values(
    input: str,
    values: Iterable[Any],
    *,
    source: str = "",
    start: int = 0,
    times: Sequence[float] | None = None,
    keys: Sequence[Any] | None = None,
) -> Iterator[Observation]:
    """Bind a sequence of values to a logical input, numbering them from `start`."""
    for i, v in enumerate(values):
        yield Observation(
            input,
            v,
            source,
            start + i,
            keys[i] if keys is not None else None,
            times[i] if times is not None else None,
        )


def from_csv(
    path: str | os.PathLike[str],
    input: str,
    value: str,
    *,
    convert: Callable[[str], Any] = float,
    source: str = "",
    seq: str | None = None,
    time: str | None = None,
    key: str | None = None,
    start: int = 0,
) -> Iterator[Observation]:
    """Read observations of one logical input from a CSV file with a header row.

    `value` names the column that holds the observation.  CSV holds text,
    so every value is converted explicitly with `convert` (float by
    default; int for counts); a value that does not convert raises
    ValueError naming the line, instead of reaching the procedure as a
    string.  `seq`, `time` and `key` name optional columns; without `seq`
    rows are numbered from `start` in file order.
    """
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for column in (value, seq, time, key):
            if column is not None and column not in header:
                raise ValueError(f"{path}: no column {column!r} (columns: {header})")
        for i, row in enumerate(reader):
            line = i + 2  # the header is line 1
            try:
                v = convert(row[value])
                n = int(row[seq]) if seq is not None else start + i
                t = float(row[time]) if time is not None else None
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}, line {line}: {error}") from error
            yield Observation(input, v, source, n, row[key] if key is not None else None, t)


def gaussian(
    input: str,
    mean: float | Callable[[int], float],
    sd: float,
    *,
    seed: int,
    n: int | None = None,
    source: str = "",
) -> Iterator[Observation]:
    """A simulator.  Uses simulation randomness, a stream separate from any procedure's R_t."""
    rng = SplitMix64.for_role(seed, f"simulation/{source or input}")
    i = 0
    while n is None or i < n:
        m = mean(i) if callable(mean) else mean
        yield Observation(input, rng.normal(m, sd), source, i, time=float(i))
        i += 1


def interleave(*sources: Iterable[Observation]) -> Iterator[Observation]:
    """Round-robin over pull sources until all are exhausted."""
    missing = object()
    for group in zip_longest(*sources, fillvalue=missing):
        for obs in group:
            if obs is not missing:
                yield cast(Observation, obs)


def merge_ordered(*sources: Iterable[Observation]) -> Iterator[Observation]:
    """Deterministic total order over time-ordered sources: (time, source, seq)."""
    return heapq.merge(*sources, key=lambda o: (o.time, o.source, o.seq))


def skip_to(source: Iterable[Observation], positions: Mapping[str, int] | Delivery) -> Iterator[Observation]:
    """Seek: drop what a restored run has already consumed, according to its source checkpoint.

    Pass the run's Delivery (``run.delivery``): sources it has not seen yet
    then start at its ``start`` number.  A bare mapping of positions treats
    unseen sources as starting at 0.
    """
    if isinstance(positions, Delivery):
        start, positions = positions.start, positions.positions()
    else:
        start = 0
    for obs in source:
        if obs.seq is None or obs.seq >= positions.get(obs.source, start):
            yield obs


async def as_async(
    source: Iterable[Observation], delay: float | Callable[[], float] = 0.0
) -> AsyncIterator[Observation]:
    """Turn a pull source into a push producer, optionally with (random) delays."""
    for obs in source:
        await asyncio.sleep(delay() if callable(delay) else delay)
        yield obs


class ObservationLog:
    """The delivered history of a run: every observation delivery passed on, in order.

    Replaying the log through the same delivery policy and topology
    reproduces the run.  The log holds what was delivered, not every
    arrival: duplicates that delivery dropped and observations it still
    holds back (the `sequence` policy) are not in it, so that the log
    position and the source positions in a checkpoint describe the same
    cut.  Invalid values are logged as they were, and a replay skips them
    as the run did.  With a path, lines are appended as JSON; without
    one, the log lives in memory.

    The log has a position -- number of records, byte offset and a hash
    chain over the records -- that a run stores in its checkpoint.
    Restoring the checkpoint restores the log to that position: the prefix
    is verified against the hash chain, and records written after the
    checkpoint are deleted.  Only their number is kept, in the provenance.
    Relative paths are resolved when the log is created.

    The file stays open between records; `sync` chooses between "flush"
    (survives a crash of the process) and "fsync" (survives a power
    failure), see `seqinfer.files`.  Whatever the mode, the log is forced
    to the disk whenever a checkpoint records its position.  Close the
    log, or use it as a context manager, when the run is over.

    seqinfer does not keep observations beyond the last checkpoint.  After a
    restore, the run receives them again from its sources (see `skip_to`).
    A source that cannot deliver past observations again -- a live
    instrument without a buffer, a stream without retention -- has to be
    buffered by the application, between the source and the run.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *, sync: str = "flush"):
        self.path = None if path is None else os.path.abspath(os.fspath(path))
        self._file = None if self.path is None else AppendFile(self.path, sync)
        self._items: list[Observation] = []
        self._position: dict | None = None  # computed lazily for an existing file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def __enter__(self) -> ObservationLog:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------- position

    @staticmethod
    def _line(obs: Observation) -> bytes:
        """One JSON line that reads back as exactly what was delivered.

        Floats keep their type and every bit (Python's shortest repr), large
        integers stay exact, and NaN or infinity are written as such: an
        invalid value has to come back invalid, so that a replay skips it
        exactly as the run did.  A value JSON cannot hold at all (a numpy
        scalar, an arbitrary object) is recorded by its repr, which the
        contracts reject just as they rejected the original.
        """
        record = obs.to_json()
        try:
            text = json.dumps(record, sort_keys=True)
        except (TypeError, ValueError):
            record["value"] = {"unserializable": repr(obs.value)}
            try:
                text = json.dumps(record, sort_keys=True)
            except (TypeError, ValueError):
                record = {k: v if isinstance(v, (str, int, float, type(None))) else repr(v) for k, v in record.items()}
                text = json.dumps(record, sort_keys=True)
        return (text + "\n").encode("utf-8")

    @staticmethod
    def _chain(digest: str, line: bytes) -> str:
        return hashlib.sha256(digest.encode("ascii") + line).hexdigest()

    def _lines(self) -> Iterator[bytes]:
        if self.path is None:
            for obs in self._items:
                yield self._line(obs)
        elif os.path.exists(self.path):
            with open(self.path, "rb") as f:
                yield from f

    def _scan(self) -> dict:
        if self._position is None:
            count, offset, digest = 0, 0, _LOG_GENESIS
            for line in self._lines():
                if not line.endswith(b"\n"):
                    raise IncompatibleCheckpoint(f"log {self.path}: last record is incomplete (torn write)")
                count, offset, digest = count + 1, offset + len(line), self._chain(digest, line)
            self._position = {"count": count, "offset": offset, "digest": digest}
        return self._position

    def is_empty(self) -> bool:
        if self.path is None:
            return not self._items
        return not os.path.exists(self.path) or os.path.getsize(self.path) == 0

    @property
    def sync(self) -> str | None:
        return None if self._file is None else self._file.sync

    def state(self) -> dict:
        """Durable position of the log, stored in the run checkpoint."""
        position = self._scan()
        if self._file is not None:
            self._file.fsync()
        return {"path": self.path, "sync": self.sync, **position}

    def verify(self, state: Mapping[str, Any]) -> int:
        """Check that the log starts with the checkpointed records; return how many follow them.

        Raises IncompatibleCheckpoint otherwise.  Changes nothing.
        """
        count, offset, digest = int(state["count"]), int(state["offset"]), state["digest"]
        where = self.path or "in-memory log"
        n, size, d, discarded = 0, 0, _LOG_GENESIS, 0
        for line in self._lines():
            if n < count:
                n, size, d = n + 1, size + len(line), self._chain(d, line)
            else:
                discarded += 1
        if n < count:
            raise IncompatibleCheckpoint(
                f"{where}: has {n} records, the checkpoint expects at least {count}; "
                f"was the log truncated, rotated or replaced?"
            )
        if size != offset or d != digest:
            raise IncompatibleCheckpoint(f"{where}: the first {count} records differ from the checkpointed ones")
        return discarded

    def restore(self, state: Mapping[str, Any]) -> int:
        """Return the log to a checkpointed position.

        Raises IncompatibleCheckpoint, and changes nothing, if the log does
        not start with the checkpointed records.  Otherwise deletes the
        records written after the checkpoint and returns how many there were.
        """
        discarded = self.verify(state)
        count, offset, digest = int(state["count"]), int(state["offset"]), state["digest"]
        if self.path is None:
            del self._items[count:]
        elif os.path.exists(self.path) and os.path.getsize(self.path) > offset:
            self.close()
            with open(self.path, "r+b") as f:
                f.truncate(offset)
                os.fsync(f.fileno())
        self._position = {"count": count, "offset": offset, "digest": digest}
        return discarded

    # ------------------------------------------------------------ records

    def append(self, obs: Observation) -> None:
        position = self._scan()
        line = self._line(obs)
        if self._file is None:
            self._items.append(obs)
        else:
            self._file.write(line)
        position["count"] += 1
        position["offset"] += len(line)
        position["digest"] = self._chain(position["digest"], line)

    def __iter__(self) -> Iterator[Observation]:
        if self.path is None:
            yield from list(self._items)
        else:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield Observation.from_json(json.loads(line))


def replay(log) -> Iterator[Observation]:
    """Replay a recorded delivered history (an ObservationLog or a path to one)."""
    return iter(log if isinstance(log, ObservationLog) else ObservationLog(log))
