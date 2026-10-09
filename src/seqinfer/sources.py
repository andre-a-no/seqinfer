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
import shutil
import time
from itertools import zip_longest
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence

from .core import Observation
from .delivery import Delivery
from .errors import IncompatibleCheckpoint
from .rng import SplitMix64

_LOG_GENESIS = hashlib.sha256(b"seqinfer.log/1").hexdigest()


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
                yield obs


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
    """The arrival history of a run: every observation, in the order the run saw it.

    Replaying the log through the same delivery policy and topology
    reproduces the run.  With a path, lines are appended as JSON; without
    one, the log lives in memory.

    The log has a position -- number of records, byte offset and a hash
    chain over the records -- that a run stores in its checkpoint.
    Restoring the checkpoint restores the log to that position: the prefix
    is verified against the hash chain, and records written after the
    checkpoint are moved to a sidecar file, because the restored run will
    receive those observations again.  Relative paths are resolved against
    the working directory at the time of use.
    """

    def __init__(self, path=None):
        self.path = None if path is None else os.fspath(path)
        self._items: list[Observation] = []
        self._position: dict | None = None  # computed lazily for an existing file

    # ----------------------------------------------------------- position

    @staticmethod
    def _line(obs: Observation) -> bytes:
        return (json.dumps(obs.to_json(), sort_keys=True) + "\n").encode("utf-8")

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

    def state(self) -> dict:
        """Durable position of the log, stored in the run checkpoint."""
        position = self._scan()
        if self.path is not None and os.path.exists(self.path):
            with open(self.path, "ab") as f:
                os.fsync(f.fileno())
        return {"path": self.path, **position}

    def restore(self, state: Mapping[str, Any]) -> str | None:
        """Return the log to a checkpointed position.

        Raises IncompatibleCheckpoint if the log does not start with the
        checkpointed records.  Returns the path of the sidecar file that
        received the records written after the checkpoint, if there were any.
        """
        count, offset, digest = int(state["count"]), int(state["offset"]), state["digest"]
        where = self.path or "in-memory log"
        n, size, d = 0, 0, _LOG_GENESIS
        lines = self._lines()
        for line in lines:
            if n == count:
                break
            n, size, d = n + 1, size + len(line), self._chain(d, line)
        if n < count:
            raise IncompatibleCheckpoint(
                f"{where}: has {n} records, the checkpoint expects at least {count}; "
                f"was the log truncated, rotated or replaced?"
            )
        if size != offset or d != digest:
            raise IncompatibleCheckpoint(f"{where}: the first {count} records differ from the checkpointed ones")
        lines.close()

        orphaned = None
        if self.path is None:
            del self._items[count:]
        elif os.path.exists(self.path) and os.path.getsize(self.path) > offset:
            orphaned = f"{self.path}.after-{count}.{int(time.time())}.jsonl"
            with open(self.path, "rb") as src, open(orphaned, "wb") as dst:
                src.seek(offset)
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
            with open(self.path, "r+b") as f:
                f.truncate(offset)
                os.fsync(f.fileno())
        self._position = {"count": count, "offset": offset, "digest": digest}
        return orphaned

    # ------------------------------------------------------------ records

    def append(self, obs: Observation) -> None:
        position = self._scan()
        line = self._line(obs)
        if self.path is None:
            self._items.append(obs)
        else:
            with open(self.path, "ab") as f:
                f.write(line)
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
    """Replay a recorded arrival history (an ObservationLog or a path to one)."""
    return iter(log if isinstance(log, ObservationLog) else ObservationLog(log))
