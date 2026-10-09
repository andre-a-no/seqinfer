"""Sources and adapters: everything that produces Observations.

A live instrument, a file of historical data, a replay log and a simulator
all look the same from the run's side.  A real adapter (DICOM, OPC UA,
MQTT, a database cursor) is a function or generator that yields
`Observation` values bound to a logical input.
"""
from __future__ import annotations

import asyncio
import heapq
import json
from itertools import zip_longest
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence

from .core import Observation
from .rng import SplitMix64


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


def skip_to(source: Iterable[Observation], positions: Mapping[str, int]) -> Iterator[Observation]:
    """Seek: drop what a restored run has already consumed, according to its source checkpoint."""
    for obs in source:
        if obs.seq is None or obs.seq >= positions.get(obs.source, 0):
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
    """

    def __init__(self, path=None):
        self.path = path
        self._items: list[Observation] = []

    def append(self, obs: Observation) -> None:
        if self.path is None:
            self._items.append(obs)
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obs.to_json()) + "\n")

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
