# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Runtimes: disposable machinery that moves observations into a run.

Nothing here is checkpointed.  Both runtimes end in the same place, a
serialized sequence of `run.offer` calls, which is why the procedure needs
no asynchronous interface and no internal locking.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterable, Iterable, Sequence

from .core import Observation
from .run import Run, RunStatus

_END = object()


class _Failure:
    def __init__(self, error: BaseException):
        self.error = error


def _attach(run: Run, mode: str) -> None:
    if run.status in (RunStatus.CREATED, RunStatus.PAUSED):
        run.start(runtime=mode)


def run_sync(run: Run, source: Iterable[Observation], *, stop_on_terminal: bool = True) -> int:
    """Pull observations from `source` and offer them in order.  Returns the number consumed."""
    _attach(run, "sync/pull")
    consumed = 0
    for obs in source:
        run.offer(obs)
        consumed += 1
        if stop_on_terminal and run.terminal:
            break
    return consumed


async def run_async(
    run: Run,
    producers: Sequence[AsyncIterable[Observation]],
    *,
    maxsize: int = 64,
    overflow: str = "block",
    stop_on_terminal: bool = True,
) -> int:
    """Concurrent acquisition, ordered delivery, serialized transitions.

    Each producer pushes into one bounded queue; a single consumer drains
    it into the run.  `overflow="block"` applies backpressure to producers.
    `overflow="drop"` discards observations when the queue is full; that
    changes the observation history, so drops are counted on the run and
    end up in its provenance.
    """
    if overflow not in ("block", "drop"):
        raise ValueError("overflow must be 'block' or 'drop'")
    if overflow == "drop" and run.delivery.policy != "arrival":
        raise ValueError(
            f"overflow='drop' cannot be combined with delivery policy {run.delivery.policy!r}: "
            f"a dropped sequence number leaves a gap that the policy waits on forever. "
            f"Use overflow='block', or Delivery('arrival')"
        )
    _attach(run, f"async/push/{overflow}")
    queue: asyncio.Queue = asyncio.Queue(maxsize)

    async def pump(producer: AsyncIterable[Observation]) -> None:
        async for obs in producer:
            if overflow == "block":
                await queue.put(obs)
            else:
                try:
                    queue.put_nowait(obs)
                except asyncio.QueueFull:
                    run.counters["dropped_by_runtime"] += 1

    async def feed() -> None:
        tasks = [asyncio.ensure_future(pump(p)) for p in producers]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise
        except Exception as error:
            for task in tasks:
                task.cancel()
            await queue.put(_Failure(error))
            return
        await queue.put(_END)

    feeder = asyncio.ensure_future(feed())
    consumed = 0
    try:
        while True:
            item = await queue.get()
            if item is _END:
                break
            if isinstance(item, _Failure):
                raise item.error
            run.offer(item)
            consumed += 1
            if stop_on_terminal and run.terminal:
                break
    finally:
        feeder.cancel()
        try:
            await feeder
        except (asyncio.CancelledError, Exception):
            pass
    return consumed
