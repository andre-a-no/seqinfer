# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""What Python 3.8 and 3.9 lack, so that the library runs on them too.

zip_strict is zip(..., strict=True): it raises ValueError when the
iterables have different lengths.  On Python 3.10 and later it is the
built-in; before, a generator that checks the same thing.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import Any

if sys.version_info >= (3, 10):
    from functools import partial
    from itertools import pairwise  # novermin: only on Python 3.10+

    zip_strict = partial(zip, strict=True)
else:  # pragma: no cover - exercised on Python 3.8 and 3.9

    def zip_strict(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
        iterators = [iter(it) for it in iterables]
        if not iterators:
            return
        sentinel = object()
        while True:
            items = [next(it, sentinel) for it in iterators]
            done = [item is sentinel for item in items]
            if all(done):
                return
            if any(done):
                raise ValueError("zip_strict: the iterables have different lengths")
            yield tuple(items)

    def pairwise(iterable: Iterable[Any]) -> Iterator[tuple[Any, Any]]:
        iterator = iter(iterable)
        previous = next(iterator, None)
        first = True
        for item in iterator:
            if first:
                first = False
            yield previous, item
            previous = item

__all__ = ["pairwise", "zip_strict"]
