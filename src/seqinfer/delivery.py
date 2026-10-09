# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Delivery: per-source ordering and deduplication.

This is a runtime concern.  It decides which observations reach the
topology and in what per-source order; it never looks at values.

Policies
--------
arrival   deliver in arrival order; duplicates (same source and seq) are
          dropped when `dedup` is on
sequence  hold observations back until the per-source sequence is
          contiguous, then release them in sequence order
strict    reject gaps with OrderingError; late duplicates are dropped when
          `dedup` is on and rejected otherwise

`positions()` is the source checkpoint: for every source, the next
sequence number that has not yet been delivered.  Observations held back
by the `sequence` policy are deliberately not persisted; after a restore
they are fetched again from the source, starting at its position.
"""
from __future__ import annotations

from .core import Observation
from .errors import OrderingError

POLICIES = ("arrival", "sequence", "strict")


class Delivery:
    def __init__(self, policy: str = "arrival", *, dedup: bool = True, start: int = 0):
        if policy not in POLICIES:
            raise ValueError(f"unknown delivery policy {policy!r}")
        self.policy = policy
        self.dedup = dedup
        self.start = start
        self.duplicates = 0
        self._next: dict[str, int] = {}
        self._ahead: dict[str, set[int]] = {}
        self._pending: dict[str, dict[int, Observation]] = {}

    def spec(self) -> dict:
        return {"policy": self.policy, "dedup": self.dedup, "start": self.start}

    def delivered(self, obs: Observation) -> bool:
        """True if this observation was already delivered (a restored run must not receive it again)."""
        if obs.seq is None:
            return False
        if obs.seq < self._next.get(obs.source, self.start):
            return True
        return obs.seq in self._ahead.get(obs.source, ())

    def positions(self) -> dict[str, int]:
        return dict(self._next)

    def pending(self) -> int:
        return sum(len(p) for p in self._pending.values())

    def _duplicate(self, obs: Observation) -> list[Observation]:
        if not self.dedup:
            raise OrderingError(f"source {obs.source!r}: sequence number {obs.seq} was already delivered")
        self.duplicates += 1
        return []

    def push(self, obs: Observation) -> list[Observation]:
        """Accept one arriving observation; return those now deliverable, in order."""
        if obs.seq is None:
            if self.policy != "arrival":
                raise OrderingError(f"policy {self.policy!r} needs sequence numbers (source {obs.source!r})")
            return [obs]
        src, seq = obs.source, obs.seq
        nxt = self._next.get(src, self.start)

        if self.policy == "arrival":
            ahead = self._ahead.setdefault(src, set())
            if seq < nxt or seq in ahead:
                if self.dedup:
                    self.duplicates += 1
                    return []
                return [obs]
            if seq == nxt:
                nxt += 1
                while nxt in ahead:
                    ahead.discard(nxt)
                    nxt += 1
            else:
                ahead.add(seq)
            self._next[src] = nxt
            return [obs]

        if seq < nxt:
            return self._duplicate(obs)

        if self.policy == "strict":
            if seq != nxt:
                raise OrderingError(f"source {src!r}: expected sequence number {nxt}, got {seq}")
            self._next[src] = nxt + 1
            return [obs]

        pending = self._pending.setdefault(src, {})
        if seq in pending:
            return self._duplicate(obs)
        pending[seq] = obs
        out: list[Observation] = []
        while nxt in pending:
            out.append(pending.pop(nxt))
            nxt += 1
        self._next[src] = nxt
        return out

    def state(self) -> dict:
        return {
            "next": dict(self._next),
            "ahead": {s: sorted(a) for s, a in self._ahead.items() if a},
            "duplicates": self.duplicates,
        }

    def restore(self, state: dict) -> None:
        self._next = {s: int(n) for s, n in state["next"].items()}
        self._ahead = {s: set(a) for s, a in state.get("ahead", {}).items()}
        self._pending = {}
        self.duplicates = int(state.get("duplicates", 0))
