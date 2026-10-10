# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Stateless transformations with an identity.

A transformation applied by a topology (`Topology.map`) or between the
stages of a `Chain` changes what the procedure sees, so it is part of the
experiment.  Like a procedure it is identified by name, version and
configuration; that identity is recorded in provenance and checked on
restore.  Change what a transformation computes, and you change its
version: a checkpoint made with the old one then refuses to restore.

A plain function can still be used together with a name, but only the
name is recorded; provenance marks such identities as "name only".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class Transform(ABC):
    """Map one statistical input (or one procedure output) to a statistical input, or None to drop it."""

    name: str = "transform"
    version: str = "1"

    @abstractmethod
    def config(self) -> dict:
        """JSON-serializable parameters that determine the transformation."""

    @abstractmethod
    def apply(self, x: Any) -> dict | None: ...

    def __call__(self, x: Any) -> dict | None:
        return self.apply(x)

    def identity(self) -> dict:
        return {"name": self.name, "version": self.version, "config": self.config()}


def identity_of(fn: Callable[[Any], Any], name: str | None) -> dict:
    """The recorded identity of a transformation given as a Transform or as a named function."""
    if isinstance(fn, Transform):
        identity = fn.identity()
        if name is not None:
            identity["label"] = name
        return identity
    if not name:
        raise ValueError("a plain function needs a name; better, make it a Transform with a version and config")
    return {"name": name, "identity": "name only"}


class Difference(Transform):
    """{a, b} -> {out: a - b}, typically after pairing two inputs."""

    name = "difference"
    version = "1"

    def __init__(self, a: str, b: str, out: str = "x"):
        self.a, self.b, self.out = a, b, out

    def config(self) -> dict:
        return {"a": self.a, "b": self.b, "out": self.out}

    def apply(self, x: Any) -> dict:
        return {self.out: x[self.a] - x[self.b]}


class Field(Transform):
    """Take one field of a procedure output (a dataclass or a dict) as the input `out`; None skips the step."""

    name = "field"
    version = "1"

    def __init__(self, field: str, out: str = "x"):
        self.field, self.out = field, out

    def config(self) -> dict:
        return {"field": self.field, "out": self.out}

    def apply(self, x: Any) -> dict | None:
        value = x[self.field] if isinstance(x, dict) else getattr(x, self.field)
        return None if value is None else {self.out: value}
