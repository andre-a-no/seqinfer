# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Composition.  A chain of procedures is itself a procedure; nothing inherits from "Pipeline"."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import InputContract, validate_input
from .core import Procedure, StatInput
from .errors import ContractViolation
from .transforms import identity_of


@dataclass(frozen=True)
class ChainOutput:
    first: Any
    second: Any


class Chain(Procedure):
    """Feed the outputs of `first` into `second`.

    `link` maps an output of `first` to a statistical input of `second`, or
    returns None when there is nothing to pass on for this step.  The chain
    is terminal as soon as either stage is.

    If `link` produces an input that violates the contract of `second`,
    the step raises ContractViolation and neither stage advances: the run
    handles the whole input X_t by its `on_invalid` policy.
    """

    def __init__(
        self,
        first: Procedure,
        second: Procedure,
        link: Callable[[Any], StatInput | None],
        link_name: str | None = None,
    ):
        self.first, self.second, self.link = first, second, link
        self.link_identity = identity_of(link, link_name)
        self.link_name = link_name or self.link_identity["name"]
        self.name = f"chain({first.name},{second.name})"
        self.version = f"{first.version}+{second.version}"
        self.randomized = first.randomized or second.randomized
        self.joint_inputs = first.joint_inputs

    def config(self) -> dict:
        return {"first": self.first.identity(), "second": self.second.identity(), "link": self.link_identity}

    def inputs(self) -> Mapping[str, InputContract]:
        return self.first.inputs()

    def initial_state(self):
        return (self.first.initial_state(), self.second.initial_state())

    def step(self, state, x, rng):
        s1, s2 = state
        s1, o1 = self.first.step(s1, x, rng)
        o2 = None
        x2 = self.link(o1)
        if x2 is not None and not self.second.is_terminal(s2):
            try:
                validate_input(self.second, x2)
            except ContractViolation as violation:
                raise ContractViolation(
                    f"{self.second.name} (second stage, via {self.link_name!r}): {violation}"
                ) from violation
            s2, o2 = self.second.step(s2, x2, rng)
        return (s1, s2), ChainOutput(o1, o2)

    def is_terminal(self, state) -> bool:
        return self.first.is_terminal(state[0]) or self.second.is_terminal(state[1])

    def encode_state(self, state):
        return {"first": self.first.encode_state(state[0]), "second": self.second.encode_state(state[1])}

    def decode_state(self, data):
        return (self.first.decode_state(data["first"]), self.second.decode_state(data["second"]))

    def encode_output(self, output):
        return {
            "first": self.first.encode_output(output.first),
            "second": None if output.second is None else self.second.encode_output(output.second),
        }
