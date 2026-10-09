"""Execute a precomputed sequential sampling plan.

Optimal sequential tests, such as the numerical solutions of the
Kiefer-Weiss problem by Novikov, Novikov and Farkhshatov, are computed
offline by backward induction.  Their result is a *plan*: for every step
n = 1, ..., H an interval of the running sum S_n in which sampling
continues.  Computing the plan is a design problem.  Executing it on a
stream is a Sequential Procedure, and that is all this class does.

Plan format
-----------
``continuation[n-1] = (lower_n, upper_n)`` for n = 1, ..., H.

    S_n <= lower_n   stop and accept H0
    S_n >= upper_n   stop and reject H0
    otherwise        continue

The acceptance side is checked first.  ``None`` means "no boundary on this
side at this step".  The last step must force a decision, so it needs
``lower_H >= upper_H``; the usual choice ``(c, c)`` reads "accept H0 if
S_H <= c, otherwise reject".  With ``reject_high=False`` the two decisions
are swapped, for problems in which large sums favour H0.

Boundary conventions (open or closed ends) differ between design codes.
Check them against the code that produced the plan; for integer-valued
data, shift a boundary by one where necessary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..contracts import InputContract
from ..core import Procedure
from ..numerics import neumaier_add, require_finite
from .sprt import ACCEPT_H0, REJECT_H0


@dataclass(frozen=True)
class PlanState:
    n: int = 0
    total: float = 0.0
    comp: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class PlanOutput:
    n: int
    statistic: float
    decision: str | None


class PlanTest(Procedure):
    name = "plan_test"
    version = "1"

    def __init__(
        self,
        continuation: Sequence[Sequence[float | None]],
        *,
        contract: InputContract | None = None,
        reject_high: bool = True,
        label: str = "",
        input: str = "x",
    ):
        plan = []
        for n, stage in enumerate(continuation, start=1):
            if len(stage) != 2:
                raise ValueError(f"step {n}: expected (lower, upper)")
            lower, upper = (None if b is None else float(b) for b in stage)
            for b in (lower, upper):
                if b is not None and not math.isfinite(b):
                    raise ValueError(f"step {n}: use None for a missing boundary, not {b!r}")
            plan.append((lower, upper))
        if not plan:
            raise ValueError("a plan needs at least one step")
        lower, upper = plan[-1]
        if lower is None or upper is None or lower < upper:
            raise ValueError("the last step must force a decision: it needs lower >= upper")
        self.plan = tuple(plan)
        self.input, self.label, self.reject_high = input, label, reject_high
        self.contract = contract if contract is not None else InputContract(input)

    @property
    def horizon(self) -> int:
        """The largest number of observations the plan can take."""
        return len(self.plan)

    def config(self) -> dict:
        return {
            "continuation": [list(stage) for stage in self.plan],
            "reject_high": self.reject_high,
            "label": self.label,
            "input": self.input,
            "contract": self.contract.spec(),
        }

    def inputs(self):
        return {self.input: self.contract}

    def initial_state(self) -> PlanState:
        return PlanState()

    def step(self, state: PlanState, x, rng=None):
        total, comp = neumaier_add(state.total, state.comp, x[self.input])
        s = require_finite(total + comp, "running sum")
        n = state.n + 1
        lower, upper = self.plan[n - 1]
        low, high = (ACCEPT_H0, REJECT_H0) if self.reject_high else (REJECT_H0, ACCEPT_H0)
        decision = None
        if lower is not None and s <= lower:
            decision = low
        elif upper is not None and s >= upper:
            decision = high
        return PlanState(n, total, comp, decision), PlanOutput(n, s, decision)

    def is_terminal(self, state: PlanState) -> bool:
        return state.decision is not None

    def decode_state(self, data) -> PlanState:
        return PlanState(**data)
