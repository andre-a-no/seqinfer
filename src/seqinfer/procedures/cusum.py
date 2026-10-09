"""Page's CUSUM change-point detector."""
from __future__ import annotations

from dataclasses import dataclass

from ..core import Procedure
from ..numerics import require_finite


@dataclass(frozen=True)
class CusumState:
    n: int = 0
    w: float = 0.0
    alarm_at: int | None = None


@dataclass(frozen=True)
class CusumOutput:
    n: int
    statistic: float
    alarm: bool


class CUSUM(Procedure):
    """W_n = max(0, W_{n-1} + log f_theta1(X_n)/f_theta0(X_n)); alarm when W_n >= threshold."""

    name = "cusum"
    version = "1"

    def __init__(self, family, theta0: float, theta1: float, threshold: float, input: str = "x"):
        family.check(theta0)
        family.check(theta1)
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.family, self.theta0, self.theta1 = family, float(theta0), float(theta1)
        self.threshold, self.input = float(threshold), input

    def config(self) -> dict:
        return {
            "family": self.family.spec(),
            "theta0": self.theta0,
            "theta1": self.theta1,
            "threshold": self.threshold,
            "input": self.input,
        }

    def inputs(self):
        return {self.input: self.family.contract(self.input)}

    def initial_state(self) -> CusumState:
        return CusumState()

    def step(self, state: CusumState, x, rng=None):
        inc = self.family.llr(x[self.input], self.theta1, self.theta0)
        w = max(0.0, require_finite(state.w + inc, "CUSUM statistic"))
        n = state.n + 1
        alarm = w >= self.threshold
        return CusumState(n, w, n if alarm else None), CusumOutput(n, w, alarm)

    def is_terminal(self, state: CusumState) -> bool:
        return state.alarm_at is not None

    def decode_state(self, data) -> CusumState:
        return CusumState(**data)
