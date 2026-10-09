"""Exponential moving average: a stateful transformation, so also a procedure."""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure


@dataclass(frozen=True)
class EMAState:
    n: int = 0
    value: float = 0.0


@dataclass(frozen=True)
class EMAOutput:
    n: int
    value: float
    residual: float  # observation minus the average before this step


class EMA(Procedure):
    name = "ema"
    version = "1"

    def __init__(self, alpha: float, input: str = "x"):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        self.alpha, self.input = float(alpha), input

    def config(self) -> dict:
        return {"alpha": self.alpha, "input": self.input}

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def initial_state(self) -> EMAState:
        return EMAState()

    def step(self, state: EMAState, x, rng=None):
        v = x[self.input]
        if state.n == 0:
            return EMAState(1, v), EMAOutput(1, v, 0.0)
        value = state.value + self.alpha * (v - state.value)
        return EMAState(state.n + 1, value), EMAOutput(state.n + 1, value, v - state.value)

    def decode_state(self, data) -> EMAState:
        return EMAState(**data)
