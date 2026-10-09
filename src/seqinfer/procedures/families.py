"""One-parameter likelihood families used by the likelihood-ratio procedures."""
from __future__ import annotations

import math

from ..contracts import InputContract


class Gaussian:
    """Normal observations with unknown mean and known standard deviation."""

    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)

    def spec(self) -> dict:
        return {"family": "gaussian", "sigma": self.sigma}

    def contract(self, name: str) -> InputContract:
        return InputContract(name, kind="real")

    def check(self, theta: float) -> None:
        if not math.isfinite(theta):
            raise ValueError("mean must be finite")

    def llr(self, x: float, num: float, den: float) -> float:
        """log f_num(x) - log f_den(x)."""
        return (num - den) / self.sigma**2 * (x - 0.5 * (num + den))

    def kl(self, a: float, b: float) -> float:
        """Kullback-Leibler divergence from theta=a to theta=b."""
        return (a - b) ** 2 / (2.0 * self.sigma**2)


class Bernoulli:
    """Binary observations with unknown success probability."""

    def spec(self) -> dict:
        return {"family": "bernoulli"}

    def contract(self, name: str) -> InputContract:
        return InputContract(name, kind="binary")

    def check(self, theta: float) -> None:
        if not 0.0 < theta < 1.0:
            raise ValueError("success probability must lie strictly between 0 and 1")

    def llr(self, x: float, num: float, den: float) -> float:
        return math.log(num / den) if x else math.log((1.0 - num) / (1.0 - den))

    def kl(self, a: float, b: float) -> float:
        return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))
