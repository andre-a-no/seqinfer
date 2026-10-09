"""Statistical random state.

The random state R_t of a randomized procedure is part of its mathematical
trajectory, so it has to be small, explicit and portable.  SplitMix64 keeps
the whole state in one 64-bit integer, which makes it trivial to checkpoint
and to re-implement bit-for-bit in another language.

Box-Muller is used without caching the second variate, so the generator
state after a draw is always just that one integer.
"""
from __future__ import annotations

import hashlib
import math

_MASK = (1 << 64) - 1


class SplitMix64:
    __slots__ = ("state",)

    def __init__(self, state: int):
        self.state = int(state) & _MASK

    @classmethod
    def for_role(cls, seed: int, role: str) -> "SplitMix64":
        """Derive an independent stream for a named role.

        Roles keep statistical, simulation and execution randomness apart:
        two generators derived from the same seed with different roles never
        share a stream, so attaching a simulator cannot perturb R_t.
        """
        digest = hashlib.sha256(f"{int(seed)}/{role}".encode()).digest()
        return cls(int.from_bytes(digest[:8], "big"))

    def copy(self) -> "SplitMix64":
        return SplitMix64(self.state)

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Uniform on [0, 1) with 53 bits of precision."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def normal(self, mean: float = 0.0, sd: float = 1.0) -> float:
        u1 = 1.0 - self.random()  # (0, 1], keeps log finite
        u2 = self.random()
        return mean + sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
