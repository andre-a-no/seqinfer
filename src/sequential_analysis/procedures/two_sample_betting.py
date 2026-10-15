# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Anytime-valid nonparametric two-sample test by betting (Shekhar and Ramdas 2023).

Observations arrive in pairs (A_t, B_t).  H0: A_t and B_t have the same
distribution -- any distribution, possibly changing over time -- and are
independent of each other given the past.  The alternative is any
difference of the distributions: location, scale, shape, tails.

Before each pair, a witness function g is formed from the past pairs:
the kernel mean embedding of the A sample minus that of the B sample,

    g(z) = (1/(t-1)) sum_{s<t} [ k(z, A_s) - k(z, B_s) ],   k(z, u) = exp(-(z - u)^2 / (2 h^2)),

which is large where A has been more frequent than B (the witness of the
maximum mean discrepancy).  The gambler then stakes a fraction lambda_t on

    v_t = clip( (g(A_t) - g(B_t)) / (2 G_t), -1, 1 ),

G_t the largest |g| seen so far.  v_t is an odd function of the swap
A_t <-> B_t, so under H0 it has mean zero whatever the distribution, and
the wealth prod (1 + lambda_t v_t) is a non-negative martingale: reject
when it reaches 1/alpha (Ville's inequality).  lambda_t in [-1/2, 1/2] is
chosen by online Newton steps (Cutkosky and Orabona 2018), as Shekhar and
Ramdas do.

The kernel width h is the median distance within the first `burn_in`
pairs (the median heuristic), on which nothing is staked; or it is given.
The state holds the past pairs: memory grows with the run and each step
costs O(n) kernel evaluations.

For two streams that arrive separately, pair them with PositionalPair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import backend as _backend
from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..numerics import ordered_sum, require_finite
from .sprt import REJECT_H0

_ONS = 2.0 / (2.0 - math.log(3.0))
_MAX_BET = 0.5


@dataclass(frozen=True)
class KernelTwoSampleState:
    n: int = 0
    a: tuple[float, ...] = ()
    b: tuple[float, ...] = ()
    bandwidth: float | None = None
    scale: float = 0.0  # largest |g| seen at the observed points
    bet: float = 0.0  # lambda for the next pair
    curvature: float = 1.0  # online Newton: 1 + sum of squared gradients
    log_wealth: float = 0.0
    max_log_wealth: float = 0.0
    decision: str | None = None


@dataclass(frozen=True)
class KernelTwoSampleOutput:
    n: int
    payoff: float | None  # v_t, None while the bandwidth is being learnt
    log_wealth: float
    p_value: float
    decision: str | None


class KernelTwoSampleTest(Procedure):
    """Is the distribution of A the same as that of B?  Anytime-valid, for any distributions.

    Use it for A/B comparisons when the data are far from normal: revenue
    with many zeros and a long tail, latencies, counts.  It detects any
    difference of the distributions, not only of the means; to test the
    means alone with bounded data use BettingMeanTest on the differences.
    """

    name = "kernel_two_sample_test"
    version = "1"
    joint_inputs = True

    def __init__(
        self,
        alpha: float = 0.05,
        *,
        bandwidth: float | None = None,
        burn_in: int = 20,
        stop_on_reject: bool = True,
        backend: str = "auto",
        a: str = "a",
        b: str = "b",
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if bandwidth is not None and not (math.isfinite(bandwidth) and bandwidth > 0):
            raise ValueError("bandwidth must be positive and finite, or None for the median heuristic")
        if type(burn_in) is not int or burn_in < (0 if bandwidth is not None else 2):
            raise ValueError("burn_in must be an int (at least 2 when the bandwidth is learnt)")
        if a == b:
            raise ValueError("the two inputs need different names")
        self.alpha, self.burn_in = float(alpha), burn_in
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.stop_on_reject, self.a, self.b = stop_on_reject, a, b
        self._log_level = math.log(1.0 / self.alpha)
        self.backend = _backend.resolve(backend)

    def config(self) -> dict:
        return {
            "alpha": self.alpha,
            "bandwidth": self.bandwidth,
            "burn_in": self.burn_in,
            "backend": self.backend,
            "stop_on_reject": self.stop_on_reject,
            "a": self.a,
            "b": self.b,
        }

    def inputs(self):
        return {self.a: InputContract(self.a), self.b: InputContract(self.b)}

    def initial_state(self) -> KernelTwoSampleState:
        return KernelTwoSampleState(bandwidth=self.bandwidth)

    @staticmethod
    def witness(z: float, a: tuple[float, ...], b: tuple[float, ...], bandwidth: float) -> float:
        """g(z): mean kernel similarity of z to the A sample minus that to the B sample, in [-1, 1]."""
        c = -0.5 / (bandwidth * bandwidth)
        terms = (math.exp(c * (z - u) ** 2) - math.exp(c * (z - w) ** 2) for u, w in zip(a, b, strict=True))
        return ordered_sum(terms) / len(a)

    def _witnesses(self, za: float, zb: float, state: KernelTwoSampleState, bandwidth: float) -> tuple[float, float]:
        if self.backend == "python":
            return self.witness(za, state.a, state.b, bandwidth), self.witness(zb, state.a, state.b, bandwidth)
        np = _backend.np
        a, b, c = np.array(state.a), np.array(state.b), -0.5 / (bandwidth * bandwidth)
        z = np.array([[za], [zb]])
        g = (np.exp(c * (z - a) ** 2) - np.exp(c * (z - b) ** 2)).sum(axis=1) / len(state.a)
        return float(g[0]), float(g[1])

    @staticmethod
    def median_heuristic(points: list[float]) -> float:
        distances = sorted(abs(u - w) for i, u in enumerate(points) for w in points[i + 1:])
        positive = [d for d in distances if d > 0.0]
        return positive[len(positive) // 2] if positive else 1.0

    def step(self, state: KernelTwoSampleState, x, rng=None):
        va, vb = x[self.a], x[self.b]
        n = state.n + 1
        bandwidth, scale, bet, curvature = state.bandwidth, state.scale, state.bet, state.curvature
        log_wealth, payoff = state.log_wealth, None
        if bandwidth is not None and state.n >= max(1, self.burn_in):
            ga, gb = self._witnesses(va, vb, state, bandwidth)
            payoff = 0.0 if scale == 0.0 else max(-1.0, min(1.0, (ga - gb) / (2.0 * scale)))
            log_wealth = require_finite(log_wealth + math.log1p(bet * payoff), "log wealth")
            gradient = payoff / (1.0 + bet * payoff)
            curvature += gradient * gradient
            bet = max(-_MAX_BET, min(_MAX_BET, bet + _ONS * gradient / curvature))
            scale = max(scale, abs(ga), abs(gb))
        elif bandwidth is not None and state.n >= 1:
            ga, gb = self._witnesses(va, vb, state, bandwidth)
            scale = max(scale, abs(ga), abs(gb))
        a, b = (*state.a, float(va)), (*state.b, float(vb))
        if bandwidth is None and n == self.burn_in:
            bandwidth = self.median_heuristic([*a, *b])
        best = max(state.max_log_wealth, log_wealth)
        decision = state.decision or (REJECT_H0 if log_wealth >= self._log_level else None)
        new = KernelTwoSampleState(n, a, b, bandwidth, scale, bet, curvature, log_wealth, best, decision)
        return new, KernelTwoSampleOutput(n, payoff, log_wealth, min(1.0, math.exp(-best)), decision)

    def is_terminal(self, state: KernelTwoSampleState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def encode_state(self, state: KernelTwoSampleState):
        data = {k: getattr(state, k) for k in KernelTwoSampleState.__dataclass_fields__}
        data["a"], data["b"] = list(state.a), list(state.b)
        return data

    def decode_state(self, data) -> KernelTwoSampleState:
        state = decode_fields(KernelTwoSampleState, data)
        require_state(len(state.a) == len(state.b) == state.n, "samples differ from n")
        require_state(abs(state.bet) <= _MAX_BET and state.curvature >= 1.0 and state.scale >= 0.0, "bad bet")
        require_state(state.bandwidth is None or state.bandwidth > 0, "bandwidth must be positive")
        return state
