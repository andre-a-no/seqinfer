# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Kalman filter for the local-level model, optionally with a known control input.

    z_t = z_{t-1} + u_t + w_t,   w_t ~ N(0, q)
    y_t = z_t + v_t,             v_t ~ N(0, r)

With `control=True` the applied action u_t is a declared logical input.
This is how feedback enters the statistical boundary: an action that
changes the observation process is itself observed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure
from ..numerics import require_finite


@dataclass(frozen=True)
class KalmanState:
    n: int
    mean: float
    var: float


@dataclass(frozen=True)
class KalmanOutput:
    n: int
    mean: float
    var: float
    innovation: float
    innovation_var: float


class LocalLevelKalman(Procedure):
    name = "local_level_kalman"
    version = "1"

    def __init__(self, q: float, r: float, m0: float = 0.0, p0: float = 1.0, control: bool = False):
        if not (0 <= q < math.inf and 0 < r < math.inf and 0 <= p0 < math.inf and math.isfinite(m0)):
            raise ValueError("need q >= 0, r > 0 and p0 >= 0")
        self.q, self.r, self.m0, self.p0, self.control = float(q), float(r), float(m0), float(p0), control

    def config(self) -> dict:
        return {"q": self.q, "r": self.r, "m0": self.m0, "p0": self.p0, "control": self.control}

    def inputs(self):
        contracts = {"y": InputContract("y")}
        if self.control:
            contracts["u"] = InputContract("u")
        return contracts

    def initial_state(self) -> KalmanState:
        return KalmanState(0, self.m0, self.p0)

    def step(self, state: KalmanState, x, rng=None):
        m_pred = state.mean + (x["u"] if self.control else 0.0)
        p_pred = state.var + self.q
        innovation = x["y"] - m_pred
        s = p_pred + self.r
        gain = p_pred / s
        mean = require_finite(m_pred + gain * innovation, "filtered mean")
        var = (1.0 - gain) * p_pred
        n = state.n + 1
        return KalmanState(n, mean, var), KalmanOutput(n, mean, var, innovation, s)

    def decode_state(self, data) -> KalmanState:
        return KalmanState(**data)
