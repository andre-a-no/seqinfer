# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Bootstrap particle filter for the local-level model: a randomized procedure.

All randomness is drawn from the statistical random state R_t that the run
passes into `step`.  The initial particle cloud is drawn in the first step,
so `initial_state` stays deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure
from ..numerics import logsumexp, ordered_sum, require_finite


@dataclass(frozen=True)
class ParticleState:
    n: int = 0
    particles: tuple[float, ...] = ()


@dataclass(frozen=True)
class ParticleOutput:
    n: int
    mean: float
    var: float
    ess: float


class BootstrapParticleFilter(Procedure):
    name = "bootstrap_particle_filter"
    version = "1"
    randomized = True

    def __init__(self, q: float, r: float, m0: float = 0.0, p0: float = 1.0, particles: int = 256):
        if not (0 <= q < math.inf and 0 < r < math.inf and 0 <= p0 < math.inf and math.isfinite(m0)) or particles < 2:
            raise ValueError("need q >= 0, r > 0, p0 >= 0 and at least two particles")
        self.q, self.r, self.m0, self.p0, self.size = float(q), float(r), float(m0), float(p0), int(particles)

    def config(self) -> dict:
        return {"q": self.q, "r": self.r, "m0": self.m0, "p0": self.p0, "particles": self.size}

    def inputs(self):
        return {"y": InputContract("y")}

    def initial_state(self) -> ParticleState:
        return ParticleState()

    def step(self, state: ParticleState, x, rng):
        y = x["y"]
        size = self.size
        prior = state.particles or tuple(rng.normal(self.m0, math.sqrt(self.p0)) for _ in range(size))
        sd = math.sqrt(self.q)
        cloud = [p + rng.normal(0.0, sd) for p in prior]
        logw = [-0.5 * (y - p) ** 2 / self.r for p in cloud]
        norm = require_finite(logsumexp(logw), "log normalizing constant")
        w = [math.exp(lw - norm) for lw in logw]
        mean = ordered_sum(wi * p for wi, p in zip(w, cloud, strict=True))
        var = ordered_sum(wi * (p - mean) ** 2 for wi, p in zip(w, cloud, strict=True))
        ess = 1.0 / ordered_sum(wi * wi for wi in w)
        # systematic resampling: one uniform draw
        u = rng.random() / size
        resampled, i, cum = [], 0, w[0]
        for j in range(size):
            target = u + j / size
            while cum < target and i < size - 1:
                i += 1
                cum += w[i]
            resampled.append(cloud[i])
        n = state.n + 1
        return ParticleState(n, tuple(resampled)), ParticleOutput(n, mean, var, ess)

    def encode_state(self, state: ParticleState):
        return {"n": state.n, "particles": list(state.particles)}

    def decode_state(self, data) -> ParticleState:
        return ParticleState(data["n"], tuple(data["particles"]))
