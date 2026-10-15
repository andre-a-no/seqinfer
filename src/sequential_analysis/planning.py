# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Planning a sequential experiment: how long will it run, how often will it decide?

A sequential test has no fixed sample size; what is planned is its
distribution.  `simulate` runs any procedure on simulated data many times
and reports how often it stopped within a budget (the power, or under
no effect the false positive rate) and the distribution of the number of
units it needed (median, 90th percentile, mean).  Everything else in the
library -- exact operating characteristics of plans (design), group
sequential boundaries -- is computed without simulation; this module
covers what has no closed form: the anytime-valid and nonparametric
tests, change detectors and A/B tests with their real data shapes.

    from sequential_analysis.planning import simulate, conversion_units
    from sequential_analysis.procedures import TwoProportionTest

    plan = simulate(TwoProportionTest, conversion_units(0.10, 0.12), runs=200, max_n=50_000)
    plan.power, plan.median_n, plan.quantile(0.9)

The simulation draws from Python's random module seeded per run, so a
plan is reproducible; Monte Carlo error is reported (`power_interval`, a
Wilson interval).

Cost: runs x (units per run) x (cost of a step); a two-proportion plan of
200 runs up to 50,000 users takes seconds.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .core import Procedure


def _stopped(output: Any) -> bool:
    """Default stopping rule: a decision (tests) or an alarm (change detectors)."""
    return getattr(output, "decision", None) is not None or bool(getattr(output, "alarm", False))


@dataclass(frozen=True)
class SimulationSummary:
    runs: int
    max_n: int
    stopped: int  # runs that stopped within max_n
    stopping_times: tuple[int, ...]  # sorted, of the runs that stopped
    alpha_level: float  # the confidence level of power_interval

    @property
    def power(self) -> float:
        """Fraction of runs that stopped within max_n: the power, or the false positive rate under no effect."""
        return self.stopped / self.runs

    @property
    def power_interval(self) -> tuple[float, float]:
        """Wilson score interval for `power` (Monte Carlo error)."""
        z = _normal_quantile(1 - self.alpha_level / 2)
        n, p = self.runs, self.power
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, centre - half), min(1.0, centre + half)

    def quantile(self, q: float) -> float | None:
        """The q-quantile of the number of units over ALL runs, runs that did not stop counting as max_n + 1.

        None when that quantile lies among the runs that did not stop: the
        budget is too small to say.
        """
        if not 0 < q < 1:
            raise ValueError("q must lie in (0, 1)")
        k = math.ceil(q * self.runs) - 1
        return float(self.stopping_times[k]) if k < self.stopped else None

    @property
    def median_n(self) -> float | None:
        return self.quantile(0.5)

    @property
    def mean_n_stopped(self) -> float | None:
        """Mean number of units among the runs that stopped (biased low when many did not)."""
        if not self.stopped:
            return None
        total = 0
        for n in self.stopping_times:
            total += n
        return total / self.stopped


def simulate(make: Callable[[], Procedure], draw: Callable[[random.Random], Mapping[str, Any]], *,
             runs: int = 200, max_n: int = 100_000, seed: int = 0,
             stop: Callable[[Any], bool] = _stopped, level: float = 0.05) -> SimulationSummary:
    """Run `make()` on `draw(rnd)` inputs until `stop(output)` or max_n inputs, `runs` times.

    `draw` receives the run's random generator and returns one statistical
    input (a dict).  Each run uses random.Random(seed + run).
    """
    if type(runs) is not int or runs < 1 or type(max_n) is not int or max_n < 1:
        raise ValueError("runs and max_n must be positive ints")
    times = []
    for r in range(runs):
        rnd = random.Random(seed + r)
        proc = make()
        state = proc.initial_state()
        rng = None
        if getattr(proc, "randomized", False):
            from .rng import SplitMix64

            rng = SplitMix64.for_role(seed + r, "statistical")
        for n in range(1, max_n + 1):
            state, out = proc.step(state, draw(rnd), rng)
            if stop(out):
                times.append(n)
                break
            if proc.is_terminal(state):
                break
    times.sort()
    return SimulationSummary(runs, max_n, len(times), tuple(times), level)


def _normal_quantile(p: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


# ---------------------------------------------------------------- scenarios


def conversion_units(p_treatment: float, p_control: float, allocation: float = 0.5, *,
                     arms: tuple[int, int] = (1, 0)) -> Callable[[random.Random], dict]:
    """Users of a 0/1 metric: {"arm", "y"}, treatment with probability `allocation`."""
    for p in (p_treatment, p_control, allocation):
        if not 0 <= p <= 1:
            raise ValueError("probabilities must lie in [0, 1]")

    def draw(rnd: random.Random) -> dict:
        treated = rnd.random() < allocation
        rate = p_treatment if treated else p_control
        return {"arm": arms[0] if treated else arms[1], "y": int(rnd.random() < rate)}

    return draw


def metric_units(treatment: Callable[[random.Random], float], control: Callable[[random.Random], float],
                 allocation: float = 0.5, *, arms: tuple[int, int] = (1, 0)) -> Callable[[random.Random], dict]:
    """Users of any metric: {"arm", "y"}, with y drawn by `treatment(rnd)` or `control(rnd)`."""
    if not 0 < allocation < 1:
        raise ValueError("allocation must lie in (0, 1)")

    def draw(rnd: random.Random) -> dict:
        treated = rnd.random() < allocation
        return {"arm": arms[0] if treated else arms[1], "y": (treatment if treated else control)(rnd)}

    return draw
