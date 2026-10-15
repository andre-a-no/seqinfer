# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Nonparametric change detection with e-detectors (Shin, Ramdas and Rinaldo 2024).

CUSUM and Shiryaev-Roberts need the density before and after the change.
An e-detector needs only a statement about the data before the change --
"the mean is m and the values lie in [lo, hi]", "the median is q0" -- and
guarantees the average run length to a false alarm, whatever the
distribution otherwise is.

For a bet lambda, a factor L_i = 1 + lambda Y_i with E[Y_i | past] = 0
before the change has conditional mean one.  The product of the factors
from a candidate change point j onwards is an e-process that grows only
after a real change, and the detectors combine all candidate j:

    Shiryaev-Roberts type   M_t = L_t (M_{t-1} + 1)        = sum_j prod_{i=j..t} L_i
    CUSUM type              M_t = L_t max(M_{t-1}, 1)      = max_j prod_{i=j..t} L_i

Before the change E[M_t | past] <= M_{t-1} + 1, so M_t - t is a
supermartingale, and by optional stopping the first t with M_t >= A has
E[tau] >= A: the average run length to a false alarm is at least `arl`.
The CUSUM type is never larger than the Shiryaev-Roberts type, so the
same holds for it.  As the size of the change is unknown, the detector
mixes a geometric grid of bets (each with its own recursion, equal
weights), which keeps the guarantee: a weighted average of e-detectors is
an e-detector.

MeanEDetector: values in [lo, hi] with mean m before the change; detects
a change of the mean.  QuantileEDetector: q0 is the p-quantile before the
change (P(X <= q0) = p); detects a change of the quantile, for data of
any shape, heavy tails included.  `direction` is "increase", "decrease"
or "both".

The two types have the same guarantee but behave differently: the
Shiryaev-Roberts type grows like t on average before the change, so its
run length to a false alarm is spread out -- with arl=1000, 14% of
simulated runs on skewed steady-state data alarmed within 400
observations -- while the CUSUM type stays near one and alarmed in 0.3 to
0.7% of them, in exchange for a somewhat later detection.  Choose CUSUM
when early false alarms are costly.

The reported change estimate is the start of the dominant bet's product
under the CUSUM rule (the last time its statistic was below one).

Cost per step (PERFORMANCE.md): O(grid) and a constant state, 6 µs for
the default grid of 10 bets per direction, both directions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure, decode_fields, refuse_terminal, require_state
from ..numerics import require_finite

DIRECTIONS = ("increase", "decrease", "both")
STATISTICS = ("shiryaev-roberts", "cusum")
_FRACTION = 0.5  # of the largest bet that keeps every factor positive


@dataclass(frozen=True)
class EDetectorState:
    n: int = 0
    values: tuple[float, ...] = ()  # M_t of each bet; empty before the first observation
    starts: tuple[int, ...] = ()  # CUSUM-rule start of each bet's product
    alarm_at: int | None = None


@dataclass(frozen=True)
class EDetectorOutput:
    n: int
    statistic: float  # the weighted mixture of M_t over the bets
    alarm: bool
    change_estimate: int | None  # observation number at which the dominant product starts


class _EDetector(Procedure):
    def _setup(self, arl: float, direction: str, statistic: str, grid: int, low: float, high: float) -> None:
        """`low` < 0 < `high` bound the centred score Y."""
        if not (math.isfinite(arl) and 1 <= arl <= 1e15):
            raise ValueError("arl must be a number of observations in [1, 1e15]")
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}")
        if type(grid) is not int or not 1 <= grid <= 60:
            raise ValueError("grid must be an int in [1, 60]")
        self.arl, self.direction, self.statistic, self.grid = float(arl), direction, statistic, grid
        bets: list[float] = []
        if direction != "decrease":
            bets += [_FRACTION / -low * 2.0**-k for k in range(grid)]
        if direction != "increase":
            bets += [-_FRACTION / high * 2.0**-k for k in range(grid)]
        self.bets = tuple(bets)
        self.weight = 1.0 / len(bets)

    def score(self, x) -> float:
        raise NotImplementedError

    def initial_state(self) -> EDetectorState:
        return EDetectorState()

    def step(self, state: EDetectorState, x, rng=None):
        refuse_terminal(self, state)
        y = self.score(x)
        n = state.n + 1
        previous = state.values or (0.0,) * len(self.bets)
        starts_before = state.starts or (1,) * len(self.bets)
        values, starts = [], []
        for bet, m, start in zip(self.bets, previous, starts_before, strict=True):
            factor = 1.0 + bet * y
            carried = m + 1.0 if self.statistic == "shiryaev-roberts" else max(m, 1.0)
            values.append(require_finite(factor * carried, "e-detector"))
            starts.append(n if m < 1.0 else start)
        total = 0.0
        for v in values:
            total += self.weight * v
        alarm = total >= self.arl
        best = max(range(len(values)), key=lambda k: (values[k], -k))
        change = starts[best] if values[best] > 1.0 else None
        new = EDetectorState(n, tuple(values), tuple(starts), n if alarm else None)
        return new, EDetectorOutput(n, total, alarm, change)

    def is_terminal(self, state: EDetectorState) -> bool:
        return state.alarm_at is not None

    def decode_state(self, data) -> EDetectorState:
        state = decode_fields(EDetectorState, data)
        size = len(self.bets) if state.n else 0
        require_state(len(state.values) == size and len(state.starts) == size, "one value per bet")
        require_state(all(v >= 0 for v in state.values), "negative statistic")
        require_state(all(1 <= s <= state.n for s in state.starts), "start out of range")
        require_state(state.alarm_at in (None, state.n), "alarm before the last observation")
        return state


class MeanEDetector(_EDetector):
    """Detects a change of the mean of values in [lo, hi] whose mean is m before the change."""

    name = "mean_e_detector"
    version = "1"

    def __init__(self, m: float, lo: float = 0.0, hi: float = 1.0, *, arl: float = 1000.0,
                 direction: str = "both", statistic: str = "shiryaev-roberts", grid: int = 10, input: str = "x"):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < m < hi):
            raise ValueError("need finite lo < m < hi")
        self.m, self.lo, self.hi, self.input = float(m), float(lo), float(hi), input
        self._setup(arl, direction, statistic, grid, self.lo - self.m, self.hi - self.m)

    def config(self) -> dict:
        return {"m": self.m, "lo": self.lo, "hi": self.hi, "arl": self.arl, "direction": self.direction,
                "statistic": self.statistic, "grid": self.grid, "input": self.input}

    def inputs(self):
        return {self.input: InputContract(self.input, lower=self.lo, upper=self.hi)}

    def score(self, x) -> float:
        return float(x[self.input] - self.m)


class QuantileEDetector(_EDetector):
    """Detects a change of the p-quantile, q0 before the change, for data of any distribution.

    The score is p - 1{X <= q0}: positive on average once the quantile has
    moved above q0.  Before the change P(X <= q0) = p is assumed (no atom
    of probability at q0).
    """

    name = "quantile_e_detector"
    version = "1"

    def __init__(self, q0: float, p: float = 0.5, *, arl: float = 1000.0, direction: str = "both",
                 statistic: str = "shiryaev-roberts", grid: int = 10, input: str = "x"):
        if not math.isfinite(q0):
            raise ValueError("q0 must be finite")
        if not 0 < p < 1:
            raise ValueError("p must lie in (0, 1)")
        self.q0, self.p, self.input = float(q0), float(p), input
        self._setup(arl, direction, statistic, grid, self.p - 1.0, self.p)

    def config(self) -> dict:
        return {"q0": self.q0, "p": self.p, "arl": self.arl, "direction": self.direction,
                "statistic": self.statistic, "grid": self.grid, "input": self.input}

    def inputs(self):
        return {self.input: InputContract(self.input)}

    def score(self, x) -> float:
        return self.p - (1.0 if x[self.input] <= self.q0 else 0.0)
