# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Exact anytime-valid comparison of two event rates: orders per user, errors per hour.

Each unit (a user, a machine, a site) reports its arm, a count k of events
and its exposure e (days observed, sessions, hours of operation); its count
is Poisson with mean lambda_arm * e.  The question is the rate ratio
rho = lambda_T / lambda_C.

Units are paired as they arrive: a unit waits for the next unit of the
other arm, and is replaced (not used) if one of its own arm comes first,
so that a pair is always two units adjacent in time.  Given the pair's
total count K, its treatment count is

    K_T ~ Binomial(K, rho E_T / (rho E_T + E_C)),

E_T and E_C the two exposures, whatever the common rate -- which may drift
over time (seasonality, a growing user base): only the two units of a
pair have to share it, and they are neighbours.  Which units are paired
depends on the arms alone, never on the counts, so discarding the
replaced units costs data (a third of the units at 50/50, almost half
at 30/70), not validity.  Conditioning makes H0: rho = rho0 a simple hypothesis, so

    E_pair = Bin(K_T; K, q(rho_hat)) / Bin(K_T; K, q(rho0)),

with rho_hat estimated from earlier pairs, has conditional mean exactly 1
under H0, and the product over pairs is a test martingale: an exact
anytime-valid test.  The confidence sequence for rho inverts the same
e-processes over a log-spaced grid of rho0 (each excluded value stays
excluded; the interval is intersected over time).  If every grid point is
excluded, the least-excluded one says where rho is: beyond the top of the
grid, the interval stays [top point, None); below the bottom, (None,
bottom point]; inside the grid (a level-alpha event) it collapses onto
that point and its neighbours, as in AnytimeLogRankTest.

The counts must be Poisson given the exposure.  Overdispersed counts
(negative binomial: some users order far more than others) break
exactness: use BoundedABTest on capped counts or ABTest instead.  As in
TwoProportionTest the allocation must not depend on outcomes.

Cost per step (PERFORMANCE.md): a few µs, plus O(grid) for each completed
pair -- 55 µs per unit on average with grid=200; the state holds at most
one waiting unit and one log-wealth per grid point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..errors import ContractViolation
from ..numerics import neumaier_add, require_finite
from .sprt import REJECT_H0

ALTERNATIVES = ("two-sided", "greater", "less")


@dataclass(frozen=True)
class RateRatioState:
    counts: tuple[int, int] = (0, 0)  # every unit seen: treatment, control
    exposures: tuple[float, float] = (0.0, 0.0)
    units: tuple[int, int] = (0, 0)
    pending: tuple[float, ...] = ()  # (arm index, count, exposure) of the unit waiting for its pair, or ()
    pairs: int = 0
    pair_counts: tuple[int, int] = (0, 0)  # the units of completed pairs: the plug-in estimate
    pair_exposures: tuple[float, float] = (0.0, 0.0)
    log_e: float = 0.0  # against rho0
    comp: float = 0.0
    max_log_e: float = 0.0
    grid_log_e: tuple[float, ...] = ()  # the confidence sequence: one e-process per grid point
    excluded: tuple[int, ...] = ()  # 1 once a grid point has been excluded
    decision: str | None = None


@dataclass(frozen=True)
class RateRatioOutput:
    units: int
    rates: tuple[float | None, float | None]  # events per unit of exposure: treatment, control
    rate_ratio: float | None
    log_e_value: float
    p_value: float
    decision: str | None
    lower: float | None  # confidence sequence for rho; None: below the grid (lower) or above it (upper)
    upper: float | None
    pairs: int


def _log_binomial_ratio(k: int, n: int, q1: float, q0: float) -> float:
    """log Bin(k; n, q1) / Bin(k; n, q0): the binomial coefficients cancel."""
    out = 0.0
    if k:
        out += k * (math.log(q1) - math.log(q0))
    if n - k:
        out += (n - k) * (math.log1p(-q1) - math.log1p(-q0))
    return out


class RateRatioTest(Procedure):
    """Exact anytime-valid test and confidence sequence for a ratio of Poisson rates; see the module docstring.

    Inputs: the arm (`arm_input`), the count (`input`, a non-negative
    integer) and the exposure (`exposure_input`, positive; omit it with
    exposure=False when every unit has exposure 1).  `rho0` is the null
    rate ratio, `alternative` "greater" tests rho <= rho0 against rho > rho0.
    `grid` points log-spaced over [ratio_range[0], ratio_range[1]] carry the
    confidence sequence (grid=0: none).
    """

    name = "rate_ratio_test"
    version = "1"
    guarantee = "exact"
    e_value_field = "log_e_value"

    def __init__(self, alpha: float = 0.05, *, rho0: float = 1.0, alternative: str = "two-sided",
                 arms: tuple[int, int] = (1, 0), exposure: bool = True, prior: float = 0.5, grid: int = 200,
                 ratio_range: tuple[float, float] = (0.1, 10.0), stop_on_reject: bool = False,
                 arm_input: str = "arm", input: str = "count", exposure_input: str = "exposure"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if not (math.isfinite(rho0) and rho0 > 0):
            raise ValueError("rho0 must be positive and finite")
        if alternative not in ALTERNATIVES:
            raise ValueError(f"alternative must be one of {ALTERNATIVES}")
        if len(arms) != 2 or any(type(a) is not int or a < 0 for a in arms) or arms[0] == arms[1]:
            raise ValueError("arms must be two different non-negative integer labels")
        if not (math.isfinite(prior) and prior > 0):
            raise ValueError("prior must be a positive number of pseudo-events")
        if type(grid) is not int or not (grid == 0 or 10 <= grid <= 100_000):
            raise ValueError("grid must be 0 (no confidence sequence) or an int in [10, 100000]")
        lo, hi = (float(v) for v in ratio_range)
        if not (0 < lo < hi < math.inf):
            raise ValueError("ratio_range must be 0 < low < high")
        names = {arm_input, input} | ({exposure_input} if exposure else set())
        if len(names) != (3 if exposure else 2):
            raise ValueError("the arm, count and exposure need different names")
        self.alpha, self.rho0, self.alternative = float(alpha), float(rho0), alternative
        self.arms, self.exposure = (arms[0], arms[1]), bool(exposure)
        self.prior, self.grid, self.ratio_range = float(prior), grid, (lo, hi)
        self.stop_on_reject, self.arm_input, self.input, self.exposure_input = (
            stop_on_reject, arm_input, input, exposure_input)
        self._log_level = -math.log(self.alpha)
        span = math.log(hi) - math.log(lo)
        self._points = tuple(math.exp(math.log(lo) + span * j / (grid - 1)) for j in range(grid)) if grid else ()

    def config(self) -> dict:
        return {"alpha": self.alpha, "rho0": self.rho0, "alternative": self.alternative, "arms": list(self.arms),
                "exposure": self.exposure, "prior": self.prior, "grid": self.grid,
                "ratio_range": list(self.ratio_range), "stop_on_reject": self.stop_on_reject,
                "arm_input": self.arm_input, "input": self.input, "exposure_input": self.exposure_input}

    def inputs(self):
        out = {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0),
               self.input: InputContract(self.input, kind="integer", lower=0)}
        if self.exposure:
            out[self.exposure_input] = InputContract(self.exposure_input)
        return out

    def initial_state(self) -> RateRatioState:
        return RateRatioState(grid_log_e=(0.0,) * self.grid, excluded=(0,) * self.grid)

    def plug_in(self, counts: tuple[int, int], exposures: tuple[float, float]) -> float:
        """rho_hat from the completed pairs, shrunk towards 1 by `prior` pseudo-events in each arm."""
        if exposures[0] <= 0 or exposures[1] <= 0:
            return 1.0
        return ((counts[0] + self.prior) / exposures[0]) / ((counts[1] + self.prior) / exposures[1])

    @staticmethod
    def share(rho: float, e_t: float, e_c: float) -> float:
        """P(an event of the pair is the treatment unit's) under rate ratio rho."""
        return rho * e_t / (rho * e_t + e_c)

    def pair_log_e(self, rho_hat: float, rho0: float, alternative: str, k_t: int, k: int, e_t: float,
                   e_c: float) -> float:
        """log of one pair's e-value against rho0 (0 when not betting)."""
        if (alternative == "greater" and rho_hat <= rho0) or (alternative == "less" and rho_hat >= rho0):
            return 0.0  # no bet on the null side
        if k == 0 or rho_hat == rho0:
            return 0.0
        return _log_binomial_ratio(k_t, k, self.share(rho_hat, e_t, e_c), self.share(rho0, e_t, e_c))

    def _interval(self, state: RateRatioState) -> tuple[float | None, float | None]:
        kept = [j for j, x in enumerate(state.excluded) if not x]
        if not kept:
            # Every point excluded.  The least-excluded point (the smallest e-value) tells where the ratio is:
            # at an end of the grid, beyond it, and the interval keeps its last bound (it stays nested);
            # inside, a level-alpha event, and the interval collapses onto that point and its neighbours.
            least = min(range(self.grid), key=state.grid_log_e.__getitem__)
            if least == self.grid - 1:
                return self._points[-1], None
            if least == 0:
                return None, self._points[0]
            return self._points[least - 1], self._points[least + 1]

        lo = None if kept[0] == 0 else self._points[kept[0] - 1]
        hi = None if kept[-1] == self.grid - 1 else self._points[kept[-1] + 1]
        return lo, hi

    def _output(self, state: RateRatioState, value: float) -> RateRatioOutput:
        (kt, kc), (et, ec) = state.counts, state.exposures
        rt = kt / et if et > 0 else None
        rc = kc / ec if ec > 0 else None
        ratio = rt / rc if rt is not None and rc else None
        lower, upper = self._interval(state) if self.grid else (None, None)
        return RateRatioOutput(state.units[0] + state.units[1], (rt, rc), ratio, value,
                               min(1.0, math.exp(-state.max_log_e)), state.decision, lower, upper, state.pairs)

    def step(self, state: RateRatioState, x, rng=None):
        label = x[self.arm_input]
        if label not in self.arms:
            return state, self._output(state, state.log_e + state.comp)
        k = self.arms.index(label)
        count = int(x[self.input])
        e = float(x[self.exposure_input]) if self.exposure else 1.0
        if not (e > 0 and math.isfinite(e)):
            raise ContractViolation(f"input {self.exposure_input!r}: exposure must be positive, got {e!r}")
        counts, exposures, units = list(state.counts), list(state.exposures), list(state.units)
        counts[k] += count
        exposures[k] += e
        units[k] += 1
        log_e, comp, pairs = state.log_e, state.comp, state.pairs
        pair_k, pair_e = state.pair_counts, state.pair_exposures
        grid_log_e, excluded = state.grid_log_e, state.excluded
        if not state.pending or int(state.pending[0]) == k:
            pending: tuple[float, ...] = (float(k), float(count), e)  # wait (replacing a unit of the same arm)
        else:
            pending = ()
            (k_t, e_t), (k_c, e_c) = ((count, e), (int(state.pending[1]), state.pending[2])) if k == 0 else \
                ((int(state.pending[1]), state.pending[2]), (count, e))
            total = k_t + k_c
            rho_hat = self.plug_in(pair_k, pair_e)
            log_e, comp = neumaier_add(log_e, comp,
                                       self.pair_log_e(rho_hat, self.rho0, self.alternative, k_t, total, e_t, e_c))
            if self.grid:
                new_grid, new_excluded = [], []
                for rho, g, x_ in zip(self._points, grid_log_e, excluded):
                    g += self.pair_log_e(rho_hat, rho, "two-sided", k_t, total, e_t, e_c)
                    new_grid.append(g)
                    new_excluded.append(1 if x_ or g >= self._log_level else 0)
                grid_log_e, excluded = tuple(new_grid), tuple(new_excluded)
            pairs += 1
            pair_k = (pair_k[0] + k_t, pair_k[1] + k_c)
            pair_e = (pair_e[0] + e_t, pair_e[1] + e_c)
        value = require_finite(log_e + comp, "log e-value")
        decision = state.decision or (REJECT_H0 if value >= self._log_level else None)
        new = RateRatioState((counts[0], counts[1]), (exposures[0], exposures[1]), (units[0], units[1]), pending,
                             pairs, pair_k, pair_e, log_e, comp, max(state.max_log_e, value), grid_log_e, excluded,
                             decision)
        return new, self._output(new, value)

    def is_terminal(self, state: RateRatioState) -> bool:
        return self.stop_on_reject and state.decision is not None

    def decode_state(self, data) -> RateRatioState:
        state = decode_fields(RateRatioState, data)
        require_state(len(state.grid_log_e) == self.grid and len(state.excluded) == self.grid,
                      "one value per grid point")
        require_state(all(x in (0, 1) for x in state.excluded), "exclusion flags are 0 or 1")
        p = state.pending
        require_state(p == () or (len(p) == 3 and p[0] in (0.0, 1.0) and p[1] == int(p[1]) and p[2] > 0),
                      "pending (arm, count, exposure)")
        require_state(all(len(v) == 2 for v in (state.counts, state.exposures, state.units, state.pair_counts,
                                                state.pair_exposures)), "two arms")
        require_state(2 * state.pairs <= state.units[0] + state.units[1], "more pairs than units")
        return state
