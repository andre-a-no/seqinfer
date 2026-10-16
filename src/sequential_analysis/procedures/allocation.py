# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Redirecting traffic during an experiment: Thompson allocation and a deployment rule.

ThompsonAllocation follows K arms of an experiment and says how to split
the next units between them.  Each arm's mean gets a posterior:

* metric="binary" (conversions): Beta(a + successes, b + failures);
* metric="normal" (revenue, time, any real outcome): with the prior
  p(mu, sigma^2) ~ 1/sigma^2, the mean's posterior is Student t with
  n - 1 degrees of freedom, location the sample mean and scale s / sqrt(n).

From these it computes, for each arm, the probability that it is the best
(the highest mean) and its expected loss, E[max_j mu_j - mu_k]: what is
lost, in units of the metric, by deploying arm k if it is not the best.
Both are one-dimensional integrals,

    P(k best) = int f_k(x) prod_{j != k} F_j(x) dx,
    E[max_j mu_j] = int x d(prod_j F_j(x)),

computed by Gauss-Legendre quadrature (16 points) on pieces cut at each
posterior's location and multiples of its scale, so that every posterior,
narrow or wide, is resolved: deterministic, no random draws.

The allocation is Thompson sampling's: arm k gets the share P(k best),
which sends most traffic to arms likely to be best and keeps exploring
the others in proportion to their chance.  `floor` reserves a minimum
share for every arm (so that every comparison keeps data coming and the
inverse-propensity weights of the tests stay bounded):

    share_k = floor + (1 - K floor - reserve) P(k best) + reserve [k = control].

`reserve` keeps a fixed extra share for a `control` arm (a holdout), so
that every variant can still be compared with the control quickly: with
traffic redirected, the comparisons the traffic left behind starve, and a
test of a variant against a control at the floor needs many units.

Until every arm has `min_units` units the split stays equal; afterwards
the shares are recomputed every `update_every` units, as a production
system would push a new split periodically.  The output's `shares` are
the split to use for the next units, and the assignment of each unit must
use them: record them with the unit (they are the propensities that
BoundedABTest(propensity_input=...) needs).

`deploy` is the expected-loss rule of Bayesian A/B testing: once some
arm's expected loss is at most `threshold` (the "threshold of caring", in
units of the metric), that arm can take all the traffic.  This is a
decision rule, not a test: it controls the expected cost of the choice
under the posterior, not an error rate.  When an error rate matters,
confirm with a test that stays valid under redirected traffic --
TwoProportionTest(method="universal") for conversions, or
BoundedABTest(propensity_input=...) for any bounded metric -- before
deploying.  Tests that assume a fixed split (TwoProportionTest with
method="grow", ABTest, RatioABTest, RateRatioTest, BoundedABTest without
propensities) are not valid once the split follows the outcomes.

Cost per step (PERFORMANCE.md): constant time and state.  The quadrature
runs once per `update_every` units and takes about 15 ms for three binary
arms (the incomplete beta function at some 800 points per arm): 0.15 ms
per unit with the default update_every=100.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .._compat import zip_strict
from ..contracts import InputContract
from ..core import Procedure, decode_fields, require_state
from ..distributions import regularized_beta, t_cdf
from ..numerics import require_finite

METRICS = ("binary", "normal")

# 16-point Gauss-Legendre rule on [-1, 1]
_NODES = (-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755404408355003, -0.6178762444026438,
          -0.45801677765722737, -0.2816035507792589, -0.09501250983763744, 0.09501250983763744, 0.2816035507792589,
          0.45801677765722737, 0.6178762444026438, 0.755404408355003, 0.8656312023878318, 0.9445750230732326,
          0.9894009349916499)
_WEIGHTS = (0.027152459411754176, 0.062253523938647456, 0.0951585116824926, 0.12462897125553407,
            0.1495959888165767, 0.16915651939500265, 0.18260341504492364, 0.18945061045506864, 0.18945061045506864,
            0.18260341504492364, 0.16915651939500265, 0.1495959888165767, 0.12462897125553407, 0.0951585116824926,
            0.062253523938647456, 0.027152459411754176)
# pieces of the integration range: each posterior's location plus these multiples of its scale
_CUTS = (-40.0, -21.0, -13.0, -8.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 40.0)


class _Posterior:
    """The posterior of one arm's mean: density, distribution function, location and scale."""

    def __init__(self, metric: str, count: int, mean: float, m2: float, prior: tuple[float, float]):
        self.metric = metric
        if metric == "binary":
            successes = float(round(mean * count))  # the running mean of 0/1 outcomes times their number
            self.a, self.b = prior[0] + successes, prior[1] + count - successes
            total = self.a + self.b
            self.location = self.a / total
            self.scale = math.sqrt(self.a * self.b / (total * total * (total + 1.0)))
            self.log_norm = math.lgamma(total) - math.lgamma(self.a) - math.lgamma(self.b)
        else:
            self.df = float(count - 1)
            self.location = mean
            self.scale = math.sqrt(m2 / (count - 1) / count)

    def density(self, x: float) -> float:
        if self.metric == "binary":
            if not 0.0 < x < 1.0:
                return 0.0
            return math.exp(self.log_norm + (self.a - 1.0) * math.log(x) + (self.b - 1.0) * math.log1p(-x))
        z = (x - self.location) / self.scale
        df = self.df
        return math.exp(math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df) - 0.5 * math.log(df * math.pi)
                        - 0.5 * (df + 1.0) * math.log1p(z * z / df)) / self.scale

    def cdf(self, x: float) -> float:
        if self.metric == "binary":
            if x <= 0.0:
                return 0.0
            if x >= 1.0:
                return 1.0
            return regularized_beta(x, self.a, self.b)
        return t_cdf((x - self.location) / self.scale, self.df)


def best_arm_probabilities(posteriors: list[_Posterior], metric: str) -> tuple[list[float], list[float]]:
    """(P(k best), expected loss of k) for each arm, by piecewise Gauss-Legendre quadrature."""
    k = len(posteriors)
    cuts = []
    for p in posteriors:
        for c in _CUTS:
            cuts.append(p.location + c * p.scale)
    if metric == "binary":
        cuts = [min(1.0, max(0.0, c)) for c in cuts] + [0.0, 1.0]
    cuts = sorted(set(cuts))
    best = [0.0] * k
    top = 0.0  # E[max_j mu_j]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
        for node, weight in zip_strict(_NODES, _WEIGHTS):
            x = mid + half * node
            dens = [p.density(x) for p in posteriors]
            dist = [p.cdf(x) for p in posteriors]
            # products of the other arms' distribution functions, without dividing by one that may be 0
            before, after = [1.0] * k, [1.0] * k
            for i in range(1, k):
                before[i] = before[i - 1] * dist[i - 1]
            for i in range(k - 2, -1, -1):
                after[i] = after[i + 1] * dist[i + 1]
            w = weight * half
            for i in range(k):
                term = w * dens[i] * before[i] * after[i]
                best[i] += term
                top += term * x
    total = 0.0
    for v in best:
        total += v
    probabilities = [v / total for v in best]
    top /= total
    return probabilities, [max(0.0, top - p.location) for p in posteriors]


@dataclass(frozen=True)
class ThompsonAllocationState:
    n: int
    counts: tuple[int, ...]
    means: tuple[float, ...]  # running means of the outcomes per arm (Welford)
    m2: tuple[float, ...]  # running sums of squared deviations per arm
    shares: tuple[float, ...]  # the split in force for the next units
    since: int  # units since the split was last computed
    best: tuple[float, ...]  # P(k best) when the split was last computed; () before
    loss: tuple[float, ...]  # expected loss of each arm, idem
    deploy: int | None = None  # the arm the deployment rule chose, once it has


@dataclass(frozen=True)
class ThompsonAllocationOutput:
    n: int
    shares: tuple[float, ...]  # how to split the next units between the arms
    counts: tuple[int, ...]
    means: tuple[float | None, ...]
    best: tuple[float, ...]  # probability that each arm is the best, at the last recomputation
    loss: tuple[float, ...]  # expected loss of deploying each arm, idem
    leader: int | None  # the arm most likely to be best
    deploy: int | None  # the arm to deploy, once one's expected loss is at most `threshold`


class ThompsonAllocation(Procedure):
    """Traffic split by Thompson sampling with a floor, and the expected-loss deployment rule.

    See the module docstring.  `arms` is the number of arms, labelled
    0..arms-1; the inputs are a unit's arm and its outcome.
    """

    name = "thompson_allocation"
    version = "1"
    guarantee = "model"

    def __init__(self, arms: int = 2, *, metric: str = "binary", floor: float = 0.05, threshold: float = 0.0,
                 min_units: int = 100, update_every: int = 100, prior: tuple[float, float] = (1.0, 1.0),
                 control: int | None = None, reserve: float = 0.0, arm_input: str = "arm", input: str = "y"):
        if type(arms) is not int or arms < 2:
            raise ValueError("arms must be an int >= 2")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if not 0.0 <= floor <= 1.0 / arms:
            raise ValueError("floor must lie in [0, 1/arms]")
        if not 0.0 <= threshold < math.inf:
            raise ValueError("threshold must be a non-negative number")
        if type(min_units) is not int or min_units < (2 if metric == "normal" else 0):
            raise ValueError("min_units must be an int (at least 2 for a normal metric)")
        if type(update_every) is not int or update_every < 1:
            raise ValueError("update_every must be a positive int")
        a, b = (float(v) for v in prior)
        if not (0.0 < a < math.inf and 0.0 < b < math.inf):
            raise ValueError("the Beta prior needs positive parameters")
        if arm_input == input:
            raise ValueError("the arm and the outcome need different names")
        if control is None:
            if reserve != 0.0:
                raise ValueError("a reserve needs a control arm")
        elif type(control) is not int or not 0 <= control < arms:
            raise ValueError("control must be one of the arms")
        if not 0.0 <= reserve <= 1.0 - arms * floor:
            raise ValueError("reserve must lie in [0, 1 - arms * floor]")
        self.control, self.reserve = control, float(reserve)
        self.arms, self.metric, self.floor, self.threshold = arms, metric, float(floor), float(threshold)
        self.min_units, self.update_every, self.prior = min_units, update_every, (a, b)
        self.arm_input, self.input = arm_input, input

    def config(self) -> dict:
        return {"arms": self.arms, "metric": self.metric, "floor": self.floor, "threshold": self.threshold,
                "min_units": self.min_units, "update_every": self.update_every, "prior": list(self.prior),
                "control": self.control, "reserve": self.reserve, "arm_input": self.arm_input, "input": self.input}

    def inputs(self):
        outcome = (InputContract(self.input, kind="binary") if self.metric == "binary"
                   else InputContract(self.input))
        return {self.arm_input: InputContract(self.arm_input, kind="integer", lower=0, upper=self.arms - 1),
                self.input: outcome}

    def initial_state(self) -> ThompsonAllocationState:
        k = self.arms
        return ThompsonAllocationState(0, (0,) * k, (0.0,) * k, (0.0,) * k, (1.0 / k,) * k, 0, (), ())

    def _posteriors(self, state: ThompsonAllocationState) -> list[_Posterior] | None:
        if any(c < self.min_units for c in state.counts):
            return None
        if self.metric == "normal" and not all(c >= 2 and v > 0.0 for c, v in zip_strict(state.counts, state.m2)):
            return None
        return [_Posterior(self.metric, c, m, v, self.prior)
                for c, m, v in zip_strict(state.counts, state.means, state.m2)]

    def step(self, state: ThompsonAllocationState, x, rng=None):
        arm, y = x[self.arm_input], float(x[self.input])
        counts, means, m2 = list(state.counts), list(state.means), list(state.m2)
        counts[arm] += 1
        delta = y - means[arm]
        means[arm] = require_finite(means[arm] + delta / counts[arm], "mean")
        m2[arm] = require_finite(m2[arm] + delta * (y - means[arm]), "sum of squares")
        new = ThompsonAllocationState(state.n + 1, tuple(counts), tuple(means), tuple(m2), state.shares,
                                      state.since + 1, state.best, state.loss, state.deploy)
        if new.since >= self.update_every or (not state.best and new.since >= 1):
            posteriors = self._posteriors(new)
            if posteriors is not None:
                best, loss = best_arm_probabilities(posteriors, self.metric)
                spread = 1.0 - self.arms * self.floor - self.reserve
                shares = tuple(self.floor + spread * p + (self.reserve if i == self.control else 0.0)
                               for i, p in enumerate(best))
                deploy = state.deploy
                if deploy is None:
                    choice = min(range(self.arms), key=lambda i: (loss[i], i))
                    if loss[choice] <= self.threshold:
                        deploy = choice
                new = ThompsonAllocationState(new.n, new.counts, new.means, new.m2, shares, 0, tuple(best),
                                              tuple(loss), deploy)
            elif new.since >= self.update_every:
                new = ThompsonAllocationState(new.n, new.counts, new.means, new.m2, new.shares, 0, new.best,
                                              new.loss, new.deploy)
        leader = max(range(self.arms), key=lambda i: (new.best[i], -i)) if new.best else None
        return new, ThompsonAllocationOutput(
            new.n, new.shares, new.counts, tuple(m if c else None for m, c in zip_strict(new.means, new.counts)),
            new.best, new.loss, leader, new.deploy)

    def decode_state(self, data) -> ThompsonAllocationState:
        state = decode_fields(ThompsonAllocationState, data)
        k = self.arms
        require_state(len(state.counts) == len(state.means) == len(state.m2) == len(state.shares) == k,
                      "one entry per arm")
        units = 0
        for c in state.counts:
            units += c
        require_state(all(c >= 0 for c in state.counts) and units == state.n, "counts")
        require_state(all(v >= 0 for v in state.m2), "sums of squares")
        require_state(all(v == 0 for c, v in zip_strict(state.counts, state.m2) if c <= 1), "a spread from one unit")
        if self.metric == "binary":
            require_state(all(0.0 <= m <= 1.0 for m in state.means), "binary means must lie in [0, 1]")
        total = 0.0
        for s in state.shares:
            total += s
        require_state(all(s >= 0 for s in state.shares) and abs(total - 1.0) < 1e-9, "shares must sum to 1")
        require_state(len(state.best) in (0, k) and len(state.loss) == len(state.best), "probabilities")
        require_state(0 <= state.since <= self.update_every, "units since the last split")
        require_state(state.deploy is None or (0 <= state.deploy < k and bool(state.best)), "deployed arm")
        return state
