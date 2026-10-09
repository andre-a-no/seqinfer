# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Monitor an experiment continuously without assuming a distribution.

Two variants are compared on paired, skewed, heavy-tailed measurements.
The sequential sign test needs no model at all, the betting test only
needs the data to be bounded, and both may be looked at after every
observation: their error rates hold at any stopping time.  For contrast,
the mixture SPRT assumes normal data and reports a confidence sequence.
"""
import math

from seqinfer import PositionalPair, Run, difference
from seqinfer.procedures import BettingMeanTest, NormalMixtureSPRT, SequentialSignTest
from seqinfer.rng import SplitMix64
from seqinfer.sources import from_values

rng = SplitMix64.for_role(2026, "simulation/experiment")


def lognormal(mu):
    return math.exp(mu + 1.2 * rng.normal())


n = 2000
control = [lognormal(0.0) for _ in range(n)]
variant = [lognormal(0.3) for _ in range(n)]

# 1. Paired sign test on the differences: is the median difference zero?
sign = Run(SequentialSignTest(0.0), topology=PositionalPair(("variant", "control")).map(
    difference("variant", "control"), "variant-control"))
for a, b in zip(from_values("variant", variant), from_values("control", control), strict=True):
    sign.offer(a)
    sign.offer(b)
    if sign.terminal:
        break
print(f"sign test:    {sign.state.decision or 'no decision'} after {sign.state.n} pairs")

# 2. Betting test on a bounded metric: share of sessions above 1.0 in the variant, against 50%.
bet = Run(BettingMeanTest(0.5, 0.0, 1.0))
for v in variant:
    event = bet.step({"x": float(v > 1.0)})
    if bet.terminal:
        break
print(f"betting test: {event.output.decision} after {event.output.n} sessions, "
      f"share {event.output.estimate:.3f}, p = {event.output.p_value:.4f}")

# 3. Normal-theory mSPRT with a confidence sequence on the log scale.
mix = Run(NormalMixtureSPRT(sigma=1.2 * math.sqrt(2), tau=0.2, stop_on_reject=False))
for i, (a, b) in enumerate(zip(variant, control, strict=True), start=1):
    out = mix.step({"x": math.log(a) - math.log(b)}).output
    if i in (100, 500, 2000):
        print(f"mSPRT n={i:4d}: effect {out.mean:+.3f}, 95% sequence [{out.lower:+.3f}, {out.upper:+.3f}], "
              f"p = {out.p_value:.4f}")
