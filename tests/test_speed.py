# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Cost of a step as the run grows (absolute times are in PERFORMANCE.md, from tools/benchmark.py).

Absolute times depend on the machine, so they are not asserted.  What is
asserted is the shape of the cost: procedures with a constant state take
as long, and checkpoint as small, after thousands of observations as after
a hundred; those that hold the sample grow linearly, as documented.
"""
import json
import random
import statistics
import time
import unittest

from sequential_analysis.procedures import (
    CUSUM,
    EMA,
    AsymptoticConfidenceSequence,
    BettingMeanTest,
    Gaussian,
    GaussianEmission,
    HiddenMarkovFilter,
    KalmanFilter,
    NormalMixtureSPRT,
    QuantileConfidenceSequence,
    TMixtureSPRT,
)

CONSTANT = {
    "normal mixture SPRT": (lambda: NormalMixtureSPRT(1.0, 0.5, stop_on_reject=False), "x"),
    "t-mixture SPRT": (lambda: TMixtureSPRT(0.0, stop_on_reject=False), "x"),
    "betting mean test": (lambda: BettingMeanTest(0.5, -5.0, 5.0, stop_on_reject=False), "x"),
    "CUSUM": (lambda: CUSUM(Gaussian(1.0), 0.0, 1.0, threshold=1e9), "x"),
    "asymptotic sequence": (lambda: AsymptoticConfidenceSequence(start=10), "x"),
    "Kalman filter": (lambda: KalmanFilter(((1.0, 1.0), (0.0, 1.0)), ((1.0, 0.0),), ((0.01, 0.0), (0.0, 0.01)),
                                           1.0, (0.0, 0.0), ((10.0, 0.0), (0.0, 10.0))), "y"),
    "hidden Markov filter": (lambda: HiddenMarkovFilter((0.5, 0.5), ((0.9, 0.1), (0.1, 0.9)),
                                                        GaussianEmission((0.0, 1.0), (1.0, 1.0))), "y"),
    "EMA": (lambda: EMA(0.1), "x"),
}


def grown(make, key, n):
    proc, rnd = make(), random.Random(1)
    state = proc.initial_state()
    for _ in range(n):
        state, _ = proc.step(state, {key: rnd.gauss(0.0, 1.0)}, None)
    return proc, state


def step_time(proc, state, key) -> float:
    times = []
    for _ in range(200):
        start = time.perf_counter()
        proc.step(state, {key: 0.3}, None)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


class CostShape(unittest.TestCase):
    def test_constant_state_procedures_do_not_slow_down(self):
        for label, (make, key) in CONSTANT.items():
            with self.subTest(label):
                early, late = (grown(make, key, n) for n in (100, 3000))
                ratio = step_time(*late, key) / step_time(*early, key)
                self.assertLess(ratio, 4.0)  # a linear cost would make it about 30

    def test_constant_state_procedures_checkpoint_in_constant_space(self):
        for label, (make, key) in CONSTANT.items():
            with self.subTest(label):
                sizes = [len(json.dumps(p.encode_state(s))) for p, s in (grown(make, key, n) for n in (100, 3000))]
                self.assertLess(sizes[1], sizes[0] + 40)  # a few more digits at most

    def test_a_sample_holding_state_grows_linearly(self):
        sizes = [len(json.dumps(p.encode_state(s))) for p, s in
                 (grown(QuantileConfidenceSequence, "x", n) for n in (500, 1000, 2000))]
        self.assertAlmostEqual(sizes[2] / sizes[1], 2.0, delta=0.1)
        self.assertAlmostEqual(sizes[1] / sizes[0], 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
