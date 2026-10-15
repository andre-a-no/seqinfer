# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Mathematical correctness of the e-detectors.

* The recursions equal their definitions: a sum (Shiryaev-Roberts) or a
  maximum (CUSUM) over every candidate change point of the product of the
  betting factors, averaged over the bets.
* Before the change E[M_t] = t exactly when the mean is exactly m: checked
  by enumerating every 0/1 sequence (the supermartingale argument behind
  the run-length guarantee holds with equality).
* Average run length to a false alarm >= arl, by simulation, for skewed,
  discrete and heavy-tailed data; detection after a change.
"""
import itertools
import math
import random
import unittest

from sequential_analysis._compat import zip_strict
from sequential_analysis.errors import LifecycleError
from sequential_analysis.procedures import MeanEDetector, QuantileEDetector


def fold(proc, values):
    state, outs = proc.initial_state(), []
    for v in values:
        if proc.is_terminal(state):
            break
        state, out = proc.step(state, {"x": v})
        outs.append(out)
    return state, outs


def run_length(proc, draw, cap):
    state = proc.initial_state()
    for t in range(1, cap + 1):
        state, out = proc.step(state, {"x": draw()})
        if out.alarm:
            return t
    return cap


class Definition(unittest.TestCase):
    def test_recursions_equal_sums_and_maxima_over_change_points(self):
        rnd = random.Random(1)
        xs = [rnd.random() for _ in range(30)]
        for statistic in ("shiryaev-roberts", "cusum"):
            for direction in ("increase", "decrease", "both"):
                proc = MeanEDetector(0.4, arl=1e12, statistic=statistic, direction=direction, grid=4)
                _, outs = fold(proc, xs)
                for t in range(1, len(xs) + 1):
                    total = 0.0
                    for bet in proc.bets:
                        products = [math.prod(1 + bet * (x - 0.4) for x in xs[j:t]) for j in range(t)]
                        total += (sum(products) if statistic == "shiryaev-roberts" else max(products)) / len(proc.bets)
                    self.assertAlmostEqual(outs[t - 1].statistic, total, delta=1e-12 * max(1.0, total))

    def test_expected_statistic_is_t_before_the_change(self):
        m = 0.3
        for direction in ("increase", "both"):
            proc = MeanEDetector(m, arl=1e12, direction=direction, grid=3)
            t_max = 10
            expected = [0.0] * t_max
            for seq in itertools.product((0, 1), repeat=t_max):
                p = math.prod(m if v else 1 - m for v in seq)
                _, outs = fold(proc, seq)
                for t in range(t_max):
                    expected[t] += p * outs[t].statistic
            for t in range(t_max):
                self.assertAlmostEqual(expected[t], t + 1, delta=1e-12 * (t + 1))

    def test_cusum_is_never_above_shiryaev_roberts(self):
        rnd = random.Random(2)
        xs = [min(1.0, max(0.0, rnd.gauss(0.5 if i > 50 else 0.3, 0.2))) for i in range(150)]
        _, sr = fold(MeanEDetector(0.3, arl=1e12), xs)
        _, cu = fold(MeanEDetector(0.3, arl=1e12, statistic="cusum"), xs)
        self.assertTrue(all(c.statistic <= s.statistic * (1 + 1e-12) for c, s in zip_strict(cu, sr)))


class RunLength(unittest.TestCase):
    def check(self, make, draw, arl, runs=200, cap=20_000):
        lengths = [run_length(make(), draw, cap) for _ in range(runs)]
        mean = math.fsum(lengths) / runs
        self.assertGreaterEqual(mean, arl, f"average run length {mean}")
        return mean

    def test_bounded_mean_with_skewed_and_discrete_data(self):
        rnd = random.Random(3)
        self.check(lambda: MeanEDetector(0.2, arl=100), lambda: rnd.betavariate(0.5, 2.0), 100)
        self.check(lambda: MeanEDetector(0.1, arl=100, statistic="cusum"), lambda: float(rnd.random() < 0.1), 100)

    def test_quantile_with_cauchy_data(self):
        rnd = random.Random(4)
        cauchy = lambda: math.tan(math.pi * (rnd.random() - 0.5))
        self.check(lambda: QuantileEDetector(0.0, arl=100), cauchy, 100)
        upper_decile = math.tan(math.pi * 0.4)
        self.check(lambda: QuantileEDetector(upper_decile, p=0.9, arl=100, direction="increase"), cauchy, 100)


class Detection(unittest.TestCase):
    def test_mean_shift_is_detected_soon_after(self):
        rnd = random.Random(5)
        delays = []
        for _ in range(50):
            xs = [rnd.betavariate(2, 6) for _ in range(300)] + [rnd.betavariate(4, 4) for _ in range(1000)]
            _state, outs = fold(MeanEDetector(0.25, arl=1000), xs)
            alarm = next((o.n for o in outs if o.alarm), None)
            if alarm is None or alarm <= 300:
                continue
            delays.append(alarm - 300)
        self.assertGreater(len(delays), 45)
        self.assertLess(statistics_median(delays), 40)

    def test_change_estimate_points_at_the_change(self):
        rnd = random.Random(6)
        xs = [rnd.gauss(0, 3) for _ in range(400)] + [rnd.gauss(4, 3) for _ in range(400)]
        _state, outs = fold(QuantileEDetector(0.0, arl=1e6, statistic="cusum", direction="increase"), xs)
        last = next(o for o in outs if o.alarm)
        self.assertLess(abs(last.change_estimate - 401), 30)

    def test_stops_at_the_alarm(self):
        proc = MeanEDetector(0.1, arl=5)
        state, outs = fold(proc, [1.0] * 10)
        self.assertTrue(outs[-1].alarm)
        self.assertLess(len(outs), 10)
        with self.assertRaises(LifecycleError):
            proc.step(state, {"x": 0.5})

    def test_arguments(self):
        for kwargs in ({"m": 0.0}, {"m": 1.0}, {"m": 0.5, "arl": 0.5}, {"m": 0.5, "grid": 0},
                       {"m": 0.5, "direction": "up"}, {"m": 0.5, "statistic": "page"}):
            with self.assertRaises(ValueError):
                MeanEDetector(**kwargs)
        with self.assertRaises(ValueError):
            QuantileEDetector(0.0, p=1.0)


def statistics_median(values):
    values = sorted(values)
    return values[len(values) // 2]


if __name__ == "__main__":
    unittest.main()
