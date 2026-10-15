# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The plots draw what the outputs say: line data, reference lines, markers and labels (not pixels).

matplotlib is optional: without it these tests are skipped, except the one
that checks the message a user without it gets.
"""
import math
import random
import re
import sys
import unittest
from pathlib import Path
from statistics import NormalDist
from unittest import mock

from sequential_analysis import Recorder, Run, run_sync, trajectory
from sequential_analysis.group_sequential import group_sequential_design
from sequential_analysis.planning import simulate
from sequential_analysis.procedures import (
    CUSUM,
    BayesianChangePoint,
    Gaussian,
    GroupSequentialTest,
    KalmanFilter,
    LocalLevelKalman,
    MeanEDetector,
    MultipleTests,
    NormalGammaModel,
    QuantileConfidenceSequence,
    ShiryaevRoberts,
    TMixtureSPRT,
)
from sequential_analysis.smoothing import kalman_smoother
from sequential_analysis.sources import from_values, gaussian

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
except ImportError:  # pragma: no cover - the CI jobs without matplotlib
    plt = None

from sequential_analysis import plotting

ROOT = Path(__file__).resolve().parent.parent
LN10 = math.log(10.0)


def same(a, b, tol=1e-12):
    """Equal sequences of floats, NaN equal to NaN."""
    a, b = list(a), list(b)
    return len(a) == len(b) and all(
        (math.isnan(x) and math.isnan(y)) or abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b)
    )


def recorded(procedure, source):
    seen = Recorder()
    run_sync(Run(procedure, consumers=[seen]), source)
    return seen


class WithoutMatplotlib(unittest.TestCase):
    def test_the_module_imports_and_a_call_names_the_extra(self):
        with mock.patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with self.assertRaises(ImportError) as caught:
                plotting.plot_stopping_times([1, 2, 3])
        self.assertIn('pip install "sequential-analysis[plot]"', str(caught.exception))

    def test_every_function_is_exported(self):
        for name in plotting.__all__:
            self.assertTrue(callable(getattr(plotting, name)), name)


@unittest.skipIf(plt is None, "matplotlib is not installed")
class Plots(unittest.TestCase):
    def setUp(self):
        show = mock.patch.object(plt, "show", side_effect=AssertionError("a plot function called plt.show()"))
        show.start()
        self.addCleanup(show.stop)
        self.addCleanup(plt.close, "all")

    def lines(self, ax, label=None):
        return [line for line in ax.get_lines() if label is None or line.get_label() == label]

    def texts(self, ax):
        return [t.get_text() for t in ax.texts]

    def legend(self, ax):
        legend = ax.get_legend()
        return [t.get_text() for t in legend.get_texts()] if legend is not None else []

    def hline_at(self, ax, y):
        return any(len(set(line.get_ydata())) == 1 and abs(line.get_ydata()[0] - y) < 1e-12
                   and list(line.get_xdata()) == [0, 1] for line in ax.get_lines())

    def red_marker(self, ax):
        dots = [line for line in ax.get_lines() if line.get_marker() == "o" and line.get_linestyle() == "None"
                and len(line.get_xdata()) == 1]
        self.assertEqual(len(dots), 1)
        return float(dots[0].get_xdata()[0]), float(dots[0].get_ydata()[0]), dots[0].get_label()


class EProcess(Plots):
    def setUp(self):
        super().setUp()
        self.seen = recorded(TMixtureSPRT(theta0=0.0, effect=0.5, alpha=0.05), gaussian("x", mean=0.5, sd=1.0,
                                                                                         seed=3, n=500))
        self.outputs = self.seen.outputs

    def test_the_e_value_on_a_log10_scale_the_threshold_and_the_stop(self):
        ax = plotting.plot_e_process(self.seen, alpha=0.05)
        curve = self.lines(ax)[0]
        self.assertEqual(list(curve.get_xdata()), [o.n for o in self.outputs])
        self.assertTrue(same(curve.get_ydata(), [o.log_bf / LN10 for o in self.outputs]))
        self.assertTrue(self.hline_at(ax, math.log10(20)))
        self.assertIn("1/α = 20", self.texts(ax))
        stop = next(o for o in self.outputs if o.decision)
        x, y, label = self.red_marker(ax)
        self.assertEqual((x, y), (stop.n, stop.log_bf / LN10))
        self.assertEqual(label, f"reject_h0 at {stop.n}")
        self.assertEqual(self.legend(ax), [label])
        self.assertEqual(ax.get_ylabel(), "e-value (log scale)")
        self.assertEqual(ax.get_xlabel(), "observations")
        self.assertEqual(ax.yaxis.get_major_formatter()(1.0, 0), "10")
        self.assertEqual(ax.yaxis.get_major_formatter()(5.0, 0), "$10^{5}$")

    def test_every_form_of_the_outputs_gives_the_same_plot(self):
        test = TMixtureSPRT(theta0=0.0, effect=0.5, alpha=0.05)
        inputs = [{"x": random.Random(1).gauss(0.5, 1.0)} for _ in range(50)]
        pairs = trajectory(test, inputs)
        seen = recorded(test, from_values("x", [x["x"] for x in inputs]))
        forms = [seen, seen.events, seen.outputs, pairs]
        drawn = [self.lines(plotting.plot_e_process(form))[0] for form in forms]
        for line in drawn[1:]:
            self.assertEqual(list(line.get_xdata()), list(drawn[0].get_xdata()))
            self.assertTrue(same(line.get_ydata(), drawn[0].get_ydata()))

    def test_a_given_axes_is_used_and_returned(self):
        _, (left, right) = plt.subplots(1, 2)
        self.assertIs(plotting.plot_e_process(self.seen, ax=right), right)
        self.assertEqual(self.lines(left), [])

    def test_field_names_another_e_value(self):
        ax = plotting.plot_e_process(self.outputs, field="t")
        self.assertTrue(same(self.lines(ax)[0].get_ydata(), [
            (o.t if o.t is not None else math.nan) / LN10 for o in self.outputs]))
        with self.assertRaisesRegex(ValueError, "no field 'nothing'"):
            plotting.plot_e_process(self.outputs, field="nothing")

    def test_outputs_without_an_e_value_are_refused(self):
        outputs = recorded(CUSUM(Gaussian(1.0), 0.0, 1.0, 5.0), from_values("x", [0.0, 1.0])).outputs
        with self.assertRaisesRegex(ValueError, "CusumOutput has no log e-value field"):
            plotting.plot_e_process(outputs)
        with self.assertRaisesRegex(ValueError, "nothing to plot"):
            plotting.plot_e_process([])
        with self.assertRaisesRegex(ValueError, "alpha"):
            plotting.plot_e_process(self.outputs, alpha=1.5)

    def test_multiple_tests_one_line_per_test_e_bh_thresholds_and_rejections(self):
        tests = MultipleTests({name: TMixtureSPRT(effect=0.5, stop_on_reject=False, input=name) for name in "ab"},
                              alpha=0.05)
        rnd = random.Random(2)
        inputs = [{"a": rnd.gauss(0.8, 1.0), "b": rnd.gauss(0.0, 1.0)} for _ in range(150)]
        outputs = [o for _, o in trajectory(tests, inputs)]
        ax = plotting.plot_e_process(outputs, alpha=0.05)
        self.assertEqual(self.legend(ax), ["a", "b"])
        self.assertTrue(same(self.lines(ax, "a")[0].get_ydata(), [o.log_e_values["a"] / LN10 for o in outputs]))
        self.assertTrue(self.hline_at(ax, math.log10(20)))
        self.assertTrue(self.hline_at(ax, math.log10(40)))
        first = next(o for o in outputs if "a" in o.rejected)
        dots = [line for line in ax.get_lines() if len(line.get_xdata()) == 1]
        self.assertEqual([float(d.get_xdata()[0]) for d in dots], [first.n])
        ax = plotting.plot_e_process(outputs, field="b")
        self.assertEqual(self.legend(ax), ["b"])


class ConfidenceSequence(Plots):
    def test_a_band_with_gaps_where_unbounded_the_estimate_and_the_truth(self):
        rnd = random.Random(7)
        outputs = recorded(QuantileConfidenceSequence(p=0.5), from_values("x", [rnd.gauss(1.0, 1.0)
                                                                              for _ in range(300)])).outputs
        self.assertIsNone(outputs[0].lower)  # unbounded at the start
        ax = plotting.plot_confidence_sequence(outputs, truth=1.0)
        lower = [o.lower if o.lower is not None else math.nan for o in outputs]
        self.assertTrue(any(same(line.get_ydata(), lower) for line in ax.get_lines()))
        estimate = self.lines(ax, "estimate (estimate)")[0]
        self.assertTrue(same(estimate.get_ydata(), [o.estimate if o.estimate is not None else math.nan
                                                    for o in outputs]))
        self.assertIn("true value 1", self.legend(ax))
        self.assertIn("confidence sequence", self.legend(ax))
        band = [c for c in ax.collections if isinstance(c, PolyCollection)]
        self.assertEqual(len(band), 1)
        self.assertEqual(ax.get_xlabel(), "observations")

    def test_the_wide_start_does_not_flatten_the_plot(self):
        outputs = recorded(TMixtureSPRT(stop_on_reject=False), gaussian("x", mean=0.3, sd=1.0, seed=3,
                                                                           n=1000)).outputs
        ax = plotting.plot_confidence_sequence(outputs)
        lo, hi = ax.get_ylim()
        widest = max(o.upper - o.lower for o in outputs if o.lower is not None)
        self.assertLess(hi - lo, widest / 3)
        self.assertLess(lo, outputs[-1].lower)
        self.assertGreater(hi, outputs[-1].upper)

    def test_estimate_false_leaves_the_line_out_and_a_str_names_it(self):
        outputs = recorded(TMixtureSPRT(stop_on_reject=False), gaussian("x", mean=0.3, sd=1.0, seed=3,
                                                                           n=50)).outputs
        ax = plotting.plot_confidence_sequence(outputs, estimate=False)
        self.assertEqual(self.legend(ax), ["confidence sequence"])
        ax = plotting.plot_confidence_sequence(outputs, estimate="sd")
        self.assertEqual(self.lines(ax, "estimate (sd)")[0].get_ydata()[-1], outputs[-1].sd)

    def test_outputs_without_bounds_are_refused(self):
        outputs = recorded(CUSUM(Gaussian(1.0), 0.0, 1.0, 5.0), from_values("x", [0.0])).outputs
        with self.assertRaisesRegex(ValueError, "not a confidence sequence"):
            plotting.plot_confidence_sequence(outputs)


class GroupSequential(Plots):
    def setUp(self):
        super().setUp()
        self.design = group_sequential_design(0.025, [0.25, 0.5, 0.75, 1.0], futility="obrien-fleming", power=0.9)

    def test_boundaries_against_information_fraction_with_the_observed_path(self):
        n_max = self.design.max_sample_size(effect=0.3, sigma=1.0)
        trial = recorded(GroupSequentialTest.from_design(self.design, n_max, sigma=1.0),
                         gaussian("x", mean=0.3, sd=1.0, seed=11))
        ax = plotting.plot_group_sequential_design(self.design, observed=trial)
        efficacy = self.lines(ax, "efficacy boundary")[0]
        self.assertEqual(list(efficacy.get_xdata()), list(self.design.fractions))
        self.assertTrue(same(efficacy.get_ydata(), self.design.bounds))
        self.assertTrue(same(self.lines(ax, "futility boundary")[0].get_ydata(), self.design.futility))
        analyses = [o for o in trial.outputs if o.analysis]
        observed = self.lines(ax, "observed")[0]
        self.assertTrue(same(observed.get_ydata(), [o.z for o in analyses]))
        self.assertEqual(list(observed.get_xdata()), list(self.design.fractions[:len(analyses)]))
        x, y, label = self.red_marker(ax)
        self.assertEqual((x, y), (self.design.fractions[len(analyses) - 1], analyses[-1].z))
        self.assertEqual(label, f"{analyses[-1].decision} at {x:.0%}")
        self.assertEqual(ax.get_xlabel(), "information fraction")
        self.assertEqual(ax.get_ylabel(), "Z statistic")
        self.assertEqual(len([c for c in ax.collections if isinstance(c, PolyCollection)]), 2)  # stopping regions

    def test_score_and_p_value_scales(self):
        ts = self.design.fractions
        ax = plotting.plot_group_sequential_design(self.design, scale="score", observed=[1.0, 2.0])
        self.assertTrue(same(self.lines(ax, "efficacy boundary")[0].get_ydata(),
                             [b * math.sqrt(t) for b, t in zip(self.design.bounds, ts)]))
        self.assertTrue(same(self.lines(ax, "observed")[0].get_ydata(), [1.0 * math.sqrt(0.25), 2.0 * math.sqrt(0.5)]))
        ax = plotting.plot_group_sequential_design(self.design, scale="p")
        self.assertTrue(same(self.lines(ax, "efficacy boundary")[0].get_ydata(),
                             [NormalDist().cdf(-b) for b in self.design.bounds]))
        self.assertEqual(ax.get_yscale(), "log")
        self.assertTrue(ax.yaxis_inverted())
        self.assertEqual(ax.get_ylabel(), "nominal p-value (one-sided)")

    def test_two_sided_mirrors_the_boundary_and_infinite_bounds_leave_gaps(self):
        design = group_sequential_design(0.05, [0.5, 1.0], spending=lambda t: 0.0 if t < 1 else 0.025,
                                         two_sided=True)
        self.assertEqual(design.bounds[0], math.inf)
        ax = plotting.plot_group_sequential_design(design)
        upper, lower = [line for line in ax.get_lines() if line.get_color() == plotting._BLUE]
        self.assertTrue(math.isnan(upper.get_ydata()[0]))
        self.assertTrue(same(lower.get_ydata(), [-v for v in upper.get_ydata()]))
        ax = plotting.plot_group_sequential_design(design, scale="p")
        self.assertAlmostEqual(self.lines(ax, "efficacy boundary")[0].get_ydata()[1], 0.05, places=6)
        self.assertEqual(ax.get_ylabel(), "nominal p-value (two-sided)")

    def test_bad_arguments(self):
        with self.assertRaisesRegex(ValueError, "scale"):
            plotting.plot_group_sequential_design(self.design, scale="t")
        with self.assertRaisesRegex(ValueError, "5 observed values"):
            plotting.plot_group_sequential_design(self.design, observed=[1.0] * 5)


class Detection(Plots):
    def test_cusum_on_its_own_scale_with_threshold_alarm_and_change(self):
        data = [0.0] * 30 + [1.5] * 30
        outputs = recorded(CUSUM(Gaussian(1.0), 0.0, 1.0, 5.0), from_values("x", data)).outputs
        ax = plotting.plot_detection(outputs, threshold=5.0, change_point=30)
        self.assertTrue(same(self.lines(ax)[0].get_ydata(), [o.statistic for o in outputs]))
        self.assertTrue(self.hline_at(ax, 5.0))
        self.assertIn("threshold 5", self.texts(ax))
        self.assertIn("true change at 30", self.texts(ax))
        x, y, label = self.red_marker(ax)
        self.assertEqual((x, y, label), (outputs[-1].n, outputs[-1].statistic, f"alarm at {outputs[-1].n}"))
        self.assertTrue(outputs[-1].alarm)
        self.assertEqual(ax.get_ylabel(), "CUSUM statistic W")

    def test_shiryaev_roberts_on_a_log_scale(self):
        data = [0.0] * 30 + [1.5] * 30
        outputs = recorded(ShiryaevRoberts(Gaussian(1.0), 0.0, 1.0, 1000.0), from_values("x", data)).outputs
        ax = plotting.plot_detection(outputs, threshold=1000.0)
        self.assertTrue(same(self.lines(ax)[0].get_ydata(), [o.log_statistic / LN10 for o in outputs]))
        self.assertTrue(self.hline_at(ax, 3.0))
        self.assertIn("threshold 1000", self.texts(ax))
        with self.assertRaisesRegex(ValueError, "positive"):
            plotting.plot_detection(outputs, threshold=0.0)

    def test_an_e_detector_draws_its_change_estimate(self):
        rnd = random.Random(4)
        data = [rnd.uniform(0.0, 0.4) for _ in range(200)] + [rnd.uniform(0.3, 1.0) for _ in range(200)]
        outputs = recorded(MeanEDetector(m=0.2, arl=1000, direction="increase", statistic="cusum"),
                           from_values("x", data)).outputs
        alarm = outputs[-1]
        self.assertTrue(alarm.alarm)
        ax = plotting.plot_detection(outputs, threshold=1000, change_point=200)
        self.assertTrue(same(self.lines(ax)[0].get_ydata(), [math.log10(o.statistic) for o in outputs]))
        self.assertIn(f"estimated change at {alarm.change_estimate}", self.texts(ax))
        self.assertTrue(any(list(line.get_xdata()) == [alarm.change_estimate] * 2 for line in ax.get_lines()))
        self.assertEqual(self.legend(ax), [f"alarm at {alarm.n}"])

    def test_bayesian_change_points_as_a_probability(self):
        data = [0.0] * 40 + [3.0] * 20
        prior = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        outputs = recorded(BayesianChangePoint(prior, hazard=0.01, alarm_probability=0.9, alarm_window=10),
                           from_values("x", data)).outputs
        ax = plotting.plot_detection(outputs, threshold=0.9)
        self.assertTrue(same(self.lines(ax)[0].get_ydata(), [o.recent_change_probability for o in outputs]))
        self.assertEqual(ax.get_ylabel(), "P(change within the alarm window)")
        first = next(o for o in outputs if o.decision)
        self.assertEqual(self.red_marker(ax)[0], first.n)


class Filters(Plots):
    def test_a_scalar_filter_with_its_band_observations_and_truth(self):
        rnd = random.Random(5)
        ys = [1.0 + rnd.gauss(0.0, 0.5) for _ in range(100)]
        outputs = recorded(LocalLevelKalman(q=0.01, r=0.25), from_values("y", ys)).outputs
        ax = plotting.plot_filter(outputs, observations=ys, truth=1.0)
        mean = self.lines(ax, "filtered mean ± 2 sd")[0]
        self.assertTrue(same(mean.get_ydata(), [o.mean for o in outputs]))
        band = next(c for c in ax.collections if isinstance(c, PolyCollection))
        vertices = band.get_paths()[0].vertices
        top = max(vertices[:, 1])
        self.assertAlmostEqual(top, max(o.mean + 2 * math.sqrt(o.var) for o in outputs))
        scatter = next(c for c in ax.collections if c.get_label() == "observations")
        self.assertTrue(same(scatter.get_offsets()[:, 1], ys))
        self.assertIn("true state", self.legend(ax))

    def test_a_vector_filter_component_and_the_smoother(self):
        rnd = random.Random(5)
        level = [0.5 + 0.01 * t for t in range(80)]
        readings = [v + rnd.gauss(0.0, 0.1) for v in level]
        model = KalmanFilter.local_linear_trend(level_noise=1e-5, slope_noise=1e-6, observation_noise=0.01)
        outputs = recorded(model, from_values("y", readings)).outputs
        smoothed = kalman_smoother(model, [{"y": y} for y in readings])
        ax = plotting.plot_filter(outputs, truth=level, smoothed=smoothed)
        self.assertTrue(same(self.lines(ax, "smoothed mean ± 2 sd")[0].get_ydata(), [m[0] for m in smoothed.means]))
        self.assertTrue(same(self.lines(ax, "true state")[0].get_ydata(), level))
        lo, hi = ax.get_ylim()
        self.assertLess(hi - lo, 10.0)  # the vague prior (variance 1e6) does not set the range
        ax = plotting.plot_filter(outputs, component=1)
        self.assertTrue(same(self.lines(ax, "filtered mean ± 2 sd")[0].get_ydata(), [o.mean[1] for o in outputs]))
        self.assertEqual(ax.get_ylabel(), "state component 1")

    def test_outputs_of_other_procedures_are_refused(self):
        outputs = recorded(CUSUM(Gaussian(1.0), 0.0, 1.0, 5.0), from_values("x", [0.0])).outputs
        with self.assertRaisesRegex(ValueError, "not a filter output"):
            plotting.plot_filter(outputs)


class StoppingTimes(Plots):
    def setUp(self):
        super().setUp()
        self.plan = simulate(lambda: TMixtureSPRT(effect=0.5), lambda rnd: {"x": rnd.gauss(0.2, 1.0)},
                             runs=60, max_n=200)

    def test_the_share_of_all_runs_stopped_ends_at_the_power(self):
        self.assertLess(self.plan.stopped, self.plan.runs)
        ax = plotting.plot_stopping_times(self.plan)
        curve = self.lines(ax)[0]
        self.assertEqual(curve.get_xdata()[-1], 200)
        self.assertAlmostEqual(curve.get_ydata()[-1], self.plan.power)
        self.assertEqual(curve.get_ydata()[0], 0.0)
        self.assertTrue(self.hline_at(ax, self.plan.power))
        self.assertIn(f"stopped within 200: {self.plan.power:.0%}", self.texts(ax))
        x, _, label = self.red_marker(ax)
        self.assertEqual(x, self.plan.median_n)
        self.assertEqual(label, f"median {self.plan.median_n:.0f}")

    def test_a_list_of_stopping_times_and_a_histogram(self):
        ax = plotting.plot_stopping_times([5, 1, 3, 3])
        curve = self.lines(ax)[0]
        self.assertEqual(list(curve.get_xdata()), [0, 1, 3, 3, 5, 5])
        self.assertEqual(list(curve.get_ydata()), [0, 0.25, 0.5, 0.75, 1.0, 1.0])
        ax = plotting.plot_stopping_times(self.plan, kind="hist")
        self.assertEqual(sum(p.get_height() for p in ax.patches), self.plan.stopped)
        self.assertEqual(ax.get_ylabel(), "runs")
        with self.assertRaisesRegex(ValueError, "kind"):
            plotting.plot_stopping_times(self.plan, kind="pie")
        with self.assertRaisesRegex(ValueError, "no stopping times"):
            plotting.plot_stopping_times([])


@unittest.skipUnless((ROOT / "docs" / "img" / "plots").is_dir(), "the documentation's pictures are not here (sdist)")
class DocumentationPictures(unittest.TestCase):
    def test_every_picture_the_page_shows_exists(self):
        text = (ROOT / "docs" / "plotting.md").read_text(encoding="utf-8")
        pictures = re.findall(r"!\[[^\]]*\]\((img/plots/[^)\s]+\.png)\)", text)
        self.assertGreaterEqual(len(pictures), 6)
        for picture in pictures:
            self.assertTrue((ROOT / "docs" / picture).is_file(), f"{picture}: run python tools/make_plots.py")


if __name__ == "__main__":
    unittest.main()
