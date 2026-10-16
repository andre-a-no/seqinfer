# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Shared fixtures: procedures, observation histories and trace helpers."""
import contextlib
import json
import logging
import math
import random

from sequential_analysis import (
    Chain,
    Delivery,
    Independent,
    KeyJoin,
    Observation,
    PositionalPair,
    Recorder,
    Run,
    TimeAlign,
    difference,
)
from sequential_analysis.procedures import (
    CUSUM,
    EMA,
    SPRT,
    ABTest,
    AnytimeLinearModel,
    AnytimeLogRankTest,
    AsymptoticConfidenceSequence,
    AutoRegressive,
    BayesianChangePoint,
    BettingConfidenceSequence,
    BettingMeanTest,
    BootstrapParticleFilter,
    BoundedABTest,
    CatoniConfidenceSequence,
    ConformalForecast,
    ExponentialSmoothing,
    ForecastComparison,
    Gaussian,
    GaussianEmission,
    GroupSequentialLogRank,
    GroupSequentialMonitor,
    GroupSequentialProportions,
    GroupSequentialTest,
    GroupSequentialTwoMeans,
    HiddenMarkovFilter,
    KalmanFilter,
    KernelTwoSampleTest,
    LocalLevelKalman,
    MeanDifference,
    MeanEDetector,
    ModelAveraging,
    MultipleTests,
    MultivariateGaussianEmission,
    NormalGammaModel,
    NormalMixtureSPRT,
    OnlineFDR,
    OnlineHMM,
    ParticleFilter,
    PlanTest,
    Poisson,
    QuantileConfidenceSequence,
    QuantileEDetector,
    QuantileTest,
    RateRatioTest,
    RatioABTest,
    RobustKalmanFilter,
    SampleRatioMismatch,
    SequentialSignTest,
    ShiryaevRoberts,
    StochasticVolatilityModel,
    StudentTEmission,
    TMixtureSPRT,
    TwoProportionTest,
    TwoSPRT,
)
from sequential_analysis.sources import from_values, gaussian, interleave


@contextlib.contextmanager
def no_logs(logger: str, level: str = "WARNING"):
    """unittest's assertNoLogs, which needs Python 3.10: fail if `logger` emits a record at `level` or above."""
    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler, log = Catch(getattr(logging, level)), logging.getLogger(logger)
    log.addHandler(handler)
    try:
        yield
    finally:
        log.removeHandler(handler)
    if records:
        raise AssertionError(f"unexpected log records: {[r.getMessage() for r in records]}")


def fixed_clock():
    return 0.0


def summary(run: Run, recorder: Recorder) -> dict:
    """Everything that defines the inferential trajectory of a run, JSON-normalized."""
    p = run.procedure
    return json.loads(
        json.dumps(
            {
                "t": run.t,
                "state": p.encode_state(run.state),
                "rng": run.rng_state,
                "terminal": run.terminal,
                "digest": run.history_digest,
                "outputs": [p.encode_output(o) for o in recorder.outputs],
            }
        )
    )


def reference(procedure, pairs) -> list:
    return json.loads(json.dumps([procedure.encode_output(o) for _, o in pairs]))


def unbalanced_pairs(n: int, lead: int, seed: int):
    """Two sources where `a` runs `lead` observations ahead of `b`."""
    a = list(gaussian("a", 0.4, 1.0, seed=seed, n=n))
    b = list(gaussian("b", 0.0, 1.0, seed=seed, n=n))
    out = a[:lead]
    for i in range(n - lead):
        out += [a[lead + i], b[i]]
    return out + b[n - lead:]


def jittered_times(n: int, seed: int):
    rnd = random.Random(seed)
    ta = [float(i) for i in range(n)]
    tb = [i + rnd.uniform(-0.6, 0.6) for i in range(n)]
    tb.sort()
    return ta, tb


def ab_units(n: int, seed: int, arms: int = 2):
    """An A/B event stream: per unit, its arm and covariate at assignment, its outcome three units later."""
    rnd = random.Random(seed)
    units = []
    for _ in range(n):
        arm = rnd.randrange(arms)
        x = rnd.gauss(0.0, 1.0)
        y = int(rnd.random() < 0.2 + 0.1 * arm + 0.05 * (x > 0))
        units.append((arm, x, y))
    out = []
    for i in range(n + 3):
        if i < n:
            out += [Observation("arm", units[i][0], seq=i, key=i), Observation("x", units[i][1], seq=i, key=i)]
        if i >= 3:
            out.append(Observation("y", units[i - 3][2], seq=i - 3, key=i - 3))
    return out


def record_stream(rows, names):
    """Rows of a table as observations: one per named column, keyed by the row number, the outcome late."""
    out = []
    for i, row in enumerate(rows):
        out += [Observation(name, row[name], seq=i, key=i) for name in names]
    return out


def rate_rows(n: int, seed: int):
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        arm = rnd.randrange(2)
        e = round(rnd.uniform(0.5, 2.0), 3)
        mean = (2.0 if arm else 1.5) * e
        k, p, limit = 0, rnd.random(), math.exp(-mean)
        while p > limit:
            k += 1
            p *= rnd.random()
        rows.append({"arm": arm, "count": k, "exposure": e})
    return rows


def chain_ema_cusum():
    return Chain(
        EMA(0.2),
        CUSUM(Gaussian(1.0), 0.0, 1.0, threshold=6.0),
        link=lambda o: {"x": o.residual},
        link_name="residual",
    )


def cases():
    """(label, procedure factory, topology factory, delivery factory, observations, seed)."""
    n = 80
    ta, tb = jittered_times(n, 5)
    level = list(gaussian("y", lambda i: 0.05 * i, 1.0, seed=3, n=n))
    shift = list(gaussian("x", lambda i: 0.0 if i < 40 else 1.5, 1.0, seed=4, n=n))
    return [
        (
            "two_sprt",
            lambda: TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.001, 0.001),
            Independent,
            Delivery,
            list(gaussian("x", 0.25, 1.0, seed=1, n=n)),
            None,
        ),
        ("kalman", lambda: LocalLevelKalman(q=0.01, r=1.0), Independent, Delivery, level, None),
        (
            "particle_filter",
            lambda: BootstrapParticleFilter(q=0.01, r=1.0, particles=64),
            Independent,
            Delivery,
            level,
            11,
        ),
        (
            "independent_two_sample",
            MeanDifference,
            Independent,
            Delivery,
            list(interleave(gaussian("a", 1.0, 1.0, seed=2, n=n), gaussian("b", 0.0, 2.0, seed=2, n=n // 2))),
            None,
        ),
        (
            "paired_difference_sprt",
            lambda: SPRT(Gaussian(1.5), 0.0, 0.5, 1e-4, 1e-4),
            lambda: PositionalPair(("a", "b")).map(difference("a", "b"), "a-b"),
            Delivery,
            unbalanced_pairs(n, lead=5, seed=6),
            None,
        ),
        (
            "time_aligned_two_sample",
            MeanDifference,
            lambda: TimeAlign(("a", "b"), tolerance=0.3),
            Delivery,
            list(
                interleave(
                    from_values("a", [0.1 * i for i in range(n)], times=ta),
                    from_values("b", [0.2 * i for i in range(n)], times=tb),
                )
            ),
            None,
        ),
        ("chain_ema_cusum", chain_ema_cusum, Independent, Delivery, shift, None),
        (
            "plan_test",
            lambda: PlanTest(TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.001, 0.001).as_plan(), label="2-SPRT"),
            Independent,
            Delivery,
            list(gaussian("x", 0.25, 1.0, seed=12, n=n)),
            None,
        ),
        (
            "mixture_sprt_confidence_sequence",
            lambda: NormalMixtureSPRT(1.0, 0.5, stop_on_reject=False),
            Independent,
            Delivery,
            list(gaussian("x", 0.3, 1.0, seed=13, n=n)),
            None,
        ),
        ("shiryaev_roberts", lambda: ShiryaevRoberts(Gaussian(1.0), 0.0, 1.5, 1e4), Independent, Delivery, shift, None),
        (
            "betting_mean_bounded",
            lambda: BettingMeanTest(0.5, 0.0, 10.0, alpha=1e-6),
            Independent,
            Delivery,
            list(from_values("x", [(i * 7919 % 101) / 10.1 for i in range(n)])),
            None,
        ),
        (
            "paired_sign_test",
            lambda: SequentialSignTest(0.0, alpha=1e-6),
            lambda: PositionalPair(("a", "b")).map(difference("a", "b"), "a-b"),
            Delivery,
            unbalanced_pairs(n, lead=5, seed=14),
            None,
        ),
        (
            "group_sequential",
            lambda: GroupSequentialTest(1.0, analyses=[20, 40, 60, 80], bounds=[4.3, 3.0, 2.5, 2.2]),
            Independent,
            Delivery,
            list(gaussian("x", 0.3, 1.0, seed=24, n=n)),
            None,
        ),
        (
            "t_mixture_unknown_variance",
            lambda: TMixtureSPRT(0.0, 0.5, stop_on_reject=False),
            Independent,
            Delivery,
            list(gaussian("x", 0.4, 3.0, seed=32, n=n)),
            None,
        ),
        (
            "poisson_sprt",
            lambda: SPRT(Poisson(), 2.0, 2.5, 1e-4, 1e-4),
            Independent,
            Delivery,
            list(from_values("x", [(i * 37) % 5 for i in range(n)])),
            None,
        ),
        (
            "quantile_confidence_sequence_heavy_tails",
            lambda: QuantileConfidenceSequence(0.5, tuned_for=50),
            Independent,
            Delivery,
            list(from_values("x", [math.tan(math.pi * ((i * 7919 % 101) / 101.0 - 0.495)) for i in range(n)])),
            None,
        ),
        (
            "quantile_test_with_ties",
            lambda: QuantileTest(1.0, p=0.5, alpha=1e-6),
            Independent,
            Delivery,
            list(from_values("x", [float(i * 37 % 3) for i in range(n)])),
            None,
        ),
        (
            "kernel_two_sample",
            lambda: KernelTwoSampleTest(alpha=1e-6, burn_in=10),
            lambda: PositionalPair(("a", "b")),
            Delivery,
            unbalanced_pairs(n, lead=5, seed=15),
            None,
        ),
        (
            "catoni_mean",
            lambda: CatoniConfidenceSequence(2.0, mean0=0.0),
            Independent,
            Delivery,
            list(gaussian("x", 0.4, 2.0, seed=33, n=n)),
            None,
        ),
        (
            "asymptotic_mean",
            lambda: AsymptoticConfidenceSequence(start=10, mean0=0.0),
            Independent,
            Delivery,
            list(gaussian("x", 0.4, 2.0, seed=34, n=n)),
            None,
        ),
        (
            "kalman_filter_tracking",
            lambda: KalmanFilter(((1.0, 1.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0)), ((0.01, 0.0), (0.0, 0.01)),
                                 ((1.0, 0.0), (0.0, 0.5)), (0.0, 0.0), ((10.0, 0.0), (0.0, 10.0))),
            Independent,
            Delivery,
            list(from_values("y", [[0.5 * i + math.sin(i), 0.5 + 0.1 * math.cos(i)] for i in range(n)])),
            None,
        ),
        (
            "robust_kalman_with_outliers",
            lambda: RobustKalmanFilter.local_linear_trend(0.01, 0.001, 1.0, df=3.0),
            Independent,
            Delivery,
            list(from_values("y", [0.2 * i + (30.0 if i % 17 == 5 else math.sin(i)) for i in range(n)])),
            None,
        ),
        (
            "hidden_markov_regimes",
            lambda: HiddenMarkovFilter(
                (0.5, 0.5), ((0.95, 0.05), (0.1, 0.9)), GaussianEmission((0.0, 2.0), (1.0, 1.5))
            ),
            Independent,
            Delivery,
            list(gaussian("y", lambda i: 0.0 if (i // 20) % 2 == 0 else 2.0, 1.0, seed=35, n=n)),
            None,
        ),
        (
            "bayesian_change_point",
            lambda: BayesianChangePoint(NormalGammaModel(), 0.02, alarm_probability=0.99, alarm_window=10),
            Independent,
            Delivery,
            shift,
            None,
        ),
        (
            "particle_filter_stochastic_volatility",
            lambda: ParticleFilter(StochasticVolatilityModel(-1.0, 0.95, 0.3), 64),
            Independent,
            Delivery,
            list(gaussian("y", 0.0, 0.6, seed=36, n=n)),
            17,
        ),
        (
            "ab_test_with_covariate",
            lambda: ABTest(covariate=True, start=20, tuned_for=50),
            lambda: KeyJoin(("arm", "y", "x")),
            Delivery,
            ab_units(n, seed=37),
            None,
        ),
        (
            "multiple_tests_versus_control",
            lambda: MultipleTests({**MultipleTests.versus_control([1, 2], 0, alpha=0.2).tests_by_name(),
                                   "srm": SampleRatioMismatch((1, 1, 1), 0.01)}, alpha=0.2),
            lambda: KeyJoin(("arm", "y")),
            Delivery,
            [o for o in ab_units(n, seed=38, arms=3) if o.input != "x"],
            None,
        ),
        (
            "two_proportion_one_sided",
            lambda: TwoProportionTest(0.01, alternative="greater"),
            lambda: KeyJoin(("arm", "y")),
            Delivery,
            [o for o in ab_units(n, seed=39) if o.input != "x"],
            None,
        ),
        (
            "mean_e_detector",
            lambda: MeanEDetector(0.3, 0.0, 1.0, arl=1e4, statistic="cusum"),
            Independent,
            Delivery,
            list(from_values("x", [((i * 7919) % 101) / 101 * (0.6 if i < 40 else 1.0) for i in range(n)])),
            None,
        ),
        (
            "quantile_e_detector_heavy_tails",
            lambda: QuantileEDetector(0.0, arl=1e5),
            Independent,
            Delivery,
            shift,
            None,
        ),
        (
            "bounded_ab_test",
            lambda: BoundedABTest(0.0, 1.0, 0.5, grid=40, backend="python"),
            lambda: KeyJoin(("arm", "y")),
            Delivery,
            record_stream([{"arm": i % 2, "y": ((i * 7919) % 101) / 101} for i in range(n)], ("arm", "y")),
            None,
        ),
        (
            "rate_ratio_test",
            lambda: RateRatioTest(grid=30),
            lambda: KeyJoin(("arm", "count", "exposure")),
            Delivery,
            record_stream(rate_rows(n, 41), ("arm", "count", "exposure")),
            None,
        ),
        (
            "anytime_linear_model",
            lambda: AnytimeLinearModel(["x"], ["z"]),
            lambda: KeyJoin(("x", "z", "y")),
            Delivery,
            record_stream([{"x": math.sin(i), "z": float(i % 2), "y": 0.5 * math.sin(i) + 0.3 * (i % 2)
                            + math.cos(3.1 * i)} for i in range(n)], ("x", "z", "y")),
            None,
        ),
        (
            "anytime_logrank_test",
            lambda: AnytimeLogRankTest(cs_grid=21),
            lambda: KeyJoin(("arm", "kind")),
            Delivery,
            # 40 patients enter, then 40 events and censorings, at most 20 per arm
            record_stream([{"arm": i % 2, "kind": 0} for i in range(40)]
                          + [{"arm": (i // 3) % 2 if i < 36 else i % 2, "kind": 1 if i % 4 else 2} for i in range(40)],
                          ("arm", "kind")),
            None,
        ),
        (
            "group_sequential_monitor",
            lambda: GroupSequentialMonitor(0.025, 100.0, "pocock", max_analyses=6),
            lambda: KeyJoin(("z", "information")),
            Delivery,
            # many small analyses: the trial continues to the sixth, which is final
            record_stream([{"z": 0.3 * math.sin(i), "information": 2.0 + 3.0 * i} for i in range(6)],
                          ("z", "information")),
            None,
        ),
        (
            "group_sequential_proportions",
            lambda: GroupSequentialProportions([20, 40, 60, 80], [4.0, 3.0, 2.6, 2.2], futility=[-1.0, 0.0, 1.0, 2.2]),
            lambda: KeyJoin(("arm", "y")),
            Delivery,
            record_stream([{"arm": i % 3, "y": int((i * 7) % 5 < 2 + (i % 3 == 1))} for i in range(n)], ("arm", "y")),
            None,
        ),
        (
            "group_sequential_two_means",
            lambda: GroupSequentialTwoMeans(None, [10, 20, 30], [3.6, 2.6, 2.1]),
            lambda: KeyJoin(("arm", "y")),
            Delivery,
            record_stream([{"arm": i % 3, "y": math.sin(1.7 * i) + 0.3 * (i % 3 == 1)} for i in range(n)],
                          ("arm", "y")),
            None,
        ),
        (
            "group_sequential_logrank",
            lambda: GroupSequentialLogRank([10, 20, 30], [3.5, 2.7, 2.2], strata=2),
            lambda: KeyJoin(("arm", "kind", "stratum")),
            Delivery,
            # 40 patients enter, 20 per stratum and 10 per arm in each, then events and censorings
            record_stream([{"arm": i % 2, "kind": 0, "stratum": (i // 2) % 2} for i in range(40)]
                          + [{"arm": i % 2, "kind": 1 if i % 5 else 2, "stratum": (i // 2) % 2} for i in range(40)],
                          ("arm", "kind", "stratum")),
            None,
        ),
        (
            "anytime_logrank_test_stratified",
            lambda: AnytimeLogRankTest(cs_grid=21, strata=2),
            lambda: KeyJoin(("arm", "kind", "stratum")),
            Delivery,
            # 40 patients enter, 20 per stratum and 10 per arm in each, then 40 events and censorings
            record_stream([{"arm": i % 2, "kind": 0, "stratum": (i // 2) % 2} for i in range(40)]
                          + [{"arm": (i // 3) % 2 if i < 36 else i % 2, "kind": 1 if i % 4 else 2,
                              "stratum": (i // 2) % 2} for i in range(40)],
                          ("arm", "kind", "stratum")),
            None,
        ),
        (
            "ratio_ab_test_relative",
            lambda: RatioABTest(start=20, tuned_for=50, scale="relative"),
            lambda: KeyJoin(("arm", "y", "d")),
            Delivery,
            record_stream([{"arm": 2 if i % 11 == 5 else i % 2, "d": float(i % 4),
                            "y": (i % 4) * (1.0 + ((i * 7919) % 101) / 50 + 0.4 * (i % 2))} for i in range(n)],
                          ("arm", "y", "d")),
            None,
        ),
        (
            "hidden_markov_multivariate",
            lambda: HiddenMarkovFilter((0.5, 0.5), ((0.95, 0.05), (0.1, 0.9)), MultivariateGaussianEmission(
                ((0.0, 0.0), (2.0, 0.0)), (((1.0, 0.2), (0.2, 0.6)), ((1.5, -0.3), (-0.3, 0.8))))),
            Independent,
            Delivery,
            list(from_values("y", [[math.sin(0.7 * i) + 2.0 * ((i // 20) % 2), math.cos(1.1 * i)] for i in range(n)])),
            None,
        ),
        (
            "hidden_markov_student_t",
            lambda: HiddenMarkovFilter((0.5, 0.5), ((0.95, 0.05), (0.1, 0.9)),
                                       StudentTEmission((0.0, 2.0), (1.0, 1.5), 4.0)),
            Independent,
            Delivery,
            list(from_values("y", [(0.0 if (i // 20) % 2 == 0 else 2.0) + math.sin(1.7 * i)
                                   + (40.0 if i % 23 == 7 else 0.0) for i in range(n)])),
            None,
        ),
        (
            "online_hmm",
            lambda: OnlineHMM(HiddenMarkovFilter([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]],
                                                 GaussianEmission([0.0, 1.0], [1.0, 2.0])), burn_in=15),
            Independent,
            Delivery,
            list(from_values("y", [math.sin(1.3 * i) * (1 + 2 * ((i // 10) % 2)) for i in range(n)])),
            None,
        ),
        (
            "exponential_smoothing",
            lambda: ExponentialSmoothing(0.3, beta=0.05, gamma=0.1, phi=0.9, trend="damped", period=5, horizon=3),
            Independent,
            Delivery,
            list(from_values("y", [10.0 + 0.1 * i + 2.0 * math.sin(1.25 * i) + math.cos(2.9 * i) for i in range(n)])),
            None,
        ),
        (
            "autoregressive",
            lambda: AutoRegressive(2, forgetting=0.98, horizon=2),
            Independent,
            Delivery,
            list(from_values("y", [math.sin(0.4 * i) + 0.3 * math.cos(2.3 * i) for i in range(n)])),
            None,
        ),
        (
            "model_averaging",
            lambda: ModelAveraging([ExponentialSmoothing(0.2, horizon=2), ExponentialSmoothing(0.7, horizon=2),
                                    AutoRegressive(1, horizon=2)], forgetting=0.95),
            Independent,
            Delivery,
            list(from_values("y", [0.05 * i + math.sin(0.9 * i) for i in range(n)])),
            None,
        ),
        (
            "conformal_forecast",
            lambda: ConformalForecast(AutoRegressive(1, coefficients=(0.6,), intercept=0.2, sigma=0.5, horizon=2)),
            Independent,
            Delivery,
            list(from_values("y", [math.sin(0.7 * i) + (3.0 if i % 17 == 4 else 0.0) for i in range(n)])),
            None,
        ),
        (
            "forecast_comparison",
            lambda: ForecastComparison((0.0, 2.0)),
            lambda: KeyJoin(("a", "b")),
            Delivery,
            record_stream([{"a": ((i * 37) % 19) / 10.0, "b": ((i * 11) % 13) / 8.0} for i in range(n)], ("a", "b")),
            None,
        ),
        (
            "online_fdr",
            lambda: OnlineFDR(0.1),
            Independent,
            Delivery,
            list(from_values("log_e", [((i * 37) % 23) / 2.0 - 3.0 for i in range(n)])),
            None,
        ),
        (
            "betting_confidence_sequence",
            lambda: BettingConfidenceSequence(0.0, 10.0, grid=50),
            Independent,
            Delivery,
            list(from_values("x", [(i * 7919 % 101) / 10.1 for i in range(n)])),
            None,
        ),
    ]
