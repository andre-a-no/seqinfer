# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Write the conformance vectors in conformance/.

An implementation of sequential_analysis in any language conforms when, for every
case, folding its procedure over `inputs` from the default initial state
(with `seed` for randomized procedures) reproduces `outputs`, `state`,
`rng` and `history_digest` exactly.  Floats are compared through their
RFC 8785 encoding, i.e. bit for bit.

    python tools/make_conformance.py          # rewrite the vectors
    python -m unittest tests.test_conformance # check this implementation

The vectors change only when a procedure's version does.
"""
from __future__ import annotations

import math
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sequential_analysis import Run  # noqa: E402
from sequential_analysis.canonical import _number, canonical_json  # noqa: E402
from sequential_analysis.group_sequential import group_sequential_design  # noqa: E402
from sequential_analysis.procedures import (  # noqa: E402
    CUSUM,
    EMA,
    SPRT,
    ABTest,
    AnytimeLinearModel,
    AnytimeLogRankTest,
    AsymptoticConfidenceSequence,
    BayesianChangePoint,
    Bernoulli,
    BetaBernoulliModel,
    BettingConfidenceSequence,
    BettingMeanTest,
    BootstrapParticleFilter,
    BoundedABTest,
    CategoricalEmission,
    CatoniConfidenceSequence,
    Exponential,
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
    LinearGaussianModel,
    LocalLevelKalman,
    MeanDifference,
    MeanEDetector,
    MultipleTests,
    NormalGammaModel,
    NormalMixtureSPRT,
    OnlineFDR,
    OnlineHMM,
    ParticleFilter,
    PlanTest,
    Poisson,
    PoissonEmission,
    PoissonGammaModel,
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
    StudentTLocalLevel,
    TMixtureSPRT,
    TwoProportionTest,
    TwoSPRT,
)

OUT = ROOT / "conformance"


def cases():
    rnd = random.Random(20261009)

    def normal(mu, n, sd=1.0):
        return [{"x": rnd.gauss(mu, sd)} for _ in range(n)]

    def poisson(lam, n):
        out = []
        for _ in range(n):
            k, prod, limit = 0, rnd.random(), 2.718281828459045 ** -lam
            while prod > limit:
                k += 1
                prod *= rnd.random()
            out.append({"x": k})
        return out

    def bernoulli(p, n):
        return [{"x": int(rnd.random() < p)} for _ in range(n)]

    two_sprt = TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.05, 0.05)
    level = [{"y": 0.05 * i + rnd.gauss(0.0, 1.0)} for i in range(40)]
    existing = [
        ("sprt_gaussian", SPRT(Gaussian(1.0), 0.0, 0.5, 0.05, 0.05), None, normal(0.25, 60)),
        ("sprt_bernoulli", SPRT(Bernoulli(), 0.3, 0.5, 0.05, 0.05), None, bernoulli(0.4, 120)),
        ("sprt_poisson", SPRT(Poisson(), 2.0, 3.0, 0.05, 0.05), None, poisson(2.5, 80)),
        ("two_sprt", TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.05, 0.05), None, normal(0.25, 60)),
        ("plan_test", PlanTest(two_sprt.as_plan(), label="2-SPRT"), None, normal(0.25, 60)),
        ("cusum", CUSUM(Gaussian(1.0), 0.0, 1.0, 8.0), None, normal(0.0, 30) + normal(1.0, 40)),
        ("shiryaev_roberts_exponential", ShiryaevRoberts(Exponential(), 1.0, 0.5, 200.0), None,
         [{"x": rnd.expovariate(1.0)} for _ in range(30)] + [{"x": rnd.expovariate(0.5)} for _ in range(60)]),
        ("ema", EMA(0.2), None, normal(1.0, 30)),
        ("kalman_control", LocalLevelKalman(0.01, 1.0, control=True), None,
         [{"y": rnd.gauss(0.1 * i, 1.0), "u": 0.1} for i in range(40)]),
        ("particle_filter", BootstrapParticleFilter(0.01, 1.0, particles=32), 7, level),
        ("mean_difference", MeanDifference(), None,
         [{"a": rnd.gauss(1.0, 1.0)} if i % 3 else {"b": rnd.gauss(0.0, 2.0)} for i in range(45)]),
        ("mixture_sprt", NormalMixtureSPRT(1.0, 0.5, stop_on_reject=False), None, normal(0.3, 80)),
        ("betting_mean", BettingMeanTest(0.3, 0.0, 2.0, alpha=1e-4, stop_on_reject=False), None,
         [{"x": 2.0 * rnd.betavariate(1.0, 2.0)} for _ in range(80)]),
        ("sign_test", SequentialSignTest(0.0, alpha=1e-4, stop_on_reject=False), None,
         [{"x": float(rnd.choice([-2, -1, 0, 1, 2, 3]))} for _ in range(80)]),
        ("group_sequential", GroupSequentialTest(1.0, [20, 40, 60], [3.71, 2.51, 1.99]), None, normal(0.2, 60)),
        ("t_mixture", TMixtureSPRT(1.0, 0.5, stop_on_reject=False), None, normal(1.3, 50, sd=2.0)),
    ]
    # Cases added later draw their data only after all of the above, so the earlier vectors stay as they were.

    def arm_rows(n, rates, weights=(0.45, 0.45, 0.1)):
        rows = []
        for _ in range(n):
            u, arm = rnd.random(), 0
            while arm < len(weights) - 1 and u >= weights[arm]:
                u -= weights[arm]
                arm += 1
            rows.append({"arm": arm, "y": int(rnd.random() < rates[arm])})
        return rows

    def srm_rows(n, split):
        rows = []
        for _ in range(n):
            u, arm = rnd.random(), 0
            while arm < len(split) - 1 and u >= split[arm]:
                u -= split[arm]
                arm += 1
            rows.append({"arm": arm})
        return rows

    def ab_rows(n, effect):
        rows = []
        for _ in range(n):
            u = rnd.random()
            arm = 1 if u < 0.45 else 0 if u < 0.9 else 2
            x = rnd.gauss(0.0, 1.0)
            y = 1.0 + 0.8 * x + (effect if arm == 1 else 0.0) + rnd.gauss(0.0, 0.6)
            rows.append({"arm": arm, "y": y, "x": x})
        return rows

    def cauchy(loc, n):
        return [{"x": loc + math.tan(math.pi * (rnd.random() - 0.5))} for _ in range(n)]

    added = [
        ("two_proportion", TwoProportionTest(0.05), None, arm_rows(120, (0.25, 0.65, 0.9))),
        ("two_proportion_greater", TwoProportionTest(0.05, alternative="greater", arms=(2, 0)), None,
         arm_rows(90, (0.5, 0.2, 0.6), weights=(0.4, 0.2, 0.4))),
        ("sample_ratio_mismatch", SampleRatioMismatch((5, 3, 2), alpha=0.001), None, srm_rows(120, (0.42, 0.33, 0.25))),
        ("ab_test_covariate", ABTest(covariate=True, start=20), None, ab_rows(110, 0.7)),
        ("mean_e_detector", MeanEDetector(0.3, 0.0, 1.0, arl=200.0), None,
         [{"x": rnd.betavariate(3.0, 7.0)} for _ in range(50)] + [{"x": rnd.betavariate(6.0, 4.0)} for _ in range(50)]),
        ("quantile_e_detector", QuantileEDetector(0.0, 0.5, arl=200.0, statistic="cusum", direction="increase", grid=8),
         None, cauchy(0.0, 50) + cauchy(1.5, 60)),
        ("group_sequential_t", GroupSequentialTest(None, [10, 20, 30], [3.71, 2.51, 1.99]), None,
         normal(0.4, 30, sd=1.5)),
    ]
    added += [
        ("two_proportion_universal", TwoProportionTest(0.05, method="universal"), None,
         arm_rows(120, (0.25, 0.6, 0.9))),
        ("two_proportion_blocks_2_1", TwoProportionTest(0.05, block=(2, 1)), None,
         arm_rows(120, (0.3, 0.55, 0.9), weights=(0.3, 0.6, 0.1))),
    ]
    # The procedures ported to the TypeScript executor later; their data are drawn after all of the above.

    def pairs(n, sd_b):
        return [{"a": rnd.gauss(0.0, 1.0), "b": rnd.gauss(0.0, sd_b)} for _ in range(n)]

    def level_rows(n, drift, noise, outliers=0.0):
        rows, z = [], 0.0
        for _ in range(n):
            z += drift + rnd.gauss(0.0, 0.1)
            y = z + rnd.gauss(0.0, noise)
            if outliers and rnd.random() < outliers:
                y += rnd.choice([-1.0, 1.0]) * 15.0
            rows.append({"y": y})
        return rows

    def regimes(n, emit, stay=0.9):
        rows, k = [], 0
        for _ in range(n):
            if rnd.random() > stay:
                k = 1 - k
            rows.append({"y": emit(k)})
        return rows

    def sv_rows(n):
        rows, h = [], 0.0
        for _ in range(n):
            h = 0.95 * h + 0.3 * rnd.gauss(0.0, 1.0)
            rows.append({"y": math.exp(h / 2) * rnd.gauss(0.0, 1.0)})
        return rows

    trend = level_rows(80, 0.05, 1.0)
    robust = [{"y": [r["y"], r["y"] + rnd.gauss(0.0, 0.5)], "u": 0.1} for r in level_rows(60, 0.0, 1.0, outliers=0.1)]
    ported = [
        ("quantile_test", QuantileTest(0.0, alpha=0.05, tuned_for=100.0, stop_on_reject=False), None, cauchy(0.8, 100)),
        ("quantile_test_greater_ties",
         QuantileTest(2.0, p=0.3, alternative="greater", tuned_for=50.0, stop_on_reject=False), None, poisson(3.0, 80)),
        ("quantile_confidence_sequence", QuantileConfidenceSequence(0.5, tuned_for=50.0), None, cauchy(1.0, 100)),
        ("kernel_two_sample", KernelTwoSampleTest(0.05, burn_in=10, stop_on_reject=False, backend="python"), None,
         pairs(100, 2.0)),
        ("kernel_two_sample_bandwidth", KernelTwoSampleTest(0.05, bandwidth=1.0, burn_in=0, backend="python"), None,
         pairs(60, 1.0)),
        ("betting_confidence_sequence", BettingConfidenceSequence(0.0, 2.0, grid=50, backend="python"), None,
         [{"x": 2.0 * rnd.betavariate(2.0, 3.0)} for _ in range(80)]),
        ("catoni_confidence_sequence", CatoniConfidenceSequence(2.0, mean0=0.0, backend="python"), None,
         normal(0.5, 60, sd=1.5)),
        ("asymptotic_confidence_sequence", AsymptoticConfidenceSequence(mean0=0.0, start=20, tuned_for=100.0), None,
         normal(0.3, 100)),
        ("kalman_filter_trend", KalmanFilter.local_linear_trend(0.01, 0.001, 1.0), None, trend),
        ("robust_kalman_filter", RobustKalmanFilter(((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (1.0, 1.0)),
                                                    ((0.01, 0.0), (0.0, 0.001)), ((1.0, 0.2), (0.2, 0.5)),
                                                    (0.0, 0.0), ((10.0, 0.0), (0.0, 1.0)), control=((1.0,), (0.0,)),
                                                    df=4.0), None, robust),
        ("hmm_categorical", HiddenMarkovFilter((0.5, 0.5), ((0.9, 0.1), (0.2, 0.8)),
                                               CategoricalEmission(((0.7, 0.2, 0.1), (0.1, 0.3, 0.6)))), None,
         regimes(60, lambda k: int(rnd.random() < (0.3 if k else 0.8)) + (k if rnd.random() < 0.5 else 0))),
        ("hmm_gaussian", HiddenMarkovFilter((0.6, 0.3, 0.1), ((0.9, 0.05, 0.05), (0.1, 0.8, 0.1), (0.0, 0.2, 0.8)),
                                            GaussianEmission((0.0, 2.0, -1.0), (1.0, 0.5, 3.0))), None,
         regimes(80, lambda k: rnd.gauss(2.0 * k, 1.0 if k == 0 else 0.5))),
        ("hmm_poisson", HiddenMarkovFilter((0.5, 0.5), ((0.95, 0.05), (0.1, 0.9)), PoissonEmission((1.0, 6.0))), None,
         [{"y": r["x"]} for r in poisson(1.0, 30) + poisson(6.0, 30)]),
        ("bocpd_normal_gamma", BayesianChangePoint(NormalGammaModel(0.0, 0.1, 2.0, 2.0), 0.02), None,
         normal(0.0, 40) + normal(3.0, 40, sd=0.5)),
        ("bocpd_poisson_gamma", BayesianChangePoint(PoissonGammaModel(2.0, 1.0), 0.05, alarm_probability=0.9,
                                                    max_hypotheses=20), None, poisson(1.0, 40) + poisson(5.0, 40)),
        ("bocpd_beta_bernoulli", BayesianChangePoint(BetaBernoulliModel(1.0, 1.0), 0.02, prune=1e-6), None,
         bernoulli(0.2, 50) + bernoulli(0.8, 50)),
        ("particle_filter_linear_gaussian",
         ParticleFilter(LinearGaussianModel(0.9, 0.1, 1.0), 64, backend="python"), 11,
         [{"y": r["y"]} for r in level_rows(50, 0.0, 1.0)]),
        ("particle_filter_stochastic_volatility",
         ParticleFilter(StochasticVolatilityModel(0.0, 0.95, 0.3), 64, backend="python"), 12, sv_rows(60)),
        ("particle_filter_student_t",
         ParticleFilter(StudentTLocalLevel(0.01, 1.0, 3.0), 64, resample_threshold=1.0, backend="python"), 13,
         level_rows(40, 0.02, 1.0, outliers=0.1)),
        ("multiple_tests", MultipleTests({
            "mean": NormalMixtureSPRT(1.0, 0.5),
            "sign": SequentialSignTest(0.0, alpha=0.01, stop_on_reject=False),
            "t": TMixtureSPRT(1.0, 0.5, stop_on_reject=False),
            "bounded": BettingMeanTest(0.0, -4.0, 4.0, stop_on_reject=False),
        }, alpha=0.1), None, [{"x": max(-4.0, min(4.0, r["x"]))} for r in normal(0.35, 100)]),
    ]
    # The procedures ported after those; their data are drawn after all of the above.

    def bounded_rows(n):
        rows = []
        for _ in range(n):
            u = rnd.random()
            arm = 1 if u < 0.36 else 0 if u < 0.9 else 2
            y = rnd.betavariate(5.0, 3.0) if arm == 1 else rnd.betavariate(3.0, 4.0)
            rows.append({"arm": arm, "y": y})
        return rows

    def poisson_count(mean):
        k, prod, limit = 0, rnd.random(), math.exp(-mean)
        while prod > limit:
            k += 1
            prod *= rnd.random()
        return k

    def rate_rows(n, rates):
        rows = []
        for _ in range(n):
            u = rnd.random()
            arm = 1 if u < 0.45 else 0 if u < 0.92 else 2
            exposure = rnd.uniform(0.5, 3.0)
            rows.append({"arm": arm, "count": poisson_count(rates[arm] * exposure), "exposure": exposure})
        return rows

    def e_value_rows(n, signal):
        return [{"log_e": rnd.uniform(5.0, 12.0) if rnd.random() < signal else rnd.gauss(-0.5, 1.0)} for _ in range(n)]

    def regression_rows(n, two_covariates, tested, effects, constant_until):
        rows = []
        for i in range(n):
            row = {"x1": rnd.gauss(0.0, 1.0)}
            if two_covariates:
                row["x2"] = rnd.uniform(-1.0, 1.0)
            for name in tested:
                row[name] = 0.0 if i < constant_until else float(rnd.random() < 0.5)
            y = 1.0 + 0.5 * row["x1"] - 0.3 * row.get("x2", 0.0) + rnd.gauss(0.0, 1.0)
            for name, effect in zip(tested, effects):
                y += effect * row[name]
            row["y"] = y
            rows.append(row)
        return rows

    def survival_rows(n, theta, start, entry, switch=None, theta_late=None):
        """Entries, events and censorings in the order they happen; no event or censoring in an empty arm.

        The hazard ratio is theta, and theta_late from row `switch` on.
        """
        rows, at_risk = [], [0, 0]  # treatment (arm 1), control (arm 0)
        for i in range(n):
            u = rnd.random()
            if u < 0.05:
                rows.append({"arm": 2, "kind": rnd.randrange(3)})
            elif i < start or u < 0.05 + entry or at_risk[0] + at_risk[1] == 0:
                k = rnd.randrange(2)
                at_risk[k] += 1
                rows.append({"arm": 1 - k, "kind": 0})
            else:
                weight = at_risk[0] * (theta if switch is None or i < switch else theta_late)
                k = 0 if rnd.random() * (weight + at_risk[1]) < weight else 1
                at_risk[k] -= 1
                rows.append({"arm": 1 - k, "kind": 1 if rnd.random() < 0.8 else 2})
        return rows

    later = [
        ("bounded_ab_test", BoundedABTest(0.0, 1.0, allocation=0.4, grid=100, backend="python"), None,
         [{"arm": 2, "y": 0.5}, *bounded_rows(120)]),
        ("rate_ratio_test", RateRatioTest(grid=40), None, rate_rows(150, (1.0, 1.6, 1.0))),
        ("online_fdr", OnlineFDR(0.1), None, e_value_rows(40, 0.25)),
        ("online_fdr_horizon", OnlineFDR(0.1, horizon=20), None, e_value_rows(20, 0.3)),
        ("anytime_linear_model", AnytimeLinearModel(["x1", "x2"], ["z"]), None,
         regression_rows(80, True, ("z",), (0.6,), 6)),
        ("anytime_linear_model_two_tested", AnytimeLinearModel(["x1"], ["z1", "z2"], delta0=[0.0, 0.2], g=0.5), None,
         regression_rows(80, False, ("z1", "z2"), (0.7, -0.2), 0)),
        ("anytime_logrank_test", AnytimeLogRankTest(cs_grid=30), None,
         survival_rows(200, 4.0, 40, 0.45, switch=90, theta_late=0.05)),
        ("anytime_logrank_test_grow", AnytimeLogRankTest(alternative="less", method="grow", confidence_sequence=False),
         None, survival_rows(150, 0.5, 40, 0.3)),
    ]

    def stratified_survival_rows(n, theta, strata, start):
        """Survival records with a stratum each; no event or censoring in an arm of a stratum nobody is at risk in."""
        rows, at_risk = [], [[0, 0] for _ in range(strata)]  # per stratum: treatment (arm 1), control (arm 0)
        for i in range(n):
            s = rnd.randrange(strata)
            r = at_risk[s]
            if i < start or rnd.random() < 0.35 or r[0] + r[1] == 0:
                k = rnd.randrange(2)
                r[k] += 1
                rows.append({"arm": 1 - k, "kind": 0, "stratum": s})
            else:
                weight = r[0] * theta
                k = 0 if rnd.random() * (weight + r[1]) < weight else 1
                r[k] -= 1
                rows.append({"arm": 1 - k, "kind": 1 if rnd.random() < 0.85 else 2, "stratum": s})
        return rows

    def analyses_rows(zs, informations, final_at=None):
        rows = [{"z": z, "information": v} for z, v in zip(zs, informations)]
        if final_at is not None:
            for i, row in enumerate(rows):
                row["final"] = int(i == final_at)
        return rows

    gs_futility = group_sequential_design(0.025, [0.25, 0.5, 0.75, 1.0], "obrien-fleming", futility="pocock",
                                          power=0.85)
    gs_power = group_sequential_design(0.05, [0.2, 0.4, 0.6, 0.8, 1.0], ("power", 2.0), two_sided=True)
    gs_three = group_sequential_design(0.025, [1 / 3, 2 / 3, 1.0], "obrien-fleming")
    round_four = [
        ("group_sequential_monitor_futility", GroupSequentialMonitor.from_design(gs_futility, 300.0,
                                                                                 final_input="final"), None,
         analyses_rows([0.4, 1.2, 1.9, 2.1], [70.0, 155.0, 230.0, 290.0], final_at=3)),
        ("group_sequential_monitor_two_sided", GroupSequentialMonitor.from_design(gs_power, 50.0), None,
         analyses_rows([-0.5, -1.1, -1.7, -2.4, -2.6, -2.0], [8.0, 17.5, 26.0, 33.0, 41.0, 55.0])),
        ("group_sequential_proportions", GroupSequentialProportions.from_design(gs_three, 240, analyses=[70, 170, 250]),
         None, [{"arm": 2 if rnd.random() < 0.05 else i % 2, "y": int(rnd.random() < (0.45 if i % 2 else 0.3))}
                for i in range(260)]),
        ("group_sequential_logrank", GroupSequentialLogRank.from_design(gs_futility, 80, alternative="less"), None,
         survival_rows(260, 0.6, 40, 0.4)),
        ("group_sequential_logrank_stratified", GroupSequentialLogRank([25, 50, 75], [3.7, 2.5, 2.0], strata=3,
                                                                       two_sided=True, alternative="greater"), None,
         stratified_survival_rows(300, 1.5, 3, 45)),
        ("group_sequential_two_means_t", GroupSequentialTwoMeans.from_design(gs_futility, 160, None), None,
         [{"arm": 2 if rnd.random() < 0.05 else i % 2, "y": rnd.gauss(0.6 if i % 2 else 0.0, 1.5)}
          for i in range(170)]),
        ("group_sequential_two_means_known", GroupSequentialTwoMeans(1.5, [30, 60, 90], [3.5, 2.6, 2.0],
                                                                     two_sided=True, alternative="less"), None,
         [{"arm": i % 2, "y": rnd.gauss(-0.4 if i % 2 else 0.0, 1.5)} for i in range(95)]),
    ]
    # The stratified anytime-valid log-rank test; its data are drawn after all of the above.
    stratified_logrank = [
        ("anytime_logrank_test_stratified", AnytimeLogRankTest(cs_grid=30, strata=3), None,
         [{"arm": 2, "kind": 1, "stratum": 1}, *stratified_survival_rows(220, 0.35, 3, 45)]),
        ("anytime_logrank_test_stratified_grow",
         AnytimeLogRankTest(alternative="less", method="grow", confidence_sequence=False, strata=2,
                            stratum_input="centre"), None,
         [{"arm": r["arm"], "kind": r["kind"], "centre": r["stratum"]}
          for r in stratified_survival_rows(160, 0.5, 2, 30)]),
    ]

    def ratio_rows(n, lift):
        rows = []
        for _ in range(n):
            u = rnd.random()
            arm = 1 if u < 0.4 else 0 if u < 0.9 else 2
            k = rnd.randrange(4)
            level = math.exp(rnd.gauss(0.0, 0.8)) * (1.0 + lift if arm == 1 else 1.0)
            y = 0.0
            for _ in range(k):
                y += level * rnd.expovariate(1.0)
            rows.append({"arm": arm, "y": y, "d": float(k)})
        return rows

    ratio = [
        ("ratio_ab_test", RatioABTest(start=30, tuned_for=50), None, ratio_rows(150, 1.0)),
        ("ratio_ab_test_relative", RatioABTest(scale="relative", effect0=0.1, start=30, tuned_for=50), None,
         ratio_rows(150, 0.6)),
        ("ratio_ab_test_means", RatioABTest(denominator=False, start=20), None,
         [{"arm": r["arm"], "y": r["y"]} for r in ratio_rows(100, 0.5)]),
    ]

    def regime_rows(n, sds, stay):
        rows, z = [], 0
        for _ in range(n):
            if rnd.random() > stay[z]:
                z = 1 - z
            rows.append({"y": rnd.gauss(0.0, sds[z])})
        return rows

    def count_rows(n, rates, stay):
        rows, z = [], 0
        for _ in range(n):
            if rnd.random() > stay[z]:
                z = 1 - z
            k, p, limit = 0, rnd.random(), math.exp(-rates[z])
            while p > limit:
                p *= rnd.random()
                k += 1
            rows.append({"y": k})
        return rows

    guess = HiddenMarkovFilter([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], GaussianEmission([0.0, 0.0], [0.5, 2.5]))
    counts = HiddenMarkovFilter([0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]], PoissonEmission([0.5, 4.0]))
    online_hmm = [
        ("online_hmm", OnlineHMM(guess, burn_in=30), None, regime_rows(160, (0.6, 2.0), (0.97, 0.93))),
        ("online_hmm_poisson_last_iterate", OnlineHMM(counts, kappa=0.8, burn_in=20, average=False), None,
         count_rows(120, (1.0, 6.0), (0.9, 0.85))),
    ]
    return existing + added + ported + later + round_four + stratified_logrank + ratio + online_hmm


def vector(label, procedure, seed, inputs):
    run = Run(procedure, seed=seed, run_id=label, clock=lambda: 0.0)
    outputs = []
    for x in inputs:
        event = run.step(x)
        if event is None:
            break
        outputs.append(procedure.encode_output(event.output))
    used = inputs[: run.t]
    return {
        "case": label,
        "procedure": procedure.identity(),
        "seed": None if seed is None else str(seed),
        "inputs": used,
        "outputs": outputs,
        "state": procedure.encode_state(run.state),
        "rng": None if run.rng_state is None else f"{run.rng_state:016x}",
        "terminal": run.terminal,
        "history_digest": run.history_digest,
    }


def number_vectors(count=500):
    rnd = random.Random(8785)
    values = [0.0, -0.0, 5e-324, 1.7976931348623157e308, 1e21, 1e-7, 1e-6, 0.1, 1 / 3, 2.0**53, 123.0, -1.5]
    while len(values) < count:
        x = struct.unpack("<d", struct.pack("<Q", rnd.getrandbits(64)))[0]
        if x == x and abs(x) != float("inf"):
            values.append(x)
    return [{"bits": f"{struct.unpack('<Q', struct.pack('<d', v))[0]:016x}", "json": _number(v)} for v in values]


def main():
    OUT.mkdir(exist_ok=True)
    vectors = [vector(*case) for case in cases()]  # novermin (not PEP 798)
    (OUT / "procedures.json").write_text(canonical_json(vectors) + "\n", encoding="utf-8")
    (OUT / "numbers.json").write_text(canonical_json(number_vectors()) + "\n", encoding="utf-8")
    print(f"wrote {len(vectors)} procedure vectors and the number vectors to {OUT}")


if __name__ == "__main__":
    main()
