# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Speed of every procedure: time per step, how the state grows, size of a checkpoint.

    python tools/benchmark.py                 # a table on the terminal
    python tools/benchmark.py --markdown      # the table in PERFORMANCE.md

For each procedure the state is first grown to n observations, then the
time of one more step is measured (the median of repeated measurements),
for each backend available.  Procedures whose state holds the sample (or
one entry per run length) are measured at several n, so that the growth
is visible.  Times depend on the machine; the ratios between procedures
and between backends do not change much.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sequential_analysis import backend  # noqa: E402
from sequential_analysis.procedures import (  # noqa: E402
    CUSUM,
    EMA,
    SPRT,
    ABTest,
    AnytimeLinearModel,
    AnytimeLogRankTest,
    AsymptoticConfidenceSequence,
    AutoRegressive,
    BayesianChangePoint,
    Bernoulli,
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
    MultipleTests,
    NormalGammaModel,
    NormalMixtureSPRT,
    OnlineFDR,
    OnlineHMM,
    ParticleFilter,
    PlanTest,
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
    TMixtureSPRT,
    TwoProportionTest,
    TwoSPRT,
    averaged_smoothing,
)
from sequential_analysis.rng import SplitMix64  # noqa: E402
from sequential_analysis.smoothing import forward_backward, kalman_smoother, viterbi  # noqa: E402


def normal(rnd):
    return {"x": rnd.gauss(0.0, 1.0)}


def bounded(rnd):
    return {"x": rnd.random()}


def binary(rnd):
    return {"x": int(rnd.random() < 0.4)}


def observation(rnd):
    return {"y": rnd.gauss(0.0, 1.0)}


def pair(rnd):
    return {"a": rnd.gauss(0.0, 1.0), "b": rnd.gauss(0.0, 1.0)}


def loss_pair(rnd):
    return {"a": rnd.random(), "b": rnd.random()}


def two_d(rnd):
    return {"y": [rnd.gauss(0.0, 1.0), rnd.gauss(0.0, 1.0)]}


def unit(rnd):
    arm = rnd.randrange(3)
    x = rnd.gauss(0.0, 1.0)
    return {"arm": arm, "x": x, "y": int(rnd.random() < 0.1 + 0.02 * arm)}


def bounded_unit(rnd):
    return {"arm": rnd.randrange(2), "y": rnd.random()}


def rate_unit(rnd):
    return {"arm": rnd.randrange(2), "count": rnd.randrange(5), "exposure": rnd.uniform(0.5, 2.0)}


def ratio_unit(rnd):
    s = rnd.randrange(5)
    return {"arm": rnd.randrange(2), "y": s * rnd.expovariate(1.0), "d": float(s)}


def e_value(rnd):
    return {"log_e": rnd.gauss(0.0, 2.0)}


def regression_unit(rnd):
    x = [rnd.gauss(0.0, 1.0) for _ in range(2)]
    z = float(rnd.random() < 0.5)
    return {"x1": x[0], "x2": x[1], "z": z, "y": x[0] - x[1] + 0.2 * z + rnd.gauss(0.0, 1.0)}


def survival_record(rnd):
    """Entries and events, never an event in an arm with nobody at risk (counted on the generator)."""
    at_risk = rnd.__dict__.setdefault("at_risk", [0, 0])
    arm = rnd.randrange(2)
    kind = 1 if at_risk[arm] > 0 and rnd.random() < 0.4 else 0
    at_risk[arm] += -1 if kind else 1
    return {"arm": arm, "kind": kind}


def analysis_record(rnd):
    """One analysis of a monitor: Z and an information that grows by one each time (counted on the generator)."""
    k = rnd.__dict__.setdefault("analyses", [0])
    k[0] += 1
    return {"z": rnd.gauss(0.0, 1.0), "information": float(k[0])}


def four_states():
    f = tuple(tuple(1.0 if j in (i, i + 1) else 0.0 for j in range(4)) for i in range(4))
    q = tuple(tuple(0.01 if i == j else 0.0 for j in range(4)) for i in range(4))
    h = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    p0 = tuple(tuple(10.0 if i == j else 0.0 for j in range(4)) for i in range(4))
    return KalmanFilter(f, h, q, ((1.0, 0.0), (0.0, 1.0)), (0.0,) * 4, p0)


#: (name, factory taking a backend, input generator, sizes at which to measure, uses the backend, randomized)
CASES = [
    ("SPRT", lambda b: SPRT(Bernoulli(), 0.3, 0.5), binary, (100,), False),
    ("TwoSPRT", lambda b: TwoSPRT(Gaussian(1.0), 0.0, 0.5, 1e-6, 1e-6), normal, (100,), False),
    ("PlanTest", lambda b: PlanTest([(-50.0, 50.0)] * 100_000 + [(0.0, 0.0)]), normal, (100,), False),
    ("GroupSequentialTest", lambda b: GroupSequentialTest(1.0, list(range(100, 100_001, 100)), [3.0] * 1000),
     normal, (100,), False),
    ("GroupSequentialTwoMeans, estimated variance",
     lambda b: GroupSequentialTwoMeans(None, list(range(10_000, 1_000_001, 10_000)), [3.0] * 100), bounded_unit,
     (100,), False),
    ("GroupSequentialProportions",
     lambda b: GroupSequentialProportions(list(range(10_000, 1_000_001, 10_000)), [3.0] * 100), unit, (100,), False),
    ("GroupSequentialLogRank",
     lambda b: GroupSequentialLogRank(list(range(10_000, 1_000_001, 10_000)), [3.0] * 100), survival_record,
     (100,), False),
    ("GroupSequentialMonitor, one analysis", lambda b: GroupSequentialMonitor(0.025, 12.0, max_analyses=100),
     analysis_record, (3, 10), False),
    ("NormalMixtureSPRT", lambda b: NormalMixtureSPRT(1.0, 0.5, stop_on_reject=False), normal, (100,), False),
    ("TMixtureSPRT", lambda b: TMixtureSPRT(0.0, stop_on_reject=False), normal, (100,), False),
    ("BettingMeanTest", lambda b: BettingMeanTest(0.5, stop_on_reject=False), bounded, (100,), False),
    ("SequentialSignTest", lambda b: SequentialSignTest(stop_on_reject=False), normal, (100,), False),
    ("QuantileTest", lambda b: QuantileTest(0.0, stop_on_reject=False), normal, (100, 10_000), False),
    ("QuantileConfidenceSequence", lambda b: QuantileConfidenceSequence(), normal, (100, 1000, 10_000), False),
    ("KernelTwoSampleTest", lambda b: KernelTwoSampleTest(alpha=1e-12, backend=b), pair, (100, 1000, 10_000), True),
    ("BettingConfidenceSequence", lambda b: BettingConfidenceSequence(backend=b), bounded, (100,), True),
    ("CatoniConfidenceSequence", lambda b: CatoniConfidenceSequence(1.0, backend=b), normal, (100, 1000, 3000), True),
    ("AsymptoticConfidenceSequence", lambda b: AsymptoticConfidenceSequence(start=10), normal, (100,), False),
    ("CUSUM", lambda b: CUSUM(Gaussian(1.0), 0.0, 1.0, threshold=1e9), normal, (100,), False),
    ("ShiryaevRoberts", lambda b: ShiryaevRoberts(Gaussian(1.0), 0.0, 0.1, 1e300), normal, (100,), False),
    ("MeanEDetector, both directions", lambda b: MeanEDetector(0.5, arl=1e15), bounded, (100,), False),
    ("QuantileEDetector, both directions", lambda b: QuantileEDetector(0.0, arl=1e15), normal, (100,), False),
    ("LocalLevelKalman", lambda b: LocalLevelKalman(0.1, 1.0), observation, (100,), False),
    ("KalmanFilter, 2 states, 2 observations",
     lambda b: KalmanFilter(((1.0, 1.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0)), ((0.01, 0.0), (0.0, 0.01)),
                            ((1.0, 0.0), (0.0, 1.0)), (0.0, 0.0), ((10.0, 0.0), (0.0, 10.0))), two_d, (100,), False),
    ("KalmanFilter, 4 states, 2 observations", lambda b: four_states(), two_d, (100,), False),
    ("RobustKalmanFilter (local linear trend)", lambda b: RobustKalmanFilter.local_linear_trend(0.01, 0.001, 1.0),
     observation, (100,), False),
    ("HiddenMarkovFilter, 3 states",
     lambda b: HiddenMarkovFilter((1 / 3,) * 3, ((0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)),
                                  GaussianEmission((-1.0, 0.0, 1.0), (1.0, 1.0, 1.0))), observation, (100,), False),
    ("OnlineHMM, 3 states, Gaussian",
     lambda b: OnlineHMM(HiddenMarkovFilter((1 / 3,) * 3, ((0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)),
                                            GaussianEmission((-1.0, 0.0, 1.0), (1.0, 1.0, 1.0))), burn_in=10),
     observation, (100,), False),
    ("ExponentialSmoothing, trend and season 12, fixed, 12 ahead",
     lambda b: ExponentialSmoothing(0.3, beta=0.01, gamma=0.1, trend="additive", period=12, initial_level=0.0,
                                    sigma=1.0, horizon=12), observation, (100,), False),
    ("ExponentialSmoothing, learning sigma (t intervals)", lambda b: ExponentialSmoothing(0.3), observation, (100,),
     False),
    ("AutoRegressive, order 5, learnt", lambda b: AutoRegressive(5, forgetting=0.99), observation, (100,), False),
    ("averaged_smoothing, trend and season 7 (63 models)",
     lambda b: averaged_smoothing(trend="additive", period=7), observation, (100,), False),
    ("ConformalForecast around learnt AutoRegressive(5)",
     lambda b: ConformalForecast(AutoRegressive(5, forgetting=0.99)), observation, (100,), False),
    ("ForecastComparison", lambda b: ForecastComparison((0.0, 1.0)), loss_pair, (100,), False),
    ("BayesianChangePoint (normal-gamma)", lambda b: BayesianChangePoint(NormalGammaModel(), 0.01), normal,
     (100, 1000, 3000), False),
    ("BootstrapParticleFilter, 256 particles", lambda b: BootstrapParticleFilter(0.1, 1.0, particles=256),
     observation, (100,), False),
    ("ParticleFilter, 1000 particles", lambda b: ParticleFilter(StochasticVolatilityModel(-1.0, 0.95, 0.3), 1000,
                                                                backend=b), observation, (100,), True),
    ("ParticleFilter, 20000 particles", lambda b: ParticleFilter(StochasticVolatilityModel(-1.0, 0.95, 0.3), 20000,
                                                                 backend=b), observation, (20,), True),
    ("SampleRatioMismatch, 3 arms", lambda b: SampleRatioMismatch((1, 1, 1)), unit, (100,), False),
    ("TwoProportionTest", lambda b: TwoProportionTest(alpha=1e-12), unit, (100,), False),
    ("ABTest with covariate", lambda b: ABTest(covariate=True, start=10), unit, (100,), False),
    ("MultipleTests, 2 arms vs control", lambda b: MultipleTests.versus_control([1, 2], 0, alpha=1e-12), unit,
     (100,), False),
    ("BoundedABTest, grid 1000", lambda b: BoundedABTest(backend=b), bounded_unit, (100,), True),
    ("RatioABTest", lambda b: RatioABTest(start=10), ratio_unit, (100,), False),
    ("RateRatioTest, grid 200", lambda b: RateRatioTest(alpha=1e-12), rate_unit, (100,), False),
    ("AnytimeLinearModel, 2 covariates", lambda b: AnytimeLinearModel(["x1", "x2"], ["z"]), regression_unit,
     (100,), False),
    ("AnytimeLogRankTest, with confidence sequence", lambda b: AnytimeLogRankTest(alpha=1e-12), survival_record,
     (100,), False),
    ("OnlineFDR (e-LOND)", lambda b: OnlineFDR(), e_value, (100,), False),
    ("MeanDifference", lambda b: MeanDifference(), pair, (100,), False),
    ("EMA", lambda b: EMA(0.1), normal, (100,), False),
]


def three_state_hmm():
    return HiddenMarkovFilter((0.4, 0.3, 0.3), ((0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)),
                              GaussianEmission((0.0, 1.0, 2.0), (1.0, 1.0, 1.0)))


#: offline passes over a whole series: (name, function of a list of inputs, input generator)
OFFLINE = [
    ("kalman_smoother, local linear trend", lambda xs: kalman_smoother(KalmanFilter.local_linear_trend(0.1, 0.01, 1.0),
                                                                        xs), observation),
    ("forward_backward, 3 states", lambda xs: forward_backward(three_state_hmm(), xs), observation),
    ("viterbi, 3 states", lambda xs: viterbi(three_state_hmm(), xs), observation),
]


def offline_rows(length: int = 5000):
    for name, run, gen in OFFLINE:
        rnd = random.Random(1)
        xs = [gen(rnd) for _ in range(length)]
        times = []
        for _ in range(3):
            start = time.perf_counter()
            run(xs)
            times.append(time.perf_counter() - start)
        yield name, length, statistics.median(times) / length


def measure(make, gen, n: int, randomized: bool, repeats: int = 30, warm=None) -> tuple[float, int]:
    """Median seconds of one step after n observations, and the size of the encoded state in bytes.

    `warm` builds the procedure that grows the state (the states of the two backends are the same
    data, so the faster one may grow it for the other).
    """
    rnd = random.Random(1)
    proc, grower = make(), (warm or make)()
    rng = SplitMix64.for_role(1, "benchmark") if randomized else None
    state = grower.initial_state()
    for _ in range(n):
        if grower.is_terminal(state):
            state = grower.initial_state()
        state, _ = grower.step(state, gen(rnd), rng)
    size = len(json.dumps(proc.encode_state(state)))
    times = []
    for _ in range(repeats):
        x = gen(rnd)
        r = rng.copy() if rng is not None else None
        start = time.perf_counter()
        proc.step(state, x, r)
        times.append(time.perf_counter() - start)
    return statistics.median(times), size


def rows():
    backends = ["python"] + (["numpy"] if backend.np is not None else [])
    for name, factory, gen, sizes, uses_backend, *_ in CASES:
        randomized = getattr(factory("python"), "randomized", False)
        for b in backends if uses_backend else ["python"]:
            warm = (lambda factory=factory: factory("numpy")) if uses_backend and "numpy" in backends else None
            for n in sizes:
                seconds, size = measure(lambda b=b, factory=factory: factory(b), gen, n, randomized,
                                        repeats=5 if seconds_hint(name) else 30, warm=warm)
                yield name, b if uses_backend else "-", n, seconds, size


def seconds_hint(name: str) -> bool:
    return "20000" in name


def human(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def human_size(size: int) -> str:
    return f"{size} B" if size < 10_000 else f"{size / 1000:.0f} kB" if size < 10_000_000 else f"{size / 1e6:.0f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", action="store_true", help="write PERFORMANCE.md instead of printing")
    args = parser.parse_args()
    table = list(rows())
    machine = (f"{platform.python_implementation()} {platform.python_version()}, "
               f"numpy {backend.numpy_version() or 'not installed'}, {platform.machine()}")
    lines = ["| Procedure | Backend | after n | time per step | checkpointed state |", "|---|---|---|---|---|"]
    for name, b, n, seconds, size in table:
        lines.append(f"| {name} | {b} | {n:,} | {human(seconds)} | {human_size(size)} |")
    offline = list(offline_rows())
    lines += ["", "Offline smoothing (sequential_analysis.smoothing), over a whole series:", "",
              "| Function | series length | time per time point |", "|---|---|---|"]
    for name, length, seconds in offline:
        lines.append(f"| {name} | {length:,} | {human(seconds)} |")
    if args.markdown:
        text = (ROOT / "PERFORMANCE.md").read_text(encoding="utf-8") if (ROOT / "PERFORMANCE.md").exists() else ""
        head = text.split("<!-- table -->")[0] if "<!-- table -->" in text else "# Performance\n\n"
        (ROOT / "PERFORMANCE.md").write_text(
            head + "<!-- table -->\n" + f"Measured with `python tools/benchmark.py --markdown` on {machine}.\n\n"
            + "\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {ROOT / 'PERFORMANCE.md'}")
    else:
        print(machine)
        print("\n".join(lines))
    assert all(math.isfinite(s) for *_, s, _ in table)
    assert all(math.isfinite(s) for *_, s in offline)


if __name__ == "__main__":
    main()
