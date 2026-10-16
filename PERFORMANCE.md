# Performance

How long one step of each procedure takes, how its state grows, and how
large a checkpoint of it is.  Regenerate the table on your machine with

    python tools/benchmark.py              # print
    python tools/benchmark.py --markdown   # rewrite the table below

Times depend on the machine; the ratios between procedures and between
backends change little.

## What determines the cost

* **Constant state** — tests of simple hypotheses, mixture and betting
  tests, CUSUM, Shiryaev–Roberts, the Kalman and hidden Markov filters,
  `QuantileTest`, `AsymptoticConfidenceSequence`: a step costs the same
  after ten observations or ten million, and so does a checkpoint (a few
  hundred bytes).  `QuantileTest` evaluates an incomplete beta function,
  whose cost grows slowly with n.
* **State that holds the sample** — `QuantileConfidenceSequence`,
  `KernelTwoSampleTest`, `CatoniConfidenceSequence`: memory and checkpoint
  grow linearly (about 20 bytes per observation); a step costs O(n)
  (sorted insertion; the kernel witness; Newton steps over the sample).
  Fine for tens of thousands of observations, not for millions; to test a
  single value, `QuantileTest` needs three counts.
* **A grid or a set of hypotheses** — `BettingConfidenceSequence` keeps two
  numbers per grid point (`grid=1000`, 40 kB); `BayesianChangePoint` one
  hypothesis per run length, capped by `max_hypotheses` (2000 by default).
* **Particles** — `ParticleFilter` costs O(N) per step for N particles and
  stores them all in the checkpoint (about 40 bytes per particle).

## Backends

`ParticleFilter`, `KernelTwoSampleTest`, `BettingConfidenceSequence` and
`CatoniConfidenceSequence` use numpy when it is installed
(`pip install "sequential-analysis[fast]"`), 2 to 12 times faster; the
others stay in Python, where numpy would be slower (a 2×2 Kalman step is
71 µs in Python and 120 µs with numpy).  See the README, *Checkpoints and
backends*, for what the backend means for checkpoints.

Optimal designs (`kiefer_weiss_plan`) are computed offline: about a second
for a Bernoulli design with horizon 80, seconds to a minute for normal
data; the least favourable θ* several times that (see the README).

<!-- table -->
Measured with `python tools/benchmark.py --markdown` on CPython 3.13.16, numpy 2.5.3, x86_64.

| Procedure | Backend | after n | time per step | checkpointed state |
|---|---|---|---|---|
| SPRT | - | 100 | 1 µs | 87 B |
| TwoSPRT | - | 100 | 2 µs | 139 B |
| PlanTest | - | 100 | 1 µs | 88 B |
| GroupSequentialTest | - | 100 | 3 µs | 152 B |
| GroupSequentialTwoMeans, estimated variance | - | 100 | 4 µs | 212 B |
| GroupSequentialProportions | - | 100 | 4 µs | 119 B |
| GroupSequentialLogRank | - | 100 | 4 µs | 146 B |
| GroupSequentialMonitor, one analysis | - | 3 | 90.3 ms | 166 B |
| GroupSequentialMonitor, one analysis | - | 10 | 515.0 ms | 414 B |
| NormalMixtureSPRT | - | 100 | 2 µs | 127 B |
| TMixtureSPRT | - | 100 | 3 µs | 171 B |
| BettingMeanTest | - | 100 | 3 µs | 179 B |
| SequentialSignTest | - | 100 | 3 µs | 177 B |
| QuantileTest | - | 100 | 26 µs | 99 B |
| QuantileTest | - | 10,000 | 83 µs | 106 B |
| QuantileConfidenceSequence | - | 100 | 153 µs | 2150 B |
| QuantileConfidenceSequence | - | 1,000 | 277 µs | 21 kB |
| QuantileConfidenceSequence | - | 10,000 | 730 µs | 206 kB |
| KernelTwoSampleTest | python | 100 | 62 µs | 4341 B |
| KernelTwoSampleTest | python | 1,000 | 532 µs | 41 kB |
| KernelTwoSampleTest | python | 10,000 | 3.6 ms | 413 kB |
| KernelTwoSampleTest | numpy | 100 | 27 µs | 4341 B |
| KernelTwoSampleTest | numpy | 1,000 | 74 µs | 41 kB |
| KernelTwoSampleTest | numpy | 10,000 | 683 µs | 413 kB |
| BettingConfidenceSequence | python | 100 | 443 µs | 40 kB |
| BettingConfidenceSequence | numpy | 100 | 127 µs | 40 kB |
| CatoniConfidenceSequence | python | 100 | 255 µs | 2167 B |
| CatoniConfidenceSequence | python | 1,000 | 2.6 ms | 21 kB |
| CatoniConfidenceSequence | python | 3,000 | 8.2 ms | 62 kB |
| CatoniConfidenceSequence | numpy | 100 | 191 µs | 2167 B |
| CatoniConfidenceSequence | numpy | 1,000 | 560 µs | 21 kB |
| CatoniConfidenceSequence | numpy | 3,000 | 1.3 ms | 62 kB |
| AsymptoticConfidenceSequence | - | 100 | 1 µs | 141 B |
| CUSUM | - | 100 | 1 µs | 38 B |
| ShiryaevRoberts | - | 100 | 1 µs | 56 B |
| MeanEDetector, both directions | - | 100 | 6 µs | 499 B |
| QuantileEDetector, both directions | - | 100 | 6 µs | 499 B |
| LocalLevelKalman | - | 100 | 1 µs | 67 B |
| KalmanFilter, 2 states, 2 observations | - | 100 | 65 µs | 236 B |
| KalmanFilter, 4 states, 2 observations | - | 100 | 186 µs | 522 B |
| RobustKalmanFilter (local linear trend) | - | 100 | 551 µs | 199 B |
| HiddenMarkovFilter, 3 states | - | 100 | 5 µs | 162 B |
| OnlineHMM, 3 states, Gaussian | - | 100 | 58 µs | 1715 B |
| ExponentialSmoothing, trend and season 12, fixed, 12 ahead | - | 100 | 10 µs | 493 B |
| ExponentialSmoothing, learning sigma (t intervals) | - | 100 | 318 µs | 217 B |
| AutoRegressive, order 5, learnt | - | 100 | 367 µs | 1128 B |
| averaged_smoothing, trend and season 7 (63 models) | - | 100 | 7.5 ms | 26 kB |
| ConformalForecast around learnt AutoRegressive(5) | - | 100 | 411 µs | 1275 B |
| ForecastComparison | - | 100 | 3 µs | 150 B |
| BayesianChangePoint (normal-gamma) | - | 100 | 160 µs | 7400 B |
| BayesianChangePoint (normal-gamma) | - | 1,000 | 1.6 ms | 75 kB |
| BayesianChangePoint (normal-gamma) | - | 3,000 | 3.9 ms | 153 kB |
| BootstrapParticleFilter, 256 particles | - | 100 | 321 µs | 5417 B |
| ParticleFilter, 1000 particles | python | 100 | 1.3 ms | 41 kB |
| ParticleFilter, 1000 particles | numpy | 100 | 132 µs | 41 kB |
| ParticleFilter, 20000 particles | python | 20 | 30.6 ms | 822 kB |
| ParticleFilter, 20000 particles | numpy | 20 | 3.0 ms | 822 kB |
| SampleRatioMismatch, 3 arms | - | 100 | 3 µs | 150 B |
| TwoProportionTest | - | 100 | 6 µs | 275 B |
| ABTest with covariate | - | 100 | 3 µs | 325 B |
| MultipleTests, 2 arms vs control | - | 100 | 28 µs | 622 B |
| BoundedABTest, grid 1000 | python | 100 | 382 µs | 41 kB |
| BoundedABTest, grid 1000 | numpy | 100 | 127 µs | 41 kB |
| RatioABTest | - | 100 | 5 µs | 254 B |
| RateRatioTest, grid 200 | - | 100 | 45 µs | 4873 B |
| AnytimeLinearModel, 2 covariates | - | 100 | 12 µs | 348 B |
| AnytimeLogRankTest, with confidence sequence | - | 100 | 429 µs | 5336 B |
| OnlineFDR (e-LOND) | - | 100 | 1 µs | 28 B |
| MeanDifference | - | 100 | 1 µs | 139 B |
| EMA | - | 100 | 1 µs | 40 B |

Offline smoothing (sequential_analysis.smoothing), over a whole series:

| Function | series length | time per time point |
|---|---|---|
| kalman_smoother, local linear trend | 5,000 | 119 µs |
| forward_backward, 3 states | 5,000 | 12 µs |
| viterbi, 3 states | 5,000 | 2 µs |
