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
| GroupSequentialTest | - | 100 | 2 µs | 152 B |
| NormalMixtureSPRT | - | 100 | 3 µs | 127 B |
| TMixtureSPRT | - | 100 | 4 µs | 171 B |
| BettingMeanTest | - | 100 | 3 µs | 179 B |
| SequentialSignTest | - | 100 | 3 µs | 177 B |
| QuantileTest | - | 100 | 29 µs | 99 B |
| QuantileTest | - | 10,000 | 116 µs | 106 B |
| QuantileConfidenceSequence | - | 100 | 153 µs | 2150 B |
| QuantileConfidenceSequence | - | 1,000 | 465 µs | 21 kB |
| QuantileConfidenceSequence | - | 10,000 | 830 µs | 206 kB |
| KernelTwoSampleTest | python | 100 | 70 µs | 4341 B |
| KernelTwoSampleTest | python | 1,000 | 638 µs | 41 kB |
| KernelTwoSampleTest | python | 10,000 | 5.6 ms | 413 kB |
| KernelTwoSampleTest | numpy | 100 | 20 µs | 4341 B |
| KernelTwoSampleTest | numpy | 1,000 | 91 µs | 41 kB |
| KernelTwoSampleTest | numpy | 10,000 | 672 µs | 413 kB |
| BettingConfidenceSequence | python | 100 | 382 µs | 40 kB |
| BettingConfidenceSequence | numpy | 100 | 103 µs | 40 kB |
| CatoniConfidenceSequence | python | 100 | 243 µs | 2167 B |
| CatoniConfidenceSequence | python | 1,000 | 2.0 ms | 21 kB |
| CatoniConfidenceSequence | python | 3,000 | 7.8 ms | 62 kB |
| CatoniConfidenceSequence | numpy | 100 | 193 µs | 2167 B |
| CatoniConfidenceSequence | numpy | 1,000 | 554 µs | 21 kB |
| CatoniConfidenceSequence | numpy | 3,000 | 1.4 ms | 62 kB |
| AsymptoticConfidenceSequence | - | 100 | 2 µs | 141 B |
| CUSUM | - | 100 | 1 µs | 38 B |
| ShiryaevRoberts | - | 100 | 1 µs | 56 B |
| LocalLevelKalman | - | 100 | 1 µs | 67 B |
| KalmanFilter, 2 states, 2 observations | - | 100 | 83 µs | 236 B |
| KalmanFilter, 4 states, 2 observations | - | 100 | 147 µs | 522 B |
| RobustKalmanFilter (local linear trend) | - | 100 | 582 µs | 199 B |
| HiddenMarkovFilter, 3 states | - | 100 | 5 µs | 162 B |
| BayesianChangePoint (normal-gamma) | - | 100 | 161 µs | 7400 B |
| BayesianChangePoint (normal-gamma) | - | 1,000 | 1.6 ms | 75 kB |
| BayesianChangePoint (normal-gamma) | - | 3,000 | 4.9 ms | 153 kB |
| BootstrapParticleFilter, 256 particles | - | 100 | 630 µs | 5417 B |
| ParticleFilter, 1000 particles | python | 100 | 2.6 ms | 41 kB |
| ParticleFilter, 1000 particles | numpy | 100 | 233 µs | 41 kB |
| ParticleFilter, 20000 particles | python | 20 | 38.7 ms | 822 kB |
| ParticleFilter, 20000 particles | numpy | 20 | 2.7 ms | 822 kB |
| MeanDifference | - | 100 | 2 µs | 139 B |
| EMA | - | 100 | 1 µs | 40 B |
