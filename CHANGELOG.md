# Changelog

All notable changes are listed here. The project follows
[semantic versioning](https://semver.org/); until 1.0 a minor version may
change the checkpoint, log and cache formats, which are versioned
separately (`sequential-analysis.checkpoint/2`, `.log/2`, `.history/2`,
`.design-cache/1`).

## 0.4.0 — 2026-10-16

Group sequential trials for any endpoint, checked against rpact and R's
survival package; ratio metrics; stratified log-rank tests; plots;
property-based tests; a guide to choosing a method.

### Added: group sequential trials for any endpoint
- `GroupSequentialTwoMeans` (continuous endpoint in two arms, known
  standard deviation or pooled t), `GroupSequentialProportions` (pooled
  two-proportion Z) and `GroupSequentialLogRank` (log-rank statistic with
  late entry, optionally stratified), analysed at unit or event counts.
- `GroupSequentialMonitor`: alpha spending at the information actually
  reached, one record (Z, information) per analysis, for any statistic
  computed elsewhere; under- and over-running; optional non-binding
  futility from the design.
- `GroupSequentialDesign.at(fractions)`: the boundaries recomputed for
  analyses that did not happen as planned (Lan–DeMets); boundaries of
  analyses already done never move.
- Sample sizes: `inflation_factor`, `max_sample_size_two_means`,
  `max_sample_size_two_proportions` (rpact's method), `max_events`
  (Schoenfeld).

### Added: ratio metrics and relative lift
- `RatioABTest`: anytime-valid A/B test for ratio metrics (revenue per
  session, click-through rate, average order value) and for relative lift.
  It uses the delta method on the totals of the randomised unit (Deng,
  Knoblich and Lu 2018) with the asymptotic confidence sequence of
  Waudby-Smith et al. (2024). It reports sequences for R_T − R_C and, on
  the log scale, for R_T / R_C − 1. `denominator=False` gives the
  relative lift of a mean per user.
- Measured with sessions correlated within users and monitoring from
  1,000 users: the time-uniform miscoverage is at most 0.047. In the same
  simulations, the naive per-session test rejected a true null in 24–26%
  of runs.
- Constant state, 7 µs per step. TypeScript port included.

### Added: stratified survival tests
- `AnytimeLogRankTest(strata=k, stratum_input="stratum")`: a stratified
  anytime-valid log-rank test, with numbers at risk per arm and stratum and
  a common hazard ratio. It is exact under proportional hazards within the
  strata, whatever each stratum's baseline hazard. Stratifying by centre
  keeps the level when centres differ in baseline hazard. In simulation,
  with unequal allocation and hazards of 0.2, 1 and 5, the unstratified
  test rejects 96% of null trials and the stratified one 2.5%. Stratifying
  by narrow entry cohorts brings the test back under its level when
  hazards depend on time since entry (21% → 3.8% of null trials with 25
  cohorts). With `strata=1`, the default, the identity, configuration,
  state and conformance vectors are unchanged. Ported to the TypeScript
  executor.
- `GroupSequentialLogRank` takes `strata=` too.
- Corrected numbers in the time-scale caveat of the log-rank test, from
  new simulations: 21% (not 24%) for Weibull shape 0.3, 14% (not 5%) for
  shape 0.5.

### Added: plotting
- `sequential_analysis.plotting` (optional extra `[plot]`, matplotlib 3.3
  or later): `plot_e_process`, `plot_confidence_sequence`,
  `plot_group_sequential_design`, `plot_detection`, `plot_filter`,
  `plot_stopping_times`. Each takes a Recorder, output events, outputs or
  trajectory pairs and returns the matplotlib Axes. Importing the module
  does not need matplotlib.
- A documentation page, "Plotting runs, designs and plans", whose
  pictures are drawn by its own code (`python tools/make_plots.py`);
  tested on matplotlib 3.3.0 with Python 3.8 and on 3.11 with Python 3.13.

### Checked against R
- `tools/r_reference.R` computes reference values with rpact 3.4.0 and
  survival 3.5.8; `tests/test_r_reference.py` compares: 56 designs
  (boundaries to 2·10⁻⁷, futility boundaries, drift, crossing and futility
  probabilities, expected sample sizes, inflation factors), boundaries at
  realised information including under- and over-running, 44 sample
  sizes, the log-rank statistic at four interim analyses of a trial with
  staggered entry (to 10⁻¹²), the chi-square and pooled t tests. A CI job
  installs R, regenerates the references and repeats the comparison.

### Changed
- The power of a two-sided group sequential design counts rejections in
  the direction of the effect, as rpact and gsDesign do (sample sizes of
  two-sided Pocock-type designs fall by up to 6·10⁻⁵ relative).
- `drift_for_power` solves by the Illinois method: a few evaluations of
  the power instead of fifty.
- The recursive integration sums in plain loops: `sum()` of floats is
  compensated from Python 3.12 on, and boundaries must not depend on the
  Python version.

### Fixed (found by the property-based tests)
- A step whose new state or output holds an integer beyond 2**53 (two
  units with 2**53 events each in RateRatioTest) is a numerical failure,
  and nothing is committed. Before, the step was committed and the next
  checkpoint failed.
- Integer literals beyond 2**53 in stored JSON are read as the binary64
  numbers they denote, as `JSON.parse` reads them (SPEC 1). This applies
  to checkpoints given to `Run.restore` or read by `load_checkpoint`, and
  to observation logs. A run continued by the TypeScript executor comes
  back to Python even when a number in its state is 1e17 or more.
- `AnytimeLinearModel` restores its own checkpoints after a covariate
  deviation below about 1e-154: the Cauchy–Schwarz check of the stored
  cross products allows for products that underflowed.
- Group sequential integration: the continuation region keeps the same
  width in standard deviations at every scale. An analysis at an
  information fraction of 1e-6 no longer builds a grid of millions of
  points. A grid beyond 20,000 points is a NumericalError, not an
  unbounded allocation.
- `RateRatioTest`: once every grid point is excluded, the confidence
  sequence keeps its last bound when the ratio lies beyond the grid, and
  collapses onto the least-excluded point otherwise. It used to jump back
  to (None, None).
- `Delivery.skip_range` from the current position drops the numbers
  delivered inside the range. Before, they stayed in the checkpoint for
  ever.

### Documentation and packaging
- "Which method?": from the question asked to the procedure, with how
  exact each guarantee is.
- The group sequential tutorial covers binary and survival endpoints and
  analyses that move.
- A conda-forge recipe (`packaging/conda-forge/meta.yaml`) and a paper for
  the Journal of Open Source Software (`paper/joss/`); RELEASING.md says
  how to submit both.

## 0.3.0 — 2026-10-15

Filters and smoothers, A/B tests, nonparametric change detection, data
source adapters, a numpy backend, a specification of the formats with a
TypeScript executor, and a documentation site.

### Added: A/B tests
- `TwoProportionTest`: exact anytime-valid comparison of conversion rates,
  one- or two-sided. By default (`method="grow"`) units are grouped into
  blocks (`block=(1, 1)`, or the allocation ratio) and the blocks' e-values
  of Turner, Ly and Grünwald multiply: exact for every common rate (checked
  by enumerating every outcome of a block), and by simulation 1.3 to 1.6
  times the units of the asymptotic test. `method="universal"` (sequential
  universal inference) needs no blocks and stays exact under
  outcome-adaptive allocation, at 2.3 to 3 times.
- `SampleRatioMismatch`: Dirichlet-multinomial test of the traffic split.
- `ABTest`: difference of means by inverse-propensity scores with a
  predictable regression on a pre-experiment covariate (CUPED without its
  in-sample bias), asymptotic confidence sequence, reported variance ratio.
- `e_bh` and `MultipleTests`: e-BH false discovery rate control over every
  procedure with an e-process (`e_value_field`); `versus_control` for
  multi-arm experiments.

- `BoundedABTest`: exact confidence sequence and test for an effect on a
  bounded metric (conversion, share, capped revenue), from
  inverse-propensity scores and betting; valid at every sample size.
- `RateRatioTest`: exact test and confidence sequence for a ratio of
  Poisson rates with exposures, from pairs of units adjacent in time, so
  that a drifting common rate (seasonality) does not matter.
- `OnlineFDR`: e-LOND for a programme of experiments decided one by one,
  FDR <= alpha under any dependence; an open-ended or planned (`horizon`)
  schedule.
- `sequential_analysis.planning`: `simulate` gives the power and the
  distribution of the stopping time of any procedure in a scenario;
  checked against exact operating characteristics.

### Added: regression and survival
- `AnytimeLinearModel`: anytime-valid tests and confidence sequences for
  regression coefficients after Lindon, Ham, Tingley and Bojinov (2022):
  covariate-adjusted effects with several covariates, exact under
  Gaussian errors; the closed form is checked against two independent
  numerical integrations of the mixture.
- `AnytimeLogRankTest`: anytime-valid log-rank test (ter Schure,
  Pérez-Ortiz, Ly and Grünwald) with a hazard-ratio confidence sequence;
  exact for any baseline hazard when hazards are proportional on the time
  scale of the records (the tutorial and the docstring describe the
  staggered-entry case where they are not).

### Added: change detection and smoothing
- `MeanEDetector`, `QuantileEDetector`: e-detectors (Shin, Ramdas and
  Rinaldo), Shiryaev–Roberts or CUSUM type, average run length at least
  `arl` for every distribution with the given mean or quantile.
- `sequential_analysis.smoothing`: Rauch–Tung–Striebel smoother,
  forward-backward and Viterbi, checked against exact conditioning and
  enumeration.

### Added: data sources
- `sequential_analysis.adapters`: `from_dataframe` (pandas, polars),
  `from_sql` (DB-API), `from_kafka` (kafka-python, confluent-kafka), with
  `next_position` and `kafka_offsets` to resume from a checkpoint.
- `Delivery.skip_range`: sequence numbers that do not exist (Kafka offset
  gaps, skipped missing values) no longer accumulate in checkpoints.

### Added: specification, TypeScript executor, documentation
- `SPEC.md`: the formats and semantics another implementation needs.
- `ts/`: a TypeScript executor of every procedure of the library (36;
  those with a numpy backend with the Python backend). It reproduces all 53
  conformance vectors -- 19 bit for bit, the others to 3.6e-13 -- and
  continues runs started in Python (and hands them back).
- 37 conformance vectors for the new procedures.
- A documentation site (mkdocs, GitHub Pages) with tutorials whose code
  runs in the tests, and an API reference generated from the docstrings.
- Release workflow: tested on Python 3.8 to 3.14; the TypeScript executor
  is published to npm (with the `NPM_TOKEN` secret) and must carry the
  same version.

### Added: filters
- `KalmanFilter`: any linear Gaussian state-space model in matrix form, with
  control input, Joseph-form covariance update and the exact likelihood;
  constructors for local linear trends and autoregressions.
- `RobustKalmanFilter`: Student-t observation noise by variational Bayes
  (Piché, Särkkä and Hartikainen 2012).
- `HiddenMarkovFilter`: the forward algorithm with categorical, Gaussian
  and Poisson emissions.
- `BayesianChangePoint` (Adams and MacKay 2007) with normal-gamma,
  Poisson-gamma and beta-Bernoulli models, pruning and an alarm.
- `ParticleFilter` for any `StateSpaceModel`: adaptive systematic
  resampling, unbiased likelihood; linear Gaussian, stochastic volatility
  and Student-t local level models.
- Tests against independent references: exact rational Gaussian
  conditioning, enumeration of paths and partitions, grid filters.

### Added: speed
- numpy backend (`backend="auto" | "python" | "numpy"`) for
  `ParticleFilter`, `KernelTwoSampleTest`, `BettingConfidenceSequence` and
  `CatoniConfidenceSequence`, 2 to 12 times faster; part of the
  procedure's identity, so checkpoints never switch backend. The numpy
  draws are exactly those of the run's SplitMix64 generator.
- `tools/benchmark.py`, `PERFORMANCE.md`, a CI benchmark job; README
  sections "Checkpoints and backends" and "Speed". The provenance records
  the numpy version; restoring under another one is reported.

### Changed
- Python 3.8 and 3.9 are supported again: the library needs Python 3.8 or
  later (it needed 3.10), tested in CI on 3.8 with numpy 1.22.0 and on 3.9;
  the TypeScript executor runs on Node.js 16 or later. Results are the same
  bits on every version.
- The four procedures above have a `backend` entry in their
  configuration, so their checkpoints from 0.2.0 are refused.

## 0.2.0 — 2026-10-13

Methods that do not assume normal data, and the fixes of a line-by-line
review of the whole package after 0.1.0 (no 0.1.1 was published). No
format changes; the Kalman filter, the t-mixture SPRT and the group
sequential test have new versions (2, 3 and 4), so their checkpoints from
0.1.0 are refused rather than continued with different arithmetic.

### Added: inference that does not assume normality
- `QuantileTest` and `QuantileConfidenceSequence` (Howard and Ramdas 2022):
  anytime-valid inference for any quantile of any distribution, by
  one-sided beta-binomial mixtures; valid with ties and atoms, where the
  sign test is not.
- `KernelTwoSampleTest` (Shekhar and Ramdas 2023): are A and B equally
  distributed? A kernel witness and online Newton bets; detects any
  difference of the distributions.
- `CatoniConfidenceSequence` (Wang and Ramdas 2023): the mean of
  heavy-tailed data, given a bound on the standard deviation.
- `AsymptoticConfidenceSequence` (Waudby-Smith et al. 2024): the mean with
  estimated variance; asymptotic coverage, reported from `start`
  observations on.
- `BettingConfidenceSequence` (Waudby-Smith and Ramdas 2024): the mean of
  bounded data by hedged capital.
- All six satisfy the conformance invariants (checkpoint, replay, batch,
  sync/async equivalence); Monte Carlo tests on Cauchy, lognormal, Pareto,
  zero-inflated and discrete data.

### Fixed
- **Checkpoints.** A refused restore no longer cuts back the observation
  log, nor changes the topology and delivery passed in. Malformed
  checkpoints (integers beyond 2**53, NaN, lone surrogates, unknown
  counters or policies, provenance of another run, a randomized run
  without seed) are `IncompatibleCheckpoint`; `true` no longer passes for
  `1`. Every procedure checks a decoded state: field types, decisions and
  its own invariants.
- **Runs.** `step()` fails the run on an unexpected error, as `offer()`
  does, and commits nothing; consumers can no longer pause or close a run
  in the middle of a transition; a failing `Chain` link is an invalid
  input.
- **Restore and continue.** `skip_to` filters by the delivery position at
  the checkpoint, so duplicates arriving later reach the restored run as
  they reached the original; with `overflow="drop"` a dropped sequence
  number no longer holds up the observations after it; an in-memory
  observation log holding an unrecordable value can be restored.
- **Kalman filter.** A vague prior (initial variance 1e16 or more) froze
  the filter at the first observation; the posterior variance is now
  computed without cancellation.
- **Stopped tests** (SPRT, 2-SPRT, plans, CUSUM, Shiryaev–Roberts, group
  sequential) refuse further observations when stepped directly; the
  group sequential test used to overwrite its decision.
- **Student's t.** With 1e7 to 1e9 degrees of freedom and t of 25 to 40
  the tail was accurate to 3e-10 only; the continued fraction is now
  evaluated without cancellation there, within 1e-12.
- **Group sequential designs.** O'Brien–Fleming spending overspent alpha
  below about 1e-10 (by 11% at 1e-15); NaN futility boundaries are refused.
- **t-mixture SPRT.** `critical_t` overflowed for extreme effect sizes and
  error rates.
- **Designs.** An infinite lattice step made `kiefer_weiss_plan` loop
  forever; invalid, too fine or too coarse steps are refused, and a step
  for Bernoulli or Poisson data is an error instead of being ignored. The
  cache verifies stored multipliers, reports files it cannot remove
  instead of raising, writes entries readable by other users, and ignores
  a relative `XDG_CACHE_HOME`.
- **Python 3.14** words math domain errors differently; they are
  `NumericalError` again. Python 3.14 is tested and supported.
- Canonical JSON refuses lone surrogates and encodes `numpy.float64` as a
  float; `InputContract` checks its bounds and shape; documentation and
  docstrings corrected where they had drifted from the code.

## 0.1.0 — 2026-10-10

First public release, dual-licensed: GNU AGPL v3, or a commercial license
from Novikov Laboratories LLC for closed-source products and services (see
COMMERCIAL.md).

### Runtime
- A sequential procedure is a transition function F(S, X, R) → (S', O, R'); the
  runtime around it is made of separate layers: sources and adapters, topology
  (independent, positional pairing, key join, time alignment, mapped
  transforms), delivery (arrival, sequence and strict policies, deduplication),
  synchronous and asynchronous runtimes, consumers.
- Transactional steps: an invalid input, a numerical error or a failing
  transition leaves the state as it was.
- Checkpoints as plain JSON with the inferential state, the 64-bit random
  state, topology buffers, source positions, the observation log position and
  provenance; restoring and continuing reproduces the uninterrupted run bit for
  bit. Damaged or foreign checkpoints are refused with a message naming the
  problem.
- Observation log with a hash chain; replay reproduces the history digest.
- Invalid input is skipped and reported as a structured incident by default;
  `on_invalid="raise"`, an exception class, or `max_invalid_streak` make it
  fatal.
- History digests and checkpoint identifiers use RFC 8785 canonical JSON;
  language-independent conformance vectors with a JavaScript check.

### Procedures
- Tests of simple hypotheses: Wald's SPRT, Lorden's 2-SPRT, execution of any
  precomputed plan.
- Anytime-valid tests: normal mixture SPRT, t-mixture SPRT (unknown variance),
  betting test for bounded means, sequential sign test.
- Group sequential tests with efficacy and non-binding futility boundaries,
  known or estimated variance.
- Change-point detection: CUSUM, Shiryaev–Roberts.
- Filtering and estimation: local-level Kalman filter (with control input),
  bootstrap particle filter, two-sample mean difference, exponential moving
  average.
- Families: normal (known variance), Bernoulli, Poisson, exponential.

### Design
- Operating characteristics of plans by exact recursion (lattice for continuous
  data).
- Optimal plans for the modified Kiefer–Weiss problem by backward induction,
  and the Kiefer–Weiss problem itself (least favourable θ*: the root of
  Lorden's characterisation for continuous data, the best candidate for
  discrete data); a precomputed table of the least favourable θ* for normal
  data; a Neyman–Pearson feasibility check.
- Design cache on disk with verification on loading.
- Group sequential designs from alpha- and beta-spending functions
  (O'Brien–Fleming, Pocock, power family).
- Optional numpy acceleration of the backward induction
  (`pip install "sequential-analysis[fast]"`).

### Numerics
- Student's t distribution from the standard library, checked against
  80-digit references; `NumericalError` instead of silent overflow or
  underflow.
