# Which method?

Start from the question you are asking and from what you are willing
to assume. Each row points to the procedure, says how exact its
guarantee is, and names the tutorial that uses it.

"Exact" means: the error rate holds at every sample size and at any
stopping time, under the stated assumptions. "Asymptotic" means it holds
approximately in long runs.

**Prefer an exact procedure.** Where an exact procedure answers the
question, it is the default here and in the tutorials. An asymptotic one
is worth its approximation only when nothing exact fits: an unbounded
metric with no variance bound, a ratio of sums, planned analyses of an
approximately normal statistic.

Every procedure states its promise in its `guarantee` attribute
(`sequential_analysis.core.GUARANTEES` lists the values):

<!-- guarantees -->
| Guarantee | Meaning | Procedures |
|---|---|---|
| **exact** | error rates, coverage or false-alarm rates hold at every sample size, under the stated assumptions | `AnytimeLinearModel`, `AnytimeLogRankTest`, `BettingConfidenceSequence`, `BettingMeanTest`, `BoundedABTest`, `CatoniConfidenceSequence`, `ConformalForecast`, `CUSUM`, `ForecastComparison`, `GroupSequentialTest` (known sigma), `GroupSequentialTwoMeans` (known sigma), `KernelTwoSampleTest`, `MeanEDetector`, `MultipleTests`, `NormalMixtureSPRT`, `OnlineFDR`, `PlanTest`, `QuantileConfidenceSequence`, `QuantileEDetector`, `QuantileTest`, `RateRatioTest`, `SampleRatioMismatch`, `SequentialSignTest`, `ShiryaevRoberts`, `SPRT`, `TMixtureSPRT`, `TwoProportionTest`, `TwoSPRT` |
| **asymptotic** | they hold approximately, the better the larger the sample | `ABTest`, `AsymptoticConfidenceSequence`, `GroupSequentialLogRank`, `GroupSequentialMonitor`, `GroupSequentialProportions`, `GroupSequentialTest` (estimated sigma), `GroupSequentialTwoMeans` (estimated sigma), `RatioABTest` |
| **model** | exact probabilities under the stated model and prior; no error rate beyond the model | `AutoRegressive`, `BayesianChangePoint`, `ExponentialSmoothing`, `HiddenMarkovFilter`, `KalmanFilter`, `LocalLevelKalman`, `ModelAveraging`, `PoissonSmoothing`, `ThompsonAllocation` |
| **approximate** | an approximation even under the model (Monte Carlo, variational, stochastic approximation) | `BootstrapParticleFilter`, `OnlineHMM`, `ParticleFilter`, `RobustKalmanFilter` |
| **none** | a transformation of the data; no inferential claim | `EMA`, `MeanDifference` |
<!-- /guarantees -->

## First: may you look at any time, or only at planned analyses?

| You want to | Use | What it costs |
|---|---|---|
| look after every observation and stop whenever the data are conclusive | an **anytime-valid** procedure (e-process, confidence sequence) | more data at the planned effect than a planned design: for two proportions, 1.3–1.6 times a single analysis |
| look a few times (a committee meets, a release cycle) | a **group sequential** design | an inflation factor of 1.01–1.2 over a single analysis; looking more often than planned is not allowed, but alpha spending lets the times move |
| test one simple hypothesis against another with the fewest observations on average | `SPRT`, `TwoSPRT`, or an optimal plan from `kiefer_weiss_plan` | needs the distribution family and both parameter values |

## Online experiments (A/B tests)

| Metric | Procedure | Guarantee | Tutorial |
|---|---|---|---|
| conversion (0/1) | `TwoProportionTest` | exact, any n | [A/B test](tutorials/ab-test.md) |
| any bounded metric (share, capped revenue) | `BoundedABTest` | exact confidence sequence for the difference | [A/B test §6](tutorials/ab-test.md) |
| mean with a pre-experiment covariate (CUPED) | `ABTest` | asymptotic | [A/B test](tutorials/ab-test.md) |
| several covariates, regression adjustment | `AnytimeLinearModel` | exact under Gaussian errors | — |
| ratio metric (revenue per session, clicks per impression), relative lift | `RatioABTest` | asymptotic, delta method on the randomised unit | [A/B test §8](tutorials/ab-test.md) |
| events per unit of exposure | `RateRatioTest` | exact for Poisson counts, robust to a drifting rate | [A/B test §7](tutorials/ab-test.md) |
| median, 90th percentile of one group (a latency target) | `QuantileTest`, `QuantileConfidenceSequence` | exact, no assumption on the distribution; one group only (a difference of two groups' medians is not covered yet) | — |
| "are A and B distributed the same at all?" | `KernelTwoSampleTest` | exact | — |
| traffic split as intended? | `SampleRatioMismatch` | exact | [A/B test](tutorials/ab-test.md) |
| many metrics or arms at once | `MultipleTests` (e-BH) | false discovery rate at any stopping time | [A/B test](tutorials/ab-test.md) |
| a series of experiments decided one by one | `OnlineFDR` (e-LOND) | false discovery rate over the series | [programme](tutorials/experiment-programme.md) |
| how long will it take, how much power? | `planning.simulate` | simulation | [A/B test §9](tutorials/ab-test.md) |
| send more traffic to the better arm during the test | `ThompsonAllocation` (shares, expected-loss deployment rule); test with `TwoProportionTest(method="universal")` or `BoundedABTest(propensity_input=...)` | the rule bounds expected loss; those tests stay exact under any split | [A/B test §10](tutorials/ab-test.md) |

## Clinical trials

| Endpoint | Planned analyses | Monitored after every event |
|---|---|---|
| normal mean, one sample or paired | `GroupSequentialTest` (known or estimated variance) | `NormalMixtureSPRT`, `TMixtureSPRT` |
| continuous endpoint, two arms | `GroupSequentialTwoMeans` (known or estimated variance) | `AnytimeLinearModel` with the arm as the tested covariate |
| two proportions (response rate) | `GroupSequentialProportions` | `TwoProportionTest` |
| time to event | `GroupSequentialLogRank` (stratified if needed) | `AnytimeLogRankTest` (stratified if needed) |
| any statistic you compute (Cox model, mixed model) | `GroupSequentialMonitor` (Z and information per analysis) | — |

For the design itself, use `group_sequential_design`, which gives:

* efficacy and non-binding futility boundaries;
* power and expected sample size;
* maximum sample sizes for means and proportions, and the number of
  events.

`design.at(...)` recomputes the boundaries when the analyses do not
happen as planned. The boundaries and sample sizes agree with rpact
([tutorial §8](tutorials/group-sequential.md)).

Two cautions:

* **Late entry with time-dependent hazards.** The anytime-valid log-rank
  test needs proportional hazards on the time scale of the records. With
  staggered entry and hazards that depend on time since entry, stratify
  by entry cohort or use the group sequential test. The
  [survival tutorial](tutorials/survival.md) has the numbers.
* **Small trials.** With few patients, the normal approximations of the
  group sequential statistics are approximate. The tests simulate the
  level at the sizes the designs give.

## Monitoring a process

| Situation | Procedure | Guarantee |
|---|---|---|
| known distribution before and after the change | `CUSUM`, `ShiryaevRoberts` | optimal detection delay for the chosen alternative |
| only the mean or a quantile before the change is known | `MeanEDetector`, `QuantileEDetector` | average run length to a false alarm at least the target, for every distribution |
| unknown parameters, several changes, a posterior wanted | `BayesianChangePoint` | Bayesian; sensitive to the model |

The [monitoring tutorial](tutorials/monitoring.md) compares them on the
same data.

## Tracking a hidden state

| Model | Procedure |
|---|---|
| linear Gaussian (trends, autoregressions, tracking) | `KalmanFilter`; outliers: `RobustKalmanFilter` |
| a few hidden regimes, model known | `HiddenMarkovFilter` |
| a few hidden regimes, parameters to learn | `fit_hmm` on a history, or `OnlineHMM` on the stream ([tutorial](tutorials/regimes.md)) |
| anything you can simulate | `ParticleFilter` |
| smoothing after the fact | `sequential_analysis.smoothing` |

## Forecasting

| Situation | Procedure |
|---|---|
| a history to fit, the process stable | `fit_ets` (level, trend, season) or `fit_ar`, then forecast with the fitted model |
| no history, or a process that changes | `averaged_smoothing` (learns the smoothing parameters), `AutoRegressive` without coefficients, with `forgetting` below 1 to follow drift |
| known drivers: promotions, holidays, prices | `regressors=` on any of them |
| counts (orders, calls, incidents) | `fit_counts` or `averaged_counts` (`PoissonSmoothing`) |
| growth, swings proportional to the level | `transform="log"` |
| several candidate models | `ModelAveraging` |
| intervals that must hold whatever the data | `ConformalForecast` around any of them |
| is one forecaster better than another? | `ForecastComparison`, valid at every time |

The [forecasting tutorial](tutorials/forecasting.md) runs both kinds of
training on the same data, before and after a change.

## Means of data that are not normal

* **Bounded data:** `BettingConfidenceSequence`, exact.
* **A known bound on the standard deviation:**
  `CatoniConfidenceSequence`, exact.
* **Anything else:** `AsymptoticConfidenceSequence`, which is valid only
  approximately and in long runs. No method can guarantee coverage of
  the mean of arbitrary unbounded data.
* **The median or another quantile:** the quantile procedures, which
  assume nothing about the distribution.
