# sequential-analysis

Sequential statistical inference, separated from the experimental runtime.

> **Procedure infers; consumers act.**

A sequential procedure — an SPRT, a change-point detector, a Kalman
filter, an optimal test from a design code — is a transition function

    (S_t, O_t, R_t) = F(S_{t-1}, X_t, R_{t-1})

and nothing else. sequential-analysis keeps everything around it in separate,
replaceable layers: where the observations come from (adapters), how inputs
are related (topology: pairing, key joins, time alignment), how they are
ordered and deduplicated (delivery), how they are acquired (a synchronous
loop or concurrent producers), how the run survives a crash (checkpoints),
and what is done with the results (consumers). The same procedure runs
unchanged on a file, a live instrument or a simulator, and a resumed run
is bit-for-bit identical to an uninterrupted one.

The library is the reference implementation of the architecture described
in [`paper/main.tex`](https://github.com/andre-a-no/sequential-analysis/blob/main/paper/main.tex) ([PDF](https://github.com/andre-a-no/sequential-analysis/blob/main/paper/main.pdf)). It has no dependencies beyond the
Python standard library (3.10+).

## Installation

```
pip install sequential-analysis          # from PyPI
pip install .                            # from a checkout
pip install "sequential-analysis[fast]"  # with numpy: faster optimal designs
```

The package is imported as `sequential_analysis`.

## A first run

```python
from sequential_analysis import Run, run_sync
from sequential_analysis.procedures import Gaussian, TwoSPRT
from sequential_analysis.sources import gaussian

test = TwoSPRT(Gaussian(sigma=1.0), theta0=0.0, theta1=0.5, alpha0=0.05, alpha1=0.05)
run = Run(test)
run_sync(run, gaussian("x", mean=0.5, sd=1.0, seed=7))
print(run.state.decision, "after", run.t, "observations")

checkpoint = run.checkpoint()          # plain JSON: state, random state, buffers, provenance
```

`examples/` has nine runnable scripts: offline use, paired data from
concurrent producers, checkpoint and resume, operating characteristics,
feedback control, executing a precomputed plan, designing an optimal plan,
nonparametric monitoring, and a group sequential trial.

## Methods

| Kind | Procedures |
|---|---|
| Tests of simple hypotheses | `SPRT` (Wald), `TwoSPRT` (Lorden's 2-SPRT for the Kiefer–Weiss problem), `PlanTest` (executes any precomputed plan) |
| Group sequential trials | `GroupSequentialTest` (planned analyses; efficacy and non-binding futility boundaries; known or estimated variance) |
| Anytime-valid inference | `NormalMixtureSPRT` (mSPRT, known variance), `TMixtureSPRT` (unknown variance); both with always-valid p-value and confidence sequence |
| Nonparametric, anytime-valid | `QuantileTest`, `QuantileConfidenceSequence` (any quantile — median, 90th percentile — of any distribution, ties and atoms included; no moments), `KernelTwoSampleTest` (are A and B equally distributed? any difference: location, scale, shape; by betting), `BettingMeanTest`, `BettingConfidenceSequence` (mean of bounded data: conversion rates, scores), `SequentialSignTest` (P(X > m) = P(X < m); paired data via topology) |
| Means without normality | `CatoniConfidenceSequence` (exact, needs only a bound on the standard deviation; heavy tails and outliers), `AsymptoticConfidenceSequence` (estimated variance; coverage asymptotic, for long runs) |
| Change-point detection | `CUSUM`, `ShiryaevRoberts` |
| Filtering and estimation | `LocalLevelKalman` (with control input), `BootstrapParticleFilter`, `MeanDifference`, `EMA` |

Likelihood-ratio procedures take a family: `Gaussian` (known variance),
`Bernoulli`, `Poisson`, `Exponential`.

**Which method for data that are not normal?** For a median or another
quantile — latencies, revenue per user, anything heavy-tailed — use the
quantile procedures: they assume nothing about the distribution. To ask
whether two groups differ at all, `KernelTwoSampleTest`. For a mean, the
data have to be bounded (`BettingConfidenceSequence`, exact) or have a known
bound on their variance (`CatoniConfidenceSequence`, exact); otherwise
`AsymptoticConfidenceSequence` is valid only approximately and for long
runs — no method can guarantee coverage of a mean of arbitrary unbounded
data. All of them may be monitored after every observation; each
docstring gives the simulated error rates on skewed, heavy-tailed and
discrete data.

**Design** (`sequential_analysis.design`), computed offline before the experiment:

- `operating_characteristic(plan, family, theta)` — error probabilities,
  stopping-time distribution and expected sample size of a plan, without
  simulation: exact for Bernoulli and Poisson data, on a fine lattice
  (error of order h²) for normal and exponential data;
- `kiefer_weiss_plan(theta0, theta1, alpha0, alpha1, horizon, family=...)` —
  the plan with a maximum sample size that minimises the expected sample
  size at θ* (modified Kiefer–Weiss problem), by backward induction, for
  Bernoulli, Poisson, normal and exponential data. With
  `theta_star="least-favourable"` it solves the Kiefer–Weiss problem
  itself: θ* is placed where the maximum expected sample size is attained
  (Lorden's characterisation) — the root of that condition for normal and
  exponential data, the best candidate of a bisection for discrete data,
  where the condition holds only approximately. For response
  rates 30% against 50% at error rates 0.05 it lowers the maximum expected
  sample size from 50.8 (Wald's SPRT calibrated to the same errors) to
  45.2, and to 45.0 with the least favourable θ*. For a normal mean 0
  against 0.5 (horizon 120) it needs 31.0 observations on average at θ*,
  against 31.3 for Lorden's 2-SPRT with thresholds calibrated to the same
  errors: the 2-SPRT is already close to optimal there. A design takes
  about a second for the Bernoulli example and from seconds to a minute
  for normal data on the default lattice. The least favourable θ* takes
  longer: for discrete data it designs about a dozen candidates and
  compares their largest expected sample sizes, some thirty times the time
  of one design; for normal data it finds the root of Lorden's
  characterisation from the precomputed table below, in a few designs. With `numpy`
  installed (optional: `pip install "sequential-analysis[fast]"`) the backward pass on
  wide lattices — normal, exponential, Poisson with larger rates — runs as
  array operations, about ten times faster, with the same plans. Targets that no test with `horizon`
  observations can meet (the Neyman–Pearson bound) are reported at once.

**Least favourable θ\* for normal data** is precomputed:
`tabulated_least_favourable(alpha0, alpha1, rho)` gives the root of Lorden's
characterisation for α₀, α₁ ∈ [0.005, 0.2] to about 3·10⁻⁴ of θ₁ − θ₀
(`rho` is how much longer the horizon is than the shortest fixed-sample
test; the root hardly depends on it above 1.5). `theta_star="least-favourable"`
starts its search from it, which makes it two to four times faster; the
search still verifies the characterisation. The table is computed by
`tools/least_favourable_table.py` on Gauss–Lobatto–Legendre nodes
(D-optimal for polynomial regression) and checked on Halton points not
used in the fit.

**Design cache** (`sequential_analysis.cache`): every plan `kiefer_weiss_plan`
computes is stored on disk and returned from there when the same question
is asked again — the key is every argument, the family, the library version
and the version of the design algorithm. A stored plan is not trusted
blindly: its operating characteristic is recomputed and must reproduce the
stored one and meet the requested error rates, otherwise it is computed
again. Every save, load and ignored file is reported on the logger
`sequential_analysis.cache` (visible by default):

```
design cache: computed kiefer_weiss_plan(0.3, 0.5, 0.05, 0.05, horizon=150, family=bernoulli, theta_star=None) in 1.2 s and saved it to /home/me/.cache/sequential-analysis/kiefer_weiss-<hash>.json
design cache: loaded kiefer_weiss_plan(...) from /home/me/.cache/sequential-analysis/kiefer_weiss-<hash>.json (computed in 1.2 s); error rates verified
```

The directory is `cache=` if given, else `$SEQUENTIAL_ANALYSIS_CACHE`, else
`$XDG_CACHE_HOME/sequential-analysis` or `~/.cache/sequential-analysis`; `cache=False` or
`SEQUENTIAL_ANALYSIS_CACHE=off` turns it off, `clear_design_cache()` empties it. A
cache that cannot be written is reported and never fails the design.

**Group sequential designs** (`sequential_analysis.group_sequential`):
`group_sequential_design(alpha, fractions, spending)` computes efficacy
boundaries for planned analyses from an alpha-spending function
(O'Brien–Fleming and Pocock types, power family, or your own), one- or
two-sided, and gives crossing probabilities, power, expected sample size
and the maximum sample size for a target power. With
`futility=..., power=...` it adds non-binding futility boundaries by beta
spending. The O'Brien–Fleming boundaries for five looks reproduce the
published 4.877, 3.357, 2.680, 2.290, 2.031.

**Inputs**: `sequential_analysis.sources` binds values, files (`from_csv`, with
explicit conversion), simulators and replayed logs to logical inputs.
Transformations between topology and procedure, or between the stages of
a `Chain`, are `Transform` objects (`Difference`, `Field`, or your own)
with a name, a version and a configuration that the checkpoint records.

Every statistical claim in the docstrings — error rates, coverage, run
lengths, optimality — is checked by a test (`tests/test_methods.py`).

## Guarantees

- **Transactional updates.** States are immutable values; a failed step
  leaves the previous state in place.
- **Exact resumption.** A checkpoint holds the inferential state, the
  64-bit random state, topology buffers, source positions, the
  observation log position and provenance. Restoring it and continuing
  gives the same trajectory and history digest as never stopping. A
  damaged or foreign checkpoint is refused with a message naming the
  problem.
- **Invalid input never stops the run by default.** It is skipped,
  counted and reported as a warning on the `sequential_analysis` logger with a
  structured `incident` record. `on_invalid="raise"` or an exception class
  makes it fatal; `max_invalid_streak=N` stops the run after N invalid
  inputs in a row, when the source rather than a message is broken. Contracts are strict: `numpy` scalars, booleans and
  strings are refused with a message saying how to convert them. Observations must be exactly recordable in JSON
  (finite numbers, integers within 2^53, UTF-8 strings, string source names, int sequence numbers); others are
  invalid input, so logs and checkpoints always replay. One that delivery cannot even order (a source that is not a
  string, a sequence number that is not an int) is rejected before delivery and is not logged.
- **Numerical failures are loud.** A transition that overflows or produces a non-finite state or output raises
  `NumericalError` and commits nothing; variance-based tests also raise it when rescaled data make the variance underflow.
- **Portable digests.** History digests and checkpoint identifiers use
  RFC 8785 canonical JSON. `conformance/` holds language-independent test
  vectors; `node conformance/check.mjs` reproduces them in JavaScript.

What sequential-analysis does **not** do: it does not keep observations beyond the
last checkpoint. After a restore they must come again from the sources; a
source that cannot replay them has to be buffered by the application.

## Checkpoints and backends

A checkpoint is a plain JSON value: the procedure's identity (name,
version, configuration), its state, the 64-bit random state, topology
buffers, source positions, the observation log position, counters and
provenance. What "the restored run continues exactly" means, and when it
holds:

- **Same procedure, same arithmetic.** `Run.restore` compares the
  procedure's identity with the checkpoint's and refuses any difference
  (`IncompatibleCheckpoint`). The identity includes the version, which
  changes whenever the arithmetic of a procedure changes, even in the last
  bits: a checkpoint of an older version is refused rather than continued
  differently. (0.2.0 changed the Kalman filter, the t-mixture SPRT and the
  group sequential test.)
- **The backend is part of the identity.** `ParticleFilter`,
  `KernelTwoSampleTest`, `BettingConfidenceSequence` and
  `CatoniConfidenceSequence` take `backend="auto" | "python" | "numpy"`;
  "auto" becomes "numpy" when numpy is installed, and the resolved value is
  stored. A run computed with numpy is continued with numpy; on a machine
  without numpy its restore is refused, not silently continued in Python.
  To move such a run, start a new one from its state
  (`Run(..., initial_state=..., initialized_from=...)`): the numbers carry
  over, the provenance says it is a new run.
- **Bit for bit, and where.** With the Python backend a run depends only
  on IEEE 754 arithmetic, sums in a fixed left-to-right order, and the
  `exp`, `log` and similar functions of the platform's C library (through
  the `math` module) — not on the Python version (3.10 to 3.14 give the
  same bits). The mainstream C libraries agree on these functions in all
  but rare cases; the conformance vectors in `conformance/` pin the bits
  down and are checked in CI, in Python and in JavaScript. With the numpy backend a run is reproducible bit for bit
  on the same machine and numpy version; numpy's `exp`, `log` and sums may
  change their last bits between numpy versions and processors. The
  provenance records the numpy version, and restoring a numpy run under
  another one logs a `numpy_version` incident: the continuation is
  statistically the same, but may differ in the last bits from the
  uninterrupted run. Where results must agree bit for bit across machines,
  choose `backend="python"`.
- **Randomness.** Randomized procedures draw from one SplitMix64 state,
  stored in the checkpoint; the numpy backend draws exactly the same
  uniforms from it, so the particle filter consumes the same random
  numbers in both backends (its results agree to about 1e-12).
- **Size.** Most procedures checkpoint a few hundred bytes whatever the
  length of the run. Those that hold the sample (`QuantileConfidenceSequence`,
  `KernelTwoSampleTest`, `CatoniConfidenceSequence`) grow by about 20 bytes
  per observation, `ParticleFilter` stores every particle, and
  `BayesianChangePoint` one entry per run length kept; see
  [PERFORMANCE.md](https://github.com/andre-a-no/sequential-analysis/blob/main/PERFORMANCE.md). A checkpoint
  costs a JSON encoding of the state, so take them as often as the state
  size allows.
- **Not part of the trajectory.** Wall-clock times in the provenance
  events and a run identifier generated when none is given differ from run
  to run; the history digest and every output do not.

## Speed

A step takes microseconds for the tests, CUSUM and the scalar filters,
about 0.1 ms for a small Kalman filter, and grows with the run for the
procedures that hold the sample. numpy accelerates the array-heavy ones
2 to 12 times (a particle filter with 20 000 particles: 35 ms → 2.7 ms per
step). [PERFORMANCE.md](https://github.com/andre-a-no/sequential-analysis/blob/main/PERFORMANCE.md) has the
measured table for every procedure and backend; `python tools/benchmark.py`
measures them on your machine, and CI reports it on every change.

## Development

```
pip install -e . ruff mypy
python -m unittest discover -s tests -t .
ruff check . && mypy
node conformance/check.mjs
cd paper && latexmk -pdf main.tex   # the paper; needs TeX Live with biber
```

See [CONTRIBUTING.md](https://github.com/andre-a-no/sequential-analysis/blob/main/CONTRIBUTING.md).

## Citing

If you use sequential-analysis in research, please cite it: the
repository's [CITATION.cff](https://github.com/andre-a-no/sequential-analysis/blob/main/CITATION.cff)
gives the details, and GitHub's *Cite this repository* button turns them
into BibTeX or APA. Releases are archived on Zenodo with a DOI.

## License

sequential-analysis is dual-licensed:

- **Free** under the [GNU AGPL v3](https://github.com/andre-a-no/sequential-analysis/blob/main/LICENSE) — for research, teaching
  and development, for open-source projects under the AGPL (or GPL v3), and
  for products and services whose complete source code is published under
  the AGPL.
- **Commercial license** — for closed-source products and for network
  services whose source code is not published. Contact Novikov Laboratories
  LLC; who needs which license is set out in
  [COMMERCIAL.md](https://github.com/andre-a-no/sequential-analysis/blob/main/COMMERCIAL.md).

Authors: Alexander Stepanov, Andrej Novikov.
Copyright © 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation).
