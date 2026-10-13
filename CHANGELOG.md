# Changelog

All notable changes are listed here. The project follows
[semantic versioning](https://semver.org/); until 1.0 a minor version may
change the checkpoint, log and cache formats, which are versioned
separately (`sequential-analysis.checkpoint/2`, `.log/2`, `.history/2`,
`.design-cache/1`).

## 0.1.1 — 2026-10-13

A line-by-line review of the whole package after the first release. No
format changes; the Kalman filter, the t-mixture SPRT and the group
sequential test have new versions (2, 3 and 4), so their checkpoints from
0.1.0 are refused rather than continued with different arithmetic.

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
