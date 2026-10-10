# Changelog

All notable changes are listed here. The project follows
[semantic versioning](https://semver.org/); until 1.0 a minor version may
change the checkpoint, log and cache formats, which are versioned
separately (`sequential-analysis.checkpoint/2`, `.log/2`, `.history/2`,
`.design-cache/1`).

## 0.1.0 — unreleased

First public release.

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
