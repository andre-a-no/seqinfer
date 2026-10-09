# seqinfer

Sequential statistical inference, separated from the experimental runtime.

> **Procedure infers; consumers act.**

A sequential procedure — an SPRT, a change-point detector, a Kalman
filter, an optimal test from a design code — is a transition function

    (S_t, O_t, R_t) = F(S_{t-1}, X_t, R_{t-1})

and nothing else. seqinfer keeps everything around it in separate,
replaceable layers: where the observations come from (adapters), how inputs
are related (topology: pairing, key joins, time alignment), how they are
ordered and deduplicated (delivery), how they are acquired (a synchronous
loop or concurrent producers), how the run survives a crash (checkpoints),
and what is done with the results (consumers). The same procedure runs
unchanged on a file, a live instrument or a simulator, and a resumed run
is bit-for-bit identical to an uninterrupted one.

The library is the reference implementation of the architecture described
in [`paper/main.tex`](paper/main.tex) ([PDF](paper/main.pdf)). It has no dependencies beyond the
Python standard library (3.10+).

## Installation

```
pip install .
```

## A first run

```python
from seqinfer import Run, run_sync
from seqinfer.procedures import Gaussian, TwoSPRT
from seqinfer.sources import gaussian

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
| Nonparametric, anytime-valid | `BettingMeanTest` (mean of bounded data, any distribution), `SequentialSignTest` (median, no moment assumptions; paired data via topology) |
| Change-point detection | `CUSUM`, `ShiryaevRoberts` |
| Filtering and estimation | `LocalLevelKalman` (with control input), `BootstrapParticleFilter`, `MeanDifference`, `EMA` |

Likelihood-ratio procedures take a family: `Gaussian` (known variance),
`Bernoulli`, `Poisson`, `Exponential`.

**Design** (`seqinfer.design`), computed offline before the experiment:

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
  (Lorden's characterisation). For response
  rates 30% against 50% at error rates 0.05 it lowers the maximum expected
  sample size from 52.9 (Wald's SPRT calibrated to the same errors) to
  45.2, and to 45.0 with the least favourable θ*. For a normal mean 0
  against 0.5 it needs 31.0 observations on average at θ*, against 31.3
  for Lorden's 2-SPRT with thresholds calibrated to the same errors: the
  2-SPRT is already close to optimal there. Seconds for discrete data,
  under ten seconds for normal data. Targets that no test with `horizon`
  observations can meet (the Neyman–Pearson bound) are reported at once.

**Group sequential designs** (`seqinfer.group_sequential`):
`group_sequential_design(alpha, fractions, spending)` computes efficacy
boundaries for planned analyses from an alpha-spending function
(O'Brien–Fleming and Pocock types, power family, or your own), one- or
two-sided, and gives crossing probabilities, power, expected sample size
and the maximum sample size for a target power. With
`futility=..., power=...` it adds non-binding futility boundaries by beta
spending. The O'Brien–Fleming boundaries for five looks reproduce the
published 4.877, 3.357, 2.680, 2.290, 2.031.

**Inputs**: `seqinfer.sources` binds values, files (`from_csv`, with
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
  counted and reported as a warning on the `seqinfer` logger with a
  structured `incident` record. `on_invalid="raise"` or an exception class
  makes it fatal; `max_invalid_streak=N` stops the run after N invalid
  inputs in a row, when the source rather than a message is broken. Contracts are strict: `numpy` scalars, booleans and
  strings are refused with a message saying how to convert them. Observations must be exactly recordable in JSON
  (finite numbers, integers within 2^53, string keys); others are invalid input, so logs and checkpoints always replay.
- **Numerical failures are loud.** A transition that overflows or produces a non-finite state or output raises
  `NumericalError` and commits nothing; variance-based tests also raise it when rescaled data make the variance underflow.
- **Portable digests.** History digests and checkpoint identifiers use
  RFC 8785 canonical JSON. `conformance/` holds language-independent test
  vectors; `node conformance/check.mjs` reproduces them in JavaScript.

What seqinfer does **not** do: it does not keep observations beyond the
last checkpoint. After a restore they must come again from the sources; a
source that cannot replay them has to be buffered by the application.

## Development

```
pip install -e . ruff mypy
python -m unittest discover -s tests -t .
ruff check . && mypy
node conformance/check.mjs
cd paper && latexmk -pdf main.tex   # the paper; needs TeX Live with biber
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

seqinfer is dual-licensed. It is free software under the
[GNU AGPL v3](LICENSE). Use in proprietary software or in services whose
source is not published requires a commercial license from
Novikov Laboratories LLC — see [COMMERCIAL.md](COMMERCIAL.md).

Authors: Alexander Stepanov, Andrej Novikov.
Copyright © 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation).
