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
in [`paper/main.tex`](paper/main.tex). It has no dependencies beyond the
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

`examples/` has eight runnable scripts: offline use, paired data from
concurrent producers, checkpoint and resume, operating characteristics,
feedback control, executing a precomputed plan, designing an optimal plan,
and nonparametric monitoring.

## Methods

| Kind | Procedures |
|---|---|
| Tests of simple hypotheses | `SPRT` (Wald), `TwoSPRT` (Lorden's 2-SPRT for the Kiefer–Weiss problem), `PlanTest` (executes any precomputed plan) |
| Anytime-valid inference | `NormalMixtureSPRT` (mSPRT: always-valid p-value and confidence sequence) |
| Nonparametric, anytime-valid | `BettingMeanTest` (mean of bounded data, any distribution), `SequentialSignTest` (median, no moment assumptions; paired data via topology) |
| Change-point detection | `CUSUM`, `ShiryaevRoberts` |
| Filtering and estimation | `LocalLevelKalman` (with control input), `BootstrapParticleFilter`, `MeanDifference`, `EMA` |

Likelihood-ratio procedures take a family: `Gaussian` (known variance),
`Bernoulli`, `Poisson`, `Exponential`.

**Design** (`seqinfer.design`), computed offline before the experiment:

- `operating_characteristic(plan, family, theta)` — exact error
  probabilities, stopping-time distribution and expected sample size of a
  plan, for Bernoulli and Poisson data, without simulation;
- `kiefer_weiss_plan(theta0, theta1, alpha0, alpha1, horizon)` — the plan
  with a maximum sample size that minimises the expected sample size at
  the least favourable point (modified Kiefer–Weiss problem), by backward
  induction, for binary outcomes. For 30% against 50% at error rates 0.05
  it lowers the maximum expected sample size from 52.9 (Wald's SPRT
  calibrated to the same errors) to 45.1.

Every statistical claim in the docstrings — error rates, coverage, run
lengths, optimality — is checked by a test (`tests/test_methods.py`).

## Guarantees

- **Transactional updates.** States are immutable values; a failed step
  leaves the previous state in place.
- **Exact resumption.** A checkpoint holds the inferential state, the
  64-bit random state, topology buffers, source positions, the
  observation log position and provenance. Restoring it and continuing
  gives the same trajectory and history digest as never stopping.
- **Invalid input never stops the run by default.** It is skipped,
  counted and reported as a warning on the `seqinfer` logger with a
  structured `incident` record. `on_invalid="raise"` or an exception class
  makes it fatal. Contracts are strict: `numpy` scalars, booleans and
  strings are refused with a message saying how to convert them.
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
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

seqinfer is dual-licensed. It is free software under the
[GNU AGPL v3](LICENSE). Use in proprietary software or in services whose
source is not published requires a commercial license from
Novikov Laboratories LLC — see [COMMERCIAL.md](COMMERCIAL.md).

Authors: Alexander Stepanov, Andrej Novikov.
Copyright © 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation).
