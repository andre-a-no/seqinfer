# sequential-analysis — TypeScript executor

Continues runs of the Python library
[sequential-analysis](https://github.com/andre-a-no/sequential-analysis)
in Node.js, and hands them back. It reads and writes the same formats,
specified in SPEC.md:

* checkpoints;
* observation logs;
* delivery and topology states.

## Agreement with the Python library

* **History digests and random states** are identical.
* **Outputs and states** are identical for 18 of the 45 conformance
  cases. For the others they agree to 5e-14 relative, and to 2.2e-13 in
  two cases (a log-evidence near zero that is a difference of large
  log-gamma terms; bounds found by a root search): V8 rounds `Math.log`,
  `log1p`, `exp` and `cos` differently from the C library.
* **Decisions** never differ.
* **Round trip.** A run checkpointed in Python, continued here and
  restored in Python again ends with the digest of the uninterrupted
  run (`tests/test_typescript.py`).

Runs on Node.js 16 or later (its tests need 18).

```bash
npm ci && npm test
node dist/src/cli.js procedures                    # what this executor implements
node dist/src/cli.js verify checkpoint.json        # structure, identity, state
node dist/src/cli.js continue checkpoint.json observations.jsonl --checkpoint next.json --outputs outputs.jsonl
```

```ts
import { restoreFromCheckpoint } from "sequential-analysis";

const run = restoreFromCheckpoint(JSON.parse(text));   // procedure, topology and delivery from the checkpoint
run.start();
for (const obs of observations) run.offer(obs);        // {input, value, source, seq, key, time}
const next = run.checkpoint();                         // continue in Python with Run.restore
```

## What is implemented

The executor has two parts:

* **The run.** Delivery, topologies, contracts, the history digest,
  checkpoints and the observation log.
* **Procedures.** Every procedure of the Python library, as listed by
  `procedures`:
  * SPRT, 2-SPRT and plans;
  * CUSUM, Shiryaev–Roberts, the e-detectors and Bayesian online
    change-point detection (normal-gamma, Poisson-gamma and
    beta-Bernoulli models);
  * the mixture SPRTs, the betting and sign tests, the quantile test and
    the kernel two-sample test;
  * confidence sequences: betting, quantile, Catoni and asymptotic;
  * group sequential tests;
  * the A/B tests, and several tests with e-BH (`MultipleTests`);
  * the local-level and general Kalman filters, the robust (Student t)
    Kalman filter, the hidden Markov filter (categorical, Gaussian and
    Poisson emissions), the bootstrap particle filters (local level;
    linear Gaussian, stochastic volatility and Student t local level),
    EMA and the mean difference.

  Procedures that offer a numpy backend are continued here only when the
  run used `backend="python"`: numpy's `exp`, `log` and sums are not
  specified to the last bit. A checkpoint with `backend="numpy"` is
  refused.

Designs, such as optimal plans and group sequential boundaries, are
computed in Python. They reach this executor as part of the
procedure's configuration.

License: AGPL-3.0-only, or a commercial license (see COMMERCIAL.md in
the repository).
