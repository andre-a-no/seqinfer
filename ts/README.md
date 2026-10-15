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
* **Outputs and states** are identical for 13 of the 23 conformance
  cases. For the others they agree to 7e-15 relative: V8 rounds `Math.log`,
  `exp` and `cos` differently from the C library.
* **Decisions** never differ.
* **Round trip.** A run checkpointed in Python, continued here and
  restored in Python again ends with the digest of the uninterrupted
  run (`tests/test_typescript.py`).

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
* **Procedures.** The procedures listed by `procedures`:
  * SPRT, 2-SPRT and plans;
  * CUSUM, Shiryaev–Roberts and the e-detectors;
  * the mixture SPRTs and the betting and sign tests;
  * group sequential tests;
  * the A/B tests;
  * the local-level Kalman filter, the bootstrap particle filter, EMA and
    the mean difference.

Designs, such as optimal plans and group sequential boundaries, are
computed in Python. They reach this executor as part of the
procedure's configuration.

License: AGPL-3.0-only, or a commercial license (see COMMERCIAL.md in
the repository).
