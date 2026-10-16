# sequential-analysis: specification of runs, checkpoints, logs and plans

Version 2 of the formats (`sequential-analysis.checkpoint/2`,
`sequential-analysis.history/2`, `sequential-analysis.log/2`), as written
by sequential-analysis 0.4.0 and its TypeScript executor (`ts/`).

This document is what another implementation needs to continue a run
started by this one, or to hand one back: the shape of every stored value
and the semantics of a step.
It is checked by two independent implementations against the
conformance vectors in `conformance/`:

* the Python library, which writes them;
* the TypeScript executor, which reproduces them.

Section 11 explains the comparison. The words MUST and SHOULD are used
as in RFC 2119.

## 1. Values and canonical JSON

All stored values are JSON. Numbers are IEEE 754 binary64.

* **Integers.** An integer must lie within ±2^53. Beyond that, binary64
  no longer holds every integer, so such a value is refused rather than
  rounded.
* **Reading.** An integer literal beyond ±2^53 in stored JSON denotes the
  binary64 number it writes, as `JSON.parse` reads it. ECMAScript's
  `Number.prototype.toString`, and therefore canonical JSON and
  `JSON.stringify`, writes the float 1e17 as `100000000000000000`. A
  reader in a language with separate integers (Python's `json`) MUST read
  such a literal as that float. The refusal above applies to integers
  that are meant as integers: counts, sequence numbers, step counts.
* **Numbers in general.** NaN and the infinities have no JSON
  representation. A state or output that contains one is a numerical
  failure of the step (section 6).
* **Wide integers.** 64-bit integers are stored as strings: the random
  state as 16 lowercase hexadecimal digits, the seed in decimal.

Hashes are taken over the **canonical JSON** of a value: RFC 8785, the
JSON Canonicalization Scheme.

* Numbers are written as ECMAScript's `Number.prototype.toString` writes
  them. -0 is written `0`, and 1 and 1.0 are the same number.
* Strings are escaped as `JSON.stringify` escapes them. Text with lone
  surrogates is refused.
* Object keys are sorted by their UTF-16 code units. There is no
  whitespace.

`conformance/numbers.json` holds 500 binary64 values, as bit patterns,
with their canonical text.

SHA-256 is written as 64 lowercase hexadecimal digits.

## 2. Observations

An observation is what an adapter delivers.

```
{"input": str, "value": V, "source": str, "seq": int | null, "key": str | int | null, "time": number | null}
```

* **`source`** defaults to `input` when it is absent or empty.

An observation that delivery cannot handle is invalid input (section 6)
and is never delivered. That is the case when:

* `input` or `source` is not a string; or
* `seq` is neither null nor an integer within 2^53.

A delivered observation that cannot be *recorded* exactly is also
invalid input:

* `value` is something other than null, a boolean, a finite number, a
  string, or arrays and objects of these;
* `key` is something other than null, a string or an integer within
  2^53; or
* `time` is something other than null or a finite number.

## 3. Statistical inputs and contracts

A statistical input `X_t` is a non-empty JSON object that maps logical
input names to values. A procedure declares a contract for each input:

```
{"name": str, "kind": "real" | "integer" | "binary", "shape": [] | [d], "lower": number | null, "upper": number | null, "unit": str | null}
```

An input is valid when all of the following hold:

* every name in it is declared;
* if the procedure has `joint_inputs`, every declared name is present;
* every value is a finite number, or for `shape: [d]` an array of d
  finite numbers;
* every number satisfies the kind: an integer value for `integer`, 0 or
  1 for `binary`;
* every number satisfies the inclusive bounds.

Booleans and strings are not numbers.

## 4. Delivery

A delivery spec is:

```
{"policy": "arrival" | "sequence" | "strict", "dedup": bool, "start": int}
```

Delivery keeps, per source, the next expected sequence number. The
number starts at `start`.

* **arrival** delivers at once. With `dedup`, an observation whose `seq`
  is below the position, or among the numbers already delivered ahead
  of it, is dropped and counted as a duplicate. Without `dedup` it is
  delivered again.
* **sequence** holds observations back until the gap before them is
  filled. Then it delivers them in `seq` order. A duplicate is dropped
  and counted with `dedup`, and is an ordering error without it.
* **strict** requires every `seq` to be the next one. A number below the
  position is a duplicate: it is dropped and counted with `dedup`, and is
  an ordering error without it. Any other number is an ordering error.

Observations without `seq` are delivered at once under `arrival`, and
are an ordering error under the other policies.

Delivery state:

```
{"next": {source: int}, "ahead": {source: [int, ...ascending]}, "duplicates": int}
```

`ahead` lists the numbers delivered beyond a gap (arrival policy).
Observations held back by the sequence policy are not part of the state:
after a restore they must come again from the source.

An implementation MAY let adapters declare sequence numbers that do not
exist, such as Kafka offset gaps. In this implementation that is
`Delivery.skip_range(source, first, stop)`. It moves the position past
them. A declaration ahead of undelivered numbers is not part of the
state.

## 5. Topologies

A topology turns delivered observations into statistical inputs. Each
spec and state is listed below, with what the topology emits.

**`{"type": "independent"}`** (state: `null`)

* Each observation `o` emits `{o.input: o.value}`.

**`{"type": "positional_pair", "inputs": [a, b, ...]}`** (state: `{name: [values...]}`)

* Appends the observation to the queue of its input.
* Emits an input made of the heads of the queues, as long as no queue is
  empty.

**`{"type": "key_join", "inputs": [...]}`** (state: `[[key, {name: value}], ...]`, in insertion order)

* Collects values per key, and emits them once every input is present.
* Seeing an input twice for one key is an ordering error.
* A missing key is invalid input.

**`{"type": "time_align", "inputs": [a, b], "tolerance": t}`** (state: `{"queues": {name: [[time, value], ...]}, "last": {name: time | null}, "unmatched": int}`)

* Times must not decrease per input; a decrease is an ordering error.
* Heads within `t` of each other are paired.
* Otherwise the earlier head is discarded and counted in `unmatched`.

**`{"type": "mapped", "map": {"name", "version", "config"}, "inner": spec}`** (state: the inner state)

* Applies the transformation to every emitted input.
* A failure of the transformation makes that input invalid.
* The built-in `difference/1` with config `{a, b, out}` maps `{a, b}` to
  `{out: a - b}`.

An observation for an input the topology does not know is invalid input.

## 6. A step and the history digest

A run holds:

* the procedure state `S_t`;
* the random state `R_t` (randomized procedures only);
* `t`, the history digest, the `terminal` flag and counters.

For each delivered observation, in order:

1. If the topology emits nothing, the step ends here. Otherwise, for each
   emitted input `X`, apply steps 2–7.
2. If the run is terminal, increment `after_terminal` and stop.
3. Validate `X` (section 3). An invalid input is handled by the
   `on_invalid` policy:
   * **`skip`**: count it in `invalid_skipped` and in `invalid_streak`,
     and fail the run when `max_invalid_streak` is reached.
   * **`raise`** or **`raise:<name>`**: fail the run.
4. Compute `(S', O) = F(S_{t-1}, X, R)`.
5. If `S'` or `O` contains a non-finite number, or an integer-valued
   count beyond ±2^53, fail the run. Nothing is committed. Such a value
   could be neither checkpointed nor decoded exactly. The Python library
   checks integers by type. An executor without separate integers checks
   the fields its procedures declare as counts.
6. Compute `digest_t = SHA-256(digest_{t-1} ‖ canonical(X))`. Here
   `digest_0 = SHA-256("sequential-analysis.history/2")`, and ‖
   concatenates the hex digest with the canonical text.
7. Commit `S'`, `R`, `t + 1`, the digest and `terminal = is_terminal(S')`.
   Then reset `invalid_streak` and notify consumers. A consumer's failure
   is recorded in `consumer_errors` and does not affect the run.

The history digest depends only on the sequence of statistical inputs.
It is the same in every implementation.

## 7. Statistical random state

Randomized procedures draw from SplitMix64. The state is one 64-bit
integer:

```
state += 0x9E3779B97F4A7C15
z = state
z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z ^ (z >> 27)) * 0x94D049BB133111EB
z = z ^ (z >> 31)
```

All operations are modulo 2^64.

* **Seeding.** `R_0` is the first 8 bytes, big-endian, of
  `SHA-256("<seed>/statistical")`, with the seed in decimal.
* **Uniform.** A uniform draw is `(z >> 11) · 2^-53`.
* **Normal.** A normal draw uses two uniforms, u1 and u2, without
  caching:
  `mean + sd · sqrt(-2 ln(1 - u1)) · cos(2π u2)`.

## 8. Checkpoints

A checkpoint is a JSON object:

| field | type | meaning |
|---|---|---|
| `format` | `"sequential-analysis.checkpoint/2"` | |
| `run_id` | non-empty string | |
| `t` | integer ≥ 0 | number of committed steps |
| `terminal` | boolean | equals `is_terminal(state)` |
| `procedure` | `{"name", "version", "config"}` | the procedure's identity; determines F |
| `state` | JSON | the procedure state, in the procedure's encoding |
| `rng` | 16 hex digits or null | `R_t`; null exactly for deterministic procedures |
| `topology` | `{"spec", "state"}` | section 5 |
| `delivery` | `{"spec", "state"}` | section 4 |
| `history_digest` | 64 hex digits | section 6 |
| `log` | null or `{"path", "sync", "count", "offset", "digest"}` | section 9 |
| `counters` | `{invalid_skipped, invalid_streak, after_terminal, dropped_by_runtime}` | non-negative integers |
| `consumer_errors` | `[{t, consumer, error}]` | |
| `provenance` | object | see below |

`provenance` MUST contain:

* `seed`: a decimal string or null;
* `on_invalid`;
* `events`: a list of `{event, at, t, ...}`.

It SHOULD contain:

* `run_id`, `created_at`, `procedure`, `inputs`, `topology`, `delivery`,
  `initialized_from`, `max_invalid_streak`;
* `sequential_analysis` (the version);
* `numerics`, which describes the arithmetic of the implementation that
  created the run.

Other fields are kept as they are.

The **checkpoint id** is the SHA-256 of the canonical JSON of the whole
checkpoint.

**Restoring.** An implementation MUST:

1. Check the structure above. The whole value must be canonical-JSON
   encodable.
2. Compare these, as canonical JSON, with the ones it is given or
   rebuilds, and refuse on any difference:
   * the procedure identity;
   * the topology spec;
   * the delivery spec.
3. Decode the state, and refuse it unless:
   * it has exactly the procedure's fields, with the right types;
   * it encodes back to the same JSON;
   * `terminal` agrees with it.
4. Restore the topology and delivery states, and refuse them unless they
   encode back to the same JSON.
5. Verify and cut back the observation log, if there is one (section 9).
6. Continue with the recorded `run_id`, `t`, digest, counters, consumer
   errors and provenance. Append to `events` a
   `{"event": "restore", "checkpoint": <checkpoint id>, "log": path | null, "log_discarded": n}`
   event.

The run is then paused and resumes on the next observation. The first
mismatch is reported as an incompatible checkpoint, and nothing is
changed.

## 9. Observation log

The log holds the delivered observations: one line per observation, in
delivery order.

* Each line is a JSON object with the six fields of section 2, followed
  by `\n` and encoded in UTF-8.
* The line layout is free, because the chain covers the bytes as
  written. This implementation writes sorted keys with `", "` and `": "`
  separators and ASCII escapes.
* An observation that cannot be recorded is written with its `value`
  replaced by `{"$unrecordable": <description>}`. A replay rejects it
  where the run did.

The position of the log is:

* `count`, the number of lines;
* `offset`, the number of bytes;
* `digest`, the hash chain
  `d_0 = SHA-256("sequential-analysis.log/2")`,
  `d_i = SHA-256(ascii(d_{i-1}) ‖ bytes(line_i))`.

On restore, the first `count` lines are verified against the position.
Lines beyond it are deleted, and their number is reported in the restore
event.

## 10. Sampling plans

`plan_test/1` runs a test given as a plan rather than as a formula.
The config is:

```
{"continuation": [[lower | null, upper | null], ...], "reject_high": bool, "label": str, "input": str, "contract": <contract>}
```

At step n, with running sum `S_n`:

* `S_n ≤ lower_n` decides for the low side;
* otherwise `S_n ≥ upper_n` decides for the high side;
* `reject_high` maps high to `reject_h0`.

The sum is a Neumaier-compensated sum. The last step must have
`lower ≥ upper`. Plans are how designs computed elsewhere, such as an
optimal 2-SPRT or a group-sequential design, travel between
implementations: the executor needs only the boundaries.

## 11. Procedures and conformance

A procedure is a name, a version and a config. Its transition F is
fixed by them. A change to F that alters any output requires a new
version.

The state and output encodings of each procedure are those of the
reference implementation. `conformance/procedures.json` pins them down:
for each case, it holds:

* the identity;
* the inputs;
* every output;
* the final state, random state and terminal flag;
* the history digest.

An implementation conforms when, for every case it implements:

* the history digest is identical;
* integers, strings, booleans, nulls and the structure are identical;
* numbers are identical, or agree to a relative 1e-12. A
  Neumaier compensation term is compared on the scale of the sum it
  belongs to: it is that sum's low-order part.

The tolerance exists because exp, log and the trigonometric functions
are not correctly rounded in the C library or in JavaScript engines.
Sums and products are reproduced exactly when they are evaluated in the
reference order. Every procedure's sums are left-to-right or
Neumaier-compensated, never pairwise. Decisions are compared exactly.

In the TypeScript executor, 28 of the 92 cases agree bit for bit. The
others differ in the last bits only: by at most 5e-14 relative, except
in twelve cases. Every such difference was traced to `Math.log`, `log1p`,
`exp`, `expm1` or `cos` rounding differently from the C library. In
six of the seven a small value is a difference of much larger terms,
so a last-bit difference in those terms is large relative to it: in
`quantile_test` (1.6e-13) a log-evidence near zero is a difference of
log-gamma terms of a few hundred; in `anytime_logrank_test_grow`
(3.6e-13), `anytime_logrank_test_stratified` (4.5e-13) and
`anytime_logrank_test_stratified_grow` (1.4e-13) the log e-value passes
0.0025, 0.001 and -0.017, sums of log likelihood ratio factors of both
signs; in `rate_ratio_test` (7.7e-14) the log e-value of one grid point
is 0.017, a sum of per-pair terms of both signs; in `online_hmm_student_t`
(6.4e-14) an estimated mean of -1.2e-4 is a weighted average of
observations of both signs. In
`catoni_confidence_sequence` (2.1e-13) the bounds are roots found to a
relative 1e-12, and the search stops at a slightly different point. In
`model_averaging` (8.3e-14), `averaged_smoothing` (5.9e-13) and
`averaged_log_and_raw` (8.9e-14) the interval ends are quantiles of a
mixture of forecasts, found by Newton steps to a relative 1e-13; a
last-bit difference in the t distribution function moves the point
where the search stops. In `averaged_counts` (8.0e-14) the mixture
weights are exponentials of differences of log-likelihoods built from
log-gamma terms. In `thompson_allocation` (9.4e-14) the probabilities
that each arm is best are quadratures of products of Beta densities and
distribution functions, which carry the last bits of `log`, `exp` and
the incomplete beta function. No
decision differs.

The TypeScript executor implements the procedures listed by
`sequential-analysis procedures` (`ts/`). Its tests:

* reproduce every vector;
* move a run from Python to TypeScript and back mid-way
  (`tests/test_typescript.py`).
