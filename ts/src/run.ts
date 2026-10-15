// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Run: one execution of a procedure, its checkpoints and their restoration (SPEC.md, "Runs" and "Checkpoints").
import { randomUUID } from "node:crypto";
import { canonicalJson, normalize, sameJson, sha256Hex, type Json } from "./canonical.js";
import { validateInput, type StatInput } from "./contracts.js";
import { Delivery } from "./delivery.js";
import {
  ContractViolation,
  IncompatibleCheckpoint,
  InvalidInputStreak,
  LifecycleError,
  NumericalError,
} from "./errors.js";
import { ObservationLog, type LogPosition } from "./log.js";
import { complete, envelopeProblem, recordingProblem, type Observation } from "./observation.js";
import { identity, type Output, type Procedure, type State } from "./procedure.js";
import { SplitMix64 } from "./rng.js";
import { Independent, InvalidInput, type Topology } from "./topology.js";
import { VERSION } from "./version.js";

export const CHECKPOINT_FORMAT = "sequential-analysis.checkpoint/2";
export const HISTORY_GENESIS = sha256Hex("sequential-analysis.history/2");

export type RunStatus = "created" | "running" | "paused" | "completed" | "failed";
export type OnInvalid = "skip" | "raise" | `raise:${string}`;

export interface OutputEvent {
  runId: string;
  t: number;
  output: Output;
  terminal: boolean;
}

export type Consumer = (event: OutputEvent) => void;

export interface RunOptions {
  topology?: Topology;
  delivery?: Delivery;
  consumers?: Consumer[];
  seed?: bigint | number | string | null;
  runId?: string;
  initialState?: State;
  initializedFrom?: Json;
  onInvalid?: OnInvalid;
  maxInvalidStreak?: number | null;
  log?: ObservationLog | null;
  clock?: () => number;
}

const COUNTERS = ["invalid_skipped", "invalid_streak", "after_terminal", "dropped_by_runtime"];
const HEX64 = /^[0-9a-f]+$/;

/** Check the structure of a checkpoint; throw IncompatibleCheckpoint naming the first problem. */
export function validateCheckpoint(cp: unknown): asserts cp is Record<string, Json> {
  const bad = (what: string) => new IncompatibleCheckpoint(`malformed checkpoint: ${what}`);
  if (cp === null || typeof cp !== "object" || Array.isArray(cp)) throw bad("expected a JSON object");
  const c = cp as Record<string, Json>;
  if (c.format !== CHECKPOINT_FORMAT) throw new IncompatibleCheckpoint(`unsupported checkpoint format ${JSON.stringify(c.format)}`);
  const types: Record<string, string[]> = {
    format: ["string"], run_id: ["string"], t: ["integer"], terminal: ["boolean"], procedure: ["object"],
    rng: ["string", "null"], topology: ["object"], delivery: ["object"], history_digest: ["string"],
    counters: ["object"], provenance: ["object"],
  };
  const typeOf = (v: Json): string =>
    v === null ? "null" : Array.isArray(v) ? "array" : typeof v === "number" ? (Number.isInteger(v) ? "integer" : "number") : typeof v;
  for (const [name, allowed] of Object.entries(types)) {
    if (!(name in c)) throw bad(`missing field '${name}'`);
    if (!allowed.includes(typeOf(c[name]))) throw bad(`field '${name}' has type ${typeOf(c[name])}`);
  }
  if (!("state" in c)) throw bad("missing field 'state'");
  if ((c.t as number) < 0) throw bad("negative step count");
  if (!c.run_id) throw bad("empty run_id");
  const rng = c.rng as string | null;
  if (rng !== null && (rng.length !== 16 || !HEX64.test(rng))) throw bad("random state is not 16 hexadecimal digits");
  const digest = c.history_digest as string;
  if (digest.length !== 64 || !HEX64.test(digest)) throw bad("history digest is not a SHA-256 hex digest");
  for (const part of ["topology", "delivery"]) {
    const p = c[part] as Record<string, Json>;
    if (!("spec" in p && "state" in p)) throw bad(`${part} needs 'spec' and 'state'`);
  }
  const prov = c.provenance as Record<string, Json>;
  for (const name of ["seed", "on_invalid", "events"]) if (!(name in prov)) throw bad(`provenance lacks '${name}'`);
  if (typeof prov.on_invalid !== "string" || !Array.isArray(prov.events)) throw bad("provenance fields have the wrong type");
  if (!["skip", "raise"].includes(prov.on_invalid) && !prov.on_invalid.startsWith("raise:"))
    throw bad(`unknown on_invalid policy ${JSON.stringify(prov.on_invalid)}`);
  if ((prov.run_id ?? c.run_id) !== c.run_id) throw bad("provenance names another run");
  if (prov.seed !== null && !(typeof prov.seed === "string" && /^-?[0-9]+$/.test(prov.seed))) throw bad("seed is not a decimal string");
  const streak = prov.max_invalid_streak;
  if (streak !== undefined && streak !== null && !(Number.isInteger(streak) && (streak as number) >= 1))
    throw bad("max_invalid_streak must be a positive integer or null");
  const log = c.log;
  if (log !== undefined && log !== null) {
    const l = log as Record<string, Json>;
    if (typeof l !== "object" || !["path", "count", "offset", "digest"].every((k) => k in l)) throw bad("log position needs path, count, offset and digest");
    if (l.path !== null && typeof l.path !== "string") throw bad("log path must be a string or null");
    for (const name of ["count", "offset"]) if (!(Number.isInteger(l[name]) && (l[name] as number) >= 0)) throw bad(`log ${name} must be a non-negative integer`);
    if (typeof l.digest !== "string" || l.digest.length !== 64 || !HEX64.test(l.digest)) throw bad("log digest is not a SHA-256 hex digest");
    if (![undefined, null, "flush", "fsync"].includes(l.sync as string)) throw bad("log sync must be 'flush' or 'fsync'");
  }
  const counters = c.counters as Record<string, Json>;
  if (!Object.values(counters).every((v) => Number.isInteger(v) && (v as number) >= 0)) throw bad("counters must be non-negative integers");
  const unknown = Object.keys(counters).filter((k) => !COUNTERS.includes(k)).sort();
  if (unknown.length) throw bad(`unknown counters ${unknown}`);
  const errors = c.consumer_errors ?? [];
  if (!Array.isArray(errors) || !errors.every((e) => e !== null && typeof e === "object" && !Array.isArray(e)))
    throw bad("consumer_errors must be a list of objects");
  try {
    canonicalJson(c);
  } catch (error) {
    throw bad(`not canonical JSON (${(error as Error).message})`);
  }
}

/** Content hash of a checkpoint. */
export function checkpointId(cp: Json): string {
  return sha256Hex(canonicalJson(cp));
}

export class Run {
  readonly procedure: Procedure;
  readonly topology: Topology;
  readonly delivery: Delivery;
  consumers: Consumer[];
  onInvalid: OnInvalid;
  maxInvalidStreak: number | null;
  log: ObservationLog | null;
  runId: string;
  state: State;
  rngState: bigint | null;
  t = 0;
  status: RunStatus = "created";
  terminal: boolean;
  historyDigest = HISTORY_GENESIS;
  counters: Record<string, number> = { invalid_skipped: 0, invalid_streak: 0, after_terminal: 0, dropped_by_runtime: 0 };
  consumerErrors: Json[] = [];
  provenance: Record<string, Json>;
  private clock: () => number;
  private busy = false;

  constructor(procedure: Procedure, options: RunOptions = {}) {
    const onInvalid = options.onInvalid ?? "skip";
    if (!(onInvalid === "skip" || onInvalid === "raise" || onInvalid.startsWith("raise:"))) throw new Error("on_invalid must be 'skip', 'raise' or 'raise:<name>'");
    const streak = options.maxInvalidStreak ?? null;
    if (streak !== null && !(Number.isInteger(streak) && streak >= 1)) throw new Error("max_invalid_streak must be a positive integer or null");
    if (options.runId !== undefined && !options.runId) throw new Error("run_id must be a non-empty string");
    if (options.log && !options.log.isEmpty()) throw new Error("a new run needs an empty log; to continue a run, use Run.restore");
    this.procedure = procedure;
    this.topology = options.topology ?? new Independent();
    this.delivery = options.delivery ?? new Delivery();
    this.consumers = options.consumers ?? [];
    this.onInvalid = onInvalid;
    this.maxInvalidStreak = streak;
    this.log = options.log ?? null;
    this.clock = options.clock ?? (() => Date.now() / 1000);
    this.runId = options.runId ?? randomUUID().replaceAll("-", "");
    this.state = options.initialState ?? procedure.initialState();
    const seed = options.seed ?? null;
    if (procedure.randomized && seed === null) throw new Error(`${procedure.name} is randomized: an explicit seed is required`);
    this.rngState = procedure.randomized ? SplitMix64.forRole(seed!, "statistical").state : null;
    this.terminal = procedure.isTerminal(this.state);
    this.provenance = {
      run_id: this.runId,
      created_at: this.clock(),
      procedure: identity(procedure) as unknown as Json,
      inputs: Object.fromEntries(Object.entries(procedure.inputs()).map(([n, c]) => [n, c.spec()])),
      topology: this.topology.spec(),
      delivery: this.delivery.spec(),
      seed: seed === null ? null : BigInt(seed).toString(),
      initialized_from: normalize(options.initializedFrom ?? "default"),
      on_invalid: onInvalid,
      max_invalid_streak: streak,
      sequential_analysis: VERSION,
      numerics: { float: "IEEE 754 binary64", summation: "Neumaier", implementation: "TypeScript", node: process.version },
      events: [],
    };
  }

  // ------------------------------------------------------------ lifecycle

  private event(name: string, extra: Record<string, Json> = {}): void {
    (this.provenance.events as Json[]).push({ event: name, at: this.clock(), t: this.t, ...extra });
  }

  start(runtime: string | null = null): void {
    if (this.status !== "created" && this.status !== "paused") throw new LifecycleError(`cannot start a run that is ${this.status}`);
    this.event(this.status === "created" ? "start" : "resume", { runtime });
    this.status = "running";
  }

  pause(): void {
    if (this.busy) throw new LifecycleError("re-entrant call: consumers must act on sources, not on the run");
    if (this.status !== "running") throw new LifecycleError(`cannot pause a run that is ${this.status}`);
    this.status = "paused";
    this.event("pause");
  }

  close(): void {
    if (this.busy) throw new LifecycleError("re-entrant call: consumers must act on sources, not on the run");
    if (this.status === "completed" || this.status === "failed") throw new LifecycleError(`run is already ${this.status}`);
    this.status = "completed";
    this.event("close", { terminal: this.terminal });
    this.log?.close();
  }

  private fail(error: Error): void {
    this.status = "failed";
    this.event("fail", { error: `${error.name}: ${error.message}` });
  }

  private enter(): void {
    if (this.busy) throw new LifecycleError("re-entrant call: consumers must act on sources, not on the run");
    if (this.status === "created") this.start();
    if (this.status !== "running") throw new LifecycleError(`run is ${this.status}, not running`);
  }

  // ------------------------------------------------------------ inference

  /** Hand one arriving observation to the run: delivery, topology, validation, transition, commit, notification. */
  offer(obs: Observation): OutputEvent[] {
    this.enter();
    this.busy = true;
    try {
      const events: OutputEvent[] = [];
      try {
        const problem = envelopeProblem(obs);
        if (problem !== null) {
          this.invalid(new ContractViolation(problem));
          return events;
        }
        for (const delivered of this.delivery.push(obs)) {
          this.log?.append(delivered);
          const recording = recordingProblem(delivered);
          if (recording !== null) {
            this.invalid(new ContractViolation(recording));
            continue;
          }
          let inputs: (StatInput | InvalidInput)[];
          try {
            inputs = this.topology.push(delivered);
          } catch (error) {
            if (error instanceof ContractViolation) {
              this.invalid(error);
              continue;
            }
            throw error;
          }
          for (const x of inputs) {
            if (x instanceof InvalidInput) {
              this.invalid(x.violation);
              continue;
            }
            const event = this.apply(x);
            if (event !== null) events.push(event);
          }
        }
      } catch (error) {
        if (this.status !== "failed") this.fail(error as Error);
        throw error;
      }
      return events;
    } finally {
      this.busy = false;
    }
  }

  offerBatch(batch: Iterable<Observation>): OutputEvent[] {
    const events: OutputEvent[] = [];
    for (const obs of batch) events.push(...this.offer(obs));
    return events;
  }

  /** Apply one statistical input directly, bypassing delivery and topology (not with a log attached). */
  step(x: StatInput): OutputEvent | null {
    if (this.log !== null) throw new LifecycleError("step() bypasses the observation log; use offer(), or a run without a log");
    this.enter();
    this.busy = true;
    try {
      try {
        if (x === null || typeof x !== "object" || Array.isArray(x)) {
          this.invalid(new ContractViolation("a statistical input must be a mapping"));
          return null;
        }
        return this.apply({ ...x });
      } catch (error) {
        if (this.status !== "failed") this.fail(error as Error);
        throw error;
      }
    } finally {
      this.busy = false;
    }
  }

  private invalid(violation: ContractViolation): void {
    if (this.onInvalid === "skip") {
      this.counters.invalid_skipped += 1;
      this.counters.invalid_streak += 1;
      const limit = this.maxInvalidStreak;
      if (limit !== null && this.counters.invalid_streak >= limit) {
        const streak = new InvalidInputStreak(`run ${this.runId}: ${limit} consecutive invalid inputs after t=${this.t}; the source is probably broken (last: ${violation.message})`);
        this.fail(streak);
        throw streak;
      }
      return;
    }
    this.fail(violation);
    throw violation;
  }

  private apply(x: StatInput): OutputEvent | null {
    if (this.terminal) {
      this.counters.after_terminal += 1;
      return null;
    }
    try {
      validateInput(this.procedure.inputs(), this.procedure.jointInputs, x);
    } catch (error) {
      if (error instanceof ContractViolation) {
        this.invalid(error);
        return null;
      }
      throw error;
    }
    const rng = this.rngState === null ? null : new SplitMix64(this.rngState);
    let newState: State, output: Output, digest: string, terminal: boolean;
    try {
      [newState, output] = this.procedure.step(this.state, x, rng);
      const encoded = canonicalJson([newState, output]); // refuses non-finite numbers
      if (encoded.length === 0) throw new NumericalError("empty state");
      digest = sha256Hex(this.historyDigest + canonicalJson(x));
      terminal = this.procedure.isTerminal(newState);
    } catch (error) {
      if (error instanceof ContractViolation) {
        this.invalid(error);
        return null;
      }
      if (error instanceof RangeError) {
        const numerical = new NumericalError(`${this.procedure.name}: the step produced a non-finite value (${error.message})`);
        this.fail(numerical);
        throw numerical;
      }
      this.fail(error as Error);
      throw error;
    }
    this.state = newState;
    if (rng !== null) this.rngState = rng.state;
    this.t += 1;
    this.counters.invalid_streak = 0;
    this.historyDigest = digest;
    this.terminal = terminal;
    const event: OutputEvent = { runId: this.runId, t: this.t, output, terminal: this.terminal };
    for (const consumer of this.consumers) {
      try {
        consumer(event);
      } catch (error) {
        const e = error as Error;
        this.consumerErrors.push({ t: this.t, consumer: consumer.name || "consumer", error: `${e.name}: ${e.message}` });
      }
    }
    return event;
  }

  // ---------------------------------------------------------- persistence

  /** A JSON value sufficient to continue this run elsewhere -- in this executor or in the Python library. */
  checkpoint(): Record<string, Json> {
    if (this.status === "failed") throw new LifecycleError("a failed run cannot be checkpointed; restore an earlier checkpoint");
    if (this.busy) throw new LifecycleError("checkpoints are taken between transitions");
    const cp = normalize({
      format: CHECKPOINT_FORMAT,
      run_id: this.runId,
      t: this.t,
      terminal: this.terminal,
      procedure: identity(this.procedure),
      state: this.state,
      rng: this.rngState === null ? null : new SplitMix64(this.rngState).hex(),
      topology: { spec: this.topology.spec(), state: this.topology.state() },
      delivery: { spec: this.delivery.spec(), state: this.delivery.state() },
      history_digest: this.historyDigest,
      log: this.log === null ? null : this.log.state(),
      counters: { ...this.counters },
      consumer_errors: this.consumerErrors,
      provenance: this.provenance,
    }) as unknown as Record<string, Json>;
    canonicalJson(cp);
    return cp;
  }

  /** Continue a run from a checkpoint written by this executor or by the Python library. */
  static restore(
    procedure: Procedure,
    cp: unknown,
    options: { topology?: Topology; delivery?: Delivery; consumers?: Consumer[]; onInvalid?: OnInvalid; log?: ObservationLog | null; clock?: () => number } = {},
  ): Run {
    validateCheckpoint(cp);
    const cid = checkpointId(cp);
    const topology = options.topology ?? new Independent();
    const delivery = options.delivery ?? new Delivery();
    for (const [label, mine, theirs] of [
      ["procedure", identity(procedure), cp.procedure],
      ["topology", topology.spec(), (cp.topology as Record<string, Json>).spec],
      ["delivery", delivery.spec(), (cp.delivery as Record<string, Json>).spec],
    ] as [string, unknown, Json][]) {
      if (!sameJson(normalize(mine), theirs))
        throw new IncompatibleCheckpoint(`${label} mismatch: checkpoint has ${canonicalJson(theirs)}, got ${canonicalJson(normalize(mine))}`);
    }
    const prov = cp.provenance as Record<string, Json>;
    const recorded = prov.on_invalid as OnInvalid;
    if (options.onInvalid !== undefined && options.onInvalid !== recorded)
      throw new IncompatibleCheckpoint(`on_invalid mismatch: checkpoint has '${recorded}', got '${options.onInvalid}'`);
    if ((cp.rng === null) === procedure.randomized)
      throw new IncompatibleCheckpoint(`malformed checkpoint: ${procedure.name} is ${procedure.randomized ? "randomized and needs" : "deterministic and has no"} a random state`);
    if (procedure.randomized && prov.seed === null) throw new IncompatibleCheckpoint(`malformed checkpoint: ${procedure.name} is randomized, but has no seed`);
    let state: State;
    try {
      state = procedure.decodeState(cp.state);
    } catch (error) {
      throw new IncompatibleCheckpoint(`malformed checkpoint: ${procedure.name} cannot decode its state (${(error as Error).message})`);
    }
    if (cp.terminal !== procedure.isTerminal(state))
      throw new IncompatibleCheckpoint(`malformed checkpoint: terminal is ${cp.terminal}, but the state says ${procedure.isTerminal(state)}`);
    const logState = (cp.log ?? null) as (LogPosition & { path: string | null; sync?: string | null }) | null;
    let log = options.log;
    if (log === undefined) {
      if (logState === null) log = null;
      else if (logState.path === null)
        throw new IncompatibleCheckpoint("the checkpointed run had an in-memory observation log; pass that log, or log: null to continue without one");
      else log = new ObservationLog(logState.path, (logState.sync as "flush" | "fsync") ?? "flush");
    }
    if (log && logState === null && !log.isEmpty()) throw new IncompatibleCheckpoint("the checkpointed run had no observation log; a log attached now must be empty");
    if (log && logState) log.verify(logState);
    const run = new Run(procedure, {
      topology, delivery, consumers: options.consumers, seed: prov.seed as string | null, runId: cp.run_id as string,
      initialState: state, onInvalid: recorded, maxInvalidStreak: (prov.max_invalid_streak as number | null) ?? null,
      clock: options.clock,
    });
    const before: [Json, Json] = [topology.state(), delivery.state()];
    try {
      topology.restore((cp.topology as Record<string, Json>).state);
      delivery.restore((cp.delivery as Record<string, Json>).state);
      for (const [part, restored] of [["topology", topology.state()], ["delivery", delivery.state()]] as [string, Json][])
        if (!sameJson(normalize(restored), (cp[part] as Record<string, Json>).state))
          throw new IncompatibleCheckpoint(`malformed checkpoint: ${part} state does not round-trip`);
    } catch (error) {
      topology.restore(before[0]);
      delivery.restore(before[1]);
      if (error instanceof IncompatibleCheckpoint) throw error;
      throw new IncompatibleCheckpoint(`malformed checkpoint: buffers cannot be restored (${(error as Error).message})`);
    }
    run.rngState = cp.rng === null ? null : BigInt("0x" + (cp.rng as string));
    run.t = cp.t as number;
    run.terminal = cp.terminal as boolean;
    run.historyDigest = cp.history_digest as string;
    run.counters = { ...run.counters, ...(cp.counters as Record<string, number>) };
    run.consumerErrors = [...((cp.consumer_errors as Json[] | undefined) ?? [])];
    run.provenance = normalize(prov);
    const discarded = log && logState ? log.restore(logState) : 0;
    run.log = log ?? null;
    run.status = "paused";
    run.event("restore", { checkpoint: cid, log: log ? log.path : null, log_discarded: discarded });
    return run;
  }
}

/** The observation with its defaults, as JSON (the form adapters and the log use). */
export function observationJson(obs: Observation): Json {
  return complete(obs) as unknown as Json;
}
