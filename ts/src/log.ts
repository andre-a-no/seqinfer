// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The observation log: the delivered history, one JSON line per observation, under a hash chain (SPEC.md, "Observation log").
import { closeSync, existsSync, fsyncSync, openSync, readFileSync, statSync, truncateSync, writeSync } from "node:fs";
import { resolve } from "node:path";
import { createHash } from "node:crypto";
import type { Json } from "./canonical.js";
import { IncompatibleCheckpoint } from "./errors.js";
import { complete, recordingProblem, UNRECORDABLE, type Observation } from "./observation.js";

export const LOG_GENESIS = createHash("sha256").update("sequential-analysis.log/2").digest("hex");

export interface LogPosition {
  count: number;
  offset: number;
  digest: string;
}

/** JSON with sorted keys, ", " and ": " separators and ASCII-only strings: the layout the Python writer uses. */
export function logLine(record: Json): string {
  const enc = (v: Json): string => {
    if (v === null || typeof v !== "object") {
      return typeof v === "string" ? JSON.stringify(v).replace(/[\u0080-\uffff]/g, (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0")) : JSON.stringify(v);
    }
    if (Array.isArray(v)) return "[" + v.map(enc).join(", ") + "]";
    return "{" + Object.keys(v).sort().map((k) => enc(k) + ": " + enc(v[k])).join(", ") + "}";
  };
  return enc(record) + "\n";
}

function chain(digest: string, line: Buffer): string {
  return createHash("sha256").update(Buffer.concat([Buffer.from(digest, "ascii"), line])).digest("hex");
}

function splitLines(data: Buffer): Buffer[] {
  const out: Buffer[] = [];
  let start = 0;
  for (let i = 0; i < data.length; i++) if (data[i] === 0x0a) { out.push(data.subarray(start, i + 1)); start = i + 1; }
  if (start < data.length) out.push(data.subarray(start));
  return out;
}

export class ObservationLog {
  readonly path: string | null;
  private memory: Buffer[] = [];
  private fd: number | null = null;
  private position: LogPosition | null = null;

  constructor(path: string | null = null, readonly sync: "flush" | "fsync" = "flush") {
    this.path = path === null ? null : resolve(path);
  }

  private lines(): Buffer[] {
    if (this.path === null) return this.memory;
    return existsSync(this.path) ? splitLines(readFileSync(this.path)) : [];
  }

  private scan(): LogPosition {
    if (this.position === null) {
      let count = 0, offset = 0, digest = LOG_GENESIS;
      for (const line of this.lines()) {
        if (line[line.length - 1] !== 0x0a) throw new IncompatibleCheckpoint(`log ${this.path}: last record is incomplete (torn write)`);
        count += 1;
        offset += line.length;
        digest = chain(digest, line);
      }
      this.position = { count, offset, digest };
    }
    return this.position;
  }

  isEmpty(): boolean {
    return this.path === null ? this.memory.length === 0 : !existsSync(this.path) || statSync(this.path).size === 0;
  }

  /** The position stored in a checkpoint; forces the file to disk. */
  state(): Json {
    const p = this.scan();
    if (this.fd !== null) fsyncSync(this.fd);
    return { path: this.path, sync: this.path === null ? null : this.sync, count: p.count, offset: p.offset, digest: p.digest };
  }

  append(obs: Observation): void {
    const p = this.scan();
    let record: Json = complete(obs) as unknown as Json;
    if (recordingProblem(obs) !== null) {
      const o = complete(obs);
      record = { input: o.input, source: o.source, seq: o.seq, key: null, time: null, value: { [UNRECORDABLE]: String(o.value).slice(0, 200) } };
    }
    const line = Buffer.from(logLine(record), "utf8");
    if (this.path === null) this.memory.push(line);
    else {
      if (this.fd === null) this.fd = openSync(this.path, "a");
      writeSync(this.fd, line);
      if (this.sync === "fsync") fsyncSync(this.fd);
    }
    p.count += 1;
    p.offset += line.length;
    p.digest = chain(p.digest, line);
  }

  /** Check that the log starts with the checkpointed records; return how many follow them. */
  verify(state: LogPosition): number {
    let n = 0, size = 0, d = LOG_GENESIS, discarded = 0;
    for (const line of this.lines()) {
      if (n < state.count) { n += 1; size += line.length; d = chain(d, line); }
      else discarded += 1;
    }
    const where = this.path ?? "in-memory log";
    if (n < state.count) throw new IncompatibleCheckpoint(`${where}: has ${n} records, the checkpoint expects at least ${state.count}`);
    if (size !== state.offset || d !== state.digest) throw new IncompatibleCheckpoint(`${where}: the first ${state.count} records differ from the checkpointed ones`);
    return discarded;
  }

  /** Cut the log back to a checkpointed position; return how many records were deleted. */
  restore(state: LogPosition): number {
    const discarded = this.verify(state);
    if (this.path === null) this.memory.length = state.count;
    else if (existsSync(this.path) && statSync(this.path).size > state.offset) {
      this.close();
      truncateSync(this.path, state.offset);
    }
    this.position = { count: state.count, offset: state.offset, digest: state.digest };
    return discarded;
  }

  /** The recorded observations, in order. */
  *[Symbol.iterator](): Iterator<Observation> {
    for (const line of this.lines()) {
      const text = line.toString("utf8").trim();
      if (text) yield JSON.parse(text) as Observation;
    }
  }

  close(): void {
    if (this.fd !== null) {
      closeSync(this.fd);
      this.fd = null;
    }
  }
}
