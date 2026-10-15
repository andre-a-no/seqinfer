// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The procedure contract, and the registry that rebuilds a procedure from its identity.
//
// A procedure's state and output are kept in their JSON form -- the form a
// checkpoint stores -- so encoding is the identity and decoding a check.
import type { Json } from "./canonical.js";
import type { InputContract, StatInput } from "./contracts.js";
import { LifecycleError } from "./errors.js";
import type { SplitMix64 } from "./rng.js";

export type State = { [key: string]: Json };
export type Output = { [key: string]: Json };

export interface Identity {
  name: string;
  version: string;
  config: { [key: string]: Json };
}

export interface Procedure {
  readonly name: string;
  readonly version: string;
  /** True if step draws from the statistical random state. */
  readonly randomized: boolean;
  /** True if every declared input is required in every step. */
  readonly jointInputs: boolean;
  config(): { [key: string]: Json };
  inputs(): Record<string, InputContract>;
  initialState(): State;
  /** (S_t, O_t) from S_{t-1} and X_t; must not modify its arguments. rng carries R_{t-1} in, R_t out. */
  step(state: State, x: StatInput, rng: SplitMix64 | null): [State, Output];
  isTerminal(state: State): boolean;
  /** Check a state read from a checkpoint; throw Error naming the problem. */
  decodeState(data: Json): State;
}

export function identity(p: Procedure): Identity {
  return { name: p.name, version: p.version, config: p.config() };
}

type Factory = (config: { [key: string]: Json }) => Procedure;
const registry = new Map<string, Factory>();

/** Register the constructor of a procedure name and version. */
export function register(name: string, version: string, factory: Factory): void {
  registry.set(`${name}/${version}`, factory);
}

/** The procedure an identity describes; the rebuilt procedure must report the same identity. */
export function fromIdentity(id: Identity): Procedure {
  const factory = registry.get(`${id.name}/${id.version}`);
  if (!factory) throw new Error(`no TypeScript implementation of ${id.name} version ${id.version}`);
  return factory(id.config);
}

export function registered(): string[] {
  return [...registry.keys()].sort();
}

export function refuseTerminal(p: Procedure, state: State): void {
  if (p.isTerminal(state)) throw new LifecycleError(`${p.name} has stopped; a terminal state takes no further observations`);
}

// ------------------------------------------------------------ state decoding

export type FieldKind = "count" | "number" | "string" | "decision" | "counts" | "numbers" | "json";
export type FieldSpec = Record<string, FieldKind | `${FieldKind}?`>;

const DECISIONS = [null, "reject_h0", "accept_h0"];

function checkField(name: string, kind: string, v: Json): void {
  const optional = kind.endsWith("?");
  const k = optional ? kind.slice(0, -1) : kind;
  if (v === null && (optional || k === "decision" || k === "json")) return;
  const finite = (x: Json) => typeof x === "number" && Number.isFinite(x);
  const ok =
    k === "count" ? finite(v) && Number.isInteger(v) && (v as number) >= 0
    : k === "number" ? finite(v)
    : k === "string" ? typeof v === "string"
    : k === "decision" ? DECISIONS.includes(v as string)
    : k === "counts" ? Array.isArray(v) && v.every((x) => finite(x) && Number.isInteger(x) && (x as number) >= 0)
    : k === "numbers" ? Array.isArray(v) && v.every(finite)
    : true;
  if (!ok) throw new Error(`state field '${name}' must be ${k}, got ${JSON.stringify(v)}`);
}

/** Check that a state has exactly the given fields, each of the given kind. */
export function decodeFields(spec: FieldSpec, data: Json): State {
  if (data === null || typeof data !== "object" || Array.isArray(data)) throw new Error("a state must be a JSON object");
  const want = Object.keys(spec).sort(), got = Object.keys(data).sort();
  if (want.join("\u0000") !== got.join("\u0000")) throw new Error(`state has fields ${want}, got ${got}`);
  for (const k of want) checkField(k, spec[k], (data as State)[k]);
  return data as State;
}

export function requireState(condition: boolean, message: string): void {
  if (!condition) throw new Error(`inconsistent state: ${message}`);
}
