// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Helpers shared by the procedures: Python's min/max, config readers, a common base.
import type { Json } from "../canonical.js";
import { sameJson } from "../canonical.js";
import type { InputContract, StatInput } from "../contracts.js";
import type { Procedure, State, Output } from "../procedure.js";
import type { SplitMix64 } from "../rng.js";

export type Config = { [key: string]: Json };
export type Decision = null | "reject_h0" | "accept_h0";
export const REJECT_H0 = "reject_h0";
export const ACCEPT_H0 = "accept_h0";

/** Python's max(a, b): a unless b is greater. */
export const pyMax = (a: number, b: number): number => (b > a ? b : a);
/** Python's min(a, b): a unless b is smaller. */
export const pyMin = (a: number, b: number): number => (b < a ? b : a);

export function invalid(message: string): never {
  throw new RangeError(message);
}

function field(c: Config, key: string): Json {
  if (!(key in c)) invalid(`config is missing '${key}'`);
  return c[key];
}
export function num(c: Config, key: string): number {
  const v = field(c, key);
  if (typeof v !== "number") invalid(`config '${key}' must be a number, got ${JSON.stringify(v)}`);
  return v;
}
export function optNum(c: Config, key: string): number | null {
  return field(c, key) === null ? null : num(c, key);
}
export function int(c: Config, key: string): number {
  const v = num(c, key);
  if (!Number.isInteger(v)) invalid(`config '${key}' must be an integer, got ${v}`);
  return v;
}
export function str(c: Config, key: string): string {
  const v = field(c, key);
  if (typeof v !== "string") invalid(`config '${key}' must be a string, got ${JSON.stringify(v)}`);
  return v;
}
export function bool(c: Config, key: string): boolean {
  const v = field(c, key);
  if (typeof v !== "boolean") invalid(`config '${key}' must be a boolean, got ${JSON.stringify(v)}`);
  return v;
}
export function nums(c: Config, key: string): number[] {
  const v = field(c, key);
  if (!Array.isArray(v) || !v.every((e) => typeof e === "number")) invalid(`config '${key}' must be a list of numbers`);
  return v as number[];
}
export function obj(c: Config, key: string): Config {
  const v = field(c, key);
  if (v === null || typeof v !== "object" || Array.isArray(v)) invalid(`config '${key}' must be an object`);
  return v;
}

/** What every procedure here shares; subclasses fill in the transition. */
export abstract class BaseProcedure implements Procedure {
  abstract readonly name: string;
  abstract readonly version: string;
  readonly randomized: boolean = false;
  readonly jointInputs: boolean = true;
  abstract config(): Config;
  abstract inputs(): Record<string, InputContract>;
  abstract initialState(): State;
  abstract step(state: State, x: StatInput, rng: SplitMix64 | null): [State, Output];
  abstract decodeState(data: Json): State;
  isTerminal(_state: State): boolean {
    return false;
  }
}

/** Python dataclass equality of two states (0.0 == -0.0, 1 == 1.0). */
export const sameState = (a: unknown, b: unknown): boolean => sameJson(a, b);
