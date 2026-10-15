// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Input contracts and the validation of a statistical input.
import type { Json } from "./canonical.js";
import { ContractViolation } from "./errors.js";

export type Kind = "real" | "integer" | "binary";
const MAX_EXACT = 2 ** 53;

export class InputContract {
  constructor(
    readonly name: string,
    readonly kind: Kind = "real",
    readonly shape: number[] = [],
    readonly lower: number | null = null,
    readonly upper: number | null = null,
    readonly unit: string | null = null,
  ) {}

  spec(): Json {
    return { name: this.name, kind: this.kind, shape: this.shape, lower: this.lower, upper: this.upper, unit: this.unit };
  }

  private checkScalar(v: unknown): void {
    if (typeof v !== "number") throw new ContractViolation(`input '${this.name}': expected a number, got ${JSON.stringify(v)}`);
    if (Number.isInteger(v) && Math.abs(v) > MAX_EXACT)
      throw new ContractViolation(`input '${this.name}': integer ${v} exceeds 2**53`);
    if (!Number.isFinite(v)) throw new ContractViolation(`input '${this.name}': non-finite value ${v}`);
    if (this.kind === "integer" && !Number.isInteger(v)) throw new ContractViolation(`input '${this.name}': expected an integer, got ${v}`);
    if (this.kind === "binary" && v !== 0 && v !== 1) throw new ContractViolation(`input '${this.name}': expected 0 or 1, got ${v}`);
    if (this.lower !== null && v < this.lower) throw new ContractViolation(`input '${this.name}': ${v} is below ${this.lower}`);
    if (this.upper !== null && v > this.upper) throw new ContractViolation(`input '${this.name}': ${v} is above ${this.upper}`);
  }

  validate(value: unknown): void {
    if (this.shape.length === 0) return this.checkScalar(value);
    if (!Array.isArray(value)) throw new ContractViolation(`input '${this.name}': expected a list of length ${this.shape[0]}`);
    if (value.length !== this.shape[0])
      throw new ContractViolation(`input '${this.name}': expected a vector of length ${this.shape[0]}, got ${value.length}`);
    for (const v of value) this.checkScalar(v);
  }
}

export type StatInput = Record<string, Json>;

/** Check one statistical input against a procedure's contracts (see SPEC.md, "Statistical inputs"). */
export function validateInput(contracts: Record<string, InputContract>, jointInputs: boolean, x: unknown): void {
  if (x === null || typeof x !== "object" || Array.isArray(x) || Object.keys(x).length === 0)
    throw new ContractViolation("a statistical input must be a non-empty mapping");
  const names = Object.keys(x);
  const unknown = names.filter((n) => !(n in contracts)).sort();
  if (unknown.length) throw new ContractViolation(`undeclared logical inputs: ${unknown.map((n) => `'${n}'`).join(", ")}`);
  if (jointInputs) {
    const missing = Object.keys(contracts).filter((n) => !names.includes(n)).sort();
    if (missing.length) throw new ContractViolation(`joint procedure is missing inputs: ${missing.map((n) => `'${n}'`).join(", ")}`);
  }
  for (const n of names) contracts[n].validate((x as Record<string, unknown>)[n]);
}
