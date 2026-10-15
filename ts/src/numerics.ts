// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
import { NumericalError } from "./errors.js";

/** One step of Neumaier summation; the represented value is total + comp. */
export function neumaierAdd(total: number, comp: number, x: number): [number, number] {
  const t = total + x;
  if (Math.abs(total) >= Math.abs(x)) comp += total - t + x;
  else comp += x - t + total;
  return [t, comp];
}

export function requireFinite(value: number, what: string): number {
  if (!Number.isFinite(value)) throw new NumericalError(`${what} is not finite: ${value}`);
  return value;
}

/** log(sum exp(v)) with the largest term factored out and a left-to-right sum. */
export function logSumExp(values: number[]): number {
  if (values.length === 0) return -Infinity;
  const m = Math.max(...values);
  if (!Number.isFinite(m)) return m;
  let total = 0;
  for (const v of values) total += Math.exp(v - m);
  return m + Math.log(total);
}
