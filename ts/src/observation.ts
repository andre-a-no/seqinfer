// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Observations as adapters deliver them, and the checks delivery and logging need.
import type { Json } from "./canonical.js";

export interface Observation {
  input: string;
  value: Json;
  /** Defaults to the input name. */
  source?: string;
  seq?: number | null;
  key?: string | number | null;
  time?: number | null;
}

/** The observation with its defaults filled in, in the JSON form of the observation log. */
export function complete(obs: Observation): Required<Observation> {
  return {
    input: obs.input,
    value: obs.value,
    source: obs.source === undefined || obs.source === "" ? obs.input : obs.source,
    seq: obs.seq ?? null,
    key: obs.key ?? null,
    time: obs.time ?? null,
  };
}

const MAX_EXACT = 2 ** 53;
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;
const isText = (v: unknown): v is string => typeof v === "string" && !LONE_SURROGATE.test(v);
const exactInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v) && Math.abs(v) <= MAX_EXACT;
export const UNRECORDABLE = "$unrecordable";

/** Why delivery cannot handle an observation, or null if it can (SPEC.md, "Observations"). */
export function envelopeProblem(obs: Observation): string | null {
  if (obs === null || typeof obs !== "object") return "expected an observation object";
  if (!isText(obs.input)) return `input name ${JSON.stringify(obs.input)} must be a string`;
  if (obs.source !== undefined && !isText(obs.source)) return "source must be a string";
  if (obs.seq !== undefined && obs.seq !== null && !exactInt(obs.seq)) return `sequence number ${obs.seq} must be null or an integer within 2**53`;
  return null;
}

function plainValue(v: unknown): boolean {
  if (typeof v === "number") return Number.isFinite(v) && (!Number.isInteger(v) || Math.abs(v) <= MAX_EXACT);
  if (typeof v === "string") return isText(v);
  if (v === null || typeof v === "boolean") return true;
  if (Array.isArray(v)) return v.every(plainValue);
  if (typeof v === "object") return !(UNRECORDABLE in (v as object)) && Object.entries(v as object).every(([k, x]) => isText(k) && plainValue(x));
  return false;
}

/** Why an observation cannot be recorded exactly, or null if it can. */
export function recordingProblem(obs: Observation): string | null {
  if (!plainValue(obs.value)) return `value ${String(obs.value)} cannot be recorded exactly`;
  if (obs.key !== undefined && obs.key !== null && !(isText(obs.key) || exactInt(obs.key))) return "key must be a string or an integer within 2**53";
  if (obs.time !== undefined && obs.time !== null && !(typeof obs.time === "number" && Number.isFinite(obs.time))) return "event time must be a finite number";
  return null;
}
