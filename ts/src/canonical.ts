// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Canonical JSON (RFC 8785) and SHA-256: what history digests and checkpoint ids hash.
//
// ECMAScript's own number formatting and string escaping are the ones RFC 8785
// prescribes; keys are sorted by UTF-16 code units, as Array.prototype.sort does.
import { createHash } from "node:crypto";

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

function text(s: string): string {
  if (LONE_SURROGATE.test(s)) throw new RangeError(`canonical JSON has no representation for ${JSON.stringify(s)}: lone surrogates are not text`);
  return JSON.stringify(s);
}

function encode(v: unknown, out: string[]): void {
  if (v === null) out.push("null");
  else if (v === true) out.push("true");
  else if (v === false) out.push("false");
  else if (typeof v === "number") {
    if (!Number.isFinite(v)) throw new RangeError(`canonical JSON has no representation for ${v}`);
    out.push(Object.is(v, -0) ? "0" : String(v));
  } else if (typeof v === "string") out.push(text(v));
  else if (Array.isArray(v)) {
    out.push("[");
    v.forEach((item, i) => {
      if (i) out.push(",");
      encode(item, out);
    });
    out.push("]");
  } else if (typeof v === "object") {
    out.push("{");
    Object.keys(v as object)
      .sort()
      .forEach((k, i) => {
        if (i) out.push(",");
        out.push(text(k), ":");
        encode((v as Record<string, unknown>)[k], out);
      });
    out.push("}");
  } else throw new TypeError(`canonical JSON cannot encode ${typeof v}`);
}

/** The RFC 8785 encoding of a JSON value. */
export function canonicalJson(v: unknown): string {
  const out: string[] = [];
  encode(v, out);
  return out.join("");
}

export function sha256Hex(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Equal as JSON values (1 equals 1.0; true does not equal 1). */
export function sameJson(a: unknown, b: unknown): boolean {
  try {
    return canonicalJson(a) === canonicalJson(b);
  } catch {
    return false;
  }
}

/** A deep copy through JSON, as a checkpoint is stored. */
export function normalize<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T;
}
