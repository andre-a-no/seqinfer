// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
//
// An independent JavaScript check of the conformance vectors.
//
//   node conformance/check.mjs
//
// It recomputes every history digest from the inputs (RFC 8785 + SHA-256
// chain) and re-runs the SPRT cases with a separate implementation of
// the transition.  Agreement shows that the vectors pin down the
// semantics, not one implementation's accidents.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const dir = new URL(".", import.meta.url);
const vectors = JSON.parse(readFileSync(new URL("procedures.json", dir), "utf8"));
const numbers = JSON.parse(readFileSync(new URL("numbers.json", dir), "utf8"));

// RFC 8785: ECMAScript number formatting, UTF-16 key order, JSON.stringify escaping.
const canonical = (v) =>
  Array.isArray(v) ? "[" + v.map(canonical).join(",") + "]"
  : v !== null && typeof v === "object"
    ? "{" + Object.keys(v).sort().map((k) => JSON.stringify(k) + ":" + canonical(v[k])).join(",") + "}"
    : JSON.stringify(v);
const sha256 = (s) => createHash("sha256").update(s, "utf8").digest("hex");

let failures = 0;
const check = (ok, what) => { if (!ok) { failures++; console.log("MISMATCH", what); } };

const bits = new BigUint64Array(1), f64 = new Float64Array(bits.buffer);
for (const { bits: b, json } of numbers) { bits[0] = BigInt("0x" + b); check(String(f64[0]) === json, `number ${b}`); }

for (const v of vectors) {
  let digest = sha256("seqinfer.history/2");
  for (const x of v.inputs) digest = sha256(digest + canonical(x));
  check(digest === v.history_digest, `${v.case}: history digest`);
}

// SPRT transition: Neumaier-compensated sum of log likelihood ratios.
const llr = {
  gaussian: (f, x, a, b) => ((a - b) / f.sigma ** 2) * (x - 0.5 * (a + b)),
  bernoulli: (f, x, a, b) => (x ? Math.log(a / b) : Math.log((1 - a) / (1 - b))),
  poisson: (f, x, a, b) => x * Math.log(a / b) - (a - b),
};
for (const v of vectors.filter((v) => v.procedure.name === "sprt")) {
  const c = v.procedure.config, fam = c.family;
  const upper = Math.log((1 - c.beta) / c.alpha), lower = Math.log(c.beta / (1 - c.alpha));
  let total = 0, comp = 0;
  v.inputs.forEach((x, i) => {
    const inc = llr[fam.family](fam, x[c.input], c.theta1, c.theta0);
    const t = total + inc;
    comp += Math.abs(total) >= Math.abs(inc) ? total - t + inc : inc - t + total;
    total = t;
    const value = total + comp;
    const decision = value >= upper ? "reject_h0" : value <= lower ? "accept_h0" : null;
    const out = v.outputs[i];
    check(out.llr === value && out.decision === decision && out.n === i + 1, `${v.case}: step ${i + 1}`);
  });
}

console.log(failures ? `${failures} mismatches` : `all ${vectors.length} digests, ${numbers.length} numbers and the SPRT outputs agree`);
process.exit(failures ? 1 : 0);
