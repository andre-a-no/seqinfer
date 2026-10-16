// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The guarantees here are the Python procedures' (conformance/guarantees.json), for every registered procedure.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { GUARANTEES, PROCEDURE_GUARANTEES } from "../src/guarantees.js";
import { registered } from "../src/procedure.js";
import "../src/procedures/index.js";

const python = JSON.parse(readFileSync(new URL("../../../conformance/guarantees.json", import.meta.url), "utf8")) as Record<string, unknown>;

test("the guarantees are Python's", () => {
  assert.deepEqual(PROCEDURE_GUARANTEES, python);
});

test("every registered procedure has a known guarantee", () => {
  for (const key of registered()) {
    const name = key.split("/")[0];
    const g = PROCEDURE_GUARANTEES[name];
    assert.ok(g !== undefined, name);
    for (const v of typeof g === "string" ? [g] : Object.values(g)) assert.ok((GUARANTEES as readonly string[]).includes(v), name);
  }
});
