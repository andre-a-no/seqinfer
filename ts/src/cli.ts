#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
//
//   sequential-analysis continue CHECKPOINT OBSERVATIONS.jsonl [--checkpoint OUT] [--outputs OUT.jsonl]
//   sequential-analysis verify CHECKPOINT
//   sequential-analysis procedures
//
// `continue` restores a run from a checkpoint (written by this executor or by the Python
// library), offers it the observations (one JSON object per line: input, value, source,
// seq, key, time) and writes the new checkpoint and the outputs.
import { readFileSync, writeFileSync } from "node:fs";
import { canonicalJson } from "./canonical.js";
import { PROCEDURE_GUARANTEES } from "./guarantees.js";
import { restoreFromCheckpoint } from "./executor.js";
import type { Observation } from "./observation.js";
import { registered } from "./procedure.js";
import { checkpointId, validateCheckpoint } from "./run.js";

function option(args: string[], name: string): string | undefined {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}

function main(argv: string[]): number {
  const [command, ...args] = argv;
  if (command === "procedures") {
    // each registered procedure with what it promises (see the which-method page)
    const describe = (name: string): string => {
      const g = PROCEDURE_GUARANTEES[name.split("/")[0]];
      if (g === undefined) return "";
      return typeof g === "string" ? g : Object.entries(g).map(([k, v]) => `${v} (${k})`).join(", ");
    };
    console.log(registered().map((name) => `${name}\t${describe(name)}`).join("\n"));
    return 0;
  }
  if (command === "verify" && args[0]) {
    const cp: unknown = JSON.parse(readFileSync(args[0], "utf8"));
    validateCheckpoint(cp);
    restoreFromCheckpoint(cp, { log: null });
    console.log(`valid; checkpoint id ${checkpointId(cp)}; t=${cp.t}`);
    return 0;
  }
  if (command === "continue" && args[0] && args[1]) {
    const cp: unknown = JSON.parse(readFileSync(args[0], "utf8"));
    const run = restoreFromCheckpoint(cp, { log: null });
    const outputs: string[] = [];
    run.start("cli");
    run.consumers.push((e) => outputs.push(canonicalJson({ t: e.t, output: e.output, terminal: e.terminal })));
    for (const line of readFileSync(args[1], "utf8").split("\n")) {
      if (line.trim()) run.offer(JSON.parse(line) as Observation);
    }
    run.pause();
    writeFileSync(option(args, "--checkpoint") ?? args[0], JSON.stringify(run.checkpoint()) + "\n");
    const out = option(args, "--outputs");
    if (out) writeFileSync(out, outputs.map((l) => l + "\n").join(""));
    console.log(`t=${run.t} terminal=${run.terminal} digest=${run.historyDigest}`);
    return 0;
  }
  console.error("usage: sequential-analysis continue CHECKPOINT OBSERVATIONS.jsonl [--checkpoint OUT] [--outputs OUT.jsonl]\n" +
    "       sequential-analysis verify CHECKPOINT\n       sequential-analysis procedures");
  return 2;
}

process.exit(main(process.argv.slice(2)));
