// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Everything a checkpoint needs to be continued here is in the checkpoint: the procedure's
// identity, the topology's and the delivery's specs.
import type { Json } from "./canonical.js";
import { Delivery, type Policy } from "./delivery.js";
import { IncompatibleCheckpoint } from "./errors.js";
import type { ObservationLog } from "./log.js";
import { fromIdentity, type Identity } from "./procedure.js";
import "./procedures/index.js";
import { Run, validateCheckpoint, type Consumer } from "./run.js";
import { topologyFromSpec } from "./topology.js";

export function deliveryFromSpec(spec: Json): Delivery {
  const s = spec as { policy: Policy; dedup: boolean; start: number };
  return new Delivery(s.policy, s.dedup, s.start);
}

/** Rebuild procedure, topology and delivery from a checkpoint and continue the run. */
export function restoreFromCheckpoint(
  cp: unknown,
  options: { consumers?: Consumer[]; log?: ObservationLog | null; clock?: () => number } = {},
): Run {
  validateCheckpoint(cp);
  let procedure, topology, delivery;
  try {
    procedure = fromIdentity(cp.procedure as unknown as Identity);
    topology = topologyFromSpec((cp.topology as Record<string, Json>).spec);
    delivery = deliveryFromSpec((cp.delivery as Record<string, Json>).spec);
  } catch (error) {
    throw new IncompatibleCheckpoint(`this executor cannot continue the run: ${(error as Error).message}`);
  }
  return Run.restore(procedure, cp, { topology, delivery, ...options });
}
