// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// TypeScript executor of sequential-analysis runs.  See README.md and ../SPEC.md.
export { canonicalJson, sameJson, sha256Hex, type Json } from "./canonical.js";
export { InputContract, validateInput, type StatInput } from "./contracts.js";
export { Delivery, type Policy } from "./delivery.js";
export * from "./errors.js";
export { ObservationLog, LOG_GENESIS } from "./log.js";
export type { Observation } from "./observation.js";
export { fromIdentity, identity, register, registered, type Identity, type Procedure, type State, type Output } from "./procedure.js";
export { SplitMix64 } from "./rng.js";
export { CHECKPOINT_FORMAT, HISTORY_GENESIS, Run, checkpointId, validateCheckpoint, type OutputEvent, type RunOptions } from "./run.js";
export { Difference, Independent, KeyJoin, PositionalPair, TimeAlign, topologyFromSpec, type Topology } from "./topology.js";
export { VERSION } from "./version.js";
export { deliveryFromSpec, restoreFromCheckpoint } from "./executor.js";
export * as procedures from "./procedures/index.js";
