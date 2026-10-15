// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// The error classes of the Python library, with the same names and meanings.

export class SequentialAnalysisError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}
/** A statistical input does not satisfy the procedure's input contract. */
export class ContractViolation extends SequentialAnalysisError {}
/** Too many consecutive invalid inputs. */
export class InvalidInputStreak extends SequentialAnalysisError {}
/** An observation violates the delivery or topology ordering policy. */
export class OrderingError extends SequentialAnalysisError {}
/** A state transition produced a non-finite or otherwise invalid value. */
export class NumericalError extends SequentialAnalysisError {}
/** An operation is not permitted in the current run status. */
export class LifecycleError extends SequentialAnalysisError {}
/** A checkpoint does not match the procedure, topology or delivery it is restored into. */
export class IncompatibleCheckpoint extends SequentialAnalysisError {}
