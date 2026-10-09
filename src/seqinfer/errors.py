# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Error classes of the package (the paper's table of errors says which layer is responsible for what).

Not every row of that table has a class: source errors are the source's own
exceptions, serialization errors those of `json` and the file system, and
consumer errors are caught and recorded rather than raised.
"""


class SeqInferError(Exception):
    """Base class for all errors raised by this package."""


class ContractViolation(SeqInferError):
    """A statistical input does not satisfy the procedure's input contract."""


class InvalidInputStreak(SeqInferError):
    """Too many consecutive invalid inputs: the source, not a message, is probably broken."""


class OrderingError(SeqInferError):
    """An observation violates the delivery or topology ordering policy."""


class NumericalError(SeqInferError):
    """A state transition produced a non-finite or otherwise invalid value."""


class LifecycleError(SeqInferError):
    """An operation is not permitted in the current run status."""


class IncompatibleCheckpoint(SeqInferError):
    """A checkpoint does not match the procedure, topology or delivery it is restored into."""
