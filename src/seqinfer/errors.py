"""Error classes, one per architectural boundary (paper, Table 2)."""


class SeqInferError(Exception):
    """Base class for all errors raised by this package."""


class ContractViolation(SeqInferError):
    """A statistical input does not satisfy the procedure's input contract."""


class OrderingError(SeqInferError):
    """An observation violates the delivery or topology ordering policy."""


class NumericalError(SeqInferError):
    """A state transition produced a non-finite or otherwise invalid value."""


class LifecycleError(SeqInferError):
    """An operation is not permitted in the current run status."""


class IncompatibleCheckpoint(SeqInferError):
    """A checkpoint does not match the procedure, topology or delivery it is restored into."""
