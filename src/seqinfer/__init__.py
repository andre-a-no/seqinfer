# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""seqinfer: sequential statistical inference, separated from experimental runtime.

    Procedure infers; consumers act.

Layer            Module
---------------  ------------------------------------------
adapters         seqinfer.sources
logical inputs   seqinfer.contracts
data topology    seqinfer.topology
runtime          seqinfer.delivery, seqinfer.runtime
run              seqinfer.run, seqinfer.persistence
procedure        seqinfer.core, seqinfer.procedures, seqinfer.pipeline
consumers        seqinfer.consumers
"""
from .consumers import JsonlSink, OutputEvent, Recorder
from .contracts import InputContract, validate_input
from .core import Observation, Procedure, trajectory
from .delivery import Delivery
from .errors import (
    ContractViolation,
    IncompatibleCheckpoint,
    LifecycleError,
    NumericalError,
    OrderingError,
    SeqInferError,
)
from .numerics import equivalent
from .persistence import load_checkpoint, save_checkpoint
from .pipeline import Chain
from .rng import SplitMix64
from .run import Run, RunStatus, checkpoint_id
from .runtime import run_async, run_sync
from .topology import Independent, KeyJoin, PositionalPair, TimeAlign, difference
from .version import __version__

__all__ = [
    "Chain",
    "ContractViolation",
    "Delivery",
    "IncompatibleCheckpoint",
    "Independent",
    "InputContract",
    "JsonlSink",
    "KeyJoin",
    "LifecycleError",
    "NumericalError",
    "Observation",
    "OrderingError",
    "OutputEvent",
    "PositionalPair",
    "Procedure",
    "Recorder",
    "Run",
    "RunStatus",
    "SeqInferError",
    "SplitMix64",
    "TimeAlign",
    "__version__",
    "checkpoint_id",
    "difference",
    "equivalent",
    "load_checkpoint",
    "run_async",
    "run_sync",
    "save_checkpoint",
    "trajectory",
    "validate_input",
]
