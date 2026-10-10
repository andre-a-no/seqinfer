# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""sequential_analysis: sequential statistical inference, separated from experimental runtime.

    Procedure infers; consumers act.

Layer            Module
---------------  ------------------------------------------
adapters         sequential_analysis.sources
logical inputs   sequential_analysis.contracts
data topology    sequential_analysis.topology
runtime          sequential_analysis.delivery, sequential_analysis.runtime
run              sequential_analysis.run, sequential_analysis.persistence
procedure        sequential_analysis.core, sequential_analysis.procedures, sequential_analysis.pipeline
consumers        sequential_analysis.consumers
"""
from .consumers import JsonlSink, OutputEvent, Recorder
from .contracts import InputContract, validate_input
from .core import Observation, Procedure, trajectory
from .delivery import Delivery
from .errors import (
    ContractViolation,
    IncompatibleCheckpoint,
    InvalidInputStreak,
    LifecycleError,
    NumericalError,
    OrderingError,
    SequentialAnalysisError,
)
from .numerics import equivalent
from .persistence import load_checkpoint, save_checkpoint
from .pipeline import Chain
from .rng import SplitMix64
from .run import Run, RunStatus, checkpoint_id, validate_checkpoint
from .runtime import run_async, run_sync
from .topology import Independent, KeyJoin, PositionalPair, TimeAlign, difference
from .transforms import Difference, Field, Transform
from .version import __version__

__all__ = [
    "Chain",
    "ContractViolation",
    "Delivery",
    "Difference",
    "Field",
    "IncompatibleCheckpoint",
    "Independent",
    "InputContract",
    "InvalidInputStreak",
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
    "SequentialAnalysisError",
    "SplitMix64",
    "TimeAlign",
    "Transform",
    "__version__",
    "checkpoint_id",
    "difference",
    "equivalent",
    "load_checkpoint",
    "run_async",
    "run_sync",
    "save_checkpoint",
    "trajectory",
    "validate_checkpoint",
    "validate_input",
]
