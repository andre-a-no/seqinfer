# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Run: one concrete execution of a procedure.

The run owns everything that must survive the process: inferential state,
statistical random state, topology buffers, source positions, the history
digest and provenance.  It is the single writer of the inferential state;
runtimes may acquire observations concurrently but hand them to
`Run.offer` one at a time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from typing import Any

from .canonical import canonical_json
from .consumers import Consumer, OutputEvent
from .contracts import validate_input
from .core import Observation, Procedure, StatInput, statistical_rng
from .delivery import Delivery
from .errors import ContractViolation, IncompatibleCheckpoint, InvalidInputStreak, LifecycleError
from .rng import SplitMix64
from .sources import ObservationLog
from .topology import Independent, Topology
from .version import __version__

#: Version 2: RFC 8785 canonical JSON in digests; 64-bit values (seed, random state) as strings.
CHECKPOINT_FORMAT = "seqinfer.checkpoint/2"
_GENESIS = hashlib.sha256(b"seqinfer.history/2").hexdigest()
#: Default of `Run.restore(log=...)`: reopen the log recorded in the checkpoint.
FROM_CHECKPOINT: Any = object()

#: Incidents (skipped inputs, failing consumers) are reported here as warnings.
logger = logging.getLogger("seqinfer")


def _policy_spec(on_invalid: Any) -> str:
    """JSON form of an on_invalid policy, recorded in provenance."""
    if on_invalid in ("skip", "raise"):
        return str(on_invalid)
    if isinstance(on_invalid, type) and issubclass(on_invalid, Exception):
        return f"raise:{on_invalid.__module__}.{on_invalid.__qualname__}"
    raise ValueError("on_invalid must be 'skip', 'raise' or an exception class")


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


def _canonical(obj: Any) -> str:
    return canonical_json(obj)


def _u64(value: int | None) -> str | None:
    """64-bit integers do not survive JSON numbers in every language; they are stored as hex strings."""
    return None if value is None else f"{value:016x}"


_HEX64 = frozenset("0123456789abcdef")

#: Required top-level fields of a checkpoint and their JSON types.
_FIELDS: dict[str, tuple[type, ...]] = {
    "format": (str,),
    "run_id": (str,),
    "t": (int,),
    "terminal": (bool,),
    "procedure": (dict,),
    "rng": (str, type(None)),
    "topology": (dict,),
    "delivery": (dict,),
    "history_digest": (str,),
    "counters": (dict,),
    "provenance": (dict,),
}


def validate_checkpoint(checkpoint: Any) -> None:
    """Check the structure of a checkpoint; raise IncompatibleCheckpoint naming the first problem.

    A checkpoint is data that may have been edited, truncated or written by
    another version, so it is checked before any of it is trusted.
    """

    def bad(what: str) -> IncompatibleCheckpoint:
        return IncompatibleCheckpoint(f"malformed checkpoint: {what}")

    if not isinstance(checkpoint, dict):
        raise bad(f"expected a JSON object, got {type(checkpoint).__name__}")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise IncompatibleCheckpoint(f"unsupported checkpoint format {checkpoint.get('format')!r}")
    for name, types in _FIELDS.items():
        if name not in checkpoint:
            raise bad(f"missing field {name!r}")
        value = checkpoint[name]
        if not isinstance(value, types) or (types == (int,) and isinstance(value, bool)):
            raise bad(f"field {name!r} has type {type(value).__name__}")
    if "state" not in checkpoint:
        raise bad("missing field 'state'")
    if checkpoint["t"] < 0:
        raise bad("negative step count")
    rng = checkpoint["rng"]
    if rng is not None and (len(rng) != 16 or not set(rng) <= _HEX64):
        raise bad("random state is not 16 hexadecimal digits")
    digest = checkpoint["history_digest"]
    if len(digest) != 64 or not set(digest) <= _HEX64:
        raise bad("history digest is not a SHA-256 hex digest")
    for part in ("topology", "delivery"):
        if not {"spec", "state"} <= set(checkpoint[part]):
            raise bad(f"{part} needs 'spec' and 'state'")
    prov = checkpoint["provenance"]
    for name in ("seed", "on_invalid", "events"):
        if name not in prov:
            raise bad(f"provenance lacks {name!r}")
    if not isinstance(prov["on_invalid"], str) or not isinstance(prov["events"], list):
        raise bad("provenance fields have the wrong type")
    if prov["seed"] is not None and not (isinstance(prov["seed"], str) and prov["seed"].lstrip("-").isdigit()):
        raise bad("seed is not a decimal string")
    log = checkpoint.get("log")
    if log is not None:
        if not isinstance(log, dict) or not {"path", "count", "offset", "digest"} <= set(log):
            raise bad("log position needs path, count, offset and digest")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in checkpoint["counters"].values()):
        raise bad("counters must be integers")


def _normalize(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


class Run:
    def __init__(
        self,
        procedure: Procedure,
        *,
        topology: Topology | None = None,
        delivery: Delivery | None = None,
        consumers: Sequence[Consumer] = (),
        seed: int | None = None,
        run_id: str | None = None,
        initial_state: Any = None,
        initialized_from: Any = "default",
        on_invalid: str | type[Exception] = "skip",
        max_invalid_streak: int | None = None,
        log: Any = None,
        clock: Callable[[], float] = time.time,
    ):
        policy = _policy_spec(on_invalid)
        if max_invalid_streak is not None and (type(max_invalid_streak) is not int or max_invalid_streak < 1):
            raise ValueError("max_invalid_streak must be a positive integer or None")
        if isinstance(log, ObservationLog) and not log.is_empty():
            raise ValueError(
                f"log {log.path or '(in memory)'} already holds records; a new run needs an empty log. "
                f"To continue a run, use Run.restore with its checkpoint"
            )
        self.procedure = procedure
        self.topology = topology if topology is not None else Independent()
        self.delivery = delivery if delivery is not None else Delivery()
        self.consumers = list(consumers)
        self.on_invalid: str | type[Exception] = on_invalid
        self.max_invalid_streak = max_invalid_streak
        self.log = log
        self._clock = clock
        self._busy = False

        self.run_id = run_id or uuid.uuid4().hex
        self.state = procedure.initial_state() if initial_state is None else initial_state
        self.rng_state = statistical_rng(procedure, seed)
        self.t = 0
        self.status = RunStatus.CREATED
        self.terminal = procedure.is_terminal(self.state)
        self.history_digest = _GENESIS
        self.counters = {"invalid_skipped": 0, "invalid_streak": 0, "after_terminal": 0, "dropped_by_runtime": 0}
        self.consumer_errors: list[dict] = []
        self.provenance = {
            "run_id": self.run_id,
            "created_at": clock(),
            "procedure": procedure.identity(),
            "inputs": {n: c.spec() for n, c in procedure.inputs().items()},
            "topology": self.topology.spec(),
            "delivery": self.delivery.spec(),
            "seed": None if seed is None else str(int(seed)),
            "initialized_from": initialized_from,
            "on_invalid": policy,
            "max_invalid_streak": max_invalid_streak,
            "seqinfer": __version__,
            "numerics": {
                "float": "IEEE 754 binary64",
                "summation": "Neumaier",
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "events": [],
        }

    # ------------------------------------------------------------ lifecycle

    def _event(self, name: str, **extra: Any) -> None:
        self.provenance["events"].append({"event": name, "at": self._clock(), "t": self.t, **extra})

    def start(self, runtime: str | None = None) -> None:
        """CREATED or PAUSED -> RUNNING.  `runtime` names the execution mode for provenance."""
        if self.status not in (RunStatus.CREATED, RunStatus.PAUSED):
            raise LifecycleError(f"cannot start a run that is {self.status.value}")
        self._event("start" if self.status is RunStatus.CREATED else "resume", runtime=runtime)
        self.status = RunStatus.RUNNING

    def pause(self) -> None:
        if self.status is not RunStatus.RUNNING:
            raise LifecycleError(f"cannot pause a run that is {self.status.value}")
        self.status = RunStatus.PAUSED
        self._event("pause")

    def close(self) -> None:
        """End the run and close its observation log.  The final state stays available for checkpointing."""
        if self.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            raise LifecycleError(f"run is already {self.status.value}")
        self.status = RunStatus.COMPLETED
        self._event("close", terminal=self.terminal)
        close_log = getattr(self.log, "close", None)
        if close_log is not None:
            close_log()

    def _fail(self, error: BaseException) -> None:
        self.status = RunStatus.FAILED
        self._event("fail", error=f"{type(error).__name__}: {error}")

    def _enter(self) -> None:
        if self._busy:
            raise LifecycleError("re-entrant call: consumers must act on sources, not on the run")
        if self.status is RunStatus.CREATED:
            self.start()
        if self.status is not RunStatus.RUNNING:
            raise LifecycleError(f"run is {self.status.value}, not running")

    # ------------------------------------------------------------ inference

    def offer(self, obs: Observation) -> list[OutputEvent]:
        """Hand one arriving observation to the run.

        delivery -> topology -> validate -> transition -> commit -> notify.
        Returns the output events produced (often none or one).

        An observation the topology cannot use (unknown input, missing key
        or event time) and a statistical input that violates the
        procedure's contract are invalid inputs, handled by `on_invalid`.
        Delivery errors (OrderingError under the strict policy) are not:
        that policy was chosen to stop the run.
        """
        self._enter()
        self._busy = True
        try:
            if self.log is not None:
                self.log.append(obs)
            events = []
            try:
                for delivered in self.delivery.push(obs):
                    try:
                        inputs = self.topology.push(delivered)
                    except ContractViolation as violation:
                        self._invalid(violation, "topology", observation=delivered)
                        continue
                    for x in inputs:
                        event = self._apply(x, delivered)
                        if event is not None:
                            events.append(event)
            except Exception as error:
                if self.status is not RunStatus.FAILED:
                    self._fail(error)
                raise
            return events
        finally:
            self._busy = False

    def offer_batch(self, batch: Iterable[Observation]) -> list[OutputEvent]:
        """Transport-level batching: exactly the element-wise fold, nothing more."""
        events = []
        for obs in batch:
            events.extend(self.offer(obs))
        return events

    def step(self, x: StatInput) -> OutputEvent | None:
        """Apply one statistical input directly, bypassing delivery and topology.

        Not available while an observation log is attached: the log would
        no longer hold the whole history, and replaying it would silently
        give a different run.
        """
        if self.log is not None:
            raise LifecycleError("step() bypasses the observation log; use offer(), or a run without a log")
        self._enter()
        self._busy = True
        try:
            return self._apply(dict(x))
        finally:
            self._busy = False

    def _invalid(self, violation: ContractViolation, stage: str, *, observation=None, x=None) -> None:
        """An invalid input: always reported as a warning, then skipped or raised per `on_invalid`."""
        where = ""
        if observation is not None:
            where = f" from source {observation.source!r}"
            if observation.seq is not None:
                where += f" seq {observation.seq}"
        action = "skipped" if self.on_invalid == "skip" else "rejected"
        logger.warning(
            "run %s: %s invalid input%s after t=%d (%s): %s",
            self.run_id, action, where, self.t, stage, violation,
            extra={
                "incident": {
                    "kind": "invalid_input",
                    "run_id": self.run_id,
                    "t": self.t,
                    "stage": stage,
                    "action": action,
                    "error": str(violation),
                    "observation": observation.to_json() if observation is not None else None,
                    "input": x,
                }
            },
        )
        policy = self.on_invalid
        if policy == "skip":
            self.counters["invalid_skipped"] += 1
            self.counters["invalid_streak"] += 1
            limit = self.max_invalid_streak
            if limit is not None and self.counters["invalid_streak"] >= limit:
                streak = InvalidInputStreak(
                    f"run {self.run_id}: {limit} consecutive invalid inputs after t={self.t}; "
                    f"the source is probably broken (last: {violation})"
                )
                self._fail(streak)
                raise streak from violation
            return
        if isinstance(policy, str):  # "raise"
            self._fail(violation)
            raise violation
        error = policy(f"run {self.run_id}: invalid input after t={self.t} ({stage}): {violation}")
        self._fail(error)
        raise error from violation

    def _apply(self, x: dict, observation: Observation | None = None) -> OutputEvent | None:
        if self.terminal:
            self.counters["after_terminal"] += 1
            return None
        try:
            validate_input(self.procedure, x)
        except ContractViolation as violation:
            self._invalid(violation, "contract", observation=observation, x=x)
            return None
        rng = SplitMix64(self.rng_state) if self.rng_state is not None else None
        try:
            new_state, output = self.procedure.step(self.state, x, rng)
            digest = hashlib.sha256((self.history_digest + _canonical(x)).encode()).hexdigest()
        except ContractViolation as violation:
            # Raised by step for a data-dependent violation (e.g. a later stage of
            # a Chain).  Nothing was committed, so the input can be skipped whole.
            self._invalid(violation, "procedure", observation=observation, x=x)
            return None
        except Exception as error:
            self._fail(error)  # nothing was committed: self.state is still S_{t-1}
            raise
        # commit
        self.state = new_state
        if rng is not None:
            self.rng_state = rng.state
        self.t += 1
        self.counters["invalid_streak"] = 0
        self.history_digest = digest
        self.terminal = self.procedure.is_terminal(new_state)
        event = OutputEvent(self.run_id, self.t, output, self.terminal)
        for consumer in self.consumers:
            try:
                consumer(event)
            except Exception as error:  # observer failures never become inference failures
                record = {"t": self.t, "consumer": type(consumer).__name__, "error": f"{type(error).__name__}: {error}"}
                self.consumer_errors.append(record)
                logger.warning(
                    "run %s: consumer %s failed at t=%d: %s", self.run_id, record["consumer"], self.t, record["error"],
                    extra={"incident": {"kind": "consumer_error", "run_id": self.run_id, **record}},
                )
        return event

    # ---------------------------------------------------------- persistence

    def checkpoint(self) -> dict:
        """A JSON-serializable value sufficient to continue this run elsewhere."""
        if self.status is RunStatus.FAILED:
            raise LifecycleError("a failed run cannot be checkpointed; restore an earlier checkpoint")
        if self._busy:
            raise LifecycleError("checkpoints are taken between transitions")
        checkpoint: dict[str, Any] = _normalize(
            {
                "format": CHECKPOINT_FORMAT,
                "run_id": self.run_id,
                "t": self.t,
                "terminal": self.terminal,
                "procedure": self.procedure.identity(),
                "state": self.procedure.encode_state(self.state),
                "rng": _u64(self.rng_state),
                "topology": {"spec": self.topology.spec(), "state": self.topology.state()},
                "delivery": {"spec": self.delivery.spec(), "state": self.delivery.state()},
                "history_digest": self.history_digest,
                "log": self.log.state() if hasattr(self.log, "state") else None,
                "counters": self.counters,
                "consumer_errors": self.consumer_errors,
                "provenance": self.provenance,
            }
        )
        return checkpoint

    @classmethod
    def restore(
        cls,
        procedure: Procedure,
        checkpoint: dict,
        *,
        topology: Topology | None = None,
        delivery: Delivery | None = None,
        consumers: Sequence[Consumer] = (),
        on_invalid: str | type[Exception] | None = None,
        log: Any = FROM_CHECKPOINT,
        clock: Callable[[], float] = time.time,
    ) -> Run:
        """Continue an existing run from a checkpoint.

        The run keeps its identity and history digest.  It is returned
        PAUSED; `start()` (or a runtime) resumes it.  To begin a *new* run
        from an old state, construct `Run(..., initial_state=...,
        initialized_from=...)` instead: the numbers may be identical, the
        provenance is not.

        The observation log is restored together with the run.  By default
        the log recorded in the checkpoint is reopened from its path; pass
        an ObservationLog to use a moved copy, or ``log=None`` to continue
        without one.  Either way the log is verified against the checkpoint
        and cut back to it: records written after the checkpoint are deleted,
        and their number is recorded in the restore event.  Observations
        after the checkpoint must come again from the sources; preserving
        them for sources that cannot replay is the application's job (see
        `ObservationLog`).

        The run keeps its `on_invalid` policy, its `max_invalid_streak` and
        the current streak.  A policy that raises a custom
        exception class is recorded by name only, so that class has to be
        passed again as ``on_invalid``.
        """
        validate_checkpoint(checkpoint)
        topology = topology if topology is not None else Independent()
        delivery = delivery if delivery is not None else Delivery()
        for label, mine, theirs in (
            ("procedure", procedure.identity(), checkpoint["procedure"]),
            ("topology", topology.spec(), checkpoint["topology"]["spec"]),
            ("delivery", delivery.spec(), checkpoint["delivery"]["spec"]),
        ):
            if _normalize(mine) != theirs:
                raise IncompatibleCheckpoint(f"{label} mismatch: checkpoint has {theirs}, got {_normalize(mine)}")
        log_state = checkpoint.get("log")
        if log is FROM_CHECKPOINT:
            if log_state is None:
                log = None
            elif log_state["path"] is None:
                raise IncompatibleCheckpoint(
                    "the checkpointed run had an in-memory observation log; pass that ObservationLog "
                    "as log=..., or log=None to continue without one"
                )
            else:
                log = ObservationLog(log_state["path"])
        if log is not None and log_state is None and not (isinstance(log, ObservationLog) and log.is_empty()):
            raise IncompatibleCheckpoint(
                "the checkpointed run had no observation log; a log attached now must be empty"
            )
        prov = checkpoint["provenance"]
        recorded = prov["on_invalid"]
        if on_invalid is None:
            if recorded.startswith("raise:"):
                raise IncompatibleCheckpoint(
                    f"the run raises {recorded[6:]} on invalid input; pass that class as on_invalid=..."
                )
            on_invalid = recorded
        elif _policy_spec(on_invalid) != recorded:
            raise IncompatibleCheckpoint(
                f"on_invalid mismatch: checkpoint has {recorded!r}, got {_policy_spec(on_invalid)!r}"
            )
        run = cls(
            procedure,
            topology=topology,
            delivery=delivery,
            consumers=consumers,
            seed=None if prov["seed"] is None else int(prov["seed"]),
            run_id=checkpoint["run_id"],
            initial_state=_decoded(procedure, checkpoint["state"]),
            on_invalid=on_invalid,
            max_invalid_streak=prov.get("max_invalid_streak"),
            clock=clock,
        )
        try:
            topology.restore(checkpoint["topology"]["state"])
            delivery.restore(checkpoint["delivery"]["state"])
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleCheckpoint(f"malformed checkpoint: buffers cannot be restored ({error!r})") from error
        run.rng_state = None if checkpoint["rng"] is None else int(checkpoint["rng"], 16)
        run.t = checkpoint["t"]
        run.terminal = checkpoint["terminal"]
        run.history_digest = checkpoint["history_digest"]
        run.counters = {**run.counters, **checkpoint["counters"]}
        run.consumer_errors = list(checkpoint.get("consumer_errors", []))
        run.provenance = json.loads(json.dumps(prov))
        discarded = log.restore(log_state) if log is not None and log_state is not None else 0
        run.log = log
        run.status = RunStatus.PAUSED
        run._event(
            "restore",
            checkpoint=checkpoint_id(checkpoint),
            log=getattr(log, "path", None) if log is not None else None,
            log_discarded=discarded,
        )
        return run


def _decoded(procedure: Procedure, data: Any) -> Any:
    try:
        return procedure.decode_state(data)
    except (KeyError, TypeError, ValueError) as error:
        raise IncompatibleCheckpoint(
            f"malformed checkpoint: {procedure.name} cannot decode its state ({error!r})"
        ) from error


def checkpoint_id(checkpoint: dict) -> str:
    """Content hash of a checkpoint."""
    return hashlib.sha256(_canonical(checkpoint).encode()).hexdigest()
