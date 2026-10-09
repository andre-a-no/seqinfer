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
import platform
import time
import uuid
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from .consumers import Consumer, OutputEvent
from .contracts import validate_input
from .core import Observation, Procedure, StatInput, statistical_rng
from .delivery import Delivery
from .errors import ContractViolation, IncompatibleCheckpoint, LifecycleError
from .rng import SplitMix64
from .sources import ObservationLog
from .topology import Independent, Topology

CHECKPOINT_FORMAT = "seqinfer.checkpoint/1"
_GENESIS = hashlib.sha256(b"seqinfer.history/1").hexdigest()
#: Default of `Run.restore(log=...)`: reopen the log recorded in the checkpoint.
FROM_CHECKPOINT: Any = object()


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


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
        on_invalid: str = "raise",
        log: Any = None,
        clock: Callable[[], float] = time.time,
    ):
        if on_invalid not in ("raise", "skip"):
            raise ValueError("on_invalid must be 'raise' or 'skip'")
        if isinstance(log, ObservationLog) and not log.is_empty():
            raise ValueError(
                f"log {log.path or '(in memory)'} already holds records; a new run needs an empty log. "
                f"To continue a run, use Run.restore with its checkpoint"
            )
        self.procedure = procedure
        self.topology = topology if topology is not None else Independent()
        self.delivery = delivery if delivery is not None else Delivery()
        self.consumers = list(consumers)
        self.on_invalid = on_invalid
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
        self.counters = {"invalid_skipped": 0, "after_terminal": 0, "dropped_by_runtime": 0}
        self.consumer_errors: list[dict] = []
        self.provenance = {
            "run_id": self.run_id,
            "created_at": clock(),
            "procedure": procedure.identity(),
            "inputs": {n: c.spec() for n, c in procedure.inputs().items()},
            "topology": self.topology.spec(),
            "delivery": self.delivery.spec(),
            "seed": seed,
            "initialized_from": initialized_from,
            "on_invalid": on_invalid,
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
        """End the run.  Its final state stays available for checkpointing."""
        if self.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            raise LifecycleError(f"run is already {self.status.value}")
        self.status = RunStatus.COMPLETED
        self._event("close", terminal=self.terminal)

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
        """
        self._enter()
        self._busy = True
        try:
            if self.log is not None:
                self.log.append(obs)
            events = []
            try:
                for delivered in self.delivery.push(obs):
                    for x in self.topology.push(delivered):
                        event = self._apply(x)
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
        """Apply one statistical input directly, bypassing delivery and topology."""
        self._enter()
        self._busy = True
        try:
            return self._apply(dict(x))
        finally:
            self._busy = False

    def _apply(self, x: dict) -> OutputEvent | None:
        if self.terminal:
            self.counters["after_terminal"] += 1
            return None
        try:
            validate_input(self.procedure, x)
        except ContractViolation as error:
            if self.on_invalid == "skip":
                self.counters["invalid_skipped"] += 1
                return None
            self._fail(error)
            raise
        rng = SplitMix64(self.rng_state) if self.rng_state is not None else None
        try:
            new_state, output = self.procedure.step(self.state, x, rng)
            digest = hashlib.sha256((self.history_digest + _canonical(x)).encode()).hexdigest()
        except Exception as error:
            self._fail(error)  # nothing was committed: self.state is still S_{t-1}
            raise
        # commit
        self.state = new_state
        if rng is not None:
            self.rng_state = rng.state
        self.t += 1
        self.history_digest = digest
        self.terminal = self.procedure.is_terminal(new_state)
        event = OutputEvent(self.run_id, self.t, output, self.terminal)
        for consumer in self.consumers:
            try:
                consumer(event)
            except Exception as error:  # observer failures never become inference failures
                self.consumer_errors.append(
                    {"t": self.t, "consumer": type(consumer).__name__, "error": f"{type(error).__name__}: {error}"}
                )
        return event

    # ---------------------------------------------------------- persistence

    def checkpoint(self) -> dict:
        """A JSON-serializable value sufficient to continue this run elsewhere."""
        if self.status is RunStatus.FAILED:
            raise LifecycleError("a failed run cannot be checkpointed; restore an earlier checkpoint")
        if self._busy:
            raise LifecycleError("checkpoints are taken between transitions")
        return _normalize(
            {
                "format": CHECKPOINT_FORMAT,
                "run_id": self.run_id,
                "t": self.t,
                "terminal": self.terminal,
                "procedure": self.procedure.identity(),
                "state": self.procedure.encode_state(self.state),
                "rng": self.rng_state,
                "topology": {"spec": self.topology.spec(), "state": self.topology.state()},
                "delivery": {"spec": self.delivery.spec(), "state": self.delivery.state()},
                "history_digest": self.history_digest,
                "log": self.log.state() if hasattr(self.log, "state") else None,
                "counters": self.counters,
                "provenance": self.provenance,
            }
        )

    @classmethod
    def restore(
        cls,
        procedure: Procedure,
        checkpoint: dict,
        *,
        topology: Topology | None = None,
        delivery: Delivery | None = None,
        consumers: Sequence[Consumer] = (),
        log: Any = FROM_CHECKPOINT,
        clock: Callable[[], float] = time.time,
    ) -> "Run":
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
        and cut back to it, and records written after the checkpoint are
        moved to a sidecar file named in the restore event.
        """
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise IncompatibleCheckpoint(f"unsupported checkpoint format {checkpoint.get('format')!r}")
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
            raise IncompatibleCheckpoint("the checkpointed run had no observation log; a log attached now must be empty")
        prov = checkpoint["provenance"]
        run = cls(
            procedure,
            topology=topology,
            delivery=delivery,
            consumers=consumers,
            seed=prov["seed"],
            run_id=checkpoint["run_id"],
            initial_state=procedure.decode_state(checkpoint["state"]),
            on_invalid=prov["on_invalid"],
            clock=clock,
        )
        topology.restore(checkpoint["topology"]["state"])
        delivery.restore(checkpoint["delivery"]["state"])
        run.rng_state = checkpoint["rng"]
        run.t = checkpoint["t"]
        run.terminal = checkpoint["terminal"]
        run.history_digest = checkpoint["history_digest"]
        run.counters = dict(checkpoint["counters"])
        run.provenance = json.loads(json.dumps(prov))
        orphaned = log.restore(log_state) if log is not None and log_state is not None else None
        run.log = log
        run.status = RunStatus.PAUSED
        run._event(
            "restore",
            checkpoint=checkpoint_id(checkpoint),
            log=getattr(log, "path", None) if log is not None else None,
            log_orphaned=orphaned,
        )
        return run


def checkpoint_id(checkpoint: dict) -> str:
    """Content hash of a checkpoint."""
    return hashlib.sha256(_canonical(checkpoint).encode()).hexdigest()
