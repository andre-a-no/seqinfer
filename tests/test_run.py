# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Run semantics: transactional updates, lifecycle, contracts, delivery, restore checks."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass

from seqinfer import (
    ContractViolation,
    Delivery,
    IncompatibleCheckpoint,
    InputContract,
    KeyJoin,
    LifecycleError,
    NumericalError,
    Observation,
    OrderingError,
    PositionalPair,
    Procedure,
    Recorder,
    Run,
    RunStatus,
    run_async,
    run_sync,
)
from seqinfer.procedures import CUSUM, EMA, SPRT, BootstrapParticleFilter, Gaussian, LocalLevelKalman, MeanDifference
from seqinfer.sources import ObservationLog, as_async, from_values, replay, skip_to

try:
    import numpy
except ImportError:  # numpy is optional: it only appears here as a source of foreign number types
    numpy = None


@dataclass(frozen=True)
class Total:
    n: int = 0
    total: float = 0.0


class FragileSum(Procedure):
    """Adds its inputs; overflows on purpose when the total exceeds a limit."""

    name = "fragile_sum"

    def config(self):
        return {"limit": 100.0}

    def inputs(self):
        return {"x": InputContract("x", lower=0.0, unit="mm")}

    def initial_state(self):
        return Total()

    def step(self, state, x, rng):
        total = state.total + x["x"]
        if total > 100.0:
            raise NumericalError("overflow")
        return Total(state.n + 1, total), total

    def decode_state(self, data):
        return Total(**data)


class TransactionalUpdates(unittest.TestCase):
    def test_failed_transition_keeps_the_previous_state(self):
        run = Run(FragileSum())
        run.step({"x": 60.0})
        before = (run.state, run.t, run.history_digest)
        with self.assertRaises(NumericalError):
            run.step({"x": 60.0})
        self.assertEqual((run.state, run.t, run.history_digest), before)
        self.assertIs(run.status, RunStatus.FAILED)
        with self.assertRaises(LifecycleError):
            run.step({"x": 1.0})
        with self.assertRaises(LifecycleError):
            run.checkpoint()

    def test_invalid_input_is_rejected_before_inference(self):
        for bad in ({"x": -1.0}, {"x": float("nan")}, {"x": "3"}, {"x": True}, {"y": 1.0}, {}):
            run = Run(FragileSum())
            with self.subTest(bad=bad), self.assertRaises(ContractViolation):
                run.step(bad)
            self.assertEqual(run.state, Total())
            self.assertIs(run.status, RunStatus.FAILED)

    def test_skip_policy_records_what_it_dropped(self):
        rec = Recorder()
        run = Run(FragileSum(), on_invalid="skip", consumers=[rec])
        run_sync(run, from_values("x", [1.0, -5.0, 2.0, float("inf"), 3.0]))
        self.assertEqual(rec.outputs, [1.0, 3.0, 6.0])
        self.assertEqual(run.counters["invalid_skipped"], 2)
        self.assertEqual(run.checkpoint()["counters"]["invalid_skipped"], 2)

    def test_joint_and_partial_inputs(self):
        with self.assertRaises(ContractViolation):
            Run(LocalLevelKalman(0.1, 1.0, control=True)).step({"y": 1.0})
        run = Run(MeanDifference())
        run.step({"a": 1.0})
        run.step({"b": 2.0})
        run.step({"a": 3.0, "b": 4.0})
        self.assertEqual((run.state.n_a, run.state.n_b), (2, 2))

    def test_terminal_states_are_absorbing(self):
        run = Run(SPRT(Gaussian(1.0), 0.0, 1.0))
        run_sync(run, from_values("x", [1.0] * 50), stop_on_terminal=False)
        self.assertTrue(run.terminal)
        self.assertEqual(run.t + run.counters["after_terminal"], 50)
        self.assertEqual(run.state.n, run.t)


class Lifecycle(unittest.TestCase):
    def test_transitions(self):
        run = Run(LocalLevelKalman(0.1, 1.0), clock=lambda: 1.0)
        self.assertIs(run.status, RunStatus.CREATED)
        run.step({"y": 1.0})
        self.assertIs(run.status, RunStatus.RUNNING)
        run.pause()
        with self.assertRaises(LifecycleError):
            run.step({"y": 1.0})
        run.start(runtime="manual")
        run.step({"y": 1.0})
        run.close()
        self.assertIs(run.status, RunStatus.COMPLETED)
        with self.assertRaises(LifecycleError):
            run.start()
        self.assertEqual([e["event"] for e in run.provenance["events"]], ["start", "pause", "resume", "close"])
        self.assertEqual(run.checkpoint()["t"], 2)

    def test_consumers_cannot_drive_the_run(self):
        run = Run(LocalLevelKalman(0.1, 1.0))
        run.consumers.append(lambda event: run.step({"y": 0.0}))
        run.step({"y": 1.0})
        self.assertEqual(run.t, 1)
        self.assertIn("LifecycleError", run.consumer_errors[0]["error"])

    def test_restoring_continues_a_run_and_initializing_starts_a_new_one(self):
        first = Run(LocalLevelKalman(0.1, 1.0))
        run_sync(first, from_values("y", [1.0, 2.0, 3.0]))
        cp = first.checkpoint()

        restored = Run.restore(LocalLevelKalman(0.1, 1.0), cp)
        self.assertEqual((restored.run_id, restored.t, restored.history_digest), (first.run_id, 3, first.history_digest))
        self.assertEqual(restored.provenance["events"][-1]["event"], "restore")

        proc = LocalLevelKalman(0.1, 1.0)
        fresh = Run(proc, initial_state=proc.decode_state(cp["state"]), initialized_from={"run": first.run_id, "t": 3})
        self.assertEqual(fresh.state, restored.state)
        self.assertNotEqual(fresh.run_id, first.run_id)
        self.assertEqual(fresh.t, 0)
        self.assertNotEqual(fresh.history_digest, first.history_digest)

    def test_checkpoints_are_checked_against_what_they_are_restored_into(self):
        run = Run(LocalLevelKalman(0.1, 1.0))
        run.step({"y": 1.0})
        cp = run.checkpoint()
        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(LocalLevelKalman(0.2, 1.0), cp)
        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(LocalLevelKalman(0.1, 1.0), cp, delivery=Delivery("strict"))
        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(LocalLevelKalman(0.1, 1.0), cp, topology=PositionalPair(("y", "u")))
        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(LocalLevelKalman(0.1, 1.0), {**cp, "format": "something/else"})

    def test_randomized_procedures_need_a_seed(self):
        with self.assertRaises(ValueError):
            Run(BootstrapParticleFilter(0.1, 1.0))

    def test_provenance_describes_the_experiment(self):
        run = Run(FragileSum(), topology=None, delivery=Delivery("strict"), seed=None, clock=lambda: 5.0)
        prov = run.checkpoint()["provenance"]
        self.assertEqual(prov["procedure"], {"name": "fragile_sum", "version": "1", "config": {"limit": 100.0}})
        self.assertEqual(prov["inputs"]["x"]["unit"], "mm")
        self.assertEqual(prov["topology"], {"type": "independent"})
        self.assertEqual(prov["delivery"]["policy"], "strict")
        self.assertEqual(prov["created_at"], 5.0)


class DeliveryPolicies(unittest.TestCase):
    def obs(self, seqs):
        return [Observation("x", float(s), seq=s) for s in seqs]

    def delivered(self, delivery, seqs):
        out = []
        for o in self.obs(seqs):
            out += [d.seq for d in delivery.push(o)]
        return out

    def test_arrival_order_with_deduplication(self):
        d = Delivery("arrival")
        self.assertEqual(self.delivered(d, [0, 2, 1, 2, 0, 3]), [0, 2, 1, 3])
        self.assertEqual(d.duplicates, 2)
        self.assertEqual(d.positions(), {"x": 4})

    def test_sequence_order_holds_back_until_contiguous(self):
        d = Delivery("sequence")
        self.assertEqual(self.delivered(d, [2, 1]), [])
        self.assertEqual(d.pending(), 2)
        self.assertEqual(self.delivered(d, [0, 0, 4, 3]), [0, 1, 2, 3, 4])
        self.assertEqual(d.duplicates, 1)

    def test_strict_rejects_gaps(self):
        d = Delivery("strict")
        self.assertEqual(self.delivered(d, [0, 1, 1]), [0, 1])
        with self.assertRaises(OrderingError):
            d.push(Observation("x", 0.0, seq=3))
        with self.assertRaises(OrderingError):
            Delivery("strict", dedup=False).push(Observation("x", 0.0, seq=-1))
        with self.assertRaises(OrderingError):
            Delivery("sequence").push(Observation("x", 0.0))

    def test_state_round_trip(self):
        d = Delivery("arrival")
        self.delivered(d, [0, 1, 5, 3])
        e = Delivery("arrival")
        e.restore(d.state())
        self.assertEqual(self.delivered(e, [5, 2, 3, 4]), [2, 4])
        self.assertEqual(e.positions(), {"x": 6})


class AsyncRuntime(unittest.TestCase):
    def test_blocking_backpressure_loses_nothing(self):
        run = Run(LocalLevelKalman(0.1, 1.0))
        n = asyncio.run(run_async(run, [as_async(from_values("y", [1.0] * 200))], maxsize=1))
        self.assertEqual((n, run.t, run.counters["dropped_by_runtime"]), (200, 200, 0))
        self.assertEqual(run.provenance["events"][0]["runtime"], "async/push/block")

    def test_dropping_is_counted(self):
        async def burst():
            for obs in from_values("y", [1.0] * 100):
                yield obs  # never waits, so the consumer cannot keep up

        run = Run(LocalLevelKalman(0.1, 1.0))
        asyncio.run(run_async(run, [burst()], maxsize=4, overflow="drop"))
        self.assertEqual(run.t, 4)
        self.assertEqual(run.counters["dropped_by_runtime"], 96)

    def test_stops_acquisition_after_a_decision(self):
        run = Run(SPRT(Gaussian(1.0), 0.0, 1.0))
        consumed = asyncio.run(run_async(run, [as_async(from_values("x", [1.0] * 10_000))]))
        self.assertTrue(run.terminal)
        self.assertEqual(consumed, run.t)
        self.assertLess(consumed, 20)

    def test_source_errors_surface(self):
        async def broken():
            yield Observation("y", 1.0)
            raise ConnectionError("instrument offline")

        run = Run(LocalLevelKalman(0.1, 1.0))
        with self.assertRaises(ConnectionError):
            asyncio.run(run_async(run, [broken()]))
        self.assertEqual(run.t, 1)
        self.assertIs(run.status, RunStatus.RUNNING)  # a source error is not an inference error


class ContractMessages(unittest.TestCase):
    """Contracts are strict; a rejection says what to change in the adapter."""

    def rejection(self, contract, value):
        with self.assertRaises(ContractViolation) as caught:
            contract.validate(value)
        return str(caught.exception)

    def test_bool_is_not_a_binary_value(self):
        message = self.rejection(InputContract("x", kind="binary"), True)
        self.assertIn("got bool True", message)
        self.assertIn("int(value)", message)

    def test_strings_are_parsed_in_the_adapter(self):
        self.assertIn("parse strings in the adapter", self.rejection(InputContract("x"), "3.5"))

    @unittest.skipIf(numpy is None, "numpy is not installed")
    def test_numpy_scalars_are_rejected_with_a_conversion_hint(self):
        # float64 subclasses float; it is rejected all the same, like float32 and int64
        for value in (numpy.float64(1.0), numpy.float32(1.0), numpy.int64(1)):
            with self.subTest(type=type(value).__name__):
                message = self.rejection(InputContract("x"), value)
                self.assertIn(f"numpy.{type(value).__name__}", message)
                self.assertIn("value.item()", message)

    @unittest.skipIf(numpy is None, "numpy is not installed")
    def test_numpy_vectors_are_rejected_with_a_conversion_hint(self):
        message = self.rejection(InputContract("v", shape=(2,)), numpy.array([1.0, 2.0]))
        self.assertIn(".tolist()", message)


class LogRestore(unittest.TestCase):
    """The observation log is part of the run's persistent state."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.source = list(from_values("x", [0.5] * 30))

    def make(self, log):
        return Run(FragileSum(), delivery=Delivery("strict"), log=log, run_id="r", clock=lambda: 0.0)

    def crash_after_checkpoint(self, path):
        """Run 20 observations, checkpoint, run 10 more, then 'crash'."""
        run = self.make(ObservationLog(path))
        run_sync(run, self.source[:20])
        checkpoint = json.loads(json.dumps(run.checkpoint()))
        run_sync(run, self.source[20:])
        return checkpoint

    def test_restore_reopens_the_log_and_deletes_the_abandoned_tail(self):
        path = self.dir / "arrivals.jsonl"
        checkpoint = self.crash_after_checkpoint(path)
        self.assertEqual(checkpoint["log"]["path"], str(path))
        self.assertEqual(checkpoint["log"]["count"], 20)

        resumed = Run.restore(FragileSum(), checkpoint, delivery=Delivery("strict"), clock=lambda: 0.0)
        self.assertEqual(resumed.log.path, str(path))
        self.assertEqual(len(list(replay(path))), 20)
        event = resumed.provenance["events"][-1]
        self.assertEqual(event["log"], str(path))
        self.assertEqual(event["log_discarded"], 10)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["arrivals.jsonl"])

        run_sync(resumed, skip_to(self.source, resumed.delivery.positions()))
        straight = self.make(ObservationLog(self.dir / "straight.jsonl"))
        run_sync(straight, self.source)
        self.assertEqual(path.read_bytes(), (self.dir / "straight.jsonl").read_bytes())
        self.assertEqual(resumed.history_digest, straight.history_digest)

        again = self.make(None)
        run_sync(again, replay(path))
        self.assertEqual(again.history_digest, straight.history_digest)

    def test_a_log_that_does_not_match_the_checkpoint_is_refused(self):
        path = self.dir / "arrivals.jsonl"
        checkpoint = self.crash_after_checkpoint(path)
        lines = path.read_text().splitlines(keepends=True)
        for label, content in (
            ("truncated", "".join(lines[:15])),
            ("edited", lines[0].replace("0.5", "0.25") + "".join(lines[1:])),
        ):
            with self.subTest(label):
                path.write_text(content)
                with self.assertRaises(IncompatibleCheckpoint):
                    Run.restore(FragileSum(), checkpoint, delivery=Delivery("strict"))
                self.assertEqual(path.read_text(), content)  # a refused restore changes nothing

    def test_a_moved_log_can_be_passed_explicitly(self):
        path = self.dir / "arrivals.jsonl"
        checkpoint = self.crash_after_checkpoint(path)
        moved = path.rename(self.dir / "moved.jsonl")
        resumed = Run.restore(FragileSum(), checkpoint, delivery=Delivery("strict"), log=ObservationLog(moved))
        self.assertEqual(len(list(replay(moved))), 20)
        self.assertEqual(resumed.log.path, str(moved))

    def test_continuing_without_a_log_is_explicit(self):
        checkpoint = self.crash_after_checkpoint(self.dir / "arrivals.jsonl")
        resumed = Run.restore(FragileSum(), checkpoint, delivery=Delivery("strict"), log=None)
        self.assertIsNone(resumed.log)
        self.assertIsNone(resumed.provenance["events"][-1]["log"])

    def test_in_memory_log(self):
        log = ObservationLog()
        run = self.make(log)
        run_sync(run, self.source[:20])
        checkpoint = run.checkpoint()
        run_sync(run, self.source[20:])
        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(FragileSum(), checkpoint, delivery=Delivery("strict"))
        Run.restore(FragileSum(), checkpoint, delivery=Delivery("strict"), log=log)
        self.assertEqual(len(list(log)), 20)

    def test_a_new_run_refuses_a_used_log(self):
        path = self.dir / "arrivals.jsonl"
        self.crash_after_checkpoint(path)
        with self.assertRaises(ValueError):
            self.make(ObservationLog(path))


class Guards(unittest.TestCase):
    """Configurations that would fail silently are refused up front."""

    def test_cusum_needs_two_distinct_parameters(self):
        with self.assertRaises(ValueError):
            CUSUM(Gaussian(), 0.5, 0.5, threshold=5.0)

    def test_ema_reports_overflow(self):
        run = Run(EMA(0.5))
        run.step({"x": 1e308})
        with self.assertRaises(NumericalError):
            run.step({"x": -1e308})

    def test_key_join_keys_must_survive_a_checkpoint(self):
        for key in (("subject-7", 2), 1.5, True, None):
            with self.subTest(key=key), self.assertRaises(ContractViolation) as caught:
                KeyJoin(("a", "b")).push(Observation("a", 1.0, key=key))
            self.assertIn("as a string in the adapter", str(caught.exception))
        run = Run(MeanDifference(), topology=KeyJoin(("a", "b")))
        run.offer(Observation("a", 1.0, key="subject-7/2"))
        restored = Run.restore(MeanDifference(), run.checkpoint(), topology=KeyJoin(("a", "b")))
        restored.start()
        restored.offer(Observation("b", 3.0, key="subject-7/2"))
        self.assertEqual(restored.state.mean_a - restored.state.mean_b, -2.0)

    def test_dropping_is_refused_when_delivery_waits_for_gaps(self):
        for policy in ("sequence", "strict"):
            run = Run(FragileSum(), delivery=Delivery(policy))
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                asyncio.run(run_async(run, [as_async(from_values("x", [1.0]))], overflow="drop"))
            self.assertIs(run.status, RunStatus.CREATED)

    def test_step_is_refused_while_a_log_is_attached(self):
        run = Run(FragileSum(), log=ObservationLog())
        with self.assertRaises(LifecycleError):
            run.step({"x": 1.0})
        self.assertEqual(run.t, 0)

    def test_skip_to_honours_the_delivery_start(self):
        source = list(from_values("x", [1.0] * 8, start=5))
        delivery = Delivery("strict", start=5)
        self.assertEqual(len(list(skip_to(source, delivery))), 8)
        delivery.push(source[0])
        self.assertEqual([o.seq for o in skip_to(source, delivery)], list(range(6, 13)))

    def test_consumer_errors_survive_restore(self):
        def broken(event):
            raise RuntimeError("disk full")

        run = Run(FragileSum(), consumers=[broken])
        run.step({"x": 1.0})
        restored = Run.restore(FragileSum(), run.checkpoint())
        self.assertEqual(restored.consumer_errors, run.consumer_errors)
        self.assertEqual(len(restored.consumer_errors), 1)

    def test_provenance_records_the_package_version(self):
        import seqinfer

        self.assertEqual(Run(FragileSum()).provenance["seqinfer"], seqinfer.__version__)


if __name__ == "__main__":
    unittest.main()
