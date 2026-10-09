# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Run semantics: transactional updates, lifecycle, contracts, delivery, restore checks."""
import asyncio
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from seqinfer import (
    Chain,
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
            run = Run(FragileSum(), on_invalid="raise")
            with self.subTest(bad=bad), self.assertRaises(ContractViolation), self.assertLogs("seqinfer", "WARNING"):
                run.step(bad)
            self.assertEqual(run.state, Total())
            self.assertIs(run.status, RunStatus.FAILED)

    def test_skip_policy_records_what_it_dropped(self):
        rec = Recorder()
        run = Run(FragileSum(), consumers=[rec])  # skipping is the default
        with self.assertLogs("seqinfer", "WARNING"):
            run_sync(run, from_values("x", [1.0, -5.0, 2.0, float("inf"), 3.0]))
        self.assertEqual(rec.outputs, [1.0, 3.0, 6.0])
        self.assertEqual(run.counters["invalid_skipped"], 2)
        self.assertEqual(run.checkpoint()["counters"]["invalid_skipped"], 2)

    def test_joint_and_partial_inputs(self):
        with self.assertRaises(ContractViolation):
            Run(LocalLevelKalman(0.1, 1.0, control=True), on_invalid="raise").step({"y": 1.0})
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
        with self.assertLogs("seqinfer", "WARNING"):
            run.step({"y": 1.0})
        self.assertEqual(run.t, 1)
        self.assertIn("LifecycleError", run.consumer_errors[0]["error"])

    def test_restoring_continues_a_run_and_initializing_starts_a_new_one(self):
        first = Run(LocalLevelKalman(0.1, 1.0))
        run_sync(first, from_values("y", [1.0, 2.0, 3.0]))
        cp = first.checkpoint()

        restored = Run.restore(LocalLevelKalman(0.1, 1.0), cp)
        self.assertEqual(
            (restored.run_id, restored.t, restored.history_digest), (first.run_id, 3, first.history_digest)
        )
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


class TransformIdentity(unittest.TestCase):
    def test_transforms_are_recorded_by_name_version_and_config(self):
        from seqinfer import Difference, Field

        topo = PositionalPair(("a", "b")).map(Difference("a", "b"))
        self.assertEqual(
            topo.spec()["map"], {"name": "difference", "version": "1", "config": {"a": "a", "b": "b", "out": "x"}}
        )
        chain = Chain(FragileSum(), FragileSum(), link=Field("n"))
        self.assertEqual(chain.config()["link"]["config"], {"field": "n", "out": "x"})

    def test_a_changed_transform_refuses_to_restore(self):
        from seqinfer import Difference

        proc = SPRT(Gaussian(1.0), 0.0, 0.5)
        run = Run(proc, topology=PositionalPair(("a", "b")).map(Difference("a", "b")))
        run.offer(Observation("a", 1.0))
        checkpoint = run.checkpoint()
        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(proc, checkpoint, topology=PositionalPair(("a", "b")).map(Difference("b", "a")))
        Run.restore(proc, checkpoint, topology=PositionalPair(("a", "b")).map(Difference("a", "b")))

    def test_plain_functions_are_marked_name_only(self):
        topo = PositionalPair(("a", "b")).map(lambda x: {"x": x["a"]}, "first")
        self.assertEqual(topo.spec()["map"], {"name": "first", "identity": "name only"})
        with self.assertRaises(ValueError):
            PositionalPair(("a", "b")).map(lambda x: x)


class CsvSource(unittest.TestCase):
    def test_reads_converts_and_numbers_rows(self):
        from seqinfer.sources import from_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("t,patient,y\n0.5,p1,1.25\n1.5,p2,-2\n")
            rows = list(from_csv(path, "y", "y", time="t", key="patient"))
            got = [(o.value, o.seq, o.time, o.key) for o in rows]
            self.assertEqual(got, [(1.25, 0, 0.5, "p1"), (-2.0, 1, 1.5, "p2")])
            run = Run(FragileSum(), on_invalid="raise")
            run_sync(run, from_csv(path, "x", "t"))
            self.assertEqual(run.state.total, 2.0)

    def test_bad_values_and_columns_name_the_line(self):
        from seqinfer.sources import from_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("y\n1\nn/a\n")
            with self.assertRaises(ValueError) as caught:
                list(from_csv(path, "y", "y"))
            self.assertIn("line 3", str(caught.exception))
            with self.assertRaises(ValueError):
                list(from_csv(path, "y", "missing"))


class MalformedCheckpoints(unittest.TestCase):
    def setUp(self):
        run = Run(FragileSum())
        run.step({"x": 1.0})
        self.good = run.checkpoint()

    def damaged(self, change):
        cp = json.loads(json.dumps(self.good))
        change(cp)
        return cp

    def test_every_kind_of_damage_is_an_incompatible_checkpoint(self):
        damages = {
            "not an object": lambda cp: cp.clear(),
            "missing state": lambda cp: cp.pop("state"),
            "missing provenance": lambda cp: cp.pop("provenance"),
            "t as string": lambda cp: cp.update(t="1"),
            "negative t": lambda cp: cp.update(t=-1),
            "bad digest": lambda cp: cp.update(history_digest="x" * 64),
            "bad rng": lambda cp: cp.update(rng="zz"),
            "topology without state": lambda cp: cp["topology"].pop("state"),
            "provenance without seed": lambda cp: cp["provenance"].pop("seed"),
            "float counter": lambda cp: cp["counters"].update(invalid_skipped=1.5),
            "state of the wrong shape": lambda cp: cp.update(state={"unexpected": 1}),
            "broken log position": lambda cp: cp.update(log={"path": None}),
        }
        for label, change in damages.items():
            with self.subTest(label), self.assertRaises(IncompatibleCheckpoint):
                Run.restore(FragileSum(), self.damaged(change))
        Run.restore(FragileSum(), self.good)  # the undamaged one still works

    def test_load_checkpoint_reports_bad_files(self):
        from seqinfer import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            path.write_text("{ not json")
            with self.assertRaises(IncompatibleCheckpoint):
                load_checkpoint(path)
            save_checkpoint(path, self.damaged(lambda cp: cp.pop("counters")))
            with self.assertRaises(IncompatibleCheckpoint) as caught:
                load_checkpoint(path)
            self.assertIn("counters", str(caught.exception))


class StreamFailure(Exception):
    """An application's own error type for invalid input."""


class InvalidInputPolicy(unittest.TestCase):
    """Invalid input is skipped by default, raised on request, and reported either way."""

    def incidents(self, logs):
        return [r.incident for r in logs.records if hasattr(r, "incident")]

    def test_skipped_input_is_reported_as_a_warning(self):
        run = Run(FragileSum(), run_id="r")
        with self.assertLogs("seqinfer", "WARNING") as logs:
            run.offer(Observation("x", 1.0, seq=0))
            run.offer(Observation("x", -2.0, seq=1))
        self.assertEqual((run.t, run.state.total, run.counters["invalid_skipped"]), (1, 1.0, 1))
        (incident,) = self.incidents(logs)
        self.assertEqual(incident["kind"], "invalid_input")
        self.assertEqual((incident["t"], incident["stage"], incident["action"]), (1, "contract", "skipped"))
        self.assertEqual(incident["observation"]["seq"], 1)
        self.assertEqual(incident["input"], {"x": -2.0})
        self.assertIn("source 'x' seq 1", logs.output[0])

    def test_raise_is_explicit_and_still_reported(self):
        run = Run(FragileSum(), on_invalid="raise")
        with self.assertLogs("seqinfer", "WARNING") as logs, self.assertRaises(ContractViolation):
            run.step({"x": -1.0})
        self.assertEqual(self.incidents(logs)[0]["action"], "rejected")
        self.assertIs(run.status, RunStatus.FAILED)

    def test_raise_a_chosen_exception(self):
        run = Run(FragileSum(), on_invalid=StreamFailure)
        self.assertEqual(run.provenance["on_invalid"], "raise:tests.test_run.StreamFailure")
        run.step({"x": 1.0})
        checkpoint = run.checkpoint()
        with self.assertLogs("seqinfer", "WARNING"), self.assertRaises(StreamFailure) as caught:
            run.step({"x": -1.0})
        self.assertIsInstance(caught.exception.__cause__, ContractViolation)
        self.assertIs(run.status, RunStatus.FAILED)

        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(FragileSum(), checkpoint)  # the class is not in the checkpoint
        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(FragileSum(), checkpoint, on_invalid="skip")
        restored = Run.restore(FragileSum(), checkpoint, on_invalid=StreamFailure)
        self.assertEqual(restored.on_invalid, StreamFailure)

    def test_invalid_input_for_a_later_chain_stage_skips_the_whole_step(self):
        chain = Chain(FragileSum(), FragileSum(), link=lambda total: {"x": total - 3.0}, link_name="total-3")
        run = Run(chain)
        with self.assertLogs("seqinfer", "WARNING") as logs:
            run.step({"x": 1.0})  # second stage would get -2.0
        self.assertEqual((run.t, run.state), (0, (Total(), Total())))
        (incident,) = self.incidents(logs)
        self.assertEqual(incident["stage"], "procedure")
        self.assertIn("second stage, via 'total-3'", incident["error"])
        run.step({"x": 5.0})
        self.assertEqual(run.state, (Total(1, 5.0), Total(1, 2.0)))

    def test_observations_the_topology_cannot_use_are_invalid_inputs(self):
        run = Run(MeanDifference(), topology=KeyJoin(("a", "b")))
        with self.assertLogs("seqinfer", "WARNING") as logs:
            run.offer(Observation("a", 1.0))  # no key
            run.offer(Observation("c", 1.0, key="k"))  # not an input of the topology
        self.assertEqual([i["stage"] for i in self.incidents(logs)], ["topology", "topology"])
        self.assertEqual(run.counters["invalid_skipped"], 2)
        self.assertIs(run.status, RunStatus.RUNNING)

    def test_a_transform_failing_on_bad_data_skips_only_that_input(self):
        from seqinfer import Difference

        run = Run(MeanDifference(), topology=PositionalPair(("a", "b")).map(Difference("a", "b", out="a")))
        for v in (1.0, "broken", 3.0):  # the "a" side runs ahead; a string slipped through an adapter
            run.offer(Observation("a", v))
        with self.assertLogs("seqinfer", "WARNING") as logs:
            for v in (0.5, 0.5, 0.5):  # three pairs are emitted, the middle one cannot be differenced
                run.offer(Observation("b", v))
        self.assertIs(run.status, RunStatus.RUNNING)
        self.assertEqual((run.t, run.counters["invalid_skipped"]), (2, 1))
        self.assertEqual(run.state.mean_a, 1.5)  # (1 - 0.5 + 3 - 0.5) / 2
        self.assertIn("difference failed", logs.output[0])

    def test_consumer_failures_are_reported_as_warnings(self):
        def broken(event):
            raise RuntimeError("disk full")

        with self.assertLogs("seqinfer", "WARNING") as logs:
            Run(FragileSum(), consumers=[broken]).step({"x": 1.0})
        self.assertEqual(self.incidents(logs)[0]["kind"], "consumer_error")

    def test_a_streak_of_invalid_inputs_stops_the_run(self):
        from seqinfer import InvalidInputStreak

        run = Run(FragileSum(), max_invalid_streak=3)
        self.assertEqual(run.provenance["max_invalid_streak"], 3)
        with self.assertLogs("seqinfer", "WARNING"):
            for value in (-1.0, -1.0, 1.0, -1.0, -1.0):  # a valid input resets the streak
                run.step({"x": value})
            checkpoint = run.checkpoint()
            self.assertEqual(run.counters["invalid_streak"], 2)
            with self.assertRaises(InvalidInputStreak) as caught:
                run.step({"x": -1.0})
        self.assertIsInstance(caught.exception.__cause__, ContractViolation)
        self.assertIs(run.status, RunStatus.FAILED)
        self.assertEqual(run.counters["invalid_skipped"], 5)

        restored = Run.restore(FragileSum(), checkpoint)  # threshold and current streak carry over
        restored.start()
        with self.assertLogs("seqinfer", "WARNING"), self.assertRaises(InvalidInputStreak):
            restored.step({"x": -1.0})

    def test_streak_threshold_must_be_positive(self):
        for bad in (0, -1, 2.5, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Run(FragileSum(), max_invalid_streak=bad)

    def test_unknown_policies_are_refused(self):
        for policy in ("ignore", ValueError("not a class"), int):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                Run(FragileSum(), on_invalid=policy)


class Files(unittest.TestCase):
    """Log and sink files stay open between records; durability is configurable."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_the_log_file_is_opened_once(self):
        for sync in ("flush", "fsync"):
            with self.subTest(sync=sync), ObservationLog(self.dir / f"{sync}.jsonl", sync=sync) as log:
                run = Run(FragileSum(), log=log)
                run.offer(Observation("x", 1.0, seq=0))
                handle = log._file._file
                for i in range(1, 50):
                    run.offer(Observation("x", 1.0, seq=i))
                self.assertIs(log._file._file, handle)
                self.assertEqual(len(list(replay(log.path))), 50)  # flushed: visible to readers
                run.close()
                self.assertTrue(log._file.closed)

    def test_unknown_sync_mode_is_refused(self):
        with self.assertRaises(ValueError):
            ObservationLog(self.dir / "x.jsonl", sync="sometimes")

    def test_jsonl_sink(self):
        from seqinfer import JsonlSink

        path = self.dir / "outputs.jsonl"
        with JsonlSink(path, encode=lambda o: o) as sink:
            run = Run(FragileSum(), consumers=[sink])
            for _ in range(3):
                run.step({"x": 1.0})
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([line["output"] for line in lines], [1.0, 2.0, 3.0])


class LogRestore(unittest.TestCase):
    """The observation log is part of the run's persistent state."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.source = list(from_values("x", [0.5] * 30))

    def make(self, log):
        if log is not None:
            self.addCleanup(log.close)
        return Run(FragileSum(), delivery=Delivery("strict"), log=log, run_id="r", clock=lambda: 0.0)

    def restore(self, checkpoint, **kwargs):
        run = Run.restore(FragileSum(), checkpoint, delivery=Delivery("strict"), **kwargs)
        if run.log is not None:
            self.addCleanup(run.log.close)
        return run

    def crash_after_checkpoint(self, path):
        """Run 20 observations, checkpoint, run 10 more, then 'crash'."""
        run = self.make(ObservationLog(path))
        run_sync(run, self.source[:20])
        checkpoint = json.loads(json.dumps(run.checkpoint()))
        run_sync(run, self.source[20:])
        run.log.close()
        return checkpoint

    def test_restore_reopens_the_log_and_deletes_the_abandoned_tail(self):
        path = self.dir / "arrivals.jsonl"
        checkpoint = self.crash_after_checkpoint(path)
        self.assertEqual(checkpoint["log"]["path"], str(path))
        self.assertEqual(checkpoint["log"]["count"], 20)

        resumed = self.restore(checkpoint, clock=lambda: 0.0)
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
                    self.restore(checkpoint)
                self.assertEqual(path.read_text(), content)  # a refused restore changes nothing

    def test_a_moved_log_can_be_passed_explicitly(self):
        path = self.dir / "arrivals.jsonl"
        checkpoint = self.crash_after_checkpoint(path)
        moved = path.rename(self.dir / "moved.jsonl")
        resumed = self.restore(checkpoint, log=ObservationLog(moved))
        self.assertEqual(len(list(replay(moved))), 20)
        self.assertEqual(resumed.log.path, str(moved))

    def test_continuing_without_a_log_is_explicit(self):
        checkpoint = self.crash_after_checkpoint(self.dir / "arrivals.jsonl")
        resumed = self.restore(checkpoint, log=None)
        self.assertIsNone(resumed.log)
        self.assertIsNone(resumed.provenance["events"][-1]["log"])

    def test_in_memory_log(self):
        log = ObservationLog()
        run = self.make(log)
        run_sync(run, self.source[:20])
        checkpoint = run.checkpoint()
        run_sync(run, self.source[20:])
        with self.assertRaises(IncompatibleCheckpoint):
            self.restore(checkpoint)
        self.restore(checkpoint, log=log)
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
        with self.assertLogs("seqinfer", "WARNING"):
            run.step({"x": 1.0})
        restored = Run.restore(FragileSum(), run.checkpoint())
        self.assertEqual(restored.consumer_errors, run.consumer_errors)
        self.assertEqual(len(restored.consumer_errors), 1)

    def test_provenance_records_the_package_version(self):
        import seqinfer

        self.assertEqual(Run(FragileSum()).provenance["seqinfer"], seqinfer.__version__)


if __name__ == "__main__":
    unittest.main()
