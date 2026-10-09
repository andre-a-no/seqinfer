"""Run semantics: transactional updates, lifecycle, contracts, delivery, restore checks."""
import asyncio
import unittest
from dataclasses import dataclass

from seqinfer import (
    ContractViolation,
    Delivery,
    IncompatibleCheckpoint,
    InputContract,
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
from seqinfer.procedures import SPRT, BootstrapParticleFilter, Gaussian, LocalLevelKalman, MeanDifference
from seqinfer.sources import as_async, from_values


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


if __name__ == "__main__":
    unittest.main()
