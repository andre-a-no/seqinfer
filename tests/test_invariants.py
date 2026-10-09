"""The conformance invariants of the paper, as executable tests.

I1 synchronous/asynchronous equivalence     I5 consumer non-interference
I2 checkpoint equivalence                    I6 runtime independence
I3 replay equivalence                        I7 topology semantics
I4 batch transparency
"""
import asyncio
import random
import tempfile
import unittest
from pathlib import Path

from seqinfer import (
    Delivery,
    Independent,
    KeyJoin,
    Observation,
    PositionalPair,
    Recorder,
    Run,
    RunStatus,
    TimeAlign,
    difference,
    load_checkpoint,
    run_async,
    run_sync,
    save_checkpoint,
    trajectory,
)
from seqinfer.procedures import SPRT, Gaussian, LocalLevelKalman, TwoSPRT
from seqinfer.sources import ObservationLog, as_async, from_values, gaussian, replay, skip_to

from .helpers import cases, fixed_clock, reference, summary


def full_run(case):
    _, make_proc, make_topo, make_delivery, observations, seed = case
    rec = Recorder()
    run = Run(make_proc(), topology=make_topo(), delivery=make_delivery(), consumers=[rec], seed=seed, clock=fixed_clock)
    run_sync(run, observations, stop_on_terminal=False)
    return summary(run, rec)


class SyncAsyncEquivalence(unittest.TestCase):
    def test_one_source_concurrent_shards_in_sequence_order(self):
        observations = list(gaussian("x", 0.25, 1.0, seed=1, n=90))
        proc = TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.001, 0.001)
        expected = reference(proc, trajectory(proc, [{"x": o.value} for o in observations]))

        rnd = random.Random(0)
        shards = [observations[i::3] for i in range(3)]
        rec = Recorder()
        run = Run(proc, delivery=Delivery("sequence"), consumers=[rec])
        producers = [as_async(s, lambda: rnd.uniform(0, 0.001)) for s in shards]
        asyncio.run(run_async(run, producers, stop_on_terminal=False))
        self.assertEqual(summary(run, rec)["outputs"], expected)

    def test_two_sources_paired(self):
        a = list(gaussian("a", 0.4, 1.0, seed=8, n=60))
        b = list(gaussian("b", 0.0, 1.0, seed=8, n=60))

        def make():
            rec = Recorder()
            topo = PositionalPair(("a", "b")).map(difference("a", "b"), "a-b")
            return Run(SPRT(Gaussian(1.5), 0.0, 0.5, 1e-4, 1e-4), topology=topo, consumers=[rec], run_id="r"), rec

        sync_run, sync_rec = make()
        run_sync(sync_run, a + b, stop_on_terminal=False)

        rnd = random.Random(1)
        async_run, async_rec = make()
        producers = [as_async(a, lambda: rnd.uniform(0, 0.001)), as_async(b, lambda: rnd.uniform(0, 0.002))]
        asyncio.run(run_async(async_run, producers, maxsize=4, stop_on_terminal=False))
        self.assertEqual(summary(async_run, async_rec), summary(sync_run, sync_rec))


class CheckpointEquivalence(unittest.TestCase):
    def test_interrupt_persist_restore_continue(self):
        for case in cases():
            label, make_proc, make_topo, make_delivery, observations, seed = case
            expected = full_run(case)
            for k in (0, 1, 17, len(observations) // 2, len(observations) - 1):
                with self.subTest(case=label, k=k), tempfile.TemporaryDirectory() as tmp:
                    rec = Recorder()
                    first = Run(
                        make_proc(),
                        topology=make_topo(),
                        delivery=make_delivery(),
                        consumers=[rec],
                        seed=seed,
                        clock=fixed_clock,
                    )
                    run_sync(first, observations[:k], stop_on_terminal=False)
                    path = Path(tmp) / "run.json"
                    save_checkpoint(path, first.checkpoint())
                    del first

                    second = Run.restore(
                        make_proc(),
                        load_checkpoint(path),
                        topology=make_topo(),
                        delivery=make_delivery(),
                        consumers=[rec],
                        clock=fixed_clock,
                    )
                    self.assertIs(second.status, RunStatus.PAUSED)
                    run_sync(second, observations[k:], stop_on_terminal=False)
                    self.assertEqual(summary(second, rec), expected)

    def test_resume_with_redelivery_from_source_position(self):
        """At-least-once redelivery after a restore must not change the trajectory."""
        observations = list(gaussian("y", 1.0, 1.0, seed=9, n=50))
        proc = LocalLevelKalman(q=0.01, r=1.0)
        expected = reference(proc, trajectory(proc, [{"y": o.value} for o in observations]))

        rec = Recorder()
        first = Run(proc, delivery=Delivery("sequence"), consumers=[rec])
        run_sync(first, observations[:20])
        cp = first.checkpoint()
        self.assertEqual(cp["delivery"]["state"]["next"], {"y": 20})

        delivery = Delivery("sequence")
        second = Run.restore(LocalLevelKalman(q=0.01, r=1.0), cp, delivery=delivery, consumers=[rec])
        run_sync(second, observations[12:])  # the source rewinds further than needed
        self.assertEqual(delivery.duplicates, 8)
        self.assertEqual(summary(second, rec)["outputs"], expected)

        rec2 = Recorder()
        third = Run.restore(LocalLevelKalman(q=0.01, r=1.0), cp, delivery=Delivery("sequence"), consumers=[rec2])
        run_sync(third, skip_to(observations, third.delivery.positions()))  # the source seeks exactly
        self.assertEqual(third.delivery.duplicates, 0)
        self.assertEqual(third.history_digest, second.history_digest)


class ReplayEquivalence(unittest.TestCase):
    def test_replay_of_a_messy_arrival_history(self):
        clean = list(gaussian("x", 0.25, 1.0, seed=2, n=70))
        rnd = random.Random(3)
        messy = []
        for i in range(0, len(clean), 5):
            block = clean[i:i + 5]
            rnd.shuffle(block)
            messy += block + [rnd.choice(block)]  # out of order, plus a duplicate

        def make(log=None):
            rec = Recorder()
            proc = TwoSPRT(Gaussian(1.0), 0.0, 0.5, 0.001, 0.001)
            return Run(proc, delivery=Delivery("sequence"), consumers=[rec], log=log, run_id="r"), rec

        with tempfile.TemporaryDirectory() as tmp:
            log = ObservationLog(Path(tmp) / "arrivals.jsonl")
            live, live_rec = make(log)
            run_sync(live, messy, stop_on_terminal=False)
            self.assertEqual(live.delivery.duplicates, 14)

            again, again_rec = make()
            run_sync(again, replay(log.path), stop_on_terminal=False)
            self.assertEqual(summary(again, again_rec), summary(live, live_rec))

        proc = live.procedure
        in_order = reference(proc, trajectory(proc, [{"x": o.value} for o in clean]))
        self.assertEqual(summary(live, live_rec)["outputs"], in_order)


class BatchTransparency(unittest.TestCase):
    def test_batched_delivery_is_the_elementwise_fold(self):
        for case in cases():
            label, make_proc, make_topo, make_delivery, observations, seed = case
            with self.subTest(case=label):
                rec = Recorder()
                run = Run(
                    make_proc(), topology=make_topo(), delivery=make_delivery(), consumers=[rec], seed=seed,
                    clock=fixed_clock,
                )
                for i in range(0, len(observations), 7):
                    run.offer_batch(observations[i:i + 7])
                self.assertEqual(summary(run, rec), full_run(case))

    def test_a_decision_inside_a_batch_is_taken_at_the_same_step(self):
        values = [1.0] * 40
        proc = SPRT(Gaussian(1.0), 0.0, 1.0)
        stop = len(trajectory(proc, [{"x": v} for v in values]))
        self.assertLess(stop, 40)
        run = Run(proc)
        run.offer_batch(from_values("x", values))
        self.assertEqual(run.t, stop)
        self.assertEqual(run.counters["after_terminal"], 40 - stop)


class ConsumerNonInterference(unittest.TestCase):
    def test_consumers_do_not_change_the_trajectory_even_when_they_crash(self):
        def crash(event):
            raise RuntimeError("display disconnected")

        for case in cases():
            label, make_proc, make_topo, make_delivery, observations, seed = case
            with self.subTest(case=label):
                rec = Recorder()
                run = Run(
                    make_proc(),
                    topology=make_topo(),
                    delivery=make_delivery(),
                    consumers=[crash, rec, lambda e: None],
                    seed=seed,
                    clock=fixed_clock,
                )
                run_sync(run, observations, stop_on_terminal=False)
                self.assertEqual(summary(run, rec), full_run(case))
                self.assertIs(run.status, RunStatus.RUNNING)
                self.assertEqual(len(run.consumer_errors), run.t)


class RuntimeIndependence(unittest.TestCase):
    def test_same_logical_history_through_different_mechanisms(self):
        observations = list(gaussian("y", 2.0, 1.0, seed=5, n=40))
        proc = LocalLevelKalman(q=0.1, r=1.0)
        expected = reference(proc, trajectory(proc, [{"y": o.value} for o in observations]))

        def outputs(drive):
            rec = Recorder()
            run = Run(LocalLevelKalman(q=0.1, r=1.0), consumers=[rec])
            drive(run)
            return summary(run, rec)["outputs"]

        log = ObservationLog()
        for obs in observations:
            log.append(obs)

        self.assertEqual(outputs(lambda r: run_sync(r, observations)), expected)
        self.assertEqual(outputs(lambda r: run_sync(r, replay(log))), expected)
        self.assertEqual(outputs(lambda r: asyncio.run(run_async(r, [as_async(observations)]))), expected)
        self.assertEqual(outputs(lambda r: [r.step({"y": o.value}) for o in observations]), expected)


class TopologySemantics(unittest.TestCase):
    A = [Observation("a", v, seq=i, time=t, key=k) for i, (v, t, k) in enumerate(
        [(1, 0.0, "p"), (2, 1.0, "q"), (3, 2.0, "r"), (4, 3.0, "s")])]
    B = [Observation("b", v, seq=i, time=t, key=k) for i, (v, t, k) in enumerate(
        [(10, 0.2, "q"), (20, 2.4, "p"), (30, 3.1, "s")])]
    ARRIVAL = [A[0], A[1], B[0], A[2], B[1], A[3], B[2]]

    def emitted(self, topology, arrival=None):
        out = []
        for obs in arrival or self.ARRIVAL:
            out += topology.push(obs)
        return out

    def test_independent(self):
        self.assertEqual(
            self.emitted(Independent()),
            [{"a": 1}, {"a": 2}, {"b": 10}, {"a": 3}, {"b": 20}, {"a": 4}, {"b": 30}],
        )

    def test_positional_pairing(self):
        topo = PositionalPair(("a", "b"))
        self.assertEqual(
            self.emitted(topo), [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]
        )
        self.assertEqual(topo.state(), {"a": [4], "b": []})

    def test_key_join(self):
        topo = KeyJoin(("a", "b"))
        self.assertEqual(
            self.emitted(topo), [{"a": 2, "b": 10}, {"a": 1, "b": 20}, {"a": 4, "b": 30}]
        )
        self.assertEqual(topo.state(), [["r", {"a": 3}]])

    def test_time_alignment(self):
        topo = TimeAlign(("a", "b"), tolerance=0.5)
        self.assertEqual(
            self.emitted(topo), [{"a": 1, "b": 10}, {"a": 3, "b": 20}, {"a": 4, "b": 30}]
        )
        self.assertEqual(topo.unmatched, 1)  # a=2 at time 1.0 has no partner

    def test_confluent_operators_ignore_interleaving(self):
        rnd = random.Random(7)
        a = list(from_values("a", range(30), times=[float(i) for i in range(30)]))
        b = list(from_values("b", range(100, 130), times=[i + rnd.uniform(-0.7, 0.7) for i in range(30)]))
        b.sort(key=lambda o: o.time)
        b = [Observation("b", o.value, seq=i, time=o.time) for i, o in enumerate(b)]
        for make in (lambda: PositionalPair(("a", "b")), lambda: TimeAlign(("a", "b"), 0.4)):
            self.assertTrue(make().confluent)
            baseline = self.emitted(make(), a + b)
            for _ in range(20):
                ia, ib, arrival = 0, 0, []
                while ia < len(a) or ib < len(b):
                    if ib >= len(b) or (ia < len(a) and rnd.random() < 0.5):
                        arrival.append(a[ia]); ia += 1
                    else:
                        arrival.append(b[ib]); ib += 1
                self.assertEqual(self.emitted(make(), arrival), baseline)


if __name__ == "__main__":
    unittest.main()
