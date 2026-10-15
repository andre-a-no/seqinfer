# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Adapters: data frames, SQL, Kafka; resuming each from a checkpoint without loss or double counting.

pandas is used when installed, polars likewise; a stand-in with the same
row interface covers polars otherwise.  SQL runs on sqlite3.  Kafka is
covered by stand-ins for the message objects of kafka-python (attributes)
and confluent-kafka (methods, (type, ms) timestamps); no broker is needed.
"""
import json
import math
import random
import sqlite3
import unittest

from sequential_analysis import Delivery, KeyJoin, Recorder, Run, run_sync
from sequential_analysis.adapters import from_dataframe, from_kafka, from_sql, kafka_offsets, next_position
from sequential_analysis.errors import OrderingError
from sequential_analysis.procedures import ABTest, NormalMixtureSPRT

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


def ab_rows(n, seed):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        arm = rnd.randrange(2)
        x = rnd.gauss(0, 1)
        rows.append({"unit": i, "arm": arm, "x": x, "y": x + 0.3 * arm + rnd.gauss(0, 1)})
    return rows


def ab_run(delivery=None):
    rec = Recorder()
    run = Run(ABTest(covariate=True, start=20), topology=KeyJoin(("arm", "y", "x")),
              delivery=delivery or Delivery(), consumers=[rec], run_id="ab", clock=lambda: 0.0)
    return run, rec


def outputs(proc, rec):
    return json.loads(json.dumps([proc.encode_output(o) for o in rec.outputs]))


class FakePolars:
    """The part of polars.DataFrame the adapter uses."""

    def __init__(self, rows):
        self.rows = rows

    def iter_rows(self, named=False):
        assert named
        return iter(self.rows)


class DataFrames(unittest.TestCase):
    def test_rows_become_one_observation_per_mapped_column(self):
        rows = ab_rows(5, 1)
        obs = list(from_dataframe(FakePolars(rows), ["arm", "y"], seq="unit", key="unit"))
        self.assertEqual(len(obs), 10)
        self.assertEqual([(o.input, o.seq, o.key) for o in obs[:2]], [("arm", 0, 0), ("y", 0, 0)])
        self.assertEqual(obs[0].source, "arm")

    @unittest.skipIf(pd is None, "pandas is not installed")
    def test_pandas_types_and_missing_values(self):
        frame = pd.DataFrame({
            "unit": [0, 1, 2],
            "arm": [1, 0, 1],
            "y": [1.5, float("nan"), 2.5],
            "when": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z"]),
        })
        obs = list(from_dataframe(frame, {"arm": "arm", "y": "outcome"}, seq="unit", key="unit", time="when"))
        self.assertEqual([o.input for o in obs], ["arm", "outcome", "arm", "arm", "outcome"])
        self.assertIs(type(obs[0].value), int)  # numpy.int64 -> int, as integer contracts require
        self.assertIs(type(obs[0].seq), int)
        self.assertEqual(obs[1].time - obs[0].time, 0.0)
        self.assertEqual(obs[-1].time - obs[0].time, 2.0)
        kept = list(from_dataframe(frame, "y", missing="keep"))
        self.assertTrue(math.isnan(kept[1].value))

    @unittest.skipIf(pd is None, "pandas is not installed")
    def test_pandas_frame_drives_a_run_like_the_records(self):
        rows = ab_rows(200, 2)
        run, rec = ab_run()
        run_sync(run, from_dataframe(pd.DataFrame(rows), ["arm", "x", "y"], seq="unit", key="unit"))
        ref, ref_rec = ab_run()
        run_sync(ref, from_dataframe(FakePolars(rows), ["arm", "x", "y"], seq="unit", key="unit"))
        self.assertEqual(outputs(run.procedure, rec), outputs(ref.procedure, ref_rec))
        self.assertEqual(rec.outputs[-1].n, 200)

    @unittest.skipIf(pl is None, "polars is not installed")
    def test_polars(self):  # pragma: no cover - when polars is installed
        rows = ab_rows(20, 3)
        obs = list(from_dataframe(pl.DataFrame(rows), ["arm", "y"], seq="unit"))
        self.assertEqual(len(obs), 40)

    def test_errors(self):
        with self.assertRaises(TypeError):
            list(from_dataframe([1, 2], "x"))
        with self.assertRaises(ValueError):
            list(from_dataframe(FakePolars([{"x": 1.0}]), "y"))
        with self.assertRaises(ValueError):
            list(from_dataframe(FakePolars([{"x": 1.0, "s": 1.5}]), "x", seq="s"))
        with self.assertRaises(ValueError):
            list(from_dataframe(FakePolars([{"x": 1.0}]), []))


class SQL(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("create table events (unit integer primary key, arm integer, x real, y real)")
        self.rows = ab_rows(300, 4)
        self.db.executemany("insert into events values (:unit, :arm, :x, :y)", self.rows)
        self.query = "select unit, arm, x, y from events where unit >= ? order by unit"

    def tearDown(self):
        self.db.close()

    def read(self, start):
        return from_sql(self.db, self.query, (start,), inputs=["arm", "x", "y"], seq="unit", key="unit", batch=7)

    def test_resume_from_a_checkpoint_reads_on_from_where_the_run_stopped(self):
        full, full_rec = ab_run()
        run_sync(full, self.read(0))

        first, rec = ab_run()
        run_sync(first, (o for o in self.read(0) if o.seq < 120))
        checkpoint = first.checkpoint()
        delivery = Delivery()
        second = Run.restore(ABTest(covariate=True, start=20), checkpoint, topology=KeyJoin(("arm", "y", "x")),
                             delivery=delivery, consumers=[rec], clock=lambda: 0.0)
        start = min(next_position(delivery, s) for s in ("arm", "x", "y"))
        self.assertEqual(start, 120)
        run_sync(second, self.read(start))
        self.assertEqual(outputs(second.procedure, rec), outputs(full.procedure, full_rec))
        self.assertEqual(delivery.duplicates, 0)

    def test_reading_too_much_is_harmless(self):
        full, full_rec = ab_run()
        run_sync(full, self.read(0))
        first, rec = ab_run()
        run_sync(first, (o for o in self.read(0) if o.seq < 150))
        delivery = Delivery()
        second = Run.restore(ABTest(covariate=True, start=20), first.checkpoint(),
                             topology=KeyJoin(("arm", "y", "x")), delivery=delivery, consumers=[rec],
                             clock=lambda: 0.0)
        run_sync(second, self.read(100))
        self.assertEqual(outputs(second.procedure, rec), outputs(full.procedure, full_rec))
        self.assertEqual(delivery.duplicates, 150)  # 50 units, three inputs each

    def test_missing_values_are_declared_absent(self):
        self.db.execute("update events set y = null where unit % 10 = 3")
        delivery = Delivery()
        rec = Recorder()
        run = Run(NormalMixtureSPRT(1.0, 1.0, input="y", stop_on_reject=False), delivery=delivery, consumers=[rec])
        run_sync(run, from_sql(self.db, "select unit, y from events order by unit", inputs="y", seq="unit",
                               delivery=delivery))
        self.assertEqual(rec.outputs[-1].n, 270)
        self.assertEqual(delivery.state()["next"], {"y": 300})
        self.assertEqual(delivery.state()["ahead"], {})

    def test_errors(self):
        with self.assertRaises(ValueError):
            list(from_sql(self.db, self.query, (0,), inputs="nope", seq="unit"))
        with self.assertRaises(ValueError):
            list(from_sql(self.db, self.query, (0,), inputs="y", batch=0))


class KafkaPython:
    """A kafka-python ConsumerRecord: plain attributes, timestamp in ms."""

    def __init__(self, topic, partition, offset, value, key=None, timestamp=0):
        self.topic, self.partition, self.offset, self.value = topic, partition, offset, value
        self.key, self.timestamp = key, timestamp


class ConfluentMessage:
    """A confluent-kafka Message: accessor methods, (type, ms) timestamps."""

    def __init__(self, topic, partition, offset, value, key=None, timestamp=0):
        self._t = (topic, partition, offset, value, key, (1, timestamp))

    def topic(self):
        return self._t[0]

    def partition(self):
        return self._t[1]

    def offset(self):
        return self._t[2]

    def value(self):
        return self._t[3]

    def key(self):
        return self._t[4]

    def timestamp(self):
        return self._t[5]

    def error(self):
        return None


def messages(kind, offsets, partition=0, topic="ab"):
    rnd = random.Random(partition)
    out = []
    for i in offsets:
        arm = rnd.randrange(2)
        record = {"arm": arm, "y": rnd.gauss(0.2 * arm, 1), "x": rnd.gauss(0, 1)}
        out.append(kind(topic, partition, i, json.dumps(record).encode(), key=f"p{partition}-{i}".encode(),
                        timestamp=1_700_000_000_000 + i))
    return out


class Kafka(unittest.TestCase):
    def test_both_client_styles_give_the_same_observations(self):
        a = list(from_kafka(messages(KafkaPython, range(5)), ["arm", "y"]))
        b = list(from_kafka(messages(ConfluentMessage, range(5)), ["arm", "y"]))
        self.assertEqual([(o.input, o.value, o.source, o.seq, o.key, o.time) for o in a],
                         [(o.input, o.value, o.source, o.seq, o.key, o.time) for o in b])
        self.assertEqual(a[0].source, "ab/0/arm")
        self.assertEqual(a[0].key, "p0-0")
        self.assertEqual(a[0].time, 1_700_000_000.0)

    def test_redelivery_after_restore_is_deduplicated(self):
        stream = messages(KafkaPython, range(0, 400, 3), 0) + messages(KafkaPython, range(0, 400, 2), 1)
        stream.sort(key=lambda m: (m.offset, m.partition))

        full, full_rec = ab_run()
        run_sync(full, from_kafka(stream, ["arm", "x", "y"], delivery=full.delivery))

        first, rec = ab_run()
        run_sync(first, from_kafka(stream[:150], ["arm", "x", "y"], delivery=first.delivery))
        checkpoint = first.checkpoint()
        delivery = Delivery()
        second = Run.restore(ABTest(covariate=True, start=20), checkpoint, topology=KeyJoin(("arm", "y", "x")),
                             delivery=delivery, consumers=[rec], clock=lambda: 0.0)
        seek = kafka_offsets(delivery, "ab")
        self.assertEqual(set(seek), {0, 1})
        # a consumer seeks each partition to its offset; the broker may hand back a little more
        resumed = [m for m in stream if m.offset >= seek[m.partition] - 4]
        run_sync(second, from_kafka(resumed, ["arm", "x", "y"], delivery=delivery))
        self.assertEqual(outputs(second.procedure, rec), outputs(full.procedure, full_rec))
        self.assertGreater(delivery.duplicates, 0)

    def test_offset_gaps_do_not_grow_the_checkpoint(self):
        offsets = sorted(random.Random(5).sample(range(100_000), 500))
        delivery = Delivery()
        rec = Recorder()
        run = Run(NormalMixtureSPRT(1.0, 1.0, input="y", stop_on_reject=False), delivery=delivery, consumers=[rec])
        run_sync(run, from_kafka(messages(KafkaPython, offsets), "y", delivery=delivery))
        self.assertEqual(rec.outputs[-1].n, 500)
        self.assertEqual(delivery.state()["ahead"], {})
        self.assertEqual(delivery.state()["next"], {"ab/0/y": offsets[-1] + 1})
        without = Delivery()
        run_sync(Run(NormalMixtureSPRT(1.0, 1.0, input="y", stop_on_reject=False), delivery=without),
                 from_kafka(messages(KafkaPython, offsets), "y"))
        self.assertEqual(len(without.state()["ahead"]["ab/0/y"]), 500)  # what the declarations save

    def test_consumer_errors_and_bad_records(self):
        class Broken(ConfluentMessage):
            def error(self):
                return "broker down"

        with self.assertRaises(RuntimeError):
            list(from_kafka([Broken("t", 0, 0, b"{}")], "y"))
        with self.assertRaises(ValueError):
            list(from_kafka([KafkaPython("t", 0, 0, b"[1, 2]")], "y"))


class SkipRange(unittest.TestCase):
    def test_gap_at_the_front_moves_the_position(self):
        d = Delivery()
        for seq in (0, 1):
            d.push(_obs(seq))
        d.skip_range("s", 2, 10)
        self.assertEqual(d.positions(), {"s": 10})

    def test_gap_ahead_waits_for_the_numbers_before_it(self):
        d = Delivery()
        d.push(_obs(0))
        d.skip_range("s", 3, 10)  # 1 and 2 not delivered yet
        self.assertEqual(d.positions(), {"s": 1})
        d.push(_obs(10))
        d.push(_obs(2))
        self.assertEqual(d.positions(), {"s": 1})
        d.push(_obs(1))
        self.assertEqual(d.positions(), {"s": 11})
        self.assertEqual(d.state()["ahead"], {})

    def test_only_the_arrival_policy(self):
        with self.assertRaises(OrderingError):
            Delivery("sequence").skip_range("s", 0, 3)


def _obs(seq):
    from sequential_analysis import Observation

    return Observation("x", 0.0, "s", seq)


if __name__ == "__main__":
    unittest.main()
