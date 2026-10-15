# Data sources

A run consumes `Observation`s. Each one carries:

* the logical input it belongs to;
* its value;
* the source it came from, with a sequence number;
* optionally a join key and an event time.

`sequential_analysis.adapters` makes them from data frames, SQL
queries and Kafka messages. None of the adapters imports the library it
reads from: pandas, polars, a database driver or a Kafka client is
needed only by the code that already uses it.

A row usually holds several inputs; an A/B log, for example, holds the
arm, the outcome and a covariate. `inputs` maps column names to input
names. Each row yields one observation per input, all with the row's
sequence number, key and time, so that `KeyJoin` or `PositionalPair`
puts them back together.

## SQL

`from_sql` takes any DB-API 2.0 connection (sqlite3, psycopg, mysqlclient,
...). It runs your query with your driver's placeholders and fetches the
result in batches:

```python
import sqlite3
from sequential_analysis import KeyJoin, Run, run_sync
from sequential_analysis.adapters import from_sql, next_position
from sequential_analysis.procedures import TwoProportionTest

db = sqlite3.connect(":memory:")
db.execute("create table events (user integer primary key, arm integer, converted integer)")
db.executemany("insert into events values (?, ?, ?)",
               [(u, u % 2, int((u * 7919) % 100 < 10 + 3 * (u % 2))) for u in range(6000)])

query = "select user, arm, converted from events where user >= ? order by user"

def read(start):
    return from_sql(db, query, (start,), inputs={"arm": "arm", "converted": "y"}, seq="user", key="user")

run = Run(TwoProportionTest(), topology=KeyJoin(("arm", "y")), run_id="conversion")
run_sync(run, (o for o in read(0) if o.seq < 4000))
checkpoint = run.checkpoint()

# later, in another process: restore, and read only what the run has not seen
restored = Run.restore(TwoProportionTest(), checkpoint, topology=KeyJoin(("arm", "y")))
start = min(next_position(restored.delivery, s) for s in ("arm", "y"))
run_sync(restored, read(start))
print("resumed at user", start, "; users counted:", sum(restored.state.trials))
```

Reading too much after a restore is harmless: delivery drops what the
run has already seen, and counts it in `delivery.duplicates`.

## pandas and polars

```py
import pandas as pd
from sequential_analysis.adapters import from_dataframe

frame = pd.read_parquet("experiment.parquet")
observations = from_dataframe(frame, {"arm": "arm", "revenue": "y", "spend_before": "x"},
                              seq="user_id", key="user_id", time="timestamp")
```

* **Conversion.** numpy scalars become Python numbers, and timestamps
  become seconds since the epoch.
* **Missing values.** NaN and None are skipped unless `missing="keep"`.
  Pass the run's delivery as `delivery=run.delivery`, and a skipped
  value's sequence number is declared absent, so it is not waited for.
* **polars.** Frames are read through `iter_rows(named=True)`.

## Kafka

`from_kafka` reads the messages of a kafka-python or confluent-kafka
consumer, or any iterable of messages:

* the value is decoded as JSON by default;
* the source is `topic/partition/input`;
* the sequence number is the offset.

```py
from kafka import KafkaConsumer, TopicPartition
from sequential_analysis.adapters import from_kafka, kafka_offsets

run = Run.restore(procedure, checkpoint, topology=KeyJoin(("arm", "y")))
consumer = KafkaConsumer(bootstrap_servers="...", enable_auto_commit=False)
partitions = [TopicPartition("ab-events", p) for p in range(6)]
consumer.assign(partitions)
for tp, offset in kafka_offsets(run.delivery, "ab-events").items():
    consumer.seek(TopicPartition("ab-events", tp), offset)
run_sync(run, from_kafka(consumer, {"arm": "arm", "converted": "y"}, key="user_id", delivery=run.delivery))
```

* **Resuming.** The checkpoint, not Kafka's committed offsets, is the
  record of what the run has consumed. Seek to `kafka_offsets` (or
  earlier) when resuming.
* **Offset gaps.** Offsets have gaps on compacted and transactional
  topics. With `delivery=run.delivery`, the adapter declares the gaps it
  crosses, so the checkpoint does not grow with them.

## Writing your own

An adapter is any iterable of `Observation`s. Your own adapter should:

* convert values to plain Python numbers, strings or lists;
* number each source's observations in order;
* give observations that belong together a common key.

The [specification](spec.md) lists exactly what the run accepts.
