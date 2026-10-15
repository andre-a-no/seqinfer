# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Adapters to data frames, SQL databases and Kafka.

Each adapter turns rows or messages into Observations; none imports the
library it talks to, so pandas, polars, a database driver or a Kafka
client are needed only by the caller who has them.

A row may carry several logical inputs (an A/B log: arm, outcome,
covariate).  `inputs` maps column (or field) names to input names; each
row yields one Observation per mapped column, all with the row's sequence
number, key and time, so KeyJoin or PositionalPair puts them back
together.  Missing values (None, NaN) are not emitted unless
missing="keep"; pass the run's Delivery as `delivery` and the sequence
number of a skipped value is declared absent (Delivery.skip_range), so
that the arrival policy does not wait for it.

Resuming after a checkpoint: a restored run knows, per source, the next
sequence number it expects (``run.delivery.positions()``).  With
`seq` naming an increasing column, ask the database for the rows from
there on (see `next_position`) -- nothing is read twice; reading too
much is harmless, as the delivery drops what it has already seen.  For
Kafka, `kafka_offsets` gives the offset to seek each partition to.

Cost (measured as in PERFORMANCE.md): one Observation per mapped column
and row, 1.4 µs per value from pandas, 1.9 µs from sqlite3, 2.3 µs from
Kafka messages including JSON decoding -- comparable to the cheapest
procedures' step, small next to the others.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from ._compat import zip_strict
from .core import Observation
from .delivery import Delivery

MISSING = ("skip", "keep")


def _plain(value: Any) -> Any:
    """numpy, pandas and datetime scalars as plain Python values."""
    if isinstance(value, (str, bytes, bool, int, float)) or value is None:
        return value
    if isinstance(value, _dt.datetime):
        return value.timestamp()
    if hasattr(value, "timestamp") and callable(value.timestamp):  # pandas.Timestamp
        return float(value.timestamp())
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        return _plain(value.item())
    if hasattr(value, "tolist") and callable(value.tolist):  # numpy array: a vector input
        return value.tolist()
    return value


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _mapping(inputs: Mapping[str, str] | Sequence[str] | str) -> dict[str, str]:
    if isinstance(inputs, str):
        return {inputs: inputs}
    if isinstance(inputs, Mapping):
        out = dict(inputs)
    else:
        out = {c: c for c in inputs}
    if not out:
        raise ValueError("no columns to read")
    return out


def _rows_to_observations(rows: Iterable[Mapping[str, Any]], columns: dict[str, str], *, seq: str | None,
                          key: str | None, time: str | None, source: str, start: int, missing: str,
                          where: str, delivery: Delivery | None) -> Iterator[Observation]:
    if missing not in MISSING:
        raise ValueError(f"missing must be one of {MISSING}")
    for i, row in enumerate(rows):
        try:
            n = _plain(row[seq]) if seq is not None else start + i
            k = _plain(row[key]) if key is not None else None
            t = _plain(row[time]) if time is not None else None
            values = {c: _plain(row[c]) for c in columns}
        except KeyError as error:
            raise ValueError(f"{where}: no column {error.args[0]!r}") from None
        if seq is not None and type(n) is not int:
            raise ValueError(f"{where}, row {i}: sequence number {n!r} is not an integer")
        for column, input_name in columns.items():
            v = values[column]
            name = f"{source}/{input_name}" if source else input_name
            if missing == "skip" and _is_missing(v):
                if delivery is not None and seq is not None:
                    delivery.skip_range(name, n, n + 1)
                continue
            yield Observation(input_name, v, name, n, k, t)


def from_dataframe(frame: Any, inputs: Mapping[str, str] | Sequence[str] | str, *, seq: str | None = None,
                   key: str | None = None, time: str | None = None, source: str = "", start: int = 0,
                   missing: str = "skip", delivery: Delivery | None = None) -> Iterator[Observation]:
    """Observations from a pandas or polars DataFrame, row by row.

    `inputs` maps column names to input names (a list or a single name maps
    a column to the input of the same name).  Without `seq` rows are
    numbered from `start` in frame order.  Timestamps become seconds since
    the epoch; numpy scalars become Python numbers.
    """
    columns = _mapping(inputs)
    if hasattr(frame, "iter_rows"):  # polars
        rows: Iterable[Mapping[str, Any]] = frame.iter_rows(named=True)
    elif hasattr(frame, "itertuples") and hasattr(frame, "columns"):  # pandas
        names = [str(c) for c in frame.columns]
        rows = (dict(zip_strict(names, r)) for r in frame.itertuples(index=False, name=None))
    else:
        raise TypeError(f"expected a pandas or polars DataFrame, got {type(frame).__qualname__}")
    return _rows_to_observations(rows, columns, seq=seq, key=key, time=time, source=source, start=start,
                                 missing=missing, where="data frame", delivery=delivery)


def from_sql(connection: Any, query: str, parameters: Sequence[Any] | Mapping[str, Any] = (), *,
             inputs: Mapping[str, str] | Sequence[str] | str, seq: str | None = None, key: str | None = None,
             time: str | None = None, source: str = "", start: int = 0, missing: str = "skip",
             batch: int = 1000, delivery: Delivery | None = None) -> Iterator[Observation]:
    """Observations from a query on a DB-API 2.0 connection (sqlite3, psycopg, mysqlclient, ...).

    The query is the caller's, with the driver's own placeholders for
    `parameters`; columns are matched by the names in the result.  Rows are
    fetched `batch` at a time.  Order the query by `seq` when it is given.
    """
    columns = _mapping(inputs)
    if type(batch) is not int or batch < 1:
        raise ValueError("batch must be a positive int")
    cursor = connection.cursor()
    try:
        cursor.execute(query, parameters)
        names = [d[0] for d in cursor.description]

        def rows() -> Iterator[dict[str, Any]]:
            while True:
                chunk = cursor.fetchmany(batch)
                if not chunk:
                    return
                for r in chunk:
                    yield dict(zip_strict(names, r))

        yield from _rows_to_observations(rows(), columns, seq=seq, key=key, time=time, source=source,
                                         start=start, missing=missing, where="query", delivery=delivery)
    finally:
        cursor.close()


def next_position(delivery: Delivery, source: str) -> int:
    """The first sequence number of `source` a restored run still needs (its delivery's start if unseen)."""
    return delivery.positions().get(source, delivery.start)


def _field(message: Any, name: str) -> Any:
    v = getattr(message, name)
    return v() if callable(v) else v  # confluent-kafka has methods, kafka-python attributes


def from_kafka(messages: Iterable[Any], inputs: Mapping[str, str] | Sequence[str] | str, *,
               decode: Callable[[bytes], Any] = json.loads, key: str | None = None, time: str | None = None,
               delivery: Delivery | None = None, missing: str = "skip") -> Iterator[Observation]:
    """Observations from Kafka messages (a kafka-python or confluent-kafka consumer, or any iterable of messages).

    Each message value is decoded (JSON by default) to a record whose
    fields `inputs` maps to input names.  The source of an Observation is
    "topic/partition/input" and its sequence number the message offset, so
    the delivery deduplicates redelivered messages and a checkpoint records
    the offset reached in every partition.  `key` names a record field to
    join on (by default the message key, decoded as UTF-8); `time` a record
    field, by default the message timestamp in seconds.

    Offsets may have gaps (compacted or transactional topics).  Pass the
    run's Delivery (arrival policy) as `delivery`, and the adapter declares
    each gap it crosses (Delivery.skip_range), so the checkpoint does not
    grow with the gaps; a skipped missing value is declared the same way.
    The offsets below the first message read from a partition are declared
    absent too: start the consumer at the beginning of the topic, or, for a
    restored run, at `kafka_offsets` (or earlier).
    """
    columns = _mapping(inputs)
    if missing not in MISSING:
        raise ValueError(f"missing must be one of {MISSING}")
    last: dict[str, int] = {}
    for message in messages:
        error = getattr(message, "error", None)
        if callable(error) and error() is not None:
            raise RuntimeError(f"Kafka consumer error: {error()}")
        topic, partition, offset = _field(message, "topic"), _field(message, "partition"), _field(message, "offset")
        raw = _field(message, "value")
        record = decode(raw) if raw is not None else {}
        if not isinstance(record, Mapping):
            raise ValueError(f"{topic}/{partition} offset {offset}: the decoded value is not a record")
        if key is not None:
            k = _plain(record.get(key))
        else:
            raw_key = _field(message, "key")
            k = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
        if time is not None:
            t = _plain(record.get(time))
        else:
            stamp = _field(message, "timestamp")
            stamp = stamp[1] if isinstance(stamp, tuple) else stamp  # confluent: (type, ms)
            t = stamp / 1000.0 if isinstance(stamp, (int, float)) and stamp >= 0 else None
        for field, input_name in columns.items():
            source = f"{topic}/{partition}/{input_name}"
            if delivery is not None:
                # offsets below the first one read in a partition (the consumer's start) and between
                # consecutive ones do not exist for this run
                first = last[source] + 1 if source in last else next_position(delivery, source)
                delivery.skip_range(source, first, offset)
            last[source] = max(offset, last.get(source, offset))
            v = _plain(record.get(field))
            if missing == "skip" and _is_missing(v):
                if delivery is not None:
                    delivery.skip_range(source, offset, offset + 1)
                continue
            yield Observation(input_name, v, source, offset, k, t)


def kafka_offsets(delivery: Delivery, topic: str) -> dict[int, int]:
    """Per partition of `topic`, the offset to seek a consumer to when resuming a restored run.

    The smallest next position over the inputs read from that partition:
    nothing still needed is skipped, and what is read twice is dropped by
    the delivery.
    """
    out: dict[int, int] = {}
    for source, position in delivery.positions().items():
        parts = source.split("/")
        if len(parts) >= 3 and "/".join(parts[:-2]) == topic and parts[-2].isdigit():
            p = int(parts[-2])
            out[p] = min(out.get(p, position), position)
    return out
