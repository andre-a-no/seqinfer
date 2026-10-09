# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Canonical JSON (RFC 8785, JSON Canonicalization Scheme).

History digests and checkpoint identifiers hash a canonical encoding, so
that an implementation in any language obtains the same bytes for the same
values.  RFC 8785 fixes the three things ordinary JSON encoders disagree
on: number formatting (that of ECMAScript, the shortest representation
that round-trips), string escaping, and key order (by UTF-16 code units).

Every number is an IEEE 754 binary64 value: 1 and 1.0 encode the same, and
an integer that a binary64 value cannot hold exactly is refused rather
than rounded.
"""
from __future__ import annotations

import json
import math
from typing import Any

_MAX_EXACT_INT = 2**53


def _number(x: float) -> str:
    """ECMAScript Number::toString for a finite binary64 value."""
    if not math.isfinite(x):
        raise ValueError(f"canonical JSON has no representation for {x!r}")
    if x == 0:
        return "0"  # also -0
    sign = "-" if x < 0 else ""
    mantissa, _, exponent = repr(abs(x)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = whole + fraction
    point = len(whole) + (int(exponent) if exponent else 0)
    stripped = digits.lstrip("0")
    point -= len(digits) - len(stripped)
    digits = stripped.rstrip("0")
    k, n = len(digits), point
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        body = digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + ("+" if e >= 0 else "-") + str(abs(e))
    return sign + body


def _encode(obj: Any, out: list[str]) -> None:
    if obj is None:
        out.append("null")
    elif obj is True:
        out.append("true")
    elif obj is False:
        out.append("false")
    elif isinstance(obj, int):
        if abs(obj) > _MAX_EXACT_INT:
            raise ValueError(f"integer {obj} is not exactly representable as a binary64 number")
        out.append(_number(float(obj)))
    elif isinstance(obj, float):
        out.append(_number(obj))
    elif isinstance(obj, str):
        out.append(json.dumps(obj, ensure_ascii=False))
    elif isinstance(obj, (list, tuple)):
        out.append("[")
        for i, item in enumerate(obj):
            if i:
                out.append(",")
            _encode(item, out)
        out.append("]")
    elif isinstance(obj, dict):
        for key in obj:
            if not isinstance(key, str):
                raise TypeError(f"canonical JSON object keys must be strings, got {type(key).__name__}")
        out.append("{")
        for i, key in enumerate(sorted(obj, key=lambda k: k.encode("utf-16-be"))):
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _encode(obj[key], out)
        out.append("}")
    else:
        raise TypeError(f"canonical JSON cannot encode {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """The RFC 8785 encoding of a JSON value."""
    out: list[str] = []
    _encode(obj, out)
    return "".join(out)
