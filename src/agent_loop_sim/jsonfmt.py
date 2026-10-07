"""JSON text that the TS port can reproduce byte for byte.

The prompt a model sees contains JSON (tool schemas, tool-call arguments, tool results), and
token counts depend on every byte of it. ``dumps`` is ``json.dumps(x, ensure_ascii=False)``
(the separators ", " and ": ", as the Hugging Face chat templates' ``tojson`` filter writes),
restricted to values JavaScript represents the same way: strings, bools, None, lists, dicts
with string keys, and integers. Floats are refused: Python writes 1.0 where JavaScript
writes 1, so a float in a prompt would break parity silently.
"""
from __future__ import annotations

import json
from typing import Any


def _check(x: Any) -> None:
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return
    if isinstance(x, int):
        if abs(x) > 2**53:
            raise ValueError(f"integer {x} is not exact in JavaScript")
        return
    if isinstance(x, float):
        raise ValueError(f"float {x!r} in prompt JSON (not portable)")
    if isinstance(x, list):
        for v in x:
            _check(v)
        return
    if isinstance(x, dict):
        for k, v in x.items():
            if not isinstance(k, str):
                raise ValueError(f"non-string key {k!r}")
            _check(v)
        return
    raise ValueError(f"unsupported JSON value {type(x).__name__}")


def dumps(x: Any) -> str:
    """One-line JSON, separators ", " and ": ", keys in insertion order, non-ASCII kept."""
    _check(x)
    return json.dumps(x, ensure_ascii=False)


def compact(x: Any) -> str:
    """One-line JSON with no spaces (the trace files' own format)."""
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
