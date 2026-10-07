"""The trace format: JSON Lines, one event per line, version 1.

``schema/trace.schema.json`` is the JSON Schema of one line. Events share ``v`` (the format
version), ``seq`` (their index), ``t`` (simulated milliseconds since the run started),
``agent`` (``main``, or ``sub1``, ``sub2`` … for sub-agents) and ``type``:

``run_start`` · ``model_call`` · ``tool_call`` · ``permission_check`` · ``hook`` · ``retry`` ·
``tool_result`` · ``compaction`` · ``error`` · ``handoff`` · ``checkpoint`` · ``run_end``.

A *recorded* trace (``traces/*.json``) is different: it holds a real model's completions
and the provenance of the recording, and the replay back end turns it back into events.
"""
from __future__ import annotations

import json
from typing import Any

from .jsonfmt import compact

EVENT_TYPES = [
    "run_start", "model_call", "tool_call", "permission_check", "hook", "retry",
    "tool_result", "compaction", "error", "handoff", "checkpoint", "run_end",
]


def dumps_jsonl(events: list[dict[str, Any]]) -> str:
    return "".join(compact(e) + "\n" for e in events)


def loads_jsonl(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.split("\n") if line.strip() != ""]
