"""The model interface and its three back ends.

Every back end answers ``complete(request) -> text``. The request carries the rendered
prompt (exactly the bytes a real model would see), the loop style, and what happened to
the previous turn's tool calls.

- ``ScriptedModel``: a deterministic policy, a list of steps (call these tools / write this
  malformed text / give this final answer) with rules for what to do when a call fails or
  is denied. It writes its steps in whichever loop style the harness uses. All the
  teaching scenarios use it.
- ``ReplayModel``: plays back a recorded trace. Before answering it checks that the prompt
  it was given hashes to the recorded prompt's hash, so a replay proves the harness
  rebuilt the same prompt, byte for byte.
- ``LocalModel`` (Python only): a small open-weights model behind llama.cpp's
  ``llama-server``, used offline to record those traces.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from .jsonfmt import dumps


def fnv1a32(text: str) -> int:
    """FNV-1a, 32-bit, over the UTF-8 bytes: the prompt fingerprint replays check."""
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def render_step(step: dict[str, Any], style: str) -> str:
    """A scripted step as the model's output text, in the given loop style."""
    if "raw" in step:
        raw = step["raw"]
        return raw[style] if isinstance(raw, dict) else raw
    if "final" in step:
        if style == "native":
            return step["final"]
        return "Thought: I now know the final answer\nFinal Answer: " + step["final"]
    thought = step.get("thought", "")
    calls = step["calls"]
    if style == "native":
        parts = [thought] if thought else []
        for c in calls:
            parts.append("<tool_call>\n" + dumps({"name": c["name"], "arguments": c["args"]}) + "\n</tool_call>")
        return "\n".join(parts)
    c = calls[0]
    return "Thought: " + thought + "\nAction: " + c["name"] + "\nAction Input: " + dumps(c["args"])


class ScriptedModel:
    """A deterministic policy over a list of steps.

    After a step's calls come back, the model moves to the next step, unless a call
    failed: then the step's ``on_error`` ("repeat", the default, or "skip") applies, or its
    ``on_denied`` ("skip", the default, or "repeat") if a call was denied or blocked.
    After malformed output or a loop-detection nudge it moves on.
    """

    name = "scripted"

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self.last: int | None = None

    def complete(self, req: dict[str, Any]) -> str:
        nxt = 0
        if self.last is not None:
            step = self.script[self.last]
            nxt = self.last + 1
            prev = req["last"]
            if prev["kind"] == "results":
                failed = [r for r in prev["results"] if not r["ok"]]
                if failed:
                    refused = any(r["kind"] in ("denied", "blocked") for r in failed)
                    action = step.get("on_denied", "skip") if refused else step.get("on_error", "repeat")
                    if action == "repeat":
                        nxt = self.last
        self.last = nxt
        if nxt >= len(self.script):
            return render_step({"final": "I could not finish the task."}, req["style"])
        return render_step(self.script[nxt], req["style"])


class ReplayMismatch(Exception):
    pass


class ReplayModel:
    """Plays back a recorded trace's completions, checking each prompt's fingerprint."""

    name = "replay"

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        self.i = 0

    def complete(self, req: dict[str, Any]) -> str:
        if self.i >= len(self.calls):
            raise ReplayMismatch(f"the trace has only {len(self.calls)} calls")
        rec = self.calls[self.i]
        got = fnv1a32(req["prompt"])
        if got != rec["prompt_fnv1a32"] or req["input_tokens"] != rec["prompt_tokens"]:
            raise ReplayMismatch(
                f"call {self.i + 1}: prompt fingerprint {got} / {req['input_tokens']} tokens, "
                f"recorded {rec['prompt_fnv1a32']} / {rec['prompt_tokens']}"
            )
        self.i += 1
        return rec["text"]


class LocalModel:
    """llama.cpp's llama-server ``/completion`` endpoint, greedy, prompt cache on.

    Each call's raw response fields are kept in ``records`` for the trace's provenance.
    """

    name = "local"

    def __init__(self, url: str = "http://127.0.0.1:8080", seed: int = 0, n_predict: int = 512) -> None:
        self.url = url.rstrip("/")
        self.seed = seed
        self.n_predict = n_predict
        self.records: list[dict[str, Any]] = []

    def complete(self, req: dict[str, Any]) -> str:
        stop = ["<|im_end|>", "<|endoftext|>"]
        if req["style"] == "react":
            stop.append("Observation:")
        body = {
            "prompt": req["prompt"],
            "n_predict": self.n_predict,
            "temperature": 0,
            "top_k": 1,
            "seed": self.seed,
            "stop": stop,
            "cache_prompt": True,
            "return_tokens": True,
        }
        r = urllib.request.Request(
            self.url + "/completion",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=3600) as f:
            out = json.loads(f.read().decode("utf-8"))
        self.records.append(
            {
                "content": out["content"],
                "tokens": out.get("tokens", []),
                "tokens_evaluated": out.get("tokens_evaluated"),
                "tokens_cached": out.get("tokens_cached"),
                "tokens_predicted": out.get("tokens_predicted"),
                "stop_type": out.get("stop_type"),
                "stopping_word": out.get("stopping_word"),
                "timings": out.get("timings", {}),
            }
        )
        return out["content"]


def summarise(dropped: list[dict[str, Any]], facts: list[dict[str, Any]]) -> str:
    """The scripted summariser used by summarising compaction: it keeps the facts marked
    important that the dropped messages contain, and nothing else."""
    lines = ["Summary of the earlier work (the harness replaced older messages with this):"]
    for f in facts:
        if f["important"] and any(f["text"] in m["content"] for m in dropped):
            lines.append("- " + f["text"])
    if len(lines) == 1:
        lines.append("- nothing important was found yet")
    return "\n".join(lines)
