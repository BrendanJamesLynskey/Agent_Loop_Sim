"""Animation states derived from a trace: what each chapter's animation draws, frame by frame.

The sites draw these frames and nothing else, so a frame is a pure function of the trace
(and the trace a pure function of the scenario and policy). The TS port has the same
functions, and the fixtures check them frame by frame.
"""
from __future__ import annotations

from typing import Any

from .accounting import PRICES

KIND_ORDER = ["system", "tools", "task", "summary", "assistant", "tool_result", "observation", "error", "nudge", "prompt"]


def agg(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Token counts by kind, in a fixed order (zero kinds left out)."""
    sums = {k: 0 for k in KIND_ORDER}
    for c in context:
        if c["kind"] in sums:
            sums[c["kind"]] += c["tokens"]
    return [{"kind": k, "tokens": sums[k]} for k in KIND_ORDER if sums[k] > 0]


def _add(parts: list[dict[str, Any]], kind: str, n: int) -> list[dict[str, Any]]:
    sums = {p["kind"]: p["tokens"] for p in parts}
    sums[kind] = sums.get(kind, 0) + n
    return [{"kind": k, "tokens": sums[k]} for k in KIND_ORDER if sums.get(k, 0) > 0]


def _drop_prompt(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in parts if p["kind"] != "prompt"]


def _total(parts: list[dict[str, Any]]) -> int:
    return sum(p["tokens"] for p in parts)


def loop_frames(events: list[dict[str, Any]], agent: str = "main") -> list[dict[str, Any]]:
    """Chapter 1: model -> tool call -> result -> model, with the context's make-up.

    Phases: ``call`` (the prompt goes to the model), ``output`` (its answer has been
    generated and joins the context), ``tool`` (a call starts), ``result`` (its result joins
    the context), ``compact``, ``error`` and ``done``.
    """
    frames: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    cost = 0.0
    result_kind = "observation" if _is_react(events) else "tool_result"
    for e in events:
        if e["agent"] != agent:
            continue
        ty = e["type"]
        f: dict[str, Any] | None = None
        if ty == "model_call" and e["purpose"] == "act":
            parts = agg(e["context"])
            f = {"phase": "call", "turn": e["turn"], "t": e["t"], "input": e["input_tokens"], "cached": e["cached_tokens"]}
            frames.append(dict(f, parts=parts, total=_total(parts), cost=cost))
            cost += e["cost"]
            parts = _add(_drop_prompt(parts), "assistant", e["message_tokens"])
            f = {"phase": "output", "turn": e["turn"], "t": e["t"] + e["dur"], "output": e["output_tokens"], "text": e["text"]}
        elif ty == "tool_call":
            f = {"phase": "tool", "turn": e["turn"], "t": e["t"], "name": e["name"], "subject": e["subject"]}
        elif ty == "tool_result":
            parts = _add(parts, result_kind, e["tokens"])
            f = {"phase": "result", "t": e["t"] + e["dur"], "name": e["name"], "ok": e["ok"], "tokens": e["tokens"]}
        elif ty == "compaction":
            parts = _drop_prompt(agg(e["context"]))
            f = {"phase": "compact", "turn": e["turn"], "t": e["t"], "before": e["before"], "after": e["after"]}
        elif ty == "error":
            if e["message_tokens"] > 0:
                parts = _add(parts, "error" if e["kind"] == "malformed" else "nudge", e["message_tokens"])
            f = {"phase": "error", "turn": e["turn"], "t": e["t"], "kind": e["kind"], "detail": e["detail"]}
        elif ty == "run_end" and agent == "main":
            f = {"phase": "done", "t": e["t"], "status": e["status"]}
        if f is not None:
            frames.append(dict(f, parts=parts, total=_total(parts), cost=cost))
    return frames


def _is_react(events: list[dict[str, Any]]) -> bool:
    return len(events) > 0 and events[0]["type"] == "run_start" and events[0]["style"] == "react"


def toolcall_frames(events: list[dict[str, Any]], agent: str = "main") -> list[dict[str, Any]]:
    """Chapter 2: what the model wrote, what the harness parsed from it, what came back."""
    frames = []
    for e in events:
        if e["agent"] != agent:
            continue
        ty = e["type"]
        if ty == "model_call" and e["purpose"] == "act":
            frames.append({"phase": "emit", "turn": e["turn"], "text": e["text"], "tokens": e["output_tokens"]})
        elif ty == "tool_call":
            frames.append({"phase": "parse", "turn": e["turn"], "call": e["call"], "name": e["name"], "args": e["args"]})
        elif ty == "error" and e["kind"] == "malformed":
            frames.append({"phase": "malformed", "turn": e["turn"], "detail": e["detail"], "tokens": e["message_tokens"]})
        elif ty == "tool_result":
            frames.append({"phase": "result", "call": e["call"], "name": e["name"], "ok": e["ok"], "text": e["text"], "tokens": e["tokens"]})
        elif ty == "run_end":
            frames.append({"phase": "final", "status": e["status"], "answer": e["answer"]})
    return frames


def budget_frames(events: list[dict[str, Any]], agent: str = "main") -> list[dict[str, Any]]:
    """Chapter 3: the context window as a budget, one frame per model call and compaction."""
    start = events[0]
    ctx = start["policy"]["context"]
    reserve = start["policy"]["max_output"]
    frames = []
    for e in events:
        if e["agent"] != agent:
            continue
        if e["type"] == "model_call" and e["purpose"] == "act":
            parts = agg(e["context"])
            frames.append({"phase": "call", "turn": e["turn"], "parts": parts, "total": _total(parts),
                           "window": ctx["window"], "trigger": ctx["trigger"] * ctx["window"] - reserve,
                           "reserve": reserve, "lost": []})
        elif e["type"] == "compaction":
            parts = agg(e["context"])
            frames.append({"phase": "compact", "turn": e["turn"], "parts": parts, "total": _total(parts),
                           "window": ctx["window"], "trigger": ctx["trigger"] * ctx["window"] - reserve,
                           "reserve": reserve, "lost": e["facts_lost"], "before": e["before"], "after": e["after"],
                           "strategy": e["strategy"], "removed": len(e["removed"]), "clipped": len(e["clipped"])})
        elif e["type"] == "error" and e["kind"] == "context_overflow":
            frames.append({"phase": "overflow", "turn": e["turn"], "parts": [], "total": 0, "window": ctx["window"],
                           "trigger": ctx["trigger"] * ctx["window"] - reserve, "reserve": reserve, "lost": []})
    return frames


def cache_frames(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chapter 4: each call's cached prefix, and the running cost with and without the cache."""
    price = PRICES[events[0]["policy"]["cache"]["price"]]
    frames = []
    cum = 0.0
    cum_plain = 0.0
    for e in events:
        if e["type"] != "model_call":
            continue
        plain = (e["input_tokens"] * price["input"] + e["output_tokens"] * price["output"]) / 1e6
        cum += e["cost"]
        cum_plain += plain
        frames.append({
            "turn": e["turn"], "agent": e["agent"], "input": e["input_tokens"], "cached": e["cached_tokens"],
            "uncached": e["input_tokens"] - e["cached_tokens"], "output": e["output_tokens"],
            "hit": e["cached_tokens"] / e["input_tokens"], "cost": e["cost"], "plain": plain,
            "cum": cum, "cum_plain": cum_plain, "ttft": e["ttft"],
        })
    return frames


def permission_frames(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chapter 5: each call's permission decision, the human's answer and wait, the outcome."""
    frames: list[dict[str, Any]] = []
    by_call: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["type"] == "tool_call":
            f = {"call": e["call"], "name": e["name"], "subject": e["subject"], "t": e["t"]}
            by_call[e["call"]] = f
            frames.append(f)
        elif e["type"] == "permission_check":
            f = by_call[e["call"]]
            f["decision"] = e["decision"]
            f["reason"] = e["reason"]
            f["answer"] = e["answer"]
            f["wait"] = e["wait"]
        elif e["type"] == "hook":
            by_call[e["call"]]["hook"] = e["result"]
        elif e["type"] == "tool_result":
            f = by_call[e["call"]]
            f["outcome"] = e["kind"]
            f["done"] = e["t"] + e["dur"]
    return frames


def timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spans for a timeline: model calls, human waits, tool runs and back-offs."""
    spans = []
    for e in events:
        ty = e["type"]
        if ty == "model_call":
            spans.append({"lane": "model", "agent": e["agent"], "start": e["t"], "end": e["t"] + e["dur"],
                          "label": "summarise" if e["purpose"] == "summary" else f"turn {e['turn']}"})
        elif ty == "permission_check" and e["wait"] > 0:
            spans.append({"lane": "human", "agent": e["agent"], "start": e["t"], "end": e["t"] + e["wait"],
                          "label": e["answer"] or ""})
        elif ty == "tool_result":
            spans.append({"lane": "tool", "agent": e["agent"], "start": e["t"], "end": e["t"] + e["dur"],
                          "label": e["name"] + ("" if e["ok"] else f" ({e['kind']})")})
        elif ty == "retry":
            spans.append({"lane": "retry", "agent": e["agent"], "start": e["t"], "end": e["t"] + e["backoff"],
                          "label": f"back-off {e['attempt']}"})
    return spans
