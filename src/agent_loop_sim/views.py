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


def agents_frames(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chapter 6: the parent's and every sub-agent's context side by side, over time.

    One frame per model call (``call``, ``output``), tool call and result, compaction, error,
    hand-off (``spawn``, ``return``) and the run's end, for whichever agent it belongs to.
    Each frame carries every agent's context make-up so far (``parts``, by agent, in the
    order the agents first appear), its total, and each agent's running count of input
    tokens sent and model time used: the token saving and the latency it costs.
    """
    frames: list[dict[str, Any]] = []
    parts: dict[str, list[dict[str, Any]]] = {}
    sent: dict[str, int] = {}
    busy: dict[str, float] = {}
    result_kind = "observation" if _is_react(events) else "tool_result"

    def snap(f: dict[str, Any]) -> None:
        f["parts"] = {a: list(p) for a, p in parts.items()}
        f["totals"] = {a: _total(p) for a, p in parts.items()}
        f["sent"] = dict(sent)
        f["busy"] = dict(busy)
        frames.append(f)

    for e in events:
        ty = e["type"]
        a = e["agent"]
        if ty == "run_start":
            parts[a] = []
            sent[a] = 0
            busy[a] = 0.0
            continue
        if ty == "handoff":
            if e["direction"] == "spawn":
                parts[e["child"]] = []
                sent[e["child"]] = 0
                busy[e["child"]] = 0.0
                snap({"phase": "spawn", "agent": a, "child": e["child"], "t": e["t"], "prompt_tokens": e["prompt_tokens"]})
            else:
                snap({"phase": "return", "agent": a, "child": e["child"], "t": e["t"], "status": e["status"],
                      "child_tokens": e["child_tokens"], "summary_tokens": e["summary_tokens"]})
            continue
        if ty == "model_call":
            sent[a] = sent[a] + e["input_tokens"]
            busy[a] = busy[a] + e["dur"]
            if e["purpose"] != "act":
                continue
            parts[a] = agg(e["context"])
            snap({"phase": "call", "agent": a, "turn": e["turn"], "t": e["t"], "input": e["input_tokens"]})
            parts[a] = _add(_drop_prompt(parts[a]), "assistant", e["message_tokens"])
            snap({"phase": "output", "agent": a, "turn": e["turn"], "t": e["t"] + e["dur"], "output": e["output_tokens"]})
        elif ty == "tool_call":
            snap({"phase": "tool", "agent": a, "turn": e["turn"], "t": e["t"], "name": e["name"], "subject": e["subject"]})
        elif ty == "tool_result":
            parts[a] = _add(parts[a], result_kind, e["tokens"])
            snap({"phase": "result", "agent": a, "t": e["t"] + e["dur"], "name": e["name"], "ok": e["ok"], "tokens": e["tokens"]})
        elif ty == "compaction":
            parts[a] = _drop_prompt(agg(e["context"]))
            snap({"phase": "compact", "agent": a, "turn": e["turn"], "t": e["t"], "before": e["before"], "after": e["after"]})
        elif ty == "error":
            if e["message_tokens"] > 0:
                parts[a] = _add(parts[a], "error" if e["kind"] == "malformed" else "nudge", e["message_tokens"])
            snap({"phase": "error", "agent": a, "turn": e["turn"], "t": e["t"], "kind": e["kind"], "detail": e["detail"]})
        elif ty == "run_end":
            snap({"phase": "done", "agent": a, "t": e["t"], "status": e["status"]})
    return frames


def pipeline_frames(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chapter 7: each tool call on its way through the harness's checkpoints.

    Stages, in the order the harness runs them: ``call`` (parsed from the model's output),
    ``permission`` (the rule table's decision, and the human's answer to an ask), ``hook``
    (a pre-tool hook blocks or rewrites it; a post-tool hook appends to the result),
    ``result`` (what came back: run, refused by a rule, blocked by a hook, or stopped at the
    sandbox boundary). Each frame also says what the call's subject is by then (a rewrite
    changes it).
    """
    frames: list[dict[str, Any]] = []
    subject: dict[str, str] = {}
    for e in events:
        ty = e["type"]
        if ty == "tool_call":
            subject[e["call"]] = e["subject"]
            frames.append({"stage": "call", "agent": e["agent"], "call": e["call"], "name": e["name"],
                           "subject": e["subject"], "t": e["t"]})
        elif ty == "permission_check":
            frames.append({"stage": "permission", "agent": e["agent"], "call": e["call"], "subject": subject[e["call"]],
                           "decision": e["decision"], "reason": e["reason"], "answer": e["answer"], "wait": e["wait"],
                           "t": e["t"]})
        elif ty == "hook":
            if e["action"] == "rewrite":
                subject[e["call"]] = e["subject"]
            frames.append({"stage": "hook", "agent": e["agent"], "call": e["call"], "subject": subject[e["call"]],
                           "hook": e["hook"], "phase": e["phase"], "action": e["action"], "result": e["result"],
                           "t": e["t"]})
        elif ty == "tool_result":
            frames.append({"stage": "result", "agent": e["agent"], "call": e["call"], "name": e["name"],
                           "subject": subject[e["call"]], "ok": e["ok"], "kind": e["kind"], "text": e["text"],
                           "t": e["t"] + e["dur"]})
    return frames
