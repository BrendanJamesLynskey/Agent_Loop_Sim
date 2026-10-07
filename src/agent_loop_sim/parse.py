"""Turning the model's text into tool calls, in the two loop styles.

- **native**: every ``<tool_call>…</tool_call>`` block holds one JSON object
  ``{"name": …, "arguments": {…}}`` (Qwen2.5's format). No block means a final answer.
- **react**: ``Action: <tool>`` then ``Action Input: <JSON>``; ``Final Answer: …`` ends the
  run. Anything after ``Observation:`` is dropped (the model must not write its own).

A call that cannot be parsed, names an unknown tool, or has arguments that do not fit the
tool's schema is *malformed*: the harness tells the model what was wrong and lets it try
again. The error messages are fixed strings (not the JSON library's own wording), so the
Python reference and the TS port produce the same feedback text, token for token.
"""
from __future__ import annotations

import json
from typing import Any

from .tools import TOOL_SPECS, trim, validate_args


def _reject_constant(name: str) -> Any:
    raise ValueError(name)


def load_json(text: str) -> tuple[bool, Any]:
    """(ok, value): strict JSON as JavaScript's JSON.parse reads it (no NaN/Infinity)."""
    try:
        return True, json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return False, None


def _check_call(obj: Any, allowed: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None, 'a tool call must be a JSON object with a "name" string and an "arguments" object'
    name = obj["name"]
    args = obj["arguments"] if "arguments" in obj else {}
    if isinstance(args, str):
        ok, inner = load_json(args)
        if ok:
            args = inner
    if name not in allowed:
        return None, f"unknown tool '{name}'; the tools are: {', '.join(allowed)}"
    why = validate_args(TOOL_SPECS[name], args)
    if why is not None:
        return None, f"invalid arguments for {name}: {why}"
    return {"name": name, "args": args}, None


def parse_native(text: str, allowed: list[str]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = text.find("<tool_call>", pos)
        if start < 0:
            break
        end = text.find("</tool_call>", start)
        if end < 0:
            return {"kind": "malformed", "error": "a <tool_call> block is not closed with </tool_call>"}
        body = trim(text[start + len("<tool_call>"):end])
        ok, obj = load_json(body)
        if not ok:
            return {"kind": "malformed", "error": "the text inside <tool_call></tool_call> is not valid JSON"}
        call, err = _check_call(obj, allowed)
        if err is not None:
            return {"kind": "malformed", "error": err}
        calls.append(call)
        pos = end + len("</tool_call>")
    if not calls:
        return {"kind": "final", "final": trim(text)}
    return {"kind": "calls", "calls": calls}


def parse_react(text: str, allowed: list[str]) -> dict[str, Any]:
    obs = text.find("Observation:")
    if obs >= 0:
        text = text[:obs]
    act = text.find("Action:")
    fin = text.find("Final Answer:")
    if fin >= 0 and (act < 0 or fin < act):
        return {"kind": "final", "final": trim(text[fin + len("Final Answer:"):])}
    if act < 0:
        return {"kind": "malformed", "error": "write either 'Action:' with 'Action Input:', or 'Final Answer:'"}
    line_end = text.find("\n", act)
    name = trim(text[act + len("Action:"): line_end if line_end >= 0 else len(text)])
    inp = text.find("Action Input:", act)
    if inp < 0:
        return {"kind": "malformed", "error": "'Action:' must be followed by 'Action Input:' with a JSON object"}
    raw = trim(text[inp + len("Action Input:"):])
    ok, args = load_json(raw)
    if not ok:
        return {"kind": "malformed", "error": "the Action Input is not a valid JSON object"}
    call, err = _check_call({"name": name, "arguments": args}, allowed)
    if err is not None:
        return {"kind": "malformed", "error": err}
    return {"kind": "calls", "calls": [call]}


def parse(text: str, style: str, allowed: list[str]) -> dict[str, Any]:
    return parse_native(text, allowed) if style == "native" else parse_react(text, allowed)
