"""The MCP sessions the engine plays and the conformance recordings replay against the real SDK.

Each scenario is a client script: the protocol mode, the client's identity and capabilities,
what the simulated user and model answer when the server asks, and a list of steps. ``raw``
steps send one hand-written JSON-RPC message (to show errors no well-behaved client makes).
"""
from __future__ import annotations

from typing import Any

CLIENT = {"name": "fixture-client", "version": "1.0.0"}
ACCEPT = {"action": "accept", "content": {"confirm": True}}
DECLINE = {"action": "decline"}
SUMMARY = {"role": "assistant", "content": {"type": "text", "text": "A tiny demo project."},
           "model": "scripted-model", "stopReason": "endTurn"}

_TOUR = [
    {"op": "list_tools"},
    {"op": "call_tool", "name": "add", "arguments": {"a": 2, "b": 3}},
    {"op": "list_resources"},
    {"op": "read_resource", "uri": "docs://readme"},
    {"op": "list_prompts"},
    {"op": "get_prompt", "name": "review", "arguments": {"code": "x = 1"}},
]
_THREE_WAYS = [
    {"op": "list_tools"},
    {"op": "call_tool", "name": "read_doc", "arguments": {"uri": "docs://readme"}},
    {"op": "list_resources"},
    {"op": "read_resource", "uri": "docs://readme"},
    {"op": "list_prompts"},
    {"op": "get_prompt", "name": "review", "arguments": {"code": "# Demo\nA tiny project."}},
]
_PROGRESS = [{"op": "call_tool", "name": "count", "arguments": {"n": 3}, "progress": True}]
_ERRORS = [
    {"op": "call_tool", "name": "fail", "arguments": {"reason": "disk full"}},
    {"op": "call_tool", "name": "nope", "arguments": {}},
    {"op": "read_resource", "uri": "docs://missing"},
]
_ELICIT = [{"op": "call_tool", "name": "delete_file", "arguments": {"path": "build/"}}]
_SAMPLE = [{"op": "call_tool", "name": "summarise", "arguments": {"text": "A tiny project with one test."}}]


def _meta(version: str) -> dict[str, Any]:
    return {"io.modelcontextprotocol/protocolVersion": version, "io.modelcontextprotocol/clientCapabilities": {}}


def _s(name: str, mode: str, steps: list, *, elicitation: bool = False, sampling: bool = False,
       answers: dict | None = None, title: str = "", server: dict | None = None) -> dict[str, Any]:
    sc = {"name": name, "title": title, "mode": mode,
          "client": {"info": CLIENT, "elicitation": elicitation, "sampling": sampling},
          "answers": answers or {}, "steps": steps}
    if server is not None:
        sc["server"] = server
    return sc


SCENARIOS: list[dict[str, Any]] = [
    _s("legacy_tour", "legacy", [{"op": "ping"}] + _TOUR, title="2025-11-25: handshake, then list and use each primitive"),
    _s("modern_tour", "2026-07-28", _TOUR, title="2026-07-28: no handshake; every request carries _meta"),
    _s("auto_tour", "auto", _TOUR[:2], title="auto: server/discover first, then requests"),
    _s("legacy_three_ways", "legacy", _THREE_WAYS, title="The README three ways: a tool the model calls, a resource the app attaches, a prompt the user picks"),
    _s("modern_three_ways", "2026-07-28", _THREE_WAYS, title="The same three ways in 2026-07-28"),
    _s("legacy_progress", "legacy", _PROGRESS, title="2025-11-25: progress notifications"),
    _s("modern_progress", "2026-07-28", _PROGRESS, title="2026-07-28: progress notifications"),
    _s("legacy_errors", "legacy", _ERRORS, title="2025-11-25: a tool error, an unknown tool, an unknown resource"),
    _s("modern_errors", "2026-07-28", _ERRORS, title="2026-07-28: the same errors"),
    _s("legacy_elicit_accept", "legacy", _ELICIT, elicitation=True, answers={"elicit": [ACCEPT]},
       title="2025-11-25: the server asks the user (elicitation/create), who accepts"),
    _s("legacy_elicit_decline", "legacy", _ELICIT, elicitation=True, answers={"elicit": [DECLINE]},
       title="2025-11-25: the user declines"),
    _s("modern_elicit_accept", "2026-07-28", _ELICIT, elicitation=True, answers={"elicit": [ACCEPT]},
       title="2026-07-28: input_required, then a retry carrying the answer"),
    _s("modern_elicit_decline", "2026-07-28", _ELICIT, elicitation=True, answers={"elicit": [DECLINE]},
       title="2026-07-28: the user declines"),
    _s("legacy_sampling", "legacy", _SAMPLE, sampling=True, answers={"sample": [SUMMARY]},
       title="2025-11-25: the server asks the client's model (sampling/createMessage)"),
    _s("modern_sampling", "2026-07-28", _SAMPLE, sampling=True, answers={"sample": [SUMMARY]},
       title="2026-07-28: sampling as an input request"),
    _s("raw_errors", "raw", [
        {"op": "raw", "line": "{not json", "reply": False},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": _meta("2099-01-01")}}},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 2, "method": "no/such/method", "params": {"_meta": _meta("2026-07-28")}}},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {"_meta": _meta("2026-07-28")}}},
], title="Malformed JSON (no reply), an unsupported version, an unknown method, no _meta, then a good request"),
    _s("raw_handshake", "raw", [
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 2, "method": "initialize",
                              "params": {"protocolVersion": "1999-01-01", "capabilities": {}, "clientInfo": CLIENT}}},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "method": "notifications/initialized"}, "reply": False},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 3, "method": "server/discover", "params": {"_meta": _meta("2026-07-28")}}},
    ], title="A request before the handshake, an unknown version (the server offers its own), then a 2026-07-28 request on a handshake connection"),
    _s("raw_old_version", "raw", [
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                              "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": CLIENT}}},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "method": "notifications/initialized"}, "reply": False},
        {"op": "raw", "msg": {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}},
    ], title="A client asking for the first revision, 2024-11-05: the server agrees"),
]


# Engine-only behaviour: the SDK fixture server does not paginate, and these paths depend on
# timing a recording cannot pin down. They are tested against the spec's rules, not the SDK.
ENGINE_SCENARIOS: list[dict[str, Any]] = [
    _s("legacy_paged", "legacy", [{"op": "list_tools"}], server={"page_size": 2},
       title="Pagination: tools/list in pages of two, following nextCursor"),
    _s("modern_paged", "2026-07-28", [{"op": "list_tools"}], server={"page_size": 4},
       title="Pagination in 2026-07-28"),
    _s("legacy_cancel", "legacy", [{"op": "call_tool", "name": "count", "arguments": {"n": 5}, "progress": True,
                                    "cancel_after": 2}],
       title="Cancellation: the client sends notifications/cancelled after two progress updates"),
    _s("legacy_list_changed", "legacy", [{"op": "list_tools"}, {"op": "add_tool"}, {"op": "call_tool", "name": "deploy",
                                                                                      "arguments": {"target": "staging"}}],
       server={"list_changed": True}, title="The server gains a tool and says so; the client lists again"),
]


def protocol_scenario(name: str) -> dict[str, Any]:
    for s in SCENARIOS + ENGINE_SCENARIOS:
        if s["name"] == name:
            return s
    raise KeyError(name)
