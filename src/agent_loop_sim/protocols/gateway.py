"""An MCP gateway: one server that the host connects to, multiplexing several downstream MCP
servers (2026-07-28 messages). It lists every downstream server's tools, merges them into one
list, and routes each ``tools/call`` to the server that owns the tool.

Three policies:

- ``flat``: expose every tool under its own name; when two servers offer the same name the first
  one registered wins and the others are shadowed (unreachable, and silently so);
- ``prefix``: expose ``<alias>.<tool>`` (the alias is the gateway's own name for the server; the
  specification's tools page says aggregators SHOULD disambiguate, for example by prefixing with
  a server identifier, and SHOULD NOT rely on ``serverInfo.name``);
- ``filtered``: prefix, and expose only an allow-list chosen for the task.

The cost the model pays is the tool list in its prompt: tokens are counted with the vendored
Qwen2.5 tokenizer over the tool definitions as a chat template writes them (``jsonfmt.dumps``).
Harnesses' chapter 3 shows the same cost from the harness's side.
"""
from __future__ import annotations

import copy
from typing import Any

from ..jsonfmt import compact, dumps
from ..tokenizer import Tokenizer
from .mcp import LATEST, META_CLIENT_CAPS, META_CLIENT_INFO, META_SERVER_INFO, META_VERSION

POLICIES = ["flat", "prefix", "filtered"]


def _tool(name: str, description: str, args: list[list[str]]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for a, desc in args:
        props[a] = {"type": "string", "description": desc}
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": props, "required": [a for a, _ in args]}}


# The downstream servers, in the order the gateway registers them (alias, server name, tools).
SERVERS: list[dict[str, Any]] = [
    {"alias": "files", "info": {"name": "files-server", "version": "1.4.0"}, "tools": [
        _tool("read_file", "Read a text file from the project.", [["path", "Path relative to the project root"]]),
        _tool("write_file", "Write a text file in the project, replacing it.",
              [["path", "Path relative to the project root"], ["text", "The new contents"]]),
        _tool("search", "Search the project's files for a regular expression.",
              [["pattern", "A regular expression"]]),
    ]},
    {"alias": "github", "info": {"name": "github-server", "version": "0.9.2"}, "tools": [
        _tool("search", "Search issues and pull requests in the repository.", [["query", "GitHub search syntax"]]),
        _tool("create_issue", "Open an issue in the repository.",
              [["title", "Issue title"], ["body", "Issue body in Markdown"]]),
        _tool("read_file", "Read a file from the default branch on GitHub.", [["path", "Path in the repository"]]),
    ]},
    {"alias": "web", "info": {"name": "web-server", "version": "2.0.1"}, "tools": [
        _tool("search", "Search the web and return the top results.", [["query", "Search terms"]]),
        _tool("fetch", "Fetch a web page and return its text.", [["url", "An https URL"]]),
    ]},
    {"alias": "calendar", "info": {"name": "calendar-server", "version": "1.1.0"}, "tools": [
        _tool("list_events", "List the user's events for a day.", [["date", "An ISO date"]]),
        _tool("create_event", "Create an event in the user's calendar.",
              [["title", "Event title"], ["start", "ISO date and time"]]),
    ]},
]

# What the model wants to do in the walk-through: search the web.
CALL = {"alias": "web", "tool": "search", "arguments": {"query": "MCP gateway tool name collisions"}}
# The filtered policy's allow-list for that task.
ALLOW = ["web.search", "web.fetch", "files.read_file"]
GATEWAY_INFO = {"name": "gateway", "version": "1.0.0"}
CLIENT_INFO = {"name": "example-host", "version": "1.0.0"}


def _meta() -> dict[str, Any]:
    return {META_VERSION: LATEST, META_CLIENT_INFO: dict(CLIENT_INFO), META_CLIENT_CAPS: {}}


def _list_result(id_: int, tools: list[dict[str, Any]], info: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": {"tools": tools, "cacheScope": "private", "ttlMs": 0,
                                                   "resultType": "complete", "_meta": {META_SERVER_INFO: dict(info)}}}


def merge(policy: str) -> dict[str, Any]:
    """The gateway's merged tool list under a policy: each downstream tool with the name it is
    exposed under (or why it is not exposed), and the names more than one server offers."""
    if policy not in POLICIES:
        raise ValueError("unknown policy " + policy)
    rows: list[dict[str, Any]] = []
    owner: dict[str, str] = {}
    offered: dict[str, list[str]] = {}
    for s in SERVERS:
        for t in s["tools"]:
            offered.setdefault(t["name"], []).append(s["alias"])
            exposed = t["name"] if policy == "flat" else s["alias"] + "." + t["name"]
            status = "exposed"
            if policy == "flat" and exposed in owner:
                status = "shadowed"
            elif policy == "filtered" and exposed not in ALLOW:
                status = "filtered"
            if status == "exposed":
                owner[exposed] = s["alias"]
            rows.append({"alias": s["alias"], "tool": t["name"], "exposed": exposed, "status": status,
                         "shadowed_by": owner[exposed] if status == "shadowed" else None})
    collisions = [[n, list(a)] for n, a in offered.items() if len(a) > 1]
    return {"policy": policy, "rows": rows, "collisions": collisions, "owner": owner}


def gateway_run(policy: str, tok: Tokenizer) -> dict[str, Any]:
    """The walk-through: the host lists tools through the gateway (which lists every downstream
    server), the merged list's token cost, then the model's call to web search routed by the
    gateway. Returns the merge, the token counts and the sequence-chart frames."""
    m = merge(policy)
    frames: list[dict[str, Any]] = []
    wire: list[dict[str, Any]] = []

    def add(frm: str, to: str, label: str, kind: str, msg: dict[str, Any] | None, note: str = "",
            virtual: bool = False) -> None:
        nbytes = 0 if msg is None else len(compact(msg).encode("utf-8"))
        if msg is not None:
            wire.append({"from": frm, "to": to, "msg": msg})
        frames.append({"from": frm, "to": to, "label": label, "kind": kind, "virtual": virtual,
                       "wire": None if msg is None else len(wire) - 1, "bytes": nbytes, "note": note})

    add("client", "gateway", "tools/list", "request",
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": _meta()}})
    per_server = []
    nid = 1
    for s in SERVERS:
        nid += 1
        add("gateway", s["alias"], "tools/list", "request",
            {"jsonrpc": "2.0", "id": nid, "method": "tools/list", "params": {"_meta": _meta()}})
    for i, s in enumerate(SERVERS):
        tokens = tok.count(dumps(s["tools"]))
        per_server.append({"alias": s["alias"], "tools": len(s["tools"]), "tokens": tokens})
        add(s["alias"], "gateway", str(len(s["tools"])) + " tools (" + str(tokens) + " tokens)", "result",
            _list_result(i + 2, copy.deepcopy(s["tools"]), s["info"]))
    merged: list[dict[str, Any]] = []
    for r in m["rows"]:
        if r["status"] != "exposed":
            continue
        for s in SERVERS:
            if s["alias"] == r["alias"]:
                for t in s["tools"]:
                    if t["name"] == r["tool"]:
                        d = copy.deepcopy(t)
                        d["name"] = r["exposed"]
                        merged.append(d)
    shadowed = [r for r in m["rows"] if r["status"] == "shadowed"]
    filtered = [r for r in m["rows"] if r["status"] == "filtered"]
    note = ""
    if shadowed:
        note = str(len(shadowed)) + " tools shadowed by a same-named tool"
    elif filtered:
        note = str(len(filtered)) + " tools filtered out"
    add("gateway", "gateway", "merge: " + str(len(merged)) + " tools" + (" (" + note + ")" if note else ""), "check",
        None, note, True)
    merged_tokens = tok.count(dumps(merged))
    add("gateway", "client", str(len(merged)) + " tools (" + str(merged_tokens) + " tokens)", "result",
        _list_result(1, merged, GATEWAY_INFO))
    # The model asks for web search.
    want = CALL["alias"] + "." + CALL["tool"]
    name = CALL["tool"] if policy == "flat" else want
    add("model", "client", "call " + name, "gate", None, "the model picks a tool by name", True)
    add("client", "gateway", "tools/call " + name, "request",
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": name, "arguments": copy.deepcopy(CALL["arguments"]), "_meta": _meta()}})
    target = m["owner"].get(name)
    routed_right = target == CALL["alias"]
    add("gateway", "gateway", "route " + name + " → " + str(target) + (" (as intended)" if routed_right
                                                                      else " (the model meant " + CALL["alias"] + ")"),
        "check" if routed_right else "error", None, "", True)
    add("gateway", str(target), "tools/call " + CALL["tool"], "request",
        {"jsonrpc": "2.0", "id": nid + 1, "method": "tools/call",
         "params": {"name": CALL["tool"], "arguments": copy.deepcopy(CALL["arguments"]), "_meta": _meta()}})
    text = ("3 web results for \"" + CALL["arguments"]["query"] + "\"" if routed_right
            else "No matches for /" + CALL["arguments"]["query"] + "/ in the project")
    res = {"content": [{"type": "text", "text": text}], "isError": False, "resultType": "complete"}
    add(str(target), "gateway", "result: " + text, "result", {"jsonrpc": "2.0", "id": nid + 1, "result": res})
    add("gateway", "client", "result: " + text, "result", {"jsonrpc": "2.0", "id": 2, "result": copy.deepcopy(res)})
    for k, f in enumerate(frames):
        f["step"] = k
    direct = 0
    for p in per_server:
        direct += p["tokens"]
    return {"policy": policy, "merge": m, "per_server": per_server, "direct_tokens": direct,
            "merged_tokens": merged_tokens, "merged_tools": len(merged), "exposed": [t["name"] for t in merged],
            "routed_to": target, "routed_right": routed_right, "frames": frames, "wire": wire}
