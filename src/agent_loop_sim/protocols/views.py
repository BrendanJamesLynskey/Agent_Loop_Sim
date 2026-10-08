"""Animation frames for the Protocols site, computed from the protocol state machines.

Each function returns plain data (lists of frames or rows); the site draws frame ``k`` and the
tests compare it with the Python reference, so every picture is the model's, not a drawing.
"""
from __future__ import annotations

from typing import Any

from ..jsonfmt import compact
from ..tokenizer import Tokenizer
from .mcp import HANDSHAKE_VERSIONS, LATEST_HANDSHAKE, MODERN_VERSIONS, kind_of


def integration_frames(n: int, m: int) -> dict[str, Any]:
    """The hero picture: N agents and M tools. Point to point, every pair needs its own adapter
    (N·M); with a shared protocol each agent implements a client once and each tool a server
    once (N+M). Frames add one agent's adapters at a time, then switch to the protocol."""
    frames: list[dict[str, Any]] = []
    edges: list[list[int]] = []
    frames.append({"phase": "apart", "p2p_edges": [], "agents_wired": 0, "tools_wired": 0, "built": 0})
    for i in range(n):
        for j in range(m):
            edges.append([i, j])
        frames.append({"phase": "p2p", "p2p_edges": [list(e) for e in edges], "agents_wired": i + 1, "tools_wired": 0,
                       "built": len(edges)})
    for i in range(n):
        frames.append({"phase": "protocol", "p2p_edges": [], "agents_wired": i + 1, "tools_wired": 0, "built": i + 1})
    for j in range(m):
        frames.append({"phase": "protocol", "p2p_edges": [], "agents_wired": n, "tools_wired": j + 1,
                       "built": n + j + 1})
    return {"n": n, "m": m, "p2p": n * m, "protocol": n + m, "frames": frames}


def _short(v: Any, limit: int = 60) -> str:
    s = compact(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def message_label(msg: dict[str, Any] | None, method: str) -> str:
    """A one-line description of a JSON-RPC message for the sequence chart."""
    if msg is None:
        return "a line that is not JSON"
    k = kind_of(msg)
    p = msg.get("params") or {}
    if k == "request":
        mt = msg["method"]
        if mt == "initialize":
            return "initialize (asks for " + str(p.get("protocolVersion")) + ")"
        if mt == "tools/call":
            extra = " + answers" if "inputResponses" in p else ""
            return "tools/call " + str(p.get("name")) + " " + _short(p.get("arguments", {}), 40) + extra
        if mt == "resources/read":
            return "resources/read " + str(p.get("uri"))
        if mt == "prompts/get":
            return "prompts/get " + str(p.get("name"))
        if mt == "elicitation/create":
            return "elicitation/create: " + str(p.get("message"))
        if mt == "sampling/createMessage":
            return "sampling/createMessage (maxTokens " + str(p.get("maxTokens")) + ")"
        return mt
    if k == "notification":
        mt = msg["method"]
        if mt == "notifications/progress":
            return "progress " + str(p.get("progress")) + "/" + str(p.get("total"))
        if mt == "notifications/cancelled":
            return "cancelled (" + str(p.get("reason")) + ")"
        return mt
    if k == "error":
        e = msg["error"]
        return "error " + str(e["code"]) + ": " + _short(e["message"], 50)
    r = msg["result"]
    if r.get("resultType") == "input_required":
        reqs = r.get("inputRequests") or {}
        return "input_required: " + ", ".join(v["method"] for v in reqs.values())
    if method == "initialize":
        return "agreed " + str(r.get("protocolVersion"))
    if method == "server/discover":
        return "speaks " + ", ".join(r.get("supportedVersions") or [])
    if method in ("tools/list", "resources/list", "prompts/list"):
        key = method.split("/")[0]
        nxt = " (more: " + r["nextCursor"] + ")" if "nextCursor" in r else ""
        return str(len(r.get(key) or [])) + " " + key + nxt
    if method == "tools/call":
        c = r.get("content") or [{}]
        t = c[0].get("text", "") if c else ""
        return ("isError: " if r.get("isError") else "") + _short(t, 40)
    if method == "resources/read":
        return str(len(r.get("contents") or [])) + " content block"
    if method == "prompts/get":
        return str(len(r.get("messages") or [])) + " message"
    if method == "elicitation/create":
        return str(r.get("action"))
    if method == "sampling/createMessage":
        c = r.get("content") or {}
        return "\"" + _short(c.get("text", ""), 40) + "\""
    if method == "ping":
        return "pong"
    return "result"


def sequence_frames(played: dict[str, Any]) -> list[dict[str, Any]]:
    """A message sequence chart. Each wire message becomes an arrow between ``client`` and
    ``server``; where the host must ask the user (elicitation) or a model (sampling), virtual
    arrows to ``user`` or ``model`` are inserted, marked ``virtual`` (they happen inside the host,
    not on the wire). Frame k shows arrows 0..k."""
    out: list[dict[str, Any]] = []
    wire = played["wire"]
    log = played["log"]
    for i, w in enumerate(wire):
        lg = log[i]
        msg = w.get("msg")
        frm = "client" if w["dir"] == "c2s" else "server"
        to = "server" if frm == "client" else "client"
        # An answer to the server (handshake era) or a retry with answers (2026-07-28) is
        # preceded by the host asking the user or the model.
        if msg is not None and w["dir"] == "c2s":
            gate = None
            if kind_of(msg) == "result" and lg["method"] in ("elicitation/create", "sampling/createMessage"):
                gate = {"method": lg["method"], "answer": msg["result"]}
            elif kind_of(msg) == "request" and "inputResponses" in (msg.get("params") or {}):
                ans = msg["params"]["inputResponses"]
                key = list(ans.keys())[0]
                gate = {"method": "elicitation/create" if key == "confirm" else "sampling/createMessage",
                        "answer": ans[key]}
            if gate is not None:
                who = "user" if gate["method"] == "elicitation/create" else "model"
                ask = "show the form, wait for the user" if who == "user" else "ask the model (the user may review)"
                reply = (str(gate["answer"].get("action")) if who == "user"
                         else "\"" + _short((gate["answer"].get("content") or {}).get("text", ""), 40) + "\"")
                out.append({"from": "client", "to": who, "label": ask, "virtual": True, "kind": "gate", "wire": None,
                            "dir": "host", "bytes": 0, "client": "asking the " + who, "server": lg["server"]})
                out.append({"from": who, "to": "client", "label": reply, "virtual": True, "kind": "gate", "wire": None,
                            "dir": "host", "bytes": 0, "client": "has the answer", "server": lg["server"]})
        out.append({"from": frm, "to": to, "label": message_label(msg, lg["method"]), "virtual": False,
                    "kind": lg["kind"], "wire": i, "dir": w["dir"], "bytes": lg["bytes"], "client": lg["client"],
                    "server": lg["server"]})
    for k, f in enumerate(out):
        f["step"] = k
    return out


FEATURES: list[list[str]] = [
    ["tools", "server"], ["resources", "server"], ["prompts", "server"],
    ["elicitation", "client"], ["sampling", "client"],
]


def negotiation(played: dict[str, Any]) -> dict[str, Any]:
    """What the two sides agreed: the protocol version (two sets of versions intersected) and
    which features each side declared, from the session's actual messages."""
    wire = played["wire"]
    client_caps: dict[str, Any] = {}
    server_caps: dict[str, Any] = {}
    era = "handshake"
    asked = None
    agreed = None
    server_versions = list(HANDSHAKE_VERSIONS)
    for w in wire:
        msg = w.get("msg")
        if msg is None:
            continue
        p = msg.get("params") or {}
        meta = p.get("_meta") or {}
        if msg.get("method") == "initialize":
            client_caps = p.get("capabilities") or {}
            asked = p.get("protocolVersion")
        elif "io.modelcontextprotocol/clientCapabilities" in meta and w["dir"] == "c2s":
            era = "modern"
            client_caps = meta["io.modelcontextprotocol/clientCapabilities"]
            asked = meta.get("io.modelcontextprotocol/protocolVersion")
        if "result" in msg:
            r = msg["result"]
            if "protocolVersion" in r and "serverInfo" in r:
                server_caps = r.get("capabilities") or {}
                agreed = r["protocolVersion"]
            if "supportedVersions" in r:
                server_caps = r.get("capabilities") or {}
                server_versions = list(r["supportedVersions"])
    client_versions = list(HANDSHAKE_VERSIONS) + list(MODERN_VERSIONS)
    discovered = era == "handshake" or len(server_caps) > 0
    if era == "modern":
        server_versions = list(MODERN_VERSIONS)
        agreed = asked
    both = [v for v in client_versions if v in server_versions]
    rows = []
    for name, side in FEATURES:
        c = name in client_caps
        s = name in server_caps
        usable: bool | None = s if side == "server" else c
        if side == "server" and not discovered:
            usable = None  # 2026-07-28 without server/discover: unknown until a request is answered
        rows.append({"feature": name, "provided_by": side, "client": c, "server": s, "usable": usable})
    return {"era": era, "asked": asked, "agreed": agreed if agreed is not None else LATEST_HANDSHAKE,
            "client_versions": client_versions, "server_versions": server_versions, "both": both,
            "client_caps": sorted(client_caps.keys()), "server_caps": sorted(server_caps.keys()), "discovered": discovered,
            "features": rows}


def journey(played: dict[str, Any], call_index: int = 0) -> list[dict[str, Any]]:
    """One tool call end to end, for chapter 1: the model asks, the host routes it to the right
    MCP client, the client sends JSON-RPC, the server runs the tool and answers, and the result
    goes back into the model's context. The JSON-RPC legs are the session's real messages."""
    wire = played["wire"]
    calls = [i for i, w in enumerate(wire)
             if w.get("msg") is not None and w["dir"] == "c2s" and w["msg"].get("method") == "tools/call"]
    ci = calls[call_index]
    req = wire[ci]["msg"]
    resp = None
    for w in wire[ci + 1:]:
        m = w.get("msg") or {}
        if w["dir"] == "s2c" and m.get("id") == req["id"] and "method" not in m:
            resp = m
            break
    name = req["params"]["name"]
    args = req["params"]["arguments"]
    text = ((resp or {}).get("result") or {}).get("content", [{}])[0].get("text", "")
    use = {"type": "tool_use", "name": name, "input": args}
    return [
        {"at": "model", "to": "host", "label": "the model asks for " + name, "payload": compact(use), "wire": False},
        {"at": "host", "to": "client", "label": "the host finds which server offers " + name, "payload": name,
         "wire": False},
        {"at": "client", "to": "server", "label": "JSON-RPC request", "payload": compact(req), "wire": True},
        {"at": "server", "to": "server", "label": "the server runs " + name, "payload": compact(args), "wire": False},
        {"at": "server", "to": "client", "label": "JSON-RPC result", "payload": compact(resp), "wire": True},
        {"at": "client", "to": "host", "label": "the client returns the content", "payload": text, "wire": False},
        {"at": "host", "to": "model", "label": "the result joins the model's context", "payload": text, "wire": False},
    ]


def three_ways(played: dict[str, Any], tok: Tokenizer) -> list[dict[str, Any]]:
    """Chapter 3: the README reaches the model three ways in one session. For each: who decides
    (the model, the application, the user), the messages it took, and how many tokens of text
    land in the model's context (Qwen2.5 tokenizer)."""
    wire = played["wire"]
    log = played["log"]
    rows: list[dict[str, Any]] = []
    spec = [["tool", "tools/call", "model", "tools/list"], ["resource", "resources/read", "application", "resources/list"],
            ["prompt", "prompts/get", "user", "prompts/list"]]
    for prim, method, who, lister in spec:
        seqs = []
        text = ""
        for i, w in enumerate(wire):
            lg = log[i]
            if lg["method"] in (method, lister):
                seqs.append(i)
                m = w.get("msg") or {}
                if lg["method"] == method and "result" in m:
                    r = m["result"]
                    if method == "tools/call":
                        text = r["content"][0]["text"]
                    elif method == "resources/read":
                        text = r["contents"][0]["text"]
                    else:
                        text = r["messages"][0]["content"]["text"]
        rows.append({"primitive": prim, "method": method, "controlled_by": who, "messages": seqs,
                     "text": text, "tokens": len(tok.encode(text)),
                     "bytes": sum(log[i]["bytes"] for i in seqs)})
    return rows


# --- added in engine 1.3.0: agent to agent, and step-by-step flows (OAuth, attacks) -----------

def _a2a_label(w: dict[str, Any]) -> str:
    if "http" in w:
        h = w["http"]
        if "method" in h:
            return "GET /.well-known/agent-card.json"
        b = h["body"]
        return ("agent card: " + str(b["name"]) + ", " + str(len(b["skills"])) + " skills"
                + (", streams" if b["capabilities"].get("streaming") else ""))
    msg = w["msg"]
    if "method" in msg:
        p = msg.get("params") or {}
        m = p.get("message")
        if isinstance(m, dict):
            text = " ".join(x.get("text", "") for x in m.get("parts", []))
            return msg["method"] + " \"" + _short(text, 30).strip("\"") + "\"" + (" (same task)" if "taskId" in m else "")
        return msg["method"]
    if "error" in msg:
        e = msg["error"]
        return str(e["code"]) + " " + _short(e["message"], 48).strip("\"")
    r = msg["result"]
    if "statusUpdate" in r:
        return "statusUpdate " + r["statusUpdate"]["status"]["state"].replace("TASK_STATE_", "")
    if "artifactUpdate" in r:
        a = r["artifactUpdate"]
        return "artifactUpdate" + (" (append" + (", last" if a.get("lastChunk") else "") + ")" if a.get("append") else
                                   " (first chunk)")
    if "message" in r:
        return "Message \"" + _short(r["message"]["parts"][0].get("text", ""), 28).strip("\"") + "\""
    t = r["task"] if "task" in r else r
    s = "Task " + t["status"]["state"].replace("TASK_STATE_", "")
    if t.get("artifacts"):
        s += " + artifact"
    return s


def a2a_frames(played: dict[str, Any]) -> list[dict[str, Any]]:
    """A message sequence chart for an A2A session: the orchestrating agent (``client``), the
    remote agent (``server``) and, for in-task authorisation, the ``user`` signing in outside
    A2A. SSE events are marked ``sse``. Each frame carries the task's state after it."""
    out: list[dict[str, Any]] = []
    for i, w in enumerate(played["wire"]):
        lg = played["log"][i]
        if w["dir"] == "host":
            out.append({"from": "client", "to": "user", "label": "ask the user to sign in", "virtual": True,
                        "kind": "gate", "wire": None, "dir": "host", "bytes": 0, "sse": False, "t": lg["t"],
                        "client": "asking the user", "server": lg["server"], "task": lg["task"]})
            out.append({"from": "user", "to": "client", "label": "signed in (outside A2A)", "virtual": True,
                        "kind": "gate", "wire": None, "dir": "host", "bytes": 0, "sse": False, "t": lg["t"],
                        "client": lg["client"], "server": lg["server"], "task": lg["task"]})
            continue
        c2s = w["dir"] == "c2s"
        kind = lg["kind"]
        if kind == "http":
            kind = "request" if c2s else "result"
        out.append({"from": "client" if c2s else "server", "to": "server" if c2s else "client",
                    "label": _a2a_label(w), "virtual": False, "kind": kind, "wire": i, "dir": w["dir"],
                    "bytes": lg["bytes"], "sse": bool(w.get("sse")), "t": lg["t"], "client": lg["client"],
                    "server": lg["server"], "task": lg["task"]})
    for k, f in enumerate(out):
        f["step"] = k
    return out


def a2a_summary(played: dict[str, Any]) -> dict[str, Any]:
    """Counts and times for the prose: messages, bytes, when the first piece of the result
    reached the orchestrator, when the session ended, and the task states in order."""
    msgs = 0
    nbytes = 0
    first = None
    states: list[str] = []
    for i, w in enumerate(played["wire"]):
        lg = played["log"][i]
        if w["dir"] == "host":
            continue
        msgs += 1
        nbytes += lg["bytes"]
        if w["dir"] == "s2c" and "msg" in w and "result" in w["msg"] and first is None:
            r = w["msg"]["result"]
            t = r.get("task") if isinstance(r, dict) else None
            if "artifactUpdate" in r or (isinstance(t, dict) and t.get("artifacts")):
                first = lg["t"]
        if lg["task"] is not None and (len(states) == 0 or states[len(states) - 1] != lg["task"]):
            states.append(lg["task"])
    last = played["log"][len(played["log"]) - 1]["t"]
    return {"messages": msgs, "bytes": nbytes, "first_result_ms": first, "end_ms": last, "states": states}


def flow_frames(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Sequence-chart frames for a step-by-step flow (``oauth.run_oauth``,
    ``security.run_security``): HTTP requests and responses between actors, and the checks,
    user actions, model steps and attack steps that happen at one actor or off the wire."""
    out: list[dict[str, Any]] = []
    for s in run["steps"]:
        k = s["kind"]
        if k == "http":
            st = s["status"]
            kind = "request" if st is None else ("error" if st >= 400 else "result")
            virtual = False
        elif k == "check":
            kind = "check" if s["ok"] else "fail"
            virtual = True
        elif k == "attack":
            kind = "attack"
            virtual = True
        else:
            kind = "gate"
            virtual = True
        out.append({"from": s["from"], "to": s["to"], "label": s["label"], "kind": kind, "virtual": virtual,
                    "seq": s["seq"], "ok": s["ok"], "step": len(out)})
    return out
