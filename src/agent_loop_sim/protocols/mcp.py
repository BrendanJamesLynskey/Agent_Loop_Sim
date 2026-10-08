"""MCP as two state machines exchanging JSON-RPC 2.0 messages, in both protocol eras.

* The **handshake era** (revisions 2024-11-05 to 2025-11-25): the client sends ``initialize``,
  the server answers with the agreed version and its capabilities, the client confirms with
  ``notifications/initialized``; then requests flow. A server that needs the user or a model
  sends its own request (``elicitation/create``, ``sampling/createMessage``) to the client.
* The **stateless era** (revision 2026-07-28): no handshake. Every request carries the
  protocol version, the client's identity and capabilities in ``params._meta``; every result
  carries ``resultType`` and the server's identity. A server that needs input returns
  ``resultType: "input_required"`` with ``inputRequests``, and the client retries the original
  request with ``inputResponses`` (multi round-trip requests, MRTR).

``FIXTURE_SERVER`` models ``conformance/sdk_server.py`` (written with the official MCP Python
SDK); ``play`` runs a scenario's client script against it and returns the wire messages.
``scripts/record_sdk.py`` records the real SDK client and server doing the same scenarios,
and the tests require ``normalise(play(sc)["wire"])`` to equal the recording.

Engine-only extensions, not exercised by the SDK recordings (labelled as such where shown):
pagination (``server.page_size``), cancellation (``cancel_after``), and a tool-list change
notification (``server.list_changed`` with the ``add_tool`` step, handshake era only).
"""
from __future__ import annotations

import copy
from typing import Any

from ..jsonfmt import compact

JSONRPC = "2.0"
HANDSHAKE_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
MODERN_VERSIONS = ["2026-07-28"]
LATEST_HANDSHAKE = "2025-11-25"
LATEST = "2026-07-28"
SPEC_ACCESSED = "2026-10-08"

META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

ERRORS = {
    "parse_error": -32700,
    "invalid_request": -32600,
    "method_not_found": -32601,
    "invalid_params": -32602,
    "internal_error": -32603,
    "header_mismatch": -32020,
    "missing_client_capability": -32021,
    "unsupported_protocol_version": -32022,
}

ERA_MESSAGE = ("this connection serves the handshake protocol era; requests carrying the 2026-07-28 envelope "
               "are not accepted on it")
ENVELOPE_MESSAGE = ("params._meta must be an object carrying the required 'io.modelcontextprotocol/protocolVersion' "
                    "and 'io.modelcontextprotocol/clientCapabilities' envelope keys")

CONFIRM_SCHEMA = {"type": "object", "properties": {"confirm": {"type": "boolean"}}, "required": ["confirm"]}


def _tool(name: str, description: str, args: list[list[str]], out: str) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for a, ty in args:
        props[a] = {"title": a[0].upper() + a[1:], "type": ty}
    return {
        "description": description,
        "inputSchema": {"type": "object", "properties": props, "required": [a for a, _ in args],
                        "title": name + "Arguments"},
        "name": name,
        "outputSchema": {"properties": {"result": {"title": "Result", "type": out}}, "required": ["result"],
                         "title": name + "Output", "type": "object"},
    }


FIXTURE_SERVER: dict[str, Any] = {
    "info": {"name": "fixture-server", "version": "1.0.0"},
    "tools": [
        _tool("add", "Add two integers.", [["a", "integer"], ["b", "integer"]], "integer"),
        _tool("read_doc", "Read a document by URI.", [["uri", "string"]], "string"),
        _tool("count", "Count to n, reporting progress.", [["n", "integer"]], "string"),
        _tool("delete_file", "Delete a file after the user confirms.", [["path", "string"]], "string"),
        _tool("summarise", "Summarise text with the client's model.", [["text", "string"]], "string"),
        _tool("fail", "Always fails (to show a tool error).", [["reason", "string"]], "string"),
    ],
    "resources": [{"description": "The project README.", "mimeType": "text/markdown", "name": "readme",
                   "uri": "docs://readme"}],
    "texts": {"docs://readme": "# Demo\nA tiny project."},
    "prompts": [{"arguments": [{"name": "code", "required": True}], "description": "Review a piece of code.",
                 "name": "review"}],
}
EXTRA_TOOL = _tool("deploy", "Deploy the project.", [["target", "string"]], "string")


def text_content(text: str) -> list[dict[str, Any]]:
    return [{"text": text, "type": "text"}]


def request(id_: Any, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    m: dict[str, Any] = {"jsonrpc": JSONRPC, "id": id_, "method": method}
    if params is not None:
        m["params"] = params
    return m


def notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    m: dict[str, Any] = {"jsonrpc": JSONRPC, "method": method}
    if params is not None:
        m["params"] = params
    return m


def response(id_: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC, "id": id_, "result": result}


def error(id_: Any, code: int, message: str, data: Any = None, has_data: bool = False) -> dict[str, Any]:
    e: dict[str, Any] = {"code": code, "message": message}
    if has_data:
        e["data"] = data
    return {"jsonrpc": JSONRPC, "id": id_, "error": e}


def kind_of(msg: dict[str, Any]) -> str:
    """request · notification · result · error."""
    if "method" in msg:
        return "request" if "id" in msg else "notification"
    return "error" if "error" in msg else "result"


class Server:
    """The server's state machine. ``receive`` takes one message (or a raw line that failed to
    parse) and returns what the server sends back, in order."""

    def __init__(self, spec: dict[str, Any], options: dict[str, Any] | None = None) -> None:
        o = options or {}
        self.spec = copy.deepcopy(spec)
        self.page_size: int | None = o.get("page_size")
        self.list_changed: bool = bool(o.get("list_changed", False))
        self.era: str | None = None  # None until the first valid request: "handshake" | "modern"
        self.version: str | None = None
        self.initialized = False
        self.client_caps: dict[str, Any] = {}
        self.next_id = 1
        self.pending: dict[str, Any] | None = None  # a handshake-era call waiting for the client's answer
        self.cancelled: list[Any] = []
        self.state = "waiting"

    # --- results -------------------------------------------------------------------------
    def _modern(self, result: dict[str, Any], cacheable: bool) -> dict[str, Any]:
        """Add the 2026-07-28 result fields (key order as the SDK writes them is irrelevant to
        equality; this order keeps byte counts stable)."""
        if self.era != "modern":
            return result
        if cacheable:
            result["cacheScope"] = "private"
            result["ttlMs"] = 0
        result["resultType"] = "complete"
        result["_meta"] = {META_SERVER_INFO: dict(self.spec["info"])}
        return result

    def _capabilities(self, discover: bool) -> dict[str, Any]:
        # The SDK advertises listChanged/subscribe as true in server/discover and as the server's
        # own setting (false here) in the initialize result.
        lc = discover or self.list_changed
        return {"prompts": {"listChanged": lc}, "resources": {"listChanged": lc, "subscribe": discover},
                "tools": {"listChanged": lc}}

    def _page(self, items: list[Any], params: dict[str, Any]) -> tuple[list[Any], str | None]:
        if self.page_size is None:
            return items, None
        start = 0
        cur = params.get("cursor")
        if isinstance(cur, str) and cur.startswith("page-"):
            start = int(cur[5:])
        end = start + self.page_size
        return items[start:end], ("page-" + str(end) if end < len(items) else None)

    def _list(self, key: str, params: dict[str, Any]) -> dict[str, Any]:
        items, nxt = self._page(copy.deepcopy(self.spec[key]), params)
        r: dict[str, Any] = {key: items}
        if nxt is not None:
            r["nextCursor"] = nxt
        return self._modern(r, True)

    def _tool_result(self, text: str, structured: bool, is_error: bool = False) -> dict[str, Any]:
        r: dict[str, Any] = {"content": text_content(text), "isError": is_error}
        if structured:
            r["structuredContent"] = {"result": text}
        return self._modern(r, False)

    # --- the state machine -----------------------------------------------------------------
    def receive(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        if "raw" in item:
            # The SDK's stdio server logs a line that is not JSON and sends nothing back
            # (JSON-RPC 2.0 would answer -32700 with a null id).
            self.state = "dropped a malformed line"
            return []
        msg = item["msg"]
        k = kind_of(msg)
        if k == "notification":
            return self._notification(msg)
        if k in ("result", "error"):
            return self._client_answer(msg)
        return self._request(msg)

    def _notification(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        if msg["method"] == "notifications/initialized":
            self.initialized = True
            self.state = "ready (" + str(self.version) + ")"
        elif msg["method"] == "notifications/cancelled":
            self.cancelled.append(msg["params"]["requestId"])
            self.state = "stopped the cancelled request"
        return []

    def _client_answer(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        p = self.pending
        if p is None or p["sid"] != msg["id"]:
            return []
        self.pending = None
        answer = msg.get("result", {})
        req = p["req"]
        name = req["params"]["name"]
        if p["kind"] == "elicit":
            ok = answer.get("action") == "accept" and bool((answer.get("content") or {}).get("confirm"))
            path = req["params"]["arguments"]["path"]
            text = ("deleted " if ok else "kept ") + path
        else:
            c = answer.get("content") or {}
            text = c.get("text", "?") if c.get("type") == "text" else "?"
        self.state = "finished " + name
        return [response(req["id"], self._tool_result(text, True))]

    def _request(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        id_ = msg["id"]
        method = msg["method"]
        params = msg.get("params") or {}
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else None
        if method == "initialize":
            self.era = "handshake"
            asked = params.get("protocolVersion")
            self.version = asked if asked in HANDSHAKE_VERSIONS else LATEST_HANDSHAKE
            self.client_caps = params.get("capabilities") or {}
            self.state = "initialising"
            return [response(id_, {"capabilities": self._capabilities(False), "protocolVersion": self.version,
                                   "serverInfo": dict(self.spec["info"])})]
        if meta is not None and META_VERSION in meta:
            if self.era == "handshake":
                return [error(id_, ERRORS["invalid_request"], ERA_MESSAGE)]
            v = meta[META_VERSION]
            if v not in MODERN_VERSIONS:
                return [error(id_, ERRORS["unsupported_protocol_version"], "Unsupported protocol version",
                              {"supported": list(MODERN_VERSIONS), "requested": v}, True)]
            if META_CLIENT_CAPS not in meta:
                return [error(id_, ERRORS["invalid_params"],
                              "params._meta is missing the required envelope key(s): " + META_CLIENT_CAPS)]
            self.era = "modern"
            self.version = v
            self.client_caps = meta[META_CLIENT_CAPS]
            self.state = "serving " + method + " (stateless)"
        elif self.era == "modern":
            return [error(id_, ERRORS["invalid_params"], ENVELOPE_MESSAGE)]
        elif self.era is None:
            return [error(id_, ERRORS["invalid_params"], "Invalid request parameters", "", True)]
        else:
            self.state = "serving " + method
        return self._dispatch(id_, method, params)

    def _dispatch(self, id_: Any, method: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        modern = self.era == "modern"
        if method == "ping" and not modern:
            return [response(id_, {})]
        if method == "server/discover" and modern:
            r: dict[str, Any] = {"capabilities": self._capabilities(True), "supportedVersions": list(MODERN_VERSIONS)}
            return [response(id_, self._modern(r, True))]
        if method == "tools/list":
            return [response(id_, self._list("tools", params))]
        if method == "resources/list":
            return [response(id_, self._list("resources", params))]
        if method == "prompts/list":
            return [response(id_, self._list("prompts", params))]
        if method == "resources/read":
            uri = params.get("uri")
            if uri not in self.spec["texts"]:
                return [error(id_, ERRORS["invalid_params"], "Unknown resource: " + str(uri), {"uri": uri}, True)]
            r = {"contents": [{"mimeType": self._mime(uri), "text": self.spec["texts"][uri], "uri": uri}]}
            return [response(id_, self._modern(r, True))]
        if method == "prompts/get":
            code = (params.get("arguments") or {}).get("code", "")
            r = {"description": "Review a piece of code.",
                 "messages": [{"content": {"text": "Please review:\n" + code, "type": "text"}, "role": "user"}]}
            return [response(id_, self._modern(r, False))]
        if method == "tools/call":
            return self._call(id_, params)
        return [error(id_, ERRORS["method_not_found"], "Method not found", method, True)]

    def _mime(self, uri: str) -> str:
        for r in self.spec["resources"]:
            if r["uri"] == uri:
                return r["mimeType"]
        return "text/plain"

    def _call(self, id_: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        name = params.get("name")
        args = params.get("arguments") or {}
        modern = self.era == "modern"
        names = [t["name"] for t in self.spec["tools"]]
        if name not in names:
            return [response(id_, self._tool_result("Unknown tool: " + str(name), False, True))]
        if name == "add":
            v = args["a"] + args["b"]
            r: dict[str, Any] = {"content": text_content(str(v)), "isError": False, "structuredContent": {"result": v}}
            return [response(id_, self._modern(r, False))]
        if name == "read_doc":
            return [response(id_, self._tool_result(self.spec["texts"][args["uri"]], True))]
        if name == "deploy":
            return [response(id_, self._tool_result("deployed to " + str(args.get("target")), True))]
        if name == "fail":
            return [response(id_, self._tool_result("Error executing tool fail", False, True))]
        if name == "count":
            out: list[dict[str, Any]] = []
            token = ((params.get("_meta") or {}).get("progressToken"))
            n = args["n"]
            if token is not None:
                for i in range(1, n + 1):
                    out.append(notification("notifications/progress", {"progressToken": token, "progress": i, "total": n}))
            out.append(response(id_, self._tool_result("counted to " + str(n), True)))
            return out
        if name in ("delete_file", "summarise"):
            kind = "elicit" if name == "delete_file" else "sample"
            if kind == "elicit":
                method = "elicitation/create"
                p: dict[str, Any] = {"mode": "form", "message": "Delete " + args["path"] + "?",
                                     "requestedSchema": copy.deepcopy(CONFIRM_SCHEMA)}
                key = "confirm"
            else:
                method = "sampling/createMessage"
                p = {"messages": [{"role": "user", "content": {"type": "text", "text": "Summarise: " + args["text"]}}],
                     "maxTokens": 50}
                key = "summary"
            if modern:
                responses = params.get("inputResponses")
                if not responses:
                    self.state = "needs input: " + method
                    r = {"inputRequests": {key: {"method": method, "params": p}}, "resultType": "input_required",
                         "_meta": {META_SERVER_INFO: dict(self.spec["info"])}}
                    return [response(id_, r)]
                answer = responses[key]
                if kind == "elicit":
                    ok = answer.get("action") == "accept" and bool((answer.get("content") or {}).get("confirm"))
                    text = ("deleted " if ok else "kept ") + args["path"]
                else:
                    c = answer.get("content") or {}
                    text = c.get("text", "?") if c.get("type") == "text" else "?"
                return [response(id_, self._tool_result(text, True))]
            sid = self.next_id
            self.next_id += 1
            self.pending = {"sid": sid, "kind": kind, "req": request(id_, "tools/call", params)}
            self.state = "waiting for the client: " + method
            return [request(sid, method, p)]
        raise ValueError("no behaviour for tool " + str(name))

    def add_tool(self) -> list[dict[str, Any]]:
        """Engine-only: the server gains a tool; with list_changed it tells the client."""
        self.spec["tools"].append(copy.deepcopy(EXTRA_TOOL))
        if self.list_changed and self.era == "handshake":
            return [notification("notifications/tools/list_changed")]
        return []


class Client:
    """The client's side of a scenario: what it sends, and how it answers the server."""

    def __init__(self, sc: dict[str, Any]) -> None:
        self.mode: str = sc["mode"]
        cl = sc["client"]
        self.info = dict(cl["info"])
        caps: dict[str, Any] = {}
        if cl.get("elicitation"):
            caps["elicitation"] = {"form": {}, "url": {}}
        if cl.get("sampling"):
            caps["sampling"] = {}
        self.caps = caps
        self.answers = {k: copy.deepcopy(v) for k, v in sc.get("answers", {}).items()}
        self.next_id = 1
        self.modern = self.mode not in ("legacy", "raw")
        self.version = LATEST if self.modern else LATEST_HANDSHAKE
        self.tools_listed = False
        self.state = "connecting"

    def meta(self) -> dict[str, Any]:
        return {META_VERSION: self.version, META_CLIENT_INFO: dict(self.info), META_CLIENT_CAPS: copy.deepcopy(self.caps)}

    def new_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def params(self, base: dict[str, Any] | None, progress_token: Any = None) -> dict[str, Any] | None:
        """A request's params: in 2026-07-28 always with the _meta envelope (a progress token, if
        any, goes last); in the handshake era only the progress token rides in _meta."""
        p = dict(base) if base is not None else None
        if self.modern:
            p = p if p is not None else {}
            m = self.meta()
            if progress_token is not None:
                m["progressToken"] = progress_token
            p["_meta"] = m
        elif progress_token is not None:
            p = p if p is not None else {}
            p["_meta"] = {"progressToken": progress_token}
        return p

    def answer(self, req: dict[str, Any]) -> dict[str, Any]:
        """The client's reply to a server request (handshake era), or the payload of an input
        response (2026-07-28)."""
        if req["method"] == "elicitation/create":
            return self.answers["elicit"].pop(0)
        if req["method"] == "sampling/createMessage":
            return self.answers["sample"].pop(0)
        raise ValueError("cannot answer " + req["method"])


def play(sc: dict[str, Any]) -> dict[str, Any]:
    """Run a scenario. Returns ``wire`` (exactly what crosses the transport, in order) and
    ``log`` (the same messages annotated for the animations: kind, method, bytes, both states)."""
    server = Server(FIXTURE_SERVER, sc.get("server"))
    client = Client(sc)
    wire: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    responses: dict[Any, dict[str, Any]] = {}
    methods: dict[str, str] = {}  # "dir:id" -> the method of the request, to label responses

    def record(d: str, item: dict[str, Any], note: str) -> None:
        wire.append(item)
        if "raw" in item:
            text = item["raw"]
            kind = "malformed"
            method = ""
            mid: Any = None
        else:
            msg = item["msg"]
            text = compact(msg)
            kind = kind_of(msg)
            mid = msg.get("id")
            if "method" in msg:
                method = msg["method"]
                if "id" in msg:
                    methods[d + ":" + str(mid)] = method
            else:
                other = "s2c" if d == "c2s" else "c2s"
                method = methods.get(other + ":" + str(mid), "")
        log.append({"seq": len(log), "dir": d, "kind": kind, "method": method, "id": mid,
                    "bytes": utf8_len(text), "note": note, "client": client.state, "server": server.state})

    def deliver(item: dict[str, Any], note: str, cancel_after: int | None = None, cancel_id: Any = None) -> None:
        queue = [(item, note)]
        while queue:
            it, nt = queue.pop(0)
            record("c2s", it, nt)
            outs = server.receive(it)
            seen_progress = 0
            for o in outs:
                if cancel_after is not None and seen_progress >= cancel_after:
                    break  # the server stopped: nothing more is sent for the cancelled request
                k = kind_of(o)
                record("s2c", {"dir": "s2c", "msg": o}, "")
                if k == "notification" and o["method"] == "notifications/progress":
                    seen_progress += 1
                    if cancel_after is not None and seen_progress == cancel_after:
                        client.state = "cancelling"
                        c = notification("notifications/cancelled", {"requestId": cancel_id, "reason": "user cancelled"})
                        record("c2s", {"dir": "c2s", "msg": c}, "the user cancels")
                        server.receive({"dir": "c2s", "msg": c})
                elif k == "request":
                    client.state = "answering " + o["method"]
                    a = client.answer(o)
                    queue.append(({"dir": "c2s", "msg": response(o["id"], a)}, "the client answers"))
                elif k in ("result", "error"):
                    responses[o["id"]] = o

    def call(method: str, base: dict[str, Any] | None, note: str, progress: bool = False,
             cancel_after: int | None = None) -> dict[str, Any] | None:
        i = client.new_id()
        p = client.params(base, i if progress else None)
        client.state = "waiting for " + method
        deliver({"dir": "c2s", "msg": request(i, method, p)}, note, cancel_after, i)
        client.state = "ready"
        return responses.get(i)

    if client.mode == "raw":
        for st in sc["steps"]:
            if "line" in st:
                deliver({"dir": "c2s", "raw": st["line"]}, "a line that is not JSON")
            else:
                deliver({"dir": "c2s", "msg": copy.deepcopy(st["msg"])}, "hand-written")
        return {"name": sc["name"], "wire": wire, "log": log}

    if client.mode == "legacy":
        client.state = "initialising"
        call("initialize", {"protocolVersion": LATEST_HANDSHAKE, "capabilities": copy.deepcopy(client.caps),
                            "clientInfo": dict(client.info)}, "the client proposes a version and its capabilities")
        client.state = "ready"
        deliver({"dir": "c2s", "msg": notification("notifications/initialized")}, "the handshake is complete")
    elif client.mode == "auto":
        call("server/discover", None, "the client asks which versions the server speaks")

    def list_all(method: str, key: str) -> None:
        cursor = None
        while True:
            r = call(method, {"cursor": cursor} if cursor is not None else None, "")
            res = (r or {}).get("result") or {}
            cursor = res.get("nextCursor")
            if cursor is None:
                break
        if key == "tools":
            client.tools_listed = True

    for st in sc["steps"]:
        op = st["op"]
        if op == "ping":
            call("ping", None, "")
        elif op == "list_tools":
            list_all("tools/list", "tools")
        elif op == "list_resources":
            list_all("resources/list", "resources")
        elif op == "list_prompts":
            list_all("prompts/list", "prompts")
        elif op == "read_resource":
            call("resources/read", {"uri": st["uri"]}, "")
        elif op == "get_prompt":
            call("prompts/get", {"name": st["name"], "arguments": copy.deepcopy(st["arguments"])}, "")
        elif op == "add_tool":
            for o in server.add_tool():
                record("s2c", {"dir": "s2c", "msg": o}, "the server's tools changed")
                if o["method"] == "notifications/tools/list_changed":
                    list_all("tools/list", "tools")
        elif op == "call_tool":
            base = {"name": st["name"], "arguments": copy.deepcopy(st["arguments"])}
            r = call("tools/call", base, "", bool(st.get("progress")), st.get("cancel_after"))
            rounds = 0
            while r is not None and "result" in r and r["result"].get("resultType") == "input_required" and rounds < 8:
                rounds += 1
                client.state = "gathering input"
                answers = {}
                for key, ir in r["result"]["inputRequests"].items():
                    answers[key] = client.answer(ir)
                retry = {"inputResponses": answers}
                retry.update(base)
                r = call("tools/call", retry, "the client retries with the answers")
            res = (r or {}).get("result") or {}
            # The SDK client validates structured output against the tool's outputSchema,
            # listing the tools first if it has not yet.
            if "structuredContent" in res and not res.get("isError") and not client.tools_listed:
                list_all("tools/list", "tools")
        else:
            raise ValueError("unknown step " + op)
    return {"name": sc["name"], "wire": wire, "log": log}


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def normalise(wire: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renumber request IDs (and progress tokens) in order of appearance, per direction:
    ``c1, c2 …`` for the client's requests, ``s1 …`` for the server's; responses take the tag
    of the request they answer. Everything else is compared exactly (as parsed JSON)."""
    ids: dict[str, str] = {}
    tokens: dict[str, str] = {}
    n = {"c2s": 0, "s2c": 0}
    out: list[dict[str, Any]] = []
    for w in wire:
        if "msg" not in w:
            out.append({"dir": w["dir"], "raw": w["raw"]})
            continue
        m = copy.deepcopy(w["msg"])
        d = w["dir"]
        if "id" in m and m["id"] is not None:
            if "method" in m:
                n[d] += 1
                tag = ("c" if d == "c2s" else "s") + str(n[d])
                ids[d + ":" + compact(m["id"])] = tag
                meta = (m.get("params") or {}).get("_meta")
                if isinstance(meta, dict) and "progressToken" in meta:
                    tokens[compact(meta["progressToken"])] = tag
                    meta["progressToken"] = tag
                m["id"] = tag
            else:
                other = "s2c" if d == "c2s" else "c2s"
                m["id"] = ids.get(other + ":" + compact(m["id"]), "?")
        if m.get("method") == "notifications/progress":
            t = compact(m["params"]["progressToken"])
            m["params"]["progressToken"] = tokens.get(t, "?")
        if m.get("method") == "notifications/cancelled":
            m["params"]["requestId"] = ids.get("c2s:" + compact(m["params"]["requestId"]), "?")
        out.append({"dir": d, "msg": m})
    return out
