"""MCP transports: how each JSON-RPC message is framed, how many bytes that costs, and how long
a session takes, for stdio and Streamable HTTP in both protocol eras.

* **stdio**: the client starts the server as a child process; each message is one line of
  JSON on the server's stdin or stdout, ended by ``\\n`` (one byte of framing).
* **Streamable HTTP, 2025-11-25**: every client message is an HTTP POST to one endpoint. The
  server answers a request with JSON, or with a Server-Sent Events stream that carries
  notifications and its own requests before the response. ``initialize`` mints a session
  (``Mcp-Session-Id``); later requests carry it and ``MCP-Protocol-Version``. Client
  notifications and answers get ``202 Accepted``. Events carry IDs, so a dropped stream can be
  resumed with ``Last-Event-ID``.
* **Streamable HTTP, 2026-07-28**: no session and no resumption. Each POST carries
  ``MCP-Protocol-Version``, ``Mcp-Method`` and (for tools/call, resources/read, prompts/get)
  ``Mcp-Name``. A broken stream loses the request; the client sends it again with a new ID.

The header sets are the minimal ones the spec requires plus Host/Content-Length; real stacks
add more (Date, Server, chunked encoding). Latency numbers are illustrative parameters, not
measurements: see ``TRANSPORTS``.
"""
from __future__ import annotations

from typing import Any

from ..jsonfmt import compact
from .mcp import kind_of, notification, request, response, text_content, utf8_len

TRANSPORTS: dict[str, dict[str, Any]] = {
    # Illustrative: a local pipe is ~tens of microseconds; spawning a Python server ~100s of ms.
    "stdio": {"label": "stdio", "one_way_ms": 0.05, "mbit_per_s": 4000, "startup_ms": 150},
    # Illustrative: a nearby cloud region over a home connection.
    "http": {"label": "Streamable HTTP", "one_way_ms": 20, "mbit_per_s": 50, "startup_ms": 80},
}
WORK_MS = {"default": 1, "tools/call": 3, "count_step": 200, "human": 4000, "model": 1500}
HOST = "mcp.example.com"
SESSION_ID = "4f1c9a7e2b6d4e83"


def _lat(t: dict[str, Any], nbytes: int) -> float:
    return t["one_way_ms"] + nbytes * 8 / (t["mbit_per_s"] * 1000)


def _http_request(body: str, headers: list[list[str]]) -> str:
    lines = ["POST /mcp HTTP/1.1", "Host: " + HOST, "Content-Type: application/json",
             "Accept: application/json, text/event-stream"]
    for k, v in headers:
        lines.append(k + ": " + v)
    lines.append("Content-Length: " + str(utf8_len(body)))
    return "\r\n".join(lines) + "\r\n\r\n"


def _mcp_name(msg: dict[str, Any]) -> str | None:
    p = msg.get("params") or {}
    if msg.get("method") in ("tools/call", "prompts/get"):
        return p.get("name")
    if msg.get("method") == "resources/read":
        return p.get("uri")
    return None


def frame_session(log: list[dict[str, Any]], wire: list[dict[str, Any]], transport: str) -> dict[str, Any]:
    """Frame a played session (``play(sc)``) on a transport: per message, the framing text,
    payload and overhead bytes, and when it leaves and arrives; plus totals."""
    t = TRANSPORTS[transport]
    era = "handshake"
    for w in wire:
        if "msg" in w and w["msg"].get("method") == "initialize":
            era = "handshake"
            break
        meta = ((w.get("msg") or {}).get("params") or {}).get("_meta") or {}
        if "io.modelcontextprotocol/protocolVersion" in meta:
            era = "modern"
            break
    version = None
    session = False
    streams: dict[str, dict[str, Any]] = {}  # request id -> its POST's response stream
    open_ids: list[str] = []
    requests: dict[str, dict[str, Any]] = {}
    frames: list[dict[str, Any]] = []
    tc = t["startup_ms"]
    ts = tc
    event_id = 0
    posts = 0
    for i, w in enumerate(wire):
        lg = log[i]
        if "msg" not in w:
            body = w["raw"]
            frames.append({"seq": i, "dir": "c2s", "framing": "\\n", "payload": utf8_len(body), "overhead": 1,
                           "depart": tc, "arrive": tc + _lat(t, utf8_len(body) + 1), "http": None})
            continue
        msg = w["msg"]
        body = compact(msg)
        payload = utf8_len(body)
        k = kind_of(msg)
        d = w["dir"]
        if d == "c2s" and k == "request":
            requests[compact(msg["id"])] = msg
        http: dict[str, Any] | None = None
        if transport == "stdio":
            framing = "\\n"
            overhead = 1
        elif d == "c2s":
            headers: list[list[str]] = []
            if era == "handshake":
                if version is not None:
                    headers.append(["MCP-Protocol-Version", version])
                if session:
                    headers.append(["Mcp-Session-Id", SESSION_ID])
            else:
                if k == "request":
                    headers.append(["MCP-Protocol-Version", "2026-07-28"])
                    headers.append(["Mcp-Method", msg["method"]])
                    nm = _mcp_name(msg)
                    if nm is not None:
                        headers.append(["Mcp-Name", nm])
            framing = _http_request(body, headers)
            overhead = utf8_len(framing)
            posts += 1
            if k == "request":
                key = compact(msg["id"])
                streams[key] = {"sse": False}
                open_ids.append(key)
                http = {"request": framing, "status": None}
            else:
                reply = "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n"
                overhead += utf8_len(reply)
                http = {"request": framing, "status": 202, "response": reply}
        else:
            # Which POST's response carries it: a response goes back on its own request's POST;
            # a notification or a server request rides the newest still-open stream.
            if k in ("result", "error"):
                key = compact(msg["id"])
            else:
                key = open_ids[-1] if open_ids else ""
            st = streams.get(key, {"sse": False})
            starts_stream = not st["sse"] and k not in ("result", "error")
            if k in ("result", "error") and not st["sse"]:
                # A plain JSON response.
                hdr = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                if era == "handshake" and requests.get(key, {}).get("method") == "initialize":
                    hdr += "Mcp-Session-Id: " + SESSION_ID + "\r\n"
                hdr += "Content-Length: " + str(payload) + "\r\n\r\n"
                framing = hdr
                http = {"status": 200, "content_type": "application/json"}
            else:
                prefix = ""
                if starts_stream:
                    prefix = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                    if era == "modern":
                        prefix += "X-Accel-Buffering: no\r\n"
                    prefix += "\r\n"
                    st["sse"] = True
                    streams[key] = st
                ev = ""
                if era == "handshake":
                    event_id += 1
                    ev = "id: " + str(event_id) + "\n"
                framing = prefix + ev + "event: message\ndata: " + "\n\n"
                http = {"status": 200, "content_type": "text/event-stream", "event_id": event_id if era == "handshake" else None,
                        "opens_stream": starts_stream}
            overhead = utf8_len(framing)
            if k in ("result", "error"):
                if key in open_ids:
                    open_ids.remove(key)
                if era == "handshake" and requests.get(key, {}).get("method") == "initialize" and k == "result":
                    session = True
                    version = msg["result"].get("protocolVersion")
        # --- time ---
        if d == "c2s":
            depart = tc
            if k == "result":
                if lg["method"] == "elicitation/create":
                    depart = tc + WORK_MS["human"]
                elif lg["method"] == "sampling/createMessage":
                    depart = tc + WORK_MS["model"]
            elif k == "request" and msg["method"] == "tools/call" and "inputResponses" in (msg.get("params") or {}):
                ir = msg["params"]["inputResponses"]
                depart = tc + (WORK_MS["human"] if "confirm" in ir else WORK_MS["model"])
            arrive = depart + _lat(t, payload + overhead)
            ts = max(ts, arrive)
            tc = depart
        else:
            work = WORK_MS["default"]
            if k == "notification" and msg["method"] == "notifications/progress":
                work = WORK_MS["count_step"]
            elif k in ("result", "error"):
                req = requests.get(compact(msg["id"]), {})
                if req.get("method") == "tools/call":
                    work = WORK_MS["tools/call"]
                    p = req.get("params") or {}
                    if p.get("name") == "count" and "progressToken" not in (p.get("_meta") or {}):
                        work = work + p["arguments"]["n"] * WORK_MS["count_step"]
            depart = ts + work
            ts = depart
            arrive = depart + _lat(t, payload + overhead)
            tc = max(tc, arrive)
        frames.append({"seq": i, "dir": d, "framing": framing, "payload": payload, "overhead": overhead,
                       "depart": depart, "arrive": arrive, "http": http})
    payload_total = 0
    overhead_total = 0
    for f in frames:
        payload_total += f["payload"]
        overhead_total += f["overhead"]
    end = t["startup_ms"]
    for f in frames:
        end = max(end, f["arrive"])
    return {"transport": transport, "era": era, "frames": frames,
            "totals": {"messages": len(frames), "payload": payload_total, "overhead": overhead_total,
                       "posts": posts, "elapsed_ms": end, "startup_ms": t["startup_ms"]}}


def stream_drop(era: str, n: int, drop_after: int, reconnect_ms: float = 1000) -> dict[str, Any]:
    """A tools/call that reports progress over Streamable HTTP, whose SSE stream breaks after
    (1 <= drop_after < n)
    ``drop_after`` events. 2025-11-25: the server keeps working and buffers; the client
    reconnects with ``GET`` + ``Last-Event-ID`` and the missed events are replayed.
    2026-07-28: the break cancels the request; the client sends it again with a new ID and the
    work starts over. Returns the steps (for the animation) and the totals."""
    t = TRANSPORTS["http"]
    step = WORK_MS["count_step"]
    steps: list[dict[str, Any]] = []
    meta_v = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
              "io.modelcontextprotocol/clientInfo": {"name": "fixture-client", "version": "1.0.0"},
              "io.modelcontextprotocol/clientCapabilities": {}}
    bytes_total = 0
    work_steps = 0

    def call_msg(id_: int) -> dict[str, Any]:
        meta: dict[str, Any] = {"progressToken": id_}
        if era == "modern":
            meta = dict(meta_v)
            meta["progressToken"] = id_
        return request(id_, "tools/call", {"name": "count", "arguments": {"n": n}, "_meta": meta})

    def result_msg(id_: int) -> dict[str, Any]:
        r: dict[str, Any] = {"content": text_content("counted to " + str(n)), "isError": False,
                             "structuredContent": {"result": "counted to " + str(n)}}
        if era == "modern":
            r["resultType"] = "complete"
            r["_meta"] = {"io.modelcontextprotocol/serverInfo": {"name": "fixture-server", "version": "1.0.0"}}
        return response(id_, r)

    def add(tm: float, actor: str, kind: str, label: str, msg: dict[str, Any] | None, event_id: int | None,
            extra: int, live: bool) -> None:
        nonlocal bytes_total
        b = (utf8_len(compact(msg)) if msg is not None else 0) + extra
        if live:
            bytes_total += b
        steps.append({"t": tm, "actor": actor, "kind": kind, "label": label, "event_id": event_id, "bytes": b,
                      "live": live, "msg": msg})

    ev_frame = len("event: message\ndata: \n\n")
    post_extra = 220  # request line and headers, about the size frame_session computes
    tc = 0.0
    # First attempt.
    first = call_msg(1)
    add(tc, "client", "post", "POST tools/call (id 1)", first, None, post_extra, True)
    ts = tc + _lat(t, post_extra)
    eid = 0
    lost_at = 0.0
    for i in range(1, n + 1):
        ts += step
        work_steps += 1
        prog = notification("notifications/progress", {"progressToken": 1, "progress": i, "total": n})
        if i <= drop_after:
            eid += 1
            add(ts + t["one_way_ms"], "server", "event", "progress " + str(i) + "/" + str(n), prog, eid,
                ev_frame + (len("id: " + str(eid) + "\n") if era == "handshake" else 0), True)
            lost_at = ts + t["one_way_ms"]
        elif era == "handshake":
            eid += 1
            add(ts, "server", "buffered", "progress " + str(i) + "/" + str(n) + " (buffered)", prog, eid, 0, False)
        else:
            break
        if i == drop_after:
            add(ts + t["one_way_ms"] + 1, "network", "drop", "the stream breaks", None, None, 0, True)
            if era == "modern":
                add(ts + t["one_way_ms"] + 1, "server", "cancel", "closing the stream cancels the request", None, None,
                    0, True)
                break
    if era == "handshake":
        final_eid = eid + 1
        add(ts + 1, "server", "buffered", "result (buffered)", result_msg(1), final_eid, 0, False)
        tc = lost_at + reconnect_ms
        get_hdr = ("GET /mcp HTTP/1.1\r\nHost: " + HOST + "\r\nAccept: text/event-stream\r\nMcp-Session-Id: "
                   + SESSION_ID + "\r\nMCP-Protocol-Version: 2025-11-25\r\nLast-Event-ID: " + str(drop_after) + "\r\n\r\n")
        add(tc, "client", "resume", "GET with Last-Event-ID: " + str(drop_after), None, None, utf8_len(get_hdr), True)
        tr = tc + _lat(t, utf8_len(get_hdr))
        replayed = 0
        kept: list[dict[str, Any]] = []
        for s in steps:
            if s["kind"] != "buffered":
                kept.append(s)
        buffered: list[dict[str, Any]] = []
        for s in steps:
            if s["kind"] == "buffered":
                buffered.append(s)
        steps[:] = kept
        for s in buffered:
            eid_r = s["event_id"]
            extra = ev_frame + len("id: " + str(eid_r) + "\n")
            if s["t"] <= tr:
                # Produced while the client was away: kept by the server, sent again on the new stream.
                replayed += 1
                add(s["t"], "server", "buffered", s["label"], s["msg"], eid_r, 0, False)
                add(tr + t["one_way_ms"], "server", "replay", "replay event " + str(eid_r), s["msg"], eid_r, extra, True)
            else:
                # Produced after the reconnect: sent live on the resumed stream.
                label = s["label"].replace(" (buffered)", "")
                add(s["t"] + t["one_way_ms"], "server", "event", label, s["msg"], eid_r, extra, True)
        # The replay ends with the result, which cannot arrive before the work has finished.
        end = max(tr + t["one_way_ms"], ts + 1 + t["one_way_ms"])
        redone = 0
        lost_requests = 0
    else:
        tc = lost_at + reconnect_ms
        again = call_msg(2)
        add(tc, "client", "post", "POST tools/call again (id 2)", again, None, post_extra, True)
        ts = tc + _lat(t, post_extra)
        for i in range(1, n + 1):
            ts += step
            work_steps += 1
            prog = notification("notifications/progress", {"progressToken": 2, "progress": i, "total": n})
            add(ts + t["one_way_ms"], "server", "event", "progress " + str(i) + "/" + str(n) + " (again)", prog, None,
                ev_frame, True)
        add(ts + 1 + t["one_way_ms"], "server", "event", "result", result_msg(2), None, ev_frame, True)
        end = ts + 1 + t["one_way_ms"]
        redone = drop_after
        replayed = 0
        lost_requests = 1
    steps.sort(key=lambda s: s["t"])
    for i, s in enumerate(steps):
        s["seq"] = i
    return {"era": era, "n": n, "drop_after": drop_after, "steps": steps,
            "totals": {"elapsed_ms": end, "bytes": bytes_total, "work_steps": work_steps, "redone_steps": redone,
                       "replayed_events": replayed, "lost_requests": lost_requests}}
