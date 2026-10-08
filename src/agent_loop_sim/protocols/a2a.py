"""Agent to agent: the A2A protocol (version 1.0, specification release 1.0.1) as two state
machines, an orchestrating agent (the A2A client) and a remote research agent (the A2A server),
over the JSON-RPC binding with Server-Sent Events for streaming.

What is modelled (A2A specification 1.0.1, https://a2a-protocol.org/latest/specification/):
discovery through the Agent Card at ``/.well-known/agent-card.json`` (with the weak ``ETag`` the
official SDK derives from it); ``SendMessage`` returning a Task or a Message;
``SendStreamingMessage`` (the Task, then ``statusUpdate`` and ``artifactUpdate`` events, the
stream closing on a terminal state); ``GetTask``; ``CancelTask``; ``SubscribeToTask`` (refused on a
terminal task); ``GetExtendedAgentCard`` (not offered); the task life cycle with its terminal
(completed, failed, canceled, rejected) and interrupted (input required, auth required) states;
multi-turn follow-ups carrying ``taskId`` and ``contextId``; the ``A2A-Version`` header (an empty
one means 0.3); and the A2A error codes (-32001 to -32009) with ``google.rpc.ErrorInfo`` details.
Push notifications, ``ListTasks`` and the gRPC and HTTP+JSON bindings are not modelled.

The research agent is scripted (see ``conformance/a2a_sdk_server.py``, the same agent written with
the official A2A Python SDK). ``scripts/record_a2a_sdk.py`` drives that SDK agent with this
module's client and records the exchange; the engine's own agent must give the same messages
once IDs and timestamps are normalised. IDs here come from the seeded generator, timestamps from
a simulated clock, so a run is reproducible in Python and TypeScript.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Callable

from ..jsonfmt import compact
from ..rng import Rng

VERSION = "1.0"
SPEC_RELEASE = "1.0.1"
AGENT = "https://research.example.com"
RPC_URL = AGENT + "/a2a"
CARD_PATH = "/.well-known/agent-card.json"

SUBMITTED = "TASK_STATE_SUBMITTED"
WORKING = "TASK_STATE_WORKING"
COMPLETED = "TASK_STATE_COMPLETED"
FAILED = "TASK_STATE_FAILED"
CANCELED = "TASK_STATE_CANCELED"
INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
REJECTED = "TASK_STATE_REJECTED"
AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"
TERMINAL = [COMPLETED, FAILED, CANCELED, REJECTED]
INTERRUPTED = [INPUT_REQUIRED, AUTH_REQUIRED]
STATES = [SUBMITTED, WORKING, INPUT_REQUIRED, AUTH_REQUIRED, COMPLETED, FAILED, CANCELED, REJECTED]
# The transitions the specification describes (TaskState, 4.1.3; multi-turn, 3.4; in-task
# authorisation, 7.6); the state diagram draws these.
TRANSITIONS: list[list[str]] = [
    [SUBMITTED, WORKING], [SUBMITTED, REJECTED], [WORKING, INPUT_REQUIRED], [WORKING, AUTH_REQUIRED],
    [INPUT_REQUIRED, WORKING], [AUTH_REQUIRED, WORKING], [WORKING, COMPLETED], [WORKING, FAILED],
    [WORKING, CANCELED], [INPUT_REQUIRED, CANCELED], [AUTH_REQUIRED, CANCELED], [WORKING, REJECTED],
]

# Simulated times (illustrative): one-way network latency, the agent's work, the orchestrator's
# model deciding what to send next, and a person signing in.
TIMES = {"net": 20, "start": 40, "chunk": 600, "decide": 30, "model": 1200, "human": 4000}

CARD: dict[str, Any] = {
    "name": "Research agent",
    "description": "Summarises documents and meetings for other agents.",
    "supportedInterfaces": [{"url": RPC_URL, "protocolBinding": "JSONRPC", "protocolVersion": VERSION}],
    "version": "1.0.0",
    "capabilities": {"streaming": True},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [
        {"id": "summarise", "name": "Summarise", "description": "Summarise a document or a meeting.",
         "tags": ["summary", "text"], "examples": ["summarise the design notes"]},
        {"id": "calendar", "name": "Calendar digest",
         "description": "Today's meetings from the user's calendar (the user signs in first).",
         "tags": ["calendar"], "examples": ["check my calendar"]},
    ],
}

SUMMARY_BODY = "three decisions, two actions, no blockers."
CALENDAR_BODY = "design review at 10:00, one-to-one at 15:00."


def sorted_compact(x: Any) -> str:
    """``json.dumps(x, sort_keys=True, separators=(",", ":"))`` for ASCII JSON without floats."""
    if isinstance(x, dict):
        return "{" + ",".join(json.dumps(k) + ":" + sorted_compact(x[k]) for k in sorted(x.keys())) + "}"
    if isinstance(x, list):
        return "[" + ",".join(sorted_compact(v) for v in x) + "]"
    return compact(x)


def card_etag(card: dict[str, Any]) -> str:
    """The weak ETag the official SDK sends with the card: SHA-256 of the card (without its
    signatures) as sorted, compact JSON."""
    unsigned = {k: v for k, v in card.items() if k != "signatures"}
    return 'W/"' + hashlib.sha256(sorted_compact(unsigned).encode("utf-8")).hexdigest() + '"'


def uuid4(rng: Rng) -> str:
    """A version-4 UUID from the seeded generator."""
    b = [rng.randint(0, 255) for _ in range(16)]
    b[6] = (b[6] & 0x0F) | 0x40
    b[8] = (b[8] & 0x3F) | 0x80
    h = "".join(format(x, "02x") for x in b)
    return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:32]


def timestamp(ms: int) -> str:
    """The simulated clock as an ISO 8601 UTC time, starting at 2026-10-08T10:00:00Z."""
    t = 36000000 + ms
    h = t // 3600000
    m = (t // 60000) % 60
    s = (t // 1000) % 60
    return ("2026-10-08T" + format(h, "02d") + ":" + format(m, "02d") + ":" + format(s, "02d") + "."
            + format((t % 1000) * 1000, "06d") + "Z")


def text_of(msg: dict[str, Any]) -> str:
    return "\n".join(p["text"] for p in msg.get("parts", []) if "text" in p)


def rpc_request(id_: Any, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    m: dict[str, Any] = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        m["params"] = params
    return m


def rpc_result(id_: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def error_info(reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason, "domain": "a2a-protocol.org",
            "metadata": metadata if metadata is not None else {}}


def rpc_error(id_: Any, code: int, message: str, data: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        e["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": e}


# The A2A error types and their JSON-RPC codes (specification 5.4).
ERRORS: dict[str, int] = {
    "TaskNotFoundError": -32001, "TaskNotCancelableError": -32002, "PushNotificationNotSupportedError": -32003,
    "UnsupportedOperationError": -32004, "ContentTypeNotSupportedError": -32005, "InvalidAgentResponseError": -32006,
    "ExtendedAgentCardNotConfiguredError": -32007, "ExtensionSupportRequiredError": -32008,
    "VersionNotSupportedError": -32009,
}


class Agent:
    """The remote research agent: an A2A server with a task store and a scripted executor."""

    def __init__(self, seed: int = 23) -> None:
        self.rng = Rng(seed)
        self.tasks: dict[str, dict[str, Any]] = {}
        self.state = "idle"
        self.clock = 0

    # --- the task store -------------------------------------------------------------------
    def _snapshot(self, task: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"id": task["id"], "contextId": task["contextId"]}
        st: dict[str, Any] = {"state": task["status"]["state"]}
        if "message" in task["status"]:
            st["message"] = copy.deepcopy(task["status"]["message"])
        if "timestamp" in task["status"]:
            st["timestamp"] = task["status"]["timestamp"]
        out["status"] = st
        if task["artifacts"]:
            out["artifacts"] = copy.deepcopy(task["artifacts"])
        out["history"] = copy.deepcopy(task["history"])
        return out

    def _archive(self, task: dict[str, Any]) -> None:
        """A status message moves into the history when the next message or status arrives."""
        if "message" in task["status"]:
            task["history"].append(task["status"]["message"])
            del task["status"]["message"]

    def _agent_message(self, task: dict[str, Any], text: str) -> dict[str, Any]:
        return {"messageId": uuid4(self.rng), "contextId": task["contextId"], "taskId": task["id"],
                "role": "ROLE_AGENT", "parts": [{"text": text}]}

    # --- events the executor emits; each is applied to the task and becomes a stream event ---
    def _status(self, task: dict[str, Any], state: str, text: str | None, dt: int) -> dict[str, Any]:
        self.clock += dt
        self._archive(task)
        st: dict[str, Any] = {"state": state}
        if text is not None:
            st["message"] = self._agent_message(task, text)
        st["timestamp"] = timestamp(self.clock)
        task["status"] = st
        self.state = "task " + state.replace("TASK_STATE_", "").lower().replace("_", " ")
        return {"statusUpdate": {"taskId": task["id"], "contextId": task["contextId"],
                                 "status": copy.deepcopy(st)}, "_t": self.clock}

    def _artifact(self, task: dict[str, Any], aid: str, text: str, append: bool, last: bool) -> dict[str, Any]:
        self.clock += TIMES["chunk"]
        if append:
            for a in task["artifacts"]:
                if a["artifactId"] == aid:
                    a["parts"].append({"text": text})
        else:
            task["artifacts"].append({"artifactId": aid, "name": "summary", "parts": [{"text": text}]})
        ev: dict[str, Any] = {"taskId": task["id"], "contextId": task["contextId"],
                              "artifact": {"artifactId": aid, "name": "summary", "parts": [{"text": text}]}}
        if append:
            ev["append"] = True
        if last:
            ev["lastChunk"] = True
        self.state = "streaming its artifact"
        return {"artifactUpdate": ev, "_t": self.clock}

    def _finish(self, task: dict[str, Any], head: str, body: str) -> list[dict[str, Any]]:
        aid = uuid4(self.rng)
        return [self._artifact(task, aid, head, False, False), self._artifact(task, aid, body, True, True),
                self._status(task, COMPLETED, None, TIMES["decide"])]

    def _execute(self, msg: dict[str, Any], task: dict[str, Any] | None) -> list[dict[str, Any]]:
        """The scripted executor (the same rules as the SDK fixture agent)."""
        text = text_of(msg)
        if task is None:
            ctx = uuid4(self.rng)
            if text.startswith("hello"):
                self.clock += TIMES["decide"]
                self.state = "answered with a message"
                return [{"message": {"messageId": uuid4(self.rng), "contextId": ctx, "role": "ROLE_AGENT",
                                     "parts": [{"text": "Hello. Ask me to summarise something."}]}, "_t": self.clock}]
            tid = uuid4(self.rng)
            m = copy.deepcopy(msg)
            m["contextId"] = ctx
            m["taskId"] = tid
            m = {"messageId": m["messageId"], "contextId": ctx, "taskId": tid, "role": m["role"], "parts": m["parts"]}
            task = {"id": tid, "contextId": ctx, "status": {"state": SUBMITTED}, "artifacts": [], "history": [m]}
            self.tasks[tid] = task
            self.clock += TIMES["decide"]
            self.state = "task submitted"
            evs: list[dict[str, Any]] = [{"task": self._snapshot(task), "_t": self.clock}]
            if text.startswith("delete"):
                evs.append(self._status(task, REJECTED, "I only summarise; I do not delete files.", TIMES["decide"]))
                return evs
            evs.append(self._status(task, WORKING, None, TIMES["start"]))
            if text == "summarise the meeting":
                evs.append(self._status(task, INPUT_REQUIRED, "Which meeting: Monday's or Tuesday's?", TIMES["decide"]))
                return evs
            if text == "check my calendar":
                evs.append(self._status(task, AUTH_REQUIRED,
                                        "Sign in to the calendar first: https://calendar.example.com/authorize",
                                        TIMES["decide"]))
                return evs
            subject = text[len("summarise "):] if text.startswith("summarise ") else text
            return evs + self._finish(task, "Summary of " + subject + ": ", SUMMARY_BODY)
        was = task["status"]["state"]
        evs = [self._status(task, WORKING, None, TIMES["start"])]
        if was == AUTH_REQUIRED:
            return evs + self._finish(task, "Today: ", CALENDAR_BODY)
        return evs + self._finish(task, "Summary of " + text + "'s meeting: ", SUMMARY_BODY)

    # --- the JSON-RPC endpoint -----------------------------------------------------------
    def card(self) -> dict[str, Any]:
        self.state = "served its card"
        return {"status": 200, "headers": [["Content-Type", "application/json"], ["ETag", card_etag(CARD)]],
                "body": copy.deepcopy(CARD)}

    def handle(self, req: dict[str, Any], headers: list[list[str]], at: int) -> list[dict[str, Any]]:
        """Answer one request. Returns the responses: one for a plain call, one per event for a
        stream. Each carries ``_t``, the simulated time it is sent."""
        self.clock = max(self.clock, at)
        id_ = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        version = ""
        for k, v in headers:
            if k.lower() == "a2a-version":
                version = v
        if version == "":
            version = "0.3"

        def out(m: dict[str, Any]) -> list[dict[str, Any]]:
            self.clock += 1
            return [dict(m, _t=self.clock)]

        if version != VERSION:
            self.state = "refused the version"
            return out(rpc_error(id_, -32009, "A2A version '" + version + "' is not supported by this handler. "
                                 "Expected version '" + VERSION + "'.", [error_info("VERSION_NOT_SUPPORTED")]))
        if method in ("SendMessage", "SendStreamingMessage"):
            msg = params.get("message") or {}
            if len(msg.get("parts") or []) == 0:
                self.state = "refused the request"
                problem = "Field must contain at least one element."
                return out(rpc_error(id_, -32602, "Validation failed", [
                    error_info("INVALID_PARAMS", {"errors": [{"field": "message.parts", "message": problem}]}),
                    {"@type": "type.googleapis.com/google.rpc.BadRequest",
                     "fieldViolations": [{"field": "message.parts", "description": problem}]}]))
            task = None
            if "taskId" in msg:
                task = self.tasks.get(msg["taskId"])
                if task is None:
                    return out(rpc_error(id_, -32001, "Task not found", [error_info("TASK_NOT_FOUND")]))
                if task["status"]["state"] in TERMINAL:
                    self.state = "refused the message"
                    return out(rpc_error(id_, -32004, "Task " + task["id"] + " is in terminal state: "
                                         + task["status"]["state"], [error_info("UNSUPPORTED_OPERATION")]))
                self._archive(task)
                task["history"].append(copy.deepcopy(msg))
            evs = self._execute(msg, task)
            if method == "SendStreamingMessage":
                res: list[dict[str, Any]] = []
                for e in evs:
                    t = e["_t"]
                    body = {k: v for k, v in e.items() if k != "_t"}
                    res.append(dict(rpc_result(id_, body), _t=t))
                return res
            last = evs[-1]["_t"]
            first = {k: v for k, v in evs[0].items() if k != "_t"}
            if "message" in first:
                return [dict(rpc_result(id_, first), _t=last)]
            tid = task["id"] if task is not None else first["task"]["id"]
            return [dict(rpc_result(id_, {"task": self._snapshot(self.tasks[tid])}), _t=last)]
        if method == "GetTask":
            task = self.tasks.get(params.get("id", ""))
            if task is None:
                self.state = "has no such task"
                return out(rpc_error(id_, -32001, "Task not found", [error_info("TASK_NOT_FOUND")]))
            return out(rpc_result(id_, self._snapshot(task)))
        if method == "CancelTask":
            task = self.tasks.get(params.get("id", ""))
            if task is None:
                return out(rpc_error(id_, -32001, "Task not found", [error_info("TASK_NOT_FOUND")]))
            if task["status"]["state"] in TERMINAL:
                self.state = "cannot cancel"
                return out(rpc_error(id_, -32002, "Task cannot be canceled", [error_info("TASK_NOT_CANCELABLE")]))
            # The task is waiting (interrupted), so nothing is running for it: like the SDK's
            # store, write CANCELED over the stored status, keeping its message and time.
            task["status"]["state"] = CANCELED
            self.state = "task canceled"
            return out(rpc_result(id_, self._snapshot(task)))
        if method == "SubscribeToTask":
            task = self.tasks.get(params.get("id", ""))
            if task is None:
                return out(rpc_error(id_, -32001, "Task not found", [error_info("TASK_NOT_FOUND")]))
            if task["status"]["state"] in TERMINAL:
                self.state = "refused to subscribe"
                return out(rpc_error(id_, -32004, "Task " + task["id"] + " is in terminal state: "
                                     + task["status"]["state"], [error_info("UNSUPPORTED_OPERATION")]))
            return out(rpc_result(id_, {"task": self._snapshot(task)}))
        if method == "GetExtendedAgentCard":
            self.state = "has no extended card"
            return out(rpc_error(id_, -32004, "The agent does not support authenticated extended cards",
                                 [error_info("UNSUPPORTED_OPERATION")]))
        self.state = "does not know the method"
        return out(rpc_error(id_, -32601, "Method not found"))


# A transport: ("card", None, headers, t) -> {status, headers, body}; ("rpc", request, headers,
# t) -> the list of response messages (each with ``_t``) and whether they came as an SSE stream.
Send = Callable[[str, Any, list[list[str]], int], Any]


def engine_send(agent: Agent) -> Send:
    def send(kind: str, req: Any, headers: list[list[str]], at: int) -> Any:
        if kind == "card":
            r = agent.card()
            return dict(r, _t=at + 1)
        res = agent.handle(req, headers, at)
        stream = req.get("method") in ("SendStreamingMessage", "SubscribeToTask") and "result" in res[0]
        return {"messages": res, "sse": stream}
    return send


def short_state(s: str) -> str:
    return s.replace("TASK_STATE_", "")


def play(sc: dict[str, Any], send: Send | None = None) -> dict[str, Any]:
    """Run an A2A scenario: the orchestrator (client) follows ``sc["steps"]``. Returns ``wire``
    (what crosses HTTP, in order: requests with their headers, responses, one item per SSE
    event), ``log`` (the same annotated: kind, method, bytes, time, both agents' states and the
    task's state) and ``card`` (what the client learnt). With ``send`` the client talks to
    another agent (the conformance recorder passes the official SDK's)."""
    agent = Agent()
    tx = send if send is not None else engine_send(agent)
    rng = Rng(11)
    wire: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    cl: dict[str, Any] = {"state": "idle", "task": None, "context": None, "task_state": None, "card": None,
                          "next_id": 1, "clock": 0}

    def server_state() -> str:
        return agent.state if send is None else "(the SDK agent)"

    def record(item: dict[str, Any], kind: str, method: str, id_: Any, nbytes: int, note: str, t: int) -> None:
        wire.append(item)
        log.append({"seq": len(log), "dir": item["dir"], "kind": kind, "method": method, "id": id_,
                    "bytes": nbytes, "note": note, "t": t, "client": cl["state"], "server": server_state(),
                    "task": cl["task_state"]})

    def track(m: dict[str, Any]) -> None:
        r = m.get("result")
        if not isinstance(r, dict):
            return
        task = r.get("task") if "task" in r else (r if "status" in r and "id" in r else None)
        if task is not None:
            cl["task"] = task["id"]
            cl["context"] = task["contextId"]
            cl["task_state"] = task["status"]["state"]
        elif "statusUpdate" in r:
            cl["task_state"] = r["statusUpdate"]["status"]["state"]
        elif "message" in r:
            cl["context"] = r["message"].get("contextId")

    def call(method: str, params: dict[str, Any] | None, note: str, headers: list[list[str]] | None = None) -> list[dict[str, Any]]:
        id_ = cl["next_id"]
        cl["next_id"] += 1
        req = rpc_request(id_, method, params)
        hs = headers if headers is not None else [["Content-Type", "application/json"], ["A2A-Version", VERSION]]
        cl["state"] = "waiting for " + method
        t0 = cl["clock"]
        record({"dir": "c2s", "msg": req, "headers": hs}, "request", method, id_, len(compact(req).encode("utf-8")),
               note, t0)
        r = tx("rpc", req, hs, t0 + TIMES["net"])
        msgs = r["messages"]
        for m in msgs:
            t = m["_t"] + TIMES["net"]
            body = {k: v for k, v in m.items() if k != "_t"}
            track(body)
            kind = "error" if "error" in body else "result"
            if r["sse"]:
                cl["state"] = "reading the stream"
                text = "data: " + compact(body) + "\n\n"
            else:
                cl["state"] = "has the answer"
                text = compact(body)
            if r["sse"] and kind == "result" and (cl["task_state"] in TERMINAL or "message" in body["result"]):
                cl["state"] = "stream closed"
            record({"dir": "s2c", "msg": body, "sse": r["sse"]}, kind, method, id_, len(text.encode("utf-8")), "", t)
            cl["clock"] = t
        cl["state"] = "ready" if cl["task_state"] is None else "task " + short_state(cl["task_state"]).lower().replace("_", " ")
        return [{k: v for k, v in m.items() if k != "_t"} for m in msgs]

    def message(text: str, follow: bool) -> dict[str, Any]:
        m: dict[str, Any] = {"messageId": uuid4(rng), "role": "ROLE_USER", "parts": [{"text": text}]}
        if follow:
            m["taskId"] = cl["task"]
            m["contextId"] = cl["context"]
        return m

    for st in sc["steps"]:
        op = st["op"]
        if op == "card":
            cl["state"] = "discovering"
            url = AGENT + CARD_PATH
            req_text = "GET " + CARD_PATH + " HTTP/1.1"
            record({"dir": "c2s", "http": {"method": "GET", "url": url}}, "http", "agent card", None,
                   len(req_text.encode("utf-8")), "the client reads the agent card", cl["clock"])
            r = tx("card", None, [], cl["clock"] + TIMES["net"])
            body = r["body"]
            iface = body["supportedInterfaces"][0]
            ok = iface["protocolBinding"] == "JSONRPC" and iface["protocolVersion"] == VERSION
            cl["card"] = {"name": body["name"], "url": iface["url"], "binding": iface["protocolBinding"],
                          "version": iface["protocolVersion"], "streaming": bool(body["capabilities"].get("streaming")),
                          "skills": [s["id"] for s in body["skills"]], "usable": ok}
            cl["state"] = "has the card" if ok else "cannot use this agent"
            t = r["_t"] + TIMES["net"]
            record({"dir": "s2c", "http": {"status": r["status"], "headers": r["headers"], "body": body}}, "http",
                   "agent card", None, len(compact(body).encode("utf-8")), "", t)
            cl["clock"] = t
        elif op in ("send", "reply"):
            follow = op == "reply"
            cl["clock"] += TIMES["model"]
            p: dict[str, Any] = {"message": message(st["text"], follow)}
            if "configuration" in st:
                p["configuration"] = copy.deepcopy(st["configuration"])
            method = "SendStreamingMessage" if st.get("stream") else "SendMessage"
            call(method, p, st.get("note", ""))
        elif op == "sign_in":
            cl["state"] = "asking the user to sign in"
            cl["clock"] += TIMES["human"]
            record({"dir": "host", "gate": "the user signs in to the calendar (outside A2A)"}, "gate", "", None, 0,
                   "out of band", cl["clock"])
            cl["state"] = "the user has signed in"
        elif op == "get":
            call("GetTask", {"id": cl["task"]}, st.get("note", ""))
        elif op == "cancel":
            call("CancelTask", {"id": cl["task"]}, st.get("note", ""))
        elif op == "subscribe":
            call("SubscribeToTask", {"id": cl["task"]}, st.get("note", ""))
        elif op == "raw":
            params = copy.deepcopy(st.get("params"))
            if st.get("task_id") and params is not None:
                params["id"] = cl["task"]
            if st.get("task_message") and params is not None:
                params["message"]["taskId"] = cl["task"]
            call(st["method"], params, st.get("note", ""), st.get("headers"))
        else:
            raise ValueError("unknown step " + op)
    return {"name": sc["name"], "wire": wire, "log": log, "card": cl["card"]}


UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def normalise(wire: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What must agree between the engine and the SDK: every message, with each UUID replaced
    by ``<id-k>`` in order of first appearance (keys visited in sorted order) and each
    timestamp by ``<time>``. Host-side steps (signing in) are not on the wire and are dropped."""
    seen: dict[str, str] = {}

    def ids(s: str) -> str:
        out = ""
        pos = 0
        for m in UUID_RE.finditer(s):
            u = m.group(0)
            if u not in seen:
                seen[u] = "<id-" + str(len(seen) + 1) + ">"
            out += s[pos:m.start()] + seen[u]
            pos = m.end()
        return out + s[pos:]

    def walk(x: Any, key: str) -> Any:
        if isinstance(x, dict):
            return {k: walk(x[k], k) for k in sorted(x.keys())}
        if isinstance(x, list):
            return [walk(v, key) for v in x]
        if isinstance(x, str):
            if key == "timestamp":
                return "<time>"
            return ids(x)
        return x

    out = []
    for w in wire:
        if w["dir"] == "host":
            continue
        item = {k: v for k, v in w.items() if k != "headers" or w["dir"] == "c2s"}
        out.append(walk(item, ""))
    return out


def _send(text: str, stream: bool = False, note: str = "") -> dict[str, Any]:
    return {"op": "send", "text": text, "stream": stream, "note": note}


def _reply(text: str, stream: bool = False, note: str = "") -> dict[str, Any]:
    return {"op": "reply", "text": text, "stream": stream, "note": note}


_CARD = {"op": "card"}
_NO_VERSION = [["Content-Type", "application/json"]]
_MSG = {"message": {"messageId": "00000000-0000-4000-8000-000000000001", "role": "ROLE_USER",
                    "parts": [{"text": "summarise the design notes"}]}}

# Every A2A scenario; each is recorded against the official SDK's agent in conformance.
A2A_SCENARIOS: dict[str, dict[str, Any]] = {
    "a2a_send": {"title": "Delegate a task and wait for the result", "steps": [
        _CARD, _send("summarise the design notes", note="the orchestrator delegates and waits"),
        {"op": "get", "note": "later, the orchestrator reads the task again"}]},
    "a2a_stream": {"title": "Delegate a task and stream its progress", "steps": [
        _CARD, _send("summarise the design notes", True, "the orchestrator delegates and streams")]},
    "a2a_input": {"title": "The remote agent needs more input", "steps": [
        _CARD, _send("summarise the meeting", note="an ambiguous request"),
        _reply("Tuesday", note="the orchestrator answers in the same task")]},
    "a2a_auth": {"title": "The remote agent needs the user to sign in", "steps": [
        _CARD, _send("check my calendar"), {"op": "sign_in"},
        _reply("signed in", note="the orchestrator tells the agent the user has signed in")]},
    "a2a_cancel": {"title": "Cancel a task, then try to use it", "steps": [
        _CARD, _send("summarise the meeting"), {"op": "cancel", "note": "the orchestrator gives up"},
        {"op": "cancel", "note": "cancelling twice"}, _reply("Tuesday", note="too late: the task is over"),
        {"op": "subscribe", "note": "nothing to stream: the task is over"}]},
    "a2a_reject": {"title": "The remote agent rejects the task", "steps": [
        _CARD, _send("delete every draft")]},
    "a2a_message": {"title": "A direct answer, no task", "steps": [_CARD, _send("hello")]},
    "a2a_errors": {"title": "Errors: no such task, no version, an old method name, no parts, no extended card",
                   "steps": [
        {"op": "raw", "method": "GetTask", "params": {"id": "00000000-0000-4000-8000-0000000000ff"},
         "note": "a task this agent never made"},
        {"op": "raw", "method": "SendMessage", "params": _MSG, "headers": _NO_VERSION,
         "note": "no A2A-Version header: the agent must assume 0.3"},
        {"op": "raw", "method": "message/send", "params": _MSG, "note": "the 0.3 method name"},
        {"op": "raw", "method": "SendMessage",
         "params": {"message": {"messageId": "00000000-0000-4000-8000-000000000002", "role": "ROLE_USER", "parts": []}},
         "note": "a message with no parts"},
        {"op": "raw", "method": "GetExtendedAgentCard", "params": None, "note": "this agent offers no extended card"},
    ]},
}


def a2a_scenario(name: str) -> dict[str, Any]:
    sc = copy.deepcopy(A2A_SCENARIOS[name])
    sc["name"] = name
    return sc
