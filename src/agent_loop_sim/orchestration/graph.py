"""A graph runtime with LangGraph's execution model: state channels with reducers, nodes, static,
join and conditional edges, Pregel-style super-steps, checkpoints after every super-step,
interrupts (dynamic and static), crash and resume, and time travel (replay and fork).

The graph is data (a JSON spec), and each node's behaviour is a short list of deterministic
operations, so the same graph runs here, in the TS port, and in real LangGraph
(``conformance/langgraph_graphs.py`` builds a ``StateGraph`` from the spec). CI runs every
conformance session in a pinned LangGraph and requires every checkpoint (step, source, parent,
values, next tasks, task results, errors and interrupts) to equal the engine's.

The semantics, as checked against LangGraph 1.2.14:
  - A run on a fresh thread writes an ``input`` checkpoint at step -1 (values: the reducer
    channels' defaults; next: ``__start__``); super-step 0 runs ``__start__``, which writes the
    input through the reducers.
  - Every node triggered by the previous super-step runs in the next one, all reading the same
    snapshot. Their writes wait at the barrier, then are applied in task-path order (for plain
    nodes, sorted by node name, not by finishing time) through each key's reducer. A key without
    a reducer accepts one write per super-step (``InvalidUpdateError`` otherwise).
  - A conditional edge's router sees the snapshot plus its own node's writes, not its siblings'.
  - A join edge (``add_edge([a, b], c)``) waits until every source has run, across super-steps.
  - ``next`` lists the triggered nodes in the order they were added to the graph.
  - A checkpoint is saved after every super-step; the recursion limit stops a run once the step
    counter passes ``start + limit + 1``.
  - A node that raises or interrupts saves no checkpoint; its siblings' writes are kept as pending
    writes on the current checkpoint and are not run again on resume. The interrupted node runs
    again from its first line when resumed.
  - Running from an earlier ``loop`` checkpoint (time travel) first copies it to a ``fork``
    checkpoint and re-runs its tasks; ``update_state(as_node=…)`` writes an ``update`` checkpoint.
  - ``Send`` (since 1.7): a router may return ``Send(node, arg)`` packets (here a leaf
    ``{"send": node, "over": key, "as": name}``: one packet per item of ``state[key]``). Each packet
    is a separate "push" task in the next super-step whose input is ``arg``, not the state. Push
    tasks are listed (``next``, ``tasks``) before the edge-triggered ("pull") tasks, in packet
    order; packets are collected in task order, each router's in its list order. Their writes are
    applied after the pull tasks' (LangGraph sorts tasks by path: ``__pregel_pull`` < ``__pregel_push``),
    in packet order. A crash or interrupt in one push task keeps its siblings' writes; resuming
    re-runs only that task.
"""
from __future__ import annotations

import copy
from typing import Any

START = "__start__"
END = "__end__"
REDUCERS = ["overwrite", "add", "max"]
SOURCES = ["input", "loop", "update", "fork"]


class GraphError(Exception):
    """A malformed graph spec."""


class InvalidUpdateError(Exception):
    """Two writes to a key without a reducer in one super-step (LangGraph's error of that name)."""


class NodeFailure(Exception):
    """A node raised (a crash injected by the session)."""


class _Interrupt(Exception):
    def __init__(self, payload: Any) -> None:
        super().__init__("interrupt")
        self.payload = payload


# ---------------------------------------------------------------------------------------------
# The spec


def validate(spec: dict[str, Any]) -> None:
    keys = [s["key"] for s in spec["state"]]
    if len(set(keys)) != len(keys):
        raise GraphError("duplicate state key")
    for s in spec["state"]:
        if s["reducer"] not in REDUCERS:
            raise GraphError(f"unknown reducer {s['reducer']}")
    names = [n["name"] for n in spec["nodes"]]
    if len(set(names)) != len(names) or START in names or END in names:
        raise GraphError("bad node names")
    for e in spec["edges"]:
        srcs = e["from"] if isinstance(e["from"], list) else [e["from"]]
        for s in srcs:
            if s != START and s not in names:
                raise GraphError(f"edge from unknown node {s}")
        tos = e["targets"] if "router" in e else [e["to"]]
        for t in tos:
            if t != END and t not in names:
                raise GraphError(f"edge to unknown node {t}")
    for n in spec.get("interrupt_before", []) + spec.get("interrupt_after", []):
        if n not in names:
            raise GraphError(f"interrupt on unknown node {n}")


def _channel(spec: dict[str, Any], key: str) -> dict[str, Any]:
    for s in spec["state"]:
        if s["key"] == key:
            return s
    raise GraphError(f"unknown state key {key}")


def default_values(spec: dict[str, Any]) -> dict[str, Any]:
    """The values before any write: a reducer channel starts at its type's empty value."""
    out: dict[str, Any] = {}
    for s in spec["state"]:
        if s["reducer"] != "overwrite":
            out[s["key"]] = [] if s["type"] == "list" else 0
    return out


def ordered(spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in spec["state"]:
        if s["key"] in values:
            out[s["key"]] = values[s["key"]]
    return out


def reduce_value(chan: dict[str, Any], cur: Any, has: bool, w: Any) -> Any:
    r = chan["reducer"]
    if r == "overwrite":
        return w
    if r == "add":
        if chan["type"] == "list":
            return list(cur) + list(w)
        return cur + w
    # max
    return w if w > cur else cur


def apply_writes(spec: dict[str, Any], values: dict[str, Any], writes: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Apply each task's update in task order (callers sort by node name) through the reducers.
    Raises InvalidUpdateError when an overwrite key gets more than one write."""
    out = dict(values)
    seen: dict[str, int] = {}
    for _, upd in writes:
        for k, v in upd.items():
            chan = _channel(spec, k)
            seen[k] = seen.get(k, 0) + 1
            if chan["reducer"] == "overwrite" and seen[k] > 1:
                raise InvalidUpdateError(f"At key '{k}': Can receive only one value per step.")
            out[k] = reduce_value(chan, out.get(k), k in out, v)
    return ordered(spec, out)


# ---------------------------------------------------------------------------------------------
# Node behaviour: deterministic operations on the snapshot


def _cond(c: dict[str, Any], state: dict[str, Any]) -> bool:
    if "len" in c:
        a = len(state.get(c["len"], []))
    else:
        a = state.get(c["key"])
    b = c["value"]
    op = c["op"]
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    raise GraphError(f"unknown comparison {op}")


def _expand(g: Any, state: dict[str, Any]) -> list[Any]:
    """A send leaf ``{"send": node, "over": key, "as": name, "with": [keys]}`` becomes one packet
    ``{"send": node, "arg": {name: item, **{k: state[k]}}}`` per item of ``state[key]``."""
    if isinstance(g, dict):
        out = []
        for item in state.get(g["over"], []):
            arg: dict[str, Any] = {g["as"]: copy.deepcopy(item)}
            for k in g.get("with", []):
                arg[k] = copy.deepcopy(state.get(k))
            out.append({"send": g["send"], "arg": arg})
        return out
    return [g]


def route(router: dict[str, Any], state: dict[str, Any]) -> list[Any]:
    """A router: ``{"goto": target(s)}`` or ``{"if": cond, "then": router|target(s), "else": …}``.
    A target is a node name or a send leaf (expanded into ``Send`` packets)."""
    if "goto" in router:
        g = router["goto"]
    elif _cond(router["if"], state):
        g = router["then"]
    else:
        g = router["else"]
    if isinstance(g, dict) and "send" not in g:
        return route(g, state)
    out: list[Any] = []
    for x in (g if isinstance(g, list) else [g]):
        out += _expand(x, state)
    return out


def run_ops(node: dict[str, Any], state: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Run a node's operations against its input snapshot; returns its update (keys in op order).
    ``ctx``: ``fail`` (node names that raise at their ``fail`` op this run), ``resume`` (the value
    an ``interrupt`` op returns, or absent), ``effects`` (the outside world's log)."""
    upd: dict[str, Any] = {}
    for op in node["ops"]:
        k = op["op"]
        if "when" in op and not _cond(op["when"], state):
            continue
        if k == "set":
            upd[op["key"]] = copy.deepcopy(op["value"])
        elif k == "append":
            upd[op["key"]] = [copy.deepcopy(op["value"])]
        elif k == "delta":
            upd[op["key"]] = op["value"]
        elif k == "inc":
            upd[op["key"]] = state.get(op["key"], 0) + op["by"]
        elif k == "len":
            upd[op["key"]] = len(state.get(op["of"], []))
        elif k == "copy":
            upd[op["key"]] = copy.deepcopy(state.get(op["from"]))
        elif k == "collect":
            upd[op["key"]] = [copy.deepcopy(state.get(op["from"]))]
        elif k == "effect":
            ctx["effects"].append(node["name"] + ":" + op["name"])
        elif k == "fail":
            if node["name"] in ctx["fail"]:
                raise NodeFailure(op.get("message", "node failed"))
        elif k == "interrupt":
            if "resume" not in ctx:
                raise _Interrupt(copy.deepcopy(op["payload"]))
            upd[op["key"]] = copy.deepcopy(ctx["resume"])
        else:
            raise GraphError(f"unknown op {k}")
    return upd


# ---------------------------------------------------------------------------------------------
# The runtime


def _node(spec: dict[str, Any], name: str) -> dict[str, Any]:
    for n in spec["nodes"]:
        if n["name"] == name:
            return n
    raise GraphError(f"unknown node {name}")


def _decl(spec: dict[str, Any]) -> list[str]:
    return [n["name"] for n in spec["nodes"]]


def _join_id(e: dict[str, Any]) -> str:
    return "join:" + "+".join(e["from"]) + ":" + e["to"]


class Thread:
    """One LangGraph thread: a tree of checkpoints in creation order.

    A checkpoint: ``id`` (creation index), ``parent``, ``step``, ``source``, ``values``, ``next``
    (task names: push tasks first, then pull tasks), ``tasks`` (per task: ``name``, ``error``,
    ``interrupts``, ``result``), and the scheduler's own state: ``barriers`` (join id → sources
    seen), ``ready`` (the pull tasks that will run: ``pull`` without join targets still waiting),
    ``pull`` (the pull part of ``next``) and ``sends`` (the pending ``Send`` packets)."""

    def __init__(self, spec: dict[str, Any]) -> None:
        validate(spec)
        self.spec = spec
        self.cps: list[dict[str, Any]] = []
        self.latest: int | None = None

    # -- checkpoints

    def _put(self, parent: int | None, step: int, source: str, values: dict[str, Any], nxt: list[str],
             barriers: dict[str, list[str]], ready: list[str], sends: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        sends = copy.deepcopy(sends or [])
        names = [s["name"] for s in sends] + list(nxt)
        cp = {"id": len(self.cps), "parent": parent, "step": step, "source": source,
              "values": ordered(self.spec, values), "next": names,
              "tasks": [{"name": n, "error": None, "interrupts": [], "result": None} for n in names],
              "barriers": {k: list(v) for k, v in barriers.items()}, "ready": list(ready),
              "pull": list(nxt), "sends": sends}
        self.cps.append(cp)
        self.latest = cp["id"]
        return cp

    def history(self) -> list[dict[str, Any]]:
        """What ``get_state_history`` shows, oldest first (the conformance record)."""
        return [{"id": c["id"], "parent": c["parent"], "step": c["step"], "source": c["source"],
                 "values": copy.deepcopy(c["values"]), "next": list(c["next"]),
                 "tasks": copy.deepcopy(c["tasks"])} for c in self.cps]

    # -- scheduling

    def _triggers(self, values: dict[str, Any], done: list[tuple[str, dict[str, Any]]],
                  barriers: dict[str, list[str]]) -> tuple[list[str], list[str], dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Which nodes come next, given the tasks that just finished (name, update) and the
        snapshot before their writes. Returns ``next`` as LangGraph lists it (every node whose
        trigger was written, including a join target whose barrier is still incomplete), the nodes
        that will actually run (``ready``), the updated barriers, the edges that fired, and the
        ``Send`` packets (in task order, each router's in its list order)."""
        trig: set[str] = set()
        sends: list[dict[str, Any]] = []
        bar = {k: list(v) for k, v in barriers.items()}
        fired: list[dict[str, Any]] = []
        for name, upd in done:
            for e in self.spec["edges"]:
                if "router" in e:
                    if e["from"] != name:
                        continue
                    own = apply_writes(self.spec, values, [(name, upd)])
                    for t in route(e["router"], own):
                        if isinstance(t, dict):
                            fired.append({"from": name, "to": t["send"], "kind": "send", "arg": t["arg"]})
                            sends.append({"name": t["send"], "arg": t["arg"]})
                            continue
                        fired.append({"from": name, "to": t, "kind": "conditional"})
                        if t != END:
                            trig.add(t)
                elif isinstance(e["from"], list):
                    if name not in e["from"]:
                        continue
                    jid = _join_id(e)
                    seen = bar.get(jid, [])
                    if name not in seen:
                        seen = seen + [name]
                    bar[jid] = seen
                    fired.append({"from": name, "to": e["to"], "kind": "join",
                                  "waiting": [s for s in e["from"] if s not in seen]})
                elif e["from"] == name:
                    fired.append({"from": name, "to": e["to"], "kind": "edge"})
                    if e["to"] != END:
                        trig.add(e["to"])
        for f in fired:
            if f["kind"] == "join":
                for e in self.spec["edges"]:
                    if "router" not in e and isinstance(e["from"], list) and e["to"] == f["to"] and f["from"] in e["from"]:
                        f["waiting"] = [s for s in e["from"] if s not in bar.get(_join_id(e), [])]
        touched: set[str] = set()
        for e in self.spec["edges"]:
            if "router" not in e and isinstance(e["from"], list):
                jid = _join_id(e)
                if any(name in e["from"] for name, _ in done):
                    touched.add(e["to"])
                if all(s in bar.get(jid, []) for s in e["from"]):
                    trig.add(e["to"])
        nxt = [n for n in _decl(self.spec) if n in trig or n in touched]
        ready = [n for n in _decl(self.spec) if n in trig]
        # a join is consumed when its target runs
        for e in self.spec["edges"]:
            if "router" not in e and isinstance(e["from"], list):
                jid = _join_id(e)
                if all(s in bar.get(jid, []) for s in e["from"]):
                    bar[jid] = []
        bar = {k: v for k, v in bar.items() if v}
        return nxt, ready, bar, fired, sends

    # -- running

    def invoke(self, inp: dict[str, Any] | None = None, *, resume: Any = None, has_resume: bool = False,
               checkpoint: int | None = None, fail: list[str] | None = None, recursion_limit: int = 25) -> dict[str, Any]:
        """Run until done, an interrupt, an error, or the recursion limit. ``inp`` starts a run on a
        fresh thread; ``None`` resumes from the latest checkpoint (or from ``checkpoint``: time
        travel); ``has_resume`` passes ``resume`` to the interrupted node (``Command(resume=…)``)."""
        ctx_fail = list(fail or [])
        effects: list[str] = []
        calls: list[str] = []
        steps: list[dict[str, Any]] = []
        if inp is not None:
            if self.cps:
                raise GraphError("input on a thread that already has checkpoints is not modelled")
            cur = self._put(None, -1, "input", default_values(self.spec), [START], {}, [START])
            stop = -1 + recursion_limit + 1
            # super-step 0: __start__ writes the input
            step = 0
            cur["tasks"][0]["result"] = copy.deepcopy(inp)
            values = apply_writes(self.spec, cur["values"], [(START, inp)])
            done = [(START, inp)]
            nxt, ready, bar, fired, sends = self._triggers(cur["values"], done, {})
            par = cur["id"]
            cur = self._put(cur["id"], 0, "loop", values, nxt, bar, ready, sends)
            steps.append({"step": 0, "tasks": [START], "writes": [[START, copy.deepcopy(inp)]], "fired": fired,
                          "parent": par, "checkpoint": cur["id"]})
            step = 1
            reuse = False
        else:
            if checkpoint is not None:
                base = self.cps[checkpoint]
                stop = base["step"] + 1 + recursion_limit + 1
                if base["source"] in ("update", "fork"):
                    cur = base
                    reuse = False
                else:
                    cur = self._put(base["id"], base["step"] + 1, "fork", base["values"], base["pull"], base["barriers"], base["ready"],
                                    base["sends"])
                step = cur["step"] + 1
                reuse = False
            else:
                if self.latest is None:
                    raise GraphError("nothing to resume")
                cur = self.cps[self.latest]
                stop = cur["step"] + 1 + recursion_limit + 1
                step = cur["step"] + 1
                reuse = True
        status = "done"
        interrupts: list[Any] = []
        error = None
        halt = None
        first = True
        while True:
            if step > stop:
                status = "recursion_limit"
                break
            # (push index or -1, node name, record): push tasks first, in packet order, then pull tasks
            npush = len(cur["sends"])
            todo: list[tuple[int, str, dict[str, Any]]] = [(i, s["name"], cur["tasks"][i]) for i, s in enumerate(cur["sends"])]
            for t in cur["ready"]:
                todo.append((-1, t, next(x for x in cur["tasks"][npush:] if x["name"] == t)))
            tasks = [t for _, t, _ in todo]
            if not tasks:
                status = "done"
                break
            # a resumed run (no input) never stops at a static interrupt on its first super-step
            ib = self.spec.get("interrupt_before", [])
            if ib and any(t in ib for t in tasks) and not (first and inp is None):
                status = "interrupt_before"
                break
            keyed: list[tuple[tuple[int, str, int], str, dict[str, Any]]] = []
            failed: list[str] = []
            first_error = None
            paused: list[Any] = []
            for i, t, rec in todo:
                # LangGraph applies writes in task-path order: pull tasks by name, then push tasks by index
                key = (1, "", i) if i >= 0 else (0, t, 0)
                if first and reuse and rec["result"] is not None:
                    keyed.append((key, t, copy.deepcopy(rec["result"])))
                    continue
                ctx: dict[str, Any] = {"fail": ctx_fail, "effects": effects}
                if first and reuse and has_resume and rec["interrupts"]:
                    ctx["resume"] = resume
                calls.append(t)
                try:
                    upd = run_ops(_node(self.spec, t), cur["sends"][i]["arg"] if i >= 0 else cur["values"], ctx)
                except NodeFailure as e:
                    rec["error"] = "NodeFailure('" + str(e) + "')"
                    failed.append(t)
                    if first_error is None:
                        first_error = rec["error"]
                    continue
                except _Interrupt as it:
                    rec["interrupts"] = [it.payload]
                    paused.append(it.payload)
                    continue
                rec["result"] = copy.deepcopy(upd)
                keyed.append((key, t, upd))
            first = False
            keyed.sort(key=lambda w: w[0])
            writes: list[tuple[str, dict[str, Any]]] = [(t, u) for _, t, u in keyed]
            halt = {"step": step, "tasks": tasks, "writes": [[n, copy.deepcopy(u)] for n, u in writes],
                    "failed": failed, "interrupts": paused, "parent": cur["id"]}
            if npush:
                halt["sends"] = copy.deepcopy(cur["sends"])
            if failed:
                status = "error"
                error = "NodeFailure: " + str(first_error)
                break
            if paused:
                status = "interrupted"
                interrupts = paused
                break
            try:
                values = apply_writes(self.spec, cur["values"], writes)
            except InvalidUpdateError as e:
                status = "error"
                error = "InvalidUpdateError: " + str(e)
                break
            halt = None
            nxt, ready, bar, fired, sends = self._triggers(cur["values"], writes, cur["barriers"])
            par = cur["id"]
            ran = copy.deepcopy(cur["sends"])
            cur = self._put(cur["id"], step, "loop", values, nxt, bar, ready, sends)
            st: dict[str, Any] = {"step": step, "tasks": tasks, "writes": [[n, copy.deepcopy(u)] for n, u in writes],
                                  "fired": fired, "parent": par, "checkpoint": cur["id"]}
            if npush:
                st["sends"] = ran
            steps.append(st)
            step += 1
            ia = self.spec.get("interrupt_after", [])
            if ia and any(t in ia for t in tasks):
                status = "interrupt_after"
                break
        return {"status": status, "values": copy.deepcopy(cur["values"]), "next": list(cur["next"]),
                "interrupts": interrupts, "error": error, "calls": calls, "effects": effects,
                "checkpoint": cur["id"], "steps": steps, "halt": halt}

    def update_state(self, checkpoint: int, values: dict[str, Any], as_node: str) -> int:
        """``update_state(config_at(checkpoint), values, as_node)``: an ``update`` checkpoint whose
        next tasks are as if ``as_node`` had just written ``values``."""
        base = self.cps[checkpoint]
        new = apply_writes(self.spec, base["values"], [(as_node, values)])
        nxt, ready, bar, _, sends = self._triggers(base["values"], [(as_node, values)], base["barriers"])
        cp = self._put(base["id"], base["step"] + 1, "update", new, nxt, bar, ready, sends)
        return cp["id"]


def comparable_history(spec: dict[str, Any], hist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The history as the conformance check compares it. One detail of LangGraph 1.2.14 is not
    deterministic: whether ``next`` lists a join target whose barrier is still incomplete (it did
    in about 9 runs of 10 in a measurement here; the channel versions it compares carry a random
    suffix). Such a task never runs from that checkpoint, so for every checkpoint that has a
    ``loop`` child, a join target whose task has no result, error or interrupt is dropped from
    ``next`` and ``tasks`` on both sides. Everything else is compared as is."""
    targets = {e["to"] for e in spec["edges"] if "router" not in e and isinstance(e["from"], list)}
    parents = {c["parent"] for c in hist if c["source"] == "loop"}
    out = []
    for c in hist:
        c = copy.deepcopy(c)
        if c["id"] in parents:
            drop = [t["name"] for t in c["tasks"] if t["name"] in targets and t["result"] is None
                    and t["error"] is None and not t["interrupts"]]
            c["next"] = [n for n in c["next"] if n not in drop]
            c["tasks"] = [t for t in c["tasks"] if t["name"] not in drop]
        out.append(c)
    return out


def outcome(r: dict[str, Any]) -> dict[str, Any]:
    """The part of an operation's result the conformance check compares with LangGraph: the
    status (a static interrupt is ``paused``), the interrupt payloads, the error type, how many
    times each node ran, and the outside world's effects (both as sorted lists: LangGraph runs a
    super-step's tasks on threads, so their order within a step is not defined)."""
    st = r["status"]
    if st in ("interrupt_before", "interrupt_after"):
        st = "paused"
    return {"status": st, "interrupts": r["interrupts"],
            "error": r["error"].split(":")[0] if r["error"] else None,
            "calls": sorted(r["calls"]), "effects": sorted(r["effects"])}


def run_session(spec: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    """A conformance session: a list of operations on one thread. Each op is ``{"invoke": input}``,
    ``{"resume": value}``, ``{"continue": true}`` (invoke(None)), ``{"replay": index}`` (invoke(None)
    from the checkpoint at that creation index), or ``{"update": values, "at": index, "as_node": n}``;
    any op may carry ``fail`` and ``limit``. Records each op's outcome and the history after it."""
    th = Thread(spec)
    out = []
    for op in ops:
        kw: dict[str, Any] = {"fail": op.get("fail"), "recursion_limit": op.get("limit", 25)}
        if "invoke" in op:
            r = th.invoke(op["invoke"], **kw)
        elif "resume" in op:
            r = th.invoke(None, resume=op["resume"], has_resume=True, **kw)
        elif "continue" in op:
            r = th.invoke(None, **kw)
        elif "replay" in op:
            r = th.invoke(None, checkpoint=op["replay"], **kw)
        elif "update" in op:
            cid = th.update_state(op["at"], op["update"], op["as_node"])
            r = {"status": "updated", "values": copy.deepcopy(th.cps[cid]["values"]), "next": list(th.cps[cid]["next"]),
                 "interrupts": [], "error": None, "calls": [], "effects": [], "checkpoint": cid, "steps": [],
                 "halt": None}
        else:
            raise GraphError(f"unknown session op {op}")
        out.append({"op": op, "result": r, "history": th.history()})
    return {"name": spec["name"], "ops": out}
