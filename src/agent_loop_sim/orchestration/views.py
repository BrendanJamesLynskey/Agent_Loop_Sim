"""Frames for the site's animations, computed from the runtime's results (so every frame shows
state the conformance-checked runtime produced): super-steps (chapters 1 and 2), the checkpoint
timeline of a session (chapter 3) and the durable-execution replay (chapter 4, in durable.py).

Durations are illustrative (each node's ``ms``, plus ``CKPT_MS`` per checkpoint write)."""
from __future__ import annotations

import copy
from typing import Any

from .graph import END, START, _channel, reduce_value, run_session

CKPT_MS = 15


def names(xs: list[str]) -> str:
    """English list: "a", "a and b", "a, b and c"."""
    if not xs:
        return "nothing"
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def layout(spec: dict[str, Any]) -> dict[str, Any]:
    """Columns by longest path from START over forward edges (a back edge, to a node already
    placed, is drawn as a loop); rows by declaration order within a column."""
    order = [n["name"] for n in spec["nodes"]]
    succ: dict[str, list[str]] = {START: []}
    for n in order:
        succ[n] = []
    for e in spec["edges"]:
        srcs = e["from"] if isinstance(e["from"], list) else [e["from"]]
        tos = e["targets"] if "router" in e else [e["to"]]
        for s in srcs:
            for t in tos:
                if t != END and t not in succ[s]:
                    succ[s].append(t)
    col: dict[str, int] = {START: 0}
    # longest path by repeated relaxation, ignoring edges that would create a cycle (back edges)
    visiting: list[str] = []

    def dfs(u: str, depth: int) -> None:
        if u in visiting:
            return
        if u in col and col[u] >= depth and u != START:
            return
        col[u] = depth
        visiting.append(u)
        for v in succ[u]:
            if v not in visiting:
                dfs(v, depth + 1)
        visiting.pop()

    dfs(START, 0)
    for n in order:
        if n not in col:
            col[n] = 1
    end_col = max(col.values()) + 1
    cols: dict[int, list[str]] = {}
    for n in [START] + order:
        cols.setdefault(col[n], []).append(n)
    pos = {}
    for c, ns in cols.items():
        for i, n in enumerate(ns):
            pos[n] = {"col": c, "row": i, "rows": len(ns)}
    pos[END] = {"col": end_col, "row": 0, "rows": 1}
    edges = []
    for e in spec["edges"]:
        srcs = e["from"] if isinstance(e["from"], list) else [e["from"]]
        tos = e["targets"] if "router" in e else [e["to"]]
        kind = "conditional" if "router" in e else ("join" if isinstance(e["from"], list) else "edge")
        for s in srcs:
            for t in tos:
                edges.append({"from": s, "to": t, "kind": kind, "back": pos[t]["col"] <= pos[s]["col"]})
    return {"nodes": pos, "cols": end_col + 1, "edges": edges}


def folds(spec: dict[str, Any], before: dict[str, Any], writes: list[list[Any]]) -> dict[str, list[dict[str, Any]]]:
    """For each key written this super-step, the reducer's fold over the writes in task order:
    the value after each write (an overwrite key with two writes stops with ``error``)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for name, upd in writes:
        for k, v in upd.items():
            chan = _channel(spec, k)
            seq = out.setdefault(k, [])
            if seq and chan["reducer"] == "overwrite":
                seq.append({"node": name, "write": copy.deepcopy(v), "value": None, "error": True})
                continue
            cur = seq[-1]["value"] if seq else before.get(k)
            seq.append({"node": name, "write": copy.deepcopy(v), "value": reduce_value(chan, cur, True, v), "error": False})
    return out


def _fmt(v: Any) -> str:
    """A value as the captions show it: JSON-like, keys unquoted."""
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(k + ": " + _fmt(x) for k, x in v.items()) + "}"
    return str(v)


def superstep_frames(spec: dict[str, Any], result: dict[str, Any], history: list[dict[str, Any]],
                     t0: int = 0) -> dict[str, Any]:
    """Three frames per super-step of one invoke: ``run`` (the tasks read one snapshot), ``barrier``
    (their writes wait, in the order they will be applied), ``apply`` (the reducers fold them in,
    a checkpoint is saved, edges fire). A halted step (error, interrupt) ends with a ``halt`` frame."""
    ms = {n["name"]: n["ms"] for n in spec["nodes"]}
    ms[START] = 0
    frames: list[dict[str, Any]] = []
    t = t0
    seq_total = 0
    steps = list(result["steps"])
    for st in steps:
        s = st["step"]
        tasks = st["tasks"]
        before = history[st["parent"]]["values"]
        after = history[st["checkpoint"]]
        dur = max(ms[x] for x in tasks)
        for x in tasks:
            seq_total += ms[x]
        if tasks == [START]:
            cap = "Super-step 0: __start__ writes the input " + _fmt(st["writes"][0][1]) + " through the reducers."
        elif len(tasks) == 1:
            cap = f"Super-step {s}: {tasks[0]} runs ({ms[tasks[0]]} ms), reading the snapshot."
        else:
            cap = (f"Super-step {s}: {names(tasks)} run in parallel, all reading the same snapshot; "
                   f"the step lasts as long as the slowest ({dur} ms).")
        frames.append({"phase": "run", "step": s, "active": tasks, "values": before, "t": t, "dur": dur,
                       "caption": cap})
        order = [w[0] for w in st["writes"]]
        if len(order) > 1:
            cap = ("Barrier: no write is visible until every task finishes; they are applied in task order (" +
                   " then ".join(order) + "), not finishing order.")
        else:
            cap = f"Barrier: {order[0]}'s write " + _fmt(st["writes"][0][1]) + " waits for the end of the step."
        frames.append({"phase": "barrier", "step": s, "active": tasks, "values": before, "writes": st["writes"],
                       "t": t + dur, "dur": dur, "caption": cap})
        t = t + dur + CKPT_MS
        fl = folds(spec, before, st["writes"])
        went = [f["to"] for f in st["fired"] if f["kind"] != "join" and f["to"] != END]
        waits = [f for f in st["fired"] if f["kind"] == "join" and f["waiting"]]
        parts = [f"Checkpoint {after['id']} saved (step {s})."]
        if after["next"]:
            ready = [n for n in after["next"] if n not in [w["to"] for w in waits] or n in went]
            if ready:
                parts.append("Next: " + names(ready) + ".")
            for w in waits:
                parts.append(f"The join into {w['to']} still waits for {names(w['waiting'])}.")
        else:
            parts.append("No node is triggered: the run is done.")
        frames.append({"phase": "apply", "step": s, "active": tasks, "values": after["values"], "writes": st["writes"],
                       "folds": fl, "fired": st["fired"], "next": after["next"], "checkpoint": after["id"],
                       "t": t, "dur": dur, "caption": " ".join(parts)})
    h = result.get("halt")
    if h is not None:
        before = history[h["parent"]]["values"]
        if h["failed"]:
            cap = (f"Super-step {h['step']}: {names(h['failed'])} raised. No checkpoint is written; "
                   + ("the writes of " + names([w[0] for w in h["writes"]]) + " are kept as pending writes." if h["writes"] else
                      "nothing is kept."))
        elif h["interrupts"]:
            cap = (f"Super-step {h['step']}: an interrupt pauses the run with " + _fmt(h["interrupts"][0]) + ". " +
                   ("Pending writes kept: " + names([w[0] for w in h["writes"]]) + "." if h["writes"] else ""))
        else:
            fl = folds(spec, before, h["writes"])
            bad = [k for k, seq in fl.items() if any(x["error"] for x in seq)]
            cap = (f"Super-step {h['step']}: {names([w[0] for w in h['writes']])} both wrote '{bad[0]}', which has no reducer: "
                   "InvalidUpdateError. No checkpoint is saved.")
        frames.append({"phase": "halt", "step": h["step"], "active": h["tasks"], "values": before, "writes": h["writes"],
                       "folds": folds(spec, before, h["writes"]) if not h["failed"] and not h["interrupts"] else {},
                       "t": t, "dur": 0, "caption": cap.strip(), "status": result["status"]})
    graph_ms = t - t0
    runs = sum(len(st["tasks"]) for st in steps)
    return {"frames": frames, "graph_ms": graph_ms, "sequential_ms": seq_total + CKPT_MS * runs,
            "steps": len(steps), "status": result["status"]}


def _op_caption(op: dict[str, Any], hist_before: list[dict[str, Any]]) -> str:
    if "invoke" in op:
        cap = "invoke(" + _fmt(op["invoke"]) + ") on a new thread"
        if op.get("fail"):
            cap += f" (worker running {names(op['fail'])} will crash)"
        return cap + "."
    if "resume" in op:
        return "invoke(Command(resume=" + _fmt(op["resume"]) + ")): the human answers; the interrupted node runs again from its first line."
    if "continue" in op:
        tail = f" ({names(op['fail'])} will crash again)" if op.get("fail") else ""
        return "invoke(None): resume the thread from its latest checkpoint" + tail + "."
    if "replay" in op:
        c = hist_before[op["replay"]]
        return (f"invoke(None, checkpoint {op['replay']}): time travel to step {c['step']} and run again from there.")
    return (f"update_state(checkpoint {op['at']}, " + _fmt(op["update"]) + f", as_node={op['as_node']}): "
            "edit the state as if that node had written it.")


def _cp_caption(c: dict[str, Any]) -> str:
    src = {"input": "the input", "loop": "after a super-step", "update": "an edit (update_state)",
           "fork": "a copy of an earlier checkpoint, so the old branch is kept"}[c["source"]]
    nxt = ("next: " + names(c["next"])) if c["next"] else "nothing next"
    return f"Checkpoint {c['id']} (step {c['step']}, {c['source']}: {src}); {nxt}."


def _outcome_caption(r: dict[str, Any], hist: list[dict[str, Any]]) -> str:
    st = r["status"]
    eff = (" Side effects this call: " + names(r["effects"]) + ".") if r["effects"] else ""
    if st == "done":
        return "Done: the run reached END." + eff
    if st == "updated":
        return "No node ran; a new checkpoint holds the edit."
    if st == "interrupted":
        h = r["halt"]
        kept = [w[0] for w in h["writes"]]
        return ("Paused by interrupt(" + _fmt(r["interrupts"][0]) + "). No new checkpoint; "
                + ("pending writes of " + names(kept) + " are saved on checkpoint " + str(r["checkpoint"]) + "." if kept
                   else "the thread waits on checkpoint " + str(r["checkpoint"]) + ".") + eff)
    if st in ("interrupt_before", "interrupt_after"):
        where = "before" if st == "interrupt_before" else "after"
        return (f"Paused by a static interrupt ({where} {names([n for n in hist[r['checkpoint']]['next']] if st == 'interrupt_before' else [])})"
                .replace(" ()", "") + f" at checkpoint {r['checkpoint']}; a human can inspect or edit the state." + eff)
    if st == "error":
        h = r["halt"]
        if h is not None and h["failed"]:
            kept = [w[0] for w in h["writes"]]
            return (f"Crash: {names(h['failed'])} raised. Checkpoint {r['checkpoint']} stays the latest; "
                    + ("the finished work of " + names(kept) + " is saved there as pending writes." if kept else "nothing else finished.")
                    + eff)
        return "Error: " + (r["error"] or "") + eff
    return f"Stopped: recursion limit reached at checkpoint {r['checkpoint']}." + eff


def timeline_frames(spec: dict[str, Any], sess: dict[str, Any]) -> dict[str, Any]:
    """Chapter 3: a session as a timeline of checkpoints. One frame per operation, one per new
    checkpoint, one for the outcome. Each frame says which op's history it shows (``h``) and how
    many of its checkpoints are visible (``n``)."""
    res = run_session(spec, sess["ops"])
    frames: list[dict[str, Any]] = []
    prev: list[dict[str, Any]] = []
    ops = []
    for i, o in enumerate(res["ops"]):
        hist = o["history"]
        ops.append({"op": o["op"], "status": o["result"]["status"], "history": hist,
                    "effects": o["result"]["effects"], "calls": o["result"]["calls"],
                    "interrupts": o["result"]["interrupts"]})
        frames.append({"kind": "op", "h": i, "n": len(prev), "focus": None, "caption": _op_caption(o["op"], prev)})
        for c in hist[len(prev):]:
            frames.append({"kind": "checkpoint", "h": i, "n": c["id"] + 1, "focus": c["id"], "caption": _cp_caption(c)})
        frames.append({"kind": "outcome", "h": i, "n": len(hist), "focus": o["result"]["checkpoint"],
                       "status": o["result"]["status"], "caption": _outcome_caption(o["result"], hist)})
        prev = hist
    return {"session": sess["name"], "graph": spec["name"], "ops": ops, "frames": frames}


def session_supersteps(spec: dict[str, Any], sess: dict[str, Any]) -> list[dict[str, Any]]:
    """Super-step frames for every invoke-like op of a session (chapter 1 and 2 use the first)."""
    res = run_session(spec, sess["ops"])
    out = []
    for o in res["ops"]:
        out.append(superstep_frames(spec, o["result"], o["history"]))
    return out
