"""Graphs and sessions: the conformance set (each one run in real LangGraph in CI) and the
teaching graphs the site animates (which are conformance graphs too, so every animated
super-step is one LangGraph produces).

A node's ``ms`` is its illustrative duration, used only by the timeline views; it plays no part in
the semantics (LangGraph applies writes in task order, not finishing order, and the conformance
sessions show it)."""
from __future__ import annotations

import copy
from typing import Any

LOG = {"key": "log", "reducer": "add", "type": "list"}


def _n(name: str, ms: int, *ops: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "ms": ms, "ops": [{"op": "append", "key": "log", "value": name}] + list(ops)}


def _e(a: Any, b: str) -> dict[str, Any]:
    return {"from": a, "to": b}


def _r(a: str, router: dict[str, Any], targets: list[str]) -> dict[str, Any]:
    return {"from": a, "router": router, "targets": targets}


GRAPHS: list[dict[str, Any]] = [
    {
        "name": "chain",
        "title": "A chain: retrieve, draft, review",
        "state": [LOG, {"key": "question", "reducer": "overwrite", "type": "str"},
                  {"key": "docs", "reducer": "overwrite", "type": "int"},
                  {"key": "draft", "reducer": "overwrite", "type": "str"}],
        "nodes": [_n("retrieve", 600, {"op": "set", "key": "docs", "value": 3}),
                  _n("draft", 1400, {"op": "set", "key": "draft", "value": "v1"}),
                  _n("review", 800, {"op": "set", "key": "draft", "value": "v1 (reviewed)"})],
        "edges": [_e("__start__", "retrieve"), _e("retrieve", "draft"), _e("draft", "review"), _e("review", "__end__")],
    },
    {
        "name": "fanout",
        "title": "Fan out to two researchers, join, write",
        "state": [LOG, {"key": "notes", "reducer": "add", "type": "int"},
                  {"key": "sources", "reducer": "add", "type": "list"},
                  {"key": "report", "reducer": "overwrite", "type": "int"}],
        "nodes": [_n("plan", 500),
                  _n("web", 1800, {"op": "delta", "key": "notes", "value": 3}, {"op": "append", "key": "sources", "value": "web"}),
                  _n("papers", 1200, {"op": "delta", "key": "notes", "value": 2}, {"op": "append", "key": "sources", "value": "papers"}),
                  _n("write", 900, {"op": "len", "key": "report", "of": "sources"})],
        "edges": [_e("__start__", "plan"), _e("plan", "web"), _e("plan", "papers"), _e(["web", "papers"], "write"),
                  _e("write", "__end__")],
    },
    {
        "name": "react_loop",
        "title": "An agent loop: model, tools, model … until done",
        "state": [LOG, {"key": "turns", "reducer": "overwrite", "type": "int"},
                  {"key": "tool_calls", "reducer": "add", "type": "int"}],
        "nodes": [_n("model", 900, {"op": "inc", "key": "turns", "by": 1}),
                  _n("tools", 400, {"op": "delta", "key": "tool_calls", "value": 1})],
        "edges": [_e("__start__", "model"),
                  _r("model", {"if": {"key": "turns", "op": "<", "value": 3}, "then": "tools", "else": "__end__"},
                     ["tools", "__end__"]),
                  _e("tools", "model")],
    },
    {
        "name": "uneven_join",
        "title": "A join waits across super-steps",
        "state": [LOG],
        "nodes": [_n("a", 300), _n("b", 300), _n("c", 300), _n("d", 300), _n("e", 300)],
        "edges": [_e("__start__", "a"), _e("a", "b"), _e("b", "c"), _e("a", "d"), _e(["c", "d"], "e"), _e("e", "__end__")],
    },
    {
        "name": "no_join",
        "title": "Two plain edges into one node: it runs twice",
        "state": [LOG],
        "nodes": [_n("b", 300), _n("a", 300), _n("c", 300), _n("e", 300)],
        "edges": [_e("__start__", "c"), _e("__start__", "a"), _e("a", "b"), _e("b", "e"), _e("c", "e"), _e("e", "__end__")],
    },
    {
        "name": "router_view",
        "title": "A router sees its own node's writes, not its sibling's",
        "state": [LOG, {"key": "n", "reducer": "overwrite", "type": "int"}, {"key": "x", "reducer": "overwrite", "type": "int"}],
        "nodes": [_n("p", 300, {"op": "inc", "key": "n", "by": 1}), _n("q", 300, {"op": "set", "key": "x", "value": 7}),
                  _n("r", 300)],
        "edges": [_e("__start__", "p"), _e("__start__", "q"),
                  _r("p", {"if": {"key": "x", "op": "==", "value": 7}, "then": "r", "else": "__end__"}, ["r", "__end__"]),
                  _e("q", "__end__"), _e("r", "__end__")],
    },
    {
        "name": "merge_overwrite",
        "title": "Two branches write one key with no reducer",
        "state": [LOG, {"key": "answer", "reducer": "overwrite", "type": "int"}],
        "nodes": [_n("plan", 400), _n("fast", 700, {"op": "set", "key": "answer", "value": 41}),
                  _n("careful", 1600, {"op": "set", "key": "answer", "value": 42}),
                  _n("merge", 300)],
        "edges": [_e("__start__", "plan"), _e("plan", "fast"), _e("plan", "careful"), _e(["fast", "careful"], "merge"),
                  _e("merge", "__end__")],
    },
    {
        "name": "merge_add",
        "title": "The same branches with an add reducer (a list)",
        "state": [LOG, {"key": "answer", "reducer": "add", "type": "list"}],
        "nodes": [_n("plan", 400), _n("fast", 700, {"op": "append", "key": "answer", "value": 41}),
                  _n("careful", 1600, {"op": "append", "key": "answer", "value": 42}),
                  _n("merge", 300)],
        "edges": [_e("__start__", "plan"), _e("plan", "fast"), _e("plan", "careful"), _e(["fast", "careful"], "merge"),
                  _e("merge", "__end__")],
    },
    {
        "name": "merge_max",
        "title": "The same branches with a custom reducer (max)",
        "state": [LOG, {"key": "answer", "reducer": "max", "type": "int"}],
        "nodes": [_n("plan", 400), _n("fast", 700, {"op": "set", "key": "answer", "value": 41}),
                  _n("careful", 1600, {"op": "set", "key": "answer", "value": 42}),
                  _n("merge", 300)],
        "edges": [_e("__start__", "plan"), _e("plan", "fast"), _e("plan", "careful"), _e(["fast", "careful"], "merge"),
                  _e("merge", "__end__")],
    },
    {
        "name": "fanout_list",
        "title": "A router that returns a list fans out",
        "state": [LOG],
        "nodes": [_n("m", 300), _n("k", 300), _n("b", 300)],
        "edges": [_e("__start__", "b"), _e("__start__", "k"), _e("__start__", "m"), _r("b", {"goto": ["m", "k"]}, ["m", "k"])],
    },
    {
        "name": "start_router",
        "title": "A conditional edge from START",
        "state": [LOG, {"key": "kind", "reducer": "overwrite", "type": "str"}],
        "nodes": [_n("billing", 300), _n("tech", 300)],
        "edges": [_r("__start__", {"if": {"key": "kind", "op": "==", "value": "bill"}, "then": "billing", "else": "tech"},
                     ["billing", "tech"]),
                  _e("billing", "__end__"), _e("tech", "__end__")],
    },
    {
        "name": "approval",
        "title": "Draft, ask a human, send (a dynamic interrupt beside a sibling)",
        "state": [LOG, {"key": "draft", "reducer": "overwrite", "type": "str"},
                  {"key": "approved", "reducer": "overwrite", "type": "str"}, {"key": "checks", "reducer": "add", "type": "int"}],
        "nodes": [_n("draft", 1200, {"op": "set", "key": "draft", "value": "refund £40"}),
                  {"name": "ask", "ms": 300, "ops": [{"op": "effect", "name": "notify_reviewer"},
                                                     {"op": "interrupt", "key": "approved",
                                                      "payload": {"question": "Send this reply?", "draft": "refund £40"}},
                                                     {"op": "append", "key": "log", "value": "ask"}]},
                  _n("check", 700, {"op": "delta", "key": "checks", "value": 1}),
                  {"name": "send", "ms": 400, "ops": [{"op": "append", "key": "log", "value": "send"},
                                                      {"op": "effect", "name": "send_email"}]}],
        "edges": [_e("__start__", "draft"), _e("draft", "ask"), _e("draft", "check"), _e(["ask", "check"], "send"),
                  _e("send", "__end__")],
    },
    {
        "name": "gated",
        "title": "Plan, then a tool call gated by a static interrupt",
        "state": [LOG, {"key": "plan", "reducer": "overwrite", "type": "str"}, {"key": "result", "reducer": "overwrite", "type": "str"}],
        "nodes": [_n("planner", 900, {"op": "set", "key": "plan", "value": "delete 3 stale branches"}),
                  {"name": "execute", "ms": 600, "ops": [{"op": "append", "key": "log", "value": "execute"},
                                                         {"op": "effect", "name": "git_push"},
                                                         {"op": "copy", "key": "result", "from": "plan"}]},
                  _n("report", 300)],
        "edges": [_e("__start__", "planner"), _e("planner", "execute"), _e("execute", "report"), _e("report", "__end__")],
        "interrupt_before": ["execute"],
    },
    {
        "name": "gated_after",
        "title": "Review after a node (interrupt_after)",
        "state": [LOG],
        "nodes": [_n("a", 300), _n("b", 300)],
        "edges": [_e("__start__", "a"), _e("a", "b"), _e("b", "__end__")],
        "interrupt_after": ["a"],
    },
    {
        "name": "crash",
        "title": "A worker crashes mid-run; resume from the checkpoint",
        "state": [LOG, {"key": "rows", "reducer": "add", "type": "int"}],
        "nodes": [_n("load", 600, {"op": "delta", "key": "rows", "value": 100}),
                  {"name": "enrich", "ms": 1500, "ops": [{"op": "effect", "name": "call_api"}, {"op": "fail", "message": "worker died"},
                                                         {"op": "append", "key": "log", "value": "enrich"},
                                                         {"op": "delta", "key": "rows", "value": 10}]},
                  {"name": "score", "ms": 900, "ops": [{"op": "effect", "name": "call_model"},
                                                       {"op": "append", "key": "log", "value": "score"},
                                                       {"op": "delta", "key": "rows", "value": 1}]},
                  _n("publish", 400)],
        "edges": [_e("__start__", "load"), _e("load", "enrich"), _e("load", "score"), _e(["enrich", "score"], "publish"),
                  _e("publish", "__end__")],
    },
    {
        "name": "counter",
        "title": "A loop that counts (recursion limit)",
        "state": [LOG, {"key": "n", "reducer": "overwrite", "type": "int"}, {"key": "stop", "reducer": "overwrite", "type": "int"}],
        "nodes": [_n("a", 200, {"op": "inc", "key": "n", "by": 1})],
        "edges": [_e("__start__", "a"),
                  _r("a", {"if": {"key": "n", "op": "<", "value": 5}, "then": "a", "else": "__end__"}, ["a", "__end__"])],
    },
    {
        "name": "branchy",
        "title": "Plan, then one of two tools (for time travel)",
        "state": [LOG, {"key": "choice", "reducer": "overwrite", "type": "str"}, {"key": "cost", "reducer": "add", "type": "int"}],
        "nodes": [_n("plan", 800, {"op": "set", "key": "choice", "value": "search"}),
                  _n("search", 1100, {"op": "delta", "key": "cost", "value": 3}),
                  _n("calculator", 200, {"op": "delta", "key": "cost", "value": 1}),
                  _n("answer", 700)],
        "edges": [_e("__start__", "plan"),
                  _r("plan", {"if": {"key": "choice", "op": "==", "value": "search"}, "then": "search", "else": "calculator"},
                     ["search", "calculator"]),
                  _e("search", "answer"), _e("calculator", "answer"), _e("answer", "__end__")],
    },
    {
        "name": "loop_join",
        "title": "A fan-out and join inside a loop (the barrier resets each round)",
        "state": [LOG, {"key": "round", "reducer": "overwrite", "type": "int"}, {"key": "found", "reducer": "add", "type": "int"}],
        "nodes": [_n("plan", 300, {"op": "inc", "key": "round", "by": 1}), _n("w1", 900, {"op": "delta", "key": "found", "value": 2}),
                  _n("w2", 500, {"op": "delta", "key": "found", "value": 1}), _n("check", 200)],
        "edges": [_e("__start__", "plan"), _e("plan", "w1"), _e("plan", "w2"), _e(["w1", "w2"], "check"),
                  _r("check", {"if": {"key": "found", "op": "<", "value": 6}, "then": "plan", "else": "__end__"},
                     ["plan", "__end__"])],
    },
]


def graph(name: str) -> dict[str, Any]:
    for g in GRAPHS:
        if g["name"] == name:
            return copy.deepcopy(g)
    raise KeyError(name)


# Each session: a graph and a list of operations on one thread (see graph.run_session).
SESSIONS: list[dict[str, Any]] = [
    {"name": "chain", "graph": "chain", "ops": [{"invoke": {"question": "Which GPU?"}}]},
    {"name": "fanout", "graph": "fanout", "ops": [{"invoke": {"log": ["start"]}}]},
    {"name": "react_loop", "graph": "react_loop", "ops": [{"invoke": {"turns": 0}}]},
    {"name": "uneven_join", "graph": "uneven_join", "ops": [{"invoke": {"log": []}}]},
    {"name": "no_join", "graph": "no_join", "ops": [{"invoke": {"log": []}}]},
    {"name": "router_view", "graph": "router_view", "ops": [{"invoke": {"log": [], "n": 0}}]},
    {"name": "merge_overwrite", "graph": "merge_overwrite", "ops": [{"invoke": {"log": []}}]},
    {"name": "merge_add", "graph": "merge_add", "ops": [{"invoke": {"log": []}}]},
    {"name": "merge_max", "graph": "merge_max", "ops": [{"invoke": {"log": [], "answer": 40}}]},
    {"name": "fanout_list", "graph": "fanout_list", "ops": [{"invoke": {"log": []}}]},
    {"name": "start_router", "graph": "start_router", "ops": [{"invoke": {"kind": "bill"}}]},
    {"name": "approval", "graph": "approval", "ops": [{"invoke": {"log": []}}, {"resume": "yes"}]},
    {"name": "approval_no", "graph": "approval", "ops": [{"invoke": {"log": []}}, {"resume": "no, change the amount"}]},
    {"name": "gated", "graph": "gated", "ops": [{"invoke": {"log": []}}, {"continue": True}]},
    {"name": "gated_edit", "graph": "gated",
     "ops": [{"invoke": {"log": []}}, {"update": {"plan": "delete 1 stale branch"}, "at": 2, "as_node": "planner"},
             {"continue": True}]},
    {"name": "gated_after", "graph": "gated_after", "ops": [{"invoke": {"log": []}}, {"continue": True}]},
    {"name": "crash", "graph": "crash", "ops": [{"invoke": {"log": []}, "fail": ["enrich"]}, {"continue": True}]},
    {"name": "crash_twice", "graph": "crash",
     "ops": [{"invoke": {"log": []}, "fail": ["enrich"]}, {"continue": True, "fail": ["enrich"]}, {"continue": True}]},
    {"name": "counter_ok", "graph": "counter", "ops": [{"invoke": {"n": 1}, "limit": 5}]},
    {"name": "counter_edge", "graph": "counter", "ops": [{"invoke": {"n": 0}, "limit": 5}]},
    {"name": "counter_over", "graph": "counter", "ops": [{"invoke": {"n": -3}, "limit": 5}, {"continue": True, "limit": 3},
                                                       {"continue": True, "limit": 25}]},
    {"name": "replay", "graph": "branchy", "ops": [{"invoke": {"log": []}}, {"replay": 2}]},
    {"name": "fork", "graph": "branchy",
     "ops": [{"invoke": {"log": []}}, {"update": {"choice": "calc"}, "at": 2, "as_node": "plan"}, {"continue": True}]},
    {"name": "fork_other_node", "graph": "branchy",
     "ops": [{"invoke": {"log": []}}, {"update": {"cost": 5}, "at": 2, "as_node": "search"}, {"continue": True}]},
    {"name": "replay_update", "graph": "branchy",
     "ops": [{"invoke": {"log": []}}, {"update": {"choice": "calc"}, "at": 2, "as_node": "plan"}, {"replay": 5}]},
    {"name": "loop_join", "graph": "loop_join", "ops": [{"invoke": {"round": 0}}]},
    {"name": "start_router_else", "graph": "start_router", "ops": [{"invoke": {"kind": "bug"}}]},
    {"name": "loop_join_crash", "graph": "loop_join",
     "ops": [{"invoke": {"round": 0}, "fail": []}, {"replay": 3}, {"update": {"found": 10}, "at": 5, "as_node": "check"},
             {"continue": True}]},
    {"name": "after_update", "graph": "gated_after",
     "ops": [{"invoke": {"log": []}}, {"update": {"log": ["human"]}, "at": 2, "as_node": "a"}, {"continue": True}]},
    {"name": "replay_interrupt", "graph": "gated", "ops": [{"invoke": {"log": []}}, {"continue": True}, {"replay": 2}]},
]


def session(name: str) -> dict[str, Any]:
    for s in SESSIONS:
        if s["name"] == name:
            return copy.deepcopy(s)
    raise KeyError(name)
