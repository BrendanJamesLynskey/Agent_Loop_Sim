"""Build a real LangGraph ``StateGraph`` from an engine graph spec, and run a session on it.

Used by ``scripts/record_langgraph.py`` (CI: the "LangGraph conformance" check). Each node is a
plain Python function that interprets the spec's operations independently of the engine, calling
LangGraph's own ``interrupt``; reducers are ``operator.add`` and a ``max`` function, exactly as a
LangGraph user would write them."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class NodeFailure(Exception):
    pass


def take_max(a: int, b: int) -> int:
    return b if b > a else a


PY = {"list": list, "int": int, "str": str}


def state_type(spec: dict[str, Any]) -> type:
    fields: dict[str, Any] = {}
    for s in spec["state"]:
        t = PY[s["type"]]
        if s["reducer"] == "overwrite":
            fields[s["key"]] = t
        elif s["reducer"] == "add":
            fields[s["key"]] = Annotated[t, operator.add]
        else:
            fields[s["key"]] = Annotated[t, take_max]
    return TypedDict("State", fields, total=False)  # type: ignore[operator]


def cond(c: dict[str, Any], state: dict[str, Any]) -> bool:
    a = len(state.get(c["len"], [])) if "len" in c else state.get(c["key"])
    return {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
            "==": operator.eq, "!=": operator.ne}[c["op"]](a, c["value"])


def router_fn(r: dict[str, Any]):
    def pick(router: dict[str, Any], state: dict[str, Any]) -> Any:
        g = router["goto"] if "goto" in router else (router["then"] if cond(router["if"], state) else router["else"])
        return pick(g, state) if isinstance(g, dict) else g
    return lambda state: pick(r, state)


class World:
    def __init__(self) -> None:
        self.fail: list[str] = []
        self.calls: list[str] = []
        self.effects: list[str] = []


def node_fn(node: dict[str, Any], world: World):
    name = node["name"]

    def fn(state: dict[str, Any]) -> dict[str, Any]:
        world.calls.append(name)
        out: dict[str, Any] = {}
        for op in node["ops"]:
            k = op["op"]
            if k == "set":
                out[op["key"]] = op["value"]
            elif k == "append":
                out[op["key"]] = [op["value"]]
            elif k == "delta":
                out[op["key"]] = op["value"]
            elif k == "inc":
                out[op["key"]] = state.get(op["key"], 0) + op["by"]
            elif k == "len":
                out[op["key"]] = len(state.get(op["of"], []))
            elif k == "copy":
                out[op["key"]] = state.get(op["from"])
            elif k == "effect":
                world.effects.append(name + ":" + op["name"])
            elif k == "fail":
                if name in world.fail:
                    raise NodeFailure(op.get("message", "node failed"))
            elif k == "interrupt":
                out[op["key"]] = interrupt(op["payload"])
            else:
                raise ValueError(k)
        return out

    return fn


def build(spec: dict[str, Any], world: World):
    g = StateGraph(state_type(spec))
    for n in spec["nodes"]:
        g.add_node(n["name"], node_fn(n, world))
    for e in spec["edges"]:
        src = START if e["from"] == "__start__" else e["from"]
        if "router" in e:
            g.add_conditional_edges(src, router_fn(e["router"]), [END if t == "__end__" else t for t in e["targets"]])
        else:
            g.add_edge(src, END if e["to"] == "__end__" else e["to"])
    return g.compile(checkpointer=InMemorySaver(), interrupt_before=spec.get("interrupt_before") or None,
                     interrupt_after=spec.get("interrupt_after") or None)


def history(app, cfg) -> list[dict[str, Any]]:
    snaps = list(reversed(list(app.get_state_history(cfg))))
    ids = {s.config["configurable"]["checkpoint_id"]: i for i, s in enumerate(snaps)}
    out = []
    for i, s in enumerate(snaps):
        parent = s.parent_config["configurable"]["checkpoint_id"] if s.parent_config else None
        out.append({"id": i, "parent": ids[parent] if parent is not None else None, "step": s.metadata["step"],
                    "source": s.metadata["source"], "values": dict(s.values), "next": list(s.next),
                    "tasks": [{"name": t.name, "error": str(t.error) if t.error else None,
                               "interrupts": [x.value for x in t.interrupts],
                               "result": dict(t.result) if t.result is not None else None} for t in s.tasks]})
    return out, snaps


def run_session(spec: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    world = World()
    app = build(spec, world)
    cfg = {"configurable": {"thread_id": "t"}}
    out = []
    for op in ops:
        world.fail = list(op.get("fail") or [])
        world.calls = []
        world.effects = []
        run_cfg = dict(cfg, recursion_limit=op.get("limit", 25))
        _, snaps = history(app, cfg)
        res: dict[str, Any] = {"status": "done", "interrupts": [], "error": None}
        try:
            if "update" in op:
                at = snaps[op["at"]].config
                app.update_state(at, op["update"], as_node=op["as_node"])
                res["status"] = "updated"
            else:
                if "invoke" in op:
                    arg: Any = op["invoke"]
                    c = run_cfg
                elif "resume" in op:
                    arg, c = Command(resume=op["resume"]), run_cfg
                elif "continue" in op:
                    arg, c = None, run_cfg
                else:
                    arg = None
                    c = dict(run_cfg, configurable=dict(snaps[op["replay"]].config["configurable"]))
                r = app.invoke(arg, c)
                if "__interrupt__" in r:
                    res["status"] = "interrupted"
                    res["interrupts"] = [x.value for x in r["__interrupt__"]]
                elif app.get_state(cfg).next:
                    res["status"] = "paused"
        except GraphRecursionError:
            res["status"] = "recursion_limit"
        except InvalidUpdateError:
            res["status"], res["error"] = "error", "InvalidUpdateError"
        except NodeFailure:
            res["status"], res["error"] = "error", "NodeFailure"
        res["calls"] = sorted(world.calls)
        res["effects"] = sorted(world.effects)
        out.append({"op": op, "outcome": res, "history": history(app, cfg)[0]})
    return {"name": spec["name"], "ops": out}
