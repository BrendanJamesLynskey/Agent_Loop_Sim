"""Multi-agent patterns as call plans on the scripted model (since 1.7).

Every pattern does the same task: answer a question that needs ``K`` sub-questions researched
(each one a search-tool call and a short sub-answer), then a final answer. A pattern is turned into
a **plan**: the model calls and tool calls it makes, which agent makes each, which calls each one
waits for, and every call's token counts from that agent's own context. Contexts are append-only,
so an agent's previous prompt is a prefix of its next one (the prompt-cache model of
``accounting``: a prefix of at least the price's ``min_tokens`` is read from the cache).

Patterns (each as commonly built with LangGraph; the topologies follow the LangGraph multi-agent
docs and the ``langgraph-supervisor`` / ``langgraph-swarm`` libraries' descriptions):
  - ``single``: one agent loops (tool call, result, …) and then answers.
  - ``supervisor``: a supervisor delegates each sub-question to a worker, one at a time; each
    worker starts from a fresh context, calls the tool and answers; the supervisor sees only the
    sub-answers and writes the final answer.
  - ``hierarchical``: a top supervisor delegates to two team leads, each a supervisor of its own
    workers; a lead returns a summary.
  - ``swarm``: specialists hand control to each other; the message history is shared, so each
    specialist reads everything said so far (under its own system prompt); the last one answers.
  - ``debate``: one researcher gathers the evidence; three debaters answer from it, then answer
    again after reading the others' answers; a judge takes the majority.
  - ``map_reduce``: a planner fans out one worker per sub-question in parallel (``Send``), and a
    reducer writes the answer from all sub-answers.

**Success model (illustrative).** A model call is correct with probability ``p``: ``p_route`` for
a routing or hand-off decision, otherwise ``p_focus`` for a prompt up to ``focus_tokens`` and
falling by ``slope_per_k`` per 1,000 tokens beyond (an illustrative stand-in for accuracy falling
with context length), floored at ``p_floor``. A plan succeeds if every critical call is correct
and, for debate, the majority of the final-round answers is correct. The debaters' first round is
charged but not credited (its benefit is not modelled). All token sizes are illustrative.
"""
from __future__ import annotations

from typing import Any

from ..accounting import LATENCY, PRICES, call_cost, call_latency

PATTERNS = ["single", "supervisor", "hierarchical", "swarm", "debate", "map_reduce"]
TITLES = {
    "single": "Single agent",
    "supervisor": "Supervisor",
    "hierarchical": "Hierarchical",
    "swarm": "Swarm (hand-offs)",
    "debate": "Debate",
    "map_reduce": "Map-reduce",
}

TASK: dict[str, Any] = {
    "question": "Which of four vendors should we pick?",
    "subtasks": ["pricing", "latency", "licensing", "support"],
    "user_tokens": 120,
    "system_tokens": 900,
    "tool_result_tokens": 700,
    "route_out": 60,
    "worker_out": 250,
    "final_out": 450,
    "handoff_tokens": 150,
    "tool_ms": 800,
    "debaters": 3,
}

MODEL: dict[str, Any] = {"p_focus": 0.98, "focus_tokens": 4000, "slope_per_k": 0.01, "p_route": 0.995, "p_floor": 0.5}


def call_p(kind: str, input_tokens: int, model: dict[str, Any]) -> float:
    """The probability a call of this kind is correct (illustrative)."""
    if kind == "route":
        return model["p_route"]
    over = input_tokens - model["focus_tokens"]
    p = model["p_focus"]
    if over > 0:
        p = p - model["slope_per_k"] * over / 1000
    return p if p > model["p_floor"] else model["p_floor"]


class _Plan:
    def __init__(self, task: dict[str, Any], model: dict[str, Any]) -> None:
        self.task = task
        self.model = model
        self.items: list[dict[str, Any]] = []
        self.ctx: dict[str, int] = {}
        self.prev: dict[str, int] = {}
        self.agents: list[str] = []

    def agent(self, name: str, base: int) -> None:
        if name not in self.agents:
            self.agents.append(name)
        self.ctx[name] = base
        self.prev[name] = 0

    def add(self, name: str, n: int) -> None:
        self.ctx[name] += n

    def call(self, agent: str, kind: str, out: int, deps: list[int], label: str, role: str = "critical",
             group: str = "") -> int:
        inp = self.ctx[agent]
        item = {"id": len(self.items), "agent": agent, "kind": kind, "label": label, "deps": list(deps),
                "input": inp, "prefix": self.prev[agent], "output": out, "p": call_p(kind, inp, self.model),
                "role": role, "group": group, "ms": 0}
        self.items.append(item)
        self.prev[agent] = inp
        self.ctx[agent] = inp + out
        return item["id"]

    def tool(self, agent: str, label: str, deps: list[int]) -> int:
        item = {"id": len(self.items), "agent": agent, "kind": "tool", "label": label, "deps": list(deps),
                "input": 0, "prefix": 0, "output": 0, "p": 1.0, "role": "free", "group": "",
                "ms": self.task["tool_ms"], "result": self.task["tool_result_tokens"]}
        self.items.append(item)
        self.ctx[agent] += self.task["tool_result_tokens"]
        return item["id"]


def plan(pattern: str, task: dict[str, Any] | None = None, model: dict[str, Any] | None = None) -> dict[str, Any]:
    """The call plan of a pattern on the task."""
    t = dict(TASK if task is None else task)
    m = dict(MODEL if model is None else model)
    P = _Plan(t, m)
    S, U, H = t["system_tokens"], t["user_tokens"], t["handoff_tokens"]
    RO, WO, FO = t["route_out"], t["worker_out"], t["final_out"]
    subs: list[str] = t["subtasks"]
    if pattern == "single":
        P.agent("agent", S + U)
        last: list[int] = []
        for s in subs:
            c = P.call("agent", "act", RO, last, "search " + s)
            last = [P.tool("agent", "search " + s, [c])]
        P.call("agent", "final", FO, last, "final answer")
    elif pattern == "supervisor":
        P.agent("supervisor", S + U)
        last = []
        for s in subs:
            d = P.call("supervisor", "route", RO, last, "delegate " + s)
            w = "worker:" + s
            P.agent(w, S + H)
            a = P.call(w, "act", RO, [d], "search " + s)
            tl = P.tool(w, "search " + s, [a])
            r = P.call(w, "work", WO, [tl], "answer " + s)
            P.add("supervisor", WO)
            last = [r]
        P.call("supervisor", "final", FO, last, "final answer")
    elif pattern == "hierarchical":
        P.agent("top", S + U)
        half = (len(subs) + 1) // 2
        teams = [subs[:half], subs[half:]]
        last = []
        for i, team in enumerate(teams):
            lead = "lead:" + ("ab"[i])
            d = P.call("top", "route", RO, last, "delegate team " + "ab"[i])
            P.agent(lead, S + H)
            inner = [d]
            for s in team:
                d2 = P.call(lead, "route", RO, inner, "delegate " + s)
                w = "worker:" + s
                P.agent(w, S + H)
                a = P.call(w, "act", RO, [d2], "search " + s)
                tl = P.tool(w, "search " + s, [a])
                r = P.call(w, "work", WO, [tl], "answer " + s)
                P.add(lead, WO)
                inner = [r]
            sm = P.call(lead, "work", WO, inner, "summarise team " + "ab"[i])
            P.add("top", WO)
            last = [sm]
        P.call("top", "final", FO, last, "final answer")
    elif pattern == "swarm":
        hist = U
        last = []
        for i, s in enumerate(subs):
            ag = "specialist:" + s
            if ag not in P.ctx:
                P.agent(ag, S + hist)
            else:
                P.ctx[ag] = S + hist
            a = P.call(ag, "act", RO, last, "search " + s)
            tl = P.tool(ag, "search " + s, [a])
            hist += RO + t["tool_result_tokens"]
            if i < len(subs) - 1:
                r = P.call(ag, "work", WO + RO, [tl], "answer " + s + ", hand off")
                hist += WO + RO
            else:
                r = P.call(ag, "final", FO, [tl], "final answer")
            last = [r]
    elif pattern == "debate":
        P.agent("researcher", S + U)
        last = []
        for s in subs:
            c = P.call("researcher", "act", RO, last, "search " + s)
            last = [P.tool("researcher", "search " + s, [c])]
        evidence = len(subs) * (RO + t["tool_result_tokens"])
        n = t["debaters"]
        first = []
        for k in range(n):
            ag = "debater:" + str(k + 1)
            P.agent(ag, S + U + evidence)
            first.append(P.call(ag, "work", WO, last, "round 1 answer", role="free"))
        second = []
        for k in range(n):
            ag = "debater:" + str(k + 1)
            P.add(ag, (n - 1) * WO)
            second.append(P.call(ag, "work", WO, first, "round 2 answer", role="vote", group="debate"))
        P.agent("judge", S + U + n * WO)
        P.call("judge", "final", FO, second, "judge: majority")
    elif pattern == "map_reduce":
        P.agent("planner", S + U)
        pl = P.call("planner", "route", RO * len(subs), [], "plan " + str(len(subs)) + " sends")
        outs = []
        for s in subs:
            w = "worker:" + s
            P.agent(w, S + H)
            a = P.call(w, "act", RO, [pl], "search " + s)
            tl = P.tool(w, "search " + s, [a])
            outs.append(P.call(w, "work", WO, [tl], "answer " + s))
        P.agent("reducer", S + U + len(subs) * WO)
        P.call("reducer", "final", FO, outs, "reduce: final answer")
    else:
        raise ValueError(f"unknown pattern {pattern}")
    return {"pattern": pattern, "title": TITLES[pattern], "agents": P.agents, "calls": P.items}


def plan_map(chunks: int, chunk_tokens: int = 1500, task: dict[str, Any] | None = None,
             model: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map-reduce over ``chunks`` document chunks (chapter 6): a split call, one summarise call per
    chunk (a ``Send`` each, input = system prompt + chunk), and a reduce call over every summary."""
    t = dict(TASK if task is None else task)
    m = dict(MODEL if model is None else model)
    P = _Plan(t, m)
    S, U, WO, FO, RO = t["system_tokens"], t["user_tokens"], t["worker_out"], t["final_out"], t["route_out"]
    P.agent("split", S + U)
    sp = P.call("split", "route", RO, [], "split into " + str(chunks))
    outs = []
    for i in range(chunks):
        w = "map:" + str(i + 1)
        P.agent(w, S + chunk_tokens)
        outs.append(P.call(w, "work", WO, [sp], "summarise chunk " + str(i + 1)))
    P.agent("reduce", S + U + chunks * WO)
    P.call("reduce", "final", FO, outs, "reduce " + str(chunks) + " summaries")
    return {"pattern": "map", "title": "Map-reduce over " + str(chunks) + " chunks", "agents": P.agents, "calls": P.items}


# ---------------------------------------------------------------------------------------------
# Accounting of a plan


def cached_tokens(call: dict[str, Any], price: dict[str, Any]) -> int:
    """The prompt prefix read from the cache: the agent's previous prompt, if long enough to be cached."""
    return call["prefix"] if call["prefix"] >= price["min_tokens"] and call["prefix"] > 0 else 0


def cost_of(call: dict[str, Any], price: dict[str, Any]) -> float:
    if call["kind"] == "tool":
        return 0.0
    return call_cost(price, call["input"], cached_tokens(call, price), call["input"] >= price["min_tokens"], call["output"])


def duration_of(call: dict[str, Any], price: dict[str, Any], profile: dict[str, Any]) -> float:
    if call["kind"] == "tool":
        return float(call["ms"])
    return call_latency(profile, call["input"], cached_tokens(call, price), call["output"])[1]


def maj3(a: float, b: float, c: float) -> float:
    """P(at least two of three independent events), probabilities a, b, c."""
    return a * b + a * c + b * c - 2 * a * b * c


def success_p(p: dict[str, Any]) -> float:
    """Closed form: the product of the critical calls' p, times the majority probability of each
    vote group (three voters)."""
    out = 1.0
    groups: dict[str, list[float]] = {}
    for c in p["calls"]:
        if c["role"] == "critical":
            out = out * c["p"]
        elif c["role"] == "vote":
            groups.setdefault(c["group"], []).append(c["p"])
    for g in groups.values():
        out = out * maj3(g[0], g[1], g[2])
    return out


def schedule(p: dict[str, Any], price: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Uncontended start and end of every call: each starts when its dependencies have finished."""
    ends: list[float] = []
    out = []
    for c in p["calls"]:
        st = 0.0
        for d in c["deps"]:
            if ends[d] > st:
                st = ends[d]
        en = st + duration_of(c, price, profile)
        ends.append(en)
        out.append({"id": c["id"], "start": st, "end": en})
    return out


def metrics(p: dict[str, Any], price_name: str = "claude-sonnet-4.6", profile_name: str = "hosted") -> dict[str, Any]:
    """Tokens (per agent and in total), cost, uncontended latency and the success probability."""
    price = PRICES[price_name]
    profile = LATENCY[profile_name]
    sched = schedule(p, price, profile)
    per: dict[str, dict[str, Any]] = {}
    for a in p["agents"]:
        per[a] = {"calls": 0, "input": 0, "cached": 0, "output": 0, "cost": 0.0, "peak": 0}
    tin = tout = tcache = calls = 0
    cost = 0.0
    serial = 0.0
    for c in p["calls"]:
        serial = serial + duration_of(c, price, profile)
        if c["kind"] == "tool":
            continue
        k = cached_tokens(c, price)
        cc = cost_of(c, price)
        a = per[c["agent"]]
        a["calls"] += 1
        a["input"] += c["input"]
        a["cached"] += k
        a["output"] += c["output"]
        a["cost"] = a["cost"] + cc
        if c["input"] > a["peak"]:
            a["peak"] = c["input"]
        tin += c["input"]
        tout += c["output"]
        tcache += k
        calls += 1
        cost = cost + cc
    latency = 0.0
    for s in sched:
        if s["end"] > latency:
            latency = s["end"]
    return {"pattern": p["pattern"], "title": p["title"], "model_calls": calls, "tool_calls": len(p["calls"]) - calls,
            "input": tin, "cached": tcache, "output": tout, "cost": cost, "latency_ms": latency, "serial_ms": serial,
            "success": success_p(p), "agents": per, "schedule": sched, "price": price_name, "profile": profile_name}


def compare(price_name: str = "claude-sonnet-4.6", profile_name: str = "hosted", model: dict[str, Any] | None = None,
            task: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every pattern's metrics on the same task."""
    return [metrics(plan(n, task, model), price_name, profile_name) for n in PATTERNS]


def pattern_frames(p: dict[str, Any], price_name: str = "claude-sonnet-4.6", profile_name: str = "hosted") -> list[dict[str, Any]]:
    """Frames for chapter 5's animation: one per call start or end (uncontended), with each agent's
    context so far, cumulative tokens and cost, and a caption."""
    price = PRICES[price_name]
    profile = LATENCY[profile_name]
    sched = schedule(p, price, profile)
    evs: list[tuple[float, int, int]] = []
    for s in sched:
        evs.append((s["start"], 1, s["id"]))
        evs.append((s["end"], 0, s["id"]))
    evs.sort(key=lambda e: (e[0], e[1], e[2]))
    ctx: dict[str, int] = {a: 0 for a in p["agents"]}
    active: list[int] = []
    done: list[int] = []
    tokens = 0
    cost = 0.0
    frames = [{"t": 0.0, "active": [], "done": [], "ctx": dict(ctx), "tokens": 0, "cost": 0.0,
               "caption": p["title"] + ": " + str(len(p["agents"])) + " agent" + ("" if len(p["agents"]) == 1 else "s")
               + ", " + str(len([c for c in p["calls"] if c["kind"] != "tool"])) + " model calls."}]
    for t, kind, cid in evs:
        c = p["calls"][cid]
        if kind == 1:
            active.append(cid)
            if c["kind"] == "tool":
                cap = c["agent"] + " runs the tool: " + c["label"] + "."
            else:
                ctx[c["agent"]] = c["input"]
                k = cached_tokens(c, price)
                cap = (c["agent"] + " calls the model (" + c["label"] + "): " + str(c["input"]) + " tokens in"
                       + (" (" + str(k) + " cached)" if k else "") + ".")
        else:
            active = [x for x in active if x != cid]
            done.append(cid)
            if c["kind"] == "tool":
                cap = c["agent"] + " gets " + str(c["result"]) + " tokens of results."
            else:
                ctx[c["agent"]] = c["input"] + c["output"]
                tokens += c["input"] + c["output"]
                cost = cost + cost_of(c, price)
                cap = c["agent"] + " answers: " + str(c["output"]) + " tokens out."
        frames.append({"t": t, "active": list(active), "done": list(done), "ctx": dict(ctx), "tokens": tokens,
                       "cost": cost, "caption": cap})
    return frames
