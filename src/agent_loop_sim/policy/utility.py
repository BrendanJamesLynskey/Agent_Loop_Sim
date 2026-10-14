"""The utility-cost model and the security–utility frontier (since 1.11).

Every defence costs something. This module measures the cost on a fixed suite of benign tasks (``make_suite``,
seeded), and the security on the same tasks, so the two can be plotted against each other.

**Cost**, per task:

- **blocked**: the policy denies a call the task needs (or a human is not available to approve one), so the
  task does not finish; a detector layer can also flag a benign result by mistake (its false-positive rate),
  which stops the task with that probability;
- **prompts**: the calls a human must approve;
- **extra calls**: model calls the defence adds (a planner call; a classifier pass per untrusted result; an
  output check per write or send).

**Security**, per task: the chance that one steering attempt in the task's untrusted input ends in a
consequential call (a write or a send) that nothing stops. Threats appear only as this abstract event: the
attempt succeeds against an undefended model at the published base rate (InjecAgent, ``DEPTH["base"]``),
passes the detector layers with ``depth.all_miss_common``, and then needs a **route**:

- an *in-plan* route: a write or send the task makes anyway whose control is untrusted (in a classic agent,
  every call after an untrusted read; planned in advance, only a step that branches on untrusted data);
- an *off-plan* route (classic agent only): a send the task never planned, chosen after an untrusted read,
  at the point that is least restricted by the policy.

A route is closed if the policy denies it, needs a human (who approves it by mistake with the fatigue-adjusted
miss rate) or is open. The task's risk is base × P(all layers miss) × the best route's pass probability.

``adaptive``: an attacker who tunes the attempt against the detectors. Published adaptive attacks bypassed all
eight defences they tested with an attack success rate over 50% (arXiv 2503.00061), so in this setting every
detector layer is passed (P(all miss) = 1) and the base rate is Spotlighting's undefended > 50%
(``DEPTH["base_range"][1]``). The label policy and the plan never read content, so they are unchanged.
Nothing here holds content: a route is a tool, a destination and a label.
"""
from __future__ import annotations

from typing import Any

from ..rng import Rng
from . import capabilities as C
from . import depth as D
from . import engine as E
from .catalogue import BROAD, DEFENCES, DEPTH, SUITE, TOOLS, UTILITY

RANK = {"allow": 0, "ask": 1, "deny": 2}
RULES = ["not-granted", "scope", "destination", "untrusted-egress-private", "private-egress", "untrusted-write",
         "needs-input"]
WHAT = {"read_notes": "notes", "read_inbox": "inbox", "read_ticket": "a ticket", "search_docs": "docs",
        "fetch_page": "a web page"}


def make_suite(spec: dict[str, Any] = SUITE) -> list[dict[str, Any]]:
    """The benign task suite: each task reads one to three sources, summarises them and finishes at a sink."""
    r = Rng(spec["seed"])
    plans = []
    for k in range(spec["n"]):
        n_reads = r.randint(spec["min_reads"], spec["max_reads"])
        pool = list(spec["read_tools"])
        steps: list[dict[str, Any]] = [{"op": "input", "out": "q", "sens": "public"}]
        outs: list[str] = []
        reads: list[str] = []
        for j in range(n_reads):
            t = pool.pop(r.randint(0, len(pool) - 1))
            st: dict[str, Any] = {"op": "call", "tool": t, "args": ["q"], "out": f"r{j + 1}"}
            if TOOLS[t]["egress"]:
                st["dest"] = TOOLS[t]["dests"][0]
            steps.append(st)
            outs.append(f"r{j + 1}")
            reads.append(t)
        steps.append({"op": "call", "tool": "summarise", "args": list(outs), "out": "s"})
        sink = spec["sinks"][r.randint(0, len(spec["sinks"]) - 1)]
        adaptive = r.random() < spec["adaptive"]
        if sink["tool"] is None:
            adaptive = False
        else:
            st = {"op": "call", "tool": sink["tool"], "args": ["s"], "out": None}
            if sink["dest"] is not None:
                st["dest"] = sink["dest"]
            if adaptive:
                st["ctrl"] = [outs[-1]]
            steps.append(st)
        title = "Read " + ", ".join(WHAT[t] for t in reads) + "; " + sink["text"] + \
            (", as the last read decides" if adaptive else "")
        plans.append({"id": f"t{k + 1:03d}", "title": title, "steps": steps, "adaptive": adaptive})
    return plans


def approval_miss(n: int, cfg: dict[str, Any] = UTILITY) -> float:
    """The chance a human approves a call they should refuse, at the n-th prompt of a task (n ≥ 1): the first
    miss rate, with each further prompt keeping only ``approval_decay`` of the remaining attention."""
    keep = 1.0 - cfg["approval_first_miss"]
    for _ in range(n - 1):
        keep = keep * cfg["approval_decay"]
    return 1.0 - keep


def consequential(tool: str) -> bool:
    return TOOLS[tool]["kind"] in ("write", "send")


def grant_of(plan: dict[str, Any], d: dict[str, Any]) -> dict[str, Any]:
    if d["grant"] == "broad":
        return BROAD
    if not d["policy"]:
        raise ValueError("least privilege is modelled with the policy on")
    return C.needed(plan)


def layer_miss(ids: list[str]) -> list[float]:
    by = {x["id"]: x["miss"] for x in DEPTH["layers"]}
    return [by[i] for i in ids]


def _human(d: dict[str, Any], tool: str, decision: str) -> bool:
    return (d["policy"] and decision == "ask") or d["approval"] == "every" or \
        (d["approval"] == "consequential" and consequential(tool))


def evaluate(plan: dict[str, Any], d: dict[str, Any], base: float = DEPTH["base"],
             cfg: dict[str, Any] = UTILITY, adaptive: bool = False) -> dict[str, Any]:
    """One task under one defence: its cost and its risk."""
    grant = grant_of(plan, d)
    steps = E.run_plan(plan, grant, d["mode"], d["approve"], d["policy"])
    calls = [s for s in steps if s["op"] == "call"]
    blocked = [s["tool"] for s in calls if not s["ran"]]
    by_rule: list[str] = []
    for s in calls:
        if not s["ran"]:
            for f in (s["fired"] if s["fired"] else [{"rule": "needs-input"}]):
                if f["rule"] not in by_rule:
                    by_rule.append(f["rule"])
    by_rule = [x for x in RULES if x in by_rule]
    untrusted = 0
    cons = 0
    prompts = 0
    human: dict[int, bool] = {}
    for s in calls:
        if not s["ran"]:
            continue
        t = TOOLS[s["tool"]]
        if t["kind"] == "read" and t["out"]["src"] == "untrusted":
            untrusted += 1
        if consequential(s["tool"]):
            cons += 1
        h = _human(d, s["tool"], s["decision"])
        human[s["i"]] = h
        if h:
            prompts += 1
    miss = approval_miss(prompts, cfg) if prompts > 0 else 0.0
    routes: list[dict[str, Any]] = []
    for s in calls:
        if s["ran"] and consequential(s["tool"]) and s["control"]["src"] == "untrusted":
            routes.append({"kind": "in-plan", "step": s["i"], "tool": s["tool"], "dest": s["dest"],
                           "decision": s["decision"], "human": human[s["i"]],
                           "pass": miss if human[s["i"]] else 1.0})
    if d["mode"] == "context":
        best: dict[str, Any] | None = None
        for s in steps:
            if s["op"] == "call" and s["ran"] and s["ctx"]["src"] == "untrusted":
                dec = E.decide("send_message", s["ctx"], "web", grant)["decision"] if d["policy"] else "allow"
                if best is None or RANK[dec] < RANK[best["decision"]]:
                    best = {"kind": "off-plan", "step": s["i"], "tool": "send_message", "dest": "web",
                            "decision": dec, "label": dict(s["ctx"])}
        if best is not None:
            closed = best["decision"] == "deny" or (best["decision"] == "ask" and not d["approve"])
            h = not closed and _human(d, "send_message", best["decision"])
            best["human"] = h
            best["pass"] = 0.0 if closed else (approval_miss(prompts + 1, cfg) if h else 1.0)
            routes.append(best)
    reach = 0.0
    for x in routes:
        if x["pass"] > reach:
            reach = x["pass"]
    m = layer_miss(d["layers"])
    det = D.all_miss_common(m, d["c"]) if m and not adaptive else 1.0
    risk = base * det * reach
    keep = 0.0 if blocked else 1.0
    extra = cfg["planner_calls"] if d["mode"] == "dataflow" else 0
    for lid in d["layers"]:
        n = cons if lid == "output_check" else untrusted
        if lid != "marking":
            extra += n
        for _ in range(n):
            keep = keep * (1.0 - cfg["fp"][lid])
    return {"task": plan.get("id"), "blocked": blocked, "by_rule": by_rule, "keep": keep, "prompts": prompts, "miss": miss,
            "extra": extra, "untrusted": untrusted, "consequential": cons, "routes": routes, "reach": reach,
            "detect": det, "risk": risk, "decisions": [s["decision"] for s in calls]}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over the suite (plain left-to-right sums, so both languages agree exactly)."""
    n = len(rows)
    s = {"utility": 0.0, "blocked": 0.0, "prompts": 0.0, "extra": 0.0, "risk": 0.0, "exposed": 0.0,
         "ungated": 0.0, "with_untrusted": 0.0}
    for r in rows:
        s["utility"] = s["utility"] + r["keep"]
        s["blocked"] = s["blocked"] + (1.0 if r["blocked"] else 0.0)
        s["prompts"] = s["prompts"] + r["prompts"]
        s["extra"] = s["extra"] + r["extra"]
        s["risk"] = s["risk"] + r["risk"]
        s["exposed"] = s["exposed"] + (1.0 if r["reach"] > 0.0 else 0.0)
        s["ungated"] = s["ungated"] + (1.0 if r["reach"] == 1.0 else 0.0)
        s["with_untrusted"] = s["with_untrusted"] + (1.0 if r["untrusted"] > 0 else 0.0)
    out: dict[str, Any] = {k: v / n for k, v in s.items()}
    out["n"] = n
    out["by_rule"] = {x: sum(1 for r in rows if x in r["by_rule"]) for x in RULES}
    return out


def run_defence(d: dict[str, Any], suite: list[dict[str, Any]], base: float = DEPTH["base"],
                cfg: dict[str, Any] = UTILITY, adaptive: bool = False) -> dict[str, Any]:
    rows = [evaluate(p, d, base, cfg, adaptive) for p in suite]
    return {"id": d["id"], "name": d["name"], "rows": rows, "summary": summarise(rows)}


def pareto(points: list[dict[str, Any]]) -> list[str]:
    """The ids no other point beats on all three costs (at least as useful, at most as risky, at most as many
    prompts per task), and strictly on one."""
    front = []
    for a in points:
        beaten = False
        for b in points:
            if b["id"] == a["id"]:
                continue
            if b["utility"] >= a["utility"] and b["risk"] <= a["risk"] and b["prompts"] <= a["prompts"] and \
                    (b["utility"] > a["utility"] or b["risk"] < a["risk"] or b["prompts"] < a["prompts"]):
                beaten = True
        if not beaten:
            front.append(a["id"])
    return front


def frontier(adaptive: bool = False, suite: list[dict[str, Any]] | None = None,
             cfg: dict[str, Any] = UTILITY) -> dict[str, Any]:
    """Every defence on the suite. Fixed attacks: the base rate is InjecAgent's; adaptive: see the module."""
    tasks = suite if suite is not None else make_suite()
    base = DEPTH["base_range"][1] if adaptive else DEPTH["base"]
    runs = [run_defence(d, tasks, base, cfg, adaptive) for d in DEFENCES]
    points = [dict(r["summary"], id=r["id"], name=r["name"]) for r in runs]
    return {"adaptive": adaptive, "base": base, "points": points, "front": pareto(points)}


def simulate(rows: list[dict[str, Any]], d: dict[str, Any], base: float, reps: int, seed: int,
             adaptive: bool = False) -> dict[str, Any]:
    """Draw every task ``reps`` times with the shared seeded generator: the attempt (base rate), the shared blind
    spot and each layer, the best route; and whether the task completes (4 + layers draws per task, always)."""
    m = layer_miss(d["layers"])
    q = D.q_given_common(m, d["c"]) if m and not adaptive else []
    r = Rng(seed)
    harmed = 0
    done = 0
    n = 0
    for _ in range(reps):
        for row in rows:
            n += 1
            steered = r.random() < base
            common = r.random() < d["c"]
            missed = True
            for i in range(len(q)):
                u = r.random()
                if not (common or u < q[i]):
                    missed = False
            route = r.random() < row["reach"]
            if steered and missed and route:
                harmed += 1
            if r.random() < row["keep"]:
                done += 1
    return {"n": n, "harmed": harmed, "done": done, "risk": harmed / n, "utility": done / n}


def defence(did: str) -> dict[str, Any]:
    for d in DEFENCES:
        if d["id"] == did:
            return d
    raise KeyError(did)


def with_threshold(d: dict[str, Any], th: str) -> dict[str, Any]:
    """Chapter 8's thresholds over a defence: refuse the policy's asks (no human), send them to a human, or also
    every write and send, or every call."""
    out = dict(d)
    out["approve"] = th != "refuse"
    out["approval"] = {"refuse": "none", "policy": "none", "consequential": "consequential", "every": "every"}[th]
    out["id"] = d["id"] + ":" + th
    return out
