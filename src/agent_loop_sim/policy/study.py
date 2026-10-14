"""What the Security site's chapters animate (since 1.10). Each study runs the policy module and returns frames,
each with a caption. Labels only: no frame holds content.

- ``boundary_frames`` (chapter 1): a plan's values crossing the boundaries between the user, the model, the
  tools and the outside world, each labelled, with the decision at each crossing.
- ``context_frames`` (chapter 2): the same plan in a classic agent (the context controls every call) and
  planned in advance (only the plan's arguments carry labels), step by step.
- ``trifecta_study`` (chapter 3): every combination of the three circles, whether a flow out exists, and what
  the policy decides on it.
- ``privilege_study`` (chapter 4): one task under a broad grant and under its least privilege, with a call
  outside the task, and the tokens each grant mints.
- ``sandbox_frames`` (chapter 5): sandbox layers added one at a time.
- ``depth_study`` / ``depth_frames`` (chapter 6): layered detectors, independent and correlated (closed forms,
  a simulation, and attempts drawn one by one through the layers).
- ``dual_frames`` (chapter 7, since 1.11): plan-then-execute: who handles each step, what the planner sees, and
  the policy's decision, beside the classic agent's.
- ``approval_study`` (chapter 8): approval thresholds over the benign suite: prompts, gated routes, fatigue.
- ``frontier_study`` (chapter 9): every defence's cost and risk, fixed and adaptive attacks.
- ``checklist`` (chapter 10): a design audited against chapters 1 to 9.
"""
from __future__ import annotations

import copy
from typing import Any

from ..context.text import fixed
from ..rng import Rng
from . import capabilities as C
from . import depth as D
from . import engine as E
from . import labels as L
from . import sandbox as S
from . import utility as U
from .catalogue import (ACTIONS, ADAPTIVE, BROAD, CIRCLE_TOOLS, DEFENCES, DEPTH, DEPTH_DOTS, DEST_ORDER,
                        DUAL_PLANS, LAYERS, PLANS, PRIVILEGE, TOOL_ORDER, TOOLS, UTILITY)


def _zones(tool: str | None, op: str) -> tuple[str, str]:
    if op == "input":
        return "user", "model"
    t = TOOLS[tool]  # type: ignore[index]
    if t["kind"] == "transform":
        return "model", "model"
    if t["egress"]:
        return "model", "outside"
    return "model", "tools"


VERB = {"allow": "allowed", "ask": "sent for approval", "deny": "denied", "skip": "skipped"}
NOUN = {"private": "private data", "untrusted": "untrusted input", "outbound": "outbound channel"}


def _lab(x: L.Label | None) -> str:
    return "(" + L.text(x) + ")" if x is not None else "(none)"


# ── chapter 1 ────────────────────────────────────────────────────────────


def boundary_frames(plan_id: str, mode: str = "context", grant: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    plan = PLANS[plan_id]
    steps = E.run_plan(plan, grant if grant is not None else BROAD, mode)
    frames: list[dict[str, Any]] = []
    known: dict[str, Any] = {}
    for r in steps:
        n = r["i"] + 1
        if r["op"] == "input":
            known[r["out"]] = r["out_label"]
            frames.append({"step": r["i"], "from": "user", "to": "model", "value": r["out"], "label": r["out_label"],
                           "decision": "allow", "values": dict(known), "ctx": r["ctx"],
                           "caption": f"Step {n}: the user's request {r['out']} enters the model, labelled "
                                      f"{_lab(r['out_label'])}."})
            continue
        a, b = _zones(r["tool"], r["op"])
        t = TOOLS[r["tool"]]
        if r["decision"] == "skip":
            frames.append({"step": r["i"], "from": a, "to": b, "value": None, "label": None, "decision": "skip",
                           "values": dict(known), "ctx": L.join_all(list(known.values())),
                           "caption": f"Step {n}: {r['tool']} is skipped: an input it needs was never produced."})
            continue
        where = {"tools": "a tool", "outside": "the outside world (" + str(r["dest"]) + ")", "model": "the model"}[b]
        # the call is made before its result returns: the context is what the model holds now
        frames.append({"step": r["i"], "from": a, "to": b, "value": ",".join(r["args"]), "label": r["effective"],
                       "decision": r["decision"], "values": dict(known), "ctx": L.join_all(list(known.values())),
                       "caption": f"Step {n}: {r['tool']} is called; its arguments and control carry "
                                  f"{_lab(r['effective'])} to {where}. {r['why'][0].upper()}{r['why'][1:]}."})
        if r["ran"] and r["out_label"] is not None:
            known[r["out"]] = r["out_label"]
            src = "model" if t["kind"] == "transform" else b
            frames.append({"step": r["i"], "from": src, "to": "model", "value": r["out"], "label": r["out_label"],
                           "decision": "allow", "values": dict(known), "ctx": r["ctx"],
                           "caption": f"Step {n}: {r['out']} comes back to the model labelled {_lab(r['out_label'])}"
                                      + (f": {t['text']}." if t["kind"] == "read" else ", the join of its inputs.")})
    return frames


# ── chapter 2 ────────────────────────────────────────────────────────────


def context_frames(plan_id: str, grant: dict[str, Any] | None = None) -> dict[str, Any]:
    g = grant if grant is not None else BROAD
    plan = PLANS[plan_id]
    a = E.run_plan(plan, g, "context")
    b = E.run_plan(plan, g, "dataflow")
    frames = []
    for x, y in zip(a, b):
        n = x["i"] + 1
        if x["op"] == "input":
            cap = f"Step {n}: the request is the only thing either agent has seen."
        else:
            cap = (f"Step {n}, {x['tool']}: in the classic agent its control is {_lab(x['control'])} and it is "
                   f"{VERB[x['decision']]}; planned in advance, its control is {_lab(y['control'])} and it is "
                   f"{VERB[y['decision']]}.")
        frames.append({"step": x["i"], "context": x, "dataflow": y, "caption": cap})
    return {"plan": plan_id, "frames": frames, "summary": {"context": E.summary(a), "dataflow": E.summary(b)}}


# ── chapter 3 ────────────────────────────────────────────────────────────


def grant_for(tools: list[str], egress: list[str] | None = None) -> dict[str, Any]:
    """A grant holding just these tools (and their scopes); egress to every destination, private data
    readable only by the user."""
    ts = [t for t in TOOL_ORDER if t in tools]
    return {"tools": ts, "scopes": sorted(TOOLS[t]["scope"] for t in ts if TOOLS[t]["scope"]),
            "egress": list(egress if egress is not None else DEST_ORDER), "private_dests": ["self"]}


def exposure_plan(path: list[str], dest: str | None) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [{"op": "input", "out": "q", "sens": "public"}]
    outs = ["q"]
    for i, t in enumerate(path):
        if i < len(path) - 1:
            o = f"v{i + 1}"
            st: dict[str, Any] = {"op": "call", "tool": t, "args": ["q"], "out": o}
            if TOOLS[t]["egress"]:
                st["dest"] = TOOLS[t]["dests"][0]
            steps.append(st)
            outs.append(o)
        else:
            steps.append({"op": "call", "tool": t, "args": list(outs), "dest": dest, "out": None})
    return {"title": "a flow out", "steps": steps}


def trifecta_case(name: str, tools: list[str]) -> dict[str, Any]:
    g = grant_for(tools + ["summarise"])
    tri = C.trifecta(g)
    ex = C.exposure(g)
    case: dict[str, Any] = {"name": name, "grant": g, "trifecta": tri, "exposure": ex, "steps": None}
    if ex["exposed"]:
        steps = E.run_plan(exposure_plan(ex["path"], ex["dest"]), g, "context")
        case["steps"] = steps
        last = steps[-1]
        case["caption"] = (f"All three circles: {' → '.join(ex['path'])} to {ex['dest']} would carry private data "
                           f"out under untrusted control. The policy's decision: {last['decision']} "
                           f"({', '.join(f['rule'] for f in last['fired'])}).")
    else:
        miss = [NOUN[k] for k in ["private", "untrusted", "outbound"] if not tri[k]]
        case["caption"] = (f"No {', no '.join(miss)}: no sequence of calls this agent holds can carry private "
                           f"data out under untrusted control ({ex['checked']} sequences checked).")
    return case


def trifecta_study() -> dict[str, Any]:
    cases = []
    for mask in range(8):
        on = [k for i, k in enumerate(["private", "untrusted", "outbound"]) if mask >> i & 1]
        tools: list[str] = []
        for k in on:
            tools += CIRCLE_TOOLS[k]
        cases.append(dict(trifecta_case("+".join(on) if on else "none", tools), mask=mask, on=on))
    double = [trifecta_case("read_inbox+send_message", ["read_inbox", "send_message"]),
              trifecta_case("read_notes+fetch_page", ["read_notes", "fetch_page"])]
    return {"circles": cases, "double": double}


# ── chapter 4 ────────────────────────────────────────────────────────────


def privilege_study() -> dict[str, Any]:
    base = PLANS[PRIVILEGE["plan"]]
    plan = {"title": base["title"] + ", and one call it does not need",
            "steps": copy.deepcopy(base["steps"]) + [copy.deepcopy(PRIVILEGE["extra"])]}
    grants = {"broad": BROAD, "task": C.needed(base)}
    runs: dict[str, Any] = {}
    for gid, g in grants.items():
        toks = C.mint(g)
        steps = E.run_plan(plan, g, "context", approve=False)
        for r in steps:
            if r["op"] == "call":
                r["token"] = C.check_token(toks.get(TOOLS[r["tool"]]["aud"] or ""), r["tool"])
        runs[gid] = {"grant": g, "tokens": toks, "steps": steps, "over": C.over_grant(g, base)}
    frames = []
    for i in range(len(plan["steps"])):
        x = runs["broad"]["steps"][i]
        y = runs["task"]["steps"][i]
        n = i + 1
        if x["op"] == "input":
            cap = f"Step {n}: the request. The broad grant holds {len(BROAD['tools'])} tools; the task needs " \
                  f"{len(grants['task']['tools'])}."
        else:
            cap = f"Step {n}, {x['tool']}: broad grant: {x['why']}; least privilege: {y['why']}."
        frames.append({"step": i, "broad": x, "task": y, "caption": cap})
    task_toks = runs["task"]["tokens"]
    first = task_toks[TOOLS[base["steps"][1]["tool"]]["aud"]]
    passthrough = {"token": first, "tool": "send_message", "check": C.check_token(first, "send_message")}
    return {"plan": plan, "runs": runs, "frames": frames, "passthrough": passthrough}


# ── chapter 5 ────────────────────────────────────────────────────────────


def sandbox_frames() -> list[dict[str, Any]]:
    frames = []
    for k in range(len(LAYERS) + 1):
        active = [x["id"] for x in LAYERS[:k]]
        rows = S.matrix(active)
        t = S.tally(rows)
        if k == 0:
            cap = (f"No sandbox: every action runs. The task's {t['works']} actions work, and so do all "
                   f"{t['open']} it does not need.")
        else:
            lay = LAYERS[k - 1]
            new = [r["id"] for r, p in zip(rows, S.matrix(active[:-1])) if r["stopped"] and not p["stopped"]]
            cap = (f"+ {lay['name']}: {lay['text']}. It newly stops {len(new)} action"
                   f"{'' if len(new) == 1 else 's'}; {t['open']} unneeded action{'' if t['open'] == 1 else 's'} "
                   f"still open, {t['broken']} needed action{'' if t['broken'] == 1 else 's'} broken.")
        frames.append({"layers": active, "rows": rows, "tally": t, "caption": cap})
    return frames


def sandbox_action_frames(active: list[str]) -> list[dict[str, Any]]:
    """One frame per action under the active layers: who stops it, and a caption."""
    names = {x["id"]: x["name"] for x in LAYERS}
    frames = []
    for a, r in zip(ACTIONS, S.matrix(active)):
        who = " and the ".join(names[b["layer"]].lower() for b in r["by"])
        why = r["by"][0]["reason"] if r["by"] else ""
        text = a["text"][0].upper() + a["text"][1:]
        if r["outcome"] == "stopped":
            cap = f"{text}: not needed, and stopped by the {who} ({why})."
        elif r["outcome"] == "broken":
            cap = f"{text}: the task needs it, but the {who} stops it ({why}), so the task breaks."
        elif r["outcome"] == "works":
            cap = f"{text}: the task needs it, and no active layer stops it."
        elif a["kind"] == "net" and "egress" in active:
            cap = (f"{text}: not needed, and nothing stops it: the allow-list checks where a connection goes, "
                   f"not what it carries.")
        else:
            cap = f"{text}: not needed, and no active layer stops it."
        frames.append(dict(r, text=a["text"], kind=a["kind"], caption=cap))
    return frames


def all_layer_sets() -> list[dict[str, Any]]:
    out = []
    for mask in range(1 << len(LAYERS)):
        active = [x["id"] for i, x in enumerate(LAYERS) if mask >> i & 1]
        rows = S.matrix(active)
        out.append({"layers": active, "rows": rows, "tally": S.tally(rows), "frames": sandbox_action_frames(active)})
    return out


# ── defence in depth (chapter 6) ─────────────────────────────────────────


def depth_study(cfg: dict[str, Any] = DEPTH) -> dict[str, Any]:
    m = [x["miss"] for x in cfg["layers"]]
    sims = [dict(D.simulate(m, c, cfg["trials"], cfg["seed"] + i), c=c) for i, c in enumerate(cfg["c"])]
    corr = [{"c": c, "pairs": [[D.pair_correlation(m[i], m[j], c) for j in range(len(m))] for i in range(len(m))]}
            for c in cfg["c"]]
    return {"table": D.depth_table(cfg["base"], m, cfg["c"], cfg["attempts"]),
            "independent": D.all_miss_independent(m), "sims": sims, "corr": corr}


def depth_frames(c: float, k: int | None = None, cfg: dict[str, Any] = DEPTH,
                 dots: dict[str, Any] = DEPTH_DOTS) -> dict[str, Any]:
    """Attempts drawn one by one: each steers an undefended model at the base rate, then must get past the first
    k layers (common-mode mixture with shared blind spot c). ``stage`` is how far each attempt gets: 0 does not
    steer the model, 1 steers it, 1 + j gets past j layers. Frame f shows the attempts at stage ≥ f, beside
    the closed form."""
    lay = cfg["layers"] if k is None else cfg["layers"][:k]
    m = [x["miss"] for x in lay]
    q = D.q_given_common(m, c)
    r = Rng(dots["seed"])
    stages = []
    for _ in range(dots["n"]):
        steer = r.random() < cfg["base"]
        common = r.random() < c
        st = 1 if steer else 0
        alive = steer
        for i in range(len(q)):
            u = r.random()
            alive = alive and (common or u < q[i])
            if alive:
                st = st + 1
        stages.append(st)
    frames = []
    n = dots["n"]
    for f in range(len(m) + 2):
        count = 0
        for x in stages:
            if x >= f:
                count += 1
        if f == 0:
            exp = float(n)
            cap = f"{n} attempts arrive in untrusted results."
        elif f == 1:
            exp = n * cfg["base"]
            cap = (f"With no defence, {count} of them steer the model (expected {fixed(exp, 1)}: the base rate "
                   f"{pct(cfg['base'], 0)}).")
        else:
            exp = n * cfg["base"] * D.all_miss_common(m[:f - 1], c)
            cap = (f"+ {lay[f - 2]['name']} (misses {pct(m[f - 2], 0)}): {count} still get through (expected "
                   f"{fixed(exp, 2)}).")
        frames.append({"f": f, "count": count, "expected": exp, "caption": cap})
    last = frames[-1]
    last["caption"] = last["caption"] + (f" A shared blind spot of {pct(c)} sets a floor of {fixed(n * cfg['base'] * c, 2)}"
                                         f" that no extra layer removes." if c > 0 else
                                         " Every layer misses some, so the product never reaches zero.")
    return {"c": c, "n": n, "stages": stages, "frames": frames}


# ── chapter 7 ────────────────────────────────────────────────────────────


def dual_plan(pid: str) -> dict[str, Any]:
    return ADAPTIVE if pid == "adaptive" else PLANS[pid]


def dual_frames(pid: str) -> dict[str, Any]:
    """Plan-then-execute with least privilege: the planner writes the plan from the request alone and sees only
    variable names; reads run as tools, transforms of untrusted values in a quarantined model, and the policy
    checks each call on the labels the interpreter tracks. The classic agent's decision is alongside."""
    plan = dual_plan(pid)
    g = C.needed(plan)
    a = E.run_plan(plan, g, "dataflow")
    b = E.run_plan(plan, g, "context")
    frames = []
    seen: list[str] = []
    for x, y in zip(a, b):
        n = x["i"] + 1
        if x["op"] == "input":
            calls = [st["tool"] for st in plan["steps"] if st["op"] == "call"]
            frames.append({"step": x["i"], "actor": "planner", "flow": x, "classic": y, "planner_sees": ["q"],
                           "caption": f"Step {n}: the planner reads only the request and writes the whole plan: "
                                      f"{' → '.join(calls)}. It will never read a tool's result."})
            seen.append("q")
            continue
        t = TOOLS[x["tool"]]
        if t["kind"] == "read":
            actor = "tool"
        elif t["kind"] == "transform":
            actor = "quarantined"
        else:
            actor = "tool"
        if x["ran"] and x["out"]:
            seen.append(x["out"])
        if x["decision"] == "skip":
            what = "is skipped: an input it needs was never produced"
        else:
            what = f"is {VERB[x['decision']]} on ({L.text(x['effective'])})"
        how = {"tool": "runs as a tool", "quarantined": "runs in the quarantined model, which has no tools"}[actor]
        ctrl = ""
        if "ctrl" in x:
            ctrl = (f" The plan branches on {', '.join(x['ctrl'])}, so the call's control is "
                    f"{_lab(x['control'])}, tracked rather than hidden.")
        frames.append({"step": x["i"], "actor": actor, "flow": x, "classic": y, "planner_sees": list(seen),
                       "caption": f"Step {n}, {x['tool']} {how} and {what}; the classic agent's call is "
                                  f"{VERB[y['decision']]}.{ctrl}"})
    return {"plan": pid, "title": plan["title"], "frames": frames, "summary": {"dataflow": E.summary(a),
                                                                               "context": E.summary(b)},
            "extra_calls": UTILITY["planner_calls"]}


# ── chapter 8 ────────────────────────────────────────────────────────────


def _route_text(r: dict[str, Any]) -> str:
    if not r["routes"]:
        return "; no steered attempt has a route."
    if r["reach"] == 0.0:
        return "; the policy closes every route a steered attempt could take."
    return f"; a steered attempt's best route passes with {pct(r['reach'], 0)}."


def approval_frames(th: str, suite: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The label policy (classic agent, broad grant) under one approval threshold, task by task over the first
    tasks of the suite, with running totals; and the whole suite's summary."""
    tasks = suite if suite is not None else U.make_suite()
    d = U.with_threshold(U.defence("policy"), th)
    rows = [U.evaluate(p, d) for p in tasks]
    frames = []
    prompts = 0
    gated = 0
    risky = 0
    for p, r in zip(tasks[:UTILITY["frames"]], rows):
        prompts += r["prompts"]
        if r["routes"]:
            risky += 1
            if r["reach"] < 1.0:
                gated += 1
        calls = [st for st in p["steps"] if st["op"] == "call"]
        steps = []
        for st, dec in zip(calls, r["decisions"]):
            steps.append({"tool": st["tool"], "dest": st.get("dest"), "decision": dec})
        if r["blocked"]:
            how = f"blocked at {r['blocked'][0]} ({', '.join(r['by_rule'])})"
        else:
            how = "completes"
        cap = (f"{p['id']}: {p['title']}. It {how}, with {r['prompts']} prompt{'' if r['prompts'] == 1 else 's'}"
               + (f" (a mistaken approval at the last: {pct(r['miss'], 0)})" if r["prompts"] > 0 else "")
               + _route_text(r))
        frames.append({"task": p["id"], "title": p["title"], "steps": steps, "row": r, "prompts": prompts,
                       "risky": risky, "gated": gated, "caption": cap})
    return {"threshold": th, "frames": frames, "summary": U.summarise(rows)}


def approval_study(suite: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    tasks = suite if suite is not None else U.make_suite()
    curve = [{"n": n, "miss": U.approval_miss(n)} for n in range(1, 11)]
    return {"thresholds": {th: approval_frames(th, tasks) for th in UTILITY["thresholds"]}, "fatigue": curve}


# ── chapter 9 ────────────────────────────────────────────────────────────


def frontier_study(suite: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Both attacker settings, and one frame per defence in order, with a caption."""
    tasks = suite if suite is not None else U.make_suite()
    out: dict[str, Any] = {"n": len(tasks), "adaptive_share": 0.0}
    k = 0
    for p in tasks:
        if p["adaptive"]:
            k += 1
    out["adaptive_share"] = k / len(tasks)
    for key, adaptive in [("fixed", False), ("adaptive", True)]:
        fr = U.frontier(adaptive, tasks)
        frames = []
        for i, pt in enumerate(fr["points"]):
            on = "on the frontier" if pt["id"] in fr["front"] else "beaten by another point"
            frames.append({"i": i, "id": pt["id"], "caption":
                           f"{pt['name']}: finishes {pct(pt['utility'])} of the benign tasks with "
                           f"{fixed(pt['prompts'], 2)} prompts and {fixed(pt['extra'], 2)} extra calls per task; "
                           f"a steered attempt succeeds in {pct(pt['risk'], 2)} of tasks ({on})."})
        out[key] = dict(fr, frames=frames)
    d = U.defence("planned")
    rows = [U.evaluate(p, d) for p in tasks]
    out["sim"] = U.simulate(rows, d, DEPTH["base"], UTILITY["reps"], UTILITY["seed"])
    return out


# ── chapter 10 ───────────────────────────────────────────────────────────

def _detector_value(n: int, policy: bool) -> str:
    layers = f"{n} detector layer{'' if n == 1 else 's'}"
    if policy:
        return layers + ", backed by the policy" if n else "no detectors needed: the policy decides on labels"
    return layers + " and nothing structural behind them" if n else "no detectors and no policy"


CHECK_DEFAULT: dict[str, Any] = {"defence": "planned", "approval": None, "sandbox": ["fs_scope", "container", "egress"]}


def checklist(design: dict[str, Any] = CHECK_DEFAULT, suite: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Audit a design (one of the frontier's defences, optionally with a stricter approval rule, plus sandbox
    layers) against chapters 1 to 9, on the benign suite. Each item: pass, warn or fail, the measured value,
    and the chapter it comes from. Risk is judged against an adaptive attacker (the worst case)."""
    tasks = suite if suite is not None else U.make_suite()
    d = dict(U.defence(design["defence"]))
    if design.get("approval"):
        d["approval"] = design["approval"]
        d["id"] = d["id"] + "+" + design["approval"]
    base = DEPTH["base_range"][1]
    rows = [U.evaluate(p, d, base, UTILITY, True) for p in tasks]
    sm = U.summarise(rows)
    tri = 0
    unneeded = 0
    for p in tasks:
        g = U.grant_of(p, d)
        if C.trifecta(g)["all"]:
            tri += 1
        unneeded += len(C.over_grant(g, p)["tools"])
    t = S.tally(S.matrix(design["sandbox"]))
    fr = U.frontier(True, tasks)
    pts = [x for x in fr["points"] if x["id"] != d["id"]] + [dict(sm, id=d["id"], name="this design")]
    on_front = d["id"] in U.pareto(pts)
    items = [
        {"ch": 1, "item": "Every value carries a label, and every call is checked against the policy",
         "verdict": "pass" if d["policy"] else "fail",
         "value": "labels tracked" if d["policy"] else "no labels: the agent cannot tell where data came from"},
        {"ch": 2, "item": "No untrusted result can choose a write or send that nothing checks",
         "verdict": "pass" if sm["ungated"] == 0.0 else ("warn" if sm["ungated"] < 0.1 else "fail"),
         "value": f"{pct(sm['ungated'])} of tasks have an unchecked route"},
        {"ch": 3, "item": "No task holds private data, untrusted input and an outbound channel unpoliced",
         "verdict": "pass" if tri == 0 else ("warn" if d["policy"] else "fail"),
         "value": f"{tri} of {len(tasks)} tasks hold all three" + (", and the policy denies the flow out"
                                                                    if tri and d["policy"] else "")},
        {"ch": 4, "item": "Each task holds only the tools and scopes it needs",
         "verdict": "pass" if unneeded == 0 else "fail",
         "value": f"{fixed(unneeded / len(tasks), 2)} unneeded tools per task"},
        {"ch": 5, "item": "Code runs in a sandbox with a file scope, a container and an egress allow-list",
         "verdict": "pass" if t["open"] <= 1 and t["broken"] == 0 else ("warn" if t["open"] <= 2 else "fail"),
         "value": f"{t['open']} unneeded action{'' if t['open'] == 1 else 's'} still open, {t['broken']} needed broken"},
        {"ch": 6, "item": "Detectors are one layer, never the only one",
         "verdict": "pass" if d["policy"] else ("warn" if d["layers"] else "fail"),
         "value": _detector_value(len(d["layers"]), d["policy"])},
        {"ch": 7, "item": "Untrusted data cannot change the plan",
         "verdict": "pass" if d["mode"] == "dataflow" else ("warn" if d["policy"] else "fail"),
         "value": "planned in advance" if d["mode"] == "dataflow" else "a classic agent: every result steers"},
        {"ch": 8, "item": "Few prompts per task, so each approval gets attention",
         "verdict": "pass" if sm["prompts"] <= 0.5 else ("warn" if sm["prompts"] <= 1.5 else "fail"),
         "value": f"{fixed(sm['prompts'], 2)} prompts per task"},
        {"ch": 9, "item": "The design sits on the security–utility frontier, within the risk budget",
         "verdict": "pass" if on_front and sm["risk"] < UTILITY["risk_budget"] else
                    ("warn" if on_front or sm["risk"] < UTILITY["risk_budget"] else "fail"),
         "value": f"{pct(sm['utility'])} of tasks finish; risk {pct(sm['risk'], 2)} per task against an adaptive attacker"},
    ]
    frames = []
    score = 0
    for i, it in enumerate(items):
        if it["verdict"] == "pass":
            score += 1
        frames.append(dict(it, i=i, passed=score, caption=f"Chapter {it['ch']}: {it['item'].lower()}? "
                                                           f"{it['verdict'].upper()}: {it['value']}."))
    return {"design": design, "items": items, "frames": frames, "summary": sm, "passed": score}


def pct(x: float, d: int = 1) -> str:
    return fixed(x * 100.0 + 0.0, d) + "%"
