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
- ``depth_study``: layered detectors, independent and correlated (closed forms, and a simulation).
"""
from __future__ import annotations

import copy
from typing import Any

from ..context.text import fixed
from . import capabilities as C
from . import depth as D
from . import engine as E
from . import labels as L
from . import sandbox as S
from .catalogue import (ACTIONS, BROAD, CIRCLE_TOOLS, DEPTH, DEST_ORDER, LAYERS, PLANS, PRIVILEGE, TOOL_ORDER,
                        TOOLS)


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


# ── defence in depth (chapter 6, in 26B) ───────────────────────────────────


def depth_study(cfg: dict[str, Any] = DEPTH) -> dict[str, Any]:
    m = [x["miss"] for x in cfg["layers"]]
    sims = [dict(D.simulate(m, c, cfg["trials"], cfg["seed"] + i), c=c) for i, c in enumerate(cfg["c"])]
    corr = [{"c": c, "pairs": [[D.pair_correlation(m[i], m[j], c) for j in range(len(m))] for i in range(len(m))]}
            for c in cfg["c"]]
    return {"table": D.depth_table(cfg["base"], m, cfg["c"], cfg["attempts"]),
            "independent": D.all_miss_independent(m), "sims": sims, "corr": corr}


def pct(x: float, d: int = 1) -> str:
    return fixed(x * 100.0 + 0.0, d) + "%"
