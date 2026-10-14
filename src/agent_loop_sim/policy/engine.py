"""The policy engine (since 1.10): run a plan over labels, and decide each call from its labels and the task's
grant. Every decision is ``allow``, ``ask`` (needs the user's approval) or ``deny``, with every rule that fired
and why. The most restrictive decision wins.

Two ways the model's choices are labelled (``mode``):

- ``context``: a classic agent. The model chooses every call, and every argument, after reading everything
  in its context, so each call is **controlled** by the join of everything returned so far.
- ``dataflow``: plan-then-execute (as in CaMeL): the plan comes from the user's request alone, values flow
  only along the plan's arguments, and a call is controlled by the user. A step may name values whose content
  decides it (``ctrl``, since 1.11: the plan branches on what a read returned); its control is then the join
  of those values' labels, so the dependency is tracked rather than hidden.

A call's **effective label** is the join of its arguments' labels and its control label. The rules:

- ``not-granted``: the tool is not in the task's grant (deny);
- ``scope``: the grant lacks the tool's scope (deny);
- ``destination``: an egress call names a destination the grant does not allow (deny);
- ``untrusted-egress-private``: an egress call whose effective label is untrusted and private, to a
  destination not allowed to read private data (deny);
- ``private-egress``: an egress call carrying private data (not untrusted) to a destination not allowed to
  read it (ask);
- ``untrusted-write``: a write whose effective label is untrusted (ask).
"""
from __future__ import annotations

from typing import Any

from . import labels as L
from .catalogue import TOOLS

RANK = {"allow": 0, "ask": 1, "deny": 2}


def decide(tool: str, eff: L.Label, dest: str | None, grant: dict[str, Any]) -> dict[str, Any]:
    t = TOOLS[tool]
    fired: list[dict[str, str]] = []
    if tool not in grant["tools"]:
        fired.append({"rule": "not-granted", "decision": "deny", "reason": f"{tool} is not in this task's grant"})
    if t["scope"] is not None and t["scope"] not in grant["scopes"]:
        fired.append({"rule": "scope", "decision": "deny", "reason": f"the grant lacks the scope {t['scope']}"})
    if t["egress"]:
        if dest not in t["dests"]:
            raise ValueError(f"{tool} cannot send to {dest}")
        if dest not in grant["egress"]:
            fired.append({"rule": "destination", "decision": "deny",
                          "reason": f"{dest} is not an allowed destination"})
        if dest not in grant["private_dests"] and eff["sens"] == "private":
            if eff["src"] == "untrusted":
                fired.append({"rule": "untrusted-egress-private", "decision": "deny",
                              "reason": "untrusted data steers a call that carries private data out to " + dest})
            else:
                fired.append({"rule": "private-egress", "decision": "ask",
                              "reason": f"private data would leave to {dest}"})
    if t["kind"] == "write" and eff["src"] == "untrusted":
        fired.append({"rule": "untrusted-write", "decision": "ask",
                      "reason": "untrusted data decides what is written"})
    decision = "allow"
    for f in fired:
        if RANK[f["decision"]] > RANK[decision]:
            decision = f["decision"]
    return {"decision": decision, "fired": fired}


def explain(tool: str, eff: L.Label, d: dict[str, Any]) -> str:
    if not d["fired"]:
        return f"allow: {tool} with ({L.text(eff)}) breaks no rule"
    return d["decision"] + ": " + "; ".join(f["reason"] for f in d["fired"])


def run_plan(plan: dict[str, Any], grant: dict[str, Any], mode: str = "context",
             approve: bool = True, enforce: bool = True) -> list[dict[str, Any]]:
    """Each step's labels and decision. A denied (or unapproved) call does not run, and its output is
    missing; a later step that needs a missing value is skipped. With ``enforce`` off (since 1.11: an agent
    with no label policy) every decision is still computed and recorded, but every call runs."""
    if mode not in ("context", "dataflow"):
        raise ValueError(mode)
    vals: dict[str, L.Label] = {}
    ctx = dict(L.BOTTOM)
    out: list[dict[str, Any]] = []
    for i, st in enumerate(plan["steps"]):
        rec: dict[str, Any] = {"i": i, "op": st["op"], "tool": st.get("tool"), "args": list(st.get("args", [])),
                               "dest": st.get("dest"), "out": st.get("out")}
        if "ctrl" in st:
            rec["ctrl"] = list(st["ctrl"])
        if st["op"] == "input":
            lab = L.label("user", st["sens"])
            vals[st["out"]] = lab
            ctx = L.join(ctx, lab)
            rec.update({"control": None, "effective": None, "out_label": lab, "decision": "allow", "fired": [],
                        "ran": True, "ctx": dict(ctx), "why": "the user's request: (" + L.text(lab) + ")"})
            out.append(rec)
            continue
        missing = [a for a in st["args"] + st.get("ctrl", []) if a not in vals]
        if mode == "context":
            control = dict(ctx)
        else:
            control = L.join_all([vals[a] for a in st.get("ctrl", []) if a in vals])
        if missing:
            rec.update({"control": control, "effective": None, "out_label": None, "decision": "skip", "fired": [],
                        "ran": False, "ctx": dict(ctx), "why": "skipped: needs " + ", ".join(missing)})
            out.append(rec)
            continue
        eff = L.join_all([vals[a] for a in st["args"]] + [control])
        t = TOOLS[st["tool"]]
        d = decide(st["tool"], eff, st.get("dest"), grant)
        ran = d["decision"] == "allow" or (d["decision"] == "ask" and approve) or not enforce
        lab = None
        if ran and t["kind"] == "read":
            lab = L.join(t["out"], eff)
        elif ran and t["kind"] == "transform":
            lab = eff
        if lab is not None and st.get("out"):
            vals[st["out"]] = lab
            ctx = L.join(ctx, lab)
        why = explain(st["tool"], eff, d)
        if not enforce:
            if d["decision"] != "allow":
                why += " (not enforced: it runs)"
        elif d["decision"] == "ask":
            why += " (approved)" if approve else " (refused)"
        rec.update({"control": control, "effective": eff, "out_label": lab, "decision": d["decision"],
                    "fired": d["fired"], "ran": ran, "ctx": dict(ctx), "why": why})
        out.append(rec)
    return out


def summary(steps: list[dict[str, Any]]) -> dict[str, int]:
    s = {"allow": 0, "ask": 0, "deny": 0, "skip": 0}
    for r in steps:
        if r["op"] == "call":
            s[r["decision"]] += 1
    return s
