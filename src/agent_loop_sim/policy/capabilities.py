"""Capabilities (since 1.10): what a task holds, what it needs, and when the combination is dangerous.

- ``needed``: the least privilege for a plan: exactly the tools, scopes and destinations its steps use.
- ``trifecta``: the agent holds private data, untrusted input and an outbound channel at once (the "lethal
  trifecta"). ``exposure`` checks the same thing independently, by enumerating every sequence of granted
  calls (up to three) for one in which an outbound call is steered by untrusted data while carrying
  private data, in a classic agent whose context taints every choice; the tests check the two agree on every
  subset of the tools.
- ``mint`` / ``check_token``: OAuth-style access tokens, one per audience (resource server), each with
  only the scopes the grant holds there. A server accepts a token only if it is the token's audience and
  the token holds the scope (RFC 8707 resource indicators; MCP forbids passing a client's token through
  to another server).
"""
from __future__ import annotations

from typing import Any

from . import labels as L
from .catalogue import DEST_ORDER, TOOL_ORDER, TOOLS


def needed(plan: dict[str, Any], private_dests: list[str] | None = None) -> dict[str, Any]:
    tools: list[str] = []
    dests: list[str] = []
    for st in plan["steps"]:
        if st["op"] != "call":
            continue
        if st["tool"] not in tools:
            tools.append(st["tool"])
        if st.get("dest") and st["dest"] not in dests:
            dests.append(st["dest"])
    return {"tools": [t for t in TOOL_ORDER if t in tools],
            "scopes": sorted(TOOLS[t]["scope"] for t in tools if TOOLS[t]["scope"]),
            "egress": [d for d in DEST_ORDER if d in dests],
            "private_dests": list(private_dests if private_dests is not None else ["self"])}


def over_grant(grant: dict[str, Any], plan: dict[str, Any]) -> dict[str, list[str]]:
    """What the grant holds that the plan never uses."""
    n = needed(plan)
    return {k: [x for x in grant[k] if x not in n[k]] for k in ["tools", "scopes", "egress"]}


def circles(grant: dict[str, Any]) -> dict[str, list[str]]:
    """Which granted tools put the agent in each circle."""
    out: dict[str, list[str]] = {"private": [], "untrusted": [], "outbound": []}
    for t in TOOL_ORDER:
        if t not in grant["tools"]:
            continue
        spec = TOOLS[t]
        if spec["kind"] == "read" and spec["out"]["sens"] == "private":
            out["private"].append(t)
        if spec["kind"] == "read" and spec["out"]["src"] == "untrusted":
            out["untrusted"].append(t)
        if spec["egress"] and any(d in grant["egress"] and d not in grant["private_dests"] for d in spec["dests"]):
            out["outbound"].append(t)
    return out


def trifecta(grant: dict[str, Any]) -> dict[str, Any]:
    c = circles(grant)
    return dict(c, all=bool(c["private"]) and bool(c["untrusted"]) and bool(c["outbound"]))


def exposure(grant: dict[str, Any], max_len: int = 3) -> dict[str, Any]:
    """Search, without any policy, for the shortest sequence of granted calls whose last call carries data out
    (to a destination not allowed to read private data) while its control is untrusted and private. In a
    classic agent every result joins the context, which controls every later call."""
    calls = [t for t in TOOL_ORDER if t in grant["tools"] and TOOLS[t]["kind"] == "read"]
    outs = [(t, d) for t in TOOL_ORDER if t in grant["tools"] and TOOLS[t]["egress"]
            for d in DEST_ORDER if d in TOOLS[t]["dests"] and d in grant["egress"] and d not in grant["private_dests"]]
    checked = 0
    for n in range(0, max_len):
        seqs: list[list[str]] = [[]]
        for _ in range(n):
            seqs = [s + [t] for s in seqs for t in calls]
        for s in seqs:
            ctx = dict(L.BOTTOM)
            for t in s:
                ctx = L.join(ctx, L.join(TOOLS[t]["out"], ctx))
            for t, d in outs:
                checked += 1
                if ctx["src"] == "untrusted" and ctx["sens"] == "private":
                    return {"exposed": True, "path": s + [t], "dest": d, "checked": checked}
    return {"exposed": False, "path": [], "dest": None, "checked": checked}


def mint(grant: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One token per audience the grant reaches, holding only the grant's scopes there."""
    toks: dict[str, dict[str, Any]] = {}
    for t in TOOL_ORDER:
        spec = TOOLS[t]
        if t not in grant["tools"] or spec["aud"] is None or spec["scope"] not in grant["scopes"]:
            continue
        tok = toks.setdefault(spec["aud"], {"aud": spec["aud"], "scopes": []})
        if spec["scope"] not in tok["scopes"]:
            tok["scopes"].append(spec["scope"])
    for tok in toks.values():
        tok["scopes"].sort()
    return {k: toks[k] for k in sorted(toks)}


def check_token(token: dict[str, Any] | None, tool: str) -> dict[str, Any]:
    spec = TOOLS[tool]
    if spec["aud"] is None:
        return {"ok": True, "reason": f"{tool} runs in the model; no token needed"}
    if token is None:
        return {"ok": False, "reason": f"no token for {spec['aud']}"}
    if token["aud"] != spec["aud"]:
        return {"ok": False, "reason": f"the token is for {token['aud']}, not {spec['aud']}: rejected"}
    if spec["scope"] not in token["scopes"]:
        return {"ok": False, "reason": f"the token lacks {spec['scope']}"}
    return {"ok": True, "reason": f"token for {spec['aud']} holds {spec['scope']}"}
