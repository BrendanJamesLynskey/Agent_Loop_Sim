"""The policy module (engine 1.10.0): the label lattice's laws, exhaustively; the engine's label propagation
against an independent provenance computation, and its soundness on every plan of up to three calls under
several grants; monotonicity of decisions in the labels; the trifecta check against an exhaustive search on
every subset of the tools; tokens' audience and scope; sandbox layers; the defence-in-depth closed forms
against simulation; and the committed fixtures."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest

from agent_loop_sim import VERSION
from agent_loop_sim.policy import capabilities as C
from agent_loop_sim.policy import catalogue as K
from agent_loop_sim.policy import depth as D
from agent_loop_sim.policy import engine as E
from agent_loop_sim.policy import labels as L
from agent_loop_sim.policy import sandbox as S
from agent_loop_sim.policy import study as T

ROOT = Path(__file__).resolve().parent.parent
LABS = L.all_labels()


# ── the lattice ──────────────────────────────────────────────────────────


def test_partial_order():
    for a in LABS:
        assert L.leq(a, a)
        for b in LABS:
            if L.leq(a, b) and L.leq(b, a):
                assert a == b
            for c in LABS:
                if L.leq(a, b) and L.leq(b, c):
                    assert L.leq(a, c)


def test_join_and_meet_are_least_upper_and_greatest_lower_bounds():
    for a, b in itertools.product(LABS, LABS):
        j, m = L.join(a, b), L.meet(a, b)
        assert L.leq(a, j) and L.leq(b, j)
        assert L.leq(m, a) and L.leq(m, b)
        for u in LABS:
            if L.leq(a, u) and L.leq(b, u):
                assert L.leq(j, u)
            if L.leq(u, a) and L.leq(u, b):
                assert L.leq(u, m)
        assert L.leq(a, b) == (j == b) == (m == a)


def test_lattice_laws():
    for a, b, c in itertools.product(LABS, LABS, LABS):
        assert L.join(a, b) == L.join(b, a) and L.meet(a, b) == L.meet(b, a)
        assert L.join(L.join(a, b), c) == L.join(a, L.join(b, c))
        assert L.meet(L.meet(a, b), c) == L.meet(a, L.meet(b, c))
        assert L.join(a, L.meet(a, b)) == a and L.meet(a, L.join(a, b)) == a
    for a in LABS:
        assert L.join(a, a) == a == L.meet(a, a)
        assert L.join(a, L.BOTTOM) == a and L.meet(a, L.TOP) == a
        assert L.leq(L.BOTTOM, a) and L.leq(a, L.TOP)
    assert L.join_all([]) == L.BOTTOM and L.join_all(LABS) == L.TOP
    assert len(LABS) == 6
    with pytest.raises(ValueError):
        L.label("someone", "public")


# ── plans: propagation and soundness ─────────────────────────────────────

GRANTS = [K.BROAD,
          {"tools": ["read_notes", "read_inbox", "summarise", "send_message"], "scopes": ["mail:read", "mail:send", "notes:read"],
           "egress": ["self", "org"], "private_dests": ["self"]},
          {"tools": ["read_ticket", "fetch_page", "summarise", "write_file", "send_message"],
           "scopes": ["files:write", "mail:send", "tickets:read", "web:fetch"], "egress": ["org", "web"], "private_dests": []},
          {"tools": list(K.TOOL_ORDER), "scopes": ["notes:read"], "egress": ["web"], "private_dests": ["self"]}]


def all_plans(max_calls: int = 3):
    """Every plan of up to three calls: each call takes the request only, the latest value or every value so far,
    and every destination its tool can name; the request is public or private."""
    choices = []
    for t in K.TOOL_ORDER:
        for argmode in ["request", "latest", "all"]:
            for d in (K.TOOLS[t]["dests"] if K.TOOLS[t]["egress"] else [None]):
                choices.append((t, argmode, d))
    for sens in ["public", "private"]:
        for n in range(1, max_calls + 1):
            for seq in itertools.product(choices, repeat=n):
                steps = [{"op": "input", "out": "q", "sens": sens}]
                outs = ["q"]
                for i, (t, argmode, d) in enumerate(seq):
                    args = {"request": ["q"], "latest": [outs[-1]], "all": list(outs)}[argmode]
                    st = {"op": "call", "tool": t, "args": args, "out": None}
                    if K.TOOLS[t]["kind"] in ("read", "transform"):
                        st["out"] = f"v{i}"
                        outs.append(f"v{i}")
                    if d is not None:
                        st["dest"] = d
                    steps.append(st)
                yield {"title": "enumerated", "steps": steps}


def provenance(plan, steps, mode):
    """Independently of the engine: the set of origins (the request, and each read tool that ran) behind each
    value and each call, following arguments (and, in a classic agent, everything seen so far)."""
    origin: dict[str, set] = {}
    seen: set = set()
    out = []
    for st, r in zip(plan["steps"], steps):
        if st["op"] == "input":
            origin[st["out"]] = {("input", st["sens"])}
            seen |= origin[st["out"]]
            out.append(None)
            continue
        if not r["ran"] and r["decision"] == "skip":
            out.append(None)
            continue
        o = set()
        for a in st["args"]:
            o |= origin[a]
        if mode == "context":
            o |= seen
        out.append(set(o))
        if r["ran"] and K.TOOLS[st["tool"]]["kind"] in ("read", "transform"):
            v = set(o)
            if K.TOOLS[st["tool"]]["kind"] == "read":
                v.add(("tool", st["tool"]))
            origin[st["out"]] = v
            seen |= v
    return out


def label_of(origins: set) -> L.Label:
    lab = dict(L.BOTTOM)
    for kind, x in origins:
        lab = L.join(lab, L.label("user", x) if kind == "input" else K.TOOLS[x]["out"])
    return lab


@pytest.mark.parametrize("mode", ["context", "dataflow"])
def test_sound_on_every_plan(mode):
    n = 0
    for g in GRANTS:
        for plan in all_plans(2 if g is not K.BROAD else 3):
            for approve in [True, False]:
                steps = E.run_plan(plan, g, mode, approve)
                prov = provenance(plan, steps, mode)
                for st, r, o in zip(plan["steps"], steps, prov):
                    if st["op"] != "call" or r["decision"] == "skip":
                        continue
                    n += 1
                    # the engine's effective label is exactly the join over the independent provenance
                    assert r["effective"] == label_of(o)
                    if not r["ran"]:
                        continue
                    t = K.TOOLS[st["tool"]]
                    assert st["tool"] in g["tools"]
                    assert t["scope"] is None or t["scope"] in g["scopes"]
                    if t["egress"]:
                        assert st["dest"] in g["egress"]
                        if st["dest"] not in g["private_dests"]:
                            # the property: no untrusted-steered call carries private data out
                            assert not (any(K.TOOLS[x]["out"]["src"] == "untrusted" for k, x in o if k == "tool")
                                        and (any(k == "input" and x == "private" for k, x in o)
                                             or any(K.TOOLS[x]["out"]["sens"] == "private" for k, x in o if k == "tool")))
                            if r["effective"]["sens"] == "private":
                                assert approve and r["decision"] == "ask"
    assert n > 100000


def test_decisions_are_monotone_in_labels():
    """Raising a call's label never makes its decision less restrictive."""
    for g in GRANTS:
        for t in K.TOOL_ORDER:
            for d in (K.TOOLS[t]["dests"] if K.TOOLS[t]["egress"] else [None]):
                for a, b in itertools.product(LABS, LABS):
                    if L.leq(a, b):
                        assert E.RANK[E.decide(t, a, d, g)["decision"]] <= E.RANK[E.decide(t, b, d, g)["decision"]]


def test_benign_plans_and_explanations():
    for p in ["notes_to_file", "inbox_digest", "share_ticket"]:
        for mode in ["context", "dataflow"]:
            assert E.summary(E.run_plan(K.PLANS[p], K.BROAD, mode)) == {"allow": 3, "ask": 0, "deny": 0, "skip": 0}
    r = E.run_plan(K.PLANS["reply_from_notes"], K.BROAD, "context")[-1]
    assert r["decision"] == "deny" and [f["rule"] for f in r["fired"]] == ["untrusted-egress-private"]
    a = E.run_plan(K.PLANS["notes_then_page"], K.BROAD, "context")
    b = E.run_plan(K.PLANS["notes_then_page"], K.BROAD, "dataflow")
    assert [x["decision"] for x in a[1:]] == ["allow", "ask", "allow", "deny"]
    assert [x["decision"] for x in b[1:]] == ["allow", "allow", "allow", "ask"]
    # a denied call's output is missing; a step that needs it is skipped
    plan = {"steps": [{"op": "input", "out": "q", "sens": "public"},
                      {"op": "call", "tool": "read_inbox", "args": ["q"], "out": "m"},
                      {"op": "call", "tool": "summarise", "args": ["m"], "out": "s"}]}
    g = dict(K.BROAD, tools=["summarise"])
    st = E.run_plan(plan, g, "dataflow")
    assert st[1]["decision"] == "deny" and st[2]["decision"] == "skip" and not st[2]["ran"]
    with pytest.raises(ValueError):
        E.decide("fetch_page", L.BOTTOM, "org", K.BROAD)
    with pytest.raises(ValueError):
        E.run_plan(plan, g, "other")


# ── capabilities ────────────────────────────────────────────────────────


def test_trifecta_agrees_with_exhaustive_search():
    for m in range(1 << len(K.TOOL_ORDER)):
        tools = [t for i, t in enumerate(K.TOOL_ORDER) if m >> i & 1]
        for egress in [["self", "org", "web"], ["self"], ["org"], ["web"], []]:
            for private_dests in [["self"], ["self", "org", "web"]]:
                g = {"tools": tools, "scopes": [], "egress": egress, "private_dests": private_dests}
                ex = C.exposure(g)
                assert C.trifecta(g)["all"] == ex["exposed"]
                if ex["exposed"]:
                    # the policy stops the flow it found (at its last call or earlier), whatever the approvals
                    g2 = T.grant_for(tools, egress)
                    g2["private_dests"] = private_dests
                    for approve in [True, False]:
                        steps = E.run_plan(T.exposure_plan(ex["path"], ex["dest"]), g2, "context", approve)
                        assert not steps[-1]["ran"]
                        # with every approval granted, a deny rule is what stops it
                        assert not approve or any(r["decision"] == "deny" for r in steps)


def test_removing_any_circle_closes_the_path():
    st = T.trifecta_study()
    for c in st["circles"]:
        assert c["exposure"]["exposed"] == (c["mask"] == 7)
    assert all(c["exposure"]["exposed"] for c in st["double"])


def test_needed_and_tokens():
    for p in K.PLAN_ORDER:
        n = C.needed(K.PLANS[p])
        assert E.summary(E.run_plan(K.PLANS[p], n, "dataflow"))["deny"] == \
            E.summary(E.run_plan(K.PLANS[p], K.BROAD, "dataflow"))["deny"]
        assert all(C.over_grant(n, K.PLANS[p])[k] == [] for k in ["tools", "scopes", "egress"])
    toks = C.mint(K.BROAD)
    assert set(toks) == {K.TOOLS[t]["aud"] for t in K.TOOL_ORDER if K.TOOLS[t]["aud"]}
    for aud, tok in toks.items():
        for t in K.TOOL_ORDER:
            ok = C.check_token(tok, t)["ok"]
            assert ok == (K.TOOLS[t]["aud"] is None or (K.TOOLS[t]["aud"] == aud and K.TOOLS[t]["scope"] in tok["scopes"]))
    assert not C.check_token(None, "read_notes")["ok"]
    assert C.mint(C.needed(K.PLANS["notes_to_file"])) == {"files": {"aud": "files", "scopes": ["files:write"]},
                                                          "notes-api": {"aud": "notes-api", "scopes": ["notes:read"]}}
    pv = T.privilege_study()
    assert [f["broad"]["decision"] for f in pv["frames"][1:]] == ["allow", "allow", "allow", "ask"]
    assert [f["task"]["decision"] for f in pv["frames"][1:]] == ["allow", "allow", "allow", "deny"]
    assert not pv["passthrough"]["check"]["ok"]


# ── sandbox ──────────────────────────────────────────────────────────────


def test_sandbox_layers():
    sets = T.all_layer_sets()
    by = {tuple(s["layers"]): s for s in sets}
    full = by[tuple(x["id"] for x in K.LAYERS)]
    assert [r["outcome"] for r in full["rows"] if not r["need"]] == ["stopped", "stopped", "stopped", "stopped", "open"]
    assert all(r["outcome"] == "works" for r in full["rows"] if r["need"])
    assert [r["id"] for r in full["rows"] if r["outcome"] == "open"] == ["net_registry_put"]
    for s in sets:
        for t in sets:
            if set(s["layers"]) <= set(t["layers"]):
                for a, b in zip(s["rows"], t["rows"]):
                    assert not a["stopped"] or b["stopped"]
    for a in K.ACTIONS:
        assert (S.stops(K.LAYERS[2], a) is not None) == (a["kind"] == "net" and a["dest"] != "registry")


# ── defence in depth ─────────────────────────────────────────────────────


def test_depth_closed_forms():
    m = [x["miss"] for x in K.DEPTH["layers"]]
    assert D.all_miss_independent(m) == pytest.approx(math.prod(m), rel=1e-15)
    assert D.all_miss_common(m, 0.0) == D.all_miss_independent(m)
    for c in K.DEPTH["c"]:
        q = D.q_given_common(m, c)
        for mi, qi in zip(m, q):
            assert c + (1 - c) * qi == pytest.approx(mi, abs=1e-15)
        prev = 1.0
        for k in range(1, len(m) + 1):
            p = D.all_miss_common(m[:k], c)
            assert c - 1e-15 <= p <= prev + 1e-15
            prev = p
    for p in [0.0, 0.001, 0.24]:
        for n in [0, 1, 7, 1000]:
            assert D.at_least_one(p, n) == pytest.approx(1 - (1 - p) ** n, abs=1e-12)
    for a, b in [(0.2, 0.3), (0.04, 0.5)]:
        assert D.pair_correlation(a, b, 0.0) == pytest.approx(0.0, abs=1e-15)
        assert D.pair_correlation(a, b, 0.01) > 0
    with pytest.raises(ValueError):
        D.q_given_common([0.01], 0.02)


def test_depth_simulation_matches():
    m = [x["miss"] for x in K.DEPTH["layers"]]
    for c in [0.0, 0.01, 0.04]:
        n = 60000
        s = D.simulate(m, c, n, 11)
        p = D.all_miss_common(m, c)
        assert abs(s["rate"] - p) < 4 * math.sqrt(p * (1 - p) / n)
        for i, mi in enumerate(m):
            assert abs(s["single"][i] / n - mi) < 4 * math.sqrt(mi * (1 - mi) / n)
        q = D.q_given_common(m, c)
        both = c + (1 - c) * q[1] * q[2]
        assert abs(s["pair"][1][2] / n - both) < 4 * math.sqrt(both * (1 - both) / n)


# ── studies and fixtures ─────────────────────────────────────────────────


def test_studies_have_captions():
    for p in K.PLAN_ORDER:
        for f in T.boundary_frames(p):
            assert f["caption"].startswith("Step ")
        for f in T.context_frames(p)["frames"]:
            assert f["caption"]
    assert len(T.sandbox_frames()) == len(K.LAYERS) + 1


def test_no_content_only_labels():
    """The module models labels, not text: no tool, plan or action carries a message body or an instruction."""
    blob = json.dumps({"tools": K.TOOLS, "plans": K.PLANS, "actions": K.ACTIONS}).lower()
    for w in ["ignore", "instruction", "password", "<", "http://", "https://"]:
        assert w not in blob


def test_fixtures_up_to_date():
    fx = json.loads((ROOT / "fixtures" / "policy_fixtures.json").read_text(encoding="utf-8"))
    assert fx["engine"] == VERSION == "1.10.0"
    assert len(fx["caps"]["subsets"]) == 256
    assert all(s["trifecta"]["all"] == s["exposure"]["exposed"] for s in fx["caps"]["subsets"])


def test_sandbox_action_frames():
    full = T.sandbox_action_frames([x["id"] for x in K.LAYERS])
    assert len(full) == len(K.ACTIONS)
    assert [f["outcome"] for f in full] == [r["outcome"] for r in S.matrix([x["id"] for x in K.LAYERS])]
    assert "not what it carries" in full[-1]["caption"]
    none = T.sandbox_action_frames([])
    assert all(f["outcome"] in ("works", "open") for f in none)


def test_decisions_are_monotone_in_grants():
    """Shrinking a grant (any of its tools, scopes, destinations or private destinations) never relaxes a decision."""
    def subsets(xs):
        return [[x for i, x in enumerate(xs) if m >> i & 1] for m in range(1 << len(xs))]
    scopes = K.BROAD["scopes"]
    big = K.BROAD
    for t in K.TOOL_ORDER:
        for d in (K.TOOLS[t]["dests"] if K.TOOLS[t]["egress"] else [None]):
            for lab in LABS:
                top = E.RANK[E.decide(t, lab, d, big)["decision"]]
                for tools in [K.TOOL_ORDER, [x for x in K.TOOL_ORDER if x != t]]:
                    for sc in [scopes, [s for s in scopes if s != K.TOOLS[t]["scope"]]]:
                        for eg in subsets(K.DEST_ORDER):
                            for pd in subsets(["self"]):
                                small = {"tools": tools, "scopes": sc, "egress": eg, "private_dests": pd}
                                assert E.RANK[E.decide(t, lab, d, small)["decision"]] >= top
