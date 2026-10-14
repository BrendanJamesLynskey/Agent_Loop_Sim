"""The policy module (engine 1.10.0), written by make_fixtures.py: ts/src/policy_data.json (the catalogue: tools,
destinations, plans, grants, sandbox actions and layers, published measurements, one source of truth for both
languages), fixtures/policy_fixtures.json (every label operation, decision, plan run, capability check, sandbox
matrix, depth computation and chapter study, for the TS port to reproduce exactly) and fixtures/policy_results.md
(the recorded run every published number comes from)."""
from __future__ import annotations

from typing import Any

from agent_loop_sim import VERSION
from agent_loop_sim.context.text import fixed
from agent_loop_sim.policy import capabilities as C
from agent_loop_sim.policy import catalogue as K
from agent_loop_sim.policy import depth as D
from agent_loop_sim.policy import engine as E
from agent_loop_sim.policy import labels as L
from agent_loop_sim.policy import sandbox as S
from agent_loop_sim.policy import study as T

NARROW = {"tools": ["read_notes", "summarise", "send_message"], "scopes": ["mail:send", "notes:read"],
          "egress": ["self", "org"], "private_dests": ["self"]}
DEPTH_M = [[0.5], [0.2, 0.3], [0.04, 0.2, 0.3], [0.1, 0.1, 0.1, 0.1], [1.0, 0.5]]


def policy_data() -> dict[str, Any]:
    return {"version": VERSION, "sources": L.SOURCES, "sensitivity": L.SENSITIVITY, "source_text": L.SOURCE_TEXT,
            "tools": K.TOOLS, "tool_order": K.TOOL_ORDER, "dests": K.DESTS, "dest_order": K.DEST_ORDER,
            "plans": K.PLANS, "plan_order": K.PLAN_ORDER, "broad": K.BROAD, "circles": K.CIRCLES,
            "circle_tools": K.CIRCLE_TOOLS, "privilege": K.PRIVILEGE, "actions": K.ACTIONS, "layers": K.LAYERS,
            "published": K.PUBLISHED, "depth": K.DEPTH}


def tool_subsets() -> list[list[str]]:
    return [[t for i, t in enumerate(K.TOOL_ORDER) if m >> i & 1] for m in range(1 << len(K.TOOL_ORDER))]


def policy_fixtures() -> tuple[dict[str, Any], dict[str, Any], str]:
    labs = L.all_labels()
    lattice = [{"a": a, "b": b, "leq": L.leq(a, b), "join": L.join(a, b), "meet": L.meet(a, b)}
               for a in labs for b in labs]
    decisions = []
    for gid, g in [("broad", K.BROAD), ("narrow", NARROW)]:
        for t in K.TOOL_ORDER:
            for lab in labs:
                for d in (K.TOOLS[t]["dests"] if K.TOOLS[t]["egress"] else [None]):
                    decisions.append({"grant": gid, "tool": t, "label": lab, "dest": d,
                                      "out": E.decide(t, lab, d, g)})
    runs = []
    for p in K.PLAN_ORDER:
        for gid, g in [("broad", K.BROAD), ("needed", C.needed(K.PLANS[p])), ("narrow", NARROW)]:
            for mode in ["context", "dataflow"]:
                for approve in [True, False]:
                    st = E.run_plan(K.PLANS[p], g, mode, approve)
                    runs.append({"plan": p, "grant": gid, "mode": mode, "approve": approve, "steps": st,
                                 "summary": E.summary(st)})
    caps = {"needed": {p: C.needed(K.PLANS[p]) for p in K.PLAN_ORDER},
            "over": {p: C.over_grant(K.BROAD, K.PLANS[p]) for p in K.PLAN_ORDER},
            "subsets": [], "mint": {"broad": C.mint(K.BROAD), "narrow": C.mint(NARROW)},
            "tokens": []}
    for tools in tool_subsets():
        g = T.grant_for(tools)
        caps["subsets"].append({"tools": tools, "trifecta": C.trifecta(g), "exposure": C.exposure(g)})
    for aud, tok in C.mint(K.BROAD).items():
        for t in K.TOOL_ORDER:
            caps["tokens"].append({"aud": aud, "tool": t, "out": C.check_token(tok, t)})
    caps["tokens"].append({"aud": None, "tool": "read_notes", "out": C.check_token(None, "read_notes")})
    caps["tokens"].append({"aud": "notes-api", "tool": "read_notes",
                           "out": C.check_token({"aud": "notes-api", "scopes": []}, "read_notes")})
    depth = {"independent": [[m, D.all_miss_independent(m)] for m in DEPTH_M],
             "common": [[m, c, D.all_miss_common(m, c)] for m in DEPTH_M for c in [0.0, 0.01, 0.05]
                        if all(x >= c for x in m)],
             "corr": [[a, b, c, D.pair_correlation(a, b, c)] for a, b in [[0.2, 0.3], [0.04, 0.2], [0.5, 0.5]]
                      for c in [0.0, 0.01, 0.04]],
             "any": [[p, n, D.at_least_one(p, n)] for p in [0.0, 0.0024, 0.01, 0.24] for n in [0, 1, 10, 1000]],
             "sim": D.simulate([0.2, 0.3, 0.5], 0.05, 500, 7)}
    fx = {
        "engine": VERSION,
        "lattice": lattice,
        "join_all": [L.join_all([]), L.join_all(labs), L.join_all(labs[:3])],
        "text": [L.text(x) for x in labs],
        "decisions": decisions,
        "runs": runs,
        "caps": caps,
        "sandbox": T.all_layer_sets(),
        "depth": depth,
        "study": {
            "boundary": {f"{p}|{m}": T.boundary_frames(p, m) for p in K.PLAN_ORDER for m in ["context", "dataflow"]},
            "context": {p: T.context_frames(p) for p in K.PLAN_ORDER},
            "trifecta": T.trifecta_study(),
            "privilege": T.privilege_study(),
            "sandbox": T.sandbox_frames(),
            "depth": T.depth_study(),
        },
    }
    return policy_data(), fx, results_md(fx)


def _p(x: float, d: int = 1) -> str:
    return fixed(x * 100.0 + 0.0, d) + "%"


def _g(x: float, d: int = 4) -> str:
    return fixed(x + 0.0, d)


def results_md(fx: dict[str, Any]) -> str:
    st = fx["study"]
    out = [f"# Policy module results (engine {VERSION})", "",
           "Written by `scripts/make_fixtures.py` (CI checks it is up to date). The tools, plans, grants, sandbox "
           "actions and every layer miss rate marked illustrative are abstract placeholders; the decisions follow "
           "from the rules. The published measurements are quoted from their papers (arXiv IDs below).", "",
           "## Plans under the broad grant (chapters 1 and 2)", "",
           "| plan | classic agent: allow / ask / deny | planned in advance: allow / ask / deny | last call, classic | "
           "last call, planned |", "|---|---|---|---|---|"]
    for p in K.PLAN_ORDER:
        a = next(r for r in fx["runs"] if r["plan"] == p and r["grant"] == "broad" and r["mode"] == "context" and r["approve"])
        b = next(r for r in fx["runs"] if r["plan"] == p and r["grant"] == "broad" and r["mode"] == "dataflow" and r["approve"])
        sa, sb = a["summary"], b["summary"]
        out.append(f"| {K.PLANS[p]['title']} | {sa['allow']} / {sa['ask']} / {sa['deny']} | {sb['allow']} / {sb['ask']} / "
                   f"{sb['deny']} | {a['steps'][-1]['decision']} | {b['steps'][-1]['decision']} |")
    sub = fx["caps"]["subsets"]
    n_tri = sum(1 for s in sub if s["trifecta"]["all"])
    agree = sum(1 for s in sub if s["trifecta"]["all"] == s["exposure"]["exposed"])
    out += ["", "## The trifecta (chapter 3)", "",
            f"Over all {len(sub)} subsets of the {len(K.TOOL_ORDER)} tools: {n_tri} hold all three circles; the "
            f"trifecta check and the exhaustive search for a flow out agree on {agree} of {len(sub)}.", "",
            "| circles held | trifecta | flow out found | sequences checked |", "|---|---|---|---|"]
    for c in st["trifecta"]["circles"] + st["trifecta"]["double"]:
        out.append(f"| {c['name']} | {'yes' if c['trifecta']['all'] else 'no'} | "
                   f"{' → '.join(c['exposure']['path']) + ' (' + str(c['exposure']['dest']) + ')' if c['exposure']['exposed'] else 'none'} | "
                   f"{c['exposure']['checked']} |")
    pv = st["privilege"]
    out += ["", "## Least privilege (chapter 4)", "",
            f"Plan: {pv['plan']['title']}. Broad grant: {len(K.BROAD['tools'])} tools; least privilege: "
            f"{len(pv['runs']['task']['grant']['tools'])} ({', '.join(pv['runs']['task']['grant']['tools'])}). "
            f"Unneeded under the broad grant: {', '.join(pv['runs']['broad']['over']['tools'])}.", "",
            "| step | broad grant | least privilege |", "|---|---|---|"]
    for f in pv["frames"][1:]:
        out.append(f"| {f['broad']['tool']} | {f['broad']['decision']} | {f['task']['decision']} |")
    out += ["", f"A token minted for {pv['passthrough']['token']['aud']} presented to {pv['passthrough']['tool']}: "
            f"{pv['passthrough']['check']['reason']}.", "", "## Sandbox layers (chapter 5)", "",
            "| layers | needed actions working | needed actions broken | unneeded stopped | unneeded still open |",
            "|---|---|---|---|---|"]
    for f in st["sandbox"]:
        t = f["tally"]
        out.append(f"| {' + '.join(f['layers']) or 'none'} | {t['works']} | {t['broken']} | {t['stopped']} | {t['open']} |")
    dp = st["depth"]
    cfg = K.DEPTH
    out += ["", "## Defence in depth (chapter 6)", "",
            f"Base rate {_p(cfg['base'])} (InjecAgent, arXiv 2403.02691; up to {_p(cfg['base_range'][1], 0)} as in "
            f"Spotlighting's baseline, arXiv 2403.14720). Layers: " + "; ".join(
                f"{x['name']} misses {_p(x['miss'])}{' (illustrative)' if x['illustrative'] else ' (ratio of Spotlighting’s bounds)'}"
                for x in cfg["layers"]) + f". Independent, all three miss: {_p(dp['independent'], 2)}.", "",
            f"| shared blind spot c | layers | all miss | end to end | at least one of {cfg['attempts']:,} gets through |",
            "|---|---|---|---|---|"]
    for r in dp["table"]:
        out.append(f"| {_p(r['c'])} | {r['layers']} | {_p(r['all_miss'], 2)} | {_p(r['end_to_end'], 3)} | {_p(r['any_of_n'])} |")
    out += ["", f"Simulation ({cfg['trials']:,} attempts per c, seed {cfg['seed']} + i): " + "; ".join(
        f"c {_p(s['c'])}: {_p(s['rate'], 2)}" for s in dp["sims"]) + ".", "", "## Published measurements", "",
            "| measurement | value | arXiv |", "|---|---|---|"]
    for x in K.PUBLISHED:
        out.append(f"| {x['what']} | {x['text']} | {x['arxiv']} |")
    return "\n".join(out) + "\n"
