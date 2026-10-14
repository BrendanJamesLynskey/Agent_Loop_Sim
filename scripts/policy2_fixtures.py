"""The policy module's part two (engine 1.11.0), written by make_fixtures.py: fixtures/policy2_fixtures.json (the
benign task suite, plans that branch on a value, runs with the policy off, every task under every defence, the
frontier for fixed and adaptive attacks, the seeded simulations, and chapters 6 to 10's studies, for the TS port to
reproduce exactly) and fixtures/policy2_results.md (the recorded run every published number comes from)."""
from __future__ import annotations

from typing import Any

from agent_loop_sim import VERSION
from agent_loop_sim.context.text import fixed
from agent_loop_sim.policy import capabilities as C
from agent_loop_sim.policy import catalogue as K
from agent_loop_sim.policy import engine as E
from agent_loop_sim.policy import study as T
from agent_loop_sim.policy import utility as U

SANDBOXES = [["fs_scope", "container", "egress"], []]


def policy2_fixtures() -> tuple[dict[str, Any], str]:
    suite = U.make_suite()
    ctrl_runs = []
    for gid, g in [("broad", K.BROAD), ("needed", C.needed(K.ADAPTIVE))]:
        for mode in ["context", "dataflow"]:
            for approve in [True, False]:
                for enforce in [True, False]:
                    ctrl_runs.append({"grant": gid, "mode": mode, "approve": approve, "enforce": enforce,
                                      "steps": E.run_plan(K.ADAPTIVE, g, mode, approve, enforce)})
    open_runs = []
    for p in K.PLAN_ORDER:
        for mode in ["context", "dataflow"]:
            open_runs.append({"plan": p, "mode": mode, "steps": E.run_plan(K.PLANS[p], K.BROAD, mode, True, False)})
    rows = {}
    summaries: dict[str, Any] = {"fixed": {}, "adaptive": {}}
    sims = {}
    for d in K.DEFENCES:
        r = U.run_defence(d, suite)
        rows[d["id"]] = r["rows"]
        summaries["fixed"][d["id"]] = r["summary"]
        ra = U.run_defence(d, suite, K.DEPTH["base_range"][1], K.UTILITY, True)
        summaries["adaptive"][d["id"]] = ra["summary"]
        sims[d["id"]] = U.simulate(r["rows"][:30], d, K.DEPTH["base"], 20, 5 + len(sims))
    checks = []
    for did in K.DEFENCE_ORDER:
        for ap in [None, "consequential"]:
            for sb in SANDBOXES:
                checks.append(T.checklist({"defence": did, "approval": ap, "sandbox": sb}, suite))
    fx = {
        "engine": VERSION,
        "suite": suite,
        "ctrl_runs": ctrl_runs,
        "open_runs": open_runs,
        "approval_miss": [[n, U.approval_miss(n)] for n in range(1, 13)],
        "rows": rows,
        "summaries": summaries,
        "frontier": {"fixed": U.frontier(False, suite), "adaptive": U.frontier(True, suite)},
        "sims": sims,
        "study": {
            "depth": {fixed(c, 3): T.depth_frames(c) for c in K.DEPTH["c"]},
            "depth_k": [T.depth_frames(0.01, k) for k in range(0, len(K.DEPTH["layers"]) + 1)],
            "dual": {p: T.dual_frames(p) for p in K.DUAL_PLANS},
            "approval": T.approval_study(suite),
            "frontier": T.frontier_study(suite),
            "checklist": checks,
        },
    }
    return fx, results_md(fx)


def _p(x: float, d: int = 1) -> str:
    return fixed(x * 100.0 + 0.0, d) + "%"


def results_md(fx: dict[str, Any]) -> str:
    st = fx["study"]
    suite = fx["suite"]
    sinks: dict[str, int] = {}
    for p in suite:
        last = p["steps"][-1]
        key = (last["tool"] + (" → " + last["dest"] if last.get("dest") else "")) if last["tool"] != "summarise" \
            else "answer the user"
        sinks[key] = sinks.get(key, 0) + 1
    n_adapt = sum(1 for p in suite if p["adaptive"])
    out = [f"# Policy module results, part two (engine {VERSION})", "",
           "Written by `scripts/make_fixtures.py` (CI checks it is up to date). The benign task suite, the "
           "detector false-positive rates of the classifier and the output check, the approval miss rates and "
           "fatigue are illustrative; the decisions follow from the rules. Base rates and provenance marking's "
           "rates come from the published measurements (arXiv IDs in `policy_results.md`); the adaptive setting "
           "follows arXiv 2503.00061 (all eight defences tested bypassed, attack success over 50%).", "",
           "## The benign suite", "",
           f"{len(suite)} tasks (seed {K.SUITE['seed']}); {n_adapt} branch on what their last read returned. "
           "Sinks: " + "; ".join(f"{k}: {v}" for k, v in sorted(sinks.items())) + ".", ""]
    for key, title in [("fixed", "fixed attacks, base rate 24% (InjecAgent)"),
                       ("adaptive", "an adaptive attacker, base rate 50%, detectors passed")]:
        fr = fx["frontier"][key]
        out += [f"## The frontier: {title} (chapter 9)", "",
                "| defence | tasks finished | blocked | prompts / task | extra calls / task | risk / task | "
                "unchecked route | on the frontier |", "|---|---|---|---|---|---|---|---|"]
        for p in fr["points"]:
            out.append(f"| {p['name']} | {_p(p['utility'])} | {_p(p['blocked'])} | {fixed(p['prompts'], 2)} | "
                       f"{fixed(p['extra'], 2)} | {_p(p['risk'], 2)} | {_p(p['ungated'])} | "
                       f"{'yes' if p['id'] in fr['front'] else 'no'} |")
        out.append("")
    out += ["## What blocks benign tasks (fixed attacks)", "", "| defence | " + " | ".join(U.RULES) + " |",
            "|---|" + "---|" * len(U.RULES)]
    for did in K.DEFENCE_ORDER:
        s = fx["summaries"]["fixed"][did]
        out.append(f"| {did} | " + " | ".join(str(s["by_rule"][r]) for r in U.RULES) + " |")
    ap = st["approval"]
    out += ["", "## Approval thresholds over the label policy (chapter 8)", "",
            "| threshold | tasks finished | prompts / task | risk / task | unchecked route |", "|---|---|---|---|---|"]
    for th in K.UTILITY["thresholds"]:
        s = ap["thresholds"][th]["summary"]
        out.append(f"| {th} | {_p(s['utility'])} | {fixed(s['prompts'], 2)} | {_p(s['risk'], 2)} | {_p(s['ungated'])} |")
    out += ["", "Approval fatigue (illustrative): a mistaken approval at prompt n of a task: " +
            "; ".join(f"{x['n']}: {_p(x['miss'])}" for x in ap["fatigue"]) + ".", "",
            "## Attempts through the layers (chapter 6)", "",
            f"{K.DEPTH_DOTS['n']} attempts, seed {K.DEPTH_DOTS['seed']}, all three layers.", "",
            "| shared blind spot c | attempts | steer the model | + marking | + classifier | + output check |",
            "|---|---|---|---|---|---|"]
    for c, d in st["depth"].items():
        out.append(f"| {_p(float(c))} | " + " | ".join(f"{f['count']} ({fixed(f['expected'], 2)})" for f in d["frames"]) + " |")
    out += ["", "(Counts drawn; expected values from the closed form in brackets.)", "",
            "## Plan-then-execute (chapter 7)", "", "| plan | planned: allow / ask / deny | classic: allow / ask / deny |",
            "|---|---|---|"]
    for p, d in st["dual"].items():
        a, b = d["summary"]["dataflow"], d["summary"]["context"]
        out.append(f"| {d['title']} | {a['allow']} / {a['ask']} / {a['deny']} | {b['allow']} / {b['ask']} / {b['deny']} |")
    fs = st["frontier"]
    out += ["", f"Simulation of plan-then-execute ({K.UTILITY['reps']} draws of each task, seed {K.UTILITY['seed']}): "
            f"risk {_p(fs['sim']['risk'], 2)} (closed form {_p(fx['summaries']['fixed']['planned']['risk'], 2)}), "
            f"tasks finished {_p(fs['sim']['utility'])}.", "", "## The checklist (chapter 10), with every sandbox layer",
            "", "| design | passed | " + " | ".join(f"ch {i}" for i in range(1, 10)) + " |", "|---|---|" + "---|" * 9]
    for c in st["checklist"]:
        if not c["design"]["sandbox"]:
            continue
        name = c["design"]["defence"] + (" + approve writes and sends" if c["design"]["approval"] else "")
        out.append(f"| {name} | {c['passed']} / 9 | " + " | ".join(i["verdict"] for i in c["items"]) + " |")
    return "\n".join(out) + "\n"
