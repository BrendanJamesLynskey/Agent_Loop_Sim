"""The orchestration module's fixtures (engine 1.6.0; Send sessions since 1.7.0), written by make_fixtures.py:
fixtures/orchestration_fixtures.json (for the TS port to reproduce exactly),
fixtures/orchestration_results.md (the recorded run every published number comes from) and
ts/src/orchestration_data.json (the graphs, sessions and workflow, shared with the port)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_sim import VERSION
from agent_loop_sim.context.text import fixed
from agent_loop_sim.orchestration import durable as D
from agent_loop_sim.orchestration import views as V
from agent_loop_sim.orchestration.graph import apply_writes, default_values, route, run_session
from agent_loop_sim.orchestration.scenarios import GRAPHS, SESSIONS, graph

ROOT = Path(__file__).resolve().parent.parent
TEACH = ["chain", "fanout", "react_loop", "uneven_join", "loop_join"]
MERGES = ["merge_overwrite", "merge_add", "merge_max"]
TIMELINES = ["crash", "crash_twice", "approval", "approval_no", "gated", "gated_edit", "fork", "replay", "replay_interrupt"]
SWEEP_Q = [0.05, 0.1, 0.2, 0.5]
SWEEP_A = [1, 2, 3, 5]
SWEEP_RUNS = 4000
SWEEP_SEED = 24


def durable_grid() -> list[dict[str, Any]]:
    out = []
    crashes: list[Any] = [None] + [{"after_event": k} for k in range(0, 11)] + \
        [{"in_activity": a} for a in ["reserve_stock", "charge_card", "ship_express", "send_receipt"]]
    for c in crashes:
        for key in [False, True]:
            for mode in D.CHOOSE_MODES:
                p: dict[str, Any] = {"idempotency": key, "choose": mode}
                if c is not None:
                    p["crash"] = c
                out.append(D.run_durable(p))
    for lost in [{"charge_card": [1]}, {"charge_card": [1, 2]}, {"charge_card": [1, 2, 3]}, {"send_receipt": [1]}]:
        for key in [False, True]:
            out.append(D.run_durable({"lost": lost, "idempotency": key, "choose": "marker"}))
    return out


def orchestration_fixtures() -> tuple[dict[str, Any], str, dict[str, Any]]:
    sessions = []
    for s in SESSIONS:
        r = run_session(graph(s["graph"]), s["ops"])
        sessions.append({"name": s["name"], "graph": s["graph"], "ops": r["ops"]})
    layouts = {g["name"]: V.layout(g) for g in GRAPHS}
    supersteps = {s["name"]: V.session_supersteps(graph(s["graph"]), s) for s in SESSIONS}
    timelines = {s["name"]: V.timeline_frames(graph(s["graph"]), s) for s in SESSIONS}
    scen = {s["name"]: D.run_durable(s["params"]) for s in D.SCENARIOS}
    grid = durable_grid()
    sweeps = [D.charge_sweep(q, a, SWEEP_RUNS, SWEEP_SEED) for q in SWEEP_Q for a in SWEEP_A]
    g0 = graph("router_view")
    fx = {
        "engine": VERSION,
        "names": [V.names(x) for x in [[], ["a"], ["a", "b"], ["a", "b", "c"]]],
        "fmt": [V._fmt(x) for x in [1, "x", [1, "a"], {"k": [True, None]}, {}, []]],
        "defaults": {g["name"]: default_values(g) for g in GRAPHS},
        "apply": [apply_writes(g0, {"log": []}, [["p", {"log": ["p"], "n": 1}], ["q", {"x": 7}]])],
        "routes": [route({"if": {"len": "log", "op": ">=", "value": 2}, "then": {"goto": ["a", "b"]}, "else": "c"}, st)
                   for st in [{"log": [1, 2]}, {"log": [1]}, {}]],
        "sessions": sessions,
        "layouts": layouts,
        "supersteps": supersteps,
        "timelines": timelines,
        "durable": scen,
        "durable_grid": grid,
        "sweeps": sweeps,
    }
    data = {"version": VERSION, "graphs": GRAPHS, "sessions": SESSIONS, "workflow": D.WORKFLOW,
            "durable_scenarios": D.SCENARIOS, "consts": {"ckpt_ms": V.CKPT_MS, "timeout_ms": D.TIMEOUT_MS,
                                                        "restart_ms": D.RESTART_MS, "replay_ms": D.REPLAY_MS,
                                                        "max_attempts": D.MAX_ATTEMPTS}}
    return fx, results_md(supersteps, timelines, scen, sweeps), data


def _f(x: float, d: int = 3) -> str:
    return fixed(x, d)


def results_md(supersteps: dict[str, Any], timelines: dict[str, Any], scen: dict[str, Any], sweeps: list[dict[str, Any]]) -> str:
    rec_path = ROOT / "fixtures" / "langgraph_recordings.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"langgraph": "?", "langgraph_checkpoint": "?", "sessions": []}
    n_ops = sum(len(s["ops"]) for s in rec["sessions"])
    n_cps = sum(len(s["ops"][-1]["history"]) for s in rec["sessions"])
    out = [f"# Orchestration module results (engine {VERSION})", "",
           "Written by `scripts/make_fixtures.py` (CI checks it is up to date). Node durations, the checkpoint write "
           f"({V.CKPT_MS} ms) and every durable-execution timing are illustrative; the semantics are checked against "
           "LangGraph.", "",
           "## Conformance with LangGraph", "",
           f"`fixtures/langgraph_recordings.json`: langgraph {rec['langgraph']} (langgraph-checkpoint "
           f"{rec['langgraph_checkpoint']}), {len(rec['sessions'])} sessions, {n_ops} operations, {n_cps} checkpoints in the "
           "final histories. CI re-records them live and requires live = committed = engine for every checkpoint after "
           "every operation.", "",
           "## Super-steps of the teaching graphs", "",
           "| graph | super-steps | graph time (ms) | one node at a time (ms, a checkpoint per node) | speed-up |", "|---|---|---|---|---|"]
    for n in TEACH:
        r = supersteps[n][0]
        out.append(f"| {n} | {r['steps']} | {r['graph_ms']} | {r['sequential_ms']} | {_f(r['sequential_ms'] / r['graph_ms'], 2)} |")
    out += ["", "## Two branches, one key", "",
            "| reducer | status | answer after the join |", "|---|---|---|"]
    for n in MERGES:
        r = supersteps[n][0]
        last = r["frames"][-1]
        ans = last["values"].get("answer") if r["status"] == "done" else "InvalidUpdateError"
        out.append(f"| {n.split('_')[1]} | {r['status']} | {json.dumps(ans)} |")
    out += ["", "## Checkpoint sessions", "",
            "| session | operations | statuses | checkpoints | node runs | side effects |", "|---|---|---|---|---|---|"]
    for n in TIMELINES:
        t = timelines[n]
        runs = sum(len(o["calls"]) for o in t["ops"])
        eff = sum(len(o["effects"]) for o in t["ops"])
        out.append(f"| {n} | {len(t['ops'])} | {', '.join(o['status'] for o in t['ops'])} | {len(t['ops'][-1]['history'])} | "
                   f"{runs} | {eff} |")
    out += ["", "## Durable execution scenarios", "",
            "| scenario | status | charges | £ charged | receipts | events | finished at (ms) |", "|---|---|---|---|---|---|---|"]
    for n, r in scen.items():
        out.append(f"| {n} | {r['status']} | {r['charges']} | {r['charged']} | {len(r['emails'])} | {len(r['history'])} | {r['t_end']} |")
    out += ["", f"## Lost replies and retries ({SWEEP_RUNS} orders per row, seed {SWEEP_SEED})", "",
            "| q | attempts | mean charges, no key | closed form (1-q^A)/(1-q) | P(charged twice or more) | closed form q | "
            "P(all attempts lost) | closed form q^A | mean charges, key |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in sweeps:
        out.append(f"| {s['q']} | {s['attempts']} | {_f(s['mean_no_key'], 4)} | {_f(s['expected_no_key'], 4)} | "
                   f"{_f(s['dup_rate'], 4)} | {_f(s['expected_dup'], 4)} | {_f(s['failed'] / s['runs'], 4)} | "
                   f"{_f(s['expected_failed'], 4)} | {_f(s['mean_key'], 4)} |")
    return "\n".join(out) + "\n"
