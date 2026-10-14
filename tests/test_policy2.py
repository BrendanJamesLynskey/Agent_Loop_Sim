"""The policy module, part two (engine 1.11.0): the benign suite; plans that branch on a value (``ctrl``) and runs
with the policy off (``enforce``); the utility-cost model's closed forms against a seeded simulation; orderings
the model must respect (least privilege never adds risk, an adaptive attacker never lowers it, more prompts make
each approval worse); the Pareto front against its definition; chapters 6 to 10's studies; the committed
fixtures."""
from __future__ import annotations

import json
import math
from pathlib import Path

from agent_loop_sim import VERSION
from agent_loop_sim.policy import capabilities as C
from agent_loop_sim.policy import catalogue as K
from agent_loop_sim.policy import depth as D
from agent_loop_sim.policy import engine as E
from agent_loop_sim.policy import labels as L
from agent_loop_sim.policy import study as T
from agent_loop_sim.policy import utility as U
from agent_loop_sim.rng import Rng

ROOT = Path(__file__).resolve().parent.parent
SUITE = U.make_suite()


# ── the suite ────────────────────────────────────────────────────────────


def test_suite_is_benign_well_formed_and_seeded():
    assert len(SUITE) == K.SUITE["n"]
    assert U.make_suite() == SUITE
    assert U.make_suite(dict(K.SUITE, seed=K.SUITE["seed"] + 1)) != SUITE
    ids = set()
    for p in SUITE:
        ids.add(p["id"])
        calls = [s for s in p["steps"] if s["op"] == "call"]
        reads = [s["tool"] for s in calls if K.TOOLS[s["tool"]]["kind"] == "read"]
        assert K.SUITE["min_reads"] <= len(reads) <= K.SUITE["max_reads"]
        assert len(set(reads)) == len(reads) and all(t in K.SUITE["read_tools"] for t in reads)
        assert calls[len(reads)]["tool"] == "summarise"
        assert len(calls) in (len(reads) + 1, len(reads) + 2)
        if p["adaptive"]:
            assert calls[-1]["ctrl"] == [calls[len(reads) - 1]["out"]]
        assert all("ctrl" not in s for s in calls[:-1])
        # every task runs with no policy, end to end
        assert all(r["ran"] for r in E.run_plan(p, K.BROAD, "context", True, False))
    assert len(ids) == len(SUITE)
    assert 0 < sum(p["adaptive"] for p in SUITE) < len(SUITE)


def test_no_content_only_labels():
    blob = json.dumps({"suite": SUITE, "spec": K.SUITE, "defences": K.DEFENCES, "adaptive": K.ADAPTIVE}).lower()
    for w in ["ignore", "instruction", "password", "<", "http://", "https://"]:
        assert w not in blob


# ── ctrl and enforce ─────────────────────────────────────────────────────


def test_ctrl_controls_a_planned_call():
    for p in SUITE + [K.ADAPTIVE]:
        g = C.needed(p)
        steps = E.run_plan(p, g, "dataflow")
        vals = {}
        for st, r in zip(p["steps"], steps):
            if r["op"] == "call" and r["decision"] != "skip":
                want = L.join_all([vals[a] for a in st.get("ctrl", [])])
                assert r["control"] == want
                assert r["effective"] == L.join_all([vals[a] for a in st["args"]] + [want])
            if r["out_label"] is not None and st.get("out"):
                vals[st["out"]] = r["out_label"]
        # dropping the branch never makes a decision stricter
        flat = {"steps": [{k: v for k, v in s.items() if k != "ctrl"} for s in p["steps"]]}
        for a, b in zip(steps, E.run_plan(flat, g, "dataflow")):
            if a["op"] == "call" and a["decision"] != "skip" and b["decision"] != "skip":
                assert E.RANK[b["decision"]] <= E.RANK[a["decision"]]
                assert L.leq(b["control"], a["control"])


def test_a_missing_branch_value_skips():
    plan = {"steps": [{"op": "input", "out": "q", "sens": "public"},
                      {"op": "call", "tool": "write_file", "args": ["q"], "ctrl": ["nope"], "out": None}]}
    assert E.run_plan(plan, K.BROAD, "dataflow")[1]["decision"] == "skip"


def test_enforce_off_runs_everything_and_records_the_decisions():
    for p in list(K.PLANS.values()) + [K.ADAPTIVE] + SUITE[:40]:
        for mode in ["context", "dataflow"]:
            off = E.run_plan(p, K.BROAD, mode, True, False)
            assert all(r["ran"] for r in off)
            for st, r in zip(p["steps"], off):
                if r["op"] == "call":
                    assert E.decide(st["tool"], r["effective"], st.get("dest"), K.BROAD)["decision"] == r["decision"]
                    assert ("not enforced" in r["why"]) == (r["decision"] != "allow")
            on = E.run_plan(p, K.BROAD, mode, True, True)
            if all(r["ran"] for r in on):
                assert [r["effective"] for r in on] == [r["effective"] for r in off]


# ── the cost and risk model ──────────────────────────────────────────────


def test_approval_miss_closed_form_and_fatigue():
    a1, r = K.UTILITY["approval_first_miss"], K.UTILITY["approval_decay"]
    prev = 0.0
    for n in range(1, 30):
        x = U.approval_miss(n)
        assert abs(x - (1 - (1 - a1) * r ** (n - 1))) < 1e-12
        assert x > prev
        prev = x
    assert U.approval_miss(1) == a1 or abs(U.approval_miss(1) - a1) < 1e-15


def test_evaluate_closed_forms():
    for d in K.DEFENCES:
        m = U.layer_miss(d["layers"])
        det = D.all_miss_common(m, d["c"]) if m else 1.0
        for p in SUITE:
            row = U.evaluate(p, d)
            assert row["detect"] == det
            assert math.isclose(row["risk"], K.DEPTH["base"] * det * row["reach"], rel_tol=0, abs_tol=1e-15)
            assert row["reach"] == max([x["pass"] for x in row["routes"]], default=0.0)
            fp = 1.0
            for lid in d["layers"]:
                n = row["consequential"] if lid == "output_check" else row["untrusted"]
                fp *= (1 - K.UTILITY["fp"][lid]) ** n
            assert math.isclose(row["keep"], 0.0 if row["blocked"] else fp, abs_tol=1e-12)
            if row["untrusted"] == 0:
                assert row["risk"] == 0.0
            if not d["policy"]:
                assert not row["blocked"]
            for x in row["routes"]:
                # a route the policy denies is closed; a human-gated route passes at the fatigue rate
                if d["policy"] and x["decision"] == "deny":
                    assert x["pass"] == 0.0
                if x["human"]:
                    assert 0.0 < x["pass"] < 1.0


def test_simulation_matches_the_closed_form():
    for d in K.DEFENCES:
        for adaptive in [False, True]:
            base = K.DEPTH["base_range"][1] if adaptive else K.DEPTH["base"]
            rows = U.run_defence(d, SUITE, base, K.UTILITY, adaptive)["rows"]
            sm = U.summarise(rows)
            s = U.simulate(rows, d, base, 300, 77, adaptive)
            n = s["n"]
            # tasks differ, so the per-draw variance is at most p(1 - p) of the mean
            for got, want in [(s["risk"], sm["risk"]), (s["utility"], sm["utility"])]:
                assert abs(got - want) <= 4 * math.sqrt(max(want * (1 - want), 1e-9) / n) + 1e-12


def test_orderings():
    pol, least, none = U.defence("policy"), U.defence("least"), U.defence("none")
    for p in SUITE:
        a, b = U.evaluate(p, pol), U.evaluate(p, least)
        # least privilege blocks no benign task the broad grant allows, and never adds risk
        assert a["blocked"] == b["blocked"] and b["risk"] <= a["risk"]
        assert U.evaluate(p, none)["risk"] >= a["risk"]
    for d in K.DEFENCES:
        f = U.run_defence(d, SUITE)["summary"]
        g = U.run_defence(d, SUITE, K.DEPTH["base_range"][1], K.UTILITY, True)["summary"]
        assert g["risk"] >= f["risk"] and g["utility"] == f["utility"] and g["prompts"] == f["prompts"]
    th = [U.run_defence(U.with_threshold(pol, t), SUITE)["summary"] for t in K.UTILITY["thresholds"]]
    assert [x["prompts"] for x in th] == sorted(x["prompts"] for x in th)
    assert th[2]["ungated"] == 0.0 and th[3]["ungated"] == 0.0
    assert th[0]["utility"] < th[1]["utility"]


def test_pareto_matches_its_definition():
    r = Rng(9)
    for _ in range(200):
        pts = [{"id": f"p{i}", "utility": r.randint(0, 4) / 4, "risk": r.randint(0, 4) / 4,
                "prompts": r.randint(0, 2)} for i in range(r.randint(1, 9))]
        front = U.pareto(pts)

        def dominates(b, a):
            ge = b["utility"] >= a["utility"] and b["risk"] <= a["risk"] and b["prompts"] <= a["prompts"]
            return ge and (b["utility"], -b["risk"], -b["prompts"]) != (a["utility"], -a["risk"], -a["prompts"])
        assert front == [a["id"] for a in pts if not any(dominates(b, a) for b in pts if b is not a)]
        assert front


# ── the chapters' studies ────────────────────────────────────────────────


def test_depth_frames():
    for c in K.DEPTH["c"]:
        d = T.depth_frames(c)
        counts = [f["count"] for f in d["frames"]]
        assert counts == sorted(counts, reverse=True) and counts[0] == K.DEPTH_DOTS["n"]
        assert all(f["caption"] for f in d["frames"])
    m = [x["miss"] for x in K.DEPTH["layers"]]
    for c in [0.0, 0.02]:
        n = 40000
        d = T.depth_frames(c, None, K.DEPTH, {"n": n, "seed": 3})
        for f in d["frames"][1:]:
            p = K.DEPTH["base"] * (D.all_miss_common(m[:f["f"] - 1], c) if f["f"] > 1 else 1.0)
            assert abs(f["count"] / n - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-9
            assert math.isclose(f["expected"], n * p, rel_tol=1e-12)


def test_dual_frames():
    for pid in K.DUAL_PLANS:
        d = T.dual_frames(pid)
        plan = T.dual_plan(pid)
        assert d["frames"][0]["actor"] == "planner" and d["frames"][0]["planner_sees"] == ["q"]
        for st, f in zip(plan["steps"], d["frames"]):
            x = f["flow"]
            if x["op"] == "call" and "ctrl" not in st:
                assert x["control"] == L.BOTTOM
            if x["op"] == "call":
                assert f["actor"] == ("quarantined" if K.TOOLS[st["tool"]]["kind"] == "transform" else "tool")
                # a planned call is never more restricted than the classic agent's
                if x["decision"] != "skip" and f["classic"]["decision"] != "skip":
                    assert E.RANK[x["decision"]] <= E.RANK[f["classic"]["decision"]]
    assert "branches on t" in T.dual_frames("adaptive")["frames"][-1]["caption"]


def test_approval_and_frontier_studies():
    a = T.approval_study(SUITE)
    assert [len(a["thresholds"][t]["frames"]) for t in K.UTILITY["thresholds"]] == [K.UTILITY["frames"]] * 4
    fs = T.frontier_study(SUITE)
    assert fs["fixed"]["front"] and fs["adaptive"]["front"]
    assert "planned" in fs["adaptive"]["front"]
    assert len(fs["fixed"]["frames"]) == len(K.DEFENCES)


def test_checklist():
    best = T.checklist(T.CHECK_DEFAULT, SUITE)
    assert len(best["items"]) == 9 and [i["ch"] for i in best["items"]] == list(range(1, 10))
    for did in K.DEFENCE_ORDER:
        c = T.checklist({"defence": did, "approval": None, "sandbox": T.CHECK_DEFAULT["sandbox"]}, SUITE)
        assert c["passed"] <= best["passed"]
        assert c["frames"][-1]["passed"] == c["passed"]
    none = T.checklist({"defence": "none", "approval": None, "sandbox": []}, SUITE)
    assert none["items"][0]["verdict"] == "fail" and none["items"][4]["verdict"] == "fail"


def test_fixtures_up_to_date():
    fx = json.loads((ROOT / "fixtures" / "policy2_fixtures.json").read_text(encoding="utf-8"))
    assert fx["engine"] == VERSION == "1.11.1"
    assert fx["suite"] == SUITE
