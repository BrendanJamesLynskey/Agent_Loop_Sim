"""The orchestration module's second part (engine 1.7.0): Send in the graph runtime, multi-agent
patterns, the discrete-event simulation and the reliability maths. The closed forms are tested
against the simulations."""
from __future__ import annotations

import math

import pytest

from agent_loop_sim.orchestration import des as D
from agent_loop_sim.orchestration import patterns as P
from agent_loop_sim.orchestration import reliability as R
from agent_loop_sim.orchestration import study as S
from agent_loop_sim.orchestration.graph import Thread, route
from agent_loop_sim.orchestration.scenarios import graph


# --- Send -------------------------------------------------------------------------------------

def test_send_packets_one_per_item():
    leaf = {"goto": [{"send": "w", "over": "xs", "as": "x", "with": ["tag"]}, "z"]}
    assert route(leaf, {"xs": [1, 2], "tag": "t"}) == [{"send": "w", "arg": {"x": 1, "tag": "t"}},
                                                         {"send": "w", "arg": {"x": 2, "tag": "t"}}, "z"]
    assert route({"goto": {"send": "w", "over": "xs", "as": "x"}}, {}) == []


def test_map_reduce_push_tasks_and_write_order():
    th = Thread(graph("map_reduce"))
    r = th.invoke({"log": []})
    assert r["status"] == "done"
    h = th.history()
    assert h[2]["next"] == ["summarise"] * 4
    assert r["values"]["summaries"] == ["ch1", "ch2", "ch3", "ch4"]
    assert r["values"]["count"] == 4
    assert r["steps"][2]["sends"] == [{"name": "summarise", "arg": {"chunk": c}} for c in ["ch1", "ch2", "ch3", "ch4"]]


def test_mixed_push_after_pull():
    th = Thread(graph("send_mixed"))
    r = th.invoke({"log": []})
    # pull tasks (w then z, by name) write before push tasks (in packet order: a's, then b's)
    assert r["values"]["log"][:7] == ["a", "b", "w", "z", "x", "y", "b"]
    assert th.history()[2]["next"] == ["w", "w", "w", "w", "z"]


def test_crash_in_one_push_task_reruns_only_it():
    th = Thread(graph("map_reduce"))
    r = th.invoke({"log": []}, fail=["summarise"])
    assert r["status"] == "error" and r["calls"].count("summarise") == 4
    r2 = th.invoke(None)
    assert r2["status"] == "done" and r2["calls"] == ["summarise", "reduce"]
    assert r2["values"]["summaries"] == ["ch1", "ch2", "ch3", "ch4"]


# --- patterns ---------------------------------------------------------------------------------

def test_plans_are_dags_and_counts():
    for n in P.PATTERNS:
        p = P.plan(n)
        for c in p["calls"]:
            assert all(d < c["id"] for d in c["deps"])
    k = len(P.TASK["subtasks"])
    calls = {m["pattern"]: m["model_calls"] for m in P.compare()}
    assert calls == {"single": k + 1, "supervisor": 3 * k + 1, "hierarchical": 3 * k + 5, "swarm": 2 * k,
                     "debate": k + 2 * P.TASK["debaters"] + 1, "map_reduce": 2 * k + 2}


def test_success_closed_form_by_hand():
    p = P.plan("debate")
    q = [c["p"] for c in p["calls"] if c["role"] == "vote"]
    crit = 1.0
    for c in p["calls"]:
        if c["role"] == "critical":
            crit *= c["p"]
    maj = q[0] * q[1] + q[0] * q[2] + q[1] * q[2] - 2 * q[0] * q[1] * q[2]
    assert P.success_p(p) == pytest.approx(crit * maj, rel=1e-12)


def test_uncontended_latency_is_critical_path():
    m = {x["pattern"]: x for x in P.compare()}
    for n in ["single", "supervisor", "hierarchical", "swarm"]:
        assert m[n]["latency_ms"] == pytest.approx(m[n]["serial_ms"])  # sequential patterns
    assert m["map_reduce"]["latency_ms"] < m["map_reduce"]["serial_ms"] / 1.5


def test_no_context_penalty_single_agent_is_most_likely_to_succeed():
    for g in S.patterns_grid([4, 8, 16], [700, 1500], [0.0]):
        best = max(g["rows"], key=lambda m: m["success"])
        assert best["pattern"] in ("single", "debate")


# --- the discrete-event simulation ------------------------------------------------------------

@pytest.mark.parametrize("name", P.PATTERNS)
def test_des_success_matches_closed_form(name):
    p = P.plan(name)
    r = D.simulate(p, {"workflows": 3000, "per_min": 1.0, "fail": 0.1, "max_attempts": 2, "timeout_ms": 1e12, "seed": 21})
    exp = D.success_closed(p, 0.1, 2)
    se = math.sqrt(exp * (1 - exp) / 3000)
    assert abs(r["success"] - exp) < 4 * se


def test_des_attempts_match_closed_form():
    p = P.plan("single")
    r = D.simulate(p, {"workflows": 3000, "per_min": 2.0, "fail": 0.3, "max_attempts": 3, "timeout_ms": 1e12,
                       "semantic": False, "seed": 4})
    done = [x for x in r["runs"] if x["status"] != "exhausted"]
    # among completed workflows each call took 1 + f + f^2 attempts conditioned on succeeding by the 3rd
    f = 0.3
    cond = (1 * (1 - f) + 2 * f * (1 - f) + 3 * f * f * (1 - f)) / (1 - f ** 3)
    mean = sum(x["attempts"] for x in done) / len(done) / r["model_calls"]
    assert mean == pytest.approx(cond, abs=0.03)
    assert D.expected_attempts(f, 3) == pytest.approx(1 + f + f * f)


def test_des_throughput_saturates_at_capacity():
    p = P.plan("supervisor")
    cap = S.capacity_per_min(p, D.params({}))
    r = D.simulate(p, {"workflows": 300, "per_min": 3 * cap["min"], "fail": 0.0, "seed": 2})
    assert r["throughput_per_min"] / r["success"] == pytest.approx(cap["min"], rel=0.12)
    light = D.simulate(p, {"workflows": 300, "per_min": 0.3 * cap["min"], "fail": 0.0, "seed": 2})
    assert light["p95"] < r["p95"] / 10


def test_littles_law_identity():
    r = D.simulate(P.plan("map_reduce"), {"workflows": 80, "per_min": 2.0, "trace": True, "fail": 0.0, "seed": 8})
    ch = r["changes"]
    area = 0.0
    for a, b in zip(ch, ch[1:]):
        area += a[3] * (b[0] - a[0])
    lat = sum(x["end"] - x["arrival"] for x in r["runs"])
    assert area == pytest.approx(lat, rel=1e-9)


def test_rate_limit_never_exceeded():
    plan = P.plan_map(32, 2000)
    r = D.simulate(plan, S.fan_params(32, 50, 40000, 8000))
    from agent_loop_sim.accounting import PRICES
    price = PRICES[D.DEFAULTS["price"]]
    starts = sorted((x["start"], x["call"]) for x in r["trace"] if x["outcome"] == "ok")
    assert len(starts) == 34
    # uncached input admitted in any window of length T never exceeds the bucket (40,000) + T's refill
    for i, (t0, _) in enumerate(starts):
        for t1, _ in starts[i:]:
            win = [c for t, c in starts if t0 <= t <= t1]
            need = sum(plan["calls"][c]["input"] - P.cached_tokens(plan["calls"][c], price) for c in win)
            assert need <= 40000 + (t1 - t0) * 40000 / 60000 + 1e-6


def test_fan_speedup_equals_amdahl_without_limits():
    f = S.fan_study(32, 2000, [1, 2, 4, 8, 16, 32], 100000, 10 ** 9, 10 ** 9)
    for r in f["rows"]:
        assert r["speedup"] == pytest.approx(r["amdahl"], rel=1e-9)


def test_fan_speedup_below_ceiling():
    for it in [40000, 80000, 200000]:
        f = S.fan_study(32, 2000, S.FAN["widths"], 50, it, 8000)
        for r in f["rows"]:
            assert r["speedup"] <= r["amdahl"] + 1e-9
            if r["ceiling"]:
                assert r["speedup"] <= r["ceiling"] + 1e-9


def test_des_is_deterministic():
    p = P.plan("swarm")
    a = D.simulate(p, {"workflows": 30, "seed": 3, "trace": True})
    b = D.simulate(p, {"workflows": 30, "seed": 3, "trace": True})
    assert a == b


# --- reliability ------------------------------------------------------------------------------

@pytest.mark.parametrize("verifier", [None, {"recall": 0.8, "false_reject": 0.05}])
@pytest.mark.parametrize("attempts", [1, 3])
@pytest.mark.parametrize("n", [1, 5, 20])
def test_reliability_closed_form_matches_simulation(n, attempts, verifier):
    s = R.simulate(n, 0.05, 0.03, attempts, verifier, 6000, 100 + n)
    exp = s["closed"]["success"]
    se = math.sqrt(exp * (1 - exp) / 6000) or 1e-9
    assert abs(s["rate"] - exp) < 4 * se + 1e-12
    assert s["mean_attempts"] == pytest.approx(s["closed"]["attempts"], rel=0.03)


def test_reliability_identities():
    c = R.closed(1, 0.1, 0.05, 4, None)
    assert c["step_ok"] + c["step_wrong"] + c["step_give_up"] == pytest.approx(1.0)
    assert R.closed(7, 0.0, 0.0, 1)["success"] == 1.0
    assert R.closed(3, 0.2, 0.0, 1)["success"] == pytest.approx(0.8 ** 3)
    assert R.closed(3, 0.2, 0.0, 2)["success"] == pytest.approx((1 - 0.2 ** 2) ** 3)
    # retries cannot fix silent errors; a verifier can
    assert R.closed(10, 0.0, 0.05, 5)["success"] == pytest.approx(0.95 ** 10)
    assert R.closed(10, 0.0, 0.05, 5, {"recall": 0.9, "false_reject": 0.0})["success"] > 0.9


def test_wilson():
    lo, hi = R.wilson(731, 1000)
    assert lo == pytest.approx(0.7028, abs=1e-3) and hi == pytest.approx(0.7575, abs=1e-3)
    assert R.wilson(0, 10)[0] == 0.0 and R.wilson(10, 10)[1] == 1.0


# --- chooser ----------------------------------------------------------------------------------

def test_choose_steps():
    tab = [{"pattern": "a", "success": 0.9, "p95": 10.0, "cost_per_success": 1.0, "capacity": 5.0},
           {"pattern": "b", "success": 0.9, "p95": 5.0, "cost_per_success": 0.5, "capacity": 1.0},
           {"pattern": "c", "success": 0.5, "p95": 1.0, "cost_per_success": 0.1, "capacity": 9.0}]
    r = S.choose(tab, {"success": 0.8})
    assert r["alive"] == ["a", "b"] and r["pick"] == "b"
    r = S.choose(tab, {"success": 0.8, "capacity": 2.0})
    assert r["pick"] == "a" and r["steps"][1]["struck"] == ["b"]
    assert S.choose(tab, {"p95": 0.5})["pick"] is None
