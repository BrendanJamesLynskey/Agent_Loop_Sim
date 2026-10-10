"""The orchestration module (engine 1.6.0): the graph runtime against the committed LangGraph
recording, its rules one by one, the views, durable execution, and the closed forms against the
seeded simulation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agent_loop_sim.orchestration import durable as D
from agent_loop_sim.orchestration import views as V
from agent_loop_sim.orchestration.graph import (GraphError, InvalidUpdateError, Thread, apply_writes,
                                                 comparable_history, outcome, run_session, validate)
from agent_loop_sim.orchestration.scenarios import GRAPHS, SESSIONS, graph, session

ROOT = Path(__file__).resolve().parent.parent
REC = json.loads((ROOT / "fixtures" / "langgraph_recordings.json").read_text())


def test_recording_is_pinned_langgraph():
    assert REC["langgraph"] == "1.2.14"
    assert [s["name"] for s in REC["sessions"]] == [s["name"] for s in SESSIONS]


@pytest.mark.parametrize("name", [s["name"] for s in SESSIONS])
def test_engine_equals_langgraph_recording(name):
    rec = next(s for s in REC["sessions"] if s["name"] == name)
    s = session(name)
    g = graph(s["graph"])
    r = run_session(g, s["ops"])
    assert len(r["ops"]) == len(rec["ops"])
    for mine, theirs in zip(r["ops"], rec["ops"]):
        assert outcome(mine["result"]) == theirs["outcome"]
        assert comparable_history(g, mine["history"]) == theirs["history"]


def test_comparable_history_only_drops_waiting_join_targets():
    g = graph("uneven_join")
    h = run_session(g, session("uneven_join")["ops"])["ops"][0]["history"]
    c = comparable_history(g, h)
    assert h[3]["next"] == ["c", "e"] and c[3]["next"] == ["c"]
    assert [x for i, x in enumerate(c) if i != 3] == [x for i, x in enumerate(h) if i != 3]


def test_writes_apply_in_node_name_order_not_finishing_order():
    s = session("merge_add")
    r = run_session(graph("merge_add"), s["ops"])["ops"][0]
    # careful (1600 ms) finishes after fast (700 ms) but sorts first
    assert r["result"]["values"]["answer"] == [42, 41]


def test_overwrite_conflict_raises_and_saves_nothing():
    th = Thread(graph("merge_overwrite"))
    r = th.invoke({"log": []})
    assert r["status"] == "error" and r["error"].startswith("InvalidUpdateError")
    assert [c["step"] for c in th.cps] == [-1, 0, 1]
    with pytest.raises(InvalidUpdateError):
        apply_writes(graph("merge_overwrite"), {}, [("a", {"answer": 1}), ("b", {"answer": 2})])


def test_router_sees_own_writes_only():
    r = run_session(graph("router_view"), session("router_view")["ops"])["ops"][0]["result"]
    assert "r" not in r["calls"]  # p's router did not see q's x = 7
    assert r["values"]["x"] == 7


def test_join_waits_and_plain_edges_double_trigger():
    r = run_session(graph("uneven_join"), session("uneven_join")["ops"])["ops"][0]["result"]
    assert r["calls"].count("e") == 1
    r = run_session(graph("no_join"), session("no_join")["ops"])["ops"][0]["result"]
    assert r["calls"].count("e") == 2


def test_interrupt_reruns_the_node_and_keeps_sibling_writes():
    r = run_session(graph("approval"), session("approval")["ops"])["ops"]
    assert r[0]["result"]["status"] == "interrupted"
    assert r[1]["result"]["calls"] == ["ask", "send"]  # check is not run again
    assert r[0]["result"]["effects"] + r[1]["result"]["effects"] == \
        ["ask:notify_reviewer", "ask:notify_reviewer", "send:send_email"]


def test_recursion_limit_counts_steps():
    ok = run_session(graph("counter"), session("counter_ok")["ops"])["ops"][0]["result"]
    edge = run_session(graph("counter"), session("counter_edge")["ops"])["ops"][0]["result"]
    assert ok["status"] == "done" and edge["status"] == "recursion_limit"


def test_validate_rejects_bad_specs():
    g = graph("chain")
    g["edges"].append({"from": "nope", "to": "draft"})
    with pytest.raises(GraphError):
        validate(g)
    g = graph("chain")
    g["state"][0]["reducer"] = "concat"
    with pytest.raises(GraphError):
        validate(g)


@pytest.mark.parametrize("name", ["chain", "fanout", "react_loop", "uneven_join", "loop_join", "merge_add"])
def test_graph_time_is_sum_over_steps_of_slowest_task(name):
    g = graph(name)
    s = session(name)
    r = run_session(g, s["ops"])["ops"][0]["result"]
    ms = {n["name"]: n["ms"] for n in g["nodes"]}
    ms["__start__"] = 0
    expect = sum(max(ms[t] for t in st["tasks"]) + V.CKPT_MS for st in r["steps"])
    v = V.superstep_frames(g, r, run_session(g, s["ops"])["ops"][0]["history"])
    assert v["graph_ms"] == expect
    assert v["sequential_ms"] == sum(ms[t] + V.CKPT_MS for st in r["steps"] for t in st["tasks"])
    assert len(v["frames"]) == 3 * len(r["steps"])


def test_folds_follow_the_reducer():
    g = graph("merge_max")
    f = V.folds(g, {"answer": 40}, [["careful", {"answer": 42}], ["fast", {"answer": 41}]])
    assert [x["value"] for x in f["answer"]] == [42, 42]
    f = V.folds(graph("merge_overwrite"), {}, [["careful", {"answer": 42}], ["fast", {"answer": 41}]])
    assert f["answer"][1]["error"] is True


def test_layout_columns():
    lay = V.layout(graph("uneven_join"))
    assert lay["nodes"]["e"]["col"] == 4 and lay["nodes"]["d"]["col"] == 2
    assert any(e["back"] for e in V.layout(graph("react_loop"))["edges"])


def test_timeline_frames_cover_every_checkpoint():
    for name in ["crash", "approval", "fork", "replay_interrupt"]:
        s = session(name)
        t = V.timeline_frames(graph(s["graph"]), s)
        cps = [f["focus"] for f in t["frames"] if f["kind"] == "checkpoint"]
        assert cps == list(range(len(t["ops"][-1]["history"])))


def _effects(r, name):
    return [f for f in r["frames"] if f["kind"] == "effect" and f["caption"].startswith(name)]


def test_replay_does_not_rerun_completed_activities():
    r = D.run_durable(D.scenario("crash_replay")["params"])
    assert r["status"] == "completed" and r["charges"] == 1
    assert len(_effects(r, "charge_card")) == 1 and len(_effects(r, "reserve_stock")) == 1


def test_crash_inside_an_activity_runs_it_twice():
    r = D.run_durable(D.scenario("crash_in_charge")["params"])
    assert len(_effects(r, "charge_card")) == 2 and r["charges"] == 2
    k = D.run_durable(D.scenario("crash_in_charge_key")["params"])
    assert len(_effects(k, "charge_card")) == 2 and k["charges"] == 1


def test_lost_reply_and_key():
    assert D.run_durable(D.scenario("lost_reply")["params"])["charges"] == 2
    assert D.run_durable(D.scenario("lost_reply_key")["params"])["charges"] == 1


def test_clock_read_breaks_replay_and_marker_fixes_it():
    r = D.run_durable(D.scenario("nondeterministic")["params"])
    assert r["status"] == "nondeterminism" and r["nondet"]["wanted"] == "schedule ship_standard"
    assert r["nondet"]["got"] == "activity_scheduled ship_express"
    assert D.run_durable(D.scenario("deterministic")["params"])["status"] == "completed"
    with pytest.raises(ValueError):
        D.run_durable({"choose": "dice"})


def test_every_crash_point_with_markers_completes_with_one_charge_when_keyed():
    for k in range(0, 11):
        r = D.run_durable({"crash": {"after_event": k}, "choose": "marker", "idempotency": True})
        assert r["status"] == "completed" and r["charges"] == 1 and len(r["emails"]) == 1


@pytest.mark.parametrize("q", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("a", [1, 2, 3, 5])
def test_charge_sweep_matches_closed_forms(q, a):
    s = D.charge_sweep(q, a, 4000, 24)
    n = s["runs"]
    sd_mean = math.sqrt(q / (1 - q) ** 2 / n)
    assert abs(s["mean_no_key"] - (1 - q ** a) / (1 - q)) <= 4 * sd_mean + 1e-12
    if a >= 2:
        assert abs(s["dup_rate"] - q) <= 4 * math.sqrt(q * (1 - q) / n)
    pf = q ** a
    assert abs(s["failed"] / n - pf) <= 4 * math.sqrt(pf * (1 - pf) / n) + 1e-12
    assert s["mean_key"] == 1.0
    assert sum(s["hist"]) == n


def test_graph_names_are_unique():
    assert len({g["name"] for g in GRAPHS}) == len(GRAPHS)
