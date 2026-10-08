"""Engine 1.5.0's additions to the context module: packing (the DP is optimal, checked by brute
force), the position curve, the lossy summariser, the compaction study, memory across sessions and
the long-context against retrieval arithmetic."""
from __future__ import annotations

import itertools

import pytest

from agent_loop_sim.accounting import PRICES, call_cost
from agent_loop_sim.context import memory as M
from agent_loop_sim.context import packing as P
from agent_loop_sim.context import tradeoff as T
from agent_loop_sim.context.corpus import DEFAULT, default_corpus
from agent_loop_sim.context.retrieval import Retriever
from agent_loop_sim.context.window import compaction_study, task_questions, window_run


@pytest.fixture(scope="module")
def d() -> Retriever:
    return Retriever(default_corpus(), DEFAULT)


def test_gain_is_ndcg_discount() -> None:
    assert P.gain(1) == 1.0
    assert P.gain(3) == pytest.approx(0.5)
    assert all(P.gain(i) > P.gain(i + 1) for i in range(1, 20))


def test_position_curve_ends_and_trough() -> None:
    pc = P.POSITION
    assert P.position_p(0.0) == pc["start"]
    assert P.position_p(pc["trough"]) == pc["middle"]
    assert P.position_p(1.0) == pc["end"]
    curve = P.position_curve()
    assert min(p for _, p in curve) == pc["middle"]


def test_knapsack_is_optimal_by_brute_force(d: Retriever) -> None:
    for qi in [0, 5, 17, 61]:
        cands = P.candidates(d, qi, 10)
        for b in [200, 400, 700]:
            best, take = P.knapsack_table(cands, b)
            pick = P.knapsack_pick(cands, take, b)
            brute = 0.0
            for n in range(len(cands) + 1):
                for s in itertools.combinations(range(len(cands)), n):
                    if sum(cands[i]["tokens"] for i in s) <= b:
                        brute = max(brute, sum(cands[i]["value"] for i in s))
            assert sum(cands[i]["tokens"] for i in pick) <= b
            assert sum(cands[i]["value"] for i in pick) == pytest.approx(brute)
            assert best[len(cands)][b] == pytest.approx(brute)
            for p in ["top", "density"]:
                g = P.greedy(cands, b, p)
                assert sum(cands[i]["tokens"] for i in g["chosen"]) <= b
                assert sum(cands[i]["value"] for i in g["chosen"]) <= brute + 1e-12


def test_placements_are_permutations(d: Retriever) -> None:
    cands = P.candidates(d, 0)
    ch = [0, 1, 2, 3, 4]
    assert P.place(cands, ch, "best-first") == [0, 1, 2, 3, 4]
    assert P.place(cands, ch, "best-last") == [4, 3, 2, 1, 0]
    assert P.place(cands, ch, "ends") == [0, 2, 4, 3, 1]
    assert P.place(cands, ch, "middle") == [4, 2, 0, 1, 3]
    xs = P.positions(cands, [0, 1, 2])
    assert xs == sorted(xs) and 0 < xs[0] < xs[-1] < 1


def test_packing_eval_more_budget_more_answers(d: Retriever) -> None:
    e = P.packing_eval(d, [256, 1024])["results"]
    for p in P.PACKERS:
        assert e["1024"][p]["answered"] >= e["256"][p]["answered"]
        assert e["1024"][p]["tokens"] <= 1024
    assert e["1024"]["optimal"]["value"] >= e["1024"]["top"]["value"]
    assert e["1024"]["optimal"]["value"] >= e["1024"]["density"]["value"]


def test_lossless_lossy_equals_perfect(d: Retriever) -> None:
    task = task_questions(d, 12)
    a = window_run(d, task, 1500, "compact")
    b = window_run(d, task, 1500, "compact", "lossy", 0.0)
    assert b["recalled"] == a["recalled"] and b["spent"] == a["spent"]
    assert [f["caption"] for f in a["frames"]] == [f["caption"] for f in b["frames"]]
    with pytest.raises(ValueError):
        window_run(d, task, 1500, "compact", "magic")


def test_truncate_caption_names_the_question(d: Retriever) -> None:
    task = task_questions(d, 12)
    caps = [f["caption"] for f in window_run(d, task, 1500, "truncate")["frames"] if f["event"] == "truncate"]
    assert any("(the fact for question 1 is lost)" in c for c in caps)
    assert any("(it held no fact)" in c for c in caps)
    assert not any("facts lost" in c for c in caps)


def test_compaction_study_tracks_the_model(d: Retriever) -> None:
    task = task_questions(d, 36)
    s = compaction_study(d, task, 1500, "compact", 0.25, list(range(1, 21)))
    assert s["survival"][0]["measured"] == pytest.approx(0.75, abs=0.03)
    assert s["survival"][1]["model"] == pytest.approx(0.5625)
    assert 0 < s["recalled"] < 36
    r = compaction_study(d, task, 1500, "compact+retrieve", 0.25, list(range(1, 6)))
    assert r["recalled"] == 36  # re-retrieval restores every fact


def test_recency_is_repeated_multiplication() -> None:
    assert M.recency(0) == 1.0
    assert M.recency(24) == pytest.approx(0.995 ** 24)


def test_memory_policies(d: Retriever) -> None:
    plan = M.memory_plan(d)
    assert sum(len(p) for p in plan["probes"]) == M.SESSIONS * M.PER_SESSION + sum(
        1 for row in plan["probes"] for p in row if p["kind"] == "neighbour")
    runs = {n: M.memory_run(d, p, plan, n) for n, p in M.POLICIES.items()}
    assert runs["none"]["recalled"] == 0 and runs["none"]["read_tokens"] == 0
    assert runs["transcript"]["recalled"] == runs["transcript"]["probes"]
    # the scratchpad answers exactly the repeats it has notes for, never a neighbour
    assert runs["scratchpad"]["neighbour"][0] == 0
    # forgetting caps the store
    assert runs["episodic-cap"]["items"] <= 8 and runs["semantic-cap"]["items"] <= 10
    assert runs["scratchpad-cap"]["stored_tokens"] <= 128
    # semantic memory reads fewer tokens than the transcript and the episodes
    assert runs["semantic"]["read_tokens"] < runs["episodic"]["read_tokens"] < runs["transcript"]["read_tokens"]
    for w in runs.values():
        assert [f["step"] for f in w["frames"]] == list(range(len(w["frames"])))


def test_tradeoff_by_hand(d: Retriever) -> None:
    sz = T.measured_sizes(d)
    run = T.run_questions(sz, "claude-sonnet-4.6", 1000000, 5, 3)
    p = PRICES["claude-sonnet-4.6"]
    q = int(sz["question"] + 0.5)
    inp = sz["system"] + 1000000 + q
    assert run["strategies"]["long"]["cost"][0] == call_cost(p, inp, 0, False, T.OUT_TOKENS)
    assert run["strategies"]["long+cache"]["cost"][0] == call_cost(p, inp, 0, True, T.OUT_TOKENS)
    assert run["strategies"]["long+cache"]["cached"][1] == sz["system"] + 1000000
    rag_in = sz["system"] + 5 * int(sz["chunk"] + 0.5) + q
    assert run["strategies"]["rag"]["input"] == [rag_in] * 3
    assert run["strategies"]["rag"]["cum"][-1] < run["strategies"]["long+cache"]["cum"][-1] < run["strategies"]["long"]["cum"][-1]
