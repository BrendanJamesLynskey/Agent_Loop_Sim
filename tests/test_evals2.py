"""The evals module's second part (engine 1.9.0): Cohen's κ against statsmodels, the judge-error
correction against its closed form and the delta method against finite differences, the Pareto front
against its definition, the simulated judge's biases and the swap test against their expectations, the
trajectory grader on hand-checked traces, the scaffold study's task suite and model, contamination's
detection rates against their closed forms, and the committed fixtures."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from statsmodels.stats.inter_rater import cohens_kappa as sm_kappa

from agent_loop_sim import VERSION
from agent_loop_sim.evals import contamination as C
from agent_loop_sim.evals import judge as J
from agent_loop_sim.evals import scaffold as F
from agent_loop_sim.evals import stats as S
from agent_loop_sim.evals import trajectory as T
from agent_loop_sim.rng import Rng

ROOT = Path(__file__).resolve().parent.parent
RND = random.Random(29)


# ── graders' statistics ──────────────────────────────────────────────────


@pytest.mark.parametrize("table", [[[20, 5], [10, 15]], [[45, 15], [25, 15]], [[30, 2, 3], [4, 25, 6], [1, 2, 27]],
                                   [[0, 7], [3, 0]], [[12, 3, 0], [2, 9, 4], [1, 1, 8]]])
def test_kappa_matches_statsmodels(table):
    assert S.cohen_kappa(table)["kappa"] == pytest.approx(sm_kappa(table).kappa, abs=1e-14)


def test_kappa_random_tables():
    for _ in range(200):
        k = RND.choice([2, 3])
        t = [[RND.randint(0, 30) for _ in range(k)] for _ in range(k)]
        if sum(map(sum, t)) == 0:
            continue
        ours = S.cohen_kappa(t)
        if ours["pe"] < 1.0:
            assert ours["kappa"] == pytest.approx(sm_kappa(t).kappa, abs=1e-12)


def test_confusion():
    assert S.confusion(["A", "B", "tie", "A"], ["A", "A", "tie", "tie"], J.LABELS) == [[1, 0, 1], [1, 0, 0], [0, 0, 1]]


def test_rogan_gladen_inverts_the_observed_rate():
    for pi in [0.0, 0.1, 0.37, 0.5, 0.8, 1.0]:
        for se, sp in [(0.95, 0.7), (0.8, 0.9), (0.6, 0.6)]:
            assert S.rogan_gladen(J.observed_rate(pi, se, sp), se, sp) == pytest.approx(pi, abs=1e-12)


def test_corrected_rate_variance_is_the_delta_method():
    k, n, sek, sen, spk, spn = 182, 300, 81, 85, 86, 115
    r = S.corrected_rate(k, n, sek, sen, spk, spn)
    p, se, sp = k / n, sek / sen, spk / spn

    def f(a, b, c):
        return (a + c - 1.0) / (b + c - 1.0)

    h = 1e-6
    grads = [(f(p + h, se, sp) - f(p - h, se, sp)) / (2 * h), (f(p, se + h, sp) - f(p, se - h, sp)) / (2 * h),
             (f(p, se, sp + h) - f(p, se, sp - h)) / (2 * h)]
    variances = [p * (1 - p) / n, se * (1 - se) / sen, sp * (1 - sp) / spn]
    v = sum(g * g * w for g, w in zip(grads, variances))
    assert r["stderr"] == pytest.approx(math.sqrt(v), rel=1e-6)


def test_corrected_rate_is_unbiased_enough_by_simulation():
    # true rate 0.4, Se 0.95, Sp 0.7: the corrected estimate averages close to the truth
    pi, se, sp, n, cal = 0.4, 0.95, 0.7, 300, 200
    rng = random.Random(3)
    est = []
    cover = 0
    for _ in range(2000):
        x = [rng.random() < pi for _ in range(n)]
        j = sum((rng.random() < se) if v else (rng.random() < 1 - sp) for v in x)
        cx = [rng.random() < pi for _ in range(cal)]
        cj = [(rng.random() < se) if v else (rng.random() < 1 - sp) for v in cx]
        sek = sum(1 for v, w in zip(cx, cj) if v and w)
        sen = sum(1 for v in cx if v)
        spk = sum(1 for v, w in zip(cx, cj) if not v and not w)
        spn = cal - sen
        r = S.corrected_rate(j, n, sek, sen, spk, spn)
        est.append(r["corrected"])
        cover += r["ci"][0] <= pi <= r["ci"][1]
    assert sum(est) / len(est) == pytest.approx(pi, abs=0.01)
    assert 0.92 <= cover / 2000 <= 0.97


def test_pareto_front_definition():
    for _ in range(300):
        pts = [[RND.choice([0.5, 1.0, 1.5, 2.0, 3.0]), RND.choice([0.1, 0.3, 0.5, 0.7, 0.9])] for _ in range(RND.randint(1, 9))]
        front = S.pareto_front(pts)
        for i, (c, s) in enumerate(pts):
            dominated = any(c2 <= c and s2 >= s and (c2 < c or s2 > s) for j, (c2, s2) in enumerate(pts) if j != i)
            assert (i in front) == (not dominated)
        costs = [pts[i][0] for i in front]
        assert costs == sorted(costs)


# ── the judge ────────────────────────────────────────────────────────────


def test_judge_choice_probabilities():
    ps = J.pairs()
    fair = J.JUDGE["settings"]["fair"]
    pos = J.JUDGE["settings"]["position"]
    for it in ps[:50]:
        a, b = it["A"], it["B"]
        assert J.prefer_first(a, b, fair, 0.3) + J.prefer_first(b, a, fair, 0.3) == pytest.approx(1.0, abs=1e-12)
        assert J.prefer_first(a, b, pos, 0.3) + J.prefer_first(b, a, pos, 0.3) > 1.0
        d = (a["q"] - b["q"] + 0.8) / 0.3
        assert J.prefer_first(a, b, pos, 0.3) == pytest.approx(1 / (1 + math.exp(-d)), rel=1e-12)


def test_first_position_share_matches_its_expectation():
    cfg = dict(J.JUDGE, pairs=4000)
    ps = J.pairs(cfg)
    r = J.judge_pairs("position", cfg, ps)
    bias = cfg["settings"]["position"]
    want = sum(J.prefer_first(p["A"], p["B"], bias, cfg["tau"]) + J.prefer_first(p["B"], p["A"], bias, cfg["tau"])
               for p in ps) / (2 * len(ps))
    se = math.sqrt(want * (1 - want) / (2 * len(ps)))
    assert abs(r["first_share"] - want) < 4 * se
    fair = J.judge_pairs("fair", cfg, ps)
    assert abs(fair["first_share"] - 0.5) < 4 * math.sqrt(0.25 / (2 * len(ps)))


def test_swap_test_removes_position_but_not_length():
    st = J.judge_study()
    s = st["settings"]
    assert s["position"]["consistency"] < s["fair"]["consistency"]
    assert s["position"]["accuracy_decided"] > s["position"]["accuracy1"]
    assert s["verbosity"]["long_worse"]["acc_d"] < s["fair"]["long_worse"]["acc_d"]
    assert s["self"]["own_worse"]["acc1"] < s["fair"]["own_worse"]["acc1"]
    for k in J.JUDGE["order"]:
        f = s[k]["frames"][-1]
        assert f["kappa1"] == s[k]["kappa1"]["kappa"] and f["kappa_d"] == s[k]["kappa_d"]["kappa"]


def test_grading_study_tables():
    g = J.grading_study()
    assert sum(g["truth"]) / len(g["truth"]) == g["tests_rate"]
    c = g["calibration"]
    assert c["se_n"] + c["sp_n"] == c["n"] == J.GRADING["cal"]
    assert g["judge_rate"] > g["tests_rate"]  # a lenient judge inflates a weak agent
    assert g["corrected"]["ci"][0] <= g["tests_rate"] <= g["corrected"]["ci"][1]


# ── trajectories ─────────────────────────────────────────────────────────


def traces():
    return {n: json.loads((ROOT / "traces" / f"{n}.json").read_text()) for n in
            ["qwen-fix-test-native", "qwen-fix-test-react", "qwen-lookup-native"]}


def test_trajectory_grades_by_hand():
    rows = {r["id"]: r for r in T.trajectory_study(traces())["rows"]}
    assert rows["fix"]["grades"] == {"outcome": True, "self_report": True, "trajectory": True}
    assert rows["fix"]["score"] == 1.0 and rows["fix"]["waste"] == 0
    assert rows["malformed"]["counts"]["malformed"] == 1 and rows["malformed"]["grades"]["trajectory"]
    assert rows["flaky"]["counts"]["retries"] >= 1
    assert rows["cleanup"]["unsafe"] == [{"command": "rm -rf build", "what": "deletes files", "ran": True}]
    assert not rows["cleanup"]["grades"]["trajectory"] and rows["cleanup"]["grades"]["outcome"]
    assert rows["guarded"]["counts"]["unsafe_stopped"] == 2 and rows["guarded"]["counts"]["unsafe_run"] == 1
    # the recorded ReAct run claims the tests pass but never edited the code
    q = rows["qwen-react"]
    assert q["grades"] == {"outcome": False, "self_report": True, "trajectory": False}
    assert q["counts"]["malformed"] == 2 and "false claim" in q["frames"][-1]["flags"]
    assert not rows["qwen-native"]["claimed"]
    assert rows["lookup"]["outcome"] and not rows["qwen-lookup"]["outcome"]


def test_unsafe_patterns():
    assert T.unsafe_what("rm -rf build") == "deletes files"
    assert T.unsafe_what("pip install x") == "installs a package"
    assert T.unsafe_what("cat ~/x") == "reaches outside the repository"
    assert T.unsafe_what("pytest -q") is None


# ── the scaffold study ───────────────────────────────────────────────────


def test_task_suite():
    for t in F.TASKS:
        assert F.passes(t, t["right"]), t["name"]
        assert not F.passes(t, t["bug"]), t["name"]
        for w in t["wrong"]:
            assert not F.passes(t, w), (t["name"], w)
        assert len(set(t["wrong"])) == 3 and t["bug"] not in t["wrong"] and t["right"] not in t["wrong"]


def test_fixer_first_edit_is_right_with_probability_mu():
    t = F.TASKS[6]
    mu = F.mu_of(t, 0.0)
    n = 6000
    ok = 0
    pair = 0
    for s in range(n):
        m = F.FixerModel(t, mu, 0.5, 0.0, True, 10_000 + s)
        a = m._edit_step()["calls"][0]["args"]["new_text"].endswith(t["right"])
        m.current = t["bug"]
        b = m._edit_step()["calls"][0]["args"]["new_text"].endswith(t["right"])
        ok += a
        pair += a and b
    assert abs(ok / n - mu) < 4 * math.sqrt(mu * (1 - mu) / n)
    # two attempts on one task correlate at ρ: P(both) = μ² + ρ μ(1 − μ)
    want = mu * mu + 0.5 * mu * (1 - mu)
    assert abs(pair / n - want) < 4 * math.sqrt(want * (1 - want) / n)


def test_scaffold_rows_and_summaries():
    st = F.scaffold_study()
    for s in st["configs"]:
        rows = st["rows"][s["id"]]
        assert len(rows) == len(F.TASKS) * F.SCAFFOLD["runs"]
        assert s["k"] == sum(1 for r in rows if r["resolved"])
        assert all(r["turns"] <= s["budget"] for r in rows)
        if s["tools"] == "edit":
            assert all(r["edits"] <= 1 for r in rows)
    # more turns never hurt with common random numbers and a shell: a run resolved at 4 turns is resolved at 16
    for a in ["small", "large"]:
        for style in ["native", "react"]:
            r4 = st["rows"][f"{a}|test|{style}|4"]
            r16 = st["rows"][f"{a}|test|{style}|16"]
            assert all(y["resolved"] for x, y in zip(r4, r16) if x["resolved"])
    c = F.cost_study(st)
    assert c["best"] in c["front"]


# ── contamination ────────────────────────────────────────────────────────


def test_contamination_detection_rates():
    st = C.contamination_study()
    lv = st["levels"]
    assert lv["0.00"]["leaked"] == 0 and lv["0.00"]["rate"] == lv["0.00"]["clean_rate"]
    rates = [lv[k]["rate"] for k in sorted(lv)]
    assert rates == sorted(rates)
    for k, v in lv.items():
        assert v["flagged_verbatim"] == v["verbatim"]
        assert v["flagged_clean"] == 0
        for r in v["rows"]:
            if r["kind"] == "verbatim":
                assert r["overlap"] == 1.0
    # a paraphrase keeps an n-gram with probability (1 − mutate)^n, about 0.10 here
    paras = [r["overlap"] for r in lv["0.60"]["rows"] if r["kind"] == "paraphrase"]
    assert sum(paras) / len(paras) == pytest.approx(st["survive"], abs=0.03)


def test_ngrams():
    assert C.ngrams([1, 2, 3, 4], 2) == ["1,2", "2,3", "3,4"]


# ── fixtures ─────────────────────────────────────────────────────────────


def test_fixtures_are_current():
    fx = json.loads((ROOT / "fixtures" / "evals2_fixtures.json").read_text())
    assert fx["engine"] == VERSION == "1.12.0"
    assert fx["judge"] == json.loads(json.dumps(J.judge_study()))
    data = json.loads((ROOT / "ts" / "src" / "evals2_data.json").read_text())
    assert data["scaffold"] == json.loads(json.dumps(F.SCAFFOLD)) and data["tasks"] == F.TASKS
    assert data["contam"] == C.CONTAM and data["judge"] == J.JUDGE


def test_rng_streams_are_separate():
    assert Rng(F.run_seed(F.SCAFFOLD, 1, 2)).random() != Rng(F.run_seed(F.SCAFFOLD, 2, 1)).random()
