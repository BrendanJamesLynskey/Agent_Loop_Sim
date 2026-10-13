"""A simulated LLM judge (since 1.9): pairwise verdicts with position, verbosity and self-preference
biases, the swap test, Cohen's κ against the truth, and a pass/fail judge whose errors are corrected
with a labelled calibration set.

**Pairwise.** Each pair holds two responses, A and B. Response r has a latent quality q_r (what a
careful human would judge), a length z-score ℓ_r, and a flag o_r for "written by the judge's own model
family". The truth is the response with the higher quality, or a tie when the two
qualities are within ``tie_band`` of each other (raters may call a tie, as MT-Bench's could). The judge sees the two in some order,
first X then Y, and prefers X with probability

    P(X) = logistic((q_X − q_Y + β_len (ℓ_X − ℓ_Y) + β_self (o_X − o_Y) + β_pos) / τ),

a Bradley–Terry choice (Bradley and Terry 1952) with three bias terms: β_pos favours whichever
response is shown first, β_len the longer one, β_self the judge's own family; τ is the judge's
sampling noise. These are the three biases Zheng et al. (2023, MT-Bench, arXiv:2306.05685) measured in
GPT-4 as a judge; the coefficients here are illustrative.

**The swap test.** Every pair is judged twice, (A, B) and then (B, A), with fresh noise. A verdict that
changes with the order is a position effect; the debiased verdict keeps the winner when both orders
agree and calls a tie when they do not (the mitigation the MT-Bench paper uses). Swapping cancels the
position term exactly; it cannot touch length or self-preference, which do not move with the order.

**Pass/fail grading.** A judge reading agent transcripts says "pass" to a true pass with probability
Se (sensitivity) and to a true fail with probability 1 − Sp (specificity). The rate it reports is
p_obs = π Se + (1 − π)(1 − Sp), so a lenient judge (low Sp) inflates a weak agent's score most.
Grading a calibration set both ways estimates Se and Sp, and ``stats.corrected_rate`` inverts the
relation (Rogan and Gladen 1978).

Every quality, length, bias and rate here is illustrative.
"""
from __future__ import annotations

import math
from typing import Any

from ..context.text import fixed
from ..rng import Rng
from . import generator as G
from . import stats as S
from .mathx import logistic, norm_ppf

LABELS = ["A", "B", "tie"]

JUDGE = {
    "pairs": 200, "seed": 2561, "tau": 0.3, "own_share": 0.3, "tie_band": 0.3,
    "settings": {
        "fair": {"position": 0.0, "verbosity": 0.0, "self": 0.0},
        "position": {"position": 0.8, "verbosity": 0.0, "self": 0.0},
        "verbosity": {"position": 0.0, "verbosity": 0.6, "self": 0.0},
        "self": {"position": 0.0, "verbosity": 0.0, "self": 0.8},
        "all": {"position": 0.8, "verbosity": 0.6, "self": 0.8},
    },
    "order": ["fair", "position", "verbosity", "self", "all"],
}

GRADING = {"theta": -0.4, "tasks": 300, "cal": 200, "se": 0.95, "sp": 0.7, "seed": 2572,
           "curve": [round(0.05 * i, 2) for i in range(21)]}

_TWO32 = 4294967296.0


def _z(r: Rng) -> float:
    return norm_ppf((r.next_u32() + 0.5) / _TWO32)


def pairs(cfg: dict[str, Any] = JUDGE) -> list[dict[str, Any]]:
    """The pairs: per pair, A's quality, length and family flag, then B's (six draws, in that order)."""
    r = Rng(cfg["seed"])
    out = []
    for i in range(cfg["pairs"]):
        side = {}
        for s in ["A", "B"]:
            q = _z(r)
            ln_ = _z(r)
            own = 1 if r.random() < cfg["own_share"] else 0
            side[s] = {"q": q, "len": ln_, "own": own, "tokens": math.floor(220.0 + 90.0 * ln_ + 0.5) if ln_ > -2.0 else 40}
        out.append({"i": i, "A": side["A"], "B": side["B"], "truth": _truth(side["A"]["q"], side["B"]["q"], cfg["tie_band"])})
    return out


def _truth(qa: float, qb: float, band: float) -> str:
    d = qa - qb
    if d < band and -d < band:
        return "tie"
    return "A" if d > 0.0 else "B"


def prefer_first(x: dict[str, Any], y: dict[str, Any], bias: dict[str, float], tau: float) -> float:
    d = (x["q"] - y["q"]) + bias["verbosity"] * (x["len"] - y["len"]) + bias["self"] * (x["own"] - y["own"]) + bias["position"]
    return logistic(d / tau)


def judge_pairs(setting: str, cfg: dict[str, Any] = JUDGE, ps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Every pair judged in both orders under one bias setting (judge stream ``seed + 1``, two draws
    per pair: order AB, then BA; the same draws in every setting). Frames: one per pair, with the
    running agreement tables."""
    bias = cfg["settings"][setting]
    tau = cfg["tau"]
    items = ps if ps is not None else pairs(cfg)
    r = Rng(cfg["seed"] + 1)
    truth: list[str] = []
    single: list[str] = []
    debiased: list[str] = []
    first_wins = 0
    consistent = 0
    frames = []
    for it in items:
        a = it["A"]
        b = it["B"]
        p_ab = prefer_first(a, b, bias, tau)
        p_ba = prefer_first(b, a, bias, tau)
        u1 = r.random()
        u2 = r.random()
        v1 = "A" if u1 < p_ab else "B"
        v2 = "B" if u2 < p_ba else "A"
        if v1 == "A":
            first_wins += 1
        if v2 == "B":
            first_wins += 1
        same = v1 == v2
        if same:
            consistent += 1
        vd = v1 if same else "tie"
        truth.append(it["truth"])
        single.append(v1)
        debiased.append(vd)
        n = len(truth)
        k1 = S.cohen_kappa(S.confusion(truth, single, LABELS))
        kd = S.cohen_kappa(S.confusion(truth, debiased, LABELS))
        cap = (f"Pair {n}: " + ("a tie by the truth" if it["truth"] == "tie" else f"{it['truth']} is better") + f". Shown A first the judge picks {v1}; shown B first it picks {v2}"
               + (". Both orders agree." if same else ": the verdict follows the order, so it counts as a tie.")
               + f" κ so far: {fixed(k1['kappa'] + 0.0, 2)} one order, {fixed(kd['kappa'] + 0.0, 2)} swapped.")
        frames.append({"i": it["i"], "p_ab": p_ab, "p_ba": p_ba, "v1": v1, "v2": v2, "debiased": vd,
                       "kappa1": k1["kappa"], "kappa_d": kd["kappa"], "consistent": consistent, "caption": cap})
    n = len(items)
    t1 = S.confusion(truth, single, LABELS)
    td = S.confusion(truth, debiased, LABELS)
    decided = [i for i in range(n) if debiased[i] != "tie"]
    right_d = 0
    for i in decided:
        if debiased[i] == truth[i]:
            right_d += 1
    right1 = 0
    for i in range(n):
        if single[i] == truth[i]:
            right1 += 1
    # where the longer response is the worse one, and where the judge's own family wrote the worse one
    long_worse = [i for i in range(n) if truth[i] != "tie" and (items[i]["A"]["len"] > items[i]["B"]["len"]) == (truth[i] == "B")]
    own_worse = [i for i in range(n) if truth[i] != "tie" and items[i]["A"]["own"] != items[i]["B"]["own"]
                 and (items[i]["A"]["own"] == 1) == (truth[i] == "B")]

    def acc(ix: list[int], v: list[str]) -> float:
        if not ix:
            return 0.0
        k = 0
        for i in ix:
            if v[i] == truth[i]:
                k += 1
        return k / len(ix)

    return {"setting": setting, "bias": bias, "table1": t1, "table_d": td,
            "kappa1": S.cohen_kappa(t1), "kappa_d": S.cohen_kappa(td),
            "accuracy1": right1 / n, "consistency": consistent / n, "first_share": first_wins / (2 * n),
            "decided": len(decided) / n, "accuracy_decided": right_d / len(decided) if decided else 0.0,
            "long_worse": {"n": len(long_worse), "acc1": acc(long_worse, single), "acc_d": acc(long_worse, debiased)},
            "own_worse": {"n": len(own_worse), "acc1": acc(own_worse, single), "acc_d": acc(own_worse, debiased)},
            "frames": frames}


def judge_study(cfg: dict[str, Any] = JUDGE) -> dict[str, Any]:
    ps = pairs(cfg)
    return {"cfg": cfg, "pairs": ps, "settings": {s: judge_pairs(s, cfg, ps) for s in cfg["order"]}}


def observed_rate(pi: float, se: float, sp: float) -> float:
    return pi * se + (1.0 - pi) * (1.0 - sp)


def grading_study(cfg: dict[str, Any] = GRADING, pop: list[dict[str, float]] | None = None) -> dict[str, Any]:
    """A weak agent graded three ways on ``tasks`` tasks: unit tests (the truth), a lenient judge, and
    the judge corrected with a calibration set of ``cal`` tasks labelled by hand. Streams: the tasks
    ``seed``, the runs ``seed + 1``, the judge's calls ``seed + 2``; the calibration set's tasks
    ``seed + 3``, runs ``seed + 4``, judge ``seed + 5``. Frames: one per task graded."""
    p = pop if pop is not None else G.population()
    theta = cfg["theta"]

    def graded(n: int, s0: int) -> tuple[list[int], list[int]]:
        mu = G.mus([p[i] for i in G.sample(n, len(p), Rng(s0))], theta)
        x = [row[0] for row in G.runs(mu, 1, 0.0, Rng(s0 + 1))]
        rj = Rng(s0 + 2)
        j = []
        for v in x:
            u = rj.random()
            j.append((1 if u < cfg["se"] else 0) if v == 1 else (1 if u < 1.0 - cfg["sp"] else 0))
        return x, j

    x, j = graded(cfg["tasks"], cfg["seed"])
    cx, cj = graded(cfg["cal"], cfg["seed"] + 3)
    se_n = 0
    se_k = 0
    sp_n = 0
    sp_k = 0
    for v, w in zip(cx, cj):
        if v == 1:
            se_n += 1
            se_k += w
        else:
            sp_n += 1
            sp_k += 1 - w
    frames = []
    kt = 0
    kj = 0
    for i in range(len(x)):
        kt += x[i]
        kj += j[i]
        n = i + 1
        cap = (f"Task {n}: the unit tests say {'pass' if x[i] else 'fail'}, the judge says {'pass' if j[i] else 'fail'}"
               + ("" if x[i] == j[i] else " (wrong)") + f". Scores so far: tests {S_pct(kt / n)}, judge {S_pct(kj / n)}.")
        frames.append({"i": i, "truth": x[i], "judge": j[i], "k_tests": kt, "k_judge": kj, "caption": cap})
    corr = S.corrected_rate(kj, len(x), se_k, se_n, sp_k, sp_n)
    curve = [{"pi": pi, "observed": observed_rate(pi, cfg["se"], cfg["sp"])} for pi in cfg["curve"]]
    return {"cfg": cfg, "truth": x, "judge": j, "tests_rate": kt / len(x), "judge_rate": kj / len(x),
            "wilson_tests": S.wilson(kt, len(x)), "wilson_judge": S.wilson(kj, len(x)),
            "calibration": {"n": len(cx), "se_k": se_k, "se_n": se_n, "sp_k": sp_k, "sp_n": sp_n},
            "corrected": corr, "kappa": S.cohen_kappa(S.confusion([str(v) for v in x], [str(v) for v in j], ["1", "0"])),
            "curve": curve, "frames": frames}


def S_pct(x: float) -> str:
    return fixed(x * 100.0 + 0.0, 1) + "%"
