"""Packing the window (added in engine 1.5.0): which retrieved chunks to put in a token budget, and
where to put them.

**What to pack.** For a question, the candidates are the reranked top ``N`` chunks of the default
chunking (RRF, then the recorded cross-encoder; chapter 4). A chunk's **value** is the gain nDCG gives
its reranked position i, 1 / log2(i + 1) (with the shared ``ln``), and its **weight** is its token
count. Packing is the 0/1 knapsack: choose a set of total weight at most the budget B with the most
value. Three packers:

- ``top``: take chunks in reranked order, skipping any that no longer fits (what most pipelines do);
- ``density``: the same, but in order of value per token (the classic greedy approximation);
- ``optimal``: dynamic programming over integer token budgets, exact for these small n
  (best[i][c] = max(best[i-1][c], best[i-1][c - w_i] + v_i); ties keep the set without item i).

A question counts as **answered in the window** when a packed chunk holds its answer span.

**Where to put it.** Liu et al. (2023, "Lost in the Middle", arXiv:2307.03172) measured that models
use information at the start or end of a long input better than in the middle. The engine models
this as a U-shaped **position curve** p(x), the chance the model uses the answer when it sits at
relative position x in the packed context: two parabolas meeting at the trough x0,
p(x) = m + (s - m) ((x0 - x) / x0)^2 for x <= x0 and m + (e - m) ((x - x0) / (1 - x0))^2 after.
The parameters are **illustrative** (the paper's own numbers depend on the model and the task). A
chunk's position is its middle token's offset over the packed total. Four placements of a packed set:
``best-first``, ``best-last``, ``ends`` (best first, second best last, and so on inwards: the weakest
in the middle) and ``middle`` (the reverse, best in the middle: what to avoid).
"""
from __future__ import annotations

from typing import Any

from .evaluate import relevant
from .mathx import ln
from .retrieval import Retriever
from .text import fixed

PACKERS = ["top", "density", "optimal"]
PLACEMENTS = ["best-first", "best-last", "ends", "middle"]
CANDIDATES = 20
POSITION = {"start": 0.75, "middle": 0.55, "end": 0.65, "trough": 0.5}


def gain(i: int) -> float:
    """1 / log2(i + 1) for the reranked position i (from 1)."""
    return ln(2.0) / ln(float(i + 1))


def position_p(x: float, pc: dict[str, float] | None = None) -> float:
    pc = pc or POSITION
    x0 = pc["trough"]
    if x <= x0:
        u = (x0 - x) / x0
        return pc["middle"] + (pc["start"] - pc["middle"]) * (u * u)
    u = (x - x0) / (1 - x0)
    return pc["middle"] + (pc["end"] - pc["middle"]) * (u * u)


def position_curve(pc: dict[str, float] | None = None, points: int = 21) -> list[list[float]]:
    return [[i / (points - 1), position_p(i / (points - 1), pc)] for i in range(points)]


def candidates(r: Retriever, qi: int, n: int = CANDIDATES) -> list[dict[str, Any]]:
    order = r.ranking(qi, "rerank", {"n": n})[:n]
    rel = set(relevant(r.chunks, r.corpus.questions[qi]))
    return [{"chunk": c, "rank": i + 1, "tokens": r.chunks[c]["tokens"], "value": gain(i + 1), "relevant": c in rel}
            for i, c in enumerate(order)]


def greedy(cands: list[dict[str, Any]], budget: int, packer: str) -> dict[str, Any]:
    """``top`` or ``density``: consider the candidates in the packer's order, take what fits.
    Returns the chosen indices (into ``cands``, in candidate order) and one step per candidate."""
    idx = list(range(len(cands)))
    if packer == "density":
        idx.sort(key=lambda i: (-(cands[i]["value"] / cands[i]["tokens"]), i))
    left = budget
    chosen: list[int] = []
    steps = []
    for i in idx:
        c = cands[i]
        take = c["tokens"] <= left
        if take:
            left -= c["tokens"]
            chosen.append(i)
        steps.append({"i": i, "take": take, "left": left})
    return {"chosen": sorted(chosen), "steps": steps}


def knapsack_table(cands: list[dict[str, Any]], budget: int) -> tuple[list[list[float]], list[list[bool]]]:
    n = len(cands)
    best = [[0.0] * (budget + 1)]
    take = [[False] * (budget + 1)]
    for i in range(1, n + 1):
        w = cands[i - 1]["tokens"]
        v = cands[i - 1]["value"]
        prev = best[i - 1]
        row = list(prev)
        tk = [False] * (budget + 1)
        for c in range(w, budget + 1):
            t = prev[c - w] + v
            if t > row[c]:
                row[c] = t
                tk[c] = True
        best.append(row)
        take.append(tk)
    return best, take


def knapsack_pick(cands: list[dict[str, Any]], take: list[list[bool]], budget: int) -> list[int]:
    c = budget
    out = []
    for i in range(len(cands), 0, -1):
        if take[i][c]:
            out.append(i - 1)
            c -= cands[i - 1]["tokens"]
    return sorted(out)


def _sum(cands: list[dict[str, Any]], chosen: list[int], key: str) -> Any:
    s: Any = 0
    for i in chosen:
        s += cands[i][key]
    return s


def place(cands: list[dict[str, Any]], chosen: list[int], placement: str) -> list[int]:
    """The packed set's order in the window (indices into ``cands``)."""
    s = sorted(chosen)  # best first: candidates are in reranked order
    if placement == "best-first":
        return s
    if placement == "best-last":
        return s[::-1]
    src = s if placement == "ends" else s[::-1]
    front: list[int] = []
    back: list[int] = []
    for j, i in enumerate(src):
        (front if j % 2 == 0 else back).append(i)
    return front + back[::-1]


def positions(cands: list[dict[str, Any]], order: list[int]) -> list[float]:
    """Relative position (0..1) of each placed chunk's middle token."""
    total = _sum(cands, order, "tokens")
    out = []
    off = 0
    for i in order:
        t = cands[i]["tokens"]
        out.append((off + t / 2) / total)
        off += t
    return out


def answer_p(cands: list[dict[str, Any]], order: list[int], pc: dict[str, float] | None = None) -> float:
    """The position curve's p at the best-placed chunk that holds the answer (0 if none is packed)."""
    best = 0.0
    for i, x in zip(order, positions(cands, order)):
        if cands[i]["relevant"]:
            p = position_p(x, pc)
            if p > best:
                best = p
    return best


def pack_all(cands: list[dict[str, Any]], budgets: list[int]) -> dict[int, dict[str, dict[str, Any]]]:
    best, take = knapsack_table(cands, max(budgets))
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for b in budgets:
        out[b] = {"top": greedy(cands, b, "top"), "density": greedy(cands, b, "density"),
                  "optimal": {"chosen": knapsack_pick(cands, take, b), "best": [best[i][b] for i in range(len(cands) + 1)]}}
    return out


def packing_eval(r: Retriever, budgets: list[int], n: int = CANDIDATES,
                 pc: dict[str, float] | None = None) -> dict[str, Any]:
    """Every question, every budget and packer: the share answered in the window, the mean tokens
    used and value packed, and for each placement the mean p (the position curve at the answer;
    0 when the answer is not packed). Means over all questions."""
    qs = r.corpus.questions
    acc: dict[str, dict[str, dict[str, float]]] = {
        str(b): {p: {"answered": 0.0, "tokens": 0.0, "value": 0.0, **{pl: 0.0 for pl in PLACEMENTS}} for p in PACKERS}
        for b in budgets}
    for qi in range(len(qs)):
        cands = candidates(r, qi, n)
        packs = pack_all(cands, budgets)
        for b in budgets:
            for p in PACKERS:
                ch = packs[b][p]["chosen"]
                a = acc[str(b)][p]
                a["answered"] += 1 if any(cands[i]["relevant"] for i in ch) else 0
                a["tokens"] += _sum(cands, ch, "tokens")
                a["value"] += _sum(cands, ch, "value")
                for pl in PLACEMENTS:
                    a[pl] += answer_p(cands, place(cands, ch, pl), pc) if ch else 0.0
    nq = len(qs)
    for b in budgets:
        for p in PACKERS:
            a = acc[str(b)][p]
            for k in list(a):
                a[k] = a[k] / nq
    return {"budgets": budgets, "candidates": n, "position": pc or POSITION, "results": acc}


def packing_view(r: Retriever, qi: int, budget: int, n: int = CANDIDATES,
                 pc: dict[str, float] | None = None) -> dict[str, Any]:
    """One question's packing, for the animation: the candidates; for each packer its steps (greedy:
    each candidate considered in turn; optimal: the best value within the budget using the first i
    candidates, and whether candidate i is in the final set) and its set; and the optimal set's
    four placements with their positions and p."""
    cands = candidates(r, qi, n)
    packs = pack_all(cands, [budget])[budget]
    out: dict[str, Any] = {"q": qi, "budget": budget, "candidates": cands, "packers": {}}
    for p in ["top", "density"]:
        g = packs[p]
        steps = []
        for s in g["steps"]:
            c = cands[s["i"]]
            steps.append(dict(s, caption=(f"{p}: chunk {c['chunk']} (rank {c['rank']}, {c['tokens']} tokens, value "
                                          f"{fixed(c['value'], 2)}) " + (f"fits: taken, {s['left']} tokens left" if s["take"]
                                                                    else f"does not fit in the {s['left']} tokens left: skipped"))))
        ch = g["chosen"]
        out["packers"][p] = {"chosen": ch, "steps": steps, "tokens": _sum(cands, ch, "tokens"),
                             "value": _sum(cands, ch, "value"), "answered": any(cands[i]["relevant"] for i in ch)}
    opt = packs["optimal"]
    ch = opt["chosen"]
    steps = []
    for i in range(1, len(cands) + 1):
        c = cands[i - 1]
        inset = (i - 1) in ch
        steps.append({"i": i - 1, "take": inset, "best": opt["best"][i],
                      "caption": f"optimal: with the first {i} candidates the best value within {budget} tokens is "
                                 f"{fixed(opt['best'][i], 2)}; chunk {c['chunk']} is " + ("in" if inset else "not in") + " the final set"})
    out["packers"]["optimal"] = {"chosen": ch, "steps": steps, "tokens": _sum(cands, ch, "tokens"),
                                 "value": _sum(cands, ch, "value"), "answered": any(cands[i]["relevant"] for i in ch)}
    out["placements"] = {}
    for pl in PLACEMENTS:
        order = place(cands, ch, pl)
        out["placements"][pl] = {"order": order, "positions": positions(cands, order) if order else [],
                                 "p": answer_p(cands, order, pc) if order else 0.0}
    out["curve"] = position_curve(pc)
    return out
