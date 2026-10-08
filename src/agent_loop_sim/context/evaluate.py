"""Labelled evaluation: which chunks answer a question, and recall@k, MRR@10 and nDCG@10.

A chunk is **relevant** to a question when it contains the whole gold answer span (SQuAD gives
the answer's character offsets). A chunker that cuts through an answer leaves the question with
no relevant chunk: it scores 0 on every metric and is counted as ``lost``.

For a ranking and relevant set R (binary gains):
- recall@k = |R ∩ top k| / |R|;
- MRR@10 = 1 / rank of the first relevant chunk, 0 if none is in the top 10;
- nDCG@10 = DCG / IDCG with DCG = sum over the top 10 of rel_i / log2(i + 1) and IDCG the same for
  an ideal ranking of |R| relevant chunks.
Means are over all questions, summed left to right.
"""
from __future__ import annotations

from typing import Any

from .mathx import ln

KS = [1, 3, 5, 10, 20]
# 1 / log2(i + 1) for ranks i = 1..10, as ln 2 / ln(i + 1) with the shared ln
DISCOUNT = [ln(2.0) / ln(float(i + 1)) for i in range(1, 11)]


def relevant(chunks: list[dict[str, Any]], q: dict[str, Any]) -> list[int]:
    return [i for i, c in enumerate(chunks)
            if c["article"] == q["article"] and c["start"] <= q["start"] and q["end"] <= c["end"]]


def metrics(order: list[int], rel: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    rs = set(rel)
    for k in KS:
        hit = 0
        for c in order[:k]:
            if c in rs:
                hit += 1
        out[f"recall@{k}"] = hit / len(rel) if rel else 0.0
    rr = 0.0
    for i, c in enumerate(order[:10]):
        if c in rs:
            rr = 1 / (i + 1)
            break
    out["mrr@10"] = rr
    dcg = 0.0
    for i, c in enumerate(order[:10]):
        if c in rs:
            dcg += DISCOUNT[i]
    idcg = 0.0
    for i in range(min(len(rel), 10)):
        idcg += DISCOUNT[i]
    out["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0
    return out


METRICS = [f"recall@{k}" for k in KS] + ["mrr@10", "ndcg@10"]


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for m in METRICS:
        s = 0.0
        for r in rows:
            s += r[m]
        out[m] = s / len(rows) if rows else 0.0
    return out


def evaluate(retriever: Any, method: str, p: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mean metrics of one method over every question, the number lost to chunking, and each
    question's first relevant rank (0 when none is relevant; ranks count from 1)."""
    rows = []
    first = []
    lost = 0
    for qi, q in enumerate(retriever.corpus.questions):
        rel = relevant(retriever.chunks, q)
        if not rel:
            lost += 1
        order = retriever.ranking(qi, method, p)
        rows.append(metrics(order, rel))
        rs = set(rel)
        fr = 0
        for i, c in enumerate(order):
            if c in rs:
                fr = i + 1
                break
        first.append(fr)
    return {"config": retriever.config, "method": method, "params": p or {}, "lost": lost,
            "mean": mean_metrics(rows), "first_rank": first}
