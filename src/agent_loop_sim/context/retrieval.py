"""Retrieval over a chunk list: BM25, dense (int8 / int4 / binary), hybrid fusion (RRF and
weighted), and a reranker stage from recorded cross-encoder scores.

Every ranking sorts by score, highest first, ties broken by chunk index, and covers every chunk.
Floating-point operations run in the same order in the TS port; sums are plain left-to-right
loops (Python's built-in ``sum`` of floats is compensated since 3.12 and would differ).
"""
from __future__ import annotations

from typing import Any

from .mathx import ln
from .text import analyse
from .vectors import prepare, similarity

K1 = 1.2
B = 0.75
RRF_K = 60


class BM25:
    """Okapi BM25 over analysed chunk texts, with Lucene's smoothed IDF
    ``ln(1 + (N - df + 0.5) / (df + 0.5))`` (never negative), with ``mathx.ln`` so both
    languages compute it bit for bit."""

    def __init__(self, texts: list[str], stop: bool = True) -> None:
        self.stop = stop
        self.docs: list[dict[str, int]] = []
        self.lens: list[int] = []
        self.df: dict[str, int] = {}
        total = 0
        for t in texts:
            terms = analyse(t, stop)
            tf: dict[str, int] = {}
            for w in terms:
                tf[w] = tf.get(w, 0) + 1
            for w in tf:
                self.df[w] = self.df.get(w, 0) + 1
            self.docs.append(tf)
            self.lens.append(len(terms))
            total += len(terms)
        self.n = len(texts)
        self.avgdl = total / self.n if self.n else 0.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return ln(1 + (self.n - df + 0.5) / (df + 0.5))

    def query_terms(self, query: str) -> list[str]:
        """The analysed query, duplicates dropped, first occurrence kept."""
        out: list[str] = []
        for w in analyse(query, self.stop):
            if w not in out:
                out.append(w)
        return out

    def term_score(self, tf: int, dl: int, idf: float, k1: float, b: float) -> float:
        return idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))

    def scores(self, query: str, k1: float = K1, b: float = B) -> list[float]:
        terms = self.query_terms(query)
        idfs = [self.idf(w) for w in terms]
        out = []
        for d in range(self.n):
            s = 0.0
            doc = self.docs[d]
            for j, w in enumerate(terms):
                tf = doc.get(w, 0)
                if tf:
                    s += self.term_score(tf, self.lens[d], idfs[j], k1, b)
            out.append(s)
        return out


def rank(scores: list[float]) -> list[int]:
    """Chunk indices by score, highest first; ties by index."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def dense_scores(q: list[int], chunk_vecs: list[list[int]], precision: str) -> list[float]:
    return [similarity(q, v, precision) for v in chunk_vecs]


def rrf(rankings: list[list[int]], n: int, k: int = RRF_K, depth: int = 50) -> list[float]:
    """Reciprocal rank fusion: each list adds ``1 / (k + rank)`` (rank from 1) for its top ``depth``."""
    s = [0.0] * n
    for r in rankings:
        for pos in range(min(depth, len(r))):
            s[r[pos]] += 1 / (k + pos + 1)
    return s


def minmax_top(scores: list[float], order: list[int], depth: int) -> list[float]:
    """Min-max normalised scores of the top ``depth`` (0 for the rest); all 1 if they are equal."""
    top = order[:depth]
    out = [0.0] * len(scores)
    if not top:
        return out
    hi = scores[top[0]]
    lo = scores[top[-1]]
    for i in top:
        out[i] = 1.0 if hi == lo else (scores[i] - lo) / (hi - lo)
    return out


def weighted(bm25: list[float], dense: list[float], alpha: float, depth: int = 50) -> list[float]:
    """``alpha * dense + (1 - alpha) * bm25`` on min-max normalised top-``depth`` scores."""
    nb = minmax_top(bm25, rank(bm25), depth)
    nd = minmax_top(dense, rank(dense), depth)
    return [alpha * nd[i] + (1 - alpha) * nb[i] for i in range(len(bm25))]


def rerank(order: list[int], ce: dict[int, int], n: int) -> list[int]:
    """Reorder the top ``n`` by the recorded cross-encoder score (ties keep their order); the rest
    stay. Every one of the top ``n`` must have a recorded score."""
    top = order[:n]
    for c in top:
        if c not in ce:
            raise ValueError(f"no reranker score recorded for chunk {c} (outside the recorded pool)")
    pos = {c: i for i, c in enumerate(top)}
    return sorted(top, key=lambda c: (-ce[c], pos[c])) + order[n:]


class Retriever:
    """Every stage for one chunking config of a corpus."""

    def __init__(self, corpus: Any, config: str, stop: bool = True) -> None:
        self.corpus = corpus
        self.config = config
        self.chunks = corpus.chunks(config)
        self.bm25 = BM25([corpus.chunk_text(c) for c in self.chunks], stop)
        self._vecs: dict[str, list[list[int]]] = {}
        self._qvecs: dict[str, list[list[int]]] = {}
        self._cache: dict[tuple, list[float]] = {}

    def vecs(self, precision: str) -> list[list[int]]:
        if precision not in self._vecs:
            self._vecs[precision] = prepare(self.corpus.vectors(self.config), precision)
            self._qvecs[precision] = prepare(self.corpus.vectors("questions"), precision)
        return self._vecs[precision]

    def bm25_scores(self, qi: int, k1: float = K1, b: float = B) -> list[float]:
        key = ("bm25", qi, k1, b)
        if key not in self._cache:
            self._cache[key] = self.bm25.scores(self.corpus.questions[qi]["question"], k1, b)
        return self._cache[key]

    def dense(self, qi: int, precision: str = "int8") -> list[float]:
        key = ("dense", qi, precision)
        if key not in self._cache:
            v = self.vecs(precision)
            self._cache[key] = dense_scores(self._qvecs[precision][qi], v, precision)
        return self._cache[key]

    def ranking(self, qi: int, method: str, p: dict[str, Any] | None = None) -> list[int]:
        """``method``: bm25 | dense | rrf | weighted | rerank (the cross-encoder over the top ``n`` of
        ``base``: rrf (default), bm25 or dense)."""
        p = p or {}
        if method == "bm25":
            return rank(self.bm25_scores(qi, p.get("k1", K1), p.get("b", B)))
        if method == "dense":
            return rank(self.dense(qi, p.get("precision", "int8")))
        rb = rank(self.bm25_scores(qi, p.get("k1", K1), p.get("b", B)))
        if method == "weighted":
            return rank(weighted(self.bm25_scores(qi, p.get("k1", K1), p.get("b", B)),
                                 self.dense(qi, p.get("precision", "int8")), p.get("alpha", 0.5), p.get("depth", 50)))
        rd = rank(self.dense(qi, p.get("precision", "int8")))
        fused = rank(rrf([rb, rd], len(self.chunks), p.get("k", RRF_K), p.get("depth", 50)))
        if method == "rrf":
            return fused
        if method == "rerank":
            if self.config != self.corpus.files["rerank.json"]["config"]:
                raise ValueError(f"reranker scores were recorded for {self.corpus.files['rerank.json']['config']} only")
            base = {"rrf": fused, "bm25": rb, "dense": rd}[p.get("base", "rrf")]
            return rerank(base, self.corpus.rerank_scores()[qi], p.get("n", 20))
        raise ValueError(f"unknown method {method!r}")
