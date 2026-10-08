"""Views the Context site animates (chapters 2 to 5), each a pure function of the engine's state,
reproduced exactly by the TS port. Chapter 1's frames come from ``window.window_run``."""
from __future__ import annotations

from typing import Any

from .evaluate import relevant
from .retrieval import B, K1, RRF_K, Retriever, rank, rerank, rrf
from .text import fixed, paragraph_starts, ranges, sentence_starts
from .vectors import cosine


# ---- chapter 2: BM25 -------------------------------------------------------------------------

def tf_curve(k1: float, b: float, ratio: float, tf_max: int = 10) -> list[list[float]]:
    """The BM25 term-frequency factor ``tf (k1 + 1) / (tf + k1 (1 - b + b dl/avgdl))`` for
    tf = 0..tf_max at a given dl/avgdl."""
    out = []
    for tf in range(tf_max + 1):
        out.append([tf, (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * ratio))])
    return out


def length_curve(k1: float, b: float, tf: int = 1) -> list[list[float]]:
    """The same factor at a fixed tf against dl/avgdl = 0.25, 0.5, ... 4."""
    out = []
    for i in range(1, 17):
        ratio = i / 4
        out.append([ratio, (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * ratio))])
    return out


def bm25_view(r: Retriever, query: str, k1: float = K1, b: float = B, top: int = 8) -> dict[str, Any]:
    """The query's terms with df and idf, and the top chunks with each term's contribution; the
    frames add one term at a time (the running top ``top``)."""
    bm = r.bm25
    terms = bm.query_terms(query)
    idfs = [bm.idf(w) for w in terms]
    cum = [0.0] * bm.n
    frames = []
    for j, w in enumerate(terms):
        for d in range(bm.n):
            tf = bm.docs[d].get(w, 0)
            if tf:
                cum[d] += bm.term_score(tf, bm.lens[d], idfs[j], k1, b)
        order = rank(cum)[:top]
        frames.append({"term": w, "idf": idfs[j], "df": bm.df.get(w, 0),
                       "top": [{"chunk": c, "score": cum[c]} for c in order],
                       "caption": f"term {j + 1} of {len(terms)}, “{w}”: in {bm.df.get(w, 0)} of {bm.n} chunks, "
                                  f"idf {fixed(idfs[j], 2)}; the leader is chunk {order[0]} at {fixed(cum[order[0]], 2)}"})
    order = rank(cum)[:top]
    rows = []
    for c in order:
        parts = []
        for j, w in enumerate(terms):
            tf = bm.docs[c].get(w, 0)
            parts.append({"term": w, "tf": tf, "score": bm.term_score(tf, bm.lens[c], idfs[j], k1, b) if tf else 0.0})
        rows.append({"chunk": c, "score": cum[c], "dl": bm.lens[c], "parts": parts})
    return {"query": query, "k1": k1, "b": b, "n": bm.n, "avgdl": bm.avgdl,
            "terms": [{"term": w, "df": bm.df.get(w, 0), "idf": idfs[j]} for j, w in enumerate(terms)],
            "top": rows, "frames": frames}


# ---- chapter 3: dense ------------------------------------------------------------------------

def dense_frames(r: Retriever, qis: list[int], precision: str = "int8", k: int = 5) -> list[dict[str, Any]]:
    """For each question in turn: its k nearest chunks and whether each holds the answer."""
    out = []
    for qi in qis:
        s = r.dense(qi, precision)
        order = rank(s)[:k]
        rel = relevant(r.chunks, r.corpus.questions[qi])
        hit = next((i + 1 for i, c in enumerate(order) if c in rel), 0)
        out.append({"q": qi, "neighbours": [{"chunk": c, "sim": s[c], "relevant": c in rel} for c in order],
                    "relevant": rel,
                    "caption": f"question {qi}: nearest chunk {order[0]} at cosine {fixed(s[order[0]], 3)}; "
                               + (f"the answer is at rank {hit}" if hit else f"the answer is not in the top {k}")})
    return out


# ---- chapter 4: hybrid and rerank ------------------------------------------------------------

def hybrid_frames(r: Retriever, qi: int, k: int = RRF_K, depth: int = 50, show: int = 10, n: int = 20) -> dict[str, Any]:
    """RRF merging the BM25 and dense lists rank by rank (one frame per rank), then the rerank."""
    rb = rank(r.bm25_scores(qi))
    rd = rank(r.dense(qi))
    rel = relevant(r.chunks, r.corpus.questions[qi])
    s = [0.0] * len(r.chunks)
    frames = []
    for pos in range(show):
        for lst in (rb, rd):
            if pos < depth:
                s[lst[pos]] += 1 / (k + pos + 1)
        top = [c for c in rank(s)[:show] if s[c] > 0]
        frames.append({"stage": "fuse", "rank": pos + 1, "adds": [rb[pos], rd[pos]], "top": [{"chunk": c, "score": s[c]} for c in top],
                       "caption": f"rank {pos + 1}: BM25 adds 1/({k}+{pos + 1}) to chunk {rb[pos]}, dense adds it to chunk {rd[pos]}"
                                  + (" (the same chunk: it doubles)" if rb[pos] == rd[pos] else "")})
    fused = rank(rrf([rb, rd], len(r.chunks), k, depth))
    full = rrf([rb, rd], len(r.chunks), k, depth)
    frames.append({"stage": "fused", "rank": show, "adds": [], "top": [{"chunk": c, "score": full[c]} for c in fused[:show]],
                   "caption": f"all {depth} ranks of both lists fused: the top {show} by RRF score"})
    rr = rerank(fused, r.corpus.rerank_scores()[qi], n)
    ce = r.corpus.rerank_scores()[qi]
    frames.append({"stage": "rerank", "rank": show, "adds": [], "top": [{"chunk": c, "score": ce[c] / 1000} for c in rr[:show]],
                   "caption": f"the cross-encoder reads the question with each of the top {n} and reorders them"})

    def first(order: list[int]) -> int:
        return next((i + 1 for i, c in enumerate(order) if c in rel), 0)

    return {"q": qi, "bm25": rb[:show], "dense": rd[:show], "fused": fused[:show], "reranked": rr[:show], "relevant": rel,
            "first": {"bm25": first(rb), "dense": first(rd), "rrf": first(fused), "rerank": first(rr)}, "frames": frames}


# ---- chapter 5: chunking ---------------------------------------------------------------------

def chunk_view(r_by_config: dict[str, Retriever], article: int, start: int, end: int) -> dict[str, Any]:
    """Each config's chunk boundaries inside [start, end) of one article, the sentence and
    paragraph boundaries, and the neighbouring-sentence similarities used by the semantic chunker."""
    any_r = next(iter(r_by_config.values()))
    corpus = any_r.corpus
    text = corpus.articles[article]["text"]
    p = corpus.pieces[article]
    sents = ranges(sentence_starts(text, p), len(p))
    paras = paragraph_starts(text, p)
    s0 = 0
    for a in range(article):
        s0 += len(sentence_starts(corpus.articles[a]["text"], corpus.pieces[a]))
    sv = corpus.vectors("sentences")[s0:s0 + len(sents)]
    sentences = []
    for i, (sa, sb) in enumerate(sents):
        cs, ce = p[sa][0], p[sb - 1][1]
        if ce <= start or cs >= end:
            continue
        sim = cosine(sv[i], sv[i + 1]) if i + 1 < len(sents) else None
        sentences.append({"i": i, "start": cs, "end": ce, "paragraph": sa in paras, "sim_next": sim})
    configs = {}
    for name, r in r_by_config.items():
        configs[name] = [{"chunk": ci, "start": c["start"], "end": c["end"], "tokens": c["tokens"]}
                         for ci, c in enumerate(r.chunks) if c["article"] == article and c["end"] > start and c["start"] < end]
    qs = [{"q": qi, "start": q["start"], "end": q["end"]} for qi, q in enumerate(corpus.questions)
          if q["article"] == article and q["start"] >= start and q["end"] <= end]
    return {"article": article, "start": start, "end": end, "sentences": sentences, "configs": configs, "answers": qs}
