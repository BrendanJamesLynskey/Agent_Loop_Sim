"""The context module's fixtures (engine 1.4.0), written by make_fixtures.py:
fixtures/context_fixtures.json (for the TS port to reproduce exactly) and
fixtures/context_results.md (the recorded run every published number comes from)."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_loop_sim import VERSION
from agent_loop_sim.context.corpus import CHUNKERS, DEFAULT, FILES, default_corpus, read_data
from agent_loop_sim.context.evaluate import DISCOUNT, KS, METRICS, evaluate, relevant
from agent_loop_sim.context.retrieval import Retriever, rank
from agent_loop_sim.context.text import analyse, fixed, paragraph_starts, sentence_starts
from agent_loop_sim.context.vectors import PRECISIONS, binary_sim, cosine, nbytes, to_bits, to_int4
from agent_loop_sim.context import views as cviews
from agent_loop_sim.context.mathx import ln
from agent_loop_sim.rng import Rng
from agent_loop_sim.context.window import POLICIES, task_questions, window_run

ANALYSE = ["The Apollo program's Saturn V, 1969!", "Ünïcödé OXYGEN and the O2 molecule", "P versus NP — is it?", ""]
FREE_QUERIES = ["when did the first crewed moon landing happen", "steam engine efficiency and condensers",
                "largest known prime number"]
EVALS: list[tuple[str, dict[str, Any]]] = [
    ("bm25", {}), ("dense", {"precision": "int8"}), ("dense", {"precision": "int4"}), ("dense", {"precision": "binary"}),
    ("rrf", {}), ("weighted", {"alpha": 0.3}), ("weighted", {"alpha": 0.5}), ("weighted", {"alpha": 0.7}),
]
DEFAULT_EVALS: list[tuple[str, dict[str, Any]]] = [
    ("rerank", {"n": 10}), ("rerank", {"n": 20}), ("rerank", {"n": 30}),
    ("rerank", {"base": "bm25", "n": 10}), ("rerank", {"base": "dense", "n": 10}),
    ("bm25", {"k1": 0.5, "b": 0.75}), ("bm25", {"k1": 2.0, "b": 0.75}), ("bm25", {"k1": 1.2, "b": 0.0}),
    ("bm25", {"k1": 1.2, "b": 1.0}), ("rrf", {"k": 10}), ("rrf", {"depth": 10}),
]
WINDOW_BUDGETS = [1000, 1500, 2000, 3000]
TASK_SIZE = 12


def ln_cases() -> list[list[float]]:
    r = Rng(11)
    xs = [1.0, 2.0, 0.5, 1e-310, 1e308, 1.0000001, 0.9999999, 3.0, 10.0, 1.5]
    xs += [1 + r.random() * 400 for _ in range(300)] + [r.random() for _ in range(100)]
    return [[x, ln(x)] for x in xs]


def label(method: str, p: dict[str, Any]) -> str:
    return method + "".join(f" {k}={v}" for k, v in p.items())


def context_fixtures() -> tuple[dict[str, Any], str]:
    c = default_corpus()
    manifest = json.loads(read_data("manifest.json"))
    rs = {name: Retriever(c, name) for name in CHUNKERS}
    d = rs[DEFAULT]
    pieces_sha = [hashlib.sha256("\n".join(f"{a}:{b}:{n}" for a, b, n in p).encode()).hexdigest() for p in c.pieces]
    qv = c.vectors("questions")
    dv = c.vectors(DEFAULT)
    evals = []
    for name in CHUNKERS:
        for method, p in EVALS:
            evals.append(evaluate(rs[name], method, p))
    for method, p in DEFAULT_EVALS:
        evals.append(evaluate(d, method, p))
    task = task_questions(d, TASK_SIZE)
    windows = [window_run(d, task, bud, pol) for bud in WINDOW_BUDGETS for pol in POLICIES]
    excerpt_end = 2400
    fx = {
        "engine": VERSION,
        "files": {n: hashlib.sha256(read_data(n).encode("utf-8")).hexdigest() for n in FILES},
        "manifest_files": manifest["files"],
        "analyse": [{"text": t, "stop": analyse(t), "all": analyse(t, False)} for t in ANALYSE],
        "fixed": [{"x": x, "d": dd, "s": fixed(x, dd)} for x, dd in [(0.125, 2), (2.675, 2), (1.0, 0), (0.5, 0), (1 / 3, 3), (12.3456, 1)]],
        "pieces_sha256": pieces_sha,
        "paragraph_starts": [paragraph_starts(a["text"], c.pieces[i]) for i, a in enumerate(c.articles)],
        "sentence_starts": [sentence_starts(a["text"], c.pieces[i]) for i, a in enumerate(c.articles)],
        "chunks": {name: [[ch["article"], ch["start"], ch["end"], ch["tokens"]] for ch in c.chunks(name)] for name in CHUNKERS},
        "spans_sha256": {name: c.spans_sha256(name) for name in CHUNKERS},
        "vectors": {"int4": [to_int4(v) for v in qv[:3]], "bits": [to_bits(v) for v in qv[:3]],
                    "cosine": [cosine(qv[i], dv[j]) for i in range(5) for j in range(5)],
                    "binary": [binary_sim(to_bits(qv[i]), to_bits(dv[j])) for i in range(5) for j in range(5)],
                    "nbytes": {p: nbytes(384, p) for p in PRECISIONS}},
        "discount": DISCOUNT,
        "ln": ln_cases(),
        "relevant": {name: [relevant(rs[name].chunks, q) for q in c.questions] for name in CHUNKERS},
        "bm25": {name: {"n": rs[name].bm25.n, "avgdl": rs[name].bm25.avgdl, "terms": len(rs[name].bm25.df),
                        "top": [[[i, rs[name].bm25_scores(qi)[i]] for i in rank(rs[name].bm25_scores(qi))[:10]] for qi in range(5)]}
                 for name in CHUNKERS},
        "dense_top": [[[i, d.dense(qi)[i]] for i in rank(d.dense(qi))[:10]] for qi in range(5)],
        "evals": evals,
        "task": task,
        "windows": windows,
        "views": {
            "tf_curve": [cviews.tf_curve(k1, b, ratio) for k1, b, ratio in [(1.2, 0.75, 1.0), (0.5, 0.75, 2.0), (2.0, 0.0, 0.5)]],
            "length_curve": [cviews.length_curve(k1, b, tf) for k1, b, tf in [(1.2, 0.75, 1), (1.2, 1.0, 3), (2.0, 0.3, 1)]],
            "bm25": [cviews.bm25_view(d, q) for q in [c.questions[0]["question"], c.questions[17]["question"]] + FREE_QUERIES]
            + [cviews.bm25_view(d, FREE_QUERIES[0], 2.0, 0.2, 5)],
            "dense": cviews.dense_frames(d, list(range(10))) + cviews.dense_frames(d, list(range(5)), "binary", 3),
            "hybrid": [cviews.hybrid_frames(d, qi) for qi in range(6)] + [cviews.hybrid_frames(d, 7, 60, 50, 8, 10)],
            "chunk": cviews.chunk_view(rs, 0, 0, excerpt_end),
        },
    }
    return fx, results_md(c, manifest, rs, evals, windows, task)


def _row(e: dict[str, Any]) -> str:
    m = e["mean"]
    return " | ".join(fixed(m[k], 3) for k in METRICS)


def results_md(c: Any, manifest: dict[str, Any], rs: dict[str, Retriever], evals: list[dict[str, Any]],
               windows: list[dict[str, Any]], task: list[int]) -> str:
    out = [f"# Context module results (engine {VERSION})", "",
           "Written by `scripts/make_fixtures.py` from the shipped data (CI checks it is up to date). Every number on the "
           "Context site and in the README comes from here or from the same engine at build time.", "",
           f"Corpus: {len(c.articles)} articles of the {manifest['corpus']['source']} ({manifest['corpus']['licence']}), "
           f"{len(c.questions)} questions (seed {manifest['corpus']['sample_seed']} from {manifest['corpus']['question_pool']}). "
           f"Embedder `{manifest['embedder']['repo']}` @ `{manifest['embedder']['revision'][:7]}` (int8, per-vector scale); "
           f"reranker `{manifest['reranker']['repo']}` @ `{manifest['reranker']['revision'][:7]}` ({DEFAULT} only).", "",
           "## Chunking", "", "| config | chunks | mean tokens | max tokens | questions lost (answer cut) | chunks over the embedder's 256 word pieces |",
           "|---|---|---|---|---|---|"]
    for name, r in rs.items():
        toks = [ch["tokens"] for ch in r.chunks]
        s = 0
        for t in toks:
            s += t
        lost = next(e["lost"] for e in evals if e["config"] == name)
        out.append(f"| {name} | {len(toks)} | {fixed(s / len(toks), 1)} | {max(toks)} | {lost} | "
                   f"{json.loads(read_data(f'emb-{name}.json'))['truncated']} |")
    out += ["", "## Retrieval (means over all questions; a question with its answer cut by chunking scores 0)", "",
            "| config | method | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for e in evals:
        out.append(f"| {e['config']} | {label(e['method'], e['params'])} | {_row(e)} |")
    out += ["", "## Offline float32 dense retrieval (recorded by scripts/build_context_data.py; not reproducible from the shipped int8 vectors)", "",
            "| config | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for name, m in manifest["offline_float32"].items():
        out.append(f"| {name} | " + " | ".join(fixed(m[k], 3) for k in METRICS) + " |")
    out += ["", f"## The window task ({len(task)} questions, BM25 over {DEFAULT}, up to 3 reads each)", "",
            "| budget | policy | facts answered | input tokens spent | model calls | reads | compactions | reads dropped | peak window |",
            "|---|---|---|---|---|---|---|---|---|"]
    for w in windows:
        out.append(f"| {w['budget']} | {w['policy']} | {w['recalled']} of {len(task)} | {w['spent']} | {w['calls']} | {w['reads']} | "
                   f"{w['compactions']} | {w['dropped']} | {w['peak']} |")
    return "\n".join(out) + "\n"


__all__ = ["context_fixtures", "KS"]
