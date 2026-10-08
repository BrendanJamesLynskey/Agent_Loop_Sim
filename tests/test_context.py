"""The context module (engine 1.4.0): the shipped data and its provenance, the chunkers, BM25,
the vector arithmetic, fusion, the metrics and the window simulation."""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from agent_loop_sim.context.corpus import CHUNKERS, DEFAULT, FILES, default_corpus, read_data
from agent_loop_sim.context.evaluate import evaluate, metrics, relevant
from agent_loop_sim.context.retrieval import BM25, Retriever, rank, rerank, rrf, weighted
from agent_loop_sim.context.text import analyse, fixed
from agent_loop_sim.context.vectors import binary_sim, cosine, to_bits, to_int4
from agent_loop_sim.context.window import POLICIES, task_questions, window_run


@pytest.fixture(scope="module")
def corpus():
    return default_corpus()


@pytest.fixture(scope="module")
def default(corpus):
    return Retriever(corpus, DEFAULT)


def test_files_match_the_manifest():
    m = json.loads(read_data("manifest.json"))
    for n in FILES:
        assert hashlib.sha256(read_data(n).encode("utf-8")).hexdigest() == m["files"][n], n
    assert m["corpus"]["licence"] == "CC BY-SA 4.0"
    assert m["embedder"]["licence"] == m["reranker"]["licence"] == "Apache-2.0"
    assert len(m["embedder"]["revision"]) == len(m["reranker"]["revision"]) == 40


def test_corpus_shape(corpus):
    assert len(corpus.articles) == 6 and len(corpus.questions) == 200
    for q in corpus.questions:
        assert corpus.articles[q["article"]]["text"][q["start"]:q["end"]] == q["answer"]
    for a in corpus.articles:
        assert all(ord(ch) <= 0xFFFF for ch in a["text"])


def test_pieces_tile_each_article(corpus):
    for a, p in zip(corpus.articles, corpus.pieces):
        assert p[0][0] == 0 and p[-1][1] == len(a["text"])
        assert all(p[i][1] == p[i + 1][0] for i in range(len(p) - 1))


@pytest.mark.parametrize("name", list(CHUNKERS))
def test_chunks_are_the_embedded_chunks(corpus, name):
    f = json.loads(read_data(f"emb-{name}.json"))
    assert corpus.spans_sha256(name) == f["spans_sha256"]
    assert f["n"] == len(corpus.chunks(name)) == len(corpus.vectors(name))


@pytest.mark.parametrize("name", list(CHUNKERS))
def test_chunks_cover_the_text_within_size(corpus, name):
    c = CHUNKERS[name]
    chunks = corpus.chunks(name)
    for ai, a in enumerate(corpus.articles):
        mine = [ch for ch in chunks if ch["article"] == ai]
        assert mine[0]["start"] == 0 and mine[-1]["end"] == len(a["text"])
        for x, y in zip(mine, mine[1:]):
            if c.get("overlap"):
                assert x["start"] < y["start"] <= x["end"]
            else:
                assert x["end"] == y["start"]
    assert max(ch["tokens"] for ch in chunks) <= c["size"]


def test_fixed_overlap_repeats_at_most_the_overlap(corpus):
    chunks = corpus.chunks("fixed-256-o64")
    p = corpus.pieces
    for x, y in zip(chunks, chunks[1:]):
        if x["article"] != y["article"]:
            continue
        shared = sum(n for s, e, n in p[x["article"]] if s >= y["start"] and e <= x["end"])
        assert shared <= 64


def test_analyse():
    assert analyse("The Apollo program's Saturn V, 1969!") == ["apollo", "program", "s", "saturn", "v", "1969"]
    assert analyse("The cat", stop=False) == ["the", "cat"]


def test_fixed_rounds_half_up_like_tofixed():
    assert fixed(0.125, 2) == "0.13"  # exactly representable tie: up, as JavaScript does
    assert fixed(2.675, 2) == "2.67"  # 2.675 is really 2.67499999...
    assert fixed(0.5, 0) == "1"


def test_bm25_by_hand():
    bm = BM25(["apple apple banana", "banana cherry", "cherry cherry cherry date"], stop=False)
    assert bm.avgdl == 3.0
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    s = bm.scores("apple")
    assert s[0] == pytest.approx(idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 3)), rel=1e-15)
    assert s[1] == s[2] == 0.0
    assert rank(bm.scores("cherry")) == [2, 1, 0]
    assert bm.query_terms("cherry apple cherry") == ["cherry", "apple"]


def test_ln_is_within_an_ulp_of_libm():
    from agent_loop_sim.context.mathx import ln

    for x in [1.0, 2.0, 0.5, 1e-310, 1e308, 1.0000001, 3.0, 208.5 / 3.5, 1 + 1e-12]:
        assert abs(ln(x) - math.log(x)) <= abs(math.nextafter(math.log(x), math.inf) - math.log(x))
    assert ln(1.0) == 0.0 and ln(0.0) == float("-inf")


def test_vector_arithmetic():
    assert to_int4([127, -127, 9, -9, 10, 0]) == [7, -7, 0, 0, 1, 0]  # 9*7/127 = 0.496, 10*7/127 = 0.551
    assert to_bits([5, -3, 0]) == [1, 0, 0]
    assert cosine([3, 4], [3, 4]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert binary_sim([1, 0, 1, 1], [1, 1, 1, 0]) == 0.0


def test_fusion():
    s = rrf([[2, 0, 1], [0, 2, 1]], 3, k=60, depth=3)
    assert s[0] == s[2] == 1 / 61 + 1 / 62 and s[1] == 2 / 63
    assert rank(s) == [0, 2, 1]
    assert weighted([3.0, 1.0, 2.0], [0.1, 0.9, 0.5], 1.0, depth=3) == [0.0, 1.0, 0.5]
    assert rerank([4, 3, 2, 1, 0], {0: 0, 1: 10, 2: 20, 3: 30, 4: 40}, 3) == [4, 3, 2, 1, 0]
    assert rerank([4, 3, 2, 1, 0], {2: 50, 3: 30, 4: 30}, 3) == [2, 4, 3, 1, 0]
    with pytest.raises(ValueError):
        rerank([4, 3, 2], {4: 1}, 2)


def test_metrics_by_hand():
    m = metrics([5, 1, 7, 2], [7, 9])
    assert m["recall@1"] == 0 and m["recall@3"] == 0.5 and m["mrr@10"] == 1 / 3
    assert m["ndcg@10"] == pytest.approx((1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert metrics([1, 2], []) == {k: 0.0 for k in m}


def test_relevance_needs_the_whole_answer(corpus):
    chunks = [{"article": 0, "start": 0, "end": 10}, {"article": 0, "start": 10, "end": 20}]
    assert relevant(chunks, {"article": 0, "start": 8, "end": 12}) == []
    assert relevant(chunks, {"article": 0, "start": 10, "end": 12}) == [1]


def test_retrieval_is_better_than_chance(default):
    for method in ["bm25", "dense", "rrf", "rerank"]:
        e = evaluate(default, method)
        assert e["lost"] == 0
        assert e["mean"]["recall@10"] > 0.5, method


def test_rerank_only_for_the_recorded_config(corpus):
    with pytest.raises(ValueError):
        Retriever(corpus, "fixed-128").ranking(0, "rerank")


def test_window_policies(default):
    task = task_questions(default, 12)
    assert len(task) == 12 and len({default.corpus.questions[q]["article"] for q in task}) == 6
    for budget in [1000, 1500, 2000]:
        runs = {p: window_run(default, task, budget, p) for p in POLICIES}
        assert runs["unbounded"]["recalled"] == 12 and runs["unbounded"]["peak"] > budget
        for p in POLICIES[1:]:
            fr = runs[p]["frames"]
            for a, b in zip(fr, fr[1:]):
                if a["event"] in ("compact", "truncate") and b["event"] != "truncate":
                    assert a["used"] <= budget  # the policy has brought the window back within budget
        # truncation is cheap and forgets; compaction keeps every fact for fewer tokens than no limit;
        # searching again recovers every fact but pays for the re-reads
        assert runs["truncate"]["recalled"] < 12 and runs["truncate"]["spent"] < runs["unbounded"]["spent"]
        assert runs["compact"]["recalled"] == 12 and runs["compact"]["spent"] < runs["unbounded"]["spent"]
        assert runs["retrieve"]["recalled"] == 12 and runs["retrieve"]["reads"] > runs["truncate"]["reads"]
