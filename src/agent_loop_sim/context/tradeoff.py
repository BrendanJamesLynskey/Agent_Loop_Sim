"""Long context or retrieval? (added in engine 1.5.0): what a run of questions over a document set
costs when the whole set goes into every prompt, with and without the prompt cache, against
retrieving k chunks per question; and how often the answer is in the prompt.

Every price, cache rule and latency comes from ``accounting`` (``PRICES``, ``call_cost``,
``call_latency``: dated list prices, **illustrative**). Token counts are measured on the corpus: the
system prompt, the mean question and the mean chunk of the default chunking (Qwen2.5's tokenizer).
A document set of ``n_ctx`` tokens larger than the corpus is hypothetical: the same arithmetic for
a bigger set; whether a model accepts a prompt that long, and any long-prompt surcharge, are not
modelled. Questions arrive ``GAP_MS`` apart (inside every cache's lifetime).

- ``long``: system + the whole set + the question, every call, no cache;
- ``long+cache``: the same prompt; the system + set prefix is written to the cache on the first
  call and read from it after (the engine's cache rule: every prompt of at least ``min_tokens`` is
  written; the cached part is the prefix rounded down to the provider's block);
- ``rag``: system + the top k chunks (after reranking) + the question, no cache (the chunks change).

Answer in the prompt: 1 for the long prompts (the whole set is there), the measured recall@k of the
reranked retrieval (chapter 4) for ``rag``.
"""
from __future__ import annotations

from typing import Any

from ..accounting import LATENCY, PRICES, call_cost, call_latency
from .evaluate import evaluate
from .retrieval import Retriever
from .window import SYSTEM

STRATEGIES = ["long", "long+cache", "rag"]
GAP_MS = 60000
OUT_TOKENS = 50
MODELS = ["claude-sonnet-4.6", "claude-haiku-4.5", "gpt-5-mini"]


def measured_sizes(r: Retriever) -> dict[str, Any]:
    tok = r.corpus.tok
    qt = 0
    for q in r.corpus.questions:
        qt += tok.count(q["question"])
    ct = 0
    for c in r.chunks:
        ct += c["tokens"]
    return {"system": tok.count(SYSTEM), "question": qt / len(r.corpus.questions), "chunk": ct / len(r.chunks),
            "corpus": ct, "chunks": len(r.chunks)}


def run_questions(sizes: dict[str, Any], model: str, n_ctx: int, k: int, questions: int,
                  latency: str = "hosted") -> dict[str, Any]:
    """Per strategy and question: input tokens, cached tokens, cost (US dollars), cumulative cost and
    time to first token (ms)."""
    price = PRICES[model]
    prof = LATENCY[latency]
    q = int(sizes["question"] + 0.5)  # round half up, as Math.round
    sysn = sizes["system"]
    out: dict[str, Any] = {"model": model, "n_ctx": n_ctx, "k": k, "questions": questions, "q_tokens": q,
                           "out_tokens": OUT_TOKENS, "strategies": {}}
    for s in STRATEGIES:
        rows: dict[str, list[Any]] = {"input": [], "cached": [], "cost": [], "cum": [], "ttft": []}
        cum = 0.0
        alive = -1.0  # when the cached prefix was last used (-1: not cached)
        for i in range(questions):
            now = i * GAP_MS
            if s == "rag":
                inp = sysn + k * int(sizes["chunk"] + 0.5) + q
                cached = 0
                stored = False
            else:
                prefix = sysn + n_ctx
                inp = prefix + q
                cached = 0
                stored = False
                if s == "long+cache":
                    if alive >= 0 and now - alive <= price["ttl_ms"]:
                        c = (prefix // price["block"]) * price["block"]
                        cached = c if c >= price["min_tokens"] else 0
                    stored = inp >= price["min_tokens"]
                    if stored or cached:
                        alive = now
            cost = call_cost(price, inp, cached, stored, OUT_TOKENS)
            cum += cost
            ttft, _ = call_latency(prof, inp, cached, OUT_TOKENS)
            rows["input"].append(inp)
            rows["cached"].append(cached)
            rows["cost"].append(cost)
            rows["cum"].append(cum)
            rows["ttft"].append(ttft)
        out["strategies"][s] = rows
    return out


def tradeoff(r: Retriever, n_ctxs: list[int], ks: list[int], questions: int,
             models: list[str] | None = None) -> dict[str, Any]:
    """The grid the chapter draws: every model x set size x k, plus each k's measured recall."""
    sizes = measured_sizes(r)
    ev = evaluate(r, "rerank", {"n": 20})["mean"]
    runs = {}
    for m in models or MODELS:
        for n in n_ctxs:
            for k in ks:
                runs[f"{m}|{n}|{k}"] = run_questions(sizes, m, n, k, questions)
    return {"sizes": sizes, "recall": {str(k): ev[f"recall@{k}"] for k in ks}, "runs": runs,
            "prices": {m: {key: PRICES[m][key] for key in ["label", "input", "output", "cache_read", "cache_write",
                                                           "min_tokens", "block", "ttl_ms", "source", "accessed"]}
                       for m in models or MODELS},
            "latency": LATENCY["hosted"]}
