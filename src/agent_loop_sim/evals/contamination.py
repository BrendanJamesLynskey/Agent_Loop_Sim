"""Contamination and leakage (since 1.9): what leaked benchmark tasks do to a score, and what three
detection methods see, on the synthetic benchmark.

**The benchmark.** ``tasks`` tasks are drawn from ``generator``'s population; each gets a text (a
sequence of ``length`` token ids from a vocabulary of ``vocab``) and a creation quarter, 0 to 7. The
model's training data was collected at the end of quarter 3, so only tasks from quarters 0 to 3 can
have leaked into it.

**Leaks.** At leak rate L, a task created before the cutoff is in the training data with probability L.
A leaked copy is verbatim, or (with probability ``paraphrase``) a paraphrase in which each token is
replaced with probability ``mutate``; a verbatim copy keeps the benchmark's canary string with
probability ``canary_kept`` (scrapers and filters often drop it). A leaked task is answered from
memory with probability ``memorise``, so P(pass) = μ + (1 − μ)·memorise instead of μ (μ from the
item-response model). Every draw is shared across leak rates (a task leaks at L if its leak draw is
below L), so the scores at different rates differ only by the leaks.

**Detection.**
- *n-gram overlap*: the share of a task's ``ngram``-grams found anywhere in the training data, flagged
  at ``threshold`` (GPT-3's contamination study used 13-grams over the training set, Brown et al.
  2020, arXiv:2005.14165; the n here is shorter because the texts are). Verbatim copies overlap
  completely; a paraphrase keeps an n-gram only if all n tokens survive, (1 − mutate)^n, and slips
  through.
- *canary strings*: BIG-bench's files carry a canary GUID (Srivastava et al. 2022,
  arXiv:2206.04615); a model that can complete it has seen the files. This says the benchmark leaked,
  not which tasks.
- *time split*: tasks created after the cutoff cannot have leaked; a score that drops at the cutoff
  is a sign of leakage (the idea behind LiveCodeBench, Jain et al. 2024, arXiv:2403.07974). The two
  sides are compared with the two-proportion test.

Every rate here is illustrative; the token "texts" are random ids, not language.
"""
from __future__ import annotations

from typing import Any

from ..context.text import fixed
from ..rng import Rng
from . import generator as G
from . import stats as S

CONTAM = {"tasks": 400, "theta": 0.0, "seed": 2591, "length": 48, "vocab": 5000, "ngram": 8, "threshold": 0.5,
          "quarters": 8, "cutoff": 4, "memorise": 0.7, "paraphrase": 0.4, "mutate": 0.25, "canary_kept": 0.6,
          "filler_docs": 300, "leaks": [0.0, 0.1, 0.3, 0.6]}


def pct(x: float) -> str:
    return fixed(x * 100.0 + 0.0, 1) + "%"


def ngrams(toks: list[int], n: int) -> list[str]:
    return [",".join(str(t) for t in toks[i:i + n]) for i in range(len(toks) - n + 1)]


def build(cfg: dict[str, Any] = CONTAM, pop: list[dict[str, float]] | None = None) -> dict[str, Any]:
    """The tasks and every draw they need (stream ``seed``, per task in order: the text, the quarter, the
    leak, the paraphrase and canary draws, the run, then one mutation draw and one replacement token
    per position), the tasks' difficulties (``seed + 2``) and the filler documents (``seed + 1``)."""
    p = pop if pop is not None else G.population()
    idx = G.sample(cfg["tasks"], len(p), Rng(cfg["seed"] + 2))
    mu = G.mus([p[i] for i in idx], cfg["theta"])
    r = Rng(cfg["seed"])
    tasks = []
    for t in range(cfg["tasks"]):
        text = [r.randint(0, cfg["vocab"] - 1) for _ in range(cfg["length"])]
        quarter = r.randint(0, cfg["quarters"] - 1)
        u_leak = r.random()
        u_para = r.random()
        u_canary = r.random()
        u_run = r.random()
        para = []
        for tok in text:
            m = r.random()
            rep = r.randint(0, cfg["vocab"] - 1)
            para.append(rep if m < cfg["mutate"] else tok)
        tasks.append({"i": t, "mu": mu[t], "text": text, "quarter": quarter, "pre": quarter < cfg["cutoff"],
                      "u_leak": u_leak, "paraphrased": u_para < cfg["paraphrase"],
                      "canary": u_para >= cfg["paraphrase"] and u_canary < cfg["canary_kept"], "u_run": u_run, "para": para})
    rf = Rng(cfg["seed"] + 1)
    filler = [[rf.randint(0, cfg["vocab"] - 1) for _ in range(cfg["length"])] for _ in range(cfg["filler_docs"])]
    return {"tasks": tasks, "filler": filler}


def at_leak(leak: float, b: dict[str, Any], cfg: dict[str, Any] = CONTAM) -> dict[str, Any]:
    n = cfg["ngram"]
    tasks = b["tasks"]
    corpus: set[str] = set()
    for doc in b["filler"]:
        for g in ngrams(doc, n):
            corpus.add(g)
    leaked = [t["pre"] and t["u_leak"] < leak for t in tasks]
    for t, lk in zip(tasks, leaked):
        if lk:
            for g in ngrams(t["para"] if t["paraphrased"] else t["text"], n):
                corpus.add(g)
    rows = []
    for t, lk in zip(tasks, leaked):
        grams = ngrams(t["text"], n)
        hit = 0
        for g in grams:
            if g in corpus:
                hit += 1
        overlap = hit / len(grams)
        p = t["mu"] + (1.0 - t["mu"]) * cfg["memorise"] if lk else t["mu"]
        rows.append({"i": t["i"], "quarter": t["quarter"], "pre": t["pre"], "leaked": lk,
                     "kind": ("paraphrase" if t["paraphrased"] else "verbatim") if lk else "none",
                     "overlap": overlap, "flagged": overlap >= cfg["threshold"], "pass": t["u_run"] < p,
                     "clean_pass": t["u_run"] < t["mu"]})
    return summarise(leak, rows, any(lk and t["canary"] for t, lk in zip(tasks, leaked)), cfg)


def summarise(leak: float, rows: list[dict[str, Any]], canary: bool, cfg: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)

    def count(f: Any) -> int:
        k = 0
        for r in rows:
            if f(r):
                k += 1
        return k

    k = count(lambda r: r["pass"])
    k_clean = count(lambda r: r["clean_pass"])
    n_leaked = count(lambda r: r["leaked"])
    n_verb = count(lambda r: r["kind"] == "verbatim")
    n_para = count(lambda r: r["kind"] == "paraphrase")
    flag_verb = count(lambda r: r["kind"] == "verbatim" and r["flagged"])
    flag_para = count(lambda r: r["kind"] == "paraphrase" and r["flagged"])
    flag_clean = count(lambda r: not r["leaked"] and r["flagged"])
    kept = [r for r in rows if not r["flagged"]]
    k_kept = 0
    for r in kept:
        if r["pass"]:
            k_kept += 1
    pre = [r for r in rows if r["pre"]]
    post = [r for r in rows if not r["pre"]]
    k_pre = 0
    for r in pre:
        if r["pass"]:
            k_pre += 1
    k_post = 0
    for r in post:
        if r["pass"]:
            k_post += 1
    by_q = []
    for q in range(cfg["quarters"]):
        rq = [r for r in rows if r["quarter"] == q]
        kq = 0
        for r in rq:
            if r["pass"]:
                kq += 1
        by_q.append({"quarter": q, "n": len(rq), "k": kq, "rate": kq / len(rq), "wilson": S.wilson(kq, len(rq))})
    split = S.two_proportion(k_pre, len(pre), k_post, len(post))
    frames = [
        {"stage": "tasks", "caption": f"{n} tasks, each with a creation date; the model's training data stops at the "
                                      f"cutoff, after quarter {cfg['cutoff'] - 1}."},
        {"stage": "leak", "caption": f"Leak rate {pct(leak)}: {n_leaked} tasks from before the cutoff are in the training "
                                     f"data, {n_verb} verbatim and {n_para} paraphrased."},
        {"stage": "scan", "caption": f"The {cfg['ngram']}-gram scan flags {flag_verb} of {n_verb} verbatim leaks and "
                                     f"{flag_para} of {n_para} paraphrases" + (f", and {flag_clean} clean tasks" if flag_clean else "")
                                     + ". " + ("The canary string comes back from the model: the benchmark leaked." if canary
                                               else "The model does not complete the canary string.")},
        {"stage": "run", "caption": f"The agent scores {pct(k / n)}; without the leaks it would score {pct(k_clean / n)}. "
                                    f"Dropping the flagged tasks gives {pct(k_kept / len(kept))}."},
        {"stage": "split", "caption": f"Before the cutoff {pct(k_pre / len(pre))}, after it {pct(k_post / len(post))}: "
                                      f"a gap of {fixed((k_pre / len(pre) - k_post / len(post)) * 100.0 + 0.0, 1)} points "
                                      f"(p = {fixed(split['p'] + 0.0, 3)})."},
    ]
    return {"leak": leak, "n": n, "rate": k / n, "clean_rate": k_clean / n, "inflation": (k - k_clean) / n,
            "leaked": n_leaked, "verbatim": n_verb, "paraphrased": n_para, "flagged_verbatim": flag_verb,
            "flagged_paraphrased": flag_para, "flagged_clean": flag_clean, "canary": canary,
            "kept": len(kept), "rate_kept": k_kept / len(kept), "pre": {"n": len(pre), "k": k_pre},
            "post": {"n": len(post), "k": k_post}, "split": split, "by_quarter": by_q, "rows": rows, "frames": frames}


def contamination_study(cfg: dict[str, Any] = CONTAM, pop: list[dict[str, float]] | None = None) -> dict[str, Any]:
    b = build(cfg, pop)
    survive = 1.0
    for _ in range(cfg["ngram"]):
        survive = survive * (1.0 - cfg["mutate"])
    return {"cfg": cfg, "survive": survive, "levels": {fixed(L + 0.0, 2): at_leak(L, b, cfg) for L in cfg["leaks"]}}
