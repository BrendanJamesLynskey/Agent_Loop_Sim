"""The window as working memory: a long research task run by a scripted agent under a token
budget, with what the agent "knows" at each step.

The task is a list of questions over the corpus. For each one the agent searches (BM25 over the
default chunking) and reads chunks, best first, until it has read the chunk holding the answer
(at most ``MAX_READS``); each read is one model call whose input is the whole window. A fact is
**known** while the chunk that holds it (or a summary line that states it) is in the window.
When the window passes its budget the policy acts:

- ``unbounded``: nothing (the budget is ignored; the reference for cost);
- ``truncate``: the oldest reads are dropped until the window fits;
- ``compact``: one extra model call summarises every read but the latest into one line per fact
  ("question -> answer"), then the oldest summary lines are dropped if it still does not fit. The
  scripted summariser is ``perfect`` (it keeps exactly the facts in those reads) or, since engine
  1.5, ``lossy``: each compaction rewrites the whole summary and keeps each line with probability
  ``1 - loss`` (a seeded draw, the same in Python and TS), so a fact summarised n times survives
  with probability (1 - loss)^n;
- ``retrieve``: truncate, and at the end search again for every fact no longer in the window;
- ``compact+retrieve``: both.

At the end the agent answers the questions in order, each from what its window holds at that
moment (the retrieve policies search again first for a fact that has gone). Token counts are Qwen2.5's.
"""
from __future__ import annotations

from typing import Any

from ..rng import Rng
from .retrieval import Retriever, rank
from .evaluate import relevant

POLICIES = ["unbounded", "truncate", "compact", "retrieve", "compact+retrieve"]
SUMMARISERS = ["perfect", "lossy"]
MAX_READS = 3
SYSTEM = ("You are a research agent. Answer every question in the task. Use the search tool to read the "
          "corpus; you can only answer from what is in your context.")


def task_questions(r: Retriever, m: int) -> list[int]:
    """``m`` questions whose answer BM25 finds within ``MAX_READS`` reads, taken round-robin over the
    articles in corpus order (so the task wanders between topics)."""
    by_article: dict[int, list[int]] = {}
    for qi, q in enumerate(r.corpus.questions):
        rel = relevant(r.chunks, q)
        order = rank(r.bm25_scores(qi))
        if rel and order.index(rel[0]) < MAX_READS:
            by_article.setdefault(q["article"], []).append(qi)
    out: list[int] = []
    arts = sorted(by_article)
    j = 0
    while len(out) < m and any(by_article[a] for a in arts):
        a = arts[j % len(arts)]
        if by_article[a]:
            out.append(by_article[a].pop(0))
        j += 1
    return out


def _used(items: list[dict[str, Any]]) -> int:
    t = 0
    for it in items:
        t += it["tokens"]
    return t


def _known(items: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    for it in items:
        for f in it["facts"]:
            if f not in out:
                out.append(f)
    return sorted(out)


def _note(r: Retriever, qi: int) -> str:
    q = r.corpus.questions[qi]
    return f"- {q['question']} -> {q['answer']}\n"


def _facts_for(questions: list[int], facts: list[int]) -> str:
    """"the fact for question 2", "the facts for questions 2 and 5", "the facts for questions 2, 5 and 7"
    (1-based positions in the task)."""
    n = [str(questions.index(f) + 1) for f in facts]
    if len(n) == 1:
        return f"the fact for question {n[0]}"
    return f"the facts for questions {', '.join(n[:-1])} and {n[-1]}"


def window_run(r: Retriever, questions: list[int], budget: int, policy: str, summariser: str = "perfect",
               loss: float = 0.25, seed: int = 23) -> dict[str, Any]:
    """One run of the task. ``summariser``, ``loss`` and ``seed`` matter only to the compact policies;
    a ``lossy`` run also reports, for every fact that reached a summary, what each compaction did to it."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if summariser not in SUMMARISERS:
        raise ValueError(f"unknown summariser {summariser!r}")
    rng = Rng(seed)
    fates: dict[int, list[bool]] = {}
    tok = r.corpus.tok
    task = "Task: answer these questions.\n" + "".join(f"{i + 1}. {r.corpus.questions[qi]['question']}\n"
                                                       for i, qi in enumerate(questions))
    items: list[dict[str, Any]] = [
        {"id": "system", "kind": "system", "tokens": tok.count(SYSTEM), "facts": []},
        {"id": "task", "kind": "task", "tokens": tok.count(task), "facts": []},
    ]
    frames: list[dict[str, Any]] = []
    st = {"spent": 0, "out": 0, "calls": 0, "reads": 0, "compactions": 0, "dropped": 0}
    learned: list[int] = []

    def frame(event: str, qi: int | None, caption: str) -> None:
        frames.append({"step": len(frames), "event": event, "q": qi,
                       "items": [{"id": it["id"], "kind": it["kind"], "tokens": it["tokens"], "facts": list(it["facts"])}
                                 for it in items],
                       "used": _used(items), "budget": budget, "known": _known(items), "learned": sorted(learned),
                       "spent": st["spent"], "calls": st["calls"], "caption": caption})

    def call(out_text: str) -> None:
        st["spent"] += _used(items)
        st["calls"] += 1
        st["out"] += tok.count(out_text)

    def truncate(qi: int | None) -> None:
        while _used(items) > budget:
            k = next((i for i, it in enumerate(items) if it["kind"] in ("read", "summary")), -1)
            if k < 0 or k == len(items) - 1:
                break
            it = items[k]
            if it["kind"] == "summary" and len(it["lines"]) > 1:
                f = it["lines"].pop(0)
                it["facts"].remove(f)
                it["tokens"] = tok.count("Summary of earlier reads:\n" + "".join(_note(r, x) for x in it["lines"]))
                frame("truncate", qi, f"window over budget: the oldest summary line (question {questions.index(f) + 1}) is dropped")
            else:
                items.pop(k)
                st["dropped"] += 1
                lost = f"{_facts_for(questions, it['facts'])} is lost" if it["facts"] else "it held no fact"
                frame("truncate", qi, f"window over budget ({_used(items) + it['tokens']} > {budget}): the oldest read, {it['id']}, "
                                      f"is dropped ({lost})")

    def compact(qi: int | None) -> None:
        reads = [it for it in items if it["kind"] == "read"]
        if len(reads) < 2:
            truncate(qi)
            return
        old = reads[:-1]
        prev = next((it for it in items if it["kind"] == "summary"), None)
        lines = list(prev["lines"]) if prev else []
        for it in old:
            for f in it["facts"]:
                if f not in lines:
                    lines.append(f)
        dropped: list[int] = []
        if summariser == "lossy":
            kept: list[int] = []
            for f in lines:
                keep_it = rng.random() >= loss
                fates.setdefault(f, []).append(keep_it)
                if keep_it:
                    kept.append(f)
                else:
                    dropped.append(f)
            lines = kept
        text = "Summary of earlier reads:\n" + "".join(_note(r, x) for x in lines)
        call(text)
        st["compactions"] += 1
        old_tokens = 0
        for it in old:
            old_tokens += it["tokens"]
        keep = [it for it in items if it["kind"] in ("system", "task")]
        summ = {"id": "summary", "kind": "summary", "tokens": tok.count(text), "facts": list(lines), "lines": lines}
        items[:] = keep + [summ] + [reads[-1]]
        frame("compact", qi, f"compaction: {len(old)} reads ({old_tokens} tokens) "
                             f"become a {summ['tokens']}-token summary of {len(lines)} facts"
                             + (f"; the summariser drops {_facts_for(questions, dropped)}" if dropped else ""))
        truncate(qi)

    def read(qi: int, c: int, phase: str) -> bool:
        ch = r.chunks[c]
        has = r.corpus.questions[qi]["start"] >= ch["start"] and r.corpus.questions[qi]["end"] <= ch["end"] \
            and ch["article"] == r.corpus.questions[qi]["article"]
        call(f'search("{r.corpus.questions[qi]["question"]}")')
        st["reads"] += 1
        items.append({"id": f"chunk {c}", "kind": "read", "tokens": ch["tokens"], "facts": [qi] if has else []})
        if has and qi not in learned:
            learned.append(qi)
        n = questions.index(qi) + 1
        frame(phase, qi, f"question {n}: reads chunk {c} ({ch['tokens']} tokens) — "
                         + ("it holds the answer" if has else "no answer in it"))
        if _used(items) > budget and policy != "unbounded":
            if policy.startswith("compact"):
                compact(qi)
            else:
                truncate(qi)
        return has

    def search(qi: int, phase: str) -> bool:
        order = rank(r.bm25_scores(qi))
        for c in order[:MAX_READS]:
            if read(qi, c, phase):
                return True
        return False

    frame("start", None, f"the window starts with the system prompt and the task: {_used(items)} of {budget} tokens")
    for qi in questions:
        search(qi, "read")
    answered: list[int] = []
    for qi in questions:
        if qi not in _known(items) and policy.endswith("retrieve"):
            frame("recall", qi, f"question {questions.index(qi) + 1}: its fact is no longer in the window, so the agent searches again")
            search(qi, "reread")
        if qi in _known(items):
            answered.append(qi)
    call("Answers:\n" + "".join(f"{questions.index(qi) + 1}. {r.corpus.questions[qi]['answer']}\n" for qi in answered))
    frame("answer", None, f"the agent answers {len(answered)} of {len(questions)} questions from its window")
    out = {"policy": policy, "budget": budget, "questions": questions, "recalled": len(answered),
           "spent": st["spent"], "out": st["out"], "calls": st["calls"], "reads": st["reads"],
           "compactions": st["compactions"], "dropped": st["dropped"], "peak": max(f["used"] for f in frames),
           "frames": frames}
    if summariser == "lossy":
        out["summariser"] = summariser
        out["loss"] = loss
        out["seed"] = seed
        out["answered"] = answered
        # each fact that reached a summary: [question, [kept at its 1st compaction, kept at its 2nd, ...]]
        out["fates"] = [[f, fates[f]] for f in sorted(fates)]
    return out


def compaction_study(r: Retriever, questions: list[int], budget: int, policy: str, loss: float,
                     seeds: list[int]) -> dict[str, Any]:
    """What survives a lossy summariser, over many seeded runs of one task: the mean facts answered
    and input tokens, the share of runs answering each question (by its place in the task), and the
    survival curve: S(n) = product over j <= n of (facts kept at their j-th compaction / facts that
    reached a j-th compaction), against the model's (1 - loss)^n."""
    answered_at = [0 for _ in questions]
    rec = 0
    spent = 0
    faced: list[int] = []
    kept: list[int] = []
    for sd in seeds:
        w = window_run(r, questions, budget, policy, "lossy", loss, sd)
        rec += w["recalled"]
        spent += w["spent"]
        for qi in w["answered"]:
            answered_at[questions.index(qi)] += 1
        for _, fate in w["fates"]:
            for j, k in enumerate(fate):
                while len(faced) <= j:
                    faced.append(0)
                    kept.append(0)
                faced[j] += 1
                if k:
                    kept[j] += 1
    survival = []
    s = 1.0
    t = 1.0
    for j in range(len(faced)):
        s *= kept[j] / faced[j]
        t *= 1 - loss
        survival.append({"n": j + 1, "faced": faced[j], "kept": kept[j], "measured": s, "model": t})
    n = len(seeds)
    return {"policy": policy, "budget": budget, "loss": loss, "runs": n, "recalled": rec / n, "spent": spent / n,
            "by_position": [a / n for a in answered_at], "survival": survival}
