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
  ("question -> answer", a scripted summariser that keeps exactly the facts in those reads), then
  the oldest summary lines are dropped if it still does not fit;
- ``retrieve``: truncate, and at the end search again for every fact no longer in the window;
- ``compact+retrieve``: both.

At the end the agent answers the questions in order, each from what its window holds at that
moment (the retrieve policies search again first for a fact that has gone). Token counts are Qwen2.5's.
"""
from __future__ import annotations

from typing import Any

from .retrieval import Retriever, rank
from .evaluate import relevant

POLICIES = ["unbounded", "truncate", "compact", "retrieve", "compact+retrieve"]
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


def window_run(r: Retriever, questions: list[int], budget: int, policy: str) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
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
                lost = ", ".join(str(questions.index(f) + 1) for f in it["facts"]) or "none"
                frame("truncate", qi, f"window over budget ({_used(items) + it['tokens']} > {budget}): the oldest read, {it['id']}, "
                                      f"is dropped (facts lost: {lost})")

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
                             f"become a {summ['tokens']}-token summary of {len(lines)} facts")
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
    return {"policy": policy, "budget": budget, "questions": questions, "recalled": len(answered),
            "spent": st["spent"], "out": st["out"], "calls": st["calls"], "reads": st["reads"],
            "compactions": st["compactions"], "dropped": st["dropped"], "peak": max(f["used"] for f in frames),
            "frames": frames}
