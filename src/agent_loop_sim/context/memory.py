"""Agent memory across sessions (added in engine 1.5.0): what a scripted agent can recall in a later
session, and how many tokens its memory costs.

**The task.** ``SESSIONS`` sessions, each a day (24 hours) apart. In each, the agent researches
``PER_SESSION`` questions as in the window task (BM25 over the default chunking, up to three reads
until it reads the chunk holding the answer); when the session ends its window is gone and only its
memory remains. At the start of every later session it is **probed**: each question it researched comes back
once (``repeat``), the session's first question one session later, the second two later, the third
three later; and, two sessions later, a ``neighbour`` when there is one: another labelled question
whose answer lies in a chunk it read that session but was never asked. A final session holds probes
only (and every probe that would fall after it). A probe is **recalled** when the memory the agent puts in that call holds the answer span.

**Policies** (memory written at the end of a session, read at a probe):
- ``none``: no memory;
- ``transcript``: every earlier read, loaded whole at each probe;
- ``scratchpad``: a notes file the agent writes, one "question -> answer" line per question it
  answered, loaded whole; with ``cap`` (tokens) the oldest lines are forgotten first;
- ``episodic``: every read is an episode (the chunk, when, and an importance rating); at a probe the
  top ``k`` by recency + importance + relevance, each min-max normalised over the store (the
  Generative Agents scoring, Park et al. 2023, arXiv:2304.03442: recency decays by 0.995 per hour
  since the episode was last retrieved); consolidation merges a re-read chunk into its episode; with
  ``cap`` episodes, the one with the lowest recency + importance is forgotten;
- ``semantic``: a scripted extractor turns each read into facts: the sentence holding the answer
  it was looking for (if the read had it) and the chunk's first sentence; consolidation merges a fact
  extracted twice; at a probe the top ``k`` facts by relevance (cosine of the int8 embeddings); with
  ``cap`` facts, the lowest recency + importance is forgotten.

The scripted importance ratings: 8 for a read (or fact) that answered its question, 3 for another
read, 5 for a chunk's first sentence. Tokens are Qwen2.5's. **Read tokens** are the memory tokens put
into probe calls; **write tokens** are what writing the memory costs in model calls (the notes the
agent writes; for episodic memory, a rating call that reads the chunk; for semantic memory, an
extraction call that reads the chunk and writes the facts).
"""
from __future__ import annotations

from typing import Any

from .corpus import DEFAULT
from .evaluate import relevant
from .retrieval import Retriever, rank
from .vectors import cosine
from .window import MAX_READS, task_questions

SESSIONS = 6
PER_SESSION = 3
HOURS = 24
DECAY = 0.995
IMPORTANCE = {"answer": 8, "other": 3, "topic": 5}
POLICIES: dict[str, dict[str, Any]] = {
    "none": {"kind": "none"},
    "transcript": {"kind": "transcript"},
    "scratchpad": {"kind": "scratchpad"},
    "scratchpad-cap": {"kind": "scratchpad", "cap": 128},
    "episodic": {"kind": "episodic", "k": 3},
    "episodic-cap": {"kind": "episodic", "k": 3, "cap": 8},
    "semantic": {"kind": "semantic", "k": 3},
    "semantic-cap": {"kind": "semantic", "k": 3, "cap": 10},
}
SWEEP: list[tuple[str, dict[str, Any]]] = [
    ("episodic k=1", {"kind": "episodic", "k": 1}), ("episodic k=2", {"kind": "episodic", "k": 2}),
    ("episodic k=5", {"kind": "episodic", "k": 5}),
    ("semantic k=1", {"kind": "semantic", "k": 1}), ("semantic k=5", {"kind": "semantic", "k": 5}),
    ("semantic k=8", {"kind": "semantic", "k": 8}),
]


def recency(hours: int) -> float:
    """DECAY ** hours, as repeated multiplication (bit-identical in Python and TS)."""
    x = 1.0
    for _ in range(hours):
        x *= DECAY
    return x


def _norm(xs: list[float]) -> list[float]:
    lo = min(xs)
    hi = max(xs)
    if hi == lo:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _within(q: dict[str, Any], span: list[int]) -> bool:
    return span[0] == q["article"] and span[1] <= q["start"] and q["end"] <= span[2]


def memory_plan(r: Retriever, sessions: int = SESSIONS, per: int = PER_SESSION) -> dict[str, Any]:
    """The task: each session's questions and reads, and each later session's probes."""
    qs = r.corpus.questions
    learn = task_questions(r, sessions * per)
    plan_s = []
    for s in range(sessions):
        reads = []
        for qi in learn[s * per:(s + 1) * per]:
            for c in rank(r.bm25_scores(qi))[:MAX_READS]:
                ch = r.chunks[c]
                has = _within(qs[qi], [ch["article"], ch["start"], ch["end"]])
                reads.append([qi, c, has])
                if has:
                    break
        plan_s.append({"learn": learn[s * per:(s + 1) * per], "reads": reads})
    used = set(learn)
    probes: list[list[dict[str, Any]]] = [[] for _ in range(sessions + 1)]
    for t in range(sessions):
        for j, qi in enumerate(plan_s[t]["learn"]):
            probes[min(t + 1 + j, sessions)].append({"q": qi, "kind": "repeat", "about": t})
        read = {c for _, c, _ in plan_s[t]["reads"]}
        for qi in range(len(qs)):
            if qi not in used and any(c in read for c in relevant(r.chunks, qs[qi])):
                used.add(qi)
                probes[min(t + 2, sessions)].append({"q": qi, "kind": "neighbour", "about": t})
                break
    for row in probes:
        row.sort(key=lambda p: (p["about"], 0 if p["kind"] == "repeat" else 1, p["q"]))
    return {"sessions": plan_s, "probes": probes, "learn": learn}


class _Mem:
    def __init__(self, r: Retriever) -> None:
        self.r = r
        c = r.corpus
        self.sent = c.sentences()
        self.qv = c.vectors("questions")
        self.cv = c.vectors(DEFAULT)
        self.sv = c.vectors("sentences")
        self._stok: dict[int, int] = {}

    def sent_tokens(self, si: int) -> int:
        if si not in self._stok:
            a, s, e = self.sent[si]
            self._stok[si] = self.r.corpus.tok.count(self.r.corpus.articles[a]["text"][s:e])
        return self._stok[si]

    def chunk_sentences(self, c: int) -> list[int]:
        ch = self.r.chunks[c]
        return [i for i, (a, s, e) in enumerate(self.sent) if a == ch["article"] and s >= ch["start"] and e <= ch["end"]]


def memory_run(r: Retriever, policy: dict[str, Any], plan: dict[str, Any] | None = None,
               name: str = "") -> dict[str, Any]:
    plan = plan or memory_plan(r)
    kind = policy["kind"]
    k = policy.get("k", 0)
    cap = policy.get("cap", 0)
    m = _Mem(r)
    tok = r.corpus.tok
    qs = r.corpus.questions
    chunks = r.chunks
    store: list[dict[str, Any]] = []   # items: {id, ref, tokens, t (last access hour), imp, born}
    notes: list[int] = []
    frames: list[dict[str, Any]] = []
    st = {"read": 0, "write": 0, "probes": 0, "recalled": 0, "repeat": 0, "neighbour": 0,
          "repeat_n": 0, "neighbour_n": 0}

    def notes_text(ls: list[int]) -> str:
        return "Notes:\n" + "".join(f"- {qs[q]['question']} -> {qs[q]['answer']}\n" for q in ls)

    def items() -> list[dict[str, Any]]:
        if kind == "scratchpad":
            return [{"id": f"note {q}", "tokens": tok.count(f"- {qs[q]['question']} -> {qs[q]['answer']}\n"), "q": q}
                    for q in notes]
        return [{"id": it["id"], "tokens": it["tokens"]} for it in store]

    def frame(session: int, event: str, q: int | None, got: list[str], ok: bool | None, caption: str) -> None:
        frames.append({"step": len(frames), "session": session, "event": event, "q": q, "store": items(),
                       "got": got, "ok": ok, "read": st["read"], "write": st["write"], "probes": st["probes"],
                       "recalled": st["recalled"], "caption": caption})

    def stored_tokens() -> int:
        if kind == "scratchpad":
            return tok.count(notes_text(notes)) if notes else 0
        t = 0
        for it in store:
            t += it["tokens"]
        return t

    def forget(now: int) -> list[str]:
        gone = []
        while cap and len(store) > cap:
            rec = _norm([recency(now - it["t"]) for it in store])
            imp = _norm([float(it["imp"]) for it in store])
            sc = [rec[i] + imp[i] for i in range(len(store))]
            j = 0
            for i in range(1, len(store)):
                if sc[i] < sc[j]:
                    j = i
            gone.append(store.pop(j)["id"])
        return gone

    def write(s: int) -> None:
        now = (s + 1) * HOURS - 1
        if kind == "none":
            frame(s, "write", None, [], None, f"session {s + 1} ends: nothing is kept")
            return
        if kind == "transcript":
            for qi, c, _ in plan["sessions"][s]["reads"]:
                store.append({"id": f"read {len(store) + 1}", "ref": c, "tokens": chunks[c]["tokens"], "t": now, "imp": 0, "born": s})
            frame(s, "write", None, [], None, f"session {s + 1} ends: its {len(plan['sessions'][s]['reads'])} reads are kept "
                                              f"verbatim; the transcript is {stored_tokens()} tokens")
            return
        if kind == "scratchpad":
            added = 0
            for qi, c, has in plan["sessions"][s]["reads"]:
                if has and qi not in notes:
                    notes.append(qi)
                    added += 1
                    st["write"] += tok.count(f"- {qs[qi]['question']} -> {qs[qi]['answer']}\n")
            dropped = 0
            while cap and len(notes) > 1 and tok.count(notes_text(notes)) > cap:
                notes.pop(0)
                dropped += 1
            frame(s, "write", None, [], None, f"session {s + 1} ends: the agent writes {added} note line"
                  + ("" if added == 1 else "s") + f"; the file is {stored_tokens()} tokens"
                  + (f" ({dropped} oldest line" + ("" if dropped == 1 else "s") + f" forgotten to fit {cap})" if dropped else ""))
            return
        merged = 0
        added = 0
        for qi, c, has in plan["sessions"][s]["reads"]:
            ch = chunks[c]
            if kind == "episodic":
                st["write"] += ch["tokens"] + 1
                imp = IMPORTANCE["answer"] if has else IMPORTANCE["other"]
                new = [(f"chunk {c}", c, ch["tokens"], imp)]
            else:
                sents = m.chunk_sentences(c)
                facts = []
                if has:
                    for si in sents:
                        if _within(qs[qi], m.sent[si]):
                            facts.append((si, IMPORTANCE["answer"]))
                if sents and all(f[0] != sents[0] for f in facts):
                    facts.append((sents[0], IMPORTANCE["topic"]))
                out_t = 0
                for si, _ in facts:
                    out_t += m.sent_tokens(si)
                st["write"] += ch["tokens"] + out_t
                new = [(f"fact {si}", si, m.sent_tokens(si), imp) for si, imp in facts]
            for id_, ref, tk, imp in new:
                old = next((it for it in store if it["id"] == id_), None)
                if old is not None:
                    old["t"] = now
                    if imp > old["imp"]:
                        old["imp"] = imp
                    merged += 1
                else:
                    store.append({"id": id_, "ref": ref, "tokens": tk, "t": now, "imp": imp, "born": s})
                    added += 1
        gone = forget(now)
        what = "episode" if kind == "episodic" else "fact"
        frame(s, "write", None, [], None, f"session {s + 1} ends: {added} new {what}" + ("" if added == 1 else "s")
              + (f", {merged} merged into existing ones" if merged else "")
              + (f"; {len(gone)} forgotten to keep {cap}" if gone else "") + f"; memory holds {len(store)} ({stored_tokens()} tokens)")

    def probe(s: int, p: dict[str, Any]) -> None:
        now = s * HOURS
        q = qs[p["q"]]
        st["probes"] += 1
        st[p["kind"] + "_n"] += 1
        got: list[str] = []
        ok = False
        paid = 0
        if kind == "transcript":
            got = [it["id"] for it in store]
            paid = stored_tokens()
            ok = any(_within(q, [chunks[it["ref"]]["article"], chunks[it["ref"]]["start"], chunks[it["ref"]]["end"]]) for it in store)
        elif kind == "scratchpad":
            got = [f"note {n}" for n in notes]
            paid = stored_tokens()
            ok = p["q"] in notes
        elif kind in ("episodic", "semantic") and store:
            if kind == "episodic":
                rel = [cosine(m.qv[p["q"]], m.cv[it["ref"]]) for it in store]
                rec = _norm([recency(now - it["t"]) for it in store])
                imp = _norm([float(it["imp"]) for it in store])
                rn = _norm(rel)
                sc = [rec[i] + imp[i] + rn[i] for i in range(len(store))]
            else:
                sc = [cosine(m.qv[p["q"]], m.sv[it["ref"]]) for it in store]
            top = sorted(range(len(store)), key=lambda i: (-sc[i], i))[:k]
            for i in top:
                it = store[i]
                it["t"] = now
                got.append(it["id"])
                paid += it["tokens"]
                span = ([chunks[it["ref"]]["article"], chunks[it["ref"]]["start"], chunks[it["ref"]]["end"]]
                        if kind == "episodic" else m.sent[it["ref"]])
                if _within(q, span):
                    ok = True
        st["read"] += paid
        if ok:
            st["recalled"] += 1
            st[p["kind"]] += 1
        frame(s, "probe", p["q"], got, ok,
              f"session {s + 1}, probe about session {p['about'] + 1} ({p['kind']}): "
              + ("no memory to consult" if kind == "none" or (not got) else
                 f"{len(got)} item" + ("" if len(got) == 1 else "s") + f" ({paid} tokens) in the call")
              + (": recalled" if ok else ": not recalled"))

    frame(0, "start", None, [], None, f"{len(plan['sessions'])} sessions of {len(plan['sessions'][0]['learn'])} questions, "
                                      f"then probes about earlier sessions")
    for s in range(len(plan["sessions"]) + 1):
        for p in plan["probes"][s]:
            probe(s, p)
        if s < len(plan["sessions"]):
            for qi in plan["sessions"][s]["learn"]:
                reads = [x for x in plan["sessions"][s]["reads"] if x[0] == qi]
                found = any(x[2] for x in reads)
                frame(s, "learn", qi, [f"chunk {x[1]}" for x in reads], found,
                      f"session {s + 1}: researching question {plan['learn'].index(qi) + 1}, {len(reads)} read"
                      + ("" if len(reads) == 1 else "s") + (", answer found" if found else ", answer not found"))
            write(s)
    return {"name": name, "policy": policy, "probes": st["probes"], "recalled": st["recalled"],
            "repeat": [st["repeat"], st["repeat_n"]], "neighbour": [st["neighbour"], st["neighbour_n"]],
            "read_tokens": st["read"], "write_tokens": st["write"], "stored_tokens": stored_tokens(),
            "items": len(items()), "frames": frames}
