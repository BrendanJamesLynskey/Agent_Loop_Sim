/**
 * Agent memory across sessions (engine 1.5.0): none, transcript, scratchpad, episodic (Generative
 * Agents scoring) and semantic memory, with forgetting caps and consolidation. A port of
 * agent_loop_sim/context/memory.py, statement for statement.
 */
import { DEFAULT, type Obj } from "./corpus";
import { relevant } from "./evaluate";
import { rank, type Retriever } from "./retrieval";
import { cosine } from "./vectors";
import { MAX_READS, taskQuestions } from "./window";

export const SESSIONS = 6;
export const PER_SESSION = 3;
export const HOURS = 24;
export const DECAY = 0.995;
export const IMPORTANCE: Record<string, number> = { answer: 8, other: 3, topic: 5 };
export const POLICIES: Record<string, Obj> = {
  none: { kind: "none" },
  transcript: { kind: "transcript" },
  scratchpad: { kind: "scratchpad" },
  "scratchpad-cap": { kind: "scratchpad", cap: 128 },
  episodic: { kind: "episodic", k: 3 },
  "episodic-cap": { kind: "episodic", k: 3, cap: 8 },
  semantic: { kind: "semantic", k: 3 },
  "semantic-cap": { kind: "semantic", k: 3, cap: 10 },
};
export const SWEEP: [string, Obj][] = [
  ["episodic k=1", { kind: "episodic", k: 1 }],
  ["episodic k=2", { kind: "episodic", k: 2 }],
  ["episodic k=5", { kind: "episodic", k: 5 }],
  ["semantic k=1", { kind: "semantic", k: 1 }],
  ["semantic k=5", { kind: "semantic", k: 5 }],
  ["semantic k=8", { kind: "semantic", k: 8 }],
];

export function recency(hours: number): number {
  let x = 1.0;
  for (let i = 0; i < hours; i++) x *= DECAY;
  return x;
}

function norm(xs: number[]): number[] {
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  if (hi === lo) return xs.map(() => 0.0);
  return xs.map((x) => (x - lo) / (hi - lo));
}

function within(q: Obj, span: number[]): boolean {
  return span[0] === q.article && span[1]! <= q.start && q.end <= span[2]!;
}

export function memoryPlan(r: Retriever, sessions = SESSIONS, per = PER_SESSION): Obj {
  const qs = r.corpus.questions;
  const learn = taskQuestions(r, sessions * per);
  const planS: Obj[] = [];
  for (let s = 0; s < sessions; s++) {
    const reads: [number, number, boolean][] = [];
    for (const qi of learn.slice(s * per, (s + 1) * per)) {
      for (const c of rank(r.bm25Scores(qi)).slice(0, MAX_READS)) {
        const ch = r.chunks[c]!;
        const has = within(qs[qi]!, [ch.article, ch.start, ch.end]);
        reads.push([qi, c, has]);
        if (has) break;
      }
    }
    planS.push({ learn: learn.slice(s * per, (s + 1) * per), reads });
  }
  const used = new Set(learn);
  const probes: Obj[][] = [];
  for (let s = 0; s <= sessions; s++) probes.push([]);
  for (let t = 0; t < sessions; t++) {
    (planS[t]!.learn as number[]).forEach((qi, j) => {
      probes[Math.min(t + 1 + j, sessions)]!.push({ q: qi, kind: "repeat", about: t });
    });
    const read = new Set((planS[t]!.reads as [number, number, boolean][]).map((x) => x[1]));
    for (let qi = 0; qi < qs.length; qi++) {
      if (!used.has(qi) && relevant(r.chunks, qs[qi]!).some((c) => read.has(c))) {
        used.add(qi);
        probes[Math.min(t + 2, sessions)]!.push({ q: qi, kind: "neighbour", about: t });
        break;
      }
    }
  }
  for (const row of probes)
    row.sort((a, b) => a.about - b.about || (a.kind === "repeat" ? 0 : 1) - (b.kind === "repeat" ? 0 : 1) || a.q - b.q);
  return { sessions: planS, probes, learn };
}

interface Item {
  id: string;
  ref: number;
  tokens: number;
  t: number;
  imp: number;
  born: number;
}

export function memoryRun(r: Retriever, policy: Obj, plan?: Obj, name = ""): Obj {
  plan = plan ?? memoryPlan(r);
  const kind = policy.kind as string;
  const k = (policy.k as number | undefined) ?? 0;
  const cap = (policy.cap as number | undefined) ?? 0;
  const corpus = r.corpus;
  const sent = corpus.sentences();
  const qv = corpus.vectors("questions");
  const cv = corpus.vectors(DEFAULT);
  const sv = corpus.vectors("sentences");
  const stok = new Map<number, number>();
  const sentTokens = (si: number): number => {
    let v = stok.get(si);
    if (v === undefined) {
      const [a, s, e] = sent[si]!;
      v = corpus.tok.count((corpus.articles[a]!.text as string).slice(s, e));
      stok.set(si, v);
    }
    return v;
  };
  const chunkSentences = (c: number): number[] => {
    const ch = r.chunks[c]!;
    const out: number[] = [];
    sent.forEach(([a, s, e], i) => {
      if (a === ch.article && s >= ch.start && e <= ch.end) out.push(i);
    });
    return out;
  };
  const tok = corpus.tok;
  const qs = corpus.questions;
  const chunks = r.chunks;
  const store: Item[] = [];
  const notes: number[] = [];
  const frames: Obj[] = [];
  const st: Record<string, number> = {
    read: 0,
    write: 0,
    probes: 0,
    recalled: 0,
    repeat: 0,
    neighbour: 0,
    repeat_n: 0,
    neighbour_n: 0,
  };
  const noteLine = (q: number) => `- ${qs[q]!.question} -> ${qs[q]!.answer}\n`;
  const notesText = (ls: number[]) => "Notes:\n" + ls.map(noteLine).join("");

  const items = (): Obj[] => {
    if (kind === "scratchpad") return notes.map((q) => ({ id: `note ${q}`, tokens: tok.count(noteLine(q)), q }));
    return store.map((it) => ({ id: it.id, tokens: it.tokens }));
  };

  const frame = (session: number, event: string, q: number | null, got: string[], ok: boolean | null, caption: string) => {
    frames.push({
      step: frames.length,
      session,
      event,
      q,
      store: items(),
      got,
      ok,
      read: st.read,
      write: st.write,
      probes: st.probes,
      recalled: st.recalled,
      caption,
    });
  };

  const storedTokens = (): number => {
    if (kind === "scratchpad") return notes.length ? tok.count(notesText(notes)) : 0;
    let t = 0;
    for (const it of store) t += it.tokens;
    return t;
  };

  const forget = (now: number): string[] => {
    const gone: string[] = [];
    while (cap && store.length > cap) {
      const rec = norm(store.map((it) => recency(now - it.t)));
      const imp = norm(store.map((it) => it.imp));
      const sc = store.map((_, i) => rec[i]! + imp[i]!);
      let j = 0;
      for (let i = 1; i < store.length; i++) if (sc[i]! < sc[j]!) j = i;
      gone.push(store.splice(j, 1)[0]!.id);
    }
    return gone;
  };

  const sessionReads = (s: number) => plan!.sessions[s].reads as [number, number, boolean][];

  const write = (s: number) => {
    const now = (s + 1) * HOURS - 1;
    if (kind === "none") {
      frame(s, "write", null, [], null, `session ${s + 1} ends: nothing is kept`);
      return;
    }
    if (kind === "transcript") {
      for (const [, c] of sessionReads(s))
        store.push({ id: `read ${store.length + 1}`, ref: c, tokens: chunks[c]!.tokens, t: now, imp: 0, born: s });
      frame(
        s,
        "write",
        null,
        [],
        null,
        `session ${s + 1} ends: its ${sessionReads(s).length} reads are kept ` + `verbatim; the transcript is ${storedTokens()} tokens`,
      );
      return;
    }
    if (kind === "scratchpad") {
      let added = 0;
      for (const [qi, , has] of sessionReads(s)) {
        if (has && !notes.includes(qi)) {
          notes.push(qi);
          added += 1;
          st.write! += tok.count(noteLine(qi));
        }
      }
      let dropped = 0;
      while (cap && notes.length > 1 && tok.count(notesText(notes)) > cap) {
        notes.shift();
        dropped += 1;
      }
      frame(
        s,
        "write",
        null,
        [],
        null,
        `session ${s + 1} ends: the agent writes ${added} note line` +
          (added === 1 ? "" : "s") +
          `; the file is ${storedTokens()} tokens` +
          (dropped ? ` (${dropped} oldest line` + (dropped === 1 ? "" : "s") + ` forgotten to fit ${cap})` : ""),
      );
      return;
    }
    let merged = 0;
    let added = 0;
    for (const [qi, c, has] of sessionReads(s)) {
      const ch = chunks[c]!;
      let fresh: [string, number, number, number][];
      if (kind === "episodic") {
        st.write! += ch.tokens + 1;
        const imp = has ? IMPORTANCE.answer! : IMPORTANCE.other!;
        fresh = [[`chunk ${c}`, c, ch.tokens, imp]];
      } else {
        const sents = chunkSentences(c);
        const facts: [number, number][] = [];
        if (has) for (const si of sents) if (within(qs[qi]!, sent[si]!)) facts.push([si, IMPORTANCE.answer!]);
        if (sents.length && facts.every((f) => f[0] !== sents[0])) facts.push([sents[0]!, IMPORTANCE.topic!]);
        let outT = 0;
        for (const [si] of facts) outT += sentTokens(si);
        st.write! += ch.tokens + outT;
        fresh = facts.map(([si, imp]) => [`fact ${si}`, si, sentTokens(si), imp]);
      }
      for (const [id, ref, tk, imp] of fresh) {
        const old = store.find((it) => it.id === id);
        if (old !== undefined) {
          old.t = now;
          if (imp > old.imp) old.imp = imp;
          merged += 1;
        } else {
          store.push({ id, ref, tokens: tk, t: now, imp, born: s });
          added += 1;
        }
      }
    }
    const gone = forget(now);
    const what = kind === "episodic" ? "episode" : "fact";
    frame(
      s,
      "write",
      null,
      [],
      null,
      `session ${s + 1} ends: ${added} new ${what}` +
        (added === 1 ? "" : "s") +
        (merged ? `, ${merged} merged into existing ones` : "") +
        (gone.length ? `; ${gone.length} forgotten to keep ${cap}` : "") +
        `; memory holds ${store.length} (${storedTokens()} tokens)`,
    );
  };

  const chunkSpan = (c: number) => [chunks[c]!.article, chunks[c]!.start, chunks[c]!.end];

  const probe = (s: number, p: Obj) => {
    const now = s * HOURS;
    const q = qs[p.q]!;
    st.probes! += 1;
    st[p.kind + "_n"]! += 1;
    const got: string[] = [];
    let ok = false;
    let paid = 0;
    if (kind === "transcript") {
      for (const it of store) got.push(it.id);
      paid = storedTokens();
      ok = store.some((it) => within(q, chunkSpan(it.ref)));
    } else if (kind === "scratchpad") {
      for (const n of notes) got.push(`note ${n}`);
      paid = storedTokens();
      ok = notes.includes(p.q);
    } else if ((kind === "episodic" || kind === "semantic") && store.length) {
      let sc: number[];
      if (kind === "episodic") {
        const rel = store.map((it) => cosine(qv[p.q]!, cv[it.ref]!));
        const rec = norm(store.map((it) => recency(now - it.t)));
        const imp = norm(store.map((it) => it.imp));
        const rn = norm(rel);
        sc = store.map((_, i) => rec[i]! + imp[i]! + rn[i]!);
      } else {
        sc = store.map((it) => cosine(qv[p.q]!, sv[it.ref]!));
      }
      const top = store
        .map((_, i) => i)
        .sort((a, b) => sc[b]! - sc[a]! || a - b)
        .slice(0, k);
      for (const i of top) {
        const it = store[i]!;
        it.t = now;
        got.push(it.id);
        paid += it.tokens;
        const span = kind === "episodic" ? chunkSpan(it.ref) : sent[it.ref]!;
        if (within(q, span)) ok = true;
      }
    }
    st.read! += paid;
    if (ok) {
      st.recalled! += 1;
      st[p.kind]! += 1;
    }
    frame(
      s,
      "probe",
      p.q,
      got,
      ok,
      `session ${s + 1}, probe about session ${p.about + 1} (${p.kind}): ` +
        (kind === "none" || !got.length ? "no memory to consult" : `${got.length} item` + (got.length === 1 ? "" : "s") + ` (${paid} tokens) in the call`) +
        (ok ? ": recalled" : ": not recalled"),
    );
  };

  const nS = plan.sessions.length as number;
  frame(0, "start", null, [], null, `${nS} sessions of ${plan.sessions[0].learn.length} questions, ` + `then probes about earlier sessions`);
  for (let s = 0; s <= nS; s++) {
    for (const p of plan.probes[s] as Obj[]) probe(s, p);
    if (s < nS) {
      for (const qi of plan.sessions[s].learn as number[]) {
        const reads = sessionReads(s).filter((x) => x[0] === qi);
        const found = reads.some((x) => x[2]);
        frame(
          s,
          "learn",
          qi,
          reads.map((x) => `chunk ${x[1]}`),
          found,
          `session ${s + 1}: researching question ${(plan.learn as number[]).indexOf(qi) + 1}, ${reads.length} read` +
            (reads.length === 1 ? "" : "s") +
            (found ? ", answer found" : ", answer not found"),
        );
      }
      write(s);
    }
  }
  return {
    name,
    policy,
    probes: st.probes,
    recalled: st.recalled,
    repeat: [st.repeat, st.repeat_n],
    neighbour: [st.neighbour, st.neighbour_n],
    read_tokens: st.read,
    write_tokens: st.write,
    stored_tokens: storedTokens(),
    items: items().length,
    frames,
  };
}
