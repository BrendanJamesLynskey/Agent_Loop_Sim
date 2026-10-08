/**
 * The window as working memory: a long research task under a token budget. A port of
 * agent_loop_sim/context/window.py, statement for statement.
 */
import type { Obj } from "./corpus";
import { relevant } from "./evaluate";
import { rank, type Retriever } from "./retrieval";

export const POLICIES = ["unbounded", "truncate", "compact", "retrieve", "compact+retrieve"];
export const MAX_READS = 3;
export const SYSTEM =
  "You are a research agent. Answer every question in the task. Use the search tool to read the " +
  "corpus; you can only answer from what is in your context.";

export function taskQuestions(r: Retriever, m: number): number[] {
  const byArticle = new Map<number, number[]>();
  r.corpus.questions.forEach((q, qi) => {
    const rel = relevant(r.chunks, q);
    const order = rank(r.bm25Scores(qi));
    if (rel.length && order.indexOf(rel[0]!) < MAX_READS) {
      if (!byArticle.has(q.article)) byArticle.set(q.article, []);
      byArticle.get(q.article)!.push(qi);
    }
  });
  const out: number[] = [];
  const arts = [...byArticle.keys()].sort((a, b) => a - b);
  let j = 0;
  while (out.length < m && arts.some((a) => byArticle.get(a)!.length)) {
    const a = arts[j % arts.length]!;
    const l = byArticle.get(a)!;
    if (l.length) out.push(l.shift()!);
    j += 1;
  }
  return out;
}

interface Item {
  id: string;
  kind: string;
  tokens: number;
  facts: number[];
  lines?: number[];
}

function used(items: Item[]): number {
  let t = 0;
  for (const it of items) t += it.tokens;
  return t;
}

function known(items: Item[]): number[] {
  const out: number[] = [];
  for (const it of items) for (const f of it.facts) if (!out.includes(f)) out.push(f);
  return out.sort((a, b) => a - b);
}

function note(r: Retriever, qi: number): string {
  const q = r.corpus.questions[qi]!;
  return `- ${q.question} -> ${q.answer}\n`;
}

export function windowRun(r: Retriever, questions: number[], budget: number, policy: string): Obj {
  if (!POLICIES.includes(policy)) throw new Error(`unknown policy '${policy}'`);
  const tok = r.corpus.tok;
  const Q = r.corpus.questions;
  const task = "Task: answer these questions.\n" + questions.map((qi, i) => `${i + 1}. ${Q[qi]!.question}\n`).join("");
  const items: Item[] = [
    { id: "system", kind: "system", tokens: tok.count(SYSTEM), facts: [] },
    { id: "task", kind: "task", tokens: tok.count(task), facts: [] },
  ];
  const frames: Obj[] = [];
  const st = { spent: 0, out: 0, calls: 0, reads: 0, compactions: 0, dropped: 0 };
  const learned: number[] = [];

  const frame = (event: string, qi: number | null, caption: string) => {
    frames.push({
      step: frames.length,
      event,
      q: qi,
      items: items.map((it) => ({ id: it.id, kind: it.kind, tokens: it.tokens, facts: [...it.facts] })),
      used: used(items),
      budget,
      known: known(items),
      learned: [...learned].sort((a, b) => a - b),
      spent: st.spent,
      calls: st.calls,
      caption,
    });
  };

  const call = (outText: string) => {
    st.spent += used(items);
    st.calls += 1;
    st.out += tok.count(outText);
  };

  const summaryText = (lines: number[]) => "Summary of earlier reads:\n" + lines.map((x) => note(r, x)).join("");

  const truncate = (qi: number | null) => {
    while (used(items) > budget) {
      const k = items.findIndex((it) => it.kind === "read" || it.kind === "summary");
      if (k < 0 || k === items.length - 1) break;
      const it = items[k]!;
      if (it.kind === "summary" && it.lines!.length > 1) {
        const f = it.lines!.shift()!;
        it.facts.splice(it.facts.indexOf(f), 1);
        it.tokens = tok.count(summaryText(it.lines!));
        frame("truncate", qi, `window over budget: the oldest summary line (question ${questions.indexOf(f) + 1}) is dropped`);
      } else {
        items.splice(k, 1);
        st.dropped += 1;
        const lost = it.facts.map((f) => String(questions.indexOf(f) + 1)).join(", ") || "none";
        frame(
          "truncate",
          qi,
          `window over budget (${used(items) + it.tokens} > ${budget}): the oldest read, ${it.id}, ` + `is dropped (facts lost: ${lost})`,
        );
      }
    }
  };

  const compact = (qi: number | null) => {
    const reads = items.filter((it) => it.kind === "read");
    if (reads.length < 2) {
      truncate(qi);
      return;
    }
    const old = reads.slice(0, -1);
    const prev = items.find((it) => it.kind === "summary");
    const lines = prev ? [...prev.lines!] : [];
    for (const it of old) for (const f of it.facts) if (!lines.includes(f)) lines.push(f);
    const text = summaryText(lines);
    call(text);
    st.compactions += 1;
    let oldTokens = 0;
    for (const it of old) oldTokens += it.tokens;
    const keep = items.filter((it) => it.kind === "system" || it.kind === "task");
    const summ: Item = { id: "summary", kind: "summary", tokens: tok.count(text), facts: [...lines], lines };
    items.splice(0, items.length, ...keep, summ, reads[reads.length - 1]!);
    frame("compact", qi, `compaction: ${old.length} reads (${oldTokens} tokens) ` + `become a ${summ.tokens}-token summary of ${lines.length} facts`);
    truncate(qi);
  };

  const read = (qi: number, c: number, phase: string): boolean => {
    const ch = r.chunks[c]!;
    const q = Q[qi]!;
    const has = q.start >= ch.start && q.end <= ch.end && ch.article === q.article;
    call(`search("${q.question}")`);
    st.reads += 1;
    items.push({ id: `chunk ${c}`, kind: "read", tokens: ch.tokens, facts: has ? [qi] : [] });
    if (has && !learned.includes(qi)) learned.push(qi);
    const n = questions.indexOf(qi) + 1;
    frame(phase, qi, `question ${n}: reads chunk ${c} (${ch.tokens} tokens) — ` + (has ? "it holds the answer" : "no answer in it"));
    if (used(items) > budget && policy !== "unbounded") {
      if (policy.startsWith("compact")) compact(qi);
      else truncate(qi);
    }
    return has;
  };

  const search = (qi: number, phase: string): boolean => {
    const order = rank(r.bm25Scores(qi));
    for (const c of order.slice(0, MAX_READS)) if (read(qi, c, phase)) return true;
    return false;
  };

  frame("start", null, `the window starts with the system prompt and the task: ${used(items)} of ${budget} tokens`);
  for (const qi of questions) search(qi, "read");
  const answered: number[] = [];
  for (const qi of questions) {
    if (!known(items).includes(qi) && policy.endsWith("retrieve")) {
      frame("recall", qi, `question ${questions.indexOf(qi) + 1}: its fact is no longer in the window, so the agent searches again`);
      search(qi, "reread");
    }
    if (known(items).includes(qi)) answered.push(qi);
  }
  call("Answers:\n" + answered.map((qi) => `${questions.indexOf(qi) + 1}. ${Q[qi]!.answer}\n`).join(""));
  frame("answer", null, `the agent answers ${answered.length} of ${questions.length} questions from its window`);
  let peak = 0;
  for (const f of frames) if (f.used > peak) peak = f.used;
  return {
    policy,
    budget,
    questions,
    recalled: answered.length,
    spent: st.spent,
    out: st.out,
    calls: st.calls,
    reads: st.reads,
    compactions: st.compactions,
    dropped: st.dropped,
    peak,
    frames,
  };
}
