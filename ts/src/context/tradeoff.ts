/**
 * Long context or retrieval? (engine 1.5.0): a run of questions with the whole document set in every
 * prompt (with and without the prompt cache) against retrieving k chunks per question. A port of
 * agent_loop_sim/context/tradeoff.py, statement for statement.
 */
import { callCost, callLatency } from "../accounting";
import { LATENCY, PRICES } from "../data";
import type { Obj } from "./corpus";
import { evaluate } from "./evaluate";
import type { Retriever } from "./retrieval";
import { SYSTEM } from "./window";

export const STRATEGIES = ["long", "long+cache", "rag"];
export const GAP_MS = 60000;
export const OUT_TOKENS = 50;
export const MODELS = ["claude-sonnet-4.6", "claude-haiku-4.5", "gpt-5-mini"];

export function measuredSizes(r: Retriever): Obj {
  const tok = r.corpus.tok;
  let qt = 0;
  for (const q of r.corpus.questions) qt += tok.count(q.question as string);
  let ct = 0;
  for (const c of r.chunks) ct += c.tokens;
  return { system: tok.count(SYSTEM), question: qt / r.corpus.questions.length, chunk: ct / r.chunks.length, corpus: ct, chunks: r.chunks.length };
}

export function runQuestions(sizes: Obj, model: string, nCtx: number, k: number, questions: number, latency = "hosted"): Obj {
  const price = PRICES[model]!;
  const prof = LATENCY[latency]!;
  const q = Math.floor(sizes.question + 0.5);
  const sysn = sizes.system as number;
  const out: Obj = { model, n_ctx: nCtx, k, questions, q_tokens: q, out_tokens: OUT_TOKENS, strategies: {} };
  for (const s of STRATEGIES) {
    const rows: Record<string, number[]> = { input: [], cached: [], cost: [], cum: [], ttft: [] };
    let cum = 0.0;
    let alive = -1.0;
    for (let i = 0; i < questions; i++) {
      const now = i * GAP_MS;
      let inp: number;
      let cached = 0;
      let stored = false;
      if (s === "rag") {
        inp = sysn + k * Math.floor(sizes.chunk + 0.5) + q;
      } else {
        const prefix = sysn + nCtx;
        inp = prefix + q;
        if (s === "long+cache") {
          if (alive >= 0 && now - alive <= price.ttl_ms) {
            const c = Math.floor(prefix / price.block) * price.block;
            cached = c >= price.min_tokens ? c : 0;
          }
          stored = inp >= price.min_tokens;
          if (stored || cached) alive = now;
        }
      }
      const cost = callCost(price, inp, cached, stored, OUT_TOKENS);
      cum += cost;
      const [ttft] = callLatency(prof, inp, cached, OUT_TOKENS);
      rows.input!.push(inp);
      rows.cached!.push(cached);
      rows.cost!.push(cost);
      rows.cum!.push(cum);
      rows.ttft!.push(ttft);
    }
    out.strategies[s] = rows;
  }
  return out;
}

export function tradeoff(r: Retriever, nCtxs: number[], ks: number[], questions: number, models: string[] = MODELS): Obj {
  const sizes = measuredSizes(r);
  const ev = evaluate(r, "rerank", { n: 20 }).mean as Obj;
  const runs: Obj = {};
  for (const m of models) for (const n of nCtxs) for (const k of ks) runs[`${m}|${n}|${k}`] = runQuestions(sizes, m, n, k, questions);
  const prices: Obj = {};
  for (const m of models) {
    const p: Obj = {};
    for (const key of ["label", "input", "output", "cache_read", "cache_write", "min_tokens", "block", "ttl_ms", "source", "accessed"])
      p[key] = PRICES[m]![key];
    prices[m] = p;
  }
  return {
    sizes,
    recall: Object.fromEntries(ks.map((k) => [String(k), ev[`recall@${k}`]])),
    runs,
    prices,
    latency: LATENCY.hosted,
  };
}
