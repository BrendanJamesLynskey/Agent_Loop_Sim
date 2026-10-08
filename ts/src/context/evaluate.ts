/**
 * Relevance from answer spans, and recall@k, MRR@10, nDCG@10: a port of
 * agent_loop_sim/context/evaluate.py (means summed left to right, as there).
 */
import type { Chunk, Obj } from "./corpus";
import { ln } from "./mathx";
import type { Retriever } from "./retrieval";

export const KS = [1, 3, 5, 10, 20];
/** 1 / log2(i + 1) for ranks 1..10, as ln 2 / ln(i + 1) with the shared ln. */
export const DISCOUNT = Array.from({ length: 10 }, (_, i) => ln(2) / ln(i + 2));
export const METRICS = KS.map((k) => `recall@${k}`).concat(["mrr@10", "ndcg@10"]);

export function relevant(chunks: Chunk[], q: Obj): number[] {
  const out: number[] = [];
  chunks.forEach((c, i) => {
    if (c.article === q.article && c.start <= q.start && q.end <= c.end) out.push(i);
  });
  return out;
}

export function metrics(order: number[], rel: number[]): Record<string, number> {
  const out: Record<string, number> = {};
  const rs = new Set(rel);
  for (const k of KS) {
    let hit = 0;
    for (const c of order.slice(0, k)) if (rs.has(c)) hit += 1;
    out[`recall@${k}`] = rel.length ? hit / rel.length : 0;
  }
  let rr = 0;
  const top = order.slice(0, 10);
  for (let i = 0; i < top.length; i++) {
    if (rs.has(top[i]!)) {
      rr = 1 / (i + 1);
      break;
    }
  }
  out["mrr@10"] = rr;
  let dcg = 0;
  top.forEach((c, i) => {
    if (rs.has(c)) dcg += DISCOUNT[i]!;
  });
  let idcg = 0;
  for (let i = 0; i < Math.min(rel.length, 10); i++) idcg += DISCOUNT[i]!;
  out["ndcg@10"] = idcg > 0 ? dcg / idcg : 0;
  return out;
}

export function meanMetrics(rows: Record<string, number>[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of METRICS) {
    let s = 0;
    for (const r of rows) s += r[m]!;
    out[m] = rows.length ? s / rows.length : 0;
  }
  return out;
}

export function evaluate(retriever: Retriever, method: string, p: Obj | null = null): Obj {
  const rows: Record<string, number>[] = [];
  const first: number[] = [];
  let lost = 0;
  retriever.corpus.questions.forEach((q, qi) => {
    const rel = relevant(retriever.chunks, q);
    if (!rel.length) lost += 1;
    const order = retriever.ranking(qi, method, p ?? {});
    rows.push(metrics(order, rel));
    const rs = new Set(rel);
    let fr = 0;
    for (let i = 0; i < order.length; i++) {
      if (rs.has(order[i]!)) {
        fr = i + 1;
        break;
      }
    }
    first.push(fr);
  });
  return { config: retriever.config, method, params: p ?? {}, lost, mean: meanMetrics(rows), first_rank: first };
}
