/**
 * Packing the window (engine 1.5.0): the 0/1 knapsack over reranked chunks (top, density, optimal
 * DP) and the illustrative "lost in the middle" position curve with four placements. A port of
 * agent_loop_sim/context/packing.py, statement for statement.
 */
import type { Obj } from "./corpus";
import { relevant } from "./evaluate";
import { ln } from "./mathx";
import type { Retriever } from "./retrieval";
import { fixed } from "./text";

export const PACKERS = ["top", "density", "optimal"];
export const PLACEMENTS = ["best-first", "best-last", "ends", "middle"];
export const CANDIDATES = 20;
export const POSITION: Record<string, number> = { start: 0.75, middle: 0.55, end: 0.65, trough: 0.5 };

export interface Cand {
  chunk: number;
  rank: number;
  tokens: number;
  value: number;
  relevant: boolean;
}

export function gain(i: number): number {
  return ln(2.0) / ln(i + 1);
}

export function positionP(x: number, pc: Record<string, number> = POSITION): number {
  const x0 = pc.trough!;
  if (x <= x0) {
    const u = (x0 - x) / x0;
    return pc.middle! + (pc.start! - pc.middle!) * (u * u);
  }
  const u = (x - x0) / (1 - x0);
  return pc.middle! + (pc.end! - pc.middle!) * (u * u);
}

export function positionCurve(pc: Record<string, number> = POSITION, points = 21): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < points; i++) out.push([i / (points - 1), positionP(i / (points - 1), pc)]);
  return out;
}

export function candidates(r: Retriever, qi: number, n = CANDIDATES): Cand[] {
  const order = r.ranking(qi, "rerank", { n }).slice(0, n);
  const rel = new Set(relevant(r.chunks, r.corpus.questions[qi]!));
  return order.map((c, i) => ({ chunk: c, rank: i + 1, tokens: r.chunks[c]!.tokens, value: gain(i + 1), relevant: rel.has(c) }));
}

export function greedy(cands: Cand[], budget: number, packer: string): Obj {
  const idx = cands.map((_, i) => i);
  if (packer === "density") {
    const key = (i: number) => -(cands[i]!.value / cands[i]!.tokens);
    idx.sort((a, b) => key(a) - key(b) || a - b);
  }
  let left = budget;
  const chosen: number[] = [];
  const steps: Obj[] = [];
  for (const i of idx) {
    const c = cands[i]!;
    const take = c.tokens <= left;
    if (take) {
      left -= c.tokens;
      chosen.push(i);
    }
    steps.push({ i, take, left });
  }
  return { chosen: chosen.sort((a, b) => a - b), steps };
}

export function knapsackTable(cands: Cand[], budget: number): [Float64Array[], Uint8Array[]] {
  const best: Float64Array[] = [new Float64Array(budget + 1)];
  const take: Uint8Array[] = [new Uint8Array(budget + 1)];
  for (let i = 1; i <= cands.length; i++) {
    const w = cands[i - 1]!.tokens;
    const v = cands[i - 1]!.value;
    const prev = best[i - 1]!;
    const row = Float64Array.from(prev);
    const tk = new Uint8Array(budget + 1);
    for (let c = w; c <= budget; c++) {
      const t = prev[c - w]! + v;
      if (t > row[c]!) {
        row[c] = t;
        tk[c] = 1;
      }
    }
    best.push(row);
    take.push(tk);
  }
  return [best, take];
}

export function knapsackPick(cands: Cand[], take: Uint8Array[], budget: number): number[] {
  let c = budget;
  const out: number[] = [];
  for (let i = cands.length; i > 0; i--) {
    if (take[i]![c]) {
      out.push(i - 1);
      c -= cands[i - 1]!.tokens;
    }
  }
  return out.sort((a, b) => a - b);
}

function sumOf(cands: Cand[], chosen: number[], key: "tokens" | "value"): number {
  let s = 0;
  for (const i of chosen) s += cands[i]![key];
  return s;
}

export function place(cands: Cand[], chosen: number[], placement: string): number[] {
  const s = [...chosen].sort((a, b) => a - b);
  if (placement === "best-first") return s;
  if (placement === "best-last") return s.reverse();
  const src = placement === "ends" ? s : s.reverse();
  const front: number[] = [];
  const back: number[] = [];
  src.forEach((i, j) => (j % 2 === 0 ? front : back).push(i));
  return front.concat(back.reverse());
}

export function positions(cands: Cand[], order: number[]): number[] {
  const total = sumOf(cands, order, "tokens");
  const out: number[] = [];
  let off = 0;
  for (const i of order) {
    const t = cands[i]!.tokens;
    out.push((off + t / 2) / total);
    off += t;
  }
  return out;
}

export function useP(cands: Cand[], order: number[], pc: Record<string, number> = POSITION): number {
  let best = 0.0;
  const xs = positions(cands, order);
  order.forEach((i, j) => {
    if (cands[i]!.relevant) {
      const p = positionP(xs[j]!, pc);
      if (p > best) best = p;
    }
  });
  return best;
}

export function packAll(cands: Cand[], budgets: number[]): Record<number, Obj> {
  const [best, take] = knapsackTable(cands, Math.max(...budgets));
  const out: Record<number, Obj> = {};
  for (const b of budgets) {
    const bestAt: number[] = [];
    for (let i = 0; i <= cands.length; i++) bestAt.push(best[i]![b]!);
    out[b] = {
      top: greedy(cands, b, "top"),
      density: greedy(cands, b, "density"),
      optimal: { chosen: knapsackPick(cands, take, b), best: bestAt },
    };
  }
  return out;
}

export function packingEval(r: Retriever, budgets: number[], n = CANDIDATES, pc: Record<string, number> = POSITION): Obj {
  const qs = r.corpus.questions;
  const acc: Record<string, Record<string, Record<string, number>>> = {};
  for (const b of budgets) {
    acc[String(b)] = {};
    for (const p of PACKERS) {
      const a: Record<string, number> = { answered: 0.0, tokens: 0.0, value: 0.0 };
      for (const pl of PLACEMENTS) a[pl] = 0.0;
      acc[String(b)]![p] = a;
    }
  }
  for (let qi = 0; qi < qs.length; qi++) {
    const cands = candidates(r, qi, n);
    const packs = packAll(cands, budgets);
    for (const b of budgets) {
      for (const p of PACKERS) {
        const ch = packs[b]![p].chosen as number[];
        const a = acc[String(b)]![p]!;
        a.answered! += ch.some((i) => cands[i]!.relevant) ? 1 : 0;
        a.tokens! += sumOf(cands, ch, "tokens");
        a.value! += sumOf(cands, ch, "value");
        for (const pl of PLACEMENTS) a[pl]! += ch.length ? useP(cands, place(cands, ch, pl), pc) : 0.0;
      }
    }
  }
  const nq = qs.length;
  for (const b of budgets)
    for (const p of PACKERS) {
      const a = acc[String(b)]![p]!;
      for (const k of Object.keys(a)) a[k] = a[k]! / nq;
    }
  return { budgets, candidates: n, position: pc, results: acc };
}

export function packingView(r: Retriever, qi: number, budget: number, n = CANDIDATES, pc: Record<string, number> = POSITION): Obj {
  const cands = candidates(r, qi, n);
  const packs = packAll(cands, [budget])[budget]!;
  const out: Obj = { q: qi, budget, candidates: cands, packers: {} };
  for (const p of ["top", "density"]) {
    const g = packs[p];
    const steps = (g.steps as Obj[]).map((s) => {
      const c = cands[s.i]!;
      return {
        ...s,
        caption:
          `${p}: chunk ${c.chunk} (rank ${c.rank}, ${c.tokens} tokens, value ${fixed(c.value, 2)}) ` +
          (s.take ? `fits: taken, ${s.left} tokens left` : `does not fit in the ${s.left} tokens left: skipped`),
      };
    });
    const ch = g.chosen as number[];
    out.packers[p] = {
      chosen: ch,
      steps,
      tokens: sumOf(cands, ch, "tokens"),
      value: sumOf(cands, ch, "value"),
      answered: ch.some((i) => cands[i]!.relevant),
    };
  }
  const opt = packs.optimal;
  const ch = opt.chosen as number[];
  const steps: Obj[] = [];
  for (let i = 1; i <= cands.length; i++) {
    const c = cands[i - 1]!;
    const inset = ch.includes(i - 1);
    steps.push({
      i: i - 1,
      take: inset,
      best: opt.best[i],
      caption:
        `optimal: with the first ${i} candidates the best value within ${budget} tokens is ` +
        `${fixed(opt.best[i], 2)}; chunk ${c.chunk} is ` +
        (inset ? "in" : "not in") +
        " the final set",
    });
  }
  out.packers.optimal = {
    chosen: ch,
    steps,
    tokens: sumOf(cands, ch, "tokens"),
    value: sumOf(cands, ch, "value"),
    answered: ch.some((i) => cands[i]!.relevant),
  };
  out.placements = {};
  for (const pl of PLACEMENTS) {
    const order = place(cands, ch, pl);
    out.placements[pl] = {
      order,
      positions: order.length ? positions(cands, order) : [],
      p: order.length ? useP(cands, order, pc) : 0.0,
    };
  }
  out.curve = positionCurve(pc);
  return out;
}
