/**
 * BM25, dense retrieval, RRF and weighted fusion, and the recorded reranker: a port of
 * agent_loop_sim/context/retrieval.py with the same floating-point operation order.
 */
import type { Corpus, Chunk, Obj } from "./corpus";
import { ln } from "./mathx";
import { analyse } from "./text";
import { prepare, similarity } from "./vectors";

export const K1 = 1.2;
export const B = 0.75;
export const RRF_K = 60;

export class BM25 {
  stop: boolean;
  docs: Map<string, number>[] = [];
  lens: number[] = [];
  df = new Map<string, number>();
  n: number;
  avgdl: number;

  constructor(texts: string[], stop = true) {
    this.stop = stop;
    let total = 0;
    for (const t of texts) {
      const terms = analyse(t, stop);
      const tf = new Map<string, number>();
      for (const w of terms) tf.set(w, (tf.get(w) ?? 0) + 1);
      for (const w of tf.keys()) this.df.set(w, (this.df.get(w) ?? 0) + 1);
      this.docs.push(tf);
      this.lens.push(terms.length);
      total += terms.length;
    }
    this.n = texts.length;
    this.avgdl = this.n ? total / this.n : 0;
  }

  idf(term: string): number {
    const df = this.df.get(term) ?? 0;
    return ln(1 + (this.n - df + 0.5) / (df + 0.5));
  }

  queryTerms(query: string): string[] {
    const out: string[] = [];
    for (const w of analyse(query, this.stop)) if (!out.includes(w)) out.push(w);
    return out;
  }

  termScore(tf: number, dl: number, idf: number, k1: number, b: number): number {
    return (idf * (tf * (k1 + 1))) / (tf + k1 * (1 - b + (b * dl) / this.avgdl));
  }

  scores(query: string, k1 = K1, b = B): number[] {
    const terms = this.queryTerms(query);
    const idfs = terms.map((w) => this.idf(w));
    const out: number[] = [];
    for (let d = 0; d < this.n; d++) {
      let s = 0;
      const doc = this.docs[d]!;
      terms.forEach((w, j) => {
        const tf = doc.get(w) ?? 0;
        if (tf) s += this.termScore(tf, this.lens[d]!, idfs[j]!, k1, b);
      });
      out.push(s);
    }
    return out;
  }
}

export function rank(scores: number[]): number[] {
  const idx = scores.map((_, i) => i);
  idx.sort((a, b) => (scores[b]! > scores[a]! ? 1 : scores[b]! < scores[a]! ? -1 : a - b));
  return idx;
}

export function denseScores(q: ArrayLike<number>, chunkVecs: ArrayLike<number>[], precision: string): number[] {
  return chunkVecs.map((v) => similarity(q, v, precision));
}

export function rrf(rankings: number[][], n: number, k = RRF_K, depth = 50): number[] {
  const s = new Array<number>(n).fill(0);
  for (const r of rankings) {
    for (let pos = 0; pos < Math.min(depth, r.length); pos++) s[r[pos]!] = s[r[pos]!]! + 1 / (k + pos + 1);
  }
  return s;
}

export function minmaxTop(scores: number[], order: number[], depth: number): number[] {
  const top = order.slice(0, depth);
  const out = new Array<number>(scores.length).fill(0);
  if (!top.length) return out;
  const hi = scores[top[0]!]!;
  const lo = scores[top[top.length - 1]!]!;
  for (const i of top) out[i] = hi === lo ? 1 : (scores[i]! - lo) / (hi - lo);
  return out;
}

export function weighted(bm25: number[], dense: number[], alpha: number, depth = 50): number[] {
  const nb = minmaxTop(bm25, rank(bm25), depth);
  const nd = minmaxTop(dense, rank(dense), depth);
  return bm25.map((_, i) => alpha * nd[i]! + (1 - alpha) * nb[i]!);
}

export function rerank(order: number[], ce: Map<number, number>, n: number): number[] {
  const top = order.slice(0, n);
  for (const c of top) if (!ce.has(c)) throw new Error(`no reranker score recorded for chunk ${c} (outside the recorded pool)`);
  const pos = new Map(top.map((c, i) => [c, i]));
  const sorted = [...top].sort((a, b) => ce.get(b)! - ce.get(a)! || pos.get(a)! - pos.get(b)!);
  return sorted.concat(order.slice(n));
}

export class Retriever {
  corpus: Corpus;
  config: string;
  chunks: Chunk[];
  bm25: BM25;
  private _vecs = new Map<string, Int8Array[]>();
  private _qvecs = new Map<string, Int8Array[]>();
  private _cache = new Map<string, number[]>();

  constructor(corpus: Corpus, config: string, stop = true) {
    this.corpus = corpus;
    this.config = config;
    this.chunks = corpus.chunks(config);
    this.bm25 = new BM25(
      this.chunks.map((c) => corpus.chunkText(c)),
      stop,
    );
  }

  vecs(precision: string): Int8Array[] {
    if (!this._vecs.has(precision)) {
      this._vecs.set(precision, prepare(this.corpus.vectors(this.config), precision));
      this._qvecs.set(precision, prepare(this.corpus.vectors("questions"), precision));
    }
    return this._vecs.get(precision)!;
  }

  bm25Scores(qi: number, k1 = K1, b = B): number[] {
    const key = `bm25|${qi}|${k1}|${b}`;
    let s = this._cache.get(key);
    if (!s) {
      s = this.bm25.scores(this.corpus.questions[qi]!.question as string, k1, b);
      this._cache.set(key, s);
    }
    return s;
  }

  dense(qi: number, precision = "int8"): number[] {
    const key = `dense|${qi}|${precision}`;
    let s = this._cache.get(key);
    if (!s) {
      const v = this.vecs(precision);
      s = denseScores(this._qvecs.get(precision)![qi]!, v, precision);
      this._cache.set(key, s);
    }
    return s;
  }

  ranking(qi: number, method: string, p: Obj = {}): number[] {
    const k1 = (p.k1 as number) ?? K1;
    const b = (p.b as number) ?? B;
    const precision = (p.precision as string) ?? "int8";
    const depth = (p.depth as number) ?? 50;
    if (method === "bm25") return rank(this.bm25Scores(qi, k1, b));
    if (method === "dense") return rank(this.dense(qi, precision));
    const rb = rank(this.bm25Scores(qi, k1, b));
    if (method === "weighted")
      return rank(weighted(this.bm25Scores(qi, k1, b), this.dense(qi, precision), (p.alpha as number) ?? 0.5, depth));
    const rd = rank(this.dense(qi, precision));
    const fused = rank(rrf([rb, rd], this.chunks.length, (p.k as number) ?? RRF_K, depth));
    if (method === "rrf") return fused;
    if (method === "rerank") {
      const rc = this.corpus.files["rerank.json"]!.config as string;
      if (this.config !== rc) throw new Error(`reranker scores were recorded for ${rc} only`);
      const base = ({ rrf: fused, bm25: rb, dense: rd } as Record<string, number[]>)[(p.base as string) ?? "rrf"]!;
      return rerank(base, this.corpus.rerankScores()[qi]!, (p.n as number) ?? 20);
    }
    throw new Error(`unknown method '${method}'`);
  }
}
