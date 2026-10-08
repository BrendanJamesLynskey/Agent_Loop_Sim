/**
 * Views the Context site animates (chapters 2 to 5): a port of agent_loop_sim/context/views.py.
 */
import type { Obj } from "./corpus";
import { relevant } from "./evaluate";
import { B, K1, RRF_K, rank, rerank, rrf, type Retriever } from "./retrieval";
import { fixed, paragraphStarts, ranges, sentenceStarts } from "./text";
import { cosine } from "./vectors";

export function tfCurve(k1: number, b: number, ratio: number, tfMax = 10): [number, number][] {
  const out: [number, number][] = [];
  for (let tf = 0; tf <= tfMax; tf++) out.push([tf, (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * ratio))]);
  return out;
}

export function lengthCurve(k1: number, b: number, tf = 1): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 1; i < 17; i++) {
    const ratio = i / 4;
    out.push([ratio, (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * ratio))]);
  }
  return out;
}

export function bm25View(r: Retriever, query: string, k1 = K1, b = B, top = 8): Obj {
  const bm = r.bm25;
  const terms = bm.queryTerms(query);
  const idfs = terms.map((w) => bm.idf(w));
  const cum = new Array<number>(bm.n).fill(0);
  const frames: Obj[] = [];
  terms.forEach((w, j) => {
    for (let d = 0; d < bm.n; d++) {
      const tf = bm.docs[d]!.get(w) ?? 0;
      if (tf) cum[d] = cum[d]! + bm.termScore(tf, bm.lens[d]!, idfs[j]!, k1, b);
    }
    const order = rank(cum).slice(0, top);
    frames.push({
      term: w,
      idf: idfs[j],
      df: bm.df.get(w) ?? 0,
      top: order.map((c) => ({ chunk: c, score: cum[c] })),
      caption:
        `term ${j + 1} of ${terms.length}, “${w}”: in ${bm.df.get(w) ?? 0} of ${bm.n} chunks, ` +
        `idf ${fixed(idfs[j]!, 2)}; the leader is chunk ${order[0]} at ${fixed(cum[order[0]!]!, 2)}`,
    });
  });
  const order = rank(cum).slice(0, top);
  const rows = order.map((c) => ({
    chunk: c,
    score: cum[c],
    dl: bm.lens[c],
    parts: terms.map((w, j) => {
      const tf = bm.docs[c]!.get(w) ?? 0;
      return { term: w, tf, score: tf ? bm.termScore(tf, bm.lens[c]!, idfs[j]!, k1, b) : 0 };
    }),
  }));
  return {
    query,
    k1,
    b,
    n: bm.n,
    avgdl: bm.avgdl,
    terms: terms.map((w, j) => ({ term: w, df: bm.df.get(w) ?? 0, idf: idfs[j] })),
    top: rows,
    frames,
  };
}

export function denseFrames(r: Retriever, qis: number[], precision = "int8", k = 5): Obj[] {
  return qis.map((qi) => {
    const s = r.dense(qi, precision);
    const order = rank(s).slice(0, k);
    const rel = relevant(r.chunks, r.corpus.questions[qi]!);
    const hi = order.findIndex((c) => rel.includes(c));
    const hit = hi >= 0 ? hi + 1 : 0;
    return {
      q: qi,
      neighbours: order.map((c) => ({ chunk: c, sim: s[c], relevant: rel.includes(c) })),
      relevant: rel,
      caption:
        `question ${qi}: nearest chunk ${order[0]} at cosine ${fixed(s[order[0]!]!, 3)}; ` +
        (hit ? `the answer is at rank ${hit}` : `the answer is not in the top ${k}`),
    };
  });
}

export function hybridFrames(r: Retriever, qi: number, k = RRF_K, depth = 50, show = 10, n = 20): Obj {
  const rb = rank(r.bm25Scores(qi));
  const rd = rank(r.dense(qi));
  const rel = relevant(r.chunks, r.corpus.questions[qi]!);
  const s = new Array<number>(r.chunks.length).fill(0);
  const frames: Obj[] = [];
  for (let pos = 0; pos < show; pos++) {
    for (const lst of [rb, rd]) if (pos < depth) s[lst[pos]!] = s[lst[pos]!]! + 1 / (k + pos + 1);
    const top = rank(s)
      .slice(0, show)
      .filter((c) => s[c]! > 0);
    frames.push({
      stage: "fuse",
      rank: pos + 1,
      adds: [rb[pos], rd[pos]],
      top: top.map((c) => ({ chunk: c, score: s[c] })),
      caption:
        `rank ${pos + 1}: BM25 adds 1/(${k}+${pos + 1}) to chunk ${rb[pos]}, dense adds it to chunk ${rd[pos]}` +
        (rb[pos] === rd[pos] ? " (the same chunk: it doubles)" : ""),
    });
  }
  const full = rrf([rb, rd], r.chunks.length, k, depth);
  const fused = rank(full);
  frames.push({
    stage: "fused",
    rank: show,
    adds: [],
    top: fused.slice(0, show).map((c) => ({ chunk: c, score: full[c] })),
    caption: `all ${depth} ranks of both lists fused: the top ${show} by RRF score`,
  });
  const ce = r.corpus.rerankScores()[qi]!;
  const rr = rerank(fused, ce, n);
  frames.push({
    stage: "rerank",
    rank: show,
    adds: [],
    top: rr.slice(0, show).map((c) => ({ chunk: c, score: ce.get(c)! / 1000 })),
    caption: `the cross-encoder reads the question with each of the top ${n} and reorders them`,
  });
  const first = (order: number[]) => {
    const i = order.findIndex((c) => rel.includes(c));
    return i >= 0 ? i + 1 : 0;
  };
  return {
    q: qi,
    bm25: rb.slice(0, show),
    dense: rd.slice(0, show),
    fused: fused.slice(0, show),
    reranked: rr.slice(0, show),
    relevant: rel,
    first: { bm25: first(rb), dense: first(rd), rrf: first(fused), rerank: first(rr) },
    frames,
  };
}

export function chunkView(rByConfig: Record<string, Retriever>, article: number, start: number, end: number): Obj {
  const anyR = Object.values(rByConfig)[0]!;
  const corpus = anyR.corpus;
  const text = corpus.articles[article]!.text as string;
  const p = corpus.pieces[article]!;
  const sents = ranges(sentenceStarts(text, p), p.length);
  const paras = paragraphStarts(text, p);
  let s0 = 0;
  for (let a = 0; a < article; a++) s0 += sentenceStarts(corpus.articles[a]!.text as string, corpus.pieces[a]!).length;
  const sv = corpus.vectors("sentences").slice(s0, s0 + sents.length);
  const sentences: Obj[] = [];
  sents.forEach(([sa, sb], i) => {
    const cs = p[sa]![0];
    const ce = p[sb - 1]![1];
    if (ce <= start || cs >= end) return;
    const sim = i + 1 < sents.length ? cosine(sv[i]!, sv[i + 1]!) : null;
    sentences.push({ i, start: cs, end: ce, paragraph: paras.includes(sa), sim_next: sim });
  });
  const configs: Record<string, Obj[]> = {};
  for (const [name, r] of Object.entries(rByConfig)) {
    configs[name] = [];
    r.chunks.forEach((c, ci) => {
      if (c.article === article && c.end > start && c.start < end)
        configs[name]!.push({ chunk: ci, start: c.start, end: c.end, tokens: c.tokens });
    });
  }
  const qs: Obj[] = [];
  corpus.questions.forEach((q, qi) => {
    if (q.article === article && q.start >= start && q.end <= end) qs.push({ q: qi, start: q.start, end: q.end });
  });
  return { article, start, end, sentences, configs, answers: qs };
}
