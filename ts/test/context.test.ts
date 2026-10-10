/**
 * Engine 1.4.0's context module against the Python reference (fixtures/context_fixtures.json),
 * recomputed here from the shipped int8 vectors and recorded reranker scores with no tolerance:
 * pieces, sentence and paragraph boundaries, every chunk of every config, BM25 scores, cosines,
 * every evaluation (each question's first relevant rank and the mean metrics), every window run
 * and every view frame.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { Tokenizer, context as C, type Obj } from "../src/index";

const ROOT = join(__dirname, "..", "..");
const DATA = join(ROOT, "src/agent_loop_sim/data/context");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/context_fixtures.json"), "utf8")) as Obj;
const tok = new Tokenizer(readFileSync(join(ROOT, "src/agent_loop_sim/data/qwen2.5-merges.txt"), "utf8"));
const raw: Record<string, string> = {};
const files: Record<string, Obj> = {};
for (const n of C.FILES) {
  raw[n] = readFileSync(join(DATA, n), "utf8");
  files[n] = JSON.parse(raw[n]!) as Obj;
}
const corpus = new C.Corpus(files["corpus.json"]!, files, tok);
const rs: Record<string, C.Retriever> = {};
for (const name of Object.keys(C.CHUNKERS)) rs[name] = new C.Retriever(corpus, name);
const d = rs[C.DEFAULT]!;
const sha = (s: string) => createHash("sha256").update(s, "utf8").digest("hex");

describe("engine version and data", () => {
  it("matches the fixtures", () => expect(fx.engine).toBe("1.7.0"));
  it("reads the same data files", () => {
    for (const n of C.FILES) expect(sha(raw[n]!)).toBe(fx.files[n]);
    for (const n of C.FILES) expect(fx.manifest_files[n]).toBe(fx.files[n]);
  });
});

describe("text", () => {
  it("analyses like Python", () => {
    for (const a of fx.analyse as Obj[]) {
      expect(C.analyse(a.text)).toEqual(a.stop);
      expect(C.analyse(a.text, false)).toEqual(a.all);
    }
  });
  it("formats like Python's fixed", () => {
    for (const f of fx.fixed as Obj[]) expect(C.fixed(f.x, f.d)).toBe(f.s);
  });
  it("cuts the same pieces, paragraphs and sentences", () => {
    corpus.pieces.forEach((p, i) => {
      expect(sha(p.map(([a, b, n]) => `${a}:${b}:${n}`).join("\n"))).toBe(fx.pieces_sha256[i]);
      const t = corpus.articles[i]!.text as string;
      expect(C.paragraphStarts(t, p)).toEqual(fx.paragraph_starts[i]);
      expect(C.sentenceStarts(t, p)).toEqual(fx.sentence_starts[i]);
    });
  });
});

describe("chunking", () => {
  for (const name of Object.keys(C.CHUNKERS)) {
    it(`${name}: the same chunks, and the ones that were embedded`, () => {
      expect(corpus.chunks(name).map((c) => [c.article, c.start, c.end, c.tokens])).toEqual(fx.chunks[name]);
      expect(corpus.spansSha256(name)).toBe(fx.spans_sha256[name]);
      expect(corpus.spansSha256(name)).toBe(files[`emb-${name}.json`]!.spans_sha256);
      expect(rs[name]!.chunks.length).toBe(corpus.vectors(name).length);
    });
  }
});

describe("vectors", () => {
  const qv = corpus.vectors("questions");
  const dv = corpus.vectors(C.DEFAULT);
  it("int4, bits and similarities", () => {
    expect(qv.slice(0, 3).map((v) => Array.from(C.toInt4(v)))).toEqual(fx.vectors.int4);
    expect(qv.slice(0, 3).map((v) => Array.from(C.toBits(v)))).toEqual(fx.vectors.bits);
    const cos: number[] = [];
    const bin: number[] = [];
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 5; j++) {
        cos.push(C.cosine(qv[i]!, dv[j]!));
        bin.push(C.binarySim(C.toBits(qv[i]!), C.toBits(dv[j]!)));
      }
    expect(cos).toEqual(fx.vectors.cosine);
    expect(bin).toEqual(fx.vectors.binary);
    for (const p of C.PRECISIONS) expect(C.nbytes(384, p)).toBe(fx.vectors.nbytes[p]);
  });
  it("nDCG discounts", () => expect(C.DISCOUNT).toEqual(fx.discount));
  it("the shared ln (fdlibm) is bit-identical", () => {
    for (const [x, y] of fx.ln as [number, number][]) expect(C.ln(x)).toBe(y);
  });
});

describe("retrieval", () => {
  for (const name of Object.keys(C.CHUNKERS)) {
    it(`${name}: relevance and BM25`, () => {
      const r = rs[name]!;
      expect(corpus.questions.map((q) => C.relevant(r.chunks, q))).toEqual(fx.relevant[name]);
      const b = fx.bm25[name];
      expect([r.bm25.n, r.bm25.avgdl, r.bm25.df.size]).toEqual([b.n, b.avgdl, b.terms]);
      for (let qi = 0; qi < 5; qi++) {
        const s = r.bm25Scores(qi);
        expect(C.rank(s).slice(0, 10).map((i) => [i, s[i]])).toEqual(b.top[qi]);
      }
    });
  }
  it("dense top 10", () => {
    for (let qi = 0; qi < 5; qi++) {
      const s = d.dense(qi);
      expect(C.rank(s).slice(0, 10).map((i) => [i, s[i]])).toEqual(fx.dense_top[qi]);
    }
  });
});

describe("evaluation: every config and method, every question", () => {
  for (const e of fx.evals as Obj[]) {
    it(`${e.config} ${e.method} ${JSON.stringify(e.params)}`, () => {
      expect(C.evaluate(rs[e.config]!, e.method, e.params)).toEqual(e);
    });
  }
});

describe("the window", () => {
  it("picks the same task", () => expect(C.taskQuestions(d, 12)).toEqual(fx.task));
  for (const w of fx.windows as Obj[]) {
    it(`${w.policy} at ${w.budget}`, () => expect(C.windowRun(d, fx.task, w.budget, w.policy)).toEqual(w));
  }
});

describe("views", () => {
  const v = fx.views;
  it("BM25 curves", () => {
    const tc: [number, number, number][] = [[1.2, 0.75, 1.0], [0.5, 0.75, 2.0], [2.0, 0.0, 0.5]];
    expect(tc.map(([k1, b, r]) => C.tfCurve(k1, b, r))).toEqual(v.tf_curve);
    const lc: [number, number, number][] = [[1.2, 0.75, 1], [1.2, 1.0, 3], [2.0, 0.3, 1]];
    expect(lc.map(([k1, b, t]) => C.lengthCurve(k1, b, t))).toEqual(v.length_curve);
  });
  it("BM25 views", () => {
    for (const want of v.bm25 as Obj[]) expect(C.bm25View(d, want.query, want.k1, want.b, want.top.length)).toEqual(want);
  });
  it("dense frames", () => {
    const got = C.denseFrames(d, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).concat(C.denseFrames(d, [0, 1, 2, 3, 4], "binary", 3));
    expect(got).toEqual(v.dense);
  });
  it("hybrid frames", () => {
    const got = [0, 1, 2, 3, 4, 5].map((qi) => C.hybridFrames(d, qi)).concat([C.hybridFrames(d, 7, 60, 50, 8, 10)]);
    expect(got).toEqual(v.hybrid);
  });
  it("chunk view", () => expect(C.chunkView(rs, 0, 0, 2400)).toEqual(v.chunk));
});
