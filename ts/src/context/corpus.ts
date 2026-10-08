/**
 * The corpus, its questions, the chunking configs and the shipped vectors: a port of
 * agent_loop_sim/context/corpus.py. The caller passes the parsed data files (the sites fetch
 * them lazily; the tests read them from src/agent_loop_sim/data/context/).
 */
import type { Tokenizer } from "../tokenizer";
import { hex, sha256, utf8Bytes } from "../protocols/sha256";
import { fixedRanges, recursiveRanges, semanticRanges, spans } from "./chunking";
import { ranges, sentenceStarts, type Piece } from "./text";
import { decode } from "./vectors";

export type Obj = { [k: string]: any }; // eslint-disable-line @typescript-eslint/no-explicit-any

export const CHUNKERS: Record<string, Obj> = {
  "fixed-128": { kind: "fixed", size: 128, overlap: 0 },
  "fixed-256": { kind: "fixed", size: 256, overlap: 0 },
  "fixed-512": { kind: "fixed", size: 512, overlap: 0 },
  "fixed-256-o64": { kind: "fixed", size: 256, overlap: 64 },
  "recursive-256": { kind: "recursive", size: 256 },
  "semantic-256": { kind: "semantic", size: 256, pct: 25 },
};
export const DEFAULT = "recursive-256";
export const FILES = ["corpus.json", "emb-questions.json", "emb-sentences.json", "rerank.json", "pca.json"].concat(
  Object.keys(CHUNKERS).map((c) => `emb-${c}.json`),
);

export interface Chunk {
  article: number;
  start: number;
  end: number;
  tokens: number;
}

export function spansSha256(sp: [number, number, number][]): string {
  return hex(sha256(utf8Bytes(sp.map((c) => `${c[0]}:${c[1]}:${c[2]}`).join("\n"))));
}

export class Corpus {
  articles: Obj[];
  questions: Obj[];
  files: Record<string, Obj>;
  tok: Tokenizer;
  private _pieces: Piece[][] | null = null;
  private _chunks = new Map<string, Chunk[]>();
  private _vecs = new Map<string, Int8Array[]>();
  private _rerank: Map<number, number>[] | null = null;

  constructor(corpus: Obj, files: Record<string, Obj>, tok: Tokenizer) {
    this.articles = corpus.articles as Obj[];
    this.questions = corpus.questions as Obj[];
    this.files = files;
    this.tok = tok;
  }

  get pieces(): Piece[][] {
    if (this._pieces === null) this._pieces = this.articles.map((a) => this.tok.pieces(a.text as string));
    return this._pieces;
  }

  sentences(): [number, number, number][] {
    const out: [number, number, number][] = [];
    this.articles.forEach((a, ai) => {
      const p = this.pieces[ai]!;
      for (const [sa, sb] of ranges(sentenceStarts(a.text as string, p), p.length)) out.push([ai, p[sa]![0], p[sb - 1]![1]]);
    });
    return out;
  }

  vectors(name: string): Int8Array[] {
    let v = this._vecs.get(name);
    if (!v) {
      const f = this.files[`emb-${name}.json`];
      if (!f) throw new Error(`no vectors '${name}' loaded`);
      v = decode(f.b64 as string, f.dim as number);
      this._vecs.set(name, v);
    }
    return v;
  }

  chunks(config: string): Chunk[] {
    const hit = this._chunks.get(config);
    if (hit) return hit;
    const c = CHUNKERS[config];
    if (!c) throw new Error(`unknown config '${config}'`);
    const out: Chunk[] = [];
    const sentVecs = c.kind === "semantic" ? this.vectors("sentences") : [];
    let s0 = 0;
    this.articles.forEach((a, ai) => {
      const p = this.pieces[ai]!;
      const text = a.text as string;
      let rs: [number, number][];
      if (c.kind === "fixed") rs = fixedRanges(p, 0, p.length, c.size, c.overlap);
      else if (c.kind === "recursive") rs = recursiveRanges(text, p, c.size);
      else {
        const ns = sentenceStarts(text, p).length;
        rs = semanticRanges(text, p, c.size, c.pct, sentVecs.slice(s0, s0 + ns));
        s0 += ns;
      }
      for (const [st, en, tk] of spans(p, rs)) out.push({ article: ai, start: st, end: en, tokens: tk });
    });
    this._chunks.set(config, out);
    return out;
  }

  chunkText(ch: Chunk): string {
    return (this.articles[ch.article]!.text as string).slice(ch.start, ch.end);
  }

  spansSha256(config: string): string {
    return spansSha256(this.chunks(config).map((c) => [c.article, c.start, c.end]));
  }

  /** The cross-encoder's scores (logit x 1000, rounded) per question, for its pool (default config). */
  rerankScores(): Map<number, number>[] {
    if (this._rerank === null) {
      const f = this.files["rerank.json"];
      if (!f) throw new Error("no reranker scores loaded");
      this._rerank = (f.scores as [number, number][][]).map((row) => new Map(row));
    }
    return this._rerank;
  }
}
