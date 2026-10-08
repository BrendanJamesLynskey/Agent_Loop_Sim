/**
 * Text units for the context module: pieces, paragraphs and sentences as piece ranges, the BM25
 * analyser and toFixed formatting. A port of agent_loop_sim/context/text.py, statement for statement.
 */
const WS = "\\t\\n\\x0b\\x0c\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const TERM = /[\p{L}\p{N}]+/gu;
/** Lucene's English stop set (EnglishAnalyzer.ENGLISH_STOP_WORDS_SET), 33 words. */
export const STOPWORDS = new Set(
  ("a an and are as at be but by for if in into is it no not of on or such that the their then there these they " +
    "this to was will with").split(" "),
);
const SENT_END = /[.!?]["'”’)\]]*[\r\n]*$/u;
const WS_START = new RegExp(`^[${WS}]`, "u");
const OPENER = /^[\p{Lu}\p{N}"'“‘(\[]/u;
const LEAD_WS = new RegExp(`^[${WS}]+`, "u");

export type Piece = [number, number, number];

export function analyse(text: string, stop = true): string[] {
  const out: string[] = [];
  for (const m of text.matchAll(TERM)) {
    const t = m[0].toLowerCase();
    if (stop && STOPWORDS.has(t)) continue;
    out.push(t);
  }
  return out;
}

export function paragraphStarts(text: string, pieces: Piece[]): number[] {
  const out = [0];
  for (let i = 1; i < pieces.length; i++) {
    if (text.slice(pieces[i - 1]![0], pieces[i - 1]![1]).includes("\n")) out.push(i);
  }
  return out;
}

function visibleStart(text: string, pieces: Piece[], i: number): string {
  let s = text.slice(pieces[i]![0], pieces[i]![1]).replace(LEAD_WS, "");
  if (s === "" && i + 1 < pieces.length) s = text.slice(pieces[i + 1]![0], pieces[i + 1]![1]);
  return s;
}

export function sentenceStarts(text: string, pieces: Piece[]): number[] {
  const out = [0];
  for (let i = 1; i < pieces.length; i++) {
    const prev = text.slice(pieces[i - 1]![0], pieces[i - 1]![1]);
    if (prev.includes("\n")) {
      out.push(i);
      continue;
    }
    const cur = text.slice(pieces[i]![0], pieces[i]![1]);
    if (SENT_END.test(prev) && WS_START.test(cur) && OPENER.test(visibleStart(text, pieces, i))) out.push(i);
  }
  return out;
}

export function ranges(starts: number[], n: number): [number, number][] {
  return starts.map((s, j) => [s, j + 1 < starts.length ? starts[j + 1]! : n]);
}

/** x with d decimals (Python's `fixed` rounds the exact binary value half up, as toFixed does). */
export function fixed(x: number, d: number): string {
  return x.toFixed(d);
}
