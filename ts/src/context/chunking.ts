/**
 * Fixed, recursive and semantic chunkers over pre-tokenizer pieces: a port of
 * agent_loop_sim/context/chunking.py, statement for statement.
 */
import { paragraphStarts, ranges, sentenceStarts, type Piece } from "./text";
import { cosine } from "./vectors";

type R = [number, number];

function tok(pieces: Piece[], a: number, b: number): number {
  let t = 0;
  for (let i = a; i < b; i++) t += pieces[i]![2];
  return t;
}

export function fixedRanges(pieces: Piece[], a0: number, n: number, size: number, overlap = 0): R[] {
  const out: R[] = [];
  let a = a0;
  while (a < n) {
    let t = 0;
    let b = a;
    while (b < n && (t + pieces[b]![2] <= size || b === a)) {
      t += pieces[b]![2];
      b += 1;
    }
    out.push([a, b]);
    if (b >= n) break;
    if (overlap > 0) {
      let j = b;
      let t2 = 0;
      while (j - 1 > a && t2 + pieces[j - 1]![2] <= overlap) {
        j -= 1;
        t2 += pieces[j]![2];
      }
      a = j;
    } else {
      a = b;
    }
  }
  return out;
}

function merge(pieces: Piece[], leaves: R[], size: number, breaks: Set<number> | null = null): R[] {
  const out: R[] = [];
  let cur: R | null = null;
  let curT = 0;
  for (const leaf of leaves) {
    const t = tok(pieces, leaf[0], leaf[1]);
    if (cur !== null && curT + t <= size && !(breaks !== null && breaks.has(leaf[0]))) {
      cur = [cur[0], leaf[1]];
      curT += t;
    } else {
      if (cur !== null) out.push(cur);
      cur = [leaf[0], leaf[1]];
      curT = t;
    }
  }
  if (cur !== null) out.push(cur);
  return out;
}

export function recursiveRanges(text: string, pieces: Piece[], size: number): R[] {
  const n = pieces.length;
  const sents = sentenceStarts(text, pieces);
  const leaves: R[] = [];
  for (const [pa, pb] of ranges(paragraphStarts(text, pieces), n)) {
    if (tok(pieces, pa, pb) <= size) {
      leaves.push([pa, pb]);
      continue;
    }
    const inner = sents.filter((s) => pa <= s && s < pb);
    for (const [sa, sb] of ranges(inner, pb)) {
      if (tok(pieces, sa, sb) <= size) leaves.push([sa, sb]);
      else leaves.push(...fixedRanges(pieces, sa, sb, size));
    }
  }
  return merge(pieces, leaves, size);
}

export function semanticBreaks(sims: number[], pct: number): [number, boolean[]] {
  if (!sims.length) return [0, []];
  const srt = [...sims].sort((a, b) => a - b);
  const thr = srt[Math.floor(((srt.length - 1) * pct) / 100)]!;
  return [thr, sims.map((s) => s <= thr)];
}

export function semanticRanges(text: string, pieces: Piece[], size: number, pct: number, sentVecs: ArrayLike<number>[]): R[] {
  const n = pieces.length;
  const sents = ranges(sentenceStarts(text, pieces), n);
  if (sentVecs.length !== sents.length) throw new Error(`${sentVecs.length} sentence vectors for ${sents.length} sentences`);
  const sims: number[] = [];
  for (let i = 0; i < sents.length - 1; i++) sims.push(cosine(sentVecs[i]!, sentVecs[i + 1]!));
  const [, brk] = semanticBreaks(sims, pct);
  const leaves: R[] = [];
  const breaks = new Set<number>();
  sents.forEach(([sa, sb], i) => {
    if (i > 0 && brk[i - 1]) breaks.add(sa);
    if (tok(pieces, sa, sb) <= size) {
      leaves.push([sa, sb]);
    } else {
      const parts = fixedRanges(pieces, sa, sb, size);
      for (const p of parts) breaks.add(p[0]);
      leaves.push(...parts);
      if (sb < n) breaks.add(sb);
    }
  });
  return merge(pieces, leaves, size, breaks);
}

export function spans(pieces: Piece[], rs: R[]): [number, number, number][] {
  return rs.map(([a, b]) => [pieces[a]![0], pieces[b - 1]![1], tok(pieces, a, b)]);
}
