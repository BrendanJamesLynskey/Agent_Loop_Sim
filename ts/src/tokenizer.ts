/**
 * Qwen2.5's byte-level BPE tokenizer, the TS twin of `agent_loop_sim/tokenizer.py`: the same
 * merges file, the same pre-tokenizer regex (with the same two rewrites, so V8's `u`-flag
 * RegExp and Python's `regex` module split text identically), NFC first, special tokens split
 * out before BPE. Token ids: bytes 0-255 in GPT-2's bytes-to-unicode order, merge r -> 256 + r,
 * the 22 added tokens from 151643.
 */
const WS =
  "\\t\\n\\x0b\\x0c\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
export const PATTERN =
  "'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])" +
  "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" +
  "|\\p{N}" +
  `| ?[^${WS}\\p{L}\\p{N}]+[\\r\\n]*` +
  `|[${WS}]*[\\r\\n]+` +
  `|[${WS}]+(?![^${WS}])` +
  `|[${WS}]+`;

export const ADDED_TOKENS = [
  "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
  "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
  "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>",
  "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
  "<|repo_name|>", "<|file_sep|>",
];
export const ADDED_BASE = 151643;

function bytesToUnicode(): { table: string[]; order: number[] } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const table = new Array<string>(256).fill("");
  for (let i = 0; i < bs.length; i++) table[bs[i]!] = String.fromCharCode(cs[i]!);
  return { table, order: bs };
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export class Tokenizer {
  readonly nMerges: number;
  private byteChar: string[];
  private ids = new Map<string, number>();
  private ranks = new Map<string, number>();
  private pre: RegExp;
  private special: RegExp;
  private cache = new Map<string, number[]>();
  private utf8 = new TextEncoder();

  constructor(mergesText: string) {
    const { table, order } = bytesToUnicode();
    this.byteChar = table;
    order.forEach((b, i) => this.ids.set(table[b]!, i));
    let r = 0;
    for (const line of mergesText.split("\n")) {
      if (!line || line.startsWith("#version")) continue;
      const sp = line.indexOf(" ");
      const a = line.slice(0, sp);
      const b = line.slice(sp + 1);
      // a NUL separator: no token contains one (bytes are mapped to printable characters)
      this.ranks.set(a + "\u0000" + b, r);
      this.ids.set(a + b, 256 + r);
      r++;
    }
    this.nMerges = r;
    this.pre = new RegExp(PATTERN, "gu");
    this.special = new RegExp(ADDED_TOKENS.map(escapeRe).join("|"), "g");
  }

  private bpe(word: string): number[] {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1;
      let bestRank = this.nMerges;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.ranks.get(parts[i] + "\u0000" + parts[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      const a = parts[best]!;
      const b = parts[best + 1]!;
      const merged: string[] = [];
      let i = 0;
      while (i < parts.length) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          merged.push(a + b);
          i += 2;
        } else {
          merged.push(parts[i]!);
          i += 1;
        }
      }
      parts = merged;
    }
    const out = parts.map((p) => this.ids.get(p)!);
    this.cache.set(word, out);
    return out;
  }

  private encodePlain(text: string, out: number[]): void {
    for (const m of text.matchAll(this.pre)) {
      let word = "";
      for (const byte of this.utf8.encode(m[0])) word += this.byteChar[byte];
      for (const id of this.bpe(word)) out.push(id);
    }
  }

  encode(text: string): number[] {
    text = text.normalize("NFC");
    const out: number[] = [];
    let pos = 0;
    for (const m of text.matchAll(this.special)) {
      if (m.index! > pos) this.encodePlain(text.slice(pos, m.index), out);
      out.push(ADDED_BASE + ADDED_TOKENS.indexOf(m[0]));
      pos = m.index! + m[0].length;
    }
    if (pos < text.length) this.encodePlain(text.slice(pos), out);
    return out;
  }

  count(text: string): number {
    return this.encode(text).length;
  }

  /** The pre-tokenizer's pieces of plain (NFC, special-token-free) text as [start, end, tokens]
   * (engine 1.4.0, for the context module's chunkers). Offsets are UTF-16 code units, equal to the
   * Python reference's code points for text without astral characters (the corpus has none). */
  pieces(text: string): [number, number, number][] {
    const out: [number, number, number][] = [];
    for (const m of text.matchAll(this.pre)) {
      let word = "";
      for (const byte of this.utf8.encode(m[0])) word += this.byteChar[byte];
      out.push([m.index!, m.index! + m[0].length, this.bpe(word).length]);
    }
    return out;
  }
}
