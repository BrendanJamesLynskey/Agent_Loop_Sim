/**
 * Quantised vectors and the similarity arithmetic: a port of agent_loop_sim/context/vectors.py.
 * Integer dot products, one correctly rounded square root and one division, so int8 and int4
 * cosines are bit-identical to the Python reference.
 */
export const PRECISIONS = ["int8", "int4", "binary"] as const;
export type Precision = (typeof PRECISIONS)[number];

export function decode(b64: string, dim: number): Int8Array[] {
  const bin = atob(b64);
  if (bin.length % dim) throw new Error("vector bytes are not a multiple of the dimension");
  const all = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) all[i] = bin.charCodeAt(i) << 24 >> 24;
  const out: Int8Array[] = [];
  for (let i = 0; i < all.length; i += dim) out.push(all.subarray(i, i + dim));
  return out;
}

type Vec = ArrayLike<number>;

export function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

export function cosine(a: Vec, b: Vec): number {
  const na = dot(a, a);
  const nb = dot(b, b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / Math.sqrt(na * nb);
}

export function toInt4(v: Vec): Int8Array {
  const out = new Int8Array(v.length);
  for (let i = 0; i < v.length; i++) {
    const q = v[i]!;
    const m = Math.floor((Math.abs(q) * 14 + 127) / 254);
    out[i] = q < 0 ? -m : m;
  }
  return out;
}

export function toBits(v: Vec): Int8Array {
  const out = new Int8Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! > 0 ? 1 : 0;
  return out;
}

export function binarySim(a: Vec, b: Vec): number {
  let h = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) h += 1;
  return (a.length - 2 * h) / a.length;
}

export function prepare(vecs: Int8Array[], precision: string): Int8Array[] {
  if (precision === "int8") return vecs;
  if (precision === "int4") return vecs.map(toInt4);
  if (precision === "binary") return vecs.map(toBits);
  throw new Error(`unknown precision '${precision}'`);
}

export function similarity(a: Vec, b: Vec, precision: string): number {
  return precision === "binary" ? binarySim(a, b) : cosine(a, b);
}

export function nbytes(dim: number, precision: string): number {
  return ({ int8: dim, int4: dim / 2, binary: dim / 8 } as Record<string, number>)[precision]!;
}
