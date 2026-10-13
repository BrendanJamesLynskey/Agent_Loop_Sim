/**
 * The statistics of evaluating agents (engine 1.8.0; graders and cost since 1.9.0): pass@k and pass^k, intervals for a success
 * rate, clustered standard errors, paired comparisons and power. A port of
 * agent_loop_sim/evals/stats.py (same operations in the same order; sums are left-to-right loops).
 */
import type { Obj } from "../data";
import { Rng } from "../rng";
import { betainc, exp, lgamma, ln, normCdf, normPpf, zcrit } from "./mathx";

export const ALPHA = 0.05;
export const METHODS = ["wald", "wilson", "exact", "bootstrap"];

export function total(xs: number[]): number {
  let s = 0.0;
  for (const x of xs) s = s + x;
  return s;
}

export function mean(xs: number[]): number {
  return total(xs) / xs.length;
}

export function variance(xs: number[], ddof = 1): number {
  const m = mean(xs);
  let s = 0.0;
  for (const x of xs) {
    const d = x - m;
    s = s + d * d;
  }
  return s / (xs.length - ddof);
}

function clip(lo: number, hi: number): number[] {
  return [lo > 0.0 ? lo : 0.0, hi < 1.0 ? hi : 1.0];
}

// ── pass@k and pass^k ──

export function passAtK(n: number, c: number, k: number): number {
  if (n - c < k) return 1.0;
  let prod = 1.0;
  for (let i = n - c + 1; i <= n; i++) prod = prod * (1.0 - k / i);
  return 1.0 - prod;
}

export function passHatK(n: number, c: number, k: number): number {
  if (c < k) return 0.0;
  let prod = 1.0;
  for (let i = 0; i < k; i++) prod = prod * ((c - i) / (n - i));
  return prod;
}

export function pluginPassAtK(n: number, c: number, k: number): number {
  const q = 1.0 - c / n;
  let v = 1.0;
  for (let i = 0; i < k; i++) v = v * q;
  return 1.0 - v;
}

function pw(x: number, k: number): number {
  let v = 1.0;
  for (let i = 0; i < k; i++) v = v * x;
  return v;
}

export function taskPassCurves(mu: number, s: number, k: number): Obj {
  const stayFailIfPass = (1.0 - s) * (1.0 - mu);
  const stayFailIfFail = s + (1.0 - s) * (1.0 - mu);
  const passIfPass = s + (1.0 - s) * mu;
  const passIfFail = (1.0 - s) * mu;
  const allFail = mu * pw(stayFailIfPass, k) + (1.0 - mu) * pw(stayFailIfFail, k);
  const allPass = mu * pw(passIfPass, k) + (1.0 - mu) * pw(passIfFail, k);
  return { pass_at_k: 1.0 - allFail, pass_hat_k: allPass };
}

// ── binomial ──

export function binomPmf(k: number, n: number, p: number): number {
  if (k < 0 || k > n) return 0.0;
  if (p <= 0.0) return k === 0 ? 1.0 : 0.0;
  if (p >= 1.0) return k === n ? 1.0 : 0.0;
  return exp(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0) + k * ln(p) + (n - k) * ln(1.0 - p));
}

export function binomCdf(k: number, n: number, p: number): number {
  if (k < 0) return 0.0;
  if (k >= n) return 1.0;
  if (p <= 0.0) return 1.0;
  if (p >= 1.0) return 0.0;
  return betainc(n - k, k + 1.0, 1.0 - p);
}

export function betaPpf(q: number, a: number, b: number): number {
  let lo = 0.0;
  let hi = 1.0;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi) break;
    if (betainc(a, b, mid) < q) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

// ── intervals ──

export function wald(k: number, n: number, alpha = ALPHA): number[] {
  if (n === 0) return [0.0, 1.0];
  const z = zcrit(alpha);
  const p = k / n;
  const h = z * Math.sqrt((p * (1.0 - p)) / n);
  return clip(p - h, p + h);
}

export function wilson(k: number, n: number, alpha = ALPHA): number[] {
  if (n === 0) return [0.0, 1.0];
  const z = zcrit(alpha);
  const p = k / n;
  const z2 = z * z;
  const den = 1.0 + z2 / n;
  const mid = p + z2 / (2.0 * n);
  const half = z * Math.sqrt((p * (1.0 - p)) / n + z2 / (4.0 * n * n));
  return clip((mid - half) / den, (mid + half) / den);
}

export function exact(k: number, n: number, alpha = ALPHA): number[] {
  if (n === 0) return [0.0, 1.0];
  const lo = k === 0 ? 0.0 : betaPpf(alpha / 2.0, k, n - k + 1.0);
  const hi = k === n ? 1.0 : betaPpf(1.0 - alpha / 2.0, k + 1.0, n - k);
  return [lo, hi];
}

export function interval(method: string, k: number, n: number, alpha = ALPHA): number[] {
  if (method === "wald") return wald(k, n, alpha);
  if (method === "wilson") return wilson(k, n, alpha);
  if (method === "exact") return exact(k, n, alpha);
  throw new Error(`no closed-form interval '${method}'`);
}

export function quantile(sorted: number[], q: number): number {
  const h = (sorted.length - 1) * q;
  const i = Math.trunc(h);
  if (i >= sorted.length - 1) return sorted[sorted.length - 1]!;
  const a = sorted[i]!;
  const b = sorted[i + 1]!;
  return a + (h - i) * (b - a);
}

export function bootstrap(values: number[], b: number, seed: number, alpha = ALPHA): Obj {
  const r = new Rng(seed);
  const t = values.length;
  const means: number[] = [];
  for (let j = 0; j < b; j++) {
    let s = 0.0;
    for (let i = 0; i < t; i++) s = s + values[r.randint(0, t - 1)]!;
    means.push(s / t);
  }
  const srt = [...means].sort((x, y) => x - y);
  return { ci: [quantile(srt, alpha / 2.0), quantile(srt, 1.0 - alpha / 2.0)], se: Math.sqrt(variance(means)), means };
}

export function coverage(method: string, n: number, p: number, alpha = ALPHA, intervals: number[][] | null = null): number {
  const ivs = intervals ?? Array.from({ length: n + 1 }, (_, k) => interval(method, k, n, alpha));
  let c = 0.0;
  for (let k = 0; k <= n; k++) {
    const [lo, hi] = ivs[k]!;
    if (lo! <= p && p <= hi!) c = c + binomPmf(k, n, p);
  }
  return c;
}

// ── clustered data ──

export function clusterSe(successes: number[], runs: number[]): Obj {
  const t = successes.length;
  let n = 0;
  let sAll = 0;
  for (let i = 0; i < t; i++) {
    n += runs[i]!;
    sAll += successes[i]!;
  }
  const p = sAll / n;
  const naive = Math.sqrt((p * (1.0 - p)) / n);
  let acc = 0.0;
  for (let i = 0; i < t; i++) {
    const d = successes[i]! - runs[i]! * p;
    acc = acc + d * d;
  }
  const se = t > 1 ? Math.sqrt((t / (t - 1)) * acc) / n : naive;
  return { p, n, tasks: t, naive_se: naive, se, design_effect: naive > 0 ? (se * se) / (naive * naive) : 1.0 };
}

export function designEffect(m: number, icc: number): number {
  return 1.0 + (m - 1) * icc;
}

// ── comparing two agents ──

export function pairedTable(a: number[], b: number[]): Record<string, number> {
  const t: Record<string, number> = { n11: 0, n10: 0, n01: 0, n00: 0 };
  for (let i = 0; i < a.length && i < b.length; i++) t[`n${a[i]}${b[i]}`]! += 1;
  return t;
}

export function mcnemar(n10: number, n01: number): Obj {
  const d = n10 + n01;
  if (d === 0) return { discordant: 0, exact_p: 1.0, chi2: 0.0, chi2_p: 1.0 };
  const lo = n10 < n01 ? n10 : n01;
  const p = 2.0 * binomCdf(lo, d, 0.5);
  const diff = Math.abs(n10 - n01) - 1.0;
  const stat = (diff * diff) / d;
  return { discordant: d, exact_p: p < 1.0 ? p : 1.0, chi2: stat, chi2_p: 2.0 * normCdf(-Math.sqrt(stat)) };
}

export function compare(a: number[], b: number[], alpha = ALPHA): Obj {
  const n = a.length;
  const d = a.map((x, i) => x - b[i]!);
  const delta = mean(d);
  const va = variance(a);
  const vb = variance(b);
  const vd = variance(d);
  const seP = Math.sqrt(vd / n);
  const seU = Math.sqrt(va / n + vb / n);
  const cov = (va + vb - vd) / 2.0;
  const corr = va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0.0;
  const z = zcrit(alpha);
  return {
    n,
    a: mean(a),
    b: mean(b),
    delta,
    se_paired: seP,
    se_unpaired: seU,
    corr,
    ci_paired: [delta - z * seP, delta + z * seP],
    ci_unpaired: [delta - z * seU, delta + z * seU],
    z_paired: seP > 0 ? delta / seP : 0.0,
    p_paired: seP > 0 ? 2.0 * normCdf(-Math.abs(delta / seP)) : 1.0,
    p_unpaired: seU > 0 ? 2.0 * normCdf(-Math.abs(delta / seU)) : 1.0,
  };
}

export function twoProportion(k1: number, n1: number, k2: number, n2: number, alpha = ALPHA): Obj {
  const p1 = k1 / n1;
  const p2 = k2 / n2;
  const pp = (k1 + k2) / (n1 + n2);
  const se0 = Math.sqrt(pp * (1.0 - pp) * (1.0 / n1 + 1.0 / n2));
  const z = (p1 - p2) / se0;
  const se = Math.sqrt((p1 * (1.0 - p1)) / n1 + (p2 * (1.0 - p2)) / n2);
  const zc = zcrit(alpha);
  return {
    p1,
    p2,
    delta: p1 - p2,
    z,
    p: 2.0 * normCdf(-Math.abs(z)),
    ci: [p1 - p2 - zc * se, p1 - p2 + zc * se],
    wilson1: wilson(k1, n1, alpha),
    wilson2: wilson(k2, n2, alpha),
  };
}

// ── power and sample size ──

export function nForMargin(p: number, halfWidth: number, alpha = ALPHA): number {
  const z = zcrit(alpha);
  return Math.ceil((z * z * p * (1.0 - p)) / (halfWidth * halfWidth));
}

export function powerUnpaired(n: number, p1: number, p2: number, alpha = ALPHA): number {
  const za = zcrit(alpha);
  const pb = (p1 + p2) / 2.0;
  const d = Math.abs(p1 - p2);
  const s0 = Math.sqrt(2.0 * pb * (1.0 - pb));
  const s1 = Math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2));
  return normCdf((d * Math.sqrt(n) - za * s0) / s1);
}

export function nUnpaired(p1: number, p2: number, alpha = ALPHA, power = 0.8): number {
  const za = zcrit(alpha);
  const zb = normPpf(power);
  const pb = (p1 + p2) / 2.0;
  const d = p1 - p2;
  const x = za * Math.sqrt(2.0 * pb * (1.0 - pb)) + zb * Math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2));
  return Math.ceil((x * x) / (d * d));
}

export function powerPaired(n: number, p10: number, p01: number, alpha = ALPHA): number {
  const za = zcrit(alpha);
  const psi = p10 + p01;
  const d = Math.abs(p10 - p01);
  return normCdf((d * Math.sqrt(n) - za * Math.sqrt(psi)) / Math.sqrt(psi - d * d));
}

export function nPaired(p10: number, p01: number, alpha = ALPHA, power = 0.8): number {
  const za = zcrit(alpha);
  const zb = normPpf(power);
  const psi = p10 + p01;
  const d = p10 - p01;
  const x = za * Math.sqrt(psi) + zb * Math.sqrt(psi - d * d);
  return Math.ceil((x * x) / (d * d));
}

export function mdeUnpaired(n: number, p: number, alpha = ALPHA, power = 0.8): number {
  return (zcrit(alpha) + normPpf(power)) * Math.sqrt((2.0 * p * (1.0 - p)) / n);
}

export function mdePaired(n: number, psi: number, alpha = ALPHA, power = 0.8): number {
  return (zcrit(alpha) + normPpf(power)) * Math.sqrt(psi / n);
}

// ── graders and cost (since 1.9) ──

/** Counts by (truth row, judged column), in the order of `labels`. */
export function confusion(truth: string[], judged: string[], labels: string[]): number[][] {
  const t = labels.map(() => labels.map(() => 0));
  for (let i = 0; i < truth.length && i < judged.length; i++) {
    const r = labels.indexOf(truth[i]!);
    const c = labels.indexOf(judged[i]!);
    t[r]![c] = t[r]![c]! + 1;
  }
  return t;
}

/** Cohen's κ from a square table: (p_o − p_e)/(1 − p_e). */
export function cohenKappa(table: number[][]): Obj {
  const k = table.length;
  let n = 0.0;
  let diag = 0.0;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) n = n + table[i]![j]!;
    diag = diag + table[i]![i]!;
  }
  const po = diag / n;
  let pe = 0.0;
  for (let i = 0; i < k; i++) {
    let row = 0.0;
    let col = 0.0;
    for (let j = 0; j < k; j++) {
      row = row + table[i]![j]!;
      col = col + table[j]![i]!;
    }
    pe = pe + (row / n) * (col / n);
  }
  const kappa = pe < 1.0 ? (po - pe) / (1.0 - pe) : 1.0;
  return { n, po, pe, kappa };
}

/** Rogan and Gladen's correction of an observed rate for a grader's sensitivity and specificity. */
export function roganGladen(pObs: number, se: number, sp: number): number {
  const x = (pObs + sp - 1.0) / (se + sp - 1.0);
  return x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x;
}

/** A judge-graded rate corrected with Se and Sp estimated on a calibration set; delta-method SE. */
export function correctedRate(k: number, n: number, seK: number, seN: number, spK: number, spN: number, alpha = ALPHA): Obj {
  const p = k / n;
  const se = seK / seN;
  const sp = spK / spN;
  const j = se + sp - 1.0;
  const pi = roganGladen(p, se, sp);
  const v = ((p * (1.0 - p)) / n + (pi * pi * se * (1.0 - se)) / seN + ((1.0 - pi) * (1.0 - pi) * sp * (1.0 - sp)) / spN) / (j * j);
  const s = Math.sqrt(v);
  const z = zcrit(alpha);
  return {
    observed: p,
    se,
    sp,
    youden: j,
    corrected: pi,
    stderr: s,
    naive_stderr: Math.sqrt((p * (1.0 - p)) / n) / j,
    ci: clip(pi - z * s, pi + z * s),
  };
}

/** Indices of the points [cost, success] no other point beats, in order of cost (then success, highest first). */
export function paretoFront(points: number[][]): number[] {
  const keep: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const [c, s] = points[i] as [number, number];
    let beaten = false;
    for (let j = 0; j < points.length; j++) {
      const [c2, s2] = points[j] as [number, number];
      if (j !== i && c2 <= c && s2 >= s && (c2 < c || s2 > s)) {
        beaten = true;
        break;
      }
    }
    if (!beaten) keep.push(i);
  }
  keep.sort((a, b) => {
    const pa = points[a]!;
    const pb = points[b]!;
    if (pa[0]! !== pb[0]!) return pa[0]! < pb[0]! ? -1 : 1;
    if (pa[1]! !== pb[1]!) return pa[1]! > pb[1]! ? -1 : 1;
    return a - b;
  });
  return keep;
}
