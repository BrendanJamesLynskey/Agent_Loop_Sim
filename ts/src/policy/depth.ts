/**
 * Defence in depth with imperfect layers (engine 1.10.0): independent and common-mode correlated misses,
 * closed forms and a seeded simulation. A port of agent_loop_sim/policy/depth.py (same operation order).
 */
import type { Obj } from "../data";
import { Rng } from "../rng";

export function allMissIndependent(m: number[]): number {
  let p = 1.0;
  for (const x of m) p = p * x;
  return p;
}

export function qGivenCommon(m: number[], c: number): number[] {
  if (c < 0.0 || c >= 1.0) throw new Error("c must be in [0, 1)");
  for (const x of m) if (x < c) throw new Error("c cannot exceed any layer's miss rate");
  return m.map((x) => (x - c) / (1.0 - c));
}

export function allMissCommon(m: number[], c: number): number {
  return c + (1.0 - c) * allMissIndependent(qGivenCommon(m, c));
}

export function pairCorrelation(mi: number, mj: number, c: number): number {
  const q = qGivenCommon([mi, mj], c);
  const both = c + (1.0 - c) * q[0]! * q[1]!;
  return (both - mi * mj) / Math.sqrt(mi * (1.0 - mi) * mj * (1.0 - mj));
}

export function atLeastOne(p: number, n: number): number {
  let s = 1.0;
  for (let i = 0; i < n; i++) s = s * (1.0 - p);
  return 1.0 - s;
}

export function simulate(m: number[], c: number, trials: number, seed: number): Obj {
  const q = qGivenCommon(m, c);
  const r = new Rng(seed);
  let through = 0;
  const single = m.map(() => 0);
  const pair = m.map(() => m.map(() => 0));
  for (let k = 0; k < trials; k++) {
    const common = r.random() < c;
    const miss: boolean[] = [];
    for (let i = 0; i < m.length; i++) {
      const u = r.random();
      miss.push(common || u < q[i]!);
    }
    let ok = true;
    for (let i = 0; i < m.length; i++) {
      if (miss[i]) {
        single[i] = single[i]! + 1;
        for (let j = i + 1; j < m.length; j++) if (miss[j]) pair[i]![j] = pair[i]![j]! + 1;
      } else ok = false;
    }
    if (ok) through += 1;
  }
  return { trials, through, rate: through / trials, single, pair };
}

export function depthTable(base: number, m: number[], cs: number[], attempts: number): Obj[] {
  const rows: Obj[] = [];
  for (const c of cs) {
    for (let k = 0; k <= m.length; k++) {
      const mk = m.slice(0, k);
      const pm = k > 0 ? allMissCommon(mk, c) : 1.0;
      rows.push({ c, layers: k, all_miss: pm, end_to_end: base * pm, any_of_n: atLeastOne(base * pm, attempts) });
    }
  }
  return rows;
}
