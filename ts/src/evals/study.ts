/**
 * What the Evals site's chapters animate (engine 1.8.0): each study runs the generator and the
 * statistics under a configuration and returns the frames, with a caption per frame. A port of
 * agent_loop_sim/evals/study.py; the configurations come from the Python reference through
 * evals_data.json.
 */
import { fixed } from "../context/text";
import type { Obj } from "../data";
import data from "../evals_data.json";
import { Rng } from "../rng";
import * as G from "./generator";
import { zcrit } from "./mathx";
import * as S from "./stats";

export const LOOP: Obj = data.loop;
export const PASSK: Obj = data.passk;
export const BIAS: Obj = data.bias;
export const UNC: Obj = data.unc;
export const COVER: Obj = data.cover;
export const TIE: Obj = data.tie;
export const PAIRED: Obj = data.paired;
export const SIZES: Obj = data.sizes;

export function pct(x: number, d = 1): string {
  return fixed(x * 100.0 + 0.0, d) + "%";
}

function clip(lo: number, hi: number): number[] {
  return [lo > 0.0 ? lo : 0.0, hi < 1.0 ? hi : 1.0];
}

// ── chapter 1 ──

export function loopStudy(cfg: Obj = LOOP, pop: G.Task[] | null = null): Obj {
  const p = pop ?? G.defaultPopulation();
  const tN: number = cfg.tasks;
  const m: number = cfg.runs;
  const idx = G.sample(tN, p.length, new Rng(cfg.seed));
  const mu = G.mus(
    idx.map((i) => p[i]!),
    cfg.theta,
  );
  const rows = G.runs(mu, m, cfg.rho, new Rng(cfg.seed + 1));
  const z = zcrit(S.ALPHA);
  const succ = new Array<number>(tN).fill(0);
  const seen = new Array<number>(tN).fill(0);
  let k = 0;
  let n = 0;
  const frames: Obj[] = [];
  for (let j = 0; j < m; j++) {
    for (let t = 0; t < tN; t++) {
      const x = rows[t]![j]!;
      succ[t] = succ[t]! + x;
      seen[t] = seen[t]! + 1;
      k += x;
      n += 1;
      const ss: number[] = [];
      const rr: number[] = [];
      for (let i = 0; i < tN; i++)
        if (seen[i]! > 0) {
          ss.push(succ[i]!);
          rr.push(seen[i]!);
        }
      const cl = S.clusterSe(ss, rr);
      const wil = S.wilson(k, n);
      const band = cl.tasks > 1 ? clip(cl.p - z * cl.se, cl.p + z * cl.se) : [0.0, 1.0];
      const cap =
        `Run ${j + 1} on task ${t + 1}: ${x === 1 ? "pass" : "fail"}. ${k} of ${n} runs pass (${pct(k / n)}). ` +
        `95% interval ${pct(wil[0]!)} to ${pct(wil[1]!)} if every run were independent; ` +
        `${pct(band[0]!)} to ${pct(band[1]!)} with tasks as the unit.`;
      frames.push({ task: t, run: j, x, k, n, p: k / n, wilson: wil, cluster: band, se: cl.se, naive_se: cl.naive_se, caption: cap });
    }
  }
  return { cfg, idx, mu, runs: rows, truth_tasks: S.mean(mu), truth_population: G.trueScore(p, cfg.theta), frames };
}

// ── chapter 3 ──

export function passkStudy(cfg: Obj = PASSK, pop: G.Task[] | null = null): Obj {
  const p = pop ?? G.defaultPopulation();
  const n: number = cfg.n;
  const idx = G.sample(cfg.tasks, p.length, new Rng(cfg.seed));
  const mu = G.mus(
    idx.map((i) => p[i]!),
    cfg.theta,
  );
  const rows = G.runs(mu, n, cfg.rho, new Rng(cfg.seed + 1));
  const c = rows.map((r) => Math.trunc(S.total(r)));
  const s = Math.sqrt(cfg.rho);
  const frames: Obj[] = [];
  for (let k = 1; k <= n; k++) {
    const at = c.map((ci) => S.passAtK(n, ci, k));
    const hat = c.map((ci) => S.passHatK(n, ci, k));
    const plug = c.map((ci) => S.pluginPassAtK(n, ci, k));
    const trueAt = S.mean(mu.map((u) => S.taskPassCurves(u, s, k).pass_at_k as number));
    const trueHat = S.mean(mu.map((u) => S.taskPassCurves(u, s, k).pass_hat_k as number));
    const a = S.mean(at);
    const h = S.mean(hat);
    const cap =
      `k = ${k}: pass@${k} = ${pct(a)} (some one of ${k} tries passes), pass^${k} = ${pct(h)} (all ${k} pass). ` +
      `True values for these tasks: ${pct(trueAt)} and ${pct(trueHat)}.`;
    frames.push({ k, at, hat, plug, pass_at_k: a, pass_hat_k: h, plugin: S.mean(plug), true_at: trueAt, true_hat: trueHat, caption: cap });
  }
  return { cfg, idx, mu, runs: rows, c, frames };
}

export function biasStudy(cfg: Obj = BIAS): Obj {
  const r = new Rng(cfg.seed);
  const n: number = cfg.n;
  const k: number = cfg.k;
  const mu: number = cfg.mu;
  let sa = 0.0;
  let sp = 0.0;
  let sh = 0.0;
  for (let i = 0; i < cfg.reps; i++) {
    let c = 0;
    for (let j = 0; j < n; j++) if (r.random() < mu) c += 1;
    sa = sa + S.passAtK(n, c, k);
    sp = sp + S.pluginPassAtK(n, c, k);
    sh = sh + S.passHatK(n, c, k);
  }
  const reps: number = cfg.reps;
  const truth = S.taskPassCurves(mu, 0.0, k);
  return { cfg, unbiased: sa / reps, plugin: sp / reps, hat: sh / reps, true_at: truth.pass_at_k, true_hat: truth.pass_hat_k };
}

// ── chapter 4 ──

export function uncertaintyStudy(size: number, mode: string, cfg: Obj = UNC, pop: G.Task[] | null = null): Obj {
  const p = pop ?? G.defaultPopulation();
  const theta: number = cfg.theta;
  const pick = (seed: number) =>
    G.mus(
      G.sample(size, p.length, new Rng(seed)).map((j) => p[j]!),
      theta,
    );
  const mu0 = pick(cfg.seed);
  const truth = mode === "tasks" ? G.trueScore(p, theta) : S.mean(mu0);
  const covered: Record<string, number> = {};
  const width: Record<string, number> = {};
  for (const m of S.METHODS) {
    covered[m] = 0;
    width[m] = 0.0;
  }
  const frames: Obj[] = [];
  for (let i = 0; i < cfg.replicates; i++) {
    const mu = mode === "tasks" ? pick(cfg.seed + 100 + i) : mu0;
    const x = G.runs(mu, 1, 0.0, new Rng(cfg.seed + 1000 + i)).map((row) => row[0]!);
    let k = 0;
    for (const v of x) k += v;
    const ivs: Record<string, number[]> = {
      wald: S.wald(k, size),
      wilson: S.wilson(k, size),
      exact: S.exact(k, size),
      bootstrap: S.bootstrap(x, cfg.boot, cfg.seed + 5000 + i).ci as number[],
    };
    const hit: Record<string, boolean> = {};
    for (const m of S.METHODS) {
      const [lo, hi] = ivs[m]!;
      hit[m] = lo! <= truth && truth <= hi!;
      if (hit[m]) covered[m]! += 1;
      width[m] = width[m]! + (hi! - lo!);
    }
    const missed = S.METHODS.filter((m) => !hit[m]);
    const cap =
      `Repeat ${i + 1}: ${k} of ${size} tasks pass (${pct(k / size)}). Wilson: ${pct(ivs.wilson![0]!)} to ` +
      `${pct(ivs.wilson![1]!)}. ` +
      (missed.length === 0 ? "Every interval covers the truth." : "Missed the truth: " + missed.join(", ") + ".");
    frames.push({ k, p: k / size, ivs, hit, covered: { ...covered }, caption: cap });
  }
  const reps: number = cfg.replicates;
  const cov: Record<string, number> = {};
  const mw: Record<string, number> = {};
  for (const m of S.METHODS) {
    cov[m] = covered[m]! / reps;
    mw[m] = width[m]! / reps;
  }
  return { size, mode, truth, frames, coverage: cov, mean_width: mw };
}

export function coverageCurves(n: number, ps: number[]): Obj {
  const out: Obj = { n, ps };
  for (const m of ["wald", "wilson", "exact"]) {
    const ivs = Array.from({ length: n + 1 }, (_, k) => S.interval(m, k, n));
    out[m] = ps.map((p) => S.coverage(m, n, p, S.ALPHA, ivs));
  }
  return out;
}

export function tieStudy(cfg: Obj = TIE): Obj {
  const n: number = cfg.n;
  const two = S.twoProportion(cfg.k1, n, cfg.k2, n);
  const p1 = cfg.k1 / n;
  const p2 = cfg.k2 / n;
  return {
    cfg,
    test: two,
    n_unpaired: S.nUnpaired(p1, p2),
    mde_unpaired: S.mdeUnpaired(n, p1),
    mde_paired: S.mdePaired(n, SIZES.psi),
    n_paired: S.nPaired(SIZES.psi / 2 + (p1 - p2) / 2, SIZES.psi / 2 - (p1 - p2) / 2),
  };
}

export function sizesTable(cfg: Obj = SIZES): Obj {
  const margins: Obj[] = [];
  for (const p of cfg.margin_ps as number[]) for (const h of cfg.margins as number[]) margins.push({ p, h, n: S.nForMargin(p, h) });
  const base: number = cfg.base;
  const psi: number = cfg.psi;
  const diffs = (cfg.deltas as number[]).map((d) => ({
    delta: d,
    unpaired: S.nUnpaired(base + d, base),
    paired: S.nPaired(psi / 2 + d / 2, psi / 2 - d / 2),
  }));
  return { cfg, margins, diffs };
}

// ── chapter 5 ──

export function varianceSplit(muA: number[], muB: number[], m: number, rho: number): Obj {
  const t = muA.length;
  const d = muA.map((x, i) => x - muB[i]!);
  const w = muA.map((x, i) => ((x * (1.0 - x) + muB[i]! * (1.0 - muB[i]!)) * (1.0 + (m - 1) * rho)) / m);
  const within = S.mean(w) / t;
  return {
    tasks_paired: S.variance(d, 0) / t,
    tasks_unpaired: (S.variance(muA, 0) + S.variance(muB, 0)) / t,
    runs: within,
    se_paired: Math.sqrt(S.variance(d, 0) / t + within),
    se_unpaired: Math.sqrt((S.variance(muA, 0) + S.variance(muB, 0)) / t + within),
  };
}

export function pairedStudy(m: number, cfg: Obj = PAIRED, pop: G.Task[] | null = null): Obj {
  const p = pop ?? G.defaultPopulation();
  const tN: number = cfg.tasks;
  const idx = G.sample(tN, p.length, new Rng(cfg.seed));
  const tasks = idx.map((i) => p[i]!);
  const muA = G.mus(tasks, cfg.theta_a);
  const muB = G.mus(tasks, cfg.theta_b);
  const ra = G.runs(muA, m, cfg.rho, new Rng(cfg.seed + 1));
  const rb = G.runs(muB, m, cfg.rho, new Rng(cfg.seed + 2));
  const a = ra.map((r) => S.mean(r));
  const b = rb.map((r) => S.mean(r));
  const a1 = ra.map((r) => r[0]!);
  const b1 = rb.map((r) => r[0]!);
  const frames: Obj[] = [];
  for (let t = cfg.batch; t <= tN; t += cfg.batch) {
    const c = S.compare(a.slice(0, t), b.slice(0, t));
    const tab = S.pairedTable(a1.slice(0, t), b1.slice(0, t));
    const mc = S.mcnemar(tab.n10!, tab.n01!);
    const cap =
      `${t} tasks: A ${pct(c.a)}, B ${pct(c.b)}, difference ${pct(c.delta)}. Paired 95% interval ` +
      `${pct(c.ci_paired[0])} to ${pct(c.ci_paired[1])}; unpaired ${pct(c.ci_unpaired[0])} to ` +
      `${pct(c.ci_unpaired[1])}. First runs: A alone passes ${tab.n10}, B alone ${tab.n01}; ` +
      `McNemar p = ${fixed(mc.exact_p, 4)}.`;
    frames.push({ t, compare: c, table: tab, mcnemar: mc, caption: cap });
  }
  return {
    cfg,
    m,
    idx,
    mu_a: muA,
    mu_b: muB,
    a,
    b,
    a1,
    b1,
    truth: S.mean(muA) - S.mean(muB),
    split: varianceSplit(muA, muB, m, cfg.rho),
    frames,
  };
}
