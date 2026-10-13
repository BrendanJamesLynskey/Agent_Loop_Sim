/**
 * Engine 1.8.0's evals module against the Python reference (fixtures/evals_fixtures.json) with no
 * tolerance: the special functions, every statistic, the generator, every chapter study (frames and
 * captions included) and the SWE-style task's runs and grading.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { evals, Rng, Tokenizer, type Obj } from "../src/index";

const { mathx: X, stats: S, generator: G, study: St, swe: W } = evals;
const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/evals_fixtures.json"), "utf8")) as Obj;
const tok = new Tokenizer(readFileSync(join(ROOT, "src/agent_loop_sim/data/qwen2.5-merges.txt"), "utf8"));

describe("version", () => {
  it("is 1.9.0", () => expect(fx.engine).toBe("1.9.0"));
});

describe("special functions, bit for bit", () => {
  const m = fx.mathx;
  it("exp", () => m.exp.forEach(([x, v]: number[]) => expect(X.exp(x!)).toBe(v)));
  it("lgamma", () => m.lgamma.forEach(([x, v]: number[]) => expect(X.lgamma(x!)).toBe(v)));
  it("betainc", () => m.betainc.forEach(([a, b, x, v]: number[]) => expect(X.betainc(a!, b!, x!)).toBe(v)));
  it("normal cdf", () => m.norm_cdf.forEach(([x, v]: number[]) => expect(X.normCdf(x!)).toBe(v)));
  it("normal quantile", () => m.norm_ppf.forEach(([p, v]: number[]) => expect(X.normPpf(p!)).toBe(v)));
  it("logistic", () => m.logistic.forEach(([x, v]: number[]) => expect(X.logistic(x!)).toBe(v)));
  it("zcrit", () => m.zcrit.forEach(([a, v]: number[]) => expect(X.zcrit(a!)).toBe(v)));
});

describe("statistics", () => {
  const s = fx.stats;
  it("pass@k, pass^k, plug-in", () => {
    s.pass_at_k.forEach(([n, c, k, v]: number[]) => expect(S.passAtK(n!, c!, k!)).toBe(v));
    s.pass_hat_k.forEach(([n, c, k, v]: number[]) => expect(S.passHatK(n!, c!, k!)).toBe(v));
    s.plugin.forEach(([n, c, k, v]: number[]) => expect(S.pluginPassAtK(n!, c!, k!)).toBe(v));
    s.task_curves.forEach(([mu, sc, k, v]: [number, number, number, Obj]) => expect(S.taskPassCurves(mu, sc, k)).toEqual(v));
  });
  it("binomial", () => {
    s.pmf.forEach(([k, n, p, v]: number[]) => expect(S.binomPmf(k!, n!, p!)).toBe(v));
    s.cdf.forEach(([k, n, p, v]: number[]) => expect(S.binomCdf(k!, n!, p!)).toBe(v));
  });
  it("intervals", () => {
    s.intervals.forEach(([k, n, w, wi, ex]: [number, number, number[], number[], number[]]) => {
      expect(S.wald(k, n)).toEqual(w);
      expect(S.wilson(k, n)).toEqual(wi);
      expect(S.exact(k, n)).toEqual(ex);
    });
    s.exact_99.forEach(([k, n, ex]: [number, number, number[]]) => expect(S.exact(k, n, 0.01)).toEqual(ex));
    s.quantile.forEach(([q, v]: number[]) => expect(S.quantile([1, 2, 4, 8, 9], q!)).toBe(v));
    expect(S.bootstrap([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], 50, 7)).toEqual(s.bootstrap);
    s.coverage.forEach(([m, n, p, v]: [string, number, number, number]) => expect(S.coverage(m, n, p)).toBe(v));
  });
  it("clusters, McNemar, comparisons", () => {
    const CL = [
      [[3, 1, 0, 2, 3], [3, 3, 3, 3, 3]],
      [[1, 0, 1, 1], [1, 1, 1, 1]],
      [[5, 0, 2], [5, 2, 4]],
    ];
    CL.forEach(([a, b], i) => expect(S.clusterSe(a!, b!)).toEqual(s.cluster[i]));
    s.mcnemar.forEach(([b, c, v]: [number, number, Obj]) => expect(S.mcnemar(b, c)).toEqual(v));
    expect(S.pairedTable([1, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0])).toEqual(s.table[0]);
    expect(S.compare([1, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0])).toEqual(s.compare[0]);
    expect(S.compare([0.5, 1.0, 0.0, 0.75, 0.25], [0.25, 1.0, 0.25, 0.5, 0.0])).toEqual(s.compare[1]);
    expect(S.twoProportion(341, 500, 339, 500)).toEqual(s.two[0]);
    expect(S.twoProportion(60, 100, 45, 100)).toEqual(s.two[1]);
  });
  it("power and sample size", () => {
    const p = s.power;
    p.margin.forEach(([pp, h, v]: number[]) => expect(S.nForMargin(pp!, h!)).toBe(v));
    p.unpaired.forEach(([a, b, n, v]: number[]) => expect(S.powerUnpaired(n!, a!, b!)).toBe(v));
    p.n_unpaired.forEach(([a, b, v, v2]: number[]) => {
      expect(S.nUnpaired(a!, b!)).toBe(v);
      expect(S.nUnpaired(a!, b!, 0.01, 0.9)).toBe(v2);
    });
    p.paired.forEach(([a, b, n, v]: number[]) => expect(S.powerPaired(n!, a!, b!)).toBe(v));
    p.n_paired.forEach(([a, b, v, v2]: number[]) => {
      expect(S.nPaired(a!, b!)).toBe(v);
      expect(S.nPaired(a!, b!, 0.01, 0.9)).toBe(v2);
    });
    p.mde.forEach(([n, u, pd]: number[]) => {
      expect(S.mdeUnpaired(n!, 0.6)).toBe(u);
      expect(S.mdePaired(n!, 0.2)).toBe(pd);
    });
  });
});

describe("generator", () => {
  const g = fx.generator;
  const pop = G.population();
  it("population", () => {
    expect(pop.slice(0, 60)).toEqual(g.population_head);
    g.true_scores.forEach(([th, v]: number[]) => expect(G.trueScore(pop, th!)).toBe(v));
  });
  it("sampling and runs", () => {
    expect(G.sample(20, pop.length, new Rng(5))).toEqual(g.sample);
    g.runs.forEach(([rho, rows]: [number, number[][]]) => expect(G.runs([0.1, 0.5, 0.9, 0.3], 6, rho, new Rng(11))).toEqual(rows));
  });
  it("benchmark and ICC", () => {
    expect(G.benchmark(0.3, 25, 4, 0.4, 99, pop)).toEqual(g.benchmark);
    expect(G.icc(g.benchmark.runs)).toBe(g.icc);
  });
});

describe("chapter studies", () => {
  it("chapter 1: runs accumulating", () => expect(St.loopStudy()).toEqual(fx.loop));
  it("chapter 3: pass@k and pass^k", () => expect(St.passkStudy()).toEqual(fx.passk));
  it("chapter 3: bias check", () => expect(St.biasStudy()).toEqual(fx.bias));
  it("chapter 4: repeats", () => {
    for (const size of St.UNC.sizes as number[])
      for (const mode of St.UNC.modes as string[]) expect(St.uncertaintyStudy(size, mode)).toEqual(fx.uncertainty[`${size}|${mode}`]);
  });
  it("chapter 4: coverage curves, the tie, sample sizes", () => {
    for (const n of St.COVER.sizes as number[]) expect(St.coverageCurves(n, St.COVER.ps)).toEqual(fx.coverage[String(n)]);
    expect(St.tieStudy()).toEqual(fx.tie);
    expect(St.sizesTable()).toEqual(fx.sizes);
  });
  it("chapter 5: two agents", () => {
    for (const m of St.PAIRED.runs as number[]) expect(St.pairedStudy(m)).toEqual(fx.paired[String(m)]);
  });
});

describe("the SWE-style task", () => {
  it("every patch under every test patch", () => {
    for (const p of Object.keys(W.PATCHES))
      for (const t of Object.keys(W.TEST_PATCHES)) expect(W.anatomy(p, t, tok)).toEqual(fx.swe[`${p}|${t}`]);
  });
});
