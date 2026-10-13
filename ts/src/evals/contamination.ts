/**
 * Contamination and leakage (engine 1.9.0): leaked tasks inflate a score; n-gram overlap, canary
 * strings and a time split each see part of it. A port of agent_loop_sim/evals/contamination.py; the
 * configuration comes from the Python reference through evals2_data.json. See that file for the model.
 */
import { fixed } from "../context/text";
import type { Obj } from "../data";
import data from "../evals2_data.json";
import { Rng } from "../rng";
import * as G from "./generator";
import * as S from "./stats";

export const CONTAM: Obj = data.contam;

function pct(x: number): string {
  return fixed(x * 100.0 + 0.0, 1) + "%";
}

export function ngrams(toks: number[], n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < toks.length - n + 1; i++) out.push(toks.slice(i, i + n).join(","));
  return out;
}

export function build(cfg: Obj = CONTAM, pop: G.Task[] | null = null): Obj {
  const p = pop ?? G.defaultPopulation();
  const idx = G.sample(cfg.tasks, p.length, new Rng(cfg.seed + 2));
  const mu = G.mus(
    idx.map((i) => p[i]!),
    cfg.theta,
  );
  const r = new Rng(cfg.seed);
  const tasks: Obj[] = [];
  for (let t = 0; t < cfg.tasks; t++) {
    const text: number[] = [];
    for (let i = 0; i < cfg.length; i++) text.push(r.randint(0, cfg.vocab - 1));
    const quarter = r.randint(0, cfg.quarters - 1);
    const uLeak = r.random();
    const uPara = r.random();
    const uCanary = r.random();
    const uRun = r.random();
    const para: number[] = [];
    for (const tok of text) {
      const m = r.random();
      const rep = r.randint(0, cfg.vocab - 1);
      para.push(m < cfg.mutate ? rep : tok);
    }
    tasks.push({
      i: t,
      mu: mu[t],
      text,
      quarter,
      pre: quarter < cfg.cutoff,
      u_leak: uLeak,
      paraphrased: uPara < cfg.paraphrase,
      canary: uPara >= cfg.paraphrase && uCanary < cfg.canary_kept,
      u_run: uRun,
      para,
    });
  }
  const rf = new Rng(cfg.seed + 1);
  const filler: number[][] = [];
  for (let d = 0; d < cfg.filler_docs; d++) {
    const doc: number[] = [];
    for (let i = 0; i < cfg.length; i++) doc.push(rf.randint(0, cfg.vocab - 1));
    filler.push(doc);
  }
  return { tasks, filler };
}

export function atLeak(leak: number, b: Obj, cfg: Obj = CONTAM): Obj {
  const n: number = cfg.ngram;
  const tasks = b.tasks as Obj[];
  const corpus = new Set<string>();
  for (const doc of b.filler as number[][]) for (const g of ngrams(doc, n)) corpus.add(g);
  const leaked = tasks.map((t) => t.pre && t.u_leak < leak);
  tasks.forEach((t, i) => {
    if (leaked[i]) for (const g of ngrams(t.paraphrased ? t.para : t.text, n)) corpus.add(g);
  });
  const rows: Obj[] = tasks.map((t, i) => {
    const lk = leaked[i]!;
    const grams = ngrams(t.text, n);
    let hit = 0;
    for (const g of grams) if (corpus.has(g)) hit += 1;
    const overlap = hit / grams.length;
    const p = lk ? t.mu + (1.0 - t.mu) * cfg.memorise : t.mu;
    return {
      i: t.i,
      quarter: t.quarter,
      pre: t.pre,
      leaked: lk,
      kind: lk ? (t.paraphrased ? "paraphrase" : "verbatim") : "none",
      overlap,
      flagged: overlap >= cfg.threshold,
      pass: t.u_run < p,
      clean_pass: t.u_run < t.mu,
    };
  });
  return summarise(
    leak,
    rows,
    tasks.some((t, i) => leaked[i] && t.canary),
    cfg,
  );
}

export function summarise(leak: number, rows: Obj[], canary: boolean, cfg: Obj): Obj {
  const n = rows.length;
  const count = (f: (r: Obj) => boolean): number => {
    let k = 0;
    for (const r of rows) if (f(r)) k += 1;
    return k;
  };
  const k = count((r) => r.pass);
  const kClean = count((r) => r.clean_pass);
  const nLeaked = count((r) => r.leaked);
  const nVerb = count((r) => r.kind === "verbatim");
  const nPara = count((r) => r.kind === "paraphrase");
  const flagVerb = count((r) => r.kind === "verbatim" && r.flagged);
  const flagPara = count((r) => r.kind === "paraphrase" && r.flagged);
  const flagClean = count((r) => !r.leaked && r.flagged);
  const kept = rows.filter((r) => !r.flagged);
  let kKept = 0;
  for (const r of kept) if (r.pass) kKept += 1;
  const pre = rows.filter((r) => r.pre);
  const post = rows.filter((r) => !r.pre);
  let kPre = 0;
  for (const r of pre) if (r.pass) kPre += 1;
  let kPost = 0;
  for (const r of post) if (r.pass) kPost += 1;
  const byQ: Obj[] = [];
  for (let q = 0; q < cfg.quarters; q++) {
    const rq = rows.filter((r) => r.quarter === q);
    let kq = 0;
    for (const r of rq) if (r.pass) kq += 1;
    byQ.push({ quarter: q, n: rq.length, k: kq, rate: kq / rq.length, wilson: S.wilson(kq, rq.length) });
  }
  const split = S.twoProportion(kPre, pre.length, kPost, post.length);
  const frames = [
    {
      stage: "tasks",
      caption: `${n} tasks, each with a creation date; the model's training data stops at the cutoff, after quarter ${cfg.cutoff - 1}.`,
    },
    {
      stage: "leak",
      caption: `Leak rate ${pct(leak)}: ${nLeaked} tasks from before the cutoff are in the training data, ${nVerb} verbatim and ${nPara} paraphrased.`,
    },
    {
      stage: "scan",
      caption:
        `The ${cfg.ngram}-gram scan flags ${flagVerb} of ${nVerb} verbatim leaks and ${flagPara} of ${nPara} paraphrases` +
        (flagClean ? `, and ${flagClean} clean tasks` : "") +
        ". " +
        (canary ? "The canary string comes back from the model: the benchmark leaked." : "The model does not complete the canary string."),
    },
    {
      stage: "run",
      caption: `The agent scores ${pct(k / n)}; without the leaks it would score ${pct(kClean / n)}. Dropping the flagged tasks gives ${pct(kKept / kept.length)}.`,
    },
    {
      stage: "split",
      caption:
        `Before the cutoff ${pct(kPre / pre.length)}, after it ${pct(kPost / post.length)}: ` +
        `a gap of ${fixed((kPre / pre.length - kPost / post.length) * 100.0 + 0.0, 1)} points (p = ${fixed(split.p + 0.0, 3)}).`,
    },
  ];
  return {
    leak,
    n,
    rate: k / n,
    clean_rate: kClean / n,
    inflation: (k - kClean) / n,
    leaked: nLeaked,
    verbatim: nVerb,
    paraphrased: nPara,
    flagged_verbatim: flagVerb,
    flagged_paraphrased: flagPara,
    flagged_clean: flagClean,
    canary,
    kept: kept.length,
    rate_kept: kKept / kept.length,
    pre: { n: pre.length, k: kPre },
    post: { n: post.length, k: kPost },
    split,
    by_quarter: byQ,
    rows,
    frames,
  };
}

export function contaminationStudy(cfg: Obj = CONTAM, pop: G.Task[] | null = null): Obj {
  const b = build(cfg, pop);
  let survive = 1.0;
  for (let i = 0; i < cfg.ngram; i++) survive = survive * (1.0 - cfg.mutate);
  const levels: Obj = {};
  for (const L of cfg.leaks as number[]) levels[fixed(L + 0.0, 2)] = atLeak(L, b, cfg);
  return { cfg, survive, levels };
}
