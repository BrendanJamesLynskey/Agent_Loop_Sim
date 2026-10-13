/**
 * A simulated LLM judge (engine 1.9.0): pairwise verdicts with position, verbosity and
 * self-preference biases, the swap test, Cohen's κ against the truth, and a pass/fail judge corrected
 * with a calibration set. A port of agent_loop_sim/evals/judge.py; the configurations come from the
 * Python reference through evals2_data.json. See that file for the model.
 */
import { fixed } from "../context/text";
import type { Obj } from "../data";
import data from "../evals2_data.json";
import { Rng } from "../rng";
import * as G from "./generator";
import { logistic, normPpf } from "./mathx";
import * as S from "./stats";

export const LABELS: string[] = data.labels;
export const JUDGE: Obj = data.judge;
export const GRADING: Obj = data.grading;

const TWO32 = 4294967296.0;

function z(r: Rng): number {
  return normPpf((r.nextU32() + 0.5) / TWO32);
}

function truthOf(qa: number, qb: number, band: number): string {
  const d = qa - qb;
  if (d < band && -d < band) return "tie";
  return d > 0.0 ? "A" : "B";
}

export function pairs(cfg: Obj = JUDGE): Obj[] {
  const r = new Rng(cfg.seed);
  const out: Obj[] = [];
  for (let i = 0; i < cfg.pairs; i++) {
    const side: Obj = {};
    for (const s of ["A", "B"]) {
      const q = z(r);
      const len = z(r);
      const own = r.random() < cfg.own_share ? 1 : 0;
      side[s] = { q, len, own, tokens: len > -2.0 ? Math.floor(220.0 + 90.0 * len + 0.5) : 40 };
    }
    out.push({ i, A: side.A, B: side.B, truth: truthOf(side.A.q, side.B.q, cfg.tie_band) });
  }
  return out;
}

export function preferFirst(x: Obj, y: Obj, bias: Obj, tau: number): number {
  const d = x.q - y.q + bias.verbosity * (x.len - y.len) + bias.self * (x.own - y.own) + bias.position;
  return logistic(d / tau);
}

function acc(ix: number[], v: string[], truth: string[]): number {
  if (ix.length === 0) return 0.0;
  let k = 0;
  for (const i of ix) if (v[i] === truth[i]) k += 1;
  return k / ix.length;
}

export function judgePairs(setting: string, cfg: Obj = JUDGE, ps: Obj[] | null = null): Obj {
  const bias = cfg.settings[setting];
  const tau: number = cfg.tau;
  const items = ps ?? pairs(cfg);
  const r = new Rng(cfg.seed + 1);
  const truth: string[] = [];
  const single: string[] = [];
  const debiased: string[] = [];
  let firstWins = 0;
  let consistent = 0;
  const frames: Obj[] = [];
  for (const it of items) {
    const a = it.A;
    const b = it.B;
    const pAb = preferFirst(a, b, bias, tau);
    const pBa = preferFirst(b, a, bias, tau);
    const u1 = r.random();
    const u2 = r.random();
    const v1 = u1 < pAb ? "A" : "B";
    const v2 = u2 < pBa ? "B" : "A";
    if (v1 === "A") firstWins += 1;
    if (v2 === "B") firstWins += 1;
    const same = v1 === v2;
    if (same) consistent += 1;
    const vd = same ? v1 : "tie";
    truth.push(it.truth);
    single.push(v1);
    debiased.push(vd);
    const n = truth.length;
    const k1 = S.cohenKappa(S.confusion(truth, single, LABELS));
    const kd = S.cohenKappa(S.confusion(truth, debiased, LABELS));
    const cap =
      `Pair ${n}: ` +
      (it.truth === "tie" ? "a tie by the truth" : `${it.truth} is better`) +
      `. Shown A first the judge picks ${v1}; shown B first it picks ${v2}` +
      (same ? ". Both orders agree." : ": the verdict follows the order, so it counts as a tie.") +
      ` κ so far: ${fixed(k1.kappa + 0.0, 2)} one order, ${fixed(kd.kappa + 0.0, 2)} swapped.`;
    frames.push({ i: it.i, p_ab: pAb, p_ba: pBa, v1, v2, debiased: vd, kappa1: k1.kappa, kappa_d: kd.kappa, consistent, caption: cap });
  }
  const n = items.length;
  const t1 = S.confusion(truth, single, LABELS);
  const td = S.confusion(truth, debiased, LABELS);
  const decided: number[] = [];
  for (let i = 0; i < n; i++) if (debiased[i] !== "tie") decided.push(i);
  let rightD = 0;
  for (const i of decided) if (debiased[i] === truth[i]) rightD += 1;
  let right1 = 0;
  for (let i = 0; i < n; i++) if (single[i] === truth[i]) right1 += 1;
  const longWorse: number[] = [];
  const ownWorse: number[] = [];
  for (let i = 0; i < n; i++) {
    const A = items[i]!.A;
    const B = items[i]!.B;
    if (truth[i] !== "tie" && A.len > B.len === (truth[i] === "B")) longWorse.push(i);
    if (truth[i] !== "tie" && A.own !== B.own && (A.own === 1) === (truth[i] === "B")) ownWorse.push(i);
  }
  return {
    setting,
    bias,
    table1: t1,
    table_d: td,
    kappa1: S.cohenKappa(t1),
    kappa_d: S.cohenKappa(td),
    accuracy1: right1 / n,
    consistency: consistent / n,
    first_share: firstWins / (2 * n),
    decided: decided.length / n,
    accuracy_decided: decided.length ? rightD / decided.length : 0.0,
    long_worse: { n: longWorse.length, acc1: acc(longWorse, single, truth), acc_d: acc(longWorse, debiased, truth) },
    own_worse: { n: ownWorse.length, acc1: acc(ownWorse, single, truth), acc_d: acc(ownWorse, debiased, truth) },
    frames,
  };
}

export function judgeStudy(cfg: Obj = JUDGE): Obj {
  const ps = pairs(cfg);
  const settings: Obj = {};
  for (const s of cfg.order as string[]) settings[s] = judgePairs(s, cfg, ps);
  return { cfg, pairs: ps, settings };
}

export function observedRate(pi: number, se: number, sp: number): number {
  return pi * se + (1.0 - pi) * (1.0 - sp);
}

function pctS(x: number): string {
  return fixed(x * 100.0 + 0.0, 1) + "%";
}

export function gradingStudy(cfg: Obj = GRADING, pop: G.Task[] | null = null): Obj {
  const p = pop ?? G.defaultPopulation();
  const theta: number = cfg.theta;
  const graded = (n: number, s0: number): [number[], number[]] => {
    const mu = G.mus(
      G.sample(n, p.length, new Rng(s0)).map((i) => p[i]!),
      theta,
    );
    const x = G.runs(mu, 1, 0.0, new Rng(s0 + 1)).map((row) => row[0]!);
    const rj = new Rng(s0 + 2);
    const j: number[] = [];
    for (const v of x) {
      const u = rj.random();
      j.push(v === 1 ? (u < cfg.se ? 1 : 0) : u < 1.0 - cfg.sp ? 1 : 0);
    }
    return [x, j];
  };
  const [x, j] = graded(cfg.tasks, cfg.seed);
  const [cx, cj] = graded(cfg.cal, cfg.seed + 3);
  let seN = 0;
  let seK = 0;
  let spN = 0;
  let spK = 0;
  for (let i = 0; i < cx.length; i++) {
    if (cx[i] === 1) {
      seN += 1;
      seK += cj[i]!;
    } else {
      spN += 1;
      spK += 1 - cj[i]!;
    }
  }
  const frames: Obj[] = [];
  let kt = 0;
  let kj = 0;
  for (let i = 0; i < x.length; i++) {
    kt += x[i]!;
    kj += j[i]!;
    const n = i + 1;
    const cap =
      `Task ${n}: the unit tests say ${x[i] ? "pass" : "fail"}, the judge says ${j[i] ? "pass" : "fail"}` +
      (x[i] === j[i] ? "" : " (wrong)") +
      `. Scores so far: tests ${pctS(kt / n)}, judge ${pctS(kj / n)}.`;
    frames.push({ i, truth: x[i], judge: j[i], k_tests: kt, k_judge: kj, caption: cap });
  }
  const corr = S.correctedRate(kj, x.length, seK, seN, spK, spN);
  const curve = (cfg.curve as number[]).map((pi) => ({ pi, observed: observedRate(pi, cfg.se, cfg.sp) }));
  return {
    cfg,
    truth: x,
    judge: j,
    tests_rate: kt / x.length,
    judge_rate: kj / x.length,
    wilson_tests: S.wilson(kt, x.length),
    wilson_judge: S.wilson(kj, x.length),
    calibration: { n: cx.length, se_k: seK, se_n: seN, sp_k: spK, sp_n: spN },
    corrected: corr,
    kappa: S.cohenKappa(
      S.confusion(
        x.map((v) => String(v)),
        j.map((v) => String(v)),
        ["1", "0"],
      ),
    ),
    curve,
    frames,
  };
}
