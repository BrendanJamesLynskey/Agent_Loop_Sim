/**
 * The studies behind chapters 5–9 (engine 1.7.0). A port of agent_loop_sim/orchestration/study.py.
 */
import { LATENCY, PRICES, type Obj } from "../data";
import * as D from "./des";
import * as P from "./patterns";
import * as R from "./reliability";

// Chapter 5

export function taskOf(k: number, resultTokens: number): Obj {
  const t: Obj = { ...P.TASK };
  t.subtasks =
    k === (P.TASK.subtasks as string[]).length
      ? [...(P.TASK.subtasks as string[])]
      : Array.from({ length: k }, (_, i) => "topic " + (i + 1));
  t.tool_result_tokens = resultTokens;
  return t;
}

export function modelOf(slope: number): Obj {
  return { ...P.MODEL, slope_per_k: slope };
}

const ROW_KEYS = ["pattern", "model_calls", "input", "cached", "output", "cost", "latency_ms", "serial_ms", "success"];

export function patternsGrid(ks: number[], results: number[], slopes: number[]): Obj[] {
  const out: Obj[] = [];
  for (const k of ks)
    for (const rt of results)
      for (const sl of slopes) {
        const ms = P.compare("claude-sonnet-4.6", "hosted", modelOf(sl), taskOf(k, rt));
        out.push({
          k,
          result_tokens: rt,
          slope: sl,
          rows: ms.map((m) => Object.fromEntries(ROW_KEYS.map((x) => [x, m[x]]))),
        });
      }
  return out;
}

// Chapter 6

export const FAN: Obj = { chunks: 32, chunk_tokens: 2000, widths: [1, 2, 4, 8, 16, 32], rpm: 50, itpm: 40000, otpm: 8000 };

export function fanParams(width: number, rpm: number, itpm: number, otpm: number): Obj {
  return {
    workflows: 1,
    arrival: "batch",
    concurrency: width,
    rpm,
    itpm,
    otpm,
    fail: 0.0,
    jitter: [1.0, 1.0],
    semantic: false,
    trace: true,
    timeout_ms: 1e12,
    seed: 1,
  };
}

export function ceilingMs(plan: Obj, rpm: number, itpm: number, otpm: number): Obj {
  const price = PRICES[D.DEFAULTS.price]!;
  let req = 0;
  let tin = 0;
  let tout = 0;
  for (const c of plan.calls as Obj[]) {
    if (c.kind === "tool") continue;
    req += 1;
    tin += c.input - P.cachedTokens(c, price);
    tout += c.output;
  }
  const over = (total: number, cap: number): number => {
    const x = total - cap;
    return x > 0 ? x / (cap / 60000) : 0.0;
  };
  return { requests: over(req, rpm), input: over(tin, itpm), output: over(tout, otpm), req, tin, tout };
}

export function fanStudy(chunks: number, chunkTokens: number, widths: number[], rpm: number, itpm: number, otpm: number): Obj {
  const plan = P.planMap(chunks, chunkTokens);
  const price = PRICES[D.DEFAULTS.price]!;
  const profile = LATENCY[D.DEFAULTS.profile]!;
  const runs = widths.map((wd) => {
    const r = D.simulate(plan, fanParams(wd, rpm, itpm, otpm));
    return { width: wd, ms: r.makespan, trace: r.trace, changes: r.changes };
  });
  const t1 = widths[0] === 1 ? runs[0]!.ms : D.simulate(plan, fanParams(1, rpm, itpm, otpm)).makespan;
  const calls = plan.calls as Obj[];
  const serial = P.durationOf(calls[0]!, price, profile) + P.durationOf(calls[calls.length - 1]!, price, profile);
  const frac = 1 - serial / t1;
  const ceil = ceilingMs(plan, rpm, itpm, otpm);
  let bound = ceil.requests;
  if (ceil.input > bound) bound = ceil.input;
  if (ceil.output > bound) bound = ceil.output;
  const rows = runs.map((r) => ({
    width: r.width,
    ms: r.ms,
    speedup: t1 / r.ms,
    amdahl: 1 / (1 - frac + frac / r.width),
    ceiling: bound > 0 ? t1 / bound : 0.0,
  }));
  return {
    chunks,
    chunk_tokens: chunkTokens,
    rpm,
    itpm,
    otpm,
    t1,
    serial_ms: serial,
    parallel_fraction: frac,
    bound_ms: bound,
    ceiling: ceil,
    rows,
    runs,
    plan,
  };
}

// Chapter 7

export const REL: Obj = {
  ns: [1, 2, 3, 5, 8, 10, 15, 20, 30],
  e: 0.05,
  w: 0.03,
  runs: 2000,
  seed: 7,
  keep_n: 10,
  keep: 48,
  verifier: { recall: 0.8, false_reject: 0.05 },
};
export const REL_CONFIGS: Obj[] = [
  { name: "none", label: "one attempt", attempts: 1, verifier: false },
  { name: "retry", label: "retries (3 attempts)", attempts: 3, verifier: false },
  { name: "verify", label: "retries + verifier", attempts: 3, verifier: true },
];

export function relStudy(e: number, w: number, cfg: Obj): Obj {
  const v = cfg.verifier ? REL.verifier : null;
  return {
    name: cfg.name,
    label: cfg.label,
    attempts: cfg.attempts,
    verifier: v,
    curve: R.curve(REL.ns, e, w, cfg.attempts, v, REL.runs, REL.seed),
    grid: R.simulate(REL.keep_n, e, w, cfg.attempts, v, REL.keep, REL.seed, REL.keep),
  };
}

// Chapter 8

export const SCALE: Obj = {
  rates: [1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
  workflows: 200,
  seed: 11,
  samples: 120,
  detail_rate: 3.0,
  patterns: ["single", "supervisor", "map_reduce", "debate"],
};

export function capacityPerMin(plan: Obj, p: Obj): Obj {
  const price = PRICES[p.price]!;
  const profile = LATENCY[p.profile]!;
  let req = 0;
  let tin = 0;
  let tout = 0;
  let busy = 0.0;
  for (const c of plan.calls as Obj[]) {
    if (c.kind === "tool") continue;
    req += 1;
    tin += c.input - P.cachedTokens(c, price);
    tout += c.output;
    busy = busy + P.durationOf(c, price, profile);
  }
  const out: Obj = { requests: p.rpm / req, input: p.itpm / tin, output: p.otpm / tout, concurrency: (p.concurrency * 60000) / busy };
  let m = out.requests;
  for (const k of ["input", "output", "concurrency"]) if (out[k] < m) m = out[k];
  out.min = m;
  return out;
}

export function scaleParams(rate: number, extra: Obj | null = null): Obj {
  return { workflows: SCALE.workflows, per_min: rate, seed: SCALE.seed, ...(extra ?? {}) };
}

const SUMMARY_KEYS = [
  "pattern",
  "workflows",
  "ok",
  "wrong",
  "exhausted",
  "success",
  "p50",
  "p95",
  "p99",
  "mean",
  "cost",
  "cost_per_success",
  "makespan",
  "throughput_per_min",
  "attempts",
  "utilisation",
];

export function summary(r: Obj): Obj {
  return Object.fromEntries(SUMMARY_KEYS.map((k) => [k, r[k]]));
}

export function scaleSweep(patterns: string[], rates: number[], extra: Obj | null = null): Obj[] {
  return patterns.map((n) => {
    const pl = P.plan(n);
    const cap = capacityPerMin(pl, D.params(scaleParams(1.0, extra)));
    return { pattern: n, capacity: cap, rows: rates.map((r) => summary(D.simulate(pl, scaleParams(r, extra)))) };
  });
}

export function scaleDetail(pattern: string, rate: number, extra: Obj | null = null): Obj {
  const p = scaleParams(rate, extra);
  p.samples = SCALE.samples;
  const r = D.simulate(P.plan(pattern), p);
  const s = summary(r);
  s.series = r.series;
  s.runs = r.runs;
  return s;
}

// Chapter 9

export const CHOOSE: Obj = { k: [4, 16], rates: [1.0, 0.25], workflows: 1000 };
export const CONSTRAINTS: Obj[] = [
  { key: "success", op: ">=", label: "success at least" },
  { key: "p95", op: "<=", label: "p95 latency at most" },
  { key: "cost_per_success", op: "<=", label: "cost per success at most" },
  { key: "capacity", op: ">=", label: "workflows per minute at least" },
];

export function chooseTable(k: number, rate: number): Obj[] {
  const t = taskOf(k, 1500);
  return P.PATTERNS.map((n) => {
    const pl = P.plan(n, t);
    const r = D.simulate(pl, scaleParams(rate, { workflows: CHOOSE.workflows }));
    const s = summary(r);
    s.ci = R.wilson(r.ok, r.workflows);
    s.closed_success = D.successClosed(pl, D.DEFAULTS.fail, D.DEFAULTS.max_attempts);
    s.capacity = capacityPerMin(pl, D.params({})).min;
    return s;
  });
}

export function choose(table: Obj[], limits: Record<string, number>): Obj {
  let alive = table.map((r) => r.pattern as string);
  const steps: Obj[] = [];
  for (const c of CONSTRAINTS) {
    if (!(c.key in limits)) continue;
    const lim = limits[c.key]!;
    const keep: string[] = [];
    const out: string[] = [];
    for (const r of table) {
      if (!alive.includes(r.pattern)) continue;
      const v = r[c.key];
      const good = c.op === ">=" ? v >= lim : v <= lim;
      (good ? keep : out).push(r.pattern);
    }
    alive = keep;
    steps.push({ key: c.key, op: c.op, limit: lim, kept: keep, struck: out });
  }
  let best: Obj | null = null;
  for (const r of table) {
    if (!alive.includes(r.pattern)) continue;
    if (
      best === null ||
      r.success > best.success ||
      (r.success === best.success && r.cost_per_success < best.cost_per_success)
    )
      best = r;
  }
  return { steps, alive, pick: best ? best.pattern : null };
}
