/**
 * A discrete-event simulation of many workflows sharing one rate-limited model endpoint (engine
 * 1.7.0). A port of agent_loop_sim/orchestration/des.py, statement for statement: the same events
 * in the same order, the same random draws, the same floating-point operations.
 */
import { callLatency } from "../accounting";
import { ln } from "../context/mathx";
import { LATENCY, PRICES, type Obj } from "../data";
import { Rng } from "../rng";
import { cachedTokens, costOf, maj3 } from "./patterns";

export const DEFAULTS: Obj = {
  workflows: 50,
  arrival: "poisson",
  per_min: 6.0,
  rpm: 50,
  itpm: 40000,
  otpm: 8000,
  concurrency: 16,
  fail: 0.03,
  timeout_ms: 60000,
  max_attempts: 3,
  backoff_ms: 1000,
  backoff_cap_ms: 30000,
  jitter: [0.8, 1.5],
  price: "claude-sonnet-4.6",
  profile: "hosted",
  seed: 1,
  semantic: true,
  trace: false,
  samples: 0,
};
const EPS = 1e-6;

export function params(p: Obj | null): Obj {
  return { ...DEFAULTS, ...(p ?? {}) };
}

function pow2(n: number): number {
  let v = 1;
  for (let i = 0; i < n; i++) v = v * 2;
  return v;
}

/** Nearest-rank percentile (q in percent); 0 for an empty list. */
export function pct(xs: number[], q: number): number {
  if (!xs.length) return 0.0;
  const s = [...xs].sort((a, b) => a - b);
  let i = Math.floor((q * s.length + 99) / 100) - 1;
  if (i < 0) i = 0;
  return s[i]!;
}

type Ev = [number, number, string, number, number, string];

/** A binary min-heap on (time, sequence number); the sequence numbers are unique. */
class Heap {
  a: Ev[] = [];
  private less(x: Ev, y: Ev): boolean {
    return x[0] < y[0] || (x[0] === y[0] && x[1] < y[1]);
  }
  push(e: Ev): void {
    const a = this.a;
    a.push(e);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(a[i]!, a[p]!)) break;
      [a[i], a[p]] = [a[p]!, a[i]!];
      i = p;
    }
  }
  pop(): Ev {
    const a = this.a;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && this.less(a[l]!, a[m]!)) m = l;
        if (r < a.length && this.less(a[r]!, a[m]!)) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m]!, a[i]!];
        i = m;
      }
    }
    return top;
  }
  get size(): number {
    return this.a.length;
  }
}

export function simulate(plan: Obj, p: Obj | null = null): Obj {
  const P = params(p);
  const price = PRICES[P.price]!;
  const profile = LATENCY[P.profile]!;
  const calls = plan.calls as Obj[];
  const nwf = P.workflows as number;
  const rng = new Rng(P.seed);
  const succ: number[][] = calls.map(() => []);
  for (const c of calls) for (const d of c.deps as number[]) succ[d]!.push(c.id);
  const arrivals: number[] = [];
  let t = 0.0;
  for (let i = 0; i < nwf; i++) {
    if (P.arrival === "batch") arrivals.push(0.0);
    else {
      if (i > 0) {
        const u = rng.random();
        t = t + (-ln(1.0 - u) * 60000) / P.per_min;
      }
      arrivals.push(t);
    }
  }
  const wf: Obj[] = arrivals.map((a, i) => ({
    id: i,
    arrival: a,
    end: null,
    status: "running",
    cost: 0.0,
    attempts: 0,
    wrong: [] as number[],
    votes: {} as Record<string, [number, number]>,
    left: calls.map((c) => (c.deps as number[]).length),
    remaining: calls.length,
  }));
  const heap = new Heap();
  let seq = 0;
  const push = (at: number, kind: string, w: number, c: number, note = ""): void => {
    heap.push([at, seq, kind, w, c, note]);
    seq += 1;
  };
  for (let i = 0; i < nwf; i++) push(arrivals[i]!, "arrive", i, -1);
  const queue: [number, number][] = [];
  const attempts = new Map<string, number>();
  const akey = (w: number, c: number): string => w + ":" + c;
  const buckets = { rq: P.rpm as number, it: P.itpm as number, ot: P.otpm as number, last: 0.0 };
  const state = { inflight: 0, wake: -1.0, busy: 0.0 };
  const trace: Obj[] = [];
  const changes: number[][] = [];
  const counts = { insys: 0, ok: 0, bad: 0 };

  const noteChange = (at: number): void => {
    const row = [at, queue.length, state.inflight, counts.insys, counts.ok, counts.bad];
    const prev = changes[changes.length - 1];
    if (prev && prev[0] === at) changes[changes.length - 1] = row;
    else if (!prev || prev.slice(1).some((v, i) => v !== row[i + 1])) changes.push(row);
  };
  const refill = (at: number): void => {
    const dt = at - buckets.last;
    if (dt > 0) {
      const rq = buckets.rq + (dt * P.rpm) / 60000;
      buckets.rq = rq < P.rpm ? rq : P.rpm;
      const it = buckets.it + (dt * P.itpm) / 60000;
      buckets.it = it < P.itpm ? it : P.itpm;
      const ot = buckets.ot + (dt * P.otpm) / 60000;
      buckets.ot = ot < P.otpm ? ot : P.otpm;
      buckets.last = at;
    }
  };
  const needIn = (c: Obj): number => {
    const n = c.input - cachedTokens(c, price);
    return n < P.itpm ? n : P.itpm;
  };
  const ready = (at: number, w: number, cid: number): void => {
    const c = calls[cid]!;
    if (c.kind === "tool") {
      if (P.trace) trace.push({ wf: w, call: cid, attempt: 1, queued: at, start: at, end: at + c.ms, outcome: "tool" });
      push(at + c.ms, "tool", w, cid);
    } else {
      queue.push([w, cid]);
      if (P.trace)
        trace.push({
          wf: w,
          call: cid,
          attempt: (attempts.get(akey(w, cid)) ?? 0) + 1,
          queued: at,
          start: null,
          end: null,
          outcome: "queued",
        });
    }
  };
  const finishWf = (at: number, w: number, status: string): void => {
    const x = wf[w]!;
    x.end = at;
    x.status = status;
    counts.insys -= 1;
    if (status === "ok") counts.ok += 1;
    else counts.bad += 1;
  };
  const dispatch = (at: number): void => {
    refill(at);
    while (queue.length && state.inflight < P.concurrency) {
      const [w, cid] = queue[0]!;
      if (wf[w]!.status !== "running") {
        queue.shift();
        continue;
      }
      const c = calls[cid]!;
      const ni = needIn(c);
      if (buckets.rq + EPS >= 1 && buckets.it + EPS >= ni && buckets.ot + EPS >= 0) {
        queue.shift();
        buckets.rq = buckets.rq - 1;
        buckets.it = buckets.it - ni;
        state.inflight += 1;
        const a = (attempts.get(akey(w, cid)) ?? 0) + 1;
        attempts.set(akey(w, cid), a);
        wf[w]!.attempts += 1;
        const u = rng.random();
        const v = rng.random();
        const jit = P.jitter[0] + (P.jitter[1] - P.jitter[0]) * u;
        let [ttft, dur] = callLatency(profile, c.input, cachedTokens(c, price), c.output);
        ttft = ttft * jit;
        dur = dur * jit;
        let outcome: string;
        let end: number;
        if (v < P.fail) [outcome, end] = ["error", at + ttft];
        else if (dur > P.timeout_ms) [outcome, end] = ["timeout", at + P.timeout_ms];
        else [outcome, end] = ["ok", at + dur];
        state.busy = state.busy + (end - at);
        if (P.trace) {
          for (let j = trace.length - 1; j >= 0; j--) {
            const r = trace[j]!;
            if (r.wf === w && r.call === cid && r.outcome === "queued") {
              r.start = at;
              r.end = end;
              r.outcome = outcome;
              r.attempt = a;
              break;
            }
          }
        }
        push(end, "done", w, cid, outcome);
      } else {
        let wait = 0.0;
        if (buckets.rq + EPS < 1) {
          const x = (1 - buckets.rq) / (P.rpm / 60000);
          wait = x > wait ? x : wait;
        }
        if (buckets.it + EPS < ni) {
          const x = (ni - buckets.it) / (P.itpm / 60000);
          wait = x > wait ? x : wait;
        }
        if (buckets.ot + EPS < 0) {
          const x = (0 - buckets.ot) / (P.otpm / 60000);
          wait = x > wait ? x : wait;
        }
        if (wait < 0.001) wait = 0.001;
        if (state.wake < 0 || state.wake > at + wait) {
          state.wake = at + wait;
          push(at + wait, "wake", -1, -1);
        }
        break;
      }
    }
  };

  while (heap.size) {
    const [at, , kind, w, cid, note] = heap.pop();
    if (kind === "arrive") {
      counts.insys += 1;
      for (const c of calls) if (!(c.deps as number[]).length) ready(at, w, c.id);
    } else if (kind === "wake") {
      if (state.wake === at) state.wake = -1.0;
    } else if (kind === "retry") {
      if (wf[w]!.status === "running") ready(at, w, cid);
    } else {
      const c = calls[cid]!;
      const x = wf[w]!;
      let ok = true;
      if (kind === "done") {
        state.inflight -= 1;
        refill(at);
        if (note !== "error") {
          buckets.ot = buckets.ot - c.output;
          x.cost = x.cost + costOf(c, price);
        }
        ok = note === "ok";
      }
      if (x.status === "running") {
        if (ok) {
          if (kind === "done" && P.semantic) {
            const right = rng.random() < c.p;
            if (c.role === "critical" && !right) x.wrong.push(cid);
            if (c.role === "vote") {
              const v = (x.votes[c.group] ??= [0, 0]);
              v[0] += right ? 1 : 0;
              v[1] += 1;
            }
          }
          x.remaining -= 1;
          for (const s of succ[cid]!) {
            x.left[s] -= 1;
            if (x.left[s] === 0) ready(at, w, s);
          }
          if (x.remaining === 0) {
            let good = !x.wrong.length;
            for (const v of Object.values(x.votes) as [number, number][]) if (v[0] * 2 <= v[1]) good = false;
            finishWf(at, w, good ? "ok" : "wrong");
          }
        } else {
          const a = attempts.get(akey(w, cid))!;
          if (a < P.max_attempts) {
            let cap = P.backoff_ms * pow2(a - 1);
            if (cap > P.backoff_cap_ms) cap = P.backoff_cap_ms;
            push(at + rng.random() * cap, "retry", w, cid);
          } else finishWf(at, w, "exhausted");
        }
      }
    }
    dispatch(at);
    noteChange(at);
  }

  const done = wf.filter((x) => x.end !== null);
  const latAll = done.filter((x) => x.status !== "exhausted").map((x) => x.end - x.arrival);
  const latOk = done.filter((x) => x.status === "ok").map((x) => x.end - x.arrival);
  const nOk = wf.filter((x) => x.status === "ok").length;
  let makespan = 0.0;
  for (const x of done) if (x.end > makespan) makespan = x.end;
  let cost = 0.0;
  for (const x of wf) cost = cost + x.cost;
  let totAtt = 0;
  for (const x of wf) totAtt += x.attempts;
  const modelCalls = calls.filter((c) => c.kind !== "tool").length;
  let meanLat = 0.0;
  for (const v of latAll) meanLat = meanLat + v;
  meanLat = latAll.length ? meanLat / latAll.length : 0.0;
  const out: Obj = {
    pattern: plan.pattern,
    params: P,
    workflows: nwf,
    ok: nOk,
    wrong: wf.filter((x) => x.status === "wrong").length,
    exhausted: wf.filter((x) => x.status === "exhausted").length,
    success: nwf ? nOk / nwf : 0.0,
    p50: pct(latAll, 50),
    p95: pct(latAll, 95),
    p99: pct(latAll, 99),
    mean: meanLat,
    p50_ok: pct(latOk, 50),
    p95_ok: pct(latOk, 95),
    cost,
    cost_per_success: nOk ? cost / nOk : 0.0,
    makespan,
    throughput_per_min: makespan > 0 ? nOk / (makespan / 60000) : 0.0,
    attempts: totAtt,
    model_calls: modelCalls,
    utilisation: makespan > 0 ? state.busy / (P.concurrency * makespan) : 0.0,
    runs: wf.map((x) => ({ id: x.id, arrival: x.arrival, end: x.end, status: x.status, cost: x.cost, attempts: x.attempts })),
  };
  if (P.trace) {
    out.trace = trace;
    out.changes = changes;
  }
  if (P.samples) out.series = resample(changes, makespan, P.samples);
  return out;
}

export function resample(changes: number[][], horizon: number, n: number): Obj[] {
  const out: Obj[] = [];
  let j = 0;
  for (let i = 0; i <= n; i++) {
    const t = (horizon * i) / n;
    while (j + 1 < changes.length && changes[j + 1]![0]! <= t) j += 1;
    const r = changes.length && changes[j]![0]! <= t ? changes[j]! : [t, 0, 0, 0, 0, 0];
    out.push({ t, queue: r[1], inflight: r[2], insys: r[3], ok: r[4], bad: r[5] });
  }
  return out;
}

export function successClosed(plan: Obj, fail: number, attempts: number): number {
  let fa = 1.0;
  for (let i = 0; i < attempts; i++) fa = fa * fail;
  let through = 1.0;
  let crit = 1.0;
  const groups: Record<string, number[]> = {};
  for (const c of plan.calls as Obj[]) {
    if (c.kind === "tool") continue;
    through = through * (1 - fa);
    if (c.role === "critical") crit = crit * c.p;
    else if (c.role === "vote") (groups[c.group] ??= []).push(c.p);
  }
  for (const g of Object.values(groups)) crit = crit * maj3(g[0]!, g[1]!, g[2]!);
  return through * crit;
}

export function expectedAttempts(fail: number, attempts: number): number {
  let s = 0.0;
  let f = 1.0;
  for (let i = 0; i < attempts; i++) {
    s = s + f;
    f = f * fail;
  }
  return s;
}
