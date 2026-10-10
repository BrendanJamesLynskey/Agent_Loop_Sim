/**
 * Reliability maths for multi-step workflows, with a seeded simulation (engine 1.7.0). A port of
 * agent_loop_sim/orchestration/reliability.py; see that file for the closed forms.
 */
import type { Obj } from "../data";
import { Rng } from "../rng";

export const Z95 = 1.96;

function pw(x: number, n: number): number {
  let v = 1.0;
  for (let i = 0; i < n; i++) v = v * x;
  return v;
}

function geo(x: number, n: number): number {
  let s = 0.0;
  let v = 1.0;
  for (let i = 0; i < n; i++) {
    s = s + v;
    v = v * x;
  }
  return s;
}

export function perAttempt(e: number, w: number, verifier: Obj | null): Record<string, number> {
  const c = 1 - e - w;
  const d = verifier ? verifier.recall : 0.0;
  const f = verifier ? verifier.false_reject : 0.0;
  return { c, r: e + c * f + w * d, a: c * (1 - f), b: w * (1 - d) };
}

export function closed(n: number, e: number, w: number, attempts: number, verifier: Obj | null = null): Record<string, number> {
  const q = perAttempt(e, w, verifier);
  const sa = geo(q.r!, attempts);
  const giveUp = pw(q.r!, attempts);
  const stepOk = q.a! * sa;
  const goesOn = 1 - giveUp;
  return {
    step_ok: stepOk,
    step_wrong: q.b! * sa,
    step_give_up: giveUp,
    attempts_per_step: sa,
    success: pw(stepOk, n),
    attempts: sa * geo(goesOn, n),
    reached_end: pw(goesOn, n),
  };
}

export function wilson(k: number, n: number, z = Z95): number[] {
  if (n === 0) return [0.0, 1.0];
  const ph = k / n;
  const z2 = z * z;
  const den = 1 + z2 / n;
  const mid = ph + z2 / (2 * n);
  const half = z * Math.sqrt((ph * (1 - ph)) / n + z2 / (4 * n * n));
  const lo = (mid - half) / den;
  const hi = (mid + half) / den;
  return [lo > 0 ? lo : 0.0, hi < 1 ? hi : 1.0];
}

export function simulate(
  n: number,
  e: number,
  w: number,
  attempts: number,
  verifier: Obj | null,
  runs: number,
  seed: number,
  keep = 0,
): Obj {
  const r = new Rng(seed);
  let ok = 0;
  let wrong = 0;
  let gaveUp = 0;
  let totalAtt = 0;
  const kept: Obj[] = [];
  for (let i = 0; i < runs; i++) {
    const steps: string[][] = [];
    let silent = false;
    let stopped = false;
    for (let s = 0; s < n; s++) {
      const log: string[] = [];
      let done = false;
      while (log.length < attempts) {
        const u = r.random();
        if (u < e) {
          log.push("error");
          continue;
        }
        const right = u >= e + w;
        if (verifier) {
          const v = r.random();
          if (right && v < verifier.false_reject) {
            log.push("rejected_ok");
            continue;
          }
          if (!right && v < verifier.recall) {
            log.push("caught");
            continue;
          }
        }
        log.push(right ? "ok" : "wrong");
        if (!right) silent = true;
        done = true;
        break;
      }
      totalAtt += log.length;
      steps.push(log);
      if (!done) {
        stopped = true;
        break;
      }
    }
    let status: string;
    if (stopped) {
      gaveUp += 1;
      status = "gave_up";
    } else if (silent) {
      wrong += 1;
      status = "wrong";
    } else {
      ok += 1;
      status = "ok";
    }
    if (i < keep) kept.push({ status, steps });
  }
  return {
    n,
    e,
    w,
    attempts,
    verifier,
    runs,
    seed,
    ok,
    wrong,
    gave_up: gaveUp,
    rate: ok / runs,
    ci: wilson(ok, runs),
    mean_attempts: totalAtt / runs,
    kept,
    closed: closed(n, e, w, attempts, verifier),
  };
}

export function curve(ns: number[], e: number, w: number, attempts: number, verifier: Obj | null, runs: number, seed: number): Obj[] {
  return ns.map((n) => {
    const s = simulate(n, e, w, attempts, verifier, runs, seed + n);
    return { n, closed: s.closed.success, rate: s.rate, ci: s.ci, attempts: s.mean_attempts, closed_attempts: s.closed.attempts };
  });
}
