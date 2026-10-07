/** Seeded sweeps: the TS twin of `agent_loop_sim/sweeps.py`. */
import { scenario, type Obj } from "./data";
import { merge, run } from "./harness";
import type { Tokenizer } from "./tokenizer";
import type { Ev } from "./harness";

/** Did the run finish *and* did its last test run pass? (see sweeps.py) */
export function verified(events: Ev[]): boolean {
  const subject: Record<string, string> = {};
  let lastOk = false;
  for (const e of events) {
    if (e.type === "tool_call") subject[e.call] = e.subject;
    else if (e.type === "tool_result" && e.name === "run_shell" && subject[e.call]!.startsWith("pytest"))
      lastOk = e.ok && (e.text as string).startsWith("exit code: 0");
  }
  return events[events.length - 1]!.status === "done" && lastOk;
}

/** For each retry budget: the runs' statuses (one per seed), the share that finished, and
 * the mean turns, model calls, retries, simulated time and cost. */
export function retrySweep(name: string, budgets: number[], seeds: number[], policy: Obj | null, tokenizer: Tokenizer): Obj[] {
  const rows: Obj[] = [];
  for (const b of budgets) {
    const pol = merge(policy ?? {}, { recovery: merge((policy ?? {}).recovery ?? {}, { retries: b }) });
    const statuses: string[] = [];
    let done = 0;
    let good = 0;
    let turns = 0;
    let calls = 0;
    let retries = 0;
    let elapsed = 0;
    let cost = 0;
    for (const s of seeds) {
      const ev = run(scenario(name), pol, tokenizer, null, s);
      const end = ev[ev.length - 1]!;
      statuses.push(end.status);
      if (end.status === "done") done += 1;
      if (verified(ev)) good += 1;
      for (const e of ev) if (e.type === "model_call" && e.agent === "main" && e.purpose === "act") turns += 1;
      calls += end.totals.model_calls;
      retries += end.totals.retries;
      elapsed = elapsed + end.elapsed;
      cost = cost + end.totals.cost;
    }
    const n = seeds.length;
    rows.push({
      retries: b, runs: n, done, success: done / n, verified: good, verified_rate: good / n, statuses,
      mean_turns: turns / n, mean_calls: calls / n, mean_retries: retries / n,
      mean_elapsed: elapsed / n, mean_cost: cost / n,
    });
  }
  return rows;
}
