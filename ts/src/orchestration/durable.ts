/**
 * Durable execution: an event history, deterministic replay, non-determinism, idempotency keys.
 * A port of agent_loop_sim/orchestration/durable.py, statement for statement.
 */
import type { Obj } from "../data";
import { Rng } from "../rng";
import data from "../orchestration_data.json";

export const WORKFLOW: Obj = data.workflow as Obj;
export const TIMEOUT_MS: number = data.consts.timeout_ms;
export const RESTART_MS: number = data.consts.restart_ms;
export const REPLAY_MS: number = data.consts.replay_ms;
export const MAX_ATTEMPTS: number = data.consts.max_attempts;
export const CHOOSE_MODES = ["clock", "marker"];
export const DURABLE_SCENARIOS: Obj[] = data.durable_scenarios as Obj[];

class Crash extends Error {}

function activityName(step: Obj, choice: string | null): string {
  if (step.name === "ship") return "ship_" + (choice || "?");
  return step.name;
}

export function runDurable(params: Obj | null = null): Obj {
  const p: Obj = { ...(params ?? {}) };
  const crash: Obj | undefined = p.crash;
  const lost: Obj = p.lost ?? {};
  const keyOn = Boolean(p.idempotency ?? false);
  const mode: string = p.choose ?? "clock";
  if (!CHOOSE_MODES.includes(mode)) throw new Error(`unknown choose mode ${mode}`);
  const wf = WORKFLOW;
  const history: Obj[] = [];
  const ledger: Obj[] = [];
  const emails: Obj[] = [];
  const frames: Obj[] = [];
  const st: Obj = { t: 0, worker: 1, crashed: false };
  const attemptsBySeq = new Map<number, number>();

  const frame = (kind: string, caption: string, mode_: string, line: number, cursor: number, extra: Obj | null = null): void => {
    const f: Obj = {
      t: st.t,
      worker: st.worker,
      mode: mode_,
      line,
      kind,
      hlen: history.length,
      cursor,
      charges: ledger.length,
      emails: emails.length,
      caption,
    };
    if (extra) Object.assign(f, extra);
    frames.push(f);
  };

  const append = (ev0: Obj, line: number, caption: string): void => {
    const ev = { ...ev0, i: history.length, t: st.t };
    history.push(ev);
    frame("event", caption, "live", line, history.length, { event: ev.i });
    if (crash && "after_event" in crash && st.worker === 1 && ev.i === crash.after_event) throw new Crash();
  };

  const effect = (step: Obj, seq: number, attempt: number): string => {
    const key = keyOn ? wf.id + "/" + String(seq) : null;
    if (step.effect === "charge") {
      if (key !== null) for (const c of ledger) if (c.key === key) return "dedup";
      ledger.push({ key, amount: step.amount, t: st.t, attempt, worker: st.worker });
      return "charged";
    }
    if (step.effect === "email") {
      if (key !== null) for (const e of emails) if (e.key === key) return "dedup";
      emails.push({ key, t: st.t, worker: st.worker });
      return "sent";
    }
    return "done";
  };

  const runActivity = (step: Obj, name: string, seq: number, line: number, liveStart: boolean): string => {
    if (liveStart)
      append({ type: "activity_scheduled", seq, activity: name }, line, `Schedule ${name} (command ${seq}); the history records it before anything runs.`);
    for (;;) {
      const attempt = (attemptsBySeq.get(seq) ?? 0) + 1;
      attemptsBySeq.set(seq, attempt);
      st.t += step.ms;
      const out = effect(step, seq, attempt);
      const key = keyOn ? wf.id + "/" + String(seq) : null;
      let cap: string;
      if (out === "charged")
        cap = `${name} attempt ${attempt}: the payment service charges £${step.amount}` + (key ? ` (key ${key}).` : " (no idempotency key).");
      else if (out === "dedup")
        cap = `${name} attempt ${attempt}: the service has already seen key ${key}, so it returns the first result and charges nothing.`;
      else if (out === "sent") cap = `${name} attempt ${attempt}: the email goes out.`;
      else cap = `${name} attempt ${attempt} runs.`;
      frame("effect", cap, "live", line, history.length, { effect: out, seq, attempt });
      if (crash && crash.in_activity === name && st.worker === 1) throw new Crash();
      if (((lost[name] ?? []) as number[]).includes(attempt)) {
        st.t += TIMEOUT_MS;
        if (attempt >= MAX_ATTEMPTS) {
          append({ type: "activity_failed", seq, attempt }, line, `${name}: no reply to attempt ${attempt}, the last allowed; the activity fails.`);
          return "failed";
        }
        append(
          { type: "activity_timed_out", seq, attempt },
          line,
          `${name}: the reply to attempt ${attempt} is lost; after ${TIMEOUT_MS} ms the worker retries.`,
        );
        continue;
      }
      append({ type: "activity_completed", seq, result: out }, line, `${name} completed (${out}); the result is in the history now.`);
      return out;
    }
  };

  const nondet = (ev: Obj, wanted: string, line: number, cursor: number): string => {
    const got = ev.type + ("activity" in ev ? " " + ev.activity : "") + (ev.type === "marker" ? " " + ev.name : "");
    frame(
      "nondeterminism",
      `Non-determinism: the code now says '${wanted}' but the history has '${got}' at event ${cursor}. ` +
        "Replay stops; the workflow is stuck until the code is fixed.",
      "replay",
      line,
      cursor,
      { wanted, got },
    );
    st.nondet = { event: cursor, wanted, got };
    return "nondeterminism";
  };

  const worker = (): string => {
    let cursor = 1;
    let choice: string | null = null;
    let seq = 0;
    const steps = wf.steps as Obj[];
    for (let line = 0; line < steps.length; line++) {
      const step = steps[line]!;
      if (step.kind === "choose") {
        if (mode === "marker") {
          if (cursor < history.length) {
            const ev = history[cursor]!;
            if (ev.type !== "marker" || ev.name !== step.name) return nondet(ev, `record marker ${step.name}`, line, cursor);
            choice = ev.value;
            cursor += 1;
            frame("replay", `Replay: the ${step.name} choice comes from the history (marker: ${choice}), not the clock.`, "replay", line, cursor);
            continue;
          }
          choice = st.t < step.cutoff_ms ? step.early : step.late;
          append({ type: "marker", name: step.name, value: choice }, line, `The clock reads ${st.t} ms, so ${step.name} = ${choice}; recorded as a marker.`);
          cursor = history.length;
          continue;
        }
        choice = st.t < step.cutoff_ms ? step.early : step.late;
        frame(
          "code",
          `The code reads the clock (${st.t} ms) and picks ${choice} shipping; nothing is recorded.`,
          cursor < history.length ? "replay" : "live",
          line,
          cursor,
        );
        continue;
      }
      const name = activityName(step, choice);
      let out: string;
      if (cursor < history.length) {
        const ev = history[cursor]!;
        if (ev.type !== "activity_scheduled" || ev.activity !== name) return nondet(ev, `schedule ${name}`, line, cursor);
        cursor += 1;
        let result: string | null = null;
        while (cursor < history.length && history[cursor]!.seq === seq) {
          if (history[cursor]!.type === "activity_completed") result = history[cursor]!.result;
          if (history[cursor]!.type === "activity_failed") result = "failed";
          cursor += 1;
        }
        if (result !== null) {
          st.t += REPLAY_MS;
          frame("replay", `Replay: ${name} is in the history (${result}); its recorded result is returned and nothing runs.`, "replay", line, cursor);
          if (result === "failed") return "failed";
          seq += 1;
          continue;
        }
        frame("replay", `Replay: ${name} was scheduled but never completed, so it runs again (at least once).`, "replay", line, cursor);
        out = runActivity(step, name, seq, line, false);
        cursor = history.length;
      } else {
        out = runActivity(step, name, seq, line, true);
        cursor = history.length;
      }
      if (out === "failed") return "failed";
      seq += 1;
    }
    append({ type: "workflow_completed" }, steps.length, "The workflow completed.");
    return "completed";
  };

  let status = "completed";
  try {
    append({ type: "workflow_started", workflow: wf.id }, -1, `Workflow ${wf.id} starts on worker 1; event 0 is written.`);
    status = worker();
  } catch (e) {
    if (!(e instanceof Crash)) throw e;
    st.crashed = true;
    frame("crash", `Worker 1 dies at ${st.t} ms. The history (${history.length} events) survives in the store.`, "live", -1, history.length);
    st.t += RESTART_MS;
    st.worker = 2;
    frame("restart", `Worker 2 picks the workflow up at ${st.t} ms and replays the code from the top.`, "replay", -1, 1);
    status = worker();
  }
  let charged = 0;
  for (const c of ledger) charged += c.amount;
  return {
    params: p,
    status,
    history,
    ledger,
    emails,
    charges: ledger.length,
    charged,
    frames,
    crashed: st.crashed,
    nondet: st.nondet ?? null,
    t_end: st.t,
  };
}

export function chargeSweep(q: number, attempts: number, runs: number, seed: number): Obj {
  const r = new Rng(seed);
  let totalNo = 0;
  let totalKey = 0;
  let dup = 0;
  let failed = 0;
  const hist: number[] = new Array(attempts + 1).fill(0);
  for (let i = 0; i < runs; i++) {
    let n = 0;
    let ok = false;
    while (n < attempts) {
      n += 1;
      if (r.random() >= q) {
        ok = true;
        break;
      }
    }
    if (!ok) failed += 1;
    totalNo += n;
    totalKey += 1;
    hist[n]! += 1;
    if (n >= 2) dup += 1;
  }
  const meanNo = totalNo / runs;
  let qa = 1.0;
  for (let i = 0; i < attempts; i++) qa = qa * q;
  const expected = (1 - qa) / (1 - q);
  return {
    q,
    attempts,
    runs,
    seed,
    mean_no_key: meanNo,
    mean_key: totalKey / runs,
    dup_rate: dup / runs,
    failed,
    hist,
    expected_no_key: expected,
    expected_dup: attempts >= 2 ? q : 0.0,
    expected_failed: qa,
  };
}

export function durableScenario(name: string): Obj {
  for (const s of DURABLE_SCENARIOS) if (s.name === name) return JSON.parse(JSON.stringify(s)) as Obj;
  throw new Error(`no durable scenario '${name}'`);
}
