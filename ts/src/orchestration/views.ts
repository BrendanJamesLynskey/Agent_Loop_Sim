/**
 * Frames for the site's animations. A port of agent_loop_sim/orchestration/views.py.
 */
import type { Obj } from "../data";
import { END, START, channel, clone, reduceValue, runSession } from "./graph";

export const CKPT_MS = 15;

export function names(xs: string[]): string {
  if (!xs.length) return "nothing";
  if (xs.length === 1) return xs[0]!;
  return xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1];
}

export function layout(spec: Obj): Obj {
  const order = (spec.nodes as Obj[]).map((n) => n.name as string);
  const succ = new Map<string, string[]>([[START, []]]);
  for (const n of order) succ.set(n, []);
  for (const e of spec.edges as Obj[]) {
    const srcs: string[] = Array.isArray(e.from) ? e.from : [e.from];
    const tos: string[] = "router" in e ? e.targets : [e.to];
    for (const s of srcs) for (const t of tos) if (t !== END && !succ.get(s)!.includes(t)) succ.get(s)!.push(t);
  }
  const col = new Map<string, number>([[START, 0]]);
  const visiting: string[] = [];
  const dfs = (u: string, depth: number): void => {
    if (visiting.includes(u)) return;
    if (col.has(u) && col.get(u)! >= depth && u !== START) return;
    col.set(u, depth);
    visiting.push(u);
    for (const v of succ.get(u)!) if (!visiting.includes(v)) dfs(v, depth + 1);
    visiting.pop();
  };
  dfs(START, 0);
  for (const n of order) if (!col.has(n)) col.set(n, 1);
  const endCol = Math.max(...col.values()) + 1;
  const cols = new Map<number, string[]>();
  for (const n of [START, ...order]) {
    const c = col.get(n)!;
    if (!cols.has(c)) cols.set(c, []);
    cols.get(c)!.push(n);
  }
  const pos: Obj = {};
  for (const [c, ns] of cols) ns.forEach((n, i) => (pos[n] = { col: c, row: i, rows: ns.length }));
  pos[END] = { col: endCol, row: 0, rows: 1 };
  const edges: Obj[] = [];
  for (const e of spec.edges as Obj[]) {
    const srcs: string[] = Array.isArray(e.from) ? e.from : [e.from];
    const tos: string[] = "router" in e ? e.targets : [e.to];
    const kind = "router" in e ? "conditional" : Array.isArray(e.from) ? "join" : "edge";
    for (const s of srcs) for (const t of tos) edges.push({ from: s, to: t, kind, back: pos[t].col <= pos[s].col });
  }
  return { nodes: pos, cols: endCol + 1, edges };
}

export function folds(spec: Obj, before: Obj, writes: [string, Obj][]): Obj {
  const out: Obj = {};
  for (const [name, upd] of writes) {
    for (const [k, v] of Object.entries(upd)) {
      const chan = channel(spec, k);
      if (!(k in out)) out[k] = [];
      const seq = out[k] as Obj[];
      if (seq.length && chan.reducer === "overwrite") {
        seq.push({ node: name, write: clone(v), value: null, error: true });
        continue;
      }
      const cur = seq.length ? seq[seq.length - 1]!.value : (before[k] ?? null);
      seq.push({ node: name, write: clone(v), value: reduceValue(chan, cur, true, v), error: false });
    }
  }
  return out;
}

export function fmt(v: unknown): string {
  if (typeof v === "string") return '"' + v + '"';
  if (typeof v === "boolean") return v ? "true" : "false";
  if (v === null || v === undefined) return "null";
  if (Array.isArray(v)) return "[" + v.map((x) => fmt(x)).join(", ") + "]";
  if (typeof v === "object")
    return "{" + Object.entries(v as Obj).map(([k, x]) => k + ": " + fmt(x)).join(", ") + "}";
  return String(v);
}

export function superstepFrames(spec: Obj, result: Obj, history: Obj[], t0 = 0): Obj {
  const ms: Record<string, number> = {};
  for (const n of spec.nodes as Obj[]) ms[n.name] = n.ms;
  ms[START] = 0;
  const frames: Obj[] = [];
  let t = t0;
  let seqTotal = 0;
  const steps = [...(result.steps as Obj[])];
  for (const st of steps) {
    const s = st.step as number;
    const tasks = st.tasks as string[];
    const before = history[st.parent]!.values;
    const after = history[st.checkpoint]!;
    const dur = Math.max(...tasks.map((x) => ms[x]!));
    for (const x of tasks) seqTotal += ms[x]!;
    let cap: string;
    if (tasks.length === 1 && tasks[0] === START)
      cap = "Super-step 0: __start__ writes the input " + fmt(st.writes[0][1]) + " through the reducers.";
    else if (tasks.length === 1) cap = `Super-step ${s}: ${tasks[0]} runs (${ms[tasks[0]!]} ms), reading the snapshot.`;
    else
      cap =
        `Super-step ${s}: ${names(tasks)} run in parallel, all reading the same snapshot; ` +
        `the step lasts as long as the slowest (${dur} ms).`;
    frames.push({ phase: "run", step: s, active: tasks, values: before, t, dur, caption: cap });
    const order = (st.writes as [string, Obj][]).map((w) => w[0]);
    if (order.length > 1)
      cap =
        "Barrier: no write is visible until every task finishes; they are applied in task order (" +
        order.join(" then ") +
        "), not finishing order.";
    else cap = `Barrier: ${order[0]}'s write ` + fmt(st.writes[0][1]) + " waits for the end of the step.";
    frames.push({ phase: "barrier", step: s, active: tasks, values: before, writes: st.writes, t: t + dur, dur, caption: cap });
    t = t + dur + CKPT_MS;
    const fl = folds(spec, before, st.writes);
    const went = (st.fired as Obj[]).filter((f) => f.kind !== "join" && f.to !== END).map((f) => f.to as string);
    const waits = (st.fired as Obj[]).filter((f) => f.kind === "join" && f.waiting.length);
    const parts = [`Checkpoint ${after.id} saved (step ${s}).`];
    if ((after.next as string[]).length) {
      const waitTo = waits.map((w) => w.to as string);
      const ready = (after.next as string[]).filter((n) => !waitTo.includes(n) || went.includes(n));
      if (ready.length) parts.push("Next: " + names(ready) + ".");
      for (const w of waits) parts.push(`The join into ${w.to} still waits for ${names(w.waiting)}.`);
    } else parts.push("No node is triggered: the run is done.");
    frames.push({
      phase: "apply",
      step: s,
      active: tasks,
      values: after.values,
      writes: st.writes,
      folds: fl,
      fired: st.fired,
      next: after.next,
      checkpoint: after.id,
      t,
      dur,
      caption: parts.join(" "),
    });
  }
  const h = result.halt as Obj | null;
  if (h !== null && h !== undefined) {
    const before = history[h.parent]!.values;
    const wn = (h.writes as [string, Obj][]).map((w) => w[0]);
    let cap: string;
    if (h.failed.length)
      cap =
        `Super-step ${h.step}: ${names(h.failed)} raised. No checkpoint is written; ` +
        (h.writes.length ? "the writes of " + names(wn) + " are kept as pending writes." : "nothing is kept.");
    else if (h.interrupts.length)
      cap =
        `Super-step ${h.step}: an interrupt pauses the run with ` +
        fmt(h.interrupts[0]) +
        ". " +
        (h.writes.length ? "Pending writes kept: " + names(wn) + "." : "");
    else {
      const fl = folds(spec, before, h.writes);
      const bad = Object.entries(fl)
        .filter(([, seq]) => (seq as Obj[]).some((x) => x.error))
        .map(([k]) => k);
      cap =
        `Super-step ${h.step}: ${names(wn)} both wrote '${bad[0]}', which has no reducer: ` +
        "InvalidUpdateError. No checkpoint is saved.";
    }
    frames.push({
      phase: "halt",
      step: h.step,
      active: h.tasks,
      values: before,
      writes: h.writes,
      folds: !h.failed.length && !h.interrupts.length ? folds(spec, before, h.writes) : {},
      t,
      dur: 0,
      caption: cap.trim(),
      status: result.status,
    });
  }
  const graphMs = t - t0;
  let runs = 0;
  for (const st of steps) runs += st.tasks.length;
  return { frames, graph_ms: graphMs, sequential_ms: seqTotal + CKPT_MS * runs, steps: steps.length, status: result.status };
}

function opCaption(op: Obj, histBefore: Obj[]): string {
  if ("invoke" in op) {
    let cap = "invoke(" + fmt(op.invoke) + ") on a new thread";
    if (op.fail && op.fail.length) cap += ` (worker running ${names(op.fail)} will crash)`;
    return cap + ".";
  }
  if ("resume" in op)
    return "invoke(Command(resume=" + fmt(op.resume) + ")): the human answers; the interrupted node runs again from its first line.";
  if ("continue" in op) {
    const tail = op.fail && op.fail.length ? ` (${names(op.fail)} will crash again)` : "";
    return "invoke(None): resume the thread from its latest checkpoint" + tail + ".";
  }
  if ("replay" in op) {
    const c = histBefore[op.replay]!;
    return `invoke(None, checkpoint ${op.replay}): time travel to step ${c.step} and run again from there.`;
  }
  return (
    `update_state(checkpoint ${op.at}, ` +
    fmt(op.update) +
    `, as_node=${op.as_node}): ` +
    "edit the state as if that node had written it."
  );
}

const SOURCE_TEXT: Record<string, string> = {
  input: "the input",
  loop: "after a super-step",
  update: "an edit (update_state)",
  fork: "a copy of an earlier checkpoint, so the old branch is kept",
};

function cpCaption(c: Obj): string {
  const nxt = c.next.length ? "next: " + names(c.next) : "nothing next";
  return `Checkpoint ${c.id} (step ${c.step}, ${c.source}: ${SOURCE_TEXT[c.source]}); ${nxt}.`;
}

function outcomeCaption(r: Obj, hist: Obj[]): string {
  const st = r.status as string;
  const eff = r.effects.length ? " Side effects this call: " + names(r.effects) + "." : "";
  if (st === "done") return "Done: the run reached END." + eff;
  if (st === "updated") return "No node ran; a new checkpoint holds the edit.";
  if (st === "interrupted") {
    const kept = (r.halt.writes as [string, Obj][]).map((w) => w[0]);
    return (
      "Paused by interrupt(" +
      fmt(r.interrupts[0]) +
      "). No new checkpoint; " +
      (kept.length
        ? "pending writes of " + names(kept) + " are saved on checkpoint " + String(r.checkpoint) + "."
        : "the thread waits on checkpoint " + String(r.checkpoint) + ".") +
      eff
    );
  }
  if (st === "interrupt_before" || st === "interrupt_after") {
    const where =
      st === "interrupt_before"
        ? "before " + names(hist[r.checkpoint]!.next)
        : "after " + names(r.steps.length ? r.steps[r.steps.length - 1].tasks : []);
    return `Paused by a static interrupt (${where}) at checkpoint ${r.checkpoint}; ` + "a human can inspect or edit the state." + eff;
  }
  if (st === "error") {
    const h = r.halt as Obj | null;
    if (h !== null && h.failed.length) {
      const kept = (h.writes as [string, Obj][]).map((w) => w[0]);
      return (
        `Crash: ${names(h.failed)} raised. Checkpoint ${r.checkpoint} stays the latest; ` +
        (kept.length ? "the finished work of " + names(kept) + " is saved there as pending writes." : "nothing else finished.") +
        eff
      );
    }
    return "Error: " + (r.error ?? "") + eff;
  }
  return `Stopped: recursion limit reached at checkpoint ${r.checkpoint}.` + eff;
}

export function timelineFrames(spec: Obj, sess: Obj): Obj {
  const res = runSession(spec, sess.ops);
  const frames: Obj[] = [];
  let prev: Obj[] = [];
  const ops: Obj[] = [];
  (res.ops as Obj[]).forEach((o, i) => {
    const hist = o.history as Obj[];
    ops.push({
      op: o.op,
      status: o.result.status,
      history: hist,
      effects: o.result.effects,
      calls: o.result.calls,
      interrupts: o.result.interrupts,
    });
    frames.push({ kind: "op", h: i, n: prev.length, focus: null, caption: opCaption(o.op, prev) });
    for (const c of hist.slice(prev.length))
      frames.push({ kind: "checkpoint", h: i, n: c.id + 1, focus: c.id, caption: cpCaption(c) });
    frames.push({
      kind: "outcome",
      h: i,
      n: hist.length,
      focus: o.result.checkpoint,
      status: o.result.status,
      caption: outcomeCaption(o.result, hist),
    });
    prev = hist;
  });
  return { session: sess.name, graph: spec.name, ops, frames };
}

export function sessionSupersteps(spec: Obj, sess: Obj): Obj[] {
  const res = runSession(spec, sess.ops);
  return (res.ops as Obj[]).map((o) => superstepFrames(spec, o.result, o.history));
}
