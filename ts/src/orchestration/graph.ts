/**
 * The graph runtime with LangGraph's execution model. A port of
 * agent_loop_sim/orchestration/graph.py, statement for statement; see that file for the semantics
 * (checked against LangGraph 1.2.14 in the engine's CI).
 */
import type { Obj } from "../data";

export const START = "__start__";
export const END = "__end__";
export const REDUCERS = ["overwrite", "add", "max"];
export const SOURCES = ["input", "loop", "update", "fork"];

export class GraphError extends Error {}
export class InvalidUpdateError extends Error {}
export class NodeFailure extends Error {}
class Interrupt extends Error {
  constructor(public payload: unknown) {
    super("interrupt");
  }
}

export function clone<T>(x: T): T {
  return x === undefined ? x : (JSON.parse(JSON.stringify(x)) as T);
}

// ---------------------------------------------------------------------------------------------
// The spec

export function validate(spec: Obj): void {
  const keys = (spec.state as Obj[]).map((s) => s.key as string);
  if (new Set(keys).size !== keys.length) throw new GraphError("duplicate state key");
  for (const s of spec.state as Obj[]) if (!REDUCERS.includes(s.reducer)) throw new GraphError(`unknown reducer ${s.reducer}`);
  const names = (spec.nodes as Obj[]).map((n) => n.name as string);
  if (new Set(names).size !== names.length || names.includes(START) || names.includes(END)) throw new GraphError("bad node names");
  for (const e of spec.edges as Obj[]) {
    const srcs: string[] = Array.isArray(e.from) ? e.from : [e.from];
    for (const s of srcs) if (s !== START && !names.includes(s)) throw new GraphError(`edge from unknown node ${s}`);
    const tos: string[] = "router" in e ? e.targets : [e.to];
    for (const t of tos) if (t !== END && !names.includes(t)) throw new GraphError(`edge to unknown node ${t}`);
  }
  for (const n of [...(spec.interrupt_before ?? []), ...(spec.interrupt_after ?? [])])
    if (!names.includes(n)) throw new GraphError(`interrupt on unknown node ${n}`);
}

export function channel(spec: Obj, key: string): Obj {
  for (const s of spec.state as Obj[]) if (s.key === key) return s;
  throw new GraphError(`unknown state key ${key}`);
}

export function defaultValues(spec: Obj): Obj {
  const out: Obj = {};
  for (const s of spec.state as Obj[]) if (s.reducer !== "overwrite") out[s.key] = s.type === "list" ? [] : 0;
  return out;
}

export function ordered(spec: Obj, values: Obj): Obj {
  const out: Obj = {};
  for (const s of spec.state as Obj[]) if (s.key in values) out[s.key] = values[s.key];
  return out;
}

export function reduceValue(chan: Obj, cur: unknown, _has: boolean, w: unknown): unknown {
  const r = chan.reducer;
  if (r === "overwrite") return w;
  if (r === "add") {
    if (chan.type === "list") return [...(cur as unknown[]), ...(w as unknown[])];
    return (cur as number) + (w as number);
  }
  return (w as number) > (cur as number) ? w : cur;
}

export function applyWrites(spec: Obj, values: Obj, writes: [string, Obj][]): Obj {
  const out: Obj = { ...values };
  const seen: Record<string, number> = {};
  for (const [, upd] of writes) {
    for (const [k, v] of Object.entries(upd)) {
      const chan = channel(spec, k);
      seen[k] = (seen[k] ?? 0) + 1;
      if (chan.reducer === "overwrite" && seen[k]! > 1)
        throw new InvalidUpdateError(`At key '${k}': Can receive only one value per step.`);
      out[k] = reduceValue(chan, out[k], k in out, v);
    }
  }
  return ordered(spec, out);
}

// ---------------------------------------------------------------------------------------------
// Node behaviour

function cond(c: Obj, state: Obj): boolean {
  const a = "len" in c ? ((state[c.len] ?? []) as unknown[]).length : (state[c.key] ?? null);
  const b = c.value;
  switch (c.op) {
    case "<":
      return a < b;
    case "<=":
      return a <= b;
    case ">":
      return a > b;
    case ">=":
      return a >= b;
    case "==":
      return JSON.stringify(a) === JSON.stringify(b);
    case "!=":
      return JSON.stringify(a) !== JSON.stringify(b);
  }
  throw new GraphError(`unknown comparison ${c.op}`);
}

export function route(router: Obj, state: Obj): string[] {
  let g: unknown;
  if ("goto" in router) g = router.goto;
  else if (cond(router.if, state)) g = router.then;
  else g = router.else;
  if (g !== null && typeof g === "object" && !Array.isArray(g)) return route(g as Obj, state);
  return Array.isArray(g) ? [...(g as string[])] : [g as string];
}

export function runOps(node: Obj, state: Obj, ctx: Obj): Obj {
  const upd: Obj = {};
  for (const op of node.ops as Obj[]) {
    const k = op.op;
    if (k === "set") upd[op.key] = clone(op.value);
    else if (k === "append") upd[op.key] = [clone(op.value)];
    else if (k === "delta") upd[op.key] = op.value;
    else if (k === "inc") upd[op.key] = ((state[op.key] ?? 0) as number) + op.by;
    else if (k === "len") upd[op.key] = ((state[op.of] ?? []) as unknown[]).length;
    else if (k === "copy") upd[op.key] = clone(state[op.from] ?? null);
    else if (k === "effect") (ctx.effects as string[]).push(node.name + ":" + op.name);
    else if (k === "fail") {
      if ((ctx.fail as string[]).includes(node.name)) throw new NodeFailure(op.message ?? "node failed");
    } else if (k === "interrupt") {
      if (!("resume" in ctx)) throw new Interrupt(clone(op.payload));
      upd[op.key] = clone(ctx.resume);
    } else throw new GraphError(`unknown op ${k}`);
  }
  return upd;
}

// ---------------------------------------------------------------------------------------------
// The runtime

function nodeSpec(spec: Obj, name: string): Obj {
  for (const n of spec.nodes as Obj[]) if (n.name === name) return n;
  throw new GraphError(`unknown node ${name}`);
}

function decl(spec: Obj): string[] {
  return (spec.nodes as Obj[]).map((n) => n.name as string);
}

function joinId(e: Obj): string {
  return "join:" + (e.from as string[]).join("+") + ":" + e.to;
}

function isJoin(e: Obj): boolean {
  return !("router" in e) && Array.isArray(e.from);
}

export interface InvokeOpts {
  resume?: unknown;
  hasResume?: boolean;
  checkpoint?: number | null;
  fail?: string[] | null;
  recursionLimit?: number;
}

export class Thread {
  cps: Obj[] = [];
  latest: number | null = null;

  constructor(public spec: Obj) {
    validate(spec);
  }

  private put(parent: number | null, step: number, source: string, values: Obj, nxt: string[], barriers: Obj, ready: string[]): Obj {
    const bar: Obj = {};
    for (const [k, v] of Object.entries(barriers)) bar[k] = [...(v as string[])];
    const cp: Obj = {
      id: this.cps.length,
      parent,
      step,
      source,
      values: ordered(this.spec, values),
      next: nxt,
      tasks: nxt.map((n) => ({ name: n, error: null, interrupts: [], result: null })),
      barriers: bar,
      ready: [...ready],
    };
    this.cps.push(cp);
    this.latest = cp.id;
    return cp;
  }

  history(): Obj[] {
    return this.cps.map((c) => ({
      id: c.id,
      parent: c.parent,
      step: c.step,
      source: c.source,
      values: clone(c.values),
      next: [...c.next],
      tasks: clone(c.tasks),
    }));
  }

  triggers(values: Obj, done: [string, Obj][], barriers: Obj): [string[], string[], Obj, Obj[]] {
    const trig = new Set<string>();
    let bar: Obj = {};
    for (const [k, v] of Object.entries(barriers)) bar[k] = [...(v as string[])];
    const fired: Obj[] = [];
    for (const [name, upd] of done) {
      for (const e of this.spec.edges as Obj[]) {
        if ("router" in e) {
          if (e.from !== name) continue;
          const own = applyWrites(this.spec, values, [[name, upd]]);
          for (const t of route(e.router, own)) {
            fired.push({ from: name, to: t, kind: "conditional" });
            if (t !== END) trig.add(t);
          }
        } else if (Array.isArray(e.from)) {
          if (!e.from.includes(name)) continue;
          const jid = joinId(e);
          let seen: string[] = bar[jid] ?? [];
          if (!seen.includes(name)) seen = [...seen, name];
          bar[jid] = seen;
          fired.push({ from: name, to: e.to, kind: "join", waiting: (e.from as string[]).filter((s) => !seen.includes(s)) });
        } else if (e.from === name) {
          fired.push({ from: name, to: e.to, kind: "edge" });
          if (e.to !== END) trig.add(e.to);
        }
      }
    }
    for (const f of fired) {
      if (f.kind === "join") {
        for (const e of this.spec.edges as Obj[]) {
          if (isJoin(e) && e.to === f.to && e.from.includes(f.from))
            f.waiting = (e.from as string[]).filter((s) => !((bar[joinId(e)] ?? []) as string[]).includes(s));
        }
      }
    }
    const touched = new Set<string>();
    for (const e of this.spec.edges as Obj[]) {
      if (isJoin(e)) {
        const jid = joinId(e);
        if (done.some(([name]) => e.from.includes(name))) touched.add(e.to);
        if ((e.from as string[]).every((s) => ((bar[jid] ?? []) as string[]).includes(s))) trig.add(e.to);
      }
    }
    const nxt = decl(this.spec).filter((n) => trig.has(n) || touched.has(n));
    const ready = decl(this.spec).filter((n) => trig.has(n));
    for (const e of this.spec.edges as Obj[]) {
      if (isJoin(e)) {
        const jid = joinId(e);
        if ((e.from as string[]).every((s) => ((bar[jid] ?? []) as string[]).includes(s))) bar[jid] = [];
      }
    }
    const kept: Obj = {};
    for (const [k, v] of Object.entries(bar)) if ((v as string[]).length) kept[k] = v;
    bar = kept;
    return [nxt, ready, bar, fired];
  }

  invoke(inp: Obj | null = null, opts: InvokeOpts = {}): Obj {
    const ctxFail = [...(opts.fail ?? [])];
    const recursionLimit = opts.recursionLimit ?? 25;
    const effects: string[] = [];
    const calls: string[] = [];
    const steps: Obj[] = [];
    let cur: Obj;
    let stop: number;
    let step: number;
    let reuse: boolean;
    if (inp !== null) {
      if (this.cps.length) throw new GraphError("input on a thread that already has checkpoints is not modelled");
      cur = this.put(null, -1, "input", defaultValues(this.spec), [START], {}, [START]);
      stop = -1 + recursionLimit + 1;
      cur.tasks[0].result = clone(inp);
      const values = applyWrites(this.spec, cur.values, [[START, inp]]);
      const done: [string, Obj][] = [[START, inp]];
      const [nxt, ready, bar, fired] = this.triggers(cur.values, done, {});
      const par = cur.id;
      cur = this.put(cur.id, 0, "loop", values, nxt, bar, ready);
      steps.push({ step: 0, tasks: [START], writes: [[START, clone(inp)]], fired, parent: par, checkpoint: cur.id });
      step = 1;
      reuse = false;
    } else if (opts.checkpoint !== undefined && opts.checkpoint !== null) {
      const base = this.cps[opts.checkpoint]!;
      stop = base.step + 1 + recursionLimit + 1;
      if (base.source === "update" || base.source === "fork") cur = base;
      else cur = this.put(base.id, base.step + 1, "fork", base.values, base.next, base.barriers, base.ready);
      step = cur.step + 1;
      reuse = false;
    } else {
      if (this.latest === null) throw new GraphError("nothing to resume");
      cur = this.cps[this.latest]!;
      stop = cur.step + 1 + recursionLimit + 1;
      step = cur.step + 1;
      reuse = true;
    }
    let status = "done";
    let interrupts: unknown[] = [];
    let error: string | null = null;
    let halt: Obj | null = null;
    let first = true;
    for (;;) {
      if (step > stop) {
        status = "recursion_limit";
        break;
      }
      const tasks: string[] = [...cur.ready];
      if (!tasks.length) {
        status = "done";
        break;
      }
      const ib: string[] = this.spec.interrupt_before ?? [];
      if (ib.length && tasks.some((t) => ib.includes(t)) && !(first && inp === null)) {
        status = "interrupt_before";
        break;
      }
      const writes: [string, Obj][] = [];
      const failed: string[] = [];
      const paused: unknown[] = [];
      for (const t of tasks) {
        const rec = (cur.tasks as Obj[]).find((x) => x.name === t)!;
        if (first && reuse && rec.result !== null) {
          writes.push([t, clone(rec.result)]);
          continue;
        }
        const ctx: Obj = { fail: ctxFail, effects };
        if (first && reuse && opts.hasResume && rec.interrupts.length) ctx.resume = opts.resume;
        calls.push(t);
        let upd: Obj;
        try {
          upd = runOps(nodeSpec(this.spec, t), cur.values, ctx);
        } catch (e) {
          if (e instanceof NodeFailure) {
            rec.error = "NodeFailure('" + e.message + "')";
            failed.push(t);
            continue;
          }
          if (e instanceof Interrupt) {
            rec.interrupts = [e.payload];
            paused.push(e.payload);
            continue;
          }
          throw e;
        }
        rec.result = clone(upd);
        writes.push([t, upd]);
      }
      first = false;
      writes.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
      halt = { step, tasks, writes: writes.map(([n, u]) => [n, clone(u)]), failed, interrupts: paused, parent: cur.id };
      if (failed.length) {
        status = "error";
        error = "NodeFailure: " + (cur.tasks as Obj[]).find((x) => x.name === failed[0])!.error;
        break;
      }
      if (paused.length) {
        status = "interrupted";
        interrupts = paused;
        break;
      }
      let values: Obj;
      try {
        values = applyWrites(this.spec, cur.values, writes);
      } catch (e) {
        if (e instanceof InvalidUpdateError) {
          status = "error";
          error = "InvalidUpdateError: " + e.message;
          break;
        }
        throw e;
      }
      halt = null;
      const [nxt, ready, bar, fired] = this.triggers(cur.values, writes, cur.barriers);
      const par = cur.id;
      cur = this.put(cur.id, step, "loop", values, nxt, bar, ready);
      steps.push({ step, tasks, writes: writes.map(([n, u]) => [n, clone(u)]), fired, parent: par, checkpoint: cur.id });
      step += 1;
      const ia: string[] = this.spec.interrupt_after ?? [];
      if (ia.length && tasks.some((t) => ia.includes(t))) {
        status = "interrupt_after";
        break;
      }
    }
    return {
      status,
      values: clone(cur.values),
      next: [...cur.next],
      interrupts,
      error,
      calls,
      effects,
      checkpoint: cur.id,
      steps,
      halt,
    };
  }

  updateState(checkpoint: number, values: Obj, asNode: string): number {
    const base = this.cps[checkpoint]!;
    const nw = applyWrites(this.spec, base.values, [[asNode, values]]);
    const [nxt, ready, bar] = this.triggers(base.values, [[asNode, values]], base.barriers);
    const cp = this.put(base.id, base.step + 1, "update", nw, nxt, bar, ready);
    return cp.id;
  }
}

export function comparableHistory(spec: Obj, hist: Obj[]): Obj[] {
  const targets = new Set<string>();
  for (const e of spec.edges as Obj[]) if (isJoin(e)) targets.add(e.to);
  const parents = new Set<number | null>();
  for (const c of hist) if (c.source === "loop") parents.add(c.parent);
  return hist.map((c0) => {
    const c = clone(c0);
    if (parents.has(c.id)) {
      const drop = (c.tasks as Obj[])
        .filter((t) => targets.has(t.name) && t.result === null && t.error === null && !t.interrupts.length)
        .map((t) => t.name as string);
      c.next = (c.next as string[]).filter((n) => !drop.includes(n));
      c.tasks = (c.tasks as Obj[]).filter((t) => !drop.includes(t.name));
    }
    return c;
  });
}

export function outcome(r: Obj): Obj {
  let st = r.status;
  if (st === "interrupt_before" || st === "interrupt_after") st = "paused";
  return {
    status: st,
    interrupts: r.interrupts,
    error: r.error ? (r.error as string).split(":")[0] : null,
    calls: [...r.calls].sort(),
    effects: [...r.effects].sort(),
  };
}

export function runSession(spec: Obj, ops: Obj[]): Obj {
  const th = new Thread(spec);
  const out: Obj[] = [];
  for (const op of ops) {
    const kw: InvokeOpts = { fail: op.fail ?? null, recursionLimit: op.limit ?? 25 };
    let r: Obj;
    if ("invoke" in op) r = th.invoke(op.invoke, kw);
    else if ("resume" in op) r = th.invoke(null, { ...kw, resume: op.resume, hasResume: true });
    else if ("continue" in op) r = th.invoke(null, kw);
    else if ("replay" in op) r = th.invoke(null, { ...kw, checkpoint: op.replay });
    else if ("update" in op) {
      const cid = th.updateState(op.at, op.update, op.as_node);
      r = {
        status: "updated",
        values: clone(th.cps[cid]!.values),
        next: [...th.cps[cid]!.next],
        interrupts: [],
        error: null,
        calls: [],
        effects: [],
        checkpoint: cid,
        steps: [],
        halt: null,
      };
    } else throw new GraphError(`unknown session op ${JSON.stringify(op)}`);
    out.push({ op, result: r, history: th.history() });
  }
  return { name: spec.name, ops: out };
}
