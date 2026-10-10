/**
 * Multi-agent patterns as call plans (engine 1.7.0). A port of
 * agent_loop_sim/orchestration/patterns.py, statement for statement; see that file for the model
 * (every token size and accuracy is illustrative).
 */
import { callCost, callLatency } from "../accounting";
import { LATENCY, PRICES, type Obj } from "../data";

export const PATTERNS = ["single", "supervisor", "hierarchical", "swarm", "debate", "map_reduce"];
export const TITLES: Record<string, string> = {
  single: "Single agent",
  supervisor: "Supervisor",
  hierarchical: "Hierarchical",
  swarm: "Swarm (hand-offs)",
  debate: "Debate",
  map_reduce: "Map-reduce",
};

export const TASK: Obj = {
  question: "Which of four vendors should we pick?",
  subtasks: ["pricing", "latency", "licensing", "support"],
  user_tokens: 120,
  system_tokens: 900,
  tool_result_tokens: 700,
  route_out: 60,
  worker_out: 250,
  final_out: 450,
  handoff_tokens: 150,
  tool_ms: 800,
  debaters: 3,
};

export const MODEL: Obj = { p_focus: 0.98, focus_tokens: 4000, slope_per_k: 0.01, p_route: 0.995, p_floor: 0.5 };

export function callP(kind: string, input: number, model: Obj): number {
  if (kind === "route") return model.p_route;
  const over = input - model.focus_tokens;
  let p = model.p_focus;
  if (over > 0) p = p - (model.slope_per_k * over) / 1000;
  return p > model.p_floor ? p : model.p_floor;
}

class Plan {
  items: Obj[] = [];
  ctx: Record<string, number> = {};
  prev: Record<string, number> = {};
  agents: string[] = [];

  constructor(
    public task: Obj,
    public model: Obj,
  ) {}

  agent(name: string, base: number): void {
    if (!this.agents.includes(name)) this.agents.push(name);
    this.ctx[name] = base;
    this.prev[name] = 0;
  }

  add(name: string, n: number): void {
    this.ctx[name] = this.ctx[name]! + n;
  }

  call(agent: string, kind: string, out: number, deps: number[], label: string, role = "critical", group = ""): number {
    const inp = this.ctx[agent]!;
    const item: Obj = {
      id: this.items.length,
      agent,
      kind,
      label,
      deps: [...deps],
      input: inp,
      prefix: this.prev[agent],
      output: out,
      p: callP(kind, inp, this.model),
      role,
      group,
      ms: 0,
    };
    this.items.push(item);
    this.prev[agent] = inp;
    this.ctx[agent] = inp + out;
    return item.id;
  }

  tool(agent: string, label: string, deps: number[]): number {
    const item: Obj = {
      id: this.items.length,
      agent,
      kind: "tool",
      label,
      deps: [...deps],
      input: 0,
      prefix: 0,
      output: 0,
      p: 1.0,
      role: "free",
      group: "",
      ms: this.task.tool_ms,
      result: this.task.tool_result_tokens,
    };
    this.items.push(item);
    this.ctx[agent] = this.ctx[agent]! + this.task.tool_result_tokens;
    return item.id;
  }
}

/** The call plan of a pattern on the task. */
export function plan(pattern: string, task: Obj | null = null, model: Obj | null = null): Obj {
  const t: Obj = { ...(task ?? TASK) };
  const m: Obj = { ...(model ?? MODEL) };
  const P = new Plan(t, m);
  const S = t.system_tokens as number;
  const U = t.user_tokens as number;
  const H = t.handoff_tokens as number;
  const RO = t.route_out as number;
  const WO = t.worker_out as number;
  const FO = t.final_out as number;
  const subs = t.subtasks as string[];
  if (pattern === "single") {
    P.agent("agent", S + U);
    let last: number[] = [];
    for (const s of subs) {
      const c = P.call("agent", "act", RO, last, "search " + s);
      last = [P.tool("agent", "search " + s, [c])];
    }
    P.call("agent", "final", FO, last, "final answer");
  } else if (pattern === "supervisor") {
    P.agent("supervisor", S + U);
    let last: number[] = [];
    for (const s of subs) {
      const d = P.call("supervisor", "route", RO, last, "delegate " + s);
      const w = "worker:" + s;
      P.agent(w, S + H);
      const a = P.call(w, "act", RO, [d], "search " + s);
      const tl = P.tool(w, "search " + s, [a]);
      const r = P.call(w, "work", WO, [tl], "answer " + s);
      P.add("supervisor", WO);
      last = [r];
    }
    P.call("supervisor", "final", FO, last, "final answer");
  } else if (pattern === "hierarchical") {
    P.agent("top", S + U);
    const half = Math.floor((subs.length + 1) / 2);
    const teams = [subs.slice(0, half), subs.slice(half)];
    let last: number[] = [];
    teams.forEach((team, i) => {
      const tag = "ab"[i]!;
      const lead = "lead:" + tag;
      const d = P.call("top", "route", RO, last, "delegate team " + tag);
      P.agent(lead, S + H);
      let inner = [d];
      for (const s of team) {
        const d2 = P.call(lead, "route", RO, inner, "delegate " + s);
        const w = "worker:" + s;
        P.agent(w, S + H);
        const a = P.call(w, "act", RO, [d2], "search " + s);
        const tl = P.tool(w, "search " + s, [a]);
        const r = P.call(w, "work", WO, [tl], "answer " + s);
        P.add(lead, WO);
        inner = [r];
      }
      const sm = P.call(lead, "work", WO, inner, "summarise team " + tag);
      P.add("top", WO);
      last = [sm];
    });
    P.call("top", "final", FO, last, "final answer");
  } else if (pattern === "swarm") {
    let hist = U;
    let last: number[] = [];
    subs.forEach((s, i) => {
      const ag = "specialist:" + s;
      if (!(ag in P.ctx)) P.agent(ag, S + hist);
      else P.ctx[ag] = S + hist;
      const a = P.call(ag, "act", RO, last, "search " + s);
      const tl = P.tool(ag, "search " + s, [a]);
      hist += RO + (t.tool_result_tokens as number);
      let r: number;
      if (i < subs.length - 1) {
        r = P.call(ag, "work", WO + RO, [tl], "answer " + s + ", hand off");
        hist += WO + RO;
      } else r = P.call(ag, "final", FO, [tl], "final answer");
      last = [r];
    });
  } else if (pattern === "debate") {
    P.agent("researcher", S + U);
    let last: number[] = [];
    for (const s of subs) {
      const c = P.call("researcher", "act", RO, last, "search " + s);
      last = [P.tool("researcher", "search " + s, [c])];
    }
    const evidence = subs.length * (RO + (t.tool_result_tokens as number));
    const n = t.debaters as number;
    const first: number[] = [];
    for (let k = 0; k < n; k++) {
      const ag = "debater:" + (k + 1);
      P.agent(ag, S + U + evidence);
      first.push(P.call(ag, "work", WO, last, "round 1 answer", "free"));
    }
    const second: number[] = [];
    for (let k = 0; k < n; k++) {
      const ag = "debater:" + (k + 1);
      P.add(ag, (n - 1) * WO);
      second.push(P.call(ag, "work", WO, first, "round 2 answer", "vote", "debate"));
    }
    P.agent("judge", S + U + n * WO);
    P.call("judge", "final", FO, second, "judge: majority");
  } else if (pattern === "map_reduce") {
    P.agent("planner", S + U);
    const pl = P.call("planner", "route", RO * subs.length, [], "plan " + subs.length + " sends");
    const outs: number[] = [];
    for (const s of subs) {
      const w = "worker:" + s;
      P.agent(w, S + H);
      const a = P.call(w, "act", RO, [pl], "search " + s);
      const tl = P.tool(w, "search " + s, [a]);
      outs.push(P.call(w, "work", WO, [tl], "answer " + s));
    }
    P.agent("reducer", S + U + subs.length * WO);
    P.call("reducer", "final", FO, outs, "reduce: final answer");
  } else throw new Error(`unknown pattern ${pattern}`);
  return { pattern, title: TITLES[pattern], agents: P.agents, calls: P.items };
}

/** Map-reduce over document chunks (chapter 6). */
export function planMap(chunks: number, chunkTokens = 1500, task: Obj | null = null, model: Obj | null = null): Obj {
  const t: Obj = { ...(task ?? TASK) };
  const m: Obj = { ...(model ?? MODEL) };
  const P = new Plan(t, m);
  const S = t.system_tokens as number;
  const U = t.user_tokens as number;
  const WO = t.worker_out as number;
  const FO = t.final_out as number;
  const RO = t.route_out as number;
  P.agent("split", S + U);
  const sp = P.call("split", "route", RO, [], "split into " + chunks);
  const outs: number[] = [];
  for (let i = 0; i < chunks; i++) {
    const w = "map:" + (i + 1);
    P.agent(w, S + chunkTokens);
    outs.push(P.call(w, "work", WO, [sp], "summarise chunk " + (i + 1)));
  }
  P.agent("reduce", S + U + chunks * WO);
  P.call("reduce", "final", FO, outs, "reduce " + chunks + " summaries");
  return { pattern: "map", title: "Map-reduce over " + chunks + " chunks", agents: P.agents, calls: P.items };
}

// ---------------------------------------------------------------------------------------------
// Accounting of a plan

export function cachedTokens(call: Obj, price: Obj): number {
  return call.prefix >= price.min_tokens && call.prefix > 0 ? call.prefix : 0;
}

export function costOf(call: Obj, price: Obj): number {
  if (call.kind === "tool") return 0.0;
  return callCost(price, call.input, cachedTokens(call, price), call.input >= price.min_tokens, call.output);
}

export function durationOf(call: Obj, price: Obj, profile: Obj): number {
  if (call.kind === "tool") return call.ms;
  return callLatency(profile, call.input, cachedTokens(call, price), call.output)[1];
}

export function maj3(a: number, b: number, c: number): number {
  return a * b + a * c + b * c - 2 * a * b * c;
}

export function successP(p: Obj): number {
  let out = 1.0;
  const groups: Record<string, number[]> = {};
  for (const c of p.calls as Obj[]) {
    if (c.role === "critical") out = out * c.p;
    else if (c.role === "vote") (groups[c.group] ??= []).push(c.p);
  }
  for (const g of Object.values(groups)) out = out * maj3(g[0]!, g[1]!, g[2]!);
  return out;
}

export function schedule(p: Obj, price: Obj, profile: Obj): Obj[] {
  const ends: number[] = [];
  const out: Obj[] = [];
  for (const c of p.calls as Obj[]) {
    let st = 0.0;
    for (const d of c.deps as number[]) if (ends[d]! > st) st = ends[d]!;
    const en = st + durationOf(c, price, profile);
    ends.push(en);
    out.push({ id: c.id, start: st, end: en });
  }
  return out;
}

export function metrics(p: Obj, priceName = "claude-sonnet-4.6", profileName = "hosted"): Obj {
  const price = PRICES[priceName]!;
  const profile = LATENCY[profileName]!;
  const sched = schedule(p, price, profile);
  const per: Record<string, Obj> = {};
  for (const a of p.agents as string[]) per[a] = { calls: 0, input: 0, cached: 0, output: 0, cost: 0.0, peak: 0 };
  let tin = 0;
  let tout = 0;
  let tcache = 0;
  let calls = 0;
  let cost = 0.0;
  let serial = 0.0;
  for (const c of p.calls as Obj[]) {
    serial = serial + durationOf(c, price, profile);
    if (c.kind === "tool") continue;
    const k = cachedTokens(c, price);
    const cc = costOf(c, price);
    const a = per[c.agent]!;
    a.calls += 1;
    a.input += c.input;
    a.cached += k;
    a.output += c.output;
    a.cost = a.cost + cc;
    if (c.input > a.peak) a.peak = c.input;
    tin += c.input;
    tout += c.output;
    tcache += k;
    calls += 1;
    cost = cost + cc;
  }
  let latency = 0.0;
  for (const s of sched) if (s.end > latency) latency = s.end;
  return {
    pattern: p.pattern,
    title: p.title,
    model_calls: calls,
    tool_calls: (p.calls as Obj[]).length - calls,
    input: tin,
    cached: tcache,
    output: tout,
    cost,
    latency_ms: latency,
    serial_ms: serial,
    success: successP(p),
    agents: per,
    schedule: sched,
    price: priceName,
    profile: profileName,
  };
}

export function compare(priceName = "claude-sonnet-4.6", profileName = "hosted", model: Obj | null = null, task: Obj | null = null): Obj[] {
  return PATTERNS.map((n) => metrics(plan(n, task, model), priceName, profileName));
}

/** Frames for chapter 5's animation: one per call start or end (uncontended). */
export function patternFrames(p: Obj, priceName = "claude-sonnet-4.6", profileName = "hosted"): Obj[] {
  const price = PRICES[priceName]!;
  const profile = LATENCY[profileName]!;
  const sched = schedule(p, price, profile);
  const evs: [number, number, number][] = [];
  for (const s of sched) {
    evs.push([s.start, 1, s.id]);
    evs.push([s.end, 0, s.id]);
  }
  evs.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] !== b[1] ? a[1] - b[1] : a[2] - b[2]));
  const ctx: Record<string, number> = {};
  for (const a of p.agents as string[]) ctx[a] = 0;
  let active: number[] = [];
  const done: number[] = [];
  let tokens = 0;
  let cost = 0.0;
  const calls = p.calls as Obj[];
  const nAgents = (p.agents as string[]).length;
  const frames: Obj[] = [
    {
      t: 0.0,
      active: [],
      done: [],
      ctx: { ...ctx },
      tokens: 0,
      cost: 0.0,
      caption:
        p.title +
        ": " +
        nAgents +
        " agent" +
        (nAgents === 1 ? "" : "s") +
        ", " +
        calls.filter((c) => c.kind !== "tool").length +
        " model calls.",
    },
  ];
  for (const [t, kind, cid] of evs) {
    const c = calls[cid]!;
    let cap: string;
    if (kind === 1) {
      active.push(cid);
      if (c.kind === "tool") cap = c.agent + " runs the tool: " + c.label + ".";
      else {
        ctx[c.agent] = c.input;
        const k = cachedTokens(c, price);
        cap = c.agent + " calls the model (" + c.label + "): " + c.input + " tokens in" + (k ? " (" + k + " cached)" : "") + ".";
      }
    } else {
      active = active.filter((x) => x !== cid);
      done.push(cid);
      if (c.kind === "tool") cap = c.agent + " gets " + c.result + " tokens of results.";
      else {
        ctx[c.agent] = c.input + c.output;
        tokens += c.input + c.output;
        cost = cost + costOf(c, price);
        cap = c.agent + " answers: " + c.output + " tokens out.";
      }
    }
    frames.push({ t, active: [...active], done: [...done], ctx: { ...ctx }, tokens, cost, caption: cap });
  }
  return frames;
}
