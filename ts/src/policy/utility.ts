/**
 * The utility-cost model and the security–utility frontier (engine 1.11.0): a seeded suite of benign tasks,
 * each defence's cost (tasks blocked, prompts, extra calls) and its risk (one steering attempt per task that
 * holds untrusted input, the detector layers, and the best route to a write or send that nothing stops), for
 * fixed and adaptive attacks. A port of agent_loop_sim/policy/utility.py (same operation order); see it for
 * the model.
 */
import type { Obj } from "../data";
import { Rng } from "../rng";
import * as C from "./capabilities";
import { BROAD, DEFENCES, DEPTH, SUITE, TOOLS, UTILITY } from "./catalogue";
import * as D from "./depth";
import * as E from "./engine";

export const RANK: Record<string, number> = { allow: 0, ask: 1, deny: 2 };
export const RULES = [
  "not-granted",
  "scope",
  "destination",
  "untrusted-egress-private",
  "private-egress",
  "untrusted-write",
  "needs-input",
];
const WHAT: Record<string, string> = {
  read_notes: "notes",
  read_inbox: "inbox",
  read_ticket: "a ticket",
  search_docs: "docs",
  fetch_page: "a web page",
};

export function makeSuite(spec: Obj = SUITE): Obj[] {
  const r = new Rng(spec.seed);
  const plans: Obj[] = [];
  for (let k = 0; k < spec.n; k++) {
    const nReads = r.randint(spec.min_reads, spec.max_reads);
    const pool: string[] = [...spec.read_tools];
    const steps: Obj[] = [{ op: "input", out: "q", sens: "public" }];
    const outs: string[] = [];
    const reads: string[] = [];
    for (let j = 0; j < nReads; j++) {
      const t = pool.splice(r.randint(0, pool.length - 1), 1)[0]!;
      const st: Obj = { op: "call", tool: t, args: ["q"], out: `r${j + 1}` };
      if (TOOLS[t]!.egress) st.dest = TOOLS[t]!.dests[0];
      steps.push(st);
      outs.push(`r${j + 1}`);
      reads.push(t);
    }
    steps.push({ op: "call", tool: "summarise", args: [...outs], out: "s" });
    const sink = spec.sinks[r.randint(0, spec.sinks.length - 1)];
    let adaptive = r.random() < spec.adaptive;
    if (sink.tool === null) adaptive = false;
    else {
      const st: Obj = { op: "call", tool: sink.tool, args: ["s"], out: null };
      if (sink.dest !== null) st.dest = sink.dest;
      if (adaptive) st.ctrl = [outs[outs.length - 1]];
      steps.push(st);
    }
    const title =
      "Read " +
      reads.map((t) => WHAT[t]).join(", ") +
      "; " +
      sink.text +
      (adaptive ? ", as the last read decides" : "");
    plans.push({ id: `t${String(k + 1).padStart(3, "0")}`, title, steps, adaptive });
  }
  return plans;
}

export function approvalMiss(n: number, cfg: Obj = UTILITY): number {
  let keep = 1.0 - cfg.approval_first_miss;
  for (let i = 0; i < n - 1; i++) keep = keep * cfg.approval_decay;
  return 1.0 - keep;
}

export function consequential(tool: string): boolean {
  return TOOLS[tool]!.kind === "write" || TOOLS[tool]!.kind === "send";
}

export function grantOf(plan: Obj, d: Obj): Obj {
  if (d.grant === "broad") return BROAD;
  if (!d.policy) throw new Error("least privilege is modelled with the policy on");
  return C.needed(plan);
}

export function layerMiss(ids: string[]): number[] {
  const by: Record<string, number> = {};
  for (const x of DEPTH.layers as Obj[]) by[x.id] = x.miss;
  return ids.map((i) => by[i]!);
}

function human(d: Obj, tool: string, decision: string): boolean {
  return (
    (d.policy && decision === "ask") ||
    d.approval === "every" ||
    (d.approval === "consequential" && consequential(tool))
  );
}

export function evaluate(
  plan: Obj,
  d: Obj,
  base: number = DEPTH.base,
  cfg: Obj = UTILITY,
  adaptive = false,
): Obj {
  const grant = grantOf(plan, d);
  const steps = E.runPlan(plan, grant, d.mode, d.approve, d.policy);
  const calls = steps.filter((s) => s.op === "call");
  const blocked = calls.filter((s) => !s.ran).map((s) => s.tool as string);
  let byRule: string[] = [];
  for (const s of calls) {
    if (s.ran) continue;
    for (const f of s.fired.length > 0 ? s.fired : [{ rule: "needs-input" }])
      if (!byRule.includes(f.rule)) byRule.push(f.rule);
  }
  byRule = RULES.filter((x) => byRule.includes(x));
  let untrusted = 0;
  let cons = 0;
  let prompts = 0;
  const hum: Record<number, boolean> = {};
  for (const s of calls) {
    if (!s.ran) continue;
    const t = TOOLS[s.tool]!;
    if (t.kind === "read" && t.out.src === "untrusted") untrusted += 1;
    if (consequential(s.tool)) cons += 1;
    const h = human(d, s.tool, s.decision);
    hum[s.i] = h;
    if (h) prompts += 1;
  }
  const miss = prompts > 0 ? approvalMiss(prompts, cfg) : 0.0;
  const routes: Obj[] = [];
  for (const s of calls)
    if (s.ran && consequential(s.tool) && s.control.src === "untrusted")
      routes.push({
        kind: "in-plan",
        step: s.i,
        tool: s.tool,
        dest: s.dest,
        decision: s.decision,
        human: hum[s.i],
        pass: hum[s.i] ? miss : 1.0,
      });
  if (d.mode === "context") {
    let best: Obj | null = null;
    for (const s of steps) {
      if (s.op === "call" && s.ran && s.ctx.src === "untrusted") {
        const dec = d.policy ? E.decide("send_message", s.ctx, "web", grant).decision : "allow";
        if (best === null || RANK[dec]! < RANK[best.decision]!)
          best = { kind: "off-plan", step: s.i, tool: "send_message", dest: "web", decision: dec, label: { ...s.ctx } };
      }
    }
    if (best !== null) {
      const closed = best.decision === "deny" || (best.decision === "ask" && !d.approve);
      const h = !closed && human(d, "send_message", best.decision);
      best.human = h;
      best.pass = closed ? 0.0 : h ? approvalMiss(prompts + 1, cfg) : 1.0;
      routes.push(best);
    }
  }
  let reach = 0.0;
  for (const x of routes) if (x.pass > reach) reach = x.pass;
  const m = layerMiss(d.layers);
  const det = m.length > 0 && !adaptive ? D.allMissCommon(m, d.c) : 1.0;
  const risk = base * det * reach;
  let keep = blocked.length > 0 ? 0.0 : 1.0;
  let extra = d.mode === "dataflow" ? cfg.planner_calls : 0;
  for (const lid of d.layers as string[]) {
    const n = lid === "output_check" ? cons : untrusted;
    if (lid !== "marking") extra += n;
    for (let i = 0; i < n; i++) keep = keep * (1.0 - cfg.fp[lid]);
  }
  return {
    task: plan.id ?? null,
    blocked,
    by_rule: byRule,
    keep,
    prompts,
    miss,
    extra,
    untrusted,
    consequential: cons,
    routes,
    reach,
    detect: det,
    risk,
    decisions: calls.map((s) => s.decision),
  };
}

export function summarise(rows: Obj[]): Obj {
  const n = rows.length;
  const s: Record<string, number> = {
    utility: 0.0,
    blocked: 0.0,
    prompts: 0.0,
    extra: 0.0,
    risk: 0.0,
    exposed: 0.0,
    ungated: 0.0,
    with_untrusted: 0.0,
  };
  for (const r of rows) {
    s.utility = s.utility! + r.keep;
    s.blocked = s.blocked! + (r.blocked.length > 0 ? 1.0 : 0.0);
    s.prompts = s.prompts! + r.prompts;
    s.extra = s.extra! + r.extra;
    s.risk = s.risk! + r.risk;
    s.exposed = s.exposed! + (r.reach > 0.0 ? 1.0 : 0.0);
    s.ungated = s.ungated! + (r.reach === 1.0 ? 1.0 : 0.0);
    s.with_untrusted = s.with_untrusted! + (r.untrusted > 0 ? 1.0 : 0.0);
  }
  const out: Obj = {};
  for (const [k, v] of Object.entries(s)) out[k] = v / n;
  out.n = n;
  out.by_rule = Object.fromEntries(RULES.map((x) => [x, rows.filter((r) => r.by_rule.includes(x)).length]));
  return out;
}

export function runDefence(d: Obj, suite: Obj[], base: number = DEPTH.base, cfg: Obj = UTILITY, adaptive = false): Obj {
  const rows = suite.map((p) => evaluate(p, d, base, cfg, adaptive));
  return { id: d.id, name: d.name, rows, summary: summarise(rows) };
}

export function pareto(points: Obj[]): string[] {
  const front: string[] = [];
  for (const a of points) {
    let beaten = false;
    for (const b of points) {
      if (b.id === a.id) continue;
      if (
        b.utility >= a.utility &&
        b.risk <= a.risk &&
        b.prompts <= a.prompts &&
        (b.utility > a.utility || b.risk < a.risk || b.prompts < a.prompts)
      )
        beaten = true;
    }
    if (!beaten) front.push(a.id);
  }
  return front;
}

export function frontier(adaptive = false, suite: Obj[] | null = null, cfg: Obj = UTILITY): Obj {
  const tasks = suite ?? makeSuite();
  const base = adaptive ? DEPTH.base_range[1] : DEPTH.base;
  const runs = DEFENCES.map((d) => runDefence(d, tasks, base, cfg, adaptive));
  const points = runs.map((r) => ({ ...r.summary, id: r.id, name: r.name }));
  return { adaptive, base, points, front: pareto(points) };
}

export function simulate(rows: Obj[], d: Obj, base: number, reps: number, seed: number, adaptive = false): Obj {
  const m = layerMiss(d.layers);
  const q = m.length > 0 && !adaptive ? D.qGivenCommon(m, d.c) : [];
  const r = new Rng(seed);
  let harmed = 0;
  let done = 0;
  let n = 0;
  for (let k = 0; k < reps; k++) {
    for (const row of rows) {
      n += 1;
      const steered = r.random() < base;
      const common = r.random() < d.c;
      let missed = true;
      for (let i = 0; i < q.length; i++) {
        const u = r.random();
        if (!(common || u < q[i]!)) missed = false;
      }
      const route = r.random() < row.reach;
      if (steered && missed && route) harmed += 1;
      if (r.random() < row.keep) done += 1;
    }
  }
  return { n, harmed, done, risk: harmed / n, utility: done / n };
}

export function defence(did: string): Obj {
  const d = DEFENCES.find((x) => x.id === did);
  if (!d) throw new Error(did);
  return d;
}

export function withThreshold(d: Obj, th: string): Obj {
  const approval: Record<string, string> = {
    refuse: "none",
    policy: "none",
    consequential: "consequential",
    every: "every",
  };
  return { ...d, approve: th !== "refuse", approval: approval[th], id: d.id + ":" + th };
}
