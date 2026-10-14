/**
 * What the Security site's chapters animate (engine 1.10.0): boundaries, a classic agent against a plan made
 * in advance, the trifecta, least privilege, sandbox layers and defence in depth, each as frames with
 * captions. A port of agent_loop_sim/policy/study.py.
 */
import { fixed } from "../context/text";
import type { Obj } from "../data";
import * as C from "./capabilities";
import { Rng } from "../rng";
import {
  ACTIONS,
  ADAPTIVE,
  BROAD,
  CIRCLE_TOOLS,
  DEPTH,
  DEPTH_DOTS,
  DEST_ORDER,
  LAYERS,
  PLANS,
  PRIVILEGE,
  TOOL_ORDER,
  TOOLS,
  UTILITY,
} from "./catalogue";
import * as D from "./depth";
import * as E from "./engine";
import * as L from "./labels";
import * as S from "./sandbox";
import * as U from "./utility";

export const VERB: Record<string, string> = {
  allow: "allowed",
  ask: "sent for approval",
  deny: "denied",
  skip: "skipped",
};
export const NOUN: Record<string, string> = {
  private: "private data",
  untrusted: "untrusted input",
  outbound: "outbound channel",
};

const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T;

function zones(tool: string | null, op: string): [string, string] {
  if (op === "input") return ["user", "model"];
  const t = TOOLS[tool!]!;
  if (t.kind === "transform") return ["model", "model"];
  if (t.egress) return ["model", "outside"];
  return ["model", "tools"];
}

function lab(x: L.Label | null): string {
  return x !== null ? "(" + L.text(x) + ")" : "(none)";
}

// ── chapter 1 ────────────────────────────────────────────────────────────

export function boundaryFrames(planId: string, mode = "context", grant: Obj | null = null): Obj[] {
  const plan = PLANS[planId]!;
  const steps = E.runPlan(plan, grant ?? BROAD, mode);
  const frames: Obj[] = [];
  const known: Obj = {};
  for (const r of steps) {
    const n = r.i + 1;
    if (r.op === "input") {
      known[r.out] = r.out_label;
      frames.push({
        step: r.i,
        from: "user",
        to: "model",
        value: r.out,
        label: r.out_label,
        decision: "allow",
        values: { ...known },
        ctx: r.ctx,
        caption: `Step ${n}: the user's request ${r.out} enters the model, labelled ${lab(r.out_label)}.`,
      });
      continue;
    }
    const [a, b] = zones(r.tool, r.op);
    const t = TOOLS[r.tool]!;
    if (r.decision === "skip") {
      frames.push({
        step: r.i,
        from: a,
        to: b,
        value: null,
        label: null,
        decision: "skip",
        values: { ...known },
        ctx: L.joinAll(Object.values(known) as L.Label[]),
        caption: `Step ${n}: ${r.tool} is skipped: an input it needs was never produced.`,
      });
      continue;
    }
    const where = ({ tools: "a tool", outside: "the outside world (" + String(r.dest) + ")", model: "the model" } as Obj)[b];
    frames.push({
      step: r.i,
      from: a,
      to: b,
      value: r.args.join(","),
      label: r.effective,
      decision: r.decision,
      values: { ...known },
      // the call is made before its result returns: the context is what the model holds now
      ctx: L.joinAll(Object.values(known) as L.Label[]),
      caption:
        `Step ${n}: ${r.tool} is called; its arguments and control carry ` +
        `${lab(r.effective)} to ${where}. ${r.why[0].toUpperCase()}${r.why.slice(1)}.`,
    });
    if (r.ran && r.out_label !== null) {
      known[r.out] = r.out_label;
      const src = t.kind === "transform" ? "model" : b;
      frames.push({
        step: r.i,
        from: src,
        to: "model",
        value: r.out,
        label: r.out_label,
        decision: "allow",
        values: { ...known },
        ctx: r.ctx,
        caption:
          `Step ${n}: ${r.out} comes back to the model labelled ${lab(r.out_label)}` +
          (t.kind === "read" ? `: ${t.text}.` : ", the join of its inputs."),
      });
    }
  }
  return frames;
}

// ── chapter 2 ────────────────────────────────────────────────────────────

export function contextFrames(planId: string, grant: Obj | null = null): Obj {
  const g = grant ?? BROAD;
  const plan = PLANS[planId]!;
  const a = E.runPlan(plan, g, "context");
  const b = E.runPlan(plan, g, "dataflow");
  const frames = a.map((x, k) => {
    const y = b[k]!;
    const n = x.i + 1;
    const cap =
      x.op === "input"
        ? `Step ${n}: the request is the only thing either agent has seen.`
        : `Step ${n}, ${x.tool}: in the classic agent its control is ${lab(x.control)} and it is ` +
          `${VERB[x.decision]}; planned in advance, its control is ${lab(y.control)} and it is ` +
          `${VERB[y.decision]}.`;
    return { step: x.i, context: x, dataflow: y, caption: cap };
  });
  return { plan: planId, frames, summary: { context: E.summary(a), dataflow: E.summary(b) } };
}

// ── chapter 3 ────────────────────────────────────────────────────────────

export function grantFor(tools: string[], egress: string[] | null = null): Obj {
  const ts = TOOL_ORDER.filter((t) => tools.includes(t));
  return {
    tools: ts,
    scopes: ts
      .filter((t) => TOOLS[t]!.scope)
      .map((t) => TOOLS[t]!.scope as string)
      .sort(),
    egress: [...(egress ?? DEST_ORDER)],
    private_dests: ["self"],
  };
}

export function exposurePlan(path: string[], dest: string | null): Obj {
  const steps: Obj[] = [{ op: "input", out: "q", sens: "public" }];
  const outs = ["q"];
  path.forEach((t, i) => {
    if (i < path.length - 1) {
      const o = `v${i + 1}`;
      const st: Obj = { op: "call", tool: t, args: ["q"], out: o };
      if (TOOLS[t]!.egress) st.dest = TOOLS[t]!.dests[0];
      steps.push(st);
      outs.push(o);
    } else steps.push({ op: "call", tool: t, args: [...outs], dest, out: null });
  });
  return { title: "a flow out", steps };
}

export function trifectaCase(name: string, tools: string[]): Obj {
  const g = grantFor([...tools, "summarise"]);
  const tri = C.trifecta(g);
  const ex = C.exposure(g);
  const c: Obj = { name, grant: g, trifecta: tri, exposure: ex, steps: null };
  if (ex.exposed) {
    const steps = E.runPlan(exposurePlan(ex.path, ex.dest), g, "context");
    c.steps = steps;
    const last = steps[steps.length - 1]!;
    c.caption =
      `All three circles: ${ex.path.join(" → ")} to ${ex.dest} would carry private data ` +
      `out under untrusted control. The policy's decision: ${last.decision} ` +
      `(${last.fired.map((f: Obj) => f.rule).join(", ")}).`;
  } else {
    const miss = ["private", "untrusted", "outbound"].filter((k) => tri[k].length === 0).map((k) => NOUN[k]);
    c.caption =
      `No ${miss.join(", no ")}: no sequence of calls this agent holds can carry private ` +
      `data out under untrusted control (${ex.checked} sequences checked).`;
  }
  return c;
}

export function trifectaStudy(): Obj {
  const cases: Obj[] = [];
  for (let mask = 0; mask < 8; mask++) {
    const on = ["private", "untrusted", "outbound"].filter((_, i) => (mask >> i) & 1);
    const tools: string[] = [];
    for (const k of on) tools.push(...CIRCLE_TOOLS[k]!);
    cases.push({ ...trifectaCase(on.length ? on.join("+") : "none", tools), mask, on });
  }
  const double = [
    trifectaCase("read_inbox+send_message", ["read_inbox", "send_message"]),
    trifectaCase("read_notes+fetch_page", ["read_notes", "fetch_page"]),
  ];
  return { circles: cases, double };
}

// ── chapter 4 ────────────────────────────────────────────────────────────

export function privilegeStudy(): Obj {
  const base = PLANS[PRIVILEGE.plan]!;
  const plan = {
    title: base.title + ", and one call it does not need",
    steps: [...clone(base.steps as Obj[]), clone(PRIVILEGE.extra)],
  };
  const grants: Record<string, Obj> = { broad: BROAD, task: C.needed(base) };
  const runs: Obj = {};
  for (const [gid, g] of Object.entries(grants)) {
    const toks = C.mint(g);
    const steps = E.runPlan(plan, g, "context", false);
    for (const r of steps) if (r.op === "call") r.token = C.checkToken(toks[TOOLS[r.tool]!.aud ?? ""] ?? null, r.tool);
    runs[gid] = { grant: g, tokens: toks, steps, over: C.overGrant(g, base) };
  }
  const frames: Obj[] = [];
  for (let i = 0; i < plan.steps.length; i++) {
    const x = runs.broad.steps[i];
    const y = runs.task.steps[i];
    const n = i + 1;
    const cap =
      x.op === "input"
        ? `Step ${n}: the request. The broad grant holds ${BROAD.tools.length} tools; the task needs ` +
          `${grants.task!.tools.length}.`
        : `Step ${n}, ${x.tool}: broad grant: ${x.why}; least privilege: ${y.why}.`;
    frames.push({ step: i, broad: x, task: y, caption: cap });
  }
  const first = runs.task.tokens[TOOLS[base.steps[1].tool]!.aud];
  const passthrough = { token: first, tool: "send_message", check: C.checkToken(first, "send_message") };
  return { plan, runs, frames, passthrough };
}

// ── chapter 5 ────────────────────────────────────────────────────────────

export function sandboxFrames(): Obj[] {
  const frames: Obj[] = [];
  for (let k = 0; k <= LAYERS.length; k++) {
    const active = LAYERS.slice(0, k).map((x) => x.id as string);
    const rows = S.matrix(active);
    const t = S.tally(rows);
    let cap: string;
    if (k === 0)
      cap = `No sandbox: every action runs. The task's ${t.works} actions work, and so do all ${t.open} it does not need.`;
    else {
      const lay = LAYERS[k - 1]!;
      const prev = S.matrix(active.slice(0, -1));
      const nw = rows.filter((r, j) => r.stopped && !prev[j]!.stopped).map((r) => r.id);
      cap =
        `+ ${lay.name}: ${lay.text}. It newly stops ${nw.length} action` +
        `${nw.length === 1 ? "" : "s"}; ${t.open} unneeded action${t.open === 1 ? "" : "s"} ` +
        `still open, ${t.broken} needed action${t.broken === 1 ? "" : "s"} broken.`;
    }
    frames.push({ layers: active, rows, tally: t, caption: cap });
  }
  return frames;
}

export function sandboxActionFrames(active: string[]): Obj[] {
  const names: Record<string, string> = {};
  for (const x of LAYERS) names[x.id] = x.name;
  const rows = S.matrix(active);
  return ACTIONS.map((a, i) => {
    const r = rows[i]!;
    const who = (r.by as Obj[]).map((b) => names[b.layer]!.toLowerCase()).join(" and the ");
    const why = r.by.length > 0 ? r.by[0].reason : "";
    const text = a.text[0].toUpperCase() + a.text.slice(1);
    let cap: string;
    if (r.outcome === "stopped") cap = `${text}: not needed, and stopped by the ${who} (${why}).`;
    else if (r.outcome === "broken") cap = `${text}: the task needs it, but the ${who} stops it (${why}), so the task breaks.`;
    else if (r.outcome === "works") cap = `${text}: the task needs it, and no active layer stops it.`;
    else if (a.kind === "net" && active.includes("egress"))
      cap = `${text}: not needed, and nothing stops it: the allow-list checks where a connection goes, not what it carries.`;
    else cap = `${text}: not needed, and no active layer stops it.`;
    return { ...r, text: a.text, kind: a.kind, caption: cap };
  });
}

export function allLayerSets(): Obj[] {
  const out: Obj[] = [];
  for (let mask = 0; mask < 1 << LAYERS.length; mask++) {
    const active = LAYERS.filter((_, i) => (mask >> i) & 1).map((x) => x.id as string);
    const rows = S.matrix(active);
    out.push({ layers: active, rows, tally: S.tally(rows), frames: sandboxActionFrames(active) });
  }
  return out;
}

// ── defence in depth (chapter 6) ─────────────────────────────────────────

export function depthStudy(cfg: Obj = DEPTH): Obj {
  const m = (cfg.layers as Obj[]).map((x) => x.miss as number);
  const sims = (cfg.c as number[]).map((c, i) => ({ ...D.simulate(m, c, cfg.trials, cfg.seed + i), c }));
  const corr = (cfg.c as number[]).map((c) => ({
    c,
    pairs: m.map((_, i) => m.map((__, j) => D.pairCorrelation(m[i]!, m[j]!, c))),
  }));
  return {
    table: D.depthTable(cfg.base, m, cfg.c, cfg.attempts),
    independent: D.allMissIndependent(m),
    sims,
    corr,
  };
}

export function depthFrames(c: number, k: number | null = null, cfg: Obj = DEPTH, dots: Obj = DEPTH_DOTS): Obj {
  const lay = (k === null ? cfg.layers : cfg.layers.slice(0, k)) as Obj[];
  const m = lay.map((x) => x.miss as number);
  const q = D.qGivenCommon(m, c);
  const r = new Rng(dots.seed);
  const stages: number[] = [];
  for (let j = 0; j < dots.n; j++) {
    const steer = r.random() < cfg.base;
    const common = r.random() < c;
    let st = steer ? 1 : 0;
    let alive = steer;
    for (let i = 0; i < q.length; i++) {
      const u = r.random();
      alive = alive && (common || u < q[i]!);
      if (alive) st = st + 1;
    }
    stages.push(st);
  }
  const frames: Obj[] = [];
  const n = dots.n as number;
  for (let f = 0; f < m.length + 2; f++) {
    let count = 0;
    for (const x of stages) if (x >= f) count += 1;
    let exp: number;
    let cap: string;
    if (f === 0) {
      exp = n;
      cap = `${n} attempts arrive in untrusted results.`;
    } else if (f === 1) {
      exp = n * cfg.base;
      cap = `With no defence, ${count} of them steer the model (expected ${fixed(exp, 1)}: the base rate ${pct(cfg.base, 0)}).`;
    } else {
      exp = n * cfg.base * D.allMissCommon(m.slice(0, f - 1), c);
      cap = `+ ${lay[f - 2]!.name} (misses ${pct(m[f - 2]!, 0)}): ${count} still get through (expected ${fixed(exp, 2)}).`;
    }
    frames.push({ f, count, expected: exp, caption: cap });
  }
  const last = frames[frames.length - 1]!;
  last.caption =
    last.caption +
    (c > 0
      ? ` A shared blind spot of ${pct(c)} sets a floor of ${fixed(n * cfg.base * c, 2)} that no extra layer removes.`
      : " Every layer misses some, so the product never reaches zero.");
  return { c, n, stages, frames };
}

// ── chapter 7 ────────────────────────────────────────────────────────────

export function dualPlan(pid: string): Obj {
  return pid === "adaptive" ? ADAPTIVE : PLANS[pid]!;
}

export function dualFrames(pid: string): Obj {
  const plan = dualPlan(pid);
  const g = C.needed(plan);
  const a = E.runPlan(plan, g, "dataflow");
  const b = E.runPlan(plan, g, "context");
  const frames: Obj[] = [];
  const seen: string[] = [];
  a.forEach((x, idx) => {
    const y = b[idx]!;
    const n = x.i + 1;
    if (x.op === "input") {
      const calls = (plan.steps as Obj[]).filter((st) => st.op === "call").map((st) => st.tool as string);
      frames.push({
        step: x.i,
        actor: "planner",
        flow: x,
        classic: y,
        planner_sees: ["q"],
        caption: `Step ${n}: the planner reads only the request and writes the whole plan: ${calls.join(" → ")}. It will never read a tool's result.`,
      });
      seen.push("q");
      return;
    }
    const t = TOOLS[x.tool]!;
    const actor = t.kind === "transform" ? "quarantined" : "tool";
    if (x.ran && x.out) seen.push(x.out);
    const what = x.decision === "skip" ? "is skipped: an input it needs was never produced" : `is ${VERB[x.decision]} on (${L.text(x.effective)})`;
    const how = actor === "tool" ? "runs as a tool" : "runs in the quarantined model, which has no tools";
    let ctrl = "";
    if ("ctrl" in x)
      ctrl = ` The plan branches on ${(x.ctrl as string[]).join(", ")}, so the call's control is ${lab(x.control)}, tracked rather than hidden.`;
    frames.push({
      step: x.i,
      actor,
      flow: x,
      classic: y,
      planner_sees: [...seen],
      caption: `Step ${n}, ${x.tool} ${how} and ${what}; the classic agent's call is ${VERB[y.decision]}.${ctrl}`,
    });
  });
  return {
    plan: pid,
    title: plan.title,
    frames,
    summary: { dataflow: E.summary(a), context: E.summary(b) },
    extra_calls: UTILITY.planner_calls,
  };
}

// ── chapter 8 ────────────────────────────────────────────────────────────

function routeText(r: Obj): string {
  if (r.routes.length === 0) return "; no steered attempt has a route.";
  if (r.reach === 0.0) return "; the policy closes every route a steered attempt could take.";
  return `; a steered attempt's best route passes with ${pct(r.reach, 0)}.`;
}

export function approvalFrames(th: string, suite: Obj[] | null = null): Obj {
  const tasks = suite ?? U.makeSuite();
  const d = U.withThreshold(U.defence("policy"), th);
  const rows = tasks.map((p) => U.evaluate(p, d));
  const frames: Obj[] = [];
  let prompts = 0;
  let gated = 0;
  let risky = 0;
  tasks.slice(0, UTILITY.frames).forEach((p, i) => {
    const r = rows[i]!;
    prompts += r.prompts;
    if (r.routes.length > 0) {
      risky += 1;
      if (r.reach < 1.0) gated += 1;
    }
    const calls = (p.steps as Obj[]).filter((st) => st.op === "call");
    const steps = calls.map((st, j) => ({ tool: st.tool, dest: st.dest ?? null, decision: r.decisions[j] }));
    const how = r.blocked.length > 0 ? `blocked at ${r.blocked[0]} (${r.by_rule.join(", ")})` : "completes";
    const cap =
      `${p.id}: ${p.title}. It ${how}, with ${r.prompts} prompt${r.prompts === 1 ? "" : "s"}` +
      (r.prompts > 0 ? ` (a mistaken approval at the last: ${pct(r.miss, 0)})` : "") +
      routeText(r);
    frames.push({ task: p.id, title: p.title, steps, row: r, prompts, risky, gated, caption: cap });
  });
  return { threshold: th, frames, summary: U.summarise(rows) };
}

export function approvalStudy(suite: Obj[] | null = null): Obj {
  const tasks = suite ?? U.makeSuite();
  const curve: Obj[] = [];
  for (let n = 1; n < 11; n++) curve.push({ n, miss: U.approvalMiss(n) });
  const thresholds: Obj = {};
  for (const th of UTILITY.thresholds as string[]) thresholds[th] = approvalFrames(th, tasks);
  return { thresholds, fatigue: curve };
}

// ── chapter 9 ────────────────────────────────────────────────────────────

export function frontierStudy(suite: Obj[] | null = null): Obj {
  const tasks = suite ?? U.makeSuite();
  const out: Obj = { n: tasks.length, adaptive_share: 0.0 };
  let k = 0;
  for (const p of tasks) if (p.adaptive) k += 1;
  out.adaptive_share = k / tasks.length;
  for (const [key, adaptive] of [
    ["fixed", false],
    ["adaptive", true],
  ] as [string, boolean][]) {
    const fr = U.frontier(adaptive, tasks);
    const frames = (fr.points as Obj[]).map((pt, i) => {
      const on = fr.front.includes(pt.id) ? "on the frontier" : "beaten by another point";
      return {
        i,
        id: pt.id,
        caption: `${pt.name}: finishes ${pct(pt.utility)} of the benign tasks with ${fixed(pt.prompts, 2)} prompts and ${fixed(pt.extra, 2)} extra calls per task; a steered attempt succeeds in ${pct(pt.risk, 2)} of tasks (${on}).`,
      };
    });
    out[key] = { ...fr, frames };
  }
  const d = U.defence("planned");
  const rows = tasks.map((p) => U.evaluate(p, d));
  out.sim = U.simulate(rows, d, DEPTH.base, UTILITY.reps, UTILITY.seed);
  return out;
}

// ── chapter 10 ───────────────────────────────────────────────────────────

function detectorValue(n: number, policy: boolean): string {
  const layers = `${n} detector layer${n === 1 ? "" : "s"}`;
  if (policy) return n ? layers + ", backed by the policy" : "no detectors needed: the policy decides on labels";
  return n ? layers + " and nothing structural behind them" : "no detectors and no policy";
}

export const CHECK_DEFAULT: Obj = { defence: "planned", approval: null, sandbox: ["fs_scope", "container", "egress"] };

export function checklist(design: Obj = CHECK_DEFAULT, suite: Obj[] | null = null): Obj {
  const tasks = suite ?? U.makeSuite();
  const d: Obj = { ...U.defence(design.defence) };
  if (design.approval) {
    d.approval = design.approval;
    d.id = d.id + "+" + design.approval;
  }
  const base = DEPTH.base_range[1];
  const rows = tasks.map((p) => U.evaluate(p, d, base, UTILITY, true));
  const sm = U.summarise(rows);
  let tri = 0;
  let unneeded = 0;
  for (const p of tasks) {
    const g = U.grantOf(p, d);
    if (C.trifecta(g).all) tri += 1;
    unneeded += C.overGrant(g, p).tools!.length;
  }
  const t = S.tally(S.matrix(design.sandbox));
  const fr = U.frontier(true, tasks);
  const pts = [...(fr.points as Obj[]).filter((x) => x.id !== d.id), { ...sm, id: d.id, name: "this design" }];
  const onFront = U.pareto(pts).includes(d.id);
  const budget = UTILITY.risk_budget as number;
  const items: Obj[] = [
    {
      ch: 1,
      item: "Every value carries a label, and every call is checked against the policy",
      verdict: d.policy ? "pass" : "fail",
      value: d.policy ? "labels tracked" : "no labels: the agent cannot tell where data came from",
    },
    {
      ch: 2,
      item: "No untrusted result can choose a write or send that nothing checks",
      verdict: sm.ungated === 0.0 ? "pass" : sm.ungated < 0.1 ? "warn" : "fail",
      value: `${pct(sm.ungated)} of tasks have an unchecked route`,
    },
    {
      ch: 3,
      item: "No task holds private data, untrusted input and an outbound channel unpoliced",
      verdict: tri === 0 ? "pass" : d.policy ? "warn" : "fail",
      value: `${tri} of ${tasks.length} tasks hold all three` + (tri && d.policy ? ", and the policy denies the flow out" : ""),
    },
    {
      ch: 4,
      item: "Each task holds only the tools and scopes it needs",
      verdict: unneeded === 0 ? "pass" : "fail",
      value: `${fixed(unneeded / tasks.length, 2)} unneeded tools per task`,
    },
    {
      ch: 5,
      item: "Code runs in a sandbox with a file scope, a container and an egress allow-list",
      verdict: t.open! <= 1 && t.broken === 0 ? "pass" : t.open! <= 2 ? "warn" : "fail",
      value: `${t.open} unneeded action${t.open === 1 ? "" : "s"} still open, ${t.broken} needed broken`,
    },
    {
      ch: 6,
      item: "Detectors are one layer, never the only one",
      verdict: d.policy ? "pass" : d.layers.length > 0 ? "warn" : "fail",
      value: detectorValue(d.layers.length, d.policy),
    },
    {
      ch: 7,
      item: "Untrusted data cannot change the plan",
      verdict: d.mode === "dataflow" ? "pass" : d.policy ? "warn" : "fail",
      value: d.mode === "dataflow" ? "planned in advance" : "a classic agent: every result steers",
    },
    {
      ch: 8,
      item: "Few prompts per task, so each approval gets attention",
      verdict: sm.prompts <= 0.5 ? "pass" : sm.prompts <= 1.5 ? "warn" : "fail",
      value: `${fixed(sm.prompts, 2)} prompts per task`,
    },
    {
      ch: 9,
      item: "The design sits on the security–utility frontier, within the risk budget",
      verdict: onFront && sm.risk < budget ? "pass" : onFront || sm.risk < budget ? "warn" : "fail",
      value: `${pct(sm.utility)} of tasks finish; risk ${pct(sm.risk, 2)} per task against an adaptive attacker`,
    },
  ];
  const frames: Obj[] = [];
  let score = 0;
  items.forEach((it, i) => {
    if (it.verdict === "pass") score += 1;
    frames.push({
      ...it,
      i,
      passed: score,
      caption: `Chapter ${it.ch}: ${(it.item as string).toLowerCase()}? ${(it.verdict as string).toUpperCase()}: ${it.value}.`,
    });
  });
  return { design, items, frames, summary: sm, passed: score };
}

export function pct(x: number, d = 1): string {
  return fixed(x * 100.0 + 0.0, d) + "%";
}
