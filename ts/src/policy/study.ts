/**
 * What the Security site's chapters animate (engine 1.10.0): boundaries, a classic agent against a plan made
 * in advance, the trifecta, least privilege, sandbox layers and defence in depth, each as frames with
 * captions. A port of agent_loop_sim/policy/study.py.
 */
import { fixed } from "../context/text";
import type { Obj } from "../data";
import * as C from "./capabilities";
import { ACTIONS, BROAD, CIRCLE_TOOLS, DEPTH, DEST_ORDER, LAYERS, PLANS, PRIVILEGE, TOOL_ORDER, TOOLS } from "./catalogue";
import * as D from "./depth";
import * as E from "./engine";
import * as L from "./labels";
import * as S from "./sandbox";

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

export function pct(x: number, d = 1): string {
  return fixed(x * 100.0 + 0.0, d) + "%";
}
