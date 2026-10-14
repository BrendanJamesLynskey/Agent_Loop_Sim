/**
 * The policy engine (engine 1.10.0; `ctrl` and `enforce` since 1.11.0): run a plan over labels and decide
 * each call (allow, ask or deny, with every rule that fired). A port of agent_loop_sim/policy/engine.py; see
 * it for the rules.
 */
import type { Obj } from "../data";
import { TOOLS } from "./catalogue";
import * as L from "./labels";

export const RANK: Record<string, number> = { allow: 0, ask: 1, deny: 2 };

export function decide(tool: string, eff: L.Label, dest: string | null | undefined, grant: Obj): Obj {
  const t = TOOLS[tool]!;
  const fired: Obj[] = [];
  if (!grant.tools.includes(tool))
    fired.push({ rule: "not-granted", decision: "deny", reason: `${tool} is not in this task's grant` });
  if (t.scope !== null && !grant.scopes.includes(t.scope))
    fired.push({ rule: "scope", decision: "deny", reason: `the grant lacks the scope ${t.scope}` });
  if (t.egress) {
    if (!t.dests.includes(dest)) throw new Error(`${tool} cannot send to ${dest}`);
    if (!grant.egress.includes(dest))
      fired.push({ rule: "destination", decision: "deny", reason: `${dest} is not an allowed destination` });
    if (!grant.private_dests.includes(dest) && eff.sens === "private") {
      if (eff.src === "untrusted")
        fired.push({
          rule: "untrusted-egress-private",
          decision: "deny",
          reason: "untrusted data steers a call that carries private data out to " + dest,
        });
      else fired.push({ rule: "private-egress", decision: "ask", reason: `private data would leave to ${dest}` });
    }
  }
  if (t.kind === "write" && eff.src === "untrusted")
    fired.push({ rule: "untrusted-write", decision: "ask", reason: "untrusted data decides what is written" });
  let decision = "allow";
  for (const f of fired) if (RANK[f.decision]! > RANK[decision]!) decision = f.decision;
  return { decision, fired };
}

export function explain(tool: string, eff: L.Label, d: Obj): string {
  if (d.fired.length === 0) return `allow: ${tool} with (${L.text(eff)}) breaks no rule`;
  return d.decision + ": " + d.fired.map((f: Obj) => f.reason).join("; ");
}

export function runPlan(plan: Obj, grant: Obj, mode = "context", approve = true, enforce = true): Obj[] {
  if (mode !== "context" && mode !== "dataflow") throw new Error(mode);
  const vals: Record<string, L.Label> = {};
  let ctx: L.Label = { ...L.BOTTOM };
  const out: Obj[] = [];
  plan.steps.forEach((st: Obj, i: number) => {
    const rec: Obj = {
      i,
      op: st.op,
      tool: st.tool ?? null,
      args: [...(st.args ?? [])],
      dest: st.dest ?? null,
      out: st.out ?? null,
    };
    if ("ctrl" in st) rec.ctrl = [...st.ctrl];
    if (st.op === "input") {
      const lab = L.label("user", st.sens);
      vals[st.out] = lab;
      ctx = L.join(ctx, lab);
      Object.assign(rec, {
        control: null,
        effective: null,
        out_label: lab,
        decision: "allow",
        fired: [],
        ran: true,
        ctx: { ...ctx },
        why: "the user's request: (" + L.text(lab) + ")",
      });
      out.push(rec);
      return;
    }
    const ctrl: string[] = st.ctrl ?? [];
    const missing = [...(st.args as string[]), ...ctrl].filter((a) => !(a in vals));
    const control: L.Label =
      mode === "context" ? { ...ctx } : L.joinAll(ctrl.filter((a) => a in vals).map((a) => vals[a]!));
    if (missing.length > 0) {
      Object.assign(rec, {
        control,
        effective: null,
        out_label: null,
        decision: "skip",
        fired: [],
        ran: false,
        ctx: { ...ctx },
        why: "skipped: needs " + missing.join(", "),
      });
      out.push(rec);
      return;
    }
    const eff = L.joinAll([...(st.args as string[]).map((a) => vals[a]!), control]);
    const t = TOOLS[st.tool]!;
    const d = decide(st.tool, eff, st.dest ?? null, grant);
    const ran = d.decision === "allow" || (d.decision === "ask" && approve) || !enforce;
    let lab: L.Label | null = null;
    if (ran && t.kind === "read") lab = L.join(t.out, eff);
    else if (ran && t.kind === "transform") lab = eff;
    if (lab !== null && st.out) {
      vals[st.out] = lab;
      ctx = L.join(ctx, lab);
    }
    let why = explain(st.tool, eff, d);
    if (!enforce) {
      if (d.decision !== "allow") why += " (not enforced: it runs)";
    } else if (d.decision === "ask") why += approve ? " (approved)" : " (refused)";
    Object.assign(rec, {
      control,
      effective: eff,
      out_label: lab,
      decision: d.decision,
      fired: d.fired,
      ran,
      ctx: { ...ctx },
      why,
    });
    out.push(rec);
  });
  return out;
}

export function summary(steps: Obj[]): Record<string, number> {
  const s: Record<string, number> = { allow: 0, ask: 0, deny: 0, skip: 0 };
  for (const r of steps) if (r.op === "call") s[r.decision] = s[r.decision]! + 1;
  return s;
}
