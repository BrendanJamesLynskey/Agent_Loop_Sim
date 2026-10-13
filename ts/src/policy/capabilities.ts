/**
 * Capabilities (engine 1.10.0): least privilege for a plan, the trifecta and an exhaustive search for a
 * flow out, and audience-bound tokens. A port of agent_loop_sim/policy/capabilities.py.
 */
import type { Obj } from "../data";
import { DEST_ORDER, TOOL_ORDER, TOOLS } from "./catalogue";
import * as L from "./labels";

export function needed(plan: Obj, privateDests: string[] | null = null): Obj {
  const tools: string[] = [];
  const dests: string[] = [];
  for (const st of plan.steps) {
    if (st.op !== "call") continue;
    if (!tools.includes(st.tool)) tools.push(st.tool);
    if (st.dest && !dests.includes(st.dest)) dests.push(st.dest);
  }
  return {
    tools: TOOL_ORDER.filter((t) => tools.includes(t)),
    scopes: tools
      .filter((t) => TOOLS[t]!.scope)
      .map((t) => TOOLS[t]!.scope as string)
      .sort(),
    egress: DEST_ORDER.filter((d) => dests.includes(d)),
    private_dests: [...(privateDests ?? ["self"])],
  };
}

export function overGrant(grant: Obj, plan: Obj): Record<string, string[]> {
  const n = needed(plan);
  const out: Record<string, string[]> = {};
  for (const k of ["tools", "scopes", "egress"]) out[k] = (grant[k] as string[]).filter((x) => !n[k].includes(x));
  return out;
}

export function circles(grant: Obj): Record<string, string[]> {
  const out: Record<string, string[]> = { private: [], untrusted: [], outbound: [] };
  for (const t of TOOL_ORDER) {
    if (!grant.tools.includes(t)) continue;
    const spec = TOOLS[t]!;
    if (spec.kind === "read" && spec.out.sens === "private") out.private!.push(t);
    if (spec.kind === "read" && spec.out.src === "untrusted") out.untrusted!.push(t);
    if (spec.egress && (spec.dests as string[]).some((d) => grant.egress.includes(d) && !grant.private_dests.includes(d)))
      out.outbound!.push(t);
  }
  return out;
}

export function trifecta(grant: Obj): Obj {
  const c = circles(grant);
  return { ...c, all: c.private!.length > 0 && c.untrusted!.length > 0 && c.outbound!.length > 0 };
}

export function exposure(grant: Obj, maxLen = 3): Obj {
  const calls = TOOL_ORDER.filter((t) => grant.tools.includes(t) && TOOLS[t]!.kind === "read");
  const outs: [string, string][] = [];
  for (const t of TOOL_ORDER) {
    if (!grant.tools.includes(t) || !TOOLS[t]!.egress) continue;
    for (const d of DEST_ORDER)
      if (TOOLS[t]!.dests.includes(d) && grant.egress.includes(d) && !grant.private_dests.includes(d)) outs.push([t, d]);
  }
  let checked = 0;
  for (let n = 0; n < maxLen; n++) {
    let seqs: string[][] = [[]];
    for (let k = 0; k < n; k++) {
      const next: string[][] = [];
      for (const s of seqs) for (const t of calls) next.push([...s, t]);
      seqs = next;
    }
    for (const s of seqs) {
      let ctx: L.Label = { ...L.BOTTOM };
      for (const t of s) ctx = L.join(ctx, L.join(TOOLS[t]!.out, ctx));
      for (const [t, d] of outs) {
        checked += 1;
        if (ctx.src === "untrusted" && ctx.sens === "private") return { exposed: true, path: [...s, t], dest: d, checked };
      }
    }
  }
  return { exposed: false, path: [], dest: null, checked };
}

export function mint(grant: Obj): Record<string, Obj> {
  const toks: Record<string, Obj> = {};
  for (const t of TOOL_ORDER) {
    const spec = TOOLS[t]!;
    if (!grant.tools.includes(t) || spec.aud === null || !grant.scopes.includes(spec.scope)) continue;
    const tok = (toks[spec.aud] ??= { aud: spec.aud, scopes: [] });
    if (!tok.scopes.includes(spec.scope)) tok.scopes.push(spec.scope);
  }
  const out: Record<string, Obj> = {};
  for (const k of Object.keys(toks).sort()) {
    toks[k]!.scopes.sort();
    out[k] = toks[k]!;
  }
  return out;
}

export function checkToken(token: Obj | null | undefined, tool: string): Obj {
  const spec = TOOLS[tool]!;
  if (spec.aud === null) return { ok: true, reason: `${tool} runs in the model; no token needed` };
  if (!token) return { ok: false, reason: `no token for ${spec.aud}` };
  if (token.aud !== spec.aud) return { ok: false, reason: `the token is for ${token.aud}, not ${spec.aud}: rejected` };
  if (!token.scopes.includes(spec.scope)) return { ok: false, reason: `the token lacks ${spec.scope}` };
  return { ok: true, reason: `token for ${spec.aud} holds ${spec.scope}` };
}
