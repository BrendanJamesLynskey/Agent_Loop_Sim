/**
 * An MCP gateway multiplexing several downstream servers (2026-07-28 messages), with three
 * naming policies and the token cost of the merged tool list: a port of
 * agent_loop_sim/protocols/gateway.py, statement for statement.
 */
import data from "../protocols_data.json";
import { dumps } from "../jsonfmt";
import type { Tokenizer } from "../tokenizer";
import { LATEST, META_CLIENT_CAPS, META_CLIENT_INFO, META_SERVER_INFO, META_VERSION, clone, compact, utf8Len, type Obj } from "./mcp";

export const POLICIES = ["flat", "prefix", "filtered"];
export const SERVERS: Obj[] = data.gateway.servers as Obj[];
export const CALL: Obj = data.gateway.call as Obj;
export const ALLOW: string[] = data.gateway.allow as string[];
export const GATEWAY_INFO = { name: "gateway", version: "1.0.0" };
export const CLIENT_INFO = { name: "example-host", version: "1.0.0" };

function meta(): Obj {
  return { [META_VERSION]: LATEST, [META_CLIENT_INFO]: { ...CLIENT_INFO }, [META_CLIENT_CAPS]: {} };
}

function listResult(id: number, tools: Obj[], info: Obj): Obj {
  return { jsonrpc: "2.0", id, result: { tools, cacheScope: "private", ttlMs: 0, resultType: "complete", _meta: { [META_SERVER_INFO]: { ...info } } } };
}

export function merge(policy: string): Obj {
  if (!POLICIES.includes(policy)) throw new Error("unknown policy " + policy);
  const rows: Obj[] = [];
  const owner: Record<string, string> = {};
  const offered = new Map<string, string[]>();
  for (const s of SERVERS) {
    for (const t of s.tools as Obj[]) {
      if (!offered.has(t.name)) offered.set(t.name, []);
      offered.get(t.name)!.push(s.alias);
      const exposed = policy === "flat" ? (t.name as string) : s.alias + "." + t.name;
      let status = "exposed";
      if (policy === "flat" && exposed in owner) status = "shadowed";
      else if (policy === "filtered" && !ALLOW.includes(exposed)) status = "filtered";
      if (status === "exposed") owner[exposed] = s.alias;
      rows.push({ alias: s.alias, tool: t.name, exposed, status, shadowed_by: status === "shadowed" ? owner[exposed] : null });
    }
  }
  const collisions: unknown[] = [];
  for (const [n, a] of offered) if (a.length > 1) collisions.push([n, [...a]]);
  return { policy, rows, collisions, owner };
}

export function gatewayRun(policy: string, tok: Tokenizer): Obj {
  const m = merge(policy);
  const frames: Obj[] = [];
  const wire: Obj[] = [];
  const add = (frm: string, to: string, label: string, kind: string, msg: Obj | null, note = "", virtual = false) => {
    const nbytes = msg === null ? 0 : utf8Len(compact(msg));
    if (msg !== null) wire.push({ from: frm, to, msg });
    frames.push({ from: frm, to, label, kind, virtual, wire: msg === null ? null : wire.length - 1, bytes: nbytes, note });
  };
  add("client", "gateway", "tools/list", "request", { jsonrpc: "2.0", id: 1, method: "tools/list", params: { _meta: meta() } });
  const perServer: Obj[] = [];
  let nid = 1;
  for (const s of SERVERS) {
    nid += 1;
    add("gateway", s.alias, "tools/list", "request", { jsonrpc: "2.0", id: nid, method: "tools/list", params: { _meta: meta() } });
  }
  SERVERS.forEach((s, i) => {
    const tokens = tok.count(dumps(s.tools));
    perServer.push({ alias: s.alias, tools: (s.tools as Obj[]).length, tokens });
    add(s.alias, "gateway", String((s.tools as Obj[]).length) + " tools (" + String(tokens) + " tokens)", "result", listResult(i + 2, clone(s.tools), s.info));
  });
  const merged: Obj[] = [];
  for (const r of m.rows as Obj[]) {
    if (r.status !== "exposed") continue;
    for (const s of SERVERS) {
      if (s.alias === r.alias) {
        for (const t of s.tools as Obj[]) {
          if (t.name === r.tool) {
            const d = clone(t);
            d.name = r.exposed;
            merged.push(d);
          }
        }
      }
    }
  }
  const shadowed = (m.rows as Obj[]).filter((r) => r.status === "shadowed");
  const filtered = (m.rows as Obj[]).filter((r) => r.status === "filtered");
  let note = "";
  if (shadowed.length > 0) note = String(shadowed.length) + " tools shadowed by a same-named tool";
  else if (filtered.length > 0) note = String(filtered.length) + " tools filtered out";
  add("gateway", "gateway", "merge: " + String(merged.length) + " tools" + (note ? " (" + note + ")" : ""), "check", null, note, true);
  const mergedTokens = tok.count(dumps(merged));
  add("gateway", "client", String(merged.length) + " tools (" + String(mergedTokens) + " tokens)", "result", listResult(1, merged, GATEWAY_INFO));
  const want = CALL.alias + "." + CALL.tool;
  const name = policy === "flat" ? (CALL.tool as string) : want;
  add("model", "client", "call " + name, "gate", null, "the model picks a tool by name", true);
  add("client", "gateway", "tools/call " + name, "request", { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name, arguments: clone(CALL.arguments), _meta: meta() } });
  const target = (m.owner as Record<string, string>)[name] ?? null;
  const routedRight = target === CALL.alias;
  add("gateway", "gateway", "route " + name + " → " + String(target) + (routedRight ? " (as intended)" : " (the model meant " + CALL.alias + ")"), routedRight ? "check" : "error", null, "", true);
  add("gateway", String(target), "tools/call " + CALL.tool, "request", { jsonrpc: "2.0", id: nid + 1, method: "tools/call", params: { name: CALL.tool, arguments: clone(CALL.arguments), _meta: meta() } });
  const text = routedRight ? "3 web results for \"" + CALL.arguments.query + "\"" : "No matches for /" + CALL.arguments.query + "/ in the project";
  const res = { content: [{ type: "text", text }], isError: false, resultType: "complete" };
  add(String(target), "gateway", "result: " + text, "result", { jsonrpc: "2.0", id: nid + 1, result: res });
  add("gateway", "client", "result: " + text, "result", { jsonrpc: "2.0", id: 2, result: clone(res) });
  frames.forEach((f, k) => (f.step = k));
  let direct = 0;
  for (const p of perServer) direct += p.tokens as number;
  return { policy, merge: m, per_server: perServer, direct_tokens: direct, merged_tokens: mergedTokens, merged_tools: merged.length, exposed: merged.map((t) => t.name), routed_to: target, routed_right: routedRight, frames, wire };
}
