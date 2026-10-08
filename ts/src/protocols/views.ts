/**
 * Animation frames for the Protocols site: a port of agent_loop_sim/protocols/views.py.
 */
import type { Tokenizer } from "../tokenizer";
import { HANDSHAKE_VERSIONS, LATEST_HANDSHAKE, MODERN_VERSIONS, compact, kindOf, type Obj } from "./mcp";

export function integrationFrames(n: number, m: number): Obj {
  const frames: Obj[] = [];
  const edges: number[][] = [];
  frames.push({ phase: "apart", p2p_edges: [], agents_wired: 0, tools_wired: 0, built: 0 });
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) edges.push([i, j]);
    frames.push({ phase: "p2p", p2p_edges: edges.map((e) => [...e]), agents_wired: i + 1, tools_wired: 0, built: edges.length });
  }
  for (let i = 0; i < n; i++) frames.push({ phase: "protocol", p2p_edges: [], agents_wired: i + 1, tools_wired: 0, built: i + 1 });
  for (let j = 0; j < m; j++) {
    frames.push({ phase: "protocol", p2p_edges: [], agents_wired: n, tools_wired: j + 1, built: n + j + 1 });
  }
  return { n, m, p2p: n * m, protocol: n + m, frames };
}

function short(v: unknown, limit = 60): string {
  const s = compact(v);
  const cps = Array.from(s);
  return cps.length <= limit ? s : cps.slice(0, limit - 1).join("") + "…";
}

export function messageLabel(msg: Obj | null, method: string): string {
  if (msg === null) return "a line that is not JSON";
  const k = kindOf(msg);
  const p: Obj = msg.params || {};
  if (k === "request") {
    const mt: string = msg.method;
    if (mt === "initialize") return "initialize (asks for " + String(p.protocolVersion) + ")";
    if (mt === "tools/call") {
      const extra = "inputResponses" in p ? " + answers" : "";
      return "tools/call " + String(p.name) + " " + short(p.arguments ?? {}, 40) + extra;
    }
    if (mt === "resources/read") return "resources/read " + String(p.uri);
    if (mt === "prompts/get") return "prompts/get " + String(p.name);
    if (mt === "elicitation/create") return "elicitation/create: " + String(p.message);
    if (mt === "sampling/createMessage") return "sampling/createMessage (maxTokens " + String(p.maxTokens) + ")";
    return mt;
  }
  if (k === "notification") {
    const mt: string = msg.method;
    if (mt === "notifications/progress") return "progress " + String(p.progress) + "/" + String(p.total);
    if (mt === "notifications/cancelled") return "cancelled (" + String(p.reason) + ")";
    return mt;
  }
  if (k === "error") {
    const e = msg.error;
    return "error " + String(e.code) + ": " + short(e.message, 50);
  }
  const r: Obj = msg.result;
  if (r.resultType === "input_required") {
    const reqs: Obj = r.inputRequests || {};
    return "input_required: " + Object.values(reqs).map((v: Obj) => v.method).join(", ");
  }
  if (method === "initialize") return "agreed " + String(r.protocolVersion);
  if (method === "server/discover") return "speaks " + (r.supportedVersions || []).join(", ");
  if (method === "tools/list" || method === "resources/list" || method === "prompts/list") {
    const key = method.split("/")[0]!;
    const nxt = "nextCursor" in r ? " (more: " + r.nextCursor + ")" : "";
    return String((r[key] || []).length) + " " + key + nxt;
  }
  if (method === "tools/call") {
    const c: Obj[] = r.content || [{}];
    const t = c.length > 0 ? (c[0]!.text ?? "") : "";
    return (r.isError ? "isError: " : "") + short(t, 40);
  }
  if (method === "resources/read") return String((r.contents || []).length) + " content block";
  if (method === "prompts/get") return String((r.messages || []).length) + " message";
  if (method === "elicitation/create") return String(r.action);
  if (method === "sampling/createMessage") {
    const c = r.content || {};
    return '"' + short(c.text ?? "", 40) + '"';
  }
  if (method === "ping") return "pong";
  return "result";
}

export function sequenceFrames(played: Obj): Obj[] {
  const out: Obj[] = [];
  const wire: Obj[] = played.wire;
  const log: Obj[] = played.log;
  for (let i = 0; i < wire.length; i++) {
    const w = wire[i]!;
    const lg = log[i]!;
    const msg: Obj | null = w.msg ?? null;
    const frm = w.dir === "c2s" ? "client" : "server";
    const to = frm === "client" ? "server" : "client";
    if (msg !== null && w.dir === "c2s") {
      let gate: Obj | null = null;
      if (kindOf(msg) === "result" && (lg.method === "elicitation/create" || lg.method === "sampling/createMessage")) {
        gate = { method: lg.method, answer: msg.result };
      } else if (kindOf(msg) === "request" && "inputResponses" in (msg.params || {})) {
        const ans: Obj = msg.params.inputResponses;
        const key = Object.keys(ans)[0]!;
        gate = { method: key === "confirm" ? "elicitation/create" : "sampling/createMessage", answer: ans[key] };
      }
      if (gate !== null) {
        const who = gate.method === "elicitation/create" ? "user" : "model";
        const ask = who === "user" ? "show the form, wait for the user" : "ask the model (the user may review)";
        const reply = who === "user" ? String(gate.answer.action) : '"' + short((gate.answer.content || {}).text ?? "", 40) + '"';
        out.push({ from: "client", to: who, label: ask, virtual: true, kind: "gate", wire: null, dir: "host", bytes: 0, client: "asking the " + who, server: lg.server });
        out.push({ from: who, to: "client", label: reply, virtual: true, kind: "gate", wire: null, dir: "host", bytes: 0, client: "has the answer", server: lg.server });
      }
    }
    out.push({
      from: frm,
      to,
      label: messageLabel(msg, lg.method),
      virtual: false,
      kind: lg.kind,
      wire: i,
      dir: w.dir,
      bytes: lg.bytes,
      client: lg.client,
      server: lg.server,
    });
  }
  out.forEach((f, k) => {
    f.step = k;
  });
  return out;
}

export const FEATURES: string[][] = [
  ["tools", "server"],
  ["resources", "server"],
  ["prompts", "server"],
  ["elicitation", "client"],
  ["sampling", "client"],
];

export function negotiation(played: Obj): Obj {
  const wire: Obj[] = played.wire;
  let clientCaps: Obj = {};
  let serverCaps: Obj = {};
  let era = "handshake";
  let asked: unknown = null;
  let agreed: unknown = null;
  let serverVersions = [...HANDSHAKE_VERSIONS];
  for (const w of wire) {
    const msg: Obj | null = w.msg ?? null;
    if (msg === null) continue;
    const p: Obj = msg.params || {};
    const meta: Obj = p._meta || {};
    if (msg.method === "initialize") {
      clientCaps = p.capabilities || {};
      asked = p.protocolVersion ?? null;
    } else if ("io.modelcontextprotocol/clientCapabilities" in meta && w.dir === "c2s") {
      era = "modern";
      clientCaps = meta["io.modelcontextprotocol/clientCapabilities"];
      asked = meta["io.modelcontextprotocol/protocolVersion"] ?? null;
    }
    if ("result" in msg) {
      const r: Obj = msg.result;
      if ("protocolVersion" in r && "serverInfo" in r) {
        serverCaps = r.capabilities || {};
        agreed = r.protocolVersion;
      }
      if ("supportedVersions" in r) {
        serverCaps = r.capabilities || {};
        serverVersions = [...r.supportedVersions];
      }
    }
  }
  const clientVersions = [...HANDSHAKE_VERSIONS, ...MODERN_VERSIONS];
  const discovered = era === "handshake" || Object.keys(serverCaps).length > 0;
  if (era === "modern") {
    serverVersions = [...MODERN_VERSIONS];
    agreed = asked;
  }
  const both = clientVersions.filter((v) => serverVersions.includes(v));
  const rows: Obj[] = [];
  for (const [name, side] of FEATURES) {
    const c = name! in clientCaps;
    const s = name! in serverCaps;
    let usable: boolean | null = side === "server" ? s : c;
    if (side === "server" && !discovered) usable = null;
    rows.push({ feature: name, provided_by: side, client: c, server: s, usable });
  }
  return {
    era,
    asked,
    agreed: agreed !== null ? agreed : LATEST_HANDSHAKE,
    client_versions: clientVersions,
    server_versions: serverVersions,
    both,
    client_caps: Object.keys(clientCaps).sort(),
    server_caps: Object.keys(serverCaps).sort(),
    discovered,
    features: rows,
  };
}

export function journey(played: Obj, callIndex = 0): Obj[] {
  const wire: Obj[] = played.wire;
  const calls: number[] = [];
  wire.forEach((w, i) => {
    if ((w.msg ?? null) !== null && w.dir === "c2s" && w.msg.method === "tools/call") calls.push(i);
  });
  const ci = calls[callIndex]!;
  const req: Obj = wire[ci]!.msg;
  let resp: Obj | null = null;
  for (const w of wire.slice(ci + 1)) {
    const m: Obj = w.msg || {};
    if (w.dir === "s2c" && m.id === req.id && !("method" in m)) {
      resp = m;
      break;
    }
  }
  const name = req.params.name;
  const args = req.params.arguments;
  const content = ((resp || {}).result || {}).content ?? [{}];
  const text = content[0].text ?? "";
  const use = { type: "tool_use", name, input: args };
  return [
    { at: "model", to: "host", label: "the model asks for " + name, payload: compact(use), wire: false },
    { at: "host", to: "client", label: "the host finds which server offers " + name, payload: name, wire: false },
    { at: "client", to: "server", label: "JSON-RPC request", payload: compact(req), wire: true },
    { at: "server", to: "server", label: "the server runs " + name, payload: compact(args), wire: false },
    { at: "server", to: "client", label: "JSON-RPC result", payload: compact(resp), wire: true },
    { at: "client", to: "host", label: "the client returns the content", payload: text, wire: false },
    { at: "host", to: "model", label: "the result joins the model's context", payload: text, wire: false },
  ];
}

export function threeWays(played: Obj, tok: Tokenizer): Obj[] {
  const wire: Obj[] = played.wire;
  const log: Obj[] = played.log;
  const rows: Obj[] = [];
  const spec = [
    ["tool", "tools/call", "model", "tools/list"],
    ["resource", "resources/read", "application", "resources/list"],
    ["prompt", "prompts/get", "user", "prompts/list"],
  ];
  for (const [prim, method, who, lister] of spec) {
    const seqs: number[] = [];
    let text = "";
    wire.forEach((w, i) => {
      const lg = log[i]!;
      if (lg.method === method || lg.method === lister) {
        seqs.push(i);
        const m: Obj = w.msg || {};
        if (lg.method === method && "result" in m) {
          const r = m.result;
          if (method === "tools/call") text = r.content[0].text;
          else if (method === "resources/read") text = r.contents[0].text;
          else text = r.messages[0].content.text;
        }
      }
    });
    let bytes = 0;
    for (const i of seqs) bytes += log[i]!.bytes;
    rows.push({ primitive: prim, method, controlled_by: who, messages: seqs, text, tokens: tok.encode(text).length, bytes });
  }
  return rows;
}

// --- added in engine 1.3.0: agent to agent, and step-by-step flows (OAuth, attacks) -----------

const unquote = (s: string) => s.replace(/^"+|"+$/g, "");

function a2aLabel(w: Obj): string {
  if ("http" in w) {
    const h = w.http;
    if ("method" in h) return "GET /.well-known/agent-card.json";
    const b = h.body;
    return "agent card: " + String(b.name) + ", " + String((b.skills as Obj[]).length) + " skills" + (b.capabilities.streaming ? ", streams" : "");
  }
  const msg = w.msg;
  if ("method" in msg) {
    const p: Obj = msg.params ?? {};
    const m = p.message;
    if (m !== null && typeof m === "object" && !Array.isArray(m)) {
      const text = ((m.parts ?? []) as Obj[]).map((x) => x.text ?? "").join(" ");
      return msg.method + ' "' + unquote(short(text, 30)) + '"' + ("taskId" in m ? " (same task)" : "");
    }
    return msg.method;
  }
  if ("error" in msg) {
    const e = msg.error;
    return String(e.code) + " " + unquote(short(e.message, 48));
  }
  const r = msg.result;
  if ("statusUpdate" in r) return "statusUpdate " + (r.statusUpdate.status.state as string).replace("TASK_STATE_", "");
  if ("artifactUpdate" in r) {
    const a = r.artifactUpdate;
    return "artifactUpdate" + (a.append ? " (append" + (a.lastChunk ? ", last" : "") + ")" : " (first chunk)");
  }
  if ("message" in r) return 'Message "' + unquote(short(r.message.parts[0].text ?? "", 28)) + '"';
  const t = "task" in r ? r.task : r;
  let s = "Task " + (t.status.state as string).replace("TASK_STATE_", "");
  if (t.artifacts && (t.artifacts as Obj[]).length > 0) s += " + artifact";
  return s;
}

export function a2aFrames(played: Obj): Obj[] {
  const out: Obj[] = [];
  (played.wire as Obj[]).forEach((w, i) => {
    const lg = played.log[i];
    if (w.dir === "host") {
      out.push({ from: "client", to: "user", label: "ask the user to sign in", virtual: true, kind: "gate", wire: null, dir: "host", bytes: 0, sse: false, t: lg.t, client: "asking the user", server: lg.server, task: lg.task });
      out.push({ from: "user", to: "client", label: "signed in (outside A2A)", virtual: true, kind: "gate", wire: null, dir: "host", bytes: 0, sse: false, t: lg.t, client: lg.client, server: lg.server, task: lg.task });
      return;
    }
    const c2s = w.dir === "c2s";
    let kind = lg.kind as string;
    if (kind === "http") kind = c2s ? "request" : "result";
    out.push({ from: c2s ? "client" : "server", to: c2s ? "server" : "client", label: a2aLabel(w), virtual: false, kind, wire: i, dir: w.dir, bytes: lg.bytes, sse: Boolean(w.sse), t: lg.t, client: lg.client, server: lg.server, task: lg.task });
  });
  out.forEach((f, k) => (f.step = k));
  return out;
}

export function a2aSummary(played: Obj): Obj {
  let msgs = 0;
  let nbytes = 0;
  let first: number | null = null;
  const states: string[] = [];
  (played.wire as Obj[]).forEach((w, i) => {
    const lg = played.log[i];
    if (w.dir === "host") return;
    msgs += 1;
    nbytes += lg.bytes as number;
    if (w.dir === "s2c" && "msg" in w && "result" in w.msg && first === null) {
      const r = w.msg.result;
      const t = r !== null && typeof r === "object" ? r.task : undefined;
      if ("artifactUpdate" in r || (t !== null && typeof t === "object" && t.artifacts && (t.artifacts as Obj[]).length > 0)) first = lg.t;
    }
    if (lg.task !== null && (states.length === 0 || states[states.length - 1] !== lg.task)) states.push(lg.task);
  });
  const last = played.log[played.log.length - 1].t;
  return { messages: msgs, bytes: nbytes, first_result_ms: first, end_ms: last, states };
}

export function flowFrames(run: Obj): Obj[] {
  const out: Obj[] = [];
  for (const s of run.steps as Obj[]) {
    const k = s.kind;
    let kind: string;
    let virtual: boolean;
    if (k === "http") {
      const st = s.status;
      kind = st === null ? "request" : st >= 400 ? "error" : "result";
      virtual = false;
    } else if (k === "check") {
      kind = s.ok ? "check" : "fail";
      virtual = true;
    } else if (k === "attack") {
      kind = "attack";
      virtual = true;
    } else {
      kind = "gate";
      virtual = true;
    }
    out.push({ from: s.from, to: s.to, label: s.label, kind, virtual, seq: s.seq, ok: s.ok, step: out.length });
  }
  return out;
}
