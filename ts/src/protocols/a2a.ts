/**
 * Agent to agent: A2A 1.0 (specification release 1.0.1) as an orchestrator (client) and a
 * remote research agent (server) over the JSON-RPC binding with SSE: a port of
 * agent_loop_sim/protocols/a2a.py, statement for statement.
 */
import data from "../protocols_data.json";
import { Rng } from "../rng";
import { clone, compact, utf8Len, type Obj } from "./mcp";
import { hex, sha256, utf8Bytes } from "./sha256";

export const VERSION = "1.0";
export const SPEC_RELEASE = "1.0.1";
export const AGENT = "https://research.example.com";
export const RPC_URL = AGENT + "/a2a";
export const CARD_PATH = "/.well-known/agent-card.json";

export const SUBMITTED = "TASK_STATE_SUBMITTED";
export const WORKING = "TASK_STATE_WORKING";
export const COMPLETED = "TASK_STATE_COMPLETED";
export const FAILED = "TASK_STATE_FAILED";
export const CANCELED = "TASK_STATE_CANCELED";
export const INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED";
export const REJECTED = "TASK_STATE_REJECTED";
export const AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED";
export const TERMINAL = [COMPLETED, FAILED, CANCELED, REJECTED];
export const INTERRUPTED = [INPUT_REQUIRED, AUTH_REQUIRED];
export const STATES: string[] = data.a2a.states as string[];
export const TRANSITIONS: string[][] = data.a2a.transitions as string[][];
export const TIMES: Record<string, number> = data.a2a.times as Record<string, number>;
export const CARD: Obj = data.a2a.card as Obj;
export const ERRORS: Record<string, number> = data.a2a.errors as Record<string, number>;
export const A2A_SCENARIOS: Record<string, Obj> = data.a2a.scenarios as Record<string, Obj>;
export const SUMMARY_BODY = "three decisions, two actions, no blockers.";
export const CALENDAR_BODY = "design review at 10:00, one-to-one at 15:00.";

/** json.dumps(x, sort_keys=True, separators=(",", ":")) for ASCII JSON without floats. */
export function sortedCompact(x: unknown): string {
  if (x !== null && typeof x === "object" && !Array.isArray(x)) {
    const o = x as Obj;
    return "{" + Object.keys(o).sort().map((k) => JSON.stringify(k) + ":" + sortedCompact(o[k])).join(",") + "}";
  }
  if (Array.isArray(x)) return "[" + x.map((v) => sortedCompact(v)).join(",") + "]";
  return compact(x);
}

export function cardEtag(card: Obj): string {
  const unsigned: Obj = {};
  for (const [k, v] of Object.entries(card)) if (k !== "signatures") unsigned[k] = v;
  return 'W/"' + hex(sha256(utf8Bytes(sortedCompact(unsigned)))) + '"';
}

export function uuid4(rng: Rng): string {
  const b: number[] = [];
  for (let i = 0; i < 16; i++) b.push(rng.randint(0, 255));
  b[6] = (b[6]! & 0x0f) | 0x40;
  b[8] = (b[8]! & 0x3f) | 0x80;
  const h = b.map((x) => x.toString(16).padStart(2, "0")).join("");
  return h.slice(0, 8) + "-" + h.slice(8, 12) + "-" + h.slice(12, 16) + "-" + h.slice(16, 20) + "-" + h.slice(20, 32);
}

const p2 = (n: number, w = 2) => String(n).padStart(w, "0");

export function timestamp(ms: number): string {
  const t = 36000000 + ms;
  const h = Math.floor(t / 3600000);
  const m = Math.floor(t / 60000) % 60;
  const s = Math.floor(t / 1000) % 60;
  return "2026-10-08T" + p2(h) + ":" + p2(m) + ":" + p2(s) + "." + p2((t % 1000) * 1000, 6) + "Z";
}

export function textOf(msg: Obj): string {
  return ((msg.parts ?? []) as Obj[]).filter((p) => "text" in p).map((p) => p.text as string).join("\n");
}

export function rpcRequest(id: unknown, method: string, params: Obj | null): Obj {
  const m: Obj = { jsonrpc: "2.0", id, method };
  if (params !== null) m.params = params;
  return m;
}

export function rpcResult(id: unknown, result: unknown): Obj {
  return { jsonrpc: "2.0", id, result };
}

export function errorInfo(reason: string, metadata: Obj | null = null): Obj {
  return { "@type": "type.googleapis.com/google.rpc.ErrorInfo", reason, domain: "a2a-protocol.org", metadata: metadata !== null ? metadata : {} };
}

export function rpcError(id: unknown, code: number, message: string, dataValue: Obj[] | null = null): Obj {
  const e: Obj = { code, message };
  if (dataValue !== null) e.data = dataValue;
  return { jsonrpc: "2.0", id, error: e };
}

const strip = (e: Obj): Obj => {
  const o: Obj = {};
  for (const [k, v] of Object.entries(e)) if (k !== "_t") o[k] = v;
  return o;
};

export class Agent {
  rng: Rng;
  tasks = new Map<string, Obj>();
  state = "idle";
  clock = 0;

  constructor(seed = 23) {
    this.rng = new Rng(seed);
  }

  snapshot(task: Obj): Obj {
    const out: Obj = { id: task.id, contextId: task.contextId };
    const st: Obj = { state: task.status.state };
    if ("message" in task.status) st.message = clone(task.status.message);
    if ("timestamp" in task.status) st.timestamp = task.status.timestamp;
    out.status = st;
    if ((task.artifacts as Obj[]).length > 0) out.artifacts = clone(task.artifacts);
    out.history = clone(task.history);
    return out;
  }

  archive(task: Obj): void {
    if ("message" in task.status) {
      (task.history as Obj[]).push(task.status.message);
      delete task.status.message;
    }
  }

  agentMessage(task: Obj, text: string): Obj {
    return { messageId: uuid4(this.rng), contextId: task.contextId, taskId: task.id, role: "ROLE_AGENT", parts: [{ text }] };
  }

  status(task: Obj, state: string, text: string | null, dt: number): Obj {
    this.clock += dt;
    this.archive(task);
    const st: Obj = { state };
    if (text !== null) st.message = this.agentMessage(task, text);
    st.timestamp = timestamp(this.clock);
    task.status = st;
    this.state = "task " + state.replace("TASK_STATE_", "").toLowerCase().replaceAll("_", " ");
    return { statusUpdate: { taskId: task.id, contextId: task.contextId, status: clone(st) }, _t: this.clock };
  }

  artifact(task: Obj, aid: string, text: string, append: boolean, last: boolean): Obj {
    this.clock += TIMES.chunk!;
    if (append) {
      for (const a of task.artifacts as Obj[]) if (a.artifactId === aid) (a.parts as Obj[]).push({ text });
    } else {
      (task.artifacts as Obj[]).push({ artifactId: aid, name: "summary", parts: [{ text }] });
    }
    const ev: Obj = { taskId: task.id, contextId: task.contextId, artifact: { artifactId: aid, name: "summary", parts: [{ text }] } };
    if (append) ev.append = true;
    if (last) ev.lastChunk = true;
    this.state = "streaming its artifact";
    return { artifactUpdate: ev, _t: this.clock };
  }

  finish(task: Obj, head: string, body: string): Obj[] {
    const aid = uuid4(this.rng);
    return [this.artifact(task, aid, head, false, false), this.artifact(task, aid, body, true, true), this.status(task, COMPLETED, null, TIMES.decide!)];
  }

  execute(msg: Obj, task: Obj | null): Obj[] {
    const text = textOf(msg);
    if (task === null) {
      const ctx = uuid4(this.rng);
      if (text.startsWith("hello")) {
        this.clock += TIMES.decide!;
        this.state = "answered with a message";
        return [{ message: { messageId: uuid4(this.rng), contextId: ctx, role: "ROLE_AGENT", parts: [{ text: "Hello. Ask me to summarise something." }] }, _t: this.clock }];
      }
      const tid = uuid4(this.rng);
      const m: Obj = { messageId: msg.messageId, contextId: ctx, taskId: tid, role: msg.role, parts: clone(msg.parts) };
      const t: Obj = { id: tid, contextId: ctx, status: { state: SUBMITTED }, artifacts: [], history: [m] };
      this.tasks.set(tid, t);
      this.clock += TIMES.decide!;
      this.state = "task submitted";
      const evs: Obj[] = [{ task: this.snapshot(t), _t: this.clock }];
      if (text.startsWith("delete")) {
        evs.push(this.status(t, REJECTED, "I only summarise; I do not delete files.", TIMES.decide!));
        return evs;
      }
      evs.push(this.status(t, WORKING, null, TIMES.start!));
      if (text === "summarise the meeting") {
        evs.push(this.status(t, INPUT_REQUIRED, "Which meeting: Monday's or Tuesday's?", TIMES.decide!));
        return evs;
      }
      if (text === "check my calendar") {
        evs.push(this.status(t, AUTH_REQUIRED, "Sign in to the calendar first: https://calendar.example.com/authorize", TIMES.decide!));
        return evs;
      }
      const subject = text.startsWith("summarise ") ? text.slice("summarise ".length) : text;
      return evs.concat(this.finish(t, "Summary of " + subject + ": ", SUMMARY_BODY));
    }
    const was = task.status.state;
    const evs = [this.status(task, WORKING, null, TIMES.start!)];
    if (was === AUTH_REQUIRED) return evs.concat(this.finish(task, "Today: ", CALENDAR_BODY));
    return evs.concat(this.finish(task, "Summary of " + text + "'s meeting: ", SUMMARY_BODY));
  }

  card(): Obj {
    this.state = "served its card";
    return { status: 200, headers: [["Content-Type", "application/json"], ["ETag", cardEtag(CARD)]], body: clone(CARD) };
  }

  handle(req: Obj, headers: string[][], at: number): Obj[] {
    this.clock = Math.max(this.clock, at);
    const id = req.id;
    const method = req.method;
    const params: Obj = req.params ?? {};
    let version = "";
    for (const [k, v] of headers) if (k!.toLowerCase() === "a2a-version") version = v!;
    if (version === "") version = "0.3";
    const out = (m: Obj): Obj[] => {
      this.clock += 1;
      return [{ ...m, _t: this.clock }];
    };
    if (version !== VERSION) {
      this.state = "refused the version";
      return out(rpcError(id, -32009, "A2A version '" + version + "' is not supported by this handler. " + "Expected version '" + VERSION + "'.", [errorInfo("VERSION_NOT_SUPPORTED")]));
    }
    if (method === "SendMessage" || method === "SendStreamingMessage") {
      const msg: Obj = params.message ?? {};
      if (((msg.parts ?? []) as Obj[]).length === 0) {
        this.state = "refused the request";
        const problem = "Field must contain at least one element.";
        return out(rpcError(id, -32602, "Validation failed", [
          errorInfo("INVALID_PARAMS", { errors: [{ field: "message.parts", message: problem }] }),
          { "@type": "type.googleapis.com/google.rpc.BadRequest", fieldViolations: [{ field: "message.parts", description: problem }] },
        ]));
      }
      let task: Obj | null = null;
      if ("taskId" in msg) {
        task = this.tasks.get(msg.taskId) ?? null;
        if (task === null) return out(rpcError(id, -32001, "Task not found", [errorInfo("TASK_NOT_FOUND")]));
        if (TERMINAL.includes(task.status.state)) {
          this.state = "refused the message";
          return out(rpcError(id, -32004, "Task " + task.id + " is in terminal state: " + task.status.state, [errorInfo("UNSUPPORTED_OPERATION")]));
        }
        this.archive(task);
        (task.history as Obj[]).push(clone(msg));
      }
      const evs = this.execute(msg, task);
      if (method === "SendStreamingMessage") {
        const res: Obj[] = [];
        for (const e of evs) res.push({ ...rpcResult(id, strip(e)), _t: e._t });
        return res;
      }
      const last = evs[evs.length - 1]!._t;
      const first = strip(evs[0]!);
      if ("message" in first) return [{ ...rpcResult(id, first), _t: last }];
      const tid = task !== null ? task.id : first.task.id;
      return [{ ...rpcResult(id, { task: this.snapshot(this.tasks.get(tid)!) }), _t: last }];
    }
    if (method === "GetTask") {
      const task = this.tasks.get(params.id ?? "");
      if (task === undefined) {
        this.state = "has no such task";
        return out(rpcError(id, -32001, "Task not found", [errorInfo("TASK_NOT_FOUND")]));
      }
      return out(rpcResult(id, this.snapshot(task)));
    }
    if (method === "CancelTask") {
      const task = this.tasks.get(params.id ?? "");
      if (task === undefined) return out(rpcError(id, -32001, "Task not found", [errorInfo("TASK_NOT_FOUND")]));
      if (TERMINAL.includes(task.status.state)) {
        this.state = "cannot cancel";
        return out(rpcError(id, -32002, "Task cannot be canceled", [errorInfo("TASK_NOT_CANCELABLE")]));
      }
      task.status.state = CANCELED;
      this.state = "task canceled";
      return out(rpcResult(id, this.snapshot(task)));
    }
    if (method === "SubscribeToTask") {
      const task = this.tasks.get(params.id ?? "");
      if (task === undefined) return out(rpcError(id, -32001, "Task not found", [errorInfo("TASK_NOT_FOUND")]));
      if (TERMINAL.includes(task.status.state)) {
        this.state = "refused to subscribe";
        return out(rpcError(id, -32004, "Task " + task.id + " is in terminal state: " + task.status.state, [errorInfo("UNSUPPORTED_OPERATION")]));
      }
      return out(rpcResult(id, { task: this.snapshot(task) }));
    }
    if (method === "GetExtendedAgentCard") {
      this.state = "has no extended card";
      return out(rpcError(id, -32004, "The agent does not support authenticated extended cards", [errorInfo("UNSUPPORTED_OPERATION")]));
    }
    this.state = "does not know the method";
    return out(rpcError(id, -32601, "Method not found"));
  }
}

export type Send = (kind: string, req: Obj | null, headers: string[][], at: number) => Obj;

export function engineSend(agent: Agent): Send {
  return (kind, req, headers, at) => {
    if (kind === "card") return { ...agent.card(), _t: at + 1 };
    const res = agent.handle(req!, headers, at);
    const stream = (req!.method === "SendStreamingMessage" || req!.method === "SubscribeToTask") && "result" in res[0]!;
    return { messages: res, sse: stream };
  };
}

export function shortState(s: string): string {
  return s.replace("TASK_STATE_", "");
}

export function play(sc: Obj, send: Send | null = null): Obj {
  const agent = new Agent();
  const tx = send !== null ? send : engineSend(agent);
  const rng = new Rng(11);
  const wire: Obj[] = [];
  const log: Obj[] = [];
  const cl: Obj = { state: "idle", task: null, context: null, task_state: null, card: null, next_id: 1, clock: 0 };
  const serverState = () => (send === null ? agent.state : "(the SDK agent)");

  const record = (item: Obj, kind: string, method: string, id: unknown, nbytes: number, note: string, t: number) => {
    wire.push(item);
    log.push({ seq: log.length, dir: item.dir, kind, method, id, bytes: nbytes, note, t, client: cl.state, server: serverState(), task: cl.task_state });
  };

  const track = (m: Obj) => {
    const r = m.result;
    if (r === null || typeof r !== "object" || Array.isArray(r)) return;
    const task = "task" in r ? r.task : "status" in r && "id" in r ? r : null;
    if (task !== null) {
      cl.task = task.id;
      cl.context = task.contextId;
      cl.task_state = task.status.state;
    } else if ("statusUpdate" in r) {
      cl.task_state = r.statusUpdate.status.state;
    } else if ("message" in r) {
      cl.context = r.message.contextId ?? null;
    }
  };

  const call = (method: string, params: Obj | null, note: string, headers: string[][] | null = null): Obj[] => {
    const id = cl.next_id;
    cl.next_id += 1;
    const req = rpcRequest(id, method, params);
    const hs = headers !== null ? headers : [["Content-Type", "application/json"], ["A2A-Version", VERSION]];
    cl.state = "waiting for " + method;
    const t0 = cl.clock;
    record({ dir: "c2s", msg: req, headers: hs }, "request", method, id, utf8Len(compact(req)), note, t0);
    const r = tx("rpc", req, hs, t0 + TIMES.net!);
    const msgs = r.messages as Obj[];
    for (const m of msgs) {
      const t = m._t + TIMES.net!;
      const body = strip(m);
      track(body);
      const kind = "error" in body ? "error" : "result";
      let text: string;
      if (r.sse) {
        cl.state = "reading the stream";
        text = "data: " + compact(body) + "\n\n";
      } else {
        cl.state = "has the answer";
        text = compact(body);
      }
      if (r.sse && kind === "result" && (TERMINAL.includes(cl.task_state) || "message" in body.result)) cl.state = "stream closed";
      record({ dir: "s2c", msg: body, sse: r.sse }, kind, method, id, utf8Len(text), "", t);
      cl.clock = t;
    }
    cl.state = cl.task_state === null ? "ready" : "task " + shortState(cl.task_state).toLowerCase().replaceAll("_", " ");
    return msgs.map(strip);
  };

  const message = (text: string, follow: boolean): Obj => {
    const m: Obj = { messageId: uuid4(rng), role: "ROLE_USER", parts: [{ text }] };
    if (follow) {
      m.taskId = cl.task;
      m.contextId = cl.context;
    }
    return m;
  };

  for (const st of sc.steps as Obj[]) {
    const op = st.op;
    if (op === "card") {
      cl.state = "discovering";
      const url = AGENT + CARD_PATH;
      const reqText = "GET " + CARD_PATH + " HTTP/1.1";
      record({ dir: "c2s", http: { method: "GET", url } }, "http", "agent card", null, utf8Len(reqText), "the client reads the agent card", cl.clock);
      const r = tx("card", null, [], cl.clock + TIMES.net!);
      const body = r.body;
      const iface = body.supportedInterfaces[0];
      const ok = iface.protocolBinding === "JSONRPC" && iface.protocolVersion === VERSION;
      cl.card = { name: body.name, url: iface.url, binding: iface.protocolBinding, version: iface.protocolVersion, streaming: Boolean(body.capabilities.streaming), skills: (body.skills as Obj[]).map((s) => s.id), usable: ok };
      cl.state = ok ? "has the card" : "cannot use this agent";
      const t = r._t + TIMES.net!;
      record({ dir: "s2c", http: { status: r.status, headers: r.headers, body } }, "http", "agent card", null, utf8Len(compact(body)), "", t);
      cl.clock = t;
    } else if (op === "send" || op === "reply") {
      const follow = op === "reply";
      cl.clock += TIMES.model!;
      const p: Obj = { message: message(st.text, follow) };
      if ("configuration" in st) p.configuration = clone(st.configuration);
      const method = st.stream ? "SendStreamingMessage" : "SendMessage";
      call(method, p, st.note ?? "");
    } else if (op === "sign_in") {
      cl.state = "asking the user to sign in";
      cl.clock += TIMES.human!;
      record({ dir: "host", gate: "the user signs in to the calendar (outside A2A)" }, "gate", "", null, 0, "out of band", cl.clock);
      cl.state = "the user has signed in";
    } else if (op === "get") {
      call("GetTask", { id: cl.task }, st.note ?? "");
    } else if (op === "cancel") {
      call("CancelTask", { id: cl.task }, st.note ?? "");
    } else if (op === "subscribe") {
      call("SubscribeToTask", { id: cl.task }, st.note ?? "");
    } else if (op === "raw") {
      const params = st.params === undefined ? null : clone(st.params);
      if (st.task_id && params !== null) params.id = cl.task;
      if (st.task_message && params !== null) params.message.taskId = cl.task;
      call(st.method, params, st.note ?? "", st.headers ?? null);
    } else {
      throw new Error("unknown step " + op);
    }
  }
  return { name: sc.name, wire, log, card: cl.card };
}

const UUID_RE = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g;

export function normalise(wire: Obj[]): Obj[] {
  const seen = new Map<string, string>();
  const ids = (s: string) =>
    s.replace(UUID_RE, (u) => {
      if (!seen.has(u)) seen.set(u, "<id-" + (seen.size + 1) + ">");
      return seen.get(u)!;
    });
  const walk = (x: unknown, key: string): unknown => {
    if (x !== null && typeof x === "object" && !Array.isArray(x)) {
      const o = x as Obj;
      const out: Obj = {};
      for (const k of Object.keys(o).sort()) out[k] = walk(o[k], k);
      return out;
    }
    if (Array.isArray(x)) return x.map((v) => walk(v, key));
    if (typeof x === "string") return key === "timestamp" ? "<time>" : ids(x);
    return x;
  };
  const out: Obj[] = [];
  for (const w of wire) {
    if (w.dir === "host") continue;
    const item: Obj = {};
    for (const [k, v] of Object.entries(w)) if (k !== "headers" || w.dir === "c2s") item[k] = v;
    out.push(walk(item, "") as Obj);
  }
  return out;
}

export function a2aScenario(name: string): Obj {
  const sc = clone(A2A_SCENARIOS[name]!);
  sc.name = name;
  return sc;
}
