/**
 * MCP as two state machines exchanging JSON-RPC 2.0 messages, in both protocol eras: a port of
 * agent_loop_sim/protocols/mcp.py, statement for statement. The tests require this port to
 * reproduce the Python fixtures exactly and the official SDK's recordings after normalising IDs.
 */
import data from "../protocols_data.json";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Obj = { [k: string]: any };

export const JSONRPC = "2.0";
export const HANDSHAKE_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];
export const MODERN_VERSIONS = ["2026-07-28"];
export const LATEST_HANDSHAKE = "2025-11-25";
export const LATEST = "2026-07-28";
export const SPEC_ACCESSED = "2026-10-08";

export const META_VERSION = "io.modelcontextprotocol/protocolVersion";
export const META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo";
export const META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities";
export const META_SERVER_INFO = "io.modelcontextprotocol/serverInfo";

export const ERRORS: Record<string, number> = {
  parse_error: -32700,
  invalid_request: -32600,
  method_not_found: -32601,
  invalid_params: -32602,
  internal_error: -32603,
  header_mismatch: -32020,
  missing_client_capability: -32021,
  unsupported_protocol_version: -32022,
};

const ERA_MESSAGE =
  "this connection serves the handshake protocol era; requests carrying the 2026-07-28 envelope " +
  "are not accepted on it";
const ENVELOPE_MESSAGE =
  "params._meta must be an object carrying the required 'io.modelcontextprotocol/protocolVersion' " +
  "and 'io.modelcontextprotocol/clientCapabilities' envelope keys";

export const CONFIRM_SCHEMA: Obj = {
  type: "object",
  properties: { confirm: { type: "boolean" } },
  required: ["confirm"],
};

export const FIXTURE_SERVER: Obj = data.fixture_server as Obj;
export const EXTRA_TOOL: Obj = data.extra_tool as Obj;

export function clone<T>(x: T): T {
  return x === undefined ? x : (JSON.parse(JSON.stringify(x)) as T);
}

/** The compact JSON text the Python reference writes (json.dumps, separators "," ":"). */
export function compact(x: unknown): string {
  return JSON.stringify(x);
}

export function utf8Len(s: string): number {
  let n = 0;
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    n += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
  }
  return n;
}

export function textContent(text: string): Obj[] {
  return [{ text, type: "text" }];
}

export function request(id: unknown, method: string, params: Obj | null = null): Obj {
  const m: Obj = { jsonrpc: JSONRPC, id, method };
  if (params !== null) m.params = params;
  return m;
}

export function notification(method: string, params: Obj | null = null): Obj {
  const m: Obj = { jsonrpc: JSONRPC, method };
  if (params !== null) m.params = params;
  return m;
}

export function response(id: unknown, result: Obj): Obj {
  return { jsonrpc: JSONRPC, id, result };
}

export function error(id: unknown, code: number, message: string, dataValue: unknown = null, hasData = false): Obj {
  const e: Obj = { code, message };
  if (hasData) e.data = dataValue;
  return { jsonrpc: JSONRPC, id, error: e };
}

/** request · notification · result · error. */
export function kindOf(msg: Obj): string {
  if ("method" in msg) return "id" in msg ? "request" : "notification";
  return "error" in msg ? "error" : "result";
}

function isEmpty(x: unknown): boolean {
  return x === null || x === undefined || (typeof x === "object" && Object.keys(x as object).length === 0);
}

export class Server {
  spec: Obj;
  pageSize: number | null;
  listChanged: boolean;
  era: string | null = null;
  version: string | null = null;
  initialized = false;
  clientCaps: Obj = {};
  nextId = 1;
  pending: Obj | null = null;
  cancelled: unknown[] = [];
  state = "waiting";

  constructor(spec: Obj, options: Obj | null = null) {
    const o = options || {};
    this.spec = clone(spec);
    this.pageSize = o.page_size ?? null;
    this.listChanged = Boolean(o.list_changed ?? false);
  }

  private modern(result: Obj, cacheable: boolean): Obj {
    if (this.era !== "modern") return result;
    if (cacheable) {
      result.cacheScope = "private";
      result.ttlMs = 0;
    }
    result.resultType = "complete";
    result._meta = { [META_SERVER_INFO]: { ...this.spec.info } };
    return result;
  }

  private capabilities(discover: boolean): Obj {
    const lc = discover || this.listChanged;
    return {
      prompts: { listChanged: lc },
      resources: { listChanged: lc, subscribe: discover },
      tools: { listChanged: lc },
    };
  }

  private page(items: Obj[], params: Obj): [Obj[], string | null] {
    if (this.pageSize === null) return [items, null];
    let start = 0;
    const cur = params.cursor;
    if (typeof cur === "string" && cur.startsWith("page-")) start = parseInt(cur.slice(5), 10);
    const end = start + this.pageSize;
    return [items.slice(start, end), end < items.length ? "page-" + String(end) : null];
  }

  private list(key: string, params: Obj): Obj {
    const [items, nxt] = this.page(clone(this.spec[key]), params);
    const r: Obj = { [key]: items };
    if (nxt !== null) r.nextCursor = nxt;
    return this.modern(r, true);
  }

  private toolResult(text: string, structured: boolean, isError = false): Obj {
    const r: Obj = { content: textContent(text), isError };
    if (structured) r.structuredContent = { result: text };
    return this.modern(r, false);
  }

  receive(item: Obj): Obj[] {
    if ("raw" in item) {
      this.state = "dropped a malformed line";
      return [];
    }
    const msg = item.msg as Obj;
    const k = kindOf(msg);
    if (k === "notification") return this.onNotification(msg);
    if (k === "result" || k === "error") return this.clientAnswer(msg);
    return this.onRequest(msg);
  }

  private onNotification(msg: Obj): Obj[] {
    if (msg.method === "notifications/initialized") {
      this.initialized = true;
      this.state = "ready (" + String(this.version) + ")";
    } else if (msg.method === "notifications/cancelled") {
      this.cancelled.push(msg.params.requestId);
      this.state = "stopped the cancelled request";
    }
    return [];
  }

  private clientAnswer(msg: Obj): Obj[] {
    const p = this.pending;
    if (p === null || p.sid !== msg.id) return [];
    this.pending = null;
    const answer: Obj = msg.result ?? {};
    const req = p.req as Obj;
    const name = req.params.name;
    let text: string;
    if (p.kind === "elicit") {
      const ok = answer.action === "accept" && Boolean((answer.content || {}).confirm);
      const path = req.params.arguments.path;
      text = (ok ? "deleted " : "kept ") + path;
    } else {
      const c = answer.content || {};
      text = c.type === "text" ? (c.text ?? "?") : "?";
    }
    this.state = "finished " + name;
    return [response(req.id, this.toolResult(text, true))];
  }

  private onRequest(msg: Obj): Obj[] {
    const id = msg.id;
    const method: string = msg.method;
    const params: Obj = msg.params || {};
    const meta: Obj | null =
      params._meta !== null && typeof params._meta === "object" && !Array.isArray(params._meta) ? params._meta : null;
    if (method === "initialize") {
      this.era = "handshake";
      const asked = params.protocolVersion;
      this.version = HANDSHAKE_VERSIONS.includes(asked) ? asked : LATEST_HANDSHAKE;
      this.clientCaps = params.capabilities || {};
      this.state = "initialising";
      return [
        response(id, {
          capabilities: this.capabilities(false),
          protocolVersion: this.version,
          serverInfo: { ...this.spec.info },
        }),
      ];
    }
    if (meta !== null && META_VERSION in meta) {
      if (this.era === "handshake") return [error(id, ERRORS.invalid_request!, ERA_MESSAGE)];
      const v = meta[META_VERSION];
      if (!MODERN_VERSIONS.includes(v)) {
        return [
          error(id, ERRORS.unsupported_protocol_version!, "Unsupported protocol version", {
            supported: [...MODERN_VERSIONS],
            requested: v,
          }, true),
        ];
      }
      if (!(META_CLIENT_CAPS in meta)) {
        return [error(id, ERRORS.invalid_params!, "params._meta is missing the required envelope key(s): " + META_CLIENT_CAPS)];
      }
      this.era = "modern";
      this.version = v;
      this.clientCaps = meta[META_CLIENT_CAPS];
      this.state = "serving " + method + " (stateless)";
    } else if (this.era === "modern") {
      return [error(id, ERRORS.invalid_params!, ENVELOPE_MESSAGE)];
    } else if (this.era === null) {
      return [error(id, ERRORS.invalid_params!, "Invalid request parameters", "", true)];
    } else {
      this.state = "serving " + method;
    }
    return this.dispatch(id, method, params);
  }

  private dispatch(id: unknown, method: string, params: Obj): Obj[] {
    const modern = this.era === "modern";
    if (method === "ping" && !modern) return [response(id, {})];
    if (method === "server/discover" && modern) {
      const r: Obj = { capabilities: this.capabilities(true), supportedVersions: [...MODERN_VERSIONS] };
      return [response(id, this.modern(r, true))];
    }
    if (method === "tools/list") return [response(id, this.list("tools", params))];
    if (method === "resources/list") return [response(id, this.list("resources", params))];
    if (method === "prompts/list") return [response(id, this.list("prompts", params))];
    if (method === "resources/read") {
      const uri = params.uri;
      if (!(uri in this.spec.texts)) {
        return [error(id, ERRORS.invalid_params!, "Unknown resource: " + String(uri), { uri }, true)];
      }
      const r: Obj = { contents: [{ mimeType: this.mime(uri), text: this.spec.texts[uri], uri }] };
      return [response(id, this.modern(r, true))];
    }
    if (method === "prompts/get") {
      const code = (params.arguments || {}).code ?? "";
      const r: Obj = {
        description: "Review a piece of code.",
        messages: [{ content: { text: "Please review:\n" + code, type: "text" }, role: "user" }],
      };
      return [response(id, this.modern(r, false))];
    }
    if (method === "tools/call") return this.call(id, params);
    return [error(id, ERRORS.method_not_found!, "Method not found", method, true)];
  }

  private mime(uri: string): string {
    for (const r of this.spec.resources as Obj[]) if (r.uri === uri) return r.mimeType;
    return "text/plain";
  }

  private call(id: unknown, params: Obj): Obj[] {
    const name = params.name;
    const args: Obj = params.arguments || {};
    const modern = this.era === "modern";
    const names = (this.spec.tools as Obj[]).map((t) => t.name);
    if (!names.includes(name)) return [response(id, this.toolResult("Unknown tool: " + String(name), false, true))];
    if (name === "add") {
      const v = args.a + args.b;
      const r: Obj = { content: textContent(String(v)), isError: false, structuredContent: { result: v } };
      return [response(id, this.modern(r, false))];
    }
    if (name === "read_doc") return [response(id, this.toolResult(this.spec.texts[args.uri], true))];
    if (name === "deploy") return [response(id, this.toolResult("deployed to " + String(args.target), true))];
    if (name === "fail") return [response(id, this.toolResult("Error executing tool fail", false, true))];
    if (name === "count") {
      const out: Obj[] = [];
      const token = (params._meta || {}).progressToken ?? null;
      const n: number = args.n;
      if (token !== null) {
        for (let i = 1; i <= n; i++) {
          out.push(notification("notifications/progress", { progressToken: token, progress: i, total: n }));
        }
      }
      out.push(response(id, this.toolResult("counted to " + String(n), true)));
      return out;
    }
    if (name === "delete_file" || name === "summarise") {
      const kind = name === "delete_file" ? "elicit" : "sample";
      let method: string;
      let p: Obj;
      let key: string;
      if (kind === "elicit") {
        method = "elicitation/create";
        p = { mode: "form", message: "Delete " + args.path + "?", requestedSchema: clone(CONFIRM_SCHEMA) };
        key = "confirm";
      } else {
        method = "sampling/createMessage";
        p = { messages: [{ role: "user", content: { type: "text", text: "Summarise: " + args.text } }], maxTokens: 50 };
        key = "summary";
      }
      if (modern) {
        const responses = params.inputResponses;
        if (isEmpty(responses)) {
          this.state = "needs input: " + method;
          const r: Obj = {
            inputRequests: { [key]: { method, params: p } },
            resultType: "input_required",
            _meta: { [META_SERVER_INFO]: { ...this.spec.info } },
          };
          return [response(id, r)];
        }
        const answer: Obj = responses[key];
        let text: string;
        if (kind === "elicit") {
          const ok = answer.action === "accept" && Boolean((answer.content || {}).confirm);
          text = (ok ? "deleted " : "kept ") + args.path;
        } else {
          const c = answer.content || {};
          text = c.type === "text" ? (c.text ?? "?") : "?";
        }
        return [response(id, this.toolResult(text, true))];
      }
      const sid = this.nextId;
      this.nextId += 1;
      this.pending = { sid, kind, req: request(id, "tools/call", params) };
      this.state = "waiting for the client: " + method;
      return [request(sid, method, p)];
    }
    throw new Error("no behaviour for tool " + String(name));
  }

  /** Engine-only: the server gains a tool; with list_changed it tells the client. */
  addTool(): Obj[] {
    this.spec.tools.push(clone(EXTRA_TOOL));
    if (this.listChanged && this.era === "handshake") return [notification("notifications/tools/list_changed")];
    return [];
  }
}

export class Client {
  mode: string;
  info: Obj;
  caps: Obj;
  answers: Obj;
  nextId = 1;
  modern: boolean;
  version: string;
  toolsListed = false;
  state = "connecting";

  constructor(sc: Obj) {
    this.mode = sc.mode;
    const cl = sc.client;
    this.info = { ...cl.info };
    const caps: Obj = {};
    if (cl.elicitation) caps.elicitation = { form: {}, url: {} };
    if (cl.sampling) caps.sampling = {};
    this.caps = caps;
    this.answers = {};
    for (const [k, v] of Object.entries(sc.answers ?? {})) this.answers[k] = clone(v);
    this.modern = !["legacy", "raw"].includes(this.mode);
    this.version = this.modern ? LATEST : LATEST_HANDSHAKE;
  }

  meta(): Obj {
    return { [META_VERSION]: this.version, [META_CLIENT_INFO]: { ...this.info }, [META_CLIENT_CAPS]: clone(this.caps) };
  }

  newId(): number {
    const i = this.nextId;
    this.nextId += 1;
    return i;
  }

  params(base: Obj | null, progressToken: unknown = null): Obj | null {
    let p: Obj | null = base !== null ? { ...base } : null;
    if (this.modern) {
      p = p !== null ? p : {};
      const m = this.meta();
      if (progressToken !== null) m.progressToken = progressToken;
      p._meta = m;
    } else if (progressToken !== null) {
      p = p !== null ? p : {};
      p._meta = { progressToken };
    }
    return p;
  }

  answer(req: Obj): Obj {
    if (req.method === "elicitation/create") return this.answers.elicit.shift();
    if (req.method === "sampling/createMessage") return this.answers.sample.shift();
    throw new Error("cannot answer " + req.method);
  }
}

/**
 * Run a scenario: `wire` is exactly what crosses the transport, in order; `log` annotates the
 * same messages for the animations (kind, method, bytes, both states).
 */
export function play(sc: Obj): Obj {
  const server = new Server(FIXTURE_SERVER, sc.server ?? null);
  const client = new Client(sc);
  const wire: Obj[] = [];
  const log: Obj[] = [];
  const responses = new Map<string, Obj>();
  const methods: Record<string, string> = {};

  const record = (d: string, item: Obj, note: string): void => {
    wire.push(item);
    let text: string;
    let kind: string;
    let method: string;
    let mid: unknown;
    if ("raw" in item) {
      text = item.raw;
      kind = "malformed";
      method = "";
      mid = null;
    } else {
      const msg = item.msg as Obj;
      text = compact(msg);
      kind = kindOf(msg);
      mid = msg.id ?? null;
      if ("method" in msg) {
        method = msg.method;
        if ("id" in msg) methods[d + ":" + String(mid)] = method;
      } else {
        const other = d === "c2s" ? "s2c" : "c2s";
        method = methods[other + ":" + String(mid)] ?? "";
      }
    }
    log.push({
      seq: log.length,
      dir: d,
      kind,
      method,
      id: mid,
      bytes: utf8Len(text),
      note,
      client: client.state,
      server: server.state,
    });
  };

  const deliver = (item: Obj, note: string, cancelAfter: number | null = null, cancelId: unknown = null): void => {
    const queue: [Obj, string][] = [[item, note]];
    while (queue.length > 0) {
      const [it, nt] = queue.shift()!;
      record("c2s", it, nt);
      const outs = server.receive(it);
      let seenProgress = 0;
      for (const o of outs) {
        if (cancelAfter !== null && seenProgress >= cancelAfter) break;
        const k = kindOf(o);
        record("s2c", { dir: "s2c", msg: o }, "");
        if (k === "notification" && o.method === "notifications/progress") {
          seenProgress += 1;
          if (cancelAfter !== null && seenProgress === cancelAfter) {
            client.state = "cancelling";
            const c = notification("notifications/cancelled", { requestId: cancelId, reason: "user cancelled" });
            record("c2s", { dir: "c2s", msg: c }, "the user cancels");
            server.receive({ dir: "c2s", msg: c });
          }
        } else if (k === "request") {
          client.state = "answering " + o.method;
          const a = client.answer(o);
          queue.push([{ dir: "c2s", msg: response(o.id, a) }, "the client answers"]);
        } else if (k === "result" || k === "error") {
          responses.set(compact(o.id), o);
        }
      }
    }
  };

  const call = (method: string, base: Obj | null, note: string, progress = false, cancelAfter: number | null = null): Obj | null => {
    const i = client.newId();
    const p = client.params(base, progress ? i : null);
    client.state = "waiting for " + method;
    deliver({ dir: "c2s", msg: request(i, method, p) }, note, cancelAfter, i);
    client.state = "ready";
    return responses.get(compact(i)) ?? null;
  };

  if (client.mode === "raw") {
    for (const st of sc.steps as Obj[]) {
      if ("line" in st) deliver({ dir: "c2s", raw: st.line }, "a line that is not JSON");
      else deliver({ dir: "c2s", msg: clone(st.msg) }, "hand-written");
    }
    return { name: sc.name, wire, log };
  }

  if (client.mode === "legacy") {
    client.state = "initialising";
    call(
      "initialize",
      { protocolVersion: LATEST_HANDSHAKE, capabilities: clone(client.caps), clientInfo: { ...client.info } },
      "the client proposes a version and its capabilities",
    );
    client.state = "ready";
    deliver({ dir: "c2s", msg: notification("notifications/initialized") }, "the handshake is complete");
  } else if (client.mode === "auto") {
    call("server/discover", null, "the client asks which versions the server speaks");
  }

  const listAll = (method: string, key: string): void => {
    let cursor: string | null = null;
    for (;;) {
      const r: Obj | null = call(method, cursor !== null ? { cursor } : null, "");
      const res: Obj = (r || {}).result || {};
      cursor = res.nextCursor ?? null;
      if (cursor === null) break;
    }
    if (key === "tools") client.toolsListed = true;
  };

  for (const st of sc.steps as Obj[]) {
    const op = st.op;
    if (op === "ping") call("ping", null, "");
    else if (op === "list_tools") listAll("tools/list", "tools");
    else if (op === "list_resources") listAll("resources/list", "resources");
    else if (op === "list_prompts") listAll("prompts/list", "prompts");
    else if (op === "read_resource") call("resources/read", { uri: st.uri }, "");
    else if (op === "get_prompt") call("prompts/get", { name: st.name, arguments: clone(st.arguments) }, "");
    else if (op === "add_tool") {
      for (const o of server.addTool()) {
        record("s2c", { dir: "s2c", msg: o }, "the server's tools changed");
        if (o.method === "notifications/tools/list_changed") listAll("tools/list", "tools");
      }
    } else if (op === "call_tool") {
      const base: Obj = { name: st.name, arguments: clone(st.arguments) };
      let r = call("tools/call", base, "", Boolean(st.progress), st.cancel_after ?? null);
      let rounds = 0;
      while (r !== null && "result" in r && r.result.resultType === "input_required" && rounds < 8) {
        rounds += 1;
        client.state = "gathering input";
        const answers: Obj = {};
        for (const [key, ir] of Object.entries(r.result.inputRequests as Obj)) answers[key] = client.answer(ir as Obj);
        const retry: Obj = { inputResponses: answers, ...base };
        r = call("tools/call", retry, "the client retries with the answers");
      }
      const res: Obj = (r || {}).result || {};
      if ("structuredContent" in res && !res.isError && !client.toolsListed) listAll("tools/list", "tools");
    } else {
      throw new Error("unknown step " + op);
    }
  }
  return { name: sc.name, wire, log };
}

/** Renumber request IDs (and progress tokens) in order of appearance, as the Python reference does. */
export function normalise(wire: Obj[]): Obj[] {
  const ids: Record<string, string> = {};
  const tokens: Record<string, string> = {};
  const n: Record<string, number> = { c2s: 0, s2c: 0 };
  const out: Obj[] = [];
  for (const w of wire) {
    if (!("msg" in w)) {
      out.push({ dir: w.dir, raw: w.raw });
      continue;
    }
    const m: Obj = clone(w.msg);
    const d: string = w.dir;
    if ("id" in m && m.id !== null) {
      if ("method" in m) {
        n[d] = n[d]! + 1;
        const tag = (d === "c2s" ? "c" : "s") + String(n[d]);
        ids[d + ":" + compact(m.id)] = tag;
        const meta = (m.params || {})._meta;
        if (meta !== null && typeof meta === "object" && "progressToken" in meta) {
          tokens[compact(meta.progressToken)] = tag;
          meta.progressToken = tag;
        }
        m.id = tag;
      } else {
        const other = d === "c2s" ? "s2c" : "c2s";
        m.id = ids[other + ":" + compact(m.id)] ?? "?";
      }
    }
    if (m.method === "notifications/progress") {
      const t = compact(m.params.progressToken);
      m.params.progressToken = tokens[t] ?? "?";
    }
    if (m.method === "notifications/cancelled") {
      m.params.requestId = ids["c2s:" + compact(m.params.requestId)] ?? "?";
    }
    out.push({ dir: d, msg: m });
  }
  return out;
}
