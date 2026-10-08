/**
 * MCP transports: framing, bytes and timing for stdio and Streamable HTTP in both eras.
 * A port of agent_loop_sim/protocols/transport.py, statement for statement.
 */
import data from "../protocols_data.json";
import { compact, kindOf, notification, request, response, textContent, utf8Len, type Obj } from "./mcp";

export const TRANSPORTS: Record<string, Obj> = data.transports as Record<string, Obj>;
export const WORK_MS: Record<string, number> = data.work_ms as Record<string, number>;
export const HOST = "mcp.example.com";
export const SESSION_ID = "4f1c9a7e2b6d4e83";

function lat(t: Obj, nbytes: number): number {
  return t.one_way_ms + (nbytes * 8) / (t.mbit_per_s * 1000);
}

function httpRequest(body: string, headers: string[][]): string {
  const lines = [
    "POST /mcp HTTP/1.1",
    "Host: " + HOST,
    "Content-Type: application/json",
    "Accept: application/json, text/event-stream",
  ];
  for (const [k, v] of headers) lines.push(k + ": " + v);
  lines.push("Content-Length: " + String(utf8Len(body)));
  return lines.join("\r\n") + "\r\n\r\n";
}

function mcpName(msg: Obj): string | null {
  const p = msg.params || {};
  if (msg.method === "tools/call" || msg.method === "prompts/get") return p.name ?? null;
  if (msg.method === "resources/read") return p.uri ?? null;
  return null;
}

/** Frame a played session on a transport: per message, the framing text, payload and overhead bytes, timing. */
export function frameSession(log: Obj[], wire: Obj[], transport: string): Obj {
  const t = TRANSPORTS[transport]!;
  let era = "handshake";
  for (const w of wire) {
    if ("msg" in w && w.msg.method === "initialize") {
      era = "handshake";
      break;
    }
    const meta = ((w.msg || {}).params || {})._meta || {};
    if ("io.modelcontextprotocol/protocolVersion" in meta) {
      era = "modern";
      break;
    }
  }
  let version: string | null = null;
  let session = false;
  const streams: Record<string, Obj> = {};
  const openIds: string[] = [];
  const requests: Record<string, Obj> = {};
  const frames: Obj[] = [];
  let tc: number = t.startup_ms;
  let ts = tc;
  let eventId = 0;
  let posts = 0;
  for (let i = 0; i < wire.length; i++) {
    const w = wire[i]!;
    const lg = log[i]!;
    if (!("msg" in w)) {
      const body: string = w.raw;
      frames.push({
        seq: i,
        dir: "c2s",
        framing: "\\n",
        payload: utf8Len(body),
        overhead: 1,
        depart: tc,
        arrive: tc + lat(t, utf8Len(body) + 1),
        http: null,
      });
      continue;
    }
    const msg = w.msg as Obj;
    const body = compact(msg);
    const payload = utf8Len(body);
    const k = kindOf(msg);
    const d: string = w.dir;
    if (d === "c2s" && k === "request") requests[compact(msg.id)] = msg;
    let http: Obj | null = null;
    let framing: string;
    let overhead: number;
    if (transport === "stdio") {
      framing = "\\n";
      overhead = 1;
    } else if (d === "c2s") {
      const headers: string[][] = [];
      if (era === "handshake") {
        if (version !== null) headers.push(["MCP-Protocol-Version", version]);
        if (session) headers.push(["Mcp-Session-Id", SESSION_ID]);
      } else if (k === "request") {
        headers.push(["MCP-Protocol-Version", "2026-07-28"]);
        headers.push(["Mcp-Method", msg.method]);
        const nm = mcpName(msg);
        if (nm !== null) headers.push(["Mcp-Name", nm]);
      }
      framing = httpRequest(body, headers);
      overhead = utf8Len(framing);
      posts += 1;
      if (k === "request") {
        const key = compact(msg.id);
        streams[key] = { sse: false };
        openIds.push(key);
        http = { request: framing, status: null };
      } else {
        const reply = "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n";
        overhead += utf8Len(reply);
        http = { request: framing, status: 202, response: reply };
      }
    } else {
      let key: string;
      if (k === "result" || k === "error") key = compact(msg.id);
      else key = openIds.length > 0 ? openIds[openIds.length - 1]! : "";
      const st: Obj = streams[key] ?? { sse: false };
      const startsStream = !st.sse && !(k === "result" || k === "error");
      if ((k === "result" || k === "error") && !st.sse) {
        let hdr = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n";
        if (era === "handshake" && (requests[key] ?? {}).method === "initialize") {
          hdr += "Mcp-Session-Id: " + SESSION_ID + "\r\n";
        }
        hdr += "Content-Length: " + String(payload) + "\r\n\r\n";
        framing = hdr;
        http = { status: 200, content_type: "application/json" };
      } else {
        let prefix = "";
        if (startsStream) {
          prefix = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n";
          if (era === "modern") prefix += "X-Accel-Buffering: no\r\n";
          prefix += "\r\n";
          st.sse = true;
          streams[key] = st;
        }
        let ev = "";
        if (era === "handshake") {
          eventId += 1;
          ev = "id: " + String(eventId) + "\n";
        }
        framing = prefix + ev + "event: message\ndata: " + "\n\n";
        http = {
          status: 200,
          content_type: "text/event-stream",
          event_id: era === "handshake" ? eventId : null,
          opens_stream: startsStream,
        };
      }
      overhead = utf8Len(framing);
      if (k === "result" || k === "error") {
        const at = openIds.indexOf(key);
        if (at >= 0) openIds.splice(at, 1);
        if (era === "handshake" && (requests[key] ?? {}).method === "initialize" && k === "result") {
          session = true;
          version = msg.result.protocolVersion ?? null;
        }
      }
    }
    let depart: number;
    let arrive: number;
    if (d === "c2s") {
      depart = tc;
      if (k === "result") {
        if (lg.method === "elicitation/create") depart = tc + WORK_MS.human!;
        else if (lg.method === "sampling/createMessage") depart = tc + WORK_MS.model!;
      } else if (k === "request" && msg.method === "tools/call" && "inputResponses" in (msg.params || {})) {
        const ir = msg.params.inputResponses;
        depart = tc + ("confirm" in ir ? WORK_MS.human! : WORK_MS.model!);
      }
      arrive = depart + lat(t, payload + overhead);
      ts = Math.max(ts, arrive);
      tc = depart;
    } else {
      let work = WORK_MS.default!;
      if (k === "notification" && msg.method === "notifications/progress") {
        work = WORK_MS.count_step!;
      } else if (k === "result" || k === "error") {
        const req: Obj = requests[compact(msg.id)] ?? {};
        if (req.method === "tools/call") {
          work = WORK_MS["tools/call"]!;
          const p = req.params || {};
          if (p.name === "count" && !("progressToken" in (p._meta || {}))) {
            work = work + p.arguments.n * WORK_MS.count_step!;
          }
        }
      }
      depart = ts + work;
      ts = depart;
      arrive = depart + lat(t, payload + overhead);
      tc = Math.max(tc, arrive);
    }
    frames.push({ seq: i, dir: d, framing, payload, overhead, depart, arrive, http });
  }
  let payloadTotal = 0;
  let overheadTotal = 0;
  for (const f of frames) {
    payloadTotal += f.payload;
    overheadTotal += f.overhead;
  }
  let end: number = t.startup_ms;
  for (const f of frames) end = Math.max(end, f.arrive);
  return {
    transport,
    era,
    frames,
    totals: {
      messages: frames.length,
      payload: payloadTotal,
      overhead: overheadTotal,
      posts,
      elapsed_ms: end,
      startup_ms: t.startup_ms,
    },
  };
}

/**
 * A tools/call reporting progress over Streamable HTTP whose SSE stream breaks after
 * `dropAfter` events (1 <= dropAfter < n): resumed with Last-Event-ID (2025-11-25), or sent
 * again with a new ID (2026-07-28).
 */
export function streamDrop(era: string, n: number, dropAfter: number, reconnectMs = 1000): Obj {
  const t = TRANSPORTS.http!;
  const step = WORK_MS.count_step!;
  let steps: Obj[] = [];
  const metaV: Obj = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": { name: "fixture-client", version: "1.0.0" },
    "io.modelcontextprotocol/clientCapabilities": {},
  };
  let bytesTotal = 0;
  let workSteps = 0;

  const callMsg = (id: number): Obj => {
    let meta: Obj = { progressToken: id };
    if (era === "modern") {
      meta = { ...metaV };
      meta.progressToken = id;
    }
    return request(id, "tools/call", { name: "count", arguments: { n }, _meta: meta });
  };

  const resultMsg = (id: number): Obj => {
    const r: Obj = {
      content: textContent("counted to " + String(n)),
      isError: false,
      structuredContent: { result: "counted to " + String(n) },
    };
    if (era === "modern") {
      r.resultType = "complete";
      r._meta = { "io.modelcontextprotocol/serverInfo": { name: "fixture-server", version: "1.0.0" } };
    }
    return response(id, r);
  };

  const add = (
    tm: number,
    actor: string,
    kind: string,
    label: string,
    msg: Obj | null,
    eventId: number | null,
    extra: number,
    live: boolean,
  ): void => {
    const b = (msg !== null ? utf8Len(compact(msg)) : 0) + extra;
    if (live) bytesTotal += b;
    steps.push({ t: tm, actor, kind, label, event_id: eventId, bytes: b, live, msg });
  };

  const evFrame = "event: message\ndata: \n\n".length;
  const postExtra = 220;
  let tc = 0.0;
  const first = callMsg(1);
  add(tc, "client", "post", "POST tools/call (id 1)", first, null, postExtra, true);
  let ts = tc + lat(t, postExtra);
  let eid = 0;
  let lostAt = 0.0;
  for (let i = 1; i <= n; i++) {
    ts += step;
    workSteps += 1;
    const prog = notification("notifications/progress", { progressToken: 1, progress: i, total: n });
    if (i <= dropAfter) {
      eid += 1;
      add(
        ts + t.one_way_ms,
        "server",
        "event",
        "progress " + String(i) + "/" + String(n),
        prog,
        eid,
        evFrame + (era === "handshake" ? ("id: " + String(eid) + "\n").length : 0),
        true,
      );
      lostAt = ts + t.one_way_ms;
    } else if (era === "handshake") {
      eid += 1;
      add(ts, "server", "buffered", "progress " + String(i) + "/" + String(n) + " (buffered)", prog, eid, 0, false);
    } else {
      break;
    }
    if (i === dropAfter) {
      add(ts + t.one_way_ms + 1, "network", "drop", "the stream breaks", null, null, 0, true);
      if (era === "modern") {
        add(ts + t.one_way_ms + 1, "server", "cancel", "closing the stream cancels the request", null, null, 0, true);
        break;
      }
    }
  }
  let end: number;
  let redone: number;
  let replayed: number;
  let lostRequests: number;
  if (era === "handshake") {
    const finalEid = eid + 1;
    add(ts + 1, "server", "buffered", "result (buffered)", resultMsg(1), finalEid, 0, false);
    tc = lostAt + reconnectMs;
    const getHdr =
      "GET /mcp HTTP/1.1\r\nHost: " +
      HOST +
      "\r\nAccept: text/event-stream\r\nMcp-Session-Id: " +
      SESSION_ID +
      "\r\nMCP-Protocol-Version: 2025-11-25\r\nLast-Event-ID: " +
      String(dropAfter) +
      "\r\n\r\n";
    add(tc, "client", "resume", "GET with Last-Event-ID: " + String(dropAfter), null, null, utf8Len(getHdr), true);
    const tr = tc + lat(t, utf8Len(getHdr));
    replayed = 0;
    const kept: Obj[] = [];
    for (const s of steps) if (s.kind !== "buffered") kept.push(s);
    const buffered: Obj[] = [];
    for (const s of steps) if (s.kind === "buffered") buffered.push(s);
    steps = kept;
    for (const s of buffered) {
      const eidR: number = s.event_id;
      const extra = evFrame + ("id: " + String(eidR) + "\n").length;
      if (s.t <= tr) {
        replayed += 1;
        add(s.t, "server", "buffered", s.label, s.msg, eidR, 0, false);
        add(tr + t.one_way_ms, "server", "replay", "replay event " + String(eidR), s.msg, eidR, extra, true);
      } else {
        const label = (s.label as string).split(" (buffered)").join("");
        add(s.t + t.one_way_ms, "server", "event", label, s.msg, eidR, extra, true);
      }
    }
    end = Math.max(tr + t.one_way_ms, ts + 1 + t.one_way_ms);
    redone = 0;
    lostRequests = 0;
  } else {
    tc = lostAt + reconnectMs;
    const again = callMsg(2);
    add(tc, "client", "post", "POST tools/call again (id 2)", again, null, postExtra, true);
    ts = tc + lat(t, postExtra);
    for (let i = 1; i <= n; i++) {
      ts += step;
      workSteps += 1;
      const prog = notification("notifications/progress", { progressToken: 2, progress: i, total: n });
      add(ts + t.one_way_ms, "server", "event", "progress " + String(i) + "/" + String(n) + " (again)", prog, null, evFrame, true);
    }
    add(ts + 1 + t.one_way_ms, "server", "event", "result", resultMsg(2), null, evFrame, true);
    end = ts + 1 + t.one_way_ms;
    redone = dropAfter;
    replayed = 0;
    lostRequests = 1;
  }
  steps.sort((a, b) => a.t - b.t);
  steps.forEach((s, i) => {
    s.seq = i;
  });
  return {
    era,
    n,
    drop_after: dropAfter,
    steps,
    totals: {
      elapsed_ms: end,
      bytes: bytesTotal,
      work_steps: workSteps,
      redone_steps: redone,
      replayed_events: replayed,
      lost_requests: lostRequests,
    },
  };
}
