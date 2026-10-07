/** Animation states derived from a trace: the TS twin of `agent_loop_sim/views.py`. */
import { PRICES, type Obj } from "./data";
import type { Ev } from "./harness";

export const KIND_ORDER = [
  "system", "tools", "task", "summary", "assistant", "tool_result", "observation", "error", "nudge", "prompt",
] as const;

export type Part = { kind: string; tokens: number };

export function agg(context: Obj[]): Part[] {
  const sums = new Map<string, number>(KIND_ORDER.map((k) => [k, 0]));
  for (const c of context) if (sums.has(c.kind)) sums.set(c.kind, sums.get(c.kind)! + c.tokens);
  return KIND_ORDER.filter((k) => sums.get(k)! > 0).map((k) => ({ kind: k, tokens: sums.get(k)! }));
}

function add(parts: Part[], kind: string, n: number): Part[] {
  const sums = new Map(parts.map((p) => [p.kind, p.tokens]));
  sums.set(kind, (sums.get(kind) ?? 0) + n);
  return KIND_ORDER.filter((k) => (sums.get(k) ?? 0) > 0).map((k) => ({ kind: k, tokens: sums.get(k)! }));
}

const dropPrompt = (parts: Part[]) => parts.filter((p) => p.kind !== "prompt");
const total = (parts: Part[]) => parts.reduce((s, p) => s + p.tokens, 0);
const isReact = (events: Ev[]) => events.length > 0 && events[0]!.type === "run_start" && events[0]!.style === "react";

export function loopFrames(events: Ev[], agent = "main"): Obj[] {
  const frames: Obj[] = [];
  let parts: Part[] = [];
  let cost = 0;
  const resultKind = isReact(events) ? "observation" : "tool_result";
  for (const e of events) {
    if (e.agent !== agent) continue;
    let f: Obj | null = null;
    if (e.type === "model_call" && e.purpose === "act") {
      parts = agg(e.context);
      f = { phase: "call", turn: e.turn, t: e.t, input: e.input_tokens, cached: e.cached_tokens };
      frames.push({ ...f, parts, total: total(parts), cost });
      cost += e.cost;
      parts = add(dropPrompt(parts), "assistant", e.message_tokens);
      f = { phase: "output", turn: e.turn, t: e.t + e.dur, output: e.output_tokens, text: e.text };
    } else if (e.type === "tool_call") {
      f = { phase: "tool", turn: e.turn, t: e.t, name: e.name, subject: e.subject };
    } else if (e.type === "tool_result") {
      parts = add(parts, resultKind, e.tokens);
      f = { phase: "result", t: e.t + e.dur, name: e.name, ok: e.ok, tokens: e.tokens };
    } else if (e.type === "compaction") {
      parts = dropPrompt(agg(e.context));
      f = { phase: "compact", turn: e.turn, t: e.t, before: e.before, after: e.after };
    } else if (e.type === "error") {
      if (e.message_tokens > 0) parts = add(parts, e.kind === "malformed" ? "error" : "nudge", e.message_tokens);
      f = { phase: "error", turn: e.turn, t: e.t, kind: e.kind, detail: e.detail };
    } else if (e.type === "run_end" && agent === "main") {
      f = { phase: "done", t: e.t, status: e.status };
    }
    if (f !== null) frames.push({ ...f, parts, total: total(parts), cost });
  }
  return frames;
}

export function toolcallFrames(events: Ev[], agent = "main"): Obj[] {
  const frames: Obj[] = [];
  for (const e of events) {
    if (e.agent !== agent) continue;
    if (e.type === "model_call" && e.purpose === "act")
      frames.push({ phase: "emit", turn: e.turn, text: e.text, tokens: e.output_tokens });
    else if (e.type === "tool_call")
      frames.push({ phase: "parse", turn: e.turn, call: e.call, name: e.name, args: e.args });
    else if (e.type === "error" && e.kind === "malformed")
      frames.push({ phase: "malformed", turn: e.turn, detail: e.detail, tokens: e.message_tokens });
    else if (e.type === "tool_result")
      frames.push({ phase: "result", call: e.call, name: e.name, ok: e.ok, text: e.text, tokens: e.tokens });
    else if (e.type === "run_end") frames.push({ phase: "final", status: e.status, answer: e.answer });
  }
  return frames;
}

export function budgetFrames(events: Ev[], agent = "main"): Obj[] {
  const start = events[0]!;
  const ctx = start.policy.context as Obj;
  const reserve = start.policy.max_output as number;
  const trigger = ctx.trigger * ctx.window - reserve;
  const frames: Obj[] = [];
  for (const e of events) {
    if (e.agent !== agent) continue;
    if (e.type === "model_call" && e.purpose === "act") {
      const parts = agg(e.context);
      frames.push({ phase: "call", turn: e.turn, parts, total: total(parts), window: ctx.window, trigger, reserve, lost: [] });
    } else if (e.type === "compaction") {
      const parts = agg(e.context);
      frames.push({
        phase: "compact", turn: e.turn, parts, total: total(parts), window: ctx.window, trigger, reserve,
        lost: e.facts_lost, before: e.before, after: e.after, strategy: e.strategy,
        removed: e.removed.length, clipped: e.clipped.length,
      });
    } else if (e.type === "error" && e.kind === "context_overflow") {
      frames.push({ phase: "overflow", turn: e.turn, parts: [], total: 0, window: ctx.window, trigger, reserve, lost: [] });
    }
  }
  return frames;
}

export function cacheFrames(events: Ev[]): Obj[] {
  const price = PRICES[events[0]!.policy.cache.price]!;
  const frames: Obj[] = [];
  let cum = 0;
  let cumPlain = 0;
  for (const e of events) {
    if (e.type !== "model_call") continue;
    const plain = (e.input_tokens * price.input + e.output_tokens * price.output) / 1e6;
    cum += e.cost;
    cumPlain += plain;
    frames.push({
      turn: e.turn, agent: e.agent, input: e.input_tokens, cached: e.cached_tokens,
      uncached: e.input_tokens - e.cached_tokens, output: e.output_tokens,
      hit: e.cached_tokens / e.input_tokens, cost: e.cost, plain, cum, cum_plain: cumPlain, ttft: e.ttft,
    });
  }
  return frames;
}

export function permissionFrames(events: Ev[]): Obj[] {
  const frames: Obj[] = [];
  const byCall = new Map<string, Obj>();
  for (const e of events) {
    if (e.type === "tool_call") {
      const f: Obj = { call: e.call, name: e.name, subject: e.subject, t: e.t };
      byCall.set(e.call, f);
      frames.push(f);
    } else if (e.type === "permission_check") {
      const f = byCall.get(e.call)!;
      f.decision = e.decision;
      f.reason = e.reason;
      f.answer = e.answer;
      f.wait = e.wait;
    } else if (e.type === "hook") byCall.get(e.call)!.hook = e.result;
    else if (e.type === "tool_result") {
      const f = byCall.get(e.call)!;
      f.outcome = e.kind;
      f.done = e.t + e.dur;
    }
  }
  return frames;
}

export function timeline(events: Ev[]): Obj[] {
  const spans: Obj[] = [];
  for (const e of events) {
    if (e.type === "model_call")
      spans.push({
        lane: "model", agent: e.agent, start: e.t, end: e.t + e.dur,
        label: e.purpose === "summary" ? "summarise" : `turn ${e.turn}`,
      });
    else if (e.type === "permission_check" && e.wait > 0)
      spans.push({ lane: "human", agent: e.agent, start: e.t, end: e.t + e.wait, label: e.answer ?? "" });
    else if (e.type === "tool_result")
      spans.push({
        lane: "tool", agent: e.agent, start: e.t, end: e.t + e.dur,
        label: e.name + (e.ok ? "" : ` (${e.kind})`),
      });
    else if (e.type === "retry")
      spans.push({ lane: "retry", agent: e.agent, start: e.t, end: e.t + e.backoff, label: `back-off ${e.attempt}` });
  }
  return spans;
}
