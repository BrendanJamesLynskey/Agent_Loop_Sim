/**
 * The harness loop: the TS twin of `agent_loop_sim/harness.py`, statement for statement, so
 * every event (and every float in it) matches the Python reference's fixtures exactly.
 */
import { PromptCache, callCost, callLatency } from "./accounting";
import { GENERATION_PROMPT, messageSegment, systemSegment } from "./chat";
import { DEFAULT_POLICY, HUMAN_SEED_OFFSET, LATENCY, PRICES, TOOL_SPECS, VERSION, scenario as getScenario, type Obj } from "./data";
import { dumps, dumpsSorted } from "./jsonfmt";
import { ReplayModel, ScriptedModel, summarise, type Model } from "./models";
import { parse } from "./parse";
import { Rng } from "./rng";
import type { Tokenizer } from "./tokenizer";
import { ToolError, World, execute, globMatch, sandboxViolation } from "./tools";

export type Ev = Obj;

const isPlainObject = (x: unknown): x is Obj =>
  typeof x === "object" && x !== null && !Array.isArray(x);

/** `over` on top of `base`, recursively for nested objects (lists are replaced). */
export function merge(base: Obj, over: Obj | null | undefined): Obj {
  const out: Obj = { ...base };
  if (!over) return out;
  for (const [k, v] of Object.entries(over)) {
    if (isPlainObject(v) && isPlainObject(out[k])) out[k] = merge(out[k], v);
    else out[k] = v;
  }
  return out;
}

const pad2 = (n: number) => String(n).padStart(2, "0");

export function clockText(t: number): string {
  const s = Math.floor(t / 1000);
  const h = 9 + Math.floor(s / 3600);
  const m = Math.floor(s / 60) % 60;
  return `${pad2(h)}:${pad2(m)}:${pad2(s % 60)}`;
}

/** [allow | ask | deny, why]: deny rules, read-only mode, ask mode, ask rules, allow rules, default. */
export function decide(perm: Obj, spec: Obj, subject: string): [string, string] {
  const name = spec.name as string;
  const mode = perm.mode as string;
  const hits = new Map<string, string>();
  for (const r of perm.rules as Obj[]) {
    const pattern = r.pattern ?? "*";
    if ((r.tool === name || r.tool === "*") && globMatch(pattern, subject))
      if (!hits.has(r.decision)) hits.set(r.decision, `rule: ${r.decision} ${r.tool}(${pattern})`);
  }
  if (hits.has("deny")) return ["deny", hits.get("deny")!];
  if (mode === "read_only" && !spec.read_only) return ["deny", "mode: read-only"];
  if (mode === "ask") return ["ask", "mode: ask every time"];
  if (hits.has("ask")) return ["ask", hits.get("ask")!];
  if (hits.has("allow")) return ["allow", hits.get("allow")!];
  if (mode === "auto" || mode === "read_only") return ["allow", `mode: ${mode === "auto" ? "auto" : "read-only"}`];
  if (spec.read_only) return ["allow", "mode: default (reads allowed)"];
  return ["ask", "mode: default (writes ask)"];
}

type Msg = Obj & { id: number; ids: number[] };
type Prompt = { text: string; ids: number[]; context: Obj[] };

export class Run {
  scenario: Obj;
  policy: Obj;
  tok: Tokenizer;
  seed: number;
  rngTools: Rng;
  rngHuman: Rng;
  world: World;
  model: Model;
  price: Obj;
  profile: Obj;
  cache: PromptCache;
  events: Ev[] = [];
  t = 0;
  msgId = 0;
  attempts = new Map<string, number>();
  spawned = 0;
  genIds: number[];
  sysCache = new Map<string, [number[], number]>();

  constructor(scenario: Obj, policy: Obj | null, model: Model | null, seed: number | null, tokenizer: Tokenizer) {
    this.scenario = scenario;
    this.policy = merge(merge(DEFAULT_POLICY, scenario.policy), policy);
    this.tok = tokenizer;
    this.seed = (seed === null ? (scenario.seed ?? 0) : seed) >>> 0;
    this.rngTools = new Rng(this.seed);
    this.rngHuman = new Rng((this.seed + HUMAN_SEED_OFFSET) >>> 0);
    this.world = new World(scenario.files ?? {}, scenario.corpus ?? []);
    this.model = model ?? new ScriptedModel(scenario.script);
    this.price = PRICES[this.policy.cache.price]!;
    this.profile = LATENCY[this.policy.latency]!;
    this.cache = new PromptCache(this.policy.cache.enabled, this.price.min_tokens, this.price.block, this.price.ttl_ms);
    this.genIds = this.tok.encode(GENERATION_PROMPT);
  }

  emit(type: string, agent: string, t: number, fields: Obj = {}): Ev {
    const ev: Ev = { v: 1, seq: this.events.length, t, agent, type, ...fields };
    this.events.push(ev);
    return ev;
  }

  message(role: string, kind: string, content: string, turn: number): Msg {
    this.msgId += 1;
    const m = { id: this.msgId, role, kind, content, turn } as unknown as Msg;
    m.ids = this.tok.encode(messageSegment(m));
    return m;
  }

  toolsFor(names: string[], turn: number): Obj[] {
    let tools = names.map((n) => TOOL_SPECS[n]!);
    if (this.policy.cache.layout === "reorder_tools" && tools.length > 1) {
      const k = turn % tools.length;
      tools = [...tools.slice(k), ...tools.slice(0, k)];
    }
    return tools;
  }

  systemFor(system: string): string {
    if (this.policy.cache.layout === "timestamp") return system + "\nCurrent time: " + clockText(this.t);
    return system;
  }

  prompt(system: string, toolNames: string[], msgs: Msg[], turn: number): Prompt {
    const style = this.policy.style as string;
    const tools = this.toolsFor(toolNames, turn);
    const [seg, bare] = systemSegment(this.systemFor(system), tools, style);
    let hit = this.sysCache.get(seg);
    if (!hit) {
      hit = [this.tok.encode(seg), this.tok.count(bare)];
      this.sysCache.set(seg, hit);
    }
    const [sysIds, bareN] = hit;
    const ids = sysIds.slice();
    const context: Obj[] = [{ id: "system", kind: "system", tokens: bareN }];
    if (sysIds.length > bareN) context.push({ id: "tools", kind: "tools", tokens: sysIds.length - bareN });
    let text = seg;
    for (const m of msgs) {
      for (const i of m.ids) ids.push(i);
      text += messageSegment(m);
      context.push({ id: m.id, kind: m.kind, tokens: m.ids.length });
    }
    for (const i of this.genIds) ids.push(i);
    context.push({ id: "gen", kind: "prompt", tokens: this.genIds.length });
    return { text: text + GENERATION_PROMPT, ids, context };
  }

  promptTokens(system: string, toolNames: string[], msgs: Msg[], turn: number): number {
    return this.prompt(system, toolNames, msgs, turn).ids.length;
  }

  factsIn(msgs: Msg[]): string[] {
    return ((this.scenario.facts ?? []) as Obj[])
      .filter((f) => msgs.some((m) => (m.content as string).includes(f.text)))
      .map((f) => f.id as string);
  }

  modelCall(agent: string, turn: number, purpose: string, p: Prompt, text: string, messageTokens: number): Ev {
    const nIn = p.ids.length;
    const cached = this.cache.lookup(p.ids, this.t);
    const stored = this.cache.store(p.ids, this.t);
    const nOut = this.tok.count(text) + 1;
    const [ttft, dur] = callLatency(this.profile, nIn, cached, nOut);
    const cost = callCost(this.price, nIn, cached, stored, nOut);
    const ev = this.emit("model_call", agent, this.t, {
      turn,
      purpose,
      input_tokens: nIn,
      cached_tokens: cached,
      output_tokens: nOut,
      cost,
      ttft,
      dur,
      window: this.policy.context.window,
      context: p.context,
      text,
      message_tokens: messageTokens,
    });
    this.t += dur;
    return ev;
  }

  compact(agent: string, system: string, toolNames: string[], msgs: Msg[], turn: number): Msg[] {
    const ctx = this.policy.context as Obj;
    const strategy = ctx.strategy as string;
    const window = ctx.window as number;
    const reserve = this.policy.max_output as number;
    const before = this.promptTokens(system, toolNames, msgs, turn);
    if (strategy === "none" || before + reserve <= ctx.trigger * window) return msgs;
    const target = ctx.target * window;
    const keep = ctx.keep_last as number;
    const factsBefore = this.factsIn(msgs);
    let removed: number[] = [];
    const clipped: number[] = [];
    msgs = msgs.slice();
    if (strategy === "clip") {
      for (let i = 1; i < Math.max(1, msgs.length - keep); i++) {
        const m = msgs[i]!;
        if ((m.kind === "tool_result" || m.kind === "observation") && !m.clipped) {
          const lines = (m.content as string).split("\n");
          if (lines.length > ctx.clip_lines) {
            const content =
              lines.slice(0, ctx.clip_lines).join("\n") +
              `\n[... ${lines.length - ctx.clip_lines} more lines clipped by the harness]`;
            const nm = { ...m, content, clipped: true } as Msg;
            nm.ids = this.tok.encode(messageSegment(nm));
            msgs[i] = nm;
            clipped.push(m.id);
          }
        }
      }
    }
    if (
      strategy === "truncate" ||
      (strategy === "clip" && this.promptTokens(system, toolNames, msgs, turn) + reserve > ctx.trigger * window)
    ) {
      while (this.promptTokens(system, toolNames, msgs, turn) + reserve > target && msgs.length > 1 + keep) {
        removed.push(msgs[1]!.id);
        msgs.splice(1, 1);
      }
    }
    if (strategy === "summarise" && msgs.length > 1 + keep) {
      const dropped = msgs.slice(1, msgs.length - keep);
      const text = summarise(dropped, this.scenario.facts ?? []);
      const sysText =
        "Summarise the conversation so far for the agent that will continue it. Keep what matters for the task.";
      const ids = this.tok.encode("<|im_start|>system\n" + sysText + "<|im_end|>\n");
      for (const m of dropped) for (const i of m.ids) ids.push(i);
      for (const i of this.genIds) ids.push(i);
      const p: Prompt = { text: "", ids, context: [{ id: "summary-input", kind: "summary_input", tokens: ids.length }] };
      const summary = this.message("user", "summary", text, turn);
      this.modelCall(agent, turn, "summary", p, text, summary.ids.length);
      removed = dropped.map((m) => m.id);
      msgs = [msgs[0]!, summary, ...msgs.slice(msgs.length - keep)];
    }
    const pa = this.prompt(system, toolNames, msgs, turn);
    const factsAfter = this.factsIn(msgs);
    this.emit("compaction", agent, this.t, {
      turn,
      strategy,
      before,
      after: pa.ids.length,
      removed,
      clipped,
      facts_lost: factsBefore.filter((f) => !factsAfter.includes(f)),
      context: pa.context,
    });
    return msgs;
  }

  fault(name: string): boolean {
    const n = (this.attempts.get(name) ?? 0) + 1;
    this.attempts.set(name, n);
    for (const f of (this.scenario.faults ?? []) as Obj[]) if (f.tool === name && f.attempt === n) return true;
    const rate = ((this.scenario.fail_rates ?? {}) as Obj)[name] ?? 0;
    return rate > 0 && this.rngTools.random() < rate;
  }

  runCall(agent: string, cid: string, call: Obj, start: number): Obj {
    const name = call.name as string;
    const args = call.args as Obj;
    const spec = TOOL_SPECS[name]!;
    const rec = this.policy.recovery as Obj;
    if (name === "task") {
      this.t = start;
      let text: string;
      try {
        text = this.spawn(agent, args.prompt);
      } catch (e) {
        if (!(e instanceof ToolError)) throw e;
        return { ok: false, kind: "error", text: `Error: ${e.message}`, dur: 0 };
      }
      return { ok: true, kind: "ok", text, dur: this.t - start };
    }
    let d = 0;
    let attempt = 0;
    if (name === "run_shell") {
      const blocked = sandboxViolation(this.policy.sandbox, args.command);
      if (blocked !== null) return { ok: false, kind: "sandboxed", dur: spec.latency_ms, text: `exit code: 1\n${blocked}` };
    }
    for (;;) {
      d += spec.latency_ms;
      if (this.fault(name)) {
        if (attempt < rec.retries) {
          const backoff = rec.backoff_ms * rec.backoff_factor ** attempt;
          attempt += 1;
          this.emit("retry", agent, start + d, {
            call: cid,
            attempt,
            backoff,
            error: `TransientError: ${name} timed out`,
          });
          d += backoff;
          continue;
        }
        return { ok: false, kind: "transient", dur: d, text: `TransientError: ${name} timed out after ${attempt + 1} attempts` };
      }
      try {
        return { ok: true, kind: "ok", text: execute(this.world, name, args), dur: d };
      } catch (e) {
        if (!(e instanceof ToolError)) throw e;
        return { ok: false, kind: "error", text: `Error: ${e.message}`, dur: d };
      }
    }
  }

  calls(agent: string, turn: number, calls: Obj[]): Obj[] {
    const perm = this.policy.permissions as Obj;
    const prepared: Obj[] = [];
    calls.forEach((c, k) => {
      const cid = `${agent}:${turn}.${k + 1}`;
      const spec = TOOL_SPECS[c.name]!;
      const args: Obj = { ...c.args };
      const raw = Object.prototype.hasOwnProperty.call(args, spec.subject) ? args[spec.subject] : "";
      let subject: string = typeof raw === "string" ? raw : dumps(raw);
      this.emit("tool_call", agent, this.t, { turn, call: cid, name: c.name, args: { ...args }, subject });
      const [decision, why] = decide(perm, spec, subject);
      let wait = 0;
      let answer: string | null = null;
      if (decision === "ask") {
        const [lo, hi] = perm.human.latency_ms as [number, number];
        wait = this.rngHuman.uniform(lo, hi);
        answer = (perm.human.deny as string[]).some((h) => globMatch(h, subject)) ? "deny" : "approve";
      }
      this.emit("permission_check", agent, this.t, { call: cid, decision, reason: why, answer, wait });
      this.t += wait;
      if (decision === "deny" || answer === "deny") {
        const who = answer === "deny" ? "the user" : why;
        prepared.push({ cid, call: c, args, outcome: "denied", text: `Permission denied (${who}): ${c.name} was not run.` });
        return;
      }
      let outcome: Obj = { cid, call: c, args, outcome: "run", text: "" };
      for (const h of this.policy.hooks as Obj[]) {
        if (h.phase !== "pre" || !(h.tool === c.name || h.tool === "*") || !globMatch(h.match ?? "*", subject)) continue;
        this.t += h.latency_ms ?? 0;
        if (h.action === "block") {
          this.emit("hook", agent, this.t, { call: cid, hook: h.name, phase: "pre", action: "block", result: "blocked" });
          outcome = { cid, call: c, args, outcome: "blocked", text: `Blocked by hook ${h.name}: ${h.message ?? "not allowed"}` };
          break;
        }
        if (h.action === "rewrite" && subject.includes(h.find)) {
          const i = subject.indexOf(h.find);
          subject = subject.slice(0, i) + h.replace + subject.slice(i + (h.find as string).length);
          args[spec.subject] = subject;
          this.emit("hook", agent, this.t, {
            call: cid,
            hook: h.name,
            phase: "pre",
            action: "rewrite",
            result: "rewritten",
            subject,
          });
        }
      }
      prepared.push(outcome);
    });
    const t0 = this.t;
    let cursor = t0;
    let end = t0;
    const results: Obj[] = [];
    for (const p of prepared) {
      const start = this.policy.parallel ? t0 : cursor;
      let r: Obj;
      if (p.outcome === "run") {
        r = this.runCall(agent, p.cid, { name: p.call.name, args: p.args }, start);
        for (const h of this.policy.hooks as Obj[]) {
          if (h.phase === "post" && (h.tool === p.call.name || h.tool === "*") && h.action === "append") {
            r.dur += h.latency_ms ?? 0;
            r.text = r.text + "\n" + h.text;
            this.emit("hook", agent, start + r.dur, {
              call: p.cid,
              hook: h.name,
              phase: "post",
              action: "append",
              result: "appended",
            });
          }
        }
      } else r = { ok: false, kind: p.outcome, text: p.text, dur: 0 };
      r.cid = p.cid;
      r.name = p.call.name;
      r.start = start;
      results.push(r);
      cursor = start + r.dur;
      if (cursor > end) end = cursor;
    }
    this.t = end;
    return results;
  }

  spawn(parent: string, prompt: string): string {
    const subs = (this.scenario.subagents ?? []) as Obj[][];
    if (this.spawned >= subs.length) throw new ToolError("no sub-agent is available");
    const script = subs[this.spawned]!;
    this.spawned += 1;
    const child = `sub${this.spawned}`;
    this.emit("handoff", parent, this.t, { direction: "spawn", child, prompt_tokens: this.tok.count(prompt) });
    const tools = (this.scenario.tools as string[]).filter((n) => n !== "task");
    const sub = this.policy.subagents as Obj;
    const [status, answer, used] = this.loop(child, sub.system, prompt, tools, new ScriptedModel(script), sub.max_turns);
    this.emit("handoff", parent, this.t, {
      direction: "return",
      child,
      status,
      child_tokens: used,
      summary_tokens: this.tok.count(answer),
    });
    return answer;
  }

  loop(agent: string, system: string, task: string, toolNames: string[], model: Model, maxTurns: number): [string, string, number] {
    const style = this.policy.style as string;
    const rec = this.policy.recovery as Obj;
    const ctx = this.policy.context as Obj;
    let msgs: Msg[] = [this.message("user", "task", task, 0)];
    let last: Obj = { kind: "none" };
    let history: string[] = [];
    let used = 0;
    let nCalls = 0;
    for (let turn = 1; turn <= maxTurns; turn++) {
      msgs = this.compact(agent, system, toolNames, msgs, turn);
      const p = this.prompt(system, toolNames, msgs, turn);
      if (p.ids.length + this.policy.max_output > ctx.window) {
        this.emit("error", agent, this.t, {
          turn,
          kind: "context_overflow",
          detail: `${p.ids.length} prompt tokens + ${this.policy.max_output} reserved > window ${ctx.window}`,
          message_tokens: 0,
        });
        return ["context_overflow", "", used];
      }
      const text = model.complete({ prompt: p.text, input_tokens: p.ids.length, style, last, turn });
      const am = this.message("assistant", "assistant", text, turn);
      const ev = this.modelCall(agent, turn, "act", p, text, am.ids.length);
      used += ev.input_tokens + ev.output_tokens;
      msgs.push(am);
      const parsed = parse(text, style, toolNames);
      if (parsed.kind === "final") {
        this.checkpoint(agent, turn, msgs, system, toolNames, nCalls);
        return ["done", parsed.final, used];
      }
      if (parsed.kind === "malformed") {
        const em = this.message("user", "error", `Error: ${parsed.error}. Please try again.`, turn);
        this.emit("error", agent, this.t, {
          turn,
          kind: "malformed",
          detail: parsed.error,
          message_tokens: rec.malformed === "stop" ? 0 : em.ids.length,
        });
        if (rec.malformed === "stop") return ["malformed", "", used];
        msgs.push(em);
        last = { kind: "malformed" };
        this.checkpoint(agent, turn, msgs, system, toolNames, nCalls);
        continue;
      }
      const calls = parsed.calls as Obj[];
      let looped = false;
      if (rec.loop_repeats > 0) {
        for (const c of calls) history.push(c.name + " " + dumpsSorted(c.args));
        const n = rec.loop_repeats as number;
        if (history.length >= n && history.slice(-n).every((h) => h === history[history.length - 1])) looped = true;
      }
      if (looped) {
        const nm = this.message(
          "user",
          "nudge",
          "You have made the same call several times without progress. Try a different approach.",
          turn,
        );
        const stop = rec.loop_action === "stop";
        this.emit("error", agent, this.t, {
          turn,
          kind: "loop",
          detail: `the same call ${rec.loop_repeats} times in a row: ${history[history.length - 1]}`,
          message_tokens: stop ? 0 : nm.ids.length,
        });
        if (stop) return ["loop", "", used];
        history = [];
        msgs.push(nm);
        last = { kind: "nudge" };
        this.checkpoint(agent, turn, msgs, system, toolNames, nCalls);
        continue;
      }
      const results = this.calls(agent, turn, calls);
      nCalls += results.length;
      for (const r of results) {
        const m =
          style === "native"
            ? this.message("tool", "tool_result", r.text, turn)
            : this.message("user", "observation", "Observation: " + r.text, turn);
        msgs.push(m);
        this.emit("tool_result", agent, r.start, {
          call: r.cid,
          name: r.name,
          ok: r.ok,
          kind: r.kind,
          tokens: m.ids.length,
          dur: r.dur,
          text: r.text,
        });
      }
      last = { kind: "results", results: results.map((r) => ({ ok: r.ok, kind: r.kind })) };
      this.checkpoint(agent, turn, msgs, system, toolNames, nCalls);
    }
    return ["max_turns", "", used];
  }

  checkpoint(agent: string, turn: number, msgs: Msg[], system: string, toolNames: string[], nCalls: number): void {
    this.emit("checkpoint", agent, this.t, {
      turn,
      context_tokens: this.promptTokens(system, toolNames, msgs, turn),
      messages: msgs.length,
      tool_calls: nCalls,
    });
  }

  run(): Ev[] {
    const sc = this.scenario;
    this.emit("run_start", "main", 0, {
      engine: VERSION,
      scenario: sc.id,
      task: sc.task,
      backend: this.model.name,
      style: this.policy.style,
      seed: this.seed,
      tools: [...sc.tools],
      policy: this.policy,
    });
    const [status, answer] = this.loop("main", sc.system, sc.task, [...sc.tools], this.model, this.policy.max_turns);
    if (this.model instanceof ReplayModel && this.model.i !== this.model.calls.length)
      throw new Error(`replay used ${this.model.i} of ${this.model.calls.length} recorded calls`);
    this.emit("run_end", "main", this.t, { status, answer, elapsed: this.t, totals: totals(this.events) });
    return this.events;
  }
}

export function totals(events: Ev[]): Obj {
  const out: Obj = {
    model_calls: 0,
    input_tokens: 0,
    cached_tokens: 0,
    output_tokens: 0,
    cost: 0,
    tool_calls: 0,
    failed_calls: 0,
    human_prompts: 0,
    human_wait: 0,
    denied: 0,
    blocked: 0,
    retries: 0,
    errors: 0,
    compactions: 0,
  };
  for (const e of events) {
    const ty = e.type;
    if (ty === "model_call") {
      out.model_calls += 1;
      out.input_tokens += e.input_tokens;
      out.cached_tokens += e.cached_tokens;
      out.output_tokens += e.output_tokens;
      out.cost += e.cost;
    } else if (ty === "tool_result") {
      out.tool_calls += 1;
      if (!e.ok) out.failed_calls += 1;
      if (e.kind === "denied") out.denied += 1;
      if (e.kind === "blocked") out.blocked += 1;
    } else if (ty === "permission_check" && e.decision === "ask") {
      out.human_prompts += 1;
      out.human_wait += e.wait;
    } else if (ty === "retry") out.retries += 1;
    else if (ty === "error") out.errors += 1;
    else if (ty === "compaction") out.compactions += 1;
  }
  return out;
}

/** Run a scenario under a policy and return its events. */
export function run(scenario: Obj, policy: Obj | null, tokenizer: Tokenizer, model: Model | null = null, seed: number | null = null): Ev[] {
  return new Run(scenario, policy, model, seed, tokenizer).run();
}

/** Replay a recorded trace through the harness (same scenario, policy and seed). */
export function replayTrace(trace: Obj, tokenizer: Tokenizer): Ev[] {
  return new Run(getScenario(trace.scenario), trace.policy, new ReplayModel(trace.calls), trace.seed, tokenizer).run();
}
