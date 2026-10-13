/**
 * The harness is part of the result (engine 1.9.0): one simulated model (a seeded stochastic policy,
 * not a language model) on a twelve-task suite under many scaffolds (tools, call style, turn budget),
 * run through the engine's harness, and what each configuration costs. A port of
 * agent_loop_sim/evals/scaffold.py; the suite and configurations come from the Python reference through
 * evals2_data.json. See that file for the model.
 */
import { fixed } from "../context/text";
import type { Obj } from "../data";
import data from "../evals2_data.json";
import { Run } from "../harness";
import { renderStep, type Model } from "../models";
import { Rng } from "../rng";
import type { Tokenizer } from "../tokenizer";
import { runPytest } from "../tools";
import { logistic } from "./mathx";
import * as S from "./stats";

export const TASKS: Obj[] = data.tasks;
export const AGENTS: Obj[] = data.agents;
export const SCAFFOLD: Obj = data.scaffold;

const MAL_OFFSET = 0x5bd1e995;

export function filesFor(task: Obj, expr: string | null = null): Record<string, string> {
  const body = expr === null ? task.bug : expr;
  const calc = `"""One function."""\n\n\ndef ${task.name}(${task.params}):\n    return ${body}\n`;
  const lines = [`from calc import ${task.name}`, "", "", `def test_${task.name}():`];
  for (const [args, want] of task.tests as [string, number][]) lines.push(`    assert ${task.name}(${args}) == ${want}`);
  return { "calc.py": calc, "test_calc.py": lines.join("\n") + "\n" };
}

export function passes(task: Obj, expr: string): boolean {
  return runPytest(filesFor(task, expr))[0] === 0;
}

export function scenarioFor(task: Obj, tools: string[], seed: number): Obj {
  return {
    id: "suite_" + task.name,
    title: "Fix " + task.name + "()",
    system: SCAFFOLD.system,
    task: `${task.name}() in calc.py returns the wrong result. Fix it.`,
    tools,
    files: filesFor(task),
    facts: [],
    script: [],
    seed,
  };
}

export function muOf(task: Obj, theta: number, cfg: Obj = SCAFFOLD): number {
  return logistic(cfg.discrimination * (theta - task.b));
}

export function runSeed(cfg: Obj, t: number, j: number): number {
  return (cfg.seed + 1000 * t + j) >>> 0;
}

export class FixerModel implements Model {
  name = "fixer";
  task: Obj;
  mu: number;
  s: number;
  malformed: number;
  canTest: boolean;
  rEdit: Rng;
  rMal: Rng;
  z: number;
  current: string;
  state = "read";
  pending: Obj | null = null;
  edits = 0;

  constructor(task: Obj, mu: number, rho: number, malformed: number, canTest: boolean, seed: number) {
    this.task = task;
    this.mu = mu;
    this.s = Math.sqrt(rho);
    this.malformed = malformed;
    this.canTest = canTest;
    this.rEdit = new Rng(seed);
    this.rMal = new Rng((seed ^ MAL_OFFSET) >>> 0);
    this.z = this.rEdit.random() < mu ? 1 : 0;
    this.current = task.bug;
  }

  editStep(): Obj {
    const c = this.rEdit.random();
    const own = this.rEdit.random();
    const pick = this.rEdit.random();
    const ok = c < this.s ? this.z : own < this.mu ? 1 : 0;
    let next: string;
    if (ok === 1) next = this.task.right;
    else {
      const wrong: string[] = this.task.wrong;
      let k = Math.floor(pick * wrong.length);
      if (wrong[k] === this.current) k = (k + 1) % wrong.length;
      next = wrong[k]!;
    }
    const old = this.current;
    this.current = next;
    this.edits += 1;
    const thought = this.edits === 1 ? "The bug is in the return line; change it." : "The tests still fail; try another fix.";
    return {
      thought,
      calls: [{ name: "edit_file", args: { path: "calc.py", old_text: "    return " + old, new_text: "    return " + next } }],
    };
  }

  next(req: Obj): Obj {
    if (this.state === "read") {
      this.state = "edit";
      return { thought: "Read the function.", calls: [{ name: "read_file", args: { path: "calc.py" } }] };
    }
    if (this.state === "edit") {
      const step = this.editStep();
      this.state = this.canTest ? "test" : "final";
      return step;
    }
    if (this.state === "test") {
      this.state = "check";
      return { thought: "Run the tests.", calls: [{ name: "run_shell", args: { command: "pytest" } }] };
    }
    if (this.state === "check") {
      const prompt: string = req.prompt;
      const i = prompt.lastIndexOf("exit code: ");
      if (i >= 0 && prompt.slice(i + 11, i + 12) === "0") {
        this.state = "done";
        return { final: `Fixed ${this.task.name}(); the tests pass.` };
      }
      const step = this.editStep();
      this.state = "test";
      return step;
    }
    this.state = "done";
    return { final: `Fixed ${this.task.name}().` };
  }

  complete(req: Obj): string {
    if (req.last.kind !== "malformed" || this.pending === null) this.pending = this.next(req);
    const style: string = req.style;
    const text = renderStep(this.pending, style);
    if ("calls" in this.pending && this.rMal.random() < this.malformed) {
      if (style === "native") return text.split("}}\n</tool_call>").join("}\n</tool_call>");
      return text.slice(0, -1);
    }
    return text;
  }
}

export function configs(cfg: Obj = SCAFFOLD): Obj[] {
  const out: Obj[] = [];
  for (const a of AGENTS)
    for (const tools of ["edit", "test"])
      for (const style of cfg.styles as string[])
        for (const budget of tools === "edit" ? (cfg.budgets as number[]).slice(0, 1) : (cfg.budgets as number[]))
          out.push({ id: `${a.id}|${tools}|${style}|${budget}`, agent: a.id, tools, style, budget });
  return out;
}

export function runOne(conf: Obj, t: number, j: number, tokenizer: Tokenizer, cfg: Obj = SCAFFOLD, events = false): Obj {
  const agent = AGENTS.find((a) => a.id === conf.agent)!;
  const task = TASKS[t]!;
  const seed = runSeed(cfg, t, j);
  const mu = muOf(task, agent.theta, cfg);
  const model = new FixerModel(task, mu, cfg.rho, cfg.malformed[conf.style], conf.tools === "test", seed);
  const policy = { style: conf.style, max_turns: conf.budget, cache: { price: agent.price }, permissions: { mode: "auto" } };
  const run = new Run(scenarioFor(task, cfg.tools[conf.tools], seed), policy, model, seed, tokenizer);
  const ev = run.run();
  const end = ev[ev.length - 1]!;
  const tot = end.totals;
  const resolved = runPytest(run.world.files as Record<string, string>)[0] === 0;
  let malformed = 0;
  for (const e of ev) if (e.type === "error" && e.kind === "malformed") malformed += 1;
  const out: Obj = {
    t,
    j,
    resolved,
    status: end.status,
    turns: tot.model_calls,
    edits: model.edits,
    malformed,
    input: tot.input_tokens,
    cached: tot.cached_tokens,
    output: tot.output_tokens,
    cost: tot.cost,
    elapsed: end.elapsed,
  };
  if (events) {
    out.events = ev;
    out.files = run.world.files;
  }
  return out;
}

export function summarise(conf: Obj, rows: Obj[], cfg: Obj = SCAFFOLD): Obj {
  void cfg;
  const n = rows.length;
  let k = 0;
  let cost = 0.0;
  let turns = 0;
  let mal = 0;
  let tokens = 0;
  for (const r of rows) {
    k += r.resolved ? 1 : 0;
    cost = cost + r.cost;
    turns += r.turns;
    mal += r.malformed;
    tokens += r.input + r.output;
  }
  const succ = new Array<number>(TASKS.length).fill(0);
  const cnt = new Array<number>(TASKS.length).fill(0);
  for (const r of rows) {
    succ[r.t] = succ[r.t]! + (r.resolved ? 1 : 0);
    cnt[r.t] = cnt[r.t]! + 1;
  }
  const cl = S.clusterSe(succ, cnt);
  const z = 1.959963984540054;
  return {
    ...conf,
    n,
    k,
    rate: k / n,
    wilson: S.wilson(k, n),
    cluster: [cl.p - z * cl.se, cl.p + z * cl.se],
    se: cl.se,
    cost_per_run: cost / n,
    cost_per_solved: k > 0 ? cost / k : 0.0,
    turns: turns / n,
    malformed: mal / n,
    tokens: tokens / n,
    per_task: succ.map((s, i) => s / cnt[i]!),
  };
}

export function scaffoldStudy(tokenizer: Tokenizer, cfg: Obj = SCAFFOLD): Obj {
  const outRows: Record<string, Obj[]> = {};
  const summ: Obj[] = [];
  for (const conf of configs(cfg)) {
    const rows: Obj[] = [];
    for (let t = 0; t < TASKS.length; t++) for (let j = 0; j < cfg.runs; j++) rows.push(runOne(conf, t, j, tokenizer, cfg));
    outRows[conf.id] = rows;
    summ.push(summarise(conf, rows, cfg));
  }
  const by: Obj = {};
  for (const s of summ) by[s.id] = s;
  const spread: Obj = {};
  for (const a of AGENTS) {
    const rates = summ.filter((s) => s.agent === a.id).map((s) => s.rate as number);
    const lo = Math.min(...rates);
    const hi = Math.max(...rates);
    spread[a.id] = { min: lo, max: hi, range: hi - lo };
  }
  const mu: Obj = {};
  for (const a of AGENTS) mu[a.id] = TASKS.map((t) => muOf(t, a.theta, cfg));
  return { cfg, tasks: TASKS, agents: AGENTS, mu, configs: summ, by_id: by, rows: outRows, spread };
}

export function pairedScaffolds(study: Obj, a: string, b: string): Obj {
  const xa = (study.rows[a] as Obj[]).map((r) => (r.resolved ? 1 : 0));
  const xb = (study.rows[b] as Obj[]).map((r) => (r.resolved ? 1 : 0));
  const tab = S.pairedTable(xa, xb);
  return { a, b, table: tab, mcnemar: S.mcnemar(tab.n10!, tab.n01!), compare: S.compare(xa, xb) };
}

export function costStudy(study: Obj): Obj {
  const summ = study.configs as Obj[];
  const pts = summ.map((s) => [s.cost_per_run as number, s.rate as number]);
  const front = S.paretoFront(pts);
  const points = summ.map((s, i) => ({
    id: s.id,
    agent: s.agent,
    tools: s.tools,
    style: s.style,
    budget: s.budget,
    cost: s.cost_per_run,
    rate: s.rate,
    wilson: s.wilson,
    per_solved: s.cost_per_solved,
    front: front.includes(i),
  }));
  let bestRate = -Infinity;
  for (const p of points) if (p.rate > bestRate) bestRate = p.rate;
  const byKey = (x: number, y: number, ia: string, ib: string): number => (x !== y ? (x < y ? -1 : 1) : ia < ib ? -1 : ia > ib ? 1 : 0);
  const best = points.filter((p) => p.rate === bestRate).sort((p, q) => byKey(p.cost, q.cost, p.id, q.id));
  const cheap = points.filter((p) => p.rate > 0).sort((p, q) => byKey(p.per_solved, q.per_solved, p.id, q.id));
  return { points, front: front.map((i) => summ[i]!.id), best: best[0]!.id, cheapest_per_solved: cheap[0]!.id };
}

export function describe(confId: string): string {
  const [agent, tools, style, budget] = confId.split("|") as [string, string, string, string];
  const a = AGENTS.find((x) => x.id === agent)!;
  return (
    `${(a.label as string).toLowerCase()}, ${tools === "edit" ? "read/edit only" : "with a shell for the tests"}, ` +
    `${style === "native" ? "native calls" : "ReAct text"}, ${budget} turns`
  );
}

export function captionRun(confId: string, r: Obj): string {
  void confId;
  return (
    `${TASKS[r.t]!.name}(), run ${r.j + 1}: ${r.resolved ? "resolved" : "not resolved"} after ` +
    `${r.turns} model calls and ${r.edits} edit${r.edits !== 1 ? "s" : ""}` +
    (r.malformed ? `, ${r.malformed} malformed` : "") +
    `; $${fixed(r.cost + 0.0, 4)}.`
  );
}

/** Chapter 10's animation: configurations placed cheapest first, with the running Pareto front. */
export function costFrames(cost: Obj, agent: string | null = null): Obj[] {
  const pts = (cost.points as Obj[]).filter((p) => agent === null || p.agent === agent);
  pts.sort((p, q) => {
    if (p.cost !== q.cost) return p.cost < q.cost ? -1 : 1;
    if (p.rate !== q.rate) return p.rate > q.rate ? -1 : 1;
    return p.id < q.id ? -1 : p.id > q.id ? 1 : 0;
  });
  const frames: Obj[] = [];
  for (let i = 0; i < pts.length; i++) {
    const seen = pts.slice(0, i + 1);
    const front = S.paretoFront(seen.map((q) => [q.cost as number, q.rate as number])).map((j) => seen[j]!.id as string);
    const p = pts[i]!;
    const on = front.includes(p.id);
    const d = describe(p.id);
    const cap =
      `${d.slice(0, 1).toUpperCase()}${d.slice(1)}: ${fixed(p.rate * 100.0 + 0.0, 1)}% resolved ` +
      `at $${fixed(p.cost + 0.0, 4)} a task, $${fixed(p.per_solved + 0.0, 4)} per solved task. ` +
      (on
        ? "On the front: nothing placed so far is both cheaper and better."
        : "Off the front: something placed so far is cheaper and at least as good.");
    frames.push({ id: p.id, front, on, caption: cap });
  }
  return frames;
}
