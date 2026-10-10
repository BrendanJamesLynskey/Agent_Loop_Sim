/**
 * Engine 1.5.0's additions to the context module against the Python reference
 * (fixtures/context2_fixtures.json), with no tolerance: the lossy summariser's window runs (every
 * frame), the long task and its compaction studies, the gain and position curve, every packing
 * evaluation and view, the memory plan and every memory run (every frame), and the long-context
 * against retrieval grid.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { Tokenizer, context as C, type Obj } from "../src/index";

const ROOT = join(__dirname, "..", "..");
const DATA = join(ROOT, "src/agent_loop_sim/data/context");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/context2_fixtures.json"), "utf8")) as Obj;
const tok = new Tokenizer(readFileSync(join(ROOT, "src/agent_loop_sim/data/qwen2.5-merges.txt"), "utf8"));
const files: Record<string, Obj> = {};
for (const n of C.FILES) files[n] = JSON.parse(readFileSync(join(DATA, n), "utf8")) as Obj;
const corpus = new C.Corpus(files["corpus.json"]!, files, tok);
const d = new C.Retriever(corpus, C.DEFAULT);

describe("engine version", () => {
  it("matches the fixtures", () => expect(fx.engine).toBe("1.6.0"));
});

describe("window task with a lossy summariser", () => {
  it("picks the same tasks", () => {
    expect(C.taskQuestions(d, 12)).toEqual(fx.task);
    expect(C.taskQuestions(d, 36)).toEqual(fx.long_task);
  });
  for (const w of fx.lossy as Obj[])
    it(`lossy ${w.policy} @${w.budget}: every frame`, () =>
      expect(C.windowRun(d, fx.task, w.budget, w.policy, "lossy")).toEqual(w));
  for (const w of fx.long_runs as Obj[])
    it(`long task ${w.policy} @${w.budget}`, () => expect(C.windowRun(d, fx.long_task, w.budget, w.policy)).toEqual(w));
  for (const w of fx.long_lossy as Obj[])
    it(`long task lossy @${w.budget}`, () => expect(C.windowRun(d, fx.long_task, w.budget, "compact", "lossy", 0.25, 1)).toEqual(w));
  it("rejects an unknown summariser", () => expect(() => C.windowRun(d, fx.task, 1500, "compact", "magic")).toThrow());
});

describe("compaction studies", () => {
  for (const s of fx.studies as Obj[])
    it(`${s.policy} @${s.budget} loss ${s.loss}`, () =>
      expect(C.compactionStudy(d, fx.long_task, s.budget, s.policy, s.loss, Array.from({ length: 20 }, (_, i) => i + 1))).toEqual(s),
    );
});

describe("packing", () => {
  it("gain and position curve", () => {
    expect(Array.from({ length: 20 }, (_, i) => C.gain(i + 1))).toEqual(fx.gain);
    const pos = fx.position as number[][];
    for (const [x, p] of pos.slice(0, -1)) expect(C.positionP(x!)).toBe(p);
    expect(C.positionP(0.3, { start: 0.9, middle: 0.2, end: 0.4, trough: 0.6 })).toBe(pos[pos.length - 1]![1]);
  });
  it("every question, budget, packer and placement", () => expect(C.packingEval(d, fx.packing.budgets)).toEqual(fx.packing));
  for (const v of fx.packing_views as Obj[]) it(`view q${v.q} @${v.budget}`, () => expect(C.packingView(d, v.q, v.budget)).toEqual(v));
});

describe("memory", () => {
  it("recency", () => {
    for (const [h, x] of fx.recency as number[][]) expect(C.recency(h!)).toBe(x);
  });
  const plan = C.memoryPlan(d);
  it("plan", () => expect(plan).toEqual(fx.memory_plan));
  const all: [string, Obj][] = [...Object.entries(C.MEMORY_POLICIES), ...C.MEMORY_SWEEP];
  it("covers every policy", () => expect(all.map(([n]) => n)).toEqual((fx.memory as Obj[]).map((m) => m.name)));
  all.forEach(([name, p], i) => it(`${name}: every frame`, () => expect(C.memoryRun(d, p, plan, name)).toEqual(fx.memory[i])));
});

describe("long context or retrieval", () => {
  it("the whole grid", () => {
    const t = fx.tradeoff as Obj;
    expect(C.tradeoff(d, [t.sizes.corpus, 200000, 1000000], [3, 5, 10, 20], 50)).toEqual(t);
  });
});
