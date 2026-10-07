/**
 * The TS port against the Python reference's fixtures (scripts/make_fixtures.py): token ids,
 * random draws, JSON text, the tools, parsing, and every event and animation frame of every
 * scripted run and recorded-trace replay — all exactly, with no tolerance.
 */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  CalcError,
  Rng,
  Tokenizer,
  World,
  agentsFrames,
  budgetFrames,
  cacheFrames,
  dumps,
  evaluate,
  fmtNumber,
  globMatch,
  loopFrames,
  parse,
  permissionFrames,
  pipelineFrames,
  replayTrace,
  retrySweep,
  run,
  runPytest,
  sandboxViolation,
  scenario,
  timeline,
  toolcallFrames,
  type Obj,
} from "../src/index";

const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/engine_fixtures.json"), "utf8")) as Obj;
const merges = readFileSync(join(ROOT, "src/agent_loop_sim/data/qwen2.5-merges.txt"), "utf8");
const tok = new Tokenizer(merges);

describe("tokenizer", () => {
  it("uses the same merges file", () => {
    expect(createHash("sha256").update(merges, "utf8").digest("hex")).toBe(fx.merges_sha256);
    expect(tok.nMerges).toBe(151387);
  });
  it("encodes every fixture text to the same ids", () => {
    for (const c of fx.tokenizer as Obj[]) expect(tok.encode(c.text), JSON.stringify(c.text)).toEqual(c.ids);
  });
});

describe("rng", () => {
  it("draws the same numbers", () => {
    for (const c of fx.rng as Obj[]) {
      const r = new Rng(c.seed);
      expect(Array.from({ length: 12 }, () => r.nextU32())).toEqual(c.u32);
      expect(Array.from({ length: 6 }, () => r.random())).toEqual(c.random);
      expect(Array.from({ length: 6 }, () => r.randint(1, 6))).toEqual(c.randint);
      expect(Array.from({ length: 3 }, () => r.uniform(3000, 9000))).toEqual(c.uniform);
    }
  });
});

describe("text and tools", () => {
  it("writes JSON like Python's json.dumps", () => {
    const xs = [{ a: 1, b: [true, null, 'ü\n"q"'] }, [], {}, " \x01", -5, { nested: { k: [1, { z: "y" }] } }];
    expect(xs.map(dumps)).toEqual(fx.dumps);
  });
  it("globs", () => {
    for (const g of fx.glob as Obj[]) expect(globMatch(g.p, g.s), `${g.p} ${g.s}`).toBe(g.m);
  });
  it("calculates", () => {
    for (const c of fx.calc as Obj[]) {
      let got: Obj;
      try {
        got = { src: c.src, ok: true, out: fmtNumber(evaluate(c.src)) };
      } catch (e) {
        if (!(e instanceof CalcError)) throw e;
        got = { src: c.src, ok: false, out: e.message };
      }
      expect(got).toEqual(c);
    }
  });
  it("runs the tests of the tiny repository", () => {
    const repo = scenario("fix_test").files as Record<string, string>;
    const cases: Record<string, Record<string, string>> = {
      buggy: repo,
      fixed: { ...repo, "calc.py": repo["calc.py"]!.replace("return a - b", "return a + b") },
      broken_syntax: { ...repo, "calc.py": "def add(a, b):\n    return a +\n" },
      missing: { "test_calc.py": repo["test_calc.py"]! },
      no_tests: { "calc.py": repo["calc.py"]! },
    };
    for (const [k, files] of Object.entries(cases)) expect(runPytest(files), k).toEqual(fx.pytest[k]);
  });
  it("simulates the shell", () => {
    const w = new World({ "README.md": "x", "docs/a.md": "a", "docs/b.md": "b", "src/main.py": "y" }, []);
    for (const c of fx.shell as Obj[]) {
      let got: Obj;
      try {
        got = { cmd: c.cmd, ok: true, out: w.shell(c.cmd) };
      } catch (e) {
        got = { cmd: c.cmd, ok: false, out: (e as Error).message };
      }
      expect(got).toEqual(c);
    }
  });
  it("parses model output identically", () => {
    const allowed = ["list_files", "read_file", "edit_file", "run_shell", "calculator"];
    for (const c of fx.parse as Obj[]) expect(parse(c.text, c.style, allowed), c.text).toEqual(c.out);
  });
});

describe("runs", () => {
  for (const [i, r] of (fx.runs as Obj[]).entries()) {
    it(`#${i} ${r.case.scenario} ${JSON.stringify(r.case.policy ?? {})} seed ${r.case.seed ?? "-"}`, () => {
      const events = run(scenario(r.case.scenario), r.case.policy ?? null, tok, null, r.case.seed ?? null);
      expect(events.length).toBe(r.events.length);
      events.forEach((e, k) => expect(e, `event ${k} (${e.type})`).toEqual(r.events[k]));
      expect(loopFrames(events)).toEqual(r.views.loop);
      expect(toolcallFrames(events)).toEqual(r.views.toolcall);
      expect(budgetFrames(events)).toEqual(r.views.budget);
      expect(cacheFrames(events)).toEqual(r.views.cache);
      expect(permissionFrames(events)).toEqual(r.views.permission);
      expect(timeline(events)).toEqual(r.views.timeline);
      expect(agentsFrames(events)).toEqual(r.views.agents);
      expect(pipelineFrames(events)).toEqual(r.views.pipeline);
    });
  }
});

describe("the sandbox around the shell", () => {
  it("stops the same commands", () => {
    for (const c of fx.sandbox as Obj[]) expect(sandboxViolation(c.sandbox, c.cmd), c.cmd).toEqual(c.out);
  });
});

describe("seeded sweeps", () => {
  for (const sw of fx.sweeps as Obj[]) {
    it(`${sw.scenario} ${JSON.stringify(sw.policy ?? {})}`, () => {
      expect(retrySweep(sw.scenario, sw.budgets, sw.seeds, sw.policy, tok)).toEqual(sw.rows);
    });
  }
});

describe("recorded traces replay identically", () => {
  const files = readdirSync(join(ROOT, "traces")).filter((f) => f.endsWith(".json"));
  it("has at least three recorded traces", () => expect(files.length).toBeGreaterThanOrEqual(3));
  for (const f of files) {
    it(f, () => {
      const t = JSON.parse(readFileSync(join(ROOT, "traces", f), "utf8")) as Obj;
      const events = replayTrace(t, tok);
      const want = fx.traces[t.id] as Obj;
      expect(events).toEqual(want.events);
      expect(loopFrames(events)).toEqual(want.loop);
      expect(toolcallFrames(events)).toEqual(want.toolcall);
      const jsonl = readFileSync(join(ROOT, "traces", f.replace(/\.json$/, ".events.jsonl")), "utf8");
      expect(jsonl.trim().split("\n").map((l) => JSON.parse(l))).toEqual(events);
    });
  }
});
