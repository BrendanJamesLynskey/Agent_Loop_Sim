/**
 * Engine 1.9.0's evals additions against the Python reference (fixtures/evals2_fixtures.json) with no
 * tolerance: κ, the judge-error correction and the Pareto front; the simulated judge (every frame); the
 * trajectory grader on scripted and replayed runs; every run of the scaffold study through the TS
 * harness, with its costs; and contamination.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { evals, Tokenizer, type Obj } from "../src/index";

const { stats: S, judge: J, trajectory: T, scaffold: F, contamination: C } = evals;
const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/evals2_fixtures.json"), "utf8")) as Obj;
const tok = new Tokenizer(readFileSync(join(ROOT, "src/agent_loop_sim/data/qwen2.5-merges.txt"), "utf8"));
const norm = (x: unknown): unknown => JSON.parse(JSON.stringify(x));

describe("version", () => {
  it("is 1.9.0", () => expect(fx.engine).toBe("1.9.0"));
});

describe("graders' statistics", () => {
  const s = fx.stats;
  it("Cohen's kappa", () => s.kappa.forEach(([t, v]: [number[][], Obj]) => expect(S.cohenKappa(t)).toEqual(v)));
  it("confusion", () =>
    expect(S.confusion(["A", "B", "tie", "A", "B", "A"], ["A", "A", "tie", "B", "B", "tie"], J.LABELS)).toEqual(s.confusion));
  it("Rogan-Gladen", () => s.rogan_gladen.forEach(([p, se, sp, v]: number[]) => expect(S.roganGladen(p!, se!, sp!)).toBe(v)));
  it("corrected rate", () =>
    s.corrected.forEach(([c, v]: [number[], Obj]) => expect(S.correctedRate(c[0]!, c[1]!, c[2]!, c[3]!, c[4]!, c[5]!)).toEqual(v)));
  it("Pareto front", () => s.pareto.forEach(([pts, v]: [number[][], number[]]) => expect(S.paretoFront(pts)).toEqual(v)));
});

describe("the judge", () => {
  it("every pair, setting and frame", () => expect(norm(J.judgeStudy())).toEqual(fx.judge));
  it("pass/fail grading and its correction", () => expect(norm(J.gradingStudy())).toEqual(fx.grading));
});

describe("trajectories", () => {
  const traces: Record<string, Obj> = {};
  for (const n of ["qwen-fix-test-native", "qwen-fix-test-react", "qwen-lookup-native"])
    traces[n] = JSON.parse(readFileSync(join(ROOT, "traces", `${n}.json`), "utf8"));
  it("every grade and frame", () => expect(norm(T.trajectoryStudy(traces, tok))).toEqual(fx.trajectory));
});

describe("the scaffold study", () => {
  const st = F.scaffoldStudy(tok);
  const sc = fx.scaffold;
  it("every run", () => expect(norm(st.rows)).toEqual(sc.rows));
  it("every configuration", () => {
    expect(norm(st.configs)).toEqual(sc.configs);
    expect(norm(st.spread)).toEqual(sc.spread);
    expect(norm(st.mu)).toEqual(sc.mu);
  });
  it("paired comparisons and descriptions", () => {
    sc.paired.forEach((p: Obj) => expect(norm(F.pairedScaffolds(st, p.a, p.b))).toEqual(p));
    for (const [id, d] of Object.entries(sc.describe as Obj)) expect(F.describe(id)).toBe(d);
    for (const t of F.TASKS) expect(F.filesFor(t)).toEqual(sc.files[t.name]);
  });
  it("full example runs", () => {
    for (const ex of sc.examples as Obj[]) {
      const conf = F.configs().find((c) => c.id === ex.config)!;
      const r = F.runOne(conf, ex.t, ex.j, tok, F.SCAFFOLD, true);
      expect(norm({ config: ex.config, caption: F.captionRun(ex.config, r), ...r })).toEqual(ex);
    }
  });
  it("cost", () => expect(norm(F.costStudy(st))).toEqual(fx.cost));
  it("cost frames", () => {
    for (const a of ["all", "small", "large"])
      expect(norm(F.costFrames(F.costStudy(st), a === "all" ? null : a))).toEqual(fx.cost_frames[a]);
  });
});

describe("contamination", () => {
  it("every level, row and frame", () => expect(norm(C.contaminationStudy())).toEqual(fx.contamination));
  it("n-grams", () => expect(C.ngrams([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 8)).toEqual(fx.ngrams));
});
