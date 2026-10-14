/**
 * Engine 1.11.0's policy module, part two, against the Python reference (fixtures/policy2_fixtures.json) with no
 * tolerance: the benign suite, plans that branch on a value, runs with the policy off, every task under every
 * defence, both frontiers, the seeded simulations, and chapters 6 to 10's studies.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { policy, type Obj } from "../src/index";

const { catalogue: K, engine: E, capabilities: C, utility: U, study: T } = policy;
const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/policy2_fixtures.json"), "utf8")) as Obj;
const norm = (x: unknown): unknown => JSON.parse(JSON.stringify(x));
const SUITE = U.makeSuite();

describe("version", () => {
  it("is 1.11.1", () => expect(fx.engine).toBe("1.11.1"));
});

describe("the suite and the engine's new options", () => {
  it("the benign suite", () => expect(norm(SUITE)).toEqual(fx.suite));
  it("plans that branch on a value", () =>
    fx.ctrl_runs.forEach((r: Obj) => {
      const g = r.grant === "broad" ? K.BROAD : C.needed(K.ADAPTIVE);
      expect(norm(E.runPlan(K.ADAPTIVE, g, r.mode, r.approve, r.enforce))).toEqual(r.steps);
    }));
  it("runs with the policy off", () =>
    fx.open_runs.forEach((r: Obj) =>
      expect(norm(E.runPlan(K.PLANS[r.plan]!, K.BROAD, r.mode, true, false))).toEqual(r.steps),
    ));
  it("approval fatigue", () => fx.approval_miss.forEach(([n, m]: [number, number]) => expect(U.approvalMiss(n)).toBe(m)));
});

describe("cost and risk", () => {
  it("every task under every defence", () => {
    for (const d of K.DEFENCES) {
      const r = U.runDefence(d, SUITE);
      expect(norm(r.rows)).toEqual(fx.rows[d.id]);
      expect(norm(r.summary)).toEqual(fx.summaries.fixed[d.id]);
      const a = U.runDefence(d, SUITE, K.DEPTH.base_range[1], K.UTILITY, true);
      expect(norm(a.summary)).toEqual(fx.summaries.adaptive[d.id]);
    }
  });
  it("both frontiers", () => {
    expect(norm(U.frontier(false, SUITE))).toEqual(fx.frontier.fixed);
    expect(norm(U.frontier(true, SUITE))).toEqual(fx.frontier.adaptive);
  });
  it("the seeded simulations", () =>
    K.DEFENCES.forEach((d, i) => {
      const rows = U.runDefence(d, SUITE).rows.slice(0, 30);
      expect(U.simulate(rows, d, K.DEPTH.base, 20, 5 + i)).toEqual(fx.sims[d.id]);
    }));
});

describe("chapters 6 to 10", () => {
  const st = fx.study;
  it("attempts through the layers", () => {
    for (const c of K.DEPTH.c as number[]) expect(norm(T.depthFrames(c))).toEqual(st.depth[c.toFixed(3)]);
    st.depth_k.forEach((d: Obj, k: number) => expect(norm(T.depthFrames(0.01, k))).toEqual(d));
  });
  it("plan-then-execute", () => {
    for (const p of K.DUAL_PLANS) expect(norm(T.dualFrames(p))).toEqual(st.dual[p]);
  });
  it("approval thresholds", () => expect(norm(T.approvalStudy(SUITE))).toEqual(st.approval));
  it("the frontier study", () => expect(norm(T.frontierStudy(SUITE))).toEqual(st.frontier));
  it("the checklist", () =>
    st.checklist.forEach((c: Obj) => expect(norm(T.checklist(c.design, SUITE))).toEqual(c)));
  it("the defaults need no suite argument", () => {
    expect(norm(T.checklist())).toEqual(st.checklist[st.checklist.length - 4]);
  });
});
