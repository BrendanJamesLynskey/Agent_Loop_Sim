/**
 * Engine 1.6.0's orchestration module against the Python reference
 * (fixtures/orchestration_fixtures.json) with no tolerance: every session's results and every
 * checkpoint history, layouts, super-step and timeline frames, every durable-execution run (frames
 * included) and the seeded charge sweeps; and against the committed LangGraph recording.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { orchestration as O, type Obj } from "../src/index";

const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/orchestration_fixtures.json"), "utf8")) as Obj;
const rec = JSON.parse(readFileSync(join(ROOT, "fixtures/langgraph_recordings.json"), "utf8")) as Obj;

describe("engine version", () => {
  it("matches the fixtures", () => expect(fx.engine).toBe("1.6.0"));
});

describe("helpers", () => {
  it("names", () => expect([[], ["a"], ["a", "b"], ["a", "b", "c"]].map((x) => O.names(x))).toEqual(fx.names));
  it("fmt", () => expect([1, "x", [1, "a"], { k: [true, null] }, {}, []].map((x) => O.fmt(x))).toEqual(fx.fmt));
  it("defaults", () => {
    for (const g of O.GRAPHS) expect(O.defaultValues(g)).toEqual(fx.defaults[g.name]);
  });
  it("applyWrites", () =>
    expect([
      O.applyWrites(O.graph("router_view"), { log: [] }, [
        ["p", { log: ["p"], n: 1 }],
        ["q", { x: 7 }],
      ]),
    ]).toEqual(fx.apply));
  it("routes", () =>
    expect(
      [{ log: [1, 2] }, { log: [1] }, {}].map((st) =>
        O.route({ if: { len: "log", op: ">=", value: 2 }, then: { goto: ["a", "b"] }, else: "c" }, st),
      ),
    ).toEqual(fx.routes));
  it("rejects an overwrite conflict", () =>
    expect(() =>
      O.applyWrites(O.graph("merge_overwrite"), {}, [
        ["a", { answer: 1 }],
        ["b", { answer: 2 }],
      ]),
    ).toThrow(O.InvalidUpdateError));
  it("rejects a bad spec", () => {
    const g = O.graph("chain");
    g.edges.push({ from: "nope", to: "draft" });
    expect(() => O.validate(g)).toThrow(O.GraphError);
  });
});

describe("sessions (every checkpoint)", () => {
  for (const s of fx.sessions as Obj[])
    it(s.name, () => expect(O.runSession(O.graph(s.graph), O.session(s.name).ops).ops).toEqual(s.ops));
});

describe("the committed LangGraph recording", () => {
  it("is langgraph 1.2.14", () => expect(rec.langgraph).toBe("1.2.14"));
  for (const s of rec.sessions as Obj[])
    it(s.name, () => {
      const g = O.graph(s.graph);
      const r = O.runSession(g, O.session(s.name).ops);
      (r.ops as Obj[]).forEach((o, i) => {
        expect(O.outcome(o.result)).toEqual(s.ops[i].outcome);
        expect(O.comparableHistory(g, o.history)).toEqual(s.ops[i].history);
      });
    });
});

describe("views", () => {
  it("layouts", () => {
    for (const g of O.GRAPHS) expect(O.layout(g)).toEqual(fx.layouts[g.name]);
  });
  for (const s of O.SESSIONS) {
    it(`super-steps ${s.name}`, () => expect(O.sessionSupersteps(O.graph(s.graph), s)).toEqual(fx.supersteps[s.name]));
    it(`timeline ${s.name}`, () => expect(O.timelineFrames(O.graph(s.graph), s)).toEqual(fx.timelines[s.name]));
  }
});

describe("durable execution", () => {
  for (const s of O.DURABLE_SCENARIOS) it(s.name, () => expect(O.runDurable(s.params)).toEqual(fx.durable[s.name]));
  it("the whole grid, every frame", () => {
    for (const r of fx.durable_grid as Obj[]) expect(O.runDurable(r.params)).toEqual(r);
  });
  it("rejects an unknown choose mode", () => expect(() => O.runDurable({ choose: "dice" })).toThrow());
  it("charge sweeps", () => {
    for (const s of fx.sweeps as Obj[]) expect(O.chargeSweep(s.q, s.attempts, s.runs, s.seed)).toEqual(s);
  });
});
