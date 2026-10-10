/**
 * Engine 1.7.0's orchestration module, part 2, against the Python reference
 * (fixtures/orchestration2_fixtures.json) with no tolerance: the patterns' plans, metrics and
 * frames, the discrete-event simulation (every event-driven number, traces and series included),
 * fan-out, reliability, scale and the chooser.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { orchestration as O, type Obj } from "../src/index";

const { patterns: P, des: D, reliability: R, study: S } = O;
const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/orchestration2_fixtures.json"), "utf8")) as Obj;

describe("engine version and constants", () => {
  it("version", () => expect(fx.engine).toBe("1.7.0"));
  it("constants", () => {
    expect(P.TASK).toEqual(fx.consts.task);
    expect(P.MODEL).toEqual(fx.consts.model);
    expect(P.PATTERNS).toEqual(fx.consts.patterns);
    expect(P.TITLES).toEqual(fx.consts.titles);
    expect(D.DEFAULTS).toEqual(fx.consts.des);
    expect(S.FAN).toEqual(fx.consts.fan);
    expect(S.REL).toEqual(fx.consts.rel);
    expect(S.REL_CONFIGS).toEqual(fx.consts.rel_configs);
    expect(S.SCALE).toEqual(fx.consts.scale);
    expect(S.CHOOSE).toEqual(fx.consts.choose);
    expect(S.CONSTRAINTS).toEqual(fx.consts.constraints);
  });
});

describe("helpers", () => {
  it("callP", () =>
    expect(["route", "act", "final"].flatMap((k) => [0, 4000, 4001, 9000, 100000].map((n) => P.callP(k, n, P.MODEL)))).toEqual(fx.call_p));
  it("maj3", () =>
    expect(
      [
        [0.9, 0.9, 0.9],
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.7],
      ].map(([a, b, c]) => P.maj3(a!, b!, c!)),
    ).toEqual(fx.maj3));
  it("pct", () =>
    expect([[], [3.0], [5.0, 1.0, 4.0, 2.0, 3.0]].flatMap((xs) => [1, 50, 95, 99, 100].map((q) => D.pct(xs, q)))).toEqual(fx.pct));
  it("expectedAttempts", () =>
    expect([0.0, 0.03, 0.5, 1.0].flatMap((f) => [1, 3].map((a) => D.expectedAttempts(f, a)))).toEqual(fx.expected_attempts));
  it("wilson", () =>
    expect(
      [
        [0, 10],
        [5, 10],
        [10, 10],
        [731, 1000],
        [0, 0],
      ].map(([k, n]) => R.wilson(k!, n!)),
    ).toEqual(fx.wilson));
});

describe("patterns", () => {
  for (const n of P.PATTERNS) {
    it(`plan ${n}`, () => expect(P.plan(n)).toEqual(fx.plans[n]));
    it(`plan ${n} (16 sub-questions)`, () => expect(P.plan(n, S.taskOf(16, 1500))).toEqual(fx.plans16[n]));
    it(`frames ${n}`, () => expect(P.patternFrames(P.plan(n))).toEqual(fx.frames[n]));
    it(`successClosed ${n}`, () => expect(D.successClosed(P.plan(n), 0.03, 3)).toEqual(fx.success_closed[n]));
  }
  it("planMap", () => expect(P.planMap(6, 1200)).toEqual(fx.plan_map));
  it("compare", () => expect(P.compare()).toEqual(fx.compare));
  it("grid", () => expect(S.patternsGrid([4, 8, 16], [700, 1500], [0.0, 0.01, 0.02])).toEqual(fx.grid));
});

describe("discrete-event simulation", () => {
  for (const n of P.PATTERNS)
    it(`small run ${n} (trace, series)`, () =>
      expect(D.simulate(P.plan(n), { workflows: 5, per_min: 4.0, trace: true, samples: 20, seed: 5 })).toEqual(fx.small[n]));
  it("overload with failures, timeouts and retries", () =>
    expect(
      D.simulate(P.plan("supervisor"), {
        workflows: 12,
        arrival: "batch",
        fail: 0.2,
        timeout_ms: 20000,
        jitter: [0.5, 3.0],
        trace: true,
        samples: 30,
        seed: 9,
      }),
    ).toEqual(fx.overload));
  [40000, 80000, 200000].forEach((it_, i) =>
    it(`fan study ITPM ${it_}`, () => {
      const f = S.fanStudy(S.FAN.chunks, S.FAN.chunk_tokens, S.FAN.widths, S.FAN.rpm, it_, S.FAN.otpm);
      delete f.plan;
      expect(f).toEqual(fx.fans[i]);
    }),
  );
  it("scale sweep", () => expect(S.scaleSweep(S.SCALE.patterns, S.SCALE.rates)).toEqual(fx.scale));
  for (const n of S.SCALE.patterns as string[])
    it(`scale detail ${n}`, () => expect(S.scaleDetail(n, S.SCALE.detail_rate)).toEqual(fx.details[n]));
  (S.CHOOSE.k as number[]).forEach((k, i) =>
    it(`choose table k=${k}`, () => expect(S.chooseTable(k, S.CHOOSE.rates[i])).toEqual(fx.tables[String(k)])),
  );
  it("choices", () => {
    for (const c of fx.choices as Obj[]) expect(S.choose(fx.tables[String(c.k)], c.limits)).toEqual(c.result);
  });
});

describe("reliability", () => {
  it("closed forms", () => {
    const out: Obj[] = [];
    for (const n of [1, 5, 10])
      for (const e of [0.0, 0.05, 0.2])
        for (const w of [0.0, 0.03]) for (const a of [1, 2, 4]) for (const v of [null, S.REL.verifier]) out.push(R.closed(n, e, w, a, v));
    expect(out).toEqual(fx.closed);
  });
  S.REL_CONFIGS.forEach((c, i) => it(`study ${c.name}`, () => expect(S.relStudy(S.REL.e, S.REL.w, c)).toEqual(fx.rel[i])));
});
