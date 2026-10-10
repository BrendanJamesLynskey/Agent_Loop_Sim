/**
 * The protocols port against the Python reference's fixtures (fixtures/protocols_fixtures.json)
 * with no tolerance, and against the official MCP SDK's recorded exchanges
 * (fixtures/sdk_exchanges.json) after renumbering IDs.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { Rng, Tokenizer, protocols as P, type Obj } from "../src/index";

const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/protocols_fixtures.json"), "utf8")) as Obj;
const sdk = JSON.parse(readFileSync(join(ROOT, "fixtures/sdk_exchanges.json"), "utf8")) as Obj;
const tok = new Tokenizer(readFileSync(join(ROOT, "src/agent_loop_sim/data/qwen2.5-merges.txt"), "utf8"));

describe("engine version", () => {
  it("matches the fixtures", () => expect(fx.engine).toBe("1.6.0"));
});

describe("the official MCP SDK's recordings", () => {
  for (const sc of P.SCENARIOS) {
    it(`${sc.name}: the TS engine reproduces the SDK exchange`, () => {
      expect(P.normalise(P.play(P.protocolScenario(sc.name)).wire)).toEqual(sdk.scenarios[sc.name]);
    });
  }
});

describe("every scenario equals the Python reference", () => {
  for (const want of fx.plays as Obj[]) {
    it(want.name, () => {
      const p = P.play(P.protocolScenario(want.name));
      expect(p.wire).toEqual(want.wire);
      expect(p.log).toEqual(want.log);
      expect(P.sequenceFrames(p)).toEqual(want.sequence);
      expect(P.negotiation(p)).toEqual(want.negotiation);
      expect(P.frameSession(p.log, p.wire, "stdio")).toEqual(want.stdio);
      if (want.http) expect(P.frameSession(p.log, p.wire, "http")).toEqual(want.http);
    });
  }
});

describe("views", () => {
  it("journeys", () => {
    for (const [n, j] of Object.entries(fx.journeys as Obj)) expect(P.journey(P.play(P.protocolScenario(n)))).toEqual(j);
  });
  it("three ways (with the Qwen tokenizer)", () => {
    for (const [n, rows] of Object.entries(fx.three_ways as Obj)) {
      expect(P.threeWays(P.play(P.protocolScenario(n)), tok)).toEqual(rows);
    }
  });
  it("integration frames", () => {
    for (const f of fx.integration as Obj[]) expect(P.integrationFrames(f.n, f.m)).toEqual(f);
  });
  it("stream drops", () => {
    for (const s of fx.stream_drop as Obj[]) expect(P.streamDrop(s.era, s.n, s.drop_after)).toEqual(s);
  });
});

describe("oauth", () => {
  it("SHA-256 and PKCE equal hashlib (incl. the RFC 7636 vector)", () => {
    for (const c of fx.pkce as Obj[]) {
      expect(P.sha256Hex(c.verifier)).toBe(c.sha256);
      expect(P.pkceChallenge(c.verifier)).toBe(c.challenge);
    }
    expect(P.pkceChallenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")).toBe("E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
  });
  it("percent-encoding and random strings", () => {
    for (const c of fx.pct as Obj[]) expect(P.pct(c.s)).toBe(c.out);
    expect([0, 7, 42].map((s) => P.randomString(new Rng(s), 43))).toEqual(fx.rng_strings);
  });
  it("every variant, step for step", () => {
    const seeds = [...Object.keys(P.VARIANTS).map((v) => [v, 7] as const), ["ok", 42] as const];
    expect(seeds.map(([v, s]) => P.runOauth(v, s))).toEqual(fx.oauth);
  });
});
