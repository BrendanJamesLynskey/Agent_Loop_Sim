/**
 * Engine 1.3.0 port against the Python reference (fixtures/protocols2_fixtures.json) with no
 * tolerance, and A2A against the official A2A SDK's recorded exchanges
 * (fixtures/a2a_sdk_exchanges.json) after renumbering IDs and blanking timestamps.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { Rng, Tokenizer, protocols as P, type Obj } from "../src/index";

const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/protocols2_fixtures.json"), "utf8")) as Obj;
const sdk = JSON.parse(readFileSync(join(ROOT, "fixtures/a2a_sdk_exchanges.json"), "utf8")) as Obj;
const tok = new Tokenizer(readFileSync(join(ROOT, "src/agent_loop_sim/data/qwen2.5-merges.txt"), "utf8"));
const A = P.a2a;

describe("engine version", () => {
  it("matches the fixtures", () => expect(fx.engine).toBe("1.4.0"));
});

describe("the official A2A SDK's recordings", () => {
  for (const name of Object.keys(A.A2A_SCENARIOS)) {
    it(`${name}: the TS engine reproduces the SDK exchange`, () => {
      expect(A.normalise(A.play(A.a2aScenario(name)).wire)).toEqual(sdk.scenarios[name]);
    });
  }
});

describe("every A2A scenario equals the Python reference", () => {
  for (const want of fx.a2a as Obj[]) {
    it(want.name, () => {
      const p = A.play(A.a2aScenario(want.name));
      expect(p.wire).toEqual(want.wire);
      expect(p.log).toEqual(want.log);
      expect(p.card).toEqual(want.card);
      expect(P.a2aFrames(p)).toEqual(want.frames);
      expect(P.a2aSummary(p)).toEqual(want.summary);
      expect(A.normalise(p.wire)).toEqual(want.normalised);
    });
  }
});

describe("A2A helpers", () => {
  it("uuid4", () => {
    const r = new Rng(3);
    expect([0, 1, 2, 3, 4].map(() => A.uuid4(r))).toEqual(fx.uuid4);
  });
  it("timestamps", () => {
    expect([0, 1, 999, 1000, 61001, 3599999, 3600000, 7891].map((ms) => A.timestamp(ms))).toEqual(fx.timestamps);
  });
  it("sorted compact JSON and the card's ETag", () => {
    expect([A.CARD, { b: [1, { d: null, c: true }], a: "x" }, [], {}].map((x) => A.sortedCompact(x))).toEqual(fx.sorted_compact);
    expect(A.cardEtag(A.CARD)).toBe(fx.card_etag);
  });
});

describe("the gateway", () => {
  for (const want of fx.gateway as Obj[]) {
    it(want.policy, () => expect(P.gatewayRun(want.policy, tok)).toEqual(want));
  }
});

describe("attacks and defences", () => {
  for (const want of fx.security as Obj[]) {
    it(`${want.attack} defended=${want.defended}`, () => {
      const r = P.runSecurity(want.attack, want.defended);
      expect({ ...r, frames: P.flowFrames(r) }).toEqual(want);
    });
  }
  it("tool hashes", () => {
    expect([P.toolHash(P.NOTES_TOOL), P.toolHash({ ...P.NOTES_TOOL, description: P.HIDDEN })]).toEqual(fx.tool_hashes);
  });
});

describe("OAuth flows as chart frames", () => {
  for (const [v, want] of Object.entries(fx.oauth_frames as Record<string, Obj[]>)) {
    it(v, () => expect(P.flowFrames(P.runOauth(v))).toEqual(want));
  }
});
