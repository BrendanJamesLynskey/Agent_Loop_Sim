/**
 * Engine 1.10.0's policy module against the Python reference (fixtures/policy_fixtures.json) with no
 * tolerance: every label operation, every decision under two grants, every plan run (three grants, both
 * modes, approvals on and off), capabilities on every subset of the tools, tokens, every sandbox layer set,
 * the defence-in-depth closed forms and simulation, and every chapter study's frames.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { policy, type Obj } from "../src/index";

const { labels: L, catalogue: K, engine: E, capabilities: C, depth: D, study: T } = policy;
const ROOT = join(__dirname, "..", "..");
const fx = JSON.parse(readFileSync(join(ROOT, "fixtures/policy_fixtures.json"), "utf8")) as Obj;
const norm = (x: unknown): unknown => JSON.parse(JSON.stringify(x));
const NARROW = {
  tools: ["read_notes", "summarise", "send_message"],
  scopes: ["mail:send", "notes:read"],
  egress: ["self", "org"],
  private_dests: ["self"],
};
const GRANTS: Record<string, Obj> = { broad: K.BROAD, narrow: NARROW };

describe("version", () => {
  it("is 1.12.0", () => expect(fx.engine).toBe("1.12.0"));
});

describe("labels", () => {
  it("every pair", () =>
    fx.lattice.forEach((r: Obj) => {
      expect(L.leq(r.a, r.b)).toBe(r.leq);
      expect(L.join(r.a, r.b)).toEqual(r.join);
      expect(L.meet(r.a, r.b)).toEqual(r.meet);
    }));
  it("join_all and text", () => {
    const labs = L.allLabels();
    expect([L.joinAll([]), L.joinAll(labs), L.joinAll(labs.slice(0, 3))]).toEqual(fx.join_all);
    expect(labs.map(L.text)).toEqual(fx.text);
  });
});

describe("the policy engine", () => {
  it("every decision", () =>
    fx.decisions.forEach((d: Obj) => expect(E.decide(d.tool, d.label, d.dest, GRANTS[d.grant]!)).toEqual(d.out)));
  it("every plan run", () =>
    fx.runs.forEach((r: Obj) => {
      const g = r.grant === "needed" ? C.needed(K.PLANS[r.plan]!) : GRANTS[r.grant]!;
      const st = E.runPlan(K.PLANS[r.plan]!, g, r.mode, r.approve);
      expect(norm(st)).toEqual(r.steps);
      expect(E.summary(st)).toEqual(r.summary);
    }));
});

describe("capabilities", () => {
  const c = fx.caps;
  it("least privilege", () => {
    for (const p of K.PLAN_ORDER) {
      expect(C.needed(K.PLANS[p]!)).toEqual(c.needed[p]);
      expect(C.overGrant(K.BROAD, K.PLANS[p]!)).toEqual(c.over[p]);
    }
  });
  it("trifecta and exposure on every subset of the tools", () =>
    c.subsets.forEach((s: Obj) => {
      const g = T.grantFor(s.tools);
      expect(C.trifecta(g)).toEqual(s.trifecta);
      expect(C.exposure(g)).toEqual(s.exposure);
    }));
  it("tokens", () => {
    expect(C.mint(K.BROAD)).toEqual(c.mint.broad);
    expect(C.mint(NARROW)).toEqual(c.mint.narrow);
    const toks = C.mint(K.BROAD);
    const extra = [null, { aud: "notes-api", scopes: [] }];
    c.tokens.forEach((t: Obj, i: number) => {
      const tok = i < c.tokens.length - 2 ? toks[t.aud] : extra[i - (c.tokens.length - 2)];
      expect(C.checkToken(tok, t.tool)).toEqual(t.out);
    });
  });
});

describe("sandbox and depth", () => {
  it("every layer set", () => expect(norm(T.allLayerSets())).toEqual(fx.sandbox));
  it("closed forms", () => {
    const d = fx.depth;
    d.independent.forEach(([m, v]: [number[], number]) => expect(D.allMissIndependent(m)).toBe(v));
    d.common.forEach(([m, c, v]: [number[], number, number]) => expect(D.allMissCommon(m, c)).toBe(v));
    d.corr.forEach(([a, b, c, v]: number[]) => expect(D.pairCorrelation(a!, b!, c!)).toBe(v));
    d.any.forEach(([p, n, v]: number[]) => expect(D.atLeastOne(p!, n!)).toBe(v));
    expect(D.simulate([0.2, 0.3, 0.5], 0.05, 500, 7)).toEqual(d.sim);
  });
});

describe("chapter studies", () => {
  const s = fx.study;
  it("boundaries", () => {
    for (const p of K.PLAN_ORDER)
      for (const m of ["context", "dataflow"]) expect(norm(T.boundaryFrames(p, m))).toEqual(s.boundary[`${p}|${m}`]);
  });
  it("context against dataflow", () => {
    for (const p of K.PLAN_ORDER) expect(norm(T.contextFrames(p))).toEqual(s.context[p]);
  });
  it("trifecta", () => expect(norm(T.trifectaStudy())).toEqual(s.trifecta));
  it("least privilege", () => expect(norm(T.privilegeStudy())).toEqual(s.privilege));
  it("sandbox", () => expect(norm(T.sandboxFrames())).toEqual(s.sandbox));
  it("depth", () => expect(norm(T.depthStudy())).toEqual(s.depth));
});
