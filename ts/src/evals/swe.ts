/**
 * One SWE-bench-style task end to end (engine 1.8.0): an issue, an agent's patch (the engine's
 * harness with a scripted model), hidden FAIL_TO_PASS and PASS_TO_PASS tests (the engine's fake
 * pytest), and the verdict. A port of agent_loop_sim/evals/swe.py; the task's data come from the
 * Python reference through evals_data.json.
 */
import type { Obj } from "../data";
import data from "../evals_data.json";
import { Run } from "../harness";
import type { Tokenizer } from "../tokenizer";
import { runPytest } from "../tools";

const SWE = data.swe as Obj;
export const ISSUE: string = SWE.issue;
export const BASE: Record<string, string> = SWE.base;
export const TEST_PATCHES: Record<string, string> = SWE.test_patches;
export const PATCHES: Record<string, Obj[]> = SWE.patches;
export const TITLES: Record<string, string> = SWE.titles;
export const FAIL_TO_PASS: string[] = SWE.fail_to_pass;
export const PASS_TO_PASS: string[] = SWE.pass_to_pass;

export function scenario(patch: string): Obj {
  return {
    id: `swe_${patch}`,
    title: `Resolve the issue (${TITLES[patch]})`,
    system: "You are a coding agent. Resolve the issue in the repository with the tools.",
    task: ISSUE,
    tools: ["list_files", "read_file", "edit_file", "run_shell"],
    files: BASE,
    facts: [],
    script: PATCHES[patch],
    seed: 25,
  };
}

export function runTest(files: Record<string, string>, testId: string, testPatch: string): Obj {
  const [path, name] = testId.split("::") as [string, string];
  const src = path === "test_issue.py" ? TEST_PATCHES[testPatch]! : files[path]!;
  const blocks = src.split("\n\n\n");
  const keep = [blocks[0]!, ...blocks.slice(1).filter((b) => b.startsWith(`def ${name}(`))];
  const impl: Record<string, string> = {};
  for (const [p, c] of Object.entries(files)) {
    const base = p.split("/").pop()!;
    if (p.endsWith(".py") && !base.startsWith("test_")) impl[p] = c;
  }
  impl[path] = keep.join("\n\n\n");
  const [code, out] = runPytest(impl);
  return { test: testId, passed: code === 0, output: out };
}

export function diffLines(before: string, after: string): Obj[] {
  const a = before.split("\n");
  const b = after.split("\n");
  const out: Obj[] = [];
  for (let i = 0; i < a.length; i++) {
    if (a[i] === b[i]) out.push({ line: i + 1, kind: "same", text: a[i] });
    else {
      out.push({ line: i + 1, kind: "del", text: a[i] });
      out.push({ line: i + 1, kind: "add", text: b[i] });
    }
  }
  return out;
}

export function grade(files: Record<string, string>, testPatch: string): Obj {
  const before: Record<string, boolean> = {};
  const after: Record<string, boolean> = {};
  for (const t of [...FAIL_TO_PASS, ...PASS_TO_PASS]) before[t] = runTest(BASE, t, testPatch).passed;
  for (const t of [...FAIL_TO_PASS, ...PASS_TO_PASS]) after[t] = runTest(files, t, testPatch).passed;
  const f2p = FAIL_TO_PASS.every((t) => after[t]);
  const p2p = PASS_TO_PASS.every((t) => after[t]);
  return { before, after, f2p_ok: f2p, p2p_ok: p2p, resolved: f2p && p2p };
}

export function anatomy(patch: string, testPatch: string, tokenizer: Tokenizer): Obj {
  const run = new Run(scenario(patch), { permissions: { mode: "auto" } }, null, null, tokenizer);
  const events = run.run();
  const files = run.world.files as Record<string, string>;
  const calls: Obj[] = [];
  for (const e of events) {
    if (e.type === "tool_call") calls.push({ name: e.name, args: e.args });
    else if (e.type === "tool_result") {
      calls[calls.length - 1]!.ok = e.ok;
      calls[calls.length - 1]!.text = e.text;
    }
  }
  const end = events[events.length - 1]!;
  const g = grade(files, testPatch);
  const frames: Obj[] = [
    {
      stage: "task",
      caption:
        "The task: a repository at its base commit and an issue. The tests that will grade the fix are hidden from the agent.",
    },
  ];
  calls.forEach((c, i) => {
    const shown = Object.entries(c.args as Obj)
      .filter(([k]) => k === "path" || k === "command")
      .map(([k, v]) => `${k}='${v}'`)
      .join(", ");
    frames.push({ stage: "agent", call: i, caption: `Agent step ${i + 1}: ${c.name}(${shown}).` });
  });
  frames.push({ stage: "patch", caption: "The agent stops; its patch is the diff of the repository against the base commit." });
  for (const t of [...FAIL_TO_PASS, ...PASS_TO_PASS]) {
    const kind = FAIL_TO_PASS.includes(t) ? "FAIL_TO_PASS" : "PASS_TO_PASS";
    const b = g.before[t] ? "passes" : "fails";
    const a = g.after[t] ? "passes" : "fails";
    frames.push({ stage: "test", test: t, caption: `${kind} ${t}: ${b} at the base commit, ${a} with the patch.` });
  }
  const verdict = g.resolved
    ? "Resolved: every FAIL_TO_PASS test now passes and every PASS_TO_PASS test still does."
    : "Not resolved: " + (!g.f2p_ok ? "a FAIL_TO_PASS test still fails" : "a PASS_TO_PASS test now fails") + ".";
  frames.push({ stage: "verdict", caption: verdict });
  return {
    patch,
    test_patch: testPatch,
    issue: ISSUE,
    base: BASE,
    hidden: TEST_PATCHES[testPatch],
    calls,
    answer: end.answer ?? null,
    files,
    diff: diffLines(BASE["calc.py"]!, files["calc.py"]!),
    grade: g,
    frames,
    fail_to_pass: FAIL_TO_PASS,
    pass_to_pass: PASS_TO_PASS,
  };
}
