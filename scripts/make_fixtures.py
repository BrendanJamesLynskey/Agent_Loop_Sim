"""Write the parity fixtures: what the Python reference produces, for the TS port to match.

    python scripts/make_fixtures.py          # write fixtures/engine_fixtures.json and traces/*.events.jsonl
    python scripts/make_fixtures.py --check  # fail if they are out of date (CI)

The TS tests (ts/test/*.test.ts) recompute every case and require equality: token ids,
random draws, JSON text, parses, and every event of every run, with no tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_sim import VERSION  # noqa: E402
from agent_loop_sim.accounting import LATENCY, PRICES  # noqa: E402
from agent_loop_sim.harness import DEFAULT_POLICY, HUMAN_SEED_OFFSET, replay_trace, run  # noqa: E402
from agent_loop_sim.scenarios import SCENARIOS  # noqa: E402
from agent_loop_sim.tools import TOOL_SPECS  # noqa: E402
from agent_loop_sim.jsonfmt import dumps  # noqa: E402
from agent_loop_sim.parse import parse  # noqa: E402
from agent_loop_sim.rng import Rng  # noqa: E402
from agent_loop_sim.scenarios import scenario  # noqa: E402
from agent_loop_sim.tokenizer import default_tokenizer, merges_text  # noqa: E402
from agent_loop_sim.tools import CalcError, World, evaluate, fmt_number, glob_match, run_pytest, sandbox_violation  # noqa: E402
from agent_loop_sim.sweeps import retry_sweep  # noqa: E402
from agent_loop_sim.trace import dumps_jsonl  # noqa: E402
from agent_loop_sim import views  # noqa: E402

TEXTS = [
    "Hello, world! It's a test. They'll've DON'T  x\n\n  y\t\tz   \n",
    "<|im_start|>user\nFix the failing test.<|im_end|>\n<|im_start|>assistant\n<tool_call>\n"
    '{"name": "read_file", "arguments": {"path": "calc.py"}}\n</tool_call><|im_end|>',
    "def add(a, b):\n    return a - b\n\n\n# 1234567 numbers 3.14159 ünïcödé 日本語テキスト 🙂 emoji\r\n\r\nend",
    "    indented\n        more\n\t\ttabs  trailing   ",
    "x y z　w \u0085 q﻿r",
    "Café is NFC-normalised to Café; Ångström; ﬁ ligature; Ｆｕｌｌｗｉｄｔｈ",
    "Thought: I need the tests.\nAction: run_shell\nAction Input: {\"command\": \"pytest -q\"}",
    "'s 's 'S 'T 're 'RE 've 'm 'll 'LL 'd I'm we'd you're",
    "a\n\n\nb \n c\r\n\r\nd   \n\t e",
    "price: $3.75 / MTok — 1,024 tokens (≈ 0.1×); x² ∑ ∞ → ≤ ≥",
    "",
]

CALC = ["1 + 2 * 3", "(1 + 2) * 3", "7.2 * 2 / 900 * 1000", "8 * 4096 * 0.5 / 1000", "24 - 14.4 - 2", "0.1 + 0.2",
        "-3 * -(2 + 1)", "10 / 4", "1 / 3", "2 / 0", "3 +", "4 ** 2", "abc", "1e5", "123456789012345 * 10",
        "0.00001 * 1", ".5 + .25", "1.2.3"]

GLOBS = [("rm *", "rm -rf build"), ("rm *", "git status"), ("*.py", "calc.py"), ("docs/*", "docs/gpu.md"),
         ("calc.?y", "calc.py"), ("*", ""), ("", ""), ("a*b*c", "aXbYc"), ("a*b*c", "aXbY"), ("pytest*", "pytest -q"),
         ("*/*", "a/b/c"), ("*build*", "rm -rf build/")]

PARSES = [
    ("native", "Done: all tests pass."),
    ("native", '<tool_call>\n{"name": "read_file", "arguments": {"path": "calc.py"}}\n</tool_call>'),
    ("native", '<tool_call>\n{"name": "read_file", "arguments": {"path": "a"}}\n</tool_call>\n<tool_call>\n'
               '{"name": "list_files", "arguments": {}}\n</tool_call>'),
    ("native", '<tool_call>\n{"name": "read_file", "arguments": {"path": "calc.py"}\n</tool_call>'),
    ("native", '<tool_call>\n{"name": "read_file", "arguments": {"path": "calc.py"}}'),
    ("native", '<tool_call>\n{"name": "delete_all", "arguments": {}}\n</tool_call>'),
    ("native", '<tool_call>\n{"name": "read_file", "arguments": {"file": "calc.py"}}\n</tool_call>'),
    ("native", '<tool_call>\n{"name": "read_file", "arguments": "{\\"path\\": \\"x\\"}"}\n</tool_call>'),
    ("native", '<tool_call>\n{"name": "read_file", "arguments": {"path": NaN}}\n</tool_call>'),
    ("native", '<tool_call>\n["read_file"]\n</tool_call>'),
    ("react", "Thought: go\nAction: run_shell\nAction Input: {\"command\": \"pytest\"}\nObservation: fake"),
    ("react", "Thought: done\nFinal Answer: 16 ms"),
    ("react", "Thought: hmm\nAction: run_shell\nAction Input: ```json\n{\"command\": \"pytest\"}\n```"),
    ("react", "Thought: hmm\nAction: run_shell"),
    ("react", "I think the answer is 16."),
    ("react", "Action: calculator\nAction Input: {\"expression\": \"1+1\"}\nFinal Answer: 2"),
]

FIXED_REPO = scenario("fix_test")["files"]
PYTEST_CASES = {
    "buggy": FIXED_REPO,
    "fixed": dict(FIXED_REPO, **{"calc.py": FIXED_REPO["calc.py"].replace("return a - b", "return a + b", 1)}),
    "broken_syntax": dict(FIXED_REPO, **{"calc.py": "def add(a, b):\n    return a +\n"}),
    "missing": {"test_calc.py": FIXED_REPO["test_calc.py"]},
    "no_tests": {"calc.py": FIXED_REPO["calc.py"]},
}

DENY_RM = {"tool": "run_shell", "pattern": "rm *", "decision": "deny"}
ALLOW_TESTS = {"tool": "run_shell", "pattern": "pytest*", "decision": "allow"}
HUMAN_DENY_RM = {"latency_ms": [3000, 9000], "deny": ["rm *"]}

RUNS: list[dict] = [
    {"scenario": "fix_test"},
    {"scenario": "fix_test", "policy": {"style": "react"}},
    {"scenario": "fix_test_react"},
    {"scenario": "fix_test", "policy": {"latency": "local-i7", "cache": {"price": "local"}}},
    {"scenario": "fix_test_malformed"},
    {"scenario": "fix_test_malformed", "policy": {"style": "react"}},
    {"scenario": "fix_test_malformed", "policy": {"recovery": {"malformed": "stop"}}},
    {"scenario": "fix_test_cleanup", "policy": {"permissions": {"mode": "auto"}}},
    {"scenario": "fix_test_cleanup", "policy": {"permissions": {"mode": "auto", "rules": [DENY_RM]}}},
    {"scenario": "fix_test_cleanup", "policy": {"permissions": {"mode": "default", "rules": [DENY_RM, ALLOW_TESTS]}}},
    {"scenario": "fix_test_cleanup", "policy": {"permissions": {"mode": "default", "rules": [], "human": HUMAN_DENY_RM}}},
    {"scenario": "fix_test_cleanup", "policy": {"permissions": {"mode": "ask", "rules": [ALLOW_TESTS], "human": HUMAN_DENY_RM}}},
    {"scenario": "fix_test_cleanup", "policy": {"permissions": {"mode": "read_only", "rules": []}}},
    {"scenario": "fix_test_cleanup", "policy": {"hooks": [
        {"name": "no-rm", "phase": "pre", "tool": "run_shell", "match": "rm *", "action": "block", "message": "deleting files is not allowed"},
        {"name": "quiet-tests", "phase": "pre", "tool": "run_shell", "match": "pytest*", "action": "rewrite", "find": "pytest", "replace": "pytest -q", "latency_ms": 5},
        {"name": "lint", "phase": "post", "tool": "edit_file", "action": "append", "text": "(lint: ok)", "latency_ms": 300},
    ]}},
    {"scenario": "fix_test_flaky"},
    {"scenario": "fix_test_flaky", "seed": 1},
    {"scenario": "fix_test_flaky", "seed": 2, "policy": {"recovery": {"retries": 0}}},
    {"scenario": "fix_test_flaky", "seed": 5, "policy": {"recovery": {"retries": 0, "loop_repeats": 2, "loop_action": "nudge"}}},
    {"scenario": "fix_test_flaky", "seed": 9, "policy": {"recovery": {"retries": 0, "loop_repeats": 2, "loop_action": "stop"}}},
    {"scenario": "fix_test", "policy": {"parallel": False, "max_turns": 3}},
    {"scenario": "research"},
    {"scenario": "research", "policy": {"context": {"window": 3000, "strategy": "none"}}},
    {"scenario": "research", "policy": {"context": {"window": 3000, "strategy": "truncate"}}},
    {"scenario": "research", "policy": {"context": {"window": 3000, "strategy": "clip"}}},
    {"scenario": "research", "policy": {"context": {"window": 3000, "strategy": "summarise"}}},
    {"scenario": "research", "policy": {"context": {"window": 3000, "strategy": "summarise", "keep_last": 2}}},
    {"scenario": "research", "policy": {"style": "react", "context": {"window": 3000, "strategy": "clip"}}},
    {"scenario": "research", "policy": {"cache": {"layout": "timestamp"}}},
    {"scenario": "research", "policy": {"cache": {"layout": "reorder_tools"}}},
    {"scenario": "research", "policy": {"cache": {"enabled": False}}},
    {"scenario": "research", "policy": {"cache": {"price": "claude-haiku-4.5"}}},
    {"scenario": "research", "policy": {"cache": {"price": "gpt-5-mini"}}},
    {"scenario": "research", "policy": {"cache": {"price": "gpt-5-mini", "layout": "timestamp"}}},
    {"scenario": "research_subagent"},
    {"scenario": "research_subagent", "policy": {"parallel": False}},
    {"scenario": "lookup"},
    {"scenario": "lookup", "policy": {"style": "react"}},
    # v1.1: the sandbox around the shell, with and without hooks and rules
    {"scenario": "fix_test_guarded"},
    {"scenario": "fix_test_guarded", "policy": {"sandbox": {"mode": "workspace"}}},
    {"scenario": "fix_test_guarded", "policy": {"sandbox": {"mode": "workspace", "network": True}}},
    {"scenario": "fix_test_guarded", "policy": {"sandbox": {"mode": "read_only"}}},
    {"scenario": "fix_test_guarded", "policy": {"sandbox": {"mode": "workspace"}, "hooks": [
        {"name": "no-rm", "phase": "pre", "tool": "run_shell", "match": "rm *", "action": "block", "message": "deleting files is not allowed"},
        {"name": "quiet-tests", "phase": "pre", "tool": "run_shell", "match": "pytest*", "action": "rewrite", "find": "pytest", "replace": "pytest -q", "latency_ms": 5},
        {"name": "lint", "phase": "post", "tool": "edit_file", "action": "append", "text": "(lint: ok)", "latency_ms": 300},
    ]}},
    {"scenario": "fix_test_guarded", "policy": {"style": "react", "parallel": False,
                                                "permissions": {"mode": "ask", "rules": [], "human": HUMAN_DENY_RM}}},
    {"scenario": "research_subagent", "policy": {"context": {"window": 3000, "strategy": "summarise"}}},
]

SANDBOX_CASES = [
    (None, "rm -rf /"), ({"mode": "off"}, "curl https://example.com"), ({"mode": "workspace"}, "curl -s https://example.com"),
    ({"mode": "workspace", "network": True}, "wget x"), ({"mode": "workspace"}, "pip install requests"),
    ({"mode": "workspace", "network": True}, "pip3 install requests"), ({"mode": "workspace"}, "pip list"),
    ({"mode": "workspace"}, "rm -rf build"), ({"mode": "workspace"}, "rm -rf ~/.cache"), ({"mode": "workspace"}, "rm -f ../x"),
    ({"mode": "workspace"}, "rm a/../../b"), ({"mode": "workspace"}, "rm -rf /tmp/x"), ({"mode": "read_only"}, "rm build"),
    ({"mode": "read_only"}, "rm -rf"), ({"mode": "read_only"}, "pytest"), ({"mode": "workspace"}, ""), ({"mode": "workspace"}, "  ls  "),
]

SWEEPS = [
    {"scenario": "fix_test_flaky", "budgets": [0, 1, 2, 3], "seeds": list(range(40)), "policy": None},
    {"scenario": "fix_test_flaky", "budgets": [0, 1], "seeds": list(range(40)),
     "policy": {"recovery": {"loop_action": "nudge"}}},
]


def run_case(case: dict) -> dict:
    events = run(scenario(case["scenario"]), case.get("policy"), seed=case.get("seed"))
    return {
        "case": case,
        "events": events,
        "views": {
            "loop": views.loop_frames(events),
            "toolcall": views.toolcall_frames(events),
            "budget": views.budget_frames(events),
            "cache": views.cache_frames(events),
            "permission": views.permission_frames(events),
            "timeline": views.timeline(events),
            "agents": views.agents_frames(events),
            "pipeline": views.pipeline_frames(events),
        },
    }


def calc_case(src: str) -> dict:
    try:
        return {"src": src, "ok": True, "out": fmt_number(evaluate(src))}
    except CalcError as e:
        return {"src": src, "ok": False, "out": str(e)}


def build() -> dict[str, str]:
    tok = default_tokenizer()
    rng_cases = []
    for seed in [0, 1, 7, 42, 4294967295, 0x9E3779B9]:
        r = Rng(seed)
        u = [r.next_u32() for _ in range(12)]
        f = [r.random() for _ in range(6)]
        i = [r.randint(1, 6) for _ in range(6)]
        g = [r.uniform(3000, 9000) for _ in range(3)]
        rng_cases.append({"seed": seed, "u32": u, "random": f, "randint": i, "uniform": g})
    world = World({"README.md": "x", "docs/a.md": "a", "docs/b.md": "b", "src/main.py": "y"}, [])
    shell = ["ls", "ls docs", "cat docs/a.md", "cat nope", "echo hi there", "rm -rf docs", "git status", "frobnicate", "python -m pytest", "",
             "pip install pytest-cov", "pip3 install -q a b", "pip install", "pip list"]
    shell_out = []
    for c in shell:
        try:
            shell_out.append({"cmd": c, "ok": True, "out": world.shell(c)})
        except Exception as e:  # ToolError
            shell_out.append({"cmd": c, "ok": False, "out": str(e)})
    traces = {}
    for path in sorted((ROOT / "traces").glob("*.json")):
        t = json.loads(path.read_text())
        events = replay_trace(t)
        traces[t["id"]] = {"events": events, "loop": views.loop_frames(events), "toolcall": views.toolcall_frames(events)}
    fx = {
        "engine": VERSION,
        "merges_sha256": hashlib.sha256(merges_text().encode("utf-8")).hexdigest(),
        "tokenizer": [{"text": t, "ids": tok.encode(t)} for t in TEXTS],
        "rng": rng_cases,
        "dumps": [dumps(x) for x in [{"a": 1, "b": [True, None, "ü\n\"q\""]}, [], {}, " \x01", -5, {"nested": {"k": [1, {"z": "y"}]}}]],
        "glob": [{"p": p, "s": s, "m": glob_match(p, s)} for p, s in GLOBS],
        "calc": [calc_case(c) for c in CALC],
        "pytest": {k: list(run_pytest(v)) for k, v in PYTEST_CASES.items()},
        "shell": shell_out,
        "parse": [{"style": st, "text": tx, "out": parse(tx, st, ["list_files", "read_file", "edit_file", "run_shell", "calculator"])} for st, tx in PARSES],
        "runs": [run_case(c) for c in RUNS],
        "sandbox": [{"sandbox": sb, "cmd": c, "out": sandbox_violation(sb, c)} for sb, c in SANDBOX_CASES],
        "sweeps": [dict(sw, rows=retry_sweep(sw["scenario"], sw["budgets"], sw["seeds"], sw["policy"], tok)) for sw in SWEEPS],
        "traces": traces,
    }
    data = {
        "version": VERSION,
        "tools": TOOL_SPECS,
        "prices": PRICES,
        "latency": LATENCY,
        "default_policy": DEFAULT_POLICY,
        "human_seed_offset": HUMAN_SEED_OFFSET,
        "scenarios": SCENARIOS,
    }
    files = {
        "fixtures/engine_fixtures.json": json.dumps(fx, ensure_ascii=False, separators=(",", ":")) + "\n",
        "ts/src/engine_data.json": json.dumps(data, ensure_ascii=False, indent=1) + "\n",
    }
    for tid, tr in traces.items():
        files[f"traces/{tid}.events.jsonl"] = dumps_jsonl(tr["events"])
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = build()
    stale = []
    for rel, text in files.items():
        p = ROOT / rel
        if a.check:
            if not p.exists() or p.read_text(encoding="utf-8") != text:
                stale.append(rel)
        else:
            p.write_text(text, encoding="utf-8")
    if a.check:
        if stale:
            sys.exit("out of date: " + ", ".join(stale) + " (run python scripts/make_fixtures.py)")
        print(f"fixtures up to date ({len(files)} files)")
    else:
        print(f"wrote {len(files)} files")


if __name__ == "__main__":
    main()
