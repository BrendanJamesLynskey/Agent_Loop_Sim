"""One SWE-bench-style task, end to end (since 1.8): an issue, an agent's patch, hidden tests.

SWE-bench (Jimenez et al. 2023, arXiv:2310.06770) gives an agent a repository at a base commit and
an issue; the agent edits the code; then tests the agent never saw are added and run. A task counts
as **resolved** when every FAIL_TO_PASS test (written for the issue: fails before the fix) passes and
every PASS_TO_PASS test (passed before) still passes. The field names are SWE-bench's.

Here the repository is the engine's tiny ``tinycalc`` (whose ``add`` subtracts), the agent is the
engine's harness with a scripted model, and the tests run in the engine's fake ``pytest``. Three
scripted agents make three patches:

- ``fix``: ``return a + b``, the right fix;
- ``overfit``: ``return a * b - 1``, which happens to give add(2, 3) = 5 and nothing else right;
- ``collateral``: fixes ``add`` but also turns ``sub`` into an addition.

Two test patches grade them: ``thorough`` (three assertions on ``add``) and ``weak`` (only the
example from the issue). With the weak tests the overfit patch counts as resolved: the "weak test
cases" problem Aleithan et al. (2024, SWE-Bench+, arXiv:2410.06992) found in real SWE-bench tasks.
"""
from __future__ import annotations

from typing import Any

from ..harness import Run
from ..tools import run_pytest

ISSUE = ("add() returns the wrong answer\n\n"
         "`add(2, 3)` returns -1, but it should return 5. sub() and mul() look fine.")

BASE = {
    "README.md": "# tinycalc\n\nThree arithmetic helpers and their tests. Run the tests with `pytest`.\n",
    "calc.py": (
        '"""Tiny calculator functions."""\n\n\n'
        "def add(a, b):\n    return a - b\n\n\n"
        "def sub(a, b):\n    return a - b\n\n\n"
        "def mul(a, b):\n    return a * b\n"
    ),
    "test_calc.py": (
        "from calc import sub, mul\n\n\n"
        "def test_sub():\n    assert sub(7, 4) == 3\n\n\n"
        "def test_mul():\n    assert mul(6, 7) == 42\n"
    ),
}

# The hidden test patch: one new file (FAIL_TO_PASS) and the existing tests (PASS_TO_PASS).
TEST_PATCHES = {
    "thorough": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n"
                "    assert add(10, 0) == 10\n",
    "weak": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
}
FAIL_TO_PASS = ["test_issue.py::test_add"]
PASS_TO_PASS = ["test_calc.py::test_sub", "test_calc.py::test_mul"]

_ADD = "def add(a, b):\n    return a - b"
_SUB = "def sub(a, b):\n    return a - b"


def _script(edits: list[tuple[str, str]], final: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {"thought": "Look at the repository.", "calls": [{"name": "list_files", "args": {}}]},
        {"thought": "Read the code the issue is about.", "calls": [{"name": "read_file", "args": {"path": "calc.py"}}]},
    ]
    for old, new in edits:
        steps.append({"thought": "Change the code.", "calls": [
            {"name": "edit_file", "args": {"path": "calc.py", "old_text": old, "new_text": new}}]})
    steps.append({"thought": "Run the tests.", "calls": [{"name": "run_shell", "args": {"command": "pytest"}}]})
    steps.append({"final": final})
    return steps


PATCHES = {
    "fix": _script([(_ADD, "def add(a, b):\n    return a + b")], "Fixed add(): it subtracted instead of adding."),
    "overfit": _script([(_ADD, "def add(a, b):\n    return a * b - 1")], "add(2, 3) now returns 5."),
    "collateral": _script([(_ADD, "def add(a, b):\n    return a + b"), (_SUB, "def sub(a, b):\n    return a + b")],
                          "Fixed the arithmetic in calc.py."),
}
TITLES = {"fix": "the right fix", "overfit": "an overfit patch", "collateral": "a fix that breaks sub()"}


def scenario(patch: str) -> dict[str, Any]:
    return {"id": f"swe_{patch}", "title": f"Resolve the issue ({TITLES[patch]})",
            "system": "You are a coding agent. Resolve the issue in the repository with the tools.",
            "task": ISSUE, "tools": ["list_files", "read_file", "edit_file", "run_shell"], "files": BASE,
            "facts": [], "script": PATCHES[patch], "seed": 25}


def run_test(files: dict[str, str], test_id: str, test_patch: str) -> dict[str, Any]:
    """Run one test (``file::name``) on ``files`` plus the hidden test file, alone, in the engine's
    pytest. The test file is cut down to the import line and the one test."""
    path, name = test_id.split("::")
    src = TEST_PATCHES[test_patch] if path == "test_issue.py" else files[path]
    blocks = src.split("\n\n\n")
    keep = [blocks[0]] + [b for b in blocks[1:] if b.startswith(f"def {name}(")]
    impl = {p: c for p, c in files.items() if p.endswith(".py") and not p.split("/")[-1].startswith("test_")}
    impl[path] = "\n\n\n".join(keep)
    code, out = run_pytest(impl)
    return {"test": test_id, "passed": code == 0, "output": out}


def diff_lines(before: str, after: str) -> list[dict[str, Any]]:
    """A line diff for same-length files (the scripted edits change lines in place)."""
    a = before.split("\n")
    b = after.split("\n")
    out: list[dict[str, Any]] = []
    for i in range(len(a)):
        if a[i] == b[i]:
            out.append({"line": i + 1, "kind": "same", "text": a[i]})
        else:
            out.append({"line": i + 1, "kind": "del", "text": a[i]})
            out.append({"line": i + 1, "kind": "add", "text": b[i]})
    return out


def grade(files: dict[str, str], test_patch: str) -> dict[str, Any]:
    before = {t: run_test(BASE, t, test_patch)["passed"] for t in FAIL_TO_PASS + PASS_TO_PASS}
    after = {t: run_test(files, t, test_patch)["passed"] for t in FAIL_TO_PASS + PASS_TO_PASS}
    f2p = all(after[t] for t in FAIL_TO_PASS)
    p2p = all(after[t] for t in PASS_TO_PASS)
    return {"before": before, "after": after, "f2p_ok": f2p, "p2p_ok": p2p, "resolved": f2p and p2p}


def anatomy(patch: str, test_patch: str, tokenizer: Any = None) -> dict[str, Any]:
    """The task as the chapter animates it: the agent's run (its tool calls, from the trace), the
    patch as a diff, the hidden tests before and after, and the verdict; ``frames`` lists them in
    order, each with a caption."""
    run = Run(scenario(patch), {"permissions": {"mode": "auto"}}, None, None, tokenizer)
    events = run.run()
    files = run.world.files
    calls = []
    for e in events:
        if e["type"] == "tool_call":
            calls.append({"name": e["name"], "args": e["args"]})
        elif e["type"] == "tool_result":
            calls[-1]["ok"] = e["ok"]
            calls[-1]["text"] = e["text"]
    end = events[-1]
    g = grade(files, test_patch)
    frames: list[dict[str, Any]] = [{"stage": "task", "caption":
                                     "The task: a repository at its base commit and an issue. The tests that will "
                                     "grade the fix are hidden from the agent."}]
    for i, c in enumerate(calls):
        what = c["name"] + ("(" + ", ".join(f"{k}={v!r}" for k, v in c["args"].items() if k == "path" or k == "command") + ")")
        frames.append({"stage": "agent", "call": i, "caption": f"Agent step {i + 1}: {what}."})
    frames.append({"stage": "patch", "caption": "The agent stops; its patch is the diff of the repository against the base commit."})
    for t in FAIL_TO_PASS + PASS_TO_PASS:
        kind = "FAIL_TO_PASS" if t in FAIL_TO_PASS else "PASS_TO_PASS"
        b = "passes" if g["before"][t] else "fails"
        a = "passes" if g["after"][t] else "fails"
        frames.append({"stage": "test", "test": t, "caption":
                       f"{kind} {t}: {b} at the base commit, {a} with the patch."})
    verdict = ("Resolved: every FAIL_TO_PASS test now passes and every PASS_TO_PASS test still does."
               if g["resolved"] else
               "Not resolved: " + ("a FAIL_TO_PASS test still fails" if not g["f2p_ok"] else "a PASS_TO_PASS test now fails")
               + ".")
    frames.append({"stage": "verdict", "caption": verdict})
    return {"patch": patch, "test_patch": test_patch, "issue": ISSUE, "base": BASE, "hidden": TEST_PATCHES[test_patch],
            "calls": calls, "answer": end.get("answer"), "files": files,
            "diff": diff_lines(BASE["calc.py"], files["calc.py"]), "grade": g, "frames": frames,
            "fail_to_pass": FAIL_TO_PASS, "pass_to_pass": PASS_TO_PASS}
