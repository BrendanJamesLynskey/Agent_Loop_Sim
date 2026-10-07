"""The scenarios: tasks, the small worlds they run in, and scripted policies for them.

- ``fix_test``: a tiny repository whose ``add`` subtracts; find it, fix it, re-run the tests.
  Variants: ``fix_test_react`` (text tool calls), ``fix_test_malformed`` (one broken call),
  ``fix_test_cleanup`` (also tries ``rm -rf build`` and ``git status``, for permissions),
  ``fix_test_flaky`` (the shell fails transiently).
- ``research``: read four design notes and work out two numbers; the notes are long, so the
  context fills up (context budget, prompt caching). ``research_subagent`` hands the
  reading to a sub-agent.
- ``lookup``: search a small corpus and use the calculator (a recorded trace's task).

Every number in the notes is invented for the exercise (``tiny-7b`` and ``dev-gpu`` are not
real products).
"""
from __future__ import annotations

import copy
from typing import Any

CODING_SYSTEM = (
    "You are a coding agent working in a small Python repository. Use the tools to look at files, "
    "run the tests and change code. Make the smallest change that fixes the problem. When you are done, "
    "reply with a one-sentence summary and no tool call."
)
RESEARCH_SYSTEM = (
    "You are a research assistant. Use the tools to read the documents you need and the calculator for "
    "arithmetic. When you have the answer, reply with it and no tool call."
)

REPO = {
    "README.md": "# tinycalc\n\nThree arithmetic helpers and their tests. Run the tests with `pytest`.\n",
    "calc.py": (
        '"""Tiny calculator functions."""\n\n\n'
        "def add(a, b):\n    return a - b\n\n\n"
        "def sub(a, b):\n    return a - b\n\n\n"
        "def mul(a, b):\n    return a * b\n"
    ),
    "test_calc.py": (
        "from calc import add, sub, mul\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_sub():\n    assert sub(7, 4) == 3\n\n\n"
        "def test_mul():\n    assert mul(6, 7) == 42\n"
    ),
    "build/cache.txt": "compiled artefacts from the last run\n",
}

FIX_SCRIPT: list[dict[str, Any]] = [
    {"thought": "First, see what is in the repository.", "calls": [{"name": "list_files", "args": {}}]},
    {"thought": "Run the tests to see what fails.", "calls": [{"name": "run_shell", "args": {"command": "pytest"}}]},
    {"thought": "test_add fails; read the implementation.", "calls": [{"name": "read_file", "args": {"path": "calc.py"}}]},
    {
        "thought": "add subtracts instead of adding. Fix that line.",
        "calls": [
            {
                "name": "edit_file",
                "args": {"path": "calc.py", "old_text": "def add(a, b):\n    return a - b", "new_text": "def add(a, b):\n    return a + b"},
            }
        ],
        "on_error": "skip",
    },
    {"thought": "Run the tests again.", "calls": [{"name": "run_shell", "args": {"command": "pytest"}}]},
    {"final": "Fixed add() in calc.py, which subtracted instead of adding; all 3 tests now pass."},
]

CLEANUP_STEPS: list[dict[str, Any]] = [
    {"thought": "Clean up the build directory.", "calls": [{"name": "run_shell", "args": {"command": "rm -rf build"}}], "on_denied": "skip"},
    {"thought": "Check what changed.", "calls": [{"name": "run_shell", "args": {"command": "git status"}}]},
]

FILLER = [
    "This note is part of the team's design notes. It is kept short on purpose, but it has grown over time as people added details.",
    "Most of the text below is background: why a choice was made, who asked for it, and what was tried first.",
    "When a number matters for capacity planning, it is written as a plain sentence so that it can be found by search.",
    "Older revisions of this note are in the history of the repository, and the meeting notes say when each change was agreed.",
    "If anything here disagrees with the measurements, the measurements win, and the note should be fixed.",
    "Questions about this note go to the platform channel; please do not edit the numbers without a measurement to back them.",
]


def _doc(title: str, facts_at: dict[int, str], lines: int) -> str:
    out = ["# " + title, ""]
    k = 0
    for i in range(lines):
        if i in facts_at:
            out.append(facts_at[i])
        else:
            out.append(FILLER[k % len(FILLER)])
            k += 1
    return "\n".join(out) + "\n"


DOCS = {
    "docs/gpu.md": _doc(
        "The dev-gpu board",
        {2: "The dev-gpu board has 24 GB of memory.", 9: "The dev-gpu memory bandwidth is 900 GB/s."},
        20,
    ),
    "docs/model.md": _doc(
        "The tiny-7b model",
        {1: "The tiny-7b model has 7.2 billion parameters, stored in 2 bytes each.", 11: "The KV cache of tiny-7b needs 0.5 MB per token."},
        20,
    ),
    "docs/serving.md": _doc(
        "How the service runs",
        {4: "The service keeps 2 GB of memory free for activations.", 12: "The service batches up to 8 users at a time."},
        20,
    ),
    "docs/meetings.md": _doc(
        "Meeting notes",
        {3: "The team meets on Tuesdays.", 8: "The note about the dev-gpu board was last updated in March."},
        20,
    ),
}

RESEARCH_FACTS = [
    {"id": "mem", "text": "The dev-gpu board has 24 GB of memory.", "important": True},
    {"id": "bw", "text": "The dev-gpu memory bandwidth is 900 GB/s.", "important": True},
    {"id": "params", "text": "The tiny-7b model has 7.2 billion parameters, stored in 2 bytes each.", "important": True},
    {"id": "kv", "text": "The KV cache of tiny-7b needs 0.5 MB per token.", "important": True},
    {"id": "free", "text": "The service keeps 2 GB of memory free for activations.", "important": True},
    {"id": "batch", "text": "The service batches up to 8 users at a time.", "important": False},
    {"id": "tuesday", "text": "The team meets on Tuesdays.", "important": False},
    {"id": "march", "text": "The note about the dev-gpu board was last updated in March.", "important": False},
]

RESEARCH_TASK = (
    "The service runs the tiny-7b model on one dev-gpu board. Using the notes in docs/, work out "
    "(1) how many milliseconds one decode step takes if it must read every weight once from memory, and "
    "(2) how much memory, in GB, the KV cache needs for 8 users with 4096 tokens each, and whether it fits."
)

READ_STEPS: list[dict[str, Any]] = [
    {"thought": "Find the notes.", "calls": [{"name": "list_files", "args": {"path": "docs"}}]},
    {"thought": "Read about the board.", "calls": [{"name": "read_file", "args": {"path": "docs/gpu.md"}}]},
    {"thought": "Read about the model.", "calls": [{"name": "read_file", "args": {"path": "docs/model.md"}}]},
    {"thought": "Read about the service.", "calls": [{"name": "read_file", "args": {"path": "docs/serving.md"}}]},
    {"thought": "Check the meeting notes for anything newer.", "calls": [{"name": "read_file", "args": {"path": "docs/meetings.md"}}]},
]

CALC_STEPS: list[dict[str, Any]] = [
    {"thought": "Weights: 7.2e9 x 2 bytes = 14.4 GB, read at 900 GB/s.", "calls": [{"name": "calculator", "args": {"expression": "7.2 * 2 / 900 * 1000"}}]},
    {"thought": "KV cache: 8 users x 4096 tokens x 0.5 MB, in GB.", "calls": [{"name": "calculator", "args": {"expression": "8 * 4096 * 0.5 / 1000"}}]},
    {"thought": "Memory left for it: 24 - 14.4 - 2 GB.", "calls": [{"name": "calculator", "args": {"expression": "24 - 14.4 - 2"}}]},
    {
        "final": "(1) About 16 ms per decode step: 14.4 GB of weights at 900 GB/s. "
        "(2) The KV cache needs about 16.4 GB for 8 users at 4096 tokens, but only 7.6 GB is left after the weights "
        "and the 2 GB reserve, so it does not fit; about 3 users would."
    },
]

CORPUS = [
    {"id": "gpu-1", "title": "dev-gpu board: memory", "text": "The dev-gpu board has 24 GB of memory. The dev-gpu memory bandwidth is 900 GB/s."},
    {"id": "gpu-2", "title": "dev-gpu board: compute", "text": "The dev-gpu board does 120 trillion 16-bit operations per second."},
    {"id": "model-1", "title": "tiny-7b model card", "text": "The tiny-7b model has 7.2 billion parameters, stored in 2 bytes each."},
    {"id": "model-2", "title": "tiny-7b context", "text": "tiny-7b supports a context window of 32768 tokens."},
    {"id": "svc-1", "title": "service limits", "text": "The service batches up to 8 users at a time and keeps 2 GB free."},
    {"id": "misc-1", "title": "team calendar", "text": "The team meets on Tuesdays; planning is on the first Monday of the month."},
]

LOOKUP_TASK = (
    "How many milliseconds does it take to read every weight of the tiny-7b model once from the memory of a "
    "dev-gpu board? Use the search tool to find the numbers and the calculator for the arithmetic."
)

SCENARIOS: dict[str, dict[str, Any]] = {
    "fix_test": {
        "id": "fix_test",
        "title": "Fix a failing test",
        "system": CODING_SYSTEM,
        "task": "The tests in this repository fail. Find out why and fix the code.",
        "tools": ["list_files", "read_file", "edit_file", "run_shell"],
        "files": REPO,
        "facts": [],
        "script": FIX_SCRIPT,
        "seed": 7,
    },
    "research": {
        "id": "research",
        "title": "Read the design notes and do the sums",
        "system": RESEARCH_SYSTEM,
        "task": RESEARCH_TASK,
        "tools": ["list_files", "read_file", "calculator"],
        "files": DOCS,
        "facts": RESEARCH_FACTS,
        "script": READ_STEPS + CALC_STEPS,
        "seed": 11,
    },
    "lookup": {
        "id": "lookup",
        "title": "Search and calculate",
        "system": RESEARCH_SYSTEM,
        "task": LOOKUP_TASK,
        "tools": ["search", "calculator"],
        "corpus": CORPUS,
        "facts": [],
        "script": [
            {"thought": "Find the board's memory bandwidth.", "calls": [{"name": "search", "args": {"query": "dev-gpu memory bandwidth"}}]},
            {"thought": "Find the model's size.", "calls": [{"name": "search", "args": {"query": "tiny-7b parameters bytes"}}]},
            {"thought": "14.4 GB at 900 GB/s, in milliseconds.", "calls": [{"name": "calculator", "args": {"expression": "7.2 * 2 / 900 * 1000"}}]},
            {"final": "About 16 ms: 7.2 billion parameters x 2 bytes = 14.4 GB, read at 900 GB/s."},
        ],
        "seed": 3,
    },
}

SCENARIOS["fix_test_react"] = dict(SCENARIOS["fix_test"], id="fix_test_react", title="Fix a failing test (ReAct)",
                                   policy={"style": "react"})
SCENARIOS["fix_test_malformed"] = dict(
    SCENARIOS["fix_test"],
    id="fix_test_malformed",
    title="Fix a failing test, with one malformed call",
    script=FIX_SCRIPT[:1]
    + [
        {
            "raw": {
                "native": 'Run the tests to see what fails.\n<tool_call>\n{"name": "run_shell", "arguments": {"command": "pytest"}\n</tool_call>',
                "react": 'Thought: Run the tests to see what fails.\nAction: run_shell\nAction Input: {"command": "pytest"',
            }
        }
    ]
    + FIX_SCRIPT[1:],
)
SCENARIOS["fix_test_cleanup"] = dict(
    SCENARIOS["fix_test"],
    id="fix_test_cleanup",
    title="Fix a failing test, then tidy up",
    script=FIX_SCRIPT[:-1] + CLEANUP_STEPS + [
        {"final": "Fixed add() in calc.py, which subtracted instead of adding; all 3 tests now pass."}
    ],
)
SCENARIOS["fix_test_flaky"] = dict(
    SCENARIOS["fix_test"], id="fix_test_flaky", title="Fix a failing test with a flaky shell",
    fail_rates={"run_shell": 0.3},
)
SCENARIOS["research_subagent"] = dict(
    SCENARIOS["research"],
    id="research_subagent",
    title="Read the design notes with a sub-agent",
    tools=["list_files", "read_file", "calculator", "task"],
    script=[
        {
            "thought": "Hand the reading to a sub-agent so the notes stay out of my context.",
            "calls": [
                {
                    "name": "task",
                    "args": {
                        "prompt": "Read every note in docs/ and report, one per line, every number about the dev-gpu board's "
                        "memory and bandwidth, the tiny-7b model's size and KV cache, and the service's memory reserve."
                    },
                }
            ],
        }
    ]
    + CALC_STEPS,
    subagents=[
        READ_STEPS
        + [
            {
                "final": "The dev-gpu board has 24 GB of memory.\nThe dev-gpu memory bandwidth is 900 GB/s.\n"
                "The tiny-7b model has 7.2 billion parameters, stored in 2 bytes each.\n"
                "The KV cache of tiny-7b needs 0.5 MB per token.\nThe service keeps 2 GB of memory free for activations."
            }
        ]
    ],
)


def scenario(name: str) -> dict[str, Any]:
    """A deep copy of a named scenario (runs never change the originals)."""
    return copy.deepcopy(SCENARIOS[name])
