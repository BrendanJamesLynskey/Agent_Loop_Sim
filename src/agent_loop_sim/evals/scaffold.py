"""The harness is part of the result (since 1.9): one simulated model, many scaffolds, measured; and what
each configuration costs.

**The task suite**: twelve one-function repositories, each with a bug in one ``return`` line and a
``test_calc.py`` of three assertions that the engine's fake ``pytest`` really evaluates. Every
configuration runs every task ``runs`` times through the engine's harness, and a run counts as
resolved when the tests pass on the files it leaves behind.

**The model** (``FixerModel``) is a seeded stochastic policy, not a language model: it reads
``calc.py``, then edits the buggy line. An edit is right with probability μ_t = logistic(a(θ − b_t)),
the item-response model of ``generator`` (task difficulty b_t, the agent's skill θ). Attempts on one
task are correlated: each copies the run's shared outcome with probability sqrt(ρ), as in
``generator.runs``, so an agent that misreads a task tends to keep misreading it. A wrong edit
writes one of the task's plausible wrong expressions. When the scaffold gives it a shell, it runs
the tests after each edit, reads the exit code from the tool result in its prompt, and edits again
until they pass or the turn budget runs out; without a shell it stops after one edit. Any output can
come out malformed (the closing brace of the call's JSON lost) with a per-style probability: text
(ReAct) calls are written by the model as text and break more often than native tool calls, which an
API constrains. Its draws come from two streams per (task, run), one for the edits and one for the
malformed outputs, so every scaffold sees the same edits in the same order (common random numbers):
differences between scaffolds are the scaffolds' doing.

**Scaffolds**: tools (``edit``: read and edit only; ``test``: also the shell), the call style
(``native`` or ``react``) and the turn budget. **Agents**: a smaller and a larger model, priced at
two list prices from ``accounting.PRICES``. Every skill, rate and price pairing is illustrative.

**Cost**: each run's cost and tokens come from the harness's own accounting (real token counts from
the vendored tokenizer, the price table, the prompt-cache simulation). ``cost_study`` puts each
configuration on the cost–success plane, with its Pareto front and dollars per solved task.
"""
from __future__ import annotations

import math
from typing import Any

from ..context.text import fixed
from ..harness import Run
from ..models import render_step
from ..rng import MASK, Rng
from ..tools import run_pytest
from . import stats as S
from .mathx import logistic

TASKS: list[dict[str, Any]] = [
    {"name": "add", "params": "a, b", "right": "a + b", "bug": "a - b", "wrong": ["a * b", "b - a", "a + b + 1"],
     "tests": [["2, 3", 5], ["10, 0", 10], ["-1, 1", 0]], "b": -1.5},
    {"name": "sub", "params": "a, b", "right": "a - b", "bug": "a + b", "wrong": ["b - a", "a - b - 1", "a * b"],
     "tests": [["7, 4", 3], ["0, 5", -5], ["3, 3", 0]], "b": -1.1},
    {"name": "mul", "params": "a, b", "right": "a * b", "bug": "a + b", "wrong": ["a * a", "a * b + 1", "a / b"],
     "tests": [["6, 7", 42], ["0, 3", 0], ["-2, 5", -10]], "b": -0.9},
    {"name": "double", "params": "a", "right": "2 * a", "bug": "a * a", "wrong": ["a + 2", "a * 3", "a / 2"],
     "tests": [["3", 6], ["0", 0], ["-4", -8]], "b": -0.6},
    {"name": "square", "params": "a", "right": "a * a", "bug": "a * 2", "wrong": ["a + a", "a * a * a", "a * a + 1"],
     "tests": [["3", 9], ["-2", 4], ["0", 0]], "b": -0.3},
    {"name": "avg", "params": "a, b", "right": "(a + b) / 2", "bug": "a + b / 2",
     "wrong": ["(a + b) * 2", "a / 2 + b", "(a - b) / 2"], "tests": [["4, 6", 5], ["-2, 2", 0], ["7, 9", 8]], "b": 0.0},
    {"name": "area", "params": "w, h", "right": "w * h / 2", "bug": "w * h", "wrong": ["w + h / 2", "(w + h) / 2", "w * h * 2"],
     "tests": [["4, 6", 12], ["3, 2", 3], ["10, 1", 5]], "b": 0.3},
    {"name": "perimeter", "params": "w, h", "right": "2 * (w + h)", "bug": "2 * w + h",
     "wrong": ["w * h * 2", "w + h", "4 * w"], "tests": [["3, 4", 14], ["1, 1", 4], ["5, 0", 10]], "b": 0.6},
    {"name": "mean3", "params": "a, b, c", "right": "(a + b + c) / 3", "bug": "a + b + c / 3",
     "wrong": ["(a + b + c) / 2", "(a + b) / 2 + c", "a + (b + c) / 3"], "tests": [["3, 3, 3", 3], ["1, 2, 3", 2], ["0, 0, 9", 3]],
     "b": 0.9},
    {"name": "celsius", "params": "f", "right": "(f - 32) * 5 / 9", "bug": "f - 32 * 5 / 9",
     "wrong": ["(f - 32) * 9 / 5", "(f + 32) * 5 / 9", "f * 5 / 9 - 32"], "tests": [["212", 100], ["32", 0], ["-40", -40]],
     "b": 1.2},
    {"name": "lerp", "params": "a, b, t", "right": "a + (b - a) * t", "bug": "a + b * t",
     "wrong": ["(a + b) * t", "a * t + b", "b + (a - b) * t"], "tests": [["0, 10, 1", 10], ["2, 4, 0", 2], ["1, 3, 2", 5]],
     "b": 1.5},
    {"name": "poly", "params": "x", "right": "x * x - 2 * x + 1", "bug": "x * x + 2 * x + 1",
     "wrong": ["x * x - 2 * x - 1", "x * x - x + 1", "(x - 1) * x"], "tests": [["1", 0], ["3", 4], ["0", 1]], "b": 1.9},
]

AGENTS: list[dict[str, Any]] = [
    {"id": "small", "label": "Smaller model", "theta": -0.2, "price": "claude-haiku-4.5"},
    {"id": "large", "label": "Larger model", "theta": 0.6, "price": "claude-sonnet-4.6"},
]

SCAFFOLD = {
    "runs": 8, "seed": 2581, "discrimination": 1.4, "rho": 0.5,
    "malformed": {"native": 0.02, "react": 0.12},
    "tools": {"edit": ["read_file", "edit_file"], "test": ["read_file", "edit_file", "run_shell"]},
    "styles": ["native", "react"], "budgets": [4, 8, 16],
    "system": ("You are a coding agent working in a small Python repository. Use the tools to fix the bug the user "
               "describes. When you are done, reply with a one-sentence summary and no tool call."),
}

_MAL_OFFSET = 0x5BD1E995


def files_for(task: dict[str, Any], expr: str | None = None) -> dict[str, str]:
    body = task["bug"] if expr is None else expr
    calc = f'"""One function."""\n\n\ndef {task["name"]}({task["params"]}):\n    return {body}\n'
    lines = [f"from calc import {task['name']}", "", "", f"def test_{task['name']}():"]
    for args, want in task["tests"]:
        lines.append(f"    assert {task['name']}({args}) == {want}")
    return {"calc.py": calc, "test_calc.py": "\n".join(lines) + "\n"}


def passes(task: dict[str, Any], expr: str) -> bool:
    return run_pytest(files_for(task, expr))[0] == 0


def scenario_for(task: dict[str, Any], tools: list[str], seed: int) -> dict[str, Any]:
    return {"id": "suite_" + task["name"], "title": "Fix " + task["name"] + "()", "system": SCAFFOLD["system"],
            "task": f"{task['name']}() in calc.py returns the wrong result. Fix it.", "tools": tools,
            "files": files_for(task), "facts": [], "script": [], "seed": seed}


def mu_of(task: dict[str, Any], theta: float, cfg: dict[str, Any] = SCAFFOLD) -> float:
    return logistic(cfg["discrimination"] * (theta - task["b"]))


def run_seed(cfg: dict[str, Any], t: int, j: int) -> int:
    return (cfg["seed"] + 1000 * t + j) & MASK


class FixerModel:
    """The stochastic policy above. Steps: read → edit → (test → edit → test …) → final."""

    name = "fixer"

    def __init__(self, task: dict[str, Any], mu: float, rho: float, malformed: float, can_test: bool, seed: int) -> None:
        self.task = task
        self.mu = mu
        self.s = math.sqrt(rho)
        self.malformed = malformed
        self.can_test = can_test
        self.r_edit = Rng(seed)
        self.r_mal = Rng((seed ^ _MAL_OFFSET) & MASK)
        self.z = 1 if self.r_edit.random() < mu else 0
        self.current = task["bug"]
        self.state = "read"
        self.pending: dict[str, Any] | None = None
        self.edits = 0

    def _edit_step(self) -> dict[str, Any]:
        c = self.r_edit.random()
        own = self.r_edit.random()
        pick = self.r_edit.random()
        ok = self.z if c < self.s else (1 if own < self.mu else 0)
        if ok == 1:
            new = self.task["right"]
        else:
            wrong = self.task["wrong"]
            k = math.floor(pick * len(wrong))
            if wrong[k] == self.current:
                k = (k + 1) % len(wrong)
            new = wrong[k]
        old = self.current
        self.current = new
        self.edits += 1
        thought = "The bug is in the return line; change it." if self.edits == 1 else "The tests still fail; try another fix."
        return {"thought": thought, "calls": [{"name": "edit_file", "args": {
            "path": "calc.py", "old_text": "    return " + old, "new_text": "    return " + new}}]}

    def _next(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.state == "read":
            self.state = "edit"
            return {"thought": "Read the function.", "calls": [{"name": "read_file", "args": {"path": "calc.py"}}]}
        if self.state == "edit":
            step = self._edit_step()
            self.state = "test" if self.can_test else "final"
            return step
        if self.state == "test":
            self.state = "check"
            return {"thought": "Run the tests.", "calls": [{"name": "run_shell", "args": {"command": "pytest"}}]}
        if self.state == "check":
            prompt = req["prompt"]
            i = prompt.rfind("exit code: ")
            if i >= 0 and prompt[i + 11: i + 12] == "0":
                self.state = "done"
                return {"final": f"Fixed {self.task['name']}(); the tests pass."}
            step = self._edit_step()
            self.state = "test"
            return step
        self.state = "done"
        return {"final": f"Fixed {self.task['name']}()."}

    def complete(self, req: dict[str, Any]) -> str:
        if req["last"]["kind"] != "malformed" or self.pending is None:
            self.pending = self._next(req)
        style = req["style"]
        text = render_step(self.pending, style)
        if "calls" in self.pending and self.r_mal.random() < self.malformed:
            if style == "native":
                return text.replace("}}\n</tool_call>", "}\n</tool_call>")
            return text[:-1]
        return text


def configs(cfg: dict[str, Any] = SCAFFOLD) -> list[dict[str, Any]]:
    out = []
    for a in AGENTS:
        for tools in ["edit", "test"]:
            for style in cfg["styles"]:
                # without a shell the agent stops after one edit, so only the smallest budget is run
                for budget in (cfg["budgets"][:1] if tools == "edit" else cfg["budgets"]):
                    out.append({"id": f"{a['id']}|{tools}|{style}|{budget}", "agent": a["id"], "tools": tools,
                                "style": style, "budget": budget})
    return out


def run_one(conf: dict[str, Any], t: int, j: int, tokenizer: Any = None, cfg: dict[str, Any] = SCAFFOLD,
            events: bool = False) -> dict[str, Any]:
    agent = next(a for a in AGENTS if a["id"] == conf["agent"])
    task = TASKS[t]
    seed = run_seed(cfg, t, j)
    mu = mu_of(task, agent["theta"], cfg)
    model = FixerModel(task, mu, cfg["rho"], cfg["malformed"][conf["style"]], conf["tools"] == "test", seed)
    policy = {"style": conf["style"], "max_turns": conf["budget"], "cache": {"price": agent["price"]},
              "permissions": {"mode": "auto"}}
    run = Run(scenario_for(task, cfg["tools"][conf["tools"]], seed), policy, model, seed, tokenizer)
    ev = run.run()
    end = ev[-1]
    tot = end["totals"]
    resolved = run_pytest(run.world.files)[0] == 0
    malformed = 0
    for e in ev:
        if e["type"] == "error" and e["kind"] == "malformed":
            malformed += 1
    out = {"t": t, "j": j, "resolved": resolved, "status": end["status"], "turns": tot["model_calls"],
           "edits": model.edits, "malformed": malformed, "input": tot["input_tokens"], "cached": tot["cached_tokens"],
           "output": tot["output_tokens"], "cost": tot["cost"], "elapsed": end["elapsed"]}
    if events:
        out["events"] = ev
        out["files"] = run.world.files
    return out


def summarise(conf: dict[str, Any], rows: list[dict[str, Any]], cfg: dict[str, Any] = SCAFFOLD) -> dict[str, Any]:
    n = len(rows)
    k = 0
    cost = 0.0
    turns = 0
    mal = 0
    tokens = 0
    for r in rows:
        k += 1 if r["resolved"] else 0
        cost = cost + r["cost"]
        turns += r["turns"]
        mal += r["malformed"]
        tokens += r["input"] + r["output"]
    succ = [0] * len(TASKS)
    cnt = [0] * len(TASKS)
    for r in rows:
        succ[r["t"]] += 1 if r["resolved"] else 0
        cnt[r["t"]] += 1
    cl = S.cluster_se(succ, cnt)
    z = 1.959963984540054
    return {**conf, "n": n, "k": k, "rate": k / n, "wilson": S.wilson(k, n),
            "cluster": [cl["p"] - z * cl["se"], cl["p"] + z * cl["se"]], "se": cl["se"],
            "cost_per_run": cost / n, "cost_per_solved": cost / k if k > 0 else 0.0, "turns": turns / n,
            "malformed": mal / n, "tokens": tokens / n, "per_task": [succ[i] / cnt[i] for i in range(len(TASKS))]}


def scaffold_study(tokenizer: Any = None, cfg: dict[str, Any] = SCAFFOLD) -> dict[str, Any]:
    """Every configuration × task × run. ``rows`` holds every run's summary; ``configs`` each
    configuration's rate, intervals, cost and turns; ``examples`` a few full traces for the page."""
    out_rows: dict[str, list[dict[str, Any]]] = {}
    summ = []
    for conf in configs(cfg):
        rows = [run_one(conf, t, j, tokenizer, cfg) for t in range(len(TASKS)) for j in range(cfg["runs"])]
        out_rows[conf["id"]] = rows
        summ.append(summarise(conf, rows, cfg))
    by = {s["id"]: s for s in summ}
    spread = {}
    for a in AGENTS:
        rates = [s["rate"] for s in summ if s["agent"] == a["id"]]
        lo = min(rates)
        hi = max(rates)
        spread[a["id"]] = {"min": lo, "max": hi, "range": hi - lo}
    return {"cfg": cfg, "tasks": TASKS, "agents": AGENTS, "mu": {a["id"]: [mu_of(t, a["theta"], cfg) for t in TASKS] for a in AGENTS},
            "configs": summ, "by_id": by, "rows": out_rows, "spread": spread}


def paired_scaffolds(study: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    """Configuration a against b on the same (task, run) pairs, which share the model's edit draws."""
    ra = study["rows"][a]
    rb = study["rows"][b]
    xa = [1 if r["resolved"] else 0 for r in ra]
    xb = [1 if r["resolved"] else 0 for r in rb]
    tab = S.paired_table(xa, xb)
    return {"a": a, "b": b, "table": tab, "mcnemar": S.mcnemar(tab["n10"], tab["n01"]),
            "compare": S.compare([float(x) for x in xa], [float(x) for x in xb])}


def cost_study(study: dict[str, Any]) -> dict[str, Any]:
    """Each configuration on the (cost per task, success rate) plane: the Pareto front, dollars per
    solved task, and the cheapest configuration that reaches each success level."""
    summ = study["configs"]
    pts = [[s["cost_per_run"], s["rate"]] for s in summ]
    front = S.pareto_front(pts)
    points = [{"id": s["id"], "agent": s["agent"], "tools": s["tools"], "style": s["style"], "budget": s["budget"],
               "cost": s["cost_per_run"], "rate": s["rate"], "wilson": s["wilson"], "per_solved": s["cost_per_solved"],
               "front": i in front} for i, s in enumerate(summ)]
    best_rate = max(p["rate"] for p in points)
    best = [p for p in points if p["rate"] == best_rate]
    best.sort(key=lambda p: (p["cost"], p["id"]))
    cheap = sorted([p for p in points if p["rate"] > 0], key=lambda p: (p["per_solved"], p["id"]))
    return {"points": points, "front": [summ[i]["id"] for i in front], "best": best[0]["id"], "cheapest_per_solved": cheap[0]["id"]}


def describe(conf_id: str) -> str:
    agent, tools, style, budget = conf_id.split("|")
    a = next(x for x in AGENTS if x["id"] == agent)
    return (f"{a['label'].lower()}, {'read/edit only' if tools == 'edit' else 'with a shell for the tests'}, "
            f"{'native calls' if style == 'native' else 'ReAct text'}, {budget} turns")


def caption_run(conf_id: str, r: dict[str, Any]) -> str:
    return (f"{TASKS[r['t']]['name']}(), run {r['j'] + 1}: {'resolved' if r['resolved'] else 'not resolved'} after "
            f"{r['turns']} model calls and {r['edits']} edit{'s' if r['edits'] != 1 else ''}"
            + (f", {r['malformed']} malformed" if r["malformed"] else "") + f"; ${fixed(r['cost'] + 0.0, 4)}.")
