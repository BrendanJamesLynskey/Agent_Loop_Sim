"""Seeded sweeps: the same scenario run over many seeds, one row per value of a policy knob.

``retry_sweep`` answers chapter 8's question: with tools that fail at random, how often does
the run finish, and what does it cost, for each retry budget? Every run is an ordinary seeded
run, so a row is reproducible in Python and in the TS port alike. Sums are plain left-to-right
additions (never Python's compensated ``sum`` of floats), so the means match exactly.
"""
from __future__ import annotations

from typing import Any

from .harness import merge, run
from .scenarios import scenario
from .tokenizer import Tokenizer


def verified(events: list[dict[str, Any]]) -> bool:
    """Did the run finish *and* did its last test run pass? A run can end "done" with the
    model claiming success after its tests never ran (a loop nudge moved it on): the
    harness, not the model, should decide when a task is done."""
    subject: dict[str, str] = {}
    last_ok = False
    for e in events:
        if e["type"] == "tool_call":
            subject[e["call"]] = e["subject"]
        elif e["type"] == "tool_result" and e["name"] == "run_shell" and subject[e["call"]].startswith("pytest"):
            last_ok = e["ok"] and e["text"].startswith("exit code: 0")
    return events[len(events) - 1]["status"] == "done" and last_ok


def retry_sweep(name: str, budgets: list[int], seeds: list[int], policy: dict[str, Any] | None = None,
                tokenizer: Tokenizer | None = None) -> list[dict[str, Any]]:
    """For each retry budget: the runs' statuses (one per seed, in order), the share that
    finished, the share that finished with passing tests (``verified``), and the mean turns,
    model calls, retries, simulated time and cost."""
    rows = []
    for b in budgets:
        pol = merge(policy or {}, {"recovery": merge((policy or {}).get("recovery", {}), {"retries": b})})
        statuses: list[str] = []
        done = 0
        good = 0
        turns = 0
        calls = 0
        retries = 0
        elapsed = 0.0
        cost = 0.0
        for s in seeds:
            ev = run(scenario(name), pol, seed=s, tokenizer=tokenizer)
            end = ev[len(ev) - 1]
            statuses.append(end["status"])
            if end["status"] == "done":
                done += 1
            if verified(ev):
                good += 1
            for e in ev:
                if e["type"] == "model_call" and e["agent"] == "main" and e["purpose"] == "act":
                    turns += 1
            calls += end["totals"]["model_calls"]
            retries += end["totals"]["retries"]
            elapsed = elapsed + end["elapsed"]
            cost = cost + end["totals"]["cost"]
        n = len(seeds)
        rows.append({
            "retries": b, "runs": n, "done": done, "success": done / n, "verified": good,
            "verified_rate": good / n, "statuses": statuses,
            "mean_turns": turns / n, "mean_calls": calls / n, "mean_retries": retries / n,
            "mean_elapsed": elapsed / n, "mean_cost": cost / n,
        })
    return rows
