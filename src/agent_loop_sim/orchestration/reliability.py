"""Reliability maths for multi-step workflows, with a seeded simulation to test it against (since 1.7).

A workflow is a chain of ``n`` steps. Each attempt at a step either raises a **detectable** error
(probability ``e``: an exception, a malformed tool call), returns a **silently wrong** result
(``w``), or is correct (``c = 1 − e − w``). Retries re-run a step after a detectable error, up to
``A`` attempts. A **verifier** (optional) checks each result: it rejects a wrong one with
probability ``d`` (its recall) and a correct one with probability ``f`` (false rejections); a
rejection is retried like an error, within the same ``A`` attempts. Per attempt:

    retry  r = e + c·f + w·d        accept-correct a = c·(1 − f)        accept-wrong b = w·(1 − d)

(without a verifier, d = f = 0). Per step, with S_A = 1 + r + … + r^(A−1) = (1 − r^A)/(1 − r):

    P(step correct) = a·S_A     P(step wrong, unnoticed) = b·S_A     P(step gives up) = r^A
    E[attempts per step] = S_A

End to end (a wrong step is only found at the end; a step that gives up stops the run):

    P(success) = (a·S_A)^n          E[attempts] = S_A · Σ_{i<n} (1 − r^A)^i

Powers and the geometric sums are computed by repeated multiplication (identical in Python and
TS). Confidence intervals are Wilson score intervals (z = 1.96). Every probability is a parameter;
the defaults on the site are illustrative.
"""
from __future__ import annotations

import math
from typing import Any

from ..rng import Rng

Z95 = 1.96


def _pow(x: float, n: int) -> float:
    v = 1.0
    for _ in range(n):
        v = v * x
    return v


def _geo(x: float, n: int) -> float:
    """1 + x + … + x^(n−1)."""
    s = 0.0
    v = 1.0
    for _ in range(n):
        s = s + v
        v = v * x
    return s


def per_attempt(e: float, w: float, verifier: dict[str, Any] | None) -> dict[str, float]:
    c = 1 - e - w
    d = verifier["recall"] if verifier else 0.0
    f = verifier["false_reject"] if verifier else 0.0
    return {"c": c, "r": e + c * f + w * d, "a": c * (1 - f), "b": w * (1 - d)}


def closed(n: int, e: float, w: float, attempts: int, verifier: dict[str, Any] | None = None) -> dict[str, float]:
    """The closed forms above for one configuration."""
    q = per_attempt(e, w, verifier)
    sa = _geo(q["r"], attempts)
    give_up = _pow(q["r"], attempts)
    step_ok = q["a"] * sa
    goes_on = 1 - give_up
    return {"step_ok": step_ok, "step_wrong": q["b"] * sa, "step_give_up": give_up, "attempts_per_step": sa,
            "success": _pow(step_ok, n), "attempts": sa * _geo(goes_on, n),
            "reached_end": _pow(goes_on, n)}


def wilson(k: int, n: int, z: float = Z95) -> list[float]:
    """The Wilson score interval for k successes in n trials."""
    if n == 0:
        return [0.0, 1.0]
    ph = k / n
    z2 = z * z
    den = 1 + z2 / n
    mid = ph + z2 / (2 * n)
    half = z * math.sqrt(ph * (1 - ph) / n + z2 / (4 * n * n))
    lo = (mid - half) / den
    hi = (mid + half) / den
    return [lo if lo > 0 else 0.0, hi if hi < 1 else 1.0]


def simulate(n: int, e: float, w: float, attempts: int, verifier: dict[str, Any] | None, runs: int, seed: int,
             keep: int = 0) -> dict[str, Any]:
    """Monte Carlo of ``runs`` chains. Each attempt draws its outcome (one draw), then, with a
    verifier, its verdict (one draw). Returns the success count, Wilson interval, mean attempts,
    how runs ended, and the first ``keep`` runs step by step (for the animation)."""
    r = Rng(seed)
    ok = 0
    wrong = 0
    gave_up = 0
    total_att = 0
    kept: list[dict[str, Any]] = []
    for i in range(runs):
        steps: list[list[str]] = []
        silent = False
        stopped = False
        for _ in range(n):
            log: list[str] = []
            done = False
            while len(log) < attempts:
                u = r.random()
                if u < e:
                    log.append("error")
                    continue
                right = u >= e + w
                if verifier:
                    v = r.random()
                    if right and v < verifier["false_reject"]:
                        log.append("rejected_ok")
                        continue
                    if not right and v < verifier["recall"]:
                        log.append("caught")
                        continue
                log.append("ok" if right else "wrong")
                if not right:
                    silent = True
                done = True
                break
            total_att += len(log)
            steps.append(log)
            if not done:
                stopped = True
                break
        if stopped:
            gave_up += 1
            status = "gave_up"
        elif silent:
            wrong += 1
            status = "wrong"
        else:
            ok += 1
            status = "ok"
        if i < keep:
            kept.append({"status": status, "steps": steps})
    return {"n": n, "e": e, "w": w, "attempts": attempts, "verifier": verifier, "runs": runs, "seed": seed,
            "ok": ok, "wrong": wrong, "gave_up": gave_up, "rate": ok / runs, "ci": wilson(ok, runs),
            "mean_attempts": total_att / runs, "kept": kept, "closed": closed(n, e, w, attempts, verifier)}


def curve(ns: list[int], e: float, w: float, attempts: int, verifier: dict[str, Any] | None, runs: int, seed: int) -> list[dict[str, Any]]:
    """P(success) against chain length: the closed form and a simulation (seed + n) at each n."""
    out = []
    for n in ns:
        s = simulate(n, e, w, attempts, verifier, runs, seed + n)
        out.append({"n": n, "closed": s["closed"]["success"], "rate": s["rate"], "ci": s["ci"],
                    "attempts": s["mean_attempts"], "closed_attempts": s["closed"]["attempts"]})
    return out
