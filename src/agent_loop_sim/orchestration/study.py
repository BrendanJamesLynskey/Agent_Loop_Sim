"""The studies behind chapters 5–9 of the Agent Orchestration site (since 1.7): patterns compared on
one task, fan-out under a rate limit against Amdahl's law and the rate-limit ceiling, reliability
curves, many workflows at scale, and a pattern chooser driven by those results. Every number is
illustrative (see patterns, des and reliability for the assumptions)."""
from __future__ import annotations

from typing import Any

from ..accounting import LATENCY, PRICES
from . import des as D
from . import patterns as P
from . import reliability as R

# ---------------------------------------------------------------------------------------------
# Chapter 5: patterns


def task_of(k: int, result_tokens: int) -> dict[str, Any]:
    t = dict(P.TASK)
    t["subtasks"] = list(P.TASK["subtasks"]) if k == len(P.TASK["subtasks"]) else ["topic " + str(i + 1) for i in range(k)]
    t["tool_result_tokens"] = result_tokens
    return t


def model_of(slope: float) -> dict[str, Any]:
    m = dict(P.MODEL)
    m["slope_per_k"] = slope
    return m


def patterns_grid(ks: list[int], results: list[int], slopes: list[float]) -> list[dict[str, Any]]:
    out = []
    for k in ks:
        for rt in results:
            for sl in slopes:
                ms = P.compare(model=model_of(sl), task=task_of(k, rt))
                out.append({"k": k, "result_tokens": rt, "slope": sl,
                            "rows": [{x: m[x] for x in ["pattern", "model_calls", "input", "cached", "output", "cost",
                                                         "latency_ms", "serial_ms", "success"]} for m in ms]})
    return out


# ---------------------------------------------------------------------------------------------
# Chapter 6: fan-out under a rate limit

FAN = {"chunks": 32, "chunk_tokens": 2000, "widths": [1, 2, 4, 8, 16, 32], "rpm": 50, "itpm": 40000, "otpm": 8000}


def fan_params(width: int, rpm: int, itpm: int, otpm: int) -> dict[str, Any]:
    return {"workflows": 1, "arrival": "batch", "concurrency": width, "rpm": rpm, "itpm": itpm, "otpm": otpm,
            "fail": 0.0, "jitter": [1.0, 1.0], "semantic": False, "trace": True, "timeout_ms": 1e12, "seed": 1}


def ceiling_ms(plan: dict[str, Any], rpm: int, itpm: int, otpm: int) -> dict[str, float]:
    """A lower bound on the run time from each bucket: what must be admitted beyond the bucket's
    initial capacity, at its refill rate (the last call cannot start before then)."""
    price = PRICES[D.DEFAULTS["price"]]
    req = 0
    tin = 0
    tout = 0
    for c in plan["calls"]:
        if c["kind"] == "tool":
            continue
        req += 1
        tin += c["input"] - P.cached_tokens(c, price)
        tout += c["output"]
    def over(total: float, cap: float) -> float:
        x = total - cap
        return x / (cap / 60000) if x > 0 else 0.0
    return {"requests": over(req, rpm), "input": over(tin, itpm), "output": over(tout, otpm), "req": req, "tin": tin, "tout": tout}


def fan_study(chunks: int, chunk_tokens: int, widths: list[int], rpm: int, itpm: int, otpm: int) -> dict[str, Any]:
    plan = P.plan_map(chunks, chunk_tokens)
    price = PRICES[D.DEFAULTS["price"]]
    profile = LATENCY[D.DEFAULTS["profile"]]
    runs = []
    for wd in widths:
        r = D.simulate(plan, fan_params(wd, rpm, itpm, otpm))
        runs.append({"width": wd, "ms": r["makespan"], "trace": r["trace"], "changes": r["changes"]})
    t1 = runs[0]["ms"] if widths[0] == 1 else D.simulate(plan, fan_params(1, rpm, itpm, otpm))["makespan"]
    serial = P.duration_of(plan["calls"][0], price, profile) + P.duration_of(plan["calls"][-1], price, profile)
    frac = 1 - serial / t1
    ceil = ceiling_ms(plan, rpm, itpm, otpm)
    bound = ceil["requests"]
    if ceil["input"] > bound:
        bound = ceil["input"]
    if ceil["output"] > bound:
        bound = ceil["output"]
    rows = []
    for r in runs:
        amdahl = 1 / ((1 - frac) + frac / r["width"])
        rows.append({"width": r["width"], "ms": r["ms"], "speedup": t1 / r["ms"], "amdahl": amdahl,
                     "ceiling": t1 / bound if bound > 0 else 0.0})
    return {"chunks": chunks, "chunk_tokens": chunk_tokens, "rpm": rpm, "itpm": itpm, "otpm": otpm, "t1": t1,
            "serial_ms": serial, "parallel_fraction": frac, "bound_ms": bound, "ceiling": ceil, "rows": rows,
            "runs": runs, "plan": plan}


# ---------------------------------------------------------------------------------------------
# Chapter 7: reliability

REL = {"ns": [1, 2, 3, 5, 8, 10, 15, 20, 30], "e": 0.05, "w": 0.03, "runs": 2000, "seed": 7, "keep_n": 10, "keep": 48,
       "verifier": {"recall": 0.8, "false_reject": 0.05}}
REL_CONFIGS = [
    {"name": "none", "label": "one attempt", "attempts": 1, "verifier": False},
    {"name": "retry", "label": "retries (3 attempts)", "attempts": 3, "verifier": False},
    {"name": "verify", "label": "retries + verifier", "attempts": 3, "verifier": True},
]


def rel_study(e: float, w: float, cfg: dict[str, Any]) -> dict[str, Any]:
    v = REL["verifier"] if cfg["verifier"] else None
    return {"name": cfg["name"], "label": cfg["label"], "attempts": cfg["attempts"], "verifier": v,
            "curve": R.curve(REL["ns"], e, w, cfg["attempts"], v, REL["runs"], REL["seed"]),
            "grid": R.simulate(REL["keep_n"], e, w, cfg["attempts"], v, REL["keep"], REL["seed"], REL["keep"])}


# ---------------------------------------------------------------------------------------------
# Chapter 8: at scale

SCALE = {"rates": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0], "workflows": 200, "seed": 11, "samples": 120, "detail_rate": 3.0,
         "patterns": ["single", "supervisor", "map_reduce", "debate"]}


def capacity_per_min(plan: dict[str, Any], p: dict[str, Any]) -> dict[str, float]:
    """Workflows per minute each limit allows (requests, input tokens, output tokens), and the
    concurrency limit by Little's law with uncontended call durations."""
    price = PRICES[p["price"]]
    profile = LATENCY[p["profile"]]
    req = 0
    tin = 0
    tout = 0
    busy = 0.0
    for c in plan["calls"]:
        if c["kind"] == "tool":
            continue
        req += 1
        tin += c["input"] - P.cached_tokens(c, price)
        tout += c["output"]
        busy = busy + P.duration_of(c, price, profile)
    out = {"requests": p["rpm"] / req, "input": p["itpm"] / tin, "output": p["otpm"] / tout,
           "concurrency": p["concurrency"] * 60000 / busy}
    m = out["requests"]
    for k in ["input", "output", "concurrency"]:
        if out[k] < m:
            m = out[k]
    out["min"] = m
    return out


def scale_params(rate: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {"workflows": SCALE["workflows"], "per_min": rate, "seed": SCALE["seed"]}
    for k, v in (extra or {}).items():
        p[k] = v
    return p


def summary(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r[k] for k in ["pattern", "workflows", "ok", "wrong", "exhausted", "success", "p50", "p95", "p99", "mean",
                              "cost", "cost_per_success", "makespan", "throughput_per_min", "attempts", "utilisation"]}


def scale_sweep(patterns: list[str], rates: list[float], extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    out = []
    for n in patterns:
        pl = P.plan(n)
        cap = capacity_per_min(pl, D.params(scale_params(1.0, extra)))
        rows = [summary(D.simulate(pl, scale_params(r, extra))) for r in rates]
        out.append({"pattern": n, "capacity": cap, "rows": rows})
    return out


def scale_detail(pattern: str, rate: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    p = scale_params(rate, extra)
    p["samples"] = SCALE["samples"]
    r = D.simulate(P.plan(pattern), p)
    s = summary(r)
    s["series"] = r["series"]
    s["runs"] = r["runs"]
    return s


# ---------------------------------------------------------------------------------------------
# Chapter 9: choosing a pattern

CHOOSE = {"k": [4, 16], "rates": [1.0, 0.25], "workflows": 1000}
CONSTRAINTS = [
    {"key": "success", "op": ">=", "label": "success at least"},
    {"key": "p95", "op": "<=", "label": "p95 latency at most"},
    {"key": "cost_per_success", "op": "<=", "label": "cost per success at most"},
    {"key": "capacity", "op": ">=", "label": "workflows per minute at least"},
]


def choose_table(k: int, rate: float) -> list[dict[str, Any]]:
    """Every pattern at one load, on a task with k sub-questions (1,500-token tool results, the
    default context penalty): the DES's success, p95 latency, cost per success and throughput, and
    the capacity the rate limits allow (``capacity_per_min``)."""
    t = task_of(k, 1500)
    out = []
    for n in P.PATTERNS:
        pl = P.plan(n, t)
        r = D.simulate(pl, scale_params(rate, {"workflows": CHOOSE["workflows"]}))
        s = summary(r)
        s["ci"] = R.wilson(r["ok"], r["workflows"])
        s["closed_success"] = D.success_closed(pl, D.DEFAULTS["fail"], D.DEFAULTS["max_attempts"])
        s["capacity"] = capacity_per_min(pl, D.params({}))["min"]
        out.append(s)
    return out


def choose(table: list[dict[str, Any]], limits: dict[str, float]) -> dict[str, Any]:
    """Apply the constraints in order (each one strikes out the patterns that miss it), then pick
    the highest success, breaking ties by lower cost per success. Returns each step's survivors."""
    alive = [r["pattern"] for r in table]
    steps = []
    for c in CONSTRAINTS:
        if c["key"] not in limits:
            continue
        lim = limits[c["key"]]
        keep = []
        out = []
        for r in table:
            if r["pattern"] not in alive:
                continue
            v = r[c["key"]]
            good = v >= lim if c["op"] == ">=" else v <= lim
            (keep if good else out).append(r["pattern"])
        alive = keep
        steps.append({"key": c["key"], "op": c["op"], "limit": lim, "kept": keep, "struck": out})
    best = None
    for r in table:
        if r["pattern"] not in alive:
            continue
        if best is None or r["success"] > best["success"] or (r["success"] == best["success"] and r["cost_per_success"] < best["cost_per_success"]):
            best = r
    return {"steps": steps, "alive": alive, "pick": best["pattern"] if best else None}
