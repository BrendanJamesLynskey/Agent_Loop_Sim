"""The orchestration module's second part (engine 1.7.0), written by make_fixtures.py:
fixtures/orchestration2_fixtures.json (multi-agent patterns, the discrete-event simulation, fan-out,
reliability, scale and the chooser, for the TS port to reproduce exactly) and
fixtures/orchestration2_results.md (the recorded run every published number comes from)."""
from __future__ import annotations

from typing import Any

from agent_loop_sim import VERSION
from agent_loop_sim.context.text import fixed
from agent_loop_sim.orchestration import des as D
from agent_loop_sim.orchestration import patterns as P
from agent_loop_sim.orchestration import reliability as R
from agent_loop_sim.orchestration import study as S

GRID_K = [4, 8, 16]
GRID_RESULTS = [700, 1500]
GRID_SLOPES = [0.0, 0.01, 0.02]
FAN_ITPM = [40000, 80000, 200000]
CHOOSE_LIMITS = [
    {"success": 0.8},
    {"success": 0.8, "p95": 30000},
    {"success": 0.4, "p95": 180000},
    {"success": 0.4, "p95": 180000, "cost_per_success": 0.6},
    {"success": 0.4, "capacity": 1.0},
]


def consts() -> dict[str, Any]:
    return {"task": P.TASK, "model": P.MODEL, "patterns": P.PATTERNS, "titles": P.TITLES, "des": D.DEFAULTS,
            "fan": S.FAN, "rel": S.REL, "rel_configs": S.REL_CONFIGS, "scale": S.SCALE, "choose": S.CHOOSE,
            "constraints": S.CONSTRAINTS}


def orchestration2_fixtures() -> tuple[dict[str, Any], str]:
    plans = {n: P.plan(n) for n in P.PATTERNS}
    plans16 = {n: P.plan(n, S.task_of(16, 1500)) for n in P.PATTERNS}
    compare = P.compare()
    frames = {n: P.pattern_frames(plans[n]) for n in P.PATTERNS}
    grid = S.patterns_grid(GRID_K, GRID_RESULTS, GRID_SLOPES)
    small = {n: D.simulate(plans[n], {"workflows": 5, "per_min": 4.0, "trace": True, "samples": 20, "seed": 5})
             for n in P.PATTERNS}
    overload = D.simulate(plans["supervisor"], {"workflows": 12, "arrival": "batch", "fail": 0.2, "timeout_ms": 20000,
                                                "jitter": [0.5, 3.0], "trace": True, "samples": 30, "seed": 9})
    fans = [S.fan_study(S.FAN["chunks"], S.FAN["chunk_tokens"], S.FAN["widths"], S.FAN["rpm"], it, S.FAN["otpm"])
            for it in FAN_ITPM]
    for f in fans:
        f.pop("plan")
    rel = [S.rel_study(S.REL["e"], S.REL["w"], c) for c in S.REL_CONFIGS]
    closed = [R.closed(n, e, w, a, v) for n in [1, 5, 10] for e in [0.0, 0.05, 0.2] for w in [0.0, 0.03]
              for a in [1, 2, 4] for v in [None, S.REL["verifier"]]]
    wil = [R.wilson(k, n) for k, n in [(0, 10), (5, 10), (10, 10), (731, 1000), (0, 0)]]
    scale = S.scale_sweep(S.SCALE["patterns"], S.SCALE["rates"])
    details = {n: S.scale_detail(n, S.SCALE["detail_rate"]) for n in S.SCALE["patterns"]}
    tables = {str(k): S.choose_table(k, rate) for k, rate in zip(S.CHOOSE["k"], S.CHOOSE["rates"])}
    choices = [{"k": k, "limits": lim, "result": S.choose(tables[str(k)], lim)} for k in S.CHOOSE["k"] for lim in CHOOSE_LIMITS]
    fx = {
        "engine": VERSION,
        "consts": consts(),
        "call_p": [P.call_p(k, n, P.MODEL) for k in ["route", "act", "final"] for n in [0, 4000, 4001, 9000, 100000]],
        "maj3": [P.maj3(a, b, c) for a, b, c in [(0.9, 0.9, 0.9), (0.5, 0.5, 0.5), (1.0, 0.0, 0.7)]],
        "pct": [D.pct(xs, q) for xs in [[], [3.0], [5.0, 1.0, 4.0, 2.0, 3.0]] for q in [1, 50, 95, 99, 100]],
        "plans": plans,
        "plans16": plans16,
        "plan_map": P.plan_map(6, 1200),
        "compare": compare,
        "frames": frames,
        "grid": grid,
        "success_closed": {n: D.success_closed(plans[n], 0.03, 3) for n in P.PATTERNS},
        "expected_attempts": [D.expected_attempts(f, a) for f in [0.0, 0.03, 0.5, 1.0] for a in [1, 3]],
        "small": small,
        "overload": overload,
        "fans": fans,
        "rel": rel,
        "closed": closed,
        "wilson": wil,
        "scale": scale,
        "details": details,
        "tables": tables,
        "choices": choices,
    }
    return fx, results_md(compare, grid, fans, rel, scale, details, tables, choices)


def _f(x: float, d: int = 3) -> str:
    return fixed(x, d)


def _s(ms: float) -> str:
    return fixed(ms / 1000, 1)


def results_md(compare, grid, fans, rel, scale, details, tables, choices) -> str:  # type: ignore[no-untyped-def]
    t = P.TASK
    out = [f"# Orchestration module results, part 2 (engine {VERSION})", "",
           "Written by `scripts/make_fixtures.py` (CI checks it is up to date). Every token size, accuracy, latency, "
           "rate limit and failure rate here is illustrative (see the module docstrings); prices are the engine's "
           "dated list prices (`accounting.PRICES`), latency is the engine's hosted profile.", "",
           f"## Multi-agent patterns on one task ({len(t['subtasks'])} sub-questions, {t['tool_result_tokens']}-token tool "
           "results, uncontended, Claude Sonnet 4.6 prices)", "",
           "| pattern | model calls | input tokens | cached | output | cost ($) | latency (s) | one call at a time (s) | P(success) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in compare:
        out.append(f"| {m['pattern']} | {m['model_calls']} | {m['input']} | {m['cached']} | {m['output']} | {_f(m['cost'], 4)} | "
                   f"{_s(m['latency_ms'])} | {_s(m['serial_ms'])} | {_f(m['success'])} |")
    out += ["", "## Patterns as the task grows (1,500-token tool results, context penalty 0.01 per 1K tokens)", "",
            "| sub-questions | pattern | model calls | input tokens | cost ($) | latency (s) | P(success) |", "|---|---|---|---|---|---|---|"]
    for g in grid:
        if g["result_tokens"] == 1500 and g["slope"] == 0.01:
            for m in g["rows"]:
                out.append(f"| {g['k']} | {m['pattern']} | {m['model_calls']} | {m['input']} | {_f(m['cost'], 4)} | "
                           f"{_s(m['latency_ms'])} | {_f(m['success'])} |")
    out += ["", "## Best pattern by P(success) across the grid", "",
            "| sub-questions | tool result tokens | penalty per 1K | best | P(success) | single agent |", "|---|---|---|---|---|---|"]
    for g in grid:
        best = max(g["rows"], key=lambda m: m["success"])
        single = next(m for m in g["rows"] if m["pattern"] == "single")
        out.append(f"| {g['k']} | {g['result_tokens']} | {g['slope']} | {best['pattern']} | {_f(best['success'])} | {_f(single['success'])} |")
    out += ["", f"## Fan-out under a rate limit ({S.FAN['chunks']} chunks of {S.FAN['chunk_tokens']} tokens, map-reduce, "
            f"{S.FAN['rpm']} RPM, {S.FAN['otpm']} OTPM)", "",
            "| ITPM | width | run time (s) | speed-up | Amdahl | rate-limit ceiling |", "|---|---|---|---|---|---|"]
    for f in fans:
        for r in f["rows"]:
            out.append(f"| {f['itpm']} | {r['width']} | {_s(r['ms'])} | {_f(r['speedup'], 2)} | {_f(r['amdahl'], 2)} | {_f(r['ceiling'], 2)} |")
    out += ["", "| ITPM | one at a time (s) | parallel fraction | bound from the buckets (s) | requests | uncached input | output |",
            "|---|---|---|---|---|---|---|"]
    for f in fans:
        c = f["ceiling"]
        out.append(f"| {f['itpm']} | {_s(f['t1'])} | {_f(f['parallel_fraction'])} | {_s(f['bound_ms'])} | {c['req']} | {c['tin']} | {c['tout']} |")
    out += ["", f"## Reliability (detectable errors e = {S.REL['e']}, silent errors w = {S.REL['w']}, {S.REL['runs']} runs per point, "
            f"verifier recall {S.REL['verifier']['recall']}, false rejections {S.REL['verifier']['false_reject']})", "",
            "| configuration | steps | closed form | simulated | 95% CI | mean attempts | closed form |", "|---|---|---|---|---|---|---|"]
    for r in rel:
        for c in r["curve"]:
            out.append(f"| {r['name']} | {c['n']} | {_f(c['closed'], 4)} | {_f(c['rate'], 4)} | [{_f(c['ci'][0], 3)}, {_f(c['ci'][1], 3)}] | "
                       f"{_f(c['attempts'], 2)} | {_f(c['closed_attempts'], 2)} |")
    out += ["", f"## At scale ({S.SCALE['workflows']} workflows per row, Poisson arrivals, seed {S.SCALE['seed']}; "
            f"{D.DEFAULTS['rpm']} RPM, {D.DEFAULTS['itpm']} ITPM, {D.DEFAULTS['otpm']} OTPM, concurrency {D.DEFAULTS['concurrency']}, "
            f"transient failure {D.DEFAULTS['fail']}, {D.DEFAULTS['max_attempts']} attempts)", "",
            "| pattern | capacity (workflows/min) | limited by | arrivals/min | success | p50 (s) | p95 (s) | p99 (s) | $ per success | completed/min |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for x in scale:
        cap = x["capacity"]
        lim = min(["requests", "input", "output", "concurrency"], key=lambda k: cap[k])
        for rate, r in zip(S.SCALE["rates"], x["rows"]):
            out.append(f"| {x['pattern']} | {_f(cap['min'], 2)} | {lim} | {rate} | {_f(r['success'])} | {_s(r['p50'])} | {_s(r['p95'])} | "
                       f"{_s(r['p99'])} | {_f(r['cost_per_success'], 4)} | {_f(r['throughput_per_min'], 2)} |")
    out += ["", "## Choosing a pattern (DES, 1,500-token tool results; arrivals per minute: "
            + ", ".join(f"{r} for {k} sub-questions" for k, r in zip(S.CHOOSE["k"], S.CHOOSE["rates"])) + ")", "",
            "| sub-questions | pattern | success | closed form | p95 (s) | $ per success | completed/min | capacity (workflows/min) |",
            "|---|---|---|---|---|---|---|---|"]
    for k, tab in tables.items():
        for r in tab:
            out.append(f"| {k} | {r['pattern']} | {_f(r['success'])} | {_f(r['closed_success'])} | {_s(r['p95'])} | "
                       f"{_f(r['cost_per_success'], 4)} | {_f(r['throughput_per_min'], 2)} | {_f(r['capacity'], 2)} |")
    out += ["", "| sub-questions | constraints | survivors | pick |", "|---|---|---|---|"]
    for c in choices:
        lim = ", ".join(f"{k} {'≥' if k in ('success', 'capacity') else '≤'} {v}" for k, v in c["limits"].items())
        out.append(f"| {c['k']} | {lim} | {', '.join(c['result']['alive']) or 'none'} | {c['result']['pick'] or 'none'} |")
    return "\n".join(out) + "\n"
