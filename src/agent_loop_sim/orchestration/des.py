"""A discrete-event simulation of many workflows sharing one rate-limited model endpoint (since 1.7).

Each workflow is a pattern's plan (``patterns.plan``): a call becomes ready when the calls it
depends on have finished. Ready model calls join one FIFO queue in front of the endpoint; tool
calls just take their time. The client admits the call at the head of the queue when

  - fewer than ``concurrency`` calls are in flight,
  - the request bucket holds one request, the input-token bucket holds the call's uncached input
    tokens, and the output-token bucket is not negative.

The buckets follow the token-bucket scheme the Claude API documents for its limits (capacity =
the per-minute limit, replenished continuously; only uncached input tokens count toward the
input limit; output tokens count as they are produced, here charged when the call ends). A
well-behaved client that waits for capacity instead of collecting 429s is assumed.

An attempt takes the latency model's duration (``accounting.call_latency``) times a uniform
jitter factor. It fails with probability ``fail`` (a transient error, returned at the time to first
token and not billed), or times out after ``timeout_ms`` (billed in full: the server finished it;
an assumption). A failed attempt is retried after a backoff with "full jitter" (uniform on
[0, min(cap, base·2^(attempt−1))]) until ``max_attempts``; then the workflow fails. A successful
model call is correct with its plan probability (drawn when ``semantic``); a workflow succeeds if
it completes and its plan's success rule holds (``patterns``).

Deterministic given the seed; Python and TS draw the same random numbers in the same order (one
mulberry32 stream; arrival gaps use the shared fdlibm ``ln``). Every limit and timing is
illustrative.
"""
from __future__ import annotations

import heapq
from typing import Any

from ..accounting import LATENCY, PRICES, call_latency
from ..context.mathx import ln
from ..rng import Rng
from .patterns import cached_tokens, cost_of, maj3

DEFAULTS: dict[str, Any] = {
    "workflows": 50,
    "arrival": "poisson",
    "per_min": 6.0,
    "rpm": 50,
    "itpm": 40000,
    "otpm": 8000,
    "concurrency": 16,
    "fail": 0.03,
    "timeout_ms": 60000,
    "max_attempts": 3,
    "backoff_ms": 1000,
    "backoff_cap_ms": 30000,
    "jitter": [0.8, 1.5],
    "price": "claude-sonnet-4.6",
    "profile": "hosted",
    "seed": 1,
    "semantic": True,
    "trace": False,
    "samples": 0,
}
EPS = 1e-6


def params(p: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    for k, v in (p or {}).items():
        out[k] = v
    return out


def _pow2(n: int) -> int:
    v = 1
    for _ in range(n):
        v = v * 2
    return v


def pct(xs: list[float], q: int) -> float:
    """Nearest-rank percentile (q in percent) of an unsorted list; 0 for an empty list."""
    if not xs:
        return 0.0
    s = sorted(xs)
    i = (q * len(s) + 99) // 100 - 1
    if i < 0:
        i = 0
    return s[i]


def simulate(plan: dict[str, Any], p: dict[str, Any] | None = None) -> dict[str, Any]:
    P = params(p)
    price = PRICES[P["price"]]
    profile = LATENCY[P["profile"]]
    calls = plan["calls"]
    nwf = P["workflows"]
    rng = Rng(P["seed"])
    succ: list[list[int]] = [[] for _ in calls]
    for c in calls:
        for d in c["deps"]:
            succ[d].append(c["id"])
    # arrivals
    arrivals: list[float] = []
    t = 0.0
    for i in range(nwf):
        if P["arrival"] == "batch":
            arrivals.append(0.0)
        else:
            if i > 0:
                u = rng.random()
                t = t + (-ln(1.0 - u)) * 60000 / P["per_min"]
            arrivals.append(t)
    wf = [{"id": i, "arrival": arrivals[i], "end": None, "status": "running", "cost": 0.0, "attempts": 0,
           "wrong": [], "votes": {}, "left": [len(c["deps"]) for c in calls], "remaining": len(calls)} for i in range(nwf)]
    heap: list[tuple[float, int, str, int, int, str]] = []
    seq = [0]

    def push(at: float, kind: str, w: int, c: int, note: str = "") -> None:
        heapq.heappush(heap, (at, seq[0], kind, w, c, note))
        seq[0] += 1

    for i in range(nwf):
        push(arrivals[i], "arrive", i, -1)
    queue: list[tuple[int, int]] = []
    attempts: dict[tuple[int, int], int] = {}
    buckets = {"rq": float(P["rpm"]), "it": float(P["itpm"]), "ot": float(P["otpm"]), "last": 0.0}
    state = {"inflight": 0, "wake": -1.0, "busy_ms": 0.0}
    trace: list[dict[str, Any]] = []
    changes: list[list[float]] = []
    counts = {"insys": 0, "ok": 0, "bad": 0}

    def note_change(at: float) -> None:
        row = [at, float(len(queue)), float(state["inflight"]), float(counts["insys"]), float(counts["ok"]), float(counts["bad"])]
        if changes and changes[-1][0] == at:
            changes[-1] = row
        elif not changes or changes[-1][1:] != row[1:]:
            changes.append(row)

    def refill(at: float) -> None:
        dt = at - buckets["last"]
        if dt > 0:
            rq = buckets["rq"] + dt * P["rpm"] / 60000
            buckets["rq"] = rq if rq < P["rpm"] else float(P["rpm"])
            it = buckets["it"] + dt * P["itpm"] / 60000
            buckets["it"] = it if it < P["itpm"] else float(P["itpm"])
            ot = buckets["ot"] + dt * P["otpm"] / 60000
            buckets["ot"] = ot if ot < P["otpm"] else float(P["otpm"])
            buckets["last"] = at

    def need_in(c: dict[str, Any]) -> float:
        n = float(c["input"] - cached_tokens(c, price))
        return n if n < P["itpm"] else float(P["itpm"])

    def ready(at: float, w: int, cid: int) -> None:
        c = calls[cid]
        if c["kind"] == "tool":
            if P["trace"]:
                trace.append({"wf": w, "call": cid, "attempt": 1, "queued": at, "start": at, "end": at + c["ms"], "outcome": "tool"})
            push(at + c["ms"], "tool", w, cid)
        else:
            queue.append((w, cid))
            if P["trace"]:
                trace.append({"wf": w, "call": cid, "attempt": attempts.get((w, cid), 0) + 1, "queued": at, "start": None,
                              "end": None, "outcome": "queued"})

    def finish_wf(at: float, w: int, status: str) -> None:
        x = wf[w]
        x["end"] = at
        x["status"] = status
        counts["insys"] -= 1
        if status == "ok":
            counts["ok"] += 1
        else:
            counts["bad"] += 1

    def dispatch(at: float) -> None:
        refill(at)
        while queue and state["inflight"] < P["concurrency"]:
            w, cid = queue[0]
            if wf[w]["status"] != "running":
                queue.pop(0)
                continue
            c = calls[cid]
            ni = need_in(c)
            if buckets["rq"] + EPS >= 1 and buckets["it"] + EPS >= ni and buckets["ot"] + EPS >= 0:
                queue.pop(0)
                buckets["rq"] = buckets["rq"] - 1
                buckets["it"] = buckets["it"] - ni
                state["inflight"] += 1
                a = attempts.get((w, cid), 0) + 1
                attempts[(w, cid)] = a
                wf[w]["attempts"] += 1
                u = rng.random()
                v = rng.random()
                jit = P["jitter"][0] + (P["jitter"][1] - P["jitter"][0]) * u
                ttft, dur = call_latency(profile, c["input"], cached_tokens(c, price), c["output"])
                ttft = ttft * jit
                dur = dur * jit
                if v < P["fail"]:
                    outcome, end = "error", at + ttft
                elif dur > P["timeout_ms"]:
                    outcome, end = "timeout", at + P["timeout_ms"]
                else:
                    outcome, end = "ok", at + dur
                state["busy_ms"] = state["busy_ms"] + (end - at)
                if P["trace"]:
                    for r in reversed(trace):
                        if r["wf"] == w and r["call"] == cid and r["outcome"] == "queued":
                            r["start"] = at
                            r["end"] = end
                            r["outcome"] = outcome
                            r["attempt"] = a
                            break
                push(end, "done", w, cid, outcome)
            else:
                wait = 0.0
                if buckets["rq"] + EPS < 1:
                    x = (1 - buckets["rq"]) / (P["rpm"] / 60000)
                    wait = x if x > wait else wait
                if buckets["it"] + EPS < ni:
                    x = (ni - buckets["it"]) / (P["itpm"] / 60000)
                    wait = x if x > wait else wait
                if buckets["ot"] + EPS < 0:
                    x = (0 - buckets["ot"]) / (P["otpm"] / 60000)
                    wait = x if x > wait else wait
                if wait < 0.001:
                    wait = 0.001
                if state["wake"] < 0 or state["wake"] > at + wait:
                    state["wake"] = at + wait
                    push(at + wait, "wake", -1, -1)
                break

    while heap:
        at, _, kind, w, cid, note = heapq.heappop(heap)
        if kind == "arrive":
            counts["insys"] += 1
            for c in calls:
                if not c["deps"]:
                    ready(at, w, c["id"])
        elif kind == "wake":
            if state["wake"] == at:
                state["wake"] = -1.0
        elif kind == "retry":
            if wf[w]["status"] == "running":
                ready(at, w, cid)
        else:
            c = calls[cid]
            x = wf[w]
            ok = True
            if kind == "done":
                state["inflight"] -= 1
                refill(at)
                if note != "error":
                    buckets["ot"] = buckets["ot"] - c["output"]
                    x["cost"] = x["cost"] + cost_of(c, price)
                ok = note == "ok"
            if x["status"] == "running":
                if ok:
                    if kind == "done" and P["semantic"]:
                        right = rng.random() < c["p"]
                        if c["role"] == "critical" and not right:
                            x["wrong"].append(cid)
                        if c["role"] == "vote":
                            v = x["votes"].setdefault(c["group"], [0, 0])
                            v[0] += 1 if right else 0
                            v[1] += 1
                    x["remaining"] -= 1
                    for s in succ[cid]:
                        x["left"][s] -= 1
                        if x["left"][s] == 0:
                            ready(at, w, s)
                    if x["remaining"] == 0:
                        good = not x["wrong"]
                        for v in x["votes"].values():
                            if v[0] * 2 <= v[1]:
                                good = False
                        finish_wf(at, w, "ok" if good else "wrong")
                else:
                    a = attempts[(w, cid)]
                    if a < P["max_attempts"]:
                        cap = P["backoff_ms"] * _pow2(a - 1)
                        if cap > P["backoff_cap_ms"]:
                            cap = P["backoff_cap_ms"]
                        push(at + rng.random() * cap, "retry", w, cid)
                    else:
                        finish_wf(at, w, "exhausted")
        dispatch(at)
        note_change(at)

    done = [x for x in wf if x["end"] is not None]
    lat_all = [x["end"] - x["arrival"] for x in done if x["status"] != "exhausted"]
    lat_ok = [x["end"] - x["arrival"] for x in done if x["status"] == "ok"]
    n_ok = len([x for x in wf if x["status"] == "ok"])
    makespan = 0.0
    for x in done:
        if x["end"] > makespan:
            makespan = x["end"]
    cost = 0.0
    for x in wf:
        cost = cost + x["cost"]
    tot_att = 0
    for x in wf:
        tot_att += x["attempts"]
    model_calls = len([c for c in calls if c["kind"] != "tool"])
    mean_lat = 0.0
    for v in lat_all:
        mean_lat = mean_lat + v
    mean_lat = mean_lat / len(lat_all) if lat_all else 0.0
    out: dict[str, Any] = {
        "pattern": plan["pattern"], "params": P, "workflows": nwf,
        "ok": n_ok, "wrong": len([x for x in wf if x["status"] == "wrong"]),
        "exhausted": len([x for x in wf if x["status"] == "exhausted"]),
        "success": n_ok / nwf if nwf else 0.0,
        "p50": pct(lat_all, 50), "p95": pct(lat_all, 95), "p99": pct(lat_all, 99), "mean": mean_lat,
        "p50_ok": pct(lat_ok, 50), "p95_ok": pct(lat_ok, 95),
        "cost": cost, "cost_per_success": cost / n_ok if n_ok else 0.0,
        "makespan": makespan, "throughput_per_min": n_ok / (makespan / 60000) if makespan > 0 else 0.0,
        "attempts": tot_att, "model_calls": model_calls,
        "utilisation": state["busy_ms"] / (P["concurrency"] * makespan) if makespan > 0 else 0.0,
        "runs": [{"id": x["id"], "arrival": x["arrival"], "end": x["end"], "status": x["status"], "cost": x["cost"],
                  "attempts": x["attempts"]} for x in wf],
    }
    if P["trace"]:
        out["trace"] = trace
        out["changes"] = changes
    if P["samples"]:
        out["series"] = resample(changes, makespan, P["samples"])
    return out


def resample(changes: list[list[float]], horizon: float, n: int) -> list[dict[str, Any]]:
    """The step functions (queue, in flight, in system, done ok, done not ok) at n+1 evenly spaced
    times from 0 to the horizon: the value of the last change at or before each time."""
    out = []
    j = 0
    for i in range(n + 1):
        t = horizon * i / n
        while j + 1 < len(changes) and changes[j + 1][0] <= t:
            j += 1
        r = changes[j] if changes and changes[j][0] <= t else [t, 0.0, 0.0, 0.0, 0.0, 0.0]
        out.append({"t": t, "queue": int(r[1]), "inflight": int(r[2]), "insys": int(r[3]), "ok": int(r[4]), "bad": int(r[5])})
    return out


def success_closed(plan: dict[str, Any], fail: float, attempts: int) -> float:
    """P(a workflow succeeds) with no timeouts: every model call must get through within
    ``attempts`` tries (1 − fail^A each) and the plan's success rule must hold."""
    fa = 1.0
    for _ in range(attempts):
        fa = fa * fail
    through = 1.0
    crit = 1.0
    groups: dict[str, list[float]] = {}
    for c in plan["calls"]:
        if c["kind"] == "tool":
            continue
        through = through * (1 - fa)
        if c["role"] == "critical":
            crit = crit * c["p"]
        elif c["role"] == "vote":
            groups.setdefault(c["group"], []).append(c["p"])
    for g in groups.values():
        crit = crit * maj3(g[0], g[1], g[2])
    return through * crit


def expected_attempts(fail: float, attempts: int) -> float:
    """E[attempts per call] = (1 − f^A)/(1 − f) (by repeated multiplication)."""
    s = 0.0
    f = 1.0
    for _ in range(attempts):
        s = s + f
        f = f * fail
    return s
