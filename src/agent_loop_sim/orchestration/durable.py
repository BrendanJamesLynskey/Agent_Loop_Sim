"""Durable execution: an append-only event history, deterministic replay after a crash, what a
non-deterministic step does to replay, and idempotency keys on side-effecting activities.

The model follows the workflow/activity split of durable-execution engines such as Temporal (and
LangGraph's own advice: put side effects in tasks or nodes, make them idempotent, and keep the code
deterministic). The workflow's code is a list of steps; running it issues commands (schedule an
activity, record a marker), and each command's outcome is appended to the history. When a worker
dies, a new worker runs the code again from the top: each command is matched against the history,
recorded results are returned without running anything, and live execution resumes where the
history ends. A step whose command differs from the recorded one (because it read the clock, say)
stops the replay with a non-determinism error.

The event names and the history's shape are simplified (illustrative); the timings are
illustrative too. Every run is deterministic given its parameters; the sweep is seeded.
"""
from __future__ import annotations

import copy
from typing import Any

from ..rng import Rng

WORKFLOW = {
    "id": "order-1042",
    "steps": [
        {"kind": "activity", "name": "reserve_stock", "ms": 300},
        {"kind": "activity", "name": "charge_card", "ms": 800, "effect": "charge", "amount": 40},
        {"kind": "choose", "name": "shipping", "cutoff_ms": 1500, "early": "express", "late": "standard"},
        {"kind": "activity", "name": "ship", "ms": 500},
        {"kind": "activity", "name": "send_receipt", "ms": 200, "effect": "email"},
    ],
}
TIMEOUT_MS = 2000
RESTART_MS = 5000
REPLAY_MS = 5
MAX_ATTEMPTS = 3
CHOOSE_MODES = ["clock", "marker"]


class _Crash(Exception):
    pass


def _activity_name(step: dict[str, Any], choice: str | None) -> str:
    if step["name"] == "ship":
        return "ship_" + (choice or "?")
    return step["name"]


def run_durable(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """One workflow run. ``params``: ``crash`` (``{"after_event": k}`` the worker dies once event
    k is written; ``{"in_activity": name}`` it dies after the outside world acted but before the
    completion is recorded), ``lost`` ({activity: [attempts whose response is lost]}),
    ``idempotency`` (send a key with each side effect), ``choose`` (``clock`` reads the clock in
    the workflow code: non-deterministic; ``marker`` records the choice in the history).
    Returns the history, the outside world (ledger of charges, emails), the status and frames."""
    p = dict(params or {})
    crash = p.get("crash")
    lost = p.get("lost", {})
    key_on = bool(p.get("idempotency", False))
    mode = p.get("choose", "clock")
    if mode not in CHOOSE_MODES:
        raise ValueError(f"unknown choose mode {mode}")
    wf = WORKFLOW
    history: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    emails: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    st = {"t": 0, "worker": 1, "crashed": False}
    attempts_by_seq: dict[int, int] = {}

    def frame(kind: str, caption: str, mode_: str, line: int, cursor: int, extra: dict[str, Any] | None = None) -> None:
        f = {"t": st["t"], "worker": st["worker"], "mode": mode_, "line": line, "kind": kind,
             "hlen": len(history), "cursor": cursor, "charges": len(ledger), "emails": len(emails), "caption": caption}
        if extra:
            f.update(extra)
        frames.append(f)

    def append(ev: dict[str, Any], line: int, caption: str) -> None:
        ev = dict(ev, i=len(history), t=st["t"])
        history.append(ev)
        frame("event", caption, "live", line, len(history), {"event": ev["i"]})
        if crash and "after_event" in crash and st["worker"] == 1 and ev["i"] == crash["after_event"]:
            raise _Crash()

    def effect(step: dict[str, Any], seq: int, attempt: int) -> str:
        key = wf["id"] + "/" + str(seq) if key_on else None
        if step.get("effect") == "charge":
            if key is not None:
                for c in ledger:
                    if c["key"] == key:
                        return "dedup"
            ledger.append({"key": key, "amount": step["amount"], "t": st["t"], "attempt": attempt, "worker": st["worker"]})
            return "charged"
        if step.get("effect") == "email":
            if key is not None:
                for e in emails:
                    if e["key"] == key:
                        return "dedup"
            emails.append({"key": key, "t": st["t"], "worker": st["worker"]})
            return "sent"
        return "done"

    def run_activity(step: dict[str, Any], name: str, seq: int, line: int, live_start: bool) -> str:
        if live_start:
            append({"type": "activity_scheduled", "seq": seq, "activity": name}, line,
                   f"Schedule {name} (command {seq}); the history records it before anything runs.")
        while True:
            attempt = attempts_by_seq.get(seq, 0) + 1
            attempts_by_seq[seq] = attempt
            st["t"] += step["ms"]
            out = effect(step, seq, attempt)
            key = wf["id"] + "/" + str(seq) if key_on else None
            if out == "charged":
                cap = f"{name} attempt {attempt}: the payment service charges £{step['amount']}" + \
                      (f" (key {key})." if key else " (no idempotency key).")
            elif out == "dedup":
                cap = f"{name} attempt {attempt}: the service has already seen key {key}, so it returns the first result and charges nothing."
            elif out == "sent":
                cap = f"{name} attempt {attempt}: the email goes out."
            else:
                cap = f"{name} attempt {attempt} runs."
            frame("effect", cap, "live", line, len(history), {"effect": out, "seq": seq, "attempt": attempt})
            if crash and crash.get("in_activity") == name and st["worker"] == 1:
                raise _Crash()
            if attempt in lost.get(name, []):
                st["t"] += TIMEOUT_MS
                if attempt >= MAX_ATTEMPTS:
                    append({"type": "activity_failed", "seq": seq, "attempt": attempt}, line,
                           f"{name}: no reply to attempt {attempt}, the last allowed; the activity fails.")
                    return "failed"
                append({"type": "activity_timed_out", "seq": seq, "attempt": attempt}, line,
                       f"{name}: the reply to attempt {attempt} is lost; after {TIMEOUT_MS} ms the worker retries.")
                continue
            append({"type": "activity_completed", "seq": seq, "result": out}, line,
                   f"{name} completed ({out}); the result is in the history now.")
            return out

    def worker() -> str:
        cursor = 1  # event 0 (workflow_started) is written before any worker runs
        choice: str | None = None
        seq = 0
        for line, step in enumerate(wf["steps"]):
            if step["kind"] == "choose":
                if mode == "marker":
                    if cursor < len(history):
                        ev = history[cursor]
                        if ev["type"] != "marker" or ev["name"] != step["name"]:
                            return _nondet(ev, f"record marker {step['name']}", line, cursor)
                        choice = ev["value"]
                        cursor += 1
                        frame("replay", f"Replay: the {step['name']} choice comes from the history (marker: {choice}), not the clock.",
                              "replay", line, cursor)
                        continue
                    choice = step["early"] if st["t"] < step["cutoff_ms"] else step["late"]
                    append({"type": "marker", "name": step["name"], "value": choice}, line,
                           f"The clock reads {st['t']} ms, so {step['name']} = {choice}; recorded as a marker.")
                    cursor = len(history)
                    continue
                choice = step["early"] if st["t"] < step["cutoff_ms"] else step["late"]
                frame("code", f"The code reads the clock ({st['t']} ms) and picks {choice} shipping; nothing is recorded.",
                      "replay" if cursor < len(history) else "live", line, cursor)
                continue
            name = _activity_name(step, choice)
            if cursor < len(history):
                ev = history[cursor]
                if ev["type"] != "activity_scheduled" or ev["activity"] != name:
                    return _nondet(ev, f"schedule {name}", line, cursor)
                cursor += 1
                result = None
                while cursor < len(history) and history[cursor].get("seq") == seq:
                    if history[cursor]["type"] == "activity_completed":
                        result = history[cursor]["result"]
                    if history[cursor]["type"] == "activity_failed":
                        result = "failed"
                    cursor += 1
                if result is not None:
                    st["t"] += REPLAY_MS
                    frame("replay", f"Replay: {name} is in the history ({result}); its recorded result is returned and nothing runs.",
                          "replay", line, cursor)
                    if result == "failed":
                        return "failed"
                    seq += 1
                    continue
                frame("replay", f"Replay: {name} was scheduled but never completed, so it runs again (at least once).",
                      "replay", line, cursor)
                out = run_activity(step, name, seq, line, False)
                cursor = len(history)
            else:
                out = run_activity(step, name, seq, line, True)
                cursor = len(history)
            if out == "failed":
                return "failed"
            seq += 1
        append({"type": "workflow_completed"}, len(wf["steps"]), "The workflow completed.")
        return "completed"

    def _nondet(ev: dict[str, Any], wanted: str, line: int, cursor: int) -> str:
        got = ev["type"] + (" " + ev["activity"] if "activity" in ev else "") + (" " + ev["name"] if ev["type"] == "marker" else "")
        frame("nondeterminism", f"Non-determinism: the code now says '{wanted}' but the history has '{got}' at event {cursor}. "
              "Replay stops; the workflow is stuck until the code is fixed.", "replay", line, cursor,
              {"wanted": wanted, "got": got})
        st["nondet"] = {"event": cursor, "wanted": wanted, "got": got}
        return "nondeterminism"

    frames_start = {"type": "workflow_started", "workflow": wf["id"]}
    status = "completed"
    try:
        append(frames_start, -1, f"Workflow {wf['id']} starts on worker 1; event 0 is written.")
        status = worker()
    except _Crash:
        st["crashed"] = True
        frame("crash", f"Worker 1 dies at {st['t']} ms. The history ({len(history)} events) survives in the store.",
              "live", -1, len(history))
        st["t"] += RESTART_MS
        st["worker"] = 2
        frame("restart", f"Worker 2 picks the workflow up at {st['t']} ms and replays the code from the top.",
              "replay", -1, 1)
        status = worker()
    return {"params": p, "status": status, "history": history, "ledger": ledger, "emails": emails,
            "charges": len(ledger), "charged": sum(c["amount"] for c in ledger), "frames": frames,
            "crashed": st["crashed"], "nondet": st.get("nondet"), "t_end": st["t"]}


def charge_sweep(q: float, attempts: int, runs: int, seed: int) -> dict[str, Any]:
    """Many orders, each charge attempt's reply lost with probability ``q``, up to ``attempts``
    tries. Counts charges with and without an idempotency key (every attempt reaches the service;
    only replies are lost). Closed forms: without a key E[charges] = (1 − q^A)/(1 − q) and
    P(charged twice or more) = q (for A ≥ 2); with a key, exactly one charge."""
    r = Rng(seed)
    total_no = 0
    total_key = 0
    dup = 0
    failed = 0
    hist = [0] * (attempts + 1)
    for _ in range(runs):
        n = 0
        ok = False
        while n < attempts:
            n += 1
            if r.random() >= q:
                ok = True
                break
        if not ok:
            failed += 1
        total_no += n
        total_key += 1
        hist[n] += 1
        if n >= 2:
            dup += 1
    mean_no = total_no / runs
    qa = 1.0
    for _ in range(attempts):
        qa = qa * q
    expected = (1 - qa) / (1 - q)
    return {"q": q, "attempts": attempts, "runs": runs, "seed": seed, "mean_no_key": mean_no,
            "mean_key": total_key / runs, "dup_rate": dup / runs, "failed": failed, "hist": hist,
            "expected_no_key": expected, "expected_dup": q if attempts >= 2 else 0.0,
            "expected_failed": qa}


SCENARIOS: list[dict[str, Any]] = [
    {"name": "clean", "title": "A clean run", "params": {}},
    {"name": "crash_replay", "title": "Crash after the charge, then replay", "params": {"crash": {"after_event": 4}, "choose": "marker"}},
    {"name": "crash_in_charge", "title": "Crash inside the charge (no key)", "params": {"crash": {"in_activity": "charge_card"}, "choose": "marker"}},
    {"name": "crash_in_charge_key", "title": "Crash inside the charge (with a key)",
     "params": {"crash": {"in_activity": "charge_card"}, "choose": "marker", "idempotency": True}},
    {"name": "lost_reply", "title": "A lost reply and a retry (no key)", "params": {"lost": {"charge_card": [1]}, "choose": "marker"}},
    {"name": "lost_reply_key", "title": "A lost reply and a retry (with a key)",
     "params": {"lost": {"charge_card": [1]}, "choose": "marker", "idempotency": True}},
    {"name": "nondeterministic", "title": "Replay with a clock read in the code", "params": {"crash": {"after_event": 5}, "choose": "clock"}},
    {"name": "deterministic", "title": "The same, with the choice recorded as a marker", "params": {"crash": {"after_event": 5}, "choose": "marker"}},
]


def scenario(name: str) -> dict[str, Any]:
    for s in SCENARIOS:
        if s["name"] == name:
            return copy.deepcopy(s)
    raise KeyError(name)
