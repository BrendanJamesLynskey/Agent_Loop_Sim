"""The harness: the loop that turns a chat model into an agent, with every choice a policy.

One turn: (compact the context if it is too full) → render the prompt → call the model →
parse its output → for each tool call: check permissions (maybe ask the simulated human) →
run the pre-tool hooks → execute (inside the shell sandbox, with injected failures, retries and back-off) → run the
post-tool hooks → append the results → next turn. The run stops on a final answer, the
turn limit, loop detection or a context overflow.

Every step appends an event to the trace (``trace.py`` holds the schema). Time is simulated:
model calls take the latency model's time, tools their latency, humans their answer time.
"""
from __future__ import annotations

import math
from typing import Any

from . import VERSION
from .accounting import LATENCY, PRICES, PromptCache, call_cost, call_latency
from .chat import GENERATION_PROMPT, message_segment, system_segment
from .models import ReplayModel, ScriptedModel, summarise
from .parse import parse
from .rng import MASK, Rng
from .tokenizer import Tokenizer, default_tokenizer
from .tools import TOOL_SPECS, ToolError, World, execute, glob_match, sandbox_violation
from .jsonfmt import dumps

DEFAULT_POLICY: dict[str, Any] = {
    "style": "native",
    "max_turns": 20,
    "parallel": True,
    "max_output": 512,
    "context": {"window": 32768, "strategy": "none", "trigger": 0.8, "target": 0.5, "keep_last": 4, "clip_lines": 6},
    "cache": {"enabled": True, "price": "claude-sonnet-4.6", "layout": "stable"},
    "latency": "hosted",
    "permissions": {"mode": "auto", "rules": [], "human": {"latency_ms": [3000, 9000], "deny": []}},
    "hooks": [],
    "recovery": {
        "retries": 2,
        "backoff_ms": 500,
        "backoff_factor": 2,
        "loop_repeats": 3,
        "loop_action": "stop",
        "malformed": "feedback",
    },
    "subagents": {
        "system": "You are a sub-agent. Complete the sub-task with the tools, then reply with a short report of what you found.",
        "max_turns": 10,
    },
}

HUMAN_SEED_OFFSET = 0x9E3779B9


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    """``over`` on top of ``base``, recursively for nested objects (lists are replaced)."""
    out = dict(base)
    if over is None:
        return out
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def clock_text(t: float) -> str:
    """The simulated wall clock (the run starts at 09:00:00) as HH:MM:SS."""
    s = math.floor(t / 1000)
    h = 9 + math.floor(s / 3600)
    m = math.floor(s / 60) % 60
    return f"{h:02d}:{m:02d}:{s % 60:02d}"


def decide(perm: dict[str, Any], spec: dict[str, Any], subject: str) -> tuple[str, str]:
    """(allow | ask | deny, why) for one call: deny rules first, then the read-only mode,
    then ask rules, then allow rules, then the mode's default."""
    name = spec["name"]
    mode = perm["mode"]
    hits: dict[str, str] = {}
    for r in perm["rules"]:
        if r["tool"] in (name, "*") and glob_match(r.get("pattern", "*"), subject):
            if r["decision"] not in hits:
                hits[r["decision"]] = f"rule: {r['decision']} {r['tool']}({r.get('pattern', '*')})"
    if "deny" in hits:
        return "deny", hits["deny"]
    if mode == "read_only" and not spec["read_only"]:
        return "deny", "mode: read-only"
    if mode == "ask":
        return "ask", "mode: ask every time"
    if "ask" in hits:
        return "ask", hits["ask"]
    if "allow" in hits:
        return "allow", hits["allow"]
    if mode == "auto" or mode == "read_only":
        return "allow", f"mode: {'auto' if mode == 'auto' else 'read-only'}"
    if spec["read_only"]:
        return "allow", "mode: default (reads allowed)"
    return "ask", "mode: default (writes ask)"


class Run:
    """One run of a scenario under a policy. ``run()`` returns its events."""

    def __init__(
        self,
        scenario: dict[str, Any],
        policy: dict[str, Any] | None = None,
        model: Any = None,
        seed: int | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.scenario = scenario
        self.policy = merge(merge(DEFAULT_POLICY, scenario.get("policy")), policy)
        self.tok = tokenizer or default_tokenizer()
        self.seed = (scenario.get("seed", 0) if seed is None else seed) & MASK
        self.rng_tools = Rng(self.seed)
        self.rng_human = Rng((self.seed + HUMAN_SEED_OFFSET) & MASK)
        self.world = World(scenario.get("files", {}), scenario.get("corpus", []))
        self.model = model if model is not None else ScriptedModel(scenario["script"])
        self.price = PRICES[self.policy["cache"]["price"]]
        self.profile = LATENCY[self.policy["latency"]]
        cache = self.policy["cache"]
        self.cache = PromptCache(cache["enabled"], self.price["min_tokens"], self.price["block"], self.price["ttl_ms"])
        self.events: list[dict[str, Any]] = []
        self.t = 0.0
        self.msg_id = 0
        self.attempts: dict[str, int] = {}
        self.spawned = 0
        self.gen_ids = self.tok.encode(GENERATION_PROMPT)
        self.sys_cache: dict[str, tuple[list[int], int]] = {}

    # ── events ───────────────────────────────────────────────────────────

    def emit(self, type_: str, agent: str, t: float, **fields: Any) -> dict[str, Any]:
        ev = {"v": 1, "seq": len(self.events), "t": t, "agent": agent, "type": type_}
        ev.update(fields)
        self.events.append(ev)
        return ev

    # ── messages and prompts ─────────────────────────────────────────────

    def message(self, role: str, kind: str, content: str, turn: int) -> dict[str, Any]:
        self.msg_id += 1
        m = {"id": self.msg_id, "role": role, "kind": kind, "content": content, "turn": turn}
        m["ids"] = self.tok.encode(message_segment(m))
        return m

    def tools_for(self, names: list[str], turn: int) -> list[dict[str, Any]]:
        tools = [TOOL_SPECS[n] for n in names]
        if self.policy["cache"]["layout"] == "reorder_tools" and len(tools) > 1:
            k = turn % len(tools)
            tools = tools[k:] + tools[:k]
        return tools

    def system_for(self, system: str, turn: int) -> str:
        if self.policy["cache"]["layout"] == "timestamp":
            return system + "\nCurrent time: " + clock_text(self.t)
        return system

    def prompt(self, system: str, tool_names: list[str], msgs: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        style = self.policy["style"]
        tools = self.tools_for(tool_names, turn)
        seg, bare = system_segment(self.system_for(system, turn), tools, style)
        hit = self.sys_cache.get(seg)
        if hit is None:
            hit = (self.tok.encode(seg), self.tok.count(bare))
            self.sys_cache[seg] = hit
        sys_ids, bare_n = hit
        ids = list(sys_ids)
        context = [{"id": "system", "kind": "system", "tokens": bare_n}]
        if len(sys_ids) > bare_n:
            context.append({"id": "tools", "kind": "tools", "tokens": len(sys_ids) - bare_n})
        text = seg
        for m in msgs:
            ids.extend(m["ids"])
            text += message_segment(m)
            context.append({"id": m["id"], "kind": m["kind"], "tokens": len(m["ids"])})
        ids.extend(self.gen_ids)
        context.append({"id": "gen", "kind": "prompt", "tokens": len(self.gen_ids)})
        return {"text": text + GENERATION_PROMPT, "ids": ids, "context": context}

    def prompt_tokens(self, system: str, tool_names: list[str], msgs: list[dict[str, Any]], turn: int) -> int:
        return len(self.prompt(system, tool_names, msgs, turn)["ids"])

    def facts_in(self, msgs: list[dict[str, Any]]) -> list[str]:
        return [f["id"] for f in self.scenario.get("facts", []) if any(f["text"] in m["content"] for m in msgs)]

    # ── model calls ──────────────────────────────────────────────────────

    def model_call(self, agent: str, turn: int, purpose: str, p: dict[str, Any], text: str, message_tokens: int) -> dict[str, Any]:
        n_in = len(p["ids"])
        cached = self.cache.lookup(p["ids"], self.t)
        stored = self.cache.store(p["ids"], self.t)
        n_out = self.tok.count(text) + 1  # + the <|im_end|> that stops generation
        ttft, dur = call_latency(self.profile, n_in, cached, n_out)
        cost = call_cost(self.price, n_in, cached, stored, n_out)
        ev = self.emit(
            "model_call",
            agent,
            self.t,
            turn=turn,
            purpose=purpose,
            input_tokens=n_in,
            cached_tokens=cached,
            output_tokens=n_out,
            cost=cost,
            ttft=ttft,
            dur=dur,
            window=self.policy["context"]["window"],
            context=p["context"],
            text=text,
            message_tokens=message_tokens,
        )
        self.t += dur
        return ev

    # ── context management ───────────────────────────────────────────────

    def compact(self, agent: str, system: str, tool_names: list[str], msgs: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
        ctx = self.policy["context"]
        strategy = ctx["strategy"]
        window = ctx["window"]
        reserve = self.policy["max_output"]
        before = self.prompt_tokens(system, tool_names, msgs, turn)
        if strategy == "none" or before + reserve <= ctx["trigger"] * window:
            return msgs
        target = ctx["target"] * window
        keep = ctx["keep_last"]
        facts_before = self.facts_in(msgs)
        removed: list[int] = []
        clipped: list[int] = []
        msgs = list(msgs)
        if strategy == "clip":
            for i in range(1, max(1, len(msgs) - keep)):
                m = msgs[i]
                if m["kind"] in ("tool_result", "observation") and not m.get("clipped"):
                    lines = m["content"].split("\n")
                    if len(lines) > ctx["clip_lines"]:
                        content = "\n".join(lines[: ctx["clip_lines"]]) + (
                            f"\n[... {len(lines) - ctx['clip_lines']} more lines clipped by the harness]"
                        )
                        nm = dict(m)
                        nm["content"] = content
                        nm["clipped"] = True
                        nm["ids"] = self.tok.encode(message_segment(nm))
                        msgs[i] = nm
                        clipped.append(m["id"])
        # truncation drops the oldest messages until the prompt is under the target; after
        # clipping it only runs if the clipped prompt would still trigger compaction
        if strategy == "truncate" or (
            strategy == "clip" and self.prompt_tokens(system, tool_names, msgs, turn) + reserve > ctx["trigger"] * window
        ):
            while self.prompt_tokens(system, tool_names, msgs, turn) + reserve > target and len(msgs) > 1 + keep:
                removed.append(msgs[1]["id"])
                msgs.pop(1)
        if strategy == "summarise" and len(msgs) > 1 + keep:
            dropped = msgs[1 : len(msgs) - keep]
            text = summarise(dropped, self.scenario.get("facts", []))
            sys_text = "Summarise the conversation so far for the agent that will continue it. Keep what matters for the task."
            p = {
                "text": "",
                "ids": self.tok.encode("<|im_start|>system\n" + sys_text + "<|im_end|>\n")
                + [i for m in dropped for i in m["ids"]]
                + self.gen_ids,
                "context": [],
            }
            p["context"] = [{"id": "summary-input", "kind": "summary_input", "tokens": len(p["ids"])}]
            summary = self.message("user", "summary", text, turn)
            self.model_call(agent, turn, "summary", p, text, len(summary["ids"]))
            removed = [m["id"] for m in dropped]
            msgs = [msgs[0], summary] + msgs[len(msgs) - keep :]
        pa = self.prompt(system, tool_names, msgs, turn)
        after = len(pa["ids"])
        facts_after = self.facts_in(msgs)
        self.emit(
            "compaction",
            agent,
            self.t,
            turn=turn,
            strategy=strategy,
            before=before,
            after=after,
            removed=removed,
            clipped=clipped,
            facts_lost=[f for f in facts_before if f not in facts_after],
            context=pa["context"],
        )
        return msgs

    # ── tool calls ───────────────────────────────────────────────────────

    def fault(self, name: str) -> bool:
        """Does this execution attempt of ``name`` fail transiently?"""
        n = self.attempts.get(name, 0) + 1
        self.attempts[name] = n
        for f in self.scenario.get("faults", []):
            if f["tool"] == name and f["attempt"] == n:
                return True
        rate = self.scenario.get("fail_rates", {}).get(name, 0)
        return rate > 0 and self.rng_tools.random() < rate

    def run_call(self, agent: str, cid: str, call: dict[str, Any], start: float) -> dict[str, Any]:
        """Execute one call at simulated time ``start``: (ok, kind, text, dur)."""
        name, args = call["name"], call["args"]
        spec = TOOL_SPECS[name]
        rec = self.policy["recovery"]
        if name == "task":
            self.t = start
            try:
                text = self.spawn(agent, args["prompt"])
            except ToolError as e:
                return {"ok": False, "kind": "error", "text": f"Error: {e}", "dur": 0.0}
            return {"ok": True, "kind": "ok", "text": text, "dur": self.t - start}
        d = 0.0
        attempt = 0
        if name == "run_shell":
            blocked = sandbox_violation(self.policy.get("sandbox"), args["command"])
            if blocked is not None:
                return {"ok": False, "kind": "sandboxed", "dur": float(spec["latency_ms"]),
                        "text": f"exit code: 1\n{blocked}"}
        while True:
            d += spec["latency_ms"]
            if self.fault(name):
                if attempt < rec["retries"]:
                    backoff = rec["backoff_ms"] * rec["backoff_factor"] ** attempt
                    attempt += 1
                    self.emit("retry", agent, start + d, call=cid, attempt=attempt, backoff=backoff,
                              error=f"TransientError: {name} timed out")
                    d += backoff
                    continue
                return {"ok": False, "kind": "transient", "dur": d,
                        "text": f"TransientError: {name} timed out after {attempt + 1} attempts"}
            try:
                return {"ok": True, "kind": "ok", "text": execute(self.world, name, args), "dur": d}
            except ToolError as e:
                return {"ok": False, "kind": "error", "text": f"Error: {e}", "dur": d}

    def calls(self, agent: str, turn: int, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Permissions and pre-hooks (one call at a time), then execution (in parallel or
        in sequence), then post-hooks. Returns the results in call order."""
        perm = self.policy["permissions"]
        prepared = []
        for k, c in enumerate(calls):
            cid = f"{agent}:{turn}.{k + 1}"
            spec = TOOL_SPECS[c["name"]]
            args = dict(c["args"])
            raw = args.get(spec["subject"], "")
            subject = raw if isinstance(raw, str) else dumps(raw)
            self.emit("tool_call", agent, self.t, turn=turn, call=cid, name=c["name"], args=dict(args), subject=subject)
            decision, why = decide(perm, spec, subject)
            wait = 0.0
            answer = None
            if decision == "ask":
                lo, hi = perm["human"]["latency_ms"]
                wait = self.rng_human.uniform(lo, hi)
                answer = "deny" if any(glob_match(h, subject) for h in perm["human"]["deny"]) else "approve"
            self.emit("permission_check", agent, self.t, call=cid, decision=decision, reason=why, answer=answer, wait=wait)
            self.t += wait
            if decision == "deny" or answer == "deny":
                who = "the user" if answer == "deny" else why
                prepared.append({"cid": cid, "call": c, "args": args, "outcome": "denied",
                                 "text": f"Permission denied ({who}): {c['name']} was not run."})
                continue
            outcome = {"cid": cid, "call": c, "args": args, "outcome": "run", "text": ""}
            for h in self.policy["hooks"]:
                if h["phase"] != "pre" or h["tool"] not in (c["name"], "*") or not glob_match(h.get("match", "*"), subject):
                    continue
                self.t += h.get("latency_ms", 0)
                if h["action"] == "block":
                    self.emit("hook", agent, self.t, call=cid, hook=h["name"], phase="pre", action="block", result="blocked")
                    outcome = {"cid": cid, "call": c, "args": args, "outcome": "blocked",
                               "text": f"Blocked by hook {h['name']}: {h.get('message', 'not allowed')}"}
                    break
                if h["action"] == "rewrite" and h["find"] in subject:
                    subject = subject.replace(h["find"], h["replace"], 1)
                    args[spec["subject"]] = subject
                    self.emit("hook", agent, self.t, call=cid, hook=h["name"], phase="pre", action="rewrite",
                              result="rewritten", subject=subject)
            prepared.append(outcome)
        t0 = self.t
        cursor = t0
        end = t0
        results = []
        for p in prepared:
            start = t0 if self.policy["parallel"] else cursor
            if p["outcome"] == "run":
                r = self.run_call(agent, p["cid"], {"name": p["call"]["name"], "args": p["args"]}, start)
                for h in self.policy["hooks"]:
                    if (h["phase"] == "post" and h["tool"] in (p["call"]["name"], "*") and h["action"] == "append"):
                        r["dur"] += h.get("latency_ms", 0)
                        r["text"] = r["text"] + "\n" + h["text"]
                        self.emit("hook", agent, start + r["dur"], call=p["cid"], hook=h["name"], phase="post",
                                  action="append", result="appended")
            else:
                r = {"ok": False, "kind": p["outcome"], "text": p["text"], "dur": 0.0}
            r["cid"] = p["cid"]
            r["name"] = p["call"]["name"]
            r["start"] = start
            results.append(r)
            cursor = start + r["dur"]
            if cursor > end:
                end = cursor
        self.t = end
        return results

    # ── sub-agents ───────────────────────────────────────────────────────

    def spawn(self, parent: str, prompt: str) -> str:
        subs = self.scenario.get("subagents", [])
        if self.spawned >= len(subs):
            raise ToolError("no sub-agent is available")
        script = subs[self.spawned]
        self.spawned += 1
        child = f"sub{self.spawned}"
        self.emit("handoff", parent, self.t, direction="spawn", child=child, prompt_tokens=self.tok.count(prompt))
        tools = [n for n in self.scenario["tools"] if n != "task"]
        sub = self.policy["subagents"]
        status, answer, used = self.loop(child, sub["system"], prompt, tools, ScriptedModel(script), sub["max_turns"])
        self.emit("handoff", parent, self.t, direction="return", child=child, status=status,
                  child_tokens=used, summary_tokens=self.tok.count(answer))
        return answer

    # ── the loop ─────────────────────────────────────────────────────────

    def loop(self, agent: str, system: str, task: str, tool_names: list[str], model: Any, max_turns: int) -> tuple[str, str, int]:
        """Run one agent to completion: (status, final answer, tokens it used)."""
        style = self.policy["style"]
        rec = self.policy["recovery"]
        ctx = self.policy["context"]
        msgs = [self.message("user", "task", task, 0)]
        last: dict[str, Any] = {"kind": "none"}
        history: list[str] = []
        used = 0
        n_calls = 0
        for turn in range(1, max_turns + 1):
            msgs = self.compact(agent, system, tool_names, msgs, turn)
            p = self.prompt(system, tool_names, msgs, turn)
            if len(p["ids"]) + self.policy["max_output"] > ctx["window"]:
                self.emit("error", agent, self.t, turn=turn, kind="context_overflow",
                          detail=f"{len(p['ids'])} prompt tokens + {self.policy['max_output']} reserved > window {ctx['window']}",
                          message_tokens=0)
                return "context_overflow", "", used
            text = model.complete({"prompt": p["text"], "input_tokens": len(p["ids"]), "style": style, "last": last, "turn": turn})
            am = self.message("assistant", "assistant", text, turn)
            ev = self.model_call(agent, turn, "act", p, text, len(am["ids"]))
            used += ev["input_tokens"] + ev["output_tokens"]
            msgs.append(am)
            parsed = parse(text, style, tool_names)
            if parsed["kind"] == "final":
                self.checkpoint(agent, turn, msgs, system, tool_names, n_calls)
                return "done", parsed["final"], used
            if parsed["kind"] == "malformed":
                em = self.message("user", "error", f"Error: {parsed['error']}. Please try again.", turn)
                self.emit("error", agent, self.t, turn=turn, kind="malformed", detail=parsed["error"],
                          message_tokens=0 if rec["malformed"] == "stop" else len(em["ids"]))
                if rec["malformed"] == "stop":
                    return "malformed", "", used
                msgs.append(em)
                last = {"kind": "malformed"}
                self.checkpoint(agent, turn, msgs, system, tool_names, n_calls)
                continue
            calls = parsed["calls"]
            looped = False
            if rec["loop_repeats"] > 0:
                for c in calls:
                    history.append(c["name"] + " " + dumps(sort_keys(c["args"])))
                n = rec["loop_repeats"]
                if len(history) >= n and all(h == history[-1] for h in history[-n:]):
                    looped = True
            if looped:
                nm = self.message("user", "nudge",
                                  "You have made the same call several times without progress. Try a different approach.", turn)
                stop = rec["loop_action"] == "stop"
                self.emit("error", agent, self.t, turn=turn, kind="loop",
                          detail=f"the same call {rec['loop_repeats']} times in a row: {history[-1]}",
                          message_tokens=0 if stop else len(nm["ids"]))
                if stop:
                    return "loop", "", used
                history = []
                msgs.append(nm)
                last = {"kind": "nudge"}
                self.checkpoint(agent, turn, msgs, system, tool_names, n_calls)
                continue
            results = self.calls(agent, turn, calls)
            n_calls += len(results)
            for r in results:
                if style == "native":
                    m = self.message("tool", "tool_result", r["text"], turn)
                else:
                    m = self.message("user", "observation", "Observation: " + r["text"], turn)
                msgs.append(m)
                self.emit("tool_result", agent, r["start"], call=r["cid"], name=r["name"], ok=r["ok"], kind=r["kind"],
                          tokens=len(m["ids"]), dur=r["dur"], text=r["text"])
            last = {"kind": "results", "results": [{"ok": r["ok"], "kind": r["kind"]} for r in results]}
            self.checkpoint(agent, turn, msgs, system, tool_names, n_calls)
        return "max_turns", "", used

    def checkpoint(self, agent: str, turn: int, msgs: list[dict[str, Any]], system: str, tool_names: list[str], n_calls: int) -> None:
        self.emit("checkpoint", agent, self.t, turn=turn, context_tokens=self.prompt_tokens(system, tool_names, msgs, turn),
                  messages=len(msgs), tool_calls=n_calls)

    def run(self) -> list[dict[str, Any]]:
        sc = self.scenario
        self.emit("run_start", "main", 0.0, engine=VERSION, scenario=sc["id"], task=sc["task"],
                  backend=getattr(self.model, "name", "scripted"), style=self.policy["style"], seed=self.seed,
                  tools=list(sc["tools"]), policy=self.policy)
        status, answer, _ = self.loop("main", sc["system"], sc["task"], list(sc["tools"]), self.model, self.policy["max_turns"])
        if isinstance(self.model, ReplayModel) and self.model.i != len(self.model.calls):
            raise ValueError(f"replay used {self.model.i} of {len(self.model.calls)} recorded calls")
        self.emit("run_end", "main", self.t, status=status, answer=answer, elapsed=self.t, totals=totals(self.events))
        return self.events


def sort_keys(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: sort_keys(x[k]) for k in sorted(x)}
    if isinstance(x, list):
        return [sort_keys(v) for v in x]
    return x


def totals(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Whole-run totals over every agent's events."""
    out = {"model_calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "cost": 0.0,
           "tool_calls": 0, "failed_calls": 0, "human_prompts": 0, "human_wait": 0.0, "denied": 0, "blocked": 0,
           "retries": 0, "errors": 0, "compactions": 0}
    for e in events:
        ty = e["type"]
        if ty == "model_call":
            out["model_calls"] += 1
            out["input_tokens"] += e["input_tokens"]
            out["cached_tokens"] += e["cached_tokens"]
            out["output_tokens"] += e["output_tokens"]
            out["cost"] += e["cost"]
        elif ty == "tool_result":
            out["tool_calls"] += 1
            if not e["ok"]:
                out["failed_calls"] += 1
            if e["kind"] == "denied":
                out["denied"] += 1
            if e["kind"] == "blocked":
                out["blocked"] += 1
        elif ty == "permission_check" and e["decision"] == "ask":
            out["human_prompts"] += 1
            out["human_wait"] += e["wait"]
        elif ty == "retry":
            out["retries"] += 1
        elif ty == "error":
            out["errors"] += 1
        elif ty == "compaction":
            out["compactions"] += 1
    return out


def run(scenario: dict[str, Any], policy: dict[str, Any] | None = None, model: Any = None,
        seed: int | None = None, tokenizer: Tokenizer | None = None) -> list[dict[str, Any]]:
    """Run a scenario under a policy and return its trace events."""
    return Run(scenario, policy, model, seed, tokenizer).run()


def replay_trace(trace: dict[str, Any], tokenizer: Tokenizer | None = None) -> list[dict[str, Any]]:
    """Replay a recorded trace (``traces/*.json``) through the harness: the same scenario,
    policy and seed, with the recorded completions in place of the model."""
    from .scenarios import scenario

    return Run(scenario(trace["scenario"]), trace["policy"], ReplayModel(trace["calls"]), trace["seed"], tokenizer).run()
