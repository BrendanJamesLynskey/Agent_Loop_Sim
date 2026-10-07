"""The Python reference: tokenizer, tools, parsing, accounting and every harness policy."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from agent_loop_sim import views
from agent_loop_sim.accounting import LATENCY, PRICES, PromptCache, call_cost, call_latency, lcp
from agent_loop_sim.chat import render, system_segment
from agent_loop_sim.harness import Run, decide, replay_trace, run
from agent_loop_sim.jsonfmt import dumps
from agent_loop_sim.models import ReplayMismatch, ReplayModel, ScriptedModel, fnv1a32
from agent_loop_sim.parse import parse
from agent_loop_sim.rng import Rng
from agent_loop_sim.scenarios import SCENARIOS, scenario
from agent_loop_sim.tokenizer import ADDED_BASE, default_tokenizer
from agent_loop_sim.tools import TOOL_SPECS, World, glob_match, run_pytest

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schema/trace.schema.json").read_text())
TRACES = sorted((ROOT / "traces").glob("*.json"))


def validate(events):
    for e in events:
        jsonschema.validate(e, SCHEMA)


# ── tokenizer ────────────────────────────────────────────────────────────


def test_tokenizer_matches_llama_cpp_on_every_scenario_prompt():
    chk = json.loads((ROOT / "fixtures/tokenizer_llama_check.json").read_text())
    assert chk["all_equal"] and chk["prompts"] >= 100 and chk["tokens"] > 50000


def test_tokenizer_matched_llama_cpp_on_every_recorded_prompt():
    for p in TRACES:
        t = json.loads(p.read_text())
        assert all(c["prompt_ids_equal"] for c in t["tokenizer_check"])
        assert all(c["prompt_tokens_ours"] == c["prompt_tokens_llama"] for c in t["tokenizer_check"])


def test_tokenizer_basics():
    tok = default_tokenizer()
    assert tok.n_merges == 151387
    assert tok.encode("<|im_start|>") == [ADDED_BASE + 1]
    assert tok.encode("<tool_call>") == [151657]
    assert tok.encode("") == []
    # NFC: a decomposed é tokenizes like the composed one
    assert tok.encode("Café") == tok.encode("Café")
    # the recorded first completion's ids are what our tokenizer gives for its text
    t = json.loads(TRACES[0].read_text())
    c = t["calls"][0]
    assert tok.encode(c["text"]) == c["completion_ids"][: len(tok.encode(c["text"]))]


def test_prompt_segments_add_up():
    """Every segment starts with a special token, so per-message counts sum to the total."""
    tok = default_tokenizer()
    for name in ["fix_test", "research"]:
        for e in run(scenario(name)):
            if e["type"] == "model_call" and e["purpose"] == "act":
                assert sum(c["tokens"] for c in e["context"]) == e["input_tokens"]


# ── rng, json, glob, tools ───────────────────────────────────────────────


def test_rng_is_mulberry32():
    r = Rng(0)
    # mulberry32(0) as Disaggregated_Inference_Sim's JavaScript function computes it (node, 2026-10-07)
    assert [r.next_u32() for _ in range(3)] == [1144304738, 1416247, 958946056]
    a, b = Rng(42), Rng(42)
    xs = [a.random() for _ in range(1000)]
    assert xs == [b.random() for _ in range(1000)]
    assert all(0 <= x < 1 for x in xs)
    assert 0.45 < sum(xs) / len(xs) < 0.55
    assert all(1 <= Rng(s).randint(1, 6) <= 6 for s in range(50))


def test_dumps_is_template_tojson_and_refuses_floats():
    assert dumps({"a": [1, True, None], "b": "é"}) == '{"a": [1, true, null], "b": "é"}'
    with pytest.raises(ValueError):
        dumps({"x": 1.0})


@pytest.mark.parametrize("p,s,want", [("rm *", "rm -rf build", True), ("rm *", "git rm x", False), ("*", "", True),
                                      ("a?c", "abc", True), ("a?c", "ac", False), ("*.py", "x/y.py", True)])
def test_glob(p, s, want):
    assert glob_match(p, s) is want


def test_pytest_simulator():
    files = scenario("fix_test")["files"]
    code, out = run_pytest(files)
    assert code == 1 and "1 failed, 2 passed" in out and "assert -1 == 5" in out
    fixed = dict(files, **{"calc.py": files["calc.py"].replace("return a - b", "return a + b", 1)})
    code, out = run_pytest(fixed)
    assert code == 0 and "3 passed" in out


def test_world_shell_and_files():
    w = World(scenario("fix_test")["files"], [])
    assert w.shell("ls").startswith("exit code: 0\nREADME.md")
    assert "command not found" in w.shell("make")
    w.shell("rm -rf build")
    assert "build/cache.txt" not in w.files
    assert "deleted: build/cache.txt" in w.shell("git status")


# ── parsing ──────────────────────────────────────────────────────────────


def test_parse_native_and_react():
    names = list(TOOL_SPECS)
    ok = parse('<tool_call>\n{"name": "read_file", "arguments": {"path": "a"}}\n</tool_call>', "native", names)
    assert ok == {"kind": "calls", "calls": [{"name": "read_file", "args": {"path": "a"}}]}
    assert parse("all done", "native", names)["kind"] == "final"
    assert parse('<tool_call>\n{"name": "read_file"', "native", names)["kind"] == "malformed"
    bad = parse('<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>', "native", names)
    assert bad["kind"] == "malformed" and "unknown tool" in bad["error"]
    r = parse('Thought: x\nAction: calculator\nAction Input: {"expression": "1+1"}\nObservation: 2', "react", names)
    assert r["calls"] == [{"name": "calculator", "args": {"expression": "1+1"}}]
    assert parse("Final Answer: 2", "react", names) == {"kind": "final", "final": "2"}
    fenced = parse('Action: calculator\nAction Input: ```json\n{"expression": "1"}\n```', "react", names)
    assert fenced["kind"] == "malformed"


# ── accounting ───────────────────────────────────────────────────────────


def test_cost_and_latency_formulas():
    p = PRICES["claude-sonnet-4.6"]
    # 2,000 input of which 1,500 cached and the rest written; 100 output
    assert call_cost(p, 2000, 1500, True, 100) == pytest.approx((1500 * 0.3 + 500 * 3.75 + 100 * 15) / 1e6)
    assert call_cost(p, 2000, 0, False, 100) == pytest.approx((2000 * 3 + 100 * 15) / 1e6)
    ttft, dur = call_latency(LATENCY["hosted"], 5400, 400, 80)
    assert ttft == pytest.approx(400 + 5000 / 5000 * 1000) and dur == pytest.approx(ttft + 1000)


def test_prompt_cache_prefix_block_min_and_ttl():
    c = PromptCache(True, min_tokens=4, block=2, ttl_ms=1000)
    assert c.lookup([1, 2, 3, 4, 5], 0) == 0
    assert c.store([1, 2, 3, 4, 5], 0)
    assert c.lookup([1, 2, 3, 4, 5, 6], 10) == 4  # 5 rounded down to the block
    assert c.lookup([1, 2, 3, 9], 20) == 0  # 3 < min
    assert c.lookup([1, 2, 3, 4, 5], 2000) == 0  # expired
    assert lcp([1, 2], [1, 2, 3]) == 2


def test_measured_local_profile_is_the_traces_median():
    pf, dc = [], []
    for p in TRACES:
        for c in json.loads(p.read_text())["calls"]:
            ll = c["llama"]
            if ll["prompt_n"] >= 100:
                pf.append(ll["prompt_n"] * 1000 / ll["prompt_ms"])
            dc.append(ll["predicted_n"] * 1000 / ll["predicted_ms"])

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    assert round(median(pf), 1) == LATENCY["local-i7"]["prefill_tps"]
    assert round(median(dc), 1) == LATENCY["local-i7"]["decode_tps"]


# ── the harness ──────────────────────────────────────────────────────────


def test_every_scenario_runs_and_validates():
    for name in SCENARIOS:
        ev = run(scenario(name))
        validate(ev)
        assert ev[0]["type"] == "run_start" and ev[-1]["type"] == "run_end"
        assert [e["seq"] for e in ev] == list(range(len(ev)))


def test_fix_test_really_fixes_the_code():
    r = Run(scenario("fix_test"))
    ev = r.run()
    assert ev[-1]["status"] == "done"
    assert "return a + b" in r.world.files["calc.py"]
    results = [e for e in ev if e["type"] == "tool_result" and e["name"] == "run_shell"]
    assert results[0]["text"].startswith("exit code: 1") and results[-1]["text"].startswith("exit code: 0")


def test_react_and_native_see_different_prompts_same_work():
    a, b = run(scenario("fix_test")), run(scenario("fix_test_react"))
    assert a[-1]["totals"]["tool_calls"] == b[-1]["totals"]["tool_calls"] == 5
    # both pay for the tool definitions: JSON schemas natively, prose plus the format in ReAct
    assert a[1]["context"][1]["kind"] == b[1]["context"][1]["kind"] == "tools"
    assert b[1]["context"][1]["tokens"] < a[1]["context"][1]["tokens"]


def test_malformed_call_is_fed_back_and_recovered():
    ev = run(scenario("fix_test_malformed"))
    errs = [e for e in ev if e["type"] == "error"]
    assert len(errs) == 1 and errs[0]["kind"] == "malformed" and ev[-1]["status"] == "done"
    stop = run(scenario("fix_test_malformed"), {"recovery": {"malformed": "stop"}})
    assert stop[-1]["status"] == "malformed"


def test_permission_rules_precedence_and_modes():
    shell, read = TOOL_SPECS["run_shell"], TOOL_SPECS["read_file"]
    deny = {"tool": "run_shell", "pattern": "rm *", "decision": "deny"}
    allow_all = {"tool": "*", "pattern": "*", "decision": "allow"}
    base = {"mode": "default", "rules": [allow_all, deny], "human": {}}
    assert decide(base, shell, "rm -rf build")[0] == "deny"  # deny beats allow, whatever the order
    assert decide(base, shell, "pytest")[0] == "allow"
    assert decide(dict(base, rules=[]), shell, "pytest") == ("ask", "mode: default (writes ask)")
    assert decide(dict(base, rules=[]), read, "calc.py")[0] == "allow"
    assert decide(dict(base, mode="read_only"), shell, "pytest")[0] == "deny"
    assert decide(dict(base, mode="ask"), read, "calc.py")[0] == "ask"
    assert decide(dict(base, mode="auto", rules=[]), shell, "rm -rf /")[0] == "allow"


def test_permission_modes_trade_time_for_safety():
    def go(perm):
        return run(scenario("fix_test_cleanup"), {"permissions": perm})[-1]

    human = {"latency_ms": [3000, 9000], "deny": ["rm *"]}
    auto = go({"mode": "auto"})
    ask = go({"mode": "ask", "human": human})
    ro = go({"mode": "read_only"})
    assert auto["totals"]["human_prompts"] == 0 and ask["totals"]["human_prompts"] == 7
    assert ask["elapsed"] > auto["elapsed"] + 7 * 3000
    assert ask["totals"]["denied"] == 1  # the human refused rm -rf build
    assert ro["totals"]["denied"] == 5  # every write and shell call


def test_hooks_block_rewrite_append():
    hooks = [
        {"name": "no-rm", "phase": "pre", "tool": "run_shell", "match": "rm *", "action": "block", "message": "no"},
        {"name": "q", "phase": "pre", "tool": "run_shell", "match": "pytest*", "action": "rewrite", "find": "pytest", "replace": "pytest -q"},
        {"name": "lint", "phase": "post", "tool": "edit_file", "action": "append", "text": "(lint: ok)"},
    ]
    ev = run(scenario("fix_test_cleanup"), {"hooks": hooks})
    hk = [e for e in ev if e["type"] == "hook"]
    assert {e["result"] for e in hk} == {"blocked", "rewritten", "appended"}
    assert any(e["type"] == "tool_result" and e["kind"] == "blocked" for e in ev)
    assert any(e["type"] == "tool_result" and e["text"].endswith("(lint: ok)") for e in ev)


def test_retries_back_off_exponentially():
    ev = run(scenario("fix_test_flaky"))
    rs = [e for e in ev if e["type"] == "retry"]
    assert rs and all(e["backoff"] == 500 * 2 ** (e["attempt"] - 1) for e in rs)


def test_loop_detection_stops_or_nudges():
    sc = scenario("fix_test")
    sc["fail_rates"] = {"run_shell": 1}
    stop = run(sc, {"recovery": {"retries": 0, "loop_repeats": 3, "loop_action": "stop"}})
    assert stop[-1]["status"] == "loop"
    sc = scenario("fix_test")
    sc["fail_rates"] = {"run_shell": 1}
    nudge = run(sc, {"max_turns": 8, "recovery": {"retries": 0, "loop_repeats": 3, "loop_action": "nudge"}})
    assert any(e["type"] == "error" and e["kind"] == "loop" for e in nudge) and nudge[-1]["status"] == "max_turns"


def test_context_strategies_keep_tokens_and_lose_facts():
    out = {}
    for s in ["none", "truncate", "clip", "summarise"]:
        out[s] = run(scenario("research"), {"context": {"window": 3000, "strategy": s}})
    assert out["none"][-1]["status"] == "context_overflow"
    lost = {s: [f for e in ev if e["type"] == "compaction" for f in e["facts_lost"]] for s, ev in out.items()}
    assert set(lost["truncate"]) >= {"mem", "bw", "params", "kv"}
    assert "mem" not in lost["clip"] and "bw" in lost["clip"]  # the first lines survive clipping
    assert lost["summarise"] == []  # the summariser kept every important fact
    for s in ["truncate", "clip", "summarise"]:
        assert out[s][-1]["status"] == "done"
        for e in out[s]:
            if e["type"] == "model_call":
                assert e["input_tokens"] + 512 <= 3000


def test_prompt_cache_layouts():
    def hits(layout, price="claude-sonnet-4.6"):
        ev = run(scenario("research"), {"cache": {"layout": layout, "price": price}})
        return ev[-1]["totals"]

    stable, ts, reorder = hits("stable"), hits("timestamp"), hits("reorder_tools")
    assert stable["cached_tokens"] > reorder["cached_tokens"] > ts["cached_tokens"] == 0
    assert stable["cost"] < reorder["cost"] < ts["cost"]
    assert hits("stable", "claude-haiku-4.5")["cached_tokens"] == 0  # prompts below its 4,096 minimum
    g = hits("stable", "gpt-5-mini")["cached_tokens"]
    assert g > 0 and g % 128 == 0  # cached tokens come in 128-token blocks


def test_parallel_calls_overlap_in_time():
    ev = run(scenario("research_subagent"))
    seq = run(scenario("research_subagent"), {"parallel": False})
    assert ev[-1]["elapsed"] == seq[-1]["elapsed"]  # one call per turn: no difference
    sc = scenario("fix_test")
    sc["script"] = [{"calls": [{"name": "read_file", "args": {"path": "calc.py"}},
                               {"name": "run_shell", "args": {"command": "pytest"}}]}, {"final": "ok"}]
    par = run(sc)
    sc2 = scenario("fix_test")
    sc2["script"] = sc["script"]
    ser = run(sc2, {"parallel": False})
    assert ser[-1]["elapsed"] - par[-1]["elapsed"] == pytest.approx(TOOL_SPECS["read_file"]["latency_ms"])


def test_subagent_keeps_the_parent_context_small():
    direct = run(scenario("research"))
    sub = run(scenario("research_subagent"))
    peak = lambda ev: max(e["input_tokens"] for e in ev if e["type"] == "model_call" and e["agent"] == "main")  # noqa: E731
    assert peak(sub) < peak(direct)
    hs = [e for e in sub if e["type"] == "handoff"]
    assert [h["direction"] for h in hs] == ["spawn", "return"] and hs[1]["child_tokens"] > hs[1]["summary_tokens"]
    assert any(e["agent"] == "sub1" for e in sub)


# ── replay ───────────────────────────────────────────────────────────────


def test_recorded_traces_replay_exactly():
    assert len(TRACES) >= 3
    for p in TRACES:
        t = json.loads(p.read_text())
        ev = replay_trace(t)
        validate(ev)
        assert ev[0]["backend"] == "replay"
        assert ev[-1]["status"] == t["outcome"]["status"]
        calls = [e for e in ev if e["type"] == "model_call"]
        assert [c["text"] for c in calls] == [c["text"] for c in t["calls"]]
        prov = t["provenance"]
        assert prov["quantisation"] == "Q4_K_M" and len(prov["llama_cpp_commit"]) == 40 and len(prov["gguf_sha256"]) == 64


def test_replay_detects_a_changed_prompt():
    t = json.loads(TRACES[0].read_text())
    sc = scenario(t["scenario"])
    sc["system"] += " Be brief."
    with pytest.raises(ReplayMismatch):
        Run(sc, t["policy"], ReplayModel(t["calls"]), t["seed"]).run()


def test_fnv1a32_known_value():
    assert fnv1a32("") == 0x811C9DC5 and fnv1a32("a") == 0xE40C292C


# ── views ────────────────────────────────────────────────────────────────


def test_loop_frames_track_the_context():
    ev = run(scenario("fix_test"))
    fr = views.loop_frames(ev)
    calls = [e["input_tokens"] for e in ev if e["type"] == "model_call"]
    assert [f["total"] for f in fr if f["phase"] == "call"] == calls
    res = [f["total"] for f in fr if f["phase"] == "result"]
    assert [r + 3 for r in res] == calls[1:]  # + the next generation prompt
    assert fr[-1]["phase"] == "done"


def test_cache_frames_and_budget_frames():
    ev = run(scenario("research"))
    cf = views.cache_frames(ev)
    assert cf[-1]["cum"] == pytest.approx(ev[-1]["totals"]["cost"]) and cf[-1]["cum_plain"] > cf[-1]["cum"]
    bf = views.budget_frames(run(scenario("research"), {"context": {"window": 3000, "strategy": "clip"}}))
    assert any(f["phase"] == "compact" for f in bf) and all(f["window"] == 3000 for f in bf)


def test_scripted_model_renders_both_styles():
    m = ScriptedModel([{"calls": [{"name": "read_file", "args": {"path": "x"}}]}])
    assert "<tool_call>" in m.complete({"style": "native", "last": {"kind": "none"}})
    m = ScriptedModel([{"calls": [{"name": "read_file", "args": {"path": "x"}}]}])
    assert "Action: read_file" in m.complete({"style": "react", "last": {"kind": "none"}})


def test_system_segment_and_render():
    tools = [TOOL_SPECS["read_file"]]
    seg, bare = system_segment("S", tools, "native")
    assert seg.startswith("<|im_start|>system\nS\n\n# Tools") and bare == "<|im_start|>system\nS<|im_end|>\n"
    assert render("S", tools, "react", []).endswith("<|im_start|>assistant\n")
