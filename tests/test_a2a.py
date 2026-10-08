"""Engine 1.3.0: A2A against the official A2A SDK's recorded exchanges, the task life cycle and
its errors, the MCP gateway's policies, and the three attacks with and without their defences."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_sim.protocols import a2a, gateway, oauth, security, views
from agent_loop_sim.tokenizer import default_tokenizer

ROOT = Path(__file__).resolve().parent.parent
SDK = json.loads((ROOT / "fixtures" / "a2a_sdk_exchanges.json").read_text(encoding="utf-8"))


def play(name: str) -> dict:
    return a2a.play(a2a.a2a_scenario(name))


@pytest.mark.parametrize("name", list(a2a.A2A_SCENARIOS))
def test_engine_reproduces_the_sdk_recording(name: str) -> None:
    assert a2a.normalise(play(name)["wire"]) == SDK["scenarios"][name]


def test_recording_metadata() -> None:
    assert SDK["sdk"] == "a2a-sdk 1.2.2" and SDK["protocol"] == "1.0"
    assert set(SDK["scenarios"]) == set(a2a.A2A_SCENARIOS)


def test_normalise_renumbers_ids_and_blanks_times() -> None:
    u1, u2 = "12345678-1234-4234-8234-123456789abc", "abcdefab-cdef-4abc-8def-abcdefabcdef"
    w = [{"dir": "s2c", "msg": {"result": {"b": u2, "a": u1, "status": {"timestamp": "x"}}, "m": "Task " + u2}}]
    n = a2a.normalise(w)
    # keys are visited in sorted order: "m" before "result", so u2 is met first
    assert n[0]["msg"]["m"] == "Task <id-1>"
    assert n[0]["msg"]["result"] == {"a": "<id-2>", "b": "<id-1>", "status": {"timestamp": "<time>"}}


def test_uuid4_and_timestamp() -> None:
    from agent_loop_sim.rng import Rng

    u = a2a.uuid4(Rng(1))
    assert a2a.UUID_RE.fullmatch(u) and u[14] == "4" and u[19] in "89ab"
    assert a2a.timestamp(0) == "2026-10-08T10:00:00.000000Z"
    assert a2a.timestamp(61001) == "2026-10-08T10:01:01.001000Z"


def test_card_etag_matches_the_sdk_recipe() -> None:
    import hashlib

    body = json.dumps(a2a.CARD, sort_keys=True, separators=(",", ":"))
    assert a2a.card_etag(a2a.CARD) == 'W/"' + hashlib.sha256(body.encode()).hexdigest() + '"'


def test_streaming_shows_every_state_and_arrives_earlier() -> None:
    s = views.a2a_summary(play("a2a_stream"))
    b = views.a2a_summary(play("a2a_send"))
    assert s["states"] == [a2a.SUBMITTED, a2a.WORKING, a2a.COMPLETED]
    assert b["states"] == [a2a.COMPLETED]  # a blocking call only ever sees the end
    assert s["first_result_ms"] < b["first_result_ms"]


def test_stream_events_and_artifact_chunks() -> None:
    p = play("a2a_stream")
    ev = [w["msg"]["result"] for w in p["wire"] if w["dir"] == "s2c" and "msg" in w]
    assert [list(e)[0] for e in ev] == ["task", "statusUpdate", "artifactUpdate", "artifactUpdate", "statusUpdate"]
    assert "append" not in ev[2]["artifactUpdate"] and ev[3]["artifactUpdate"]["append"] is True
    assert ev[3]["artifactUpdate"]["lastChunk"] is True
    assert all(w.get("sse") for w in p["wire"] if w["dir"] == "s2c" and "msg" in w)


def test_multi_turn_keeps_one_task() -> None:
    p = play("a2a_input")
    tasks = [w["msg"]["result"]["task"] for w in p["wire"] if w["dir"] == "s2c" and "msg" in w]
    assert tasks[0]["status"]["state"] == a2a.INPUT_REQUIRED and tasks[1]["status"]["state"] == a2a.COMPLETED
    assert tasks[0]["id"] == tasks[1]["id"]
    assert [m["role"] for m in tasks[1]["history"]] == ["ROLE_USER", "ROLE_AGENT", "ROLE_USER"]


def test_auth_required_and_the_host_gate() -> None:
    p = play("a2a_auth")
    assert [lg["kind"] for lg in p["log"]].count("gate") == 1
    f = views.a2a_frames(p)
    assert [x["to"] for x in f if x["virtual"]] == ["user", "client"]


def test_cancel_and_terminal_errors() -> None:
    p = play("a2a_cancel")
    errs = [w["msg"]["error"]["code"] for w in p["wire"] if w["dir"] == "s2c" and "error" in w.get("msg", {})]
    assert errs == [-32002, -32004, -32004]
    assert p["log"][-1]["task"] == a2a.CANCELED


def test_error_scenario_codes() -> None:
    p = play("a2a_errors")
    errs = [w["msg"]["error"]["code"] for w in p["wire"] if w["dir"] == "s2c"]
    assert errs == [-32001, -32009, -32601, -32602, -32004]
    assert all(c in a2a.ERRORS.values() or c in (-32601, -32602) for c in errs)


def test_reject_and_direct_message() -> None:
    r = [w for w in play("a2a_reject")["wire"] if w["dir"] == "s2c" and "msg" in w][0]
    assert r["msg"]["result"]["task"]["status"]["state"] == a2a.REJECTED
    m = [w for w in play("a2a_message")["wire"] if w["dir"] == "s2c" and "msg" in w][0]
    assert "message" in m["msg"]["result"] and m["msg"]["result"]["message"]["role"] == "ROLE_AGENT"


def test_transitions_cover_every_state_streamed() -> None:
    # a stream shows every state change; a blocking call only the state it ended in
    edges = {tuple(e) for e in a2a.TRANSITIONS}
    for name in ["a2a_stream"]:
        p = play(name)
        prev = None
        for lg in p["log"]:
            if lg["task"] is not None and lg["task"] != prev:
                if prev is not None:
                    assert (prev, lg["task"]) in edges, (name, prev, lg["task"])
                prev = lg["task"]


@pytest.fixture(scope="module")
def tok():
    return default_tokenizer()


def test_gateway_policies(tok) -> None:
    flat = gateway.gateway_run("flat", tok)
    pre = gateway.gateway_run("prefix", tok)
    fil = gateway.gateway_run("filtered", tok)
    assert flat["merge"]["collisions"] == [["read_file", ["files", "github"]], ["search", ["files", "github", "web"]]]
    assert flat["merged_tools"] == 7 and pre["merged_tools"] == 10 and fil["merged_tools"] == 3
    assert flat["routed_to"] == "files" and not flat["routed_right"]
    assert pre["routed_right"] and fil["routed_right"]
    assert fil["merged_tokens"] < flat["merged_tokens"] < pre["merged_tokens"]
    assert pre["direct_tokens"] == sum(p["tokens"] for p in pre["per_server"])
    assert fil["exposed"] == ["files.read_file", "web.search", "web.fetch"]


@pytest.mark.parametrize("attack", list(security.ATTACKS))
def test_each_attack_succeeds_undefended_and_is_stopped_defended(attack: str) -> None:
    bad = security.run_security(attack, False)
    good = security.run_security(attack, True)
    assert bad["outcome"]["harm"] and bad["steps"][-1]["kind"] == "attack"
    assert not good["outcome"]["harm"]
    failed = [s for s in good["steps"] if s["kind"] == "check" and not s["ok"]]
    assert len(failed) == 1
    assert "attack" not in [s["kind"] for s in good["steps"][good["steps"].index(failed[0]):]]


def test_tool_pin_detects_the_change() -> None:
    assert security.tool_hash(security.NOTES_TOOL) != security.tool_hash(
        dict(security.NOTES_TOOL, description=security.HIDDEN))


def test_flow_frames() -> None:
    f = views.flow_frames(oauth.run_oauth("wrong_audience"))
    assert f[-1]["kind"] == "error" and any(x["kind"] == "fail" for x in f)
    assert [x["step"] for x in f] == list(range(len(f)))
    ok = views.flow_frames(oauth.run_oauth("ok"))
    assert not any(x["kind"] in ("fail", "error") for x in ok[2:])
