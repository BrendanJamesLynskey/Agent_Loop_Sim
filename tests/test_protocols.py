"""The protocols module (engine 1.2.0): MCP state machines against the official SDK's recorded
exchanges, transports, OAuth 2.1 variants, and the views."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_sim import VERSION
from agent_loop_sim.protocols import mcp, oauth, scenarios, transport, views
from agent_loop_sim.rng import Rng
from agent_loop_sim.tokenizer import default_tokenizer

ROOT = Path(__file__).resolve().parent.parent
SDK = json.loads((ROOT / "fixtures" / "sdk_exchanges.json").read_text(encoding="utf-8"))


def test_version():
    assert VERSION == "1.5.0"


@pytest.mark.parametrize("sc", scenarios.SCENARIOS, ids=[s["name"] for s in scenarios.SCENARIOS])
def test_engine_reproduces_the_sdk_recording(sc):
    """The engine's messages equal the real SDK client/server exchange (IDs renumbered)."""
    assert mcp.normalise(mcp.play(sc)["wire"]) == SDK["scenarios"][sc["name"]]


def test_recording_covers_every_scenario_and_pins_the_sdk():
    assert sorted(SDK["scenarios"]) == sorted(s["name"] for s in scenarios.SCENARIOS)
    assert SDK["sdk"] == "mcp 2.3.0"


def test_normalise_renumbers_ids_and_tokens():
    wire = [
        {"dir": "c2s", "msg": mcp.request(7, "tools/call", {"name": "count", "arguments": {"n": 1},
                                                            "_meta": {"progressToken": 7}})},
        {"dir": "s2c", "msg": mcp.request(99, "elicitation/create", {})},
        {"dir": "c2s", "msg": mcp.response(99, {"action": "accept"})},
        {"dir": "s2c", "msg": mcp.notification("notifications/progress", {"progressToken": 7, "progress": 1})},
        {"dir": "s2c", "msg": mcp.response(7, {})},
    ]
    n = mcp.normalise(wire)
    assert [w["msg"].get("id") for w in n] == ["c1", "s1", "s1", None, "c1"]
    assert n[0]["msg"]["params"]["_meta"]["progressToken"] == "c1"
    assert n[3]["msg"]["params"]["progressToken"] == "c1"


def test_handshake_era_rules():
    p = mcp.play(scenarios.protocol_scenario("legacy_tour"))
    methods = [w["msg"].get("method") for w in p["wire"]]
    assert methods[:3] == ["initialize", None, "notifications/initialized"]
    # Requests carry no _meta envelope in the handshake era.
    for w in p["wire"]:
        assert "_meta" not in (w["msg"].get("params") or {})


def test_modern_era_rules():
    p = mcp.play(scenarios.protocol_scenario("modern_tour"))
    for w in p["wire"]:
        m = w["msg"]
        if w["dir"] == "c2s":
            meta = m["params"]["_meta"]
            assert meta[mcp.META_VERSION] == "2026-07-28" and mcp.META_CLIENT_CAPS in meta
        else:
            assert m["result"]["resultType"] == "complete"
            assert m["result"]["_meta"][mcp.META_SERVER_INFO] == mcp.FIXTURE_SERVER["info"]
    assert "initialize" not in [w["msg"].get("method") for w in p["wire"]]


def test_server_to_client_request_vs_input_required():
    legacy = mcp.play(scenarios.protocol_scenario("legacy_elicit_accept"))["wire"]
    modern = mcp.play(scenarios.protocol_scenario("modern_elicit_accept"))["wire"]
    s2c_requests = [w for w in legacy if w["dir"] == "s2c" and mcp.kind_of(w["msg"]) == "request"]
    assert [w["msg"]["method"] for w in s2c_requests] == ["elicitation/create"]
    assert not [w for w in modern if w["dir"] == "s2c" and mcp.kind_of(w["msg"]) == "request"]
    retry = [w for w in modern if "inputResponses" in (w["msg"].get("params") or {})]
    assert len(retry) == 1 and retry[0]["msg"]["id"] != modern[0]["msg"]["id"]


def test_engine_only_pagination_cancel_and_list_changed():
    p = mcp.play(scenarios.protocol_scenario("legacy_paged"))
    pages = [w["msg"]["result"] for w in p["wire"] if w["dir"] == "s2c" and "tools" in w["msg"].get("result", {})]
    assert [len(r["tools"]) for r in pages] == [2, 2, 2] and "nextCursor" not in pages[-1]
    c = mcp.play(scenarios.protocol_scenario("legacy_cancel"))["wire"]
    assert c[-1]["msg"]["method"] == "notifications/cancelled"
    assert not [w for w in c if w["dir"] == "s2c" and w["msg"].get("id") == c[3]["msg"]["id"]]
    lc = [w["msg"].get("method") for w in mcp.play(scenarios.protocol_scenario("legacy_list_changed"))["wire"]]
    i = lc.index("notifications/tools/list_changed")
    assert lc[i + 1] == "tools/list"


def test_transport_framing():
    p = mcp.play(scenarios.protocol_scenario("legacy_elicit_accept"))
    st = transport.frame_session(p["log"], p["wire"], "stdio")
    assert st["totals"]["overhead"] == st["totals"]["messages"]
    h = transport.frame_session(p["log"], p["wire"], "http")
    assert h["totals"]["payload"] == st["totals"]["payload"]
    assert h["totals"]["posts"] == len([w for w in p["wire"] if w["dir"] == "c2s"])
    init_resp = h["frames"][1]["framing"]
    assert "Mcp-Session-Id: " + transport.SESSION_ID in init_resp
    later = h["frames"][3]["framing"]
    assert "Mcp-Session-Id" in later and "MCP-Protocol-Version: 2025-11-25" in later
    # The elicitation request rides the SSE stream of the tools/call POST, with an event id.
    assert h["frames"][4]["http"]["content_type"] == "text/event-stream" and h["frames"][4]["http"]["event_id"] == 1
    m = mcp.play(scenarios.protocol_scenario("modern_tour"))
    hm = transport.frame_session(m["log"], m["wire"], "http")
    call = hm["frames"][2]["framing"]
    assert "Mcp-Method: tools/call" in call and "Mcp-Name: add" in call and "Mcp-Session-Id" not in call
    for f in hm["frames"]:
        assert f["arrive"] > f["depart"]


def test_stream_drop():
    a = transport.stream_drop("handshake", 5, 2)
    b = transport.stream_drop("modern", 5, 2)
    assert a["totals"]["replayed_events"] == 4 and a["totals"]["redone_steps"] == 0 and a["totals"]["work_steps"] == 5
    assert b["totals"]["redone_steps"] == 2 and b["totals"]["lost_requests"] == 1 and b["totals"]["work_steps"] == 7
    assert b["totals"]["elapsed_ms"] > a["totals"]["elapsed_ms"]
    assert [s["t"] for s in a["steps"]] == sorted(s["t"] for s in a["steps"])
    assert any(s["kind"] == "resume" and "Last-Event-ID: 2" in s["label"] for s in a["steps"])


def test_pkce_rfc7636_vector():
    assert oauth.pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    assert oauth.sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert oauth.pct("https://a.b/c d") == "https%3A%2F%2Fa.b%2Fc%20d"
    v = oauth.random_string(Rng(1), 43)
    assert len(v) == 43 and all(ch in oauth.UNRESERVED for ch in v)


EXPECTED_FAILURE = {
    "no_pkce_support": "PKCE S256 supported",
    "pkce_skipped": "400 invalid_request",
    "wrong_verifier": "400 invalid_grant",
    "issuer_mismatch": "issuer in the metadata equals the issuer asked",
    "iss_mismatch": "iss equals the recorded issuer",
    "wrong_audience": "401 invalid_token",
    "insufficient_scope": "403 insufficient_scope",
    "token_passthrough": "401 invalid_token",
}


@pytest.mark.parametrize("variant", list(oauth.VARIANTS))
def test_oauth_variants_fail_where_the_spec_says(variant):
    r = oauth.run_oauth(variant)
    if variant in ("ok", "dcr"):
        assert r["outcome"]["ok"] and r["steps"][-1]["status"] == 200
        tok = r["token"]["claims"]
        assert tok["aud"] == oauth.RESOURCE and tok["iss"] == oauth.AS
        assert oauth.pkce_challenge(r["pkce"]["verifier"]) == r["pkce"]["challenge"]
        auth = [s for s in r["steps"] if s["label"] == "GET /authorize"][0]["url"]
        assert "code_challenge_method=S256" in auth and "resource=https%3A%2F%2Fmcp.example.com%2Fmcp" in auth
    else:
        assert not r["outcome"]["ok"]
        last = r["steps"][r["outcome"]["failed_at"]]
        assert last["label"] == EXPECTED_FAILURE[variant]
        assert last["ok"] is False


def test_views():
    f = views.integration_frames(3, 4)
    assert f["p2p"] == 12 and f["protocol"] == 7
    assert f["frames"][3]["built"] == 12 and f["frames"][-1]["built"] == 7
    p = mcp.play(scenarios.protocol_scenario("legacy_elicit_accept"))
    seq = views.sequence_frames(p)
    assert [s["to"] for s in seq if s["virtual"]] == ["user", "client"]
    assert len([s for s in seq if not s["virtual"]]) == len(p["wire"])
    n = views.negotiation(mcp.play(scenarios.protocol_scenario("raw_old_version")))
    assert n["agreed"] == "2024-11-05"
    tw = views.three_ways(mcp.play(scenarios.protocol_scenario("legacy_three_ways")), default_tokenizer())
    assert [r["controlled_by"] for r in tw] == ["model", "application", "user"]
    assert tw[0]["text"] == tw[1]["text"] == mcp.FIXTURE_SERVER["texts"]["docs://readme"]
    j = views.journey(mcp.play(scenarios.protocol_scenario("legacy_tour")))
    assert [s["wire"] for s in j].count(True) == 2
