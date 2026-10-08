"""Record real A2A exchanges with an agent written with the official A2A Python SDK.

    python scripts/record_a2a_sdk.py            # write fixtures/a2a_sdk_exchanges.json
    python scripts/record_a2a_sdk.py --check    # record again; fail if anything differs from the
                                                # committed recording or from the engine (CI)

Each scenario in ``agent_loop_sim.protocols.a2a.A2A_SCENARIOS`` is played by the engine's
orchestrator (the A2A client) against ``conformance/a2a_sdk_server.py``, the SDK's JSON-RPC
binding served in-process by Starlette's test client (no socket). Every request, response and
SSE event is recorded and normalised (UUIDs renumbered in order of appearance, timestamps
blanked); the engine's own agent must produce the same messages: ``tests/test_protocols.py``
and the TS port both check it. Needs ``pip install -e ".[conformance]"`` (the pinned SDK).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "conformance"))

from agent_loop_sim.protocols.a2a import A2A_SCENARIOS, CARD_PATH, a2a_scenario, normalise, play  # noqa: E402

OUT = ROOT / "fixtures" / "a2a_sdk_exchanges.json"


def sdk_send():  # noqa: ANN201
    """A transport to a fresh SDK agent (one per scenario, like the engine's)."""
    warnings.simplefilter("ignore")
    import logging

    logging.disable(logging.CRITICAL)
    from starlette.testclient import TestClient

    import a2a_sdk_server as S

    client = TestClient(S.app(), base_url=S.AGENT)

    def send(kind: str, req: Any, headers: list[list[str]], at: int) -> Any:
        if kind == "card":
            r = client.get(CARD_PATH)
            return {"status": r.status_code,
                    "headers": [["Content-Type", r.headers["content-type"]], ["ETag", r.headers["etag"]]],
                    "body": r.json(), "_t": at + 1}
        hs = {k: v for k, v in headers}
        with client.stream("POST", S.RPC_PATH, json=req, headers=hs) as r:
            sse = r.headers["content-type"].startswith("text/event-stream")
            if sse:
                msgs = [json.loads(line[len("data: "):]) for line in r.iter_lines() if line.startswith("data: ")]
            else:
                r.read()
                msgs = [r.json()]
        return {"messages": [dict(m, _t=at + 1 + i) for i, m in enumerate(msgs)], "sse": sse}

    return send


def record() -> dict[str, Any]:
    out: dict[str, Any] = {"sdk": "a2a-sdk " + version("a2a-sdk"), "protocol": "1.0", "scenarios": {}}
    for name in A2A_SCENARIOS:
        played = play(a2a_scenario(name), sdk_send())
        out["scenarios"][name] = normalise(played["wire"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    live = record()
    if a.check:
        committed = json.loads(OUT.read_text(encoding="utf-8"))
        bad = []
        for name in A2A_SCENARIOS:
            engine = normalise(play(a2a_scenario(name))["wire"])
            if live["scenarios"][name] != committed["scenarios"].get(name):
                bad.append(name + ": live SDK != committed recording")
            if engine != live["scenarios"][name]:
                bad.append(name + ": engine != live SDK")
        if live["sdk"] != committed["sdk"]:
            bad.append("SDK version " + live["sdk"] + " != committed " + committed["sdk"])
        if bad:
            sys.exit("\n".join(bad))
        n = sum(len(v) for v in live["scenarios"].values())
        print(f"{len(A2A_SCENARIOS)} A2A scenarios ({n} messages): live {live['sdk']} == committed recording == engine")
    else:
        OUT.write_text(json.dumps(live, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
