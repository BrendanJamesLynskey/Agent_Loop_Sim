"""Record real JSON-RPC exchanges between the official MCP Python SDK's client and server.

    python scripts/record_sdk.py            # write fixtures/sdk_exchanges.json
    python scripts/record_sdk.py --check    # record again; fail if anything differs from the
                                            # committed recording or from the engine (CI)

Each scenario in ``agent_loop_sim.protocols.scenarios`` is played by the SDK's ``Client``
(or, for ``raw`` scenarios, by writing hand-made lines) against ``conformance/sdk_server.py``
run as a stdio subprocess behind a small tee that logs every line in both directions. The
recording is normalised (request IDs renumbered in order of appearance) and the engine must
produce the same messages: ``tests/test_protocols.py`` and the TS port both check it.
Needs ``pip install -e ".[conformance]"`` (the pinned SDK).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import warnings
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_sim.protocols.mcp import normalise, play  # noqa: E402
from agent_loop_sim.protocols.scenarios import SCENARIOS  # noqa: E402

SERVER = ROOT / "conformance" / "sdk_server.py"
OUT = ROOT / "fixtures" / "sdk_exchanges.json"


def tee(log_path: str, cmd: list[str]) -> None:
    """Run cmd, pass stdin/stdout through, and log each line with its direction."""
    log = open(log_path, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    lock = threading.Lock()

    def rec(d: str, line: bytes) -> None:
        with lock:
            log.write(json.dumps({"dir": d, "line": line.decode("utf-8")}) + "\n")
            log.flush()

    def up() -> None:
        for line in sys.stdin.buffer:
            rec("c2s", line)
            p.stdin.write(line)  # type: ignore[union-attr]
            p.stdin.flush()  # type: ignore[union-attr]
        p.stdin.close()  # type: ignore[union-attr]

    threading.Thread(target=up, daemon=True).start()
    for line in p.stdout:  # type: ignore[union-attr]
        rec("s2c", line)
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
    p.wait()


async def play_sdk(sc: dict, log: str) -> None:
    import mcp_types as types
    from mcp import Client, StdioServerParameters

    answers = {k: list(v) for k, v in sc["answers"].items()}

    async def elicit(_ctx, _params):  # noqa: ANN001
        return types.ElicitResult.model_validate(answers["elicit"].pop(0))

    async def sample(_ctx, _params):  # noqa: ANN001
        return types.CreateMessageResult.model_validate(answers["sample"].pop(0))

    async def progress(_p, _t, _m):  # noqa: ANN001
        return None

    cl = sc["client"]
    params = StdioServerParameters(command=sys.executable, args=[__file__, "--tee", log, sys.executable, str(SERVER)])
    kw: dict = {"mode": sc["mode"], "cache": None,
                "client_info": types.Implementation(name=cl["info"]["name"], version=cl["info"]["version"])}
    if cl["elicitation"]:
        kw["elicitation_callback"] = elicit
    if cl["sampling"]:
        kw["sampling_callback"] = sample
    async with Client(params, **kw) as c:
        for st in sc["steps"]:
            op = st["op"]
            if op == "ping":
                await c.send_ping()
            elif op == "list_tools":
                await c.list_tools()
            elif op == "list_resources":
                await c.list_resources()
            elif op == "list_prompts":
                await c.list_prompts()
            elif op == "call_tool":
                try:
                    await c.call_tool(st["name"], st["arguments"], progress_callback=progress if st.get("progress") else None)
                except Exception:  # noqa: BLE001 - a protocol error is part of the recording
                    pass
            elif op == "read_resource":
                try:
                    await c.read_resource(st["uri"])
                except Exception:  # noqa: BLE001
                    pass
            elif op == "get_prompt":
                await c.get_prompt(st["name"], st["arguments"])
            else:
                raise ValueError(op)


def play_raw(sc: dict, log: str) -> None:
    p = subprocess.Popen([sys.executable, __file__, "--tee", log, sys.executable, str(SERVER)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for st in sc["steps"]:
        line = st["line"] if "line" in st else json.dumps(st["msg"], separators=(",", ":"))
        p.stdin.write(line.encode("utf-8") + b"\n")  # type: ignore[union-attr]
        p.stdin.flush()  # type: ignore[union-attr]
        if st.get("reply", True):
            p.stdout.readline()  # type: ignore[union-attr]  # the step's one reply
    p.stdin.close()  # type: ignore[union-attr]
    p.wait()


def record(sc: dict) -> list[dict]:
    with TemporaryDirectory() as d:
        log = str(Path(d) / "wire.jsonl")
        if sc["mode"] == "raw":
            play_raw(sc, log)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                asyncio.run(play_sdk(sc, log))
        lines = [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for r in lines:
        text = r["line"].rstrip("\n")
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            msg = None
        out.append({"dir": r["dir"], "msg": msg} if msg is not None else {"dir": r["dir"], "raw": text})
    return out


def build() -> dict:
    return {
        "sdk": f"mcp {version('mcp')}",
        "server": "conformance/sdk_server.py",
        "note": "Recorded over stdio by scripts/record_sdk.py; request IDs renumbered in order of appearance.",
        "scenarios": {sc["name"]: normalise(record(sc)) for sc in SCENARIOS},
    }


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--tee":
        tee(sys.argv[2], sys.argv[3:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fx = build()
    text = json.dumps(fx, ensure_ascii=False, indent=1) + "\n"
    if not a.check:
        OUT.write_text(text, encoding="utf-8")
        print(f"recorded {len(fx['scenarios'])} scenarios with {fx['sdk']}")
        return
    bad = []
    committed = json.loads(OUT.read_text(encoding="utf-8"))
    for name, msgs in fx["scenarios"].items():
        if committed["scenarios"].get(name) != msgs:
            bad.append(f"{name}: live SDK recording differs from the committed fixture")
        sc = next(s for s in SCENARIOS if s["name"] == name)
        if normalise(play(sc)["wire"]) != msgs:
            bad.append(f"{name}: the engine differs from the live SDK recording")
    if committed.get("sdk") != fx["sdk"]:
        bad.append(f"SDK version {fx['sdk']} != committed {committed.get('sdk')}")
    if bad:
        sys.exit("\n".join(bad))
    print(f"{len(fx['scenarios'])} scenarios: live {fx['sdk']} == committed recording == engine")


if __name__ == "__main__":
    main()
