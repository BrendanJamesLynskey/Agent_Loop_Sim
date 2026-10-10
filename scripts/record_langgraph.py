"""Run every orchestration conformance session in real LangGraph and compare it with the engine.

    python scripts/record_langgraph.py          # record: write fixtures/langgraph_recordings.json
    python scripts/record_langgraph.py --check  # CI: live LangGraph == committed recording == engine

For each operation of each session it compares the outcome (status, interrupt payloads, error
type, node runs, effects) and the whole checkpoint history after it: every checkpoint's step,
source, parent, values, next tasks and each task's result, error and interrupts. LangGraph is
pinned in pyproject.toml's ``conformance`` extra."""
from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "conformance"))

from agent_loop_sim.orchestration.graph import comparable_history, outcome, run_session  # noqa: E402
from agent_loop_sim.orchestration.scenarios import SESSIONS, graph  # noqa: E402

OUT = ROOT / "fixtures" / "langgraph_recordings.json"


def engine_sessions() -> list[dict]:
    out = []
    for s in SESSIONS:
        g = graph(s["graph"])
        r = run_session(g, s["ops"])
        out.append({"name": s["name"], "graph": s["graph"],
                    "ops": [{"op": o["op"], "outcome": outcome(o["result"]),
                             "history": comparable_history(g, o["history"])} for o in r["ops"]]})
    return out


def live_sessions() -> list[dict]:
    import langgraph_graphs as LG
    out = []
    for s in SESSIONS:
        g = graph(s["graph"])
        r = LG.run_session(g, s["ops"])
        out.append({"name": s["name"], "graph": s["graph"],
                    "ops": [dict(o, history=comparable_history(g, o["history"])) for o in r["ops"]]})
    return out


def diff(a: list[dict], b: list[dict], la: str, lb: str) -> list[str]:
    bad = []
    for x, y in zip(a, b):
        for i, (ox, oy) in enumerate(zip(x["ops"], y["ops"])):
            for part in ["outcome", "history"]:
                if json.dumps(ox[part], sort_keys=True) != json.dumps(oy[part], sort_keys=True):
                    bad.append(f"{x['name']} op {i} {part}: {la} {json.dumps(ox[part], sort_keys=True)}\n   {lb} {json.dumps(oy[part], sort_keys=True)}")
    if len(a) != len(b):
        bad.append("session count differs")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--engine-only", action="store_true", help="compare the committed recording with the engine (no LangGraph)")
    a = ap.parse_args()
    eng = engine_sessions()
    if a.engine_only:
        rec = json.loads(OUT.read_text())
        bad = diff(rec["sessions"], eng, "recorded", "engine")
        if bad:
            sys.exit("\n".join(bad))
        print(f"{len(eng)} sessions: committed LangGraph recording == engine")
        return
    live = live_sessions()
    doc = {"langgraph": version("langgraph"), "langgraph_checkpoint": version("langgraph-checkpoint"), "sessions": live}
    bad = diff(live, eng, "langgraph", "engine")
    if a.check:
        rec = json.loads(OUT.read_text())
        if rec["langgraph"] != doc["langgraph"]:
            bad.append(f"recorded with langgraph {rec['langgraph']}, running {doc['langgraph']}")
        bad += diff(rec["sessions"], live, "recorded", "live")
    else:
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    if bad:
        sys.exit(f"{len(bad)} differences:\n" + "\n".join(bad))
    n = sum(len(s["ops"]) for s in live)
    cps = sum(len(s["ops"][-1]["history"]) for s in live)
    print(f"{len(live)} sessions, {n} operations, {cps} checkpoints: live langgraph {doc['langgraph']} == "
          + ("committed recording == " if a.check else "") + "engine")


if __name__ == "__main__":
    main()
