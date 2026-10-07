"""Check the vendored tokenizer against llama.cpp's (offline; needs llama-server running).

    python scripts/tokenizer_check.py [--url http://127.0.0.1:8080]

Tokenizes every prompt that every scenario sends (under several policies) with both, and
writes ``fixtures/tokenizer_llama_check.json`` (counts and whether the ids are equal).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_sim.harness import Run  # noqa: E402
from agent_loop_sim.scenarios import SCENARIOS, scenario  # noqa: E402
from agent_loop_sim.tokenizer import default_tokenizer  # noqa: E402


class Spy:
    def __init__(self, inner):
        self.inner = inner
        self.name = "scripted"
        self.prompts = []

    def complete(self, req):
        self.prompts.append(req["prompt"])
        return self.inner.complete(req)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    a = ap.parse_args()
    from agent_loop_sim.models import ScriptedModel

    tok = default_tokenizer()
    rows = []
    for name in SCENARIOS:
        for style in ("native", "react"):
            sc = scenario(name)
            spy = Spy(ScriptedModel(sc["script"]))
            Run(sc, {"style": style, "cache": {"layout": "timestamp"}}, model=spy).run()
            for p in spy.prompts:
                body = json.dumps({"content": p, "add_special": False, "parse_special": True}).encode()
                r = urllib.request.Request(a.url + "/tokenize", data=body, headers={"Content-Type": "application/json"})
                theirs = json.loads(urllib.request.urlopen(r).read())["tokens"]
                ours = tok.encode(p)
                rows.append({"scenario": name, "style": style, "ours": len(ours), "llama": len(theirs), "equal": ours == theirs})
    out = {"prompts": len(rows), "all_equal": all(r["equal"] for r in rows),
           "tokens": sum(r["ours"] for r in rows), "rows": rows}
    (ROOT / "fixtures" / "tokenizer_llama_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(out["prompts"], "prompts,", out["tokens"], "tokens, all equal:", out["all_equal"])


if __name__ == "__main__":
    main()
