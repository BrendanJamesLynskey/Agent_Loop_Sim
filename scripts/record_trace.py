"""Record a real trace: run a scenario with the local model behind llama-server, offline.

    python scripts/record_trace.py <scenario> <trace-id> [--style native|react] [--url http://127.0.0.1:8080]

Writes ``traces/<trace-id>.json``: the model's completions with each prompt's fingerprint and
token count, llama.cpp's own counts and timings, and the provenance of the recording (model
file and its SHA-256, quantisation, seed, sampling, llama.cpp commit, CPU). It also checks
the vendored tokenizer against llama.cpp's on every prompt (``/tokenize``) and records the
result. The site and CI only ever *replay* these files; nothing calls a model there.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_sim import VERSION  # noqa: E402
from agent_loop_sim.harness import Run  # noqa: E402
from agent_loop_sim.models import LocalModel, fnv1a32  # noqa: E402
from agent_loop_sim.scenarios import scenario  # noqa: E402
from agent_loop_sim.tokenizer import default_tokenizer  # noqa: E402

LLAMA = Path.home() / ".local/opt/llama.cpp"
GGUF = Path.home() / ".local/opt/models/qwen2.5-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf"


def post(url: str, body: dict) -> dict:
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.loads(f.read().decode())


class Recorder(LocalModel):
    """LocalModel that also keeps each prompt (for the fingerprint and tokenizer checks)."""

    def __init__(self, url: str, seed: int) -> None:
        super().__init__(url, seed=seed, n_predict=384)
        self.prompts: list[dict] = []

    def complete(self, req: dict) -> str:
        self.prompts.append({"prompt": req["prompt"], "input_tokens": req["input_tokens"]})
        return super().complete(req)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("trace_id")
    ap.add_argument("--style", default="native")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-turns", type=int, default=12)
    a = ap.parse_args()

    policy = {"style": a.style, "max_turns": a.max_turns, "latency": "local-i7", "cache": {"price": "local"}}
    sc = scenario(a.scenario)
    model = Recorder(a.url, a.seed)
    started = dt.datetime.now(dt.timezone.utc)
    events = Run(sc, policy, model=model).run()
    tok = default_tokenizer()

    calls = []
    tok_checks = []
    for p, rec in zip(model.prompts, model.records):
        ours = tok.encode(p["prompt"])
        theirs = post(a.url + "/tokenize", {"content": p["prompt"], "add_special": False, "parse_special": True})["tokens"]
        gen_ours = tok.encode(rec["content"])
        tok_checks.append({
            "prompt_tokens_ours": len(ours),
            "prompt_tokens_llama": len(theirs),
            "prompt_ids_equal": ours == theirs,
            "completion_ids_equal": gen_ours == rec["tokens"][: len(gen_ours)] and len(rec["tokens"]) - len(gen_ours) <= 1,
        })
        tm = rec["timings"]
        calls.append({
            "prompt_fnv1a32": fnv1a32(p["prompt"]),
            "prompt_sha256": hashlib.sha256(p["prompt"].encode()).hexdigest(),
            "prompt_tokens": p["input_tokens"],
            "text": rec["content"],
            "completion_ids": rec["tokens"],
            "llama": {
                "tokens_evaluated": rec["tokens_evaluated"],
                "tokens_cached": rec["tokens_cached"],
                "tokens_predicted": rec["tokens_predicted"],
                "stop_type": rec["stop_type"],
                "stopping_word": rec["stopping_word"],
                "prompt_n": tm.get("prompt_n"),
                "cache_n": tm.get("cache_n"),
                "prompt_ms": tm.get("prompt_ms"),
                "predicted_n": tm.get("predicted_n"),
                "predicted_ms": tm.get("predicted_ms"),
            },
        })
    commit = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sha = hashlib.sha256(GGUF.read_bytes()).hexdigest()
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "")
    end = events[-1]
    trace = {
        "format": "agent_loop_sim.recorded/1",
        "id": a.trace_id,
        "scenario": a.scenario,
        "policy": policy,
        "seed": sc.get("seed", 0),
        "provenance": {
            "model": "Qwen2.5-1.5B-Instruct",
            "model_repo": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "model_revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
            "gguf": GGUF.name,
            "gguf_sha256": sha,
            "quantisation": "Q4_K_M",
            "licence": "Apache-2.0",
            "runtime": "llama.cpp llama-server, /completion endpoint, CPU only",
            "llama_cpp_commit": commit,
            "sampling": {"temperature": 0, "top_k": 1, "seed": a.seed, "n_predict": 384},
            "cpu": cpu,
            "os": platform.platform(),
            "engine": VERSION,
            "recorded_at": started.isoformat(timespec="seconds"),
        },
        "calls": calls,
        "tokenizer_check": tok_checks,
        "outcome": {"status": end["status"], "answer": end["answer"], "turns": sum(1 for e in events if e["type"] == "model_call")},
    }
    out = ROOT / "traces" / f"{a.trace_id}.json"
    out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(calls)} calls, status {end['status']}")
    print("tokenizer:", all(c["prompt_ids_equal"] for c in tok_checks), [c["completion_ids_equal"] for c in tok_checks])
    for e in events:
        if e["type"] in ("model_call",):
            print("---", e["turn"], repr(e["text"][:300]))
        if e["type"] in ("tool_result", "error"):
            print("   ", e["type"], e.get("kind"), repr(e.get("text", e.get("detail", ""))[:160]))
    print("END", end["status"], repr(end["answer"][:300]))


if __name__ == "__main__":
    main()
