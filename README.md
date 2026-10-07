# Agent_Loop_Sim

A deterministic simulator of an **agent harness**: the loop that turns a chat model into an agent.
It runs the loop (model → tool calls → results → model), with every harness choice a switchable policy:
the tool-calling style, parallel calls and stop conditions; permission rules, modes and a simulated human;
context management (truncation, tool-result clipping, summarising compaction); prompt-cache-aware prompt
layout; sub-agents; pre- and post-tool hooks; a sandbox around the shell; and error recovery (retries with
back-off, malformed-call feedback, loop detection). Every run produces a versioned **event trace** (JSON Lines) that the
companion sites animate.

- **Python is the reference** (`src/agent_loop_sim`); a **TypeScript port** (`ts/`) reproduces its fixtures
  exactly: token ids, random draws, every event of every run, every animation frame. No tolerance.
- **Token counts are real**: Qwen2.5's tokenizer is vendored and checked against llama.cpp's on every prompt.
- **No live model anywhere** in CI or on the sites. Teaching scenarios use a scripted model; three **real
  traces**, recorded once from a small open-weights model on a CPU, are replayed token-exactly.

Used by **[Agent Harnesses Explained](https://agent-harnesses-explained.vercel.app)**, which vendors the TS
port at a pinned commit.

## The pieces

| Module | What it does |
| --- | --- |
| `rng.py` / `rng.ts` | mulberry32, the seeded generator Disaggregated_Inference_Sim's browser engine uses: 32-bit integer maths, so both languages draw the same numbers |
| `tokenizer.py` / `tokenizer.ts` | Qwen2.5's byte-level BPE (the vendored `merges.txt`; ids follow from merge ranks) |
| `chat.py` / `chat.ts` | the prompt the model sees: Qwen2.5's ChatML template, native tool calls (`<tool_call>` JSON) or ReAct text |
| `tools.py` / `tools.ts` | JSON-Schema tools over a fake world: a file system, a shell simulator (its `pytest` really evaluates the tiny repository's tests), search over a fixed corpus, a calculator |
| `parse.py` / `parse.ts` | model text → tool calls; malformed output gets fixed-wording feedback |
| `accounting.py` / `accounting.ts` | prompt-cache simulation (prefix match, block size, minimum length, TTL), cost from a dated price table, latency (TTFT + tokens/s) |
| `models.py` / `models.ts` | the model interface: `scripted`, `replay` (checks each prompt's fingerprint) and, Python only, `local` (llama.cpp's `llama-server`) |
| `harness.py` / `harness.ts` | the loop and every policy |
| `views.py` / `views.ts` | the animation frames the sites draw, derived from a trace (the loop, tool calls, the context budget, the cache, permissions, a timeline; since 1.1, every agent's context side by side and each call's path through permissions, hooks and the sandbox) |
| `sweeps.py` / `sweeps.ts` | seeded sweeps: one scenario over many seeds per value of a knob (since 1.1: the retry budget) |
| `schema/trace.schema.json` | JSON Schema of one trace event |

## Policies

```python
from agent_loop_sim.harness import run
from agent_loop_sim.scenarios import scenario

events = run(scenario("fix_test_cleanup"), {
    "style": "native",                                     # or "react"
    "parallel": True, "max_turns": 20,
    "context": {"window": 3000, "strategy": "summarise"},  # none | truncate | clip | summarise
    "cache": {"price": "claude-sonnet-4.6", "layout": "stable"},  # stable | timestamp | reorder_tools
    "permissions": {"mode": "default",                     # default | auto | ask | read_only
                    "rules": [{"tool": "run_shell", "pattern": "rm *", "decision": "deny"}],
                    "human": {"latency_ms": [3000, 9000], "deny": []}},
    "hooks": [{"name": "quiet", "phase": "pre", "tool": "run_shell", "match": "pytest*",
               "action": "rewrite", "find": "pytest", "replace": "pytest -q"}],
    "recovery": {"retries": 2, "backoff_ms": 500, "backoff_factor": 2,
                 "loop_repeats": 3, "loop_action": "stop", "malformed": "feedback"},
    "sandbox": {"mode": "workspace", "network": False},     # off | workspace | read_only (1.1; optional)
})
```

The sandbox (1.1) wraps the shell only, as the public harnesses' sandboxes do; the file tools stay under
the permission rules. `workspace` lets a command write inside the repository only and blocks the network;
`read_only` blocks every write. A stopped command comes back as a failed `tool_result` of kind `sandboxed`.
The checks look at the command line, which is all a fake shell has; a real sandbox is enforced by the
operating system. Leaving `sandbox` out keeps every 1.0 run byte-identical (apart from the `engine` version).

Permission rules: deny rules first, then the read-only mode, then ask rules, then allow rules, then the
mode's default (reads allowed, writes ask). A rule's pattern is a glob over the call's subject (a path or a
command line). The defaults are in `harness.DEFAULT_POLICY`.

## The trace format (version 1)

One JSON object per line. Every event has `v`, `seq`, `t` (simulated milliseconds), `agent` (`main`, `sub1` …)
and `type`, one of `run_start`, `model_call`, `tool_call`, `permission_check`, `hook`, `retry`, `tool_result`,
`compaction`, `error`, `handoff`, `checkpoint`, `run_end`. A `model_call` carries the prompt's token count, its
cached prefix, the output tokens, the cost, the time to first token and the duration, and the context's
make-up message by message. See `schema/trace.schema.json`; `traces/*.events.jsonl` are examples.

## Scenarios

| Scenario | Task |
| --- | --- |
| `fix_test` (+ `_react`, `_malformed`, `_cleanup`, `_flaky`, `_guarded`) | a tiny repository whose `add()` subtracts: find it, fix it, re-run the tests (`_guarded`, 1.1: then `pip install`, `rm -rf build` and `rm -rf ~/.cache/pytest`, for hooks and the sandbox) |
| `research` (+ `_subagent`) | read four long design notes and work out two numbers (fills the context) |
| `lookup` | search a small corpus and use the calculator |

The notes' numbers are invented for the exercise (`tiny-7b` and `dev-gpu` are not real products).

## Versions

- **1.1.0** (2026-10-07): the shell sandbox (`policy.sandbox`, result kind `sandboxed`), `pip install` in the
  shell simulator, the `fix_test_guarded` scenario, `views.agents_frames` / `pipeline_frames`,
  `sweeps.retry_sweep` (finished, and finished with passing tests: `verified`). Every 1.0 fixture run is unchanged apart from the `engine` field.
- **1.0.0** (2026-10-07): the first release.

## Recorded traces

Recorded on 2026-10-07 with **Qwen2.5-1.5B-Instruct, Q4_K_M GGUF** (Hugging Face
`Qwen/Qwen2.5-1.5B-Instruct-GGUF` revision `91cad51`, file SHA-256 `6a1a2eb6…407e`), llama.cpp `llama-server`
commit `b9acf13`, CPU only (Intel i7-3770), greedy decoding (temperature 0, top-k 1, seed 42), at most 384
output tokens per call. Each file in `traces/` records the provenance, every completion and its token ids,
and llama.cpp's own counts and timings. Our tokenizer gave the same prompt token ids as llama.cpp on every
recorded prompt (and on all 102 prompts of the scripted scenarios: `fixtures/tokenizer_llama_check.json`).

Where the small model fails, plainly:

| Trace | What happened |
| --- | --- |
| `qwen-fix-test-native` | listed the files, then claimed `calc.py` was missing (it was in the listing) and stopped without running the tests. |
| `qwen-fix-test-react` | ran the tests correctly, misread the failure as `add` missing from the test file, wrapped its `Action Input` in a Markdown code fence twice (both rejected as malformed), then declared success without changing anything. |
| `qwen-lookup-native` | made two well-chosen searches in one turn (parallel calls), then skipped the calculator, did the arithmetic in prose with wrong conversions, and ran into the 384-token output limit mid-sentence. |

Replaying a trace rebuilds every prompt and checks its fingerprint (FNV-1a over the UTF-8 bytes) and token
count against the recording, so a replay proves the harness reproduces the recorded prompts byte for byte —
in Python and in TypeScript.

## What is illustrative

- **Prices** (`accounting.PRICES`): list prices copied from the
  [Claude API pricing page](https://platform.claude.com/docs/en/about-claude/pricing),
  [its prompt-caching page](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) and the
  [OpenAI API pricing page](https://developers.openai.com/api/docs/pricing) /
  [prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching), accessed 2026-10-07.
  Prices change; and token counts here are Qwen2.5's, not those providers' tokenizers. A cost is "what this
  many tokens would cost at that price", not a quote.
- **Latency**: the `hosted` profile is round illustrative numbers; `local-i7` uses the medians of the
  recorded traces' llama.cpp timings (prefill 42.7 tokens/s, decode 16.5 tokens/s; the 50 ms overhead is
  assumed). Tool latencies and the human's answer times are illustrative.
- The prompt cache is a simple prefix model (longest common prefix with a live entry, rounded down to a
  block, above a minimum, with a TTL), not any provider's implementation.
- The shell, test runner and search are simulators of the few commands the scenarios use.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                                   # the reference
.venv/bin/python scripts/make_fixtures.py --check  # fixtures, engine data and replays up to date
cd ts && pnpm install && pnpm test                 # the port, against the fixtures
```

Recording (offline, needs a local `llama-server` on port 8080):
`python scripts/record_trace.py fix_test qwen-fix-test-native --style native`, then
`python scripts/tokenizer_check.py` and `python scripts/make_fixtures.py`.

## Licence

MIT (`LICENSE`). The vendored Qwen2.5 merges file is Apache-2.0; see `NOTICE`.

Part of the [LLMs](https://github.com/BrendanJamesLynskey/LLMs) collection.
