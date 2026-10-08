# Agent_Loop_Sim

A deterministic simulator of an **agent harness**: the loop that turns a chat model into an agent.
It runs the loop (model → tool calls → results → model), with every harness choice a switchable policy:
the tool-calling style, parallel calls and stop conditions; permission rules, modes and a simulated human;
context management (truncation, tool-result clipping, summarising compaction); prompt-cache-aware prompt
layout; sub-agents; pre- and post-tool hooks; a sandbox around the shell; and error recovery (retries with
back-off, malformed-call feedback, loop detection). Every run produces a versioned **event trace** (JSON Lines) that the
companion sites animate. Since 1.2 it also models **agent protocols** on the wire: MCP's client and server state
machines over JSON-RPC 2.0 in both protocol eras, its transports and its OAuth 2.1 authorisation, checked
message for message against the official MCP Python SDK; since 1.3, **A2A** (agent to agent) checked the same way
against the official A2A Python SDK, an MCP gateway, and three protocol attacks with their defences; since 1.4,
**context engineering**: chunking, BM25, dense retrieval over shipped int8 embeddings, hybrid fusion and a recorded
reranker, measured with recall@k, MRR and nDCG on a fixed, openly licensed corpus, and the window as working memory.

- **Python is the reference** (`src/agent_loop_sim`); a **TypeScript port** (`ts/`) reproduces its fixtures
  exactly: token ids, random draws, every event of every run, every animation frame. No tolerance.
- **Token counts are real**: Qwen2.5's tokenizer is vendored and checked against llama.cpp's on every prompt.
- **No live model anywhere** in CI or on the sites. Teaching scenarios use a scripted model; three **real
  traces**, recorded once from a small open-weights model on a CPU, are replayed token-exactly.

Used by **[Agent Harnesses Explained](https://agent-harnesses-explained.vercel.app)** and
**[Agent Protocols Explained](https://agent-protocols-explained.vercel.app)** and
**[Agent Context Explained](https://agent-context-explained.vercel.app)**, which vendor the TS port at a pinned
commit.

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
| `protocols/mcp.py` / `protocols/mcp.ts` | MCP's client and server as state machines exchanging JSON-RPC 2.0 messages, in both protocol eras (below) |
| `protocols/transport.py` / `.ts` | framing, bytes and timing on stdio and Streamable HTTP; a dropped SSE stream resumed (2025-11-25) or re-sent (2026-07-28) |
| `protocols/oauth.py` / `.ts` | the OAuth 2.1 flow MCP requires, as a state machine, with mis-configured variants that fail where the spec says |
| `protocols/views.py` / `.ts` | frames for the Protocols site: sequence charts, version and capability negotiation, a tool call end to end, N×M vs N+M; since 1.3, A2A charts and summaries, and any step-by-step flow (OAuth, attacks) as a chart |
| `protocols/a2a.py` / `.ts` | A2A 1.0: an orchestrating agent (client) and a remote research agent (server) over the JSON-RPC binding with SSE; the task life cycle (since 1.3) |
| `protocols/gateway.py` / `.ts` | an MCP gateway over four servers: tool-name collisions, prefixing, an allow-list, the tool list's token cost, routing (since 1.3) |
| `context/corpus.py` / `.ts` | the fixed corpus, its labelled questions, six chunking configurations and the shipped vectors (since 1.4) |
| `context/chunking.py` / `.ts` | fixed (with optional overlap), recursive and semantic chunkers, cutting only between tokenizer pieces |
| `context/retrieval.py` / `.ts` | BM25, dense retrieval (int8, int4, binary), RRF and weighted fusion, the recorded reranker |
| `context/evaluate.py` / `.ts` | relevance from answer spans; recall@k, MRR@10, nDCG@10 |
| `context/window.py` / `.ts` | a long research task under a token budget: truncate, compact, search again |
| `context/views.py` / `.ts` | frames for the Context site: BM25 term by term, nearest neighbours, RRF rank by rank and the rerank, chunk boundaries |
| `context/mathx.py` / `.ts` | fdlibm's natural logarithm, bit for bit in both languages (for BM25's IDF and nDCG) |
| `protocols/security.py` / `.ts` | tool poisoning, a confused deputy and state-handle hijacking, each with and without its defence (since 1.3) |

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

## Protocols (since 1.2)

**The spec revision modelled is MCP [2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28)**
(the current revision on [modelcontextprotocol.io](https://modelcontextprotocol.io/specification/versioning),
accessed 2026-10-08), together with the handshake era it replaced (2024-11-05 → 2025-11-25):

| | Handshake era (≤ 2025-11-25) | 2026-07-28 |
| --- | --- | --- |
| Start | `initialize` → result (agreed version, server capabilities) → `notifications/initialized` | nothing: every request carries `_meta` with the version, the client's info and capabilities; `server/discover` is optional |
| Results | plain | every result has `resultType`; list and read results add `ttlMs` / `cacheScope`; the server names itself in `_meta` |
| Server needs the user or a model | the server sends its own request (`elicitation/create`, `sampling/createMessage`) and waits | the result is `resultType: "input_required"` with `inputRequests`; the client retries with `inputResponses` (multi round-trip requests) |
| Streamable HTTP | sessions (`Mcp-Session-Id`), SSE events with IDs, resume with `Last-Event-ID` | no sessions, no resumption; `Mcp-Method` / `Mcp-Name` headers mirror the body |

**Conformance with the official SDK.** `conformance/sdk_server.py` is a small server written with the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (`mcp` 2.3.0, pinned in the
`conformance` extra). `scripts/record_sdk.py` runs the SDK's own `Client` against it over stdio, through a tee
that logs every line, for each scenario in `protocols/scenarios.py` (18: both eras, `auto` discovery, progress,
errors, elicitation accepted and declined, sampling, malformed and out-of-era requests, an old client) and writes
`fixtures/sdk_exchanges.json`. The engine must produce the same messages, compared as parsed JSON with request
IDs and progress tokens renumbered in order of appearance (`mcp.normalise`). CI re-records live and checks
live recording = committed recording = engine; the TS port is checked against the same recording.

Things the recordings taught the model (each is SDK 2.3.0 behaviour, not a spec rule): the stdio server sends
nothing back for a line that is not JSON (JSON-RPC 2.0 would answer `-32700`); a connection is locked to the era
of its first valid request; after a tool result with `structuredContent` the client lists the tools (to validate
the output against `outputSchema`) if it has not yet; `server/discover` advertises `listChanged: true` while
`initialize` reports the server's own `false`.

**Engine-only** (not in the SDK recordings): pagination (`server.page_size`, the fixture server does not
paginate), cancellation (`notifications/cancelled`, whose timing a recording cannot pin down), a
`notifications/tools/list_changed` after the server gains a tool, the HTTP framing and timing, and OAuth
(`protocols/oauth.py`: protected-resource metadata, authorisation-server metadata, issuer check, PKCE S256
with real SHA-256 (the RFC 7636 test vector is in the tests), Client ID Metadata Documents or Dynamic Client
Registration, `resource`, `iss`, audience and scope checks, and variants: no PKCE support, PKCE skipped, wrong
verifier, issuer mismatch, `iss` mix-up, wrong audience, insufficient scope, token passthrough).

## A2A, gateways and attacks (since 1.3)

**The A2A version modelled is 1.0** ([specification](https://a2a-protocol.org/latest/specification/), release
[v1.0.1](https://github.com/a2aproject/A2A/releases/tag/v1.0.1) of 2026-05-28, accessed 2026-10-08): the Agent
Card at `/.well-known/agent-card.json` (with the weak `ETag` the SDK sends), the JSON-RPC binding's PascalCase
methods (`SendMessage`, `SendStreamingMessage`, `GetTask`, `CancelTask`, `SubscribeToTask`,
`GetExtendedAgentCard`), the `A2A-Version` header (an empty one means 0.3), `TASK_STATE_*` states (terminal:
completed, failed, canceled, rejected; interrupted: input required, auth required), multi-turn follow-ups with
`taskId`/`contextId`, SSE streams of `task` → `statusUpdate` / `artifactUpdate` (chunks with `append` and
`lastChunk`) that close on a terminal state, and the error codes -32001…-32009 with `google.rpc.ErrorInfo` details.
Not modelled: push notifications, `ListTasks`, the gRPC and HTTP+JSON bindings, signed cards.

**Conformance with the official A2A SDK.** `conformance/a2a_sdk_server.py` is the research agent written with the
[A2A Python SDK](https://github.com/a2aproject/a2a-python) (`a2a-sdk[http-server]` 1.2.2, pinned in the
`conformance` extra). `scripts/record_a2a_sdk.py` drives it with the engine's own orchestrator, in process
(Starlette's test client, no socket), for each of the 8 scenarios in `a2a.A2A_SCENARIOS` (blocking and streamed
delegation, input required, auth required with the user signing in outside A2A, cancel and the errors after it, a
rejection, a direct message, and five errors) and writes `fixtures/a2a_sdk_exchanges.json` (56 messages). The
engine's agent must give the same messages once UUIDs are renumbered and timestamps blanked (`a2a.normalise`); CI
re-records live in the conformance job. SDK 1.2.2 behaviour the recordings taught the model: cancelling a task that
is waiting for input (so nothing is running for it) writes `TASK_STATE_CANCELED` over its stored status and keeps
the status message; a message to a finished task is refused with "Task … is in terminal state: …"; a status message
moves into the history when the next message arrives.

**The gateway** (`gateway.py`) lists four downstream servers and merges their tools under three policies: `flat`
(same names collide: the first server wins and `search` goes to the wrong server), `prefix` (`web.search`, as the
spec's tools page recommends for aggregators) and `filtered` (an allow-list). Token counts of the merged list use
the vendored tokenizer. **The attacks** (`security.py`) are inert walk-throughs against `*.example.com`: a tool whose
description changes after approval (defence: pin a hash of the approved definition), the OAuth proxy confused
deputy from MCP's security best practices (defence: per-client consent at the proxy), and state-handle hijacking
(defence: key the state by the verified token's user).

## Context (since 1.4)

**The corpus** is six articles of the **SQuAD v1.1 development set** (Rajpurkar et al., *SQuAD: 100,000+ Questions
for Machine Comprehension of Text*, EMNLP 2016, [arXiv:1606.05250](https://arxiv.org/abs/1606.05250); licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)): Computational complexity theory, Packet
switching, Steam engine, Oxygen, Prime number and Apollo program, 37,499 Qwen2.5 tokens, with 200 of their 1,160
questions picked by the seeded generator (seed 23). Each question keeps its first gold answer's character span,
so relevance needs no judgement: **a chunk is relevant when it contains the whole answer**, and a chunker that
cuts an answer in two loses that question. `data/context/corpus.json` is the derived corpus (CC BY-SA 4.0; the
changes are listed in `manifest.json`).

**The models run offline, once** (`scripts/build_context_data.py`, CPU, ONNX Runtime 1.30.0, two threads):
- embeddings: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  at revision `1110a24` (Apache-2.0; MiniLM, [arXiv:2002.10957](https://arxiv.org/abs/2002.10957); trained with the
  Sentence-BERT recipe, [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)), `onnx/model.onnx` SHA-256
  `6fd5d72f…6452`, mean pooling then L2 normalisation, 384 dimensions, 256 word pieces at most;
- reranker: [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
  at revision `233902d` (Apache-2.0; trained on MS MARCO, [arXiv:1611.09268](https://arxiv.org/abs/1611.09268)),
  `onnx/model.onnx` SHA-256 `5d3e70fd…4d4a`.

What ships (`src/agent_loop_sim/data/context/`, 1.7 MB, each file's SHA-256 in `manifest.json`): the **int8
embeddings** of the 200 questions, of all 1,234 sentences and of every chunk of the six configurations (each
vector scaled so its largest component is ±127); the cross-encoder's score for each question's **candidate pool**
(the RRF top 30 with the BM25 and dense top 10s of the default chunking: 6,226 pairs; scoring all 41,600 pairs
would take hours on this CPU); and a 2-D PCA of the default chunks. Each embedding file names the chunks it embeds
(a SHA-256 of their spans), and the tests require the engine's chunkers to produce exactly those chunks.

**Everything else is recomputed from those files by the engine, identically in both languages**: chunk
boundaries (from Qwen2.5 pre-tokenizer pieces; the semantic chunker reads the int8 sentence vectors), BM25 (with
its IDF through `mathx.ln`, a transcription of fdlibm's logarithm, because libm's and V8's `log` differ in the
last bit for some inputs), integer cosines, int4 and binary re-quantisation, fusion, the rerank, and every metric.
The TS tests compare every evaluation (each question's first relevant rank and the means) with no tolerance. The
only numbers that need the unquantised vectors are the float32 baselines, recorded by the offline run in
`manifest.json` and labelled as such.

Results (all in [`fixtures/context_results.md`](fixtures/context_results.md), the recorded run):

| default chunking (recursive, 256 tokens) | recall@1 | recall@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| BM25 (k1 1.2, b 0.75) | 0.785 | 0.985 | 0.869 | 0.899 |
| dense, int8 | 0.705 | 0.950 | 0.806 | 0.849 |
| dense, binary | 0.600 | 0.870 | 0.720 | 0.778 |
| RRF (k 60) | 0.810 | 0.980 | 0.883 | 0.912 |
| RRF, then the cross-encoder on the top 20 | 0.890 | 0.995 | 0.937 | 0.952 |

On this corpus BM25 beats the small embedder: SQuAD's questions were written by people looking at the paragraph,
so they share its words. Fixed 128- and 256-token chunks cut 5 and 4 answers in two; 512-token chunks exceed the
embedder's 256 word pieces (73 of 77 are truncated), and dense recall@5 falls to 0.755.

## Packing, compaction, memory and long context (since 1.5)

Four more modules in `context/`, each ported statement for statement to `ts/src/context/` and checked against
[`fixtures/context2_fixtures.json`](fixtures/context2_fixtures.json) with no tolerance (every frame of every run).
Every number is in [`fixtures/context2_results.md`](fixtures/context2_results.md), the recorded run.

- **`packing`**: what to put in a token budget. The candidates are a question's reranked top 20 chunks; a chunk's
  value is the nDCG gain of its rank, 1/log2(rank + 1); its weight is its token count. Three packers: `top`
  (reranked order, skip what does not fit), `density` (value per token) and `optimal` (0/1 knapsack by dynamic
  programming, checked against brute force in the tests). Then **where** to put the packed chunks: a U-shaped
  position curve in the spirit of Liu et al., *Lost in the Middle* ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)),
  with **illustrative** parameters, and four placements (`best-first`, `best-last`, `ends`, `middle`). At 512
  tokens the answer is in the window for 0.965 of the questions with `top`, 0.940 with `optimal`: the knapsack
  maximises the value it was given, which is not the same thing as holding the answer.
- **`window`** gains a **lossy summariser** (`summariser="lossy"`, `loss`, `seed`): each compaction rewrites the
  summary and keeps each line with probability 1 − loss, drawn from the shared seeded generator. `compaction_study`
  runs a 36-question task over 20 seeds and measures survival: at loss 0.25 the share of facts left after five
  compactions is 0.251 (budget 1,500) against the model's 0.237, and the agent answers 10.70 of 36 on average;
  searching again for what was lost (`compact+retrieve`) answers all 36 for about twice the tokens. The
  truncation caption now names the question whose fact is lost ("the fact for question 2 is lost").
- **`memory`**: six sessions of three questions, then probes in later sessions (each question again, plus
  "neighbour" questions whose answer was in a chunk the agent read but was not asked about). Policies: none,
  transcript, scratchpad (a notes file), episodic (Generative Agents scoring, recency 0.995 per hour + importance
  + relevance; Park et al., [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)), semantic (facts extracted by a
  scripted extractor, retrieved by cosine), each with forgetting caps and consolidation (merging a re-read chunk or
  a fact extracted twice). Semantic memory recalls 19 of 22 probes for 2,231 memory tokens; the full transcript 22
  of 22 for 67,498.
- **`tradeoff`**: 50 questions over a document set in one prompt (with and without the prompt cache) against the
  reranked top k chunks, priced by `accounting`. With Claude Sonnet 4.6's list prices, a 1M-token set costs
  $150.05 uncached and $18.49 cached for the 50 questions; top-5 retrieval costs $0.18 with the answer in the prompt
  for 0.995 of the questions.

## Versions

- **1.5.0** (2026-10-08): `context/packing.py`, `context/memory.py`, `context/tradeoff.py`; a lossy summariser
  and `compaction_study` in `context/window.py`; new fixture files `fixtures/context2_fixtures.json` and
  `fixtures/context2_results.md`. Every 1.4 fixture is unchanged apart from the `engine` field, except one
  deliberate wording change: the window task's truncation caption said "(facts lost: 2)", meaning question 2,
  and now says "(the fact for question 2 is lost)" or "(it held no fact)" (99 captions in
  `context_fixtures.json`; no number changed).
- **1.4.0** (2026-10-08): the `context` package (corpus, chunkers, retrieval, evaluation, the window task,
  views), its data and offline build script, `Tokenizer.pieces`; new fixture files `fixtures/context_fixtures.json`
  and `fixtures/context_results.md`. Every 1.3 fixture is unchanged apart from the `engine` field.
- **1.3.0** (2026-10-08): `protocols/a2a.py` (A2A 1.0) and its SDK conformance recordings, `protocols/gateway.py`,
  `protocols/security.py`, `views.a2a_frames` / `a2a_summary` / `flow_frames`; new fixture file
  `fixtures/protocols2_fixtures.json`. Every 1.2 fixture is unchanged apart from the `engine` field.
- **1.2.0** (2026-10-08): the `protocols` package (MCP state machines in both eras, transports, OAuth 2.1,
  views), the SDK conformance recordings and their CI job. Every 1.1 fixture is unchanged apart from the
  `engine` field.
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
- **Protocols**: transport latencies (`transport.TRANSPORTS`: a 0.05 ms pipe and 150 ms process start for
  stdio; 20 ms one way, 50 Mbit/s and 80 ms to connect for HTTP), server work times and the user's and model's
  answer times (`transport.WORK_MS`) are illustrative. HTTP header sets are the minimal ones the spec requires
  plus Host and Content-Length. OAuth tokens are opaque strings with their claims kept beside them; the hosts
  (`*.example.com`) are invented.
- **A2A**: the research agent's replies are scripted; network, work, model and sign-in times (`a2a.TIMES`: 20 ms
  one way, 40 ms to start, 600 ms per artifact chunk, 1.2 s for the orchestrator's model, 4 s for a person to sign
  in) are illustrative. IDs come from the seeded generator and timestamps from a simulated clock.
- **Gateway and attacks**: the servers, tools, handles, secrets and tokens are invented placeholders.
- **Context**: the corpus and questions are real (SQuAD) and the embeddings and reranker scores are real model
  outputs; the window task is a scripted agent (BM25 search, at most three reads a question) with budgets of 1,000
  to 3,000 tokens, scaled down so a twelve-question task overflows them, and its default summariser is perfect (it
  keeps one line per fact), so compaction there loses nothing by construction. The lossy summariser's loss rates
  (0.1, 0.25, 0.5) are illustrative.
- **Packing, memory, long context (1.5)**: the position curve's parameters (start 0.75, middle 0.55, end 0.65,
  trough at the middle) are illustrative, shaped like the U that *Lost in the Middle* reports, not its numbers;
  chunk values are a rank gain, not a probability. The memory agent, its importance ratings (8, 3, 5) and its
  extractor are scripted. The long-context comparison uses the dated list prices and the `hosted` latency profile;
  sets larger than the corpus are hypothetical (the same arithmetic), and whether a model accepts such a prompt,
  and any long-prompt surcharge, are not modelled.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                                   # the reference
.venv/bin/python scripts/make_fixtures.py --check  # fixtures, engine data and replays up to date
cd ts && pnpm install && pnpm test                 # the port, against the fixtures
```

SDK conformance: `.venv/bin/pip install -e ".[conformance]"`, then `python scripts/record_sdk.py --check` (MCP)
and `python scripts/record_a2a_sdk.py --check` (A2A): each records its SDK live and compares; without `--check`
they re-record.

Context data (offline, downloads the SQuAD file and two ONNX models, about 190 MB):
`.venv/bin/pip install -e ".[offline]"`, then
`python scripts/build_context_data.py --cache ~/.cache/agent-context-offline` and `python scripts/make_fixtures.py`.

Recording (offline, needs a local `llama-server` on port 8080):
`python scripts/record_trace.py fix_test qwen-fix-test-native --style native`, then
`python scripts/tokenizer_check.py` and `python scripts/make_fixtures.py`.

## Licence

MIT (`LICENSE`). The vendored Qwen2.5 merges file is Apache-2.0; the context corpus (`data/context/corpus.json`)
is derived from SQuAD and is CC BY-SA 4.0; the embeddings and reranker scores are outputs of Apache-2.0 models;
see `NOTICE`.

Part of the [LLMs](https://github.com/BrendanJamesLynskey/LLMs) collection.
