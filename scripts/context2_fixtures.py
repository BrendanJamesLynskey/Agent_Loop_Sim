"""The context module's second fixtures (engine 1.5.0: packing, compaction with a lossy summariser,
memory across sessions, long context against retrieval), written by make_fixtures.py:
fixtures/context2_fixtures.json (for the TS port to reproduce exactly) and
fixtures/context2_results.md (the recorded run every published number comes from)."""
from __future__ import annotations

from typing import Any

from agent_loop_sim import VERSION
from agent_loop_sim.context.corpus import DEFAULT, default_corpus
from agent_loop_sim.context.retrieval import Retriever
from agent_loop_sim.context.text import fixed
from agent_loop_sim.context import memory as M
from agent_loop_sim.context import packing as P
from agent_loop_sim.context import tradeoff as T
from agent_loop_sim.context.window import compaction_study, task_questions, window_run

LOSSY_BUDGETS = [1000, 1500, 2000, 3000]
LONG_TASK = 36
LONG_BUDGETS = [1500, 2000]
LOSSES = [0.1, 0.25, 0.5]
SEEDS = list(range(1, 21))
PACK_BUDGETS = [256, 512, 1024, 2048]
PACK_VIEWS = [(0, 512), (17, 512), (61, 1024), (92, 1024), (140, 512)]
TRADE_NCTX = [200000, 1000000]
TRADE_KS = [3, 5, 10, 20]
TRADE_Q = 50


def context2_fixtures() -> tuple[dict[str, Any], str]:
    c = default_corpus()
    d = Retriever(c, DEFAULT)
    task = task_questions(d, 12)
    long_task = task_questions(d, LONG_TASK)
    lossy = [window_run(d, task, b, p, "lossy") for b in LOSSY_BUDGETS for p in ["compact", "compact+retrieve"]]
    long_runs = [window_run(d, long_task, b, p) for b in LONG_BUDGETS for p in ["unbounded", "truncate", "compact", "compact+retrieve"]]
    long_lossy = [window_run(d, long_task, b, "compact", "lossy", 0.25, 1) for b in LONG_BUDGETS]
    studies = [compaction_study(d, long_task, b, p, loss, SEEDS) for b in LONG_BUDGETS
               for p in ["compact", "compact+retrieve"] for loss in LOSSES]
    pack = P.packing_eval(d, PACK_BUDGETS)
    views = [P.packing_view(d, qi, b) for qi, b in PACK_VIEWS]
    plan = M.memory_plan(d)
    mem = [M.memory_run(d, p, plan, name) for name, p in list(M.POLICIES.items()) + M.SWEEP]
    sizes = T.measured_sizes(d)
    trade = T.tradeoff(d, [sizes["corpus"]] + TRADE_NCTX, TRADE_KS, TRADE_Q)
    fx = {
        "engine": VERSION,
        "task": task,
        "long_task": long_task,
        "lossy": lossy,
        "long_runs": long_runs,
        "long_lossy": long_lossy,
        "studies": studies,
        "gain": [P.gain(i) for i in range(1, 21)],
        "position": [[x / 40, P.position_p(x / 40)] for x in range(41)]
        + [[0.3, P.position_p(0.3, {"start": 0.9, "middle": 0.2, "end": 0.4, "trough": 0.6})]],
        "packing": pack,
        "packing_views": views,
        "recency": [[h, M.recency(h)] for h in [0, 1, 23, 24, 100, 143]],
        "memory_plan": plan,
        "memory": mem,
        "tradeoff": trade,
    }
    return fx, results_md(task, long_task, lossy, long_runs, studies, pack, mem, trade)


def _f(x: float, d: int = 3) -> str:
    return fixed(x, d)


def results_md(task: list[int], long_task: list[int], lossy: list[dict[str, Any]], long_runs: list[dict[str, Any]],
               studies: list[dict[str, Any]], pack: dict[str, Any], mem: list[dict[str, Any]],
               trade: dict[str, Any]) -> str:
    out = [f"# Context module results, part 2 (engine {VERSION})", "",
           "Written by `scripts/make_fixtures.py` (CI checks it is up to date). Packing, compaction with a lossy "
           "summariser, memory across sessions, and long context against retrieval. Part 1 is `context_results.md`.", "",
           f"## The window task with a lossy summariser ({len(task)} questions; loss 0.25, seed 23)", "",
           "| budget | policy | facts answered | input tokens spent | compactions |", "|---|---|---|---|---|"]
    for w in lossy:
        out.append(f"| {w['budget']} | {w['policy']} (lossy) | {w['recalled']} of {len(task)} | {w['spent']} | {w['compactions']} |")
    out += ["", f"## The long task ({len(long_task)} questions, six per article)", "",
            "| budget | policy | facts answered | input tokens spent | model calls | compactions |", "|---|---|---|---|---|---|"]
    for w in long_runs:
        out.append(f"| {w['budget']} | {w['policy']} | {w['recalled']} of {len(long_task)} | {w['spent']} | {w['calls']} | {w['compactions']} |")
    out += ["", f"## What survives a lossy summariser (long task, {len(SEEDS)} seeds each)", "",
            "S(n): the share of facts still in the summary after their n-th compaction (product of each compaction's "
            "kept / faced), against the model (1 - loss)^n.", "",
            "| budget | policy | loss | mean facts answered | mean input tokens | S(1) | S(2) | S(3) | S(5) | (1-loss)^5 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for s in studies:
        sv = s["survival"]

        def at(n: int, key: str) -> str:
            return _f(sv[n - 1][key]) if len(sv) >= n else "-"
        out.append(f"| {s['budget']} | {s['policy']} | {s['loss']} | {_f(s['recalled'], 2)} of {len(long_task)} | "
                   f"{_f(s['spent'], 0)} | {at(1, 'measured')} | {at(2, 'measured')} | {at(3, 'measured')} | {at(5, 'measured')} | "
                   f"{at(5, 'model')} |")
    out += ["", f"## Packing (reranked top {pack['candidates']} candidates, value 1/log2(rank+1); means over all questions)", "",
            "Position curve (illustrative): start " + _f(pack["position"]["start"], 2) + ", middle " + _f(pack["position"]["middle"], 2)
            + ", end " + _f(pack["position"]["end"], 2) + ", trough at " + _f(pack["position"]["trough"], 2) + ".", "",
            "| budget | packer | answer in window | tokens used | value | p best-first | p best-last | p ends | p middle |",
            "|---|---|---|---|---|---|---|---|---|"]
    for b in pack["budgets"]:
        for p in P.PACKERS:
            a = pack["results"][str(b)][p]
            out.append(f"| {b} | {p} | {_f(a['answered'])} | {_f(a['tokens'], 1)} | {_f(a['value'])} | {_f(a['best-first'])} | "
                       f"{_f(a['best-last'])} | {_f(a['ends'])} | {_f(a['middle'])} |")
    out += ["", f"## Memory across sessions ({M.SESSIONS} sessions of {M.PER_SESSION} questions, then a probe-only session)", "",
            "| policy | recalled | repeats | neighbours | memory tokens read | write tokens | stored at the end |",
            "|---|---|---|---|---|---|---|"]
    for m in mem:
        out.append(f"| {m['name']} | {m['recalled']} of {m['probes']} | {m['repeat'][0]} of {m['repeat'][1]} | "
                   f"{m['neighbour'][0]} of {m['neighbour'][1]} | {m['read_tokens']} | {m['write_tokens']} | {m['stored_tokens']} |")
    sz = trade["sizes"]
    out += ["", f"## Long context or retrieval ({TRADE_Q} questions, {T.GAP_MS // 1000} s apart, {T.OUT_TOKENS} output tokens each)", "",
            f"Measured: system prompt {sz['system']} tokens, mean question {_f(sz['question'], 2)}, mean chunk {_f(sz['chunk'], 1)} "
            f"({sz['chunks']} chunks), corpus {sz['corpus']} tokens. Retrieval: reranked top k; answer in the prompt = recall@k "
            + ", ".join(f"k={k} {_f(v)}" for k, v in trade["recall"].items()) + ".", "",
            "| model | set (tokens) | k | strategy | first call $ | later call $ | all questions $ | first TTFT (s) | later TTFT (s) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for key, run in trade["runs"].items():
        for s, rows in run["strategies"].items():
            if s != "rag" and run["k"] != TRADE_KS[1]:
                continue  # the long prompts do not depend on k
            if s == "rag" and run["n_ctx"] != trade["sizes"]["corpus"]:
                continue  # nor retrieval on the set's size
            out.append(f"| {run['model']} | {run['n_ctx'] if s != 'rag' else 'any'} | {run['k'] if s == 'rag' else '-'} | {s} | {_f(rows['cost'][0], 4)} | "
                       f"{_f(rows['cost'][1], 4)} | {_f(rows['cum'][-1], 3)} | {_f(rows['ttft'][0] / 1000, 2)} | {_f(rows['ttft'][1] / 1000, 2)} |")
    return "\n".join(out) + "\n"
