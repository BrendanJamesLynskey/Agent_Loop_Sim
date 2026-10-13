"""The evals module's second part (engine 1.9.0), written by make_fixtures.py: ts/src/evals2_data.json (the
configurations, one source of truth for both languages), fixtures/evals2_fixtures.json (graders, the judge,
trajectories, the scaffold study and its costs, contamination: for the TS port to reproduce exactly) and
fixtures/evals2_results.md (the recorded run every published number comes from)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_sim import VERSION
from agent_loop_sim.context.text import fixed
from agent_loop_sim.evals import contamination as C
from agent_loop_sim.evals import judge as J
from agent_loop_sim.evals import scaffold as F
from agent_loop_sim.evals import stats as S
from agent_loop_sim.evals import trajectory as T

ROOT = Path(__file__).resolve().parent.parent
TRACES = ["qwen-fix-test-native", "qwen-fix-test-react", "qwen-lookup-native"]

KAPPA = [[[20, 5], [10, 15]], [[45, 15], [25, 15]], [[10, 0, 0], [0, 10, 0], [0, 0, 10]], [[5, 5], [5, 5]],
         [[30, 2, 3], [4, 25, 6], [1, 2, 27]], [[0, 7], [3, 0]]]
RG = [[0.6, 0.95, 0.7], [0.3, 0.9, 0.9], [0.05, 0.95, 0.7], [0.99, 0.8, 0.8], [0.5, 0.5, 0.75]]
CORR = [[182, 300, 81, 85, 86, 115], [50, 100, 45, 50, 40, 50], [10, 200, 30, 30, 60, 70]]
PARETO = [[[1.0, 0.5], [2.0, 0.7], [1.5, 0.4], [3.0, 0.7], [0.5, 0.2], [2.0, 0.8]],
          [[1.0, 0.5], [1.0, 0.5], [0.8, 0.3]], [[3.0, 0.9]]]
EXAMPLES = [["small|edit|native|4", 9, 0], ["small|test|native|16", 9, 0], ["small|test|react|16", 9, 0],
            ["large|edit|native|4", 9, 0]]
PAIRS = [["small|test|native|16", "small|edit|native|4"], ["small|test|native|16", "large|edit|native|4"],
         ["small|test|react|8", "small|test|native|8"], ["large|test|native|16", "large|test|native|4"]]


def evals2_data() -> dict[str, Any]:
    return {"version": VERSION, "labels": J.LABELS, "judge": J.JUDGE, "grading": J.GRADING,
            "trajectories": T.TRAJECTORIES, "unsafe": T.UNSAFE, "claim_words": T.CLAIM_WORDS,
            "lookup_answers": T.LOOKUP_ANSWERS, "weights": T.WEIGHTS, "stopped": T.STOPPED,
            "scaffold": F.SCAFFOLD, "tasks": F.TASKS, "agents": F.AGENTS, "contam": C.CONTAM}


def load_traces() -> dict[str, Any]:
    return {n: json.loads((ROOT / "traces" / f"{n}.json").read_text(encoding="utf-8")) for n in TRACES}


def evals2_fixtures(tok: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    stats = {
        "kappa": [[t, S.cohen_kappa(t)] for t in KAPPA],
        "confusion": S.confusion(["A", "B", "tie", "A", "B", "A"], ["A", "A", "tie", "B", "B", "tie"], J.LABELS),
        "rogan_gladen": [[p, se, sp, S.rogan_gladen(p, se, sp)] for p, se, sp in RG],
        "corrected": [[c, S.corrected_rate(*c)] for c in CORR],
        "pareto": [[pts, S.pareto_front(pts)] for pts in PARETO],
    }
    sc = F.scaffold_study(tok)
    examples = []
    for cid, t, j in EXAMPLES:
        conf = next(c for c in F.configs() if c["id"] == cid)
        r = F.run_one(conf, t, j, tok, F.SCAFFOLD, True)
        examples.append({"config": cid, "caption": F.caption_run(cid, r), **r})
    fx = {
        "engine": VERSION,
        "stats": stats,
        "judge": J.judge_study(),
        "grading": J.grading_study(),
        "trajectory": T.trajectory_study(load_traces(), tok),
        "scaffold": {"configs": sc["configs"], "rows": sc["rows"], "spread": sc["spread"], "mu": sc["mu"],
                     "describe": {c["id"]: F.describe(c["id"]) for c in sc["configs"]},
                     "paired": [F.paired_scaffolds(sc, a, b) for a, b in PAIRS], "examples": examples,
                     "files": {t["name"]: F.files_for(t) for t in F.TASKS}},
        "cost": F.cost_study(sc),
        "cost_frames": {a: F.cost_frames(F.cost_study(sc), None if a == "all" else a) for a in ["all", "small", "large"]},
        "contamination": C.contamination_study(),
        "ngrams": C.ngrams([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 8),
    }
    return evals2_data(), fx, results_md(fx)


def _p(x: float, d: int = 1) -> str:
    return fixed(x * 100.0 + 0.0, d) + "%"


def _f(x: float, d: int = 3) -> str:
    return fixed(x + 0.0, d)


def results_md(fx: dict[str, Any]) -> str:
    jc = J.JUDGE
    out = [f"# Evals module results, part 2 (engine {VERSION})", "",
           "Written by `scripts/make_fixtures.py` (CI checks it is up to date). Every quality, bias, skill, rate, "
           "leak and task here is synthetic and illustrative; token counts and costs come from the harness's own "
           "accounting (the vendored Qwen2.5 tokenizer, the dated price table).", "",
           f"## Chapter 6: a simulated judge ({jc['pairs']} pairs, noise τ = {jc['tau']}, ties within "
           f"{jc['tie_band']} of quality)", "",
           "| biases | accuracy, one order | κ, one order | first shown wins | consistent across the swap | "
           "κ, swapped (ties when inconsistent) | accuracy where decided | longer-but-worse pairs right | "
           "own-family-but-worse pairs right |", "|---|---|---|---|---|---|---|---|---|"]
    for s in jc["order"]:
        r = fx["judge"]["settings"][s]
        b = r["bias"]
        out.append(f"| {s} (pos {b['position']}, len {b['verbosity']}, self {b['self']}) | {_p(r['accuracy1'])} | "
                   f"{_f(r['kappa1']['kappa'], 2)} | {_p(r['first_share'])} | {_p(r['consistency'])} | "
                   f"{_f(r['kappa_d']['kappa'], 2)} | {_p(r['accuracy_decided'])} | {_p(r['long_worse']['acc1'])} "
                   f"(n = {r['long_worse']['n']}) | {_p(r['own_worse']['acc1'])} (n = {r['own_worse']['n']}) |")
    g = fx["grading"]
    gc = g["cfg"]
    cr = g["corrected"]
    cal = g["calibration"]
    out += ["", f"Pass/fail grading ({gc['tasks']} tasks, skill {gc['theta']}; judge Se {gc['se']}, Sp {gc['sp']}): unit "
            f"tests {_p(g['tests_rate'])}, judge {_p(g['judge_rate'])}, κ {_f(g['kappa']['kappa'], 2)}. Calibration "
            f"({cal['n']} tasks labelled): Se {cal['se_k']}/{cal['se_n']}, Sp {cal['sp_k']}/{cal['sp_n']}. Corrected "
            f"{_p(cr['corrected'])}, 95% {_p(cr['ci'][0])} to {_p(cr['ci'][1])} (SE {_f(cr['stderr'], 4)}; "
            f"{_f(cr['naive_stderr'], 4)} if Se and Sp were known).", "",
            "## Chapter 7: grading trajectories", "",
            "| trajectory | status | outcome | self-report | trajectory | verified | wasted steps | unsafe run / stopped | "
            "process score | tokens |", "|---|---|---|---|---|---|---|---|---|---|"]
    yn = {True: "pass", False: "fail"}
    for r in fx["trajectory"]["rows"]:
        c = r["counts"]
        out.append(f"| {r['title']} | {r['status']} | {yn[r['grades']['outcome']]} | {yn[r['grades']['self_report']]} | "
                   f"{yn[r['grades']['trajectory']]} | {'yes' if r['verified'] else 'no'} | {r['waste']} | "
                   f"{c['unsafe_run']} / {c['unsafe_stopped']} | {_f(r['score'], 2)} | {r['tokens']:,} |")
    sc = fx["scaffold"]
    cf = F.SCAFFOLD
    out += ["", f"## Chapter 8: one model, many harnesses ({len(F.TASKS)} tasks × {cf['runs']} runs per configuration; "
            f"within-task correlation {cf['rho']}; malformed-output rates {cf['malformed']['native']} native, "
            f"{cf['malformed']['react']} ReAct)", "",
            "| agent | tools | style | turns | resolved | Wilson 95% | cluster 95% | model calls | malformed / run | "
            "tokens / run | $ / run | $ / solved |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sc["configs"]:
        out.append(f"| {s['agent']} | {s['tools']} | {s['style']} | {s['budget']} | {_p(s['rate'])} ({s['k']}/{s['n']}) | "
                   f"{_p(s['wilson'][0])} to {_p(s['wilson'][1])} | {_p(s['cluster'][0])} to {_p(s['cluster'][1])} | "
                   f"{_f(s['turns'], 2)} | {_f(s['malformed'], 2)} | {s['tokens']:,.0f} | {_f(s['cost_per_run'], 5)} | "
                   f"{_f(s['cost_per_solved'], 5)} |")
    out += ["", "Spread across scaffolds: " + "; ".join(
        f"{a}: {_p(v['min'])} to {_p(v['max'])} ({_f(v['range'] * 100.0, 1)} points)" for a, v in sc["spread"].items()) + ".",
        "", "| paired comparison | A only | B only | McNemar exact p | difference | paired 95% |", "|---|---|---|---|---|---|"]
    for p in sc["paired"]:
        c = p["compare"]
        out.append(f"| {p['a']} vs {p['b']} | {p['table']['n10']} | {p['table']['n01']} | {_f(p['mcnemar']['exact_p'], 4)} | "
                   f"{_p(c['delta'])} | {_p(c['ci_paired'][0])} to {_p(c['ci_paired'][1])} |")
    co = fx["cost"]
    out += ["", "## Chapter 10: cost", "", f"Pareto front (cheapest first): {', '.join(co['front'])}. Highest success: "
            f"{co['best']}. Fewest dollars per solved task: {co['cheapest_per_solved']}.", ""]
    ct = fx["contamination"]
    cc = ct["cfg"]
    out += [f"## Chapter 9: contamination ({cc['tasks']} tasks, skill {cc['theta']}, memorise {cc['memorise']}, "
            f"paraphrased share {cc['paraphrase']}, mutation {cc['mutate']}, {cc['ngram']}-grams flagged at "
            f"{cc['threshold']})", "",
            f"Chance an {cc['ngram']}-gram survives a paraphrase: {_f(ct['survive'], 4)}.", "",
            "| leak rate | leaked (verbatim / paraphrased) | flagged verbatim | flagged paraphrased | canary | score | "
            "clean score | after dropping flagged | before cutoff | after cutoff | split p |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, lv in ct["levels"].items():
        out.append(f"| {_p(lv['leak'], 0)} | {lv['leaked']} ({lv['verbatim']} / {lv['paraphrased']}) | {lv['flagged_verbatim']} | "
                   f"{lv['flagged_paraphrased']} | {'yes' if lv['canary'] else 'no'} | {_p(lv['rate'])} | {_p(lv['clean_rate'])} | "
                   f"{_p(lv['rate_kept'])} | {_p(lv['pre']['k'] / lv['pre']['n'])} | {_p(lv['post']['k'] / lv['post']['n'])} | "
                   f"{_f(lv['split']['p'], 4)} |")
    return "\n".join(out) + "\n"
