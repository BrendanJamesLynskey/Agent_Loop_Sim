"""The evals module (engine 1.8.0), written by make_fixtures.py: ts/src/evals_data.json (the
configurations and the SWE-style task, one source of truth for both languages),
fixtures/evals_fixtures.json (special functions, statistics, the generator, every chapter study and
the SWE-style anatomies, for the TS port to reproduce exactly) and fixtures/evals_results.md (the
recorded run every published number comes from)."""
from __future__ import annotations

from typing import Any

from agent_loop_sim import VERSION
from agent_loop_sim.context.text import fixed
from agent_loop_sim.evals import generator as G
from agent_loop_sim.evals import mathx as X
from agent_loop_sim.evals import stats as S
from agent_loop_sim.evals import study as St
from agent_loop_sim.evals import swe as W
from agent_loop_sim.rng import Rng

EXP_X = [0.0, 1e-30, -1e-30, 1e-9, 0.25, -0.25, 0.3465735902799726, 0.5, -0.5, 1.0, -1.0, 1.0397, 2.0, -2.0, 5.5,
         -7.25, 10.0, -20.0, 50.0, -100.0, 300.5, -700.0, 709.0, -740.0, -745.2]
LGAMMA_X = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 7.25, 10.0, 19.5, 20.0, 21.0, 100.0, 501.0, 1001.0, 12345.5]
BETA_CASES = [(1.0, 1.0, 0.3), (2.0, 5.0, 0.2), (5.0, 2.0, 0.9), (0.5, 0.5, 0.1), (10.0, 491.0, 0.02),
              (341.0, 160.0, 0.64), (342.0, 159.0, 0.72), (100.0, 100.0, 0.5), (1000.0, 3.0, 0.999), (3.0, 1000.0, 0.001)]
CDF_X = [-40.0, -10.0, -5.0, -3.5, -3.0, -2.5, -1.959963984540054, -1.0, -0.1, 0.0, 0.1, 1.0, 1.96, 2.5, 3.0, 3.01, 4.0,
         6.0, 8.5]
PPF_P = [1e-300, 1e-20, 1e-12, 1e-6, 0.001, 0.025, 0.05, 0.075, 0.1, 0.3, 0.5, 0.7, 0.8, 0.925, 0.95, 0.975, 0.999,
         0.999999, 1 - 1e-12]
NCK = [(10, 0), (10, 3), (10, 9), (10, 10), (20, 7), (100, 37), (200, 1), (200, 199), (5, 5)]
WIL = [(0, 10), (1, 10), (5, 10), (10, 10), (0, 1), (341, 500), (339, 500), (731, 1000), (3, 20), (0, 0)]
PMF = [(0, 10, 0.3), (3, 10, 0.3), (10, 10, 0.3), (250, 500, 0.5), (341, 500, 0.682), (1, 20, 0.0), (20, 20, 1.0)]
CLUSTERS = [([3, 1, 0, 2, 3], [3, 3, 3, 3, 3]), ([1, 0, 1, 1], [1, 1, 1, 1]), ([5, 0, 2], [5, 2, 4])]
MCN = [(0, 0), (5, 0), (0, 5), (12, 4), (49, 40), (56, 34), (120, 118), (1, 1)]
PAIRS = [([1, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0]),
         ([0.5, 1.0, 0.0, 0.75, 0.25], [0.25, 1.0, 0.25, 0.5, 0.0])]
POWER = [(0.6, 0.5), (0.7, 0.65), (0.682, 0.678), (0.3, 0.35)]
POWER_N = [50, 200, 500, 2000]
PAIRED_P = [(0.15, 0.05), (0.12, 0.08), (0.102, 0.098), (0.3, 0.1)]


def evals_data() -> dict[str, Any]:
    return {"version": VERSION, "population": G.POPULATION, "loop": St.LOOP, "passk": St.PASSK, "bias": St.BIAS,
            "unc": St.UNC, "cover": St.COVER, "tie": St.TIE, "paired": St.PAIRED, "sizes": St.SIZES, "alpha": S.ALPHA,
            "methods": S.METHODS,
            "swe": {"issue": W.ISSUE, "base": W.BASE, "test_patches": W.TEST_PATCHES, "patches": W.PATCHES,
                    "titles": W.TITLES, "fail_to_pass": W.FAIL_TO_PASS, "pass_to_pass": W.PASS_TO_PASS}}


def evals_fixtures(tok: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    pop = G.population()
    mathx = {
        "exp": [[x, X.exp(x)] for x in EXP_X],
        "lgamma": [[x, X.lgamma(x)] for x in LGAMMA_X],
        "betainc": [[a, b, x, X.betainc(a, b, x)] for a, b, x in BETA_CASES],
        "norm_cdf": [[x, X.norm_cdf(x)] for x in CDF_X],
        "norm_ppf": [[p, X.norm_ppf(p)] for p in PPF_P],
        "logistic": [[x, X.logistic(x)] for x in [-30.0, -2.0, 0.0, 0.7, 3.0, 40.0]],
        "zcrit": [[a, X.zcrit(a)] for a in [0.01, 0.05, 0.1, 0.32]],
    }
    stats = {
        "pass_at_k": [[n, c, k, S.pass_at_k(n, c, k)] for n, c in NCK for k in [1, 2, 5, 10]],
        "pass_hat_k": [[n, c, k, S.pass_hat_k(n, c, k)] for n, c in NCK for k in [1, 2, 5, 10]],
        "plugin": [[n, c, k, S.plugin_pass_at_k(n, c, k)] for n, c in NCK for k in [1, 2, 5]],
        "task_curves": [[mu, s, k, S.task_pass_curves(mu, s, k)] for mu in [0.1, 0.5, 0.9] for s in [0.0, 0.5, 1.0]
                        for k in [1, 3, 8]],
        "pmf": [[k, n, p, S.binom_pmf(k, n, p)] for k, n, p in PMF],
        "cdf": [[k, n, p, S.binom_cdf(k, n, p)] for k, n, p in PMF],
        "intervals": [[k, n, S.wald(k, n), S.wilson(k, n), S.exact(k, n)] for k, n in WIL],
        "exact_99": [[k, n, S.exact(k, n, 0.01)] for k, n in WIL],
        "quantile": [[q, S.quantile([1.0, 2.0, 4.0, 8.0, 9.0], q)] for q in [0.0, 0.025, 0.25, 0.5, 0.9, 0.975, 1.0]],
        "bootstrap": S.bootstrap([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0], 50, 7),
        "coverage": [[m, 30, p, S.coverage(m, 30, p)] for m in ["wald", "wilson", "exact"] for p in [0.05, 0.5, 0.93]],
        "cluster": [S.cluster_se(s, r) for s, r in CLUSTERS],
        "mcnemar": [[b, c, S.mcnemar(b, c)] for b, c in MCN],
        "table": [S.paired_table(a, b) for a, b in PAIRS[:1]],
        "compare": [S.compare(a, b) for a, b in PAIRS],
        "two": [S.two_proportion(341, 500, 339, 500), S.two_proportion(60, 100, 45, 100)],
        "power": {
            "margin": [[p, h, S.n_for_margin(p, h)] for p in [0.5, 0.8] for h in [0.05, 0.01]],
            "unpaired": [[p1, p2, n, S.power_unpaired(n, p1, p2)] for p1, p2 in POWER for n in POWER_N],
            "n_unpaired": [[p1, p2, S.n_unpaired(p1, p2), S.n_unpaired(p1, p2, 0.01, 0.9)] for p1, p2 in POWER],
            "paired": [[a, b, n, S.power_paired(n, a, b)] for a, b in PAIRED_P for n in POWER_N],
            "n_paired": [[a, b, S.n_paired(a, b), S.n_paired(a, b, 0.01, 0.9)] for a, b in PAIRED_P],
            "mde": [[n, S.mde_unpaired(n, 0.6), S.mde_paired(n, 0.2)] for n in POWER_N],
        },
    }
    bench = G.benchmark(0.3, 25, 4, 0.4, 99, pop)
    gen = {
        "population_head": pop[:60],
        "true_scores": [[th, G.true_score(pop, th)] for th in [-1.0, 0.0, 0.4, 0.6, 1.5]],
        "sample": G.sample(20, len(pop), Rng(5)),
        "runs": [[rho, G.runs([0.1, 0.5, 0.9, 0.3], 6, rho, Rng(11))] for rho in [0.0, 0.3, 1.0]],
        "benchmark": bench,
        "icc": G.icc(bench["runs"]),
    }
    unc = {f"{size}|{mode}": St.uncertainty_study(size, mode, St.UNC, pop) for size in St.UNC["sizes"]
           for mode in St.UNC["modes"]}
    cover = {str(n): St.coverage_curves(n, St.COVER["ps"]) for n in St.COVER["sizes"]}
    paired = {str(m): St.paired_study(m, St.PAIRED, pop) for m in St.PAIRED["runs"]}
    swe = {f"{p}|{t}": W.anatomy(p, t, tok) for p in W.PATCHES for t in W.TEST_PATCHES}
    fx = {
        "engine": VERSION,
        "mathx": mathx,
        "stats": stats,
        "generator": gen,
        "loop": St.loop_study(St.LOOP, pop),
        "passk": St.passk_study(St.PASSK, pop),
        "bias": St.bias_study(St.BIAS),
        "uncertainty": unc,
        "coverage": cover,
        "tie": St.tie_study(St.TIE),
        "sizes": St.sizes_table(St.SIZES),
        "paired": paired,
        "swe": swe,
    }
    return evals_data(), fx, results_md(fx)


def _f(x: float, d: int = 3) -> str:
    return fixed(x + 0.0, d)


def _p(x: float, d: int = 1) -> str:
    return St.pct(x, d)


def results_md(fx: dict[str, Any]) -> str:
    lp = fx["loop"]
    lc = lp["cfg"]
    last = lp["frames"][-1]
    pk = fx["passk"]
    b = fx["bias"]
    t = fx["tie"]
    out = [f"# Evals module results (engine {VERSION})", "",
           "Written by `scripts/make_fixtures.py` (CI checks it is up to date). Every difficulty, skill, correlation "
           "and benchmark here is synthetic and illustrative (see `evals/generator.py`); the statistics are exact "
           "under the stated assumptions.", "",
           f"Population: {G.POPULATION['size']} tasks, difficulty sd {G.POPULATION['difficulty_sd']}, discrimination "
           f"uniform on [{G.POPULATION['a_lo']}, {G.POPULATION['a_hi']}], seed {G.POPULATION['seed']}.", "",
           f"## Chapter 1: runs accumulating ({lc['tasks']} tasks, {lc['runs']} runs each, skill {lc['theta']}, "
           f"within-task correlation {lc['rho']})", "",
           "| after | runs | rate | Wilson 95% (runs independent) | cluster-robust 95% (tasks the unit) |", "|---|---|---|---|---|"]
    for i in [lc["tasks"] - 1, 2 * lc["tasks"] - 1, 3 * lc["tasks"] - 1]:
        f = lp["frames"][i]
        out.append(f"| run {f['run'] + 1} of every task | {f['n']} | {_p(f['p'])} | {_p(f['wilson'][0])} to "
                   f"{_p(f['wilson'][1])} | {_p(f['cluster'][0])} to {_p(f['cluster'][1])} |")
    out += ["", f"True score of these {lc['tasks']} tasks: {_p(lp['truth_tasks'])}; of the population: "
            f"{_p(lp['truth_population'])}. Design effect after all runs: "
            f"{_f((last['se'] / last['naive_se']) ** 2, 2)}.", "",
            f"## Chapter 3: pass@k and pass^k ({pk['cfg']['tasks']} tasks, n = {pk['cfg']['n']} samples, skill "
            f"{pk['cfg']['theta']}, correlation {pk['cfg']['rho']})", "",
            "| k | pass@k | true | plug-in | pass^k | true |", "|---|---|---|---|---|---|"]
    for f in pk["frames"]:
        out.append(f"| {f['k']} | {_p(f['pass_at_k'])} | {_p(f['true_at'])} | {_p(f['plugin'])} | {_p(f['pass_hat_k'])} | "
                   f"{_p(f['true_hat'])} |")
    out += ["", f"Bias check (one task, μ = {b['cfg']['mu']}, n = {b['cfg']['n']}, k = {b['cfg']['k']}, "
            f"{b['cfg']['reps']} repeats): unbiased pass@k averages {_f(b['unbiased'], 4)}, the plug-in "
            f"{_f(b['plugin'], 4)}, the truth is {_f(b['true_at'], 4)}; pass^k averages {_f(b['hat'], 5)} against "
            f"{_f(b['true_hat'], 5)}.", "",
            f"## Chapter 4: repeating the benchmark ({St.UNC['replicates']} repeats, skill {St.UNC['theta']}, one run "
            f"per task, bootstrap {St.UNC['boot']} resamples)", "",
            "| tasks | what changes | truth | Wald | Wilson | exact | bootstrap | mean Wilson width |", "|---|---|---|---|---|---|---|---|"]
    for k, u in fx["uncertainty"].items():
        c = u["coverage"]
        out.append(f"| {u['size']} | {'tasks and runs' if u['mode'] == 'tasks' else 'runs only'} | {_p(u['truth'])} | "
                   f"{_p(c['wald'], 0)} | {_p(c['wilson'], 0)} | {_p(c['exact'], 0)} | {_p(c['bootstrap'], 0)} | "
                   f"{_p(u['mean_width']['wilson'])} |")
    out += ["", "Exact coverage of nominal 95% intervals (minimum over p = 0.02 … 0.98):", "",
            "| n | Wald | Wilson | exact |", "|---|---|---|---|"]
    for n, c in fx["coverage"].items():
        out.append(f"| {n} | {_p(min(c['wald']))} | {_p(min(c['wilson']))} | {_p(min(c['exact']))} |")
    tt = t["test"]
    out += ["", f"The tie: {t['cfg']['k1']}/{t['cfg']['n']} ({_p(tt['p1'])}) against {t['cfg']['k2']}/{t['cfg']['n']} "
            f"({_p(tt['p2'])}): z = {_f(tt['z'], 3)}, p = {_f(tt['p'], 3)}; Wilson {_p(tt['wilson1'][0])} to "
            f"{_p(tt['wilson1'][1])} and {_p(tt['wilson2'][0])} to {_p(tt['wilson2'][1])}. Tasks per agent to detect "
            f"that difference at 80% power: {t['n_unpaired']:,} (unpaired), {t['n_paired']:,} (paired, discordance "
            f"{St.SIZES['psi']}). Smallest difference {t['cfg']['n']} tasks detect: {_p(t['mde_unpaired'])} unpaired, "
            f"{_p(t['mde_paired'])} paired.", "",
            "| margin ± | n at p = 0.5 | n at p = 0.8 |", "|---|---|---|"]
    m = fx["sizes"]["margins"]
    for h in St.SIZES["margins"]:
        a = next(x for x in m if x["h"] == h and x["p"] == 0.5)
        c = next(x for x in m if x["h"] == h and x["p"] == 0.8)
        out.append(f"| {_p(h)} | {a['n']:,} | {c['n']:,} |")
    out += ["", f"| difference (base {St.SIZES['base']}) | tasks per agent, unpaired | shared tasks, paired (ψ = "
            f"{St.SIZES['psi']}) |", "|---|---|---|"]
    for d in fx["sizes"]["diffs"]:
        out.append(f"| {_p(d['delta'])} | {d['unpaired']:,} | {d['paired']:,} |")
    pc = St.PAIRED
    out += ["", f"## Chapter 5: two agents on the same {pc['tasks']} tasks (skills {pc['theta_a']} and {pc['theta_b']}, "
            f"correlation {pc['rho']})", "",
            "| runs per task | A | B | difference | paired 95% | unpaired 95% | McNemar p (first runs) | SE² tasks (paired) | SE² tasks (unpaired) | SE² runs |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k, p in fx["paired"].items():
        f = p["frames"][-1]
        c = f["compare"]
        sp = p["split"]
        out.append(f"| {k} | {_p(c['a'])} | {_p(c['b'])} | {_p(c['delta'])} | {_p(c['ci_paired'][0])} to "
                   f"{_p(c['ci_paired'][1])} | {_p(c['ci_unpaired'][0])} to {_p(c['ci_unpaired'][1])} | "
                   f"{_f(f['mcnemar']['exact_p'], 4)} | {sp['tasks_paired']:.2e} | {sp['tasks_unpaired']:.2e} | "
                   f"{sp['runs']:.2e} |")
    out += ["", f"True difference on these tasks: {_p(fx['paired']['1']['truth'])}.", "",
            "## Chapter 2: one SWE-bench-style task", "", "| patch | hidden tests | FAIL_TO_PASS | PASS_TO_PASS | resolved |",
            "|---|---|---|---|---|"]
    for k, a in fx["swe"].items():
        g = a["grade"]
        out.append(f"| {W.TITLES[a['patch']]} | {a['test_patch']} | {'pass' if g['f2p_ok'] else 'fail'} | "
                   f"{'pass' if g['p2p_ok'] else 'fail'} | {'yes' if g['resolved'] else 'no'} |")
    return "\n".join(out) + "\n"
