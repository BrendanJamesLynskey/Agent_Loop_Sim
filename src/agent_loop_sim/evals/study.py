"""What the Evals site's chapters animate (since 1.8): each study runs the generator and the statistics
under a configuration and returns the frames, with a caption per frame.

- ``loop_study`` (chapter 1): runs accumulate on a 40-task benchmark; after each run, the rate and
  two 95% intervals: Wilson's, as if every run were independent, and the cluster-robust one that
  treats tasks as the unit.
- ``passk_study`` (chapter 3): n samples on each of 30 tasks; for k = 1 … n, the unbiased pass@k and
  pass^k on each task and their means, against the true curves of the generator.
- ``uncertainty_study`` (chapter 4): the benchmark repeated (new tasks and new runs, or the same tasks
  with new runs); each repeat's four intervals and whether they cover the truth.
- ``coverage_curves`` (chapter 4): the exact coverage of the Wald, Wilson and exact intervals.
- ``paired_study`` (chapter 5): two agents on the same tasks, tasks added in batches; the paired and
  unpaired intervals for the difference, McNemar's table and test on the first runs, and the variance
  split between tasks and runs.

Every configuration value is illustrative.
"""
from __future__ import annotations

import math
from typing import Any

from ..context.text import fixed
from ..rng import Rng
from . import generator as G
from . import stats as S
from .mathx import zcrit

LOOP = {"theta": 0.6, "tasks": 40, "runs": 3, "rho": 0.5, "seed": 2511}
PASSK = {"theta": -0.2, "tasks": 30, "n": 10, "rho": 0.3, "seed": 2531}
BIAS = {"mu": 0.3, "n": 10, "k": 5, "reps": 4000, "seed": 2533}
UNC = {"theta": 0.4, "sizes": [50, 200, 500], "modes": ["tasks", "runs"], "replicates": 30, "boot": 200, "seed": 2541}
COVER = {"sizes": [20, 100], "ps": [round(0.02 * i, 2) for i in range(1, 50)]}
TIE = {"k1": 341, "k2": 339, "n": 500}
PAIRED = {"theta_a": 0.45, "theta_b": 0.1, "tasks": 300, "runs": [1, 3, 5], "rho": 0.5, "batch": 10, "seed": 2551}
SIZES = {"margins": [0.05, 0.03, 0.02, 0.01], "margin_ps": [0.5, 0.8], "deltas": [0.1, 0.05, 0.03, 0.01],
         "base": 0.6, "psi": 0.2}


def pct(x: float, d: int = 1) -> str:
    return fixed(x * 100.0 + 0.0, d) + "%"


def _clip(lo: float, hi: float) -> list[float]:
    return [lo if lo > 0.0 else 0.0, hi if hi < 1.0 else 1.0]


# ── chapter 1 ────────────────────────────────────────────────────────────


def loop_study(cfg: dict[str, Any] = LOOP, pop: list[dict[str, float]] | None = None) -> dict[str, Any]:
    p = pop if pop is not None else G.population()
    t_n = cfg["tasks"]
    m = cfg["runs"]
    idx = G.sample(t_n, len(p), Rng(cfg["seed"]))
    mu = G.mus([p[i] for i in idx], cfg["theta"])
    rows = G.runs(mu, m, cfg["rho"], Rng(cfg["seed"] + 1))
    z = zcrit(S.ALPHA)
    succ = [0] * t_n
    seen = [0] * t_n
    k = 0
    n = 0
    frames = []
    for j in range(m):
        for t in range(t_n):
            x = rows[t][j]
            succ[t] += x
            seen[t] += 1
            k += x
            n += 1
            ss = [succ[i] for i in range(t_n) if seen[i] > 0]
            rr = [seen[i] for i in range(t_n) if seen[i] > 0]
            cl = S.cluster_se(ss, rr)
            wil = S.wilson(k, n)
            band = _clip(cl["p"] - z * cl["se"], cl["p"] + z * cl["se"]) if cl["tasks"] > 1 else [0.0, 1.0]
            cap = (f"Run {j + 1} on task {t + 1}: {'pass' if x == 1 else 'fail'}. {k} of {n} runs pass ({pct(k / n)}). "
                   f"95% interval {pct(wil[0])} to {pct(wil[1])} if every run were independent; "
                   f"{pct(band[0])} to {pct(band[1])} with tasks as the unit.")
            frames.append({"task": t, "run": j, "x": x, "k": k, "n": n, "p": k / n, "wilson": wil, "cluster": band,
                           "se": cl["se"], "naive_se": cl["naive_se"], "caption": cap})
    return {"cfg": cfg, "idx": idx, "mu": mu, "runs": rows, "truth_tasks": S.mean(mu),
            "truth_population": G.true_score(p, cfg["theta"]), "frames": frames}


# ── chapter 3 ────────────────────────────────────────────────────────────


def passk_study(cfg: dict[str, Any] = PASSK, pop: list[dict[str, float]] | None = None) -> dict[str, Any]:
    p = pop if pop is not None else G.population()
    n = cfg["n"]
    idx = G.sample(cfg["tasks"], len(p), Rng(cfg["seed"]))
    mu = G.mus([p[i] for i in idx], cfg["theta"])
    rows = G.runs(mu, n, cfg["rho"], Rng(cfg["seed"] + 1))
    c = [int(S.total([float(x) for x in r])) for r in rows]
    s = math.sqrt(cfg["rho"])
    frames = []
    for k in range(1, n + 1):
        at = [S.pass_at_k(n, ci, k) for ci in c]
        hat = [S.pass_hat_k(n, ci, k) for ci in c]
        plug = [S.plugin_pass_at_k(n, ci, k) for ci in c]
        true_at = S.mean([S.task_pass_curves(u, s, k)["pass_at_k"] for u in mu])
        true_hat = S.mean([S.task_pass_curves(u, s, k)["pass_hat_k"] for u in mu])
        a = S.mean(at)
        h = S.mean(hat)
        cap = (f"k = {k}: pass@{k} = {pct(a)} (some one of {k} tries passes), pass^{k} = {pct(h)} (all {k} pass). "
               f"True values for these tasks: {pct(true_at)} and {pct(true_hat)}.")
        frames.append({"k": k, "at": at, "hat": hat, "plug": plug, "pass_at_k": a, "pass_hat_k": h,
                       "plugin": S.mean(plug), "true_at": true_at, "true_hat": true_hat, "caption": cap})
    return {"cfg": cfg, "idx": idx, "mu": mu, "runs": rows, "c": c, "frames": frames}


def bias_study(cfg: dict[str, Any] = BIAS) -> dict[str, Any]:
    """One task with success probability mu, sampled n times, ``reps`` times over (independent runs):
    the mean of the unbiased pass@k, of the plug-in, and of pass^k, against the truth."""
    r = Rng(cfg["seed"])
    n = cfg["n"]
    k = cfg["k"]
    mu = cfg["mu"]
    sa = 0.0
    sp = 0.0
    sh = 0.0
    for _ in range(cfg["reps"]):
        c = 0
        for _ in range(n):
            if r.random() < mu:
                c += 1
        sa = sa + S.pass_at_k(n, c, k)
        sp = sp + S.plugin_pass_at_k(n, c, k)
        sh = sh + S.pass_hat_k(n, c, k)
    reps = cfg["reps"]
    truth = S.task_pass_curves(mu, 0.0, k)
    return {"cfg": cfg, "unbiased": sa / reps, "plugin": sp / reps, "hat": sh / reps,
            "true_at": truth["pass_at_k"], "true_hat": truth["pass_hat_k"]}


# ── chapter 4 ────────────────────────────────────────────────────────────


def uncertainty_study(size: int, mode: str, cfg: dict[str, Any] = UNC,
                      pop: list[dict[str, float]] | None = None) -> dict[str, Any]:
    """``replicates`` repeats of a ``size``-task benchmark, one run per task. ``mode`` "tasks": each
    repeat samples new tasks (the truth is the population's mean); "runs": the same tasks, new runs
    (the truth is those tasks' mean). Streams: the fixed tasks ``seed``, repeat i's tasks
    ``seed + 100 + i``, its runs ``seed + 1000 + i``, its bootstrap ``seed + 5000 + i``."""
    p = pop if pop is not None else G.population()
    theta = cfg["theta"]
    mu0 = G.mus([p[i] for i in G.sample(size, len(p), Rng(cfg["seed"]))], theta)
    truth = G.true_score(p, theta) if mode == "tasks" else S.mean(mu0)
    covered = {m: 0 for m in S.METHODS}
    width = {m: 0.0 for m in S.METHODS}
    frames = []
    for i in range(cfg["replicates"]):
        mu = G.mus([p[j] for j in G.sample(size, len(p), Rng(cfg["seed"] + 100 + i))], theta) if mode == "tasks" else mu0
        x = [row[0] for row in G.runs(mu, 1, 0.0, Rng(cfg["seed"] + 1000 + i))]
        k = 0
        for v in x:
            k += v
        ivs = {"wald": S.wald(k, size), "wilson": S.wilson(k, size), "exact": S.exact(k, size),
               "bootstrap": S.bootstrap([float(v) for v in x], cfg["boot"], cfg["seed"] + 5000 + i)["ci"]}
        hit = {}
        for m in S.METHODS:
            lo, hi = ivs[m]
            hit[m] = lo <= truth <= hi
            if hit[m]:
                covered[m] += 1
            width[m] = width[m] + (hi - lo)
        missed = [m for m in S.METHODS if not hit[m]]
        cap = (f"Repeat {i + 1}: {k} of {size} tasks pass ({pct(k / size)}). Wilson: {pct(ivs['wilson'][0])} to "
               f"{pct(ivs['wilson'][1])}. " + ("Every interval covers the truth." if not missed else
                                                 "Missed the truth: " + ", ".join(missed) + "."))
        frames.append({"k": k, "p": k / size, "ivs": ivs, "hit": hit, "covered": dict(covered),
                       "caption": cap})
    reps = cfg["replicates"]
    return {"size": size, "mode": mode, "truth": truth, "frames": frames,
            "coverage": {m: covered[m] / reps for m in S.METHODS},
            "mean_width": {m: width[m] / reps for m in S.METHODS}}


def coverage_curves(n: int, ps: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": n, "ps": ps}
    for m in ["wald", "wilson", "exact"]:
        ivs = [S.interval(m, k, n) for k in range(n + 1)]
        out[m] = [S.coverage(m, n, p, S.ALPHA, ivs) for p in ps]
    return out


def tie_study(cfg: dict[str, Any] = TIE) -> dict[str, Any]:
    n = cfg["n"]
    two = S.two_proportion(cfg["k1"], n, cfg["k2"], n)
    p1 = cfg["k1"] / n
    p2 = cfg["k2"] / n
    return {"cfg": cfg, "test": two, "n_unpaired": S.n_unpaired(p1, p2), "mde_unpaired": S.mde_unpaired(n, p1),
            "mde_paired": S.mde_paired(n, SIZES["psi"]),
            "n_paired": S.n_paired(SIZES["psi"] / 2 + (p1 - p2) / 2, SIZES["psi"] / 2 - (p1 - p2) / 2)}


def sizes_table(cfg: dict[str, Any] = SIZES) -> dict[str, Any]:
    margins = [{"p": p, "h": h, "n": S.n_for_margin(p, h)} for p in cfg["margin_ps"] for h in cfg["margins"]]
    base = cfg["base"]
    psi = cfg["psi"]
    diffs = []
    for d in cfg["deltas"]:
        diffs.append({"delta": d, "unpaired": S.n_unpaired(base + d, base), "paired": S.n_paired(psi / 2 + d / 2, psi / 2 - d / 2)})
    return {"cfg": cfg, "margins": margins, "diffs": diffs}


# ── chapter 5 ────────────────────────────────────────────────────────────


def variance_split(mu_a: list[float], mu_b: list[float], m: int, rho: float) -> dict[str, float]:
    """SE² of the difference in mean score on T tasks (tasks drawn from the population, m runs each),
    split into the part from which tasks were drawn and the part from the runs:
    paired   [Var_t(μa − μb) + E(wa + wb)] / T,  unpaired [Var_t(μa) + Var_t(μb) + E(wa + wb)] / T,
    with w = μ(1 − μ)(1 + (m − 1)ρ)/m the variance of one task's mean over its runs."""
    t = len(mu_a)
    d = [mu_a[i] - mu_b[i] for i in range(t)]
    w = [(mu_a[i] * (1.0 - mu_a[i]) + mu_b[i] * (1.0 - mu_b[i])) * (1.0 + (m - 1) * rho) / m for i in range(t)]
    within = S.mean(w) / t
    return {"tasks_paired": S.var(d, 0) / t, "tasks_unpaired": (S.var(mu_a, 0) + S.var(mu_b, 0)) / t,
            "runs": within, "se_paired": math.sqrt(S.var(d, 0) / t + within),
            "se_unpaired": math.sqrt((S.var(mu_a, 0) + S.var(mu_b, 0)) / t + within)}


def paired_study(m: int, cfg: dict[str, Any] = PAIRED, pop: list[dict[str, float]] | None = None) -> dict[str, Any]:
    p = pop if pop is not None else G.population()
    t_n = cfg["tasks"]
    idx = G.sample(t_n, len(p), Rng(cfg["seed"]))
    tasks = [p[i] for i in idx]
    mu_a = G.mus(tasks, cfg["theta_a"])
    mu_b = G.mus(tasks, cfg["theta_b"])
    ra = G.runs(mu_a, m, cfg["rho"], Rng(cfg["seed"] + 1))
    rb = G.runs(mu_b, m, cfg["rho"], Rng(cfg["seed"] + 2))
    a = [S.mean([float(x) for x in r]) for r in ra]
    b = [S.mean([float(x) for x in r]) for r in rb]
    a1 = [r[0] for r in ra]
    b1 = [r[0] for r in rb]
    frames = []
    for t in range(cfg["batch"], t_n + 1, cfg["batch"]):
        c = S.compare(a[:t], b[:t])
        tab = S.paired_table(a1[:t], b1[:t])
        mc = S.mcnemar(tab["n10"], tab["n01"])
        cap = (f"{t} tasks: A {pct(c['a'])}, B {pct(c['b'])}, difference {pct(c['delta'])}. Paired 95% interval "
               f"{pct(c['ci_paired'][0])} to {pct(c['ci_paired'][1])}; unpaired {pct(c['ci_unpaired'][0])} to "
               f"{pct(c['ci_unpaired'][1])}. First runs: A alone passes {tab['n10']}, B alone {tab['n01']}; "
               f"McNemar p = {fixed(mc['exact_p'], 4)}.")
        frames.append({"t": t, "compare": c, "table": tab, "mcnemar": mc, "caption": cap})
    return {"cfg": cfg, "m": m, "idx": idx, "mu_a": mu_a, "mu_b": mu_b, "a": a, "b": b, "a1": a1, "b1": b1,
            "truth": S.mean(mu_a) - S.mean(mu_b), "split": variance_split(mu_a, mu_b, m, cfg["rho"]), "frames": frames}
