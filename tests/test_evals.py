"""The evals module (engine 1.8.0): special functions, estimators, intervals, clustered errors, paired
tests and power against scipy, statsmodels or a closed form; the generator's correlation; the chapter
studies' invariants; the SWE-style task's grading; the committed fixtures."""
from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats as sst
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.regression.linear_model import OLS

from agent_loop_sim import VERSION
from agent_loop_sim.evals import generator as G
from agent_loop_sim.evals import mathx as X
from agent_loop_sim.evals import stats as S
from agent_loop_sim.evals import study as St
from agent_loop_sim.evals import swe as W
from agent_loop_sim.rng import Rng

ROOT = Path(__file__).resolve().parent.parent
RND = random.Random(25)


# ── special functions ────────────────────────────────────────────────────


def test_exp_matches_libm():
    xs = [RND.uniform(-745, 709) for _ in range(5000)] + [RND.uniform(-2, 2) for _ in range(5000)] + [0.0, 1e-20, -1e-20]
    for x in xs:
        want = math.exp(x)
        if want > 1e-300:
            assert X.exp(x) == pytest.approx(want, rel=1e-15)
    assert X.exp(-800.0) == 0.0 and X.exp(800.0) == math.inf and math.isnan(X.exp(math.nan))


def test_lgamma_matches_libm():
    for x in [RND.uniform(0.01, 30) for _ in range(2000)] + [RND.uniform(30, 1e5) for _ in range(2000)]:
        assert X.lgamma(x) == pytest.approx(math.lgamma(x), rel=1e-13, abs=1e-13)


def test_betainc_matches_scipy():
    for _ in range(2000):
        a, b, x = RND.randint(1, 2000), RND.randint(1, 2000), RND.random()
        assert X.betainc(a, b, x) == pytest.approx(special.betainc(a, b, x), rel=1e-9, abs=1e-11)
    for a, b, x in [(0.5, 0.5, 0.2), (2.5, 7.25, 0.4), (30.5, 2.0, 0.97)]:
        assert X.betainc(a, b, x) == pytest.approx(special.betainc(a, b, x), rel=1e-12)


def test_normal_cdf_and_quantile_match_scipy():
    for x in [RND.uniform(-37, 37) for _ in range(5000)]:
        assert X.norm_cdf(x) == pytest.approx(sst.norm.cdf(x), rel=1e-12, abs=1e-300)
        assert X.norm_sf(x) == pytest.approx(sst.norm.sf(x), rel=1e-12, abs=1e-300)
    for p in [RND.random() for _ in range(5000)] + [1e-300, 1e-15, 1 - 1e-15, 0.025, 0.975]:
        assert X.norm_ppf(p) == pytest.approx(sst.norm.ppf(p), rel=1e-15, abs=1e-15)
    assert X.zcrit(0.05) == pytest.approx(1.959963984540054, rel=1e-15)
    assert X.norm_ppf(0.0) == -math.inf and X.norm_ppf(1.0) == math.inf


# ── pass@k and pass^k ────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 5, 10, 20, 100])
def test_pass_at_k_and_hat_k_are_the_combinatorial_ratios(n):
    for c in range(n + 1):
        for k in range(1, n + 1):
            want_at = 1 - Fraction(comb(n - c, k), comb(n, k))
            want_hat = Fraction(comb(c, k), comb(n, k))
            assert S.pass_at_k(n, c, k) == pytest.approx(float(want_at), rel=1e-12, abs=1e-15)
            assert S.pass_hat_k(n, c, k) == pytest.approx(float(want_hat), rel=1e-12, abs=1e-15)


def test_pass_at_k_matches_the_papers_numpy_form():
    def paper(n, c, k):  # Chen et al. 2021, Figure 3
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    for n, c, k in [(10, 3, 1), (10, 3, 5), (200, 13, 100), (200, 1, 1), (50, 49, 2)]:
        assert S.pass_at_k(n, c, k) == pytest.approx(paper(n, c, k), rel=1e-13)


def test_task_curves_closed_form_against_enumeration():
    # enumerate every outcome of Z and the copy/own draws for k runs
    for mu in [0.2, 0.7]:
        for s in [0.0, 0.4, 1.0]:
            for k in [1, 2, 4]:
                p_fail = 0.0
                p_pass = 0.0
                for z, pz in [(1, mu), (0, 1 - mu)]:
                    one_pass = s * z + (1 - s) * mu
                    p_pass += pz * one_pass**k
                    p_fail += pz * (1 - one_pass) ** k
                got = S.task_pass_curves(mu, s, k)
                assert got["pass_at_k"] == pytest.approx(1 - p_fail, rel=1e-12)
                assert got["pass_hat_k"] == pytest.approx(p_pass, rel=1e-12)


def test_unbiased_estimators_are_unbiased_and_plugin_is_not():
    # exact expectation over c ~ Binomial(n, mu)
    n, k, mu = 10, 5, 0.3
    e_at = sum(sst.binom.pmf(c, n, mu) * S.pass_at_k(n, c, k) for c in range(n + 1))
    e_hat = sum(sst.binom.pmf(c, n, mu) * S.pass_hat_k(n, c, k) for c in range(n + 1))
    e_plug = sum(sst.binom.pmf(c, n, mu) * S.plugin_pass_at_k(n, c, k) for c in range(n + 1))
    assert e_at == pytest.approx(1 - (1 - mu) ** k, rel=1e-12)
    assert e_hat == pytest.approx(mu**k, rel=1e-12)
    assert e_plug < 1 - (1 - mu) ** k - 0.05
    b = St.bias_study()
    se = 0.5 / math.sqrt(b["cfg"]["reps"])
    assert abs(b["unbiased"] - b["true_at"]) < 4 * se
    assert b["true_at"] - b["plugin"] > 5 * se


# ── binomial and intervals ───────────────────────────────────────────────


def test_binomial_pmf_cdf_match_scipy():
    for n in [1, 7, 50, 500, 3000]:
        for p in [0.01, 0.3, 0.5, 0.682, 0.99]:
            for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
                assert S.binom_pmf(k, n, p) == pytest.approx(sst.binom.pmf(k, n, p), rel=1e-9, abs=1e-300)
                assert S.binom_cdf(k, n, p) == pytest.approx(sst.binom.cdf(k, n, p), rel=1e-9, abs=1e-14)


def test_intervals_match_statsmodels():
    for k, n in [(0, 10), (1, 10), (5, 10), (10, 10), (3, 20), (341, 500), (731, 1000), (999, 1000), (0, 1)]:
        for alpha in [0.05, 0.01]:
            assert S.wald(k, n, alpha) == pytest.approx(list(np.clip(proportion_confint(k, n, alpha, "normal"), 0, 1)), abs=1e-14)
            assert S.wilson(k, n, alpha) == pytest.approx(list(proportion_confint(k, n, alpha, "wilson")), abs=1e-14)
            assert S.exact(k, n, alpha) == pytest.approx(list(proportion_confint(k, n, alpha, "beta")), abs=1e-11)
            if 0 < k < n:
                assert S.exact(k, n, alpha)[0] == pytest.approx(sst.beta.ppf(alpha / 2, k, n - k + 1), rel=1e-10)


def test_quantile_and_bootstrap_match_numpy():
    xs = [RND.random() for _ in range(37)]
    srt = sorted(xs)
    for q in [0.0, 0.025, 0.3, 0.5, 0.975, 1.0]:
        assert S.quantile(srt, q) == pytest.approx(float(np.percentile(xs, 100 * q)), rel=1e-15, abs=1e-16)
    vals = [float(RND.random() < 0.6) for _ in range(80)]
    b = S.bootstrap(vals, 300, 17)
    assert b["ci"] == pytest.approx([float(np.percentile(b["means"], 2.5)), float(np.percentile(b["means"], 97.5))], rel=1e-15)
    assert b["se"] == pytest.approx(float(np.std(b["means"], ddof=1)), rel=1e-12)
    # the resamples really are draws of the tasks, in order
    r = Rng(17)
    first = sum(vals[r.randint(0, 79)] for _ in range(80)) / 80
    assert b["means"][0] == pytest.approx(first, rel=1e-15)
    # and the bootstrap SE approximates the binomial SE
    p = sum(vals) / 80
    assert b["se"] == pytest.approx(math.sqrt(p * (1 - p) / 80), rel=0.15)


def test_coverage_is_the_binomial_sum_and_exact_never_undercovers():
    for n in [10, 30]:
        for m in ["wald", "wilson", "exact"]:
            ivs = [S.interval(m, k, n) for k in range(n + 1)]
            for p in [0.03, 0.2, 0.5, 0.91]:
                want = sum(sst.binom.pmf(k, n, p) for k in range(n + 1) if ivs[k][0] <= p <= ivs[k][1])
                assert S.coverage(m, n, p, 0.05, ivs) == pytest.approx(want, rel=1e-9)
    c = St.coverage_curves(20, St.COVER["ps"])
    assert min(c["exact"]) >= 0.95
    assert min(c["wald"]) < 0.5 < min(c["wilson"])


# ── clustered data ───────────────────────────────────────────────────────


def test_cluster_se_matches_statsmodels_cluster_robust():
    rows = G.runs([0.2, 0.5, 0.9, 0.6, 0.4, 0.75, 0.1, 0.85], 5, 0.5, Rng(3))
    y = np.array([x for r in rows for x in r], dtype=float)
    g = np.array([i for i, r in enumerate(rows) for _ in r])
    fit = OLS(y, np.ones_like(y)).fit(cov_type="cluster", cov_kwds={"groups": g})
    got = S.cluster_se([sum(r) for r in rows], [len(r) for r in rows])
    assert got["p"] == pytest.approx(float(fit.params[0]), rel=1e-14)
    assert got["se"] == pytest.approx(float(fit.bse[0]), rel=1e-12)
    assert got["naive_se"] == pytest.approx(math.sqrt(got["p"] * (1 - got["p"]) / len(y)), rel=1e-15)


def test_generator_correlation_is_rho():
    for rho in [0.0, 0.3, 0.8]:
        rows = G.runs([0.4] * 6000, 4, rho, Rng(8))
        assert G.icc(rows) == pytest.approx(rho, abs=0.03)
        rate = sum(sum(r) for r in rows) / (6000 * 4)
        assert rate == pytest.approx(0.4, abs=0.015)


def test_generator_population_is_logistic_irt():
    pop = G.population()
    bs = [t["b"] for t in pop]
    assert abs(np.mean(bs)) < 0.1 and np.std(bs) == pytest.approx(G.POPULATION["difficulty_sd"], rel=0.05)
    assert all(G.POPULATION["a_lo"] <= t["a"] <= G.POPULATION["a_hi"] for t in pop)
    t = pop[0]
    assert G.success_p(t, 0.3) == pytest.approx(1 / (1 + math.exp(-t["a"] * (0.3 - t["b"]))), rel=1e-15)
    # higher skill, higher score
    assert G.true_score(pop, 0.0) < G.true_score(pop, 0.5) < G.true_score(pop, 1.0)


def test_design_effect_predicts_cluster_variance():
    # one task type (equal mu), m runs, icc rho: the cluster SE^2 / naive SE^2 is about 1 + (m-1) rho
    rows = G.runs([0.5] * 4000, 5, 0.4, Rng(21))
    got = S.cluster_se([sum(r) for r in rows], [5] * 4000)
    assert got["design_effect"] == pytest.approx(S.design_effect(5, 0.4), rel=0.08)


# ── comparing two agents ─────────────────────────────────────────────────


def test_mcnemar_matches_statsmodels():
    for b, c in [(5, 0), (12, 4), (49, 40), (56, 34), (120, 118), (1, 1), (0, 3)]:
        got = S.mcnemar(b, c)
        ex = sm_mcnemar([[10, b], [c, 10]], exact=True)
        ch = sm_mcnemar([[10, b], [c, 10]], exact=False, correction=True)
        assert got["exact_p"] == pytest.approx(ex.pvalue, rel=1e-9)
        assert got["exact_p"] == pytest.approx(min(1.0, sst.binomtest(min(b, c), b + c, 0.5).pvalue), rel=1e-9)
        assert got["chi2"] == pytest.approx(ch.statistic, rel=1e-12)
        assert got["chi2_p"] == pytest.approx(ch.pvalue, rel=1e-10)
    assert S.mcnemar(0, 0)["exact_p"] == 1.0


def test_compare_matches_scipy_t_statistics():
    a = [RND.random() for _ in range(60)]
    b = [x * 0.7 + 0.3 * RND.random() for x in a]
    c = S.compare(a, b)
    assert c["z_paired"] == pytest.approx(sst.ttest_rel(a, b).statistic, rel=1e-12)
    assert c["delta"] / c["se_unpaired"] == pytest.approx(sst.ttest_ind(a, b, equal_var=False).statistic, rel=1e-12)
    assert c["corr"] == pytest.approx(float(np.corrcoef(a, b)[0, 1]), rel=1e-12)
    assert c["se_paired"] < c["se_unpaired"]


def test_two_proportion_matches_statsmodels():
    for k1, k2, n in [(341, 339, 500), (60, 45, 100), (12, 30, 40)]:
        z, p = proportions_ztest([k1, k2], [n, n])
        got = S.two_proportion(k1, n, k2, n)
        assert got["z"] == pytest.approx(z, rel=1e-12)
        assert got["p"] == pytest.approx(p, rel=1e-10)


# ── power and sample size ────────────────────────────────────────────────


def test_sample_sizes_are_the_smallest_meeting_the_power():
    for p1, p2 in [(0.6, 0.5), (0.7, 0.65), (0.3, 0.35)]:
        n = S.n_unpaired(p1, p2)
        assert S.power_unpaired(n, p1, p2) >= 0.8 - 1e-12 > S.power_unpaired(n - 1, p1, p2)
    for a, b in [(0.15, 0.05), (0.12, 0.08), (0.3, 0.1)]:
        n = S.n_paired(a, b)
        assert S.power_paired(n, a, b) >= 0.8 - 1e-12 > S.power_paired(n - 1, a, b)
    # the familiar textbook value: ± 3 points at 95% needs 1,068
    assert S.n_for_margin(0.5, 0.03) == 1068 and S.n_for_margin(0.5, 0.05) == 385


def test_unpaired_power_by_simulation():
    rng = np.random.default_rng(4)
    n, p1, p2, reps = 300, 0.6, 0.5, 4000
    k1 = rng.binomial(n, p1, reps)
    k2 = rng.binomial(n, p2, reps)
    pp = (k1 + k2) / (2 * n)
    z = (k1 - k2) / n / np.sqrt(pp * (1 - pp) * 2 / n)
    sim = float(np.mean(np.abs(z) > X.zcrit(0.05)))
    assert sim == pytest.approx(S.power_unpaired(n, p1, p2), abs=4 * math.sqrt(0.25 / reps))


def test_paired_power_by_simulation():
    rng = np.random.default_rng(5)
    n, p10, p01, reps = 200, 0.12, 0.05, 4000
    cells = rng.multinomial(n, [p10, p01, 1 - p10 - p01], reps)
    b, c = cells[:, 0], cells[:, 1]
    d = b - c
    stat = np.where(b + c > 0, d / np.sqrt(np.maximum(b + c, 1)), 0)
    sim = float(np.mean(np.abs(stat) > X.zcrit(0.05)))
    assert sim == pytest.approx(S.power_paired(n, p10, p01), abs=0.04)


# ── chapter studies ──────────────────────────────────────────────────────


def test_loop_study_bands():
    s = St.loop_study()
    f = s["frames"]
    assert len(f) == St.LOOP["tasks"] * St.LOOP["runs"]
    last = f[-1]
    assert last["k"] == sum(sum(r) for r in s["runs"])
    # extra runs on the same tasks shrink the naive band faster than the clustered one
    one = f[St.LOOP["tasks"] - 1]
    w = lambda iv: iv[1] - iv[0]  # noqa: E731
    assert w(last["wilson"]) / w(one["wilson"]) < w(last["cluster"]) / w(one["cluster"])
    assert last["se"] > last["naive_se"]


def test_passk_study_curves_diverge():
    s = St.passk_study()
    fr = s["frames"]
    assert fr[0]["pass_at_k"] == fr[0]["pass_hat_k"]
    for a, b in zip(fr, fr[1:]):
        assert b["pass_at_k"] >= a["pass_at_k"] and b["pass_hat_k"] <= a["pass_hat_k"]
        assert b["true_at"] >= a["true_at"] and b["true_hat"] <= a["true_hat"]


def test_uncertainty_study_is_consistent():
    u = St.uncertainty_study(200, "tasks")
    assert u["truth"] == pytest.approx(G.true_score(G.population(), St.UNC["theta"]))
    for f in u["frames"]:
        assert f["ivs"]["wilson"] == S.wilson(f["k"], 200)
    assert all(0.8 <= v <= 1.0 for v in u["coverage"].values())


def test_paired_study_variance_split_by_simulation():
    """The closed-form SE of the paired difference matches the spread over fresh runs on the same
    tasks plus fresh task samples (here: fresh runs only, so compare the runs part)."""
    pop = G.population()
    cfg = St.PAIRED
    idx = G.sample(cfg["tasks"], len(pop), Rng(cfg["seed"]))
    tasks = [pop[i] for i in idx]
    ma, mb = G.mus(tasks, cfg["theta_a"]), G.mus(tasks, cfg["theta_b"])
    m = 3
    ds = []
    for rep in range(400):
        ra = G.runs(ma, m, cfg["rho"], Rng(10_000 + rep))
        rb = G.runs(mb, m, cfg["rho"], Rng(20_000 + rep))
        ds.append(np.mean([np.mean(x) for x in ra]) - np.mean([np.mean(x) for x in rb]))
    split = St.variance_split(ma, mb, m, cfg["rho"])
    assert float(np.var(ds, ddof=1)) == pytest.approx(split["runs"], rel=0.2)
    s = St.paired_study(m)
    assert s["split"]["se_paired"] < s["split"]["se_unpaired"]


def test_tie_is_a_tie():
    t = St.tie_study()
    assert t["test"]["p"] > 0.5
    assert t["test"]["wilson1"][0] < t["test"]["p2"] < t["test"]["wilson1"][1]
    assert t["n_unpaired"] > 100_000


# ── the SWE-style task ───────────────────────────────────────────────────


def test_swe_grading():
    want = {("fix", "thorough"): True, ("fix", "weak"): True, ("overfit", "thorough"): False,
            ("overfit", "weak"): True, ("collateral", "thorough"): False, ("collateral", "weak"): False}
    for (p, t), resolved in want.items():
        a = W.anatomy(p, t)
        assert a["grade"]["resolved"] is resolved
        assert a["grade"]["before"]["test_issue.py::test_add"] is False
        assert all(a["grade"]["before"][x] for x in W.PASS_TO_PASS)
        assert a["frames"][0]["stage"] == "task" and a["frames"][-1]["stage"] == "verdict"
        assert sum(1 for f in a["frames"] if f["stage"] == "agent") == len(a["calls"])


# ── fixtures ─────────────────────────────────────────────────────────────


def test_fixtures_are_current_engine():
    fx = json.loads((ROOT / "fixtures" / "evals_fixtures.json").read_text())
    assert fx["engine"] == VERSION == "1.10.0"
    assert fx["loop"] == json.loads(json.dumps(St.loop_study()))
    data = json.loads((ROOT / "ts" / "src" / "evals_data.json").read_text())
    assert data["loop"] == St.LOOP and data["swe"]["base"] == W.BASE
