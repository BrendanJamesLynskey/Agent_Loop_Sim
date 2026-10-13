"""The statistics of evaluating agents (since 1.8): pass@k and pass^k, intervals for a success rate,
clustered standard errors, paired comparisons and power.

Every function is plain arithmetic over the special functions in ``mathx`` (exact in both languages),
and every sum is a left-to-right loop (Python's built-in ``sum`` of floats is compensated, which
TypeScript's loop is not). The tests check each one against scipy, statsmodels or a closed form.

- **pass@k** (Chen et al. 2021, arXiv:2107.03374): from n samples of which c pass, the unbiased
  estimate of P(at least one of k passes) is 1 − C(n−c, k)/C(n, k), computed as the paper's product
  1 − ∏_{i=n−c+1}^{n} (1 − k/i).
- **pass^k** (Yao et al. 2024, τ-bench, arXiv:2406.12045): P(all k pass), estimated without bias by
  C(c, k)/C(n, k) = ∏_{i<k} (c − i)/(n − i).
- **Intervals** for k successes in n trials: Wald p̂ ± z·sqrt(p̂(1−p̂)/n); Wilson's score interval
  (Wilson 1927); Clopper–Pearson's exact interval (1934), from beta quantiles found by bisection on
  the incomplete beta function; and the percentile bootstrap (Efron 1979), resampling tasks.
- **Clustered standard errors**: when each task is run m times the runs are not independent; the
  cluster-robust (CR1) standard error of the overall rate treats tasks as the independent units.
- **Paired comparisons**: McNemar's test (1947) on the discordant tasks, exact (binomial) and with
  the continuity-corrected chi-square; the paired difference with its standard error, against the
  unpaired one.
- **Power and sample size**: the normal approximations for two independent proportions (Fleiss)
  and for paired proportions (Connor 1987).
"""
from __future__ import annotations

import math
from typing import Any

from ..rng import Rng
from .mathx import betainc, exp, lgamma, ln, norm_cdf, norm_ppf, zcrit

ALPHA = 0.05
METHODS = ["wald", "wilson", "exact", "bootstrap"]


def total(xs: list[float]) -> float:
    s = 0.0
    for x in xs:
        s = s + x
    return s


def mean(xs: list[float]) -> float:
    return total(xs) / len(xs)


def var(xs: list[float], ddof: int = 1) -> float:
    m = mean(xs)
    s = 0.0
    for x in xs:
        d = x - m
        s = s + d * d
    return s / (len(xs) - ddof)


def _clip(lo: float, hi: float) -> list[float]:
    return [lo if lo > 0.0 else 0.0, hi if hi < 1.0 else 1.0]


# ── pass@k and pass^k ────────────────────────────────────────────────────


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod = prod * (1.0 - k / i)
    return 1.0 - prod


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k from n samples with c correct: C(c, k) / C(n, k)."""
    if c < k:
        return 0.0
    prod = 1.0
    for i in range(k):
        prod = prod * ((c - i) / (n - i))
    return prod


def plugin_pass_at_k(n: int, c: int, k: int) -> float:
    """The naive plug-in 1 − (1 − c/n)^k, biased low for k > 1."""
    q = 1.0 - c / n
    v = 1.0
    for _ in range(k):
        v = v * q
    return 1.0 - v


def _pow(x: float, k: int) -> float:
    v = 1.0
    for _ in range(k):
        v = v * x
    return v


def task_pass_curves(mu: float, s: float, k: int) -> dict[str, float]:
    """True pass@k and pass^k on one task whose runs succeed with probability ``mu`` and are tied
    together by the generator's copy probability ``s`` (``generator.runs``): each run copies the
    task's shared outcome with probability s, else draws its own. Given the shared outcome, runs are
    independent, so P(all k fail) and P(all k pass) are two-term mixtures."""
    stay_fail_if_pass = (1.0 - s) * (1.0 - mu)
    stay_fail_if_fail = s + (1.0 - s) * (1.0 - mu)
    pass_if_pass = s + (1.0 - s) * mu
    pass_if_fail = (1.0 - s) * mu
    all_fail = mu * _pow(stay_fail_if_pass, k) + (1.0 - mu) * _pow(stay_fail_if_fail, k)
    all_pass = mu * _pow(pass_if_pass, k) + (1.0 - mu) * _pow(pass_if_fail, k)
    return {"pass_at_k": 1.0 - all_fail, "pass_hat_k": all_pass}


# ── binomial ─────────────────────────────────────────────────────────────


def binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return exp(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0) + k * ln(p) + (n - k) * ln(1.0 - p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) for X ~ Binomial(n, p) = I_{1−p}(n − k, k + 1)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    return betainc(n - k, k + 1.0, 1.0 - p)


def beta_ppf(q: float, a: float, b: float) -> float:
    """The q-quantile of Beta(a, b), by bisection on I_x(a, b) until the bracket stops shrinking."""
    lo = 0.0
    hi = 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── intervals ────────────────────────────────────────────────────────────


def wald(k: int, n: int, alpha: float = ALPHA) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z = zcrit(alpha)
    p = k / n
    h = z * math.sqrt(p * (1.0 - p) / n)
    return _clip(p - h, p + h)


def wilson(k: int, n: int, alpha: float = ALPHA) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z = zcrit(alpha)
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    mid = p + z2 / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return _clip((mid - half) / den, (mid + half) / den)


def exact(k: int, n: int, alpha: float = ALPHA) -> list[float]:
    """Clopper–Pearson: the beta quantiles B(α/2; k, n−k+1) and B(1−α/2; k+1, n−k)."""
    if n == 0:
        return [0.0, 1.0]
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2.0, k, n - k + 1.0)
    hi = 1.0 if k == n else beta_ppf(1.0 - alpha / 2.0, k + 1.0, n - k)
    return [lo, hi]


def interval(method: str, k: int, n: int, alpha: float = ALPHA) -> list[float]:
    if method == "wald":
        return wald(k, n, alpha)
    if method == "wilson":
        return wilson(k, n, alpha)
    if method == "exact":
        return exact(k, n, alpha)
    raise ValueError(f"no closed-form interval '{method}'")


def quantile(sorted_xs: list[float], q: float) -> float:
    """numpy's default ("linear") quantile of an ascending list."""
    h = (len(sorted_xs) - 1) * q
    i = int(h)
    if i >= len(sorted_xs) - 1:
        return sorted_xs[-1]
    a = sorted_xs[i]
    b = sorted_xs[i + 1]
    return a + (h - i) * (b - a)


def bootstrap(values: list[float], b: int, seed: int, alpha: float = ALPHA) -> dict[str, Any]:
    """Percentile bootstrap of the mean of per-task values: resample the tasks with replacement
    ``b`` times (one ``randint`` per draw, in order). Returns the interval, the bootstrap standard
    error and the resampled means in the order drawn."""
    r = Rng(seed)
    t = len(values)
    means = []
    for _ in range(b):
        s = 0.0
        for _ in range(t):
            s = s + values[r.randint(0, t - 1)]
        means.append(s / t)
    srt = sorted(means)
    return {"ci": [quantile(srt, alpha / 2.0), quantile(srt, 1.0 - alpha / 2.0)], "se": math.sqrt(var(means)),
            "means": means}


def coverage(method: str, n: int, p: float, alpha: float = ALPHA, intervals: list[list[float]] | None = None) -> float:
    """P(the interval covers p) when X ~ Binomial(n, p): Σ_k P(X = k)·[p ∈ CI(k)]."""
    ivs = intervals if intervals is not None else [interval(method, k, n, alpha) for k in range(n + 1)]
    c = 0.0
    for k in range(n + 1):
        lo, hi = ivs[k]
        if lo <= p <= hi:
            c = c + binom_pmf(k, n, p)
    return c


# ── clustered data ───────────────────────────────────────────────────────


def cluster_se(successes: list[int], runs: list[int]) -> dict[str, float]:
    """The rate over all runs, its naive binomial SE (runs independent) and the cluster-robust
    CR1 SE with tasks as clusters: sqrt(T/(T−1) · Σ_t (S_t − m_t p̂)²) / N."""
    t = len(successes)
    n = 0
    s_all = 0
    for i in range(t):
        n += runs[i]
        s_all += successes[i]
    p = s_all / n
    naive = math.sqrt(p * (1.0 - p) / n)
    acc = 0.0
    for i in range(t):
        d = successes[i] - runs[i] * p
        acc = acc + d * d
    se = math.sqrt(t / (t - 1) * acc) / n if t > 1 else naive
    return {"p": p, "n": n, "tasks": t, "naive_se": naive, "se": se,
            "design_effect": (se * se) / (naive * naive) if naive > 0 else 1.0}


def design_effect(m: int, icc: float) -> float:
    """Kish's design effect for clusters of m runs with intra-cluster correlation ``icc``."""
    return 1.0 + (m - 1) * icc


# ── comparing two agents ─────────────────────────────────────────────────


def paired_table(a: list[int], b: list[int]) -> dict[str, int]:
    t = {"n11": 0, "n10": 0, "n01": 0, "n00": 0}
    for x, y in zip(a, b):
        t["n" + str(x) + str(y)] += 1
    return t


def mcnemar(n10: int, n01: int) -> dict[str, float]:
    """McNemar's test on the discordant counts: the exact two-sided p (binomial at 1/2) and the
    continuity-corrected chi-square statistic with its p."""
    d = n10 + n01
    if d == 0:
        return {"discordant": 0, "exact_p": 1.0, "chi2": 0.0, "chi2_p": 1.0}
    lo = n10 if n10 < n01 else n01
    p = 2.0 * binom_cdf(lo, d, 0.5)
    diff = abs(n10 - n01) - 1.0  # Edwards' correction, as R's mcnemar.test and statsmodels (no clamp at 0)
    stat = diff * diff / d
    return {"discordant": d, "exact_p": p if p < 1.0 else 1.0, "chi2": stat, "chi2_p": 2.0 * norm_cdf(-math.sqrt(stat))}


def compare(a: list[float], b: list[float], alpha: float = ALPHA) -> dict[str, Any]:
    """Agent A against agent B on the same tasks (per-task scores: 0/1 or a mean over runs): the
    difference, its paired SE (from the per-task differences) and the unpaired SE (as if the two
    had been run on different tasks), the correlation between the agents' per-task scores, and
    both intervals."""
    n = len(a)
    d = [a[i] - b[i] for i in range(n)]
    delta = mean(d)
    va = var(a)
    vb = var(b)
    vd = var(d)
    se_p = math.sqrt(vd / n)
    se_u = math.sqrt(va / n + vb / n)
    cov = (va + vb - vd) / 2.0
    corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
    z = zcrit(alpha)
    return {"n": n, "a": mean(a), "b": mean(b), "delta": delta, "se_paired": se_p, "se_unpaired": se_u, "corr": corr,
            "ci_paired": [delta - z * se_p, delta + z * se_p], "ci_unpaired": [delta - z * se_u, delta + z * se_u],
            "z_paired": delta / se_p if se_p > 0 else 0.0,
            "p_paired": 2.0 * norm_cdf(-abs(delta / se_p)) if se_p > 0 else 1.0,
            "p_unpaired": 2.0 * norm_cdf(-abs(delta / se_u)) if se_u > 0 else 1.0}


def two_proportion(k1: int, n1: int, k2: int, n2: int, alpha: float = ALPHA) -> dict[str, Any]:
    """Two independent proportions: the pooled z test and the Wald interval for the difference."""
    p1 = k1 / n1
    p2 = k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(pp * (1.0 - pp) * (1.0 / n1 + 1.0 / n2))
    z = (p1 - p2) / se0
    se = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    zc = zcrit(alpha)
    return {"p1": p1, "p2": p2, "delta": p1 - p2, "z": z, "p": 2.0 * norm_cdf(-abs(z)),
            "ci": [p1 - p2 - zc * se, p1 - p2 + zc * se], "wilson1": wilson(k1, n1, alpha),
            "wilson2": wilson(k2, n2, alpha)}


# ── power and sample size ────────────────────────────────────────────────


def n_for_margin(p: float, half_width: float, alpha: float = ALPHA) -> int:
    """Tasks needed for a 1−α interval of ± half_width around a rate near p (normal approximation)."""
    z = zcrit(alpha)
    return math.ceil(z * z * p * (1.0 - p) / (half_width * half_width))


def power_unpaired(n: int, p1: float, p2: float, alpha: float = ALPHA) -> float:
    """Power of the two-sided pooled z test with n tasks per agent, each agent on its own tasks."""
    za = zcrit(alpha)
    pb = (p1 + p2) / 2.0
    d = abs(p1 - p2)
    s0 = math.sqrt(2.0 * pb * (1.0 - pb))
    s1 = math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    return norm_cdf((d * math.sqrt(n) - za * s0) / s1)


def n_unpaired(p1: float, p2: float, alpha: float = ALPHA, power: float = 0.8) -> int:
    """Tasks per agent (Fleiss, without continuity correction)."""
    za = zcrit(alpha)
    zb = norm_ppf(power)
    pb = (p1 + p2) / 2.0
    d = p1 - p2
    x = za * math.sqrt(2.0 * pb * (1.0 - pb)) + zb * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    return math.ceil(x * x / (d * d))


def power_paired(n: int, p10: float, p01: float, alpha: float = ALPHA) -> float:
    """Power of McNemar's test (normal approximation) on n shared tasks, where a task is passed by
    A only with probability p10 and by B only with probability p01 (Connor 1987)."""
    za = zcrit(alpha)
    psi = p10 + p01
    d = abs(p10 - p01)
    return norm_cdf((d * math.sqrt(n) - za * math.sqrt(psi)) / math.sqrt(psi - d * d))


def n_paired(p10: float, p01: float, alpha: float = ALPHA, power: float = 0.8) -> int:
    """Shared tasks needed (Connor 1987): (z_a sqrt(ψ) + z_b sqrt(ψ − δ²))² / δ²."""
    za = zcrit(alpha)
    zb = norm_ppf(power)
    psi = p10 + p01
    d = p10 - p01
    x = za * math.sqrt(psi) + zb * math.sqrt(psi - d * d)
    return math.ceil(x * x / (d * d))


def mde_unpaired(n: int, p: float, alpha: float = ALPHA, power: float = 0.8) -> float:
    """The smallest difference n tasks per agent detect with the given power, near rate p."""
    return (zcrit(alpha) + norm_ppf(power)) * math.sqrt(2.0 * p * (1.0 - p) / n)


def mde_paired(n: int, psi: float, alpha: float = ALPHA, power: float = 0.8) -> float:
    """The same on n shared tasks with discordance ψ (small-difference approximation)."""
    return (zcrit(alpha) + norm_ppf(power)) * math.sqrt(psi / n)
