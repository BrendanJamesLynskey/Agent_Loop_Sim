"""A synthetic benchmark (since 1.8): tasks with a latent difficulty, agents with a latent skill,
and seeded runs whose correlation within a task is a parameter.

**Tasks and agents** follow the two-parameter logistic model of item response theory (IRT; Lord and
Novick 1968; applied to LLM benchmarks by Polo et al. 2024, tinyBenchmarks, arXiv:2402.14992): task t
has a difficulty b_t and a discrimination a_t, an agent has a skill θ, and one run of the agent on
the task succeeds with probability

    μ_t(θ) = 1 / (1 + exp(−a_t (θ − b_t))).

A **population** of tasks is drawn once from a seed: b_t = σ_b · Φ⁻¹(u) (u from the shared mulberry32
generator, offset by half a step so it is never 0 or 1) and a_t uniform on [a_lo, a_hi]. A
**benchmark** is a sample of T tasks from the population, drawn with replacement, so its score is an
estimate of the population's mean μ (the "super-population" view of Miller 2024, arXiv:2411.00640).

**Runs.** Each task has one shared outcome Z_t ~ Bernoulli(μ_t), drawn first. Run j then copies Z_t
with probability s = sqrt(ρ), and otherwise draws its own Bernoulli(μ_t). Two runs on the same task
both copy with probability s², so their correlation is exactly ρ, and each run still succeeds with
probability μ_t. Every run draws the same two numbers (copy?, own outcome) whatever ρ is, so changing
ρ keeps the other draws fixed (common random numbers). ρ = 0 gives independent runs; ρ = 1 gives a
task that always succeeds or always fails.

Every number here is illustrative: no difficulty, skill or correlation is fitted to a real
benchmark.
"""
from __future__ import annotations

import math
from typing import Any

from ..rng import Rng
from .mathx import logistic, norm_ppf

POPULATION = {"size": 4000, "difficulty_sd": 1.3, "a_lo": 0.7, "a_hi": 1.9, "seed": 2501}

_TWO32 = 4294967296.0


def population(size: int = POPULATION["size"], difficulty_sd: float = POPULATION["difficulty_sd"],
               a_lo: float = POPULATION["a_lo"], a_hi: float = POPULATION["a_hi"],
               seed: int = POPULATION["seed"]) -> list[dict[str, float]]:
    """``size`` tasks, each {b, a}: two draws per task, in order."""
    r = Rng(seed)
    out = []
    for _ in range(size):
        u = (r.next_u32() + 0.5) / _TWO32
        b = difficulty_sd * norm_ppf(u)
        a = r.uniform(a_lo, a_hi)
        out.append({"b": b, "a": a})
    return out


def success_p(task: dict[str, float], theta: float) -> float:
    return logistic(task["a"] * (theta - task["b"]))


def mus(tasks: list[dict[str, float]], theta: float) -> list[float]:
    return [success_p(t, theta) for t in tasks]


def true_score(tasks: list[dict[str, float]], theta: float) -> float:
    s = 0.0
    for t in tasks:
        s = s + success_p(t, theta)
    return s / len(tasks)


def sample(n_tasks: int, size: int, r: Rng) -> list[int]:
    """Task indices drawn with replacement from a population of ``size``."""
    return [r.randint(0, size - 1) for _ in range(n_tasks)]


def runs(mu_list: list[float], m: int, rho: float, r: Rng) -> list[list[int]]:
    """``m`` runs on each task (rows), correlated within a task as above. Draws, per task: Z_t, then
    per run (copy, own)."""
    s = math.sqrt(rho)
    out = []
    for mu in mu_list:
        z = 1 if r.random() < mu else 0
        row = []
        for _ in range(m):
            c = r.random()
            own = r.random()
            row.append(z if c < s else (1 if own < mu else 0))
        out.append(row)
    return out


def benchmark(theta: float, n_tasks: int, m: int, rho: float, seed: int,
              pop: list[dict[str, float]] | None = None) -> dict[str, Any]:
    """Sample ``n_tasks`` tasks (stream ``seed``) and run an agent of skill ``theta`` ``m`` times on
    each (stream ``seed + 1``)."""
    p = pop if pop is not None else population()
    idx = sample(n_tasks, len(p), Rng(seed))
    tasks = [p[i] for i in idx]
    mu = mus(tasks, theta)
    return {"theta": theta, "idx": idx, "mu": mu, "runs": runs(mu, m, rho, Rng(seed + 1)), "rho": rho, "m": m}


def icc(rows: list[list[int]]) -> float:
    """The pairwise correlation of runs on the same task across all pairs (j < l), pooled: the
    sample analogue of ρ when μ is constant across tasks (used by the tests)."""
    s = 0.0
    n = 0
    pairs = 0.0
    for row in rows:
        for x in row:
            s = s + x
            n += 1
    p = s / n
    for row in rows:
        for j in range(len(row)):
            for k in range(j + 1, len(row)):
                pairs = pairs + (row[j] - p) * (row[k] - p)
    m = len(rows[0])
    npairs = len(rows) * m * (m - 1) / 2
    return pairs / npairs / (p * (1.0 - p))
