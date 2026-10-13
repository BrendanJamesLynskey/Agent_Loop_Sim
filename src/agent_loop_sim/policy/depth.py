"""Defence in depth with imperfect layers (since 1.10). An attempt that would steer the model must pass every
layer; layer i misses it with probability m_i.

- **Independent** layers: P(all miss) = m_1 · m_2 · … · m_n, never zero while every m_i > 0.
- **Correlated** layers, as a common-mode mixture: with probability c an attempt evades every layer together
  (all layers share a blind spot); otherwise layer i misses independently with q_i = (m_i − c) / (1 − c),
  which keeps each layer's own miss rate at m_i. Then P(all miss) = c + (1 − c) · q_1 · … · q_n ≥ c: stacking
  more layers cannot push the rate below the shared blind spot. The pairwise correlation of two layers' misses
  is (P(both) − m_i m_j) / √(m_i(1 − m_i) m_j(1 − m_j)), with P(both) = c + (1 − c) q_i q_j.
- Over N attempts, P(at least one gets through) = 1 − (1 − p · P(all miss))^N, where p is the base rate at
  which an attempt would steer an undefended model.

``simulate`` draws the mixture with the shared seeded generator; the tests check it against the closed forms.
Products and powers are left-to-right loops, so both languages compute them in the same order.
"""
from __future__ import annotations

import math
from typing import Any

from ..rng import Rng


def all_miss_independent(m: list[float]) -> float:
    p = 1.0
    for x in m:
        p = p * x
    return p


def q_given_common(m: list[float], c: float) -> list[float]:
    if c < 0.0 or c >= 1.0:
        raise ValueError("c must be in [0, 1)")
    for x in m:
        if x < c:
            raise ValueError("c cannot exceed any layer's miss rate")
    return [(x - c) / (1.0 - c) for x in m]


def all_miss_common(m: list[float], c: float) -> float:
    return c + (1.0 - c) * all_miss_independent(q_given_common(m, c))


def pair_correlation(mi: float, mj: float, c: float) -> float:
    q = q_given_common([mi, mj], c)
    both = c + (1.0 - c) * q[0] * q[1]
    return (both - mi * mj) / math.sqrt(mi * (1.0 - mi) * mj * (1.0 - mj))


def at_least_one(p: float, n: int) -> float:
    """1 − (1 − p)^n by repeated multiplication."""
    s = 1.0
    for _ in range(n):
        s = s * (1.0 - p)
    return 1.0 - s


def simulate(m: list[float], c: float, trials: int, seed: int) -> dict[str, Any]:
    """Draw ``trials`` attempts against the layers; count those every layer misses, and each pair's joint misses."""
    q = q_given_common(m, c)
    r = Rng(seed)
    through = 0
    single = [0] * len(m)
    pair = [[0] * len(m) for _ in m]
    for _ in range(trials):
        common = r.random() < c
        miss = []
        for i in range(len(m)):
            u = r.random()
            miss.append(common or u < q[i])
        ok = True
        for i in range(len(m)):
            if miss[i]:
                single[i] += 1
                for j in range(i + 1, len(m)):
                    if miss[j]:
                        pair[i][j] += 1
            else:
                ok = False
        if ok:
            through += 1
    return {"trials": trials, "through": through, "rate": through / trials, "single": single, "pair": pair}


def depth_table(base: float, m: list[float], cs: list[float], attempts: int) -> list[dict[str, Any]]:
    """For each c and each number of layers k (the first k layers): P(all miss), the end-to-end rate p·P and
    the chance at least one of ``attempts`` attempts gets through."""
    rows = []
    for c in cs:
        for k in range(len(m) + 1):
            mk = m[:k]
            pm = all_miss_common(mk, c) if k > 0 else 1.0
            rows.append({"c": c, "layers": k, "all_miss": pm, "end_to_end": base * pm,
                         "any_of_n": at_least_one(base * pm, attempts)})
    return rows
