"""A seeded random-number generator shared by the Python reference and the TS port.

mulberry32, as Disaggregated_Inference_Sim's browser engine uses it: 32-bit integer
arithmetic only, so Python and TypeScript produce the same stream bit for bit. Every
stochastic choice in the engine (injected tool failures, tool latency jitter, a simulated
human's answer time) draws from one of these, so a seed reproduces a run exactly in both
languages.
"""
from __future__ import annotations

MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """JavaScript's Math.imul: the low 32 bits of a * b, as unsigned."""
    return (a * b) & MASK


class Rng:
    """mulberry32. ``random()`` returns a float in [0, 1) with 32 bits of randomness."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    def next_u32(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & MASK
        a = self.state
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & MASK) ^ t
        return (t ^ (t >> 14)) & MASK

    def random(self) -> float:
        return self.next_u32() / 4294967296

    def randint(self, lo: int, hi: int) -> int:
        """An integer in [lo, hi], inclusive."""
        return lo + int(self.random() * (hi - lo + 1))

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()
