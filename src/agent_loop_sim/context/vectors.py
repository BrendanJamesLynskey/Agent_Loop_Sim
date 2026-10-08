"""Quantised embedding vectors and the similarity arithmetic both languages share.

The shipped vectors are int8: each float embedding (unit length, from the offline run) scaled so
its largest component is +-127 and rounded (per-vector scale; cosine does not care about a
vector's length). From them, with integer arithmetic only:
- **int8** cosine: ``dot / sqrt(|a|^2 |b|^2)``; the dot product and the squared norms are exact
  integers, and a square root and a division are correctly rounded in IEEE 754, so the Python
  and TS scores are bit-identical;
- **int4**: each component re-quantised to -7..7, ``sign(q) * round_half_up(|q| * 7 / 127)``,
  then the same cosine;
- **binary**: one bit per component (``q > 0``); similarity ``(d - 2 * hamming) / d``.
"""
from __future__ import annotations

import base64
import math

PRECISIONS = ["int8", "int4", "binary"]


def decode(b64: str, dim: int) -> list[list[int]]:
    """Base64 int8 bytes to a list of vectors."""
    raw = base64.b64decode(b64)
    if len(raw) % dim:
        raise ValueError("vector bytes are not a multiple of the dimension")
    vals = [b - 256 if b > 127 else b for b in raw]
    return [vals[i:i + dim] for i in range(0, len(vals), dim)]


def dot(a: list[int], b: list[int]) -> int:
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def cosine(a: list[int], b: list[int]) -> float:
    na = dot(a, a)
    nb = dot(b, b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / math.sqrt(na * nb)


def to_int4(v: list[int]) -> list[int]:
    out = []
    for q in v:
        m = (abs(q) * 14 + 127) // 254
        out.append(-m if q < 0 else m)
    return out


def to_bits(v: list[int]) -> list[int]:
    return [1 if q > 0 else 0 for q in v]


def binary_sim(a: list[int], b: list[int]) -> float:
    h = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            h += 1
    return (len(a) - 2 * h) / len(a)


def prepare(vecs: list[list[int]], precision: str) -> list[list[int]]:
    if precision == "int8":
        return vecs
    if precision == "int4":
        return [to_int4(v) for v in vecs]
    if precision == "binary":
        return [to_bits(v) for v in vecs]
    raise ValueError(f"unknown precision {precision!r}")


def similarity(a: list[int], b: list[int], precision: str) -> float:
    return binary_sim(a, b) if precision == "binary" else cosine(a, b)


def nbytes(dim: int, precision: str) -> int:
    """Storage per vector: int8 one byte a component, int4 half, binary one bit."""
    return {"int8": dim, "int4": dim // 2, "binary": dim // 8}[precision]
