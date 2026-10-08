"""A natural logarithm both languages compute bit for bit: fdlibm's ``__ieee754_log`` (the
algorithm behind JavaScript engines' ``Math.log``), transcribed with explicit bit manipulation.

The platform's ``math.log`` (C libm) and V8's ``Math.log`` disagree in the last bit for some
inputs, which made BM25 scores differ by one ulp between Python and TS. BM25's IDF and nDCG's
discount use this ``ln`` instead, so scores, ties and rankings are identical everywhere.
"""
from __future__ import annotations

import struct

LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10
TWO54 = 1.80143985094819840000e16
LG1 = 6.666666666666735130e-01
LG2 = 3.999999999940941908e-01
LG3 = 2.857142874366239149e-01
LG4 = 2.222219843214978396e-01
LG5 = 1.818357216161805012e-01
LG6 = 1.531383769920937332e-01
LG7 = 1.479819860511658591e-01


def _words(x: float) -> tuple[int, int]:
    b = struct.unpack("<Q", struct.pack("<d", x))[0]
    hi = b >> 32
    if hi >= 0x80000000:
        hi -= 0x100000000
    return hi, b & 0xFFFFFFFF


def _with_high(x: float, hi: int) -> float:
    lo = struct.unpack("<Q", struct.pack("<d", x))[0] & 0xFFFFFFFF
    return struct.unpack("<d", struct.pack("<Q", ((hi & 0xFFFFFFFF) << 32) | lo))[0]


def ln(x: float) -> float:
    hx, lx = _words(x)
    k = 0
    if hx < 0x00100000:
        if ((hx & 0x7FFFFFFF) | lx) == 0:
            return float("-inf")
        if hx < 0:
            return float("nan")
        k -= 54
        x *= TWO54
        hx, _ = _words(x)
    if hx >= 0x7FF00000:
        return x + x
    k += (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = _with_high(x, hx | (i ^ 0x3FF00000))
    k += i >> 20
    f = x - 1.0
    if (0x000FFFFF & (2 + hx)) < 3:
        if f == 0.0:
            if k == 0:
                return 0.0
            dk = float(k)
            return dk * LN2_HI + dk * LN2_LO
        r = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0:
            return f - r
        dk = float(k)
        return dk * LN2_HI - ((r - dk * LN2_LO) - f)
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147A
    w = z * z
    j = 0x6B851 - hx
    t1 = w * (LG2 + w * (LG4 + w * LG6))
    t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)))
    i |= j
    r = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + r))
        return dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
    if k == 0:
        return f - s * (f - r)
    return dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
