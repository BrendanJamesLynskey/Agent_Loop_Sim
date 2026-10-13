"""Special functions both languages compute bit for bit (since 1.8).

The evals module needs ``exp``, ``ln``, the log-gamma function, the regularised incomplete beta
function and the normal distribution's CDF and quantile. Platform libraries (C libm in Python,
V8 in TypeScript) may disagree in the last bit, so these are written out from published
algorithms using only IEEE-754 arithmetic (``+ − × ÷``, ``sqrt``, which both languages round
correctly) and explicit bit manipulation:

- ``exp``: fdlibm's ``__ieee754_exp`` (e_exp.c, Sun Microsystems 1993), error < 1 ulp;
- ``ln``: fdlibm's ``__ieee754_log`` (shared with the context module);
- ``lgamma``: shift the argument above 20 by the recurrence Γ(x+1) = xΓ(x), then Stirling's series
  to the x⁻⁷ term (truncation error below 2e-15 relative at x = 20);
- ``betainc``: the continued fraction for I_x(a, b) evaluated by the modified Lentz method
  (Press et al., Numerical Recipes, §6.4), with the usual symmetry switch;
- ``norm_cdf``: Marsaglia's series for |x| ≤ 3 (Marsaglia 2004, "Evaluating the normal
  distribution", J. Stat. Softw. 11(4)) and Laplace's continued fraction for the tail beyond,
  evaluated backwards with a fixed 120 terms, so tail probabilities keep their relative accuracy;
- ``norm_ppf``: Wichura's algorithm AS 241 (PPND16, Appl. Statist. 37(3), 1988), accurate to about
  1 part in 10¹⁶, transcribed from R's qnorm.c. For p within exp(−729) of 0 or 1 (r > 27) the
  r ≤ 27 rational approximation is used as is; no evaluation here comes near it.

The tests check each against scipy and the standard library with a tolerance; the TS port must
reproduce every value exactly.
"""
from __future__ import annotations

import math
import struct

from ..context.mathx import ln

__all__ = ["exp", "ln", "lgamma", "betainc", "norm_cdf", "norm_sf", "norm_pdf", "norm_ppf", "logistic", "zcrit"]

_HALF = (0.5, -0.5)
_O_THRESHOLD = 7.09782712893383973096e02
_U_THRESHOLD = -7.45133219101941108420e02
_LN2_HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)
_LN2_LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)
_INVLN2 = 1.44269504088896338700e00
_TWOM1000 = 9.33263618503218878990e-302
_P1 = 1.66666666666666019037e-01
_P2 = -2.77777777770155933842e-03
_P3 = 6.61375632143793436117e-05
_P4 = -1.65339022054652515390e-06
_P5 = 4.13813679705723846039e-08

HALF_LN_2PI = 0.91893853320467274178  # ln(sqrt(2π))


def _words(x: float) -> tuple[int, int]:
    b = struct.unpack("<Q", struct.pack("<d", x))[0]
    hi = b >> 32
    if hi >= 0x80000000:
        hi -= 0x100000000
    return hi, b & 0xFFFFFFFF


def _from_words(hi: int, lo: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)))[0]


def exp(x: float) -> float:
    """fdlibm's ``__ieee754_exp``."""
    hx, lx = _words(x)
    xsb = 1 if hx < 0 else 0
    hx &= 0x7FFFFFFF
    if hx >= 0x40862E42:
        if hx >= 0x7FF00000:
            if ((hx & 0xFFFFF) | lx) != 0:
                return x + x
            return x if xsb == 0 else 0.0
        if x > _O_THRESHOLD:
            return math.inf
        if x < _U_THRESHOLD:
            return 0.0
    hi = 0.0
    lo = 0.0
    k = 0
    if hx > 0x3FD62E42:
        if hx < 0x3FF0A2B2:
            hi = x - _LN2_HI[xsb]
            lo = _LN2_LO[xsb]
            k = 1 - xsb - xsb
        else:
            k = int(_INVLN2 * x + _HALF[xsb])
            t = float(k)
            hi = x - t * _LN2_HI[0]
            lo = t * _LN2_LO[0]
        x = hi - lo
    elif hx < 0x3E300000:
        return 1.0 + x
    t = x * x
    c = x - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    yh, yl = _words(y)
    if k >= -1021:
        return _from_words(yh + (k << 20), yl)
    return _from_words(yh + ((k + 1000) << 20), yl) * _TWOM1000


# Stirling's series coefficients B_2j / (2j (2j − 1)), as divisions both languages round alike.
_S1 = 1.0 / 12.0
_S2 = 1.0 / 360.0
_S3 = 1.0 / 1260.0
_S4 = 1.0 / 1680.0


def lgamma(x: float) -> float:
    """ln Γ(x) for x > 0."""
    p = 1.0
    while x < 20.0:
        p = p * x
        x = x + 1.0
    z = 1.0 / (x * x)
    s = (((-_S4 * z + _S3) * z - _S2) * z + _S1) / x
    return (x - 0.5) * ln(x) - x + HALF_LN_2PI + s - ln(p)


_FPMIN = 1e-300
_CF_EPS = 1e-15
_CF_MAXIT = 2000


def _betacf(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h = h * (d * c)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h = h * de
        if abs(de - 1.0) < _CF_EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function I_x(a, b), a, b > 0."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * ln(x) + b * ln(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x - HALF_LN_2PI)


_CF_TERMS = 120


def _upper_tail(x: float) -> float:
    """Q(x) = 1 − Φ(x) for x > 3: φ(x) / (x + 1/(x + 2/(x + 3/(x + …)))), evaluated backwards."""
    r = x
    for k in range(_CF_TERMS, 0, -1):
        r = x + k / r
    return norm_pdf(x) / r


def norm_cdf(x: float) -> float:
    """Φ(x)."""
    if x != x:
        return x
    if x > 3.0:
        return 1.0 - _upper_tail(x)
    if x < -3.0:
        return _upper_tail(-x)
    s = x
    t = 0.0
    b = x
    q = x * x
    i = 1.0
    while s != t:
        t = s
        i = i + 2.0
        b = b * q / i
        s = t + b
    return 0.5 + s * norm_pdf(x)


def norm_sf(x: float) -> float:
    """1 − Φ(x), accurate in the upper tail."""
    return norm_cdf(-x)


def norm_ppf(p: float) -> float:
    """Φ⁻¹(p): Wichura's AS 241."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r
                        + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
                     + 133.14166789178437745) * r + 3.387132872796366608) / (
            ((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r
                 + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
             + 42.313330701600911252) * r + 1.0)
    r = math.sqrt(-ln(1.0 - p if q > 0 else p))
    if r <= 5.0:
        r = r - 1.6
        val = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
                   + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                + 4.6303378461565452959) * r + 1.42343711074968357734) / (
            (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
                  + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r
              + 2.05319162663775882187) * r + 1.0))
    else:
        r = r - 5.0
        val = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
                   + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r
                + 5.4637849111641143699) * r + 6.6579046435011037772) / (
            (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                  + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
              + 0.59983220655588793769) * r + 1.0))
    return -val if q < 0 else val


def logistic(z: float) -> float:
    """1 / (1 + e^−z)."""
    return 1.0 / (1.0 + exp(-z))


def zcrit(alpha: float) -> float:
    """The two-sided critical value z_{1−α/2} (1.959964… for α = 0.05)."""
    return norm_ppf(1.0 - alpha / 2.0)
