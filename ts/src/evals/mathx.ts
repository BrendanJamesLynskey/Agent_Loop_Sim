/**
 * Special functions computed bit for bit as the Python reference does (engine 1.8.0): fdlibm's exp
 * and log, ln Γ by shift and Stirling's series, the regularised incomplete beta function by Lentz's
 * continued fraction, the normal CDF (Marsaglia's series and Laplace's continued fraction) and its
 * quantile (Wichura's AS 241). A port of agent_loop_sim/evals/mathx.py; see that file for sources.
 */
import { ln } from "../context/mathx";

export { ln };

const HALF = [0.5, -0.5];
const O_THRESHOLD = 7.09782712893383973096e2;
const U_THRESHOLD = -7.45133219101941108420e2;
const LN2_HI = [6.9314718036912381649e-1, -6.9314718036912381649e-1];
const LN2_LO = [1.90821492927058770002e-10, -1.90821492927058770002e-10];
const INVLN2 = 1.442695040888963387;
const TWOM1000 = 9.3326361850321887899e-302;
const P1 = 1.66666666666666019037e-1;
const P2 = -2.77777777770155933842e-3;
const P3 = 6.61375632143793436117e-5;
const P4 = -1.6533902205465251539e-6;
const P5 = 4.13813679705723846039e-8;

export const HALF_LN_2PI = 0.91893853320467274178;

const buf = new DataView(new ArrayBuffer(8));

function words(x: number): [number, number] {
  buf.setFloat64(0, x);
  return [buf.getInt32(0), buf.getUint32(4)];
}

function fromWords(hi: number, lo: number): number {
  buf.setInt32(0, hi | 0);
  buf.setUint32(4, lo >>> 0);
  return buf.getFloat64(0);
}

/** fdlibm's __ieee754_exp. */
export function exp(x: number): number {
  let [hx] = words(x);
  const lx = words(x)[1];
  const xsb = hx < 0 ? 1 : 0;
  hx &= 0x7fffffff;
  if (hx >= 0x40862e42) {
    if (hx >= 0x7ff00000) {
      if (((hx & 0xfffff) | lx) !== 0) return x + x;
      return xsb === 0 ? x : 0.0;
    }
    if (x > O_THRESHOLD) return Infinity;
    if (x < U_THRESHOLD) return 0.0;
  }
  let hi = 0.0;
  let lo = 0.0;
  let k = 0;
  if (hx > 0x3fd62e42) {
    if (hx < 0x3ff0a2b2) {
      hi = x - LN2_HI[xsb]!;
      lo = LN2_LO[xsb]!;
      k = 1 - xsb - xsb;
    } else {
      k = Math.trunc(INVLN2 * x + HALF[xsb]!);
      const t = k;
      hi = x - t * LN2_HI[0]!;
      lo = t * LN2_LO[0]!;
    }
    x = hi - lo;
  } else if (hx < 0x3e300000) {
    return 1.0 + x;
  }
  const t = x * x;
  const c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  if (k === 0) return 1.0 - ((x * c) / (c - 2.0) - x);
  const y = 1.0 - (lo - (x * c) / (2.0 - c) - hi);
  const [yh, yl] = words(y);
  if (k >= -1021) return fromWords(yh + (k << 20), yl);
  return fromWords(yh + ((k + 1000) << 20), yl) * TWOM1000;
}

const S1 = 1.0 / 12.0;
const S2 = 1.0 / 360.0;
const S3 = 1.0 / 1260.0;
const S4 = 1.0 / 1680.0;

/** ln Γ(x) for x > 0. */
export function lgamma(x: number): number {
  let p = 1.0;
  while (x < 20.0) {
    p = p * x;
    x = x + 1.0;
  }
  const z = 1.0 / (x * x);
  const s = (((-S4 * z + S3) * z - S2) * z + S1) / x;
  return (x - 0.5) * ln(x) - x + HALF_LN_2PI + s - ln(p);
}

const FPMIN = 1e-300;
const CF_EPS = 1e-15;
const CF_MAXIT = 2000;

function betacf(a: number, b: number, x: number): number {
  const qab = a + b;
  const qap = a + 1.0;
  const qam = a - 1.0;
  let c = 1.0;
  let d = 1.0 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1.0 / d;
  let h = d;
  for (let m = 1; m <= CF_MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1.0 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1.0 / d;
    h = h * (d * c);
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1.0 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1.0 / d;
    const de = d * c;
    h = h * de;
    if (Math.abs(de - 1.0) < CF_EPS) break;
  }
  return h;
}

/** The regularised incomplete beta function I_x(a, b), a, b > 0. */
export function betainc(a: number, b: number, x: number): number {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  const bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * ln(x) + b * ln(1.0 - x));
  if (x < (a + 1.0) / (a + b + 2.0)) return (bt * betacf(a, b, x)) / a;
  return 1.0 - (bt * betacf(b, a, 1.0 - x)) / b;
}

export function normPdf(x: number): number {
  return exp(-0.5 * x * x - HALF_LN_2PI);
}

const CF_TERMS = 120;

function upperTail(x: number): number {
  let r = x;
  for (let k = CF_TERMS; k > 0; k--) r = x + k / r;
  return normPdf(x) / r;
}

/** Φ(x). */
export function normCdf(x: number): number {
  if (x !== x) return x;
  if (x > 3.0) return 1.0 - upperTail(x);
  if (x < -3.0) return upperTail(-x);
  let s = x;
  let t = 0.0;
  let b = x;
  const q = x * x;
  let i = 1.0;
  while (s !== t) {
    t = s;
    i = i + 2.0;
    b = (b * q) / i;
    s = t + b;
  }
  return 0.5 + s * normPdf(x);
}

/** 1 − Φ(x), accurate in the upper tail. */
export function normSf(x: number): number {
  return normCdf(-x);
}

/** Φ⁻¹(p): Wichura's AS 241. */
export function normPpf(p: number): number {
  if (p <= 0.0) return -Infinity;
  if (p >= 1.0) return Infinity;
  const q = p - 0.5;
  if (Math.abs(q) <= 0.425) {
    const r = 0.180625 - q * q;
    return (
      (q *
        (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r +
          45921.953931549871457) *
          r +
          13731.693765509461125) *
          r +
          1971.5909503065514427) *
          r +
          133.14166789178437745) *
          r +
          3.387132872796366608)) /
      (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r +
        21213.794301586595867) *
        r +
        5394.1960214247511077) *
        r +
        687.1870074920579083) *
        r +
        42.313330701600911252) *
        r +
        1.0)
    );
  }
  let r = Math.sqrt(-ln(q > 0 ? 1.0 - p : p));
  let val: number;
  if (r <= 5.0) {
    r = r - 1.6;
    val =
      (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r +
        1.27045825245236838258) *
        r +
        3.64784832476320460504) *
        r +
        5.7694972214606914055) *
        r +
        4.6303378461565452959) *
        r +
        1.42343711074968357734) /
      (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r +
        0.14810397642748007459) *
        r +
        0.68976733498510000455) *
        r +
        1.6763848301838038494) *
        r +
        2.05319162663775882187) *
        r +
        1.0);
  } else {
    r = r - 5.0;
    val =
      (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r +
        0.026532189526576123093) *
        r +
        0.29656057182850489123) *
        r +
        1.7848265399172913358) *
        r +
        5.4637849111641143699) *
        r +
        6.6579046435011037772) /
      (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
        7.868691311456132591e-4) *
        r +
        0.0148753612908506148525) *
        r +
        0.13692988092273580531) *
        r +
        0.59983220655588793769) *
        r +
        1.0);
  }
  return q < 0 ? -val : val;
}

/** 1 / (1 + e^−z). */
export function logistic(z: number): number {
  return 1.0 / (1.0 + exp(-z));
}

/** The two-sided critical value z_{1−α/2}. */
export function zcrit(alpha: number): number {
  return normPpf(1.0 - alpha / 2.0);
}
