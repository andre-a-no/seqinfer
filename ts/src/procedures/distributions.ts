// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Student's t tail and its inverse; mirrors distributions.py, with the two functions JavaScript lacks:
// lgamma (CPython's Lanczos m_lgamma, which math.lgamma is) and erfc (fdlibm's, which glibc's is).
// Agreement with Python is to rounding level, not bit for bit: Math.log/exp/pow are not glibc's.

// ------------------------------------------------------------------ lgamma (CPython mathmodule.c)

const LANCZOS_G = 6.024680040776729583740234375;
const LANCZOS_NUM = [
  23531376880.410759688572007674451636754734846804940, 42919803642.649098768957899047001988850926355848959,
  35711959237.355668049440185451547166705960488635843, 17921034426.037209699919755754458931112671403265390,
  6039542586.3520280050642916443072979210699388420708, 1439720407.3117216736632230727949123939715485786772,
  248874557.86205415651146038641322942321632125127801, 31426415.585400194380614231628318205362874684987640,
  2876370.6289353724412254090516208496135991145378768, 186056.26539522349504029498971604569928220784236328,
  8071.6720023658162106380029022722506138218516325024, 210.82427775157934587250973392071336271166969580291,
  2.5066282746310002701649081771338373386264310793408,
];
const LANCZOS_DEN = [0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0, 13339535.0, 2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0];

function lanczosSum(x: number): number {
  let num = 0.0, den = 0.0;
  if (x < 5.0) {
    for (let i = LANCZOS_NUM.length - 1; i >= 0; i--) {
      num = num * x + LANCZOS_NUM[i];
      den = den * x + LANCZOS_DEN[i];
    }
  } else {
    for (let i = 0; i < LANCZOS_NUM.length; i++) {
      num = num / x + LANCZOS_NUM[i];
      den = den / x + LANCZOS_DEN[i];
    }
  }
  return num / den;
}

/** log Gamma(x) for x > 0, as CPython computes it. */
export function lgamma(x: number): number {
  if (!(x > 0)) throw new RangeError(`lgamma is only implemented for positive arguments, got ${x}`);
  if (x === Infinity) return Infinity;
  if (x === Math.floor(x) && x <= 2.0) return 0.0;
  if (x < 1e-20) return -Math.log(x);
  let r = Math.log(lanczosSum(x)) - LANCZOS_G;
  r += (x - 0.5) * (Math.log(x + LANCZOS_G - 0.5) - 1);
  return r;
}

// ------------------------------------------------------------------ erfc (fdlibm s_erf.c)

const view = new DataView(new ArrayBuffer(8));
const high = (x: number): number => (view.setFloat64(0, x), view.getInt32(0));
const lowCleared = (x: number): number => (view.setFloat64(0, x), view.setUint32(4, 0), view.getFloat64(0));

const ERX = 8.45062911510467529297e-01;
const PP = [1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02, -5.77027029648944159157e-03, -2.37630166566501626084e-05];
const QQ = [0, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03, 1.32494738004321644526e-04, -3.96022827877536812320e-06];
const PA = [-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01, 3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02, -2.16637559486879084300e-03];
const QA = [0, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02, 1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02];
const RA = [-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e+01, -6.23753324503260060396e+01, -1.62396669462573470355e+02, -1.84605092906711035994e+02, -8.12874355063065934246e+01, -9.81432934416914548592e+00];
const SA = [0, 1.96512716674392571292e+01, 1.37657754143519042600e+02, 4.34565877475229228821e+02, 6.45387271733267880336e+02, 4.29008140027567833386e+02, 1.08635005541779435134e+02, 6.57024977031928170135e+00, -6.04244152148580987438e-02];
const RB = [-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e+01, -1.60636384855821916062e+02, -6.37566443368389627722e+02, -1.02509513161107724954e+03, -4.83519191608651397019e+02];
const SB = [0, 3.03380607434824582924e+01, 3.25792512996573918826e+02, 1.53672958608443695994e+03, 3.19985821950859553908e+03, 2.55305040643316442583e+03, 4.74528541206955367215e+02, -2.24409524465858183362e+01];

export function erfc(x: number): number {
  const hx = high(x), ix = hx & 0x7fffffff;
  if (ix >= 0x7ff00000) return (hx < 0 ? 2 : 0) + 1 / x;
  if (ix < 0x3feb0000) {
    if (ix < 0x3c700000) return 1 - x;
    const z = x * x;
    const r = PP[0] + z * (PP[1] + z * (PP[2] + z * (PP[3] + z * PP[4])));
    const s = 1 + z * (QQ[1] + z * (QQ[2] + z * (QQ[3] + z * (QQ[4] + z * QQ[5]))));
    const y = r / s;
    if (hx < 0x3fd00000) return 1 - (x + x * y);
    let rr = x * y;
    rr += x - 0.5;
    return 0.5 - rr;
  }
  if (ix < 0x3ff40000) {
    const s = Math.abs(x) - 1;
    const P = PA[0] + s * (PA[1] + s * (PA[2] + s * (PA[3] + s * (PA[4] + s * (PA[5] + s * PA[6])))));
    const Q = 1 + s * (QA[1] + s * (QA[2] + s * (QA[3] + s * (QA[4] + s * (QA[5] + s * QA[6])))));
    return hx >= 0 ? 1 - ERX - P / Q : 1 + (ERX + P / Q);
  }
  if (ix < 0x403c0000) {
    const ax = Math.abs(x);
    const s = 1 / (ax * ax);
    let R: number, S: number;
    if (ix < 0x4006db6d) {
      R = RA[0] + s * (RA[1] + s * (RA[2] + s * (RA[3] + s * (RA[4] + s * (RA[5] + s * (RA[6] + s * RA[7]))))));
      S = 1 + s * (SA[1] + s * (SA[2] + s * (SA[3] + s * (SA[4] + s * (SA[5] + s * (SA[6] + s * (SA[7] + s * SA[8])))))));
    } else {
      if (hx < 0 && ix >= 0x40180000) return 2 - 1e-300;
      R = RB[0] + s * (RB[1] + s * (RB[2] + s * (RB[3] + s * (RB[4] + s * (RB[5] + s * RB[6])))));
      S = 1 + s * (SB[1] + s * (SB[2] + s * (SB[3] + s * (SB[4] + s * (SB[5] + s * (SB[6] + s * SB[7]))))));
    }
    const z = lowCleared(ax);
    const r = Math.exp(-z * z - 0.5625) * Math.exp((z - ax) * (z + ax) + R / S);
    return hx > 0 ? r / ax : 2 - r / ax;
  }
  return hx > 0 ? 0 : 2 - 1e-300;
}

// ------------------------------------------------------------------ distributions.py

function betaContinuedFraction(a: number, b: number, x: number, y: number): number {
  const tiny = 1e-300, eps = 2.3e-16;
  const qab = a + b, qap = a + 1.0, qam = a - 1.0;
  let c = 1.0, d = ((a + 1.0) * y + (1.0 - b) * x) / qap;
  d = 1.0 / (Math.abs(d) > tiny ? d : tiny);
  let h = d;
  for (let m = 1; m < 20000; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    d = 1.0 / (Math.abs(d) > tiny ? d : tiny);
    c = 1.0 + aa / c;
    c = Math.abs(c) > tiny ? c : tiny;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    d = 1.0 / (Math.abs(d) > tiny ? d : tiny);
    c = 1.0 + aa / c;
    c = Math.abs(c) > tiny ? c : tiny;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1.0) < eps) return h;
  }
  throw new RangeError("incomplete beta continued fraction did not converge");
}

function betaFractionNearOne(a: number, b: number, x: number, y: number): number | null {
  const value = (terms: number): number => {
    let u = 1.0, e = 0.0;
    for (let k = terms; k > 0; k--) {
      const m = Math.floor(k / 2);
      if (k % 2) {
        const onePlusD = (2.0 * a * m + 3.0 * m * m + a * (1.0 - b) + m * (2.0 - b) + (a + m) * (a + b + m) * y)
          / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        u = (e + onePlusD) / u;
      } else {
        e = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)) / u;
        u = 1.0 + e;
      }
    }
    return 1.0 / u;
  };
  let terms = 32, previous = value(16);
  while (terms <= 4096) {
    const current = value(terms);
    if (Math.abs(current - previous) <= 1e-16 * Math.abs(current)) return current;
    [terms, previous] = [2 * terms, current];
  }
  return null;
}

function betaFactor(a: number, b: number, x: number, y: number): number {
  if (b <= 1.0 && x <= 0.5) {
    let total = 1.0, term = 1.0, n = 0.0;
    for (;;) {
      term *= ((a + b + n) / (a + 1.0 + n)) * x;
      total += term;
      n += 1.0;
      if (term <= 1e-17 * total) return total;
    }
  }
  if (b <= 1.0 && a > 1e4) {
    const near = betaFractionNearOne(a, b, x, y);
    if (near !== null) return near;
  }
  return betaContinuedFraction(a, b, x, y);
}

const SQRT2 = Math.sqrt(2.0);
const LOG_SQRT_2PI = 0.5 * Math.log(2.0 * Math.PI);
const HUGE_DF = 1e15;
const TINY = 1e-280;

function lgammaRatio(z: number, a: number): number {
  if (z < 100.0) return lgamma(z + a) - lgamma(z);
  const w = z + a;
  const iw = 1.0 / w, iz = 1.0 / z;
  const main = (z - 0.5) * Math.log1p(a / z) + a * Math.log(w) - a;
  return main + (iw - iz) / 12.0 - (Math.pow(iw, 3) - Math.pow(iz, 3)) / 360.0 + (Math.pow(iw, 5) - Math.pow(iz, 5)) / 1260.0;
}

function logInverseBeta(a: number, b: number): number {
  const [big, small] = a >= b ? [a, b] : [b, a];
  return lgammaRatio(big, small) - lgamma(small);
}

export function regularizedBeta(x: number, a: number, b: number, y: number | null = null, logs: [number, number] | null = null): number {
  if (y === null) y = 1.0 - x;
  if (x <= 0.0) return 0.0;
  if (y <= 0.0) return 1.0;
  const [logX, logY] = logs ?? [Math.log(x), Math.log(y)];
  const logFront = logInverseBeta(a, b) + a * logX + b * logY;
  if (x < (a + 1.0) / (a + b + 2.0)) return (Math.exp(logFront) * betaFactor(a, b, x, y)) / a;
  return 1.0 - (Math.exp(logFront) * betaFactor(b, a, y, x)) / b;
}

export function logNormalUpper(z: number): number {
  const q = 0.5 * erfc(z / SQRT2);
  if (q > TINY) return Math.log(q);
  const w = 1.0 / (z * z);
  const series = 1.0 - w * (1.0 - w * (3.0 - w * (15.0 - 105.0 * w)));
  return -0.5 * z * z - Math.log(z) - LOG_SQRT_2PI + Math.log(series);
}

function fisherApplies(t: number, df: number): boolean {
  return df >= 1e5 && (t === 0.0 || 16.0 * Math.log(t) <= Math.log(6e-11) + 4.0 * Math.log(df));
}

function fisherCorrection(t: number, df: number): number {
  const root = Math.sqrt(df);
  const r2 = t < 1e150 ? (t * t) / root : Math.pow(t / Math.sqrt(root), 2);
  const u = r2 / root, w = r2 * r2, v = 1.0 / df;
  const first = (u + v) / 4.0;
  const second = (3.0 * w * u - 7.0 * u * u - 5.0 * u * v - 3.0 * v * v) / 96.0;
  const third = (w * w * u - 11.0 * w * u * u + 14.0 * Math.pow(u, 3) + 6.0 * u * u * v - 3.0 * u * v * v - 15.0 * Math.pow(v, 3)) / 384.0;
  return t * (first + second + third);
}

function normalApplies(t: number, df: number): boolean {
  return df === Infinity || (t * t + 1.0) * Math.max(1.0, t * t) < 4e-17 * df;
}

function logHugeDfTail(t: number, df: number): number {
  const s = t / Math.sqrt(df);
  const log1pS2 = s < 1e150 ? Math.log1p(s * s) : 2.0 * Math.log(s);
  const logDensity = lgammaRatio(0.5 * df, 0.5) - 0.5 * (Math.log(df) + Math.log(Math.PI)) - 0.5 * (df + 1.0) * log1pS2;
  const correction = Math.log1p(-(1.0 / (t * t) - 1.0 / df) / (1.0 + 1.0 / df));
  return logDensity - Math.log(t) + log1pS2 - Math.log1p(1.0 / df) + correction;
}

function upperTail(t: number, df: number): number {
  if (t === 0.0) return 0.5;
  if (normalApplies(t, df)) return 0.5 * erfc(t / SQRT2);
  if (df > HUGE_DF && !fisherApplies(t, df)) return Math.exp(logHugeDfTail(t, df));
  if (fisherApplies(t, df)) {
    const phi = Math.exp(-0.5 * t * t - LOG_SQRT_2PI);
    return 0.5 * erfc(t / SQRT2) + phi * fisherCorrection(t, df);
  }
  return betaTailIsTiny(t, df) ? Math.exp(logUpperTail(t, df)) : betaUpperTail(t, df);
}

const logS = (t: number, df: number): number => Math.log(t) - 0.5 * Math.log(df);
const betaTailIsTiny = (t: number, df: number): boolean => t > 0.0 && logS(t, df) > Math.log(1e100);

function betaArguments(t: number, df: number): [number, number, [number, number]] {
  const s = t / Math.sqrt(df);
  if (s < 1.0) {
    const s2 = s * s;
    const logs: [number, number] = s > 0.0 ? [-Math.log1p(s2), 2.0 * Math.log(s) - Math.log1p(s2)] : [0.0, -Infinity];
    return [1.0 / (1.0 + s2), s2 / (1.0 + s2), logs];
  }
  const r2 = 1.0 / (s * s);
  return [r2 / (1.0 + r2), 1.0 / (1.0 + r2), [-2.0 * Math.log(s) - Math.log1p(r2), -Math.log1p(r2)]];
}

function betaUpperTail(t: number, df: number): number {
  const [x, y, logs] = betaArguments(t, df);
  return 0.5 * regularizedBeta(x, 0.5 * df, 0.5, y, logs);
}

function logUpperTail(t: number, df: number): number {
  if (t === 0.0) return Math.log(0.5);
  if (normalApplies(t, df)) return logNormalUpper(t);
  if (df > HUGE_DF && !fisherApplies(t, df)) return logHugeDfTail(t, df);
  if (fisherApplies(t, df)) {
    const logPhi = -0.5 * t * t - LOG_SQRT_2PI;
    const logQ = logNormalUpper(t);
    return logQ + Math.log1p(Math.exp(logPhi - logQ) * fisherCorrection(t, df));
  }
  const a = 0.5 * df;
  const ls = logS(t, df);
  if (ls > Math.log(1e100)) return Math.log(0.5) + logInverseBeta(a, 0.5) + a * (-2.0 * ls) - Math.log(a);
  const tail = betaUpperTail(t, df);
  if (tail > TINY) return Math.log(tail);
  const [x, y, [logX, logY]] = betaArguments(t, df);
  const logFront = logInverseBeta(a, 0.5) + a * logX + 0.5 * logY;
  return Math.log(0.5) + logFront + Math.log(betaFactor(a, 0.5, x, y)) - Math.log(a);
}

function checkDf(df: number): void {
  if (!(df > 0)) throw new RangeError("degrees of freedom must be positive");
}

/** P(T <= t) for Student's t with df degrees of freedom. */
export function tCdf(t: number, df: number): number {
  checkDf(df);
  if (Number.isNaN(t)) return NaN;
  if (t === 0.0) return 0.5;
  if (!Number.isFinite(t)) return t > 0 ? 1.0 : 0.0;
  const tail = upperTail(Math.abs(t), df);
  return t > 0 ? 1.0 - tail : tail;
}

/** The t >= 0 with log P(T > t) = logTail (logTail <= log 1/2); Infinity if beyond every float. */
export function tIsfLog(logTail: number, df: number): number {
  checkDf(df);
  if (Number.isNaN(logTail)) return NaN;
  if (logTail >= Math.log(0.5)) return 0.0;
  let lo = 0.0, hi = 1.0;
  while (logUpperTail(hi, df) > logTail) {
    [lo, hi] = [hi, 2.0 * hi];
    if (hi === Infinity) return Infinity;
  }
  for (let i = 0; i < 400; i++) {
    const mid = lo > 0.0 && hi > 4.0 * lo ? Math.sqrt(lo * hi) : 0.5 * (lo + hi);
    if (logUpperTail(mid, df) > logTail) lo = mid;
    else hi = mid;
    if (hi - lo <= 2e-16 * hi) break;
  }
  return 0.5 * (lo + hi);
}
