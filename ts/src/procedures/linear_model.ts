// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Anytime-valid tests and confidence sequences for regression coefficients; mirrors procedures/linear_model.py
// (AnytimeLinearModel), every sum in the same order.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, invalid, num, nums, pyMax, pyMin, REJECT_H0, str } from "./util.js";

/** A pivot of the Cholesky factorisation smaller than this fraction of its diagonal entry counts as zero. */
const RANK_TOLERANCE = 1e-10;
/** A residual sum of squares below this fraction of the total is rounding. */
const EXACT_FIT = 1e-13;

type Rows = number[][];

/** Lower-triangular L with L L' = A[:size, :size], or null if A is (numerically) singular. */
function cholesky(a: Rows, size: number): Rows | null {
  const low = Array.from({ length: size }, () => new Array<number>(size).fill(0.0));
  for (let i = 0; i < size; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j];
      for (let k = 0; k < j; k++) s -= low[i][k] * low[j][k];
      if (i === j) {
        if (!(a[i][i] > 0.0 && s > RANK_TOLERANCE * a[i][i])) return null;
        low[i][i] = Math.sqrt(s);
      } else low[i][j] = s / low[j][j];
    }
  }
  return low;
}

type LinearModelState = { n: number; mean: number[]; comoment: number[]; max_log_e_value: number; decision: Decision };

type Fit = { estimates: number[] | null; f: number | null; logE: number; lower: number | null; upper: number | null };

const SINGULAR: Fit = { estimates: null, f: null, logE: 0.0, lower: null, upper: null };

function names(value: Json, what: string): string[] {
  if (!Array.isArray(value)) return invalid(`${what} must be a list of input names`);
  if (!value.every((v) => typeof v === "string" && v)) invalid(`${what} must be non-empty strings`);
  return [...(value as string[])];
}

/** Anytime-valid F-test (t-test for d = 1) and confidence sequence for regression coefficients. */
export class AnytimeLinearModel extends BaseProcedure {
  readonly name = "anytime_linear_model";
  readonly version = "1";
  readonly eValueField = "log_e_value";
  readonly covariates: string[];
  readonly tested: string[];
  readonly delta0: number[];
  private readonly columns: string[];
  private readonly k: number;
  private readonly d: number;
  private readonly m: number;
  private readonly p: number;
  private readonly logLevel: number;

  constructor(
    covariates: Json,
    tested: Json,
    delta0: number[] | null = null,
    readonly g = 1.0,
    readonly alpha = 0.05,
    readonly intercept = true,
    readonly stopOnReject = false,
    readonly input = "y",
  ) {
    super();
    this.covariates = names(covariates, "covariates");
    this.tested = names(tested, "tested");
    if (!this.tested.length) invalid("at least one coefficient must be tested");
    if (!(typeof input === "string" && input)) invalid("input must be a non-empty string");
    const everything = [...this.covariates, ...this.tested, input];
    if (new Set(everything).size !== everything.length) invalid("the outcome, covariates and tested inputs need different names");
    const d = this.tested.length;
    if (delta0 === null) delta0 = this.tested.map(() => 0.0);
    if (delta0.length !== d) invalid(`delta0 must be a list of ${d} numbers, one per tested input`);
    if (!delta0.every(Number.isFinite)) invalid("delta0 must be finite numbers");
    if (!(0 < g && g <= 1e100)) invalid("g (prior variance of a standardised coefficient) must lie in (0, 1e100]");
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    this.delta0 = [...delta0];
    this.columns = [...this.covariates, ...this.tested];
    this.k = this.covariates.length;
    this.d = d;
    this.m = this.columns.length + 1;
    this.p = this.k + (intercept ? 1 : 0);
    this.logLevel = Math.log(1.0 / alpha);
  }

  static fromConfig(c: Config): AnytimeLinearModel {
    if (!("covariates" in c) || !("tested" in c)) invalid("config is missing 'covariates' or 'tested'");
    return new AnytimeLinearModel(c.covariates, c.tested, nums(c, "delta0"), num(c, "g"), num(c, "alpha"), bool(c, "intercept"),
      bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return {
      covariates: [...this.covariates], tested: [...this.tested], delta0: [...this.delta0], g: this.g, alpha: this.alpha,
      intercept: this.intercept, stop_on_reject: this.stopOnReject, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    const out: Record<string, InputContract> = {};
    for (const name of [...this.columns, this.input]) out[name] = new InputContract(name);
    return out;
  }

  initialState(): LinearModelState {
    const m = this.m;
    return {
      n: 0, mean: this.intercept ? new Array<number>(m).fill(0.0) : [], comoment: new Array<number>((m * (m + 1)) / 2).fill(0.0),
      max_log_e_value: 0.0, decision: null,
    };
  }

  private unpack(packed: number[]): Rows {
    const m = this.m;
    const a = Array.from({ length: m }, () => new Array<number>(m).fill(0.0));
    let at = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j <= i; j++) {
        a[i][j] = a[j][i] = packed[at];
        at += 1;
      }
    }
    return a;
  }

  /** t*^2 / nu at which B_n reaches 1/alpha (d = 1, logRInv = log(1 + g s)), or null. */
  criticalLogRatio(nu: number, logRInv: number): number | null {
    const logK = (2.0 * (this.logLevel + 0.5 * logRInv)) / (nu + 1);
    if (logK >= logRInv) return null;
    return Math.expm1(logK) / -Math.expm1(logK - logRInv);
  }

  /** Estimates, F, log B_n and (d = 1) the interval from the sufficient statistics. */
  fit(n: number, comoment: number[]): Fit {
    const a = this.unpack(comoment);
    const q = this.m - 1, k = this.k, d = this.d;
    const low = cholesky(a, q);
    if (low === null) return SINGULAR;
    const c = new Array<number>(q).fill(0.0);
    for (let i = 0; i < q; i++) {
      let s = a[q][i];
      for (let j = 0; j < i; j++) s -= low[i][j] * c[j];
      c[i] = s / low[i][i];
    }
    const est = new Array<number>(d).fill(0.0);
    for (let i = d - 1; i >= 0; i--) {
      let s = c[k + i];
      for (let j = i + 1; j < d; j++) s -= low[k + j][k + i] * est[j];
      est[i] = s / low[k + i][k + i];
    }
    const estimates = est.map((v) => requireFinite(v, "coefficient estimate"));
    const nu = n - this.p - d;
    if (nu < 1) return { estimates, f: null, logE: 0.0, lower: null, upper: null };
    const total = a[q][q];
    let rss = total;
    for (const value of c) rss -= value * value;
    if (!(rss > EXACT_FIT * total)) rss = 0.0;
    const c2 = new Array<number>(d).fill(0.0);
    for (let i = 0; i < d; i++) {
      let s = c[k + i];
      for (let j = i; j < d; j++) s -= low[k + j][k + i] * this.delta0[j];
      c2[i] = s;
    }
    const b = Array.from({ length: d }, () => new Array<number>(d).fill(0.0));
    for (let i = 0; i < d; i++) {
      for (let j = 0; j <= i; j++) {
        let s = 0.0;
        for (let t = pyMax(i, j); t < d; t++) s += low[k + t][k + i] * low[k + t][k + j];
        b[i][j] = b[j][i] = (i === j ? 1.0 : 0.0) + this.g * s;
      }
    }
    const lb = cholesky(b, d);
    if (lb === null) throw new RangeError("I + g S is not positive definite");
    let logDet = 0.0;
    for (let i = 0; i < d; i++) logDet += 2.0 * Math.log(lb[i][i]);
    const v = new Array<number>(d).fill(0.0);
    let q0 = 0.0, q1 = 0.0;
    for (let i = 0; i < d; i++) {
      let s = c2[i];
      for (let j = 0; j < i; j++) s -= lb[i][j] * v[j];
      v[i] = s / lb[i][i];
      q0 += c2[i] * c2[i];
      q1 += v[i] * v[i];
    }
    const half = 0.5 * (nu + d);
    let logRatio: number, f: number | null;
    if (rss > 0.0) {
      logRatio = Math.log1p(q0 / rss) - Math.log1p(q1 / rss);
      f = q0 / d / (rss / nu);
    } else {
      logRatio = q0 > 0.0 ? Math.log(q0) - Math.log(q1) : 0.0;
      f = null;
    }
    const logE = requireFinite(-0.5 * logDet + half * logRatio, "log Bayes factor");
    let lower: number | null = null, upper: number | null = null;
    if (d === 1) {
      const ratio = this.criticalLogRatio(nu, logDet);
      if (ratio !== null) {
        const width = Math.sqrt(ratio * rss) / low[k][k];
        [lower, upper] = [estimates[0] - width, estimates[0] + width];
      }
    }
    return { estimates, f, logE, lower, upper };
  }

  step(state: State, x: StatInput): [LinearModelState, State] {
    const s = state as LinearModelState;
    const w = this.columns.map((name) => x[name] as number);
    w.push(x[this.input] as number);
    const m = this.m, n = s.n + 1;
    let mean: number[], dev: number[], scaled: number[];
    if (this.intercept) {
      dev = w.map((v, i) => v - s.mean[i]);
      mean = dev.map((dv, i) => requireFinite(s.mean[i] + dv / n, "running mean"));
      const factor = s.n / n;
      scaled = dev.map((v) => v * factor);
    } else {
      mean = [];
      dev = w;
      scaled = w;
    }
    const comoment = [...s.comoment];
    let at = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j <= i; j++) {
        comoment[at] = requireFinite(comoment[at] + scaled[i] * dev[j], "cross product");
        at += 1;
      }
    }
    const fit = this.fit(n, comoment);
    const maxLogE = pyMax(s.max_log_e_value, fit.logE);
    const decision = s.decision || (fit.logE >= this.logLevel ? REJECT_H0 : null);
    const pValue = pyMin(1.0, Math.exp(-maxLogE));
    return [
      { n, mean, comoment, max_log_e_value: maxLogE, decision },
      { n, estimates: fit.estimates, f: fit.f, log_e_value: fit.logE, p_value: pValue, lower: fit.lower, upper: fit.upper, decision },
    ];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): LinearModelState {
    const s = decodeFields({
      n: "count", mean: "numbers", comoment: "numbers", max_log_e_value: "number", decision: "decision",
    }, data) as LinearModelState;
    const m = this.m;
    requireState(s.mean.length === (this.intercept ? m : 0), "wrong number of means");
    requireState(s.comoment.length === (m * (m + 1)) / 2, "wrong size of the cross-product matrix");
    const a = this.unpack(s.comoment);
    let diagonal = true, schwarz = true;
    for (let i = 0; i < m; i++) {
      if (!(a[i][i] >= 0.0)) diagonal = false;
      for (let j = 0; j < i; j++) if (!(Math.abs(a[i][j]) <= Math.sqrt(a[i][i] * a[j][j]) * (1.0 + 1e-9))) schwarz = false;
    }
    requireState(diagonal, "negative sum of squares");
    requireState(schwarz, "cross products violate the Cauchy-Schwarz inequality");
    if (s.n === 0) requireState([...s.mean, ...s.comoment].every((v) => v === 0.0), "statistics without observations");
    requireState(s.decision === null || s.decision === REJECT_H0, "unknown decision");
    requireState((s.decision !== null) === (s.max_log_e_value >= this.logLevel), "the decision does not match the largest e-value");
    requireState(s.max_log_e_value >= 0.0, "the largest log e-value is below its initial 0");
    return s;
  }
}

register("anytime_linear_model", "1", (c) => AnytimeLinearModel.fromConfig(c));
