// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Anytime-valid log-rank test and hazard-ratio confidence sequence; mirrors procedures/survival.py (AnytimeLogRankTest:
// methods "mixture" and "grow", the confidence sequence grid with its convolution and log-domain fallback).
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { ContractViolation } from "../errors.js";
import { logSumExp, neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, int, invalid, num, nums, pyMax, pyMin, REJECT_H0, str } from "./util.js";

const ALTERNATIVES = ["two-sided", "less", "greater"];
const METHODS = ["mixture", "grow"];
const ENTER = 0, EVENT = 1, CENSOR = 2;
const STEPS = 8; // grid points per prior standard deviation
const SPAN = 4; // prior standard deviations on each side
const HALF = STEPS * SPAN; // 32: the alternative grid has 2 * 32 + 1 points
const TINY = 1e-280; // below this a convolution is recomputed in the log domain

type LogRankState = {
  at_risk: number[]; n_events: number; log_lik: number[]; log_e: number; comp: number; cs_log_lik: number[];
  cs_kept: number[]; max_log_e: number; decision: Decision;
};

/** Python's round(x) for a float: to the nearest integer, ties to even. */
function pyRound(x: number): number {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff < 0.5) return f;
  if (diff > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/** Python's max(values): the first largest. */
function firstMax(values: number[]): number {
  let m = values[0];
  for (let i = 1; i < values.length; i++) if (values[i] > m) m = values[i];
  return m;
}

/** log LR_t: x delta + log(n_T theta0 + n_C) - log(n_T theta1 + n_C). */
function logFactor(x: number, nT: number, nC: number, delta: number, theta0: number, theta1: number): number {
  return x * delta + Math.log(nT * theta0 + nC) - Math.log(nT * theta1 + nC);
}

/** Anytime-valid log-rank test and confidence sequence for a hazard ratio. */
export class AnytimeLogRankTest extends BaseProcedure {
  readonly name = "anytime_logrank_test";
  readonly version = "1";
  readonly eValueField = "log_e_value";
  readonly csRange: [number, number];
  readonly arms: [number, number];
  private readonly logLevel: number;
  private readonly step_: number;
  private readonly deltas: number[];
  private readonly thetas: number[];
  private readonly penalty: number[];
  private readonly mixIndex: number[];
  private readonly mixLogw: number[];
  private readonly h: number;
  private readonly c: number;
  private readonly pad: number;
  private readonly csBetas: number[];
  private readonly csExp: number[];
  private readonly csPoints: number[];
  private readonly csOffsets: number[];
  private readonly csLogw: number[];
  private readonly csW: number[];

  constructor(
    readonly alpha = 0.05,
    readonly theta0 = 1.0,
    readonly alternative = "two-sided",
    readonly method = "mixture",
    readonly priorSd = 0.5,
    readonly confidenceSequence = true,
    csRange: number[] = [0.1, 10.0],
    readonly csGrid = 101,
    arms: number[] = [1, 0],
    readonly stopOnReject = false,
    readonly armInput = "arm",
    readonly input = "kind",
  ) {
    super();
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (!(Number.isFinite(theta0) && theta0 > 0)) invalid("theta0 must be a positive finite hazard ratio");
    if (!ALTERNATIVES.includes(alternative)) invalid(`alternative must be one of ${ALTERNATIVES}`);
    if (!METHODS.includes(method)) invalid(`method must be one of ${METHODS}`);
    if (!(Number.isFinite(priorSd) && 0.01 <= priorSd && priorSd <= 5))
      invalid("prior_sd must be a number in [0.01, 5] (a standard deviation of the log hazard ratio)");
    if (csRange.length !== 2 || !csRange.every((v) => Number.isFinite(v) && v > 0) || !(csRange[0] < csRange[1]))
      invalid("cs_range must be two positive finite hazard ratios, lower < upper");
    if (!Number.isInteger(csGrid) || !(3 <= csGrid && csGrid <= 2000)) invalid("cs_grid must be an int in [3, 2000]");
    if (arms.length !== 2 || arms.some((a) => !Number.isInteger(a) || a < 0) || arms[0] === arms[1])
      invalid("arms must be two different non-negative integer labels");
    if (armInput === input) invalid("the arm and the kind need different names");
    this.csRange = [csRange[0], csRange[1]];
    this.arms = [arms[0], arms[1]];
    this.logLevel = Math.log(1.0 / alpha);

    const tau = priorSd;
    const step = tau / STEPS;
    this.step_ = step;
    this.deltas = [];
    for (let j = -HALF; j <= HALF; j++) this.deltas.push(j * step);
    this.thetas = this.deltas.map((d) => theta0 * Math.exp(d));
    this.penalty = this.deltas.map((d) => (-d * d) / (2.0 * tau * tau));
    const side: number[] = [];
    if (alternative === "less") for (let j = -HALF; j < 0; j++) side.push(j);
    else if (alternative === "greater") for (let j = 1; j <= HALF; j++) side.push(j);
    else for (let j = -HALF; j <= HALF; j++) if (j !== 0) side.push(j);
    const norm = logSumExp(side.map((j) => this.penalty[j + HALF]));
    this.mixIndex = side.map((j) => j + HALF);
    this.mixLogw = side.map((j) => this.penalty[j + HALF] - norm);

    // confidence sequence: points b_i = log(lo) + i h, offsets m_j = j c h, one grid for all b_i + m_j
    const [lo, hi] = this.csRange;
    this.h = Math.log(hi / lo) / (csGrid - 1);
    this.c = pyMax(1, pyRound(step / this.h));
    this.pad = pyMax(1, Math.trunc((SPAN * tau) / (this.c * this.h))) * this.c;
    const width = csGrid + 2 * this.pad;
    this.csBetas = [];
    for (let u = 0; u < width; u++) this.csBetas.push(Math.log(lo) + (u - this.pad) * this.h);
    this.csExp = this.csBetas.map((b) => Math.exp(b));
    this.csPoints = [];
    for (let i = 0; i < csGrid; i++) this.csPoints.push(lo * Math.exp(i * this.h));
    const offsets: number[] = [];
    const reach = this.pad / this.c; // pad is a multiple of c
    for (let j = -reach; j <= reach; j++) if (j !== 0) offsets.push(j * this.c);
    const logs = offsets.map((m) => {
      const mh = m * this.h;
      return -(mh * mh) / (2.0 * tau * tau);
    });
    const csNorm = logSumExp(logs);
    this.csOffsets = offsets;
    this.csLogw = logs.map((v) => v - csNorm);
    this.csW = this.csLogw.map((v) => Math.exp(v));
  }

  static fromConfig(c: Config): AnytimeLogRankTest {
    return new AnytimeLogRankTest(num(c, "alpha"), num(c, "theta0"), str(c, "alternative"), str(c, "method"), num(c, "prior_sd"),
      bool(c, "confidence_sequence"), nums(c, "cs_range"), int(c, "cs_grid"), nums(c, "arms"), bool(c, "stop_on_reject"),
      str(c, "arm_input"), str(c, "input"));
  }

  config(): Config {
    return {
      alpha: this.alpha, theta0: this.theta0, alternative: this.alternative, method: this.method, prior_sd: this.priorSd,
      confidence_sequence: this.confidenceSequence, cs_range: [...this.csRange], cs_grid: this.csGrid, arms: [...this.arms],
      stop_on_reject: this.stopOnReject, arm_input: this.armInput, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return {
      [this.armInput]: new InputContract(this.armInput, "integer", [], 0),
      [this.input]: new InputContract(this.input, "integer", [], ENTER, CENSOR),
    };
  }

  initialState(): LogRankState {
    let cs: number[] = [], kept = [0, 0];
    if (this.confidenceSequence) {
      cs = this.csBetas.map(() => 0.0);
      kept = [0, this.csGrid - 1];
    }
    return {
      at_risk: [0, 0], n_events: 0, log_lik: this.deltas.map(() => 0.0), log_e: 0.0, comp: 0.0, cs_log_lik: cs, cs_kept: kept,
      max_log_e: 0.0, decision: null,
    };
  }

  // ------------------------------------------------------------ e-values and estimates

  logEValue(s: LogRankState): number {
    if (this.method === "grow") return s.log_e + s.comp;
    return logSumExp(this.mixLogw.map((w, j) => w + s.log_lik[this.mixIndex[j]]));
  }

  /** d: the penalised partial-likelihood estimate of log(theta / theta0), from the grid. */
  estimate(logLik: number[]): number {
    const f = logLik.map((v, i) => v + this.penalty[i]);
    let best = 0;
    for (let i = 1; i < f.length; i++) if (f[i] > f[best]) best = i; // the first maximum
    if (0 < best && best < f.length - 1) {
      const left = f[best - 1], mid = f[best], right = f[best + 1];
      const curvature = left - 2.0 * mid + right;
      const shift = curvature < 0.0 ? (0.5 * (left - right)) / curvature : 0.0;
      return (best - HALF + shift) * this.step_;
    }
    return (best - HALF) * this.step_;
  }

  private bet(logLik: number[]): number {
    const d = this.estimate(logLik);
    if (this.alternative === "less") return pyMin(d, 0.0);
    if (this.alternative === "greater") return pyMax(d, 0.0);
    return d;
  }

  /** log E_n(theta_i) of the two-sided mixture for the CS grid points first..last. */
  csLogEValues(csLogLik: number[], first: number, last: number): number[] {
    const pad = this.pad, offsets = this.csOffsets;
    const window = csLogLik.slice(first, last + 2 * pad + 1);
    const top = firstMax(window);
    const scaled = window.map((v) => Math.exp(v - top));
    const out: number[] = [];
    for (let i = first; i <= last; i++) {
      const centre = i + pad;
      let total = 0.0;
      for (let j = 0; j < offsets.length; j++) total += this.csW[j] * scaled[centre + offsets[j] - first];
      if (total > TINY) out.push(Math.log(total) + top - csLogLik[centre]);
      else {
        const base = csLogLik[centre];
        out.push(logSumExp(offsets.map((m, j) => this.csLogw[j] + csLogLik[centre + m] - base)));
      }
    }
    return out;
  }

  private bounds(kept: number[]): [number | null, number | null] {
    if (!this.confidenceSequence) return [null, null];
    const [first, last] = kept;
    const lower = first > 0 ? this.csPoints[first - 1] : null;
    const upper = last < this.csGrid - 1 ? this.csPoints[last + 1] : null;
    return [lower, upper];
  }

  private output(s: LogRankState, value: number): State {
    const [lower, upper] = this.bounds(s.cs_kept);
    const estimate = this.theta0 * Math.exp(this.estimate(s.log_lik));
    return {
      n_events: s.n_events, at_risk: [...s.at_risk], log_e_value: value, p_value: pyMin(1.0, Math.exp(-s.max_log_e)),
      decision: s.decision, hazard_ratio_estimate: estimate, lower, upper,
    };
  }

  // ------------------------------------------------------------ the step

  step(state: State, x: StatInput): [LogRankState, State] {
    const s = state as LogRankState;
    const label = x[this.armInput] as number;
    if (!this.arms.includes(label)) return [s, this.output(s, this.logEValue(s))];
    const k = this.arms.indexOf(label);
    const kind = x[this.input] as number;
    const [nT, nC] = s.at_risk;
    if (kind === ENTER) {
      const next: LogRankState = { ...s, at_risk: k === 0 ? [nT + 1, nC] : [nT, nC + 1] };
      return [next, this.output(next, this.logEValue(next))];
    }
    if (s.at_risk[k] === 0) {
      const what = kind === EVENT ? "an event" : "a censoring";
      throw new ContractViolation(`${this.name}: ${what} in arm ${label} with nobody at risk`);
    }
    const atRisk = k === 0 ? [nT - 1, nC] : [nT, nC - 1];
    if (kind === CENSOR) {
      const next: LogRankState = { ...s, at_risk: atRisk };
      return [next, this.output(next, this.logEValue(next))];
    }
    let logLik = s.log_lik, logE = s.log_e, comp = s.comp;
    let csLogLik = s.cs_log_lik, kept = s.cs_kept;
    if (nT > 0 && nC > 0) { // otherwise the arm of the event is certain: every factor is 1
      const xt = k === 0 ? 1 : 0;
      const theta0 = this.theta0;
      if (this.method === "grow") {
        const d = this.bet(logLik);
        if (d !== 0.0) [logE, comp] = neumaierAdd(logE, comp, logFactor(xt, nT, nC, d, theta0, theta0 * Math.exp(d)));
      }
      logLik = logLik.map((v, j) => v + logFactor(xt, nT, nC, this.deltas[j], theta0, this.thetas[j]));
      if (this.confidenceSequence) {
        csLogLik = csLogLik.map((v, u) => v + logFactor(xt, nT, nC, this.csBetas[u], 1.0, this.csExp[u]));
        const [first, last] = kept;
        const values = this.csLogEValues(csLogLik, first, last);
        const inside: number[] = [];
        values.forEach((v, i) => {
          if (v < this.logLevel) inside.push(first + i);
        });
        if (inside.length) kept = [inside[0], inside[inside.length - 1]];
        else { // every point excluded: collapse onto the least-excluded one
          let least = 0;
          for (let i = 1; i < values.length; i++) if (values[i] < values[least]) least = i;
          kept = [first + least, first + least];
        }
      }
    }
    const partial: LogRankState = {
      at_risk: atRisk, n_events: s.n_events + 1, log_lik: logLik, log_e: logE, comp, cs_log_lik: csLogLik, cs_kept: kept,
      max_log_e: s.max_log_e, decision: s.decision,
    };
    const value = requireFinite(this.logEValue(partial), "log e-value");
    const decision = s.decision || (value >= this.logLevel ? REJECT_H0 : null);
    const next: LogRankState = { ...partial, max_log_e: pyMax(s.max_log_e, value), decision };
    return [next, this.output(next, value)];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): LogRankState {
    const s = decodeFields({
      at_risk: "counts", n_events: "count", log_lik: "numbers", log_e: "number", comp: "number", cs_log_lik: "numbers",
      cs_kept: "counts", max_log_e: "number", decision: "decision",
    }, data) as LogRankState;
    requireState(s.at_risk.length === 2 && s.cs_kept.length === 2, "two arms, two CS indices");
    requireState(s.log_lik.length === this.deltas.length, "one log-likelihood per alternative");
    requireState(s.log_lik[HALF] === 0.0, "the null point has likelihood ratio one");
    if (this.method === "mixture") requireState(s.log_e === 0.0 && s.comp === 0.0, "a mixture keeps no plug-in product");
    if (this.confidenceSequence) {
      requireState(s.cs_log_lik.length === this.csBetas.length, "one log-likelihood per CS grid point");
      const [first, last] = s.cs_kept;
      requireState(first <= last && last <= this.csGrid - 1, "CS indices out of range");
    } else {
      requireState(s.cs_log_lik.length === 0 && s.cs_kept[0] === 0 && s.cs_kept[1] === 0, "no confidence sequence");
    }
    if (s.n_events === 0) {
      requireState([...s.log_lik, ...s.cs_log_lik].every((v) => v === 0.0) && s.log_e === 0.0 && s.comp === 0.0
        && s.max_log_e === 0.0, "no events, no evidence");
      if (this.confidenceSequence)
        requireState(s.cs_kept[0] === 0 && s.cs_kept[1] === this.csGrid - 1, "no events, nothing excluded");
    }
    const value = this.logEValue(s);
    requireState(s.max_log_e >= 0.0 && s.max_log_e >= value, "the running maximum");
    requireState(s.decision === null || s.decision === REJECT_H0, "the test only rejects");
    requireState((s.decision !== null) === (s.max_log_e >= this.logLevel), "the decision follows the running maximum");
    return s;
  }
}

register("anytime_logrank_test", "1", (c) => AnytimeLogRankTest.fromConfig(c));
