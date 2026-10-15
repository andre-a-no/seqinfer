// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Confidence sequences for a mean without normality; mirrors procedures/robust_mean.py (Catoni: backend "python").
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, int, invalid, num, optNum, pyMax, pyMin, pythonBackend, REJECT_H0, str } from "./util.js";

function check(alpha: number, mean0: number | null): void {
  if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
  if (mean0 !== null && !Number.isFinite(mean0)) invalid("mean0 must be finite or None");
}

function orderedSum(values: number[]): number {
  let total = 0.0;
  for (const v of values) total += v;
  return total;
}

// ---------------------------------------------------------------- Catoni

type CatoniState = { n: number; sample: number[]; lower: number | null; upper: number | null; decision: Decision };

export class CatoniConfidenceSequence extends BaseProcedure {
  readonly name = "catoni_confidence_sequence";
  readonly version = "1";
  readonly backend = "python";
  private readonly logLevel: number;

  constructor(
    readonly sigma: number,
    readonly alpha = 0.05,
    readonly mean0: number | null = null,
    readonly stopOnReject = false,
    readonly input = "x",
  ) {
    super();
    if (!(Number.isFinite(sigma) && sigma > 0)) invalid("sigma must be a positive, finite bound on the standard deviation");
    check(alpha, mean0);
    this.logLevel = Math.log(2.0 / alpha);
  }

  static fromConfig(c: Config): CatoniConfidenceSequence {
    pythonBackend(c, "catoni_confidence_sequence");
    return new CatoniConfidenceSequence(num(c, "sigma"), num(c, "alpha"), optNum(c, "mean0"), bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return {
      sigma: this.sigma, alpha: this.alpha, mean0: this.mean0, stop_on_reject: this.stopOnReject, backend: this.backend,
      input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): CatoniState {
    return { n: 0, sample: [], lower: null, upper: null, decision: null };
  }

  /** lambda_i, the predictable stake on the i-th observation. */
  rate(i: number): number {
    return Math.sqrt((2.0 * this.logLevel) / (this.sigma * this.sigma * i * Math.log(i + 1.0)));
  }

  /** The means kept after the sample (in arrival order), before intersection over time. */
  interval(sample: number[], guess: [number, number] | null = null): [number, number] {
    const rates: number[] = [];
    for (let i = 1; i <= sample.length; i++) rates.push(this.rate(i));
    const budget = 0.5 * (this.sigma * this.sigma) * orderedSum(rates.map((r) => r * r)) + this.logLevel;
    const scale = pyMax(sample.reduce(pyMax) - sample.reduce(pyMin), this.sigma);

    const score = (m: number): [number, number] => {
      let value = 0.0, slope = 0.0;
      for (let k = 0; k < rates.length; k++) {
        const r = rates[k];
        const y = r * (sample[k] - m);
        if (y >= 0.0) {
          value += Math.log1p(y + 0.5 * y * y);
          slope -= (r * (1.0 + y)) / (1.0 + y + 0.5 * y * y);
        } else {
          value -= Math.log1p(-y + 0.5 * y * y);
          slope -= (r * (1.0 - y)) / (1.0 - y + 0.5 * y * y);
        }
      }
      return [value, slope];
    };

    const root = (target: number, start: number): number => {
      let lo = start, hi = start;
      let width = scale;
      while (score(lo)[0] < target) {
        lo -= width;
        width *= 2.0;
      }
      width = scale;
      while (score(hi)[0] > target) {
        hi += width;
        width *= 2.0;
      }
      let m = lo <= start && start <= hi ? start : 0.5 * (lo + hi);
      for (let it = 0; it < 200; it++) {
        let [value, slope] = score(m);
        value -= target;
        if (value > 0.0) lo = m;
        else if (value < 0.0) hi = m;
        else return m;
        if (hi - lo <= 1e-12 * (Math.abs(lo) + Math.abs(hi) + scale)) break;
        const nxt = slope < 0.0 ? m - value / slope : 0.5 * (lo + hi);
        m = lo < nxt && nxt < hi ? nxt : 0.5 * (lo + hi);
      }
      return 0.5 * (lo + hi);
    };

    const centre = orderedSum(sample) / sample.length;
    const [loGuess, hiGuess] = guess !== null ? guess : [centre, centre];
    return [root(budget, loGuess), root(-budget, hiGuess)];
  }

  step(state: State, x: StatInput): [CatoniState, State] {
    const s = state as CatoniState;
    const sample = [...s.sample, x[this.input] as number];
    const n = s.n + 1;
    const guess: [number, number] | null = s.lower === null || s.upper === null ? null : [s.lower, s.upper];
    let [lo, hi] = this.interval(sample, guess);
    if (s.lower !== null) lo = pyMax(lo, s.lower);
    if (s.upper !== null) hi = pyMin(hi, s.upper);
    let decision = s.decision;
    if (decision === null && this.mean0 !== null && !(lo <= this.mean0 && this.mean0 <= hi)) decision = REJECT_H0;
    return [
      { n, sample, lower: lo, upper: hi, decision },
      { n, mean: orderedSum(sample) / n, lower: lo, upper: hi, decision },
    ];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): CatoniState {
    const s = decodeFields({ n: "count", sample: "numbers", lower: "number?", upper: "number?", decision: "decision" }, data) as CatoniState;
    requireState(s.sample.length === s.n, "sample size differs from n");
    requireState((s.lower === null) === (s.n === 0) && (s.n === 0) === (s.upper === null), "bounds without data");
    return s;
  }
}

// ---------------------------------------------------------------- asymptotic

type AsymptoticState = { n: number; mean: number; m2: number; lower: number | null; upper: number | null; decision: Decision };

export class AsymptoticConfidenceSequence extends BaseProcedure {
  readonly name = "asymptotic_confidence_sequence";
  readonly version = "1";
  private readonly rho2: number;

  constructor(
    readonly alpha = 0.05,
    readonly mean0: number | null = null,
    readonly start = 1000,
    readonly tunedFor = 1000.0,
    readonly stopOnReject = false,
    readonly input = "x",
  ) {
    super();
    check(alpha, mean0);
    if (!(Number.isFinite(tunedFor) && tunedFor >= 1)) invalid("tuned_for must be a number of observations, at least 1");
    if (!Number.isInteger(start) || start < 2) invalid("start must be an int, at least 2");
    const c = -2.0 * Math.log(alpha);
    this.rho2 = (c + Math.log1p(c)) / tunedFor;
  }

  static fromConfig(c: Config): AsymptoticConfidenceSequence {
    return new AsymptoticConfidenceSequence(num(c, "alpha"), optNum(c, "mean0"), int(c, "start"), num(c, "tuned_for"),
      bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return {
      alpha: this.alpha, mean0: this.mean0, start: this.start, tuned_for: this.tunedFor, stop_on_reject: this.stopOnReject,
      input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  initialState(): AsymptoticState {
    return { n: 0, mean: 0.0, m2: 0.0, lower: null, upper: null, decision: null };
  }

  halfWidth(n: number, sd: number): number {
    const v = n * this.rho2 + 1.0;
    return sd * Math.sqrt(((2.0 * v) / (n * n * this.rho2)) * (0.5 * Math.log(v) - Math.log(this.alpha)));
  }

  step(state: State, x: StatInput): [AsymptoticState, State] {
    const s = state as AsymptoticState;
    const v = x[this.input] as number;
    const n = s.n + 1;
    const delta = v - s.mean;
    const mean = s.mean + delta / n;
    const m2 = s.m2 + delta * (v - mean);
    let lower: number | null = null, upper: number | null = null;
    if (n >= this.start && m2 > 0.0) {
      const h = this.halfWidth(n, Math.sqrt(m2 / (n - 1)));
      [lower, upper] = [mean - h, mean + h];
    }
    let decision = s.decision;
    if (decision === null && this.mean0 !== null && lower !== null && upper !== null) {
      if (!(lower <= this.mean0 && this.mean0 <= upper)) decision = REJECT_H0;
    }
    return [{ n, mean, m2, lower, upper, decision }, { n, mean, lower, upper, decision }];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): AsymptoticState {
    const s = decodeFields({ n: "count", mean: "number", m2: "number", lower: "number?", upper: "number?", decision: "decision" },
      data) as AsymptoticState;
    requireState(s.m2 >= 0, "negative sum of squares");
    return s;
  }
}

register("catoni_confidence_sequence", "1", (c) => CatoniConfidenceSequence.fromConfig(c));
register("asymptotic_confidence_sequence", "1", (c) => AsymptoticConfidenceSequence.fromConfig(c));
