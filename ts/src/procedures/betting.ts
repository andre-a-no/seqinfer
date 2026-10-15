// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Nonparametric anytime-valid tests by betting; mirrors procedures/betting.py (class _Betting, its two tests,
// and BettingConfidenceSequence with backend "python").
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, int, invalid, num, pyMax, pyMin, pythonBackend, REJECT_H0, str } from "./util.js";

const ALTERNATIVES = ["two-sided", "greater", "less"];
const TRUNCATION = 0.5;
const LOG2 = Math.log(2.0);

type BettingState = {
  n: number; mean: number; m2: number; log_up: number; log_down: number;
  max_log_wealth: number; ties: number; decision: Decision;
};
type BettingOutput = { n: number; estimate: number; log_wealth: number; p_value: number; decision: Decision };

function logAdd(a: number, b: number): number {
  const m = pyMax(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/** The shared wealth process (Python's _Betting); subclasses map an input to a value in [0, 1]. */
abstract class Betting extends BaseProcedure {
  readonly eValueField = "log_wealth";
  protected readonly logLevel: number;

  constructor(readonly m: number, readonly alpha: number, readonly alternative: string, readonly stopOnReject: boolean) {
    super();
    if (!(0 < m && m < 1)) invalid("the null value must lie strictly inside the range of the data");
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (!ALTERNATIVES.includes(alternative)) invalid(`alternative must be one of ${ALTERNATIVES}`);
    this.logLevel = Math.log(1.0 / alpha);
  }

  initialState(): BettingState {
    return { n: 0, mean: 0.0, m2: 0.0, log_up: 0.0, log_down: 0.0, max_log_wealth: 0.0, ties: 0, decision: null };
  }

  private bets(s: BettingState): [number, number] {
    const t = s.n;
    const mu = (0.5 + t * s.mean) / (t + 1);
    const s2 = (0.25 + s.m2) / (t + 1);
    const d = mu - this.m;
    const lam = d / (s2 + d * d);
    const up = this.alternative !== "less" ? pyMin(pyMax(lam, 0.0), TRUNCATION / this.m) : 0.0;
    const down = this.alternative !== "greater" ? pyMin(pyMax(-lam, 0.0), TRUNCATION / (1.0 - this.m)) : 0.0;
    return [up, down];
  }

  protected logWealth(logUp: number, logDown: number): number {
    if (this.alternative === "greater") return logUp;
    if (this.alternative === "less") return logDown;
    return logAdd(logUp, logDown) - LOG2;
  }

  protected bet(s: BettingState, y: number, ties: number): [BettingState, BettingOutput] {
    const [up, down] = this.bets(s);
    const logUp = s.log_up + Math.log1p(up * (y - this.m));
    const logDown = s.log_down + Math.log1p(-down * (y - this.m));
    const n = s.n + 1;
    const delta = y - s.mean;
    const mean = s.mean + delta / n;
    const m2 = s.m2 + delta * (y - mean);
    const logWealth = requireFinite(this.logWealth(logUp, logDown), "log wealth");
    const maxLogWealth = pyMax(s.max_log_wealth, logWealth);
    const decision = s.decision || (logWealth >= this.logLevel ? REJECT_H0 : null);
    const next: BettingState = { n, mean, m2, log_up: logUp, log_down: logDown, max_log_wealth: maxLogWealth, ties, decision };
    return [next, this.output(next, logWealth)];
  }

  protected output(s: BettingState, logWealth: number): BettingOutput {
    return { n: s.n, estimate: s.mean, log_wealth: logWealth, p_value: pyMin(1.0, Math.exp(-s.max_log_wealth)), decision: s.decision };
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): BettingState {
    const s = decodeFields({
      n: "count", mean: "number", m2: "number", log_up: "number", log_down: "number",
      max_log_wealth: "number", ties: "count", decision: "decision",
    }, data) as BettingState;
    requireState(s.m2 >= 0, "negative sum of squares");
    return s;
  }
}

/** Anytime-valid test of E[X] = mean0 for observations bounded in [lower, upper]. */
export class BettingMeanTest extends Betting {
  readonly name = "betting_mean_test";
  readonly version = "1";

  constructor(
    readonly mean0: number,
    readonly lower = 0.0,
    readonly upper = 1.0,
    alpha = 0.05,
    alternative = "two-sided",
    stopOnReject = true,
    readonly input = "x",
  ) {
    if (!(Number.isFinite(lower) && Number.isFinite(upper) && lower < upper)) invalid("need finite bounds lower < upper");
    super((mean0 - lower) / (upper - lower), alpha, alternative, stopOnReject);
  }

  static fromConfig(c: Config): BettingMeanTest {
    return new BettingMeanTest(num(c, "mean0"), num(c, "lower"), num(c, "upper"), num(c, "alpha"), str(c, "alternative"),
      bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return {
      mean0: this.mean0, lower: this.lower, upper: this.upper, alpha: this.alpha, alternative: this.alternative,
      stop_on_reject: this.stopOnReject, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input, "real", [], this.lower, this.upper) };
  }

  step(state: State, x: StatInput): [BettingState, State] {
    const s = state as BettingState;
    const y = ((x[this.input] as number) - this.lower) / (this.upper - this.lower);
    const [next, out] = this.bet(s, y, s.ties);
    return [next, { ...out, estimate: this.lower + next.mean * (this.upper - this.lower) }];
  }
}

/** Anytime-valid sign test: H0 says P(X > median0) = P(X < median0). */
export class SequentialSignTest extends Betting {
  readonly name = "sequential_sign_test";
  readonly version = "1";

  constructor(readonly median0 = 0.0, alpha = 0.05, alternative = "two-sided", stopOnReject = true, readonly input = "x") {
    if (!Number.isFinite(median0)) invalid("median0 must be finite");
    super(0.5, alpha, alternative, stopOnReject);
  }

  static fromConfig(c: Config): SequentialSignTest {
    return new SequentialSignTest(num(c, "median0"), num(c, "alpha"), str(c, "alternative"), bool(c, "stop_on_reject"), str(c, "input"));
  }

  config(): Config {
    return { median0: this.median0, alpha: this.alpha, alternative: this.alternative, stop_on_reject: this.stopOnReject, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input) };
  }

  step(state: State, x: StatInput): [BettingState, State] {
    const s = state as BettingState;
    const v = x[this.input] as number;
    if (v === this.median0) {
      const next: BettingState = { ...s, ties: s.ties + 1 };
      return [next, this.output(next, this.logWealth(s.log_up, s.log_down))];
    }
    return this.bet(s, v > this.median0 ? 1.0 : 0.0, s.ties);
  }
}

// ---------------------------------------------------------------- confidence sequence

type BettingCSState = {
  n: number; total: number; squares: number; log_up: number[]; log_down: number[]; lower: number | null; upper: number | null;
};

/** Confidence sequence for the mean of observations bounded in [lower, upper] (Waudby-Smith and Ramdas 2024). */
export class BettingConfidenceSequence extends BaseProcedure {
  readonly name = "betting_confidence_sequence";
  readonly version = "1";
  readonly backend = "python";
  private readonly points: number[];
  private readonly logLevel: number;
  private readonly c: number;

  constructor(readonly lower = 0.0, readonly upper = 1.0, readonly alpha = 0.05, readonly grid = 1000, readonly input = "x") {
    super();
    if (!(Number.isFinite(lower) && Number.isFinite(upper) && lower < upper)) invalid("need finite bounds lower < upper");
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (!Number.isInteger(grid) || grid < 10) invalid("grid must be an int, at least 10");
    this.points = [];
    for (let j = 0; j < grid; j++) this.points.push((j + 0.5) / grid);
    this.logLevel = Math.log(1.0 / alpha);
    this.c = Math.log(2.0 / alpha);
  }

  static fromConfig(c: Config): BettingConfidenceSequence {
    pythonBackend(c, "betting_confidence_sequence");
    return new BettingConfidenceSequence(num(c, "lower"), num(c, "upper"), num(c, "alpha"), int(c, "grid"), str(c, "input"));
  }

  config(): Config {
    return { lower: this.lower, upper: this.upper, alpha: this.alpha, grid: this.grid, backend: this.backend, input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: new InputContract(this.input, "real", [], this.lower, this.upper) };
  }

  initialState(): BettingCSState {
    const zeros = this.points.map(() => 0.0);
    return { n: 0, total: 0.0, squares: 0.0, log_up: zeros, log_down: [...zeros], lower: null, upper: null };
  }

  private scale(y: number): number {
    return this.lower + y * (this.upper - this.lower);
  }

  step(state: State, x: StatInput): [BettingCSState, State] {
    const s = state as BettingCSState;
    const y = ((x[this.input] as number) - this.lower) / (this.upper - this.lower);
    const i = s.n + 1;
    const mu = (0.5 + s.total) / i;
    const s2 = (0.25 + s.squares) / i;
    const stake = Math.sqrt((2.0 * this.c) / (s2 * i * Math.log1p(i)));
    const logUp: number[] = [], logDown: number[] = [], kept: number[] = [];
    this.points.forEach((mj, j) => {
      const up = s.log_up[j] + Math.log1p(pyMin(stake, TRUNCATION / mj) * (y - mj));
      const down = s.log_down[j] + Math.log1p(-pyMin(stake, TRUNCATION / (1.0 - mj)) * (y - mj));
      logUp.push(up);
      logDown.push(down);
      if (logAdd(up, down) - LOG2 < this.logLevel) kept.push(mj);
    });
    const stepSize = 1.0 / this.grid;
    let lo = kept.length ? pyMax(0.0, kept[0] - stepSize) : mu;
    let hi = kept.length ? pyMin(1.0, kept[kept.length - 1] + stepSize) : mu;
    lo = this.scale(lo);
    hi = this.scale(hi);
    if (s.lower !== null && s.upper !== null) {
      lo = pyMax(lo, s.lower);
      hi = pyMin(hi, s.upper);
    }
    const total = s.total + y;
    const d = y - mu;
    return [
      { n: i, total, squares: s.squares + d * d, log_up: logUp, log_down: logDown, lower: lo, upper: hi },
      { n: i, mean: this.scale(total / i), lower: lo, upper: hi },
    ];
  }

  decodeState(data: Json): BettingCSState {
    const s = decodeFields({
      n: "count", total: "number", squares: "number", log_up: "numbers", log_down: "numbers", lower: "number?", upper: "number?",
    }, data) as BettingCSState;
    requireState(s.log_up.length === this.grid && s.log_down.length === this.grid, "one wealth per grid point");
    requireState(s.squares >= 0 && 0 <= s.total && s.total <= s.n, "running sums out of range");
    return s;
  }
}

register("betting_mean_test", "1", (c) => BettingMeanTest.fromConfig(c));
register("sequential_sign_test", "1", (c) => SequentialSignTest.fromConfig(c));
register("betting_confidence_sequence", "1", (c) => BettingConfidenceSequence.fromConfig(c));
