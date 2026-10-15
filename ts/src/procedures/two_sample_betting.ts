// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Anytime-valid kernel two-sample test by betting; mirrors procedures/two_sample_betting.py (backend "python").
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, bool, type Config, type Decision, int, invalid, num, optNum, pyMax, pyMin, pythonBackend, REJECT_H0, str } from "./util.js";

const ONS = 2.0 / (2.0 - Math.log(3.0));
const MAX_BET = 0.5;

type KernelState = {
  n: number; a: number[]; b: number[]; bandwidth: number | null; scale: number; bet: number; curvature: number;
  log_wealth: number; max_log_wealth: number; decision: Decision;
};

export class KernelTwoSampleTest extends BaseProcedure {
  readonly name = "kernel_two_sample_test";
  readonly version = "1";
  readonly eValueField = "log_wealth";
  readonly backend = "python";
  private readonly logLevel: number;

  constructor(
    readonly alpha = 0.05,
    readonly bandwidth: number | null = null,
    readonly burnIn = 20,
    readonly stopOnReject = true,
    readonly a = "a",
    readonly b = "b",
  ) {
    super();
    if (!(0 < alpha && alpha < 1)) invalid("alpha must lie in (0, 1)");
    if (bandwidth !== null && !(Number.isFinite(bandwidth) && bandwidth > 0))
      invalid("bandwidth must be positive and finite, or None for the median heuristic");
    if (!Number.isInteger(burnIn) || burnIn < (bandwidth !== null ? 0 : 2))
      invalid("burn_in must be an int (at least 2 when the bandwidth is learnt)");
    if (a === b) invalid("the two inputs need different names");
    this.logLevel = Math.log(1.0 / alpha);
  }

  static fromConfig(c: Config): KernelTwoSampleTest {
    pythonBackend(c, "kernel_two_sample_test");
    return new KernelTwoSampleTest(num(c, "alpha"), optNum(c, "bandwidth"), int(c, "burn_in"), bool(c, "stop_on_reject"),
      str(c, "a"), str(c, "b"));
  }

  config(): Config {
    return {
      alpha: this.alpha, bandwidth: this.bandwidth, burn_in: this.burnIn, backend: this.backend,
      stop_on_reject: this.stopOnReject, a: this.a, b: this.b,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.a]: new InputContract(this.a), [this.b]: new InputContract(this.b) };
  }

  initialState(): KernelState {
    return {
      n: 0, a: [], b: [], bandwidth: this.bandwidth, scale: 0.0, bet: 0.0, curvature: 1.0,
      log_wealth: 0.0, max_log_wealth: 0.0, decision: null,
    };
  }

  /** g(z): mean kernel similarity of z to the A sample minus that to the B sample. */
  static witness(z: number, a: number[], b: number[], bandwidth: number): number {
    if (a.length !== b.length) throw new Error("zip_strict: the iterables have different lengths");
    const c = -0.5 / (bandwidth * bandwidth);
    let total = 0.0;
    for (let i = 0; i < a.length; i++) {
      const du = z - a[i], dw = z - b[i];
      total += Math.exp(c * (du * du)) - Math.exp(c * (dw * dw));
    }
    return total / a.length;
  }

  static medianHeuristic(points: number[]): number {
    const distances: number[] = [];
    for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) distances.push(Math.abs(points[i] - points[j]));
    distances.sort((x, y) => x - y);
    const positive = distances.filter((d) => d > 0.0);
    return positive.length ? positive[Math.floor(positive.length / 2)] : 1.0;
  }

  step(state: State, x: StatInput): [KernelState, State] {
    const s = state as KernelState;
    const va = x[this.a] as number, vb = x[this.b] as number;
    const n = s.n + 1;
    let { bandwidth, scale, bet, curvature } = s;
    let logWealth = s.log_wealth;
    let payoff: number | null = null;
    if (bandwidth !== null && s.n >= pyMax(1, this.burnIn)) {
      const ga = KernelTwoSampleTest.witness(va, s.a, s.b, bandwidth);
      const gb = KernelTwoSampleTest.witness(vb, s.a, s.b, bandwidth);
      const v = scale === 0.0 ? 0.0 : pyMax(-1.0, pyMin(1.0, (ga - gb) / (2.0 * scale)));
      payoff = v;
      logWealth = requireFinite(logWealth + Math.log1p(bet * v), "log wealth");
      const gradient = v / (1.0 + bet * v);
      curvature += gradient * gradient;
      bet = pyMax(-MAX_BET, pyMin(MAX_BET, bet + (ONS * gradient) / curvature));
      scale = pyMax(pyMax(scale, Math.abs(ga)), Math.abs(gb));
    } else if (bandwidth !== null && s.n >= 1) {
      const ga = KernelTwoSampleTest.witness(va, s.a, s.b, bandwidth);
      const gb = KernelTwoSampleTest.witness(vb, s.a, s.b, bandwidth);
      scale = pyMax(pyMax(scale, Math.abs(ga)), Math.abs(gb));
    }
    const a = [...s.a, va], b = [...s.b, vb];
    if (bandwidth === null && n === this.burnIn) bandwidth = KernelTwoSampleTest.medianHeuristic([...a, ...b]);
    const best = pyMax(s.max_log_wealth, logWealth);
    const decision = s.decision || (logWealth >= this.logLevel ? REJECT_H0 : null);
    return [
      { n, a, b, bandwidth, scale, bet, curvature, log_wealth: logWealth, max_log_wealth: best, decision },
      { n, payoff, log_wealth: logWealth, p_value: pyMin(1.0, Math.exp(-best)), decision },
    ];
  }

  override isTerminal(state: State): boolean {
    return this.stopOnReject && state.decision !== null;
  }

  decodeState(data: Json): KernelState {
    const s = decodeFields({
      n: "count", a: "numbers", b: "numbers", bandwidth: "number?", scale: "number", bet: "number", curvature: "number",
      log_wealth: "number", max_log_wealth: "number", decision: "decision",
    }, data) as KernelState;
    requireState(s.a.length === s.b.length && s.b.length === s.n, "samples differ from n");
    requireState(Math.abs(s.bet) <= MAX_BET && s.curvature >= 1.0 && s.scale >= 0.0, "bad bet");
    requireState(s.bandwidth === null || s.bandwidth > 0, "bandwidth must be positive");
    return s;
  }
}

register("kernel_two_sample_test", "1", (c) => KernelTwoSampleTest.fromConfig(c));
