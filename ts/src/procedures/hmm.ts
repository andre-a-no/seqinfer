// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Forward filtering in a hidden Markov model; mirrors procedures/hmm.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { NumericalError } from "../errors.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { fsum } from "./ab.js";
import { lgamma } from "./distributions.js";
import { BaseProcedure, type Config, invalid, nums, obj, str } from "./util.js";

const LOG_SQRT_2PI = 0.5 * Math.log(2.0 * Math.PI);

function distribution(values: unknown, size: number | null, name: string): number[] {
  if (!Array.isArray(values) || !values.every((v) => typeof v === "number")) invalid(`${name} must be a list of numbers`);
  const p = [...(values as number[])];
  if (size !== null && p.length !== size) invalid(`${name} must have ${size} entries`);
  if (!p.length || p.some((v) => !(Number.isFinite(v) && v >= 0)) || Math.abs(fsum(p) - 1.0) > 1e-12)
    invalid(`${name} must be a probability distribution (non-negative, summing to 1)`);
  return p;
}

export interface Emission {
  readonly states: number;
  spec(): Config;
  contract(name: string): InputContract;
  logDensity(k: number, y: number): number;
}

export class CategoricalEmission implements Emission {
  readonly p: number[][];
  readonly states: number;

  constructor(probabilities: unknown) {
    if (!Array.isArray(probabilities)) invalid("probabilities must be a list of rows");
    this.p = probabilities.map((row) => distribution(row, null, "emission row"));
    if (new Set(this.p.map((r) => r.length)).size !== 1) invalid("every state needs a distribution over the same symbols");
    this.states = this.p.length;
  }

  spec(): Config {
    return { model: "categorical", probabilities: this.p.map((r) => [...r]) };
  }

  contract(name: string): InputContract {
    return new InputContract(name, "integer", [], 0, this.p[0].length - 1);
  }

  logDensity(k: number, y: number): number {
    const p = this.p[k][y];
    return p > 0 ? Math.log(p) : -Infinity;
  }
}

export class GaussianEmission implements Emission {
  readonly states: number;

  constructor(readonly means: number[], readonly sds: number[]) {
    if (means.length !== sds.length || !means.length) invalid("one mean and one standard deviation per state");
    if (!means.every(Number.isFinite) || !sds.every((s) => 0 < s && s < Infinity))
      invalid("means must be finite and standard deviations positive");
    this.states = means.length;
  }

  spec(): Config {
    return { model: "gaussian", means: [...this.means], sds: [...this.sds] };
  }

  contract(name: string): InputContract {
    return new InputContract(name);
  }

  logDensity(k: number, y: number): number {
    const z = (y - this.means[k]) / this.sds[k];
    return -0.5 * z * z - Math.log(this.sds[k]) - LOG_SQRT_2PI;
  }
}

export class PoissonEmission implements Emission {
  readonly states: number;

  constructor(readonly rates: number[]) {
    if (!rates.length || !rates.every((r) => 0 < r && r < Infinity)) invalid("rates must be positive and finite");
    this.states = rates.length;
  }

  spec(): Config {
    return { model: "poisson", rates: [...this.rates] };
  }

  contract(name: string): InputContract {
    return new InputContract(name, "integer", [], 0);
  }

  logDensity(k: number, y: number): number {
    return y * Math.log(this.rates[k]) - this.rates[k] - lgamma(y + 1.0);
  }
}

export function emissionFromSpec(spec: Config): Emission {
  switch (str(spec, "model")) {
    case "categorical":
      if (!("probabilities" in spec)) invalid("config is missing 'probabilities'");
      return new CategoricalEmission(spec.probabilities);
    case "gaussian":
      return new GaussianEmission(nums(spec, "means"), nums(spec, "sds"));
    case "poisson":
      return new PoissonEmission(nums(spec, "rates"));
    default:
      return invalid(`unknown emission model ${JSON.stringify(spec.model)}`);
  }
}

type HMMState = { n: number; probabilities: number[]; log_likelihood: number; comp: number };

export class HiddenMarkovFilter extends BaseProcedure {
  readonly name = "hidden_markov_filter";
  readonly version = "1";
  readonly initial: number[];
  readonly transition: number[][];
  readonly states: number;

  constructor(initial: unknown, transition: unknown, readonly emission: Emission, readonly input = "y") {
    super();
    this.initial = distribution(initial, null, "initial");
    const k = this.initial.length;
    if (!Array.isArray(transition)) invalid("transition must be a list of rows");
    this.transition = transition.map((row) => distribution(row, k, "transition row"));
    if (this.transition.length !== k) invalid(`transition must be ${k} x ${k}`);
    if (emission.states !== k) invalid(`the emission model must describe ${k} states`);
    this.states = k;
  }

  static fromConfig(c: Config): HiddenMarkovFilter {
    for (const key of ["initial", "transition"]) if (!(key in c)) invalid(`config is missing '${key}'`);
    return new HiddenMarkovFilter(c.initial, c.transition, emissionFromSpec(obj(c, "emission")), str(c, "input"));
  }

  config(): Config {
    return { initial: [...this.initial], transition: this.transition.map((r) => [...r]), emission: this.emission.spec(), input: this.input };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.emission.contract(this.input) };
  }

  initialState(): HMMState {
    return { n: 0, probabilities: [], log_likelihood: 0.0, comp: 0.0 };
  }

  /** P(z_{n+1} = k | y_1..y_n). */
  predicted(s: HMMState): number[] {
    if (s.n === 0) return this.initial;
    const out: number[] = [];
    for (let k = 0; k < this.states; k++) {
      let total = 0.0;
      for (let j = 0; j < this.states; j++) total += s.probabilities[j] * this.transition[j][k];
      out.push(total);
    }
    return out;
  }

  step(state: State, x: StatInput): [HMMState, State] {
    const s = state as HMMState;
    const y = x[this.input] as number;
    const prior = this.predicted(s);
    const logs = prior.map((p, k) => (p > 0 ? this.emission.logDensity(k, y) : -Infinity));
    const top = Math.max(...logs);
    if (top === -Infinity) throw new NumericalError(`${this.name}: observation ${y} is impossible in every reachable state`);
    const joint = prior.map((p, k) => p * Math.exp(logs[k] - top));
    let total = 0.0;
    for (const v of joint) total += v;
    const posterior = joint.map((v) => v / total);
    const [logLik, comp] = neumaierAdd(s.log_likelihood, s.comp, top + Math.log(total));
    const n = s.n + 1;
    let best = 0;
    for (let k = 1; k < this.states; k++) if (posterior[k] > posterior[best]) best = k;
    return [
      { n, probabilities: posterior, log_likelihood: logLik, comp },
      { n, probabilities: [...posterior], most_likely: best, log_likelihood: requireFinite(logLik + comp, "log likelihood") },
    ];
  }

  decodeState(data: Json): HMMState {
    const s = decodeFields({ n: "count", probabilities: "numbers", log_likelihood: "number", comp: "number" }, data) as HMMState;
    requireState(s.probabilities.length === (s.n ? this.states : 0), "one probability per state");
    requireState(s.probabilities.every((p) => p >= 0), "negative probability");
    requireState(s.n === 0 || Math.abs(fsum(s.probabilities) - 1.0) < 1e-9, "probabilities must sum to 1");
    return s;
  }
}

register("hidden_markov_filter", "1", (c) => HiddenMarkovFilter.fromConfig(c));
