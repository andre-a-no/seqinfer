// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Bootstrap particle filter with adaptive systematic resampling (randomized); mirrors procedures/particle_filter.py
// with backend "python" and the scalar models LinearGaussianModel, StochasticVolatilityModel, StudentTLocalLevel.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { NumericalError } from "../errors.js";
import { logSumExp, neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import type { SplitMix64 } from "../rng.js";
import { lgamma } from "./distributions.js";
import { BaseProcedure, type Config, int, invalid, num, obj, pythonBackend, str } from "./util.js";

const LOG_SQRT_2PI = 0.5 * Math.log(2.0 * Math.PI);

/** A scalar state-space model: how the hidden state starts, moves, and explains an observation. */
export interface StateSpaceModel {
  readonly name: string;
  readonly version: string;
  config(): Config;
  initial(rng: SplitMix64): number;
  transition(state: number, rng: SplitMix64): number;
  logLikelihood(state: number, y: number): number;
  contract(name: string): InputContract;
}

function normalLogDensity(y: number, mean: number, sd: number): number {
  const z = (y - mean) / sd;
  return -0.5 * z * z - Math.log(sd) - LOG_SQRT_2PI;
}

export class LinearGaussianModel implements StateSpaceModel {
  readonly name = "linear_gaussian";
  readonly version = "1";

  constructor(readonly phi: number, readonly q: number, readonly r: number, readonly m0 = 0.0, readonly p0 = 1.0) {
    if (!(Number.isFinite(phi) && Number.isFinite(m0) && 0 <= q && q < Infinity && 0 < r && r < Infinity && 0 <= p0 && p0 < Infinity))
      invalid("need finite phi and m0, q >= 0, r > 0, p0 >= 0");
  }

  config(): Config {
    return { phi: this.phi, q: this.q, r: this.r, m0: this.m0, p0: this.p0 };
  }

  initial(rng: SplitMix64): number {
    return rng.normal(this.m0, Math.sqrt(this.p0));
  }

  transition(state: number, rng: SplitMix64): number {
    return rng.normal(this.phi * state, Math.sqrt(this.q));
  }

  logLikelihood(state: number, y: number): number {
    return normalLogDensity(y, state, Math.sqrt(this.r));
  }

  contract(name: string): InputContract {
    return new InputContract(name);
  }
}

export class StochasticVolatilityModel implements StateSpaceModel {
  readonly name = "stochastic_volatility";
  readonly version = "1";

  constructor(readonly mu: number, readonly phi: number, readonly sigma: number) {
    if (!(Number.isFinite(mu) && -1 < phi && phi < 1 && 0 < sigma && sigma < Infinity)) invalid("need a finite mu, |phi| < 1 and sigma > 0");
  }

  config(): Config {
    return { mu: this.mu, phi: this.phi, sigma: this.sigma };
  }

  initial(rng: SplitMix64): number {
    return rng.normal(this.mu, this.sigma / Math.sqrt(1.0 - this.phi * this.phi));
  }

  transition(state: number, rng: SplitMix64): number {
    return rng.normal(this.mu + this.phi * (state - this.mu), this.sigma);
  }

  logLikelihood(state: number, y: number): number {
    return normalLogDensity(y, 0.0, Math.exp(0.5 * state));
  }

  contract(name: string): InputContract {
    return new InputContract(name);
  }
}

export class StudentTLocalLevel implements StateSpaceModel {
  readonly name = "student_t_local_level";
  readonly version = "1";
  private readonly constant: number;

  constructor(readonly q: number, readonly scale: number, readonly df: number, readonly m0 = 0.0, readonly p0 = 1.0) {
    if (!(0 <= q && q < Infinity && 0 < scale && scale < Infinity && 0 < df && df < Infinity && Number.isFinite(m0) && 0 <= p0 && p0 < Infinity))
      invalid("need q >= 0, positive scale and df, finite m0, p0 >= 0");
    this.constant = lgamma(0.5 * (df + 1.0)) - lgamma(0.5 * df) - 0.5 * Math.log(df * Math.PI) - Math.log(scale);
  }

  config(): Config {
    return { q: this.q, scale: this.scale, df: this.df, m0: this.m0, p0: this.p0 };
  }

  initial(rng: SplitMix64): number {
    return rng.normal(this.m0, Math.sqrt(this.p0));
  }

  transition(state: number, rng: SplitMix64): number {
    return rng.normal(state, Math.sqrt(this.q));
  }

  logLikelihood(state: number, y: number): number {
    const z = (y - state) / this.scale;
    return this.constant - 0.5 * (this.df + 1.0) * Math.log1p((z * z) / this.df);
  }

  contract(name: string): InputContract {
    return new InputContract(name);
  }
}

/** A model from its spec {name, version, config}. */
export function stateSpaceModelFromSpec(spec: Config): StateSpaceModel {
  const name = str(spec, "name"), version = str(spec, "version");
  const c = obj(spec, "config");
  if (version !== "1") invalid(`no TypeScript implementation of state-space model ${name} version ${version}`);
  switch (name) {
    case "linear_gaussian":
      return new LinearGaussianModel(num(c, "phi"), num(c, "q"), num(c, "r"), num(c, "m0"), num(c, "p0"));
    case "stochastic_volatility":
      return new StochasticVolatilityModel(num(c, "mu"), num(c, "phi"), num(c, "sigma"));
    case "student_t_local_level":
      return new StudentTLocalLevel(num(c, "q"), num(c, "scale"), num(c, "df"), num(c, "m0"), num(c, "p0"));
    default:
      return invalid(`no TypeScript implementation of state-space model ${name}`);
  }
}

type ParticleFilterState = { n: number; particles: number[]; log_weights: number[]; log_likelihood: number; comp: number };

export class ParticleFilter extends BaseProcedure {
  readonly name = "particle_filter";
  readonly version = "1";
  override readonly randomized = true;
  readonly backend = "python";

  constructor(readonly model: StateSpaceModel, readonly size = 500, readonly threshold = 0.5, readonly input = "y") {
    super();
    if (!Number.isInteger(size) || size < 2) invalid("particles must be an int, at least 2");
    if (!(0 <= threshold && threshold <= 1)) invalid("resample_threshold must lie in [0, 1] (1 resamples at every step)");
  }

  static fromConfig(c: Config): ParticleFilter {
    pythonBackend(c, "particle_filter");
    return new ParticleFilter(stateSpaceModelFromSpec(obj(c, "model")), int(c, "particles"), num(c, "resample_threshold"), str(c, "input"));
  }

  config(): Config {
    return {
      model: { name: this.model.name, version: this.model.version, config: this.model.config() }, particles: this.size,
      resample_threshold: this.threshold, backend: this.backend, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.model.contract(this.input) };
  }

  initialState(): ParticleFilterState {
    return { n: 0, particles: [], log_weights: [], log_likelihood: 0.0, comp: 0.0 };
  }

  step(state: State, x: StatInput, rng: SplitMix64 | null): [ParticleFilterState, State] {
    if (rng === null) throw new Error(`${this.name} is randomized: it needs the statistical random state`);
    const s = state as ParticleFilterState;
    const y = x[this.input] as number;
    const size = this.size;
    let previous: number[], logW: number[];
    if (s.n === 0) {
      previous = [];
      for (let i = 0; i < size; i++) previous.push(this.model.initial(rng));
      logW = new Array<number>(size).fill(-Math.log(size));
    } else {
      previous = s.particles;
      logW = s.log_weights;
    }
    let moved = previous.map((p) => this.model.transition(p, rng));
    logW = logW.map((w, i) => w + this.model.logLikelihood(moved[i], y));
    const increment = logSumExp(logW);
    if (!Number.isFinite(increment)) throw new NumericalError(`${this.name}: every particle gives the observation zero likelihood`);
    logW = logW.map((w) => w - increment);
    const weights = logW.map((w) => Math.exp(w));
    let square = 0.0;
    for (const w of weights) square += w * w;
    const ess = 1.0 / square;
    let mean = 0.0;
    for (let i = 0; i < size; i++) mean += weights[i] * moved[i];
    let variance = 0.0;
    for (let i = 0; i < size; i++) {
      const d = moved[i] - mean;
      variance += weights[i] * (d * d);
    }
    const resampled = ess < this.threshold * size;
    if (resampled) {
      const u = rng.random() / size;
      const chosen: number[] = [];
      let i = 0, cum = weights[0];
      for (let j = 0; j < size; j++) {
        const target = u + j / size;
        while (cum < target && i < size - 1) {
          i += 1;
          cum += weights[i];
        }
        chosen.push(moved[i]);
      }
      moved = chosen;
      logW = new Array<number>(size).fill(-Math.log(size));
    }
    const [total, comp] = neumaierAdd(s.log_likelihood, s.comp, increment);
    const n = s.n + 1;
    return [
      { n, particles: moved, log_weights: logW, log_likelihood: total, comp },
      { n, mean, variance, ess, log_likelihood: requireFinite(total + comp, "log likelihood"), resampled },
    ];
  }

  decodeState(data: Json): ParticleFilterState {
    const s = decodeFields({ n: "count", particles: "numbers", log_weights: "numbers", log_likelihood: "number", comp: "number" },
      data) as ParticleFilterState;
    const expected = s.n ? this.size : 0;
    requireState(s.particles.length === expected && s.log_weights.length === expected, `${expected} particles and weights`);
    requireState(s.n === 0 || Math.abs(logSumExp(s.log_weights)) < 1e-9, "weights must be normalized");
    return s;
  }
}

register("particle_filter", "1", (c) => ParticleFilter.fromConfig(c));
