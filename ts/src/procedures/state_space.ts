// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Linear state-space filters (Kalman, and Student-t robust by variational Bayes); mirrors procedures/state_space.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { NumericalError } from "../errors.js";
import * as la from "../linalg.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { BaseProcedure, type Config, invalid, num, str } from "./util.js";

const LOG_2PI = Math.log(2.0 * Math.PI);

function checkCovariance(a: la.Matrix, name: string, definite: boolean): void {
  if (!la.isSymmetric(a)) invalid(`${name} must be symmetric`);
  if (definite) {
    try {
      la.cholesky(a);
    } catch {
      invalid(`${name} must be positive definite`);
    }
  } else if (a.some((row, i) => row[i] < 0)) invalid(`${name} must have a non-negative diagonal`);
}

type KalmanFilterState = { n: number; mean: number[]; covariance: number[][]; log_likelihood: number; comp: number };

/** The matrices of KalmanFilter, in the order of its Python constructor. */
export interface StateSpaceMatrices {
  transition: unknown;
  observation: unknown;
  processNoise: unknown;
  observationNoise: unknown;
  initialMean: unknown;
  initialCovariance: unknown;
  control?: unknown;
}

export class KalmanFilter extends BaseProcedure {
  readonly name: string = "kalman_filter";
  readonly version: string = "1";
  readonly F: la.Matrix;
  readonly H: la.Matrix;
  readonly Q: la.Matrix;
  readonly R: la.Matrix;
  readonly m0: la.Vector;
  readonly P0: la.Matrix;
  readonly B: la.Matrix | null;
  readonly nState: number;
  readonly nObs: number;
  protected readonly rInverse: la.Matrix;

  constructor(m: StateSpaceMatrices, readonly input = "y", readonly controlInput = "u") {
    super();
    const wrap = <T>(f: () => T): T => {
      try {
        return f();
      } catch (e) {
        return invalid((e as Error).message);
      }
    };
    this.F = wrap(() => la.matrix(m.transition, null, null, "transition"));
    const n = this.F.length;
    if (this.F[0].length !== n) invalid("transition must be square");
    this.H = wrap(() => la.matrix(m.observation, null, n, "observation"));
    const k = this.H.length;
    this.Q = wrap(() => la.matrix(m.processNoise, n, n, "process_noise"));
    this.R = wrap(() => la.matrix(m.observationNoise, k, k, "observation_noise"));
    this.m0 = wrap(() => la.vector(m.initialMean, n, "initial_mean"));
    this.P0 = wrap(() => la.matrix(m.initialCovariance, n, n, "initial_covariance"));
    this.B = m.control === undefined || m.control === null ? null : wrap(() => la.matrix(m.control, n, null, "control"));
    checkCovariance(this.Q, "process_noise", false);
    checkCovariance(this.P0, "initial_covariance", false);
    checkCovariance(this.R, "observation_noise", true);
    if (input === controlInput) invalid("the observation and the control need different names");
    this.nState = n;
    this.nObs = k;
    this.rInverse = la.inverseSpd(this.R);
  }

  static matricesFromConfig(c: Config): StateSpaceMatrices {
    for (const key of ["transition", "observation", "process_noise", "observation_noise", "initial_mean", "initial_covariance", "control"])
      if (!(key in c)) invalid(`config is missing '${key}'`);
    return {
      transition: c.transition, observation: c.observation, processNoise: c.process_noise, observationNoise: c.observation_noise,
      initialMean: c.initial_mean, initialCovariance: c.initial_covariance, control: c.control,
    };
  }

  static fromConfig(c: Config): KalmanFilter {
    return new KalmanFilter(KalmanFilter.matricesFromConfig(c), str(c, "input"), str(c, "control_input"));
  }

  config(): Config {
    const rows = (a: la.Matrix) => a.map((r) => [...r]);
    return {
      transition: rows(this.F), observation: rows(this.H), process_noise: rows(this.Q), observation_noise: rows(this.R),
      initial_mean: [...this.m0], initial_covariance: rows(this.P0), control: this.B === null ? null : rows(this.B),
      input: this.input, control_input: this.controlInput,
    };
  }

  inputs(): Record<string, InputContract> {
    const contract = (name: string, size: number) => (size === 1 ? new InputContract(name) : new InputContract(name, "real", [size]));
    const contracts: Record<string, InputContract> = { [this.input]: contract(this.input, this.nObs) };
    if (this.B !== null) contracts[this.controlInput] = contract(this.controlInput, this.B[0].length);
    return contracts;
  }

  initialState(): KalmanFilterState {
    return { n: 0, mean: [...this.m0], covariance: this.P0.map((r) => [...r]), log_likelihood: 0.0, comp: 0.0 };
  }

  protected predict(s: KalmanFilterState, x: StatInput): [la.Vector, la.Matrix] {
    let mean = la.matvec(this.F, s.mean);
    if (this.B !== null) mean = la.vadd(mean, la.matvec(this.B, la.vector(x[this.controlInput], this.B[0].length, "control")));
    return [mean, la.add(la.quadratic(this.F, s.covariance), this.Q)];
  }

  /** Kalman update with observation noise covariance `noise`: (mean, cov, innovation, S, S^-1). */
  protected update(mean: la.Vector, cov: la.Matrix, y: la.Vector, noise: la.Matrix): [la.Vector, la.Matrix, la.Vector, la.Matrix, la.Matrix] {
    const innovation = la.vsub(y, la.matvec(this.H, mean));
    const s = la.add(la.quadratic(this.H, cov), noise);
    let sInverse: la.Matrix;
    try {
      sInverse = la.inverseSpd(s);
    } catch {
      throw new NumericalError(`${this.name}: the innovation covariance is not positive definite`);
    }
    const gain = la.matmul(la.matmul(cov, la.transpose(this.H)), sInverse);
    const newMean = la.vadd(mean, la.matvec(gain, innovation));
    const reduction = la.sub(la.identity(this.nState), la.matmul(gain, this.H));
    const newCov = la.add(la.quadratic(reduction, cov), la.quadratic(gain, noise));
    return [newMean, newCov, innovation, s, sInverse];
  }

  step(state: State, x: StatInput): [KalmanFilterState, State] {
    const st = state as KalmanFilterState;
    const y = la.vector(x[this.input], this.nObs, "observation");
    const [mean, cov] = this.predict(st, x);
    const [newMean, newCov, innovation, s, sInverse] = this.update(mean, cov, y, this.R);
    const term = -0.5 * (this.nObs * LOG_2PI + la.logDetSpd(s) + la.dot(innovation, la.matvec(sInverse, innovation)));
    const [total, comp] = neumaierAdd(st.log_likelihood, st.comp, term);
    const n = st.n + 1;
    const logLik = requireFinite(total + comp, "log likelihood");
    return [
      { n, mean: newMean, covariance: newCov, log_likelihood: total, comp },
      {
        n, mean: [...newMean], covariance: newCov.map((r) => [...r]), innovation, innovation_covariance: s,
        log_likelihood: logLik, weight: 1.0,
      },
    ];
  }

  decodeState(data: Json): KalmanFilterState {
    const s = decodeFields({ n: "count", mean: "numbers", covariance: "json", log_likelihood: "number", comp: "number" },
      data) as KalmanFilterState;
    requireState(s.mean.length === this.nState, "state dimension");
    const cov = s.covariance as unknown;
    requireState(Array.isArray(cov) && cov.length === this.nState
      && cov.every((r) => Array.isArray(r) && r.length === this.nState && r.every((v) => typeof v === "number" && Number.isFinite(v))),
      "covariance dimension");
    requireState(la.isSymmetric(s.covariance), "covariance not symmetric");
    requireState(s.covariance.every((r, i) => r[i] >= 0), "negative variance");
    return s;
  }
}

export class RobustKalmanFilter extends KalmanFilter {
  override readonly name: string = "robust_kalman_filter";
  override readonly version: string = "1";

  constructor(m: StateSpaceMatrices, readonly df = 4.0, input = "y", controlInput = "u") {
    super(m, input, controlInput);
    if (!(Number.isFinite(df) && df > 0)) invalid("df must be positive and finite");
  }

  static override fromConfig(c: Config): RobustKalmanFilter {
    return new RobustKalmanFilter(KalmanFilter.matricesFromConfig(c), num(c, "df"), str(c, "input"), str(c, "control_input"));
  }

  override config(): Config {
    return { ...super.config(), df: this.df };
  }

  override step(state: State, x: StatInput): [KalmanFilterState, State] {
    const st = state as KalmanFilterState;
    const y = la.vector(x[this.input], this.nObs, "observation");
    const [mean, cov] = this.predict(st, x);
    let weight = 1.0;
    for (let it = 0; it < 100; it++) {
      const [newMean, newCov] = this.update(mean, cov, y, la.scale(this.R, 1.0 / weight));
      const residual = la.vsub(y, la.matvec(this.H, newMean));
      const expected = la.dot(residual, la.matvec(this.rInverse, residual))
        + la.trace(la.matmul(la.quadratic(this.H, newCov), this.rInverse));
      const updated = (this.df + this.nObs) / (this.df + expected);
      if (Math.abs(updated - weight) <= 1e-12 * weight) {
        weight = updated;
        break;
      }
      weight = updated;
    }
    const [newMean, newCov, innovation, s] = this.update(mean, cov, y, la.scale(this.R, 1.0 / weight));
    const n = st.n + 1;
    return [
      { n, mean: newMean, covariance: newCov, log_likelihood: 0.0, comp: 0.0 },
      {
        n, mean: [...newMean], covariance: newCov.map((r) => [...r]), innovation, innovation_covariance: s,
        log_likelihood: null, weight,
      },
    ];
  }
}

register("kalman_filter", "1", (c) => KalmanFilter.fromConfig(c));
register("robust_kalman_filter", "1", (c) => RobustKalmanFilter.fromConfig(c));
