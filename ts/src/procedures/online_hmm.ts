// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Online EM for a hidden Markov model (Cappé 2011); mirrors procedures/online_hmm.py operation for operation.
import type { Json } from "../canonical.js";
import type { InputContract, StatInput } from "../contracts.js";
import { NumericalError } from "../errors.js";
import { neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { fsum } from "./ab.js";
import { type Emission, type EmissionInput, HiddenMarkovFilter } from "./hmm.js";
import { BaseProcedure, bool, type Config, int, invalid, num, obj } from "./util.js";

type OnlineState = {
  n: number; probabilities: number[]; transitions: number[]; emissions: number[]; log_likelihood: number; comp: number;
  averaged: number[];
};

function probe(contract: InputContract): EmissionInput {
  if (contract.shape.length) return Array.from({ length: contract.shape[0] }, () => 0.0);
  return 0;
}

export class OnlineHMM extends BaseProcedure {
  readonly name = "online_hmm";
  readonly version = "1";
  readonly states: number;
  readonly input: string;
  private readonly m: number;

  constructor(
    readonly start: HiddenMarkovFilter, readonly kappa: number, readonly burnIn: number, readonly minScale: number,
    readonly average: boolean,
  ) {
    super();
    if (!(kappa > 0.5 && kappa <= 1.0)) invalid("kappa must lie in (0.5, 1]");
    if (!(Number.isInteger(burnIn) && burnIn >= 1)) invalid("burn_in must be a positive int");
    if (!(Number.isFinite(minScale) && minScale >= 0)) invalid("min_scale must be a non-negative number");
    this.states = start.states;
    this.input = start.input;
    this.m = start.emission.statistics(0, probe(start.inputs()[this.input])).length;
  }

  static fromConfig(c: Config): OnlineHMM {
    return new OnlineHMM(HiddenMarkovFilter.fromConfig(obj(c, "start")), num(c, "kappa"), int(c, "burn_in"), num(c, "min_scale"),
      bool(c, "average"));
  }

  config(): Config {
    return { start: this.start.config(), kappa: this.kappa, burn_in: this.burnIn, min_scale: this.minScale, average: this.average };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.start.inputs()[this.input] };
  }

  initialState(): OnlineState {
    return { n: 0, probabilities: [], transitions: [], emissions: [], log_likelihood: 0.0, comp: 0.0, averaged: [] };
  }

  /** S_n = sum_k rho_n(k) phi_n(k): transition counts (K x K) and emission totals (K x (m + 1)). */
  statistics(s: OnlineState): [number[][], number[][]] {
    const K = this.states, m1 = this.m + 1;
    const trans = Array.from({ length: K }, () => Array.from({ length: K }, () => 0.0));
    const emis = Array.from({ length: K }, () => Array.from({ length: m1 }, () => 0.0));
    s.probabilities.forEach((pk, k) => {
      if (pk === 0.0) return;
      let base = k * K * K;
      for (let j = 0; j < K; j++) for (let l = 0; l < K; l++) trans[j][l] += pk * s.transitions[base + j * K + l];
      base = k * K * m1;
      for (let j = 0; j < K; j++) for (let i = 0; i < m1; i++) emis[j][i] += pk * s.emissions[base + j * m1 + i];
    });
    return [trans, emis];
  }

  private unflatten(flat: number[]): [number[][], number[][]] {
    const K = this.states, m1 = this.m + 1;
    const trans = Array.from({ length: K }, (_, j) => flat.slice(j * K, (j + 1) * K));
    const emis = Array.from({ length: K }, (_, j) => flat.slice(K * K + j * m1, K * K + (j + 1) * m1));
    return [trans, emis];
  }

  /** The transition matrix and emissions used for the next observation. */
  model(s: OnlineState): [number[][], Emission] {
    if (s.n < this.burnIn) return [this.start.transition, this.start.emission];
    const [trans, emis] = this.average ? this.unflatten(s.averaged) : this.statistics(s);
    const transition = trans.map((row, j) => {
      let total = 0.0;
      for (const v of row) total += v;
      return total > 0 ? row.map((v) => v / total) : [...this.start.transition[j]];
    });
    return [transition, this.start.emission.fromStatistics(emis, this.minScale)];
  }

  step(state: State, x: StatInput): [OnlineState, State] {
    const s = state as OnlineState;
    const y = x[this.input] as EmissionInput;
    const K = this.states, m1 = this.m + 1;
    const [a, emission] = this.model(s);
    const n = s.n + 1;
    let prior: number[];
    if (s.n === 0) prior = [...this.start.initial];
    else {
      prior = [];
      for (let k = 0; k < K; k++) {
        let total = 0.0;
        for (let j = 0; j < K; j++) total += s.probabilities[j] * a[j][k];
        prior.push(total);
      }
    }
    const logs = prior.map((p, k) => (p > 0 ? emission.logDensity(k, y) : -Infinity));
    const top = Math.max(...logs);
    if (top === -Infinity) throw new NumericalError(`${this.name}: observation ${JSON.stringify(y)} is impossible in every reachable state`);
    const joint = prior.map((p, k) => p * Math.exp(logs[k] - top));
    let c = 0.0;
    for (const v of joint) c += v;
    const phi = joint.map((v) => v / c);
    const [logLik, comp] = neumaierAdd(s.log_likelihood, s.comp, top + Math.log(c));
    const stats = Array.from({ length: K }, (_, k) => emission.statistics(k, y));
    const transitions = Array.from({ length: K * K * K }, () => 0.0);
    const emissions = Array.from({ length: K * K * m1 }, () => 0.0);
    if (s.n === 0) {
      for (let k = 0; k < K; k++) {
        const base = (k * K + k) * m1;
        emissions[base] = 1.0;
        stats[k].forEach((v, i) => (emissions[base + 1 + i] = v));
      }
    } else {
      const gamma = n ** -this.kappa;
      const keep = 1.0 - gamma;
      for (let k = 0; k < K; k++) {
        if (prior[k] === 0.0) continue;
        const r = Array.from({ length: K }, (_, j) => (s.probabilities[j] * a[j][k]) / prior[k]);
        const tk = k * K * K;
        for (let j = 0; j < K; j++) {
          const rj = r[j];
          if (rj === 0.0) continue;
          const old = j * K * K;
          for (let p = 0; p < K * K; p++) transitions[tk + p] += keep * rj * s.transitions[old + p];
          transitions[tk + j * K + k] += gamma * rj;
        }
        const ek = k * K * m1;
        for (let j = 0; j < K; j++) {
          const rj = r[j];
          if (rj === 0.0) continue;
          const old = j * K * m1;
          for (let p = 0; p < K * m1; p++) emissions[ek + p] += keep * rj * s.emissions[old + p];
        }
        const base = ek + k * m1;
        emissions[base] += gamma;
        stats[k].forEach((v, i) => (emissions[base + 1 + i] += gamma * v));
      }
    }
    let next: OnlineState = { n, probabilities: phi, transitions, emissions, log_likelihood: logLik, comp, averaged: [] };
    if (this.average && n >= this.burnIn) {
      const [tNow, eNow] = this.statistics(next);
      const flat = [...tNow.flat(), ...eNow.flat()];
      let averaged: number[];
      if (n === this.burnIn) averaged = flat;
      else {
        const weight = 1.0 / (n - this.burnIn + 1);
        averaged = s.averaged.map((v, i) => v + weight * (flat[i] - v));
      }
      next = { ...next, averaged };
    }
    const [transition, current] = this.model(next);
    let best = 0;
    for (let k = 1; k < K; k++) if (phi[k] > phi[best]) best = k;
    return [next, {
      n, probabilities: phi, most_likely: best, transition: transition.map((row) => [...row]), emission: current.spec(),
      log_likelihood: requireFinite(logLik + comp, "log likelihood"),
    }];
  }

  decodeState(data: Json): OnlineState {
    const s = decodeFields({
      n: "count", probabilities: "numbers", transitions: "numbers", emissions: "numbers", log_likelihood: "number", comp: "number",
      averaged: "numbers",
    }, data) as OnlineState;
    const K = this.states, m1 = this.m + 1;
    if (s.n === 0) {
      requireState(!(s.probabilities.length || s.transitions.length || s.emissions.length || s.averaged.length), "statistics before data");
      return s;
    }
    requireState(s.probabilities.length === K, "one probability per state");
    requireState(s.transitions.length === K ** 3 && s.emissions.length === K * K * m1, "statistics' sizes");
    requireState(s.probabilities.every((p) => p >= 0), "negative probability");
    requireState(Math.abs(fsum(s.probabilities) - 1.0) < 1e-9, "probabilities must sum to 1");
    requireState(s.transitions.every((v) => v >= 0), "negative transition statistics");
    const averaging = this.average && s.n >= this.burnIn;
    requireState(s.averaged.length === (averaging ? K * K + K * m1 : 0), "averaged statistics' size");
    return s;
  }
}

register("online_hmm", "1", (c) => OnlineHMM.fromConfig(c));
