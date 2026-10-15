// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Bayesian online change-point detection (Adams and MacKay 2007); mirrors procedures/bocpd.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { logSumExp, neumaierAdd, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { lgamma } from "./distributions.js";
import { BaseProcedure, type Config, type Decision, int, invalid, num, obj, optNum, REJECT_H0, str } from "./util.js";

const positive = (...values: number[]): boolean => values.every((v) => Number.isFinite(v) && v > 0);

export interface ConjugateModel {
  readonly empty: number[];
  spec(): Config;
  contract(name: string): InputContract;
  update(stats: number[], y: number): number[];
  logPredictive(stats: number[], y: number): number;
  mean(stats: number[]): number;
}

export class NormalGammaModel implements ConjugateModel {
  readonly empty = [0.0, 0.0, 0.0];

  constructor(readonly mu0 = 0.0, readonly kappa0 = 1.0, readonly alpha0 = 1.0, readonly beta0 = 1.0) {
    if (!(Number.isFinite(mu0) && positive(kappa0, alpha0, beta0))) invalid("need a finite mu0 and positive kappa0, alpha0, beta0");
  }

  spec(): Config {
    return { model: "normal_gamma", mu0: this.mu0, kappa0: this.kappa0, alpha0: this.alpha0, beta0: this.beta0 };
  }

  contract(name: string): InputContract {
    return new InputContract(name);
  }

  update(stats: number[], y: number): number[] {
    let [n, mean] = stats;
    const ss = stats[2];
    n += 1.0;
    const delta = y - mean;
    mean += delta / n;
    return [n, mean, ss + delta * (y - mean)];
  }

  posterior(stats: number[]): [number, number, number, number] {
    const [n, mean, ss] = stats;
    const kappa = this.kappa0 + n;
    const mu = (this.kappa0 * this.mu0 + n * mean) / kappa;
    const alpha = this.alpha0 + 0.5 * n;
    const d = mean - this.mu0;
    const beta = this.beta0 + 0.5 * ss + (this.kappa0 * n * (d * d)) / (2.0 * kappa);
    return [mu, kappa, alpha, beta];
  }

  logPredictive(stats: number[], y: number): number {
    const [mu, kappa, alpha, beta] = this.posterior(stats);
    const nu = 2.0 * alpha, s2 = (beta * (kappa + 1.0)) / (alpha * kappa);
    const d = y - mu;
    const z2 = (d * d) / s2;
    return lgamma(0.5 * (nu + 1.0)) - lgamma(0.5 * nu) - 0.5 * Math.log(nu * Math.PI * s2) - 0.5 * (nu + 1.0) * Math.log1p(z2 / nu);
  }

  mean(stats: number[]): number {
    return this.posterior(stats)[0];
  }
}

export class PoissonGammaModel implements ConjugateModel {
  readonly empty = [0.0, 0.0];

  constructor(readonly shape = 1.0, readonly rate = 1.0) {
    if (!positive(shape, rate)) invalid("shape and rate must be positive");
  }

  spec(): Config {
    return { model: "poisson_gamma", shape: this.shape, rate: this.rate };
  }

  contract(name: string): InputContract {
    return new InputContract(name, "integer", [], 0);
  }

  update(stats: number[], y: number): number[] {
    return [stats[0] + 1.0, stats[1] + y];
  }

  logPredictive(stats: number[], y: number): number {
    const r = this.shape + stats[1], b = this.rate + stats[0];
    return lgamma(r + y) - lgamma(r) - lgamma(y + 1.0) + r * Math.log(b / (b + 1.0)) - y * Math.log1p(b);
  }

  mean(stats: number[]): number {
    return (this.shape + stats[1]) / (this.rate + stats[0]);
  }
}

export class BetaBernoulliModel implements ConjugateModel {
  readonly empty = [0.0, 0.0];

  constructor(readonly a = 1.0, readonly b = 1.0) {
    if (!positive(a, b)) invalid("a and b must be positive");
  }

  spec(): Config {
    return { model: "beta_bernoulli", a: this.a, b: this.b };
  }

  contract(name: string): InputContract {
    return new InputContract(name, "binary");
  }

  update(stats: number[], y: number): number[] {
    return [stats[0] + 1.0, stats[1] + y];
  }

  logPredictive(stats: number[], y: number): number {
    const p = (this.a + stats[1]) / (this.a + this.b + stats[0]);
    return y ? Math.log(p) : Math.log1p(-p);
  }

  mean(stats: number[]): number {
    return (this.a + stats[1]) / (this.a + this.b + stats[0]);
  }
}

export function conjugateModelFromSpec(spec: Config): ConjugateModel {
  switch (str(spec, "model")) {
    case "normal_gamma":
      return new NormalGammaModel(num(spec, "mu0"), num(spec, "kappa0"), num(spec, "alpha0"), num(spec, "beta0"));
    case "poisson_gamma":
      return new PoissonGammaModel(num(spec, "shape"), num(spec, "rate"));
    case "beta_bernoulli":
      return new BetaBernoulliModel(num(spec, "a"), num(spec, "b"));
    default:
      return invalid(`unknown conjugate model ${JSON.stringify(spec.model)}`);
  }
}

type BOCPDState = {
  n: number; lengths: number[]; log_probs: number[]; stats: number[][]; log_evidence: number; comp: number; decision: Decision;
};

export class BayesianChangePoint extends BaseProcedure {
  readonly name = "bayesian_change_point";
  readonly version = "1";
  private readonly logH: number;
  private readonly log1mH: number;

  constructor(
    readonly model: ConjugateModel,
    readonly hazard = 0.01,
    readonly alarmProbability: number | null = null,
    readonly alarmWindow = 5,
    readonly prune = 1e-12,
    readonly maxHypotheses: number | null = 2000,
    readonly input = "x",
  ) {
    super();
    if (!(0 < hazard && hazard < 1)) invalid("hazard must lie in (0, 1): the probability that a segment ends at each step");
    if (alarmProbability !== null && !(0 < alarmProbability && alarmProbability < 1)) invalid("alarm_probability must lie in (0, 1), or be None");
    if (!Number.isInteger(alarmWindow) || alarmWindow < 1) invalid("alarm_window must be a positive int");
    if (!(0 <= prune && prune < 1)) invalid("prune must lie in [0, 1)");
    if (maxHypotheses !== null && (!Number.isInteger(maxHypotheses) || maxHypotheses < 2))
      invalid("max_hypotheses must be an int of at least 2, or None");
    this.logH = Math.log(hazard);
    this.log1mH = Math.log1p(-hazard);
  }

  static fromConfig(c: Config): BayesianChangePoint {
    const maxHypotheses = optNum(c, "max_hypotheses");
    return new BayesianChangePoint(conjugateModelFromSpec(obj(c, "model")), num(c, "hazard"), optNum(c, "alarm_probability"),
      int(c, "alarm_window"), num(c, "prune"), maxHypotheses === null ? null : int(c, "max_hypotheses"), str(c, "input"));
  }

  config(): Config {
    return {
      model: this.model.spec(), hazard: this.hazard, alarm_probability: this.alarmProbability, alarm_window: this.alarmWindow,
      prune: this.prune, max_hypotheses: this.maxHypotheses, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    return { [this.input]: this.model.contract(this.input) };
  }

  initialState(): BOCPDState {
    return { n: 0, lengths: [], log_probs: [], stats: [], log_evidence: 0.0, comp: 0.0, decision: null };
  }

  step(state: State, x: StatInput): [BOCPDState, State] {
    const s = state as BOCPDState;
    const y = x[this.input] as number;
    const model = this.model;
    let lengths: number[], logs: number[], stats: number[][];
    if (s.n === 0) {
      lengths = [1];
      logs = [model.logPredictive(model.empty, y)];
      stats = [model.update(model.empty, y)];
    } else {
      lengths = [1];
      logs = [this.logH + model.logPredictive(model.empty, y)];
      stats = [model.update(model.empty, y)];
      if (s.lengths.length !== s.log_probs.length || s.lengths.length !== s.stats.length)
        throw new Error("zip_strict: the iterables have different lengths");
      for (let i = 0; i < s.lengths.length; i++) {
        lengths.push(s.lengths[i] + 1);
        logs.push(s.log_probs[i] + this.log1mH + model.logPredictive(s.stats[i], y));
        stats.push(model.update(s.stats[i], y));
      }
    }
    const increment = requireFinite(logSumExp(logs), "evidence");
    logs = logs.map((v) => v - increment);
    const [logEvidence, comp] = neumaierAdd(s.log_evidence, s.comp, increment);
    [lengths, logs, stats] = this.pruned(lengths, logs, stats);
    const probs = logs.map((v) => Math.exp(v));
    const n = s.n + 1;
    const change = n > 1 && lengths[0] === 1 ? probs[0] : 0.0;
    let recent = 0.0;
    if (n > this.alarmWindow) {
      for (let i = 0; i < lengths.length; i++) if (lengths[i] <= this.alarmWindow) recent += probs[i];
    }
    let best = 0;
    for (let i = 1; i < lengths.length; i++)
      if (probs[i] > probs[best] || (probs[i] === probs[best] && lengths[i] < lengths[best])) best = i;
    let expected = 0.0, segMean = 0.0;
    for (let i = 0; i < lengths.length; i++) {
      expected += probs[i] * lengths[i];
      segMean += probs[i] * model.mean(stats[i]);
    }
    let decision = s.decision;
    if (decision === null && this.alarmProbability !== null && recent >= this.alarmProbability) decision = REJECT_H0;
    return [
      { n, lengths, log_probs: logs, stats, log_evidence: logEvidence, comp, decision },
      {
        n, change_probability: change, recent_change_probability: recent, most_likely_run_length: lengths[best],
        expected_run_length: expected, segment_mean: segMean, log_evidence: requireFinite(logEvidence + comp, "evidence"), decision,
      },
    ];
  }

  private pruned(lengths: number[], logs: number[], stats: number[][]): [number[], number[], number[][]] {
    let keep = lengths.map((_, i) => i);
    if (this.prune > 0) {
      const cut = Math.log(this.prune);
      const above = keep.filter((i) => logs[i] >= cut);
      if (above.length) keep = above;
      else keep = [keep.reduce((b, i) => (logs[i] > logs[b] ? i : b))];
    }
    if (this.maxHypotheses !== null && keep.length > this.maxHypotheses) {
      const ranked = [...keep].sort((i, j) => (logs[i] > logs[j] ? -1 : logs[i] < logs[j] ? 1 : i - j));
      keep = ranked.slice(0, this.maxHypotheses).sort((i, j) => i - j);
    }
    if (keep.length === lengths.length) return [lengths, logs, stats];
    const norm = logSumExp(keep.map((i) => logs[i]));
    return [keep.map((i) => lengths[i]), keep.map((i) => logs[i] - norm), keep.map((i) => stats[i])];
  }

  override isTerminal(state: State): boolean {
    return state.decision !== null;
  }

  decodeState(data: Json): BOCPDState {
    const s = decodeFields({
      n: "count", lengths: "counts", log_probs: "numbers", stats: "json", log_evidence: "number", comp: "number", decision: "decision",
    }, data) as BOCPDState;
    const k = s.lengths.length;
    const stats = s.stats as unknown;
    requireState(Array.isArray(stats) && stats.every((r) => Array.isArray(r) && r.every((v) => typeof v === "number" && Number.isFinite(v))),
      "sufficient statistics must be lists of numbers");
    requireState(s.log_probs.length === k && s.stats.length === k, "one probability and statistic per run length");
    requireState((k === 0) === (s.n === 0), "hypotheses without observations");
    requireState(s.lengths.every((length) => 1 <= length && length <= s.n), "run length out of range");
    let total = 0.0;
    for (const v of s.log_probs) total += Math.exp(v);
    requireState(k === 0 || Math.abs(total - 1.0) < 1e-9, "probabilities must sum to 1");
    requireState(s.stats.every((r) => r.length === this.model.empty.length), "sufficient statistics of the wrong size");
    return s;
  }
}

register("bayesian_change_point", "1", (c) => BayesianChangePoint.fromConfig(c));
