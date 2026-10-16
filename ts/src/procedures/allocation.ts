// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Redirecting traffic during an experiment; mirrors procedures/allocation.py (ThompsonAllocation,
// best_arm_probabilities): P(arm k is the best) and its expected loss by piecewise Gauss-Legendre quadrature.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import { lgamma, regularizedBeta, tCdf } from "./distributions.js";
import { BaseProcedure, type Config, int, invalid, num, nums, pyMax, pyMin, str } from "./util.js";

/** An optional integer of a configuration: null or an int. */
function optInt(c: Config, key: string): number | null {
  return c[key] === null ? null : int(c, key);
}

const METRICS = ["binary", "normal"];

// 16-point Gauss-Legendre rule on [-1, 1]
const NODES = [
  -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755404408355003, -0.6178762444026438,
  -0.45801677765722737, -0.2816035507792589, -0.09501250983763744, 0.09501250983763744, 0.2816035507792589,
  0.45801677765722737, 0.6178762444026438, 0.755404408355003, 0.8656312023878318, 0.9445750230732326,
  0.9894009349916499,
];
const WEIGHTS = [
  0.027152459411754176, 0.062253523938647456, 0.0951585116824926, 0.12462897125553407,
  0.1495959888165767, 0.16915651939500265, 0.18260341504492364, 0.18945061045506864, 0.18945061045506864,
  0.18260341504492364, 0.16915651939500265, 0.1495959888165767, 0.12462897125553407, 0.0951585116824926,
  0.062253523938647456, 0.027152459411754176,
];
// pieces of the integration range: each posterior's location plus these multiples of its scale
const CUTS = [-40.0, -21.0, -13.0, -8.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 40.0];

/** The posterior of one arm's mean: density, distribution function, location and scale (Python's _Posterior). */
export class Posterior {
  readonly location: number;
  readonly scale: number;
  readonly a: number = 0.0;
  readonly b: number = 0.0;
  private readonly logNorm: number = 0.0;
  private readonly df: number = 0.0;

  constructor(readonly metric: string, count: number, mean: number, m2: number, prior: [number, number]) {
    if (metric === "binary") {
      // the running mean of 0/1 outcomes times their number: within rounding of an integer, so no .5 ties
      const successes = Math.round(mean * count);
      this.a = prior[0] + successes;
      this.b = prior[1] + count - successes;
      const total = this.a + this.b;
      this.location = this.a / total;
      this.scale = Math.sqrt((this.a * this.b) / (total * total * (total + 1.0)));
      this.logNorm = lgamma(total) - lgamma(this.a) - lgamma(this.b);
    } else {
      this.df = count - 1;
      this.location = mean;
      this.scale = Math.sqrt(m2 / (count - 1) / count);
    }
  }

  density(x: number): number {
    if (this.metric === "binary") {
      if (!(0.0 < x && x < 1.0)) return 0.0;
      return Math.exp(this.logNorm + (this.a - 1.0) * Math.log(x) + (this.b - 1.0) * Math.log1p(-x));
    }
    const z = (x - this.location) / this.scale;
    const df = this.df;
    return Math.exp(lgamma(0.5 * (df + 1.0)) - lgamma(0.5 * df) - 0.5 * Math.log(df * Math.PI)
      - 0.5 * (df + 1.0) * Math.log1p((z * z) / df)) / this.scale;
  }

  cdf(x: number): number {
    if (this.metric === "binary") {
      if (x <= 0.0) return 0.0;
      if (x >= 1.0) return 1.0;
      return regularizedBeta(x, this.a, this.b);
    }
    return tCdf((x - this.location) / this.scale, this.df);
  }
}

/** [P(k best), expected loss of k] for each arm, by piecewise Gauss-Legendre quadrature (Python's best_arm_probabilities). */
export function bestArmProbabilities(posteriors: Posterior[], metric: string): [number[], number[]] {
  const k = posteriors.length;
  let cuts: number[] = [];
  for (const p of posteriors) for (const c of CUTS) cuts.push(p.location + c * p.scale);
  if (metric === "binary") cuts = [...cuts.map((c) => pyMin(1.0, pyMax(0.0, c))), 0.0, 1.0];
  // sorted(set(cuts)): numeric order, exact duplicates removed (0.0 and -0.0 are one value)
  cuts.sort((u, v) => u - v);
  const unique: number[] = [];
  for (const c of cuts) if (unique.length === 0 || c !== unique[unique.length - 1]) unique.push(c);
  const best: number[] = new Array<number>(k).fill(0.0);
  let top = 0.0; // E[max_j mu_j]
  for (let piece = 0; piece + 1 < unique.length; piece++) {
    const lo = unique[piece], hi = unique[piece + 1];
    const half = 0.5 * (hi - lo), mid = 0.5 * (hi + lo);
    for (let j = 0; j < NODES.length; j++) {
      const x = mid + half * NODES[j];
      const dens = posteriors.map((p) => p.density(x));
      const dist = posteriors.map((p) => p.cdf(x));
      // products of the other arms' distribution functions, without dividing by one that may be 0
      const before = new Array<number>(k).fill(1.0), after = new Array<number>(k).fill(1.0);
      for (let i = 1; i < k; i++) before[i] = before[i - 1] * dist[i - 1];
      for (let i = k - 2; i >= 0; i--) after[i] = after[i + 1] * dist[i + 1];
      const w = WEIGHTS[j] * half;
      for (let i = 0; i < k; i++) {
        const term = w * dens[i] * before[i] * after[i];
        best[i] += term;
        top += term * x;
      }
    }
  }
  let total = 0.0;
  for (const v of best) total += v;
  const probabilities = best.map((v) => v / total);
  top /= total;
  return [probabilities, posteriors.map((p) => pyMax(0.0, top - p.location))];
}

type ThompsonAllocationState = {
  n: number;
  counts: number[];
  means: number[]; // running means of the outcomes per arm (Welford)
  m2: number[]; // running sums of squared deviations per arm
  shares: number[]; // the split in force for the next units
  since: number; // units since the split was last computed
  best: number[]; // P(k best) when the split was last computed; [] before
  loss: number[]; // expected loss of each arm, idem
  deploy: number | null; // the arm the deployment rule chose, once it has
};

/** Traffic split by Thompson sampling with a floor, and the expected-loss deployment rule (Python's ThompsonAllocation). */
export class ThompsonAllocation extends BaseProcedure {
  readonly name = "thompson_allocation";
  readonly version = "1";
  readonly prior: [number, number];

  constructor(
    readonly arms = 2,
    readonly metric = "binary",
    readonly floor = 0.05,
    readonly threshold = 0.0,
    readonly minUnits = 100,
    readonly updateEvery = 100,
    prior: number[] = [1.0, 1.0],
    readonly control: number | null = null,
    readonly reserve = 0.0,
    readonly armInput = "arm",
    readonly input = "y",
  ) {
    super();
    if (!Number.isInteger(arms) || arms < 2) invalid("arms must be an int >= 2");
    if (!METRICS.includes(metric)) invalid(`metric must be one of ${METRICS}`);
    if (!(0.0 <= floor && floor <= 1.0 / arms)) invalid("floor must lie in [0, 1/arms]");
    if (!(0.0 <= threshold && threshold < Infinity)) invalid("threshold must be a non-negative number");
    if (!Number.isInteger(minUnits) || minUnits < (metric === "normal" ? 2 : 0))
      invalid("min_units must be an int (at least 2 for a normal metric)");
    if (!Number.isInteger(updateEvery) || updateEvery < 1) invalid("update_every must be a positive int");
    if (prior.length !== 2) invalid("the Beta prior needs two parameters");
    const [a, b] = prior;
    if (!(0.0 < a && a < Infinity && 0.0 < b && b < Infinity)) invalid("the Beta prior needs positive parameters");
    if (armInput === input) invalid("the arm and the outcome need different names");
    if (control === null) {
      if (reserve !== 0.0) invalid("a reserve needs a control arm");
    } else if (!Number.isInteger(control) || !(0 <= control && control < arms)) {
      invalid("control must be one of the arms");
    }
    if (!(0.0 <= reserve && reserve <= 1.0 - arms * floor)) invalid("reserve must lie in [0, 1 - arms * floor]");
    this.prior = [a, b];
  }

  static fromConfig(c: Config): ThompsonAllocation {
    return new ThompsonAllocation(int(c, "arms"), str(c, "metric"), num(c, "floor"), num(c, "threshold"), int(c, "min_units"),
      int(c, "update_every"), nums(c, "prior"), optInt(c, "control"), num(c, "reserve"), str(c, "arm_input"), str(c, "input"));
  }

  config(): Config {
    return {
      arms: this.arms, metric: this.metric, floor: this.floor, threshold: this.threshold, min_units: this.minUnits,
      update_every: this.updateEvery, prior: [...this.prior], control: this.control,
      reserve: this.reserve, arm_input: this.armInput, input: this.input,
    };
  }

  inputs(): Record<string, InputContract> {
    const outcome = this.metric === "binary" ? new InputContract(this.input, "binary") : new InputContract(this.input);
    return {
      [this.armInput]: new InputContract(this.armInput, "integer", [], 0, this.arms - 1),
      [this.input]: outcome,
    };
  }

  initialState(): ThompsonAllocationState {
    const k = this.arms;
    return {
      n: 0, counts: new Array<number>(k).fill(0), means: new Array<number>(k).fill(0.0), m2: new Array<number>(k).fill(0.0),
      shares: new Array<number>(k).fill(1.0 / k), since: 0, best: [], loss: [], deploy: null,
    };
  }

  private posteriors(s: ThompsonAllocationState): Posterior[] | null {
    if (s.counts.some((c) => c < this.minUnits)) return null;
    if (this.metric === "normal" && !s.counts.every((c, i) => c >= 2 && s.m2[i] > 0.0)) return null;
    return s.counts.map((c, i) => new Posterior(this.metric, c, s.means[i], s.m2[i], this.prior));
  }

  step(state: State, x: StatInput): [ThompsonAllocationState, State] {
    const s = state as ThompsonAllocationState;
    const arm = x[this.armInput] as number, y = x[this.input] as number;
    const counts = [...s.counts], means = [...s.means], m2 = [...s.m2];
    counts[arm] += 1;
    const delta = y - means[arm];
    means[arm] = requireFinite(means[arm] + delta / counts[arm], "mean");
    m2[arm] = requireFinite(m2[arm] + delta * (y - means[arm]), "sum of squares");
    let next: ThompsonAllocationState = {
      n: s.n + 1, counts, means, m2, shares: s.shares, since: s.since + 1, best: s.best, loss: s.loss, deploy: s.deploy,
    };
    if (next.since >= this.updateEvery || (s.best.length === 0 && next.since >= 1)) {
      const posteriors = this.posteriors(next);
      if (posteriors !== null) {
        const [best, loss] = bestArmProbabilities(posteriors, this.metric);
        const spread = 1.0 - this.arms * this.floor - this.reserve;
        const shares = best.map((p, i) => this.floor + spread * p + (i === this.control ? this.reserve : 0.0));
        let deploy = s.deploy;
        if (deploy === null) {
          // min(range(k), key=lambda i: (loss[i], i)): the smallest loss, the first arm among ties
          let choice = 0;
          for (let i = 1; i < this.arms; i++) if (loss[i] < loss[choice]) choice = i;
          if (loss[choice] <= this.threshold) deploy = choice;
        }
        next = { ...next, shares, since: 0, best, loss, deploy };
      } else if (next.since >= this.updateEvery) {
        next = { ...next, since: 0 };
      }
    }
    let leader: number | null = null;
    if (next.best.length) {
      // max(range(k), key=lambda i: (best[i], -i)): the largest probability, the first arm among ties
      leader = 0;
      for (let i = 1; i < this.arms; i++) if (next.best[i] > next.best[leader]) leader = i;
    }
    return [next, {
      n: next.n, shares: next.shares, counts: next.counts, means: next.means.map((m, i) => (next.counts[i] ? m : null)),
      best: next.best, loss: next.loss, leader, deploy: next.deploy,
    }];
  }

  decodeState(data: Json): ThompsonAllocationState {
    const s = decodeFields({
      n: "count", counts: "counts", means: "numbers", m2: "numbers", shares: "numbers", since: "count", best: "numbers",
      loss: "numbers", deploy: "count?",
    }, data) as ThompsonAllocationState;
    const k = this.arms;
    requireState(s.counts.length === k && s.means.length === k && s.m2.length === k && s.shares.length === k, "one entry per arm");
    let units = 0;
    for (const c of s.counts) units += c;
    requireState(s.counts.every((c) => c >= 0) && units === s.n, "counts");
    requireState(s.m2.every((v) => v >= 0), "sums of squares");
    requireState(s.counts.every((c, i) => c > 1 || s.m2[i] === 0), "a spread from one unit");
    if (this.metric === "binary") requireState(s.means.every((m) => 0.0 <= m && m <= 1.0), "binary means must lie in [0, 1]");
    let total = 0.0;
    for (const v of s.shares) total += v;
    requireState(s.shares.every((v) => v >= 0) && Math.abs(total - 1.0) < 1e-9, "shares must sum to 1");
    requireState((s.best.length === 0 || s.best.length === k) && s.loss.length === s.best.length, "probabilities");
    requireState(0 <= s.since && s.since <= this.updateEvery, "units since the last split");
    requireState(s.deploy === null || (0 <= s.deploy && s.deploy < k && s.best.length > 0), "deployed arm");
    return s;
  }
}

register("thompson_allocation", "1", (c) => ThompsonAllocation.fromConfig(c));
