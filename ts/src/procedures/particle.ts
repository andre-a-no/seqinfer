// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Bootstrap particle filter for the local-level model (randomized); mirrors procedures/particle.py.
import type { Json } from "../canonical.js";
import { InputContract, type StatInput } from "../contracts.js";
import { logSumExp, requireFinite } from "../numerics.js";
import { decodeFields, register, requireState, type State } from "../procedure.js";
import type { SplitMix64 } from "../rng.js";
import { BaseProcedure, type Config, int, invalid, num } from "./util.js";

type ParticleState = { n: number; particles: number[] };

export class BootstrapParticleFilter extends BaseProcedure {
  readonly name = "bootstrap_particle_filter";
  readonly version = "1";
  override readonly randomized = true;

  constructor(readonly q: number, readonly r: number, readonly m0 = 0.0, readonly p0 = 1.0, readonly size = 256) {
    super();
    if (!(0 <= q && q < Infinity && 0 < r && r < Infinity && 0 <= p0 && p0 < Infinity && Number.isFinite(m0)) || size < 2)
      invalid("need q >= 0, r > 0, p0 >= 0 and at least two particles");
  }

  static fromConfig(c: Config): BootstrapParticleFilter {
    return new BootstrapParticleFilter(num(c, "q"), num(c, "r"), num(c, "m0"), num(c, "p0"), int(c, "particles"));
  }

  config(): Config {
    return { q: this.q, r: this.r, m0: this.m0, p0: this.p0, particles: this.size };
  }

  inputs(): Record<string, InputContract> {
    return { y: new InputContract("y") };
  }

  initialState(): ParticleState {
    return { n: 0, particles: [] };
  }

  step(state: State, x: StatInput, rng: SplitMix64 | null): [ParticleState, State] {
    if (rng === null) throw new Error(`${this.name} is randomized: it needs the statistical random state`);
    const s = state as ParticleState;
    const y = x.y as number;
    const size = this.size;
    let prior = s.particles;
    if (!prior.length) {
      const sd0 = Math.sqrt(this.p0);
      prior = [];
      for (let i = 0; i < size; i++) prior.push(rng.normal(this.m0, sd0));
    }
    const sd = Math.sqrt(this.q);
    const cloud = prior.map((p) => p + rng.normal(0.0, sd));
    const logw = cloud.map((p) => (-0.5 * ((y - p) * (y - p))) / this.r);
    const norm = requireFinite(logSumExp(logw), "log normalizing constant");
    const w = logw.map((lw) => Math.exp(lw - norm));
    let mean = 0.0;
    for (let i = 0; i < size; i++) mean += w[i] * cloud[i];
    let v = 0.0;
    for (let i = 0; i < size; i++) v += w[i] * ((cloud[i] - mean) * (cloud[i] - mean));
    let sw = 0.0;
    for (const wi of w) sw += wi * wi;
    const ess = 1.0 / sw;
    // systematic resampling: one uniform draw
    const u = rng.random() / size;
    const resampled: number[] = [];
    let i = 0, cum = w[0];
    for (let j = 0; j < size; j++) {
      const target = u + j / size;
      while (cum < target && i < size - 1) {
        i += 1;
        cum += w[i];
      }
      resampled.push(cloud[i]);
    }
    const n = s.n + 1;
    return [{ n, particles: resampled }, { n, mean, var: v, ess }];
  }

  decodeState(data: Json): ParticleState {
    const s = decodeFields({ n: "count", particles: "numbers" }, data) as ParticleState;
    const expected = s.n ? this.size : 0;
    requireState(s.particles.length === expected, `${s.particles.length} particles, expected ${expected}`);
    return s;
  }
}

register("bootstrap_particle_filter", "1", (c) => BootstrapParticleFilter.fromConfig(c));
