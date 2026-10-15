// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// One-parameter likelihood families; mirrors procedures/families.py.
import { InputContract } from "../contracts.js";
import { type Config, invalid, num, str } from "./util.js";

export interface Family {
  spec(): Config;
  contract(name: string): InputContract;
  check(theta: number): void;
  /** log f_num(x) - log f_den(x). */
  llr(x: number, num: number, den: number): number;
  /** Kullback-Leibler divergence from theta=a to theta=b. */
  kl(a: number, b: number): number;
}

export class Gaussian implements Family {
  constructor(readonly sigma = 1.0) {
    if (!(sigma > 0 && Number.isFinite(sigma))) invalid("sigma must be positive");
  }
  spec(): Config {
    return { family: "gaussian", sigma: this.sigma };
  }
  contract(name: string): InputContract {
    return new InputContract(name, "real");
  }
  check(theta: number): void {
    if (!Number.isFinite(theta)) invalid("mean must be finite");
  }
  llr(x: number, num: number, den: number): number {
    return ((num - den) / (this.sigma * this.sigma)) * (x - 0.5 * (num + den));
  }
  kl(a: number, b: number): number {
    return ((a - b) * (a - b)) / (2.0 * (this.sigma * this.sigma));
  }
}

export class Bernoulli implements Family {
  spec(): Config {
    return { family: "bernoulli" };
  }
  contract(name: string): InputContract {
    return new InputContract(name, "binary");
  }
  check(theta: number): void {
    if (!(0.0 < theta && theta < 1.0)) invalid("success probability must lie strictly between 0 and 1");
  }
  llr(x: number, num: number, den: number): number {
    return x ? Math.log(num / den) : Math.log((1.0 - num) / (1.0 - den));
  }
  kl(a: number, b: number): number {
    return a * Math.log(a / b) + (1.0 - a) * Math.log((1.0 - a) / (1.0 - b));
  }
}

export class Poisson implements Family {
  spec(): Config {
    return { family: "poisson" };
  }
  contract(name: string): InputContract {
    return new InputContract(name, "integer", [], 0);
  }
  check(theta: number): void {
    if (!(Number.isFinite(theta) && theta > 0)) invalid("rate must be positive and finite");
  }
  llr(x: number, num: number, den: number): number {
    return x * Math.log(num / den) - (num - den);
  }
  kl(a: number, b: number): number {
    return a * Math.log(a / b) - a + b;
  }
}

export class Exponential implements Family {
  spec(): Config {
    return { family: "exponential" };
  }
  contract(name: string): InputContract {
    return new InputContract(name, "real", [], 0);
  }
  check(theta: number): void {
    if (!(Number.isFinite(theta) && theta > 0)) invalid("rate must be positive and finite");
  }
  llr(x: number, num: number, den: number): number {
    return Math.log(num / den) - (num - den) * x;
  }
  kl(a: number, b: number): number {
    return Math.log(a / b) + b / a - 1.0;
  }
}

/** The family a config's {"family": ..., "sigma"?} describes. */
export function familyFromSpec(spec: Config): Family {
  const kind = str(spec, "family");
  const keys = Object.keys(spec).sort().join(",");
  if (kind === "gaussian" && keys === "family,sigma") return new Gaussian(num(spec, "sigma"));
  if (keys !== "family") invalid(`unexpected family spec ${JSON.stringify(spec)}`);
  if (kind === "bernoulli") return new Bernoulli();
  if (kind === "poisson") return new Poisson();
  if (kind === "exponential") return new Exponential();
  return invalid(`unknown family '${kind}'`);
}
