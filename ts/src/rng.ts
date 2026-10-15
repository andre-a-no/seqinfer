// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// SplitMix64: the statistical random state, one 64-bit integer.
import { createHash } from "node:crypto";

const MASK = (1n << 64n) - 1n;
const GOLDEN = 0x9e3779b97f4a7c15n;

export class SplitMix64 {
  state: bigint;

  constructor(state: bigint) {
    this.state = state & MASK;
  }

  /** An independent stream for a named role: the first 8 bytes of SHA-256("<seed>/<role>"), big-endian. */
  static forRole(seed: bigint | number | string, role: string): SplitMix64 {
    const digest = createHash("sha256").update(`${BigInt(seed)}/${role}`, "utf8").digest();
    return new SplitMix64(digest.readBigUInt64BE(0));
  }

  copy(): SplitMix64 {
    return new SplitMix64(this.state);
  }

  nextU64(): bigint {
    this.state = (this.state + GOLDEN) & MASK;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK;
    return z ^ (z >> 31n);
  }

  /** Uniform on [0, 1) with 53 bits of precision. */
  random(): number {
    return Number(this.nextU64() >> 11n) * (1 / 2 ** 53);
  }

  /** Box-Muller without caching the second variate. */
  normal(mean = 0, sd = 1): number {
    const u1 = 1 - this.random();
    const u2 = this.random();
    return mean + sd * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  hex(): string {
    return this.state.toString(16).padStart(16, "0");
  }

  static fromHex(hex: string): SplitMix64 {
    return new SplitMix64(BigInt("0x" + hex));
  }
}
