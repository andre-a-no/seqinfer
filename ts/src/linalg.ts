// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
// Commercial licenses for use outside the AGPL: see COMMERCIAL.md
// Small dense matrices for the state-space filters; mirrors linalg.py, every sum left to right in the same order.

export type Vector = number[];
export type Matrix = number[][];

/** An array of finite numbers of the given size (any size if null); a number is a vector of one. */
export function vector(value: unknown, size: number | null, name: string): Vector {
  if (typeof value === "number") value = [value];
  if (!Array.isArray(value)) throw new RangeError(`${name} must be a sequence of numbers`);
  if (!value.every((v) => typeof v === "number")) throw new RangeError(`${name} must be a sequence of numbers`);
  const out = [...(value as number[])];
  if (size !== null && out.length !== size) throw new RangeError(`${name} must have ${size} entries, got ${out.length}`);
  if (!out.length || !out.every(Number.isFinite)) throw new RangeError(`${name} must be non-empty and finite`);
  return out;
}

/** An array of rows of finite numbers; a number is a 1x1 matrix. */
export function matrix(value: unknown, rows: number | null, cols: number | null, name: string): Matrix {
  if (typeof value === "number") value = [[value]];
  if (!Array.isArray(value) || !value.length) throw new RangeError(`${name} must be a non-empty sequence of rows`);
  const out = value.map((row) => vector(row, cols, `${name} row`));
  if (rows !== null && out.length !== rows) throw new RangeError(`${name} must have ${rows} rows, got ${out.length}`);
  const width = out[0].length;
  if (out.some((row) => row.length !== width)) throw new RangeError(`${name} rows differ in length`);
  return out;
}

function sameLength(a: { length: number }, b: { length: number }): void {
  if (a.length !== b.length) throw new Error("zip_strict: the iterables have different lengths");
}

export function dot(a: Vector, b: Vector): number {
  sameLength(a, b);
  let total = 0.0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}

export function transpose(a: Matrix): Matrix {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const bt = transpose(b);
  return a.map((row) => bt.map((col) => dot(row, col)));
}

export function matvec(a: Matrix, v: Vector): Vector {
  return a.map((row) => dot(row, v));
}

export function add(a: Matrix, b: Matrix): Matrix {
  sameLength(a, b);
  return a.map((r, i) => (sameLength(r, b[i]), r.map((x, j) => x + b[i][j])));
}

export function sub(a: Matrix, b: Matrix): Matrix {
  sameLength(a, b);
  return a.map((r, i) => (sameLength(r, b[i]), r.map((x, j) => x - b[i][j])));
}

export function vadd(a: Vector, b: Vector): Vector {
  sameLength(a, b);
  return a.map((x, i) => x + b[i]);
}

export function vsub(a: Vector, b: Vector): Vector {
  sameLength(a, b);
  return a.map((x, i) => x - b[i]);
}

export function scale(a: Matrix, c: number): Matrix {
  return a.map((row) => row.map((x) => c * x));
}

export function identity(n: number): Matrix {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1.0 : 0.0)));
}

/** (A + A') / 2: removes the asymmetry that rounding leaves in a covariance. */
export function symmetrize(a: Matrix): Matrix {
  const n = a.length;
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => 0.5 * (a[i][j] + a[j][i])));
}

export function isSymmetric(a: Matrix): boolean {
  const n = a.length;
  if (a[0].length !== n) return false;
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) if (a[i][j] !== a[j][i]) return false;
  return true;
}

/** A B A', symmetrized. */
export function quadratic(a: Matrix, b: Matrix): Matrix {
  return symmetrize(matmul(matmul(a, b), transpose(a)));
}

/** Lower-triangular L with L L' = A; throws RangeError if A is not positive definite. */
export function cholesky(a: Matrix): Matrix {
  const n = a.length;
  const low = Array.from({ length: n }, () => new Array<number>(n).fill(0.0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j];
      for (let k = 0; k < j; k++) s -= low[i][k] * low[j][k];
      if (i === j) {
        if (!(s > 0.0)) throw new RangeError("matrix is not positive definite");
        low[i][i] = Math.sqrt(s);
      } else low[i][j] = s / low[j][j];
    }
  }
  return low;
}

/** x with L L' x = b. */
export function solveCholesky(low: Matrix, b: Vector): Vector {
  const n = low.length;
  const y = new Array<number>(n).fill(0.0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= low[i][k] * y[k];
    y[i] = s / low[i][i];
  }
  const x = new Array<number>(n).fill(0.0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= low[k][i] * x[k];
    x[i] = s / low[i][i];
  }
  return x;
}

/** Inverse of a symmetric positive definite matrix, symmetrized. */
export function inverseSpd(a: Matrix): Matrix {
  const low = cholesky(a);
  const n = a.length;
  const columns = Array.from({ length: n }, (_, j) => solveCholesky(low, Array.from({ length: n }, (_, i) => (i === j ? 1.0 : 0.0))));
  return symmetrize(transpose(columns));
}

export function logDetSpd(a: Matrix): number {
  const low = cholesky(a);
  let total = 0.0;
  for (let i = 0; i < a.length; i++) total += Math.log(low[i][i]);
  return 2.0 * total;
}

export function trace(a: Matrix): number {
  let total = 0.0;
  for (let i = 0; i < a.length; i++) total += a[i][i];
  return total;
}
