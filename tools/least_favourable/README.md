# Computed roots behind `src/seqinfer/data/least_favourable_normal.json`

Each line is one root of Lorden's characterisation for normal data, computed
by `tools/least_favourable_table.py` (fields: the point `u` in [-1, 1]^3, the
error rates as `x1 = log alpha0`, `x2 = log alpha1`, `x3 = rho`, the horizon,
`lambda` the root, `lambda1` Lorden's first-order point, `y = lambda - lambda1`,
and the number of designs and seconds it took).

- `pilot.jsonl` — lines through the domain and other horizons; chose the model
  (degree 5 in each log alpha, 1 in rho, rho >= 1.5) and showed the dependence
  on delta at fixed rho to be below 1.5e-3.
- `design.jsonl` — 84 Gauss-Lobatto-Legendre nodes (degrees 6, 6, 3), alpha0 < alpha1.
- `check.jsonl` — 16 Halton points, not used in the fit.

Reproduce the table:

    python tools/least_favourable_table.py fit tools/least_favourable/design.jsonl \
        tools/least_favourable/check.jsonl --degrees 5 5 1 --rho-min 1.5 \
        --table src/seqinfer/data/least_favourable_normal.json
