# A programme of experiments: online false discovery rate

A product team ships an experiment every few days. Each test has its own
5% false positive rate, but over a year of a hundred tests in which most
ideas do nothing, several "wins" are noise. `MultipleTests` controls the
false discovery rate among tests that run *together*. Tests that run
*one after another*, each decided when it ends, need an online
procedure.

`OnlineFDR` is e-LOND (Xu and Ramdas 2024). Each finished experiment
hands over its e-value: the `log_e_value` of any exact test here, read
when the experiment stopped. The experiment is a discovery if the e-value
clears a level that depends on its position in the series and on the
discoveries so far. The expected share of false discoveries stays below
`alpha` at every point of the series, whatever the dependence between
the experiments.

## A year of experiments, simulated

The simulation assumes 40 experiments a year, planned in advance
(`horizon=40`), of which one in four has a real effect: conversion goes
from 10% to 14%. The level of experiment t is known when it starts.
So each experiment runs its test at exactly that level, until it
decides or reaches 20,000 users:

```python
import random
from sequential_analysis.planning import conversion_units
from sequential_analysis.procedures import OnlineFDR, TwoProportionTest

def run_experiment(lift, level, seed, budget=20_000):
    test = TwoProportionTest(alpha=level)
    state, draw, rnd = test.initial_state(), conversion_units(0.10 + lift, 0.10), random.Random(seed)
    for _ in range(budget):
        state, out = test.step(state, draw(rnd))
        if out.decision:
            break
    return out.log_e_value

rnd = random.Random(2026)
truth = [rnd.random() < 0.25 for _ in range(40)]

programme = OnlineFDR(alpha=0.1, horizon=40)
state, online, naive = programme.initial_state(), [], []
for seed, real in enumerate(truth):
    level = programme.level(state.t + 1, state.discoveries)  # known before the experiment starts
    log_e = run_experiment(0.04 if real else 0.0, level, seed)
    state, out = programme.step(state, {"log_e": log_e})
    online.append(out.rejected)
    naive.append(run_experiment(0.04 if real else 0.0, 0.05, 1000 + seed) >= 2.996)  # each at 5% alone

for name, found in (("each test at 5% on its own", naive), ("e-LOND, FDR 10%", online)):
    false = sum(f and not real for f, real in zip(found, truth))
    print(f"{name}: {sum(found)} discoveries, {false} of them false, "
          f"{sum(f and real for f, real in zip(found, truth))} of {sum(truth)} real effects found")
```

On its own, each test is a valid 5% test. Across the series, though, the
false discoveries add up with the number of null ideas. e-LOND asks
each experiment for stronger evidence, a level of 0.1/40 at first, and each
discovery lowers the bar for the experiments that follow. In exchange
the experiments run longer, and that is the honest cost of a programme
whose discoveries can be trusted.

## In production

* **The e-value to pass.** Give `OnlineFDR` the experiment's final
  `log_e_value`, from whichever test the experiment used. For
  experiments with several metrics, first combine them with
  `MultipleTests`, or pick the primary metric's e-value in advance.
* **The order.** Decide experiments in the order they finish, and never
  revisit one.
* **The schedule.** Without a planned number of experiments, leave
  `horizon` unset. The default schedule then spends most of the budget
  on the first experiments and demands more and more evidence from the
  later ones, unless discoveries raise the level again.
* **Checkpoints.** The state is two integers, and the run checkpoints
  like any other: keep the programme's run next to the experiments'.
