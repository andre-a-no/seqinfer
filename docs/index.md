# sequential-analysis

Sequential statistical inference in Python: tests and estimates that may
be looked at after every observation and stopped at any time, with their
error guarantees intact.

The library has four kinds of methods:

* **Classical sequential tests:** Wald's SPRT, Lorden's 2-SPRT, optimal
  Kiefer–Weiss plans, group sequential trials with spending functions.
* **Anytime-valid inference:** mixture SPRTs, confidence sequences,
  nonparametric tests by betting, and A/B tests with e-values and
  false-discovery-rate control.
* **Change detection:** CUSUM, Shiryaev–Roberts, nonparametric
  e-detectors and Bayesian online change points.
* **Filters and smoothers:** Kalman (including a robust one), hidden
  Markov models and particle filters.

Every run can be checkpointed as plain JSON and resumed exactly. A run
can also be resumed by the [TypeScript executor](typescript.md), and
the formats are [specified](spec.md).

```bash
pip install sequential-analysis            # pure Python
pip install "sequential-analysis[fast]"    # with numpy for the array-heavy procedures
```

## A first run

```python
from sequential_analysis import Run, run_sync
from sequential_analysis.procedures import Gaussian, TwoSPRT
from sequential_analysis.sources import gaussian

test = TwoSPRT(Gaussian(sigma=1.0), theta0=0.0, theta1=0.5, alpha0=0.05, alpha1=0.05)
run = Run(test)
run_sync(run, gaussian("x", mean=0.5, sd=1.0, seed=7))
print(run.state.decision, "after", run.t, "observations")

checkpoint = run.checkpoint()   # plain JSON: state, random state, buffers, provenance
```

A procedure is only a transition function: it takes a state and an
observation and returns a new state and an output. Everything around it
is kept apart:

* **adapters** produce observations;
* **delivery** orders and deduplicates them;
* **topology** pairs or joins them;
* the **run** commits each step, keeps the history digest and writes
  checkpoints;
* **consumers** act on the outputs.

## Where to go next

* [A/B test, monitored continuously](tutorials/ab-test.md):
  * conversion and revenue;
  * traffic split checks;
  * covariate adjustment;
  * many metrics and many arms.
* [Group sequential trial](tutorials/group-sequential.md):
  * a design with efficacy and futility boundaries;
  * its operating characteristics;
  * the trial run against it.
* [Process monitoring and change detection](tutorials/monitoring.md):
  * parametric and nonparametric detectors;
  * filtering;
  * smoothing afterwards.
* [Methods](methods.md) lists every procedure, and [Speed](performance.md)
  gives the cost of a step of each.
