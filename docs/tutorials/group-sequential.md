# A group sequential trial

A clinical trial cannot look at its data after every patient. A data
monitoring committee meets a few times, and at each meeting decides
whether to stop. Stopping is possible either for efficacy, when the
treatment is clearly better, or for futility, when it clearly will not
be shown to be.

Group sequential designs (Pocock; O'Brien and Fleming; Lan and DeMets)
place the boundaries of those analyses so that the overall type I error
stays at the planned level.

This tutorial:

1. designs such a trial;
2. checks its operating characteristics;
3. runs it on simulated patients;
4. shows how a checkpoint carries it between meetings.

## 1. The design

Plan four analyses, after 25%, 50%, 75% and 100% of the patients:

* a one-sided type I error of 2.5%;
* 90% power;
* O'Brien–Fleming-like alpha spending, conservative early;
* non-binding futility boundaries from a beta-spending function.

```python
from sequential_analysis.group_sequential import group_sequential_design

design = group_sequential_design(
    0.025, [0.25, 0.5, 0.75, 1.0], spending="obrien-fleming",
    futility="obrien-fleming", power=0.9,
)
for k, (t, b, f) in enumerate(zip(design.fractions, design.bounds, design.futility), start=1):
    print(f"analysis {k} at {t:.0%} of the patients: stop for efficacy if Z >= {b:.3f}, for futility if Z <= {f:.3f}")
```

The boundaries are for the standardised statistic Z_k at analysis k. The
futility boundaries are *non-binding*: the efficacy boundaries are
computed as if they did not exist. The type I error therefore stays at
2.5% even if the committee decides to continue past one.

## 2. How many patients, and what to expect

With a treatment effect of 0.3 standard deviations to detect:

```python
n_max = design.max_sample_size(effect=0.3, sigma=1.0)
print("maximum number of patients:", n_max)

drift = design.drift
print(f"power at the planned effect: {design.power(drift):.3f}")
print(f"expected share of the patients used, no effect:      {design.expected_fraction(0.0):.2f}")
print(f"expected share of the patients used, planned effect: {design.expected_fraction(drift):.2f}")
print("chance of stopping for efficacy at each analysis, planned effect:",
      [round(p, 3) for p in design.crossing(drift)])
```

These numbers are computed by numerical integration, not by
simulation:

* the power;
* the expected share of the patients used;
* the chance of stopping at each analysis.

The futility boundaries are what saves patients when the treatment does
not work. Under the planned effect, most trials stop at the second or
third analysis.

## 3. Running the trial

`GroupSequentialTest.from_design` turns the design into a procedure.
The procedure takes one patient's outcome at a time and, at the analyses,
compares Z_k with the boundaries:

```python
from sequential_analysis import Recorder, Run, run_sync
from sequential_analysis.procedures import GroupSequentialTest
from sequential_analysis.sources import gaussian

test = GroupSequentialTest.from_design(design, n_max, sigma=1.0)
analyses = Recorder()
trial = Run(test, consumers=[lambda e: analyses(e) if e.output.analysis else None])
run_sync(trial, gaussian("x", mean=0.3, sd=1.0, seed=11))
for e in analyses.events:
    o = e.output
    print(f"analysis {o.analysis} (n = {o.n}): Z = {o.z:.2f}, efficacy bound {o.bound:.2f}, decision {o.decision}")
print("stopped after", trial.t, "of", n_max, "patients")
```

When the standard deviation is not known, pass `sigma=None`. Z is then
the t statistic, and the boundaries are converted to the t distribution
at each analysis.

## 4. Between meetings

In a real trial, outcomes arrive over months, and the analysis runs
whenever the data manager uploads a batch. The run can be checkpointed
after every batch and restored for the next. The checkpoint holds:

* the running sums;
* which analyses have been done;
* the history digest of every outcome so far.

A checkpoint that does not belong to this design is refused:

```python
import json
from sequential_analysis import IncompatibleCheckpoint, Run

first = Run(GroupSequentialTest.from_design(design, n_max, sigma=1.0), run_id="trial-42")
run_sync(first, gaussian("x", mean=0.3, sd=1.0, seed=12, n=n_max // 4))
saved = json.dumps(first.checkpoint())         # store it with the trial master file

resumed = Run.restore(GroupSequentialTest.from_design(design, n_max, sigma=1.0), json.loads(saved))
print("resumed at patient", resumed.t, "with analyses done:", resumed.state.analysis)

other = group_sequential_design(0.025, [0.5, 1.0], spending="pocock")
try:
    Run.restore(GroupSequentialTest.from_design(other, n_max, sigma=1.0), json.loads(saved))
except IncompatibleCheckpoint as error:
    print("refused:", str(error)[:60], "...")
```

The checkpoint records the boundaries as part of the procedure's
identity. An analysis run with boundaries other than the planned ones
cannot pass for the planned trial.
