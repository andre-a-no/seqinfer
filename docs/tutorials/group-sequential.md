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

## 5. A binary endpoint: two proportions

Most trials do not measure a normal mean. Take a response rate of 10%
under control, and a treatment worth detecting at 12%. The same kind of
design gives the boundaries; the sample size comes from the two
proportions:

```python
from sequential_analysis.procedures import GroupSequentialProportions

binary = group_sequential_design(0.025, [1 / 3, 2 / 3, 1.0], spending="obrien-fleming")
n_max = binary.max_sample_size_two_proportions(p_control=0.10, p_treatment=0.12, power=0.8)
print(f"patients in both arms together: {n_max} "
      f"({binary.inflation_factor(0.8):.3f} times a single analysis)")

import random
rnd = random.Random(3)
test = GroupSequentialProportions.from_design(binary, n_max)
state = test.initial_state()
for patient in range(n_max):
    arm = patient % 2                                       # 1 treatment, 0 control
    response = int(rnd.random() < (0.12 if arm == 1 else 0.10))
    state, out = test.step(state, {"arm": arm, "y": response})
    if out.analysis:
        print(f"analysis {out.analysis} after {out.n} patients: Z = {out.z:.2f}, bound {out.bound:.2f}")
    if out.decision:
        print("decision:", out.decision)
        break
```

Z is the pooled two-proportion statistic. Its square is the chi-square
test without continuity correction. For a continuous endpoint in two
arms, `GroupSequentialTwoMeans` works the same way. It uses a known
standard deviation, or the pooled t statistic, which the tests check
against R's `t.test`. Its sample size comes from
`max_sample_size_two_means`. The sample size uses the method of
rpact's `getSampleSizeRates`. The tests check it against rpact for 18
combinations of rates, allocation ratios and designs.

## 6. A survival endpoint: events, not patients

In a survival trial the information is in the events. The design asks
for a number of events (Schoenfeld's formula). The analyses happen after
a third, two thirds and all of them:

```python
from sequential_analysis.procedures import GroupSequentialLogRank

events = binary.max_events(hazard_ratio=0.7, power=0.8)
print("events needed:", events)

def trial(n, hazard_ratio, seed):
    """Entries, events and censorings of a trial in calendar order."""
    rnd = random.Random(seed)
    happenings = []
    for patient in range(n):
        arm = patient % 2
        entry = rnd.uniform(0.0, 12.0)
        event = entry + rnd.expovariate(0.05 * (hazard_ratio if arm == 1 else 1.0))
        end = min(event, entry + rnd.expovariate(0.01), 48.0)
        happenings += [(entry, arm, 0), (end, arm, 1 if end == event else 2)]
    return [{"arm": arm, "kind": kind} for _, arm, kind in sorted(happenings)]

logrank = GroupSequentialLogRank.from_design(binary, events)
state = logrank.initial_state()
for record in trial(400, 0.7, seed=8):
    state, out = logrank.step(state, record)
    if out.analysis:
        print(f"analysis {out.analysis} after {out.events} events: Z = {out.z:.2f}, bound {out.bound:.2f}, "
              f"hazard ratio about {out.hazard_ratio_estimate:.2f}")
    if out.decision:
        print("decision:", out.decision)
        break
```

The records are those of `AnytimeLogRankTest`: a patient enters (0),
has the event (1) or is censored (2). Z is the log-rank statistic with
late entry. The tests check it against the score test of a Cox model in
R's survival package, at four interim analyses, to 1e-12. With
`strata=` the test is stratified: by centre, by a prognostic factor, or
by entry cohort.

## 7. When the meetings move

A committee rarely meets exactly at a third of the events. Alpha
spending lets the analyses happen whenever they do. The spending
function is evaluated at the information actually reached, and the
boundaries already used never change. A trial that ends early
(under-running) or late (over-running) spends what is left at its final
analysis.

`GroupSequentialMonitor` takes one record per analysis: the statistic Z
and its information. It works for any endpoint whose statistic you
compute yourself, such as a Cox model with covariates, a mixed model or
a ratio of means. For a log-rank statistic the information is about a
quarter of the events:

```python
from sequential_analysis.procedures import GroupSequentialMonitor

monitor = GroupSequentialMonitor.from_design(binary, max_information=events / 4, final_input="final")
state = monitor.initial_state()
meetings = [(0.9, 70), (1.8, 150), (2.3, 230)]              # (Z, events) at three meetings
for k, (z, d) in enumerate(meetings):
    state, out = monitor.step(state, {"z": z, "information": d / 4, "final": int(k == len(meetings) - 1)})
    print(f"meeting {out.analysis} at {out.fraction:.0%} of the information: bound {out.bound:.3f}, "
          f"alpha spent {out.spent:.2g}, decision {out.decision}")
```

The last meeting comes at 230 of the planned 250 events, and is declared
final. It spends all the alpha left. The boundaries are those of
`binary.at([0.28, 0.6, 0.92])`. The tests check them against rpact,
including under- and over-running.

## 8. Checked against rpact

`tools/r_reference.R` computes reference values with rpact and R's
survival package, and `tests/test_r_reference.py` compares with them.
The comparison covers 56 designs, combining:

* O'Brien–Fleming, Pocock and power spending;
* one- and two-sided tests;
* two to five analyses;
* with and without futility boundaries.

The comparison includes:

* the boundaries;
* the drift for the power;
* the crossing and futility probabilities;
* the expected sample size and the inflation factors;
* boundaries at realised information;
* 44 sample sizes for means, rates and events.

They agree to 2e-7 or better. A CI job regenerates the references with
R and repeats the comparison on every change.
