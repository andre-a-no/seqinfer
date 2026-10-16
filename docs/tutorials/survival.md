# A survival trial, monitored after every event

In a trial with a time-to-event endpoint (death, relapse, failure of a
device), the classical analysis is the log-rank test, run once or at a
few planned interim analyses. `AnytimeLogRankTest` can be looked at
after every event instead. It answers whether the hazard of the
treatment arm differs from that of the control arm, and it keeps its
error rate however often the trial is looked at and whenever it is
stopped.

The test needs no assumption about the baseline hazard (ter Schure,
Pérez-Ortiz, Ly and Grünwald 2024). At each event, given who is still at
risk, the event falls in the treatment arm with probability
`n_T θ / (n_T θ + n_C)`, where θ is the hazard ratio. Under H0: θ = 1
these conditional coin flips have a known law, whatever the hazard
itself does over time. Their likelihood ratio, mixed over θ, is a test
martingale.

## The trial

Patients enter over a year and are randomised 1:1. Each one is followed
until an event, dropping out, or the end of the study. Hazards are
exponential here, and the treatment has a hazard ratio of 0.6.

The records reach the test in the order they happen in calendar time:

* a patient enters (`kind` 0);
* a patient has an event (`kind` 1);
* a patient is censored (`kind` 2).

```python
import random
from sequential_analysis import KeyJoin, Observation, Recorder, Run, run_sync
from sequential_analysis.procedures import AnytimeLogRankTest

def trial(n=600, hazard_ratio=0.6, seed=1):
    rnd = random.Random(seed)
    happenings = []
    for patient in range(n):
        arm = patient % 2
        entry = rnd.uniform(0.0, 12.0)                      # months
        rate = 0.05 * (hazard_ratio if arm == 1 else 1.0)  # events per month
        event = entry + rnd.expovariate(rate)
        dropout = entry + rnd.expovariate(0.01)
        end = min(event, dropout, 36.0)                     # the study ends at month 36
        happenings += [(entry, arm, 0), (end, arm, 1 if end == event else 2)]
    happenings.sort()
    stream = []
    for i, (_, arm, kind) in enumerate(happenings):
        stream += [Observation("arm", arm, seq=i, key=i), Observation("kind", kind, seq=i, key=i)]
    return stream

seen = Recorder()
logrank = Run(AnytimeLogRankTest(alpha=0.05), topology=KeyJoin(("arm", "kind")), consumers=[seen])
run_sync(logrank, trial())
first = next(o for o in seen.outputs if o.decision)
print(f"decision after {first.n_events} events: {first.decision}; hazard ratio about "
      f"{first.hazard_ratio_estimate:.2f}, 95% confidence sequence [{first.lower:.2f}, {first.upper:.2f}]")
last = seen.outputs[-1]
print(f"at the end of the study, {last.n_events} events: [{last.lower:.2f}, {last.upper:.2f}]")
```

The confidence sequence for the hazard ratio holds simultaneously at
every event. An interval read at the moment of stopping is valid without
any correction for the looks before it.

The interval is computed on a grid of hazard ratios and widened by one
grid step on each side. So at the moment the test rejects θ = 1, its
upper end can still read 1.00.

## Is it safe? The level under no effect

Under no effect, the test should reject in at most 5% of trials:

```python
false_alarms = 0
for seed in range(40):
    run = Run(AnytimeLogRankTest(alpha=0.05, confidence_sequence=False), topology=KeyJoin(("arm", "kind")))
    run_sync(run, trial(n=300, hazard_ratio=1.0, seed=100 + seed))
    false_alarms += run.state.decision is not None
print(f"no effect: {false_alarms} of 40 trials rejected")
```

## The time scale: read this before using it

The guarantee needs proportional hazards on the time scale in which the
records arrive. Here that scale is calendar time. Two cases are fine:

* **Hazards constant in time.** With exponential survival, as above,
  the scale makes no difference.
* **Hazards that depend on calendar time.** A season, or a change of
  standard care, affects both arms at once.

Late entry is then left truncation, which the conditional coin flips
handle exactly.

The case that is not fine: hazards that depend on time since the
patient's own entry (high right after surgery, for example) while
patients enter at different times. Then the patients at risk at a
calendar moment are at different stages, the coin flips are not fair
under H0, and the test can be badly anti-conservative. In a simulation
with Weibull shape 0.3 in time since entry and entry spread over the
study, it rejected 21% of 4000 null trials at a nominal 5%.

For such trials, choose one of these:

* order the records by time since entry, if that is how the trial is
  analysed;
* stratify by narrow entry cohorts (next section);
* use a group sequential design (`GroupSequentialTest` with the
  log-rank statistic at planned analyses).

## Stratified by centre or by entry cohort

Patients in different centres, or with a different stage of disease, can
have very different hazards. The stratified test compares like with
like. Each record carries its stratum, and `strata` says how many there
are. The test keeps the numbers at risk of each arm in each stratum. An
event in stratum s falls in the treatment arm with probability
`n_T^s θ / (n_T^s θ + n_C^s)`.

Each stratum may have a baseline hazard of any shape. The hazard ratio θ
is common to all of them. The guarantee is exact, as before.

Stratifying matters most when the strata differ in both hazard and
allocation. In this trial the riskiest centre also treats most of its
patients:

```python
def records_to_stream(happenings, names):
    happenings.sort()
    stream = []
    for i, (_, *values) in enumerate(happenings):
        stream += [Observation(name, value, seq=i, key=i) for name, value in zip(names, values)]
    return stream

def multicentre_trial(n=300, hazard_ratio=1.0, seed=1):
    rnd = random.Random(seed)
    rates = [0.01, 0.05, 0.25]    # events per month in centres 0, 1, 2
    treated = [0.25, 0.5, 0.75]   # share of patients randomised to treatment there
    happenings = []
    for _ in range(n):
        centre = rnd.randrange(3)
        arm = 1 if rnd.random() < treated[centre] else 0
        entry = rnd.uniform(0.0, 12.0)
        event = entry + rnd.expovariate(rates[centre] * (hazard_ratio if arm == 1 else 1.0))
        dropout = entry + rnd.expovariate(0.01)
        end = min(event, dropout, 36.0)
        happenings += [(entry, arm, 0, centre), (end, arm, 1 if end == event else 2, centre)]
    return records_to_stream(happenings, ("arm", "kind", "centre"))

def rejects(procedure, stream, inputs):
    run = Run(procedure, topology=KeyJoin(inputs))
    run_sync(run, [o for o in stream if o.input in inputs])
    return run.state.decision is not None

plain = stratified = 0
for seed in range(10):
    stream = multicentre_trial(seed=200 + seed)
    plain += rejects(AnytimeLogRankTest(confidence_sequence=False), stream, ("arm", "kind"))
    stratified += rejects(AnytimeLogRankTest(confidence_sequence=False, strata=3, stratum_input="centre"),
                          stream, ("arm", "kind", "centre"))
print(f"no effect, 10 trials: unstratified rejected {plain}, stratified by centre {stratified}")
```

There is no effect, yet the unstratified test rejects every one of these
trials. It sees more events among treated patients and blames the
treatment, not the centre. The stratified test rejects none.

With a real effect, the stratified run reads like the unstratified one:

```python
seen = Recorder()
run = Run(AnytimeLogRankTest(strata=3, stratum_input="centre"), topology=KeyJoin(("arm", "kind", "centre")),
          consumers=[seen])
run_sync(run, multicentre_trial(hazard_ratio=0.6, seed=7))
first = next(o for o in seen.outputs if o.decision)
last = seen.outputs[-1]
print(f"rejected after {first.n_events} events; at the end the hazard ratio is about "
      f"{last.hazard_ratio_estimate:.2f}, [{last.lower:.2f}, {last.upper:.2f}]")
```

The outputs' `at_risk` is the total of each arm over the strata. The
state keeps two counts per stratum, so its size stays fixed.

Even with balanced allocation, stratifying can buy a lot of power when
the strata differ widely. In a simulation of 600 trials with two centres
whose hazards differ a hundredfold, power at a hazard ratio of 0.6 was
82% stratified against 18% unstratified.

### Entry cohorts

For hazards that depend on time since entry, use the month (or week) of
entry as the stratum. Within a narrow cohort, the patients at risk at a
given moment are at about the same stage.

```python
def staggered_trial(n=200, seed=1, cohorts=25):
    rnd = random.Random(seed)
    happenings = []
    for _ in range(n):
        arm = rnd.randrange(2)
        entry = rnd.uniform(0.0, 25.0)                          # months
        event = entry + rnd.expovariate(1.0) ** (1 / 0.3)       # Weibull, shape 0.3: high right after entry
        dropout = entry + rnd.expovariate(0.3)
        cohort = min(cohorts - 1, int(entry / 25.0 * cohorts))  # the month of entry
        end = min(event, dropout)
        happenings += [(entry, arm, 0, cohort), (end, arm, 1 if end == event else 2, cohort)]
    return records_to_stream(happenings, ("arm", "kind", "cohort"))

plain = stratified = 0
for seed in range(20):
    stream = staggered_trial(seed=300 + seed)
    plain += rejects(AnytimeLogRankTest(confidence_sequence=False), stream, ("arm", "kind"))
    stratified += rejects(AnytimeLogRankTest(confidence_sequence=False, strata=25, stratum_input="cohort"),
                          stream, ("arm", "kind", "cohort"))
print(f"no effect, 20 trials: unstratified rejected {plain}, stratified by month of entry {stratified}")
```

This is an approximation, not an exact remedy. Within a cohort, entry
still varies, so the cohorts must be narrow compared with how fast the
hazard changes after entry. In 4000 simulated trials like these, at a
nominal 5%, the share of null trials rejected was:

| entry cohorts | 1 (unstratified) | 4   | 8   | 16   | 25 (one month) | 50   |
|---------------|------------------|-----|-----|------|----------------|------|
| rejected      | 21%              | 16% | 11% | 5.8% | 3.8%           | 1.5% |

An exact analysis, with the records ordered by time since entry, rejects
1.7%. Narrow cohorts also cost power: an event in a cohort with one arm
empty says nothing. At a hazard ratio of 0.6, 25 cohorts detected the
effect in 39% of trials and 50 cohorts in 21%. The exact analysis in
time since entry detected it in 54%.

## Method and cost

`method="mixture"`, the default, mixes the likelihood ratio over a
normal prior on log θ with standard deviation `prior_sd` = 0.5. This is
our choice, not the paper's, and it is most sensitive at about 130
events, the size of a classical trial for a hazard ratio of 0.6.
`method="grow"` bets on a penalised partial-likelihood estimate instead.
It is as powerful two-sided and slower one-sided.

An event costs about 50 µs without the confidence sequence, and 0.2 to
0.6 ms with it. The state is constant: about 5 kB of JSON in each
checkpoint.
