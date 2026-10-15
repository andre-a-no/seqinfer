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
study, it rejected 24% of null trials at a nominal 5%.

For such trials, choose one of these:

* order the records by time since entry, if that is how the trial is
  analysed;
* stratify by entry cohort;
* use a group sequential design (`GroupSequentialTest` with the
  log-rank statistic at planned analyses).

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
