# An A/B test, monitored continuously

A classical A/B test fixes its sample size in advance and looks once.
Looking at a t-test every day and stopping when p < 0.05 inflates the
false positive rate several times over.

The tests here are built to be looked at after every user. They have no
sample size to fix, they may be stopped whenever one likes, and their
error rates hold at whatever time they are stopped.

This tutorial goes through:

* checking the traffic split;
* testing conversion exactly;
* estimating the effect on revenue with a pre-experiment covariate;
* many metrics at once;
* several treatment arms.

## The data

Each user produces events with the same key: the assignment (`arm`),
the user's spend in the month before the experiment (`x`, the
covariate), whether the user converted, and the revenue. The events
arrive separately, the outcome later than the assignment, as they would
from an event log. Here they are simulated:

```python
import random
from sequential_analysis import Observation

def events(n, conversion=(0.10, 0.13), lift=0.4, split=0.5, seed=1):
    rnd = random.Random(seed)
    for user in range(n):
        arm = int(rnd.random() < split)
        spend_before = rnd.gauss(10.0, 3.0)
        converted = int(rnd.random() < conversion[arm])
        revenue = spend_before + rnd.gauss(lift * arm, 2.0)
        for name, value in (("arm", arm), ("x", spend_before), ("converted", converted), ("revenue", revenue)):
            yield Observation(name, value, seq=user, key=user)

stream = list(events(20_000))
```

`KeyJoin` puts the events of one user back together, whatever order
they arrive in.

## 1. Is the traffic split as intended?

Run a sample-ratio-mismatch check first, and keep it running throughout.
A mismatch means the assignment or the logging is broken, and then no
other result of the experiment can be trusted.

```python
from sequential_analysis import Recorder, Run, run_sync
from sequential_analysis.procedures import SampleRatioMismatch

seen = Recorder()
srm = Run(SampleRatioMismatch(weights=(0.5, 0.5), alpha=0.001), consumers=[seen])
run_sync(srm, (o for o in stream if o.input == "arm"))
last = seen.outputs[-1]
print("split:", [round(p, 4) for p in last.proportions], "mismatch:", last.decision)
```

The check is exact at any time, and so is a broken split it would
catch. Here is a 50/50 design that actually sends 52% of users to
treatment:

```python
broken = Run(SampleRatioMismatch(weights=(0.5, 0.5), alpha=0.001, stop_on_reject=True))
run_sync(broken, (o for o in events(50_000, split=0.52, seed=2) if o.input == "arm"))
print("broken split:", broken.state.decision, "after", broken.t, "users")
```

## 2. Did conversion change?

`TwoProportionTest` is exact. It needs no normal approximation and no
minimum sample size, and it stays valid when the allocation is
unbalanced or changes during the experiment.

```python
from sequential_analysis import KeyJoin
from sequential_analysis.procedures import TwoProportionTest

conversion = Run(TwoProportionTest(alpha=0.05, input="converted"), topology=KeyJoin(("arm", "converted")))
run_sync(conversion, (o for o in stream if o.input in ("arm", "converted")))
state = conversion.state
print("users:", sum(state.trials), "conversion rates:",
      [round(s / n, 4) for s, n in zip(state.successes, state.trials)],
      "decision:", state.decision)
```

The decision `reject_h0` means conversion differs, at an error rate of
at most 5%, even though the test was looked at after every user.

**What exactness costs.** Valid at every sample size, the exact test
pays a price in users. By simulation, a 10% → 13% conversion lift is
detected after a median of about 10,500 users. The asymptotic test below
needs about 4,000, and a fixed-sample test planned for 80% power needs
3,500, but that test may be looked at only once.

For a 10% → 12% lift the medians are 22,000 against 9,500 users, about
2.5 times. The extra cost of exactness is a logarithmic penalty for
estimating both conversion rates.

When thousands of users arrive anyway, the asymptotic confidence
sequence of section 3 also works for a 0/1 outcome, and gives an
interval for the lift:

```python
from sequential_analysis.procedures import ABTest

lift = Run(ABTest(input="converted"), topology=KeyJoin(("arm", "converted")))
run_sync(lift, (o for o in stream if o.input in ("arm", "converted")))
print("asymptotic test of the lift:", lift.state.decision)
```

* **The output.** `log_e_value` is the evidence, and
  `p_value = 1/max e` is an always-valid p-value.
* **Direction.** Use `alternative="greater"` to look for an improvement
  only.

## 3. How much did revenue change? Adjusting for the covariate

Revenue is noisy, but much of the noise is explained by how much the
user spent before the experiment. `ABTest(covariate=True)` subtracts a
regression on that covariate, fitted only on earlier users. Its
confidence sequence for the effect is then as narrow as it would be with
fewer users of variance:

$$\psi_t = \left(\frac{T_t}{\pi} - \frac{C_t}{1-\pi}\right)\,\bigl(y_t - \hat f_{t-1}(x_t)\bigr),$$

which estimates the effect without bias for any $\hat f$ fitted on the
past. This is CUPED's idea, without its in-sample bias.

```python
from sequential_analysis.procedures import ABTest

plain = Run(ABTest(allocation=0.5, input="revenue"), topology=KeyJoin(("arm", "revenue")))
adjusted = Run(ABTest(allocation=0.5, covariate=True, input="revenue", covariate_input="x"),
               topology=KeyJoin(("arm", "revenue", "x")))
seen = Recorder()
adjusted.consumers.append(seen)
run_sync(plain, (o for o in stream if o.input in ("arm", "revenue")))
run_sync(adjusted, (o for o in stream if o.input in ("arm", "revenue", "x")))

last = seen.outputs[-1]
print(f"effect {last.estimate:.3f}, 95% confidence sequence [{last.lower:.3f}, {last.upper:.3f}]")
print(f"the covariate saves {1 - last.variance_ratio:.0%} of the sample size")
```

The interval is valid uniformly over time once a thousand or so users
have been seen. Before `start` users it is not reported. The
`variance_ratio` says how much the covariate helped. 1 − ratio is the
fraction of users it saves: with spend explaining most of the revenue
variance, that is most of them.

## 4. Many metrics: control the false discovery rate

With ten metrics each tested at 5%, about one in two experiments without
any effect will show something. `MultipleTests` runs several tests on the
same users and applies the e-BH procedure after every step. Whenever you
stop, the expected share of false discoveries among the reported ones is
at most `alpha`, under any dependence between the metrics.

```python
from sequential_analysis.procedures import MultipleTests

metrics = MultipleTests({
    "conversion": TwoProportionTest(input="converted"),
    "traffic split": SampleRatioMismatch((0.5, 0.5)),
}, alpha=0.05)
many = Run(metrics, topology=KeyJoin(("arm", "converted")))
log = Recorder()
many.consumers.append(log)
run_sync(many, (o for o in stream if o.input in ("arm", "converted")))
print("discoveries:", log.outputs[-1].rejected)
```

Every exact test of the library declares its e-process and can be
combined this way. That includes the mixture SPRTs, the betting and sign
tests, the quantile test, the kernel two-sample test, and the A/B tests
above.

## 5. Several treatment arms

`MultipleTests.versus_control` compares each arm with the control, with
the false discovery rate controlled across arms:

```python
rnd = random.Random(3)
rates = {0: 0.10, 1: 0.10, 2: 0.14}
arms_stream = []
for user in range(30_000):
    arm = rnd.randrange(3)
    arms_stream += [Observation("arm", arm, seq=user, key=user),
                    Observation("y", int(rnd.random() < rates[arm]), seq=user, key=user)]

arms = Run(MultipleTests.versus_control([1, 2], control=0, alpha=0.05), topology=KeyJoin(("arm", "y")))
log = Recorder()
arms.consumers.append(log)
run_sync(arms, arms_stream)
print("arms better or worse than control:", log.outputs[-1].rejected)
```

## Running it for real

* **Live events.** Feed the run from your event log as it grows, with
  `from_kafka` or `from_sql` (see [data sources](../adapters.md)).
* **Restarts.** Checkpoint after every batch. A restarted service
  restores the run and asks the source for the events after the
  checkpointed position, and nothing is counted twice.
* **Stopping.** Stop when the test decides or when the budget runs out.
  The guarantees hold either way.
