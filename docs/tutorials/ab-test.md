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
* several treatment arms;
* ratio metrics, such as revenue per session, and relative lift.

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
pays a price in users. It works on blocks of one treatment and one
control user. Use `block=(2, 1)` for a 2:1 split.

By simulation, the median number of users before a decision is:

| Lift | Exact test (`method="grow"`, the default) | Asymptotic test (below) | Universal inference (`method="universal"`) |
|---|---|---|---|
| 10% → 15% | 2,000 | 1,500 | 4,800 |
| 10% → 13% | 6,200 | 4,000 | 10,500 |
| 10% → 12% | 14,000 | 9,500 | 22,000 |

So the default exact test needs 1.3 to 1.6 times the users of the
asymptotic test. A fixed-sample test planned for 80% power at 10% → 13%
needs 3,500 users, but it may be looked at only once.

`method="universal"` is for allocation that adapts to the outcomes, as a
bandit's does. It needs no blocks and stays exact under any allocation
rule, at the price shown in the last column.

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

## 6. An exact interval for a bounded metric

The interval of section 3 is asymptotic. For a metric with known bounds
`BoundedABTest` gives an interval that holds exactly, at every sample
size and for every distribution. Examples are a conversion, a share of
sessions, or a revenue capped at 100.

It works on the same inverse-propensity scores, with betting in place of
the normal approximation:

```python
from sequential_analysis.procedures import BoundedABTest

capped = [Observation(o.input, min(max(o.value, 0.0), 30.0) if o.input == "revenue" else o.value,
                      seq=o.seq, key=o.key)
          for o in stream if o.input in ("arm", "revenue")]
exact = Run(BoundedABTest(lower=0.0, upper=30.0, allocation=0.5, input="revenue", grid=300),
            topology=KeyJoin(("arm", "revenue")))
seen = Recorder()
exact.consumers.append(seen)
run_sync(exact, capped)
last = seen.outputs[-1]
print(f"exact 95% interval for the revenue effect: [{last.lower:.3f}, {last.upper:.3f}]")
```

It is wider than the asymptotic one, which is the price of holding at
n = 50 as well as at n = 50,000. Its `log_e_value` can join a
`MultipleTests` family.

## 7. Counts with exposure: orders per user-day

When users stay in the experiment for different lengths of time, compare
event rates rather than means. `RateRatioTest` takes each user's count
and exposure and is exact for Poisson counts. It stays exact when the
common rate drifts over the weeks, because it only compares users who
arrived next to each other:

```python
import math
from sequential_analysis.procedures import RateRatioTest

rnd = random.Random(8)
rate_stream = []
for user in range(6000):
    arm = int(rnd.random() < 0.5)
    days = rnd.uniform(1, 14)
    weekly = 1 + 0.3 * math.sin(user / 400)  # seasonality, shared by both arms
    mean = 0.2 * days * weekly * (1.15 if arm else 1.0)
    k, p, limit = 0, rnd.random(), math.exp(-mean)
    while p > limit:
        k += 1
        p *= rnd.random()
    rate_stream += [Observation(n, v, seq=user, key=user) for n, v in (("arm", arm), ("count", k), ("exposure", days))]

rates = Run(RateRatioTest(), topology=KeyJoin(("arm", "count", "exposure")))
seen = Recorder()
rates.consumers.append(seen)
run_sync(rates, rate_stream)
last = seen.outputs[-1]
print(f"orders per user-day, treatment / control: {last.rate_ratio:.3f}, "
      f"95% confidence sequence [{last.lower:.3f}, {last.upper:.3f}], decision {last.decision}")
```

The counts have to be Poisson. If a few users order far more than the
rest (overdispersion), cap the counts and use `BoundedABTest`, or use
`ABTest`.

## 8. Ratio metrics: revenue per session

Many product metrics are ratios measured on something other than the
user: revenue per session, click-through rate (clicks per impression),
average order value (revenue per order). The users are randomised, but
the metric counts sessions. A user's sessions are not independent: a
big spender is a big spender in every session. A test that treats
sessions as observations therefore has a standard error that is too
small, and it finds effects that are not there.

`RatioABTest` works per user. Each user reports a numerator (their
revenue) and a denominator (their number of sessions). The metric of an
arm is the ratio of the sums, and its variance comes from the delta
method on the user totals (Deng, Knoblich and Lu 2018). The confidence
sequence is the asymptotic one of section 3.

Here 20,000 users have Poisson(2) sessions, so some have none. Each user
has a spend level that all their sessions share, and treatment raises
spend per session by 5%:

```python
from sequential_analysis.procedures import RatioABTest

rnd = random.Random(11)
session_stream, session_spend = [], []
for user in range(20_000):
    arm = int(rnd.random() < 0.5)
    level = math.exp(rnd.gauss(0.0, 1.0)) * (1.05 if arm else 1.0)  # the user's spend per session
    k, p, limit = 0, rnd.random(), math.exp(-2.0)  # Poisson(2) sessions
    while p > limit:
        k += 1
        p *= rnd.random()
    spend = [level * math.exp(rnd.gauss(0.0, 0.5)) for _ in range(k)]
    session_spend += [(arm, s) for s in spend]
    # a user with no session stays in: dropping them would select on an outcome of the treatment
    session_stream += [Observation(n, v, seq=user, key=user)
                       for n, v in (("arm", arm), ("revenue", sum(spend)), ("sessions", k))]

per_session = Run(RatioABTest(input="revenue", denominator_input="sessions"),
                  topology=KeyJoin(("arm", "revenue", "sessions")))
seen = Recorder()
per_session.consumers.append(seen)
run_sync(per_session, session_stream)
last = seen.outputs[-1]
print(f"revenue per session: treatment {last.ratios[0]:.3f}, control {last.ratios[1]:.3f}")
print(f"difference {last.difference:+.3f}, 95% confidence sequence [{last.lower:+.3f}, {last.upper:+.3f}]")
print(f"relative lift {last.lift:+.1%}, 95% confidence sequence [{last.lift_lower:+.1%}, {last.lift_upper:+.1%}]")
```

The 5% lift is not yet detected after 20,000 users of so noisy a
metric: both sequences still contain zero.

The relative lift has its own confidence sequence. It is computed on
the log scale, so it is not symmetric about the estimate. Both
sequences are reported at every step. The decision follows the
difference by default; with `scale="relative"` it follows the lift.
`effect0` sets a null value other than zero.

Compare the standard errors with the naive one, which treats every
session as an independent observation:

```python
def per_session_standard_error(arm):
    values = [s for a, s in session_spend if a == arm]
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values))

naive = math.hypot(per_session_standard_error(1), per_session_standard_error(0))
print(f"standard error of the difference: per user {last.std_error:.4f}, per session (naive) {naive:.4f}")
```

The naive standard error is smaller, by a factor of about 1.6 here.
The same data give the same estimate, but the naive interval is too
narrow. The module docstring reports simulations of this kind of data,
monitored after every user from the thousandth on, with no effect. The
naive per-session test rejected in 24% to 26% of the runs.
`RatioABTest` rejected in 1.4% to 3.5% of them.

**Relative lift of a mean.** Without a denominator (`denominator=False`)
every user's denominator is 1. The ratio is then the mean per user, so
the same test gives the relative lift of revenue per user, which is how
results are often reported:

```python
per_user = Run(RatioABTest(denominator=False, scale="relative", input="revenue"),
               topology=KeyJoin(("arm", "revenue")))
seen = Recorder()
per_user.consumers.append(seen)
run_sync(per_user, (o for o in stream if o.input in ("arm", "revenue")))
last = seen.outputs[-1]
print(f"revenue per user: {last.lift:+.1%}, 95% confidence sequence "
      f"[{last.lift_lower:+.1%}, {last.lift_upper:+.1%}], decision {last.decision}")
```

Like `ABTest`, these intervals are asymptotic. Nothing is reported
before `start` users (1,000 by default). With unbalanced arms and heavy
tails, monitoring from the first few dozen users misses more often than
`alpha`, and the lift misses more than the difference. The module
docstring has the measured rates.

## 9. Before the experiment: how many users?

A sequential test has no fixed sample size, but its distribution can be
planned. `planning.simulate` runs any procedure on simulated users and
reports how often it decides within a budget, and how many users it
needed:

```python
from sequential_analysis.planning import conversion_units, simulate

for lift in (0.12, 0.13, 0.15):
    plan = simulate(TwoProportionTest, conversion_units(lift, 0.10), runs=60, max_n=40_000, seed=1)
    print(f"10% -> {lift:.0%}: decides within 40,000 users in {plan.power:.0%} of runs, "
          f"median {plan.median_n:,.0f}, 90th percentile {plan.quantile(0.9) or float('nan'):,.0f}")
```

Plan with the effect you would be sorry to miss. The 90th percentile, not
the median, is what to budget for.

## Running it for real

* **Live events.** Feed the run from your event log as it grows, with
  `from_kafka` or `from_sql` (see [data sources](../adapters.md)).
* **Restarts.** Checkpoint after every batch. A restarted service
  restores the run and asks the source for the events after the
  checkpointed position, and nothing is counted twice.
* **Stopping.** Stop when the test decides or when the budget runs out.
  The guarantees hold either way.
