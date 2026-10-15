# Process monitoring and change detection

A machine, a service or a sensor runs in a steady state until something
changes. The task is to raise an alarm soon after a change, and rarely
when nothing has changed. The second requirement is measured by the
*average run length* (ARL) to a false alarm.

This tutorial compares four ways to do it:

* CUSUM, when both distributions are known;
* an e-detector, when only the mean or a quantile before the change is
  known;
* Bayesian online change points, when the parameters are unknown on both
  sides;
* a Kalman filter for a slowly drifting level, smoothed afterwards to
  see what happened.

## The data

Response times of a service:

* 400 requests from a skewed distribution with mean 0.2 s;
* then a regression: the mean rises to 0.3 s.

The values are clipped to [0, 1], which is also what makes the
nonparametric detector applicable.

```python
import random

rnd = random.Random(4)
before = [min(1.0, rnd.gammavariate(2.0, 0.1)) for _ in range(400)]
after = [min(1.0, rnd.gammavariate(2.0, 0.15)) for _ in range(400)]
latency = before + after
```

## 1. CUSUM, when both distributions are known

CUSUM is optimal when the densities before and after the change are
known, here taken to be normal. Its threshold is chosen for an ARL of
about 1000:

```python
from sequential_analysis import Recorder, Run, run_sync
from sequential_analysis.procedures import CUSUM, Gaussian
from sequential_analysis.sources import from_values

cusum = Run(CUSUM(Gaussian(sigma=0.14), theta0=0.2, theta1=0.3, threshold=6.9))
run_sync(cusum, from_values("x", latency))
print("CUSUM alarm at request", cusum.t)
```

Here it raises its alarm long before request 400: a false alarm. The
planned ARL holds only if the data really are normal with that variance.
These data are skewed, and their long right tail pushes the statistic up
far more often than a normal tail would.

## 2. An e-detector, when only the mean is known

`MeanEDetector` needs only two facts about the steady state:

* the values lie in [0, 1];
* their mean is 0.2.

Its ARL is at least `arl` *for every distribution* with that mean:

```python
from sequential_analysis.procedures import MeanEDetector

detector = Run(MeanEDetector(m=0.2, lo=0.0, hi=1.0, arl=1000, direction="increase", statistic="cusum"))
seen = Recorder()
detector.consumers.append(seen)
run_sync(detector, from_values("x", latency))
alarm = seen.outputs[-1]
print("e-detector alarm at request", alarm.n, "-- change estimated to start at request", alarm.change_estimate)
```

For data with no useful bound, use `QuantileEDetector`. It monitors a
quantile, such as the median or the 95th percentile, and assumes nothing
else. Its guarantee holds for Cauchy data too:

```python
from sequential_analysis.procedures import QuantileEDetector

p95 = Run(QuantileEDetector(q0=0.4744, p=0.95, arl=1000, direction="increase", statistic="cusum"))
run_sync(p95, from_values("x", latency))
print("95th-percentile detector alarm at request", p95.t)
```

Both detectors here use the CUSUM-type statistic. The Shiryaev–Roberts
type, the default, has the same guarantee E[τ] ≥ `arl`, but its run
length to a false alarm is far more spread out.

On these steady-state data, with `arl=1000`, it raised a false alarm
within the first 400 requests in 14% of simulated runs. The CUSUM type
did so in 0.3% to 0.7%. In exchange, the CUSUM type reacts somewhat later
to a real change.

## 3. Bayesian change points, when nothing is known

`BayesianChangePoint` puts a prior on the parameters of every segment
and computes the posterior of the time since the last change:

```python
from sequential_analysis.procedures import BayesianChangePoint, NormalGammaModel

prior = NormalGammaModel(mu0=0.2, kappa0=1.0, alpha0=2.0, beta0=0.04)  # a mean near 0.2, a variance near 0.02
bocpd = Run(BayesianChangePoint(prior, hazard=1 / 500, alarm_probability=0.9, alarm_window=50))
posterior = Recorder()
bocpd.consumers.append(posterior)
run_sync(bocpd, from_values("x", latency))
first = next(o for o in posterior.outputs if o.decision)
print("Bayesian alarm at request", first.n, "-- most probable run length", first.most_likely_run_length)
```

It needs no knowledge of either segment, and gives a full posterior over
where the change was. In exchange, it has no frequentist ARL guarantee.

Here it alarms at about request 200, a false alarm. Its model says the
data are normal within a segment. These are skewed, and a cluster of
large values looks more like a new segment than like the tail of the old
one. The posterior is only as good as the model, which is the
difference from the e-detector above.

## 4. A drifting level: filter now, smooth later

Some processes do not jump; they drift. A local-linear-trend Kalman
filter tracks the level and its slope as the data arrive. Afterwards the
Rauch–Tung–Striebel smoother re-estimates every past level from all the
data:

```python
from sequential_analysis.procedures import KalmanFilter
from sequential_analysis.smoothing import kalman_smoother

rnd = random.Random(5)
level = [0.5 * (1 + (t > 150) * (t - 150) / 100) for t in range(300)]
readings = [v + rnd.gauss(0.0, 0.1) for v in level]

model = KalmanFilter.local_linear_trend(level_noise=1e-5, slope_noise=1e-6, observation_noise=0.01)
smoothed = kalman_smoother(model, [{"y": y} for y in readings])

def error(estimates):
    return (sum((m[0] - v) ** 2 for m, v in zip(estimates, level)) / len(level)) ** 0.5

print(f"filtered error {error(smoothed.filtered_means):.4f}, smoothed error {error(smoothed.means):.4f}")
print(f"slope at the end: {smoothed.means[-1][1] * 100:.3f} per 100 readings")
```

The smoothed estimates use the future as well as the past, so they are
more accurate. They answer the question asked afterwards: when did the
drift start, and how fast was it?

For regimes rather than levels, use these, from
`sequential_analysis.smoothing`:

* the `HiddenMarkovFilter` while the data arrive;
* `forward_backward` and `viterbi` afterwards.

## Choosing

| You know | Use | Guarantee |
|---|---|---|
| both distributions | `CUSUM`, `ShiryaevRoberts` | optimal; ARL by design, if the model holds |
| a bound and the mean, or a quantile, before the change | `MeanEDetector`, `QuantileEDetector` | ARL ≥ `arl` for every such distribution |
| nothing | `BayesianChangePoint` | a posterior; no ARL guarantee |
| a model of how the state moves | `KalmanFilter`, `ParticleFilter`, `HiddenMarkovFilter` | exact or consistent filtering; smoothing afterwards |
