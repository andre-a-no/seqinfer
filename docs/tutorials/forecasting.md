# Forecasting: fixed and continuous training

A forecaster here is a procedure like any other. It takes one value at a
time and, after each, gives the forecast of the next 1..`horizon` values:
a location, a scale, a distribution (normal or Student t) and a central
interval. Before that, it scores the new value against the forecast it
made for it. `error` is the miss and `log_score` the log predictive
density, and their running sum `log_likelihood` is a fair score for
comparing forecasters on the same data.

There are two ways to train one:

* **Fixed training.** Estimate the parameters from a history (`fit_ets`,
  `fit_ar`), then forecast with them held fixed. The state (level, trend,
  season, recent values) still moves with every value.
* **Continuous training.** Learn the parameters from the stream itself.
  `AutoRegressive` without coefficients is Bayesian regression on the
  lags. `averaged_smoothing` weights a grid of exponential smoothing
  models by how well each has forecast so far. Both can forget the
  distant past, so they follow a change.

This tutorial uses both on daily demand, then puts distribution-free
intervals around a forecaster (`ConformalForecast`) and decides which of
two forecasters is better while monitoring (`ForecastComparison`).

## The data

Demand has a wandering level with a slight trend, a weekly pattern and
heavy-tailed noise (Student t with 3 degrees of freedom). On day 400
something changes: the weekly pattern is different and the level wanders
three times as fast.

```python
import math
import random


def demand(n, seed, change_at=None):
    """Daily demand; from day `change_at` a new weekly pattern and a faster-moving level."""
    rnd = random.Random(seed)
    week, new_week = (0.0, 4.0, 6.0, 5.0, 8.0, 18.0, -41.0), (-20.0, 0.0, 5.0, 5.0, 10.0, 25.0, -25.0)
    walk, out = 0.0, []
    for day in range(n):
        changed = change_at is not None and day >= change_at
        walk += rnd.gauss(0.0, 3.0 if changed else 1.0)
        noise = rnd.gauss(0.0, 5.0) / math.sqrt(rnd.gammavariate(1.5, 1 / 1.5))  # Student t, 3 df
        out.append(round(100.0 + 0.05 * day + walk + (new_week if changed else week)[day % 7] + noise, 1))
    return out


series = demand(1460, seed=3, change_at=400)
history = series[:365]
```

## 1. Fixed training

`fit_ets` fits exponential smoothing with an additive trend and a weekly
season by maximum likelihood. It returns the model, its parameters and
its state at the end of the history, ready to forecast from:

```python
from sequential_analysis.forecasting import fit_ets

fit = fit_ets(history, trend="additive", period=7, horizon=7)
print({k: round(v, 3) for k, v in fit.parameters.items() if k in ("alpha", "beta", "gamma", "sigma")})
print("weekly effects:", [round(s, 1) for s in fit.parameters["initial_seasonal"]])

model = fit.model
forecast = model.predictive(fit.state)
lower, upper = model.interval(forecast)
for h in range(7):
    print(f"day {365 + h}: {forecast.mean[h]:6.1f}   90% interval [{lower[h]:6.1f}, {upper[h]:6.1f}]")
```

The forecasts and their intervals are those of statsmodels' `ETSModel`
with the same parameters (checked in the tests to 1e-9), and `fit_ets`
reaches at least statsmodels' maximum likelihood. The best initial level,
trend and season are found exactly for each set of smoothing parameters,
by least squares.

To continue live, run the model from that state, for instance with
`Run(model, initial_state=fit.state)`. The level, trend and season keep
updating with every value, but the smoothing parameters and sigma stay
those of the history.

## 2. Continuous training

`averaged_smoothing` builds 63 exponential smoothing models, one for each
combination of 7 values of alpha, 3 of beta and 3 of gamma. Each learns
its own noise level. They are weighted by how well each has predicted
every value so far: with the default `forgetting=1` the weight is the
exact posterior probability of the model. With `forgetting` below 1 an
old record fades, so the weight can move to a model that has become
better. `AutoRegressive(8, forgetting=0.99)` learns an AR(8) by
exponentially weighted regression on the last eight values.

None of these see the history before they start: they learn from day 0,
on the same stream. Below, every forecaster runs over the whole series,
and its mean absolute error is measured on three stretches: before the
change, its first 50 days, and after them.

```python
from sequential_analysis.procedures import AutoRegressive, averaged_smoothing


def one_step_errors(forecaster, ys):
    """|y - forecast| for each value, None while the forecaster cannot forecast yet."""
    state, errors, forecast = forecaster.initial_state(), [], None
    for y in ys:
        errors.append(None if forecast is None else abs(y - forecast))
        state, out = forecaster.step(state, {"y": y})
        forecast = out.mean[0] if out.mean else None
    return errors


def mean_error(errors, start, stop):
    values = [e for e in errors[start:stop] if e is not None]
    return sum(values) / len(values)


forecasters = {
    "fixed (fitted on days 0-364)": fit_ets(history, trend="additive", period=7).model,
    "learnt smoothing": averaged_smoothing(trend="additive", period=7),
    "learnt smoothing, forgetting 0.98": averaged_smoothing(trend="additive", period=7, forgetting=0.98),
    "learnt AR(8), forgetting 0.99": AutoRegressive(8, forgetting=0.99),
}
errors = {name: one_step_errors(f, series) for name, f in forecasters.items()}
print(f"{'':36} days 365-399  400-449  450-1459")
for name, e in errors.items():
    print(f"{name:36} {mean_error(e, 365, 400):11.2f} {mean_error(e, 400, 450):8.2f} {mean_error(e, 450, 1460):9.2f}")
```

Before the change, the fixed model is best (6.5 against 7.7 to 11.4): a
year of history estimates its parameters better than any grid. After
the change it is worst among the smoothers (11.4 against 8.0 to 8.2),
because its seasonal smoothing gamma was fitted on a stable weekly
pattern and is too small to learn a new one. The learnt models picked
larger smoothing parameters as soon as the data asked for them. The
AR(8) has no notion of a season or a trend and is the weakest here.

Neither kind always wins. A fixed model is the right choice while the
process stays as it was. Continuous training costs a little accuracy
while nothing changes and pays when something does. It also needs no
history to start.

## 3. Intervals that hold without a model

A model's intervals are right only if the model is. After the change
the fixed model's 90% intervals hold far less often. `ConformalForecast`
puts its own intervals around any forecaster. For each horizon it keeps
a half width, in units of the forecaster's scale, and nudges it after
every value: up by `step_size × 0.9` after a miss, down by
`step_size × 0.1` after a hit. Whatever the data,

    fraction missed - alpha = (half width now - at start) / (step_size × number judged),

and the half width stays within `step_size` of the range of the scores,
so the fraction missed tends to alpha. This holds for any data and any
forecaster, with no distributional assumption. It is a long-run
guarantee, not a probability for each interval:

```python
from sequential_analysis.procedures import ConformalForecast


def held(forecaster, ys, start, stop):
    """Fraction of one-step intervals issued for days start..stop-1 that held."""
    state, issued, hits, total = forecaster.initial_state(), None, 0, 0
    for day, y in enumerate(ys[:stop]):
        if issued is not None and day >= start:
            hits += issued[0] <= y <= issued[1]
            total += 1
        state, out = forecaster.step(state, {"y": y})
        issued = (out.lower[0], out.upper[0]) if out.lower else None
    return hits / total


fixed = fit_ets(history, trend="additive", period=7).model
conformal = ConformalForecast(fixed, alpha=0.1)
for name, f in (("the model's intervals", fixed), ("conformal intervals", conformal)):
    print(f"{name:22} held {held(f, series, 365, 400):.0%} before the change, "
          f"{held(f, series, 400, 1460):.0%} after it")
```

Before the change both hold 91% of the time. After it the model's own
intervals hold 61%: they are as wide as on the stable process. The
conformal ones hold 86%, because they widened. They needed some time to
do so: with `step_size=0.05` the width moves slowly, and a larger step
adapts faster at the price of more jitter.

## 4. Which forecaster is better, while monitoring

After the change, is the learnt forecaster really better than the fixed
one, or lucky? `ForecastComparison` takes the two forecasters' losses on
each value, here absolute errors clipped at 20. It keeps a confidence
sequence for the difference in expected loss, averaged over the days so
far, that is valid at every day at once (Choe and Ramdas 2024). So it can
be read after every value, and a decision taken as soon as the sequence
excludes zero, without inflating the error rate. No stationarity is
assumed: the difference may drift.

```python
from sequential_analysis.procedures import ForecastComparison

comparison = ForecastComparison((0.0, 20.0))
state = comparison.initial_state()
learnt, fixed_errors = errors["learnt smoothing, forgetting 0.98"], errors["fixed (fitted on days 0-364)"]
decided = None
for day in range(365, 1460):
    state, verdict = comparison.step(state, {"a": min(20.0, learnt[day]), "b": min(20.0, fixed_errors[day])})
    if verdict.better and decided is None:
        decided = day, verdict
day, verdict = decided
print(f"decided on day {day}: {verdict.better!r} is better, "
      f"difference in mean absolute error in [{verdict.lower:.2f}, {verdict.upper:.2f}]")
```

It decides on day 960, after 596 days of monitoring: the learnt
forecaster's mean absolute error is lower, by between 0.0 and 3.7.
Monitoring from day 365 includes the 35 days before the change, when the
fixed model was better. A guarantee that holds at every time costs
data. The bound needs only the range of the losses, so a tighter
`bounds` (a smaller cap on the errors) gives a faster decision, as long
as the losses truly stay within it.

## Choosing

* `ExponentialSmoothing` covers level, trend (additive or damped) and an
  additive season. It is fitted with `fit_ets` or learnt with
  `averaged_smoothing`.
* `AutoRegressive` is for series driven by their own recent values. It is
  fitted with `fit_ar` or learnt with `forgetting`.
* `ModelAveraging` combines any of these, for instance a smoother and an
  autoregression, and lets the data choose.
* `ConformalForecast` is for when the intervals must hold whatever the
  data, and `ForecastComparison` for choosing between forecasters with a
  guarantee.

Not covered: multiplicative seasons, exogenous regressors, and ARIMA
moving-average terms. The last two can be had from `KalmanFilter` with a
model written as a state space.
