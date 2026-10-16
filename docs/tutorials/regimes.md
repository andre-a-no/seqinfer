# Regimes: hidden Markov models

Many processes switch between a few regimes. A market can be calm or
volatile, a machine healthy or wearing out, a patient in remission or in
relapse. The regime itself is not observed: only its effect on the data
is. A hidden Markov model describes this. The regime z_t moves from one
step to the next by a Markov chain. The observation y_t depends only on
the regime at that step.

This tutorial goes from a recorded history to live monitoring:

1. fit the model to the history (`fit_hmm`);
2. follow the regime as new data arrive (`HiddenMarkovFilter`), with an
   alarm;
3. reconstruct the regimes afterwards (`forward_backward`, `viterbi`);
4. learn the model on the stream itself, without a history (`OnlineHMM`).

## The data

Daily returns alternate between a calm regime (standard deviation 0.6)
and a volatile one (2.0). Calm spells last 50 days on average, volatile
ones 20:

```python
import random

def returns(n, seed):
    """Simulated daily returns and the true regime of each day (0 calm, 1 volatile)."""
    rnd = random.Random(seed)
    stay, sds, means = (0.98, 0.95), (0.6, 2.0), (0.05, -0.1)
    z, ys, regimes = 0, [], []
    for _ in range(n):
        if rnd.random() > stay[z]:
            z = 1 - z
        ys.append(round(rnd.gauss(means[z], sds[z]), 4))
        regimes.append(z)
    return ys, regimes

history, _ = returns(2000, seed=1)
```

## 1. Fit the model to the history

`fit_hmm` estimates the transition matrix and the emission parameters by
maximum likelihood (the Baum–Welch algorithm). It starts from a rough
guess, which fixes the number of regimes, the emission family and the
labels. Here the guess is "a quiet regime and a wild one". EM finds the
nearest local maximum, so `restarts` tries further random starting points
and keeps the best fit:

```python
from sequential_analysis.hmm_fit import fit_hmm
from sequential_analysis.procedures import GaussianEmission, HiddenMarkovFilter

guess = HiddenMarkovFilter(
    initial=[0.5, 0.5],
    transition=[[0.9, 0.1], [0.1, 0.9]],
    emission=GaussianEmission(means=[0.0, 0.0], sds=[0.5, 2.5]),
)
fit = fit_hmm([{"y": y} for y in history], guess, restarts=3, seed=7)
model = fit.model
print(f"{fit.iterations} iterations, log-likelihood {fit.log_likelihood:.1f}")
print("transition matrix:", [[round(p, 3) for p in row] for row in model.transition])
print("standard deviations:", [round(s, 2) for s in model.emission.sds])
print(f"expected spell lengths: calm {1 / model.transition[0][1]:.0f} days, "
      f"volatile {1 / model.transition[1][0]:.0f} days")
```

The labels follow the guess: regime 0 is the one whose parameters are
nearest the guess's quiet regime. One EM step agrees with hmmlearn to
1e-10, and the log-likelihood never decreases from one iteration to the
next. Both are checked in the tests.

## 2. Follow the regime live

The fitted model is an ordinary procedure. It takes one return at a time
and gives the probability of each regime given everything so far. An
alarm with hysteresis (enter above 0.9, leave below 0.1) avoids
flickering:

```python
live, truth = returns(300, seed=2)
state, volatile, switches = model.initial_state(), False, []
for day, y in enumerate(live):
    state, out = model.step(state, {"y": y})
    p = out.probabilities[1]
    if not volatile and p > 0.9 or volatile and p < 0.1:
        volatile = not volatile
        switches.append((day, "volatile" if volatile else "calm"))
print("alarms:", switches[:6])
true_switches = [day for day in range(1, len(truth)) if truth[day] != truth[day - 1]]
print("true switches:", true_switches[:6])
```

Each alarm comes a few days after the true switch: the filter needs a
few large (or small) returns before it is confident. A checkpoint of the
run holds the regime probabilities and the log-likelihood, so monitoring
continues across restarts like any other run.

## 3. Afterwards: smoothing and the most likely path

Once a period is over, every day can be judged with the data after it
too. `forward_backward` gives the smoothed probabilities, and `viterbi`
the single most probable sequence of regimes:

```python
from sequential_analysis.smoothing import forward_backward, viterbi

records = [{"y": y} for y in live]
smoothed = forward_backward(model, records)
path = viterbi(model, records)
accuracy = sum(a == b for a, b in zip(path.states, truth)) / len(truth)
print(f"Viterbi path matches the true regime on {accuracy:.0%} of the days")
```

## 4. No history: learn on the stream

Without a recorded history, `OnlineHMM` filters and estimates at the same
time (online EM, Cappé 2011). It keeps a constant-size summary of the
expected sufficient statistics, updates it with every observation, and
re-estimates the parameters from it after a burn-in. The estimates
approach those of the batch fit as data accumulate:

```python
from sequential_analysis.procedures import OnlineHMM

learner = OnlineHMM(guess)
state = learner.initial_state()
for y in history:
    state, out = learner.step(state, {"y": y})
print("online transition matrix:", [[round(p, 3) for p in row] for row in out.transition])
print("online standard deviations:", [round(s, 2) for s in out.emission["sds"]])
```

The tests check two things. With step sizes 1/t and the parameters held
fixed, its statistics are exactly the batch E-step's averages. On long
series, its estimates are within a few hundredths of the batch
maximum-likelihood ones.

Every output also carries the prequential log-likelihood: the sum of
log p(y_t | y_1..y_{t-1}), each term under the parameters learnt before
y_t. It is a fair score for comparing learners.

## Choosing the emissions

* `GaussianEmission` is for one real value per step.
* `PoissonEmission` is for counts.
* `CategoricalEmission` is for symbols.

A regime can only be told apart by what the emission model sees. Two
regimes with the same mean and different variances, as here, need the
variance in the model, which the Gaussian has.
