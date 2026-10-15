# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Plots of runs, designs and plans, with matplotlib.

    pip install "sequential-analysis[plot]"

Every function draws on a matplotlib Axes and returns it: pass `ax=` to
draw into a figure of your own, or leave it out for a new figure.  None
of them calls `plt.show()` or saves anything; that is up to the caller
(`ax.figure.savefig("run.png")`).

plot_e_process              evidence against H0 (e-value, log scale), 1/alpha and the stop
plot_confidence_sequence    the confidence sequence as a band, with the estimate
plot_group_sequential_design  efficacy and futility boundaries, and an observed path
plot_detection              a change detector's statistic, its threshold and the alarm
plot_filter                 a filter's mean with a +-2 sd band (and a smoother's)
plot_stopping_times         the distribution of stopping times from planning.simulate

The outputs of a run are accepted in any of the forms the library
produces them: a `Recorder`, a list of `OutputEvent`s, a list of
outputs, or the (state, output) pairs of `trajectory`.

matplotlib is an optional dependency, imported only when a function is
called: importing this module works without it.
"""
from __future__ import annotations

import math
import typing
from collections.abc import Sequence
from typing import Any

__all__ = [
    "plot_confidence_sequence",
    "plot_detection",
    "plot_e_process",
    "plot_filter",
    "plot_group_sequential_design",
    "plot_stopping_times",
]

# One palette for every plot: the estimate or statistic in blue, a second
# series (futility, smoother) in orange, events (a stop, an alarm) in red,
# reference lines (thresholds, the truth) in neutral ink.
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_RED = "#e34948"
_INK = "#3d3c38"
_GREY = "#8a8984"
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

# Field names under which the procedures report the log of their e-value.
_E_FIELDS = ("log_e_value", "log_lambda", "log_bf", "log_evidence", "log_wealth")
_ESTIMATE_FIELDS = ("estimate", "mean", "rate_ratio", "hazard_ratio_estimate", "difference", "estimates")
_COUNTERS = (("n", "observations"), ("units", "units"), ("n_events", "events"), ("t", "step"))
_LN10 = math.log(10.0)

Number = typing.Union[int, float]  # typing: evaluated at import, also on Python 3.8


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found, unused-ignore]  # optional
    except ImportError as error:
        raise ImportError(
            'sequential_analysis.plotting needs matplotlib: pip install "sequential-analysis[plot]"'
        ) from error
    return plt


def _axes(ax: Any) -> Any:
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    return ax


# ------------------------------------------------------------------ inputs


def _outputs(data: Any) -> tuple[list[float], list[Any], str]:
    """(x, outputs, x label) from a Recorder, OutputEvents, outputs or (state, output) pairs.

    x is the procedure's own counter (n, units, n_events, t) when every
    output has one, else the run's step t of the events, else 1, 2, ...
    """
    events = getattr(data, "events", None)
    items = list(events) if events is not None and not isinstance(data, (list, tuple)) else list(data)
    steps: list[float] = []
    outputs: list[Any] = []
    for i, item in enumerate(items, start=1):
        step: float = i
        if hasattr(item, "output") and hasattr(item, "terminal"):  # an OutputEvent
            step, item = item.t, item.output
        elif isinstance(item, tuple) and len(item) == 2:  # (state, output) from trajectory
            item = item[1]
        if item is None:
            continue
        steps.append(step)
        outputs.append(item)
    if not outputs:
        raise ValueError("nothing to plot: no outputs")
    for name, label in _COUNTERS:
        values = [getattr(o, name, None) for o in outputs]
        if all(type(v) is int for v in values):
            return [float(v) for v in values], outputs, label  # type: ignore[arg-type]
    return steps, outputs, "step"


def _number(value: Any) -> float:
    """A float, NaN for None or a value that is not a finite number (matplotlib leaves a gap)."""
    if isinstance(value, (tuple, list)) and len(value) == 1:  # a one-coefficient estimate (AnytimeLinearModel)
        value = value[0]
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _field(outputs: list[Any], field: str | None, candidates: Sequence[str], what: str) -> str:
    if field is not None:
        if not all(hasattr(o, field) for o in outputs):
            raise ValueError(f"the outputs have no field {field!r}")
        return field
    for name in candidates:
        if all(hasattr(o, name) for o in outputs):
            return name
    kind = type(outputs[0]).__name__
    raise ValueError(f"{kind} has no {what} field (looked for {', '.join(candidates)}); name one with field=...")


def _first_stop(xs: list[float], outputs: list[Any]) -> tuple[float, Any, str] | None:
    """(x, output, what) of the first output with a decision or an alarm."""
    for x, o in zip(xs, outputs):
        if getattr(o, "alarm", False) is True:
            return x, o, "alarm"
        decision = getattr(o, "decision", None)
        if decision:
            return x, o, str(decision)
    return None


def _power_of_ten_axis(axis: Any) -> None:
    """Ticks at integer log10 values, labelled as the numbers themselves: 0.1, 1, 10, 100, 10^4."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator  # type: ignore[import-not-found, unused-ignore]

    def label(value: float, _position: Any) -> str:
        k = round(value)
        if abs(value - k) > 1e-9:
            return ""
        if -2 <= k <= 3:
            return f"{10.0 ** k:g}"
        return f"$10^{{{k}}}$"

    axis.set_major_locator(MaxNLocator(integer=True))
    axis.set_major_formatter(FuncFormatter(label))


def _style(ax: Any, xlabel: str | None, ylabel: str | None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend(ax: Any, **kwargs: Any) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, **kwargs)


_LABEL_BOX = {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}


def _hline(ax: Any, y: float, label: str, style: str = "--", *, direct: bool = True) -> None:
    """A horizontal reference line, labelled at its left end (or in the legend: direct=False)."""
    if not direct:
        ax.axhline(y, color=_INK, linestyle=style, linewidth=1.2, label=label)
        return
    ax.axhline(y, color=_INK, linestyle=style, linewidth=1.2)
    ax.annotate(label, xy=(0.0, y), xycoords=ax.get_yaxis_transform(), xytext=(4, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize="small", color=_INK, bbox=_LABEL_BOX)


def _vline(ax: Any, x: float, label: str, style: str = ":", color: str = _INK, *, left: bool = False) -> None:
    """A vertical reference line, labelled along it at the top (on its left side with left=True)."""
    ax.axvline(x, color=color, linestyle=style, linewidth=1.2)
    ax.annotate(label, xy=(x, 1.0), xycoords=ax.get_xaxis_transform(), xytext=(-3 if left else 3, -4),
                textcoords="offset points", rotation=90, ha="right" if left else "left", va="top",
                fontsize="small", color=_INK, bbox=_LABEL_BOX)


def _mark(ax: Any, x: float, y: float, label: str) -> None:
    """An event (a stop, an alarm) as a red dot, named in the legend."""
    ax.plot([x], [y], marker="o", markersize=8, color=_RED, markeredgecolor="white", markeredgewidth=1.5,
            linestyle="none", label=label, zorder=5)


def _fmt(x: float) -> str:
    return f"{x:g}" if x != int(x) else str(int(x))


def _band_limits(ax: Any, lower: list[float], upper: list[float], always: list[float]) -> None:
    """Keep a band's first, very wide steps from flattening the rest of the plot.

    The y range covers `always` (estimate, observations, truth) and the band
    from the first step at which it is at most 20 times as wide as at its
    narrowest; earlier steps are drawn but run off the axes.  Applied only
    when it shrinks the range at least three times.
    """
    widths = [u - lo for lo, u in zip(lower, upper)]
    finite = [w for w in widths if math.isfinite(w)]
    edges = [v for v in lower + upper if math.isfinite(v)]
    centre = [v for v in always if math.isfinite(v)]
    if not finite or not edges:
        return
    narrow = max(min(finite), 1e-300) * 20.0
    start = next(i for i, w in enumerate(widths) if math.isfinite(w) and w <= narrow)
    kept = [v for v in lower[start:] + upper[start:] if math.isfinite(v)] + centre
    lo, hi = min(kept), max(kept)
    full_lo, full_hi = min(edges + centre), max(edges + centre)
    if hi > lo and (full_hi - full_lo) > 3.0 * (hi - lo):
        margin = 0.06 * (hi - lo)
        ax.set_ylim(lo - margin, hi + margin)


# --------------------------------------------------------------- e-process


def plot_e_process(outputs: Any, *, alpha: float | None = None, field: str | None = None, ax: Any = None) -> Any:
    """The e-value of a test against the number of observations, on a log scale.

    The e-value is exp of the output's log e-value field, found
    automatically (log_e_value, log_lambda, log_bf, log_evidence,
    log_wealth) or named by `field`.  With `alpha` the rejection threshold
    1/alpha is drawn; the first output with a decision is marked.

    For `MultipleTests` every test's e-value is drawn (or only the test
    named by `field`), with the e-BH thresholds K/alpha (the first
    rejection) and 1/alpha (all K), and each test marked where e-BH first
    rejects it.

    The y axis is log10 of the e-value (ticks labelled 1, 10, 100, ...),
    so that runs of any length stay readable without overflow.
    """
    ax = _axes(ax)
    xs, outs, xlabel = _outputs(outputs)
    if alpha is not None and not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if all(isinstance(getattr(o, "log_e_values", None), dict) for o in outs):
        _plot_multiple(ax, xs, outs, alpha, field)
    else:
        name = _field(outs, field, _E_FIELDS, "log e-value")
        ys = [_number(getattr(o, name)) / _LN10 for o in outs]
        ax.plot(xs, ys, color=_BLUE, linewidth=2.0)
        if alpha is not None:
            _hline(ax, -math.log10(alpha), f"1/α = {_fmt(1 / alpha)}")
        stop = _first_stop(xs, outs)
        if stop is not None:
            x, o, what = stop
            _mark(ax, x, _number(getattr(o, name)) / _LN10, f"{what} at {_fmt(x)}")
    _power_of_ten_axis(ax.yaxis)
    _style(ax, xlabel, "e-value (log scale)")
    _legend(ax, loc="best")
    return ax


def _plot_multiple(ax: Any, xs: list[float], outs: list[Any], alpha: float | None, field: str | None) -> None:
    names = list(outs[0].log_e_values)
    if field is not None:
        if field not in names:
            raise ValueError(f"no test named {field!r}; the tests are {', '.join(names)}")
        names = [field]
    for i, name in enumerate(names):
        color = _SERIES[i % len(_SERIES)]
        ys = [_number(o.log_e_values.get(name)) / _LN10 for o in outs]
        ax.plot(xs, ys, color=color, linewidth=1.8, label=name)
        for x, y, o in zip(xs, ys, outs):
            if name in getattr(o, "rejected", ()):
                ax.plot([x], [y], marker="o", markersize=8, color=color, markeredgecolor="white",
                        markeredgewidth=1.5, linestyle="none", zorder=5)
                break
    if alpha is not None:
        k = len(outs[0].log_e_values)
        if k > 1:
            _hline(ax, math.log10(k / alpha), f"K/α = {_fmt(k / alpha)} (e-BH, first rejection)", ":")
        _hline(ax, -math.log10(alpha), f"1/α = {_fmt(1 / alpha)}")


# ---------------------------------------------------- confidence sequence


def plot_confidence_sequence(outputs: Any, *, estimate: Any = None, truth: float | None = None,
                             ax: Any = None) -> Any:
    """The confidence sequence (lower, upper) as a band against the number of observations.

    `estimate`: None finds the estimate field (estimate, mean, rate_ratio,
    hazard_ratio_estimate, difference, or the one coefficient of an
    AnytimeLinearModel's estimates), a str names it, False leaves the line
    out.  `truth` draws the true value as a reference line.

    A bound that is None or infinite (an unbounded start) leaves a gap; the
    band is drawn where both bounds are finite.  The y range leaves out the
    first, very wide part of the band when that would flatten the rest.
    """
    ax = _axes(ax)
    xs, outs, xlabel = _outputs(outputs)
    for name in ("lower", "upper"):
        if not all(hasattr(o, name) for o in outs):
            raise ValueError(f"{type(outs[0]).__name__} has no {name!r} field: not a confidence sequence")
    lower = [_number(o.lower) for o in outs]
    upper = [_number(o.upper) for o in outs]
    both = [math.isfinite(lo) and math.isfinite(u) for lo, u in zip(lower, upper)]
    ax.fill_between(xs, [v if ok else 0.0 for v, ok in zip(lower, both)],
                    [v if ok else 0.0 for v, ok in zip(upper, both)],
                    where=both, color=_BLUE, alpha=0.18, linewidth=0, label="confidence sequence")
    ax.plot(xs, lower, color=_BLUE, linewidth=1.0)
    ax.plot(xs, upper, color=_BLUE, linewidth=1.0)
    centre: list[float] = []
    if estimate is not False:
        if estimate is not None:
            found: str | None = _field(outs, str(estimate), (), "estimate")
        else:
            found = next((f for f in _ESTIMATE_FIELDS if all(hasattr(o, f) for o in outs)), None)
        if found is not None:
            centre = [_number(getattr(o, found)) for o in outs]
            ax.plot(xs, centre, color=_BLUE, linewidth=2.0, label=f"estimate ({found})")
    if truth is not None:
        _hline(ax, float(truth), f"true value {_fmt(float(truth))}", ":", direct=False)
        centre.append(float(truth))
    _band_limits(ax, lower, upper, centre)
    _style(ax, xlabel, None)
    _legend(ax, loc="best")
    return ax


# ----------------------------------------------------- group sequential


def plot_group_sequential_design(design: Any, *, observed: Any = None, scale: str = "z", ax: Any = None) -> Any:
    """Efficacy (and futility) boundaries of a group sequential design against information fraction.

    `design` is a `GroupSequentialDesign`.  `scale`: "z" the standardised
    statistic Z_k (the default), "score" B_k = Z_k sqrt(t_k), "p" the
    nominal p-value 1 - Phi(Z_k) (two-sided: 2 (1 - Phi(|Z_k|))), on a log
    scale.  Infinite boundaries (no alpha spent at an analysis) are left out.

    `observed` adds the path of a trial: a sequence of Z values for the
    first analyses, or the outputs of a `GroupSequentialTest` run (any of
    the forms `plot_e_process` takes), from which the analyses are taken
    and the decision, if any, is marked.  (With an estimated variance the
    run's statistic is a t statistic, drawn against the Z boundaries.)
    The regions where the trial stops are shaded.
    """
    from statistics import NormalDist

    if scale not in ("z", "score", "p"):
        raise ValueError('scale must be "z", "score" or "p"')
    ax = _axes(ax)
    ts = [float(t) for t in design.fractions]
    two_sided = bool(design.two_sided)
    normal = NormalDist()

    def convert(z: float, t: float) -> float:
        if not math.isfinite(z):
            return math.nan
        if scale == "score":
            return z * math.sqrt(t)
        if scale == "p":
            return 2.0 * normal.cdf(-abs(z)) if two_sided else normal.cdf(-z)
        return z

    efficacy = [convert(b, t) for b, t in zip(design.bounds, ts)]
    lines: list[tuple[list[float], str, str | None]] = [(efficacy, _BLUE, "efficacy boundary")]
    if two_sided and scale != "p":
        lines.append(([-v for v in efficacy], _BLUE, None))
    futility = None
    if design.futility is not None:
        futility = [convert(f, t) for f, t in zip(design.futility, ts)]
        lines.append((futility, _ORANGE, "futility boundary"))
    for values, color, label in lines:
        ax.plot(ts, values, color=color, linewidth=2.0, marker="o", markersize=7, label=label)
    if observed is not None:
        path = _observed_path(observed, ts)
        if path:
            ys = [convert(z, t) for t, z, _ in path]
            ax.plot([t for t, _, _ in path], ys, color=_INK, linewidth=1.5, marker="s", markersize=6,
                    label="observed")
            t, _, decision = path[-1]
            if decision:
                _mark(ax, t, ys[-1], f"{decision} at {t:.0%}")
    ax.set_xlim(0.0, 1.05)
    ax.xaxis.set_major_formatter(_percent())
    if scale == "p":
        ax.set_yscale("log")
    elif not two_sided or scale == "score":
        ax.axhline(0.0, color=_GREY, linewidth=0.8)
    # The stopping regions, lightly shaded: beyond the efficacy boundary, below the futility boundary.
    lo, hi = ax.get_ylim()
    strong, weak = (lo, hi) if scale == "p" else (hi, lo)
    ax.fill_between(ts, efficacy, strong, color=_BLUE, alpha=0.08, linewidth=0)
    if two_sided and scale != "p":
        ax.fill_between(ts, [-v for v in efficacy], lo, color=_BLUE, alpha=0.08, linewidth=0)
    if futility is not None:
        ax.fill_between(ts, futility, weak, color=_ORANGE, alpha=0.08, linewidth=0)
    ax.set_ylim(lo, hi)
    if scale == "p" and not two_sided:
        ax.invert_yaxis()  # smaller p-values, like larger Z, at the top
    labels = {"z": "Z statistic", "score": "score B = Z √t",
              "p": "nominal p-value" + (" (two-sided)" if two_sided else " (one-sided)")}
    _style(ax, "information fraction", labels[scale])
    _legend(ax, loc="best")
    return ax


def _percent() -> Any:
    from matplotlib.ticker import FuncFormatter  # type: ignore[import-not-found, unused-ignore]

    return FuncFormatter(lambda v, _position: f"{v:.0%}")


def _observed_path(observed: Any, ts: list[float]) -> list[tuple[float, float, Any]]:
    """(information fraction, Z, decision) of each analysis of a trial."""
    items = list(getattr(observed, "events", observed)) if not isinstance(observed, (list, tuple)) else observed
    if all(isinstance(z, (int, float)) and not isinstance(z, bool) for z in items):
        if len(items) > len(ts):
            raise ValueError(f"{len(items)} observed values for a design with {len(ts)} analyses")
        return [(t, float(z), None) for t, z in zip(ts, items)]
    _, outs, _ = _outputs(items)
    path = []
    for o in outs:
        k, z = getattr(o, "analysis", None), getattr(o, "z", None)
        if k is not None and z is not None:
            if not 1 <= k <= len(ts):
                raise ValueError(f"analysis {k} is not in a design with {len(ts)} analyses")
            path.append((ts[k - 1], float(z), getattr(o, "decision", None)))
    return path


# --------------------------------------------------------------- detection


def plot_detection(outputs: Any, *, threshold: float | None = None, change_point: float | None = None,
                   field: str | None = None, ax: Any = None) -> Any:
    """A change detector's statistic against the number of observations, with the threshold and the alarm.

    CUSUM: W_n and its threshold h on a linear scale.  Shiryaev-Roberts
    (log_statistic) and the e-detectors (statistic with change_estimate):
    R_n or M_t on a log scale, with `threshold` the A or `arl` given to
    the procedure.  BayesianChangePoint: the probability of a change within
    the alarm window, against `alarm_probability`.  `field` names another
    statistic, drawn on a linear scale.

    The first alarm is marked; an e-detector's estimate of where the
    change began is drawn at it, and `change_point` (the true change, when
    known) as a reference line.
    """
    ax = _axes(ax)
    xs, outs, xlabel = _outputs(outputs)
    first = outs[0]
    log_scale = False
    if field is not None:
        name, ylabel = _field(outs, field, (), "statistic"), field
        values = [_number(getattr(o, name)) for o in outs]
    elif hasattr(first, "recent_change_probability"):
        ylabel = "P(change within the alarm window)"
        values = [_number(o.recent_change_probability) for o in outs]
    elif hasattr(first, "log_statistic"):
        ylabel, log_scale = "Shiryaev–Roberts statistic R (log scale)", True
        values = [_number(o.log_statistic) / _LN10 for o in outs]
    elif hasattr(first, "statistic") and hasattr(first, "change_estimate"):
        ylabel, log_scale = "e-detector statistic M (log scale)", True
        values = [math.log10(o.statistic) if o.statistic > 0 else math.nan for o in outs]
    else:
        name = _field(outs, None, ("statistic",), "detection statistic")
        ylabel = "CUSUM statistic W" if hasattr(first, "alarm") else "statistic"
        values = [_number(getattr(o, name)) for o in outs]
    ax.plot(xs, values, color=_BLUE, linewidth=1.8)
    if threshold is not None:
        if log_scale:
            if not threshold > 0:
                raise ValueError("the threshold of a statistic on a log scale must be positive")
            _hline(ax, math.log10(threshold), f"threshold {_fmt(float(threshold))}")
        else:
            _hline(ax, float(threshold), f"threshold {_fmt(float(threshold))}")
    if change_point is not None:
        _vline(ax, float(change_point), f"true change at {_fmt(float(change_point))}")
    stop = _first_stop(xs, outs)
    if stop is not None:
        x, o, _ = stop
        _mark(ax, x, values[xs.index(x)], f"alarm at {_fmt(x)}")
        estimate = getattr(o, "change_estimate", None)
        if estimate is not None:
            _vline(ax, float(estimate), f"estimated change at {estimate}", "--", _RED, left=True)
    if log_scale:
        _power_of_ten_axis(ax.yaxis)
    _style(ax, xlabel, ylabel)
    _legend(ax, loc="best")
    return ax


# ----------------------------------------------------------------- filters


def plot_filter(outputs: Any, *, observations: Sequence[Number] | None = None, truth: Any = None,
                component: int = 0, smoothed: Any = None, ax: Any = None) -> Any:
    """A filter's mean with a band of +-2 standard deviations, against the number of observations.

    Works for the Kalman filters (KalmanFilter, RobustKalmanFilter,
    LocalLevelKalman) and the particle filters; for a vector state,
    `component` picks the coordinate.  `observations` are drawn as dots,
    `truth` (a number or one value per step) as a reference line, and
    `smoothed` -- a `KalmanSmoothing` from `smoothing.kalman_smoother` over
    the same inputs -- as a second line and band.

    The y range leaves out the first, very wide part of a band that starts
    from a vague prior.
    """
    ax = _axes(ax)
    xs, outs, xlabel = _outputs(outputs)
    means, sds = [], []
    for o in outs:
        mean, var = _moments(o, component)
        means.append(mean)
        sds.append(math.sqrt(var) if var >= 0 else math.nan)
    lower = [m - 2 * s for m, s in zip(means, sds)]
    upper = [m + 2 * s for m, s in zip(means, sds)]
    always = list(means)
    if observations is not None:
        ys = [_number(y) for y in observations]
        ax.scatter(xs[:len(ys)], ys[:len(xs)], s=6, color=_GREY, alpha=0.7, linewidths=0, label="observations")
        always += ys
    _band(ax, xs, means, lower, upper, _BLUE, "filtered mean ± 2 sd")
    if smoothed is not None:
        s_means = [float(m[component]) for m in smoothed.means]
        s_sds = [math.sqrt(max(0.0, float(c[component][component]))) for c in smoothed.covariances]
        s_xs = xs[:len(s_means)]
        s_lower = [m - 2 * s for m, s in zip(s_means, s_sds)]
        s_upper = [m + 2 * s for m, s in zip(s_means, s_sds)]
        _band(ax, s_xs, s_means[:len(s_xs)], s_lower, s_upper, _ORANGE, "smoothed mean ± 2 sd")
        lower, upper = lower + s_lower, upper + s_upper
        always += s_means
    if truth is not None:
        if isinstance(truth, (int, float)):
            _hline(ax, float(truth), "true state", ":", direct=False)
            always.append(float(truth))
        else:
            values = [_number(v) for v in truth]
            ax.plot(xs[:len(values)], values[:len(xs)], color=_INK, linestyle=":", linewidth=1.5, label="true state")
            always += values
    _band_limits(ax, lower, upper, always)
    _style(ax, xlabel, "state" if component == 0 else f"state component {component}")
    _legend(ax, loc="best")
    return ax


def _band(ax: Any, xs: list[float], centre: list[float], lower: list[float], upper: list[float],
          color: str, label: str) -> None:
    ax.fill_between(xs, lower, upper, color=color, alpha=0.18, linewidth=0)
    ax.plot(xs, centre, color=color, linewidth=2.0, label=label)


def _moments(output: Any, component: int) -> tuple[float, float]:
    """(mean, variance) of one coordinate of a filter output."""
    if hasattr(output, "covariance"):  # KalmanFilterOutput
        return float(output.mean[component]), float(output.covariance[component][component])
    variance = getattr(output, "var", getattr(output, "variance", None))
    if variance is None or not hasattr(output, "mean"):
        raise ValueError(f"{type(output).__name__} is not a filter output: it needs a mean and a variance")
    mean = output.mean
    if isinstance(mean, (tuple, list)):
        return float(mean[component]), float(variance[component])
    if component != 0:
        raise ValueError("component must be 0 for a scalar state")
    return float(mean), float(variance)


# ---------------------------------------------------------- stopping times


def plot_stopping_times(summary_or_samples: Any, *, kind: str = "ecdf", ax: Any = None) -> Any:
    """The distribution of the stopping time, from `planning.simulate` or a list of stopping times.

    kind="ecdf" (the default): the fraction of ALL runs stopped by n, a
    step function that ends at the power (runs that did not stop within
    max_n never count), with the median marked.  kind="hist": a histogram
    of the stopping times of the runs that stopped.
    """
    if kind not in ("ecdf", "hist"):
        raise ValueError('kind must be "ecdf" or "hist"')
    ax = _axes(ax)
    times = getattr(summary_or_samples, "stopping_times", None)
    if times is not None:
        times = sorted(float(n) for n in times)
        runs, max_n = int(summary_or_samples.runs), float(summary_or_samples.max_n)
    else:
        times = sorted(float(n) for n in summary_or_samples)
        if not times:
            raise ValueError("nothing to plot: no stopping times")
        runs, max_n = len(times), times[-1]
    if kind == "hist":
        if not times:
            raise ValueError("no run stopped: nothing to draw a histogram of")
        ax.hist(times, bins="auto", color=_BLUE, edgecolor="white", linewidth=0.8)
        _style(ax, "sample size at the stop", "runs")
        return ax
    xs = [0.0, *times, max_n]
    ys = [i / runs for i in range(len(times) + 1)] + [len(times) / runs]
    ax.step(xs, ys, where="post", color=_BLUE, linewidth=2.0)
    power = len(times) / runs
    if runs > len(times):
        _hline(ax, power, f"stopped within {_fmt(max_n)}: {power:.0%}", ":")
    k = math.ceil(0.5 * runs) - 1
    if k < len(times):
        median = times[k]
        _mark(ax, median, (k + 1) / runs, f"median {_fmt(median)}")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, max_n)
    ax.yaxis.set_major_formatter(_percent())
    _style(ax, "sample size at the stop", "runs stopped")
    _legend(ax, loc="best")
    return ax
