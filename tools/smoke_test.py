# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Smoke test of an installed sequential-analysis: what a new user does first.

    python tools/smoke_test.py            # the installed package
    python tools/smoke_test.py 0.1.0      # ... and check that this version is installed

Run it outside the source tree, against a package installed from a wheel,
TestPyPI or PyPI: it fails if the package was imported from a source
checkout instead. Exits non-zero on the first failure.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import tempfile
import time
from importlib import metadata


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'ok  ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
    if not condition:
        sys.exit(1)


def main() -> None:
    started = time.perf_counter()
    import sequential_analysis as sa

    check("imported from an installation", "site-packages" in sa.__file__ or "dist-packages" in sa.__file__,
          sa.__file__)
    installed = metadata.version("sequential-analysis")
    check("version", sa.__version__ == installed, f"{sa.__version__} (metadata {installed})")
    if len(sys.argv) > 1:
        check("expected version", installed == sys.argv[1], f"wanted {sys.argv[1]}")
    licence = metadata.metadata("sequential-analysis").get("License-Expression", "")
    check("license metadata", "AGPL-3.0-only" in licence, licence)

    # the README's first run
    from sequential_analysis import Run, run_sync
    from sequential_analysis.procedures import Gaussian, TwoSPRT
    from sequential_analysis.sources import gaussian

    test = TwoSPRT(Gaussian(sigma=1.0), theta0=0.0, theta1=0.5, alpha0=0.05, alpha1=0.05)
    run = Run(test)
    run_sync(run, gaussian("x", mean=0.5, sd=1.0, seed=7))
    check("README run", run.state.decision == "reject_h0" and run.t == 18, f"{run.state.decision} after {run.t}")

    # checkpoint, JSON round trip, restore, and the same continuation
    checkpoint = json.loads(json.dumps(run.checkpoint()))
    again = Run.restore(test, checkpoint)
    check("checkpoint restores", again.t == run.t and again.history_digest == run.history_digest)

    # the precomputed table ships with the wheel
    from sequential_analysis.design import kiefer_weiss_plan, tabulated_least_favourable

    lam = tabulated_least_favourable(0.01, 0.1, 2.0)
    check("least favourable table", lam is not None and 0.6 < lam < 0.75, f"{lam}")

    # a small optimal design, through the cache in a temporary directory, with its messages
    from sequential_analysis.procedures import Bernoulli

    with tempfile.TemporaryDirectory() as cache:
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logging.getLogger("sequential_analysis.cache").addHandler(handler)
        first = kiefer_weiss_plan(0.3, 0.5, 0.05, 0.05, 80, family=Bernoulli(), cache=cache)
        second = kiefer_weiss_plan(0.3, 0.5, 0.05, 0.05, 80, family=Bernoulli(), cache=cache)
        actions = [getattr(r, "incident", {}).get("action") for r in records]
        errors = f"errors {first.at_theta0.reject:.4f}/{first.at_theta1.accept:.4f}"
        check("design", first.at_theta0.reject <= 0.05 and first.at_theta1.accept <= 0.05,
              f"{errors}, E[N] {first.at_theta_star.expected_n:.2f}")
        check("design cache", actions == ["saved", "loaded"] and second.plan.plan == first.plan.plan, str(actions))

    # Student t, anytime-valid and group sequential procedures
    from sequential_analysis.distributions import t_cdf
    from sequential_analysis.group_sequential import group_sequential_design

    check("Student t", abs(t_cdf(2.0, 10) - 0.9633059826146299) < 1e-12)
    design = group_sequential_design(0.025, [0.2, 0.4, 0.6, 0.8, 1.0])
    check("group sequential design", abs(design.bounds[-1] - 2.031) < 1e-3, f"{[round(b, 3) for b in design.bounds]}")

    try:
        import numpy  # noqa: F401
        fast = True
    except ImportError:
        fast = False
    print(f"numpy installed (the [fast] extra): {fast}")
    check("finished", math.isfinite(time.perf_counter() - started), f"{time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
