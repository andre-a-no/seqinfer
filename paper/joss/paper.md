---
title: 'sequential-analysis: sequential tests, anytime-valid inference and group sequential designs with checkpoints that resume exactly'
tags:
  - Python
  - TypeScript
  - statistics
  - sequential analysis
  - A/B testing
  - clinical trials
  - anytime-valid inference
  - change-point detection
authors:
  - name: Alexander Stepanov
    affiliation: 1
  - name: Andrej Novikov
    corresponding: true
    affiliation: 2
affiliations:
  - name: Sobolev Institute of Mathematics, Russian Federation
    index: 1
  - name: Novikov Laboratories LLC, Russian Federation
    index: 2
date: 16 October 2026
bibliography: paper.bib
---

# Summary

A sequential procedure looks at its data as they arrive and may stop as
soon as they are conclusive. Online experiments, clinical trials with
interim analyses, quality control and the monitoring of production
systems all work this way. `sequential-analysis` is a Python library
(Python 3.8 or later, no required dependencies) of such procedures:

* Wald's sequential probability ratio test [@wald1945], Lorden's 2-SPRT
  [@lorden1976], and optimal truncated tests for the Kiefer–Weiss problem
  [@kiefer1957] by backward induction;
* group sequential designs with alpha and beta spending [@lan1983;
  @jennison2000] for normal, binary and time-to-event endpoints, with
  boundaries recomputed at the information actually reached;
* anytime-valid tests and confidence sequences [@ramdas2023], parametric
  [@robbins1970; @lindon2022linear], nonparametric [@howard2022;
  @waudbysmith2024] and for survival data [@terschure2024];
* A/B tests:
  * exact tests for conversion rates [@turner2024];
  * bounded metrics and rate ratios;
  * regression adjustment;
  * false discovery rate control over metrics and over a series of
    experiments [@wang2022ebh; @xu2024online];
* change-point detection by CUSUM [@page1954], Shiryaev–Roberts, e-detectors
  [@shin2024edetectors] and Bayesian online change points [@adams2007];
* Kalman, particle and hidden Markov filters with smoothing.

Each procedure is a pure state-transition function
$F(S, X, R) \to (S', O, R')$ of its state, an observation and a random
state. A separate runtime acquires, aligns, orders, deduplicates and
delivers observations, and writes checkpoints. A run restored from a
checkpoint continues bit for bit as if it had never stopped. A history
digest in RFC 8785 canonical JSON [@rfc8785] records exactly which data
produced a decision. A specification of the formats and a TypeScript
executor let a run move between languages: 58 conformance vectors are
reproduced, about a third bit for bit and the rest to $4\cdot10^{-13}$.

# Statement of need

Sequential methods are scattered across languages and assumptions:

* Group sequential designs live mainly in R (`gsDesign`, `rpact`
  [@rpact]) and in SAS.
* Anytime-valid methods are spread over research code in several
  languages.
* Experimentation platforms implement their own variants [@johari2022;
  @spotifyconfidence].

A team that runs experiments in Python, or that needs to stop, restart
and audit a monitoring process, has to assemble these pieces itself.
The pieces then disagree on what a "sample" is, and nothing records
which data a decision was based on.

`sequential-analysis` offers the methods behind one interface, from the
classical tests to the anytime-valid ones. It is written for:

* statisticians designing trials and experiments, who need boundaries,
  sample sizes and operating characteristics;
* engineers running them on streams, who need checkpoints, replays and
  exact resumption;
* researchers comparing methods, who need the same data delivered
  identically to every procedure.

The library states the cost of each guarantee. For example, the exact
two-proportion test needs 1.3 to 1.6 times the users of the asymptotic
test in simulation. A log-rank test monitored after every event loses its
level under staggered entry when hazards depend on time since entry. The
documentation says so where the method is offered.

# Quality control

Every statistical claim in a docstring is checked by a test:

* error rates, coverage and run lengths by simulation;
* martingale properties by exact enumeration;
* filters against exact Gaussian conditioning in rational arithmetic and
  against enumeration of hidden paths.

Group sequential designs are checked against `rpact` and the log-rank
statistic against R's `survival` package. A CI job regenerates these
references with R on every change. The comparison covers 56 designs
(boundaries to $2\cdot10^{-7}$), 44 sample sizes and interim analyses
of a trial with staggered entry.

The test suite runs on Python 3.8 to 3.14, on the oldest supported
numpy, and against the installed wheel. Property-based tests generate
streams, split points and canonical JSON values. Conformance vectors tie
the Python and TypeScript implementations together.

# Acknowledgements

We thank the authors of `rpact` and `survival`, whose implementations
serve as independent references in the tests.

# References
