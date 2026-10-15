# Contributing

Thank you for your interest in sequential-analysis.

## Contributor License Agreement

sequential-analysis is offered under the GNU AGPL v3 and under commercial licenses
(see [COMMERCIAL.md](COMMERCIAL.md)). To keep that possible, Novikov
Laboratories LLC must hold the rights needed to license every part of the
code under both. We can therefore merge a contribution only after its
author has signed the Contributor License Agreement (CLA).

The agreement is in [CLA.md](CLA.md) (Russian and English; a draft until it
has been reviewed by a lawyer). Before opening a pull request with code,
please sign it and send it to andrei@hobukob.ru. Small fixes of typos in
documentation are welcome without it.

If you write code as part of your job, your employer may own it; in that
case the employer has to sign the CLA as well.

## Development

```
python -m pip install -e .
python -m unittest discover -s tests -t .
```

Every change to inference, persistence or delivery needs a test. New
source files start with the SPDX header used throughout the project.

## Minimum versions

The library asks for as little as it can: Python 3.8 or later, nothing
else; numpy 1.22 or later only for the optional `[fast]` extra; Node.js 16
or later for the TypeScript executor (18 to run its tests). Users should
not have to upgrade their interpreter to use it.

These floors are tested, not assumed. CI runs the whole test suite on
Python 3.8 with numpy 1.22.0 and on 3.9 without numpy, from the installed
wheel (job `oldest`), and the Python-TypeScript round trip on Node 16;
`vermin` fails the lint job if code needs a newer Python than 3.8.

When writing code:

- use `sequential_analysis._compat.zip_strict` instead of
  `zip(..., strict=True)`, and `_compat.pairwise`;
- keep `from __future__ import annotations` at the top of every module,
  and write type aliases evaluated at import with `typing`
  (`typing.Tuple[float, ...]`), not `tuple[float, ...]`;
- no `match`, no parenthesised context managers, no
  `importlib.resources.files` (use `pkgutil.get_data`).

A floor is raised only when a dependency or a language feature is worth
it, and the reason is written in the changelog.

