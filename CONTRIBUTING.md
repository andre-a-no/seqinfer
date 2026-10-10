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
