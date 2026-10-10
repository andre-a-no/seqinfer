# Releasing

Releases are published to PyPI by `.github/workflows/release.yml` when a
version tag is pushed. Nothing secret is stored: PyPI trusts uploads from
that workflow in this repository ("trusted publishing").

## One-time setup

1. Create an account on https://pypi.org (two-factor authentication is
   required) — preferably one belonging to Novikov Laboratories LLC.
2. On PyPI: *Your account → Publishing → Add a new pending publisher*:
   - PyPI project name: `sequential-analysis`
   - Owner: `andre-a-no`, repository: `sequential-analysis`
     (the repository's name after renaming it on GitHub)
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. On GitHub: *Settings → Environments → New environment* `pypi`. Optionally
   add yourself as a required reviewer, so that every upload waits for a
   click.

## Each release

1. Update `__version__` in `src/sequential_analysis/version.py` (the only
   place it is written; `pyproject.toml` reads it from there) and move the
   "unreleased" section of `CHANGELOG.md` under the version and date.
2. Merge to `main` with CI green.
3. Tag and push:

       git tag v0.1.0
       git push origin v0.1.0

The workflow checks that the tag matches the version, builds the wheel and
the source distribution, runs the test suite against the installed wheel and
publishes. A version cannot be uploaded twice; fix mistakes with a new
version.

## Russian summary / Кратко по-русски

Один раз: завести аккаунт на pypi.org (лучше на компанию, с двухфакторной
аутентификацией), добавить там «pending publisher» с параметрами выше и
создать на GitHub окружение `pypi`. Каждый выпуск: поднять версию, влить в
`main`, поставить тег `vX.Y.Z` и отправить его — публикация пройдёт сама.
