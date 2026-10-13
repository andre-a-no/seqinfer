# Releasing

A release is a GitHub release with a version tag. Publishing it starts
`.github/workflows/release.yml`, which tests the package and uploads it to
PyPI; Zenodo archives the same release and gives it a DOI. Nothing secret is
stored: PyPI trusts uploads from that workflow in this repository ("trusted
publishing").

## One-time setup (done for 0.1.0)

1. An account on https://pypi.org with two-factor authentication —
   preferably one belonging to Novikov Laboratories LLC — and, under
   *Your account → Publishing*, a trusted publisher for the project
   `sequential-analysis`: owner `andre-a-no`, repository
   `sequential-analysis`, workflow `release.yml`, environment `pypi`.
2. On GitHub: *Settings → Environments* `pypi`, optionally with yourself as a
   required reviewer, so that every upload waits for a click (*Review
   deployments → Approve and deploy* on the workflow run).
3. On https://zenodo.org: *Log in with GitHub → GitHub* and switch the
   repository on. Zenodo archives every GitHub release published after that
   — releases, not bare tags — and takes the authors, title and license from
   `CITATION.cff`.

## Rehearsal before a release

A version cannot be uploaded to PyPI twice, so rehearse first:

- **Dry run** — *Actions → release → Run workflow*, `publish_to: none`. Builds
  the wheel and the source distribution, installs the wheel into clean
  environments on Python 3.10–3.14, with and without numpy, and runs the test
  suite and `tools/smoke_test.py` (what a new user does first: the README run,
  a checkpoint, the precomputed table, a design through the cache). Nothing
  is published; no setup needed.
- **TestPyPI** — the same with `publish_to: testpypi`: the package is also
  uploaded to https://test.pypi.org, installed from there with `pip` and
  smoke-tested. It needs the same setup as PyPI above, on test.pypi.org (a
  separate account) with the environment name `testpypi`, plus a GitHub
  environment `testpypi`. TestPyPI, too, takes each version only once: a
  second rehearsal of the same version installs the first upload, so give
  every TestPyPI rehearsal its own version (`0.3.0rc1`, `0.3.0rc2`, …).

The smoke test can be run by hand against any installation:

    python tools/smoke_test.py 0.2.0

After the real upload the workflow installs the package from PyPI and runs the
smoke test again.

## Each release

1. Set the version in `src/sequential_analysis/version.py` (the only place
   it is written; `pyproject.toml` reads it from there), add a section
   `## X.Y.Z — YYYY-MM-DD` to `CHANGELOG.md`, and set `version` and
   `date-released` in `CITATION.cff` to the same version and date. The
   workflow refuses a release whose changelog or citation file names
   another version.
2. Merge to `main` with CI green.
3. *Releases → Draft a new release*: tag `vX.Y.Z` ("Create new tag on
   publish"), target `main`, title `sequential-analysis X.Y.Z`, the
   changelog section as the description, then **Publish release** (not
   *Save draft*: a draft creates no tag and starts nothing).

The workflow checks that the tag matches the version, builds the wheel and
the source distribution, runs the test suite against the installed wheel and
publishes. A version cannot be uploaded twice; fix mistakes with a new
version.

## Russian summary / Кратко по-русски

Выпуск — это GitHub-релиз с тегом версии: при его публикации workflow
тестирует пакет и загружает его на PyPI, а Zenodo архивирует релиз и
присваивает DOI. Настройка PyPI, окружения `pypi` и Zenodo уже сделана.
Перед выпуском — репетиция: Actions → release → Run workflow, `none`
(сборка и тесты установленного пакета). Каждый выпуск: поднять версию в
`version.py`, добавить датированный раздел в `CHANGELOG.md`, обновить
`version` и `date-released` в `CITATION.cff`, влить в `main`, затем Releases
→ Draft a new release → тег `vX.Y.Z` → **Publish release** (не *Save draft*)
и одобрить загрузку на PyPI.
