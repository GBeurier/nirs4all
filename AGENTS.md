# Repository Guidelines

## Project Structure & Module Organization

`nirs4all/` contains the Python package. Key areas include `data/`, `operators/`, `pipeline/`, `controllers/`, `analysis/`, `visualization/`, `workspace/`, `api/`, and the CLI in `cli/`. Tests live in `tests/unit/` and `tests/integration/`, with reusable data and pipeline fixtures in `tests/fixtures/`. Documentation is built from `docs/source/`; runnable examples and sample YAML/JSON configs are under `examples/`. Packaging and tool settings are centralized in `pyproject.toml`.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"`: install the package in editable mode with developer tools.
- `python -m pip install -r requirements-test.txt`: install the broader CI test dependency set.
- `ruff check .`: run lint checks used by CI.
- `mypy nirs4all`: type-check package code.
- `pytest tests/unit/`: run the fast unit test suite.
- `pytest tests/integration/`: run integration tests for pipelines, storage, artifacts, APIs, and examples.
- `pytest tests/ --cov=nirs4all --cov-report=xml`: run the full suite with coverage.
- `sphinx-build -b html docs/source docs/_build/html --keep-going`: build documentation.
- `python -m build --sdist --wheel --outdir dist`: validate package builds.

## Coding Style & Naming Conventions

Target Python 3.11+. Follow PEP 8, Google-style docstrings, and existing module patterns. Ruff is the primary linter; the configured line length is 220, with import sorting enabled. Prefer explicit types for public APIs and keep `mypy nirs4all` clean. Use `snake_case` for functions, variables, modules, config keys, and test functions; use `PascalCase` for classes.

## Testing Guidelines

Place focused module tests in the matching `tests/unit/<area>/` directory and cross-component workflows in `tests/integration/<area>/`. Name test files `test_*.py`, classes `Test*`, and functions `test_*`. Use existing pytest markers such as `slow`, `sklearn`, `tensorflow`, `torch`, `keras`, `jax`, `gpu`, and `stress` when dependencies or runtime matter. Keep fixtures small and reusable under `tests/fixtures/`.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages, often Conventional Commit style such as `feat(dataviz): ...` or `fix(loaders): ...`; prefer that format for scoped changes. Pull requests should include a summary, change list, test results, and notes for reviewers. Link related issues when applicable, update docs/examples for user-facing changes, and include screenshots only for visualization or documentation UI changes.

## Security & Configuration Tips

Do not commit generated workspaces, caches, credentials, or large local datasets. Keep dependency and license changes reflected in `requirements*.txt`, `pyproject.toml`, and license notice files when relevant.
