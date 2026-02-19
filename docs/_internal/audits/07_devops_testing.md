# Beta-Readiness Audit: DevOps, CI/CD, Testing, and Packaging - Release Blockers Only (2026-02-19)

## Active Findings

(none)

## Beta Release Tasks (Open)

(none)

## Resolved Findings
- `D05 [RESOLVED]` macOS added to CI test matrix with `timeout-minutes: 30` and `continue-on-error: true` so failures don't block merges.
- `D01 [RESOLVED]` mypy type-check gate added to CI.yaml (`type-check` job, runs before `tests`).
- `D06 [RESOLVED]` Python 3.12 added to CI test matrix (`['3.11', '3.12', '3.13']`).
- `D09 [RESOLVED]` Direct unit tests added for `nirs4all/workspace/__init__.py` in `tests/unit/workspace/test_workspace_init.py`.
- `D13 [RESOLVED]` CodeQL static security workflow added at `.github/workflows/codeql.yml`.
- `D17 [RESOLVED]` `pre-publish.yml` and `publish.yml` refactored to share jobs via `shared-test-and-docs.yml` `workflow_call`.
