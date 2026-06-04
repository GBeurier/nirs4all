# nirs4all — Regression Baseline (frozen 2026-06-04)

This is the **before** state. The zero-regression gate for the debt campaign is:
**keep the 7313 passing tests green; introduce no new failure; the 6 known failures below are the accepted pre-existing baseline.**

## Test suite baseline

Command (pinned determinism env):
```
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  pytest tests -n auto --dist worksteal
```
Result: **7313 passed · 6 failed · 129 skipped** in ~295s (after removing dead `test_new_operators.py`; collection clean = 7446 collected).

## The 6 pre-existing failures (NOT regressions — known debt)

| # | Test | Error | Root cause | Kind |
|---|------|-------|-----------|------|
| 1 | `unit/operators/models/test_aom_pls.py::TestSklearnCompat::test_estimator_type` | `AttributeError: 'AOMPLSRegressor' object has no attribute '_estimator_type'` | vendored `_aom_nirs` estimator doesn't expose legacy sklearn `_estimator_type` (related to the MRO/tags issue Codex flagged) | AOM bug / sklearn-API drift |
| 2 | `unit/operators/models/test_aom_pls_classifier.py::TestSklearnCompat::test_estimator_type` | same | same | AOM bug |
| 3 | `unit/operators/models/test_pop_pls.py::TestSklearnCompat::test_estimator_type` | `POPPLSRegressor` same | same | AOM bug |
| 4 | `unit/operators/models/test_pop_pls.py::TestPOPPLSClassifier::test_estimator_type` | same | same | AOM bug |
| 5 | `integration/parity/test_parity_smoke.py::test_explain_path[explain_path_baseline]` | `KeyError: 'step_idx'` → `RuntimeError: Pipeline step 3 failed` (`executor.py:936`) | real bug in the explain path | **real bug** |
| 6 | `integration/parity/test_parity_smoke.py::test_pipeline_runs_end_to_end[generator_grid_n_components_scale]` | `AssertionError: expected >= 18 predictions, got 17` | generator grid expansion off-by-one (or stale expectation) | generator bug / drift |

## Gate interpretation
- **Regression** = any of the 7313 passing tests starts failing, OR a new failure appears.
- The 4 AOM `_estimator_type` failures must stay **unchanged** through the aom Option-A lint fix (it is behavior-preserving — it must not flip them either way unless we explicitly decide to fix `_estimator_type`, which is a separate Tier-R behavior change).
- Failures 5 and 6 are real bugs and candidate Tier-R fixes later — out of the "identical results" campaign.

## Environment
- python `.venv` 3.11 · ruff 0.15.15 · mypy 2.1.0 · pytest via `.venv`
- node/codex resolved via `/home/delete/.nvm/versions/node/v22.21.1/bin` + `/home/delete/.local/bin`.
