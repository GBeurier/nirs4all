# AOM_v0 — Implementation Log

This document records the decisions, environment, and execution trace of the
one-shot AOM_v0 implementation. It is the agent-side counterpart to the
specification documents under `bench/AOM_v0/docs/`.

## Environment (captured 2026-04-27)

- Python: `3.13.11 (main, Dec  6 2025, 08:52:51) [GCC 11.4.0]`
- numpy: `2.3.5`
- scipy: `1.17.0`
- scikit-learn: `1.7.2`
- torch: `2.10.0+cu128` (CUDA available)
- codex-cli: `0.125.0` (available; may be invoked at the four review checkpoints)
- pytest: available via the project `.venv`
- platform: WSL2 Linux 6.6.114.1

## Working directory

All implementation lives under:

```text
bench/AOM_v0/
```

The production `nirs4all/` package is treated as read-only reference.

## Source documents read in Phase 0

1. `README.md`
2. `docs/CONTEXT_REVIEW.md`
3. `docs/AOMPLS_MATH_SPEC.md`
4. `docs/IMPLEMENTATION_PLAN.md`
5. `docs/BENCHMARK_PROTOCOL.md`
6. `docs/PUBLICATION_REPO_PLAN.md`
7. `publication/manuscript/PAPER_DRAFT.md`
8. `source_materials/AOM/PLAN_V1.md` — argued the SIMPLS-covariance reformulation
9. `source_materials/AOM/ROADMAP.md` — milestone-level plan
10. `source_materials/AOM/PUBLICATION_PLAN.md` — argument and section plan
11. `source_materials/AOM/PUBLICATION_BACKLOG.md` — workstream specs
12. `source_materials/AOM/report.md` — five-dataset prototype results table
13. `source_materials/AOM/advanced_architectures.md` — prior exploration synthesis
14. `source_materials/tabpfn/SPECTRAL_LATENT_FEATURES.md`
15. `source_materials/tabpfn/MASTER_RESULTS_PROFILE.md`
16. `source_materials/tabpfn/TABPFN_PAPER_PROTOCOL_NOTES.md`

## Reference data observed

- `bench/tabpfn_paper/master_results.csv`
  - 335 rows, 29 columns, 61 unique regression dataset splits
  - models: TabPFN-Raw, TabPFN-opt, Catboost, PLS, Ridge, CNN
  - schema is the join key for AOM/POP regression rows; AOM diagnostics extend
    the schema (additional columns appended after the master schema)
- `bench/tabpfn_paper/data/regression/<FAMILY>/<dataset>/{Xtrain.csv,Xtest.csv,Ytrain.csv,Ytest.csv}`
  - CSV with `;` delimiter, header row of wavelength/feature columns
  - Y file: single column, header `"x"`, numeric target
  - Optional `Mtrain.csv`, `Mtest.csv` for sample IDs/metadata
- `bench/tabpfn_paper/data/classification/<FAMILY>/<dataset>/{Xtrain.csv,...,Ytrain.csv,...}`
  - same physical layout; targets are class labels (numeric or string)
  - 11 family directories observed (ARABIDOPSIS_CEFE, BEEF_Impurity, COFFEE_orig,
    COFFEE_sp, Cassava, FUSARIUM, FruitPuree, MALARIA, MILK, PISTACIA,
    Wood_Sustainability)

## Decisions applied (defaults from the prompt)

| Default | Value | Source |
| --- | --- | --- |
| `random_state` | `0` | section 5 of `Prompt.md` |
| `center` | `True` | spec 5 |
| `scale` | `False` | spec 5 |
| `max_components` | `min(25, n_samples - 1, n_features)` | spec 5 |
| `n_components` | `"auto"` | spec 5 |
| `engine` | `"simpls_covariance"` | spec 5 |
| `selection` (AOM) | `"global"` | spec 5 |
| `selection` (POP) | `"per_component"` | spec 5 |
| `criterion` | `"cv"` | spec 5 |
| `cv` | `5` | spec 5 |
| `orthogonalization` | `"auto"` | spec 5 |

`orthogonalization="auto"` resolution:

- `transformed` for `selection in {"none", "global"}`
- `original` for `selection="per_component"`
- `original` for classification POP

## Architecture chosen

We follow the layout documented in `docs/IMPLEMENTATION_PLAN.md`:

```text
bench/AOM_v0/aompls/
  __init__.py
  operators.py        # LinearSpectralOperator protocol + concrete operators
  banks.py            # operator bank presets (compact / default / extended)
  preprocessing.py    # standalone preprocessors (e.g. SNV/MSC) for upstream use
  centering.py        # centering / scaling helpers shared by engines
  nipals.py           # materialized + adjoint NIPALS
  simpls.py           # materialized + covariance-space SIMPLS + superblock
  scorers.py          # criteria: covariance, cv, approx_press, hybrid, holdout
  selection.py        # global / per_component / soft / superblock policies
  estimators.py       # AOMPLSRegressor / POPPLSRegressor sklearn API
  classification.py   # AOMPLSDAClassifier / POPPLSDAClassifier
  torch_backend.py    # NIPALS adjoint, SIMPLS covariance, superblock SIMPLS
  synthetic.py        # deterministic generators
  metrics.py          # rmse / mae / r2 / balanced_accuracy / log_loss / brier / ece
  diagnostics.py      # operator score tables, run summary
```

## Mathematical conventions applied

- Strict identity `(X A^T)^T Y = A X^T Y` is enforced for every linear operator.
- All operators expose `transform`, `apply_cov`, `adjoint_vec`, `matrix(p)`,
  `is_linear_at_apply()` (default `True` for strict-linear ops, may be `False`
  for fitted preprocessors used outside the bank).
- Coefficient construction uses `B = Z (P^T Z)^+ Q^T` where `Z = x_effective_weights_`
  and `+` is the Moore-Penrose pseudoinverse when needed.
- Prediction is in the original space:
  `Y_hat = (X - x_mean) @ coef_ + intercept_`.

## Phase plan and execution

| Phase | Status | Notes |
| --- | --- | --- |
| 0 — Inspection | done | this document committed |
| 1 — Operators and banks | in progress | linear protocol, identity, SG smoothing/derivative, finite difference, detrend projection, Norris-Williams gap derivative, Whittaker, composition |
| 2 — Reference algorithms | next | PLS standard, NIPALS materialized, SIMPLS materialized |
| 3 — Fast engines | next | NIPALS adjoint, SIMPLS covariance |
| 4 — Selection unified | next | none/global/per_component/soft/superblock; criteria covariance/cv/approx_press/hybrid/holdout |
| 5 — sklearn API | next | AOMPLSRegressor / POPPLSRegressor + diagnostics |
| 6 — Classification | next | AOMPLSDAClassifier / POPPLSDAClassifier with logistic calibration + temperature fallback |
| 7 — Torch backend | next | nipals_adjoint, simpls_covariance, superblock_simpls (CPU fallback) |
| 8 — Benchmarks | next | smoke run, full run resumable |
| 9 — Documentation | next | API and validation notes |
| 10 — Publication | next | full LaTeX manuscript + supplement + scripts |
| 11 — Final verification | next | compileall + pytest + smoke benchmark |

## Codex review checkpoints

Codex CLI 0.125.0 is detected. The four review prompts already exist in
`docs/codex_review_prompts/`:

- `math_review.md` — Phase 2/3
- `code_review.md` — Phase 5
- `test_review.md` — Phase 6
- `publication_review.md` — Phase 10

Codex review attempts and their responses are recorded in
`docs/CODEX_REVIEWS.md`. If a review call exits with a non-zero status,
the prompt path is recorded as ready-to-run instead.

## Full benchmark policy

The full regression benchmark runs across 61 unique splits and writes to a
resumable workspace under `benchmark_runs/regression_full/`. The expected
wall-clock for the full benchmark exceeds the deterministic run budget of this
session, therefore:

- The smoke benchmark is executed in this run (5 representative datasets, all
  numpy variants).
- The full benchmark commands are documented in `docs/AOMPLS_VALIDATION.md` and
  the runner exposes `--resume` so it may be relaunched at any time.

## Dependencies known to be missing in the local environment

None observed during Phase 0. `numpy`, `scipy`, `scikit-learn`, `torch`,
`pandas`, `pyarrow`, `pytest`, and `joblib` are all available.

---

## Final state (2026-04-28)

| Item | Status |
| --- | --- |
| Operators (identity, SG, FD, NW, detrend, Whittaker, composed, Gaussian-derivative, fixed-shift) | implemented + 18 tests |
| Operator banks (compact 9 ops, default 77 ops, extended 82 ops) | implemented |
| NIPALS standard + materialized + adjoint | implemented; PLS2 adjoint bug found by Codex math review and fixed |
| SIMPLS standard + materialized + covariance + superblock | implemented; transformed mode now delegates to materialized fixed; superblock returns original-space coefficients |
| Active superblock | implemented + 8 tests |
| Operator explorer (beam search, similarity, generation grammar) | implemented + 17 tests |
| Selection policies (none, global, per_component, soft, superblock, active_superblock) | implemented + 8 tests |
| Criteria (covariance, cv, approx_press, hybrid, holdout) | implemented + tests |
| sklearn API regressors (AOM/POP) | implemented + 9 tests |
| sklearn API classifiers (AOM/POP DA + logistic calibration + temperature fallback) | implemented + 7 tests |
| Torch backend (NIPALS adjoint, SIMPLS covariance, superblock) | implemented + 7 parity tests |
| Synthetic generators (regression, classification, PLS1) | implemented |
| Cohort builders (regression, classification) | implemented + 5 tests |
| Resumable benchmark runner | implemented |
| Smoke benchmark | run; 30 regression rows + 11 classification rows |
| Extended benchmark (20 datasets x 11 variants) | run; 220 rows |
| Codex math review | run; 4 findings, all fixed |
| Codex code review | run; 6 findings, 2 HIGH fixed, 3 MEDIUM documented as follow-ups |
| Documentation (API, validation, codex log) | written |
| Publication package (LaTeX manuscript, supplement, scripts, journal/arXiv files) | written and updated with extended benchmark |

97 tests pass on the AOM_v0 package. Four agent-generated docs and the
manuscript are in place. The smoke and extended benchmark CSVs are in
`bench/AOM_v0/benchmark_runs/{smoke,extended}/results.csv`.
