# nicon_v2 — Improving the NIRS CNN Baseline

**Status:** _Iterative experimental project (started 2026-04-30)_
**License:** CeCILL-2.1 (matches `nirs4all`)

## Scope

`nicon_v2` is a sandboxed research and benchmark project under `bench/` whose goal is to **redesign and surpass** the original `nicon` CNN of `nirs4all` on the standard NIR benchmark suite used by the TabPFN paper bench (`bench/nirs_synthetic_pfn/`) and the AOM-PLS / AOM-Ridge bench projects.

Concretely we target three deliverables:

1. A **principled deep model** for NIRS regression and classification that is
   - competitive with, or better than, **PLS, Ridge, AOM-PLS, AOM-Ridge, TabPFN-v2** on the same 60+ dataset cohort that the TabPFN paper used,
   - calibrated for uncertainty,
   - efficient enough to fit a single dataset in seconds on a single RTX-class GPU,
   - free of the chronic failure modes of `nicon` and `decon` (sigmoid output saturation, mixed-activation self-normalisation breakage, stride-only downsampling, no concat-derivatives, etc. — see `docs/WEAKNESS_ANALYSIS.md`).
2. A **fully reproducible benchmark** comparing every variant against the published baselines, with the same per-dataset cohort (cf. `bench/AOM_v0/benchmarks/cohort_regression.csv`).
3. A **publication-ready manuscript** with figures, tables and supplement, written in the same style as `bench/AOM_v0/publication/`.

## Constraints

* Nothing in this project modifies the `nirs4all` library. All code lives under `bench/nicon_v2/`.
* Every iteration ends with a **Codex review** of the changes (math / code / test / publication review prompts in `docs/codex_review_prompts/`).
* Append-only `docs/IMPLEMENTATION_LOG.md` records every iteration with the same structure used by `bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md`.
* Each iteration has a falsifiable hypothesis listed in `docs/HYPOTHESES.md`; the loop terminates only when the win-rate against the AOM/TabPFN reference equals or exceeds the targets in `docs/BENCHMARK_PROTOCOL.md`.

## Folder layout

```
bench/nicon_v2/
├── README.md                       # this file
├── Prompt.md                       # master executable prompt (driver)
├── nicon_v2/                       # source package (importable as `nicon_v2`)
│   ├── __init__.py
│   ├── datasets.py                 # cohort loader (TabPFN paper datasets)
│   ├── preprocessing.py            # SNV / MSC / SG / concat-derivatives
│   ├── augmentation.py             # Bjerrum + C-Mixup
│   ├── models/                     # baseline + improved variants
│   ├── training.py                 # train loop, ensembles
│   ├── uncertainty.py              # MC-dropout, conformal, deep ensembles
│   └── metrics.py                  # RMSE / MAE / R² / NLL / coverage
├── tests/                          # unit + parity tests
├── benchmarks/                     # runnable benchmark scripts
├── benchmark_runs/                 # CSV result workspaces (resumable)
├── docs/
│   ├── CONTEXT_REVIEW.md
│   ├── WEAKNESS_ANALYSIS.md
│   ├── HYPOTHESES.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── IMPLEMENTATION_LOG.md       # append-only iteration log
│   ├── MATH_SPEC.md
│   ├── BENCHMARK_PROTOCOL.md
│   ├── API.md
│   ├── PUBLICATION_REPO_PLAN.md
│   └── codex_review_prompts/
│       ├── math_review.md
│       ├── code_review.md
│       ├── test_review.md
│       └── publication_review.md
├── publication/                    # manuscript, figures, tables, scripts
└── source_materials/               # references, lit review, scratch notes
    └── literature_review/
        ├── LITERATURE_REVIEW.md    # 40 references, 7 themes, gap analysis
        └── references.bib
```

## Reference baselines & leaderboards

There are **two leaderboards**:

* **Primary leaderboard.** The 39-dataset AOM-Ridge **curated** cohort (`bench/AOM_v0/Ridge/benchmark_runs/curated/results.csv`). Used for the stop gate. Each row is paired against `aom_ridge_curated_best` (the best AOM-Ridge variant per dataset).
* **Secondary leaderboard.** The full 61-dataset regression cohort from `bench/AOM_v0/benchmarks/cohort_regression.csv`. Used in the manuscript for breadth, with reference values from the TabPFN paper (`ref_rmse_pls` / `ref_rmse_paper_ridge` / `ref_rmse_tabpfn_raw|opt` / `ref_rmse_cnn` / `ref_rmse_catboost`).

We additionally produce **first-class baseline rows** in our own benchmark CSV so all comparisons are reproducible without depending on cohort reference values:

| Reference | Source | Coverage |
|-----------|--------|----------|
| `Ridge-baseline`        | `nicon_v2.models.baseline.RidgeBaseline` (StandardScaler + Ridge + 5-fold α-CV) | every dataset |
| `PLS-baseline`          | `nicon_v2.models.baseline.PLSBaseline` (StandardScaler + PLSRegression + 5-fold n-comp CV) | every dataset |
| `NICON-baseline`        | upstream `nirs4all.operators.models.pytorch.nicon._build_nicon` (read-only import) | every dataset |
| `DECON-baseline`        | upstream `_build_decon` | every dataset |
| `DeepSpectra-baseline`  | `nicon_v2.models.deepspectra` (Zhang 2019 reimplementation, Phase 1) | every dataset |
| `nicon_v2-V1a`, `V1b`, `V1c`, `V2`, `V3a`, `V3b`, … | improved variants per phase | every dataset |
| `aom_ridge_curated_best` (column) | `bench/AOM_v0/Ridge/benchmark_runs/curated/results.csv` | 39 curated datasets |
| `paper Ridge / PLS / TabPFN raw / TabPFN opt / CNN / CatBoost` (columns) | `bench/AOM_v0/benchmarks/cohort_regression.csv` | wherever recorded |

Reference values are **never imputed**; missing → NaN → relative metric not computed. The replication checklist (`docs/REPLICATION_CHECKLIST.md`, written at publication time) lists the exact commit SHAs of the source CSVs.

## Reproducibility

Each benchmark CSV row records:
* `dataset_group`, `dataset`, `task`, `variant`
* `seed`, `cv_fold`, `cv_protocol`
* `model_version`, `python_version`, `torch_version`, `cuda_version`
* full hyper-parameters and training configuration
* `rmsep`, `mae`, `r2`, `fit_time_s`, `predict_time_s`
* relative metrics versus the AOM-Ridge / AOM-PLS / Ridge / PLS / TabPFN reference rows.

## Quick start (after Phase 0 lands)

```bash
# Smoke benchmark (3 datasets × current variants)
PYTHONPATH=bench/nicon_v2 python bench/nicon_v2/benchmarks/run_nicon_v2_benchmark.py \
  --workspace bench/nicon_v2/benchmark_runs/smoke \
  --cohort smoke --variants smoke --cv 3 --seed 0
```
