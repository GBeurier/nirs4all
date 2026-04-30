# AOM-Ridge Reproducibility Guide

This document lists the exact steps to reproduce every artefact in the
manuscript from a fresh checkout.

## Environment

- Python 3.11+
- Virtual env at the repository root: `.venv/`
- Install dependencies (already done if you cloned with `.venv` populated):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install numpy scipy scikit-learn pandas matplotlib pytest ruff
  pip install -e .  # nirs4all (provides SPXYFold)
  ```
- LaTeX (TeX Live 2022+) for the manuscript build.

## One-shot rebuild

From the repository root:

```bash
cd bench/AOM_v0/Ridge
make all
```

This runs lint, tests, the curated benchmark (39 datasets, 10 lean
variants, ~1.5 h), regenerates all figures and tables, and rebuilds the
manuscript PDF.

## Step-by-step

### 1. Tests + lint

```bash
cd bench/AOM_v0/Ridge
make test
make lint
```

Expected: 96/96 tests pass, ruff clean.

### 2. Curated cohort (39 NIRS datasets with paper-Ridge HPO baseline)

The cohort is filtered from the upstream TabPFN-paper regression cohort
to datasets with a valid `ref_rmse_ridge`, `n_train <= 400`, and
`p <= 2500`:

```bash
make benchmark_runs/curated_cohort.csv
```

### 3. Run the benchmark

```bash
make bench-curated
```

Outputs `benchmark_runs/curated_v2/results.csv` with one row per
(dataset, variant). Variants are listed in
`benchmarks/run_aomridge_benchmark.py::LEAN_VARIANTS`. SPXYFold(3)
inner CV.

### 4. Build figures + tables

```bash
make figures tables
```

Outputs:

- `publication/figures/fig_aomridge_framework.{pdf,png}`
- `publication/figures/fig_alpha_grid.{pdf,png}`
- `publication/figures/fig_per_dataset_delta_vs_paper_ridge.{pdf,png}`
- `publication/figures/fig_critical_difference.{pdf,png}`
- `publication/figures/fig_heatmap_methods_x_datasets.{pdf,png}`
- `publication/figures/fig_cumulative_irmsep.{pdf,png}`
- `publication/figures/fig_irmsep_vs_time.{pdf,png}`
- `publication/tables/table_per_dataset_results.tex`
- `publication/tables/table_summary.tex`
- `publication/tables/table_per_method_summary.tex`

### 5. Compile the paper

```bash
make paper
```

Output: `publication/manuscript/aomridge_paper.pdf` (~16 pages).

## Layout

```
bench/AOM_v0/Ridge/
  aomridge/                  # the package
    kernels.py
    solvers.py
    selection.py
    estimators.py
    branches.py
    mkl.py
    cv.py
    preprocessing.py
  tests/                     # 96 tests
  benchmarks/
    run_aomridge_benchmark.py
    summarize_aomridge_results.py
  benchmark_runs/
    curated_cohort.csv       # cached cohort filter
    curated_v2/results.csv   # main results
    smoke6/results.csv       # 6-dataset smoke
  publication/
    scripts/
      make_aomridge_figures.py
      make_aomridge_tables.py
      make_cumulative_irmsep.py
    figures/                 # generated PDFs + PNGs
    tables/                  # generated .tex files
    manuscript/
      aomridge_paper.tex
      abstract.tex
      refs.bib
  docs/
    AOM_RIDGE_MATH_SPEC.md
    AOM_RIDGE_API.md
    IMPLEMENTATION_LOG.md
    CODEX_BACKLOG_*.md
  Makefile                   # this guide's pipeline
  REPRODUCIBILITY.md         # this file
```

## Reproducibility caveats

- The benchmark relies on `SPXYFold` from the `nirs4all` package. With
  default `pca_components=None`, SPXY produces identical folds for the
  same data; so `RepeatedSPXYFold` adds a small per-repeat row jitter
  to break the tie. This is documented in `aomridge/cv.py`.
- A handful of datasets (e.g. Tleaf) have NaN values in the input
  spectra; the benchmark imputes column means at load time
  (`_load_csv_array` in `run_aomridge_benchmark.py`).
- QUARTZ is dropped from the headline metrics: its paper Ridge ref
  RMSEP ~ 3e-9 makes relative ratios meaningless.
- The paper baselines in `bench/AOM_v0/publication/tables/master_pivot.csv`
  are taken verbatim from the TabPFN-paper protocol (CNN, CatBoost,
  PLS, Ridge, TabPFN-Raw, TabPFN-opt). They are not re-run here.
