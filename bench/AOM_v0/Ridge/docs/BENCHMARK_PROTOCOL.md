# AOM-Ridge Benchmark Protocol

## Goals

The benchmark should answer:

1. Does AOM-Ridge beat tuned raw Ridge?
2. Is superblock AOM-Ridge more stable than hard global selection?
3. Does strict-linear operator composition help once Ridge is tuned?
4. How much gain comes from nonlinear branches such as SNV/MSC?

## Baselines

```text
Ridge-raw
Ridge-SNV
Ridge-MSC
Ridge-SG-d1
Ridge-SG-d2
```

Use the same alpha grid and CV protocol as AOM-Ridge.

## AOM-Ridge Variants

Phase 1:

```text
AOMRidge-global-compact
AOMRidge-superblock-compact
```

Phase 2:

```text
AOMRidge-global-default
AOMRidge-superblock-default
AOMRidge-active-compact
AOMRidge-active-default
```

Phase 3:

```text
AOMRidge-branches-raw-snv-msc
AOMRidge-oof-experts-compact
```

## Alpha Selection

For each train fold:

```text
base = trace(K_train) / n_train
alphas = base * logspace(-6, 6, 50)
```

Select the alpha with lowest mean validation RMSE.

## Metrics

Per dataset and variant:

```text
rmsep
mae
r2
relative_rmsep_vs_Ridge_raw
relative_rmsep_vs_PLS_standard
fit_time_s
predict_time_s
status
error
```

Summary:

```text
median relative RMSEP
wins versus Ridge-raw
wins versus PLS-standard
failure count
median fit time
```

## Output Schema

Minimum CSV columns:

```text
dataset_group
dataset
task
variant
status
error
selection
operator_bank
alpha
block_scaling
active_operator_names
selected_operator_names
rmsep
mae
r2
relative_rmsep_vs_ridge_raw
relative_rmsep_vs_pls_standard
fit_time_s
predict_time_s
random_state
version
```

## Commands

Smoke:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge python \
  bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py \
  --workspace bench/AOM_v0/Ridge/benchmark_runs/smoke \
  --cohort smoke \
  --variants smoke \
  --cv 3
```

Full:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge python \
  bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py \
  --workspace bench/AOM_v0/Ridge/benchmark_runs/full \
  --cohort full \
  --variants full \
  --cv 5 \
  --resume
```

## Interpretation Rules

- Compare primarily against tuned raw Ridge.
- Treat `global` as a baseline, not expected champion.
- Require block scaling in all headline superblock variants.
- Do not claim nonlinear branch gains as strict-linear AOM gains.

