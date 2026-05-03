#!/usr/bin/env bash
# Iter 4: Codex-prioritised score-method ablation + cross-over branches.
# Run AFTER iter3 completes.

set -e

.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_full.py \
  --cohort bench/AOM_v0/Multi-kernel/benchmark_runs/iter2_cohort.csv \
  --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/iter4_score_methods \
  --variants \
    mkR-softmax_cv-snv-default-active15-kta \
    MKM-reml-asls-default-active15-kta \
    mkR-softmax_cv-snv-default-active15-blend \
    MKM-reml-asls-default-active15-blend \
    mkR-softmax_cv-asls-default-active15 \
    mkR-softmax_cv-default-active15-tuned \
  --n-jobs 2 2>&1 | tail -3
