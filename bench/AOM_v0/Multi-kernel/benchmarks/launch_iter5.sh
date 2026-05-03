#!/usr/bin/env bash
# Iter 5: Sparse softmax (post-hoc top-k sparsification of softmax_cv weights).
# Codex P2 priority — sparse weight selection.
#
# Quick test on BEER showed:
#   - sparse on SNV branch: NO-OP (softmax already 0.978 concentrated on 1 block)
#   - sparse on ASLS branch: BIG WIN (eta spread across 5 blocks; sparse3 drops 2 noise blocks)
#     None: rmse=0.2267 (rel-PLS=0.886)
#     sparse3: rmse=0.2184 (rel-PLS=0.853)  ← matches BLUP-reml-asls champion
#
# Goal: see if this generalises across TIC/ALPINE/MANURE_MgO.

set -e

.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_full.py \
  --cohort bench/AOM_v0/Multi-kernel/benchmark_runs/iter2_cohort.csv \
  --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/iter5_sparse \
  --variants \
    mkR-softmax_cv-asls-default-active15-sparse5 \
    mkR-softmax_cv-asls-default-active15-sparse3 \
    mkR-softmax_cv-asls-default-active15-sparse2 \
    mkR-softmax_cv-asls-default-active15-sparse1 \
    mkR-softmax_cv-msc-default-active15-sparse5 \
    mkR-softmax_cv-msc-default-active15-sparse3 \
    mkR-softmax_cv-snv-default-active15-sparse3 \
    mkR-softmax_cv-default-active15-sparse5 \
    mkR-softmax_cv-default-active15-sparse3 \
  --n-jobs 2 2>&1 | tail -3
