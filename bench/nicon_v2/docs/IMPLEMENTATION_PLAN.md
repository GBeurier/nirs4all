# nicon_v2 — Implementation Plan

This file is the phase blueprint. Phases are numbered; each phase is an iteration of the loop in `Prompt.md`. Each phase ends with a Codex review checkpoint.

Acceptance gates are **stricter than the Hypothesis tests** — they protect against regressions on prior gains and on per-dataset coverage.

---

## Phase 0 — Skeleton & baseline reproducibility (no model changes)

**Goal.** Be able to reproduce the existing `nicon` and `decon` scores on the cohort under our own benchmark harness. No model changes; we are only verifying the harness, the dataset loaders and the reference numbers.

**Files.**
* `nicon_v2/datasets.py` — load TabPFN-paper / AOM cohort CSVs (reuse `bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py::_load_csv_array` style).
* `nicon_v2/metrics.py` — RMSE / MAE / R² / NLL / coverage on raw or scaled y.
* `nicon_v2/training.py` — train + cross-validate + ensemble entry-point (initially trivial single network).
* `nicon_v2/models/baseline.py` — wraps `nirs4all.operators.models.pytorch.nicon.nicon` to produce a reference parity baseline. (We import from nirs4all; we never modify it.)
* `benchmarks/run_baseline_benchmark.py` — driver, resumable CSV; runs `nicon`, `decon`, `pls`, `ridge` baselines.
* `tests/test_datasets.py`, `tests/test_metrics.py`, `tests/test_training_smoke.py`.
* `docs/codex_review_prompts/code_review.md`.

**Acceptance.**
* Smoke benchmark on `[ALPINE_P_291_KS, Beer_OriginalExtract_60_KS, Rice_Amylose_313_YbasedSplit]` × `[nicon, decon, ridge, pls]` runs end-to-end and produces a CSV identical in schema to `bench/AOM_v0/Ridge/benchmark_runs/smoke/results.csv` (subset of columns).
* Tests green: `pytest bench/nicon_v2/tests -q`.
* Codex code review run on the new files; findings logged in IMPLEMENTATION_LOG.

**Why this comes first.** Without our own harness reproducing existing numbers, no later improvement can be attributed cleanly.

---

## Phase 1a — Minimal head & activation repair (H1 + H2)

**Goal.** Smallest-possible architectural change that addresses W1 (sigmoid output) and W2 (mixed activations) without growing capacity. Variant: a 1-line patch on top of `NICON-baseline` — same backbone, replace the final `Dense(1, sigmoid)` with linear and the SELU / ReLU / ELU mix with a single GELU + LayerNorm.

**Hypotheses.** H1, H2.

**Files.** `nicon_v2/models/v1a_minimal_repair.py`, `tests/test_v1a_*.py`.

**Acceptance.** Median rmsep − previous (NICON-baseline) ≤ −3 %, paired Wilcoxon p < 0.05 across the curated cohort, no per-dataset regression > 5 %.

## Phase 1b — Concat-derivatives + augmentation (H5 + H6 + H7)

**Goal.** Add the SOTA NIRS recipe (concat-derivatives input, Bjerrum-style augmentation, C-Mixup) **before** any capacity-increasing architecture change, so subsequent capacity wins can be attributed to architecture, not augmentation. Ranks H6/H7 high because the cohort is dominated by small-n datasets (Beer 40, Biscuit 40, etc.).

**Hypotheses.** H5, H6, H7.

**Files.**
* `nicon_v2/preprocessing.py` — fixed Savitzky-Golay derivative kernels (Conv1D, no_grad), SNV, MSC, concat helper.
* `nicon_v2/augmentation.py` — Bjerrum offset/slope/multiplicative, C-Mixup, contiguous-band masking.
* `nicon_v2/models/v1b_concat_aug.py`.
* tests for preprocessing parity vs scipy and seed-determinism for augmentation.

**Acceptance.** Median rmsep − Phase-1a ≤ −4 %, p < 0.05; ablations report H5, H6, H7 individual effects; H7 beats vanilla mixup on ≥ 60 % of datasets *or* is rejected.

## Phase 1c — Small-kernel + GAP backbone (H3 + H4)

**Goal.** Replace the stride-only large-kernel backbone with the (Cui & Fearn 2018) recipe: 4 small-kernel blocks + max-pool 2 + GAP + linear head, with norm choice (LayerNorm / GroupNorm / BatchNorm) studied via H4.

**Hypotheses.** H3, H4.

**Files.** `nicon_v2/models/v1c_gap_backbone.py`, `tests/test_v1c_*.py`, length-robustness tests on `{401, 576, 700, 1154, 2151}`.

**Acceptance.** Median rmsep − Phase-1b ≤ −5 %, p < 0.05; no per-dataset regression > 5 %; LayerNorm vs BatchNorm A/B documented; the new variant is the **Phase-1 winner** that subsequent phases are paired against.

## Phase 2 — Multi-scale Inception (H8) with capacity guard

**Goal.** Add a single Inception block (parallel 1×1, 1×3, 1×5, pool branches) as in DeepSpectra (Zhang 2019). Capacity-guarded: gradient norm, train/val gap, and per-dataset regressions are logged. Reject if `n_train ≤ 500` regresses > 5 %.

**Hypothesis.** H8.

**Files.** `nicon_v2/models/v2_inception.py`, capacity-guard logging in `training.py`.

**Acceptance.** On `n_train > 500` sub-cohort, median rmsep − Phase-1c ≤ −3 %, p < 0.05. On `n_train ≤ 500` sub-cohort, no significant regression (p > 0.05 or median Δ ≥ −1 %).

## Phase 3a — Deep ensembles (H9 — accuracy half)

**Goal.** Train 5-seed deep ensembles (Lakshminarayanan 2017). Reports `aleatoric_var` and `epistemic_var` separately.

**Hypothesis.** H9 (RMSEP arm only).

**Files.** `nicon_v2/training.py::train_ensemble`, `nicon_v2/uncertainty.py::ensemble_predict`.

**Acceptance.** Median rmsep − Phase-2 ≤ −2 %, p < 0.05.

## Phase 3b — Conformal calibration (H9 — UQ half)

**Goal.** Add split-conformal calibration on the held-out `C` set; report `coverage_90`, `width_90`, `interval_score_90`, `crps`.

**Hypothesis.** H9 (UQ arm).

**Files.** `nicon_v2/uncertainty.py::conformal_calibrate`, `tests/test_conformal.py`.

**Acceptance.** Empirical 90 % coverage ∈ [0.85, 0.95] on ≥ 80 % of cohort datasets; median `interval_score_90` ≤ a fixed-σ baseline by ≥ 5 %; no rmsep regression vs Phase 3a.

## Phase 4 — Learnable preprocessing (H10)

**Goal.** Replace fixed SG kernels with learnable EMSC + learnable SG (Helin 2022) and study the trade-off vs the deterministic concat-derivatives front.

**Hypothesis.** H10.

**Files.** `nicon_v2/preprocessing.py::LearnableEMSC`, `LearnableSGWindow`; `nicon_v2/models/v4_learnable_preproc.py`.

**Acceptance.** Match or beat the deterministic Phase-1b H5 within 1 % rmsep median, no per-dataset regression > 2 %.

## Phase 5 — TabPFN head / stacking (H11, H12)

**Goal.** Use TabPFN-v2 or AOM-PLS as a meta-learner over GAP features.

**Hypotheses.** H11, H12.

**Files.** `nicon_v2/models/v5_tabpfn_head.py`, `nicon_v2/models/v6_stacking.py`.

**Acceptance.** On `n_train ≤ 200` sub-cohort, median rmsep − Phase-3a ≤ −1 %, p < 0.05; H12 stacking improves median rmsep ≥ 1 % over either parent.

---

## Pre-registered ablation matrix

Before any phase runs we lock the ablation table. Each row is a **single-control change** vs a frozen "minimal viable baseline" (`nicon_v2-V0 = NICON-baseline + linear head`, the Phase-1a variant).

| Cell | Backbone | Norm | Activation | Input | Augmentation | Mixup | Head | Notes |
|------|----------|------|------------|-------|--------------|-------|------|-------|
| MVB  | nicon-baseline | mixed | mixed | raw | none | none | linear | Phase-1a accepted variant |
| A1 (H2) | mvb | LayerNorm | GELU | raw | none | none | linear | activation/norm cohesion |
| A2 (H5) | mvb | LayerNorm | GELU | concat-deriv | none | none | linear | concat-derivatives |
| A3 (H6) | mvb | LayerNorm | GELU | concat-deriv | Bjerrum | none | linear | EMSC-aug |
| A4 (H7v) | A3 | – | – | – | – | mixup | – | vanilla mixup |
| A4 (H7c) | A3 | – | – | – | – | C-Mixup | – | label-aware mixup |
| A4 (H6m) | A3 | – | – | – | + band-mask | – | – | spectral band masking |
| A5 (H3) | gap-4-block | LayerNorm | GELU | concat-deriv | Bjerrum | C-Mixup | linear | small-kernel backbone |
| A6 (H4-LN) | A5 | LayerNorm | GELU | – | – | – | – | norm A/B |
| A6 (H4-BN) | A5 | BatchNorm | GELU | – | – | – | – | norm A/B |
| A6 (H4-GN) | A5 | GroupNorm | GELU | – | – | – | – | norm A/B |
| A7 (H8)  | A5+inception | LayerNorm | GELU | – | – | – | – | multi-scale |
| A8 (H10) | A5 | LayerNorm | GELU | learnable-EMSC | Bjerrum | C-Mixup | linear | learnable preproc |
| A9 (H9)  | 5-ensemble of A5 | – | – | – | – | – | – | deep-ensemble |
| A10 (H11) | A5 | – | – | – | – | – | TabPFN head | meta-learner |
| A11 (H12) | A5 ⊕ AOM-PLS | – | – | – | – | – | Ridge meta | stacking |

Each ablation run is paired against `MVB` (and against the immediately preceding accepted variant) on every cohort dataset and seed. The matrix is appended to `publication/tables/ablation.csv`.

---

## Phase 6 — Final benchmark, manuscript, figures

**Goal.** Frozen final variant; full cohort run; compute LaTeX tables and figures (mirroring AOM-PLS publication scaffolding).

**Files.**
* `publication/scripts/make_figures.py` — critical-difference, per-dataset delta, cumulative RMSEP, cost-vs-precision, calibration curves.
* `publication/scripts/make_tables.py` — main table, ablation table, operator/architecture comparison table.
* `publication/manuscript/PAPER_DRAFT.md`, `publication/manuscript/main.tex`.
* `publication/manuscript/references.bib` — start from `source_materials/literature_review/references.bib`.

**Acceptance.**
* All cohort runs reproduce within 1 % rmsep across `seeds = [0, 1, 2]`.
* Manuscript draft passes Codex publication review.

---

## Cross-cutting

* **Codex review prompts** are kept under `docs/codex_review_prompts/`. Each iteration sends the relevant prompt to Codex via the `codex:rescue` agent.
* **Implementation log** (`docs/IMPLEMENTATION_LOG.md`) is append-only; each iteration adds a section.
* **Wall-clock budget** (Prompt.md): max 8 phases, max 12 GPU-hours per phase on the curated cohort, max 5-member ensemble; futility stop after 2 consecutive non-significant phases.
* **No edits to `nirs4all`.**
