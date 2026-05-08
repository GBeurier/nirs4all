# AOM-Ridge Big-n OOM — Diagnosis and Mitigation

**Owner**: Agent A (Classical Production Pack)
**Date**: 2026-05-05 (revised 17:30 CEST after Codex round 1)
**Status**: `DECISION_LOCKED` (Codex round 1: CONFIRM substitute) with two corrections applied below.
**Scope**: full-57 cohort runs of `AOMRidgeRegressor` (kernel ridge formulation) and the `aom-ridge-fast` integration in `bench/AOM_v0/multiview/` hetero-stack.

This document closes the partial-run question raised in `bench/PLAN_REPRISE_2026-05.md` §6 A3 ("AOM-Ridge full-57 | OOM LMA | diagnostiquer mémoire, relancer ou produire variante downsampled `exploratory`").

---

## 1. Observed coverage

Source: `bench/benchmark_master_results.csv` (after P0 freeze, 2026-05-05 14:30).

| Variant | source_run | Datasets | Status |
|---|---|---:|---|
| `aom-ridge-fast` (multiview hetero stack) | `bench/AOM_v0/multiview/results/full57.csv` | 36 / 57 | `protocol_maturity=locked` (per freeze) |
| `aom-ridge-standalone` | same source | 7 / 57 | `protocol_maturity=locked` |
| `AOMRidge-global-compact-none` (Ridge benchmark family) | `bench/AOM_v0/Ridge/benchmark_runs/all54_*` | 54 / 57 | `protocol_maturity=locked` |

The standalone Ridge benchmark `all54_*` covers 54 datasets (the canonical `RETHOUGHT_SUBSETS.md` 4-dataset coverage gap accounts for the missing 3 — see §3 below). The big-n problem is **specific to the multiview hetero-stack runs** that fold `aom-ridge-fast` as a base into NNLS / Ridge stacking, where the cumulative kernel-matrix memory across folds × atoms blows up.

## 2. Failure mechanism

`aomridge.estimators.AOMRidgeRegressor` operates on a kernel matrix of shape `(n, n)` per AOM block (because the linear ridge dual solution stores `α ∈ R^n`). For `n = 39 225` (LMA_spxyG_block2deg), the dense kernel takes:

```text
n × n × 8 bytes = 39 225² × 8 ≈ 12.3 GB per block
× num_blocks (compact bank: 9 operators) = ~110 GB peak
```

Even with `--n-max=8000` cap, multiview hetero stack cumulates concurrent blocks across atoms (`AOM-PLS-compact + multiK-3-5-7 + moe-preproc-soft + lazy-V2-AOM`) plus 5-fold OOF, producing peak working sets that exceed the RTX 4090 box's 64 GB system RAM.

Per `bench/AOM_v0/multiview/docs/SUMMARY.md §11`:

- **`LMA_spxyG_block2deg`** (n = 39 225, p = 2 151) → OOM, killed.
- **`LUCAS_SOC_Cropland_8731_NocitaKS`** (n = 6 111, p = 4 200) → 40+ minute stall, killed.

The remaining ~21 missing datasets in `aom-ridge-fast` come from:
- the `RETHOUGHT_SUBSETS.md` 4-dataset coverage gap (Brix_spxy70 etc.),
- a handful of mid-n datasets (n ≈ 1 500-3 000) that exceeded the per-dataset 5-min budget in the multiview launcher.

## 3. Affected dataset classes

| Class | Threshold | Datasets in cohort |
|---|---|---|
| Big-n (kernel infeasible) | `n_train > 3 000` | LMA_spxyG_block2deg, LUCAS_SOC_Cropland_8731_NocitaKS, brix_groupSampleID_*, ph_*, ta_*, Beef_Marbling_RandomSplit, Milk_Lactose/Urea_1224, Ccar_spxyG_block2deg, Chla+b_spxyG_*, COLZA C_woOutlier / N_w*, BERRY brix |
| Mid-n (feasible but slow) | `1 500 < n_train ≤ 3 000` | a subset of LUCAS variants, manure rows |
| Coverage caution | small but X-format incompatible with current pipeline | Brix_spxy70, LUCAS_SOC_Cropland_8731_NocitaKS, Malaria_Oocist_333_Maia, Malaria_Sporozoite_229_Maia |

Local-knn50 (`aomridge.local_ridge.AOMLocalRidge`, class verified at
`local_ridge.py:404`) runs on every dataset in the cohort because its memory
is `O(n · k · p)` with `k = 50`: for n = 39 225, p = 2 151 → peak ~330 MB
per block, fits in cache.

## 4. Locked remediation — two-entry registry, not auto-fallback

**Codex round 1 verdict (D-A-003)**: confirm substitute, but the registry
**must expose two separate entries** rather than a single canonical entry
with `n_train`-routed auto-fallback. Rationale: pairing a kernel-ridge row
with a local-knn50 row inside one statistical group hides a model-family
change behind a sample-size boundary. §10.2 paired tests (Wilcoxon,
Nadeau-Bengio) require apples-to-apples comparison, so the two estimators
must remain reported and compared as distinct candidates.

The current registry already implements this (`bench/scenarios/model_registry.yaml`):

- `AOMRidge-global-compact-none` — `model_class: AOMRidgeRegressor`,
  `module: aomridge.estimators`, `input_constraints: {min_n: 30, max_n: 3000}`.
  Stays out of scope above n=3000.
- `AOMRidge-Local-compact-knn50` — `model_class: AOMLocalRidge`,
  `module: aomridge.local_ridge`, `input_constraints: {min_n: 50}`. Eligible
  on every dataset in the cohort, including big-n.

The harness skips a candidate when `n_train > max_n` per the input_constraints
contract. No new registry entry is needed; only the **completion run** below.

### 4.1 Completion run for big-n datasets

Run the local-knn50 candidate on the 21 datasets where the kernel-ridge
candidate is skipped or absent:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge \
python bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py \
    --variants local_only \
    --datasets bigN_completion_cohort.csv \
    --workspace bench/AOM_v0/Ridge/benchmark_runs/bigN_local_completion \
    --seed 0
```

`bigN_completion_cohort.csv` lists the 21 datasets where `aom-ridge-fast`
did not produce a row.

### 4.2 Reporting rule

Score-cards (per plan §10.1) report both candidates separately. The kernel
ridge row is **absent** on big-n datasets and the synthesis must surface
this gap explicitly (Agent C exporter's `coverage_fraction` already does so
via the `low_coverage` penalty when fraction < 0.40). The local-knn50 row
provides full coverage.

### 4.3 Master CSV tagging

- Local-knn50 rows on the 21 big-n datasets → `protocol_maturity=locked`.
- Kernel-ridge `aom-ridge-fast` rows for those datasets stay absent. No
  synthetic placeholder is needed because the registry's `max_n: 3000`
  constraint documents the rationale and the exporter's coverage penalty
  surfaces it.

## 5. Alternative (rejected by Codex)

The previous draft considered (a) a single canonical registry entry with
auto-fallback at `n_train > 3000` and (b) capping `--n-max=3 000` on the
kernel runner with synthetic placeholder rows. Both are **rejected** by
Codex round 1 because they conflate two estimator families inside one
statistical comparison group. The two-entry approach above replaces them.

## 6. Acceptance criteria

The big-n issue is closed when **all** of the following hold:

- Every full-57 cohort dataset has at least an `AOMRidge-Local-compact-knn50`
  row in the master with `protocol_maturity=locked`.
- The kernel-ridge entry `AOMRidge-global-compact-none` keeps its
  `max_n: 3000` constraint; the harness honours it via the input_constraints
  contract.
- Score-cards report kernel-ridge and local-knn50 as **separate** candidates
  (no statistical pooling).
- A two-paragraph note in `bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md`
  records the locked decision and the date.

## 7. Decision history

- 2026-05-05 — D-A-003 posted with status `DECISION_PENDING_CODEX_REVIEW`,
  default substitute recommendation.
- 2026-05-05 17:00 CEST — Codex round 1: CONFIRM substitute, with two
  corrections (class name `AOMLocalRidge` not `AOMRidgeLocalRegressor`;
  registry must use two separate entries). Both corrections applied above.
- 2026-05-05 17:30 CEST — D-A-003 status flipped to `DECISION_LOCKED`.
  Implementation = the completion run §4.1.
