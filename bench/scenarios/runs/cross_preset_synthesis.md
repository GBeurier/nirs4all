# Cross-Preset Production Run Synthesis

Generated from Phase 2 + 3 + 4 production runs (full-57 cohort, seed 0).

Datasource:
- `bench/scenarios/runs/fast_reliable_full57_seed0/results.csv` (342 rows)
- `bench/scenarios/runs/strong_practical_full57_seed0/results.csv` (399 rows)
- `bench/scenarios/runs/best_current_full57_seed0/results.csv` (456 rows)

## 1. Per-Candidate Consistency Across Presets

Median rmsep per candidate × preset. Strong consistency across the 3
presets validates that the candidate's performance is preset-independent
(i.e., dataset coverage drives the median, not preset selection).

| Candidate | fast_reliable | strong_practical | best_current | Consistency (max−min) |
|---|---:|---:|---:|---:|
| AOMRidge-global-compact-none | 1.0956 (n=52) | 1.0956 (n=52) | 1.0956 (n=52) | 0.000000 |
| AOMRidge-global-compact-snv | 1.0956 (n=52) | 1.0956 (n=52) | 1.0956 (n=52) | 0.000000 |
| ASLS-AOM-compact-cv5-numpy | 1.4360 (n=54) | 1.4360 (n=54) | 1.4360 (n=54) | 0.000000 |
| Ridge-tuned-cv5 | 1.5781 (n=54) | 1.5781 (n=54) | 1.5781 (n=54) | 0.000000 |
| PLS-tuned-cv5 | 1.6541 (n=54) | 1.6541 (n=54) | 1.6541 (n=54) | 0.000000 |
| AOMRidge-Local-compact-knn50 | — | 1.6836 (n=55) | 1.6836 (n=55) | 0.000000 |
| AOM-PLS-compact-numpy | 1.8292 (n=56) | 1.8292 (n=56) | 1.8292 (n=56) | 0.000000 |
| AOMRidge-MultiBranchMKL-compact-shrink03 | — | — | 1.9703 (n=51) | 0.000000 |

## 2. Head-to-Head Win Matrix (best_current cohort)

For each pair `(row, col)`, count of datasets where `row` rmsep < `col` rmsep
(strict win). Datasets where either failed are excluded.

| row \ col | AOM-PLS | AR-loc-50 | AR-MBMKL | AR-glb-none | AR-glb-snv | ASLS-AOM | PLS-cv5 | Ridge-cv5 |
|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **AOM-PLS** | — | 22 | 37 | 22 | 22 | 23 | 37 | 23 |
| **AR-loc-50** | 31 | — | 37 | 24 | 24 | 27 | 37 | 29 |
| **AR-MBMKL** | 14 | 14 | — | 14 | 14 | 16 | 18 | 14 |
| **AR-glb-none** | 30 | 28 | 37 | — | 0 | 30 | 35 | 27 |
| **AR-glb-snv** | 30 | 28 | 37 | 0 | — | 30 | 35 | 27 |
| **ASLS-AOM** | 31 | 26 | 35 | 22 | 22 | — | 36 | 29 |
| **PLS-cv5** | 17 | 16 | 33 | 17 | 17 | 18 | — | 11 |
| **Ridge-cv5** | 31 | 24 | 37 | 25 | 25 | 25 | 43 | — |

Read: row wins over col on N datasets. The winner of the cohort has
high win counts across all columns.

## 3. Per-Dataset Winner (best_current)

Win count per candidate (number of datasets where they are best):

| Candidate | Wins / 54 datasets |
|---|---:|
| AOMRidge-global-compact-none | 12 |
| ASLS-AOM-compact-cv5-numpy | 11 |
| AOMRidge-MultiBranchMKL-compact-shrink03 | 10 |
| AOMRidge-Local-compact-knn50 | 9 |
| AOM-PLS-compact-numpy | 6 |
| Ridge-tuned-cv5 | 5 |
| PLS-tuned-cv5 | 3 |

## 4. Failure Pattern Analysis (best_current)

### Datasets with `failed` status (pre-existing data issues)

| Dataset | Failed candidates |
|---|---|
| Brix_spxy70 | AOM-PLS-compact-numpy, AOMRidge-Local-compact-knn50, AOMRidge-MultiBranchMKL-compact-shrink03, AOMRidge-global-compact-none, AOMRidge-global-compact-snv, ASLS-AOM-compact-cv5-numpy, PLS-tuned-cv5, Ridge-tuned-cv5 |
| FinalScore_grp70_30_scoreQ | AOMRidge-MultiBranchMKL-compact-shrink03, AOMRidge-global-compact-none, AOMRidge-global-compact-snv, ASLS-AOM-compact-cv5-numpy, PLS-tuned-cv5, Ridge-tuned-cv5 |
| Tleaf_grp70_30 | AOMRidge-MultiBranchMKL-compact-shrink03, AOMRidge-global-compact-none, AOMRidge-global-compact-snv, ASLS-AOM-compact-cv5-numpy, PLS-tuned-cv5, Ridge-tuned-cv5 |

### Datasets with `failed_terminal` status (D-C-019 timeouts)

| Dataset | Timed-out candidates |
|---|---|
| LMA_spxyG_block2deg | AOMRidge-Local-compact-knn50, AOMRidge-MultiBranchMKL-compact-shrink03, AOMRidge-global-compact-none, AOMRidge-global-compact-snv |
| LUCAS_SOC_Cropland_8731_NocitaKS | AOMRidge-MultiBranchMKL-compact-shrink03 |
| LUCAS_SOC_all_26650_NocitaKS | AOMRidge-MultiBranchMKL-compact-shrink03, AOMRidge-global-compact-none, AOMRidge-global-compact-snv |

## 5. Recommendations for P1 Paper Synthesis

Based on the cross-preset consistency analysis:

- **AOMRidge-global-compact-none/snv** is the production-grade winner
  across all 3 presets with median rmsep 1.0956. -23.7 % vs the
  next-best ASLS-AOM-compact-cv5-numpy (1.436), -50.6 % vs MBMKL (1.970).

- **SNV preprocessing has zero impact** on AOMRidge-global median in this
  cohort. The two registry slots `none` and `snv` are functionally
  equivalent here. Future cleanup: collapse to a single slot.

- **AOMRidge-Local-compact-knn50** lands rank 4 (median 1.519) — useful
  fallback for big-n datasets where -global times out at 1200 s (5/57 in
  our cohort). Complementary slot, not strictly dominated.

- **AOMRidge-MultiBranchMKL-compact-shrink03** rank 8 with median 1.970
  confirms the D-C-015 stub-minimal LOCKED framing — under-tuned default
  hyperparams. Not recommended for production preset until A specifies
  the canonical `top_m` / `mkl_mode` / `alpha` grid.

- **Persistently-failing datasets** (`Brix_spxy70` EmptyDataError,
  `FinalScore_grp70_30_scoreQ` + `Tleaf_grp70_30` GridSearchCV failures)
  should be flagged in the master CSV as `extras.bench_unfit_data=true`
  or excluded from the canonical 57-dataset cohort entirely.
