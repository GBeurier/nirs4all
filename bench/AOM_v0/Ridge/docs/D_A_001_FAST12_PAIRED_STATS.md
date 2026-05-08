# D-A-001 fast12 paired statistics (Codex round-6 GATE)

Source: `bench/AOM_v0/Ridge/benchmark_runs/da001_partial_fast12_seeds012/results.csv`

Cohort: fast12_transfer_core x seeds 0/1/2 (12 datasets x 3 seeds = 36 rows per candidate).

Wilcoxon = paired one-sided (target < baseline) on log RMSEP deltas. Primary unit = per-dataset seed-mean (N=12). Row-level (N=36) reported as sensitivity. Holm correction across the 4 x 2 = 8 comparisons.

Win threshold conventions (from Codex round-6, SYNC 04:10 CEST):
- median_delta_pct_ds <= -3 % (preferably <= -5 % for headline language)
- |cliffs_delta_ds| >= 0.147 in the favourable direction
- q90_ratio_ds <= 1.10 (no-harm sanity check)
- p_wilcoxon_ds_holm < 0.05

## Summary table

| Selector | Baseline | N_ds | Wins/12 | Wins/36 | Median Δ% | q75 ratio | q90 ratio | Worst ratio (dataset) | Cliff's δ | p (ds, Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AOMRidge-Blender-headline-spxy3 | Ridge-tuned-cv5 | 12 | 8 | 24 | -13.55 | 1.002 | 1.045 | 1.107 (Biscuit_Sucrose_40_RandomSplit) | +0.333 | 0.0671 | NO_WIN |
| AOMRidge-Blender-headline-spxy3 | ASLS-AOM-compact-cv5-numpy | 12 | 10 | 30 | -11.29 | 0.945 | 1.088 | 1.114 (MP_spxyG) | +0.667 | 0.0483 | WIN_strong |
| AOMRidge-Blender-headline-spxy3 | AOMRidge-global-compact-none | 12 | 9 | 27 | -14.17 | 0.983 | 1.095 | 1.397 (Biscuit_Sucrose_40_RandomSplit) | +0.500 | 0.1543 | NO_WIN |
| AOMRidge-Blender-headline-spxy3 | AOMRidge-Local-compact-knn50 | 12 | 9 | 27 | -4.63 | 0.984 | 1.032 | 1.036 (Biscuit_Sucrose_40_RandomSplit) | +0.500 | 0.0427 | WIN_practical |
| AOMRidge-AutoSelect-headline-spxy3 | Ridge-tuned-cv5 | 12 | 8 | 24 | -10.44 | 1.009 | 1.149 | 1.189 (Ccar_spxyG_block2deg) | +0.333 | 0.1543 | NO_WIN |
| AOMRidge-AutoSelect-headline-spxy3 | ASLS-AOM-compact-cv5-numpy | 12 | 10 | 30 | -17.58 | 0.931 | 1.053 | 1.076 (TIC_spxy70) | +0.667 | 0.0195 | WIN_strong |
| AOMRidge-AutoSelect-headline-spxy3 | AOMRidge-global-compact-none | 12 | 9 | 27 | -10.65 | 1.030 | 1.195 | 1.466 (Biscuit_Sucrose_40_RandomSplit) | +0.500 | 0.1543 | NO_WIN |
| AOMRidge-AutoSelect-headline-spxy3 | AOMRidge-Local-compact-knn50 | 12 | 9 | 27 | -3.54 | 0.998 | 1.034 | 1.088 (Biscuit_Sucrose_40_RandomSplit) | +0.500 | 0.1543 | NO_WIN |

## Per-comparison detail

### AOMRidge-Blender-headline-spxy3

#### vs Ridge-tuned-cv5

- Rows kept: 36 (out of 36 per side); datasets kept: 12 (out of 12)
- Wins (per-row, N=36): 24/36 ; Wins (per-dataset, N=12): 8/12
- Median ratio (ds): 0.864  (median Δ% = -13.55 %)
- q75 / q90 / worst ratio (ds): 1.002 / 1.045 / 1.107
- Worst-regression dataset: **Biscuit_Sucrose_40_RandomSplit** (ratio = 1.107)
- Cliff's δ (ds, paired): +0.333
- Wilcoxon (ds, one-sided less): p = 0.0134 -> Holm-adjusted = 0.0671
- Wilcoxon (rows, one-sided less): p = 0.0001 -> Holm-adjusted = 0.0003

#### vs ASLS-AOM-compact-cv5-numpy

- Rows kept: 36 (out of 36 per side); datasets kept: 12 (out of 12)
- Wins (per-row, N=36): 30/36 ; Wins (per-dataset, N=12): 10/12
- Median ratio (ds): 0.887  (median Δ% = -11.29 %)
- q75 / q90 / worst ratio (ds): 0.945 / 1.088 / 1.114
- Worst-regression dataset: **MP_spxyG** (ratio = 1.114)
- Cliff's δ (ds, paired): +0.667
- Wilcoxon (ds, one-sided less): p = 0.0081 -> Holm-adjusted = 0.0483
- Wilcoxon (rows, one-sided less): p = 0.0000 -> Holm-adjusted = 0.0001

#### vs AOMRidge-global-compact-none

- Rows kept: 36 (out of 36 per side); datasets kept: 12 (out of 12)
- Wins (per-row, N=36): 27/36 ; Wins (per-dataset, N=12): 9/12
- Median ratio (ds): 0.858  (median Δ% = -14.17 %)
- q75 / q90 / worst ratio (ds): 0.983 / 1.095 / 1.397
- Worst-regression dataset: **Biscuit_Sucrose_40_RandomSplit** (ratio = 1.397)
- Cliff's δ (ds, paired): +0.500
- Wilcoxon (ds, one-sided less): p = 0.0386 -> Holm-adjusted = 0.1543
- Wilcoxon (rows, one-sided less): p = 0.0009 -> Holm-adjusted = 0.0037

#### vs AOMRidge-Local-compact-knn50

- Rows kept: 36 (out of 36 per side); datasets kept: 12 (out of 12)
- Wins (per-row, N=36): 27/36 ; Wins (per-dataset, N=12): 9/12
- Median ratio (ds): 0.954  (median Δ% = -4.63 %)
- q75 / q90 / worst ratio (ds): 0.984 / 1.032 / 1.036
- Worst-regression dataset: **Biscuit_Sucrose_40_RandomSplit** (ratio = 1.036)
- Cliff's δ (ds, paired): +0.500
- Wilcoxon (ds, one-sided less): p = 0.0061 -> Holm-adjusted = 0.0427
- Wilcoxon (rows, one-sided less): p = 0.0000 -> Holm-adjusted = 0.0001

### AOMRidge-AutoSelect-headline-spxy3

#### vs Ridge-tuned-cv5

- Rows kept: 36 (out of 36 per side); datasets kept: 12 (out of 12)
- Wins (per-row, N=36): 24/36 ; Wins (per-dataset, N=12): 8/12
- Median ratio (ds): 0.896  (median Δ% = -10.44 %)
- q75 / q90 / worst ratio (ds): 1.009 / 1.149 / 1.189
- Worst-regression dataset: **Ccar_spxyG_block2deg** (ratio = 1.189)
- Cliff's δ (ds, paired): +0.333
- Wilcoxon (ds, one-sided less): p = 0.0461 -> Holm-adjusted = 0.1543
- Wilcoxon (rows, one-sided less): p = 0.0011 -> Holm-adjusted = 0.0037

#### vs ASLS-AOM-compact-cv5-numpy

- Rows kept: 36 (out of 36 per side); datasets kept: 12 (out of 12)
- Wins (per-row, N=36): 30/36 ; Wins (per-dataset, N=12): 10/12
- Median ratio (ds): 0.824  (median Δ% = -17.58 %)
- q75 / q90 / worst ratio (ds): 0.931 / 1.053 / 1.076
- Worst-regression dataset: **TIC_spxy70** (ratio = 1.076)
- Cliff's δ (ds, paired): +0.667
- Wilcoxon (ds, one-sided less): p = 0.0024 -> Holm-adjusted = 0.0195
- Wilcoxon (rows, one-sided less): p = 0.0000 -> Holm-adjusted = 0.0000

#### vs AOMRidge-global-compact-none

- Rows kept: 36 (out of 36 per side); datasets kept: 12 (out of 12)
- Wins (per-row, N=36): 27/36 ; Wins (per-dataset, N=12): 9/12
- Median ratio (ds): 0.893  (median Δ% = -10.65 %)
- q75 / q90 / worst ratio (ds): 1.030 / 1.195 / 1.466
- Worst-regression dataset: **Biscuit_Sucrose_40_RandomSplit** (ratio = 1.466)
- Cliff's δ (ds, paired): +0.500
- Wilcoxon (ds, one-sided less): p = 0.0881 -> Holm-adjusted = 0.1543
- Wilcoxon (rows, one-sided less): p = 0.0081 -> Holm-adjusted = 0.0081

#### vs AOMRidge-Local-compact-knn50

- Rows kept: 36 (out of 36 per side); datasets kept: 12 (out of 12)
- Wins (per-row, N=36): 27/36 ; Wins (per-dataset, N=12): 9/12
- Median ratio (ds): 0.965  (median Δ% = -3.54 %)
- q75 / q90 / worst ratio (ds): 0.998 / 1.034 / 1.088
- Worst-regression dataset: **Biscuit_Sucrose_40_RandomSplit** (ratio = 1.088)
- Cliff's δ (ds, paired): +0.500
- Wilcoxon (ds, one-sided less): p = 0.0386 -> Holm-adjusted = 0.1543
- Wilcoxon (rows, one-sided less): p = 0.0009 -> Holm-adjusted = 0.0037

## Pre-registered descriptive Friedman-Nemenyi (5 AOMRidge variants)

Variants pre-registered before looking at ranks: `AOMRidge-global-compact-none`, `AOMRidge-Local-compact-knn50`, `AOMRidge-MultiBranchMKL-compact-shrink03`, `AOMRidge-Blender-headline-spxy3`, `AOMRidge-AutoSelect-headline-spxy3`.

Rows with all 5 variants OK: 36 (out of 36 possible).

Friedman: chi^2 = 51.800, p = 0.0000 (descriptive only; omnibus is reserved for the production/full-57 escalation).

Mean rank per variant (1 = best):
- `AOMRidge-global-compact-none`: 3.333
- `AOMRidge-Local-compact-knn50`: 3.167
- `AOMRidge-MultiBranchMKL-compact-shrink03`: 4.333
- `AOMRidge-Blender-headline-spxy3`: 2.083
- `AOMRidge-AutoSelect-headline-spxy3`: 2.083

## Selector diagnostics

Selector diagnostics (AutoSelect chosen-candidate counts, Blender weight mean/std, OOF fold RMSE variance) are not surfaced in this results.csv schema; the existing harness logs them only to per-run JSON sidecars under `benchmark_runs/.../<canonical_name>/`. A follow-up pass to aggregate those into a markdown table is staged but out of scope for this initial GATE deliverable.