# D-A-001 audit20 paired statistics (Codex round-8 GATE)

Source: `bench/AOM_v0/Ridge/benchmark_runs/da001_audit20_seeds012/results.csv`

Cohort: audit20_transfer_core x seeds 0/1/2 (20 datasets x 3 seeds = 60 rows per candidate).

Wilcoxon = paired one-sided (target < baseline) on log RMSEP deltas. Primary unit = per-dataset seed-mean (N=20). Row-level (N=60) reported as sensitivity. Holm correction across the 4 x 2 = 8 comparisons.

Win threshold conventions (from Codex round-6/7):
- median_delta_pct_ds <= -3 % (preferably <= -5 % for headline language)
- |cliffs_delta_ds| >= 0.147 in the favourable direction
- q90_ratio_ds <= 1.10 (no-harm sanity check)
- p_wilcoxon_ds_holm < 0.05

## Summary table

| Selector | Baseline | N_ds | Wins/20 | Wins/60 | Median Δ% | q75 ratio | q90 ratio | Worst ratio (dataset) | Cliff's δ | p (ds, Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AOMRidge-Blender-headline-spxy3 | Ridge-tuned-cv5 | 20 | 17 | 50 | -8.26 | 0.972 | 1.012 | 127.254 (Quartz_spxy70) | +0.700 | 0.0148 | WIN_strong |
| AOMRidge-Blender-headline-spxy3 | ASLS-AOM-compact-cv5-numpy | 20 | 15 | 45 | -4.71 | 1.012 | 1.100 | 16851.112 (Quartz_spxy70) | +0.500 | 0.1238 | NO_WIN |
| AOMRidge-Blender-headline-spxy3 | AOMRidge-global-compact-none | 20 | 15 | 45 | -11.40 | 0.985 | 1.078 | 37059.028 (Quartz_spxy70) | +0.500 | 0.0479 | WIN_strong |
| AOMRidge-Blender-headline-spxy3 | AOMRidge-Local-compact-knn50 | 20 | 14 | 42 | -5.38 | 1.035 | 1.106 | 254.700 (Quartz_spxy70) | +0.400 | 0.1238 | NO_WIN |
| AOMRidge-AutoSelect-headline-spxy3 | Ridge-tuned-cv5 | 20 | 18 | 53 | -7.22 | 0.959 | 1.012 | 1.189 (Ccar_spxyG_block2deg) | +0.800 | 0.0126 | WIN_strong |
| AOMRidge-AutoSelect-headline-spxy3 | ASLS-AOM-compact-cv5-numpy | 20 | 14 | 42 | -5.52 | 1.025 | 1.104 | 1.437 (Biscuit_Sucrose_40_RandomSplit) | +0.400 | 0.1238 | NO_WIN |
| AOMRidge-AutoSelect-headline-spxy3 | AOMRidge-global-compact-none | 20 | 16 | 48 | -9.75 | 0.976 | 1.043 | 1.466 (Biscuit_Sucrose_40_RandomSplit) | +0.600 | 0.0283 | WIN_strong |
| AOMRidge-AutoSelect-headline-spxy3 | AOMRidge-Local-compact-knn50 | 20 | 14 | 42 | -4.46 | 1.008 | 1.027 | 1.088 (Biscuit_Sucrose_40_RandomSplit) | +0.400 | 0.0283 | WIN_practical |

## Per-comparison detail

### AOMRidge-Blender-headline-spxy3

#### vs Ridge-tuned-cv5

- Rows kept: 60 (out of 60 per side); datasets kept: 20 (out of 20)
- Wins (per-row, N=60): 50/60 ; Wins (per-dataset, N=20): 17/20
- Median ratio (ds): 0.917  (median Δ% = -8.26 %)
- q75 / q90 / worst ratio (ds): 0.972 / 1.012 / 127.254
- Worst-regression dataset: **Quartz_spxy70** (ratio = 127.254)
- Cliff's δ (ds, paired): +0.700
- Wilcoxon (ds, one-sided less): p = 0.0021 -> Holm-adjusted = 0.0148
- Wilcoxon (rows, one-sided less): p = 0.0000 -> Holm-adjusted = 0.0000

#### vs ASLS-AOM-compact-cv5-numpy

- Rows kept: 60 (out of 60 per side); datasets kept: 20 (out of 20)
- Wins (per-row, N=60): 45/60 ; Wins (per-dataset, N=20): 15/20
- Median ratio (ds): 0.953  (median Δ% = -4.71 %)
- q75 / q90 / worst ratio (ds): 1.012 / 1.100 / 16851.112
- Worst-regression dataset: **Quartz_spxy70** (ratio = 16851.112)
- Cliff's δ (ds, paired): +0.500
- Wilcoxon (ds, one-sided less): p = 0.0448 -> Holm-adjusted = 0.1238
- Wilcoxon (rows, one-sided less): p = 0.0017 -> Holm-adjusted = 0.0037

#### vs AOMRidge-global-compact-none

- Rows kept: 60 (out of 60 per side); datasets kept: 20 (out of 20)
- Wins (per-row, N=60): 45/60 ; Wins (per-dataset, N=20): 15/20
- Median ratio (ds): 0.886  (median Δ% = -11.40 %)
- q75 / q90 / worst ratio (ds): 0.985 / 1.078 / 37059.028
- Worst-regression dataset: **Quartz_spxy70** (ratio = 37059.028)
- Cliff's δ (ds, paired): +0.500
- Wilcoxon (ds, one-sided less): p = 0.0120 -> Holm-adjusted = 0.0479
- Wilcoxon (rows, one-sided less): p = 0.0001 -> Holm-adjusted = 0.0002

#### vs AOMRidge-Local-compact-knn50

- Rows kept: 60 (out of 60 per side); datasets kept: 20 (out of 20)
- Wins (per-row, N=60): 42/60 ; Wins (per-dataset, N=20): 14/20
- Median ratio (ds): 0.946  (median Δ% = -5.38 %)
- q75 / q90 / worst ratio (ds): 1.035 / 1.106 / 254.700
- Worst-regression dataset: **Quartz_spxy70** (ratio = 254.700)
- Cliff's δ (ds, paired): +0.400
- Wilcoxon (ds, one-sided less): p = 0.0413 -> Holm-adjusted = 0.1238
- Wilcoxon (rows, one-sided less): p = 0.0012 -> Holm-adjusted = 0.0037

### AOMRidge-AutoSelect-headline-spxy3

#### vs Ridge-tuned-cv5

- Rows kept: 60 (out of 60 per side); datasets kept: 20 (out of 20)
- Wins (per-row, N=60): 53/60 ; Wins (per-dataset, N=20): 18/20
- Median ratio (ds): 0.928  (median Δ% = -7.22 %)
- q75 / q90 / worst ratio (ds): 0.959 / 1.012 / 1.189
- Worst-regression dataset: **Ccar_spxyG_block2deg** (ratio = 1.189)
- Cliff's δ (ds, paired): +0.800
- Wilcoxon (ds, one-sided less): p = 0.0016 -> Holm-adjusted = 0.0126
- Wilcoxon (rows, one-sided less): p = 0.0000 -> Holm-adjusted = 0.0000

#### vs ASLS-AOM-compact-cv5-numpy

- Rows kept: 60 (out of 60 per side); datasets kept: 20 (out of 20)
- Wins (per-row, N=60): 42/60 ; Wins (per-dataset, N=20): 14/20
- Median ratio (ds): 0.945  (median Δ% = -5.52 %)
- q75 / q90 / worst ratio (ds): 1.025 / 1.104 / 1.437
- Worst-regression dataset: **Biscuit_Sucrose_40_RandomSplit** (ratio = 1.437)
- Cliff's δ (ds, paired): +0.400
- Wilcoxon (ds, one-sided less): p = 0.0448 -> Holm-adjusted = 0.1238
- Wilcoxon (rows, one-sided less): p = 0.0015 -> Holm-adjusted = 0.0037

#### vs AOMRidge-global-compact-none

- Rows kept: 60 (out of 60 per side); datasets kept: 20 (out of 20)
- Wins (per-row, N=60): 48/60 ; Wins (per-dataset, N=20): 16/20
- Median ratio (ds): 0.902  (median Δ% = -9.75 %)
- q75 / q90 / worst ratio (ds): 0.976 / 1.043 / 1.466
- Worst-regression dataset: **Biscuit_Sucrose_40_RandomSplit** (ratio = 1.466)
- Cliff's δ (ds, paired): +0.600
- Wilcoxon (ds, one-sided less): p = 0.0047 -> Holm-adjusted = 0.0283
- Wilcoxon (rows, one-sided less): p = 0.0000 -> Holm-adjusted = 0.0000

#### vs AOMRidge-Local-compact-knn50

- Rows kept: 60 (out of 60 per side); datasets kept: 20 (out of 20)
- Wins (per-row, N=60): 42/60 ; Wins (per-dataset, N=20): 14/20
- Median ratio (ds): 0.955  (median Δ% = -4.46 %)
- q75 / q90 / worst ratio (ds): 1.008 / 1.027 / 1.088
- Worst-regression dataset: **Biscuit_Sucrose_40_RandomSplit** (ratio = 1.088)
- Cliff's δ (ds, paired): +0.400
- Wilcoxon (ds, one-sided less): p = 0.0053 -> Holm-adjusted = 0.0283
- Wilcoxon (rows, one-sided less): p = 0.0000 -> Holm-adjusted = 0.0000

## Pre-registered descriptive Friedman-Nemenyi (5 AOMRidge variants)

Variants pre-registered before looking at ranks: `AOMRidge-global-compact-none`, `AOMRidge-Local-compact-knn50`, `AOMRidge-MultiBranchMKL-compact-shrink03`, `AOMRidge-Blender-headline-spxy3`, `AOMRidge-AutoSelect-headline-spxy3`.

Rows with all 5 variants OK: 60 (out of 60 possible).

Friedman: chi^2 = 67.560, p = 0.0000 (descriptive only; omnibus is reserved for the production/full-57 escalation).

Mean rank per variant (1 = best):
- `AOMRidge-global-compact-none`: 3.300
- `AOMRidge-Local-compact-knn50`: 3.200
- `AOMRidge-MultiBranchMKL-compact-shrink03`: 4.150
- `AOMRidge-Blender-headline-spxy3`: 2.150
- `AOMRidge-AutoSelect-headline-spxy3`: 2.200

## Selector diagnostics

Selector diagnostics (AutoSelect chosen-candidate counts, Blender weight mean/std, OOF fold RMSE variance) are not surfaced in this results.csv schema; the existing harness logs them only to per-run JSON sidecars under `benchmark_runs/.../<canonical_name>/`. A follow-up pass to aggregate those into a markdown table is staged but out of scope for this audit20 GATE deliverable.