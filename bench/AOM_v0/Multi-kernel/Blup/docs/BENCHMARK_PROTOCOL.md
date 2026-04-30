# BLUP — Benchmark Protocol

## Cohorts

Same as MKM (smoke / extended / full).

## Variants

- `BLUP-compact-none` (E-BLUP from MKM-REML)
- `BLUP-compact-snv`
- `BLUP-compact-msc`
- `BLUP-compact-asls`
- `BLUP-default-none`

## Output CSV columns

Same as MKM, plus:

```text
contribution_train_<block_name_1>_norm,    # ||hat u_b||_2 / ||y - y_mean||_2 on train
contribution_train_<block_name_2>_norm,
...,
contribution_test_<block_name_1>_norm,     # ||hat u_b_*||_2 / ||hat y_* - y_mean||_2 on test
...,
shrinkage_block_<name_1>, ...
```

## Predictive Stop Conditions

BLUP prediction == MKM prediction by construction. The "win" criterion is
**diagnostic informativeness**:

- Per-block contributions consistently above 0.1 of total norm: **shipped**.
- Per-individual decomposition reproducible across folds (Spearman rank
  correlation `> 0.9` between any two folds' contributions): **shipped**.

## Diagnostic Outputs

Beyond CSV, BLUP produces (Phase 7+):

- `figures/blup_contributions_heatmap.pdf` — heatmap of
  `relative_contribution_block_b` per dataset.
- `figures/blup_per_individual_<dataset>.pdf` — stacked-bar of contributions
  for each test sample (top-20 most-deviating from mean).
- `tables/blup_contribution_summary.csv` — mean / std of per-block
  contributions across the 57-dataset cohort.

## Comparison targets

BLUP beats MKM (= same predictions) on **interpretability** only.
The publication framing positions BLUP as the interpretation tool, MKM as
the variance-decomposition tool, and mkR as the prediction-optimised tool.
