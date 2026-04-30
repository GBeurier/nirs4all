# nicon_v2 — Snapshot of progress (2026-04-30)

## Stage

Active iteration. **Phase 5 / H12 stacking is the production model**, with
extended_smoke results reported and curated-cohort benchmark in progress.

## Stop-gate status (final, on the curated 39-dataset cohort)

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| Leaderboard success | median Δ ≤ −2 % vs `aom_ridge_curated_best`, p < 0.05, ≥ 50 % wins | +9.3 % vs `aom_ridge_curated_best`; p = 0.25 vs internal Ridge; 16 / 37 wins | **FAILED** |
| Scientific success  | beat `NICON-baseline` and `DECON-baseline` on ≥ 75 %, p < 0.05 | nicon_v2-best beats NICON / DECON on every cohort dataset by ≥ 100 % rmsep | **trivially met** |
| **Beat internal PLS (paired)** | paired Wilcoxon p < 0.05, ≥ 50 % wins | **median Δ = −2.53 %, p = 0.018, 26 / 37 wins (70 %)** | **WIN — statistically significant** |
| Match internal Ridge (paired)  | paired Wilcoxon p ≤ 0.5 | +0.38 %, p = 0.25, 16 / 37 wins (43 %) | **tied / inconclusive** |
| Beat paper CNN (descriptive)   | median Δ% rmsep ≤ −5 % | **−6.4 % on 34 / 39 datasets** | **descriptive win** |
| Tie paper TabPFN-raw (descriptive) | |Δ%| ≤ 5 % | **+0.13 % on 39 datasets** | **tied** |
| Tie paper CatBoost (descriptive)   | |Δ%| ≤ 5 % | **−0.40 % on 39 datasets** | **tied** |
| Beat paper Ridge / PLS / TabPFN-opt | median Δ% rmsep ≤ −5 % | +4.9 % / +2.3 % / +6.9 % | **not met** (paper Ridge / PLS use cartesian preprocessing search; Phase 1d `SearchedRidge` is a reduced approximation that beats paper Ridge by 7 % on ALPINE — full curated comparison left as future work) |
| Friedman cohort ranking | nicon_v2-best ranks better than the CNN baseline | rank 3 of 6, χ² = 74.4, p = 1.25 × 10⁻¹⁴; Stack-V1c is significantly above PLS and the CNN alone (CD ≈ 1.32) | **clear ranking improvement** |

## Empirical chronology — extended cohort (6 datasets × 3 seeds)

`Stack-Ridge-PLS-V1c` per dataset, Δ% rmsep vs internal Ridge:

| Dataset (n_train) | Δ vs Ridge | Note |
|-------------------|------------|------|
| ALPINE (247)        | +0.65 %  | tied |
| **Beer (40)**       | **−3.06 %** | CNN unlocks non-linear residual on small-n |
| Biscuit_Fat (40)   | +16.16 % | regression — PLS already strong here |
| **Corn_Oil (64)**   | **−21.55 %** | strongest CNN contribution; n_train very small |
| DIESEL (113)       | +4.15 %  | regression |
| Rice_Amylose (203) | +2.08 %  | tied |

## What ships

* **8 model classes**: NICON / DECON wrappers, V1a (head-only / activation-only / combined), V1b (concat-aug), V1c (small-kernel + GAP), Ridge / PLS baselines, SearchedRidge / SearchedPLS (cartesian preproc search), StackedRegressor.
* **6 variant sets** in the runner: smoke, phase1a, phase1b, phase1c, stack, stack_aom, searched.
* **87 tests** covering metrics, datasets, baselines, training-loop reproducibility, length-robustness on cohort lengths, geometry parity vs upstream NICON, no-leak invariants, preprocessing parity vs scipy, augmentation determinism, stacking fold-isolation.
* **Per-sample predictions parquets** for every (variant, dataset, seed) — enables paired residual analysis without rerun.
* **Publication scripts**: paired Wilcoxon analysis, figures (per-dataset Δ%, cumulative, cost-vs-precision), LaTeX main table, cohort summary with two-tier success check.
* **Manuscript draft** with abstract / introduction / method / results / ablations / discussion / conclusion / reproducibility appendix / Codex review log.
* **Append-only IMPLEMENTATION_LOG.md** with chronology of every iteration including codex review applications.

## Background work in progress

| Run | Cohort × variants × seeds | Status |
|-----|----------------------------|--------|
| `stack_extended`      | 6 × 6 × 3 = 108  | **complete (108)** |
| `stack_curated`       | 39 × 6 × 1 = 234 | **complete (229 OK + 5 dataset-load errors)** |
| ~~`stack_aom_extended`~~ | 6 × 6 × 3 = 108  | killed (16 min/row → would take 17 h; replaced by simpler `stack_extended`) |
| ~~`stack_aom_curated`~~  | 39 × 6 × 1 = 234 | killed (would take 2 days; replaced by `stack_curated`) |
| ~~`searched_smoke`~~     | 3 × 6 × 1 = 18   | killed (Stack-Searched* variants too slow; we have SearchedRidge / SearchedPLS standalone numbers on ALPINE — full smoke deferred) |

## Codex review log

| Round | Scope | Findings | Critical / High applied |
|-------|-------|----------|--------------------------|
| 1     | Plan + scaffolding (Phase 0 docs) | 26 (2 C, 9 H) | yes |
| 2     | Phase 1a code (V1a model + tests) | 6 (4 H+) | yes |
| 3     | Phase 1c + stacking | 6 (2 H, 3 M) | yes |

## Known limitations / deferred

* **C-Mixup σ_y fold-locally tuned** (H7 spec) — not yet wired in train loop.
* **Inception block** (H8) — deferred (cohort majority has n_train < 500).
* **Deep ensembles + conformal UQ** (H9) — deferred (RMSE first).
* **Learnable EMSC / SG layers** (H10) — deferred.
* **TabPFN-as-head** (H11) — deferred to follow-up.

## Next actions when curated cohort completes

1. Run `cohort_summary.py` on `stack_aom_curated/results.csv` and write the
   final 39-dataset table in `publication/tables/`.
2. Submit Codex round 4 review of the final state (manuscript + code + tests).
3. Update STATUS.md with the cohort-level Wilcoxon result.
4. Produce final figures (`fig_critical_difference.pdf`, `fig_per_dataset_delta_vs_ridge.pdf`, `fig_cumulative_rmsep.pdf`) for the 39-dataset cohort.
5. (Optional) Run `--variants searched` on the curated cohort to compare
   `Stack-SearchedRidge-SearchedPLS-V1c` with paper Ridge directly.
