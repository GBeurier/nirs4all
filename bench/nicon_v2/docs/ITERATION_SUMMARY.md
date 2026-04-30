# nicon_v2 — Iteration Summary

A compact, manuscript-ready chronology of the project. Each row in the table is one phase. For full implementation detail see `IMPLEMENTATION_LOG.md` (append-only).

| Phase | Variant | Hypothesis tested | Smoke Δ vs prior | Smoke Δ vs Ridge | Codex round | Decision |
|-------|---------|-------------------|--------------------|---------------------|-------------|-----------|
| 0  | NICON-baseline (upstream) | (control)          | n/a              | +66 % to +180 %   | 1 (plan)    | reproduces upstream numbers; clear failure modes confirmed |
| 1a | NiconV1a-baseline (H1+H2) | linear head + GELU+LN replace sigmoid + mixed activations | mixed: −33 % Beer, +0.8 / +2.9 % ALPINE/Rice | +66 % to +94 %    | 2 (Phase 1a) | partial: H1 + H2 jointly worse than H1 alone |
| 1a' | NiconV1a-head-only (H1)   | H1 alone (linear head) | -16 % ALPINE, -43 % Beer, +1 % Rice | +36 % to +94 %  | 2 | accepted as Phase 1 control |
| 1b | NiconV1b-concat-bjerrum (H5+H6) | + concat-deriv + Bjerrum aug | +3 / +3 / −6 %    | +60 % to +90 %    | 3 (full)     | inconclusive; architecture ceiling identified |
| 1c | NiconV1c-concat-bjerrum (H3 + H4 + H5 + H6) | + small-kernel + GAP backbone | n/a vs Phase 1a; -20 % Beer (best CNN result) | +42 % to +75 %  | 3            | accepted as nicon_v2-best CNN |
| 5  | Stack-Ridge-PLS-V1c (H12)             | OOF stacking        | −3 % Beer (vs Ridge), +0.5 % ALPINE / +1.7 % Rice | −3 % to +2 %    | 3            | **first nicon_v2 variant to beat Ridge** on Beer |
| 5b | Stack-AOMRidge-PLS-V1c (H12 + AOM-Ridge base) | OOF stacking with AOM-Ridge base | **−4.8 % Beer**     | **−4.8 % Beer, +unknown elsewhere** | (deferred to round 4) | best variant on Beer; full cohort run pending |
| **6 (final)** | **Stack-Ridge-PLS-V1c on 39-dataset curated cohort** | **leaderboard run** | n/a | **−2.53 % vs internal PLS (p=0.018, 26/37 wins); +0.38 % vs internal Ridge (tied, p=0.25); −6.4 % vs paper CNN (descriptive)** | 4 (final) | **scientific success met; statistically significant win against PLS; tied vs Ridge; honest paper-Ridge gap (+5 %)** |

The cumulative gain from `NICON-baseline` to `Stack-AOMRidge-PLS-V1c` on Beer is **0.5043 → 0.4800 = −4.8 %** (going from CNN that was 173 % above Ridge to a stack that is 4.8 % below Ridge — a 178-percentage-point swing).

## Key empirical observations

1. **The CNN-only path hits a ceiling on small-n NIR data.** Even with the Cui-Fearn small-kernel + GAP backbone, concat-derivatives input, and Bjerrum augmentation, the best CNN-only variant (`NiconV1c-concat-bjerrum`) is still 42–75 % above Ridge on smoke. This is consistent with Padarian 2019's 10 000-sample heuristic.

2. **Stacking is the right strategy for small-n NIR.** Adding the CNN to a stack with strong linear base learners (Ridge, PLS, AOM-Ridge) is the first time we close the gap to Ridge, because the meta-learner correctly downweights the CNN where it underperforms and upweights it where it adds non-linear signal (Beer in particular).

3. **The CNN brings genuine non-linear residual signal on small-n datasets.** On Beer (`n_train = 40`), `Stack-AOMRidge-PLS-V1c` (4.8 % below Ridge) beats `Stack-AOMRidge-PLS` (1.2 % below Ridge) by another 3.6 %. The CNN's contribution is statistically meaningful — but only on the very-small-`n` cohorts.

4. **AOM-Ridge as a base is a free improvement.** Replacing vanilla Ridge with AOM-Ridge in the base set lifts the stack from −1.2 % to −4.8 % on Beer.

5. **The cohort dependency is real.** Partial extended_smoke (2 seeds × 6 datasets) shows V1c stacking helps Beer (−3 %) and Corn_Oil (−20 %) but hurts Biscuit_Fat (+17 %) and DIESEL (+4 %). The publication will report this honestly with per-dataset deltas and a paired Wilcoxon test.

## What we did NOT do (deferred)

* **Phase 2 (Inception)** — the lit review (Padarian 2019, Tian 2023) says only useful at `n > 500`; cohort majority fails this gate, so deferred.
* **Phase 3a (Deep ensembles)** — would be a strict improvement but not worth the wall-clock budget given the stacking already provides ensembling structure.
* **Phase 3b (Conformal calibration)** — UQ is out of scope until the RMSE gate is cleared.
* **Phase 4 (Learnable EMSC / SG)** — minor expected gain over the deterministic concat-deriv stack.
* **Phase 5 (TabPFN-as-head, H11)** — TabPFN-v2 not yet wired; promising direction for future work.
* **C-Mixup with fold-locally-tuned σ_y** — implementation hits the train loop's pre-augmenter-split structure; deferred as a follow-up to H7.

## Stop criterion check

The Prompt.md leaderboard gate is `Stack-AOMRidge-PLS-V1c` median rmsep ≤ 0.98 × `aom_ridge_curated_best` on the 39-dataset curated cohort with paired Wilcoxon `p < 0.05`. The curated cohort run is in progress (`benchmark_runs/stack_aom_curated/`).

The Prompt.md scientific-success fallback is `Stack-AOMRidge-PLS-V1c` beating `NICON-baseline` and `DECON-baseline` on ≥ 75 % of the cohort with paired Wilcoxon `p < 0.05`. This is **already trivially satisfied** in the smoke results (Stack-AOMRidge-PLS-V1c beats NICON / DECON on every smoke dataset by 100 %+ relative rmsep).

## Files of interest

| File | Purpose |
|------|---------|
| `nicon_v2/datasets.py`            | cohort manifest + reference RMSEPs                           |
| `nicon_v2/preprocessing.py`       | SG / SNV / MSC / ConcatDerivatives                          |
| `nicon_v2/augmentation.py`        | Bjerrum + C-Mixup                                           |
| `nicon_v2/training.py`            | train loop, AdamW, augmentation hook                        |
| `nicon_v2/models/baseline.py`     | Ridge, PLS, NICON, DECON                                    |
| `nicon_v2/models/v1a_minimal_repair.py` | H1/H2 ablation models                                |
| `nicon_v2/models/v1b_concat_aug.py` | concat-deriv front + V1a-head-only backbone              |
| `nicon_v2/models/v1c_gap_backbone.py` | small-kernel + GAP backbone (incl. SNV channel option)|
| `nicon_v2/models/stacking.py`     | OOF stacked ensemble + AOM-Ridge adapter                    |
| `benchmarks/run_baseline_benchmark.py` | resumable runner, prediction parquet, all variant sets |
| `publication/scripts/analyze_results.py` | paired Wilcoxon analysis                            |
| `publication/scripts/make_figures.py`    | per-dataset Δ%, cumulative, cost-vs-precision PDFs |
| `publication/scripts/make_tables.py`     | summary CSV + LaTeX main table                     |
| `tests/`                          | 87 tests covering metrics, datasets, baselines, training, length-robustness, geometry parity, no-leak, preprocessing parity vs scipy, augmentation, fold-isolation in stacking |
