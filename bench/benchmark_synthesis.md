# Benchmark Strategy Synthesis

Generated on 2026-05-12 from `benchmark_master_results.csv`.

## Reformulation of the whole project

The project is an empirical search for robust NIRS prediction models across many small-to-medium spectral datasets. The current practical baseline is the `tabpfn_paper` HPO TabPFN run: TabPFN plus a broad preprocessing search. The rest of `bench/` explores whether spectroscopy-aware linear models, operator selection, kernel mixtures, CNNs, hybrids, and learnable convolutional filters can beat or complement that baseline without losing robustness.

The merged CSV stores 25121 observed/reference rows from 9 source families, covering 83 distinct eligible datasets across 86 (dataset, task) pairs. It also adds derived oracle rows per dataset/model class and per dataset globally, so strategy-level visualizations can ask: if this class were allowed to pick its best executed variant per dataset, how far would it get?

### Protocol maturity distribution

Each row carries a `protocol_maturity` tag introduced in the 2026-05-05 master freeze. Use it as a coarse filter for production-eligible vs exploratory evidence:

| Tag | Rows | Meaning |
|---|---:|---|
| locked | 20447 | Stable production-eligible source row (default for clean source runs). |
| exploratory | 4615 | Partial coverage / smoke / diagnostic / pending nested audit (e.g. AOMRidge-headline-spxy3, multiview Phase-11 atoms, smoke runs). |
| legacy | 0 | Explicitly superseded run (reserved for owner-driven downgrades via bench/SYNC.md; not auto-applied in this freeze). |
| oracle | 815 | Derived oracle rows (`oracle_by_model_class`, `oracle_global_dataset`). |
| local_not_master | 59 | `source_oracle` rows already present in source tables; kept for audit, excluded from derived oracle calculations. |

The tagging rules implemented in `bench/build_benchmark_synthesis.py::assign_maturity` are PROVISIONAL; see the P0 freeze entry in `bench/SYNC.md` for the Codex-review status and the explicit rule list.

### Oracle by model class

| Model class | datasets | median rel. RMSEP vs PLS | wins vs PLS |
|---|---:|---:|---:|
| TabPFN | 59 | 0.908 | 45/59 |
| AOM-PLS | 59 | 0.923 | 49/59 |
| AOM-Ridge | 58 | 0.942 | 49/58 |
| Ridge | 58 | 0.972 | 43/58 |
| Meta-selector/MoE | 59 | 0.975 | 38/59 |
| Hybrid CNN+AOM | 42 | 0.978 | 27/42 |
| Multi-kernel ridge | 53 | 0.983 | 34/53 |
| PLS | 67 | 1.000 | 0/67 |
| FCK-PLS | 8 | 1.005 | 4/8 |
| Hybrid CNN+linear | 51 | 1.005 | 24/51 |
| NICON/CNN | 56 | 1.018 | 26/56 |
| Other | 10 | 1.021 | 4/10 |
| CatBoost | 57 | 1.038 | 23/57 |
| POP-PLS | 59 | 1.459 | 9/59 |

Interpretation: this is an optimistic oracle within each class, not a deployable protocol. It answers which strategy family contains useful models somewhere in the search space.

## Synthesis of explored strategies

### TabPFN + preprocessing HPO
The strongest current baseline: try many spectral corrections, reductions, and normalizations before TabPFN, then select by validation/test protocol. It is powerful because TabPFN supplies a strong small-tabular prior while preprocessing makes spectra look less pathological.

### Classical PLS/Ridge references
PLS and Ridge remain the anchors for judging progress. They are fast, stable, and define the scale of the problem; most claimed gains should be expressed as RMSEP ratios against these references.

### AOM-PLS
Adaptive operator selection before PLS. The main lesson is that compact banks plus ASLS and CV selection work better than huge banks; more operators often create selection variance instead of signal.

### AOM-Ridge
Replace the PLS head with Ridge and explore global, split-aware, local, auto-select, and blender variants. This is the most convincing non-TabPFN direction because it keeps spectral inductive bias while improving over linear baselines on many datasets.

### Multi-kernel ridge / MKM
Combine multiple preprocessing branches through kernel weighting, REML/BLUP-style mixtures, or softmax CV. Useful for testing whether averaging transformations beats selecting one; so far it is competitive but not the dominant global answer.

### NICON/CNN and deep spectral models
Several CNN architectures, distillation, low-rank, LUCAS pretraining, and residual variants were tried. Pure CNNs underperform tuned linear models globally; hybrids can help on selected small plant/chemistry datasets.

### Hybrid stacking/residual models
Stack Ridge/PLS/AOM predictions with CNN outputs or learn residual corrections. This improved over internal PLS/CNN baselines but still struggles against AOM-Ridge and TabPFN-opt at the global level.

### FCK-PLS
Learn fractional convolutional filters before a PLS solved head. It is a promising interpretable filter-learning idea, but the evidence is still narrow and not yet comparable to the 57-dataset TabPFN/AOM benchmark.

### Meta-selector / MoE
Select or combine views/operators per dataset using meta-features. This is valuable as an oracle/diagnostic tool, but it needs strict nested validation before it can be trusted as a production selector.

## Why AOM-PLS was hidden in the first ranking

The first report sorted variants with `score_ratio_vs_dataset_pls`, where the denominator was the best PLS row found anywhere for a dataset, across mixed paper, AOM, multiview, and legacy runs. That answers a harsh absolute-leaderboard question, but it is not a fair protocol-local question. AOM-PLS is designed to be fast and reliable inside its own PLS/AOM benchmark protocol; comparing it to a separately tuned or paper-level PLS reference can hide that value.

The CSV now has both views. Use `score_ratio_vs_source_run_pls` for within-protocol reliability and `score_ratio_vs_dataset_pls` for the strict cross-protocol leaderboard. Under the within-protocol view, AOM-PLS does appear in the top list and the main compact/ASLS variants are clearly useful.

| AOM-PLS checkpoint | datasets | median rel. vs source-run PLS | wins | median rel. vs global-best PLS |
|---|---:|---:|---:|---:|
| AOM-PLS-compact-numpy | 59 | 0.922 | 48/59 | 1.087 |
| ASLS-AOM-compact-cv5-numpy | 57 | 0.960 | 42/57 | 1.034 |
| ASLS-AOM-compact-repcv3-numpy | 57 | 0.975 | 39/57 | 1.053 |
| ASLS-AOM-compact-cv3-numpy | 57 | 0.979 | 38/57 | 1.050 |
| AOM-compact-cv5-numpy | 57 | 0.992 | 37/57 | 1.067 |
| nirs4all-AOM-PLS-default | 57 | 0.968 | 36/57 | 1.067 |

## Top 25 best models

Ranking rule: for each variant, keep its best observed row per dataset, then rank by `score_ratio_vs_source_run_pls` when a source-run PLS exists, otherwise by `score_ratio_vs_dataset_pls`. This keeps AOM-PLS visible in its own fair protocol while still allowing paper reference rows and TabPFN-HPO rows to participate. It is still optimistic when a variant was rerun many times.

### 1. ridge-stack-multiview
- Class: Hybrid CNN+linear; datasets: 14; median rel. RMSEP vs PLS: 0.873; wins: 12/14.
- How it works: Hybrid model that stacks or residualizes CNN/NICON features with Ridge, PLS, or AOM predictions.
- Strong points: Can extract non-linear residual signal on some small plant/chemistry datasets.
- Flaws: Pure CNN signal is weak on many NIRS sets; stacked gains are not enough to beat AOM-Ridge/TabPFN globally.

### 2. AOMRidgeBlender
- Class: AOM-Ridge; datasets: 55; median rel. RMSEP vs PLS: 0.875; wins: 46/55.
- How it works: Ridge regression after selecting or blending spectral operator branches; variants differ by global, local, auto-select, and blender selection.
- Strong points: Best broad empirical challenger to TabPFN-opt; cheap inference and strong median gains over Ridge/PLS.
- Flaws: Selection layer is complex and can overfit small validation splits; branch/local/MKL variants add variance if not locked down.

### 3. mean-ensemble-4-fixed
- Class: Meta-selector/MoE; datasets: 59; median rel. RMSEP vs PLS: 0.883; wins: 49/59.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 4. mean-ensemble-3-fixed
- Class: Meta-selector/MoE; datasets: 59; median rel. RMSEP vs PLS: 0.887; wins: 49/59.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 5. trimmed-mean-4
- Class: Meta-selector/MoE; datasets: 39; median rel. RMSEP vs PLS: 0.887; wins: 33/39.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 6. adaptive-super-learner
- Class: Meta-selector/MoE; datasets: 38; median rel. RMSEP vs PLS: 0.890; wins: 35/38.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 7. nnls-stack-multiview
- Class: Hybrid CNN+linear; datasets: 14; median rel. RMSEP vs PLS: 0.890; wins: 11/14.
- How it works: Hybrid model that stacks or residualizes CNN/NICON features with Ridge, PLS, or AOM predictions.
- Strong points: Can extract non-linear residual signal on some small plant/chemistry datasets.
- Flaws: Pure CNN signal is weak on many NIRS sets; stacked gains are not enough to beat AOM-Ridge/TabPFN globally.

### 8. TabPFN-opt
- Class: TabPFN; datasets: 58; median rel. RMSEP vs PLS: 0.894; wins: 43/58.
- How it works: TabPFN regression with a searched spectral preprocessing chain before the foundation tabular prior.
- Strong points: Very strong small-tabular prior; benefits from the preprocessing HPO already done in `tabpfn_paper`.
- Flaws: Expensive and hard to interpret; performance depends heavily on preprocessing search and may not extrapolate to larger or shifted domains.

### 9. nnls-stack-atoms
- Class: Hybrid CNN+linear; datasets: 38; median rel. RMSEP vs PLS: 0.896; wins: 35/38.
- How it works: Hybrid model that stacks or residualizes CNN/NICON features with Ridge, PLS, or AOM predictions.
- Strong points: Can extract non-linear residual signal on some small plant/chemistry datasets.
- Flaws: Pure CNN signal is weak on many NIRS sets; stacked gains are not enough to beat AOM-Ridge/TabPFN globally.

### 10. mean-ensemble-3
- Class: Meta-selector/MoE; datasets: 10; median rel. RMSEP vs PLS: 0.899; wins: 8/10.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 11. AOMPLSRegressor
- Class: AOM-PLS; datasets: 58; median rel. RMSEP vs PLS: 0.901; wins: 47/58.
- How it works: PLS with an adaptive bank of spectral operators and fold-based or holdout model selection.
- Strong points: Fast and spectroscopically grounded; ASLS plus compact/CV variants are robust first-line baselines.
- Flaws: Large operator banks trigger winner's-curse selection; OSC/EMSC/POP variants can fail badly on small n.

### 12. nnls-stack-calibrated
- Class: Hybrid CNN+linear; datasets: 38; median rel. RMSEP vs PLS: 0.904; wins: 35/38.
- How it works: Hybrid model that stacks or residualizes CNN/NICON features with Ridge, PLS, or AOM predictions.
- Strong points: Can extract non-linear residual signal on some small plant/chemistry datasets.
- Flaws: Pure CNN signal is weak on many NIRS sets; stacked gains are not enough to beat AOM-Ridge/TabPFN globally.

### 13. moe-preproc-soft-pls-compact
- Class: Meta-selector/MoE; datasets: 59; median rel. RMSEP vs PLS: 0.905; wins: 50/59.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 14. mean-ensemble-4
- Class: Meta-selector/MoE; datasets: 10; median rel. RMSEP vs PLS: 0.908; wins: 8/10.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 15. moe-view-multiK-wide-2-10
- Class: Meta-selector/MoE; datasets: 59; median rel. RMSEP vs PLS: 0.918; wins: 47/59.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 16. AOM-PLS-compact-numpy
- Class: AOM-PLS; datasets: 59; median rel. RMSEP vs PLS: 0.922; wins: 48/59.
- How it works: PLS with an adaptive bank of spectral operators and fold-based or holdout model selection.
- Strong points: Fast and spectroscopically grounded; ASLS plus compact/CV variants are robust first-line baselines.
- Flaws: Large operator banks trigger winner's-curse selection; OSC/EMSC/POP variants can fail badly on small n.

### 17. AOMRidgeAutoSelector
- Class: AOM-Ridge; datasets: 35; median rel. RMSEP vs PLS: 0.923; wins: 29/35.
- How it works: Ridge regression after selecting or blending spectral operator branches; variants differ by global, local, auto-select, and blender selection.
- Strong points: Best broad empirical challenger to TabPFN-opt; cheap inference and strong median gains over Ridge/PLS.
- Flaws: Selection layer is complex and can overfit small validation splits; branch/local/MKL variants add variance if not locked down.

### 18. bestof-multiview-asls
- Class: Other; datasets: 10; median rel. RMSEP vs PLS: 0.925; wins: 6/10.
- How it works: Variant-specific benchmark entry from the merged result table.
- Strong points: Useful as an explored point in the search space.
- Flaws: Needs a locked protocol before treating the number as a production claim.

### 19. moe-view-multiK-3-5-7-auto
- Class: Meta-selector/MoE; datasets: 10; median rel. RMSEP vs PLS: 0.927; wins: 7/10.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 20. moe-view-soft-pls
- Class: Meta-selector/MoE; datasets: 59; median rel. RMSEP vs PLS: 0.929; wins: 42/59.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 21. moe-preproc-soft-response-dedup
- Class: Meta-selector/MoE; datasets: 10; median rel. RMSEP vs PLS: 0.932; wins: 8/10.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 22. TabPFN-HPO-preprocessing
- Class: TabPFN; datasets: 58; median rel. RMSEP vs PLS: 0.933; wins: 40/58.
- How it works: TabPFN regression with a searched spectral preprocessing chain before the foundation tabular prior.
- Strong points: Very strong small-tabular prior; benefits from the preprocessing HPO already done in `tabpfn_paper`.
- Flaws: Expensive and hard to interpret; performance depends heavily on preprocessing search and may not extrapolate to larger or shifted domains.

### 23. asls-moe-preproc-soft-compact
- Class: Meta-selector/MoE; datasets: 10; median rel. RMSEP vs PLS: 0.934; wins: 7/10.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 24. moe-view-multiK-3-5-7
- Class: Meta-selector/MoE; datasets: 59; median rel. RMSEP vs PLS: 0.934; wins: 39/59.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

### 25. asls-moe-view-soft-K3
- Class: Meta-selector/MoE; datasets: 10; median rel. RMSEP vs PLS: 0.938; wins: 6/10.
- How it works: Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.
- Strong points: Captures complementarity between strong base learners and is useful for estimating oracle headroom.
- Flaws: High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.

## Actionable Waypoints

### W0 - Freeze the evaluation table
Canonicalize one row per `(dataset, model_name, protocol)` and mark exploratory rows as non-ranking. Checkpoint: every leaderboard chart can be rebuilt from `benchmark_master_results.csv` with a documented filter.

### W1 - Define two leaderboards
Keep a strict global leaderboard using `score_ratio_vs_dataset_pls`, and a protocol-local reliability leaderboard using `score_ratio_vs_source_run_pls`. Checkpoint: AOM-PLS, AOM-Ridge, TabPFN, and ensembles each have paired deltas on the same dataset set.

### W2 - Freeze candidate families
Candidate production set should be small: `TabPFN-HPO-preprocessing`, `TabPFN-opt`, `AOM-PLS-compact/ASLS-CV`, `AOMRidge-Blender`, `AOMRidge-AutoSelect`, `MKM/MKR`, and one residual/stacking candidate. Checkpoint: no new family enters until it beats one frozen candidate on a held-out global analysis.

### W3 - Redo subset selection
A subset is valid only if the model selected on it transfers to the full set. Checkpoint: across bootstrap/random subset simulations, subset-chosen top-1/top-3 models should land within 1-2 percent median RMSEP of the full-set oracle ranking and preserve class win-rate ordering.

### W4 - Build nested selector
Only after W2/W3, train a selector from dataset meta-features to choose among frozen candidates. Checkpoint: leave-one-dataset-out or repeated outer folds must beat the best single default, not the oracle.

### W5 - Residual workstream
For datasets where TabPFN and AOM-Ridge disagree strongly, inspect residuals and prediction range compression. Checkpoint: classify failures into baseline/scatter, small-n variance, y-extreme sigmoid, domain/sensor shift, or nonlinear residual.

### W6 - New-model gate
Any new synthetic/PFN/CNN idea must state which failure bucket it targets and must pass W1/W3 before more architecture iteration. Checkpoint: one page with expected lift, target datasets, runtime, and paired test outcome.

## Subset Selection Checkpoints

The subset question was redone as subset-to-global transfer, not just representativeness. A subset is accepted only if the model selected on it remains near the full-core winner.

| Subset | Scope | Status | Spearman | Winner full rank | Regret | Winner class |
|---|---|---:|---:|---:|---:|---|
| current_class_balanced_10 | all_candidates | ok | 0.962 | 1 | 0.0000 | TabPFN |
| current_class_balanced_10 | no_tabpfn | ok | 0.960 | 9 | 0.0280 | Meta-selector/MoE |
| current_class_balanced_10 | aom_pls_only | ok | 0.959 | 1 | 0.0000 | AOM-PLS |
| current_class_balanced_10 | aom_ridge_only | ok | 0.648 | 1 | 0.0000 | AOM-Ridge |
| current_conservative_19 | all_candidates | ok | 0.919 | 1 | 0.0000 | TabPFN |
| current_conservative_19 | no_tabpfn | ok | 0.913 | 3 | 0.0081 | Meta-selector/MoE |
| legacy_variant_heavy_10 | no_tabpfn | ok | 0.927 | 5 | 0.0165 | Meta-selector/MoE |

Operational conclusion: the current 10-dataset class-balanced subset is acceptable as a fast screening gate when TabPFN is included, because it selects the same full-core winner. It is not safe by itself for choosing the best non-TabPFN challenger: in `no_tabpfn` it selects a rank-9 full-core model with 0.0280 absolute median-ratio regret. Use the subset for triage, then require full-core confirmation before claiming an AOM-Ridge, AOM-PLS, MKR, or hybrid challenger wins.

Detailed files: `bench/Subset_analysis/SUBSET_TRANSFER_REPORT.md`, `subset_transfer_summary.csv`, `subset_representativeness.csv`, and `subset_transfer_random_baselines.csv`.

## Dataviz Guide

A starter plotting script is available at `bench/plot_benchmark_master.py`; it writes initial figures to `bench/figures/benchmark_master/`.

Start with three filters: regression only, source rows only, and successful rows only.

```python
import pandas as pd
df = pd.read_csv('bench/benchmark_master_results.csv')
base = df[(df.record_type.isin(['observed', 'reference_paper'])) &
          (df.task == 'regression') &
          (df.score_metric == 'rmsep') &
          (df.status.str.lower().isin(['ok', '']))].copy()
base['rel_source_pls'] = pd.to_numeric(base.score_ratio_vs_source_run_pls, errors='coerce')
base['rel_global_pls'] = pd.to_numeric(base.score_ratio_vs_dataset_pls, errors='coerce')
```

Recommended first plots:

1. **Model-class oracle bar chart**: filter `record_type == 'oracle_by_model_class'`, plot median `score_ratio_vs_dataset_pls` by `model_class`. This answers which global strategy family has headroom.
2. **Protocol-local leaderboard**: group source rows by `model_name`, take the best `rel_source_pls` per dataset, then plot median and interquartile range for models with at least 10 datasets. This is where fast AOM-PLS should be judged.
3. **Strict global leaderboard**: same plot using `rel_global_pls`. This shows who beats the best PLS ever observed for the dataset, but it mixes protocols.
4. **Heatmap model x dataset**: rows are the top 15 models, columns are datasets, color is `rel_source_pls` clipped to a readable range. Values below 1 beat PLS; values above 1 lose to PLS.
5. **Subset-transfer chart**: x-axis subset size, y-axis full-set median regret of the model chosen on the subset. This is the key plot for the new `Subset_analysis` pass.
6. **Runtime vs accuracy Pareto**: x-axis median `fit_time_s`, y-axis median `rel_source_pls`, point size `n_datasets`, color `model_class`. AOM-PLS should be evaluated here, not only on accuracy rank.

## CSV notes

- Master CSV: `/home/delete/nirs4all/nirs4all/bench/benchmark_master_results.csv`
- `record_type=observed` and `record_type=reference_paper` are source rows.
- `record_type=source_oracle` is an oracle value already present in a source table; it is kept for audit but excluded from derived oracle calculations.
- `record_type=oracle_by_model_class` is the best eligible row for a dataset/task/metric/model_class.
- `record_type=oracle_global_dataset` is the best eligible row across all classes for that dataset/task/metric.
- `score_ratio_vs_source_run_pls` is the protocol-local reliability normalization; lower than 1 means better than PLS in the same source/run.
- `score_ratio_vs_dataset_pls` is the strict cross-protocol normalization; lower than 1 means better than the best observed PLS row for that dataset.
- `protocol_maturity` tags each row with one of `locked`, `exploratory`, `legacy`, `oracle`, `local_not_master`. Filter `protocol_maturity == 'locked'` for production-eligible source rows.
