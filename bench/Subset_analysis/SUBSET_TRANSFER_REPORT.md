# Subset-to-global transfer analysis

Generated from `bench/benchmark_master_results.csv`.

## Question

Does selecting a model on a subset of datasets transfer to better global results?

A subset cannot beat the full-core oracle by definition. The useful test is whether the subset selects the same concrete model, or a near-oracle model, when evaluated on the full 57-dataset core.

## Protocol

- Universe: 57 regression datasets from `main_class_score_matrix.csv`, used as the existing Subset_analysis core.
- Rows: `record_type in {observed, reference_paper}`, regression, `evaluation_split=test`, `status=ok`.
- Score: `score_ratio_vs_dataset_pls`; lower is better and values below 1 beat the dataset PLS anchor.
- Concrete candidate: `source_family | model_class | model_name | variant`; duplicate rows per candidate/dataset are aggregated by median.
- Candidate eligibility per subset/scope: at least 90% full-core dataset coverage and complete coverage on the evaluated subset.
- Primary transfer diagnostics: subset-vs-global Spearman rank, top-5 overlap, selected winner's full-core rank, and full-core regret against the best eligible candidate.

## Main findings

- Current class-balanced subset selects the full-core winner in `all_candidates`: `TabPFN` / `tabpfn_paper | TabPFN | TabPFN-HPO-preprocessing | TabPFN-HPO-preprocessing`.
- Its full-core median ratio is 0.9368; regret is 0.0000; subset-vs-global Spearman is 0.962.
- Excluding TabPFN, the current subset does not recover the full-core oracle: it selects rank 9 with regret 0.0280, despite a high Spearman of 0.960.
- The conservative 19-dataset subset is not automatically safer for incomplete families: for AOM-Ridge it leaves too few candidates with complete subset coverage.
- The legacy variant-heavy subset is usable for the top TabPFN decision, but without TabPFN it selects full-core rank 5, not rank 1.

## Current Subset Safety

| Subset | Scope | Status | Eligible | Spearman | Winner full rank | Regret | Winner class |
|---|---|---:|---:|---:|---:|---:|---|
| `current_class_balanced_10` | `all_candidates` | ok | 178 | 0.962 | 1 | 0.0000 | `TabPFN` |
| `current_class_balanced_10` | `no_tabpfn` | ok | 174 | 0.960 | 9 | 0.0280 | `Meta-selector/MoE` |
| `current_class_balanced_10` | `tabpfn_only` | ok | 4 | 1.000 | 1 | 0.0000 | `TabPFN` |
| `current_class_balanced_10` | `aom_pls_only` | ok | 130 | 0.959 | 1 | 0.0000 | `AOM-PLS` |
| `current_class_balanced_10` | `aom_ridge_only` | ok | 13 | 0.648 | 1 | 0.0000 | `AOM-Ridge` |
| `current_class_balanced_10` | `meta_selector_only` | ok | 10 | 0.952 | 1 | 0.0000 | `Meta-selector/MoE` |
| `current_class_balanced_10` | `ridge_only` | ok | 4 | 1.000 | 1 | 0.0000 | `Ridge` |
| `current_class_balanced_10` | `pls_only` | ok | 5 | 1.000 | 1 | 0.0000 | `PLS` |
| `current_conservative_19` | `all_candidates` | ok | 160 | 0.919 | 1 | 0.0000 | `TabPFN` |
| `current_conservative_19` | `no_tabpfn` | ok | 156 | 0.913 | 3 | 0.0081 | `Meta-selector/MoE` |
| `current_conservative_19` | `tabpfn_only` | ok | 4 | 1.000 | 1 | 0.0000 | `TabPFN` |
| `current_conservative_19` | `aom_pls_only` | ok | 130 | 0.906 | 2 | 0.0092 | `AOM-PLS` |
| `current_conservative_19` | `aom_ridge_only` | too_few_candidates | 0 |  |  |  |  |
| `current_conservative_19` | `meta_selector_only` | ok | 10 | 0.927 | 3 | 0.0081 | `Meta-selector/MoE` |
| `current_conservative_19` | `ridge_only` | too_few_candidates | 1 |  |  |  |  |
| `current_conservative_19` | `pls_only` | ok | 3 | 1.000 | 1 | 0.0000 | `PLS` |
| `legacy_variant_heavy_10` | `all_candidates` | ok | 165 | 0.929 | 1 | 0.0000 | `TabPFN` |
| `legacy_variant_heavy_10` | `no_tabpfn` | ok | 161 | 0.927 | 5 | 0.0165 | `Meta-selector/MoE` |
| `legacy_variant_heavy_10` | `tabpfn_only` | ok | 4 | 1.000 | 1 | 0.0000 | `TabPFN` |
| `legacy_variant_heavy_10` | `aom_pls_only` | ok | 130 | 0.929 | 10 | 0.0324 | `AOM-PLS` |
| `legacy_variant_heavy_10` | `aom_ridge_only` | too_few_candidates | 0 |  |  |  |  |
| `legacy_variant_heavy_10` | `meta_selector_only` | ok | 10 | 0.952 | 3 | 0.0081 | `Meta-selector/MoE` |
| `legacy_variant_heavy_10` | `ridge_only` | too_few_candidates | 1 |  |  |  |  |
| `legacy_variant_heavy_10` | `pls_only` | ok | 3 | 0.866 | 1 | 0.0000 | `PLS` |
| `legacy_variant_heavy_19` | `all_candidates` | ok | 165 | 0.968 | 2 | 0.0160 | `TabPFN` |
| `legacy_variant_heavy_19` | `no_tabpfn` | ok | 161 | 0.966 | 5 | 0.0165 | `Meta-selector/MoE` |
| `legacy_variant_heavy_19` | `tabpfn_only` | ok | 4 | 0.333 | 2 | 0.0160 | `TabPFN` |
| `legacy_variant_heavy_19` | `aom_pls_only` | ok | 130 | 0.958 | 10 | 0.0324 | `AOM-PLS` |
| `legacy_variant_heavy_19` | `aom_ridge_only` | too_few_candidates | 0 |  |  |  |  |
| `legacy_variant_heavy_19` | `meta_selector_only` | ok | 10 | 0.867 | 3 | 0.0081 | `Meta-selector/MoE` |
| `legacy_variant_heavy_19` | `ridge_only` | too_few_candidates | 1 |  |  |  |  |
| `legacy_variant_heavy_19` | `pls_only` | ok | 3 | 0.866 | 1 | 0.0000 | `PLS` |

## Representativeness Diagnostics

| Subset | Size | Numeric z-delta | Group TV | Winner-class TV | Missing groups |
|---|---:|---:|---:|---:|---|
| `current_class_balanced_10` | 10 | 0.781 | 0.619 | 0.251 | 17 |
| `current_conservative_19` | 19 | 0.360 | 0.368 | 0.175 | 12 |
| `legacy_variant_heavy_10` | 10 | 0.161 | 0.567 | 0.232 | 17 |
| `legacy_variant_heavy_19` | 19 | 0.123 | 0.439 | 0.158 | 13 |

## Random Baseline Check

For a good subset, transfer should beat or at least sit near the favorable tail of random subsets of the same size.

| Scope | Size | Oracle hit rate | Spearman p50 | Spearman p95 | Regret p50 | Regret p95 |
|---|---:|---:|---:|---:|---:|---:|
| `all_candidates` | 10 | 0.624 | 0.933 | 0.965 | 0.0000 | 0.0579 |
| `no_tabpfn` | 10 | 0.206 | 0.928 | 0.961 | 0.0157 | 0.0627 |
| `all_candidates` | 19 | 0.700 | 0.955 | 0.975 | 0.0000 | 0.0160 |
| `no_tabpfn` | 19 | 0.234 | 0.951 | 0.974 | 0.0081 | 0.0586 |

## Actionable Plan

1. Freeze the candidate universe before using any subset result. A subset is valid only for candidates that have complete scores on the subset and high global/core coverage in the audit.
2. Use the current 10-dataset class-balanced subset for fast first-pass screening; it recovers the current full-core TabPFN winner and beats same-size random medians on rank transfer.
3. Do not use subset selection alone to choose among non-TabPFN challengers. The current subset has good rank correlation but picks no-TabPFN rank 9 under the broad 90%-coverage audit.
4. Do not treat the 19-dataset conservative subset as universally safer. It improves one representativeness metric but can exclude incomplete families such as AOM-Ridge from fair complete-subset comparisons.
5. Before accepting a subset-selected winner, check `subset_transfer_summary.csv`: require `status=ok`, winner full rank <= 3, low regret, and subset-vs-global Spearman above the same-size random median.
6. For new model families, rerun this script after adding results. If candidate coverage drops below the subset size, either reduce the subset to datasets all candidates cover or evaluate that family separately.
7. Use the subset only as a screening gate. Final claims still require full-core evaluation or a nested selection protocol that did not use full-core test RMSEP to design the subset.

## Generated Files

- `subset_transfer_candidate_matrix.csv`
- `subset_transfer_summary.csv`
- `subset_representativeness.csv`
- `subset_transfer_random_baselines.csv`
- `SUBSET_TRANSFER_REPORT.md`
