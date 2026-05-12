# Preset Audit — Time-Budget Pool Selection

Data source: `bench/benchmark_master_results.csv` (locked + exploratory `observed`/`reference_paper` rmsep rows) + `bench/scenarios/runs/exhaustive_research_full57_seed0/results.csv` (recent production fit-times).

Methodology:
1. Aggregate per-(canonical, dataset) the best rmsep across master CSV.
2. Compute per-(dataset) PLS-tuned-cv5 rmsep as baseline.
3. Compute ratio = best_rmsep_model / pls_baseline_per_dataset (clip 5×).
4. Greedy pool selection: minimise mean over datasets of `min_{m∈pool} ratio_m`, subject to `Σ_m q90_fit_time_m ≤ budget`.
5. Always include `PLS-tuned-cv5` as anchor.
6. Non-research presets restricted to models with ≥25 datasets of evidence and ≥50 % success rate in latest production run.

## Time budgets (per dataset, sequential upper bound)

| Preset | Budget |
|---|---:|
| fast_reliable | 30s |
| strong_practical | 10 min |
| best_current | 2 h |
| exhaustive_research | 12 h |

## Current vs proposed pools — side-by-side

| Preset | Pool | n_models | seq_time_s | mean_ratio | median_ratio | q90_ratio | coverage |
|---|---|---:|---:|---:|---:|---:|---|
| fast_reliable | **current** | 6 | 212 | 0.9224 | 0.9658 | 1.0000 | 63 |
| fast_reliable | **proposed** | 3 | 22 | 0.9311 | 0.9749 | 1.0000 | 63 |
| strong_practical | **current** | 7 | 336 | 0.9215 | 0.9622 | 1.0000 | 63 |
| strong_practical | **proposed** | 8 | 439 | 0.9212 | 0.9622 | 1.0000 | 63 |
| best_current | **current** | 8 | 1165 | 0.9211 | 0.9622 | 1.0000 | 63 |
| best_current | **proposed** | 9 | 5998 | 0.8230 | 0.8671 | 1.0000 | 63 |
| exhaustive_research | **current** | 24 | 7126 | 0.8966 | 0.9445 | 1.0000 | 63 |
| exhaustive_research | **proposed** | 12 | 9001 | 0.8152 | 0.8669 | 1.0000 | 63 |

## Proposed pools (detail)

### fast_reliable (budget = 30s; expected seq time = 22s)

| canonical | median_ratio | q90_ratio | wins/n_datasets | q90_fit_s | runnability |
|---|---:|---:|---:|---:|---|
| `PLS-tuned-cv5` | 1.0 | 1.0 | 0/63 | 5.749746962671634 | 95 % |
| `AOM-PLS-compact-numpy` | 0.9978474339694489 | 1.1587019025165086 | 35/59 | 7.8607731384470005 | 98 % |
| `Ridge-tuned-cv5` | 0.9883603087329805 | 1.2806485620078478 | 30/55 | 8.72020387048291 | 95 % |

### strong_practical (budget = 600s; expected seq time = 439s)

| canonical | median_ratio | q90_ratio | wins/n_datasets | q90_fit_s | runnability |
|---|---:|---:|---:|---:|---|
| `PLS-tuned-cv5` | 1.0 | 1.0 | 0/63 | 5.749746962671634 | 95 % |
| `AOM-PLS-compact-numpy` | 0.9978474339694489 | 1.1587019025165086 | 35/59 | 7.8607731384470005 | 98 % |
| `Ridge-tuned-cv5` | 0.9883603087329805 | 1.2806485620078478 | 30/55 | 8.72020387048291 | 95 % |
| `AOMRidge-global-compact-snv` | 1.037970429926344 | 1.4991013582022772 | 17/53 | 101.19831128251273 | 92 % |
| `AOMRidge-global-compact-none` | 0.9996835986324695 | 1.1780218190578162 | 29/56 | 79.92803952076501 | 92 % |
| `ASLS-AOM-compact-cv5-numpy` | 1.033522675173786 | 1.3437437600717528 | 18/57 | 8.967219726297968 | 95 % |
| `AOMRidge-Local-compact-knn50` | 1.0214407384006352 | 1.4537129235424227 | 24/58 | 123.63198793520002 | 97 % |
| `AOM-default-nipals-adjoint-numpy` | 1.0618147947712604 | 1.3775896935689462 | 17/57 | 103.13914703598712 | 95 % |

### best_current (budget = 7200s; expected seq time = 5998s)

| canonical | median_ratio | q90_ratio | wins/n_datasets | q90_fit_s | runnability |
|---|---:|---:|---:|---:|---|
| `PLS-tuned-cv5` | 1.0 | 1.0 | 0/63 | 5.749746962671634 | 95 % |
| `TabPFN-HPO-preprocessing` | 0.9020581411591121 | 1.1286800768299794 | 44/58 | 3600.0 | n/a |
| `Ridge-tuned-cv5` | 0.9883603087329805 | 1.2806485620078478 | 30/55 | 8.72020387048291 | 95 % |
| `AOM-PLS-compact-numpy` | 0.9978474339694489 | 1.1587019025165086 | 35/59 | 7.8607731384470005 | 98 % |
| `AOMRidge-Local-compact-knn50` | 1.0214407384006352 | 1.4537129235424227 | 24/58 | 123.63198793520002 | 97 % |
| `AOMRidge-Blender-headline-spxy3` | 0.9897405309300205 | 1.2090422147867717 | 29/53 | 2051.32761536 | 94 % |
| `AOMRidge-global-compact-none` | 0.9996835986324695 | 1.1780218190578162 | 29/56 | 79.92803952076501 | 92 % |
| `TabPFN-Raw` | 1.0058584060430331 | 1.9202415116993188 | 28/59 | 20.0 | n/a |
| `AOMRidge-global-compact-snv` | 1.037970429926344 | 1.4991013582022772 | 17/53 | 101.19831128251273 | 92 % |

### exhaustive_research (budget = 43200s; expected seq time = 9001s)

| canonical | median_ratio | q90_ratio | wins/n_datasets | q90_fit_s | runnability |
|---|---:|---:|---:|---:|---|
| `PLS-tuned-cv5` | 1.0 | 1.0 | 0/63 | 5.749746962671634 | 95 % |
| `TabPFN-HPO-preprocessing` | 0.9020581411591121 | 1.1286800768299794 | 44/58 | 3600.0 | n/a |
| `Ridge-tuned-cv5` | 0.9883603087329805 | 1.2806485620078478 | 30/55 | 8.72020387048291 | 95 % |
| `V2L-Boost-AOMPLS` | 1.1680545210697126 | 2.8838870034596398 | 10/42 | 895.7655568596 | n/a |
| `AOM-PLS-compact-numpy` | 0.9978474339694489 | 1.1587019025165086 | 35/59 | 7.8607731384470005 | 98 % |
| `AOMRidge-Local-compact-knn50` | 1.0214407384006352 | 1.4537129235424227 | 24/58 | 123.63198793520002 | 97 % |
| `AdaptiveSuperLearner-recipe-nnls` | 0.9961405809257642 | 1.1989782760115748 | 22/39 | 91.12472257920001 | n/a |
| `AOMRidge-AutoSelect-headline-spxy3` | 0.9942306041462361 | 1.2416326136317348 | 30/53 | 2015.7797141139997 | n/a |
| `AOMRidge-Blender-headline-spxy3` | 0.9897405309300205 | 1.2090422147867717 | 29/53 | 2051.32761536 | 94 % |
| `AOMRidge-global-compact-none` | 0.9996835986324695 | 1.1780218190578162 | 29/56 | 79.92803952076501 | 92 % |
| `TabPFN-Raw` | 1.0058584060430331 | 1.9202415116993188 | 28/59 | 20.0 | n/a |
| `AOMRidge-global-compact-snv` | 1.037970429926344 | 1.4991013582022772 | 17/53 | 101.19831128251273 | 92 % |

## TabPFN runtime estimates (external — paper-derived)

Master CSV rows for TabPFN-Raw / TabPFN-opt / TabPFN-HPO-preprocessing come from the `tabpfn_paper` ingest and do not contain `fit_time_s` (n=58-61 rows, evaluation only). We use the following wall-clock estimates per dataset:

| canonical | median_fit_time_s | q90_fit_time_s | source |
|---|---:|---:|---|
| TabPFN-Raw | 3 | 10 | TabPFN v2.5 paper Table 2 (small datasets) |
| TabPFN-opt | 60 | 180 | ensemble of 8 configs |
| TabPFN-HPO-preprocessing | 900 | 1800 | HPO over preprocessing budget |

Audit JSON: `bench/scenarios/preset_audit/preset_audit.json`
Per-model evidence CSV: `bench/scenarios/preset_audit/per_model_evidence.csv`