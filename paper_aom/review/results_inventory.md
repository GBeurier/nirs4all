# AOM Results Inventory

Scope: existing artefacts only, no new benchmark runs. Directories inspected:
`bench/AOM`, `bench/AOM_v0`, `bench/AOM_v0/Ridge`,
`bench/AOM_v0/Multi-kernel`, and `bench/tabpfn_paper`.

## Recommended Manuscript Numbers

### AOM-PLS v0, main regression result

Best general AOM-PLS result: `ASLS-AOM-compact-cv5-numpy` on the final
57-regression-dataset cohort.

- Median relative RMSEP vs `PLS-standard`: `0.960026` (about 4.0% lower).
- Wins vs PLS: `42/57`.
- Median fit time: `1.359730 s`.
- Practical comparison to production AOM-PLS: production
  `nirs4all-AOM-PLS-default` is reported as median relative RMSEP
  about `0.999`, `29/57` wins, so the stabilized ASLS+CV recipe adds
  13 wins.
- Exact source: `bench/AOM_v0/publication/tables/relative_rmsep_per_variant.csv`.
- Narrative source: `bench/AOM_v0/Multi-kernel/Summary.md` and
  `bench/AOM_v0/publication/tables/table_regression_main.tex`.

Good secondary AOM-PLS variants from the same table:

| Variant | n | median rel-RMSEP vs PLS | wins | median fit s |
|---|---:|---:|---:|---:|
| `ASLS-AOM-response-dedup-cv3-numpy` | 57 | 0.963777 | 37 | 4.567066 |
| `ASLS-AOM-family-pruned-cv3-numpy` | 57 | 0.963847 | 38 | 1.602201 |
| `ASLS-AOM-compact-repcv3-numpy` | 57 | 0.975305 | 39 | 2.210765 |
| `ASLS-AOM-compact-cv3-numpy` | 57 | 0.978506 | 38 | 0.912129 |

Use these as robustness/Pareto points, not as separate headline claims.

### AOM-Ridge, strongest headline result

Best general Ridge-family result: headline AOM-Ridge blender on the
52-dataset cohort after excluding QUARTZ.

- Versus Ridge: median delta `-4.73%`, capped mean delta `-12.37%`,
  win-rate `86.5%`, wins `45/52`.
- Versus PLS: median delta `-7.73%`, capped mean delta `-14.66%`,
  win-rate `90.4%`, wins `47/52`.
- Versus CNN: median delta `-12.56%`, win-rate `74.5%`, wins `35/47`.
- Versus CatBoost: median delta `-13.27%`, win-rate `73.1%`, wins `38/52`.
- Versus TabPFN-Raw: median delta `-6.51%`, win-rate `71.2%`, wins `37/52`.
- Versus TabPFN-opt: median delta `-0.21%`, capped mean delta `+1.55%`,
  win-rate `51.9%`, wins `27/52`.
- Exact source: `bench/AOM_v0/Ridge/publication/tables/table_summary.tex`.
- Result rows: `bench/AOM_v0/Ridge/benchmark_runs/all54_headline/results.csv`.
- Caveat source: `bench/AOM_v0/Ridge/REPRODUCIBILITY.md` says QUARTZ is
  dropped because its paper Ridge RMSEP is near `3e-9`, making relative
  ratios meaningless.

Variant-level source for AOM-Ridge:

| AOM-Ridge variant | wins / N | win-rate | mean rank | median delta vs Ridge |
|---|---:|---:|---:|---:|
| `AOMRidge-Blender-headline-spxy3` | 35/52 | 67.3% | 3.56 | -2.22% |
| `AOMRidge-AutoSelect-headline-spxy3` | 27/52 | 51.9% | 4.61 | -0.61% |
| `AOMRidge-global-compact-none` | 29/52 | 55.8% | 4.79 | -0.63% |

Exact source: `bench/AOM_v0/Ridge/publication/tables/table_per_method_summary.tex`.
Use the baseline comparison table for the main paper; use this table only if
discussing which AOM-Ridge component drives the aggregate result.

### Multi-kernel AOM, best sister result

Best full multi-kernel table in publication artefacts:

| Variant | n | median rel-PLS | median rel-Ridge | median rel-TabPFN-opt | wins vs PLS | median fit s |
|---|---:|---:|---:|---:|---:|---:|
| `MKM-reml-msc` | 51 | 0.9821 | 1.0231 | 1.1186 | 29 | 6.4352 |
| `MKM-reml` | 51 | 0.9838 | 1.0012 | 1.0704 | 30 | 7.6420 |
| `mkR-softmax_cv` | 51 | 0.9841 | 1.0045 | 1.0669 | 27 | 11.4990 |

Exact source: `bench/AOM_v0/Multi-kernel/publication/tables/table_per_variant_multikernel.csv`.

The stronger but more iterative `iter8_full54_champions` result reports:

- `mkR-softmax_cv-default-active15-sparse3`: n=50, median rel-PLS
  `0.968`, median rel-TabPFN-opt `1.082`, wins vs PLS `33/50`,
  wins vs TabPFN-opt `10/50`, median fit `70.065 s`.
- `MKM-reml-asls-default-active15`: n=50, median rel-PLS `0.970`,
  median rel-TabPFN-opt `1.095`, wins vs PLS `32/50`, wins vs
  TabPFN-opt `13/50`, median fit `68.316 s`.
- Exact source:
  `bench/AOM_v0/Multi-kernel/publication/tables/iter8_summary_per_variant.csv`.

Use multi-kernel results as follow-up/sister-method evidence, not as the
main general AOM headline, unless the manuscript is explicitly about
multi-kernel AOM. The best multi-kernel variants are competitive with PLS
but remain worse than TabPFN-opt in the median.

### TabPFN-paper baseline context

Use these only to contextualize AOM against the existing TabPFN paper
baselines:

- Regression mean ranks: TabPFN-opt `1.749`, Ridge `3.086`,
  TabPFN-Raw `3.449`, CatBoost `3.684`, PLS `4.049`, CNN `4.340`.
- TabPFN-opt pairwise wins: vs CNN `17/21`, CatBoost `21/24`,
  PLS `18/22`, Ridge `18/22`, TabPFN-Raw `22/24`.
- Exact sources:
  `bench/tabpfn_paper/article/Tables/table_global_performance_summary.tex`
  and `bench/tabpfn_paper/article/Tables/table_win_tie_loss_summary.tex`.

Do not mix raw mean RMSEP across datasets as a headline here; means are
scale-dominated and dataset coverage differs by model (`N` ranges from 51
to 61 in the global table).

## Useful But Caveated

### Original `bench/AOM` quick prototype report

`bench/AOM/report.md` compares five prototypes on only five datasets:
baseline AOM-PLS, Bandit AOM-PLS, DARTS PLS, Zero-Shot Router, and MoE PLS.
This is useful for historical motivation only.

Potentially quotable with a clear "pilot" label:

- DARTS PLS improves several small examples, e.g. `LP_spxyG` RMSE
  `0.162523` vs baseline `0.172817`, but is slower (`23.9846 s` vs
  `7.85011 s` on that row).
- MoE PLS has the best RMSE on some rows, e.g. Milk Lactose `0.0549695`
  vs baseline `0.0579551`, but is very slow (`347.07 s` on Milk,
  `715.66 s` on LP).

Do not use these as general benchmark evidence: n=5 datasets, prototype
implementations, no broad paired statistics.

### Classification

Classification artefacts are incomplete for a general AOM claim.

- AOM_v0 publication classification table has no rows:
  `bench/AOM_v0/publication/tables/table_classification_main.tex`.
- Multi-kernel publication classification table also has no rows:
  `bench/AOM_v0/Multi-kernel/publication/tables/table_classification_main.tex`.
- Ridge classification run exists:
  `bench/AOM_v0/Ridge/benchmark_runs/classification_all17/results.csv`.
  It contains 70 OK rows across 14 datasets. Median balanced accuracy is
  roughly `0.608` for `AOMRidgeCls-mkl-compact`, `0.607` for
  `AOMRidgeCls-global-compact`, and `0.606` for active/superblock variants.

Use classification only as "preliminary classification extension" unless
new summary tables and baselines are generated.

## Avoid Or Caveat Strongly

- Avoid absolute mean RMSEP from
  `bench/AOM_v0/publication/tables/table_regression_main.tex` as a primary
  performance number. The table has means around 750-800 and medians near
  0.7-1.0 because raw RMSEP is pooled across different target scales.
  Prefer `relative_rmsep_per_variant.csv`.
- Avoid QUARTZ-relative ratios in AOM-Ridge headline metrics. The official
  AOM-Ridge table excludes QUARTZ for this reason.
- Avoid `bench/AOM_v0/Multi-kernel/STATUS.md` as a sole numeric source for
  final claims. It is a working status document; use the CSV tables it points
  to for exact numbers.
- Avoid smoke/curated iteration runs as manuscript-wide evidence:
  `smoke*`, `curated10`, `curated11`, `diverse10`, and early `iter*` runs
  are useful for development traces but have small or changing cohorts.
- Avoid `bench/tabpfn_paper/early_results.txt` for final comparisons. It is
  a raw early log, not a consolidated benchmark table.
- Avoid `bench/tabpfn_paper/table_results_tabpfn_final_light.csv` unless a
  parser/encoding issue is explicitly handled. It is Latin-1/Windows-style
  encoded and contains an extra markdown-like column; the curated TeX tables
  are safer sources.

## Suggested Manuscript Framing

For a general AOM paper, the cleanest hierarchy is:

1. Main AOM-PLS claim: ASLS + compact AOM + CV-5 gives median relative
   RMSEP `0.960` vs PLS on 57 datasets, with `42/57` wins.
2. Ridge extension claim: AOM-Ridge headline improves over Ridge, PLS,
   CNN, CatBoost, and TabPFN-Raw, and is essentially tied with TabPFN-opt
   by median delta (`-0.21%`, `27/52` wins), after excluding QUARTZ.
3. Multi-kernel claim: multi-kernel variants are a promising follow-up,
   with best publication-table median rel-PLS around `0.982-0.984` on
   51 datasets; they do not yet beat TabPFN-opt in the median.
4. Classification: preliminary only.

