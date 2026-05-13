# AOM final statistics summary

## Workspaces ingested
- aom_pls_full_seed0 (7888 rows)
- aom_pls_paper_seed0 (495 rows)
- aom_pls_paper_seeds012 (1485 rows)
- aom_pls_da_seeds012 (240 rows)
- aom_ridge_audit_seeds012 (540 rows)
- aom_ridge_paper_seeds012 (86 rows)
- aom_ridge_cls_seeds012 (58 rows)
- linear_hpo_seed0 (300 rows)

## Pre-registered paired comparisons

| Comparison | N | Median effect | 95% CI | Wins | Wilcoxon p_Holm | Cliff's delta |
| --- | ---: | --- | --- | --- | --- | ---: |
| ASLS-AOM-compact-cv5 vs PLS-TabPFN-HPO | 25 | ratio=0.980 | 0.907-1.034 | 14/25 (ties 0) | 1 | -0.021 |
| ASLS-AOM-compact-cv5 vs PLS-default | 25 | ratio=0.972 | 0.907-1.038 | 15/25 (ties 0) | 1 | -0.024 |
| AOM-compact-cv5 vs PLS-default | 25 | ratio=1.000 | 0.971-1.046 | 13/25 (ties 0) | 1 | -0.024 |
| AOMRidge-global-compact-none vs Ridge-TabPFN-HPO | 9 | ratio=1.162 | 0.814-1.343 | 4/9 (ties 0) | 1 | 0.037 |
| AOMRidge-Local-compact-knn50 vs Ridge-TabPFN-HPO | 9 | ratio=1.076 | 0.986-1.305 | 2/9 (ties 0) | 1 | 0.111 |
| AOMRidge-Blender vs Ridge-TabPFN-HPO | 9 | ratio=0.914 | 0.829-1.039 | 6/9 (ties 0) | 1 | -0.037 |
| AOMRidge-AutoSelect vs Ridge-TabPFN-HPO | 9 | ratio=0.968 | 0.852-1.091 | 5/9 (ties 0) | 1 | -0.012 |

## Friedman / Nemenyi
- 9 candidates on 9 datasets, chi^2=10.163, p=0.254, CD@0.05=4.005.
- mean ranks (smaller is better):
  - AOMRidge-Blender-headline-spxy3: 3.00
  - AOMRidge-AutoSelect-headline-spxy3: 3.78
  - ridge-tabpfn-hpo-60trials: 4.67
  - AOMRidge-global-compact-none: 5.11
  - ASLS-AOM-compact-cv5-numpy: 5.11
  - AOMRidge-Local-compact-knn50: 5.67
  - pls-tabpfn-hpo-25trials: 5.78
  - pls-default-cv5: 5.89
  - AOM-compact-cv5-numpy: 6.00

## Runtime summary
| Variant | N | median fit | q75 fit | q90 fit | median total | failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AOMRidge-Blender-headline-spxy3 | 60 | 152.19 | 313.73 | 799.98 | 152.31 | 0 |
| AOMRidge-AutoSelect-headline-spxy3 | 60 | 116.89 | 388.51 | 806.94 | 116.96 | 0 |
| ASLS-AOM-compact-cv5-numpy | 339 | 1.65 | 4.42 | 7.66 | 1.73 | 0 |
| AOMRidge-global-compact-none | 60 | 8.45 | 24.45 | 47.44 | 8.45 | 0 |
| AOMRidge-Local-compact-knn50 | 60 | 0.97 | 2.00 | 3.92 | 1.06 | 0 |
| AOM-compact-cv5-numpy | 279 | 1.60 | 4.24 | 6.94 | 1.60 | 0 |
| Ridge-tabpfn-hpo-60trials | 75 | 0.04 | 0.09 | 0.50 | 17.77 | 0 |
| PLS-tabpfn-hpo-25trials | 75 | 0.04 | 0.12 | 0.72 | 7.31 | 0 |
| PLS-default-cv5 | 75 | 0.04 | 0.10 | 0.37 | 2.51 | 0 |

## Seed stability
| Variant | datasets | seeds | median | IQR | std | full-seed datasets | winner changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AOMRidge-Blender-headline-spxy3 | 20 | 3 | 0.3300 | 2.2792 | 11.0459 | 20 | 27 |
| AOMRidge-AutoSelect-headline-spxy3 | 20 | 3 | 0.3455 | 2.2845 | 15.8999 | 20 | 27 |
| ASLS-AOM-compact-cv5-numpy | 59 | 3 | 0.4515 | 3.3055 | 4682.3817 | 54 | 27 |
| AOMRidge-global-compact-none | 20 | 3 | 0.4153 | 2.7123 | 13.1837 | 20 | 27 |
| AOMRidge-Local-compact-knn50 | 20 | 3 | 0.5549 | 2.5818 | 15.4551 | 20 | 27 |
| AOM-compact-cv5-numpy | 59 | 3 | 0.6327 | 3.3118 | 4689.4677 | 53 | 27 |
| Ridge-tabpfn-hpo-60trials | 25 | 3 | 0.6392 | 2.6554 | 27.3673 | 25 | 27 |
| PLS-tabpfn-hpo-25trials | 25 | 3 | 0.5214 | 3.1355 | 23.5487 | 25 | 27 |
| PLS-default-cv5 | 25 | 3 | 0.5714 | 2.7199 | 22.6196 | 25 | 27 |
