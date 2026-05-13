# TabPFN Paper Protocol Notes

Source inspected:

```text
bench/tabpfn_paper/Robin_s_article-1.pdf
```

Relevant protocol points extracted with `pdftotext`:

- The benchmark is designed for NIRS regression and classification tasks.
- Regression collection is described in the paper draft as 54 datasets.
- Classification collection is described as 15 datasets.
- Whenever available, original train/test split protocols are preserved.
- Otherwise deterministic SPXY or stratified variants are used.
- Preprocessing and hyperparameter selection are performed on calibration data
  only.
- The independent test set is kept untouched until final evaluation.
- Regression primary metrics:
  - `RMSECV`
  - `RMSEP`
  - `iRMSEP` relative to a reference, especially PLS.
- Classification primary metric:
  - balanced accuracy.
- Classification final metric in the paper draft is denoted `ACCP`.
- The draft reports classification comparisons against PLS-DA, CatBoost,
  CNN-1D, TabPFN-Raw, and TabPFN-opt.

Local mismatch to handle:

- `bench/tabpfn_paper/data/DatabaseDetail.xlsx` currently lists regression rows
  only.
- Classification files exist under `bench/tabpfn_paper/data/classification`,
  but there is no local classification `master_results.csv`.

Implementation consequence:

- Regression benchmark must compare directly to `master_results.csv`.
- Classification benchmark must build a local cohort and compare AOM/POP-PLS-DA
  against locally run PLS-DA.
