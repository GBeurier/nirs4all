# Synthetic multimodal software qualification

Executed on 2026-09-17 with
[U07_multimodal_qualification.py](../examples/user/02_data_handling/U07_multimodal_qualification.py).
All 14 cases completed through `engine="dag-ml"`, exported an archive and
reproduced their test predictions from that archive without training. The
fixture is synthetic and deliberately low dimensional; these observations
qualify software behavior and do not establish scientific method rankings.

## Reproduce

```bash
.venv/bin/python examples/user/02_data_handling/U07_multimodal_qualification.py \
  --output /tmp/nirs4all-multimodal-qualification
```

The output includes `report.json` with schemas, sample IDs, folds, pipelines,
selected variants, versions and measurements; `predictions.csv` with every
out-of-fold/test prediction and residual; `summary.csv`; `report.md`; and one
`.n4a` archive per case. Generated workspaces and archives remain outside Git.
This execution used `/tmp/nirs4all-multimodal-qualification-final-20260917`.

## Fixed protocol

- Seeds 17 and 23 reuse `make_cohort` from the main U07 example. Each cohort has
  48 observations from 24 groups, with unequal repetitions of one to three
  observations. Training uses 36 observations/18 groups; final test uses
  12 observations/6 disjoint groups. Targets use synthetic arbitrary units.
- Every method uses the same outer `GroupKFold(3)` splits. Each validation fold
  has 12 observations. The script checks exact sample-ID coverage and fold
  agreement, then independently recomputes RMSE from native predictions.
- Raw source shapes are NIR `(48, 24)`, RGB `(48, 8, 8, 3)`, series
  `(48, 16, 2)` and mixed metadata `(48, 2)`. Source encoders come from
  `make_pipeline`: scaling, two-component TensorPCA, and numeric/categorical
  column processing. Encoders learn within the folds.
- Unimodal and early-fusion Ridge use two native grid candidates, alpha
  `[0.1, 1.0]`. Intermediate fusion uses true multiblock
  `MBPLS(standardize=False)` with components `[1, 2]`. Native selection minimizes
  pooled out-of-fold RMSE, followed by training-set refit.
- Late fusion has one fixed recipe: Ridge alpha 1.0 for each source and for
  the prediction combiner. Native nested `GroupKFold(2)` produces the inner
  out-of-fold predictions. Its search budget and training workload therefore
  differ from the other cases.
- Archive predictions match the stored test predictions at `rtol=atol=1e-9`;
  replay reports `training_performed=False`. The CSV contains 672 predictions
  (36 validation and 12 test observations for each of 14 cases).

## Observed measurements

Time covers run, evidence checks, export and test replay in the current Python
process. Peak bytes come from `tracemalloc`; they measure traced Python
allocations, not process RSS or all native allocations. Startup/cache effects
and concurrent activity can affect time and memory. Archive sizes include
serialization metadata and can vary slightly on rerun. Two seeds and one timed
execution per case are a smoke qualification, not a performance benchmark.

| Seed | Case | Candidates | CV RMSE | Test RMSE | Seconds | Python peak bytes | Archive bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | NIR | 2 | 0.061205 | 0.046474 | 1.088403 | 4781955 | 3770 |
| 17 | Image | 2 | 0.726220 | 0.792766 | 0.994639 | 1832602 | 7563 |
| 17 | Series | 2 | 0.733999 | 0.668869 | 1.140674 | 1746132 | 4024 |
| 17 | Metadata | 2 | 1.673942 | 1.825788 | 0.734406 | 1798328 | 3294 |
| 17 | Early | 2 | 0.045054 | 0.034547 | 0.971212 | 1928031 | 11680 |
| 17 | Intermediate MBPLS | 2 | 0.650533 | 0.416322 | 0.912307 | 1974427 | 13630 |
| 17 | Late | 1 | 0.216600 | 0.116487 | 2.039072 | 3537978 | 14495 |
| 23 | NIR | 2 | 0.059878 | 0.051053 | 0.565027 | 1739780 | 3777 |
| 23 | Image | 2 | 1.418528 | 2.010090 | 0.588097 | 1814969 | 7532 |
| 23 | Series | 2 | 0.706398 | 0.395754 | 1.139283 | 1744541 | 4031 |
| 23 | Metadata | 2 | 1.771782 | 1.754996 | 0.674739 | 1756705 | 3267 |
| 23 | Early | 2 | 0.041794 | 0.037164 | 0.858899 | 1952330 | 11702 |
| 23 | Intermediate MBPLS | 2 | 0.870074 | 0.950314 | 0.895062 | 1928085 | 13633 |
| 23 | Late | 1 | 0.165644 | 0.079420 | 1.835058 | 3476013 | 14490 |

The environment reports nirs4all 1.0.1, nirs4all-io 0.1.18, dag-ml 0.3.25,
NumPy 2.4.6 and scikit-learn 1.9.1. The working trees include the multimodal
implementation under development; these version numbers do not assert that
the same capabilities are already in published releases.

The script passes targeted Ruff and mypy checks. The related adapter,
early/intermediate archive and durable-tuning tests passed together:
`28 passed` across `test_multimodal_adapter.py`, `test_multimodal_dagml.py` and
`test_multimodal_tuning.py`. Late-fusion leakage and schema checks live in
the separate `test_multimodal_late_fusion.py` suite.
