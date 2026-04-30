# Multi-kernel smoke benchmark

## Per-variant median performance

| variant             |   median_rel_pls |   median_rel_ridge |   median_rel_tabpfn_opt |   median_rmsep |   median_fit_time_s |   n_datasets |
|:--------------------|-----------------:|-------------------:|------------------------:|---------------:|--------------------:|-------------:|
| mkR-softmax_cv      |           0.9502 |             1.0033 |                  1.3712 |         0.2414 |             29.7552 |            3 |
| mkR-softmax_cv-snv  |           0.9774 |             1.032  |                  1.2287 |         0.1445 |             19.1272 |            3 |
| mkR-softmax_cv-msc  |           0.9815 |             1.0363 |                  1.2143 |         0.1447 |             20.003  |            3 |
| mkR-softmax_cv-asls |           0.9829 |             1.0378 |                  1.4097 |         0.2189 |             15.0302 |            3 |
| MKM-reml            |           0.9915 |             1.0469 |                  1.4221 |         0.2341 |             46.2657 |            3 |
| MKM-reml-asls       |           1.0024 |             1.0352 |                  1.4377 |         0.223  |             52.2678 |            3 |
| MKM-reml-msc        |           1.0258 |             1.0713 |                  1.3694 |         0.1779 |             27.1597 |            3 |
| MKM-reml-snv        |           1.0307 |             1.0883 |                  1.3653 |         0.1774 |             40.4908 |            3 |

## Best variant per dataset

| dataset_group   | dataset                      | variant            |   rmsep |   rel_rmsep_vs_pls |   rel_rmsep_vs_ridge |   rel_rmsep_vs_tabpfn_opt |   fit_time_s |
|:----------------|:-----------------------------|:-------------------|--------:|-------------------:|---------------------:|--------------------------:|-------------:|
| ALPINE          | ALPINE_P_291_KS              | mkR-softmax_cv     |  0.0592 |             0.9502 |               1.0033 |                    1.3628 |      57.9895 |
| AMYLOSE         | Rice_Amylose_313_YbasedSplit | MKM-reml-asls      |  1.948  |             1.0224 |               1.0352 |                    1.1933 |      52.2678 |
| BEER            | Beer_OriginalExtract_60_KS   | mkR-softmax_cv-snv |  0.1445 |             0.3818 |               0.3863 |                    1.1127 |       2.522  |
