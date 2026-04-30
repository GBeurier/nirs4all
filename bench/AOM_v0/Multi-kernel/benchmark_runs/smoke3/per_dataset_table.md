# Multi-kernel smoke benchmark

## Per-variant median performance

| variant        |   median_rel_pls |   median_rel_ridge |   median_rel_tabpfn_opt |   median_rmsep |   median_fit_time_s |   n_datasets |
|:---------------|-----------------:|-------------------:|------------------------:|---------------:|--------------------:|-------------:|
| mkR-softmax_cv |           0.9502 |             1.0033 |                  1.3712 |         0.2414 |             39.0484 |            3 |
| BLUP-reml      |           0.9915 |             1.0469 |                  1.4221 |         0.2341 |             46.2874 |            3 |
| MKM-reml       |           0.9915 |             1.0469 |                  1.4221 |         0.2341 |             54.1315 |            3 |
| mkR-kta        |           1.1705 |             1.1842 |                  2.1163 |         0.4431 |             17.6377 |            3 |
| mkR-uniform    |           1.3548 |             1.3706 |                  2.1543 |         0.5128 |             22.1375 |            3 |
| Ridge-raw      |           2.3703 |             2.3979 |                  3.0863 |         0.8973 |              0.0814 |            3 |

## Best variant per dataset

| dataset_group   | dataset                      | variant        |   rmsep |   rel_rmsep_vs_pls |   rel_rmsep_vs_ridge |   rel_rmsep_vs_tabpfn_opt |   fit_time_s |
|:----------------|:-----------------------------|:---------------|--------:|-------------------:|---------------------:|--------------------------:|-------------:|
| ALPINE          | ALPINE_P_291_KS              | mkR-softmax_cv |  0.0592 |             0.9502 |               1.0033 |                    1.3628 |      39.0484 |
| AMYLOSE         | Rice_Amylose_313_YbasedSplit | MKM-reml       |  2.2379 |             1.1745 |               1.1893 |                    1.3708 |      54.1315 |
| BEER            | Beer_OriginalExtract_60_KS   | MKM-reml       |  0.2341 |             0.6183 |               0.6255 |                    1.8018 |      29.4572 |
