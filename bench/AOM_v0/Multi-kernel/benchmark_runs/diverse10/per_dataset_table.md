# Multi-kernel smoke benchmark

## Per-variant median performance

| variant            |   median_rel_pls |   median_rel_ridge |   median_rel_tabpfn_opt |   median_rmsep |   median_fit_time_s |   n_datasets |
|:-------------------|-----------------:|-------------------:|------------------------:|---------------:|--------------------:|-------------:|
| mkR-softmax_cv-snv |           0.9932 |             1.0321 |                  1.2272 |         1.7362 |             38.6075 |            9 |
| MKM-reml-asls      |           1.0024 |             1.0869 |                  1.2894 |         1.5263 |             36.4874 |            9 |
| MKM-reml-msc       |           1.0258 |             1.0766 |                  1.2772 |         1.778  |             35.6583 |            9 |
| mkR-softmax_cv     |           1.0536 |             1.0545 |                  1.2509 |         1.7305 |             33.2024 |            9 |
| MKM-reml           |           1.0956 |             1.0965 |                  1.3067 |         1.7065 |             20.2375 |            9 |
| Ridge-raw          |           1.1477 |             1.1814 |                  1.3165 |         3.0868 |              0.0907 |           10 |

## Best variant per dataset

| dataset_group         | dataset                                   | variant            |    rmsep |   rel_rmsep_vs_pls |   rel_rmsep_vs_ridge |   rel_rmsep_vs_tabpfn_opt |   fit_time_s |
|:----------------------|:------------------------------------------|:-------------------|---------:|-------------------:|---------------------:|--------------------------:|-------------:|
| ALPINE                | ALPINE_P_291_KS                           | mkR-softmax_cv     |   0.0592 |             0.9509 |               1.004  |                    1.3639 |      31.0601 |
| BEER                  | Beer_OriginalExtract_60_YbaseSplit        | MKM-reml-asls      |   0.2511 |             0.9812 |               1.3147 |                    1.6542 |       3.4943 |
| COLZA                 | N_woOutlier                               | mkR-softmax_cv-snv |   0.224  |             0.9412 |               0.9899 |                    1.2625 |     642.966  |
| ECOSIS_LeafTraits     | Chla+b_spxyG_block2deg                    | Ridge-raw          |  34.4972 |             0.5037 |               0.4733 |                    0.4911 |       0.1437 |
| ECOSIS_LeafTraits     | Chla+b_spxyG_species                      | Ridge-raw          |  38.9957 |             0.599  |               0.6122 |                    0.7958 |       0.26   |
| GRAPEVINES            | grapevine_chloride_556_KS                 | mkR-softmax_cv-snv | 946.489  |             1.0336 |               1.0345 |                    1.2272 |      86.3984 |
| GRAPEVINE_LeafTraits  | An_spxyG70_30_byCultivar_NeoSpectra       | Ridge-raw          |   4.3764 |             0.8882 |               0.9112 |                    0.9759 |       0.0315 |
| IncombustibleMaterial | TIC_spxy70                                | mkR-softmax_cv-snv |   3.6871 |             1.3226 |               1.4708 |                    1.2449 |       1.3792 |
| MANURE21              | All_manure_MgO_SPXY_strat_Manure_type     | mkR-softmax_cv-snv |   0.7423 |             0.9484 |               0.9595 |                    0.9886 |      40.5647 |
| MANURE21              | All_manure_Total_N_SPXY_strat_Manure_type | MKM-reml-asls      |   1.5263 |             0.868  |               0.8944 |                    0.9574 |      44.909  |
