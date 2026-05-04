# Multi-kernel smoke benchmark

## Per-variant median performance

| variant            |   median_rel_pls |   median_rel_ridge |   median_rel_tabpfn_opt |   median_rmsep |   median_fit_time_s |   n_datasets |
|:-------------------|-----------------:|-------------------:|------------------------:|---------------:|--------------------:|-------------:|
| MKM-reml-msc       |           0.9518 |             1.0026 |                  1.1421 |         2.8306 |             25.9372 |           10 |
| MKM-reml-asls      |           0.9638 |             1.0023 |                  1.0961 |         2.777  |             26.1308 |           10 |
| mkR-softmax_cv     |           0.9755 |             1.0068 |                  1.0804 |         2.763  |             28.1588 |           10 |
| mkR-softmax_cv-msc |           0.9765 |             1.0559 |                  1.1616 |         3.3954 |             44.2699 |           10 |
| mkR-softmax_cv-snv |           0.9773 |             1.0492 |                  1.1643 |         2.7993 |             23.9327 |           10 |
| MKM-reml           |           0.9853 |             0.9996 |                  1.1004 |         2.7156 |             31.1012 |           10 |
| Ridge-raw          |           1.2631 |             1.3212 |                  1.4752 |         6.7155 |              0.0451 |           10 |

## Best variant per dataset

| dataset_group        | dataset                                | variant        |      rmsep |   rel_rmsep_vs_pls |   rel_rmsep_vs_ridge |   rel_rmsep_vs_tabpfn_opt |   fit_time_s |
|:---------------------|:---------------------------------------|:---------------|-----------:|-------------------:|---------------------:|--------------------------:|-------------:|
| BEEFMARBLING         | Beef_Marbling_RandomSplit              | MKM-reml-msc   |    70.423  |             0.9462 |               0.96   |                    1.1421 |     162.174  |
| BERRY                | ta_groupSampleID_stratDateVar_balRows  | mkR-softmax_cv |     1.8953 |             1.0058 |               1.0522 |                    1.2442 |     462.985  |
| DIESEL               | DIESEL_bp50_246_b-a                    | MKM-reml-msc   |     2.7909 |             0.8488 |               0.9833 |                    0.6449 |      24.1507 |
| DIESEL               | DIESEL_bp50_246_hla-b                  | MKM-reml       |     2.6005 |             0.8786 |               0.9561 |                    0.6185 |      19.4215 |
| FUSARIUM             | Fv_Fm_grp70_30                         | Ridge-raw      |     0.0302 |             0.9557 |               1.0634 |                    1.1805 |       0.5152 |
| GRAPEVINE_LeafTraits | An_spxyG70_30_byCultivar_MicroNIR      | MKM-reml-msc   |     3.5519 |             0.9518 |               0.9739 |                    0.951  |      11.4482 |
| MALARIA              | Malaria_Sporozoite_229_Maia            | MKM-reml       | 33995.2    |           nan      |             nan      |                  nan      |       2.3757 |
| MANURE21             | All_manure_CaO_SPXY_strat_Manure_type  | MKM-reml-msc   |     7.0575 |             0.9374 |               1.0231 |                    1.2554 |      27.7237 |
| MANURE21             | All_manure_P2O5_SPXY_strat_Manure_type | MKM-reml-asls  |     2.3776 |             1.0336 |               0.9335 |                    1.0359 |      24.8534 |
| WOOD_density         | WOOD_N_402_Olale                       | mkR-softmax_cv |     0.0468 |             0.9232 |               0.9531 |                    0.9903 |      22.1247 |
