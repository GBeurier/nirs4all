# AOM-PLS Next Generation Benchmark Report

Comparison of baseline AOM-PLS with 4 improved prototypes.

## Models

- **Baseline AOM-PLS**: Standard AOM-PLS with default operator bank
- **Bandit AOM-PLS**: Two-phase bandit (quick R² screen → full NIPALS for top-K), pseudo-linear SNV
- **DARTS PLS**: Gumbel-Softmax architecture search over 18 transforms, temperature annealing, adaptive hard-snap/blend
- **Zero-Shot Router**: Spectral feature extraction (10+ features), soft scoring of 9 pipelines, conservative validation
- **MoE PLS**: 25 expert chains (1st-4th order), two-phase OOF (sklearn screen → AOM-PLS), diversity filtering, Ridge meta-model with safety fallback

## Results

| Model            | Dataset                      |      RMSE |   Time (s) |
|:-----------------|:-----------------------------|----------:|-----------:|
| Baseline AOM-PLS | Rice_Amylose_313_YbasedSplit | 1.88725   |    7.56271 |
| Baseline AOM-PLS | Beer_OriginalExtract_60_KS   | 0.204183  |    1.87629 |
| Baseline AOM-PLS | Firmness_spxy70              | 0.377072  |    1.7899  |
| Baseline AOM-PLS | Milk_Lactose_1224_KS         | 0.0579551 |   11.505   |
| Baseline AOM-PLS | LP_spxyG                     | 0.172817  |    7.85011 |
| Bandit AOM-PLS   | Rice_Amylose_313_YbasedSplit | 1.92344   |    5.68846 |
| Bandit AOM-PLS   | Beer_OriginalExtract_60_KS   | 0.165715  |    2.34131 |
| Bandit AOM-PLS   | Firmness_spxy70              | 0.432015  |    2.15372 |
| Bandit AOM-PLS   | Milk_Lactose_1224_KS         | 0.0612111 |   11.1987  |
| Bandit AOM-PLS   | LP_spxyG                     | 0.168017  |    7.43883 |
| DARTS PLS        | Rice_Amylose_313_YbasedSplit | 1.85952   |   18.4797  |
| DARTS PLS        | Beer_OriginalExtract_60_KS   | 0.195442  |    5.91843 |
| DARTS PLS        | Firmness_spxy70              | 0.358417  |    5.58967 |
| DARTS PLS        | Milk_Lactose_1224_KS         | 0.0588347 |   19.783   |
| DARTS PLS        | LP_spxyG                     | 0.162523  |   23.9846  |
| Zero-Shot Router | Rice_Amylose_313_YbasedSplit | 1.88725   |   42.3012  |
| Zero-Shot Router | Beer_OriginalExtract_60_KS   | 0.204183  |   13.6221  |
| Zero-Shot Router | Firmness_spxy70              | 0.386765  |   12.4522  |
| Zero-Shot Router | Milk_Lactose_1224_KS         | 0.0579551 |   32.0298  |
| Zero-Shot Router | LP_spxyG                     | 0.172817  |   56.5191  |
| MoE PLS          | Rice_Amylose_313_YbasedSplit | 1.85265   |  486.293   |
| MoE PLS          | Beer_OriginalExtract_60_KS   | 0.19594   |  146.745   |
| MoE PLS          | Firmness_spxy70              | 0.513039  |  130.885   |
| MoE PLS          | Milk_Lactose_1224_KS         | 0.0549695 |  347.07    |
| MoE PLS          | LP_spxyG                     | 0.172472  |  715.66    |
