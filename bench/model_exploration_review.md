# NIRS Model Exploration Review And Next Steps

Generated on 2026-05-05 from the current checkout.

Primary evidence:

- `bench/benchmark_master_results.csv`
- `bench/benchmark_synthesis.md`
- `bench/AOM_v0/Summary.md`
- `bench/AOM_v0/multiview/docs/SUMMARY.md`
- `bench/nicon_v2/docs/STATUS.md`
- `bench/nicon_v2/benchmark_runs/r20_curated_oof/results.csv`
- `bench/fck_pls/README.md`
- `bench/fck_pls/fckpls_torch.py`

The root master file has 21,505 rows, including 20,671 observed/reference rows and 20,452 OK observed/reference rows. It covers 83 OK dataset/task pairs, mostly regression. Lower RMSEP ratios are better. The most important caveat is that the master has two different comparison views:

- `score_ratio_vs_dataset_pls`: strict cross-protocol ratio to the best PLS row found for the dataset.
- `score_ratio_vs_source_run_pls`: within-protocol ratio when a matching source-run PLS exists.

The strict ratio is the leaderboard view. The within-protocol ratio is better for judging whether a local method improved its own baseline.

## Executive Conclusion

TabPFN plus preprocessing HPO is currently the strongest single direction. The class oracle for TabPFN has median RMSEP ratio 0.908 vs global PLS, with 45 wins across 61 datasets. In the global observed winners, TabPFN variants win 30 of 83 dataset/task pairs.

AOM remains the best non-TabPFN engineering direction. AOM-PLS has class-oracle median 0.929 with 49/62 wins, and AOM-Ridge has class-oracle median 0.942 with 45/68 wins. AOM is not usually the absolute champion when compared against the harshest global denominator, but it is cheap, interpretable, stable, and gives strong operator priors.

Pure neural nets are not competitive as replacement predictors on the current NIRS cohort. The NICON/CNN class oracle is only 1.018 vs global PLS, and pure `V2L-learnableRMS` is much worse in both the master and local curated runs. The only credible NN signal is residual learning over a strong chemometric teacher. The local `r20_curated_oof` run, not yet ingested into `benchmark_master`, shows `V2L-Residual-AOMPLS` closes most of the pure-CNN gap: median delta +0.60% vs paper PLS, -9.69% vs paper CNN, but still +7.36% vs curated AOM-Ridge and +9.51% vs TabPFN-opt.

FCK is promising but not yet comparable. The master only contains 8 FCK-source datasets. FCK-PLS class oracle is 1.005 vs global PLS with 4/8 wins. Some old FCK rows are very strong, but the evidence is too narrow and partially synthetic/legacy. FCK should be reintroduced as a small, nested, AOM-compatible operator family and as a residual branch, not treated as a proven global champion.

## Current State By Project

| Project | Current state | Score conclusion |
|---|---|---|
| `bench/tabpfn_paper` | TabPFN raw/opt and HPO preprocessing runs on the TabPFN paper datasets. | Best overall class. `tabpfn_paper` HPO final-light has median 0.933 vs global PLS over 58 datasets; paper `TabPFN-opt` rows are also top-tier. |
| `bench/AOM_v0` | Mature AOM-PLS, AOM-Ridge, multi-kernel, multiview/MoE experiments with tests and publication outputs. | Strongest classical route. ASLS + compact AOM + fold CV is the robust PLS-side recipe; AOM-Ridge oracle is close behind TabPFN. |
| `bench/nicon_v2` | Broad CNN redesign, stacking, distillation, LUCAS pretraining, residual hybrids. | Pure CNN path failed. Residual-AOMPLS is the only promising NN path. Local `r20_curated_oof` must be merged into the master before using it as leaderboard evidence. |
| `bench/fck_pls` | Old standalone FCK study plus PyTorch learnable-kernel prototype. | Narrow evidence. Static FCK has useful wins, but it needs the same 57/61-dataset protocol and leakage-safe nested selection. |

## Score Conclusions From `benchmark_master`

Class-oracle view, strict vs global PLS:

| Class | Datasets | Median ratio | Wins |
|---|---:|---:|---:|
| TabPFN | 61 | 0.908 | 45 |
| AOM-PLS | 62 | 0.929 | 49 |
| AOM-Ridge | 68 | 0.942 | 45 |
| Ridge | 56 | 0.970 | 42 |
| Meta-selector/MoE | 62 | 0.972 | 39 |
| Multi-kernel ridge | 53 | 0.983 | 34 |
| Hybrid CNN+AOM | 12 | 0.986 | 7 |
| Hybrid CNN+linear | 51 | 1.002 | 25 |
| FCK-PLS | 8 | 1.005 | 4 |
| NICON/CNN | 59 | 1.018 | 26 |
| CatBoost | 57 | 1.038 | 23 |
| POP-PLS | 60 | 1.457 | 9 |

Interpretation:

1. The project is not short of individual wins; the global dataset oracle has median 0.848 vs PLS. The unsolved problem is selecting the right family per dataset without leakage.
2. TabPFN is the current default champion because it combines a strong small-tabular prior with preprocessing search.
3. AOM is the strongest spectroscopy-specific prior. It should be the first classical baseline and the main teacher for residual/hybrid models.
4. Meta-selector/MoE and ensemble rows show real complementarity, but they must be treated as optimistic unless all selection is nested inside each outer train split.
5. Pure CNNs do not have enough data or prior. Their useful role is residual correction, feature extraction, or pretraining, not direct replacement.

## What To Explore Next

### 1. Lock a nested champion ensemble

Build a deployable nested protocol that compares and optionally combines:

- `TabPFN-HPO-preprocessing` / `TabPFN-opt`
- `ASLS-AOM-compact-cv5-numpy`
- top AOM-Ridge variants (`Blender`, `AutoSelect`, split-aware/global compact)
- Ridge/PLS references
- residual AOMPLS-CNN if the clean OOF run is included
- FCK static branch once reintroduced

Rationale: the master shows class-level complementarity. The theory is bias-variance plus prior diversity: TabPFN supplies a strong tabular prior, AOM supplies spectroscopy-specific linear operators, and residual branches only model remaining structure. The key empirical risk is leakage, so every selector or stacker must be trained inside the outer split using OOF predictions only.

### 2. Keep the AOM search small and stable

The AOM result is clear: compact/family-pruned banks plus ASLS and fold-based CV beat larger banks. Do not expand operator banks blindly.

Recommended default classical ladder:

1. PLS/Ridge references.
2. ASLS -> AOM compact with CV-5.
3. AOM-Ridge compact/global/split-aware.
4. Multi-view/MoE only under strict nested validation.

Rationale: wide banks create winner's-curse selection variance on small-n splits. The useful spectral transformations are real, but redundant candidates hurt selection.

### 3. Turn `benchmark_master` into the only leaderboard gate

Before another modeling sprint:

- Ingest `bench/nicon_v2/benchmark_runs/r20_curated_oof/results.csv`.
- Reingest FCK only after it runs on the same TabPFN/AOM regression cohort.
- Mark rows by protocol maturity: `locked`, `exploratory`, `legacy`, `oracle`.

Rationale: the local NICON status contains important OOF residual results that the master does not yet include. Conversely, FCK has strong local rows that are not cohort-comparable. The master should make this provenance visible.

## Reintroducing And Improving FCK

FCK should come back, but as a controlled operator family, not as a standalone broad search.

### FCK v1: static operator branch

Implement a small fixed FCK bank as an AOM-compatible transformer:

- fractional orders `alpha in {0.5, 1.0, 1.5, 2.0}`
- 1-2 scales per order
- odd kernel sizes, e.g. 15 or 31
- zero-mean / smooth normalized filters

Then test:

- `FCK -> PLS`
- `ASLS -> FCK -> PLS`
- `FCK -> Ridge`
- `AOM compact + FCK filters -> PLS/Ridge`
- `FCK features/scores -> TabPFN`

Rationale: FCK is a spectroscopy prior similar to fractional derivatives. AOM shows that compact, qualitatively distinct filters help; FCK adds filters that are not exactly SG/FD/ASLS. Keeping the bank small avoids repeating the wide-bank AOM failure mode.

### FCK v2: residual FCK

Use FCK to predict residuals of a strong teacher:

```text
teacher = ASLS-AOM-compact-cv5 or AOM-Ridge
residual = y - OOF_teacher_prediction
final_prediction = teacher_test_prediction + shrinkage * FCK_residual_prediction
```

The shrinkage coefficient must be selected by inner CV, with zero allowed. This creates a do-no-harm path.

Rationale: FCK may be best at local spectral corrections left behind by linear latent models. Residual targets have lower variance and are easier to regularize. This mirrors the only successful NN pattern in `nicon_v2`.

### FCK v3: learned kernels only after static baseline

The PyTorch FCK prototype has the right ingredients: validation-based kernel learning, full-batch training, smoothness and zero-mean regularization, and two interpretable modes:

- V1 free learnable kernels: more stable.
- V2 alpha/sigma parametric kernels: more interpretable.

Before using it in the main benchmark:

- audit feature reporting and transformed dimensions; some legacy rows report `n_features=1` for learned variants and should not be trusted without inspection;
- enforce nested train/validation/test separation;
- add multi-seed stability reporting;
- add kernel diversity regularization so multiple filters do not collapse;
- compare learned kernels to static fractional references;
- keep V1/V2 as residual models first.

Success gate: over the same 57/61 regression cohort, static or learned FCK must improve median RMSEP by at least 2% over ASLS-AOM or AOM-Ridge on a locked nested protocol, or improve a predefined subset without hurting the global median.

## What To Do With Neural Nets

### Stop pure CNN replacement work

The current evidence is strong enough: pure CNN architecture search is a bad use of time for these small-to-medium NIRS datasets. In the master, `NICON-baseline` has median 1.653 vs global PLS, and `V2L-learnableRMS` remains poor. In local curated results, `V2L-learnableRMS` is +32.38% vs PLS and +40.46% vs curated AOM-Ridge.

Theoretical reason: most datasets are small-n/high-p. A raw CNN has too many degrees of freedom and too weak a chemical prior. TabPFN wins because it brings a strong prior; AOM wins because it hard-codes spectral operators. A generic CNN brings neither unless it is pretrained or residualized.

### Keep NNs only in safe roles

Recommended NN roles:

1. Residual corrector over AOM/PLS/Ridge/TabPFN, using OOF teacher residuals only.
2. Frozen or mostly frozen spectral encoder whose outputs feed Ridge/PLS/TabPFN.
3. Denoising/imputation/preprocessing model for X, evaluated by downstream RMSEP.
4. Synthetic-prior or self-supervised pretraining, after realism gates pass.

Every NN predictor should have:

- baseline skip connection: `prediction = base + shrinkage * nn_residual`;
- inner-CV shrinkage with `0` allowed;
- robust loss for residuals;
- fold-local augmentation only;
- dataset-regime gate based on residual CV improvement;
- automatic fallback to the teacher when residual gain is not significant.

### Improve the residual NN path

The local `r20_curated_oof` result is the useful direction. It shows that residualizing over AOMPLS closes most of the pure-CNN gap and beats paper CNN by about 10% median, but it is still behind AOM-Ridge and TabPFN-opt. The next residual sprint should focus on catastrophic-loss prevention, not architecture novelty.

Concrete actions:

- Add shrinkage/gating to `V2L-Residual-AOMPLS`.
- Add residual diagnostics: only train the NN if OOF residuals show structured signal.
- Use FCK/static AOM filters as the residual front-end.
- Run multi-seed on the full curated cohort; the current `r20_curated_oof_multiseed` file is only partial.
- Report paired Wilcoxon vs Ridge, AOM-Ridge, TabPFN-raw, TabPFN-opt, and paper CNN.

### If pursuing a real NN champion, pretrain on priors

A pure NN will need a prior comparable to TabPFN's. That means:

- gather all real X spectra across datasets for self-supervised masked spectral modeling;
- use synthetic spectra only after the realism/adversarial gates in `bench/nirs_synthetic_pfn` improve;
- train an encoder on X-only tasks first;
- evaluate frozen encoder features with Ridge/PLS/TabPFN before end-to-end supervised fine-tuning;
- only then attempt a NIRS-PFN/ICL architecture.

Rationale: the synthetic PFN reports explicitly say current real/synthetic scorecards are diagnostics and do not establish downstream transfer benefit. Do not spend large training budget on synthetic-pretrained supervised NNs until the realism gates and transfer validation are positive.

## Concrete Backlog

1. Update `benchmark_master_results.csv` to include `nicon_v2/r20_curated_oof`.
2. Add a protocol maturity field to the master synthesis.
3. Implement `FCKStaticTransformer` with a small alpha/sigma bank.
4. Run FCK static on the AOM/TabPFN regression cohort with PLS, Ridge, AOM, and TabPFN heads.
5. Add `FCKResidualRegressor` with OOF teacher residuals and CV shrinkage.
6. Add residual shrinkage/gating to `V2L-Residual-AOMPLS`.
7. Build the locked nested champion ensemble across TabPFN, AOM, AOM-Ridge, FCK, and residual NN.
8. Deprioritize pure CNN architecture variants unless they are part of a pretraining or residual-gated experiment.
9. Treat RandomForest/AOM-leaf ideas as selector diagnostics, not as first-priority champion candidates; tree ensembles are not leading in the current master.

