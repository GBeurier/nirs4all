# X-Realism Discriminator Strategy (active goal)

Date: 2026-05-01

## Single goal

Produce synthetic NIRS spectra whose distribution is indistinguishable
from a real dataset's spectra, evaluated by an adversarial discriminator
on the spectra alone (no targets, no labels, no splits).

A dataset is treated here as an abstract distribution of spectra over a
wavelength axis. Content meaning (which molecule, which target, which
split policy) is intentionally ignored at this stage. Once we can fool
the discriminator on spectrum shape, content-realism becomes the next
problem.

The terminal target is a generator that drives discriminator AUC near
0.5 across all datasets in `bench/tabpfn_paper/data`. Single-dataset
mastery first; full panel after.

## Doctrine revisions

This goal supersedes the per-family Y/content-realism doctrine for the
critical path. The prior M2 stack (docs 15/16/17 and `exp31`) is not
deleted; it becomes parallel work for the older content-realism goal
and is off the critical path here.

| Old rule | New status under this goal |
|---|---|
| No statistics, no PCA, no covariance, no quantile/marginal capture | Relaxed. PCA on spectrum residuals and per-channel noise are explicit ingredients of the hybrid generator. |
| No noise capture | Relaxed. Per-channel residual noise is captured statistically as the tail of the hybrid model. |
| No adversarial AUC as tuning oracle | Reversed. Adversarial AUC IS the oracle. AUC near 0.5 is the win condition. |
| No labels, no targets, no splits as tuning oracle | Kept. The generator never sees Y. The discriminator's internal CV split is a methodology choice, not a dataset-level oracle. |
| Exhaust mechanistic before stats before ML | Kept in spirit, but interpreted per-dataset. The generator is hybrid (mechanistic skeleton + statistical residual) from the start because pure mechanistic was already shown to plateau on DIESEL. |
| No `nirs4all/` edits in this lane | Kept. All work stays under `bench/nirs_synthetic_pfn/`. |
| No DIESEL-only generalization | Kept. The discriminator is run per dataset. A pass on one dataset is not a pass on the panel. |

## Generator family (hybrid)

- Mechanistic skeleton: low-degree polynomial baseline plus a small bank
  of parametric absorption peaks fit to the dataset mean spectrum.
- Statistical residual: low-rank PCA of (real - mechanistic mean), with
  scores sampled from per-component Gaussians whose variances are
  fitted to the real PCA scores.
- Per-channel residual noise: Gaussian with per-channel std fit to the
  tail after PCA reconstruction.

The mechanistic component carries the prior shape. The statistical
component fills the gap. As PCA rank grows, the statistical layer takes
over more of the structure; the mechanistic skeleton sets the prior.

## Discriminator harness

- Real spectra pool: union of `Xtrain.csv` and `Xtest.csv` for the
  chosen dataset (no Y file is read).
- Synthetic spectra: same row count as the real pool, sampled from the
  fitted hybrid generator.
- Discriminators: RandomForest (canonical, non-linear, the win
  condition is RF AUC near 0.5) and LogisticRegression (linear sanity
  baseline).
- Cross-validation: stratified shuffle splits inside the
  real-vs-synthetic pool; the dataset's official train/test split is
  never used because Y is irrelevant here.
- Reported metric: mean and std of AUC across splits, both classifiers.

## Scope of the script work

- `bench/nirs_synthetic_pfn/experiments/exp32_hybrid_xrealism_discriminator.py`
  -- single-dataset evaluator. Sweeps PCA rank, reports per-rank AUC.
- `bench/nirs_synthetic_pfn/experiments/exp33_panel_xrealism_discriminator.py`
  -- multi-dataset runner that walks `bench/tabpfn_paper/data` and
  applies the single-dataset evaluator to every leaf dataset.
- `bench/nirs_synthetic_pfn/tests/test_exp32_hybrid_xrealism_discriminator.py`
  -- generator and AUC sanity tests.
- Reports under `bench/nirs_synthetic_pfn/reports/xrealism_*` (gitignored).

No `nirs4all/` import. No Y file read. No content/family classification.

## Iteration plan

1. Lock the new strategy (this document) and the harness (`exp32`).
2. Smoke on one easy dataset (recommended start: ECOSIS Chla+b leaves
   for sample volume and broad axis).
3. Sweep PCA rank from 0 to a high value and observe AUC trajectory.
4. If RF AUC plateaus above target on the smoke dataset, upgrade the
   generator (more peaks, non-Gaussian score sampling, derivative
   features, multiplicative scattering augment, etc.).
5. Move to the panel (`exp33`) only after at least one dataset is at or
   under the target AUC.
6. For datasets where AUC stays high after generator upgrades, document
   the failure mode (heavy tails, multi-modal score distributions,
   strong non-linear coupling) before adding ML/DL components.

The doctrine ordering still holds inside this loop: hybrid first, then
ML/DL only if hybrid is documented as exhausted on the failing
datasets.

## Iteration log (ECOSIS Chla+b leaves smoke, 2026-05-02)

Subsampled smoke on 1500 ECOSIS Chla+b leaf spectra (196 features,
nm axis 450-2400), `n_pca=50`, `n_splits=3`, RandomForest n_estimators=80,
LogisticRegression linear discriminator. Per-variant adversarial AUC:

| variant | RF AUC | LR AUC | notes |
|---|---|---|---|
| v0 Gaussian PCA scores + Gaussian per-channel noise | 0.9186 | 0.4694 | Linear/second-moment matched (LR ~ 0.5); RF easy. |
| v1 Per-component empirical marginal sampling | 0.9063 | 0.4804 | Marginal upgrade does not move RF. |
| v2 GMM(K=10, diag) on PCA scores | 0.8502 | 0.4615 | GMM helps. |
| v2 GMM(K=20, diag) | 0.8036 | 0.4567 | More components help. |
| v2 GMM(K=10, full) | 0.7816 | 0.4655 | Full covariance is the best GMM. |
| v3 GMM(K=10, diag) + joint_bootstrap noise | 0.8550 | 0.4516 | Bootstrap noise does not help. |
| v3 GMM(K=10, diag) + multiplicative scattering deg=2 | 0.9447 | 0.4720 | Scattering augmentation HURTS — introduces detectable patterns. |
| v4 Gaussian copula + joint_bootstrap noise | 0.8763 | 0.4680 | Copula not better than diag GMM. |
| v5 joint_bootstrap (jitter=0.05) | 0.2816 | 0.4840 | Bootstrap leakage cheat — RF inverts (well below 0.5). Not a real generative win. |
| v5 joint_bootstrap (jitter=0.15) | 0.4069 | 0.4888 | Less leakage but still cheat-ish. |
| v6 knn_mixup (k=5, alpha=1) + joint_bootstrap noise | **0.5783** | 0.4776 | **Best honest result.** Real generation, RF AUC just 8 points above 0.5. |
| v6 knn_mixup (k=10, alpha=2) | 0.7838 | 0.4744 | Smoother mixup -> easier to detect. |
| v6 knn_mixup (k=20, alpha=0.5) | 0.8252 | 0.4898 | Sparse with many neighbors collapses; not better. |

Reading:

- Marginal-only methods (v1, copula) cannot close the RF gap; the gap is
  joint multivariate non-Gaussianity in PCA score space.
- GMM with full covariance provides the best parametric fit (RF 0.78).
- Multiplicative scattering augmentation introduces patterns the
  discriminator catches, so it is removed from the recommended config.
- joint_bootstrap variants drive AUC under 0.5 only because synthetic
  spectra are jittered copies of real spectra; that is a leakage cheat
  rather than a true match of the real distribution.
- knn_mixup with `k=5, alpha=1` (Dirichlet-weighted convex combinations
  of the 5 nearest real neighbors in PCA score space) is the leading
  honest generator on this dataset: RF AUC 0.578, LR AUC 0.478. The
  synthetic samples are new convex interpolations on the data manifold,
  and the discriminator can barely tell them apart from the real data.

## Cross-dataset confirmation (manure, 2026-05-02)

The k=5 alpha=1 knn_mixup recipe was re-tested on
`MANURE21/All_manure_MgO_SPXY_strat_Manure_type` (343 spectra, 1003
features). Variant scan focused on knn-mixup hyperparameters with both
joint_bootstrap and Gaussian noise tails:

| variant | RF AUC | LR AUC | notes |
|---|---|---|---|
| v0 Gaussian baseline | 0.7185 | 0.4562 | Manure is already easier than ECOSIS for v0 (more features per sample, weaker discriminator). |
| v6 knn_mixup k=5 alpha=1 + joint_bootstrap noise | **0.5194** | 0.4711 | Within 2 points of 0.5. |
| v7 knn_mixup k=5 alpha=1 + Gaussian noise | **0.5093** | 0.4724 | **Within 1 point of 0.5.** |
| v7 knn_mixup k=3 alpha=1 + Gaussian noise | 0.4439 | 0.4728 | Slight cheat (too few neighbors -> near-duplicate). |
| v7 knn_mixup k=5 alpha=0.5 + Gaussian noise | 0.4301 | 0.4816 | Sparser Dirichlet -> more cheat. |
| v7 knn_mixup k=7 alpha=0.5 + Gaussian noise | 0.4883 | 0.4727 | Borderline cheat. |
| v7 knn_mixup k=5 alpha=2 + Gaussian noise | 0.5758 | 0.4622 | Smoother Dirichlet -> easier to detect. |
| v7 knn_mixup k=10 alpha=0.5 + Gaussian noise | 0.5674 | 0.4506 | More neighbors with sparse Dirichlet collapses. |

The winning configuration is **knn_mixup score sampling with k=5 nearest
neighbors, Dirichlet alpha=1 (uniform on simplex), Gaussian per-channel
noise tail, and PCA rank ~30-50**. Both ECOSIS and manure converge to
RF AUC within 1-8 points of 0.5 with this recipe. Larger k or larger
alpha smooths samples into RF-detectable centroids; smaller k or
smaller alpha collapses toward single-neighbor near-duplicates and
flips RF AUC under 0.5 (bootstrap leakage).

## Recommended generator configuration

For X-realism on a single dataset:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src python \
  bench/nirs_synthetic_pfn/experiments/exp32_hybrid_xrealism_discriminator.py \
  --dataset <path/to/Xtrain_dir> \
  --pca-range 30 \
  --score-sampling-mode knn_mixup \
  --noise-sampling-mode gaussian \
  --score-knn-mixup-k 5 \
  --score-knn-mixup-dirichlet-alpha 1.0 \
  --report bench/nirs_synthetic_pfn/reports/xrealism_<name>.md \
  --csv bench/nirs_synthetic_pfn/reports/xrealism_<name>.csv
```

For the panel sweep across `bench/tabpfn_paper/data/regression`:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src python \
  bench/nirs_synthetic_pfn/experiments/exp33_panel_xrealism_discriminator.py \
  --root bench/tabpfn_paper/data/regression \
  --pca-range 30 \
  --score-sampling-mode knn_mixup \
  --noise-sampling-mode gaussian \
  --score-knn-mixup-k 5 \
  --score-knn-mixup-dirichlet-alpha 1.0 \
  --report bench/nirs_synthetic_pfn/reports/xrealism_panel_knn_winner.md \
  --csv bench/nirs_synthetic_pfn/reports/xrealism_panel_knn_winner.csv
```

## Panel sweep (M2 representative panel, 2026-05-02)

Configuration: knn_mixup k=5 alpha=1, Gaussian noise tail, PCA rank=30,
subsample=800 rows, n_splits=2, n_estimators=30. 10 datasets evaluated.

| dataset | n_real | RF AUC | Δ from 0.5 |
|---|---:|---:|---:|
| MANURE21/MgO | 490 | 0.5460 | 0.046 |
| GRAPEVINE_LeafTraits/An_NeoSpectra | 119 | **0.5112** | **0.011** |
| IncombustibleMaterial/TIC_spxy70 | 62 | 0.8006 | 0.301 |
| ECOSIS Chla+b_species | 800 | 0.5265 | 0.027 |
| ALPINE_P_291_KS | 291 | 0.5713 | 0.071 |
| BEER_OriginalExtract | 60 | 0.5725 | 0.073 |
| MANURE21/Total_N | 490 | **0.4944** | **0.006** |
| ECOSIS Chla+b_block2deg | 800 | 0.5730 | 0.073 |
| COLZA/N_woOutlier (cm-1) | 800 | 0.5492 | 0.049 |
| grapevine_chloride_556_KS | 555 | 0.7696 | 0.270 |

**8/10 datasets within 8 points of 0.5.** Two failures (TIC and
grapevine_chloride) are small-N high-feature regimes. Full panel report
in `bench/nirs_synthetic_pfn/reports/xrealism_panel_m2_winner.md`.

## What this document does not do

- It does not select a profile, a gate, a promotion, a threshold, or a
  metric for downstream Y-prediction.
- It does not authorize edits to `nirs4all/`.
- It does not retire the M2/exp29/exp31 work; that lane stays
  available for content-realism but is off the critical path.

## Cross-references

- `bench/nirs_synthetic_pfn/docs/13_HANDOFF_STATUS_AND_RESUME_POINT.md`
- `bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py`
  (descriptive panel inventory, still useful)
- `bench/nirs_synthetic_pfn/docs/14_MULTIDATASET_REALISM_REPLAN.md`
  (the prior content-realism plan, parallel lane)
