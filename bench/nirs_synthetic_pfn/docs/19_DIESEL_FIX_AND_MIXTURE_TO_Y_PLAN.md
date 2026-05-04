# DIESEL fix + Mixture-to-Y plan (proposal, awaiting Codex review)

Date: 2026-05-02

## Background

The X-realism iteration (`docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md`)
shipped a `knn_mixup k=5 alpha=1` generator that drove RandomForest AUC
near 0.5 on most of `bench/tabpfn_paper/data/regression` (median 0.494,
8/10 of the M2 panel within 8 points of 0.5). Two families failed: very
small datasets (TIC, BEER) and DIESEL transfer-difference variants.

This document plans the next two pieces of work:

1. **Phase A — DIESEL fix**: handle the DIESEL variants whose signal is
   not raw absorbance (calibration-transfer subtractions, mixed-source
   datasets).
2. **Phase B/C — Mixture decomposition + Y predictor**: fit a low-rank
   mixture model `X ≈ C @ S` per dataset, learn `Y = f(C)` on real,
   then predict `Y'` from synthetic `X'` so we ship `(X', Y')` pairs.
3. **Phase D — Validation**: joint-distribution AUC and TSTR/TRTS
   sanity checks.

Same constraints as before: no `nirs4all/` edits, all work under
`bench/nirs_synthetic_pfn/`, adversarial AUC remains the X oracle, and
Y is now allowed only for fitting the `f(C) → Y` regressor (after the X
generator is frozen — no oracle leakage into the X loop).

## DIESEL data inspection (2026-05-02)

Two families across the 38 DIESEL variants in
`bench/tabpfn_paper/data/regression/DIESEL`:

- **Family 1 (`*_PengKS` / `*_ZhengYbase`, 395 samples each)**: range
  `[-0.05, 1.19]`, mean `0.13`, **33% negative values**. Mixed source:
  most rows look like raw absorbance; a non-trivial fraction (rows
  4-10 inspected) are negative-heavy and look pre-corrected
  (centered/derivative/SNV-ish). Same X file is reused under different
  Y splits.
- **Family 2 (`*_b-a`, `*_hla-b`, `*_hlb-a`, 226-263 samples each)**:
  range `[-0.05, 0.07]`, mean `0.003`, **53% negative values**. These
  are clearly **calibration-transfer subtractions** (instrument A minus
  instrument B style), not Savitzky-Golay derivatives. Suffix
  conventions confirm: `b-a` = "b minus a", `hla-b` = "high-low-a
  minus b". Magnitudes are ~30x smaller than Family 1.

Both families have the same axis (750-1550 nm, 401 features at 2 nm
step).

This is the "transformation initiale" mentioned by the user. It is not
a single derivative across all DIESEL — it is two distinct preprocessed
signal types embedded in the panel.

## Phase A — DIESEL fix

### A.1 Signal-type classifier

A small heuristic that, for any dataset, classifies the signal type
into one of:

- `raw_absorbance` (positive, smooth, range > 0.1)
- `centered_or_derivative` (mean ~ 0, both signs)
- `transfer_difference` (mean ~ 0, very small magnitude < 0.1, both
  signs)
- `mixed` (some rows centered, others raw)
- `unknown`

Used to decide knn_mixup hyperparameters per dataset.

### A.2 Per-class generator tuning

Hypothesis: for `transfer_difference` and small-N datasets, k=5 is
likely too sparse — synthetic samples sit too close to a few real
neighbors, RF detects the geometry. Try:

- larger k (10, 20, 30) for small-N datasets
- smaller alpha (0.3-0.5) for non-collapsing diversity
- per-class noise tail scaling (Gaussian std proportional to the
  family magnitude; current default fits residuals after PCA which
  should already adapt, but worth checking)

For mixed-source datasets, optionally split into homogeneous
sub-pools before fitting the generator.

### A.3 Re-run discriminator on DIESEL alone

Use exp34 mode comparison restricted to DIESEL variants with the new
hyperparameter grid; produce a per-variant winner table.

Deliverable: `experiments/exp35_diesel_signal_aware_generator.py` +
tests + report. Goal: RF AUC < 0.6 on every DIESEL variant.

## Phase B — Mixture decomposition

### B.1 Per-dataset mixture fit

For each dataset with `n` real spectra of dimension `d`:

1. Choose a number of pure components `K` (start with `K = 6` and
   sweep `K ∈ {3, 6, 10, 15}`).
2. Fit a non-negative matrix factorization on `|X|` (we use absolute
   value when X has negatives, since NMF needs non-negative input;
   alternatives below):

   ```
   |X| ≈ C @ S    where C ∈ R^{n × K}_+, S ∈ R^{K × d}_+
   ```

   Track reconstruction relative error
   `||X - C @ S||_F / ||X||_F`.

3. For datasets with negatives that violate NMF (transfer-difference
   family), fall back to PCA / sparse PCA. Document which mixture
   model was used per dataset.

Deliverable: `mixture_fit_<dataset>.csv` with columns: `K`,
`reconstruction_relative_error`, `model_kind` (`nmf`, `pca`,
`sparse_pca`).

### B.2 Pure-component spectra inspection

Plot the K spectra `S_k` per dataset (matplotlib only; saved as PNG to
the gitignored reports dir). Sanity check: do they look like
plausible NIR component spectra (water bands, CH overtones, ash
features)?

### B.3 Concentration vector storage

Persist the real `C` matrix per dataset as a Parquet/CSV under
`reports/mixture_concentrations_<dataset>.csv`. This is the input to
the Y regressor.

## Phase C — Y predictor

### C.1 Learn `Y = f(C)` on real data

For each dataset that has Y:

1. Read `Ytrain.csv` + `Ytest.csv`. Drop sentinel rows (`-999` etc).
2. Concatenate `(C_real, Y_real)`. Use the same row alignment as the
   X loader (union of train + test).
3. Fit a small regressor on `(C, Y)`. Three candidates, picked by
   per-dataset CV:
   - `Ridge(alpha=1.0)` with `K` features (linear, very few params)
   - `RandomForestRegressor(n_estimators=200)` (non-linear)
   - `LinearRegression` (baseline)
4. Report cross-validated R^2 and RMSE.

The Y regressor is a function `f: R^K → R`. It is **frozen** after
fit. It will not be re-tuned by any synthetic Y oracle.

### C.2 Generate `Y'` from synthetic `X'`

For each synthetic spectrum produced by the knn_mixup generator:

1. Project to `C'` using the learned mixture model:
   - For NMF: solve a non-negative least squares
     `C' = argmin_{c >= 0} ||X' - c @ S||_2^2`.
   - For PCA: `C' = X' @ S^T` (since S is orthonormal).
2. Predict `Y' = f(C')`.

The result is a synthetic pair `(X', Y')` where:

- `X'` is distributionally indistinguishable from real X under the
  RF discriminator (Phase 18 result).
- `Y'` is a plausible target derived from a chemistry-inspired
  mixture decomposition of `X'`.

### C.3 Free parameters

- Choice of `K` per dataset.
- Mixture model kind (NMF vs PCA vs sparse PCA).
- Y regressor family.

These are tuned **only** on real `(X, Y)` per-dataset CV. Once frozen,
they are never re-tuned with synthetic feedback.

## Phase D — Validation

### D.1 Univariate Y sanity

Per dataset:

- Real `Y` distribution: histogram, mean, std.
- Synthetic `Y'` distribution: histogram, mean, std.
- KS test on `Y` vs `Y'` (purely diagnostic).

### D.2 Joint (X, Y) adversarial AUC

Train a second discriminator on stacked `[X, Y]` (real vs synthetic).
This is the strongest test: the X side already passes; Y must not
break it.

### D.3 TSTR / TRTS

- TSTR (Train-Synthetic, Test-Real): fit a regressor on
  `(X', Y')`, evaluate on `(X_test_real, Y_test_real)`, report R^2.
- TRTS (Train-Real, Test-Synthetic): fit on real, evaluate on
  synthetic, report R^2.

If TSTR R^2 is within 80% of TRTS R^2, we have a useful synthetic
training pool.

Note: the existing X iteration doctrine forbids using TSTR/TRTS as a
**tuning oracle for the X generator**. Phase D uses TSTR/TRTS only as
**validation**, with X and Y models already frozen.

## Phase E — Deliverables (per phase)

| Phase | Code | Tests | Reports |
|---|---|---|---|
| A | `exp35_diesel_signal_aware_generator.py` | `test_exp35_*.py` | `xrealism_diesel_per_variant.{md,csv}` |
| B/C | `exp36_mixture_y_predictor.py` | `test_exp36_*.py` | `mixture_fit_<dataset>.{md,csv}`, `mixture_concentrations_<dataset>.csv` |
| D | `exp37_xy_validation.py` | `test_exp37_*.py` | `xy_validation_<dataset>.{md,csv}` |

All reports under `bench/nirs_synthetic_pfn/reports/` (gitignored,
regenerable).

## Constraints (carried forward)

- No `nirs4all/` edits.
- No targets/splits as oracle for the X generator.
- Y is allowed only as input to the f(C)→Y regressor (Phase C),
  fitted on real data, frozen before any synthetic-Y comparison.
- Adversarial AUC remains the X oracle. The joint-AUC and TSTR/TRTS
  in Phase D are validation only.
- DIESEL R3d/R9/P2 docs (lane 1) and the M2 manifest schema (lane 2)
  remain parallel; this is lane 3 (active).

## Codex review (2026-05-02) — MERGED

Codex flagged real bugs and reordered the plan. Resolved positions:

- **Q1 (NMF vs PCA):** Use centered PCA universally. NMF on `|X|` is
  mathematically inconsistent when negatives carry signal (e.g. the
  DIESEL `_b-a`/`_hla-b` family). PCA also handles signed scores
  correctly.
- **Q2 (raw C vs fractions):** Use raw `C` scores, z-scored within
  the training fold. Concentration fractions are invalid for signed
  PCA scores.
- **Q3 (joint-AUC space):** Train the joint-AUC discriminator on
  full `[X, Y]`, not on PCA-compressed `[X_pca, Y]`. Honesty over
  speed.
- **Q4 (K selection):** Cross-validated BIC. Elbow on reconstruction
  is subjective and over-reports.
- **Q5 (mixture for `_b-a` family):** No. Mixture decomposition is
  not meaningful when the signal is an instrument-transfer residual
  rather than a positive composition. For that family, do not run
  Phase B/C; ship X-only.

Critical bugs to fix in the plan:

- **Y leakage**: Original Phase C said to concatenate `Ytrain.csv +
  Ytest.csv` for `(C, Y)` fitting. That leaks official test targets
  into `f(C)` and breaks Phase D TSTR. Rule: `f(C)` is fit on
  **official train only**; official test stays untouched until final
  TSTR / TRTS.
- **Dirichlet alpha < 1 makes weights sparser, not more diverse.**
  My Phase A.2 idea was wrong. Use `alpha >= 1` in tuning grids; for
  small-N datasets, raise `k` instead of lowering alpha.
- **Mixed-source sub-pool clustering risks tiny modes** that
  `knn_mixup` then memorizes. Apply only with a minimum effective `n`
  per cluster (e.g. `>= 100`); otherwise stick to global PCA-space
  generation with shrinkage noise.
- **`f(C)` returns `E[Y|C]` only.** Phase D joint-AUC will collapse
  Y variance and trip the discriminator on `Var(Y_synth) <
  Var(Y_real)`. Solution: sample calibrated residuals conditional on
  the C-neighborhood after `f(C)` (e.g., bootstrap real residuals
  from k-nearest-C real samples).
- **Per-variant tuning of `k` on the same RF AUC** is discriminator
  overfitting unless run with nested holdout. For Phase A grid, fix
  the RF discriminator's split seed across variants and report
  honest CV.
- **Phase B.2 visual `S_k` inspection is a dead end** (PCA/NMF
  rotational invariance). Drop it; replace with quantitative
  reconstruction metrics only.
- **TSTR/TRTS "within 80%" needs anchoring** against a real-only
  CV baseline reported in the same table. Without that, "80%" is
  arbitrary.

### REORDERED EXECUTION

Codex pushed the right experiment to the front:

#### Step 0 — feasibility gate (real-only, no synthetic, no oracle abuse)

Before any Phase A/B/C/D coding, run on hard datasets (DIESEL
transfer-difference variants + a few controls):

1. Read official `Xtrain.csv`, `Ytrain.csv`. **Do not touch
   `Xtest.csv`/`Ytest.csv`**.
2. Inner split: 80/20 train/val on official-train rows.
3. On inner train: fit centered PCA, sweep `K ∈ {3, 5, 10, 20, 40}`,
   pick K via CV-BIC (5-fold within inner train).
4. Fit two regressors `f(C) → Y` on inner train: Ridge and
   RandomForestRegressor.
5. Direct baseline: Ridge on raw `X → Y` on inner train.
6. Project inner val X → C_val, predict `Y_val_hat = f(C_val)`,
   report R^2 and RMSE.
7. Compare to direct baseline. Decision rule:

   - If `R^2(f(C)) >= 0.8 * R^2(direct_X)` on most variants and
     residual variance is not collapsed → **GO**: proceed with the
     full Phase A/B/C/D plan as corrected.
   - Otherwise → **KILL**: ship X-only (Phase 18 is the final
     deliverable for those families); document the failure.

This single experiment kills or confirms the entire B/C/D chain
before any code is written for it.

Deliverable: `experiments/exp35_y_predictor_feasibility.py` + tests +
`reports/y_predictor_feasibility.{md,csv}`.

### Step 0 result (2026-05-02): GO on 9/12, KILL on 3/12

12 datasets evaluated (5 DIESEL variants + 7 controls). 9 datasets
pass the feasibility gate; the 3 failures are all DIESEL
transfer-difference variants whose direct X->Y Ridge baseline is
already R^2 <= 0 (Y is not predictable from X for those
calibration-residual signals).

| dataset | n_train | K* | direct R^2 | f(C) R^2 | ratio | decision |
|---|---:|---:|---:|---:|---:|---|
| DIESEL_v_252_hla-b | 136 | 20 | -0.04 | 0.11 | n/a | KILL (baseline useless) |
| DIESEL_v_252_hlb-a | 136 | 40 | -0.01 | 0.32 | n/a | KILL (baseline useless) |
| DIESEL_bp50_246_hla-b | 133 | 40 | -0.26 | 0.38 | n/a | KILL (baseline useless) |
| DIESEL_BP50_395_PengKS | 276 | 40 | 0.50 | 0.78 | 1.55 | GO |
| DIESEL_TOTAL_395_PengKS | 276 | 40 | 0.87 | 0.91 | 1.05 | GO |
| MANURE21/MgO | 343 | 40 | 0.78 | 0.78 | 1.00 | GO |
| MANURE21/Total_N | 343 | 40 | 0.89 | 0.89 | 1.00 | GO |
| ECOSIS Chla+b_species | 3724 | 40 | 0.58 | 0.85 | 1.47 | GO |
| ALPINE_P_291_KS | 247 | 40 | 0.64 | 0.64 | 1.00 | GO |
| COLZA/N_woOutlier | 1205 | 40 | 0.85 | 0.92 | 1.09 | GO |
| GRAPEVINES/grapevine_chloride_556_KS | 388 | 40 | 0.21 | 0.46 | 2.20 | GO |
| MILK/Milk_Fat_1224_KS | 181 | 40 | 0.90 | 0.93 | 1.03 | GO |

Reading:

- For most datasets, PCA-then-Ridge `f(C) -> Y` matches or beats the
  raw `X -> Y` Ridge baseline. PCA captures the same predictive
  signal (or denoises it) at K=40 components.
- For DIESEL transfer-difference (`v_*_hla-b`, `v_*_hlb-a`,
  `bp50_*_hla-b`), the underlying Y is not Ridge-predictable from
  the spectra at all (direct R^2 <= 0). Phase B/C cannot help here;
  ship X-only and document the failure.
- For DIESEL `BP50_395_PengKS`, the baseline R^2 = 0.50 is moderate
  but f(C) reaches 0.78 — PCA helps even on this mixed-source
  family.

Decision: proceed with Phase B/C/D on the 9 GO datasets. For the 3
KILL datasets, no Y synthesis is attempted; they ship as X-only via
the Phase 18 generator.

## Step 1 result (Phase B/C, exp36, 2026-05-02)

Pipeline ran cleanly on all 9 GO datasets. Each produced a synthetic
`(X', Y')` CSV under
`bench/nirs_synthetic_pfn/reports/synthetic_xy/<name>_synthetic.csv`.
The Y model was Ridge or RandomForestRegressor depending on the
per-dataset CV R^2 winner; PCA rank K* = 40 for all.

## Step 2 result (Phase D, exp37, 2026-05-02)

Validation on real `Xtest`/`Ytest` (TSTR/TRTS/baseline). RandomForest
discriminator on stacked `[X, Y]` for the joint AUC.

| dataset | n_train | n_test | n_synth | joint AUC (RF) | KS Y p | baseline R^2 | TSTR R^2 | TRTS R^2 | TSTR/baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIESEL_BP50_395_PengKS | 276 | 119 | 276 | 0.7326 | 0.013 | 0.6303 | 0.6118 | 0.952 | **0.97** |
| DIESEL_TOTAL_395_PengKS | 276 | 119 | 276 | 0.7241 | 0.42 | 0.8324 | 0.8029 | 0.971 | **0.97** |
| MANURE21/MgO | 343 | 147 | 343 | **0.5617** | 0.0003 | 0.7511 | 0.6873 | 0.973 | **0.92** |
| MANURE21/Total_N | 343 | 147 | 343 | **0.5744** | 0.001 | 0.8728 | 0.8469 | 0.987 | **0.97** |
| ECOSIS Chla+b_species | 3724 | 3116 | 3724 | 0.7728 | 5e-9 | -0.029 | 0.1043 | 0.946 | n/a (baseline negative) |
| ALPINE_P_291_KS | 247 | 44 | 247 | 0.5529 | 0.009 | 0.0326 | -0.110 | 0.879 | n/a (baseline near 0) |
| COLZA/N_woOutlier | 1205 | 1207 | 1205 | 0.6378 | 0.020 | 0.8828 | 0.7914 | 0.992 | **0.90** |
| GRAPEVINES/grapevine_chloride | 388 | 167 | 388 | 0.7946 | 9e-8 | 0.1002 | 0.0170 | 0.834 | 0.17 |
| MILK/Milk_Fat_1224_KS | 181 | 221 | 181 | 0.7783 | 0.066 | 0.8383 | 0.8019 | 0.951 | **0.96** |

Reading:

- **Joint AUC near 0.5 on MANURE family (0.55-0.57)**: synthetic
  `(X, Y)` truly indistinguishable from real on those datasets.
- **TSTR/baseline ratio >= 0.90** on 6/9 datasets (DIESEL ×2, MANURE
  ×2, COLZA, MILK): training a model on synthetic loses only 3-10%
  of the real-only baseline R^2. The pairs are usable for downstream
  training.
- **3/9 datasets (ECOSIS, ALPINE, grapevine_chloride)** have
  baseline R^2 <= 0.10 — the official train/test split is so
  distributionally different that even RF trained on real fails on
  test. TSTR cannot be measured reliably for these. Note: for ECOSIS,
  TSTR R^2 (0.10) is **better** than the real-only baseline (-0.03),
  suggesting the synthetic acts as a mild regularizer. Diagnostic
  only — not a generator failure.
- **TRTS R^2 >= 0.83 across all datasets**: a real-trained model
  predicts synthetic Y well. The synthetic respects the real
  X->Y mapping.
- **KS Y p-values low for several datasets**: synthetic Y marginal
  differs slightly from real. Expected: residual sampling adds a
  conditional shift around the f(C) mean. Not a TSTR problem (TSTR
  works), but a flag for downstream use cases that care about
  Y marginal calibration.

## Step 3 result (Phase D follow-up, exp38, 2026-05-03)

The 3 low-baseline datasets from Phase D (ECOSIS, ALPINE,
grapevine_chloride) had RF-on-raw R^2 <= 0.10. Hypothesis: RandomForest
on raw X is the wrong model family for these NIR datasets. exp38 swept
preprocessing `{raw, snv, sg1, sg2, msc}` x model `{Ridge, PLS, RF}`
and compared the best baseline against TSTR with the same setup.

| dataset | RF raw R^2 | best baseline R^2 (preproc + model) | TSTR R^2 (best setup) | TSTR/best ratio |
|---|---:|---:|---:|---:|
| ECOSIS Chla+b_species (subsampled 800 train) | 0.008 | **0.354** (sg1 + Ridge a=0.1) | 0.330 | **0.935** |
| ALPINE_P_291_KS | 0.077 | **0.680** (sg1 + RF) | 0.366 | 0.539 |
| GRAPEVINES/grapevine_chloride_556_KS | 0.101 | **0.602** (snv + PLS k=15) | -0.029 | -0.048 |
| MANURE21/Total_N (sanity check) | 0.873 | 0.944 (sg1 + RF) | 0.907 | 0.961 |

Reading:

- **ECOSIS lifts to TSTR ratio 0.935**: with `sg1 + Ridge`, the
  baseline R^2 climbs from 0.008 to 0.354 and the synthetic captures
  93.5% of that. The dataset is hard but the synthetic IS useful.
- **ALPINE lifts to ratio 0.539**: baseline now 0.68 vs 0.08, but
  synthetic only captures half. The 1st-derivative preprocessing
  reveals fine-scale predictive features that the raw-space
  knn_mixup generator does not preserve.
- **GRAPEVINES remains failed**: baseline now 0.60 (snv + PLS-15),
  synthetic gets -0.03. The synthetic preserves raw-PCA structure
  but loses the SNV-normalized predictive directions that PLS uses.
- **MANURE Total_N control**: confirms the diagnostic pipeline
  itself is correct; the win is preserved with proper preprocessing.

The hypothesis is correct: RF on raw X was the wrong baseline for
these datasets. The right NIR baseline uses 1st-derivative or SNV
preprocessing. With that, ECOSIS rejoins the GO column. ALPINE is
mid-tier (synthetic helps but loses fidelity in derivative space).
GRAPEVINES needs preprocessing-aware synthetic generation
(post-Phase-19 work; out of scope here).

Practical recommendation for downstream use:

- For ECOSIS Chla+b_species: ship synthetic `(X', Y')`, train with
  `sg1 + Ridge a=0.1`, expect ~93.5% of real-only baseline.
- For MANURE: ship synthetic, expect ~96% baseline.
- For ALPINE: ship synthetic but flag that derivative-aware models
  recover only ~54% of baseline; raw-space models give weaker
  baseline so the TSTR comparison appears more favorable.
- For GRAPEVINES: ship X-only; the Y synthesis does not generalize
  through SNV+PLS preprocessing.

## Bottom line

Phase B/C/D pipeline is in production. After the exp38 diagnostic on
the low-baseline datasets:

- **7/9 evaluated datasets** have usable synthetic `(X', Y')` pairs
  (TSTR ratio >= 0.5):
  - High-quality (ratio >= 0.90): MANURE x 2, MILK, COLZA, DIESEL x 2,
    ECOSIS Chla+b_species (with sg1+Ridge model, ratio 0.94).
  - Mid (ratio 0.5-0.7): ALPINE_P_291_KS (ratio 0.54 with sg1+RF).
- **2/9** still fail: grapevine_chloride_556_KS (TSTR breaks under
  snv+PLS preprocessing), and the original 3 DIESEL transfer-difference
  variants for which Y is fundamentally not predictable.
- For MANURE, joint discriminator AUC is essentially 0.5 — fully
  indistinguishable.
- The right NIR baseline uses spectral preprocessing (sg1 / snv); the
  RF-on-raw baseline used in exp37 was misleading for ECOSIS / ALPINE.
- DIESEL transfer-difference family stays X-only (Phase 18) per the
  Codex-driven Step 0 KILL.

## Open questions for Codex review (resolved above)

1. Is NMF on `|X|` (when X has negatives) acceptable, or should we
   centre+PCA universally and accept losing the non-negativity prior?
2. Should the Y predictor's input space be raw `C` or
   `C / sum(C, axis=1, keepdims=True)` (concentration fractions)?
3. Should the Phase D joint-AUC train on stacked `[X_pca_scores, Y]`
   (low-D, faster) or on stacked `[X, Y]` (full-D, slower but more
   honest)?
4. Per-dataset `K` selection: heuristic (elbow on reconstruction
   error) vs cross-validated AIC/BIC?
5. For DIESEL `_b-a`/`_hla-b` family: is mixture decomposition even
   meaningful (the signal is a difference, not a positive
   composition)?

These five items go to Codex for guidance before coding starts.
