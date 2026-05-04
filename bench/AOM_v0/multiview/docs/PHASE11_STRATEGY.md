# Phase 11 strategy — pushing past Phase 10 mean-ensemble

## Current state (after Phase 10)

| Best by | Variant | Score on full-57 |
|---------|---------|------------------|
| Median rel-RMSEP | `mean-ensemble-4-fixed` | 0.883 |
| Wins vs PLS-std | `mean-ensemble-4-fixed` | 49/61 (80%) |
| Wins vs AOM-PLS | `mean-ensemble-4-fixed` | 46/61 (75%) |
| Wins vs TabPFN-opt | `moe-view-multiK-wide-2-10` | 16/61 (26%) |
| Oracle (multi-view) | per-dataset best | 22/58 vs TabPFN-opt |
| Meta-selector | `meta-logreg` | 15/58 vs TabPFN-opt |

The mean-ensemble (equal-weight average of test predictions across 4 bases)
achieved a 4.6% median improvement over the previous best by exploiting
uncorrelated errors. But: equal weights may be suboptimal; the meta-selector
ceiling is +6 wins over best single but we only realise +1.

## Hypothesis

Three improvements should push the frontier:

1. **Per-dataset learned ensemble weights** beat equal-weights when bases
   have systematically different performance on different datasets.
2. **Per-sample weights** beat per-dataset weights when bases have
   different reliability on different samples within a dataset.
3. **Bagged bases** (multi-seed averaged) reduce variance for each base
   before they are averaged in the outer ensemble — variance reduction
   is multiplicative across the two layers.

## Proposed Phase 11 directions

### Direction A: Cross-fitted Ridge stacking (most principled)

**Algorithm:**
1. Fit each base via 5-fold inner CV on training data → OOF predictions
   `Z ∈ R^(n_train × n_bases)`.
2. Fit `Ridge(α=auto)` meta on `(Z, y_train)` — learns per-base weights
   from training-only OOF predictions, no leakage.
3. Refit each base on full training data.
4. At predict: get base predictions on test, apply learned Ridge weights.

**Bases to include (Phase 10 winners + diverse complements):**
- `multiK-wide-2-10` (best TabPFN, 16/61)
- `mean-ensemble-3-fixed` (best median, 0.887) — note: ensemble-of-ensembles
- `moe-preproc-soft-pls-compact` (47/61 vs PLS)
- `lazy-V2-AOM-combined-compact` (block-aware variant)
- `AOM-PLS-compact-numpy` (single-LV operator AOM)

**Cost:** 5 bases × 5 OOF folds + 5 final fits = 30 fits per dataset.
At ~2s per AOM fit, ~60s per dataset, total ~60 min on full-57.

**Risk:** Ridge meta overfits on small datasets (n_train < 100). Mitigation:
fall back to equal-weight when CV R² of Ridge weights is below 0 (i.e.
Ridge is no better than equal-weight on training OOF). Smoke-4 has Beer at
n=40 — high risk.

### Direction B: Per-sample blending via Ridge on `[X_pca | OOF_preds]`

Same as `AOMMoEStacked` from Phase 9 (which underperformed) but with the
Phase 10 ensemble as a *base*, not a meta. Idea: learn that for some
samples mean-ensemble works, for others a single specialist works.

**Algorithm:**
1. For each dataset, generate OOF predictions of each base.
2. Train `Ridge` on `[PCA(X) | OOF_base_preds]` → y. Allows per-sample
   reweighting via the linear interaction with PCA features.
3. At predict: append `PCA(X_test)` and base predictions, run Ridge.

**Risk:** PCA features may dominate the Ridge if their variance scales
out-of-balance with prediction variance. Mitigation: standardise both
inputs to unit variance before Ridge.

### Direction C: Multi-seed bagging per base

Each base in mean-ensemble-4-fixed is fit once with `random_state=0`. Bag
each base over 5 seeds (0..4), average within each base, then average
across bases. Total 4 × 5 = 20 sub-fits per dataset.

**Cost:** 5x the Phase 10 mean-ensemble cost.
**Risk:** AOM-PLS / multiK might be insensitive to seed (deterministic
operator selection) — bagging only reduces variance from the OOF gate.
Need to verify base-level seed sensitivity on smoke-4 first.

## Comparative pros/cons

| Approach | Median ↑? | Wins ↑? | Cost | Risk |
|----------|----------:|--------:|-----:|------|
| A (Ridge stack) | likely | likely | 60 min | overfits at n<100 |
| B (per-sample) | maybe | maybe | 60 min | already failed in Phase 9 |
| C (multi-seed) | small | small | 30 min | base may be deterministic |

## Recommendation (pre-Codex)

Implement A first; smoke-4 → smoke-10 → full-57 protocol. If A doesn't
beat mean-ensemble-4-fixed by ≥ 1% median, try C. Skip B unless A and C
both fail (Phase 9 stacked-K3 already showed per-sample Ridge is fragile).

Add `aom-ridge-fast` to A's base set if budget allows — it covers DIESEL/brix
where PLS-family loses to Ridge (~5 datasets where AOM-Ridge wins outright).

## Codex review questions

1. **§A — Ridge meta regularisation**: should we use `RidgeCV(alphas=...)`
   inside the meta to auto-tune α per dataset, or a fixed α=1.0? On
   small datasets (n=40 Beer), CV-tuned α may collapse to very high
   regularisation and degrade to equal-weights. Is auto-α actually
   safer, or should we hard-code conservative α?

2. **§A — Negative weights**: should we allow Ridge to assign negative
   weights to bases (boosting/anti-correlation effects), or constrain
   to non-negative (NNLS)? NNLS is more interpretable but can leave
   accuracy on the table. Phase 4 stack-ridge vs stack-nnls both ran;
   Ridge marginally better.

3. **§A — Including ensemble as base**: `mean-ensemble-3-fixed` is itself
   an average of 3 bases. Including it in the Ridge meta alongside its
   constituents creates collinearity (ensemble = mean of three columns
   that are also in the base set). Drop the constituents when including
   the ensemble, or vice versa?

4. **§A — Generalisation**: with 5 bases × 5 OOF folds, Ridge is fit on
   `(n_train, 5)` data. For n_train ≤ 50 (Beer), this is heavy
   regularisation territory. Empirically more reliable to:
   (a) use equal weights as anchor and learn small deltas,
   (b) skip Ridge meta and use equal weights when n_train < threshold.
   What threshold makes sense — 100? 200?

5. **§C — base seed sensitivity**: AOM-PLS uses deterministic
   operator selection given the data; multiK has random fold splits in
   the inner OOF but the experts themselves are deterministic. Is bagging
   actually buying variance reduction here, or just averaging
   deterministically-similar predictions and adding compute for nothing?

6. **General**: am I missing a stronger ingredient? E.g., calibrated
   linear regression of a single base's prediction against y on a tiny
   held-out set per dataset (a "1-parameter post-fit calibration"). Or
   conformal-prediction-style per-sample weighting. Are there published
   approaches I should consider?
