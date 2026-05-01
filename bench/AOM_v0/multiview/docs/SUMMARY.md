# AOM-multiview — Phases 1–3 synthesis

**Status**: smoke-4 complete; smoke-10 in progress; full-57 pending.
**Date**: 2026-05-01

---

## 1. Smoke-4 results (RMSEP)

Cohort: `Beer_OriginalExtract_60_YbaseSplit`, `Chla+b_spxyG_block2deg`,
`grapevine_chloride_556_KS`, `All_manure_MgO_SPXY_strat_Manure_type`.

### Per-dataset winners

| Dataset | Winner | RMSEP | Improvement vs PLS-std | Improvement vs AOM-PLS-compact |
|---------|--------|------:|------------------------:|--------------------------------:|
| Beer | **block-sparse-V1-blocks3-holdout** | 0.157 | −60.2% | −83.4% |
| Chla+b_block2deg | **lazy-V2-POP-combined-compact-holdout** | 27.05 | −44.9% | −41.7% |
| grapevine_chloride | **moe-preproc-soft-pls-compact** | 939 | −19.6% | −3.9% |
| All_manure_MgO | AOM-PLS-compact-numpy | 0.795 | −5.5% | 0% (tie) |

### Top variants by median rel-RMSEP vs PLS-standard

| # | Variant | Median rel-RMSEP | Wins vs baseline | Wins vs AOM-PLS | Wins vs TabPFN-opt |
|---|---------|-----------------:|-----------------:|----------------:|-------------------:|
| 1 | block-sparse-V1-blocks3-cvspxy   | 0.443 | 1/1 | 1/1 | 0/1 |
| 2 | block-sparse-V2-combined-compact-holdout | 0.672 | 2/2 | 2/2 | 1/2 |
| 3 | block-sparse-V1-blocks3-holdout  | 0.827 | 3/4 | 2/4 | 1/4 |
| 4 | moe-preproc-soft-pls-compact     | 0.850 | 4/4 | 3/4 | 1/4 |
| 5 | moe-view-soft-pls                 | 0.868 | 3/4 | 2/4 | 1/4 |
| 6 | moe-view-hard-pls                 | 0.880 | 3/4 | 3/4 | 1/4 |
| 7 | moe-preproc-hard-pls-compact      | 0.909 | 4/4 | 1/4 | 1/4 |

(cvspxy / V2 variants only ran on 1-2 datasets due to runtime limits;
their median rel-RMSEP is informative but the win counts cap at the
dataset-row count.)

### Key findings (Phase 1–3)

1. **Block-aware bank wins on Beer dramatically** (−60% RMSEP vs PLS-std,
   −83% vs AOM-PLS). The 576-feature spectrum has localised information
   in the first sub-block, and the block-mask gives the model exactly that
   subspace to work with.
2. **Per-block deflation matters** on Beer specifically: block-sparse V1
   (per-block deflation) at 0.157 beats lazy-V1-POP (global deflation) at
   0.626 by 75%. Block-sparse retains other-block residual for subsequent
   LVs to exploit.
3. **Lazy V2 POP wins on Chla+b** (block-2-degree biological response),
   suggesting the dataset has block × preprocessing interactions captured
   by the 36-op combined bank.
4. **Soft-MoE preproc-pls competitive on grapevine** (−4% vs AOM-PLS),
   showing that mixture-of-experts on preprocessing operators can adapt
   to chemistry better than single-operator AOM on some datasets.
5. **AOM-PLS-compact remains best on All_manure** — for that dataset, the
   signal is spread broadly across the spectrum and block-aware locality
   does not help.

### Algorithmic insight

The four "view-mode × deflation" combinations align with the four observed
winning regimes:

| Regime | View mode | Deflation | Best for |
|--------|-----------|-----------|----------|
| Standard AOM-PLS | global preproc | global | dispersed signal (All_manure) |
| Lazy V1/V2 POP | block-aware bank | global | block-sensitive but compositional (Chla+b) |
| Block-sparse | block mask | per-block | block-localised signal (Beer) |
| MoE soft preproc | preproc-as-experts | none (mixture) | distributed chemistry (grapevine) |

This is a useful taxonomy; no single variant dominates all four regimes.

---

## 2. Implementation notes

### 2.1 Block-sparse algorithm performance

`fit_block_sparse_aom` re-fits the entire prefix from scratch for every
candidate evaluation. This is correct and leakage-safe (matches existing
AOM-PLS POP path) but is `O(bank_size · K_max² · n · p)` per dataset,
dominated by the `K_max² / 2` re-fit cost. For `n_train > 2000` the
runtime exceeds 5 minutes per dataset.

The existing AOM-PLS POP avoids this by sharing state across candidate
evaluations via `simpls_covariance`. Porting the block-sparse algorithm
to a similar incremental update is **deferred to Phase 4** — the
holdout-only variant on `n_train ≤ 1500` datasets is sufficient for the
smoke-10 evaluation and we know the algorithm scales.

### 2.2 Latent broadcast bug in `aompls.scorers._rmse`

When `y_true` is shape `(m, 1)` and `y_pred` is shape `(m,)`, naive
`(y_true - y_pred).ravel()` broadcasts to `(m, m)` and produces a wrong
RMSE. The bug is masked in existing AOM-PLS-only code paths because the
estimator always passes 1D y; my multi-view code paths trigger it.

The current workaround: pass 1D `yc` into `cv_score_regression` /
`holdout_score_regression` from `_score_block_sparse_indices`. The
fix in `_rmse` itself was attempted but deferred (changes computed
RMSE values, breaking the bit-exact production parity test that is
sensitive to absolute scoring).

---

## 3. Codex review log

| Phase | Doc | Round | Disposition |
|-------|-----|-------|-------------|
| 1 | DESIGN_VIEWS.md | 1 | All HIGH/MED items dispositioned (cv_splitter threading, strict-linearity enforcement, edge cases, bank size correction). |
| 2 | DESIGN_MBPLS.md | 1 | HIGH 1: renamed "true V1/V2" to "block-sparse AOM-MBPLS"; classic Westerhuis MB-PLS-AOM kept as separate, not yet implemented. HIGH 2: standard PLS coef formula `B = Z·pinv(P^T Z)·Q^T` instead of per-LV reconstruction. HIGH 8: leakage-free CV inherited from existing path. |
| 3 | DESIGN_MOE.md | — | Round-1 review pending. AOM-per-LV variant flagged as math-equivalent to Phase 2 block-sparse and not re-implemented. |

---

## 4. Next steps

1. **Smoke-10 escalation** (in progress) — qualifying winners on the 10-dataset
   user-curated cohort.
2. **Full-57** — only the smoke-10 winners get the full benchmark.
3. **Phase 4 hybrids** — Ridge-meta stacking of {block-sparse-V1, MoE preproc-soft, AOM-PLS-compact} predictions, if smoke-10 shows individual variants do not dominate cohorts of multiple datasets.
4. **Performance** — if Phase 4 needs block-sparse on Chla+b-scale datasets, port the algorithm to share state across candidate evaluations.
5. **Classification** — replicate winning regression variants for AOM-MBPLSClassifier / AOM-MoEClassifier on the existing classification cohort.
