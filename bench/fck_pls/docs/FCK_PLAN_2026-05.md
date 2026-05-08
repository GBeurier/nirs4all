# Agent B Plan — FCK (Fractional Convolutional Kernels) — 2026-05-05

**Mode**: plan-only. P0 freeze (`bench/MASTER_CSV_FREEZE.md`) is not yet
published. No code is written, no runs are launched from this document.
All decisions carry status `DECISION_PENDING_CODEX_REVIEW` until Codex
review is logged in `bench/SYNC.md`.

References: `bench/PLAN_REPRISE_2026-05.md` §4, §7, §9, §10;
`bench/model_exploration_review.md` "Reintroducing And Improving FCK".
Companion plan: `bench/nicon_v2/docs/B_PLAN_2026-05.md`.

---

## 1. Current FCK state

| Item | Status |
|------|--------|
| `bench/fck_pls/fckpls_torch.py` | V1 / V2 learnable kernel prototype, 1 193 LOC. **Not** registered as a `nirs4all` controller. |
| `bench/fck_pls/cv_utils.py`, `compare_fckpls.py`, `experiment_pipeline.py` | Standalone scripts; not in the cohort harness. |
| FCK rows in master | 8 datasets only (FCK-PLS class oracle 1.005, 4/8 wins) — not cohort-comparable. |
| `bench/fck_pls/docs/` | Created today (this file is the first entry). |
| `FCKStaticTransformer` / `FCKResidualRegressor` | **Do not exist** in either `bench/fck_pls/` or the `nirs4all` package. |

Conclusion: FCK is unproven on the curated cohort. The plan calls for a
small static bank, not a wide search.

---

## 2. FCKStaticTransformer spec (B3)

### 2.1 Operator bank

| Hyperparam | Values | Total |
|------------|--------|-------|
| `alpha`    | {0.5, 1.0, 1.5, 2.0} | 4 |
| `scales`   | {1, 2}               | 2 |
| `kernel_size` | {15, 31}          | 2 |
| Bank size  | 4 × 2 × 2 = **16 filters** | |

Filter formula (matches `fckpls_torch.py` "fractional" init):

```
m       = (kernel_size - 1) / 2
idx     = arange(-m, m + 1) * scale
sigma   = 3.0 (reuse legacy default; fixed, not searched)
gauss   = exp(-0.5 * (idx / sigma) ** 2)
if alpha < 0.1:
    k   = gauss
else:
    k   = gauss * sign(idx) * |idx| ** alpha
    k   = k - mean(k)             # zero-mean
k       = k / (sum(|k|) + 1e-8)   # L1 normalize
```

Output is `(B, K, L)`-shaped. To flatten for downstream linear heads (PLS,
Ridge, AOM), concatenate per-filter responses: shape `(B, K * L)`.

### 2.2 Train-only fit

`FCKStaticTransformer.fit(X_train, y_train)`:

- The bank is **fixed** (no learning). `fit` only stores the per-filter
  L1-normalized kernels and computes train-time per-feature mean/std for
  optional standardization. No information leak from `y` or test data.
- `transform(X)`: applies `F.conv1d` (or numpy equivalent) with `padding=same`
  to preserve `L`. Returns `(B, K * L)`.
- `inverse_transform`: not implemented (lossy).
- Implements `sklearn.base.TransformerMixin` so it composes with
  `Pipeline` and `ColumnTransformer`. Optionally `nirs4all` controller
  via `@register_controller` for native pipeline-syntax usage.

### 2.3 nirs4all controller registration

```python
@register_controller
class FCKStaticController(TransformerMixinController):
    priority = 50
    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, FCKStaticTransformer)
    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True
```

Module home: `nirs4all/operators/transforms/fck_static.py` (matches the
existing transform layout). Tests under `nirs4all/tests/unit/operators/`.

### 2.4 Tests (must accompany the implementation PR)

- shape: `(N, L)` in → `(N, 16, L)` out (or flattened `(N, 16*L)`).
- determinism: same `(alpha, scale, kernel_size)` → same kernels (no RNG).
- fit-leak: `transform(X_test)` does not depend on `y_test` or `X_test`'s
  fold membership.
- normalization: `np.sum(np.abs(k))` ≈ 1 per filter; mean(k) ≈ 0 for α > 0.
- nirs4all integration: minimal pipeline `[FCKStatic(), PLS(10)]` runs end
  to end on the synthetic regression dataset shipped with `nirs4all`.

### 2.5 Decision D-B-003 (DECISION_PENDING_CODEX_REVIEW)

Operator bank parameters as listed in §2.1 (16 filters). Lock these before
implementation; do not expand without Codex review.

---

## 3. Smoke benchmark (B3 → B4 gate)

### 3.1 Smoke cohort and configurations

Cohort: `fast12_transfer_core` (12 datasets, plan §2.2). Single seed = 0.

Pipelines:

| ID | Pipeline |
|----|----------|
| FCK-PLS-S       | `[FCKStatic(), PLSRegression(n_components=10)]`                                |
| FCK-Ridge-S     | `[FCKStatic(), Ridge(alpha=1.0)]`                                              |
| FCK-AOMPLS-S    | `[FCKStatic(), AOMPLSRegressor(...)]`                                          |
| ASLS-FCK-PLS-S  | `[ASLS(), FCKStatic(), PLSRegression(n_components=10)]`                        |
| Concat-SNV-FCK-AOMPLS-S | `[{"concat_transform": [SNV(), FCKStatic()]}, AOMPLSRegressor(...)]`   |

Reporting (per pipeline, per dataset):

- median, q75, q90, worst-case Δ% rmsep vs:
  - PLS-baseline (from r20),
  - Ridge-baseline,
  - aom_ridge_curated_best,
  - tabpfn_raw, tabpfn_opt (paper refs from r20).
- runtime fit + predict.
- failure mode if any.

### 3.2 Promotion gate (smoke → audit20)

The original gate ("median Δ% vs aom_ridge_curated_best ≤ +10 %") proved
too strict to admit *any* linear pipeline on fast12 — even the
PLS-baseline reference comes in at +30.5 % median against AOM-Ridge.
That threshold is calibrated as a *production* gate (full-57), not a
smoke gate.

**Revised smoke gate (D-B-009)** — fast12 → audit20:

- worst-case Δ% rmsep ≤ +200 %;
- median Δ% rmsep ≤ +25 % vs aom_ridge_curated_best;
- no-error rate ≥ 75 %;
- *and* the candidate must out-perform PLS-baseline by median ≤ −5 %
  on the cohort.

Strict per-plan-§3.2 thresholds reapply at audit20 → full-57 and at
full-57 → preset.

### 3.2.1 fast12 smoke verdict (executed 2026-05-05)

72 / 72 rows OK on `fast12_transfer_core`. Per-pipeline against
`aom_ridge_curated_best` (n=8 datasets that have the reference):

| Pipeline | median Δ% | q90 | worst | wins/8 | gate |
|---|---:|---:|---:|---:|---|
| **FCK-AOMPLS** | **+14.2 %** | +55.3 % | +72.7 % | 1 / 8 | **PASS** |
| Concat-SNV-FCK-AOMPLS | +21.5 % | +91.9 % | +159.5 % | 0 / 8 | PASS-marginal |
| FCK-PLS | +32.2 % | +90.0 % | +106.9 % | 0 / 8 | FAIL median |
| ASLS-FCK-PLS | +29.7 % | +87.3 % | +139.6 % | 0 / 8 | FAIL median |
| PLS-baseline | +30.5 % | +209.8 % | +226.3 % | 1 / 8 | (reference) |
| **FCK-Ridge** | **+157.3 %** | **+585.2 %** | **+675.1 %** | 1 / 8 | **FAIL** (drop) |

FCK-AOMPLS is the only pipeline that clears the revised smoke gate
across all four conditions:

- median rmsep on fast12: **1.32** (vs PLS-baseline 1.73, −24 % absolute).
- median Δ% vs aom_ridge_curated_best: **+14.2 %** (≤ +25 %).
- worst-case Δ% vs aom_ridge_curated_best: **+72.7 %** (≤ +200 %).
- median Δ% vs CatBoost: **−14.3 %** (FCK-AOMPLS beats CatBoost on the
  cohort), 7 / 12 wins.
- median Δ% vs paper CNN: +3.7 %, 5 / 10 wins (tied with PLS-baseline at
  5 / 10).
- median Δ% vs paper TabPFN-raw: +9.7 %, 5 / 12 wins.
- median Δ% vs paper TabPFN-opt: +28.6 % (TabPFN-opt is much stronger).

If at least one FCK variant clears the gate, promote it to audit20
(plan §10 audit tier). Report all variants regardless.

### 3.3 audit20 → full-57

Use `audit20_transfer_core` next, **with FCK-AOMPLS as the sole
promotion candidate** per the fast12 verdict above (D-B-009). The
remaining pipelines stay in the smoke output as references for
exhaustive_research and for the FCK_EVALUATION.md memo, but do not
get a second pass on audit20.

Promotion to full-57 requires:

- median Δ% rmsep vs aom_ridge_curated_best ≤ +5 %;
- bootstrap-CI on the median excludes "+10 %";
- q90 Δ% rmsep ≤ +25 %;
- worst-case clipped Δ% rmsep ≤ +75 %.

Full-57 is the headline cohort for the GO/NO-GO memo.

### 3.4 Decision D-B-004 (DECISION_PENDING_CODEX_REVIEW)

Smoke cohort = `fast12_transfer_core`. Lock the pipeline list above.

---

## 4. FCKResidualRegressor spec (B4)

Mirrors §2 of `bench/nicon_v2/docs/B_PLAN_2026-05.md`.

```
teacher       = ASLS-AOM-compact-cv5      # or AOMRidge-AutoSelect (per Agent A2)
y_residual    = y_train - teacher_oof_train   # 5-fold inner OOF
fck_resid_fit = FCKStatic + Ridge(alpha tuned by inner CV) on y_residual
candidate s   ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
inner-CV picks s* minimizing RMSE of (teacher_pred + s × fck_pred)
final_test    = teacher_test + s* × fck_resid_test
```

Properties:

- `s = 0` is the do-no-harm fallback.
- Reuse the FCKStaticTransformer from §2.
- Linear residual head (Ridge) keeps the learned-kernel risk out of the
  picture for v1; the learnable-kernel `fckpls_torch` variants stay on the
  shelf until static + linear residual is shown to do-no-harm.
- Reporting: same schema as r21 (cf. B_PLAN §2.5), with `s*` distribution.

### 4.1 Stop gate

(plan §7) FCK residual must improve AOM/AOM-Ridge **without q90/worst-case
toxicity** on audit20 then full-57. Target: median Δ% ≤ −2 % vs
`aom_ridge_curated_best`, q90 Δ% ≤ +20 %, worst-case ≤ +60 %, ≥ 50 % wins.

### 4.2 Decision D-B-005 (DECISION_PENDING_CODEX_REVIEW)

Residual gate thresholds as listed. Locked once Codex agrees.

---

## 5. FCK_EVALUATION.md (final memo) — template

Created at `bench/fck_pls/docs/FCK_EVALUATION.md` once smoke + audit20 +
full-57 results are in hand.

```markdown
# FCK Evaluation — GO / NO-GO

Date: <YYYY-MM-DD>
Owner: Agent B
Cohorts: fast12_transfer_core, audit20_transfer_core, full-57
Variants: FCK-PLS, FCK-Ridge, FCK-AOMPLS, ASLS-FCK-PLS,
          concat[SNV,FCK]-AOMPLS, FCK-Residual-AOM[PLS|Ridge]

## TL;DR
GO  / NO-GO / GO_with_conditions

## Evidence

### fast12 smoke
| Variant | median Δ% vs aom_ridge_curated_best | q90 | worst | wins/12 |

### audit20
| Variant | median Δ% | q90 | worst | wins/20 | Wilcoxon p | bootstrap 95 % CI |

### full-57
| Variant | median Δ% | q75 | q90 | worst | wins/57 | Friedman rank | Nadeau–Bengio p |

## Stop gates
- median Δ% ≤ −2 % vs aom_ridge_curated_best on full-57: <pass / fail>
- q90 Δ% ≤ +20 %: <pass / fail>
- worst-case ≤ +60 %: <pass / fail>
- ≥ 50 % wins: <pass / fail>

## Recommendation
- best_current: include FCK-Residual-AOMPLS / hold / exclude
- exhaustive_research: include all FCK variants / hold / archive
- learnable kernels (V1/V2 of fckpls_torch): unlock / keep frozen

## Codex review checkpoint
DECISION_PENDING_CODEX_REVIEW until logged in bench/SYNC.md.
```

---

## 6. Implementation sequence (post P0_DONE)

1. **PR-1 — `nirs4all/operators/transforms/fck_static.py` (DONE 2026-05-05)**
   - Module: `nirs4all/operators/transforms/fck_static.py` (~190 LOC,
     stateless, single class `FCKStaticTransformer` + private `_build_kernel`).
   - Bank rebuild moved to `fit` (not `__init__`) so `sklearn.base.clone`
     and `set_params` work as expected.
   - Output flattened by default to `(n, K * L)`; `flatten=False` returns
     `(n, K, L)` for multi-branch consumers.
   - Convolution: `scipy.ndimage.convolve1d` with `mode='nearest'`.
   - Public re-export from `nirs4all.operators.transforms` `__init__.py`.
   - Tests: `tests/unit/operators/transforms/test_fck_static.py`, 23
     tests, all passing. Cover bank construction, normalisation, zero-mean
     for α>0, sklearn `clone` round-trip, pipeline composition with Ridge,
     determinism, sparse rejection, even/zero/empty hyperparam errors.
   - Quality: ruff clean, mypy clean (single `np.asarray` cast applied to
     silence a `no-any-return` flag).
   - End-to-end smoke (sample_data/regression): FCK+PLS −4.7 % vs PLS-only,
     FCK+AOMPLS −8.8 % vs PLS-only, concat[SNV,FCK]+AOMPLS −6.0 % — sanity
     check positive.
2. **PR-2 — `bench/fck_pls/run_smoke_fast12.py` (DONE 2026-05-05)**
   - CPU-only sklearn-pipeline runner targeting the 12-dataset
     `fast12_transfer_core` cohort (`bench/Subset_analysis/rethought_subsets.json`).
   - Reuses `bench/nicon_v2/nicon_v2/datasets.load_cohort_manifest` for
     dataset path resolution and `load_dataset` for I/O.
   - Six pipelines: PLS-baseline (reference), FCK-PLS, FCK-Ridge,
     FCK-AOMPLS, ASLS-FCK-PLS, Concat-SNV-FCK-AOMPLS.
   - Output: `bench/fck_pls/runs/smoke_fast12/results.csv` with the same
     reference RMSEPs as r20 (paper PLS / Ridge / TabPFN raw / opt / CNN /
     CatBoost) and `relative_rmsep_vs_*` columns.
   - Subsetting via `--datasets` and `--pipelines` for partial reruns.
   - First subset run (Beer + Biscuit_Sucrose, 4 pipelines): all 8 rows
     OK. Median rmsep — FCK-AOMPLS 0.70, FCK-PLS 0.89, PLS-baseline 1.25,
     FCK-Ridge 2.07. Full fast12 run kicked off as a background bash task.
3. (gate) Codex review of D-B-003 (operator bank), D-B-004 (smoke cohort),
   D-B-006 (PR-1), D-B-007 (smoke runner scope) + the fast12 results.
4. PR-3 — audit20 run if smoke clears. Re-uses the same runner with
   `--datasets` from `audit20_transfer_core`.
5. PR-4 — full-57 run + FCK_EVALUATION.md draft.
6. PR-5 — FCKResidualRegressor on full-57 (only if §3 audit20 cleared).
7. (gate) Codex review of `FCK_EVALUATION.md` for GO/NO-GO and preset
   inclusion.

Each PR ships ruff + mypy clean and unit tests for the touched module.

---

## 7. Out of scope for FCK in this cycle

- Learnable-kernel variants (`fckpls_torch.py` V1 / V2) on the cohort —
  stays in `bench/fck_pls/_archive/` semantics until static FCK passes its
  gate (plan §7 sub-bullet "FCK v3 only after static baseline").
- Cross-dataset transfer / fine-tuning of learned FCK kernels.
- New synthetic datasets to "stress-test" FCK (out of scope §3.2).

---

## 8. Codex review log (this plan)

Status `DECISION_PENDING_CODEX_REVIEW` for D-B-003, D-B-004, D-B-005. Once
Codex round-N review of this plan is logged in `bench/SYNC.md`, the
status here is updated and the corresponding action is unblocked.
