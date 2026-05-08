# Nested-CV Audit — `AOMRidge-AutoSelect-headline-spxy3` and `AOMRidge-Blender-headline-spxy3`

**Owner**: Agent A (Classical Production Pack)
**Date**: 2026-05-05 (Codex round 1 verdict: 17:00 CEST)
**Status**: `DECISION_LOCKED` (Codex round 1: CONFIRM, with strengthened
promotion gate — see §7).
**Verdict (locked)**: **NESTED — no leakage between the predefined held-out
test set `Xte` and the selector's inner mechanism.** Codex independently
verified the §3-§5 code-trail line numbers against the source files and
confirmed the anti-leakage invariants hold under all current code paths.

This document is the formal companion to `bench/SYNC.md` 2026-05-05 (Agent A) entry §A. It records exact code paths, line numbers, and the five caveats Codex must weigh before promoting either variant out of `protocol_maturity=exploratory`.

---

## 1. What is being audited

Two estimators in `bench/AOM_v0/Ridge/aomridge/`:

- `AOMRidgeAutoSelector` (`auto_selector.py:420`) — picks one of 8 HEADLINE candidates by outer-CV mean RMSE, refits the winner on full `Xtr`, predicts `Xte`.
- `AOMRidgeBlender` (`blender.py:163`) — computes OOF predictions per candidate via outer-CV, solves a regularised simplex QP for blend weights, refits all candidates on full `Xtr`, blends predictions on `Xte`.

Both are evaluated by `_run_variant(variant, Xtr, ytr, Xte, yte, ...)` in `bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py:706`.

The "spxy3" suffix encodes the **selector's internal outer-CV depth** (`SPXYFold(3)`), not a 3-fold test split: the master CSV reports a single `Xte` RMSE per `(dataset, seed)` with `evaluation_split == "test"` (verified on 135 Blender + 121 AutoSelect rows).

## 2. Outer protocol (predefined test split)

`run_aomridge_benchmark.py:706-797`:

```text
_run_variant(variant, Xtr, ytr, Xte, yte, seed, cv_obj, cv_splits=3, dataset_name=None):
    ...
    if variant.selection in ("auto_select", "blender"):
        ...
        est = _Selector(candidates=cand_specs, outer_cv=outer_splits,
                        outer_cv_kind=outer_kind, outer_cv_repeats=outer_repeats,
                        random_state=seed, n_jobs=n_jobs, **extra)
        est.fit(Xtr, ytr)              # selector sees Xtr only
        yhat = est.predict(Xte)        # Xte never reaches selector internals
```

`Xte` is never seen during fit. `evaluation_split=="test"` in the master CSV confirms this is the held-out partition.

## 3. AutoSelector inner mechanism

`auto_selector.py:_score_candidate`, lines **371-412**:

```text
for tr_idx, va_idx in folds:
    X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    est, branch = _dispatch_candidate(spec, seed=seed, inner_cv=3)
    X_tr, X_va = _apply_branch(branch, X_tr_raw, y_tr, X_va_raw)
    est.fit(X_tr, y_tr)                # candidate sees outer-train only
    y_pred = est.predict(X_va)
```

Three anti-leakage guarantees inside this loop:

- **Branch preprocessing fitted on outer-train only.** `_apply_branch` (`auto_selector.py:193-222`) calls `fit_transform_branch(preproc, X_train, y_train)` on outer-train rows; outer-validation is only `transform`-ed.
- **Inner CV materialised against outer-train slice only.** When the candidate is `AOMRidgePLSCV` with `cv_kind="spxy_repeated"`, dispatch (`auto_selector.py:142-144`) builds:

  ```text
  cv_for_inner = RepeatedSPXYFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed)
  ```

  This splitter is then handed to the candidate which calls `cv_for_inner.split(X[tr_idx], y[tr_idx])` — outer-validation rows never enter the splitter.
- **Fresh estimator per fold.** `_dispatch_candidate(spec, seed, inner_cv)` returns a brand-new `BaseEstimator` instance every call; no fitted state survives across folds.

Refit step (`auto_selector.py:561-573`):

```text
refit_est, branch = _dispatch_candidate(best_spec, seed=seed, inner_cv=3)
if branch:
    X_refit = fit_transform_branch(refit_branch_preproc, X, y_arr)  # X = full Xtr
refit_est.fit(X_refit, y_arr)                                       # full Xtr only
```

Refit operates on full `Xtr`; `Xte` is never visible at any point of `fit`.

## 4. Blender OOF construction

`blender.py:_oof_predictions_for_candidate`, lines **57-95**:

```text
for tr_idx, va_idx in folds:
    X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
    y_tr = y[tr_idx]
    est, branch = _dispatch_candidate(spec, seed=seed, inner_cv=3)
    X_tr, X_va = _apply_branch(branch, X_tr_raw, y_tr, X_va_raw)
    est.fit(X_tr, y_tr)
    y_pred = est.predict(X_va)
    oof[va_idx] = y_pred
```

Same fold-level pattern as AutoSelector.

QP solve (`blender.py:_solve_simplex_qp`, lines **103-155**):

```text
def _solve_simplex_qp(Z, y, regularizer):
    # Z: OOF predictions stacked from outer-CV. y: y_train.
    ... minimise 0.5 * ||y - Z w||^2 + 0.5 * lambda * ||w - 1/K||^2 ...
```

Inputs `Z` and `y` are both training-side; `y_te` does not enter the optimisation.

Final-blend step (`blender.py:333-365`):

```text
for spec in candidates:
    est, branch = _dispatch_candidate(spec, seed=seed, inner_cv=3)
    if branch:
        X_refit = fit_transform_branch(branch_preproc, X, y_arr)   # X = full Xtr
    est.fit(X_refit, y_arr)
    refit_estimators.append(est)
```

Every candidate is refit on full `Xtr` (not on outer-train slices). `predict(Xte)` (`blender.py:367-397`) then stacks per-candidate predictions and applies `weights_`.

## 5. Recursion guard

`blender.py:_normalise_candidates`, lines **264-272**:

```text
out = [
    spec for spec in base
    if spec.get("selection") not in ("auto_select", "blender", "residual_tabpfn")
]
```

The Blender drops aggregator candidates from its own pool. The benchmark dispatcher (`run_aomridge_benchmark.py:766-771`) repeats the same filter:

```text
cand_specs = [
    _variant_to_spec(v) for v in HEADLINE_VARIANTS
    if v.label != variant.label
    and v.selection not in ("auto_select", "blender", "residual_tabpfn")
]
```

No recursion path observed.

## 6. Caveats Codex must weigh

| # | Caveat | Concern |
|---|---|---|
| C1 | **Single `seed=0`** for all 135 Blender + 121 AutoSelect rows in master CSV (`run_seed` empty, `cv_protocol` empty). §10.2 production tier requires Nadeau-Bengio + Friedman-Nemenyi which need ≥3 seeds | **HIGH** — blocks `best_current` claim |
| C2 | Outer-CV inside selector = `SPXYFold(3)` (3 folds, single repeat). Selector variance is high on small-n datasets; QP weights / variant ranking can flip on a different seed | **MEDIUM** — does not invalidate the nesting verdict |
| C3 | Naming `-spxy3` encodes selector inner-CV depth, not test-split protocol. Master CSV reports a single `Xte` RMSE per `(dataset, seed)` — predefined split, not nested-CV around the selector | **MEDIUM** — clarify in the registry |
| C4 | Coverage = 53/57 cohort. Missing: `Brix_spxy70`, `LUCAS_SOC_Cropland_8731_NocitaKS`, `Malaria_Oocist_333_Maia`, `Malaria_Sporozoite_229_Maia` (same set as `RETHOUGHT_SUBSETS.md` "Coverage Caution") | **LOW** — already known |
| C5 | A few datasets have 3-5 rows from re-runs across `source_run` (`all54_combined`, `all54_headline`, `diverse_iter3_*`, `v5b_*`). De-duplication or per-`source_run` realisation tagging needed | **MEDIUM** — depends on P0 rules |

C1 is the gate: even with the nesting verdict, promotion to `strong_practical` requires a multi-seed re-run on `fast12_transfer_core` per the protocol in `bench/PLAN_REPRISE_2026-05.md` §10.2.

## 7. Promotion gate (locked, Codex-strengthened)

| Tier | Required evidence | Today |
|---|---|---|
| `exploratory` | Verdict NESTED, single seed | **Met** — current state |
| `audit` (`audit20_transfer_core`) | NESTED + ≥3 seeds on **both** `fast12_transfer_core` AND `audit20_transfer_core` + Wilcoxon paired vs `ASLS-AOM-compact-cv5-numpy` and `Ridge-tuned-cv5` | Pending C-delivered harness hardening |
| `strong_practical` | Audit-tier evidence + full-57 multi-seed completed + sign tests stable | Pending |
| `best_current` | Strong_practical + Friedman-Nemenyi + Nadeau-Bengio per plan §10.2 | Pending |

Codex round 1 strengthened the original gate: ≥3 seeds were originally
required on `fast12_transfer_core` only; Codex requires them on both fast12
and audit20 before any promotion. Full-57 stats are required before
`best_current` (no shortcut from a strong audit20 result).

The freeze (Agent C-bootstrap, 2026-05-05 14:30) tagged both variants
`protocol_maturity=exploratory`. This audit confirms keeping `exploratory`
until the multi-seed evidence above lands.

## 8. Reproducing this audit

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge \
  python -c "from aomridge.auto_selector import AOMRidgeAutoSelector; \
             import inspect; print(inspect.getsourcefile(AOMRidgeAutoSelector))"

# Outer-test-split semantics (rows in master CSV):
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("bench/benchmark_master_results.csv", low_memory=False)
for label in ("AOMRidge-AutoSelect-headline-spxy3", "AOMRidge-Blender-headline-spxy3"):
    sub = df[df.variant == label]
    assert (sub.evaluation_split == "test").all()
    print(label, "rows", len(sub), "datasets", sub.dataset.nunique(),
          "seeds", sorted(sub.seed.dropna().unique()))
PY
```

## 9. Decision history

- 2026-05-05 14:04 CEST — D-A-001 posted with status
  `DECISION_PENDING_CODEX_REVIEW`. Verdict proposed: NESTED.
- 2026-05-05 17:00 CEST — Codex round 1 verdict: CONFIRM. Independently
  verified line numbers and code paths. Strengthened promotion gate (§7)
  to require ≥3 seeds on **both** fast12 and audit20.
- 2026-05-05 17:30 CEST — D-A-001 status flipped to `DECISION_LOCKED`.
  Registry `notes:` field for both spxy3 entries already point here.

## 10. New decisions arising from Codex round 1

### D-A-008 — selector-level `branch_preproc` regression guard

Codex flagged a blind spot: future selector variants that set
`variant.branch_preproc` (i.e., `Variant.branch_preproc` populated at the
**variant** level, not at the **candidate** level inside `extra`) trigger
the runner path at `run_aomridge_benchmark.py:740-753`, which fits the
branch preprocessor on the **full `Xtr`** before entering
`AutoSelector.fit` / `Blender.fit`. That would let preprocessing state leak
across the selector's internal folds.

Today no headline-spxy3 variant uses selector-level `branch_preproc`
(verified at `run_aomridge_benchmark.py:451`, `:463`); the variant-level
field is `None` and all branch preprocessing happens fold-local inside
`_apply_branch` (`auto_selector.py:209`, `:214`). So this was a **future
trap**, not a current bug.

**Implementation (locked 2026-05-05 18:30 CEST)**:

- New module `bench/AOM_v0/Ridge/aomridge/guards.py` —
  `SELECTOR_VARIANTS = frozenset({"auto_select", "blender", "residual_tabpfn"})`
  and `check_no_selector_branch_leak(label, selection, branch_preproc, *,
  allow_selector_level_branch_preproc=False)`.
- Runner integration: `_run_variant` calls the guard at the top of the
  function, before any preprocessor fitting (`run_aomridge_benchmark.py`
  near line 740, after `cv_obj` resolution).
- Tests: `bench/AOM_v0/Ridge/tests/test_no_selector_branch_leak.py` — 48
  parametrised tests covering selector × branch_preproc raise, selector
  with no branch_preproc passes, non-selector × branch_preproc passes,
  error message names label/selection/preproc/doc, exhaustive bidirectional
  union assertion against the runner's `selection=` literals (so a typo or
  a new selection variant fails the test loudly), opt-in escape hatch via
  `allow_selector_level_branch_preproc=True` for future Codex-approved
  dataset-level preprocessing.

**Codex review log**:

- Round 2 (17:55 CEST) — guard logic CONFIRM, test coverage REVISE
  (missing `branch_global` in non-selector matrix; missing exhaustive union
  assertion), migration path REVISE (escape hatch missing).
- Round 3 (18:10 CEST) — branch_global PASS, escape hatch PASS, exhaustive
  union FAIL (only one direction checked).
- Round 4 (18:25 CEST) — bidirectional equality PASS. **LOCK D-A-008**.

**Quality gates** (all green at lock time):
- 48 / 48 tests pass on `test_no_selector_branch_leak.py`.
- 279 / 279 tests pass on the full `bench/AOM_v0/Ridge/tests/` suite (no
  regression in `test_auto_selector.py` / `test_blender.py`).
- ruff clean on `aomridge/guards.py`, the new test, and the edited runner.
- mypy clean on the same files.

Status: `DECISION_LOCKED` (Codex round 4).

## 11. Open registry questions (non-blocking)

- **D-A-Q1** — `-spxy3` suffix in canonical name vs config template path:
  Codex did not weigh in. Default = keep current naming (matches 245 master
  rows).
- **D-A-Q2** — single-seed promotion blocker: Codex confirmed multi-seed
  gate; superseded by §7.
- **D-A-Q3** — big-n strategy: Codex verdict is in `AOMRIDGE_BIGN_OOM.md`
  (two separate entries, not auto-fallback).

## 12. Fast12 multi-seed evidence and scoped headline

LOCKED 2026-05-07 by Codex round 7. Source: `D_A_001_FAST12_PAIRED_STATS.md` /
`D_A_001_fast12_paired_stats.csv` produced from
`bench/AOM_v0/Ridge/benchmark_runs/da001_partial_fast12_seeds012/results.csv`
(414 OK / 936 total, 12 candidates fully iterated × 36 dataset×seed pairs).

### 12.1 Scoped headline (production-ready language)

> On `fast12_transfer_core` with seeds 0/1/2, `AOMRidge-Blender-headline-spxy3`
> and `AOMRidge-AutoSelect-headline-spxy3` remain nested/no-leakage selectors
> and show Holm-controlled wins versus `ASLS-AOM-compact-cv5-numpy`; Blender
> additionally clears a practical win versus `AOMRidge-Local-compact-knn50`.
> This is not a broad AOMRidge-family dominance claim: comparisons versus
> `AOMRidge-global-compact-none` and `Ridge-tuned-cv5` remain audit-pending
> because the full Holm/no-harm gate does not clear and
> `Biscuit_Sucrose_40_RandomSplit` is a known tail regression.

This claim is the only language Codex round 7 authorises for production use;
broader phrasings (e.g. "best AOMRidge variant", "production-ready vs all
baselines") are explicitly out of scope until audit20 evidence lands.

### 12.2 Wins that cleared the gate

Per-dataset (N=12, primary unit), Holm-corrected across 8 comparisons:

| Selector vs Baseline | Median Δ% | q90 ratio | Cliff's δ | p_Holm (ds) | Verdict |
|---|---|---|---|---|---|
| Blender vs ASLS-AOM-compact-cv5-numpy | -11.29 % | 1.088 | +0.667 | 0.048 | WIN_strong |
| Blender vs AOMRidge-Local-compact-knn50 | -4.63 % | 1.032 | +0.500 | 0.043 | WIN_practical |
| AutoSelect vs ASLS-AOM-compact-cv5-numpy | -17.58 % | 1.053 | +0.667 | 0.020 | WIN_strong |

Descriptive Friedman (5 AOMRidge variants, 36 rows): chi^2 = 51.8, p < 0.0001.
Mean rank (1=best): Blender 2.083, AutoSelect 2.083, Local-knn50 3.167,
global-none 3.333, MultiBranchMKL 4.333. Omnibus F-N is reserved for the
production/full-57 escalation per §7.

### 12.3 Comparisons that did NOT clear the gate

| Selector vs Baseline | Median Δ% | Failure mode |
|---|---|---|
| Blender vs Ridge-tuned-cv5 | -13.55 % | p_Holm = 0.067 (borderline) |
| Blender vs AOMRidge-global-compact-none | -14.17 % | p_Holm = 0.154 ; Biscuit tail = 1.397 |
| AutoSelect vs Ridge-tuned-cv5 | -10.44 % | p_Holm = 0.154 ; q90 = 1.149 (> 1.10 no-harm threshold) |
| AutoSelect vs AOMRidge-global-compact-none | -10.65 % | p_Holm = 0.154 ; Biscuit tail = 1.466 |
| AutoSelect vs AOMRidge-Local-compact-knn50 | -3.54 % | p_Holm = 0.154 ; median barely clears -3% |

These are described as "favourable medians but audit-pending /
not Holm-confirmed" in any external communication.

### 12.4 Known regressions

- **`Biscuit_Sucrose_40_RandomSplit`**: worst regression for both selectors
  versus `AOMRidge-global-compact-none` (Blender ratio = 1.397, AutoSelect
  ratio = 1.466). Same dataset is the worst regression for Blender vs
  Ridge-tuned-cv5 (1.107) and for both selectors vs Local-knn50 (1.036 /
  1.088). Repeated worst-case across pairings ⇒ likely fold-instability
  (random split + SPXY3 inner-CV interaction) rather than a true selector
  failure mode (per §6 caveat on single-repeat SPXY3 selector variance on
  small-n datasets), but a cheap diagnostic via per-run sidecars
  (AutoSelect chosen candidates, Blender weights, OOF fold RMSE variance)
  is required before this characterisation is final.
- **`Ccar_spxyG_block2deg`**: worst regression for AutoSelect vs
  Ridge-tuned-cv5 (ratio = 1.189). Single-pair tail, monitored on audit20.

### 12.5 Promotion boundary (unchanged)

§7 still gates `strong_practical` and `best_current` maturity behind:
- Multi-seed fast12 evidence (this section closes only the Blender vs ASLS-AOM,
  AutoSelect vs ASLS-AOM, and Blender vs Local-knn50 sub-gates).
- audit20_transfer_core × seeds 0/1/2 evidence (next escalation per Codex
  round-6 §iv ; staged but not yet launched).
- A re-run of the §10.2 Wilcoxon / Cliff's δ / q90 protocol on the audit20
  rows.

### 12.6 Audit20 update (Codex round 8)

Source: `D_A_001_AUDIT20_PAIRED_STATS.md` /
`D_A_001_audit20_paired_stats.csv` produced from
`bench/AOM_v0/Ridge/benchmark_runs/da001_audit20_seeds012/results.csv`
(540 OK / 540 total, 9 candidates × 20 datasets × 3 seeds).

Codex round 8 chooses **Option B**: audit20 confirms the need for a scoped
headline, but does not lift §12.1 into an unqualified production /
AOMRidge-family dominance claim. The audit20 cohort is larger than fast12
(20 datasets × 3 seeds = 60 rows per candidate, versus 12 × 3 = 36), and it
moves the evidence: Ridge/global caveats are no longer merely audit-pending,
while ASLS-AOM and Local-knn50 remain bounded caveats.

Per-dataset (N=20, primary unit), Holm-corrected across 8 comparisons:

| Selector vs Baseline | Median Δ% | q90 ratio | Cliff's δ | p_Holm (ds) | Verdict |
|---|---|---|---|---|---|
| Blender vs Ridge-tuned-cv5 | -8.26 % | 1.012 | +0.700 | 0.0148 | WIN_strong |
| Blender vs AOMRidge-global-compact-none | -11.40 % | 1.078 | +0.500 | 0.0479 | WIN_strong |
| AutoSelect vs Ridge-tuned-cv5 | -7.22 % | 1.012 | +0.800 | 0.0126 | WIN_strong |
| AutoSelect vs AOMRidge-global-compact-none | -9.75 % | 1.043 | +0.600 | 0.0283 | WIN_strong |
| AutoSelect vs AOMRidge-Local-compact-knn50 | -4.46 % | 1.027 | +0.400 | 0.0283 | WIN_practical |

The §12.3 "did NOT clear" table is updated by audit20 as follows:

| Selector vs Baseline | Median Δ% | Failure mode |
|---|---|---|
| Blender vs ASLS-AOM-compact-cv5-numpy | -4.71 % | p_Holm = 0.1238 ; Quartz_spxy70 worst ratio = 16851.112 |
| Blender vs AOMRidge-Local-compact-knn50 | -5.38 % | p_Holm = 0.1238 ; q90 = 1.106 (> 1.10 no-harm threshold) ; Quartz_spxy70 worst ratio = 254.700 |
| AutoSelect vs ASLS-AOM-compact-cv5-numpy | -5.52 % | p_Holm = 0.1238 ; q90 = 1.104 (> 1.10 no-harm threshold) |

The audit20 known-regression / caution table is:

| Dataset | Selector | Audit20 finding | Action |
|---|---|---|---|
| `Quartz_spxy70` | Blender | Catastrophic audit20 worst case: ratio = 127.254 vs `Ridge-tuned-cv5`, 16851.112 vs `ASLS-AOM-compact-cv5-numpy`, 37059.028 vs `AOMRidge-global-compact-none`, and 254.700 vs `AOMRidge-Local-compact-knn50`. | Promote to explicit Blender caution. AutoSelect does not share this failure mode; its audit20 worst ratio is 1.189 on `Ccar_spxyG_block2deg`. |
| `Biscuit_Sucrose_40_RandomSplit` | AutoSelect | Known AutoSelect tail remains: worst ratio = 1.437 vs `ASLS-AOM-compact-cv5-numpy`, 1.466 vs `AOMRidge-global-compact-none`, and 1.088 vs `AOMRidge-Local-compact-knn50`. | Preserve the §12.4 known-regression footnote and continue the sidecar diagnostic path. |

Additional audit20 flag: `AOMRidge-global-compact-snv` is bit-identical to
`AOMRidge-global-compact-none` in this protocol; keep the M0-M1 unification
follow-up open.

Promotion boundary: §12.1 stays scoped. The audit20 evidence closes the
audit20 re-run requirement in §12.5, but does not move Blender or AutoSelect
to an unqualified production, `strong_practical`, or `best_current` claim.
Full-57/sign-test evidence and a resolution of the remaining ASLS-AOM /
Local-knn50 caveats are still outside this §12 update.

Registry maturity: Blender stays at `exploratory`. AutoSelect's avoidance of
the Quartz failure mode is evidence for AutoSelect's meta-design, not for
Blender being safer. Blender card should gain a separate note warning about
the Quartz_spxy70 failure mode.
