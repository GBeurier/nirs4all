# DESIGN_MBPLS — AOM-MBPLS algorithms (Phase 2)

**Status**: Codex review round 1 disposed (see §7); ready for implementation.
**Phase**: 2 (after `DESIGN_VIEWS.md` Phase 1).
**Out of scope**: MoE routing (DESIGN_MOE.md), classification (Phase 6).

**Naming change post-Codex (HIGH 1)**: my original "true V1/V2" was *not*
faithful Westerhuis MB-PLS — it deflates only the winning block. I now split
it into two distinct algorithms with different semantics:

- **block-sparse AOM-MBPLS**: hard-gated, only the winning block participates
  at each LV. Block-supported components, AOM-style locality.
- **classic MB-PLS-AOM**: Westerhuis-faithful super-score from all blocks +
  per-block deflation against block-side loadings. AOM principle is in the
  per-LV operator selection within blocks, not in the block routing.

Both are implemented; they are not redundant.

---

## 1. Goal

Define **four** AOM-MBPLS variants, ordered by implementation complexity, and
benchmark them against the Phase-1 baselines (`PLS-standard`, `AOM-PLS-compact`,
`MBPLS-blocks3-vanilla`):

| # | Name                                   | Bank composition                       | Selection                       | Deflation                                  | New code? |
|---|----------------------------------------|----------------------------------------|---------------------------------|--------------------------------------------|-----------|
| 1 | **lazy V1 (per-block)**                | `[I, M_1, ..., M_K]`                   | POP (existing)                  | Global (X residual)                        | None — reuses POP |
| 2 | **lazy V2 (per-block × per-op)**       | `[I, M_k, op_i, M_k . op_i]`           | POP (existing)                  | Global                                     | None |
| 3 | **block-sparse V1 (hard-gated AOM)**   | `[M_1, ..., M_K]`                      | Per-LV block winner (PRESS)     | Winning block only (`t_a · p_{k*,a}^T`)    | New `select_block_sparse_aom` |
| 4 | **block-sparse V2 (with per-block bank)** | K blocks, each with own AOM bank    | Per-LV (block, op) winner       | Winning block only                         | New `select_block_sparse_aom` w/ banks |
| 5 | **classic MB-PLS-AOM V1**              | `[M_1, ..., M_K]`                      | Super-score from all blocks     | All blocks deflate by `t_a · p_{k,a}^T`    | New `select_classic_mbpls_aom` |
| 6 | **classic MB-PLS-AOM V2**              | K blocks, each with own AOM bank       | Super-score, per-LV op per blk  | All blocks deflate                         | New `select_classic_mbpls_aom` w/ banks |

Lazy variants ship first (zero new algorithm code, just bank assembly).
True variants require a new selection policy in `bench/AOM_v0/multiview/multiview/selection_mbpls.py`.

---

## 2. Lazy variants — already implementable on Phase-1 infra

### 2.1 Lazy V1 (per-block)

```python
from aompls.estimators import POPPLSRegressor
from multiview.views import ViewBuilder

bank = ViewBuilder.blocks_only(K=3, strategy="equal_width").build(p=Xtr.shape[1])
est = POPPLSRegressor(
    operator_bank=bank,            # explicit bank, bypasses bank_by_name
    selection="per_component",
    criterion="holdout",           # or "cv" with cv_splitter=SPXYFold(...)
    engine="simpls_covariance",
    max_components=15,
)
est.fit(Xtr, ytr)
```

Mechanics:

- The bank is `[Identity, BlockMask_0, BlockMask_1, BlockMask_2]`.
- POP's `select_per_component` (`bench/AOM_v0/aompls/selection.py:457`)
  greedily picks the bank entry that best-improves the criterion at every LV.
- Effective behaviour: at each LV, the model picks **one block (or the
  identity = "all blocks at once")** to drive the next direction.
- Deflation is global on `X` (POP's standard behaviour) — no per-block
  residual maintenance. This is "lazy" because true MB-PLS uses per-block
  deflation (§3 below).

### 2.2 Lazy V2 (per-block × per-op)

```python
bank = ViewBuilder.combined(
    bank_name="compact", K=3, strategy="equal_width", include_global=True,
).build(p=Xtr.shape[1])
est = POPPLSRegressor(operator_bank=bank, selection="per_component", ...)
```

Bank size: 36 ops (compact + K=3, see DESIGN_VIEWS §6.1). Mechanics identical
to lazy V1 except the candidate space includes preproc × block compositions.

### 2.3 What lazy variants test

The hypothesis behind lazy variants: **per-LV view selection is the
fundamental AOM principle, and adapting the bank from "all preproc on the
full spectrum" to "preproc × block restrictions" should expose **localised
chemometric signals** the standard AOM bank misses.

If lazy V1/V2 already match or beat AOM-PLS-compact on smoke-4, we have
strong evidence that block-aware views are useful, and we can decide whether
the more expensive true MB-PLS adds further benefit.

---

## 3. Block-sparse AOM-MBPLS (V3 / V4) — hard-gated

### 3.1 Notation

- `X ∈ R^{n × p}` centered training spectra.
- `y ∈ R^n` (regression) or `Y ∈ R^{n × q}` (multi-target). Phase 2 ships
  `q = 1` only; multi-target is a stub.
- `K` blocks defined by `[a_k, b_k)`, with `M_k = diag(1_{a_k ≤ i < b_k})`.
- `X_k = X · M_k^T` is block `k` — same shape as `X` but features outside
  the block are zero.
- Block-restricted operator banks: `B_k = {A_{k,1}, A_{k,2}, …}`. For V1, every
  bank is `{Identity}` (block masking only). For V2, `B_k` is a copy of the
  user-specified preproc bank (default: `compact` = 9 ops including identity)
  composed with `M_k`.
- `T ∈ R^{n × K_max}` super-score matrix (n_components_selected = K_max).
- `t_a` is the super-score at LV `a` (one column of `T`).

### 3.2 Algorithm at a glance

```
for a = 1 to n_components_max:
    # Per-block: candidate weight, score, deflated block residual.
    for k in 1..K:
        for b in B_k:
            # Compute block weight w_{k,b} = (b · X_k)^T · y_res / ||...||
            # Compute candidate block-score t_{k,b} = X_k · A_{k,b}^T · w_{k,b}
            # Score by PRESS on a held-out fold OR holdout split:
            score_{k,b} = press(t_{k,b}, y_res)
    (k*, b*) = argmin score_{k,b}
    # Commit winner.
    op_indices[a] = (k*, b*)
    block_weights_per_lv[a] = (k*, w_{k*, b*})
    super_score[a] = t_{k*, b*}
    # Per-block deflation: ONLY the winning block deflates by t_a.
    p_{k*, a} = X_{k*}^T · t_a / (t_a^T · t_a)
    X_{k*} = X_{k*} - t_a · p_{k*, a}^T
    # Y deflation (global).
    q_a = y_res^T · t_a / (t_a^T · t_a)
    y_res = y_res - t_a · q_a
```

Final coefficient (regression in original space):

```
W ∈ R^{p × K_max}    where W[:, a] = (operator-induced direction in original space)
P ∈ R^{p × K_max}    block loadings (zero outside winning block features)
Q ∈ R^{q × K_max}    Y loadings
B = W · pinv(P^T · W) · Q^T     # standard PLS coef formula
```

The crucial design choice is **per-block deflation**: only the winning block
loses its rank-1 update at LV `a`. Other blocks keep their full residual.
This contrasts with POP (lazy V1) which deflates the **full** `X`.

### 3.3 Why per-block deflation matters

Consider K = 2 blocks with disjoint chemistry signal. POP's global deflation
removes information from BOTH blocks at every LV, even when only one block
contributed. Subsequent LVs see a doubly-residualised spectrum and may find
spurious signal.

Per-block deflation preserves the unused blocks intact, so subsequent LVs
can re-pick block 1 if it still has structure, or pick block 2 fresh. This is
the standard MB-PLS contract (`bench/AOM_v0/aompls/simpls.py:413` /
`nirs4all/operators/models/sklearn/mbpls.py:124` — both use per-block
deflation in their respective conventions).

### 3.4 Coefficient computation (Codex HIGH 2 disposition)

**Original draft was wrong**: `r_a = M_{k*} A_{k*,b*}^T w_{k*,b*}` only holds
at LV 1. After deflation, later block scores live in residual space, so
direct reconstruction from `(X - x_mean)` is no longer valid.

**Correct approach — match standard PLS coefficient formula**:

For each LV `a`, store:

- **Raw weight** `w_{a}^{raw} ∈ R^p` — the `r_a = M_{k*} A_{k*,b*}^T w_{k*,b*}`
  vector computed against the *current residual block* `X_{k*}^{(a-1)}`. This
  is the "raw" weight in PLS terminology.
- **Block loading** `p_a ∈ R^p` — derived as `X_{k*}^{(a-1)T} · t_a / (t_a^T t_a)`.
  Block-supported (zero outside block `k*`).
- **Y loading** `q_a ∈ R^q`.
- **Super-score** `t_a ∈ R^n`.

After all `K_max` LVs are committed, assemble matrices `W^{raw} ∈ R^{p × K_max}`,
`P ∈ R^{p × K_max}`, `Q ∈ R^{q × K_max}` and compute the standard PLS
coefficient

```
B = W^{raw} · pinv(P^T · W^{raw}) · Q^T          ∈ R^{p × q}
```

with the same regularised pinv used in `bench/AOM_v0/aompls/simpls.py`
(eps `1e-10` on the diagonal of `P^T W^{raw}`).

This is the same formula PLS / SIMPLS / MB-PLS use; Codex was right that
"per-LV reconstruction from original X" was off by the deflation history.

Block locality is preserved because every column of `W^{raw}` and `P` is
zero outside its winning block.

### 3.5 PRESS scoring

Three criteria, mirroring AOM-PLS:

- **`holdout`**: single split, fit on train block residuals, score on val.
- **`cv`**: K-fold (with optional `cv_splitter`, e.g. SPXYFold). At each fold,
  recompute the entire greedy sequence on the train block residuals; score
  on val. This is expensive: `n_folds × n_components_max × K × |B_k|` engine
  fits per dataset. With `K = 3`, `B_k = compact (9 ops, V2)`, `n_folds = 5`,
  `n_components_max = 15` → 2025 fits per dataset. Acceptable on CPU.
- **`approx_press`**: closed-form leverage-based PRESS approximation; cheapest
  but biased on small datasets.

For Phase 2, **default = holdout**, **escalation = SPXY-CV** (only the best
holdout variant gets re-run with SPXY-CV).

### 3.6 Auto-prefix early stop (one-SE rule)

After committing all `n_components_max` LVs, score every prefix `1..K_max`
on the criterion, pick the smallest k whose score is within `best + SE` of
the optimum (Pragmatic 1-SE). Same logic as `select_per_component`
(`bench/AOM_v0/aompls/selection.py:537-560`), just adapted to the
per-block residuals.

### 3.7 Edge cases

| Case | Behaviour |
|------|-----------|
| `K = 1` | Reduces to standard AOM-PLS on the full spectrum (no diversity). |
| Block fully deflated (rank exhausted) | Drop block from candidate set for remaining LVs. |
| All blocks rank-exhausted before reaching `n_components_max` | Stop early; return current `n_components_selected`. |
| Same block wins all LVs | Equivalent to AOM-PLS on block `M_{k*} · X` — sane fallback. |

### 3.8 Soft-gate variant (alpha-blend)

Optional Phase-2 extension if the hard winner gate plateaus. At each LV,
compute scores `s_k` per block (or per (k, b) pair), softmax them with
temperature `τ`, and form a **mixture super-score**:

```
w_k = exp(-s_k / τ) / Σ exp(-s_j / τ)
t_a = Σ_k w_k · t_{k, b_k*}        (b_k* = best b within block k)
```

Per-block deflation uses `w_k · t_a` for block `k`. This is conceptually
similar to existing `select_soft` but multi-block. Default `τ = 1.0`.

---

## 3a. Classic MB-PLS-AOM (V5 / V6) — Westerhuis-faithful

To preserve a true MB-PLS reference (Codex HIGH 1 fallback path), we also
implement the classic algorithm with AOM-style operator selection within
each block.

```
for a = 1 to n_components_max:
    for k in 1..K:
        for b in B_k:                         # AOM operator pool for block k
            # candidate block weight & score within block k
            w_{k,b} = (A_{k,b} X_k^{(a-1)})^T y_res / ||...||
            t_{k,b} = X_k^{(a-1)} A_{k,b}^T w_{k,b}
            score_{k,b} = press(t_{k,b}, y_res)
        # AOM choice within block k
        b_k* = argmin_b score_{k,b}
        t_k = t_{k, b_k*}
    # Westerhuis super-score: aggregation of all block scores.
    t_a = aggregate(t_1, ..., t_K)             # mean (default) or weighted
    # Per-block deflation: ALL participating blocks deflate against t_a.
    for k in 1..K:
        p_{k,a} = X_k^{(a-1)T} · t_a / (t_a^T t_a)
        X_k^{(a)} = X_k^{(a-1)} - t_a · p_{k,a}^T
    q_a = y_res^T · t_a / (t_a^T t_a)
    y_res = y_res - t_a · q_a
```

Differences vs §3 block-sparse:

- **All blocks contribute** their best operator at every LV.
- **All blocks deflate**, mirroring `nirs4all/operators/models/sklearn/mbpls.py:_mbpls_fit_multiblock_numpy`.
- **Aggregation** is `mean` by default (Westerhuis); a `weighted` option using
  the per-block PRESS-derived weights is exposed.

Coefficient assembly: same standard formula `B = W^{raw} · pinv(P^T W^{raw}) · Q^T`,
where `W^{raw}` and `P` now have non-zero entries on every block per column
(loadings from each participating block at each LV).

## 4. Implementation plan

```
bench/AOM_v0/multiview/multiview/selection_mbpls.py
    select_mbpls_aom_global(...)    # for per-block hard winner (V1/V2 true)
    _press_block_candidate(...)     # per-block PRESS evaluation
    _per_block_deflate(...)         # in-place residual update
    _soft_alpha_mixture(...)        # Phase 2.5 if needed

bench/AOM_v0/multiview/multiview/estimators_mbpls.py
    AOMMBPLSRegressor               # sklearn-compat wrapper
        - n_components, max_components, K, strategy, bank_name (=identity for V1)
        - cv, cv_splitter, criterion, engine
        - selection_mode = "lazy" | "true_v1" | "true_v2"

bench/AOM_v0/multiview/tests/test_mbpls.py
    - V1 lazy: matches POP with blocks_only bank
    - V1 true: per-block deflation, super-score correctness
    - V1 true vs V1 lazy: differ when blocks have orthogonal signal (numerical demo)
    - K=1 reduction: matches AOM-PLS-compact

bench/AOM_v0/multiview/benchmarks/run_smoke4_v2.py
    - Append Phase 2 variants to the smoke-4 CSV.
    - Variants: lazy_v1, lazy_v2, true_v1, true_v2 (all on holdout).
```

Step ordering:

1. **Implement lazy V1/V2** (zero new code, just bank assembly into existing
   POP estimator). Run smoke-4. ~30 min.
2. **Codex review** of this DESIGN_MBPLS.md, sections 3.x. Block on Codex
   approval before implementing true variants.
3. **Implement true V1**: `select_mbpls_aom` policy + AOMMBPLSRegressor wrapper.
4. **Implement true V2**: extends V1 with per-block operator banks.
5. **Run smoke-4** for all four; compare to Phase 1 baselines.
6. **Escalate to smoke-10** for variants meeting the criterion (≥ 2/4 wins
   AND median rel-RMSEP ≤ 1.00 vs PLS-standard).
7. Commit Phase 2.

---

### 4.1 Leakage-free CV path (Codex HIGH 8 disposition)

The new `select_*` policies must reuse the same per-fold pattern as
`bench/AOM_v0/aompls/selection.py:_criterion_score_at_indices` (lines
612-627): each fold re-centers train data, refits operators, recomputes
block residuals, and predicts on val using fold-specific means/loadings.
This is the existing leakage-free contract — the new code inherits it by
calling `cv_score_regression(...)` with a `_fp` closure that captures the
fold-fitted state.

Concretely, the candidate-scoring inner loop becomes:

```python
def _candidate_score(operators, indices, ...):
    def _fp(X_tr, y_tr, X_va):
        x_mean = X_tr.mean(axis=0)
        y_mean = y_tr.mean()
        Xtc = X_tr - x_mean
        ytc = y_tr - y_mean
        # Fit operators on training only, run the (block-sparse or classic)
        # selection algorithm on (Xtc, ytc) with `indices`, derive coef_, then
        # predict on (X_va - x_mean).
        ...
        return pred + y_mean
    return cv_score_regression(Xc, yc, _fp, criterion.cv,
                                criterion.random_state,
                                cv_splitter=criterion.cv_splitter)
```

Block scaling (Codex MEDIUM 7) is added as an optional knob: per-block
column standardisation `X_k → (X_k - μ_k) / σ_k`. Default off (mirrors
AOM-PLS-compact). When enabled, the inverse scaling must be applied to
the final coefficient `B` so predictions live in the original feature
space; this matches `_mbpls_fit_multiblock_numpy:159-193`.

### 4.2 Auto-prefix (Codex MEDIUM 4 disposition)

Renamed from "1-SE rule" to "pragmatic curve shrinkage" to reflect that
the noise proxy is the prefix-score curve std, not a CV-derived SE. When
`criterion = "cv"`, the per-fold scores at each prefix can compute a
proper SE from fold-level errors; we use that when available, else fall
back to the curve-std proxy. Same fallback as
`bench/AOM_v0/aompls/selection.py:546-559`.

### 4.3 P^T W conditioning (Codex HIGH 8b disposition)

Block-supported `W^{raw}` and `P` make `P^T W^{raw}` close to block-diagonal,
so it is easier to invert than the dense AOM-PLS case. The pinv with
`eps · I` regularisation handles rank-deficient cases robustly. We log
`np.linalg.cond(P^T W^{raw})` per fit when `verbose >= 1` and warn if
condition number exceeds `1e8`.

## 7. Codex review — disposition (round 1, 2026-05-01)

| # | Codex severity | Issue | Disposition |
|---|---------------|-------|-------------|
| 1 | HIGH | §3.2 hard-winner is not classic Westerhuis MB-PLS | **Renamed to "block-sparse AOM-MBPLS" (V3/V4); added separate "classic MB-PLS-AOM" (V5/V6) per §3a.** |
| 2 | HIGH | §3.4 reconstruction wrong for LV > 1 | **Replaced with standard PLS coef formula `B = W^{raw} · pinv(P^T W^{raw}) · Q^T` per §3.4.** |
| 3 | MED  | §3.5 CV cost claim ungrounded | **Phase 2 ships holdout only. SPXY-CV documented as "profile-then-escalate" in §4.1.** |
| 4 | MED  | §3.6 1-SE rule is heuristic | **Renamed to "pragmatic curve shrinkage" in §4.2; uses fold-SE when CV criterion, fallback to curve-std otherwise.** |
| 5 | OK   | §3.7 rank exhaustion handling | No change. |
| 6 | MED  | §3.8 soft mixture loses block locality | **Documented in §3.8 explicitly: soft mixture creates union-of-blocks components, deferred to Phase 2.5.** |
| 7 | MED  | block centering vs scaling | **Added optional per-block standardization in §4.1 with inverse-scale recovery for coefficient.** |
| 8 | HIGH | leakage-free CV scoring | **§4.1 explicitly reuses `cv_score_regression` + `_fp` closure pattern from existing AOM-PLS POP path.** |
| 8b | HIGH | `P^T W` conditioning | **§4.3 logs condition number with warn at 1e8; pinv eps regularisation kept.** |

## 8. Codex review focus (round 2 — implementation phase)

1. **§3.2 algorithm** — is the per-block deflation correctly capturing the
   MB-PLS contract? Specifically: should we deflate the winning block by the
   block-side loading `p_{k*, a}` instead of by the super-score? In classic
   MB-PLS (Westerhuis 1998), deflation uses block scores `t_{k, a}` per block
   and super-score is the *aggregation*. My algorithm picks ONE block winner
   and deflates only that block by `t_a`. Is this a valid AOM-style adaptation
   or a step away from MB-PLS semantics?

2. **§3.4 super-score reconstruction** — verify `r_a = M_{k*} A_{k*,b*}^T
   w_{k*,b*}` correctly produces `t_a = (X - x_mean) · r_a`. The mask `M_{k*}`
   is required because operator `A_{k*,b*}` was applied to `X_{k*} = X · M_{k*}`,
   not to `X` directly. So `r_a` must include the mask to recover the original
   block restriction at predict time.

3. **§3.5 CV cost** — 2025 engine fits per dataset under V2/CV/SPXYFold/
   K_max=15 is the back-of-envelope; on smoke-4 this scales to ~8000 fits.
   On a 5-min budget per dataset, is `simpls_covariance` fast enough? Or
   should we ship V2 with `holdout` only and document SPXY-CV as Phase-3 work?

4. **§3.6 auto-prefix** — the 1-SE rule applied to the per-block-deflation
   case: does the SE proxy (std of curve / sqrt(n)) make sense when the
   curve was generated by a non-stationary deflation process (block residuals
   change LV-by-LV)? Or do we need a different shrinkage rule?

5. **§3.7 edge cases** — when a block is rank-exhausted, my plan is to drop
   it from the candidate set. But the Y residual is still updated by the
   global super-score, and the block's information might effectively
   "re-enter" via correlations with remaining blocks. Is this the right
   behaviour, or should we hard-stop the model when any block exhausts rank?

6. **§3.8 soft-gate alpha-blend** — the soft mixture super-score `t_a = Σ w_k
   t_{k, b_k*}`: the column lives in R^n, but the corresponding direction in
   R^p is `r_a = Σ_k w_k M_k A_{k,b_k*}^T w_{k,b_k*}` (no longer block-supported).
   Is this a faithful "soft AOM-MBPLS" or does it violate block locality?

7. **§3 missed issues** — am I neglecting block centering (each block has
   its own mean)? Standard MB-PLS centers AND standardises each block
   separately. AOM-PLS-compact only centers globally. Phase 2 default:
   global centering only (mirrors AOM-PLS); Phase 2.5: per-block centering
   as ablation. Is global-only acceptable for the per-block-deflation
   algorithm, or does the math break?

---

## 6. Decision table for Phase 2 → Phase 3

| Outcome on smoke-4 | Action |
|---------------------|--------|
| Lazy V1 wins ≥ 2/4 vs AOM-PLS-compact | Escalate lazy V1 to smoke-10. Skip true variants if lazy is already strong. |
| Lazy V1 fails, true V1 wins ≥ 2/4 | Per-block deflation is the differentiator. Escalate true V1. |
| All lazy + true variants tie within 5% RMSEP | "AOM-MBPLS plateau" — document negative result, move to MoE. |
| One specific dataset has clear MBPLS advantage (e.g. block2deg-style) | Investigate the dataset's spectral structure for the win mechanism. |
| One specific bank/criterion combination dominates | Champion variant for Phase 5 full-57 run. |
