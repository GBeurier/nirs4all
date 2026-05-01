# DESIGN_VIEWS — View abstraction for AOM-multiview

**Status**: design draft, awaiting Codex review of math + API
**Phase**: 1 (Foundation)
**Out of scope**: MB-PLS algorithm (DESIGN_MBPLS.md), MoE routing (DESIGN_MOE.md).

---

## 1. Goal

Define what a "view" is for AOM-multiview, expand the existing
`LinearSpectralOperator` abstraction to cover wavelength-block masking, and
provide a `ViewBuilder` that assembles operator banks for the three view-modes
the project will benchmark: preprocessing-as-views, wavelength-block-as-views,
and combined (preproc × block).

The deliverable is purely the **view abstraction**. The MB-PLS / MoE
selection algorithms that consume these banks are designed in subsequent docs.

---

## 2. View := LinearSpectralOperator

Every view of the spectrum is a strict linear operator
`A ∈ R^{p × p}` acting on row spectra: `X_b = X A^T`.

Why reuse `LinearSpectralOperator`:

- `bench/AOM_v0/aompls/operators.py:LinearSpectralOperator` already provides
  fast covariance scoring via `apply_cov(S) = A S` and adjoint `A^T v`, which
  lets `simpls_covariance` / `nipals_adjoint` evaluate view candidates without
  materializing transformed spectra.
- Composition is closed under the existing `ComposedOperator` (which by the
  docstring at `operators.py:513-516` applies operators leftmost-first). So
  Cartesian products of preproc × block stay in the same protocol.
- All existing AOM-PLS selection policies (`global`, `per_component`,
  `superblock`, `active_superblock`) accept any `LinearSpectralOperator` as a
  bank entry, so the multi-view extension reduces to: build the right banks.

---

## 3. New operator: BlockMaskOperator

### 3.1 Definition

For a wavelength block `[a, b) ⊂ {0, …, p-1}` (half-open, `numpy`-slicing
convention):

```
M = diag(m)         m_i = 1 if a ≤ i < b else 0
```

Properties:

- Symmetric: `M^T = M`.
- Idempotent: `M^2 = M`.
- Rank: `b - a`.
- Strict-linear, no fitted parameters.

### 3.2 Protocol implementation

| Method            | Implementation                                                |
|-------------------|---------------------------------------------------------------|
| `transform(X)`    | `X · M^T = X · M` — equivalent to zeroing columns outside `[a, b)` |
| `apply_cov(S)`    | `M · S` — zeroes rows of `S` outside `[a, b)`                 |
| `adjoint_vec(v)`  | `M^T · v = M · v` (symmetric)                                 |
| `matrix(p)`       | `diag(m)` — explicit `p × p`                                  |

For an `(n × p)` input the transform output keeps shape `(n × p)` (full feature
dimensionality is preserved; out-of-block columns are zero). This matches the
existing operator API and makes block masks composable with full-spectrum
preprocessing.

### 3.3 Math sanity checks

**Cross-covariance identity** (required by all fast engines):

```
(X · M^T)^T · Y = M · X^T · Y    ✓   (since M = M^T)
```

This is the identity exploited by `apply_cov` to score view candidates without
materializing `X · M^T`.

**Composition with a derivative operator `D`** (e.g. SG-d1):

`M · D ≠ D · M` in general. The convention adopted here is *mask **after**
preprocessing*: `M · D`. Justification in §5.

The composed operator `M · D` remains strict-linear, so the engine fast paths
(`simpls_covariance`, `nipals_adjoint`) keep working.

**Cross-block spectral dependence** (Codex MEDIUM): when `D` is a finite-window
operator (SG kernel, finite difference), the composed `M · D` produces in-block
values that depend on raw wavelengths **just outside the block edge** (within
the kernel's half-width). This is *not* CV leakage — folds are still computed
on the full centered spectrum and applied uniformly to all candidates — but it
means the block-masked view is "the restriction of the *globally* preprocessed
spectrum", not "the preprocessing of the restricted spectrum". This is the
intended semantics (§5), but is surfaced here so reviewers do not mistake it
for an isolation property.

### 3.4 Edge cases (Codex HIGH — enforced at construction & in `ViewBuilder`)

`BlockMaskOperator`:

| Case            | Behaviour |
|-----------------|-----------|
| `a == b` (empty)| Construction raises `ValueError`. |
| `[0, p)` (full) | Construction raises `ValueError` (use `IdentityOperator`). |
| `a < 0` or `b > p` | Construction raises `ValueError`. |
| `a >= b`        | Construction raises `ValueError`. |

`ViewBuilder.blocks_only` / `combined`:

| Case             | Behaviour |
|------------------|-----------|
| `K < 2`          | `ValueError` — needs at least 2 blocks to have any view diversity. |
| `K > p`          | `ValueError` — would produce empty blocks. |
| `K == p`         | `ValueError` — every block is a single feature, degenerate. |
| `p < 2 * K`      | `ValueError` — blocks too narrow to be meaningful (heuristic guardrail). |

---

## 4. WL-block strategies

Phase 1 ships **`equal_width`** only. The other strategies are stubbed with
`NotImplementedError` so the API is fixed; they are evaluated in Phase 2 if
Phase 1 shows signal.

### 4.1 `equal_width` (Phase 1 default)

`K` contiguous blocks of approximately equal feature count:

```
block_k = [⌊k·p/K⌋, ⌊(k+1)·p/K⌋)   for k = 0, …, K-1
```

Default `K = 3`, reflecting the typical NIRS scenario where the spectrum is
reconstructed from 3 stitched detector segments (UV-VIS / SWIR / NIR or similar).
`K` is exposed as a parameter; `K = 4` and `K = 5` will be probed in Phase 2.

### 4.2 `quantile_width` (Phase 2 stub)

Split points `t_1 < … < t_{K-1}` chosen so the cumulative spectral energy
`E[i] = sum_{j ≤ i} mean_n(X[n, j]^2)` is partitioned into `K` equal-energy
blocks. Robust to instruments where signal is concentrated in narrow regions.

### 4.3 `chemistry_NIR` (Phase 2 stub)

Hard-coded NIR overtone regions, used only if wavelength metadata is exposed by
the dataset configuration:

- `[700, 1100) nm` — 3rd overtones
- `[1100, 1800) nm` — 2nd overtones / combinations
- `[1800, 2500) nm` — 1st overtones / fundamentals (where present)

Skipped if no wavelength axis is available.

---

## 5. Composition order: BlockMask AFTER preprocessing

**Convention**: a "block × preproc" view is `M · A_preproc`, not
`A_preproc · M`.

Justification:

- `A_preproc · M`: applies preprocessing to a vector that is zero outside the
  block. SG smoothing/derivatives zero-pad the boundary, so derivatives at the
  block edge are computed against fictional zeros. SNV/MSC (when added) would
  compute mean/variance over the zero-padded vector. **Wrong.**
- `M · A_preproc`: preprocess on the full spectrum first (boundaries treated
  exactly as in standard AOM-PLS), then mask to the block. The block-masked
  output is a legitimate restriction of the preprocessed signal. **Correct.**

In code, this is `ComposedOperator([preproc, blockmask])` — per the
operators-leftmost-first convention at `operators.py:513-516`, this corresponds
to the matrix `M · A_preproc`.

Rederivation:

```
transform(X) = blockmask.transform(preproc.transform(X))
             = blockmask.transform(X · A_preproc^T)
             = (X · A_preproc^T) · M^T
             = X · (M · A_preproc)^T               (since M = M^T)
```

So the effective operator matrix is `M · A_preproc`. ✓

---

## 6. ViewBuilder API

```python
from .views import ViewBuilder

# Mode 1 — preprocessing views (parity with existing AOM banks)
builder = ViewBuilder.preproc_only(bank_name="compact")
bank = builder.build(p=200)
# → [identity, sg_smooth, sg_d1, snv, ..., compact_bank_ops]

# Mode 2 — WL-block views (block masks only, identity retained)
builder = ViewBuilder.blocks_only(K=3, strategy="equal_width")
bank = builder.build(p=200)
# → [identity, mask_0, mask_1, mask_2]

# Mode 3 — combined (Cartesian product preproc × block + globals + identity)
builder = ViewBuilder.combined(
    bank_name="compact",
    K=3,
    strategy="equal_width",
    include_global=True,
)
bank = builder.build(p=200)
# Notation: `mask_k ∘ op_i` reads "mask after op" — preprocessing is applied
# first to the full spectrum, then the result is masked to block k. This
# matches the matrix product M_k · A_op (see §5).
# → [identity,
#    mask_0, mask_1, mask_2,                                    # block-only
#    op_1, op_2, ..., op_8,                                     # global preproc (compact = 8 non-identity ops)
#    mask_0 ∘ op_1, mask_0 ∘ op_2, ..., mask_0 ∘ op_8,          # block 0 × preproc
#    mask_1 ∘ op_1, ...,                                        # block 1 × preproc
#    mask_2 ∘ op_1, ...]                                        # block 2 × preproc
```

`include_global=True` makes `combined` a strict superset of `preproc_only`,
which keeps ablation comparisons clean.

**Strict-linearity enforcement** (Codex HIGH): the `ViewBuilder` must reject
any operator that is not `is_strict_linear`. Existing `bank_by_name("compact")`
returns 9 strict-linear operators (identity + SG variants + detrend + finite
difference); SNV/MSC are deliberately absent because they are non-linear.
`ViewBuilder.build()` checks `op.is_linear_at_apply()` for every entry and
raises if a non-strict-linear operator slips in (e.g. via a custom user bank).

### 6.1 Bank-size budget (corrected against `bank_by_name`)

Actual sizes from `bench/AOM_v0/aompls/banks.py:bank_by_name(name, p=200)`
(verified 2026-05-01):

| Mode + bank                              | K | size                            |
|------------------------------------------|---|---------------------------------|
| preproc_only(compact)                    | – | 9 (identity + 8 ops)            |
| preproc_only(family_pruned)              | – | 15                              |
| preproc_only(response_dedup)             | – | 47                              |
| blocks_only                              | 3 | 1 + 3 = 4                       |
| combined(compact, +global)               | 3 | 1 + 3 + 8 + 24 = 36             |
| combined(family_pruned, +global)         | 3 | 1 + 3 + 14 + 42 = 60            |
| combined(response_dedup, +global)        | 3 | 1 + 3 + 46 + 138 = 188          |

(Identity is counted once in column "size" — it is added by the existing
`_resolve_bank` helper if missing, and we deduplicate `mask_k ∘ identity ≡
mask_k` and `identity ∘ op ≡ op` at builder time.)

**Complexity (corrected, Codex HIGH)**: POP greedy selection at component `k`
evaluates `bank_size` candidates against the partial sequence
`op_indices[:k-1] + [b]`. Each candidate evaluation is one engine fit at
depth `k` plus its CV cost. The total per-seed cost for POP+CV is

```
work ≈ Σ_{k=1..K_max} bank_size · n_folds · cost(engine_fit at depth k)
     ≈ bank_size · n_folds · K_max(K_max+1)/2 · cost_per_fit_at_depth_1
```

`simpls_covariance` and `nipals_adjoint` use `op.apply_cov(S)` and never
materialize `X · A^T`. The `apply_cov` cost is `O(p^2)` for dense operators
(detrend), `O(p · w)` for banded ones (SG with window `w`), and `O(p)` for
diagonal ones (BlockMask). For `combined(family_pruned, K=3) = 60` ops and
`K_max = 25`, `n_folds = 5`, the work is `~9 750` evaluations per seed —
acceptable on CPU. Phase 1 only runs `compact` (36 ops) to keep latency low.

**Active-bank prescreen alternative**: if the brute-force POP cost becomes a
bottleneck on `response_dedup × K=3` (188 ops), the existing
`select_active_superblock` covariance-screen + diversity-prune
(`bench/AOM_v0/aompls/selection.py:880`) can be ported to the multi-view bank
to drop redundant entries before POP. Deferred to Phase 2.

---

## 7. Smoke-4 cohort

Subset of `bench/AOM_v0/benchmarks/run_smoke_cohort.py:SMOKE_DATASETS`, chosen
to span small/medium/large `n` and chemistry/biology, validated against
smoke-10 representativity.

| Dataset                                       | Domain     | Why include              |
|-----------------------------------------------|------------|--------------------------|
| `Beer_OriginalExtract_60_YbaseSplit`          | chemistry  | small-`n` stress test    |
| `All_manure_MgO_SPXY_strat_Manure_type`       | agronomic  | medium-`n`, SPXY split   |
| `Chla+b_spxyG_block2deg`                      | biology    | wide-`p` biological resp |
| `grapevine_chloride_556_KS`                   | agronomic  | large-`n`, KS split      |

Smoke-4 is for Phase 1 iteration speed only. Smoke-10 remains the actual
escalation gate to full-57.

---

## 8. Phase 1 deliverables

1. **Foundational fix (Codex HIGH)** — thread `cv_splitter` through
   `bench/AOM_v0/aompls/selection.py:_criterion_score_at_indices` (currently
   line 627 calls `cv_score_regression` with no splitter, so POP candidate
   scoring and global fixed-`n` scoring silently fall back to random KFold
   even when the user passes SPXYFold). Required because Phase 2 AOM-MBPLS
   V1 (per-block POP-style selection) relies on this path. Also extend
   `cv_score_regression` in `bench/AOM_v0/aompls/scorers.py` to accept the
   splitter.
2. **`bench/AOM_v0/multiview/multiview/views.py`**
   - `BlockMaskOperator` subclassing `LinearSpectralOperator`.
   - `ViewBuilder` with `.preproc_only()`, `.blocks_only()`, `.combined()`
     factory methods and `.build(p)`.
   - `equal_width` strategy implemented; `quantile_width` /
     `chemistry_NIR` raise `NotImplementedError`.
   - Strict-linearity check on every entry; rejects non-`is_strict_linear`
     operators.
   - Identity-composition dedup (`mask_k ∘ identity ≡ mask_k`,
     `identity ∘ op ≡ op`).
3. **`bench/AOM_v0/multiview/tests/test_views.py`**
   - Strict-linearity: `transform == apply_cov(eye(p))^T`, adjoint identities.
   - `BlockMaskOperator`: idempotency, symmetry, edge cases.
   - Composition order: `M · D ≠ D · M` numerically; `ComposedOperator`
     produces `M · A_preproc` matrix.
   - `ViewBuilder` bank-size invariants and `K`/`p` rejection cases.
   - Cross-block dependence around block edges (numerical demo, not asserted
     as zero — documented expected non-zero edge values).
4. **Regression test for the cv_splitter fix**: a POP+SPXY-CV+holdout-criterion
   smoke, asserting that the splitter emitted to `_criterion_score_at_indices`
   is honoured (e.g. fold partitions are deterministic and not reshuffled).
5. **`bench/AOM_v0/multiview/benchmarks/run_smoke4.py`**
   - Vanilla `nirs4all.operators.models.sklearn.mbpls.MBPLS` baseline (3
     equal-width blocks fed as a list).
   - PLS-standard reference.
   - Best AOM-PLS variant from `bench/AOM_v0/Summary.md` (compact,
     `none+asls+none`, holdout) as the comparator.
   - Outputs `bench/AOM_v0/multiview/results/smoke4_baseline.csv`.
6. Commit Phase 1: `feat(multiview): Phase 1 — ViewBuilder + BlockMaskOperator + smoke-4 baselines + cv_splitter threading fix`.

---

## 9. Codex review — disposition (round 1, 2026-05-01)

| # | Codex severity | Issue | Disposition |
|---|---------------|-------|-------------|
| 1 | OK   | §3 BlockMaskOperator definition consistent. | No change. |
| 2 | MED  | §3.3 cross-block spectral dependence near block edges. | Added explicit note in §3.3 documenting the boundary semantics. |
| 3 | LOW  | §5 composition order math correct, but §6 notation `op_1 ∘ mask_0` reads reversed. | Renamed to `mask_k ∘ op_i` in §6 (mask after op, matches matrix `M_k · A_op`). |
| 4 | LOW  | §4 equal_width detector-segment justification thin. | Kept `equal_width` as Phase-1 metadata-free default; added §10 note that Phase 2 evaluates `quantile_width` if signal is weak. |
| 5 | HIGH | §6.1 bank sizes wrong (compact=15 should be 9; family_pruned=46 should be 15). Complexity formula too coarse. | Fixed §6.1 with verified numbers from `bank_by_name`; rewrote complexity expression to `bank_size · n_folds · K_max(K_max+1)/2 · cost_per_fit_at_depth_1`. |
| 6 | HIGH | `cv_splitter` not threaded through `_criterion_score_at_indices` (line 627 of `selection.py` calls `cv_score_regression` without it). | Added as **Phase 1 foundational fix** in §8.1 — must land before Phase 2 POP+SPXY-CV usage. |
| 7 | HIGH | Strict-linearity not enforced at bank construction; SNV/MSC examples misleading. | Removed SNV from §6 example (compact bank has only strict-linear SG/detrend/FD ops anyway); added explicit `is_linear_at_apply` check in `ViewBuilder.build()` per §6. |
| 7 | HIGH | `K/p` edge cases not rejected up front. | Added §3.4 builder-level validation table (`K < 2`, `K > p`, `K == p`, `p < 2K`). |

## 10. Open follow-ups (Phase 2+)

- `quantile_width` and `chemistry_NIR` strategies — only if Phase 1 shows
  Phase-2 viability with `equal_width`. Quantile-width semantics: split points
  fitted on the **train fold only** to avoid leakage.
- Active prescreen for `response_dedup × K=3` (188 ops): port
  `select_active_superblock` covariance + diversity prune to the multi-view
  bank.
- `ComposedOperator.fit` currently fits each child on the original `X` (Codex
  note). Fine for the strict-linear, parameter-free banks Phase 1 ships.
  Becomes a problem only if a future operator learns parameters from the
  transformed-input distribution; out of scope for Phase 1.
