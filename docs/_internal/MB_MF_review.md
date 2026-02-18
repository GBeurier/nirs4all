# AOM-PLS Design Document: Full Critical Review

**Reviewer**: ML Architecture Analysis
**Document reviewed**: `MB_MF.md` (AOM-PLS: Adaptive Operator-Mixture Multi-Block PLS)
**Date**: 2026-02-15
**Codebase context**: nirs4all v0.7.0 — existing MBPLS, SIMPLS, FCKPLS, OPLS, branching/stacking infrastructure

---

## 1. Executive Summary

The AOM-PLS proposal describes a GPU-accelerated model that jointly learns preprocessing selection and PLS regression in a single pass, replacing the traditional "grid search over preprocessing configurations + PLS" workflow. The central mathematical insight — computing block-specific cross-covariance via operator adjoints to avoid materializing preprocessed views — is sound and genuinely novel within the nirs4all context.

**Verdict**: The core idea is strong. The document is well-structured and implementation-ready. However, it has mathematical gaps, underestimates overlap with existing codebase capabilities, and risks over-engineering a solution for a problem that nirs4all's modular pipeline already handles well. Below is a full dissection.

---

## 2. What the Document Proposes (Plain English)

Traditional NIRS workflow: try many preprocessing combos (SNV/MSC + SG derivatives + detrend + OSC) × PLS component counts → pick the best via cross-validation. This is a Cartesian explosion.

AOM-PLS says: **don't search — learn.** Treat each preprocessing as a "view" (block) of the data. Instead of building all views, exploit the fact that SG filters and detrend are *linear operators*. This means:

```
(preprocessed X)^T @ y  =  A^T @ (X^T @ y)
```

So you only need **one** expensive matrix-vector product `X^T @ y`, then cheaply project it through each operator's adjoint. A sparse gating mechanism (entmax/sparsemax) learns *per PLS component* which preprocessing blocks matter, producing a weighted mixture.

The result: an explicit MB-PLS model where block weights `gamma_{b,k}` tell you "component 1 prefers SG(21,2,1)+detrend, component 2 prefers SG(11,3,2) alone", etc.

---

## 3. Mathematical Analysis

### 3.1 The Adjoint Trick (Section 4.2) — Correct and Efficient

The identity `X_b^T u = A_b^T (X^T u)` is exact when `A_b` is a linear operator. This holds for:

- **Savitzky-Golay**: Convolution with a fixed kernel → adjoint is convolution with flipped kernel. Correct.
- **Detrend**: Projection `I - P(P^T P)^{-1} P^T` which is symmetric → adjoint is itself. Correct.
- **Composition SG + detrend**: `(A_SG @ A_det)^T = A_det^T @ A_SG^T`. Correct in exact arithmetic.

**Concern**: Boundary effects in SG convolution (padding modes: reflect, zero, constant) break strict linearity at spectrum edges. The adjoint of "conv with reflect padding" is NOT "conv with flipped kernel and reflect padding." For spectra with ~1000+ wavelengths, edge effects are negligible (~1-2% of features). For short spectra (~100 wavelengths), this could matter. The document does not address this.

**Recommendation**: Use "valid" or "same" convolution with zero padding for the adjoint computation. Accept minor edge discrepancy. Document the approximation.

### 3.2 Block Scoring and Gating (Section 4.2) — Needs Normalization

The block score is defined as:
```
s_{b,k} = ||g_{b,k}||_2^2    where g_{b,k} = A_b^T c_k
```

**Problem**: This is scale-dependent. If operator `A_b` has a larger spectral norm (e.g., a second derivative amplifies high-frequency components), its `g_{b,k}` will have larger magnitude regardless of predictive relevance. The gating will be biased toward high-norm operators.

Consider: SG(11,2,2) (second derivative) produces larger gradient norms than SG(21,2,0) (smoothing) simply because differentiation amplifies noise/high-frequency content. The squared norm conflates "strong response" with "useful response."

**Fix**: Normalize by operator norm:
```
s_{b,k} = ||g_{b,k}||_2^2 / ||A_b||_F^2
```
Or normalize the block loading direction before computing score:
```
s_{b,k} = (g_{b,k}^T c_k)^2 / (||g_{b,k}||^2 * ||c_k||^2)   [cosine similarity squared]
```

The cosine variant measures *alignment* between the block gradient and the cross-covariance, independent of scale. This is more principled for gating.

### 3.3 Effective Weight Construction (Section 4.2) — Subtle Issue

```
w_k = sum_b  gamma_{b,k} * A_b * (g_{b,k} / ||g_{b,k}||)
```

This constructs `w_k` in the *original wavelength space*, which is correct for computing `t_k = X @ w_k`. But this formulation means:

1. The weight is a mixture of *back-projected normalized gradients*.
2. Orthogonalization happens *after* the mixture, in the original space.
3. The gating decisions (`gamma`) are made on non-orthogonalized gradients from possibly non-deflated blocks.

In standard MB-PLS (Westerhuis et al., 1998), each block is deflated independently. Here, only the effective `X` (in the original space) is deflated via SIMPLS projections. This means **block-specific information removal doesn't happen** — component 2's block gradients `g_{b,2}` still contain variation explained by component 1's mixture, just not the component captured by `w_1`.

**This is not necessarily wrong** — SIMPLS deflates the covariance matrix, not X — but it's a departure from classical MB-PLS that should be acknowledged and validated. The document claims "explicit multiblock" but the deflation strategy is single-block SIMPLS applied to an implicitly constructed view. The block weights are interpretable post-hoc, but the blocks don't interact during deflation the way they do in true MB-PLS.

### 3.4 SIMPLS Choice — Good but Verify

SIMPLS (de Jong 1993) deflates `S = X^T Y` rather than `X` and `Y` separately. This is a good choice because:
- Numerically more stable for many components
- Avoids sequential deflation of large matrices
- The existing nirs4all `SIMPLS` implementation (`operators/models/sklearn/simpls.py`) can serve as reference

But the hybrid "SIMPLS deflation + operator-mixture weights" is novel. There's no theorem guaranteeing this converges to the same solution as any standard PLS variant. The document should:
1. Prove or cite that the extracted components maximize covariance (or acknowledge they approximate it).
2. Show that the gating doesn't create degenerate solutions (all weight on one block for all components).

### 3.5 OPLS Integration (Section 5, Step C) — Reasonable

Using OPLS-style orthogonal components as a built-in option (rather than preprocessing) is sound. The existing `OPLS` implementation uses the Trygg & Wold (2002) approach. The proposal to sweep `(n_orth, K)` prefixes is practical.

**Minor concern**: OPLS orthogonal components should be extracted *before* the predictive components. The document's Step C ordering is ambiguous — it reads as if OPLS components are extracted *after* all K predictive components. Clarify: is it `OPLS filter → AOM-PLS`, or `AOM-PLS with embedded orthogonal deflation`? The former is simpler and well-understood.

---

## 4. Comparison with Existing nirs4all Infrastructure

This is the critical question: **does AOM-PLS offer something the existing codebase can't already achieve?**

### 4.1 Current Approaches to "Preprocessing Search + PLS"

**Approach A: Cartesian pipeline search**
```python
pipeline = [
    {"_or_": [SNV, MSC, None]},                    # 3 scatter options
    {"_or_": [SG(11,2,1), SG(21,2,2), Detrend()]}, # 3 operators
    PLSRegression({"_range_": [1, 25, 1]}),         # 25 component counts
]
# Total: 3 × 3 × 25 = 225 configurations, all evaluated with CV
```
This is already fast in nirs4all with step caching enabled. For 225 configs on typical NIRS data (200 samples × 1000 wavelengths), runtime is ~30 seconds on CPU.

**Approach B: Branching + stacking**
```python
pipeline = [
    {"branch": [
        [SNV(), SG(11,2,1), PLSRegression(10)],
        [MSC(), SG(21,2,2), PLSRegression(10)],
        [Detrend(), PLSRegression(10)],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
```
This is essentially an ensemble over preprocessing views — MB-PLS by another name.

**Approach C: Explicit MBPLS**
```python
# Precompute blocks (in a custom step or pre-pipeline)
X_snv_sg1 = SG(11,2,1).fit_transform(SNV().fit_transform(X))
X_msc_sg2 = SG(21,2,2).fit_transform(MSC().fit_transform(X))
model = MBPLS(n_components=10)
model.fit([X_snv_sg1, X_msc_sg2], y)
```

**Approach D: FCKPLS**
```python
model = FCKPLS(
    alphas=[0.0, 1.0, 2.0],  # smoothing, 1st deriv, 2nd deriv
    sigmas=[1.0, 2.0, 4.0],  # scale parameters
    n_components=10,
)
model.fit(X, y)
```
FCKPLS already does "filter bank → expanded features → PLS" in a single estimator.

### 4.2 What AOM-PLS Adds Over Existing Approaches

| Feature | Cartesian Search | Branching+Stack | MBPLS | FCKPLS | **AOM-PLS** |
|---------|-----------------|-----------------|-------|--------|-------------|
| Automatic preprocessing selection | Via grid | Manual | Manual | Implicit (learned) | **Explicit (sparse gating)** |
| Memory efficiency | High (sequential) | High (parallel branches) | O(n × Bp) | O(n × Lp) | **O(np + Bp) via adjoint** |
| Per-component preprocessing | No (global) | No (global) | Yes (block weights) | No (global PLS) | **Yes (gamma_{b,k})** |
| Interpretability | Best config | Branch predictions | Block weights | Filter weights | **Block weights + virtual recipe** |
| GPU acceleration | Not needed | Not needed | JAX backend | JAX backend | **Torch (native)** |
| One-shot (no search) | No | No | Yes (given blocks) | Yes | **Yes** |
| Handles scatter correction | In grid | In branches | Pre-applied | No | **Global select (crude)** |

**Genuine advantages of AOM-PLS**:
1. **Per-component preprocessing selection** — This is the killer feature. Standard approaches pick one global preprocessing; AOM-PLS can use SG(11,2,1) for component 1 and SG(21,2,2) for component 5. This is theoretically more flexible.
2. **Memory-efficient adjoint computation** — No need to materialize B copies of X. With 30 blocks and 1000-wavelength spectra, this saves ~30× memory.
3. **Sparse interpretability** — Entmax zeros out irrelevant blocks, giving a clear "this component uses these preprocessings" story.

**Marginal or absent advantages**:
1. **Speed** — For typical NIRS datasets (n < 1000, p < 2000, B < 50), the Cartesian search with step caching is already fast. GPU overhead (data transfer, kernel launch) may negate gains.
2. **Accuracy** — No evidence (theoretical or empirical) that per-component preprocessing beats global preprocessing. If the best SG config is globally best, per-component gating adds unnecessary flexibility.
3. **Scatter handling** — The "global select" for scatter undermines the "one-shot" claim. You're still doing a mini-search for scatter.

### 4.3 Overlap with FCKPLS

FCKPLS is the closest existing relative. Both do "filter bank → PLS". Key differences:

| Aspect | FCKPLS | AOM-PLS |
|--------|--------|---------|
| Filter type | Fractional order (continuous α ∈ [0,2]) | SG configs (discrete set) |
| Materialization | Explicit (builds X_feat ∈ R^{n × Lp}) | Implicit (adjoint trick) |
| Feature space | Concatenated filtered views | Weighted mixture in original space |
| Selection | None (PLS handles it implicitly) | Explicit sparse gating |
| Detrend | Not included | Included in operator bank |

The two approaches are complementary and could be unified (see Section 7).

---

## 5. Can AOM-PLS Overshadow Standard PLS and MB-PLS?

### 5.1 Against Standard PLS

**Theoretical case**: Yes. AOM-PLS strictly generalizes PLS. If you include the identity operator in the bank (pass-through, no preprocessing), AOM-PLS can recover standard PLS as a special case where `gamma_{identity,k} = 1` for all components. Any improvement from preprocessing is additive.

**Practical case**: Marginal. On well-studied NIRS problems:
- **Corn dataset**: Best PLS with SNV+SG(11,2,1) achieves RMSECV ≈ 0.03-0.05 for moisture. The preprocessing is well-known. AOM-PLS might match but unlikely to beat significantly.
- **Shootout dataset**: Similar story. The community has converged on effective preprocessing.
- **Novel/difficult datasets**: Here AOM-PLS could shine — when the optimal preprocessing is unknown or varies by target property.

**Expected improvement**: 1-5% RMSECV reduction over "good human-selected preprocessing + PLS" on typical NIRS datasets. Larger gains (5-15%) possible on:
- Multi-property datasets (different properties may need different preprocessing)
- Datasets with complex baseline variation
- High-dimensional spectra (>2000 wavelengths) where preprocessing matters more

### 5.2 Against MB-PLS

AOM-PLS is a **strict superset** of MB-PLS:
- MB-PLS requires pre-computed blocks → AOM-PLS constructs blocks implicitly
- MB-PLS has no block selection → AOM-PLS has sparse gating
- MB-PLS super score averages block scores → AOM-PLS weighted mixture of back-projected gradients

The per-component gating is the differentiator. In classical MB-PLS, all blocks contribute to every component. In AOM-PLS, component 1 might use only blocks {3, 7} while component 2 uses only block {12}. This is more informative and potentially more accurate.

**However**: The existing nirs4all `MBPLS` with manually selected blocks is transparent, debuggable, and well-understood. AOM-PLS's automatic block construction adds a black-box layer. For users who understand their spectral data, manual MB-PLS may be preferred.

### 5.3 The Real Competition: nirs4all Pipeline Search

The elephant in the room: nirs4all's `_cartesian_` / `_or_` combinatorial pipeline syntax **already solves this problem** from a user perspective:

```python
pipeline = [
    {"_or_": [SNV(), MSC(), EMSC(), None]},
    {"_or_": [SG(11,2,0), SG(11,2,1), SG(21,2,1), SG(11,3,2), Detrend(), None]},
    PLSRegression({"_range_": [1, 30, 1]}),
]
result = nirs4all.run(pipeline=pipeline, dataset=dataset)
# Explores 4 × 6 × 30 = 720 configurations with CV
```

This is:
- Transparent (every configuration is explicit)
- Well-integrated (works with all nirs4all features: workspace, export, visualization)
- Fast enough (with step caching, shared preprocessing across PLS component counts)
- Interpretable (best pipeline is directly readable)

AOM-PLS's advantage is going from O(B × K) evaluations to O(K) component extractions. But B × K is already small for NIRS (B < 50, K < 30 → <1500 evals → <1 minute with caching).

**AOM-PLS becomes compelling when**:
- B is large (>100 blocks) — e.g., fine-grained SG parameter sweeps
- n is very large (>10,000 samples) — makes per-PLS-fit expensive
- Real-time/online learning is needed — one-shot is essential
- Deployed models need adaptive preprocessing — the operator bank + gating can be updated

---

## 6. Critical Issues and Gaps in the Document

### 6.1 No Theoretical Convergence Guarantee

The hybrid "sparse gating + SIMPLS deflation" has no convergence proof. Standard PLS maximizes covariance; MB-PLS maximizes block-weighted covariance; but AOM-PLS's mixture-gated deflation doesn't obviously optimize a well-defined objective. This is the biggest theoretical gap.

**Recommendation**: Frame AOM-PLS as a **heuristic** inspired by MB-PLS, not as "a PLS variant." Validate empirically rather than claiming theoretical properties.

### 6.2 Scatter Handling is a Weak Point

The document proposes global scatter selection using `||X^{(s)T} y||_2` as criterion. This:
- Breaks the "one-shot" narrative
- Is a crude criterion (high cross-covariance norm ≠ good prediction)
- MSC/EMSC are sample-wise normalizations that depend on a reference spectrum — they're genuinely nonlinear and can't be folded into the operator bank

**Better approach**: Accept scatter as a separate concern. Let the nirs4all pipeline handle scatter selection upstream:
```python
pipeline = [
    {"_or_": [SNV(), EMSC(), None]},   # Pipeline handles scatter search
    TorchAOMPLSRegressor(k_max=25),     # AOM-PLS handles linear operator search
]
```
This is cleaner, more modular, and leverages existing infrastructure.

### 6.3 Missing: Identity Block

The operator bank should always include the identity operator (no preprocessing). This:
- Guarantees AOM-PLS can recover standard PLS
- Acts as a baseline block that other blocks must beat
- Makes the "sanity test" (Section 10, Epic 8) trivially satisfiable

### 6.4 Missing: Interval/Region Selection

NIRS practitioners often select wavelength regions. The operator bank could include region-selection operators:
```
A_region = diag(mask_vector)   # Linear! Adjoint = itself
```
This would subsume interval PLS (iPLS) into the framework.

### 6.5 Entmax/Sparsemax Dependency

Entmax (Peters et al., 2019) is not in standard PyTorch. Options:
- `entmax` pip package (small, maintained)
- Implement sparsemax manually (simpler, no dependency)
- Use Gumbel-softmax with hard straight-through as alternative

**Recommendation**: Start with sparsemax (easy to implement: Euclidean projection onto simplex). Add entmax as optional upgrade.

### 6.6 Mixed Precision Risks

The document proposes bf16/fp16 for GEMMs. For PLS:
- Cross-covariance `X^T y` is the critical computation — fp16 can lose 3-4 significant digits
- With n=200 samples (common in NIRS), the cross-covariance is a sum of 200 outer products — cumulative rounding error is real
- Orthogonalization in reduced precision is numerically dangerous

**Recommendation**: Use fp32 throughout. The matrices are small enough (p < 3000 typically) that fp32 GEMV is already fast. Mixed precision adds complexity for negligible gain on NIRS-scale data.

### 6.7 GPU Overhead vs Gain

For typical NIRS datasets:
- n ∈ [100, 1000], p ∈ [500, 2000]
- The dominant operation `c = X^T y` is a GEMV of size (p × n) @ (n,) → (p,)
- On CPU: ~0.1ms. On GPU: ~0.05ms + ~0.5ms transfer overhead.

GPU is only beneficial when:
- n > 5000 (large datasets from process spectroscopy)
- Running many components (K > 20) where operator bank ops accumulate
- Batch prediction on large datasets

**Recommendation**: Implement CPU-first with NumPy/SciPy. Add Torch backend as optional accelerator. Don't require GPU.

---

## 7. Proposed Enhancements

### 7.1 Unified Filter-Bank PLS Framework

Instead of AOM-PLS as a standalone estimator, implement a composable framework:

```python
# Filter bank (reusable component)
bank = OperatorBank([
    SGOperator(11, 2, 1),
    SGOperator(21, 2, 2),
    DetrendOperator(degree=1),
    IdentityOperator(),        # Always include
    # RegionOperator(mask),    # Optional interval selection
])

# Gating strategy (pluggable)
gate = SparsemaxGate(temperature=1.0)
# gate = EntmaxGate(alpha=1.5)
# gate = TopKGate(k=3)
# gate = UniformGate()  # Reduces to standard MB-PLS

# Core PLS engine (existing)
pls_core = "simpls"  # or "nipals"

# Composed model
model = FilterBankPLS(
    bank=bank,
    gate=gate,
    pls_algorithm=pls_core,
    n_components=25,
    n_orth=0,
)
```

This is more modular, reusable, and consistent with nirs4all's philosophy.

### 7.2 Unify with FCKPLS

FCKPLS and AOM-PLS share the same structure: `{filter bank} → {feature aggregation} → {PLS}`. The filter bank is the only difference:

| Component | FCKPLS | AOM-PLS | Unified |
|-----------|--------|---------|---------|
| Bank | Fractional kernels | SG + detrend | Any linear 1D operator |
| Aggregation | Concatenation | Sparse-gated mixture | Pluggable (concat, gate, attention) |
| PLS | Standard PLSRegression | SIMPLS with custom weights | Any PLS variant |

A unified `FilterBankPLS` that accepts any bank + any aggregation strategy would subsume both.

### 7.3 Block Gradient Normalization

Replace raw squared norms with normalized scores:

```python
# Current (scale-dependent)
s_b = torch.sum(g_b ** 2)

# Proposed (scale-invariant)
s_b = torch.sum(g_b ** 2) / (operator_norms[b] ** 2 + eps)

# Or cosine similarity
s_b = (g_b @ c) ** 2 / (torch.sum(g_b ** 2) * torch.sum(c ** 2) + eps)
```

### 7.4 Validation-Based Component Selection

The document proposes prefix selection (try K=1..K_max, pick best). This is standard in PLS. Enhancement: use an efficient leave-one-out formula (PRESS statistic) that doesn't require held-out data:

```
PRESS_k = sum_i (y_i - y_hat_{-i,k})^2
```

For PLS, PRESS can be computed in closed form from the training decomposition. This gives model selection without wasting data on a validation split.

### 7.5 Warm Starting and Incremental Updates

If the model is deployed for process monitoring:
- New data arrives → update gating weights without full refit
- Operator bank can grow → add new blocks without discarding old components

This is unique to AOM-PLS (standard PLS can't do this) and could be a strong selling point.

---

## 8. Recommended Implementation Strategy

### Phase 0: Groundwork (Low effort, immediate value)
1. Add identity operator to all filter bank discussions
2. Implement `OperatorBank` as a standalone utility class (SG kernels + detrend projections)
3. Implement `sparsemax` as a standalone function (10 lines of code)
4. Add to existing `MBPLS`: option to pass a bank + auto-construct blocks

### Phase 1: CPU-first AOM-PLS (Medium effort, core value)
1. NumPy implementation of the adjoint-based component extraction
2. Sparsemax gating with normalized block scores
3. SIMPLS core (reuse existing `simpls.py` logic)
4. Prefix selection for K* using PRESS or held-out validation
5. sklearn-compatible API: `fit(X, y)`, `predict(X)`, `get_block_weights()`
6. Scatter handled externally (pipeline concern, not model concern)

### Phase 2: Torch GPU backend (High effort, niche value)
1. Port Phase 1 to Torch with grouped conv1d for operator bank
2. Mixed precision (fp32 only — see Section 6.6)
3. Batch prediction for large datasets
4. Device management (CPU/GPU auto-detection)

### Phase 3: Advanced features (Future)
1. OPLS integration
2. Entmax gating
3. Region-selection operators (interval PLS fusion)
4. Warm starting / incremental updates
5. Unify with FCKPLS into `FilterBankPLS`

---

## 9. Benchmarking Requirements

Before investing in Phase 2+, validate the core hypothesis with Phase 1:

### Must-run experiments:

1. **Sanity check**: Identity-only bank → compare against sklearn `PLSRegression`. Must match within numerical tolerance.

2. **Value of per-component preprocessing**: Compare on 3+ real NIRS datasets:
   - AOM-PLS (per-component gating)
   - Best global preprocessing + PLS (nirs4all Cartesian search)
   - Fixed preprocessing + PLS (SNV + SG(11,2,1))

   If AOM-PLS doesn't beat the Cartesian search by a meaningful margin, the complexity isn't justified.

3. **Gating interpretability**: On a well-understood dataset, check if block weights `gamma_{b,k}` match domain knowledge (e.g., "first derivative is important for moisture prediction in grain").

4. **Scaling study**: Measure runtime vs (n, p, B, K) to establish when GPU backend becomes worthwhile.

### Success criteria:
- RMSECV improvement > 3% over best Cartesian search result on at least 2/3 test datasets
- Training time < 2× the single best PLS fit (amortized vs grid search, it should be much faster)
- Block weights are interpretable and stable across CV folds

---

## 10. Summary of Recommendations

| # | Recommendation | Priority | Rationale |
|---|---------------|----------|-----------|
| 1 | Handle scatter externally (pipeline concern) | **High** | Cleaner separation, leverages existing search |
| 2 | Always include identity operator in bank | **High** | Guarantees PLS recovery, baseline block |
| 3 | Normalize block scores by operator norm | **High** | Prevents scale-dependent gating bias |
| 4 | Implement CPU-first (NumPy), GPU optional | **High** | NIRS data is small; GPU overhead may dominate |
| 5 | Start with sparsemax, not entmax | **Medium** | Simpler, no dependency, sufficient sparsity |
| 6 | Use fp32 throughout (skip mixed precision) | **Medium** | NIRS matrices are small; numerical risk > speed gain |
| 7 | Clarify OPLS ordering (filter then predict) | **Medium** | Avoids ambiguity in component extraction |
| 8 | Add region-selection operators | **Low** | Subsumes iPLS, broadens applicability |
| 9 | Unify with FCKPLS into FilterBankPLS | **Low** | Architectural elegance, long-term maintainability |
| 10 | Validate before GPU investment | **Critical** | Phase 1 benchmarks must justify Phase 2 effort |

---

## 11. Final Assessment

### Strengths of the proposal:
- **Mathematically principled** adjoint computation trick
- **Per-component preprocessing** is a genuinely novel capability
- **Sparse interpretability** via entmax/sparsemax gating
- **Well-structured** document with clear epics and milestones

### Weaknesses:
- **No convergence theory** for the hybrid gating+SIMPLS formulation
- **Scatter handling** breaks the "one-shot" narrative
- **Significant overlap** with existing nirs4all pipeline search
- **GPU investment** may not pay off for typical NIRS data sizes
- **Missing normalization** in block scoring creates gating bias
- **FCKPLS overlap** not discussed

### Bottom line:

AOM-PLS is a **theoretically interesting and potentially valuable** addition to nirs4all, but it should be implemented **incrementally with validation gates**. The CPU-only Phase 1 implementation (estimated 2-3 weeks of focused development) will answer the critical question: does learned per-component preprocessing selection actually beat systematic search on real NIRS data?

If yes: proceed to GPU backend and full integration.
If marginal: consider the simpler enhancement of adding sparse gating to the existing `MBPLS` class instead.

The worst outcome would be building the full GPU pipeline (Phases 1-3) without Phase 1 benchmarking, only to discover the improvement over `nirs4all.run(pipeline=[{"_or_": [...]}, PLSRegression()])` is negligible.
