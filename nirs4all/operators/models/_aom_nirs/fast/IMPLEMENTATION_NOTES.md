# FastAOM Implementation Notes

This document records design decisions and fix history for FastAOM. It complements
the user-facing `README.md`.

## Reuse from `bench/AOM_v0/aompls`

To avoid duplication, FastAOM reuses the parent `aompls` infrastructure:

| Reused from `aompls` | Used in FastAOM |
|---|---|
| `LinearSpectralOperator` base + subclasses (Identity, SG, FD, Detrend, NW, Whittaker, FCK, Composed) | `operator_chain.py`, `chain_generator.py`, `bases.py` |
| `ComposedOperator` (N-deep chains, `apply_cov` / `adjoint_vec` / `matrix`) | `operator_chain.OperatorChain` wraps it |
| `operator_generation.canonicalize / chain_signature / family_signature / grammar_allows` | `operator_chain.simplify`, `grammar.is_extension_valid` |
| `banks.compact_bank / default_bank / extended_bank / deep_bank` | `chain_generator.generate_chains` (bank source) |
| `preprocessing.StandardNormalVariate / MultiplicativeScatterCorrection` | `bases.SNVBase`, `bases.MSCBase` |
| `metrics.rmse / mae / r2 / balanced_accuracy / macro_f1` | `benchmarks/run_fast_aom_benchmark.py` |

What's new (not reused):

- `OperatorChain` wrapper with grammar-aware simplification + stable signature
- `BaseTransform` protocol with raw / absorbance / SNV / MSC / EMSC / ASLS / Whittaker-baseline implementations
- Typed `ChainGrammar` with `role_of()` mapping (baseline/scatter/smoothing/derivative/projection_mask)
- `LowRankBase` (SVD per base) + screening (`||A^T B^T y||^2` / denominator from SVD)
- Diversity-aware `top_k` filter (global / per-family / per-base caps)
- Four model classes (Single, Hard, Soft, Sparse-MKR) — all sklearn-style
- End-to-end `FastAOMPLSRidge` orchestrator
- `run_fast_aom_benchmark.py` runner mirroring `run_aompls_benchmark.py`
- `compare_to_baselines.py` for cross-CSV leaderboards

## Mathematical convention

The parent `aompls.operators.LinearSpectralOperator` uses

```
transform(X) = X @ A.T        # A is the operator's "matrix()"
apply_cov(S) = A @ S
adjoint_vec(v) = A.T @ v
```

The user-facing spec uses `X_s = X @ A_s`, so `A_s = A.T` and `A_s^T = A`.
Therefore screening's `||A_s^T B^T y||^2 = ||A @ B^T y||^2 = ||chain.apply_cov(g_0)||^2`
where `g_0 = B^T y`.

**This was a critical correctness bug in v0**: an initial implementation
used `chain.adjoint_vec(g_0)` (codebase `A.T @ g_0`), which is the wrong
direction for asymmetric chains. Fixed in `screening.py`.

## Codex review fixes (round 1)

After the first batch of edits, an independent Codex review flagged:

| Severity | Issue | Fix |
|---|---|---|
| CRITICAL | `screening.py` numerator used `adjoint_vec` instead of `apply_cov` | Switched to `apply_cov(g_0)` |
| HIGH | `operator_chain.simplify` dropped trailing detrend after derivative — but zero-padded SG/FD derivatives don't annihilate boundary trends, so this discarded valid candidates | Removed the aggressive simplification |
| HIGH | `SparseMultiKernelRidge` greedy selection used stale `K_s @ y_centred` against the residual instead of refreshing `K_s @ residual` | Compute `Kr = K_s @ residual` fresh each step |
| HIGH | Constant-y fold crashed the orchestrator (empty finalists) | Added constant-y guard in `FastAOMPLSRidge.fit` that falls back to a mean predictor |
| MINOR | Soft-LASSO threshold convention | Documented `0.5 * ||y - U a||^2 + rho * ||a||_1` as the canonical objective |

## Codex review fixes (round 2)

After the benchmark runner was added:

| Severity | Issue | Fix |
|---|---|---|
| HIGH | `_run_variant` forced `n_components = max_components`, overriding variants that explicitly chose smaller values (e.g. soft-chain at 12) | Now uses `min(cfg.n_components, max_components)` |
| HIGH | aompls' "cv" mode auto-picks `n_components`; FastAOM uses fixed — comparisons conflate model and selection policy | Documented in README; HardAOM relies on `early_stop_patience=3` to stop adding components when train loss plateaus |
| HIGH | `DetrendProjectionOperator` builds dense `p × p` matrix — RAM risk on very large `p` | OK for the 11-dataset cohort (max `p = 2177` → 37 MB per matrix); flagged in README |
| MINOR | `sparse_mkr` didn't log `theta` weights in per-component diagnostics | Added `theta` to the diagnostic dict |
| MINOR | RESULT_COLUMNS evolves across runs; old CSV headers stay in place | Documented; users should clear the CSV before adding columns |

## Codex review fixes (round 3 — caught at runtime)

| Severity | Issue | Fix |
|---|---|---|
| CRITICAL | `SparseMultiKernelRidge` infinite loop when NNLS zeroes the freshly selected chain: the chain is dropped, residual unchanged, same chain re-selected next iteration, NNLS still drops it, forever. Manifested as the smoke benchmark stalling indefinitely on the 6th variant. | Added a `blacklisted` set: any chain whose NNLS theta drops to zero is permanently rejected so the greedy outer loop can converge. Regression test in `tests/test_models.py::test_sparse_mkr_does_not_infinite_loop_on_redundant_candidates`. |

## Codex review fixes (round 4)

| Severity | Issue | Fix |
|---|---|---|
| HIGH | `OperatorChain.simplify` had a dormant "consecutive same-family smoother collapse" rule that would silently drop the narrower smoother. The grammar currently rejects two consecutive same-family smoothers, so the rule never fired, but a future grammar change could activate it and discard valid candidates. | Removed the rule — the simplifier now only applies the parent `aompls.canonicalize` (identity drops + consecutive detrend collapse). Documented the explicit non-rules in the docstring. |
| MINOR | `FastAOMConfig.n_components` and the other shape-like ints had no validation; passing `0` or negative would silently misbehave downstream (the benchmark runner's ceiling fix would clip to `max_components`). | Added `__post_init__` with `_validate_positive_int` for `n_components`, `rank`, `top_global`, `max_chain_depth`, and `sparse_mkr_max_chains`. |

## Train/test consistency in AOM models

The first version of `HardAOMChainPLSRidge.predict` had a subtle bug: the
orthogonalisation projection coefficient was *recomputed* in predict against
the already-orthogonalised `u_train` (which is approximately zero against
`T_prev`). The fix:

1. Store the projection coefficient `coef_proj` from `T_prev.T @ u_train_raw`
   at fit time (per component).
2. In predict, reuse the stored `coef_proj` to orthogonalise *both* train
   and test directions: `u_train = u_raw - T_prev @ coef_proj`,
   `u_test = u_raw_test - T_test[:, :h] @ coef_proj`.

The same pattern is used in `SoftAOMChainPLSRidge` for the mixed direction.

## Train/test kernel representation

Initial implementations mixed the SVD low-rank kernel approximation (used in
fit) with the exact kernel (used in predict's cross-kernel). This caused
silent mis-predictions. Both `HardAOMChainPLSRidge` and
`SparseMultiKernelRidge` now use the **exact** `K_s = X_centred A^T A X_centred^T`
representation throughout fit and predict; the SVD low-rank machinery is
reserved for the upstream screening stage only (where the approximation is
inherent to the score's purpose).

## Determinism

FastAOM is fully deterministic given `(X_train, y_train, FastAOMConfig)`:

- Chain enumeration is deterministic (DFS over a sorted bank).
- Diversity `top_k` ties are broken by score order (which is deterministic).
- The PLS-NIPALS in `_common.extract_pls_scores` is deterministic for a
  fixed `(X, y)`.
- The Ridge-GCV uses a deterministic eigendecomposition.

This means running with `--seeds 0,1,2` against the same train/test split
produces three identical rows. For the benchmark, use `--seeds 0` only (or
resample train/test splits via the cohort).

## Performance profile (compact bank)

On Beef_Marbling (`n = 554, p = 331`):

| Variant | fit_time | n_finalists | n_chains_enumerated |
|---|---|---|---|
| single_chain | 9.4 s | 120 | 145 |
| hard_aom_chain (compact d3) | 13.4 s | 120 | 145 |
| hard_aom_chain (compact d4) | 13.8 s | 120 | 145 |
| soft_aom_chain | 16.9 s | 80 | 145 |
| sparse_mkr | ~20 s | 60 | 145 |
| hard_aom_chain (default bank, d2) | ~268 s | 200 | ~3000 (with 100-op bank) |

The Python loop in `aompls.operators._xcorr_zero_pad` is the dominant
bottleneck (called ~300 times per chain in screening's `chain.apply_cov(V*S)`).
Replacing it with a vectorised `scipy.signal` convolution would yield a 5-10x
speedup but is out of scope for this experimental framework (the parent
`aompls` is shared with several other benchmarks).

## Open follow-ups

1. **CV-based n_components selection** for fair comparison with aompls'
   `cv` mode (HardAOM has early-stop but Single / Soft do not).
2. **Vectorised xcorr** in the parent codebase to bring screening from
   seconds to milliseconds.
3. **Wavelet / FFT-bandpass projection-mask operators** to enrich the
   projection role (currently relies on detrend + shift only).
4. **Group-aware nested CV** for the headline numbers (`delta_rmsep_vs_master_pls`
   uses a fixed train/test split inherited from the cohort).
5. **Multi-seed runs** require resampling the train/test split or adding
   internal CV randomness (currently deterministic).
