# Claude Code Prompt: Active Superblock Mode

## Proposed Mode Name

Use:

```text
selection="active_superblock"
```

Working name:

```text
Active Superblock PLS
```

Rationale: this is not just a raw concatenation of all transformed views. The
mode first builds an active, diverse, fold-local subset of operators, then
fits a block-balanced superblock PLS/SIMPLS on that subset.

## Prompt To Give To Claude Code

```text
You are working in the repository at /home/delete/nirs4all/nirs4all.

Scope:
- Work only in bench/AOM_v0.
- Do not modify the production nirs4all package.
- Implement and test a new AOM_v0 selection mode named selection="active_superblock".

Read these files first:
- bench/AOM_v0/docs/AOMPLS_MATH_SPEC.md
- bench/AOM_v0/docs/AOMPLS_OPERATOR_EXPLORATION.md
- bench/AOM_v0/aompls/operators.py
- bench/AOM_v0/aompls/banks.py
- bench/AOM_v0/aompls/simpls.py
- bench/AOM_v0/aompls/selection.py
- bench/AOM_v0/aompls/estimators.py
- bench/AOM_v0/tests/test_selection.py
- bench/AOM_v0/tests/test_estimators.py

Goal:
Add a scientifically usable "Active Superblock PLS" mode:

1. Build an active bank from the provided operator bank using only training data.
2. Prune operators by covariance response score and response diversity.
3. Build a block-balanced superblock from that active bank.
4. Fit SIMPLS on the wide matrix.
5. Map wide-space coefficients and effective weights back to original feature space so estimator.predict(X) works on raw X with shape (n, p).
6. Add tests for selection-level behavior, estimator-level prediction, diagnostics, and leakage-safe fold behavior where applicable.

Important mathematical conventions:
- Each strict linear operator A acts on row spectra as X_b = X A^T.
- Superblock with block weights alpha_b is:
  X_wide = [alpha_1 X A_1^T | alpha_2 X A_2^T | ...]
- If wide SIMPLS gives coefficient blocks beta_b, original-space prediction is:
  y_hat = X sum_b alpha_b A_b^T beta_b
- Therefore the original-space coefficient matrix must be:
  B_original = sum_b alpha_b A_b^T beta_b
- Use op.adjoint_vec(...) to apply A_b^T column-wise.
- The same mapping applies to wide-space component weights Z_wide:
  Z_original[:, a] = sum_b alpha_b A_b^T Z_wide_block_b[:, a]
- Recompute original-space scores/loadings from X and Z_original so the estimator attributes remain coherent:
  T = X @ Z_original
  P[:, a] = X.T @ t_a / (t_a.T @ t_a)
  Q[:, a] = Y.T @ t_a / (t_a.T @ t_a)

Do not leave active_superblock coefficients in wide feature space. The current estimator predicts with:
  (X - x_mean_) @ coef_ + y_mean_
so coef_ must have shape (p, q), not (p * n_blocks, q).

Implementation requirements:

1. Add active-bank screening helpers.
   Prefer new private helpers in selection.py or a small new module if cleaner.
   Minimum behavior:
   - Compute S = Xc.T @ yc.
   - For each operator b:
       R_b = op.apply_cov(S)
       raw_score_b = -||R_b||_F  # lower is better, match covariance_score convention
   - Sort by score.
   - Keep the first candidate, then keep later candidates only if their response cosine similarity to already-kept responses is below a threshold.
   - Stop at active_top_m.
   - Always keep identity if present, unless active_top_m is too small; prefer active_top_m >= 2 in validation.

2. Add block scaling.
   Implement at least:
   - block_scaling="frobenius" default
   - block_scaling="none"
   For frobenius scaling:
   - alpha_b = 1 / (||X_b||_F + eps)
   - optionally multiply by sqrt(n_samples) or another constant if needed, but keep relative block balancing correct and document it in code comments.
   The goal is that high-gain derivative/high-pass blocks do not dominate only by scale.

3. Add a weighted superblock implementation.
   Either extend superblock_simpls(...) or add a new function, for example:
     active_superblock_simpls(...)
     weighted_superblock_simpls(...)
   It should return enough information for:
   - original-space NIPALSResult
   - groups
   - active operator indices
   - active operator names
   - block weights
   - group importance
   - covariance screening scores
   - response diversity diagnostics

4. Add selection="active_superblock" to select(...).
   Suggested signature:
   - keep existing public select(...) signature stable if possible.
   - Use conservative constants if estimator parameters are not yet available:
       active_top_m = min(20, len(operators))
       diversity_threshold = 0.98
       block_scaling = "frobenius"
   If adding estimator parameters is straightforward and sklearn-compatible, add:
       active_top_m=20
       active_diversity_threshold=0.98
       superblock_scaling="frobenius"
   Make sure get_params/set_params still work.

5. Fix or preserve existing selection="superblock".
   If current raw superblock returns wide-space coefficients that break estimator.predict, fix it too or ensure active_superblock has its own safe path.
   Add a regression test that would fail if coef_.shape[0] != p for superblock-like modes.

6. Diagnostics.
   active_superblock diagnostics should include:
   - selection: "active_superblock"
   - active_operator_indices
   - active_operator_names
   - active_operator_scores
   - block_weights
   - group_importance
   - active_top_m
   - diversity_threshold
   - block_scaling
   - original_feature_space: true

7. Tests.
   Add pytest tests covering at least:

   a. Selection-level active bank:
      - selection="active_superblock" returns diagnostics["selection"] == "active_superblock".
      - active_operator_indices length <= active_top_m.
      - identity is retained when present.
      - block_weights length equals active bank length.
      - group_importance keys match active operator names.

   b. Estimator-level prediction:
      - AOMPLSRegressor(selection="active_superblock", operator_bank="compact", max_components=3, criterion="covariance") fits.
      - est.coef_.shape == (p, 1).
      - est.predict(X_test) returns shape (n_test,).
      - est.transform(X_test) returns shape (n_test, est.n_components_).
      - diagnostics include active_superblock extras.

   c. Existing superblock safety:
      - AOMPLSRegressor(selection="superblock", operator_bank="compact", max_components=3, criterion="covariance") either fits and predicts with coef shape (p, q), or explicitly remains selection-level only with a clear test/exception.
      Prefer fixing it to produce original-space coefficients.

   d. Block scaling:
      - Create two duplicate/scaled operator blocks, or use identity plus a high-gain explicit operator.
      - Verify frobenius scaling gives finite weights and finite predictions.
      - Verify diagnostics expose the weights.

   e. Diversity pruning:
      - Use duplicated operators or near-identical explicit operators.
      - Verify active bank prunes duplicates when diversity_threshold is strict.

8. Leakage rules.
   Active bank generation must be based on the Xc/yc passed to select(...).
   Do not inspect validation/test data.
   If CV scoring is used internally, active bank generation must happen inside each training fold, not once globally using all fold data. If this is too large for this task, document the limitation and restrict active_superblock's first implementation to criterion="covariance"/"approx_press" until fold-local generation is implemented.

9. Benchmark variants.
   Add one regression benchmark variant:
     label: "ActiveSuperblock-simpls-numpy"
     selection: "active_superblock"
     engine: "simpls_covariance"
     operator_bank: "default" or "compact"
     backend: "numpy"
     experimental: True
   Prefer compact first if runtime is a concern.

10. Run tests.
   At minimum run:
     pytest bench/AOM_v0/tests/test_selection.py bench/AOM_v0/tests/test_estimators.py -q
   If there are operator or SIMPLS changes, also run:
     pytest bench/AOM_v0/tests/test_simpls.py bench/AOM_v0/tests/test_operators.py -q

Acceptance criteria:
- selection="active_superblock" is implemented and tested.
- Estimator predictions work in original feature space.
- coef_ shape is (p, q), not wide-space.
- Diagnostics clearly identify active operators and block weights.
- No production nirs4all files are modified.
- Tests pass or any failing tests are explained with concrete reasons.
```

## Design Notes For Review

The key difference from the existing `selection="superblock"` is not the PLS
engine. It is the selection and scaling before concatenation:

```text
superblock:
    concatenate all operator views

active_superblock:
    covariance-screen operators
    prune redundant responses
    normalize each retained block
    concatenate only the active blocks
    map coefficients back to original feature space
```

This mode is useful as a diagnostic comparator against:

```text
global AOM      -> one operator for the model
POP             -> one operator per component
soft            -> covariance-space operator mixture
active_superblock -> simultaneous multi-view combination
```

If `active_superblock` consistently wins against AOM/POP on the same active
bank, the dataset likely benefits from multiple preprocessing views
contributing at the same time. If it does not, AOM/POP remains preferable
because it is simpler and more interpretable.
