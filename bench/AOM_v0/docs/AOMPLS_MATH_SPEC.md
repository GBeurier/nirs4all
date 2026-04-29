# AOMPLS Mathematical Specification

## Matrix Conventions

Use these conventions everywhere in code, tests, docs, and paper:

- `X in R^{n x p}` is samples by wavelengths/features.
- `Y in R^{n x q}` is response matrix. PLS1 is `q=1`; PLS2 is `q>1`.
- Models center `X` and `Y` during fit and store means for prediction.
- Per-feature scaling is optional and defaults to `False` for spectra.
- A spectral operator `A_b in R^{p x p}` acts on row spectra as:

```text
X_b = X A_b^T
```

Therefore:

```text
X_b^T Y = A_b X^T Y
```

This identity is mandatory for covariance-space SIMPLS and must be tested for
every strict linear operator.

## Selection Policies

The framework implements a single object family with explicit policies:

- `selection="none"`: standard PLS, equivalent to identity-only bank.
- `selection="global"`: AOM global hard selection, one operator for the whole
  model.
- `selection="per_component"`: POP, one operator per extracted component.
- `selection="soft"`: experimental convex mixture.
- `selection="superblock"`: concatenate transformed blocks and fit PLS.

For `global`, the default scientific selection is full candidate evaluation by
inner CV or exact/approximate PRESS, then refit on all calibration data. A
single-component covariance proxy is allowed only when explicitly named
`criterion="covariance"`.

For `per_component`, each extraction step evaluates all candidates from the
current residual/projected state and stores the full candidate score table.

## Effective Weights and Prediction

Regardless of engine, prediction must use original-space coefficients:

```text
Y_hat = (X - x_mean) B + y_mean
```

If component `a` is extracted through transformed-space direction `r_a` under
operator `A_{b_a}`, define the original-space effective weight:

```text
z_a = A_{b_a}^T r_a
```

With score matrix `T = X Z`, loadings:

```text
p_a = X^T t_a / (t_a^T t_a)
q_a = Y^T t_a / (t_a^T t_a)
```

The default coefficient construction is:

```text
B = Z (P^T Z)^+ Q^T
```

where `+` is inverse when full-rank and Moore-Penrose pseudoinverse otherwise.
Every implementation must expose `x_effective_weights_ = Z`.

## Orthogonalization Modes

Two modes are required:

- `orthogonalization="transformed"`: orthogonalize in the transformed space of
  the current operator. This must match materialized PLS/SIMPLS on `X A_b^T`
  when a single fixed operator is used.
- `orthogonalization="original"`: map directions back to original-space weights
  and build orthogonality using `X z_a`. This is the default for
  `selection="per_component"` because operators may change across components.

The modes are not assumed equivalent. Tests must demonstrate identity
equivalence, fixed-operator equivalence for transformed mode, and stable
prediction for original mode.

## NIPALS Engines

Materialized NIPALS reference:

1. Build `X_b = X A_b^T`.
2. Run a standard deterministic NIPALS extraction on `X_b, Y`.
3. Map transformed weights back to original weights with `A_b^T`.

Adjoint NIPALS:

1. At each residual state compute `S = X_res^T Y_res`.
2. For PLS1 use `s = S[:, 0]`; for PLS2 use the leading left singular vector
   scaled by its singular value.
3. For candidate `b`, transformed covariance is `A_b s`.
4. Build the transformed direction and map to original-space effective weight.
5. Extract score `t = X_res z`.

The convention `X_b^T y = A_b X^T y` must be tested on explicit matrices.

## SIMPLS Engines

Materialized SIMPLS reference:

1. Build `X_b = X A_b^T`.
2. Run standard SIMPLS on `X_b, Y`.
3. Map all component weights to original space with `A_b^T`.

Covariance-space SIMPLS:

1. Start with `S = X^T Y`.
2. For candidate `b`, use `S_b = A_b S`.
3. For PLS1, direction is normalized `S_b[:, 0]`.
4. For PLS2, direction is the dominant left singular vector of `S_b`.
5. Map transformed direction to effective original-space weight.
6. Update the orthogonal basis using the chosen orthogonalization mode.

For `selection="per_component"`, the selected operator sequence is a greedy
operator pursuit in covariance space. Both materialized and covariance engines
must agree in fixed-operator transformed mode on small deterministic examples.

## Selection Criteria

Supported criteria:

- `criterion="covariance"`: fast screening, not default for final science.
- `criterion="cv"`: leakage-safe K-fold or repeated K-fold, default for
  benchmark model selection.
- `criterion="approx_press"`: approximate PRESS only if named explicitly.
- `criterion="press"`: only if the implemented formula is mathematically
  validated by tests.
- `criterion="hybrid"`: covariance top-m prescreen followed by CV.
- `criterion="holdout"`: legacy/debug only, never default.

All fitted operators and calibrators must be fitted on training folds only.

## Soft Mixture

Soft mixture defines:

```text
A_alpha = sum_b alpha_b A_b
alpha_b >= 0
sum_b alpha_b = 1
```

Implementation requirements:

- `gate="softmax"` and `gate="sparsemax"` are supported.
- Sparsemax must project exactly onto the simplex.
- Soft selection is experimental and cannot be the default benchmark winner
  unless it beats hard policies under the full protocol.

## Superblock

The superblock baseline uses:

```text
X_super = [X A_1^T, X A_2^T, ..., X A_B^T]
```

It must report group-wise coefficient norms and group importances mapped back
to operator names. It is a baseline, not the main AOM/POP contribution.

## Classification: AOM-PLS-DA and POP-PLS-DA

For `task="classification"`:

1. Encode labels with class-balanced one-hot coding:

```text
Y_ic = 1 / sqrt(pi_c) if y_i = c, else 0
```

where `pi_c` is the training class prior. Center columns after coding.

2. Fit the same operator-adaptive PLS2 engine using this `Y`.
3. Transform training samples to latent scores `T`.
4. Fit `LogisticRegression(class_weight="balanced", max_iter=2000)` on `T`.
5. `predict_proba(X)` transforms `X` to latent scores and calls the logistic
   calibrator.
6. If logistic calibration fails, fallback is temperature-scaled softmax on raw
   PLS scores, with `temperature` fitted on training folds only.

Selection criteria for classification:

- default benchmark criterion: inner-CV balanced log loss, tie-broken by balanced
  accuracy.
- smoke criterion: covariance/class-separation hybrid.

Report:

- balanced accuracy,
- macro-F1,
- log loss,
- Brier score for binary tasks,
- expected calibration error.
