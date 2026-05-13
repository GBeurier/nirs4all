# AOM-Ridge Test Plan

## Test 1: Identity Equals Standard Ridge

With `operator_bank=[IdentityOperator()]`, AOM-Ridge must match
`sklearn.linear_model.Ridge` with the same alpha.

Checks:

- predictions match;
- `coef_` matches;
- `intercept_` matches.

## Test 2: Single Operator Dual Equals Materialized Ridge

For strict operator `A`, compare:

```text
Ridge on Z = Xc A^T
dual Ridge on K = Z Z^T
original beta = A^T A Xc^T C
```

Checks:

- train predictions match;
- test predictions match;
- `K_cross @ C == Xtest_c @ beta`.

## Test 3: Superblock Dual Equals Explicit Concatenation

Compare explicit:

```text
Phi = [s_1 X A_1^T | ... | s_B X A_B^T]
```

against:

```text
K = sum_b s_b^2 X A_b^T A_b X^T
beta = sum_b s_b^2 A_b^T A_b X^T C
```

Checks:

- train/test predictions match;
- `coef_` has original shape `(p, q)`;
- wide coefficients are never assigned to estimator `coef_`.

## Test 4: Coefficient Original-Space Identity

After fit:

```text
Xc @ coef_ == K @ dual_coef_
```

and for test samples:

```text
Xtest_c @ coef_ == Ktest @ dual_coef_
```

## Test 5: Block Scaling

Use identity plus a high-gain explicit operator such as `10 * I`.

Checks:

- `block_scaling="rms"` gives finite scales;
- scaled duplicate blocks do not dominate only by gain;
- diagnostics expose block scales.

## Test 6: Fold-Local CV No Leakage

Use a spy operator or spy scaler.

Checks:

- each fold fit sees only training rows;
- validation rows are not observed during `fit`;
- block scales are computed only from training folds;
- `x_mean` and `y_mean` are fold-local;
- implementation does not slice a globally centered full kernel.

## Test 7: Global Selection

Checks:

- identity-only bank selects identity;
- selected operator index is valid;
- selected alpha is in `alphas_`;
- `operator_scores_` contains every candidate;
- predictions are finite.

## Test 8: Operator Cloning

Use a mutable custom operator.

Checks:

- repeated estimator fits start from fresh state;
- CV folds do not share fitted operator instances;
- final refit does not reuse a fold-fitted instance.

## Test 9: Active Superblock

Use duplicated or near-duplicated operators.

Checks:

- active bank size is `<= active_top_m`;
- identity is retained when present;
- duplicates are pruned under a strict diversity threshold;
- diagnostics include active names, scores, and pruned count.

## Test 10: Multi-Output Regression

Checks:

- `coef_.shape == (p, 2)`;
- `intercept_.shape == (2,)`;
- `predict(X).shape == (n, 2)`;
- dual solver handles `Y` as a matrix.

## Test 11: Branch Anti-Leakage

Only after branch phase starts:

- MSC reference is fitted on train folds only;
- branch scales are fitted on train folds only;
- nonlinear branch model marks `coef_available=False`.

## Minimum Command

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge pytest bench/AOM_v0/Ridge/tests -q
```

