# AOM-Ridge Plan Review And Corrections

## Verdict

The proposed plan is correct in its main direction: AOM-Ridge should be built
as a dual/kernel Ridge method, not as a direct reuse of the PLS shortcut
`A X^T Y`. Ridge depends on the geometry of `X`, so the core objects are:

```text
K_b = X A_b^T A_b X^T
K_super = sum_b s_b^2 X A_b^T A_b X^T
```

The best first implementation is:

```text
AOM-Ridge-global      : hard CV over (operator, alpha), useful baseline
AOM-Ridge-superblock : all strict-linear operators in one dual Ridge model
```

`AOM-Ridge-superblock` should be treated as the primary model.

## Required Corrections

### 1. Include identity exactly once

The AOM-PLS banks already include `identity` as the first operator. Do not add
a second raw block. Implementation rule:

```text
Use the resolved operator bank as-is.
If identity is absent, prepend it.
Do not add a duplicate raw block.
```

### 2. CV kernels must be fold-local

Do not compute one globally centered kernel and slice it for CV. That leaks
validation data through `x_mean`, block RMS scales, fitted operator state, and
branch references.

For every fold:

```text
x_mean_f = mean(X_train)
y_mean_f = mean(Y_train)
Xtr_c = X_train - x_mean_f
Xva_c = X_valid - x_mean_f
fit operators/scales on Xtr_c, Ytr_c only
Ktr = Xtr_c M_f Xtr_c^T
Kva = Xva_c M_f Xtr_c^T
```

### 3. Superblock coefficient scaling is `s_b^2`

If the explicit block is:

```text
Z_b = s_b X A_b^T
```

then:

```text
M = sum_b s_b^2 A_b^T A_b
beta = M X^T C
```

Tests must compare the dual implementation against explicit concatenated
Ridge.

### 4. The prior statement needs a range condition

For invertible `A`, Ridge on `X A^T` is equivalent to original-space Ridge
with penalty:

```text
alpha ||A^{-T} beta||_2^2
```

For non-invertible operators such as derivatives and detrend projections, the
proper statement is:

```text
min_beta ||Y - X beta||_F^2 + alpha beta^T M^+ beta
subject to beta in range(M)
```

where:

```text
M = sum_b s_b^2 A_b^T A_b
```

If identity is included with nonzero scale, `M` is positive definite and the
range caveat disappears.

### 5. Active scoring should be normalized

The score `Y^T K_b Y = ||A_b X^T Y||_F^2` is valid but scale-sensitive. Use:

```text
score_b = ||s_b A_b X^T Y||_F^2
```

or kernel-target alignment:

```text
align_b = <K_b, Y Y^T>_F / (||K_b||_F ||Y Y^T||_F)
```

### 6. Nonlinear branches are a later phase

SNV, MSC, EMSC, OSC, and ASLS can be included by materializing branches:

```text
Z_b = T_b(X)
K = sum_b s_b^2 Z_b Z_b^T
```

They cannot use `M = sum_b A_b^T A_b` unless represented by a fixed linear map
learned strictly inside the training fold. They also do not yield a clean
original-spectrum coefficient.

### 7. Operator instances must be cloned per fit/fold

The existing AOM-PLS operators store `p` and may cache matrices. Custom banks
may contain fitted state. Do not reuse mutable operator instances across CV
folds.

Rule:

```text
resolve bank -> clone fresh operators for every estimator fit
CV fold      -> clone/fresh-fit operators inside the fold
final refit  -> clone/fresh-fit operators on the full calibration set
```

Use bank factories for named banks; use `sklearn.base.clone` or
`copy.deepcopy` for custom banks.

### 8. Numerical solver requirements

The final implementation should:

- symmetrize train kernels with `0.5 * (K + K.T)`;
- require `alpha > 0`;
- use Cholesky for normal PSD cases;
- fall back to eigenvalue solve or add a tiny jitter when Cholesky fails;
- clip tiny negative eigenvalues only as a documented numerical safeguard;
- select `alpha` on a scale relative to `trace(K) / n`.

### 9. POP-Ridge is not phase 1

Ridge has no latent components, so POP-Ridge is not a natural analogue of
POP-PLS. A sequential residual model would be closer to boosting over operator
views. Keep it out of the first implementation.

## Recommended Implementation Order

1. Strict-linear kernel utilities and dual Ridge solver.
2. `AOMRidgeRegressor(selection="superblock")` with `operator_bank="compact"`.
3. Equivalence tests versus sklearn Ridge on explicit concatenated blocks.
4. `selection="global"` with fold-local CV over `(operator, alpha)`.
5. Active-superblock prescreening with normalized scores.
6. Benchmarks against Ridge raw, SNV-Ridge, MSC-Ridge, and AOM-PLS variants.
7. Optional branch-kernel Ridge and OOF Ridge experts.

