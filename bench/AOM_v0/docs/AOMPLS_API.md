# AOM_v0 — AOM-PLS API Reference

This document is the API reference for the AOM_v0 reference implementation of
Operator-Adaptive Partial Least Squares. It targets the package living at
`bench/AOM_v0/aompls/` and is independent from the production `nirs4all`
library.

The framework provides standard PLS, **AOM** (Adaptive Operator-Mixture, one
operator selected globally), **POP** (Per-Operator-Per-component, one
operator selected per latent component), soft mixtures, and superblock
baselines, all dispatched through a single estimator family. Both NIPALS and
SIMPLS engines are available, with materialized references and fast
covariance / adjoint variants.

The mathematical convention is the cross-covariance identity
`(X A^T)^T Y = A X^T Y`, which lets the fast engines evaluate operator
candidates directly in covariance space without materializing the transformed
spectra.

---

## 1. Quick Start

### Regression: AOM-PLS and POP-PLS

```python
import numpy as np
from aompls.estimators import AOMPLSRegressor, POPPLSRegressor
from aompls.synthetic import make_regression

ds = make_regression(n_train=80, n_test=40, p=200, random_state=0)

# AOM-PLS: one operator picked for the whole model.
aom = AOMPLSRegressor(max_components=15, criterion="cv", cv=5)
aom.fit(ds.X_train, ds.y_train)
print("AOM RMSEP:", np.sqrt(np.mean((aom.predict(ds.X_test) - ds.y_test) ** 2)))
print("AOM operator:", aom.get_selected_operators()[0])

# POP-PLS: per-component operator selection.
pop = POPPLSRegressor(max_components=15, criterion="cv", cv=5)
pop.fit(ds.X_train, ds.y_train)
print("POP RMSEP:", np.sqrt(np.mean((pop.predict(ds.X_test) - ds.y_test) ** 2)))
print("POP sequence:", pop.get_selected_operators())
```

### Classification: AOM-PLS-DA and POP-PLS-DA

```python
from aompls.classification import AOMPLSDAClassifier, POPPLSDAClassifier
from aompls.synthetic import make_classification
from aompls.metrics import balanced_accuracy, log_loss

ds = make_classification(n_train=120, n_test=60, p=200, n_classes=3, random_state=4)

clf = AOMPLSDAClassifier(max_components=10, criterion="cv", cv=5)
clf.fit(ds.X_train, ds.y_train)
proba = clf.predict_proba(ds.X_test)
print("Balanced accuracy:", balanced_accuracy(ds.y_test, clf.predict(ds.X_test)))
print("Log loss:", log_loss(ds.y_test, proba, classes=clf.classes_))

pop_clf = POPPLSDAClassifier(max_components=10, criterion="cv", cv=5)
pop_clf.fit(ds.X_train, ds.y_train)
print("POP-DA balanced accuracy:",
      balanced_accuracy(ds.y_test, pop_clf.predict(ds.X_test)))
```

All four estimators follow the sklearn protocol (`fit`, `predict`,
`transform`, `score`, `get_params`, `set_params`) and add the diagnostic
helpers `get_selected_operators()` and `get_diagnostics()`.

---

## 2. Estimator Constructor Parameters

All four estimators (`AOMPLSRegressor`, `POPPLSRegressor`,
`AOMPLSDAClassifier`, `POPPLSDAClassifier`) share the same constructor
signature and parameter semantics. They are defined in
`bench/AOM_v0/aompls/estimators.py` and `bench/AOM_v0/aompls/classification.py`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | `int \| "auto"` | `"auto"` | Number of latent components. `"auto"` enables prefix scoring: every prefix `k = 1..max_components` is evaluated and the optimum is selected (for non-`covariance` criteria). An explicit `int` clamps to `min(n_components, max_components)` and disables prefix scoring. |
| `max_components` | `int` | `25` | Hard upper bound on components. Always clamped to `min(max_components, n - 1, p)` at fit time. |
| `engine` | `str` | `"simpls_covariance"` | Computation engine. See [Engines](#5-engines). |
| `selection` | `str` | `"global"` (AOM, AOM-DA) / `"per_component"` (POP, POP-DA) | Selection policy. One of `"none"`, `"global"`, `"per_component"`, `"soft"`, `"superblock"`. |
| `criterion` | `str` | `"cv"` | Selection criterion. One of `"covariance"`, `"cv"`, `"approx_press"`, `"hybrid"`, `"holdout"`. |
| `operator_bank` | `str \| Sequence[LinearSpectralOperator]` | `"compact"` | Either a preset name (`"identity"`, `"compact"`, `"default"`, `"extended"`) or a custom list of `LinearSpectralOperator` instances. |
| `orthogonalization` | `str` | `"auto"` | One of `"transformed"`, `"original"`, `"auto"`. See [resolution rule](#orthogonalization-resolution). |
| `center` | `bool` | `True` | Center `X` and `Y` before fitting. |
| `scale` | `bool` | `False` | Reserved for future use; centering only is currently applied. |
| `cv` | `int` | `5` | Number of CV folds for `criterion="cv"` and `criterion="hybrid"`. |
| `random_state` | `int` | `0` | Seed used by `KFold` / `StratifiedKFold`, holdout splits, and the `LogisticRegression` calibrator (classification). |
| `backend` | `str` | `"numpy"` | Recorded in diagnostics. The torch engines are dispatched explicitly via `aompls.torch_backend`; setting `"torch"` here only labels the run. |

### Orthogonalization resolution

`orthogonalization="auto"` resolves at the start of `fit` as follows:

| `selection` | Resolved `orthogonalization` |
|-------------|------------------------------|
| `"none"` | `"transformed"` |
| `"global"` | `"transformed"` |
| `"per_component"` | `"original"` |
| `"soft"` | `"original"` |
| `"superblock"` | `"original"` |

`"transformed"` deflates the SIMPLS basis in the operator's transformed
space; this is only well-defined for a single fixed operator (AOM /
identity-only), and the engines explicitly raise `ValueError` when a
multi-operator sequence is combined with `orthogonalization="transformed"`.
`"original"` deflates in the original spectrum space using a Gram-Schmidt
basis on the loadings, and is required when the operator changes per
component (POP, soft mixture, superblock).

### Defaults summary

| Estimator | Default `selection` | Default `orthogonalization` (after auto resolution) |
|-----------|---------------------|------------------------------------------------------|
| `AOMPLSRegressor` | `"global"` | `"transformed"` |
| `POPPLSRegressor` | `"per_component"` | `"original"` |
| `AOMPLSDAClassifier` | `"global"` | `"transformed"` |
| `POPPLSDAClassifier` | `"per_component"` | `"original"` |

---

## 3. Public Attributes After Fit

Both regressors and classifiers expose the following sklearn-style
attributes once `fit` has run.

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `x_mean_` | `(p,)` | Column mean of `X` used for centering. |
| `y_mean_` | `(q,)` | Column mean of `Y`. For classifiers `Y` is the class-balanced one-hot encoding. |
| `x_weights_` | `(p, K)` | Original-space effective weights `Z = A^T R`. Equals `x_effective_weights_`. |
| `x_effective_weights_` | `(p, K)` | Same as above; named for emphasis on the original-space role. Used by `transform`. |
| `x_loadings_` | `(p, K)` | `P_a = X^T t_a / (t_a^T t_a)`. |
| `y_loadings_` | `(q, K)` | `Q_a = Y^T t_a / (t_a^T t_a)`. |
| `x_scores_` | `(n, K)` | Training latent scores `T = X Z`. |
| `rotations_` | `(p, K)` | Alias of `Z`, exposed for sklearn-API compatibility. |
| `coef_` | `(p, q)` | Regression coefficient matrix `B = Z (P^T Z)^+ Q^T`. |
| `intercept_` | `(q,)` | `y_mean - x_mean @ coef_`. |
| `selected_operators_` | `List[str]` | Operator names, length `K`. |
| `selected_operator_indices_` | `List[int]` | Operator indices into the bank, length `K`. |
| `operator_scores_` | `dict` | Per-candidate selection scores. For AOM keyed by operator name; for POP keyed by `f"component_{a+1}"` -> sub-dict per operator. |
| `n_components_` | `int` | Number of components actually retained. |
| `engine_`, `selection_`, `criterion_`, `orthogonalization_` | `str` | Resolved configuration. |
| `diagnostics_` | `RunDiagnostics` | Structured run record, see [section 9](#9-diagnostics). |

Classifiers additionally expose:

| Attribute | Description |
|-----------|-------------|
| `classes_` | `np.unique(y_train)`. |
| `_calibrator` | Fitted `LogisticRegression(class_weight="balanced", max_iter=2000)` on the latent training scores. |
| `_calibrator_kind` | `"logistic"` (primary) or `"temperature"` (fallback, golden-section temperature scaling on training scores). |

---

## 4. `LinearSpectralOperator` Protocol

Defined in `bench/AOM_v0/aompls/operators.py`. A spectral operator
`A in R^{p x p}` acts on row spectra as `X_b = X A^T`. The protocol class is
not a Python `Protocol` in the typing sense; subclasses inherit from it and
override the implementation hooks.

### Required public methods

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `fit(X=None, y=None)` | `(LinearSpectralOperator)` | Bind to feature dimensionality. Strict-linear operators record `p = X.shape[1]` only. Supervised linear operators (none in v0) may learn parameters here. |
| `transform(X)` | `(n, p) -> (n, p)` | Apply to row spectra: `X_b = X A^T`. |
| `apply_cov(S)` | `(p,) or (p, q) -> same` | Apply to the cross-covariance: `A S`. Used by all covariance-space engines. |
| `adjoint_vec(v)` | `(p,) or (p, k) -> same` | Apply the adjoint: `A^T v`. Used to map transformed-space directions back to the original space. |
| `matrix(p=None)` | `() -> (p, p)` | Return the explicit dense matrix of `A`. Cached after first call. |
| `is_linear_at_apply()` | `() -> bool` | Returns `is_strict_linear` (default `True`). Soft scatter corrections (SNV, MSC) report `False` and live outside the bank. |
| `fitted_parameters()` | `() -> dict` | Optional. Default empty. |

### Strict-linearity guarantee

Every concrete operator in v0 satisfies:

1. Linearity: `A(a x + b y) = a A x + b A y`.
2. Adjoint identity: `<A x, y> = <x, A^T y>` for all `x, y in R^p`.
3. Matrix consistency: `transform(X) = X @ matrix(p).T`, `apply_cov(S) = matrix(p) @ S`, `adjoint_vec(v) = matrix(p).T @ v`.
4. Cross-covariance identity: `transform(X).T @ Y = apply_cov(X.T @ Y)`.

These are exhaustively checked in `tests/test_operators.py`
(`test_linearity`, `test_adjoint_identity`, `test_covariance_identity`,
`test_matrix_consistency`).

---

## 5. Available Operators

All operators live in `bench/AOM_v0/aompls/operators.py` and use **zero-padded
boundaries** so that the resulting map is strictly linear in the input (no
data-dependent boundary modes).

| Class | Constructor | Notes |
|-------|-------------|-------|
| `IdentityOperator` | `IdentityOperator(p=None)` | Always present in every default bank; used as the reference for "AOM with identity-only bank reduces to standard PLS" tests. |
| `SavitzkyGolayOperator` | `SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=None)` | Centered SG smoothing or derivative. `window_length` must be odd `>= 3`; `0 <= deriv <= polyorder`. Coefficients computed via least-squares on `np.vander` with explicit pseudo-inverse. |
| `FiniteDifferenceOperator` | `FiniteDifferenceOperator(order=1, p=None)` | Centered first or second difference. `order in {1, 2}`. |
| `DetrendProjectionOperator` | `DetrendProjectionOperator(degree=1, p=None)` | Symmetric orthogonal projector onto the polynomial complement: `A = I - Q Q^T` with `Q` from QR of `[1, t, ..., t^d]` evenly spaced on `[-1, 1]`. |
| `NorrisWilliamsOperator` | `NorrisWilliamsOperator(gap=5, smoothing=5, order=1, p=None)` | Composition of moving-average smoothing (`smoothing` odd `>= 1`) and gap derivative (`gap >= 1`). `order in {1, 2}`. |
| `WhittakerOperator` | `WhittakerOperator(lam=1e3, p=None)` | Whittaker smoother `(I + lam D^T D)^{-1}` solved via banded LU (`scipy.linalg.solve_banded`). Symmetric: `A = A^T`. Falls back to dense for `p < 4`. |
| `ComposedOperator` | `ComposedOperator(operators, name=None)` | `transform` calls each child in left-to-right order. Equivalent to `A = A_K ... A_2 A_1` so that `transform(X) = X (A_K ... A_1)^T`. The matrix is built as `A_K @ ... @ A_1` and the adjoint reverses the chain. |
| `ExplicitMatrixOperator` | `ExplicitMatrixOperator(matrix, name="explicit")` | Used in tests to verify the protocol on arbitrary linear maps. |

### Strictly linear, no exceptions

Every operator above satisfies the four invariants in [section 4](#strict-linearity-guarantee). The
SG, NW, and FD operators implement `_apply_cov_impl` and `_adjoint_vec_impl`
through cheap symmetric Toeplitz cross-correlations (`_xcorr_zero_pad`) so
that `apply_cov` and `adjoint_vec` never materialize the `p x p` dense
matrix in covariance-space engines.

---

## 6. Operator Bank Presets

Defined in `bench/AOM_v0/aompls/banks.py`. Every preset starts with the
identity operator so that AOM/POP can always reduce to standard PLS.

### `compact_bank(p=None)` — 9 operators

```
identity
sg_smooth_w11_p2
sg_smooth_w21_p3
sg_d1_w11_p2
sg_d1_w21_p3
sg_d2_w11_p2
detrend_d1
detrend_d2
fd_d1
```

This is the default bank used by the four estimators and by the smoke
benchmark. Small enough to keep selection signal strong on small NIRS
datasets.

### `default_bank(p=None)` — 13 operators

`compact_bank` plus:

```
fd_d2
nw_g5_s5_d1
nw_g11_s5_d1
compose(detrend_d2 | sg_d1_w11_p2)   # named "detrend2_then_sg_d1"
```

### `extended_bank(p=None)` — 18 operators

`default_bank` plus:

```
whittaker_l10
whittaker_l1000
whittaker_l100000
sg_d2_w31_p3
nw_g5_s5_d2
```

### `bank_by_name(name, p=None)`

Resolves `"identity"` (singleton), `"compact"`, `"default"`, or `"extended"`
as in the table above. The estimator's `operator_bank` parameter accepts
either one of these strings or an explicit list of `LinearSpectralOperator`
instances.

`fit_bank(bank, X, y=None)` binds every operator to `X.shape[1]` in one
pass. The estimator does this automatically inside `fit`.

---

## 7. SNV / MSC are NOT in the bank

`StandardNormalVariate` and `MultiplicativeScatterCorrection` are defined in
`bench/AOM_v0/aompls/preprocessing.py` and **must be applied as standalone
preprocessors upstream of AOM/POP**, never inside the operator bank.

Both are *not strictly linear* in the input:

- **SNV** divides each row by its own per-row standard deviation. The
  per-row scale depends on the row data, breaking linearity in `X`.
- **MSC** fits a per-row affine `(slope, intercept)` to a reference
  spectrum. The reference is data-dependent and the per-row affine
  rescaling is not a single matrix `A` shared across samples.

If you need scatter correction, call `StandardNormalVariate().fit_transform(X)`
or `MultiplicativeScatterCorrection().fit_transform(X)` first, then pass the
corrected matrix to `AOMPLSRegressor.fit`. The bank presets are guaranteed
to contain only operators that satisfy the cross-covariance identity, so
that the covariance-space engines remain mathematically equivalent to their
materialized references.

---

## 8. Engines

Two engine families are implemented (NIPALS in `nipals.py`, SIMPLS in
`simpls.py`), each in three flavours: standard, materialized through a fixed
operator, and per-component. The SIMPLS family additionally provides a
covariance-space variant that is the recommended default. A NumPy reference
is provided for every engine and a PyTorch parity engine is provided for the
fast variants.

| Engine string | Module / function | Selection compatibility | Notes |
|---------------|-------------------|--------------------------|-------|
| `"pls_standard"` | `nipals_pls_standard` | `selection="none"` only | Identity-only sanity reference. Routes through `nipals_materialized_fixed(X, Y, IdentityOperator(), K)`. |
| `"nipals_materialized"` | `nipals_materialized_fixed` (single op) / `nipals_materialized_per_component` (sequence) | All | Slow reference. Builds `X_b = X A^T` explicitly. |
| `"nipals_adjoint"` | `nipals_adjoint` | All | Fast NIPALS using `S = X^T Y`, `r = u_1(S)`, transformed direction `A s` and weight `z = A^T r / ||A s||`. No materialisation of `X A^T`. |
| `"simpls_materialized"` | `simpls_materialized_fixed` (single op) / `simpls_materialized_per_component` (sequence) | All | de Jong's SIMPLS on `Xb = X A^T`. Slow reference with original-space loadings recompute. |
| `"simpls_covariance"` | `simpls_covariance` | All | **Default**. Computes `S = X^T Y` once, applies `S_b = A_b S` per component, deflates with original-space Gram-Schmidt basis. |

### Torch backend (`aompls/torch_backend.py`)

Three engines are mirrored in PyTorch:

| Function | Equivalent NumPy engine |
|----------|--------------------------|
| `nipals_adjoint_torch` | `nipals_adjoint` |
| `simpls_covariance_torch` | `simpls_covariance` |
| `superblock_simpls_torch` | `superblock_simpls` |

Behaviour:

- Accepts NumPy arrays and returns NumPy arrays so callers stay
  backend-agnostic.
- Uses `torch.linalg.svd` and `torch.linalg.norm` for the heavy reductions.
- Routes the cheap operator calls (`apply_cov`, `adjoint_vec`, `transform`)
  through the operator's NumPy implementation, transferring small tensors
  back and forth. This is intentional: it avoids re-implementing every
  Toeplitz convolution on the GPU and keeps the numerical reference
  identical to the NumPy backend.
- `device=None` resolves to CUDA when `torch.cuda.is_available()` and CPU
  otherwise. CPU fallback is exercised by the parity tests.
- `dtype` accepts `"float64"` (default) or `"float32"`.

### When to prefer which engine

| Situation | Recommended engine |
|-----------|--------------------|
| Default (regression / classification) | `simpls_covariance` (NumPy) |
| Sanity reference / unit testing | `pls_standard` for identity-only, `nipals_materialized` or `simpls_materialized` for AOM with a single fixed operator |
| Auditing POP runs against a slow reference | `simpls_materialized` with the fixed sequence emitted by POP |
| GPU acceleration on large `p` (no per-row metadata) | `simpls_covariance_torch` via `aompls.torch_backend` |
| Wide feature concatenation experiments | `superblock_simpls` (selection `"superblock"`) |

The materialized engines are kept as references; the covariance engine is
the production code path because the cross-covariance identity guarantees
numerical equivalence (verified by `test_simpls_per_component_materialized_vs_covariance`).

---

## 9. Selection Policies

Defined in `bench/AOM_v0/aompls/selection.py`. The dispatcher is
`select(Xc, yc, operators, engine, selection, n_components_max, criterion,
orthogonalization, auto_prefix)` and it returns a `SelectionResult` with the
final `NIPALSResult`, the operator sequence, the per-candidate score table,
and a diagnostics dict.

| Policy | Meaning |
|--------|---------|
| `"none"` | No selection. The bank must be a singleton; the engine runs on that single operator. |
| `"global"` | **AOM**. Each operator is scored at the requested prefix(es). The argmin operator is committed for every component. |
| `"per_component"` | **POP**. Greedy operator pursuit: at component `a`, every candidate `(committed prefix + [b])` is scored and the argmin is appended. |
| `"soft"` | Experimental convex mixture. Per component, candidate covariance scores are converted to weights via softmax (or sparsemax via the internal helper), and the convex combination of operators is applied. Often degenerates to hard selection on covariance objectives. |
| `"superblock"` | Concatenate operator views into a wide block and run standard SIMPLS. The diagnostics include per-operator coefficient-norm "group importance". |

### Criteria

Defined in `bench/AOM_v0/aompls/scorers.py` via `CriterionConfig`.

| `criterion` | Score (lower is better) | Notes |
|-------------|--------------------------|-------|
| `"covariance"` | `-||A_b S||` (Frobenius for `q > 1`) | Cheap proxy. Used for smoke benchmarks and AOM prescreening. |
| `"cv"` | K-fold RMSE (regression) or balanced log-loss (classification) | Default. Re-centers per fold. |
| `"approx_press"` | Approximate PRESS with leverage correction `h = diag(U U^T)` | Single full-fit pass scores every prefix. |
| `"hybrid"` | Covariance prescreening (`prescreen_top_m=5`) + CV refinement on the survivors | Useful when the bank is large. |
| `"holdout"` | Single train/val split RMSE | Legacy debug only. Never the default. |

`CriterionConfig` fields: `kind`, `cv` (default `5`), `prescreen_top_m`
(default `5`), `random_state`, `task` (`"regression"` or
`"classification"`), `holdout_fraction` (default `0.2`).

### Auto prefix

When `n_components="auto"`, the estimator sets `auto_prefix=True` and
`select` evaluates every prefix `k = 1..max_components` for non-`covariance`
criteria. The chosen `k` is the prefix with the lowest score; the resulting
model has `n_components_ = k`. For `covariance`, the prefix scorer is a no-op
and the engine runs at the requested `n_components_max`.

---

## 10. Diagnostics

Every fit produces a `RunDiagnostics` instance accessible via
`estimator.diagnostics_` and `estimator.get_diagnostics()` (the latter
returns the dataclass as a plain `dict`). The dataclass is defined in
`bench/AOM_v0/aompls/diagnostics.py`.

| Key | Type | Description |
|-----|------|-------------|
| `engine` | `str` | The engine string passed to the constructor. |
| `selection` | `str` | The selection policy. |
| `criterion` | `str` | The criterion used for selection. |
| `orthogonalization` | `str` | Resolved orthogonalization (`"transformed"` or `"original"`). |
| `operator_bank` | `str` | Bank preset name, or `"custom"` if a list was passed. |
| `selected_operator_indices` | `List[int]` | Bank indices, length `K`. |
| `selected_operator_names` | `List[str]` | Bank names, length `K`. |
| `operator_scores` | `dict` | For AOM: `{name: score}` per candidate. For POP: `{f"component_{a+1}": {name: score}}`. |
| `n_components_selected` | `int` | `K`. |
| `max_components` | `int` | Effective cap after `min(max_components, n - 1, p)`. |
| `fit_time_s` | `float` | Wall-clock fit time. |
| `predict_time_s` | `float` | Wall-clock predict time on the most recent `predict` call. |
| `backend` | `str` | `"numpy"` or `"torch"` (label only). |
| `extras` | `dict` | Engine- and selection-specific extras: `score_curve`, `candidates`, `operator_sequence`, `weights` (soft), `groups` / `group_importance` (superblock), and `task="classification"` for classifiers. |

The helper `aompls.diagnostics.operator_sequence_string(diag, max_len=6)`
formats the selected sequence as a compact human-readable string (e.g.
`"sg_d1_w11_p2 x 6"` for AOM or `"identity | sg_d1_w11_p2 | detrend_d1 | fd_d1"`
for POP).

---

## 11. Other Public Entry Points

### `aompls.synthetic`

- `make_regression(n_train=80, n_test=40, p=200, n_targets=1, noise=0.05, random_state=0)` — deterministic synthetic regression dataset with smooth bands, baseline drift, and multiplicative scatter.
- `make_classification(n_train=90, n_test=60, p=180, n_classes=3, noise=0.05, random_state=0)` — separable multi-class dataset.
- `small_pls1_dataset(p=24, n=30, K=3, noise=0.02, random_state=0)` — tiny PLS1 dataset used in `test_small_pls1_recovers_known_factors`.

### `aompls.metrics`

`rmse`, `mae`, `r2`, `balanced_accuracy`, `macro_f1`, `log_loss`,
`brier_score_binary`, `expected_calibration_error`.

### `aompls.scorers`

Low-level scorer functions: `cv_score_regression`, `cv_score_classification`,
`approx_press_regression`, `holdout_score_regression`, `covariance_score`.
Useful when wiring custom selection policies on top of the same engines.

### `aompls.simpls.superblock_simpls`

Concatenates `[X A_b^T for b in operators]` into a wide matrix and runs
`simpls_standard` on top. Returns the `NIPALSResult` and the per-column
group membership vector.

---

## 12. Related Documents

- `bench/AOM_v0/docs/AOMPLS_MATH_SPEC.md` — formal mathematical specification.
- `bench/AOM_v0/docs/AOMPLS_VALIDATION.md` — test inventory, mathematical equivalences, and benchmark evidence.
- `bench/AOM_v0/docs/AOMPLS_IMPLEMENTATION_LOG.md` — phase-by-phase implementation notes.
- `bench/AOM_v0/docs/BENCHMARK_PROTOCOL.md` — benchmark cohort and column schema.

---

## Active Superblock and Operator Explorer (added 2026-04-28)

The framework gained two additional selection modes after the initial release:

### `selection="active_superblock"`

Concatenates a *small, fold-local, diverse* subset of the operator bank into
a single wide matrix and runs SIMPLS on top, with per-block Frobenius scaling
so high-gain operators do not dominate by amplitude. Coefficients are mapped
back to the original `(p, q)` feature space:

```text
Z_orig[:, a] = sum_b alpha_b A_b^T Z_wide_b[:, a]
B_orig = Z_orig (P_orig^T Z_orig)^+ Q_orig^T
```

Estimator usage:

```python
AOMPLSRegressor(
    selection="active_superblock",
    operator_bank="default",          # the active subset is selected internally
    max_components=12,
    criterion="covariance",
)
```

Diagnostics (`get_diagnostics()`):

- `selection`: `"active_superblock"`
- `active_operator_indices`, `active_operator_names`, `active_operator_scores`
- `block_weights` (Frobenius-normalised per active block)
- `group_importance` (norm of original-space coefficients per active block)
- `active_top_m`, `diversity_threshold`, `block_scaling`
- `original_feature_space`: `True`

Default knobs (hard-coded for stability; no estimator parameter yet):

```text
active_top_m         = min(20, len(operators))
diversity_threshold  = 0.98 (response cosine)
block_scaling        = "frobenius"
```

### `selection="superblock"` (legacy, now original-space)

The legacy raw superblock mode now also produces `(p, q)` coefficients via
the same mapping. It is the no-screening, no-scaling baseline against which
`active_superblock` is compared in the paper.

### Operator explorer (`AOM-explorer-*` benchmark variant)

The explorer lives in `aompls/operator_explorer.py`. It implements a
deterministic beam search over operator chains in covariance space:

```python
from aompls.operator_explorer import build_active_bank_from_training

active_bank = build_active_bank_from_training(
    X_train, y_train,
    max_degree=2,
    beam_width=24,
    final_top_m=20,
    cosine_threshold=0.98,
)
est = AOMPLSRegressor(operator_bank=active_bank, selection="global")
```

Primitive families (`aompls/operator_generation.py`):

- Savitzky-Golay scale-space (windows × polyorder × derivatives).
- Whittaker smoothers across a logarithmic `lambda` grid.
- Gaussian-derivative filters (sigma × order).
- Detrend projections.
- Finite differences (order 1, 2).
- Norris-Williams gap derivatives.
- Fixed integer-pixel shifts.

Grammar rules (`grammar_allows`) reject obviously redundant compositions:
no consecutive smoothers of the same family, no double-detrend, no double
derivative of the same order, identity not appended to a non-empty chain.

Canonicalisation (`canonicalize`) drops identity stages and collapses
detrend pairs. The chain signature is stable and used for deduplication.

Similarity tools (`aompls/operator_similarity.py`):

- `make_probe_basis(p)` — deterministic probe matrix for intrinsic
  similarity (Diracs, polynomials, Gaussian peaks, multi-frequency
  cosines, white noise).
- `response_cosine(a, b)` — absolute cosine similarity.
- `keep_top_diverse(items, top_m, cosine_threshold)` — score-then-prune.
- `prune_by_intrinsic_similarity(operators, p, threshold)` — offline
  removal of near-duplicate operators.

