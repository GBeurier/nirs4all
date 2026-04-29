# Implementation Plan

## Scope Boundary

Implementation lives in `bench/AOM_v0`. Do not modify the production
`nirs4all` package during this one-shot run. Existing package code is read-only
reference material unless a later explicit user request asks for porting.

## Package Layout

Create this exact layout:

```text
bench/AOM_v0/
  aompls/
    __init__.py
    operators.py
    banks.py
    preprocessing.py
    centering.py
    nipals.py
    simpls.py
    scorers.py
    selection.py
    estimators.py
    classification.py
    torch_backend.py
    synthetic.py
    metrics.py
    diagnostics.py
  benchmarks/
    build_cohorts.py
    run_aompls_benchmark.py
    summarize_results.py
    run_smoke_benchmark.py
  tests/
    test_operators.py
    test_nipals.py
    test_simpls.py
    test_estimators.py
    test_selection.py
    test_classification.py
    test_torch_parity.py
    test_benchmark_schema.py
  docs/
    AOMPLS_IMPLEMENTATION_LOG.md
    AOMPLS_API.md
    AOMPLS_VALIDATION.md
    CODEX_REVIEWS.md
  publication/
    manuscript/
      main.tex
      references.bib
    supplement/
    figures/
    tables/
    scripts/
```

## Public API

Implement sklearn-like classes:

```python
AOMPLSRegressor(
    n_components="auto",
    max_components=25,
    engine="simpls_covariance",
    selection="global",
    criterion="cv",
    operator_bank="compact",
    orthogonalization="auto",
    center=True,
    scale=False,
    cv=5,
    random_state=0,
    backend="numpy",
)

POPPLSRegressor(... same parameters ..., selection="per_component")

AOMPLSDAClassifier(... same core parameters ...)
POPPLSDAClassifier(... same core parameters ..., selection="per_component")
```

Required methods:

- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)` for classifiers
- `transform(X)`
- `fit_transform(X, y)`
- `score(X, y)`
- `get_selected_operators()`
- `get_diagnostics()`
- `get_params()`
- `set_params()`

Required attributes:

- `x_mean_`, `y_mean_`
- `x_weights_`
- `x_effective_weights_`
- `x_loadings_`
- `y_loadings_`
- `x_scores_`
- `rotations_`
- `coef_`
- `intercept_`
- `selected_operators_`
- `selected_operator_indices_`
- `operator_scores_`
- `n_components_`
- `engine_`
- `selection_`
- `criterion_`
- `orthogonalization_`
- `diagnostics_`

## Engines to Implement

Mandatory numpy engines:

- `pls_standard`
- `nipals_materialized`
- `nipals_adjoint`
- `simpls_materialized`
- `simpls_covariance`
- `superblock_simpls`

Mandatory torch engines:

- `nipals_adjoint`
- `simpls_covariance`
- `superblock_simpls`

Torch implementations must run on CPU when CUDA is unavailable. Tests must check
numpy/torch parity with identity-only and small explicit operators.

## Operators

Base protocol:

```python
class LinearSpectralOperator:
    name: str
    def fit(self, X, y=None): ...
    def transform(self, X): ...
    def apply_cov(self, S): ...
    def adjoint_vec(self, v): ...
    def matrix(self, p): ...
    def is_linear_at_apply(self): return True
    def fitted_parameters(self): return {}
```

Required strict-linear operators:

- identity
- Savitzky-Golay smoothing
- Savitzky-Golay derivative order 1 and 2
- finite-difference derivative
- detrend projection degree 0, 1, 2
- Norris-Williams derivative
- Whittaker smoother if robust with scipy sparse solve
- composition/chain operator

SNV, MSC, EMSC, OSC, and EPO are not strict fixed operators by default. They may
exist only as fitted preprocessors or experimental fitted linear-at-apply
wrappers, and they must be excluded from covariance-SIMPLS strict equivalence
tests unless their matrix identity is proven.

## Defaults

Use deterministic defaults:

- `random_state=0`
- `center=True`
- `scale=False`
- `max_components=min(25, n_samples - 1, n_features)`
- `n_components="auto"`
- `engine="simpls_covariance"` for regression
- `selection="global"` for AOM, `"per_component"` for POP
- `criterion="cv"` for benchmark/full mode
- `criterion="covariance"` only for smoke or prescreen
- `orthogonalization="transformed"` when `selection in {"none", "global"}` and
  one fixed operator is used
- `orthogonalization="original"` when `selection="per_component"`

## Phase Gates

1. Inspection and implementation log.
2. Operator abstraction and unit tests.
3. Reference materialized PLS engines.
4. Covariance/adjoint fast engines.
5. Unified estimators.
6. Selection criteria and anti-leakage CV.
7. Classification and probability calibration.
8. Torch parity.
9. Benchmarks.
10. Paper and publication repository.

Do not advance a phase unless the phase tests pass. If a full benchmark is too
long, run smoke benchmarks and leave the full command resumable.

## Test Commands

Minimum local checks:

```bash
PYTHONPATH=bench/AOM_v0 pytest bench/AOM_v0/tests -q
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/run_smoke_benchmark.py
```

Optional package comparison:

```bash
pytest tests/unit/operators/models/test_aom_pls.py tests/unit/operators/models/test_pop_pls.py -q
```
