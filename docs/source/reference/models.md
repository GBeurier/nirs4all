# Built-in Models Reference

nirs4all includes a comprehensive library of PLS variants and spectroscopy-specific models, all sklearn-compatible.

```{note}
Any sklearn estimator (`RandomForestRegressor`, `Ridge`, `SVR`, `GradientBoostingRegressor`, etc.) can be used directly as a model step. Deep learning models are available via `nirs4all.operators.models.tensorflow`, `nirs4all.operators.models.pytorch`, and `nirs4all.operators.models.jax` (lazy-loaded).
```

## Usage in Pipeline

```python
from nirs4all.operators.models import AOMPLSRegressor, AOMRidgeBlender, PLSDA
from nirs4all.operators.transforms import SNV
from sklearn.model_selection import ShuffleSplit

pipeline = [
    SNV(),
    ShuffleSplit(n_splits=5),
    {"model": AOMPLSRegressor(n_components="auto", operator_bank="compact")},
    {
        "model": AOMRidgeBlender(outer_cv=5, inner_cv=5),
        "train_params": {"use_pipeline_folds_for_aom": "required"},
    },
]
```

All models below are imported from `nirs4all.operators.models`.

---

## Adaptive PLS (Auto-Preprocessing)

These models automatically select the best preprocessing operator for each PLS component from a built-in operator bank.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `AOMPLSRegressor` | `n_components="auto"`, `operator_bank="compact"`, `criterion="cv"`, `cv=5` | Adaptive Operator-Mixture PLS -- auto-selects preprocessing from an operator bank |
| `AOMPLSClassifier` | `n_components="auto"`, `operator_bank="compact"`, `cv=5` | AOM-PLS for classification with probability calibration |
| `POPPLSRegressor` | `n_components=15`, `auto_select=True`, `bank=None` | Per-Operator-Per-component PLS -- selects a different operator per component via PRESS criterion |
| `POPPLSClassifier` | `n_components=15`, `auto_select=True`, `bank=None` | POP-PLS for classification with probability calibration |

AOM-PLS provides `default_operator_bank()` and `extended_operator_bank()` helper functions for customizing the preprocessing bank. POP-PLS provides `pop_pls_operator_bank()`.

---

## AOM-Ridge and FastAOM

These models extend Ridge/PLS-Ridge calibration with AOM operator banks and
variant aggregation.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `AOMRidgeRegressor` | `selection="global"`, `operator_bank="compact"`, `cv=5` | Single AOM-Ridge estimator with operator selection and alpha CV |
| `AOMRidgeAutoSelector` | `candidates=None`, `outer_cv=3`, `inner_cv=3`, `scoring="rmse_mean"` | Runs outer CV over AOM-Ridge variants and refits the best one |
| `AOMRidgeBlender` | `candidates=None`, `outer_cv=3`, `inner_cv=3`, `regularizer=0.01` | Convex non-negative blend of AOM-Ridge variants; strongest general AOM-Ridge recipe |
| `FastAOMPLSRidge` | `config=FastAOMConfig(...)` | Fast screened chain-search family for PLS/Ridge calibration |

Split-aware usage:

```python
from sklearn.model_selection import GroupKFold
from nirs4all.operators.models import AOMRidgeBlender

pipeline = [
    {"split": GroupKFold(n_splits=5), "group_by": "batch_id"},
    {
        "model": AOMRidgeBlender(outer_cv=5, inner_cv=5, random_state=42),
        "train_params": {"use_pipeline_folds_for_aom": "required"},
    },
]
```

When enabled, nirs4all forwards the pipeline folds to `cv`, `cv_splitter`,
`outer_cv`, `inner_cv`, or `external_folds` depending on the estimator API.

See {doc}`../user_guide/models/aom_models` for the full user guide.

---

## Standard PLS

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `PLSDA` | `n_components=5` | PLS Discriminant Analysis classifier (binary and multi-class) |
| `IKPLS` | `n_components=10`, `algorithm=1`, `center=True`, `scale=True`, `backend="numpy"` | Improved Kernel PLS -- fast PLS via ikpls package; supports JAX GPU backend |
| `SIMPLS` | `n_components=10`, `scale=True` | SIMPLS algorithm for PLS regression |
| `RobustPLS` | `n_components=10` | Robust PLS resistant to outliers |
| `RecursivePLS` | `n_components=10`, `forgetting_factor=0.99`, `scale=True`, `center=True`, `backend="numpy"` | Online PLS with exponential forgetting for drifting processes |

---

## Orthogonal PLS

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `OPLS` | `n_components=1`, `scale=True` | Orthogonal PLS -- removes Y-orthogonal variation before PLS regression (via pyopls) |
| `OPLSDA` | `n_components=1`, `pls_components=5`, `scale=True` | OPLS-DA classifier -- OPLS filtering + PLS-DA classification |
| `KOPLS` | `n_components=5`, `n_ortho_components=1`, `kernel="rbf"`, `gamma=None`, `degree=3` | Kernel Orthogonal PLS -- nonlinear OPLS using kernel methods |

---

## Multi-Block and Domain-Invariant

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `MBPLS` | `n_components=5`, `method="NIPALS"`, `standardize=True`, `max_tol=1e-14`, `backend="numpy"` | Multi-Block PLS -- fuses multiple X blocks (sensors, preprocessing variants) into one model |
| `DiPLS` | `n_components=5`, `lags=1` | Domain-Invariant PLS -- handles dynamic systems with time-lagged variables (via trendfitter) |

---

## Sparse and Interval

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `SparsePLS` | `n_components=10` | Sparse PLS -- produces sparse loadings for feature selection |
| `IntervalPLS` | `n_components=10` | Interval PLS -- selects optimal wavelength intervals for PLS |

---

## Kernel PLS

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `KernelPLS` (alias: `KPLS`) | `n_components=10`, `kernel="rbf"`, `gamma=None`, `degree=3`, `coef0=1.0`, `backend="numpy"` | Kernel PLS -- maps X to kernel space then applies PLS on the kernel matrix |
| `OKLMPLS` | `n_components=10`, `featurizer=None` | Online Kernel Learning Machine PLS -- adaptive kernel PLS with pluggable featurizers |
| `FCKPLS` | `n_components=10` | Fractional Convolution Kernel PLS |

Available featurizers for `OKLMPLS`: `IdentityFeaturizer`, `PolynomialFeaturizer`, `RBFFeaturizer`.

Available featurizers for `FCKPLS`: `FractionalConvFeaturizer`.

---

## Locally Weighted

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `LWPLS` | `n_components=10`, `lambda_in_similarity=1.0`, `scale=True`, `backend="numpy"` | Locally Weighted PLS -- builds a local PLS model per query sample weighted by proximity |

---

## Nonlinear PLS

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `NLPLS` | `n_components=10`, `kernel="rbf"` | Nonlinear PLS using kernel methods (alias for KernelPLS) |

---

## Meta-Model Stacking

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `MetaModel` | `config=StackingConfig(...)` | Meta-model for stacking branch predictions; configurable via `StackingConfig` |

Configuration classes: `StackingConfig`, `CoverageStrategy`, `TestAggregation`, `BranchScope`, `StackingLevel`.

Source model selectors: `SourceModelSelector`, `AllPreviousModelsSelector`, `ExplicitModelSelector`, `TopKByMetricSelector`, `DiversitySelector`, `SelectorFactory`.

---

## AOM-PLS Operator Bank

The AOM-PLS operator bank contains preprocessing operators that can be applied per-component:

| Operator Class | Description |
|----------------|-------------|
| `IdentityOperator` | No-op (pass-through) |
| `SavitzkyGolayOperator` | Savitzky-Golay smoothing/derivatives |
| `DetrendProjectionOperator` | Detrending projection |
| `NorrisWilliamsOperator` | Norris-Williams gap derivative |
| `FiniteDifferenceOperator` | Finite difference derivative |
| `WaveletProjectionOperator` | Wavelet-based projection |
| `FFTBandpassOperator` | FFT bandpass filtering |
| `LinearOperator` | General linear operator |
| `ComposedOperator` | Composition of multiple operators |

---

## See Also

- {doc}`../reference/transforms` -- Preprocessing transforms
- {doc}`../reference/splitters` -- Cross-validation splitters
- {doc}`../reference/pipeline_keywords` -- Pipeline keyword syntax (including `model`)
