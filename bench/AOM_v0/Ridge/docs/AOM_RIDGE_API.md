# AOM-Ridge API Reference Draft

## Quick Start

```python
from aomridge.estimators import AOMRidgeRegressor

est = AOMRidgeRegressor(
    selection="superblock",
    operator_bank="compact",
    alphas="auto",
    cv=5,
    block_scaling="rms",
    random_state=0,
)
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
```

## Constructor

```python
AOMRidgeRegressor(
    selection="superblock",
    operator_bank="compact",
    alphas="auto",
    alpha_grid_size=50,
    alpha_grid_low=-6.0,
    alpha_grid_high=6.0,
    alpha=None,
    cv=5,
    scoring="rmse",
    block_scaling="rms",
    center=True,
    scale=False,
    active_top_m=20,
    active_diversity_threshold=0.98,
    random_state=0,
    solver="auto",
)
```

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `selection` | `"superblock"` | `"global"`, `"superblock"`, `"active_superblock"`, later `"branches"` or `"oof_experts"`. |
| `operator_bank` | `"compact"` | Bank name or sequence of strict linear operators. |
| `alphas` | `"auto"` | Explicit alpha sequence or trace-scaled grid. |
| `alpha` | `None` | If set, skip alpha CV. |
| `cv` | `5` | Fold count for alpha/operator selection. |
| `block_scaling` | `"rms"` | `"rms"` or `"none"`. |
| `center` | `True` | Center `X` and `Y`. |
| `scale` | `False` | Reserved; raise until implemented if `True`. |
| `active_top_m` | `20` | Maximum active operators. |
| `active_diversity_threshold` | `0.98` | Response-cosine pruning threshold. |
| `solver` | `"auto"` | `"auto"`, `"cholesky"`, or `"eigh"`. |

## Methods

```python
fit(X, y)
predict(X)
score(X, y)
get_diagnostics()
get_selected_operators()
get_params(deep=True)
set_params(**params)
```

## Fitted Attributes

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `coef_` | `(p, q)` or `(p,)` | Original-space coefficient for strict-linear models. |
| `intercept_` | `(q,)` or scalar | `y_mean_ - x_mean_ @ coef_`. |
| `alpha_` | scalar | Selected Ridge alpha. |
| `alphas_` | `(m,)` | Candidate alpha grid. |
| `dual_coef_` | `(n, q)` | `C = (K + alpha I)^-1 Yc`. |
| `x_mean_` | `(p,)` | Training feature mean. |
| `y_mean_` | `(q,)` | Training target mean. |
| `block_scales_` | `(B,)` | Final block scales. |
| `selected_operators_` | list | Operator names. |
| `selected_operator_indices_` | list | Bank indices. |
| `diagnostics_` | dict | JSON-serializable diagnostics. |

## Diagnostics Contract

`get_diagnostics()` should return:

```python
{
    "model": "AOMRidgeRegressor",
    "selection": "superblock",
    "operator_bank": "compact",
    "alpha": 12.3,
    "alphas": [...],
    "cv": 5,
    "block_scaling": "rms",
    "block_scales": [...],
    "selected_operator_names": [...],
    "selected_operator_indices": [...],
    "operator_scores": {...},
    "block_importance": {...},
    "fit_time_s": 0.0,
    "predict_time_s": 0.0,
    "coef_available": true,
    "original_feature_space": true
}
```

For active mode, also include:

```python
{
    "active_top_m": 20,
    "active_diversity_threshold": 0.98,
    "active_operator_names": [...],
    "active_operator_indices": [...],
    "active_operator_scores": {...},
    "active_pruned_count": 0
}
```

## Shape Rules

Univariate `y`:

```text
fit accepts (n,) or (n, 1)
predict returns (n,)
```

Multi-output `Y`:

```text
fit accepts (n, q)
predict returns (n, q)
coef_ shape is (p, q)
```

