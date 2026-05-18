# TabPFN for NIRS — fixed recipe (no HPO)

## TL;DR

```python
from nirs4all.operators.models import TabPFNNIRSRegressor

est = TabPFNNIRSRegressor(n_estimators=16)
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
```

One TabPFN training per dataset. No hyperparameter tuning, no
preprocessing CV, no stacking. The preprocessing pipeline is **fixed**
and was selected empirically on a 57-dataset NIRS regression cohort.

On that cohort the median test-set RMSE lands at **~+2.5 %** relative to
the TabPFN paper's 72-chain per-dataset HPO baseline `TabPFN_opt` —
making this the simplest recipe that gets close to the paper without
the HPO cost.

## What does it do?

`TabPFNNIRSRegressor` chains the following steps inside `fit`:

| Step | Operator | Why |
|---|---|---|
| 1 | `SavitzkyGolay(window_length=11, polyorder=2, deriv=1)` | Smoothing + first derivative — the most universally robust SG kernel in the TabPFN-paper winners |
| 2 | `OSC(n_components=1)` | Removes one orthogonal-to-y signal direction. Net win on the cohort (helps more than it hurts) |
| 3 | `linspace` subsample to `max_features` | TabPFN v3 has a 2000-feature hard limit. Uniformly-spaced column indices keep spectral structure |
| 4 | `StandardScaler(with_mean=True, with_std=False)` | Center, do not scale (TabPFN is scale-aware) |
| 5 | `TabPFNRegressor(n_estimators=16, ignore_pretraining_limits=True)` | TabPFN ensemble. `n_estimators=16` is the sweet spot from a sweep on the cohort |

At `predict` time the same pipeline is applied to `X_test` (re-using the
fitted SG / OSC / scaler / subsample indices).

## When to use this vs the alternatives

| Recipe | When | Cost per dataset |
|---|---|---|
| **`TabPFNNIRSRegressor`** (this class) | Default choice. No HPO available, want to match `TabPFN_opt` cheaply | **1× TabPFN** |
| `TabPFN_opt` (paper, 72-chain HPO) | Need absolute best test RMSE, willing to pay the HPO cost | ~72× TabPFN |
| Mini-HPO over 5 chains via TabPFN-OOF | More accurate than fixed recipe on some datasets, but **introduces new catastrophes on SPXY / Kennard-Stone splits** (OOF picks the wrong chain when train/test distributions diverge) | ~26× TabPFN |
| `AOMPLSRegressor` / `POPPLSRegressor` | Pure linear baselines, no GPU needed, interpretable | ~1× PLS |
| `nicon` / `decon` CNN | Lots of data (n > 1000), heavy non-linearity | Long training |

## Hyperparameter knobs

| Parameter | Default | What it does |
|---|---:|---|
| `n_estimators` | `16` | TabPFN ensemble size at fit time. 4 is faster but noisier; 16 is the empirical sweet spot |
| `max_features` | `2000` | TabPFN v3 cap. Wider spectra get uniformly subsampled |
| `sg_window_length` | `11` | SG smoothing window |
| `sg_polyorder` | `2` | SG polynomial order |
| `sg_deriv` | `1` | SG derivative order |
| `osc_n_components` | `1` | Number of OSC directions to remove |
| `random_state` | `0` | Seed for TabPFN |
| `device` | `"auto"` | TabPFN device. `"auto"` / `"cpu"` / `"cuda"` |
| `model_path` | `"auto"` | TabPFN checkpoint. `"auto"` loads the default v3 |

After `.fit`, three book-keeping attributes are populated:

- `n_features_in_` — number of input features seen at fit time
- `n_features_used_` — number of features actually fed to TabPFN (after the `max_features` cap)
- `subsample_idx_` — column indices used (or `None` when no subsample)

## Performance on a 57-dataset NIRS cohort

The recipe was validated on the full TabPFN-paper regression cohort
(60 datasets minus a few too-large ones for HPO comparisons):

| Metric | Value |
|---|---|
| Median test RMSE Δ vs `TabPFN_opt` | **~+2.5 %** |
| Wins vs `TabPFN_opt` (lower RMSE) | ~21 / 57 datasets |
| Worst case | +47 % on `Brix_spxy70` (n_train = 35, structural limit) |
| Datasets within ±10 % of `TabPFN_opt` | majority |

The full sweep is in `bench/nicon_v2/` (see `test_full_50_results.csv`
and `test_minihpo5_untested_results.csv` for raw numbers).

## Use in a `nirs4all.run` pipeline

```python
import nirs4all
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.models import TabPFNNIRSRegressor

pipeline = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"model": TabPFNNIRSRegressor(n_estimators=16)},
]
result = nirs4all.run(pipeline=pipeline, dataset="path/to/data", verbose=1)
print(f"best RMSE: {result.best_rmse:.4f}")
```

The estimator is sklearn-compatible (`get_params`, `set_params`,
`clone`), so it integrates with `nirs4all`'s cross-validation,
hyperparameter search, and result-tracking machinery the same way as
any other regressor.

## Requirements

- Install the optional dependency: `pip install tabpfn`
- A GPU is strongly recommended for `n_train > 200`. TabPFN runs on CPU
  but is much slower (the import is done lazily inside `fit`, so the
  module itself imports without the dependency).

## Known limitations

- On a handful of datasets (Brix_spxy70, Corn_Oil_80, Biscuit_Sucrose,
  TIC_spxy70, DIESEL_hlb-a) the fixed recipe is +20 to +47 % above
  `TabPFN_opt`. These are typically `n_train < 100` with very specific
  per-dataset preprocessing requirements that no fixed recipe can match.
- On SPXY / Kennard-Stone test splits where the test set is designed to
  be far from the train set, cheap CV-based preprocessing selectors
  (the alternative to this fixed recipe) frequently mis-rank chains.
  Sticking with the fixed recipe is safer on those splits.

## See also

- {doc}`hyperparameter_tuning` — when you want per-dataset HPO instead
- {doc}`/user_guide/preprocessing/index` — building custom chains
- {doc}`/reference/operator_catalog` — all available transforms
- `examples/user/04_models/U06_tabpfn_nirs.py` — runnable walkthrough
