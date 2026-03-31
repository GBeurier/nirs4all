# AI Coding Assistant Onboarding

**Complete reference for AI assistants working with nirs4all.** This document is self-contained -- everything needed to build, train, export, and deploy NIRS pipelines is here.

**Version**: 0.8.5 | **Python**: 3.11+ | **License**: CeCILL-2.1

---

## 1. Core Concepts (30-Second Overview)

**nirs4all** is a Python library for Near-Infrared Spectroscopy (NIRS) data analysis with ML pipelines.

| Concept | Description |
|---------|-------------|
| **Pipeline** | List of steps (preprocessing, splitting, model) executed sequentially |
| **SpectroDataset** | Core container holding `X` (spectra), `y` (targets), `metadata`, `folds` |
| **Operators** | Transformers, models, splitters -- anything sklearn-compatible plus NIRS-specific |
| **Controllers** | Registry pattern that dispatches operator execution (extensible via `@register_controller`) |
| **Bundles (.n4a)** | Serialized pipelines for deployment with full preprocessing chain |
| **Generators** | Keywords like `_or_`, `_range_`, `_grid_` that expand a single pipeline spec into many variants |

**Typical Workflow**:

```
Load data --> Define pipeline --> nirs4all.run() --> Analyze RunResult --> Export .n4a bundle
```

---

## 2. Primary API

All functions live directly on the `nirs4all` module.

```python
import nirs4all
```

### 2.1 nirs4all.run()

```python
nirs4all.run(
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    *,
    name: str = "",
    session: Session | None = None,
    verbose: int = 1,                # 0=quiet, 1=info, 2=debug, 3=trace
    save_artifacts: bool = True,
    save_charts: bool = True,
    plots_visible: bool = False,
    random_state: int | None = None,
    refit: bool | dict | list[dict] | None = True,
    cache: CacheConfig | None = None,
    project: str | None = None,
    report_naming: str = "nirs",     # "nirs" | "ml" | "auto"
    **runner_kwargs,
) -> RunResult
```

**`pipeline`** accepts:

- List of steps (most common): `[MinMaxScaler(), PLSRegression(10)]`
- Dict with steps: `{"steps": [...], "name": "my_pipeline"}`
- Path to YAML/JSON config file: `"configs/my_pipeline.yaml"`
- List of pipelines (batch): `[pipeline1, pipeline2]` -- each runs independently

**`dataset`** accepts:

- Path to data folder: `"sample_data/regression"`
- Numpy arrays: `(X, y)` or `X` alone
- Dict with arrays: `{"X": X, "y": y, "metadata": meta}`
- SpectroDataset instance
- List of datasets (batch): `[dataset1, dataset2]` -- Cartesian product with pipelines

**`refit`** controls post-CV retraining on the full training set:

- `True` (default): Refit top 1 by RMSECV
- `False` or `None`: Disable refit
- `dict`: Single criterion, e.g. `{"top_k": 3, "ranking": "mean_val"}`
- `list[dict]`: Multiple criteria (union selection)

Ranking methods: `"rmsecv"` (OOF concatenated val score), `"mean_val"` (mean of individual fold val scores).

**Common `runner_kwargs`**:

- `workspace_path`: Workspace root directory
- `n_jobs`: Parallel variant execution (-1 = all cores)
- `continue_on_error`: Whether to continue on step failures
- `max_generation_count`: Max pipeline combinations (for generators)

### 2.2 nirs4all.predict()

```python
nirs4all.predict(
    model: ModelSpec | None = None,
    data: DataSpec | None = None,
    *,
    chain_id: str | None = None,
    workspace_path: str | Path | None = None,
    name: str = "prediction_dataset",
    all_predictions: bool = False,
    session: Session | None = None,
    verbose: int = 0,
    **runner_kwargs,
) -> PredictResult
```

**Two prediction paths**:

1. **Store-based** (preferred): pass `chain_id` + raw numpy `data`
2. **Model-based**: pass `model` (path to `.n4a` bundle or prediction dict) + `data`

**`model`** accepts:

- Prediction dict from `result.best` or `result.top()`
- Path to exported bundle: `"exports/model.n4a"`
- Path to pipeline config directory

**`data`** accepts:

- Path to data folder
- Numpy array `X_new` (n_samples, n_features)
- Tuple `(X,)` or `(X, y)` for evaluation
- Dict: `{"X": X, "metadata": meta}`
- SpectroDataset instance

### 2.3 nirs4all.explain()

```python
nirs4all.explain(
    model: ModelSpec,
    data: DataSpec,
    *,
    name: str = "explain_dataset",
    session: Session | None = None,
    verbose: int = 1,
    plots_visible: bool = True,
    n_samples: int | None = None,
    explainer_type: str = "auto",    # "auto" | "tree" | "kernel" | "deep" | "linear"
    **shap_params,
) -> ExplainResult
```

**`shap_params`** common options:

- `feature_names`: List of feature names
- `background_samples`: Number of background samples
- `max_display`: Max features to show in plots

### 2.4 nirs4all.retrain()

```python
nirs4all.retrain(
    source: SourceSpec,
    data: DataSpec,
    *,
    mode: str = "full",              # "full" | "transfer" | "finetune"
    name: str = "retrain_dataset",
    new_model: Any | None = None,
    epochs: int | None = None,
    session: Session | None = None,
    verbose: int = 1,
    save_artifacts: bool = True,
    **kwargs,
) -> RunResult
```

**Modes**:

| Mode | Behavior |
|------|----------|
| `"full"` | Train everything from scratch (same pipeline structure) |
| `"transfer"` | Use existing preprocessing, train new model |
| `"finetune"` | Continue training existing model (neural networks) |

Additional `kwargs`: `learning_rate`, `freeze_layers`, `step_modes` (per-step mode overrides).

### 2.5 nirs4all.session()

Two usage patterns:

**Pattern 1: Resource sharing** (context manager, no pipeline):

```python
with nirs4all.session(verbose=2, save_artifacts=True) as s:
    r1 = nirs4all.run(pipeline1, data1, session=s)
    r2 = nirs4all.run(pipeline2, data2, session=s)
    # Both runs share workspace and configuration
```

**Pattern 2: Stateful pipeline** (with pipeline):

```python
session = nirs4all.Session(pipeline=pipeline, name="MyModel")
result = session.run("sample_data/regression")
predictions = session.predict(new_data)
session.save("exports/my_model.n4a")
```

**Session class**:

```python
class Session:
    def __init__(self, pipeline=None, name="", **runner_kwargs): ...

    # Properties
    name: str
    pipeline: list | None
    status: str            # "initialized" | "trained" | "error"
    is_trained: bool
    runner: PipelineRunner
    workspace_path: Path | None
    history: list[dict]

    # Methods
    def run(self, dataset, *, plots_visible=False, **kwargs) -> RunResult: ...
    def predict(self, dataset, **kwargs) -> PredictResult: ...
    def retrain(self, dataset, mode="full", **kwargs) -> RunResult: ...
    def save(self, path) -> Path: ...
    def close(self) -> None: ...
```

**Load a saved session**:

```python
session = nirs4all.load_session("exports/model.n4a")
predictions = session.predict(new_data)
```

### 2.6 nirs4all.generate()

Both callable and a namespace:

```python
# Direct call
dataset = nirs4all.generate(n_samples=1000, random_state=42)

# Namespace functions
dataset = nirs4all.generate.regression(n_samples=500)
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)
builder = nirs4all.generate.builder(n_samples=1000)
```

**Main function signature**:

```python
nirs4all.generate(
    n_samples: int = 1000,
    *,
    random_state: int | None = None,
    complexity: str = "simple",         # "simple" | "realistic" | "complex"
    wavelength_range: tuple[float, float] | None = None,
    components: list[str] | None = None,
    target_range: tuple[float, float] | None = None,
    train_ratio: float = 0.8,
    as_dataset: bool = True,            # False returns (X, y) tuple
    name: str = "synthetic_nirs",
) -> SpectroDataset | tuple[np.ndarray, np.ndarray]
```

Components: `"water"`, `"protein"`, `"lipid"`, `"starch"`, `"cellulose"`, `"chlorophyll"`, `"oil"`, `"nitrogen_compound"`.

**Namespace functions**:

| Function | Purpose | Key Extra Parameters |
|----------|---------|---------------------|
| `generate.regression()` | Regression dataset | `target_component`, `distribution` ("dirichlet", "uniform", "lognormal", "correlated") |
| `generate.classification()` | Classification dataset | `n_classes`, `class_separation`, `class_weights` |
| `generate.builder()` | SyntheticDatasetBuilder for fluent API | Returns builder for method chaining |
| `generate.multi_source()` | Multi-source dataset | `sources` (list of source dicts with name, type, wavelength_range, n_features) |
| `generate.to_folder()` | Export to folder | `path`, `format` ("standard", "single", "fragmented") |
| `generate.to_csv()` | Export to single CSV | `path` |
| `generate.from_template()` | Mimic real data | `template` (path, array, or SpectroDataset) |
| `generate.product()` | Product template | `template` (e.g. "milk_variable_fat"), `target`, `instrument_wavelength_grid` |
| `generate.category()` | Multiple templates | `templates` (list), `samples_per_template` |

---

## 3. Result Objects

### 3.1 RunResult (from `run()`)

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `best` | `dict` | Best prediction entry, preferring refit (final) over CV |
| `best_score` | `float` | Primary test score from best entry (NaN if unavailable) |
| `best_rmse` | `float` | RMSE from best entry (regression; NaN if unavailable) |
| `best_r2` | `float` | R-squared from best entry (regression; NaN if unavailable) |
| `best_accuracy` | `float` | Accuracy from best entry (classification; NaN if unavailable) |
| `best_final` | `dict` | Best refit entry (fold_id="final") |
| `final` | `dict \| None` | Refit model prediction entry, or None if no refit |
| `final_score` | `float \| None` | Refit model's test score, or None |
| `cv_best` | `dict` | Best CV prediction entry (excludes refit entries) |
| `cv_best_score` | `float` | Best CV validation score (NaN if unavailable) |
| `models` | `dict[str, ModelRefitResult]` | Per-model refit results |
| `predictions` | `Predictions` | Full predictions object |
| `per_dataset` | `dict` | Per-dataset execution details |
| `artifacts_path` | `Path \| None` | Path to workspace artifacts directory |
| `num_predictions` | `int` | Total number of prediction entries |

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `top()` | `top(n=5, **kwargs)` | Top N predictions by ranking. Supports `group_by`, `rank_metric`, `return_grouped`. |
| `export()` | `export(output_path, format="n4a", source=None, chain_id=None)` | Export model to `.n4a` bundle |
| `export_model()` | `export_model(output_path, source=None, format=None, fold=None)` | Export raw model artifact only |
| `filter()` | `filter(**kwargs)` | Filter predictions by `dataset_name`, `model_name`, `partition`, `fold_id` |
| `get_datasets()` | `get_datasets()` | List unique dataset names |
| `get_models()` | `get_models()` | List unique model names |
| `summary()` | `summary()` | Multi-line summary string |
| `validate()` | `validate(check_nan_metrics=True, raise_on_failure=True)` | Validate result for common issues |

**Lifecycle methods**: `detach()`, `close()`. Supports context manager (`with`).

### 3.2 PredictResult (from `predict()`)

```python
@dataclass
class PredictResult:
    y_pred: np.ndarray
    metadata: dict[str, Any]
    sample_indices: np.ndarray | None
    model_name: str
    preprocessing_steps: list[str]
```

| Property / Method | Description |
|-------------------|-------------|
| `values` | Alias for `y_pred` |
| `shape` | Shape of prediction array |
| `is_multioutput` | True if predictions have multiple outputs |
| `to_numpy()` | Get as numpy array |
| `to_list()` | Get as Python list (flattened) |
| `to_dataframe()` | Get as pandas DataFrame |
| `flatten()` | Get flattened 1D predictions |

### 3.3 ExplainResult (from `explain()`)

```python
@dataclass
class ExplainResult:
    shap_values: Any            # shap.Explanation or np.ndarray
    feature_names: list[str] | None
    base_value: float | np.ndarray | None
    visualizations: dict[str, Path]
    explainer_type: str
    model_name: str
```

| Property / Method | Description |
|-------------------|-------------|
| `values` | Raw SHAP values array |
| `shape` | Shape of SHAP values |
| `mean_abs_shap` | Mean absolute SHAP values per feature |
| `top_features` | Feature names sorted by importance |
| `get_feature_importance(top_n)` | Feature importance ranking |
| `get_sample_explanation(idx)` | Explanation for single sample |
| `to_dataframe()` | SHAP values as DataFrame |

---

## 4. Pipeline Syntax

### 4.1 Step Formats

A pipeline is a Python list. Each element can be:

```python
# Instance (most common)
MinMaxScaler()

# Class (instantiated with defaults)
MinMaxScaler

# Dict with keyword
{"model": PLSRegression(n_components=10)}
{"y_processing": StandardScaler()}
```

### 4.2 Execution Order

A typical pipeline executes in this order:

1. **Preprocessing transforms** -- applied to X
2. **y_processing** -- applied to y (target scaling)
3. **Splitter** -- defines cross-validation folds
4. **Fold loop**: for each fold:
   - Apply transforms (fit on train, transform on train+test)
   - Fit model on train
   - Predict on test
5. **Refit** -- retrain winning variant on full training set

### 4.3 Pipeline Keywords (Complete Reference)

| Keyword | Purpose | Syntax | Example |
|---------|---------|--------|---------|
| `model` | Explicit model definition | `{"model": instance}` | `{"model": PLSRegression(10)}` |
| `y_processing` | Target (y) scaling | `{"y_processing": instance}` | `{"y_processing": MinMaxScaler()}` |
| `tag` | Mark samples without removing | `{"tag": filter}` | `{"tag": YOutlierFilter(method="iqr")}` |
| `exclude` | Exclude samples from training | `{"exclude": filter}` or `{"exclude": [f1, f2], "mode": "any"}` | `{"exclude": YOutlierFilter()}` |
| `branch` | Parallel sub-pipelines (duplication) or sample partitioning (separation) | List (duplication) or dict with `by_*` key (separation) | See Section 11 |
| `merge` | Combine branch outputs | `{"merge": strategy}` | `{"merge": "predictions"}` |
| `merge_sources` | Merge multi-source datasets | `{"merge_sources": strategy}` | `{"merge_sources": "concat"}` |
| `merge_predictions` | Late fusion (merge predictions from multiple models without branches) | `{"merge_predictions": strategy}` | `{"merge_predictions": "all"}` |
| `sample_augmentation` | Data augmentation (training only) | `{"sample_augmentation": augmenter}` | `{"sample_augmentation": GaussianAdditiveNoise()}` |
| `feature_augmentation` | Feature-level augmentation | `{"feature_augmentation": augmenter}` | `{"feature_augmentation": ...}` |
| `concat_transform` | Concatenate multiple transform outputs | `{"concat_transform": [t1, t2]}` | `{"concat_transform": [SNV(), SavitzkyGolay()]}` |
| `rep_to_sources` | Repetitions to multi-source format | `{"rep_to_sources": column}` | `{"rep_to_sources": "Sample_ID"}` |
| `rep_to_pp` | Repetitions to preprocessing pipelines | `{"rep_to_pp": column}` | `{"rep_to_pp": "Sample_ID"}` |
| `name` | Label a step for identification | `{"name": "my_step", ...}` | `{"name": "scaler", "model": Ridge()}` |

---

## 5. Generator Keywords (Complete Reference)

Generator keywords expand a single pipeline specification into multiple pipeline variants. The orchestrator runs all variants and selects the best.

### 5.1 `_or_` -- Try Alternatives

Creates N pipeline variants, one per choice.

```python
# Try 3 different preprocessors
{"_or_": [SNV, MSC, Detrend]}
# Expands to 3 variants: one with SNV(), one with MSC(), one with Detrend()

# Try different models
{"_or_": [PLSRegression(5), PLSRegression(10), PLSRegression(15)]}
# Expands to 3 variants
```

### 5.2 `_range_` -- Linear Parameter Sweep

Generates a linear sequence: `[start, stop, step]` (exclusive stop).

```python
{"_range_": [1, 30, 5]}
# Expands to: [1, 6, 11, 16, 21, 26]

# Applied to a model parameter via nesting
PLSRegression(n_components={"_range_": [5, 26, 5]})
# Creates PLSRegression(5), PLSRegression(10), PLSRegression(15), PLSRegression(20), PLSRegression(25)
```

### 5.3 `_log_range_` -- Logarithmic Sweep

Generates logarithmically spaced values: `[start, stop, num_points]`.

```python
{"_log_range_": [1e-3, 1e0, 4]}
# Expands to: [0.001, 0.01, 0.1, 1.0]
```

### 5.4 `_grid_` -- Cartesian Product of Parameters

Produces all combinations of parameter values (like sklearn's GridSearchCV).

```python
{"_grid_": {"n_components": [5, 10, 15], "scale": [True, False]}}
# Expands to 6 combinations:
#   {"n_components": 5, "scale": True}, {"n_components": 5, "scale": False},
#   {"n_components": 10, "scale": True}, ...
```

### 5.5 `_cartesian_` -- Pipeline Stage Combinations

Produces Cartesian product across pipeline stages. Unlike `_grid_` (which produces parameter dicts), `_cartesian_` produces pipeline step sequences.

```python
{"_cartesian_": [
    {"_or_": [SNV, MSC]},
    {"_or_": [PLSRegression(5), PLSRegression(10)]}
]}
# Expands to 4 pipelines:
#   [SNV(), PLSRegression(5)]
#   [SNV(), PLSRegression(10)]
#   [MSC(), PLSRegression(5)]
#   [MSC(), PLSRegression(10)]
```

### 5.6 `_zip_` -- Parallel Paired Iteration

Like Python's `zip()`: pairs values at the same index.

```python
{"_zip_": {"n_components": [5, 10, 15], "scale": [True, False, True]}}
# Expands to 3 pairs:
#   {"n_components": 5, "scale": True}
#   {"n_components": 10, "scale": False}
#   {"n_components": 15, "scale": True}
```

### 5.7 `_chain_` -- Sequential Ordered Choices

Like `_or_` but preserves insertion order (important when expansion order matters).

```python
{"_chain_": [config1, config2, config3]}
# Expands to 3 variants in the exact order given
```

### 5.8 `_sample_` -- Random Distribution Sampling

Draw random parameter values from statistical distributions.

```python
{"_sample_": {
    "distribution": "log_uniform",  # "uniform", "log_uniform", "normal"
    "from": 1e-4,
    "to": 1e-1,
    "num": 20
}}
# Draws 20 values from log-uniform distribution between 1e-4 and 1e-1
```

### 5.9 Modifiers

Modifiers can be combined with any generator keyword:

| Modifier | Purpose | Example |
|----------|---------|---------|
| `count` | Limit number of generated variants | `{"_or_": [...], "count": 5}` |
| `pick` | Unordered selection (combinations) | `{"_or_": [A, B, C, D], "pick": 2}` -- C(4,2)=6 combos |
| `arrange` | Ordered arrangement (permutations) | `{"_or_": [A, B, C], "arrange": 2}` -- P(3,2)=6 arrangements |

### 5.10 Generator Keyword Summary Table

| Keyword | Input | Expansion | Typical Use |
|---------|-------|-----------|-------------|
| `_or_` | `[A, B, C]` | 3 variants | Compare preprocessing methods |
| `_range_` | `[start, stop, step]` | Linear sequence | Sweep n_components |
| `_log_range_` | `[start, stop, num]` | Log-spaced values | Sweep regularization alpha |
| `_grid_` | `{param: [vals]}` | Cartesian product of params | Multi-parameter grid search |
| `_cartesian_` | `[stage1, stage2]` | Cartesian product of stages | Multi-stage pipeline combos |
| `_zip_` | `{param: [vals]}` | Parallel pairs | Coupled parameters |
| `_chain_` | `[c1, c2, c3]` | Ordered sequence | Explicit ordering |
| `_sample_` | `{distribution, from, to, num}` | Random draws | Random search |

---

## 6. NIRS Transforms (Complete Reference)

Import path: `from nirs4all.operators.transforms import <Class>`

### Scatter Correction and Normalization

| Class | Alias | Key Parameters | Description |
|-------|-------|----------------|-------------|
| `StandardNormalVariate` | `SNV` | (none) | Row-wise mean-center and scale by std. Removes scatter effects. |
| `RobustStandardNormalVariate` | `RNV` | (none) | Robust version of SNV using median/MAD. Outlier-resistant. |
| `LocalStandardNormalVariate` | -- | (none) | Local SNV applied to sub-regions |
| `MultiplicativeScatterCorrection` | `MSC` | (none) | Correct multiplicative/additive scatter using reference spectrum |
| `ExtendedMultiplicativeScatterCorrection` | `EMSC` | (none) | Extended MSC with polynomial terms |
| `AreaNormalization` | -- | (none) | Normalize spectra by area under curve |
| `Normalize` | -- | (none) | General normalization |
| `SimpleScale` | -- | (none) | Simple scaling transform |

### Smoothing

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `SavitzkyGolay` | `window_length=11`, `polyorder=3`, `deriv=0` | Savitzky-Golay filter for smoothing and derivatives |
| `Gaussian` | (none) | Gaussian smoothing filter |
| `WaveletDenoise` | `wavelet="db4"`, `level=None`, `mode="soft"` | Multi-level wavelet denoising with thresholding |

### Derivatives

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `FirstDerivative` | (none) | First derivative of spectra |
| `SecondDerivative` | (none) | Second derivative of spectra |
| `NorrisWilliams` | `gap`, `segment` | Gap derivative with segment smoothing |
| `Derivate` | (none) | General derivative transform |

### Baseline Correction

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `Baseline` | (none) | Baseline removal |
| `Detrend` | (none) | Remove linear/polynomial trend from spectra |
| `ASLSBaseline` | (none) | Asymmetric Least Squares baseline correction |
| `AirPLS` | (none) | Adaptive Iteratively Reweighted Penalized Least Squares |
| `ArPLS` | (none) | Asymmetrically Reweighted Penalized Least Squares |
| `IModPoly` | (none) | Improved Modified Polynomial baseline |
| `ModPoly` | (none) | Modified Polynomial baseline |
| `SNIP` | (none) | Statistics-sensitive Non-linear Iterative Peak-clipping |
| `RollingBall` | (none) | Rolling ball baseline estimation |
| `IASLS` | (none) | Improved Asymmetric Least Squares |
| `BEADS` | (none) | Baseline Estimation And Denoising with Sparsity |
| `PyBaselineCorrection` | `method` | Generic wrapper for pybaselines methods |

### Orthogonalization

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `OSC` | (none) | Orthogonal Signal Correction (DOSC variant) |
| `EPO` | (none) | External Parameter Orthogonalization |

### Signal Conversion

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `ReflectanceToAbsorbance` | (none) | Convert reflectance to absorbance via Beer-Lambert |
| `ToAbsorbance` | (none) | Convert to absorbance |
| `FromAbsorbance` | (none) | Convert from absorbance |
| `PercentToFraction` | (none) | Convert percentage to fraction |
| `FractionToPercent` | (none) | Convert fraction to percentage |
| `KubelkaMunk` | (none) | Kubelka-Munk transformation |
| `SignalTypeConverter` | (none) | General signal type converter |
| `LogTransform` | (none) | Logarithmic transformation |

### Wavelet Transforms

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `Haar` | (none) | Haar wavelet decomposition |
| `Wavelet` | (none) | Generic wavelet transform |
| `WaveletFeatures` | (none) | Wavelet-based feature extraction |
| `WaveletPCA` | (none) | Wavelet + PCA combination |
| `WaveletSVD` | (none) | Wavelet + SVD combination |

### Feature Selection

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `CARS` | (none) | Competitive Adaptive Reweighted Sampling |
| `MCUVE` | (none) | Monte Carlo Uninformative Variable Elimination |
| `FlexiblePCA` | (none) | PCA with flexible component selection |
| `FlexibleSVD` | (none) | SVD with flexible component selection |

### Resampling and Feature Engineering

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `Resampler` | (none) | Wavelength interpolation / resampling |
| `CropTransformer` | (none) | Crop spectral region |
| `ResampleTransformer` | (none) | Resample features |
| `FlattenPreprocessing` | (none) | Flatten multi-dimensional features |

### Target Discretizers

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `IntegerKBinsDiscretizer` | (none) | Discretize continuous targets into integer bins |
| `RangeDiscretizer` | (none) | Discretize by range boundaries |

---

## 7. Augmentation Operators (Complete Reference)

Import path: `from nirs4all.operators.augmentation import <Class>`

Use with the `sample_augmentation` or `feature_augmentation` pipeline keywords.

### Noise and Distortion

| Class | Description |
|-------|-------------|
| `GaussianAdditiveNoise` | Add Gaussian noise to spectra |
| `MultiplicativeNoise` | Apply random gain factors |
| `SpikeNoise` | Add spike artifacts |
| `HeteroscedasticNoiseAugmenter` | Signal-dependent noise (varies with intensity) |

### Baseline Effects

| Class | Description |
|-------|-------------|
| `LinearBaselineDrift` | Add linear baseline drift |
| `PolynomialBaselineDrift` | Add polynomial baseline drift |

### Wavelength Distortions

| Class | Description |
|-------|-------------|
| `WavelengthShift` | Shift spectra along wavelength axis |
| `WavelengthStretch` | Stretch/compress wavelength axis |
| `LocalWavelengthWarp` | Local wavelength distortions via spline control points |
| `SmoothMagnitudeWarp` | Smooth multiplicative warping |

### Spectral Distortions

| Class | Description |
|-------|-------------|
| `BandPerturbation` | Perturb random wavelength bands |
| `GaussianSmoothingJitter` | Random Gaussian kernel broadening |
| `UnsharpSpectralMask` | High-pass spectral enhancement |
| `BandMasking` | Multiplicative masking of random bands |
| `ChannelDropout` | Random channel zeroing |
| `LocalClipping` | Local spectrum clipping |

### Mixing

| Class | Description |
|-------|-------------|
| `MixupAugmenter` | Blend samples with KNN neighbors |
| `LocalMixupAugmenter` | Band-local mixup with KNN neighbors |
| `ScatterSimulationMSC` | MSC-style scatter simulation |

### Spline-Based

| Class | Description |
|-------|-------------|
| `Spline_Smoothing` | Smoothing spline augmentation |
| `Spline_X_Perturbations` | Wavelength axis perturbations via B-spline |
| `Spline_Y_Perturbations` | Intensity axis perturbations via B-spline |
| `Spline_X_Simplification` | Spectrum simplification on x-axis |
| `Spline_Curve_Simplification` | Spectrum simplification along curve length |

### Environmental Effects (require wavelengths)

| Class | Description |
|-------|-------------|
| `TemperatureAugmenter` | Simulate temperature-induced spectral changes |
| `MoistureAugmenter` | Simulate moisture/water activity effects |

### Scattering Effects (require wavelengths)

| Class | Description |
|-------|-------------|
| `ParticleSizeAugmenter` | Simulate particle size scattering |
| `EMSCDistortionAugmenter` | Apply EMSC-style distortions |

### Edge Artifacts (require wavelengths)

| Class | Description |
|-------|-------------|
| `DetectorRollOffAugmenter` | Simulate detector sensitivity roll-off at edges |
| `StrayLightAugmenter` | Simulate stray light effects (peak truncation) |
| `EdgeCurvatureAugmenter` | Simulate edge curvature/baseline bending |
| `TruncatedPeakAugmenter` | Add truncated peaks at spectral boundaries |
| `EdgeArtifactsAugmenter` | Combined edge artifacts augmenter |

### Synthesis-Derived

| Class | Description |
|-------|-------------|
| `PathLengthAugmenter` | Optical path length scaling |
| `BatchEffectAugmenter` | Batch/session/instrument effects |
| `InstrumentalBroadeningAugmenter` | Spectral resolution broadening |
| `DeadBandAugmenter` | Detector saturation simulation |

### Random / Geometric

| Class | Description |
|-------|-------------|
| `Rotate_Translate` | Rotation and translation augmentation |
| `Random_X_Operation` | Random multiplicative/additive operations |

---

## 8. Built-in Models (Complete Reference)

Import path: `from nirs4all.operators.models import <Class>`

### PLS Variants

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `PLSDA` | `n_components` | PLS Discriminant Analysis |
| `IKPLS` | `n_components` | Improved Kernel PLS (fast algorithm) |
| `OPLS` | `n_components` | Orthogonal PLS |
| `OPLSDA` | `n_components` | Orthogonal PLS Discriminant Analysis |
| `MBPLS` | `n_components` | Multi-Block PLS |
| `DiPLS` | `n_components` | Domain-Invariant PLS |
| `SparsePLS` | `n_components` | Sparse PLS (feature selection) |
| `SIMPLS` | `n_components` | SIMPLS algorithm |
| `LWPLS` | `n_components` | Locally Weighted PLS |
| `IntervalPLS` | `n_components` | Interval PLS (wavelength selection) |
| `RobustPLS` | `n_components` | Robust PLS (resistant to outliers) |
| `RecursivePLS` | `n_components` | Recursive PLS |
| `KOPLS` | `n_components` | Kernel OPLS |

### Adaptive PLS

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `AOMPLSRegressor` | `n_components`, `gate="hard"` | Adaptive Operator-Mixture PLS -- auto-selects best preprocessing per component from operator bank |
| `AOMPLSClassifier` | `n_components`, `gate="hard"` | AOM-PLS for classification with probability calibration |
| `POPPLSRegressor` | `n_components`, `auto_select=True` | Per-Operator-Per-component PLS -- different operator per component via PRESS minimization |
| `POPPLSClassifier` | `n_components`, `auto_select=True` | POP-PLS for classification with probability calibration |

### Kernel and Non-Linear PLS

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `KernelPLS` / `KPLS` | `n_components` | Kernel PLS |
| `NLPLS` | `n_components` | Non-linear PLS |
| `FCKPLS` | `n_components` | Fractional Convolution Kernel PLS |
| `OKLMPLS` | `n_components` | Online Kernel Learning Machine PLS |

### Meta-Model / Stacking

| Class | Description |
|-------|-------------|
| `MetaModel` | Meta-model for stacking configuration |
| `StackingConfig` | Configuration for stacking level |

### Model Selection

| Class | Description |
|-------|-------------|
| `SourceModelSelector` | Select models from source |
| `AllPreviousModelsSelector` | Select all previous models |
| `ExplicitModelSelector` | Explicitly select specific models |
| `TopKByMetricSelector` | Select top K models by metric |
| `DiversitySelector` | Select diverse set of models |

**Note**: Any sklearn-compatible estimator can be used as a model step. Common sklearn models:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
```

Deep learning models (TensorFlow, PyTorch) are lazy-loaded:

```python
from nirs4all.operators.models.tensorflow import nicon, generic
```

---

## 9. Splitters (Complete Reference)

### NIRS-Specific Splitters

Import path: `from nirs4all.operators.splitters import <Class>`

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `KennardStoneSplitter` | `test_size` | Kennard-Stone algorithm -- maximally covers feature space |
| `SPXYSplitter` | `test_size` | Sample Partitioning based on joint X and Y distances |
| `SPXYFold` | `n_splits` | SPXY-based K-Fold cross-validation |
| `SPXYGFold` | `n_splits` | Group-aware SPXY K-Fold |
| `KMeansSplitter` | `n_clusters` | K-means clustering based split |
| `KBinsStratifiedSplitter` | `n_bins` | Binned stratification for continuous targets |
| `SystematicCircularSplitter` | (none) | Systematic circular assignment |
| `BinnedStratifiedGroupKFold` | `n_splits`, `n_bins` | Binned stratified group K-fold |
| `GroupedSplitterWrapper` | `splitter`, `group_column` | Wraps any splitter for group-aware splitting |

### Common sklearn Splitters

These are directly usable as pipeline steps:

```python
from sklearn.model_selection import (
    KFold,                       # Standard K-fold
    StratifiedKFold,             # Stratified K-fold (classification)
    ShuffleSplit,                # Random train/test splits
    RepeatedKFold,               # Repeated K-fold
    LeaveOneOut,                 # Leave-one-out
    GroupKFold,                  # Group-aware K-fold
)
```

---

## 10. Filters (Complete Reference)

Import path: `from nirs4all.operators.filters import <Class>`

Filters are used with the `tag` and `exclude` pipeline keywords.

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `YOutlierFilter` | `method="iqr"`, `threshold=1.5` | Filter by outlier target values. Methods: `"iqr"`, `"zscore"`, `"percentile"`, `"mad"` |
| `XOutlierFilter` | `method="mahalanobis"`, `threshold=3.0` | Filter by outlier spectra. Methods: `"mahalanobis"`, `"robust_mahalanobis"`, `"pca_residual"`, `"pca_leverage"`, `"isolation_forest"`, `"lof"` |
| `SpectralQualityFilter` | (none) | Filter poor spectral quality (NaN, zeros, low variance) |
| `HighLeverageFilter` | (none) | Filter high-leverage samples |
| `MetadataFilter` | `column`, `values` | Filter by metadata column values |
| `CompositeFilter` | `filters`, `mode` | Combine multiple filters with AND/OR logic |

### Usage with `tag` and `exclude`

```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter

# Tag: marks samples but does NOT remove them (for downstream branching/analysis)
{"tag": YOutlierFilter(method="iqr")}

# Exclude: removes samples from training
{"exclude": YOutlierFilter(method="iqr")}

# Exclude with multiple filters
{"exclude": [YOutlierFilter(), XOutlierFilter(method="mahalanobis")], "mode": "any"}
# "any" = exclude if ANY filter flags the sample
# "all" = exclude only if ALL filters flag the sample
```

---

## 11. Branching Patterns

### 11.1 Duplication Branches

All samples are processed by ALL branches. Each branch applies different preprocessing.

**List syntax**:

```python
{"branch": [
    [SNV(), PCA(n_components=10)],           # Branch 0
    [MSC(), FirstDerivative()],              # Branch 1
]}
```

### 11.2 Separation Branches

Samples are partitioned into non-overlapping subsets. Each sample goes to exactly one branch.

**By metadata column**:

```python
{"branch": {
    "by_metadata": "site",
    "steps": [SNV(), PLSRegression(10)],     # Same steps for all groups
}}
```

**By tag** (requires a prior `tag` step):

```python
{"tag": YOutlierFilter(method="iqr")},       # Creates tag "y_outlier_iqr"
{"branch": {
    "by_tag": "y_outlier_iqr",
    "values": {
        "outliers": True,
        "inliers": False,
    },
    "steps": [PLSRegression(n_components=5)],
}}
```

**By source** (multi-source datasets):

```python
{"branch": {
    "by_source": True,
    "steps": {
        "NIR": [SNV(), SavitzkyGolay()],
        "markers": [MinMaxScaler()],
    },
}}
```

**By filter**:

```python
{"branch": {"by_filter": SampleFilter(...)}}
```

---

## 12. Merge Patterns

Merge exits branch mode and combines outputs. The strategy depends on the branch type.

### For Duplication Branches (Stacking)

| Strategy | Syntax | What It Does |
|----------|--------|-------------|
| `"predictions"` | `{"merge": "predictions"}` | Collects OOF (out-of-fold) predictions from each branch as features for a meta-model |
| `"features"` | `{"merge": "features"}` | Concatenates transformed feature matrices from each branch |
| `"all"` | `{"merge": "all"}` | Combines both features and predictions |

### For Separation Branches (Reassembly)

| Strategy | Syntax | What It Does |
|----------|--------|-------------|
| `"concat"` | `{"merge": "concat"}` | Reassembles samples in original order from disjoint branches |

### For Multi-Source Datasets

| Strategy | Syntax | What It Does |
|----------|--------|-------------|
| Source concat | `{"merge_sources": "concat"}` | Concatenate features from all sources into one |
| Source stack | `{"merge_sources": "stack"}` | Stack sources |
| Source dict | `{"merge_sources": "dict"}` | Keep as dict |

### Late Fusion (No Branches)

```python
{"merge_predictions": "all"}
# Combines predictions from multiple model steps without branching
```

---

## 13. Stacking Pattern

The canonical stacking pattern: branch into base learners, merge their OOF predictions, train a meta-model.

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from nirs4all.operators.transforms import SNV, MSC

pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],    # Base learner 1
        [MSC(), RandomForestRegressor()],            # Base learner 2
    ]},
    {"merge": "predictions"},        # OOF predictions become features
    {"model": Ridge()},              # Meta-model trained on OOF predictions
]

result = nirs4all.run(pipeline, dataset)
```

**How it works**:

1. Each branch processes all samples with its own preprocessing + model
2. During CV, each branch produces out-of-fold (OOF) predictions
3. `{"merge": "predictions"}` collects these OOF predictions as new features
4. The meta-model (Ridge) is trained on the stacked OOF features
5. This prevents information leakage because OOF predictions use held-out data

**Feature merge variant** (for feature-level stacking):

```python
pipeline = [
    {"branch": [
        [SNV()],
        [MSC()],
        [SavitzkyGolay(deriv=1)],
    ]},
    {"merge": "features"},           # Concatenated transformed features
    PLSRegression(n_components=10),
]
```

---

## 14. Scoring and Refit

### Primary Metric

nirs4all automatically detects the task type (regression vs classification) and selects the appropriate metric:

- **Regression**: RMSE (lower is better)
- **Classification**: Accuracy (higher is better)

### NIRS Terminology

When `report_naming="nirs"` (default):

| Internal Name | NIRS Name | Description |
|---------------|-----------|-------------|
| CV validation score | RMSECV | Root Mean Square Error of Cross-Validation |
| Test score (refit) | RMSEP | Root Mean Square Error of Prediction |
| Train score | RMSEC | Root Mean Square Error of Calibration |

### Refit Behavior

After cross-validation identifies the best pipeline variant:

1. The winning variant is retrained on the full training set (refit)
2. `result.best` prefers the refit entry (`fold_id="final"`) over CV entries
3. `result.cv_best` gives the best CV entry (without refit)
4. `result.final` gives the refit entry specifically

```python
result = nirs4all.run(pipeline, dataset, refit=True)

# CV performance (what was measured during cross-validation)
print(f"RMSECV: {result.cv_best_score:.4f}")

# Refit performance (retrained on all training data, tested on holdout)
print(f"RMSEP: {result.final_score:.4f}")

# Best overall (prefers refit if available)
print(f"Best: {result.best_score:.4f}")
```

---

## 15. Dataset Loading

### Input Formats for `dataset` Parameter

```python
# Path to folder (auto-detected)
result = nirs4all.run(pipeline, dataset="sample_data/regression")

# Numpy arrays as tuple
result = nirs4all.run(pipeline, dataset=(X, y))

# Numpy array (X only, for unsupervised)
result = nirs4all.run(pipeline, dataset=X)

# Dict with arrays
result = nirs4all.run(pipeline, dataset={"X": X, "y": y, "metadata": meta})

# SpectroDataset instance
from nirs4all.data import SpectroDataset
ds = SpectroDataset(...)
result = nirs4all.run(pipeline, dataset=ds)

# Batch: list of datasets (Cartesian product with pipelines)
result = nirs4all.run(pipeline, dataset=[dataset1, dataset2])
```

### Synthetic Data

```python
import nirs4all

# Quick dataset
dataset = nirs4all.generate(n_samples=500, complexity="realistic")

# Regression-specific
dataset = nirs4all.generate.regression(n_samples=500, target_range=(0, 100))

# Classification
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)

# Multi-source
dataset = nirs4all.generate.multi_source(
    n_samples=500,
    sources=[
        {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
        {"name": "markers", "type": "aux", "n_features": 15}
    ]
)

# Just arrays (no SpectroDataset)
X, y = nirs4all.generate(n_samples=500, as_dataset=False)
```

---

## 16. Session API

### Pattern 1: Resource Sharing

Share a PipelineRunner across multiple `nirs4all.run()` calls for efficiency.

```python
import nirs4all

with nirs4all.session(verbose=2, save_artifacts=True) as s:
    r1 = nirs4all.run(pipeline_pls, data, session=s)
    r2 = nirs4all.run(pipeline_rf, data, session=s)
    print(f"PLS: {r1.best_score:.4f}, RF: {r2.best_score:.4f}")
```

### Pattern 2: Stateful Pipeline

Manage a single pipeline lifecycle: train, predict, save, load.

```python
import nirs4all

# Create session with pipeline
session = nirs4all.Session(
    pipeline=[MinMaxScaler(), KFold(n_splits=5), PLSRegression(10)],
    name="MyModel",
    verbose=1,
)

# Train
result = session.run("sample_data/regression")
print(f"RMSE: {result.best_rmse:.4f}")

# Predict on new data
predictions = session.predict(X_new)
print(predictions.values)

# Save
session.save("exports/my_model.n4a")

# Later: load and predict
loaded = nirs4all.load_session("exports/my_model.n4a")
new_preds = loaded.predict(X_new)
```

---

## 17. Common Recipes

### Recipe 1: Basic Regression

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

dataset = nirs4all.generate.regression(n_samples=500, random_state=42)

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"RMSE: {result.best_rmse:.4f}")
print(f"R2: {result.best_r2:.4f}")
```

### Recipe 2: Basic Classification

```python
import nirs4all
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

dataset = nirs4all.generate.classification(n_samples=500, n_classes=3, random_state=42)

pipeline = [
    StandardScaler(),
    StratifiedKFold(n_splits=5),
    RandomForestClassifier(n_estimators=100),
]

result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"Accuracy: {result.best_accuracy:.4f}")
```

### Recipe 3: Compare Preprocessings with `_or_`

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from nirs4all.operators.transforms import SNV, MSC, Detrend

dataset = nirs4all.generate.regression(n_samples=500, random_state=42)

pipeline = [
    {"_or_": [SNV(), MSC(), Detrend()]},
    KFold(n_splits=5),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"Best RMSE: {result.best_rmse:.4f}")
# Check which preprocessing won
print(f"Best model: {result.best}")
```

### Recipe 4: Grid Search n_components

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from nirs4all.operators.transforms import SNV

dataset = nirs4all.generate.regression(n_samples=500, random_state=42)

pipeline = [
    SNV(),
    KFold(n_splits=5),
    {"_or_": [PLSRegression(n_components=n) for n in range(2, 21)]},
]

result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"Best RMSE: {result.best_rmse:.4f}")
for entry in result.top(3):
    print(f"  {entry.get('model_name')}: {entry.get('test_score'):.4f}")
```

### Recipe 5: Model Stacking (branch --> merge --> meta)

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from nirs4all.operators.transforms import SNV, MSC

dataset = nirs4all.generate.regression(n_samples=500, random_state=42)

pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), RandomForestRegressor(n_estimators=50)],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]

result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"Stacking RMSE: {result.best_rmse:.4f}")
```

### Recipe 6: Multi-Source Pipeline

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from nirs4all.operators.transforms import SNV

dataset = nirs4all.generate.multi_source(
    n_samples=500,
    sources=[
        {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
        {"name": "markers", "type": "aux", "n_features": 10},
    ],
    random_state=42,
)

pipeline = [
    {"branch": {
        "by_source": True,
        "steps": {
            "NIR": [SNV()],
            "markers": [MinMaxScaler()],
        },
    }},
    {"merge_sources": "concat"},
    KFold(n_splits=5),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"RMSE: {result.best_rmse:.4f}")
```

### Recipe 7: Augmentation Pipeline

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from nirs4all.operators.transforms import SNV
from nirs4all.operators.augmentation import GaussianAdditiveNoise, WavelengthShift

dataset = nirs4all.generate.regression(n_samples=300, random_state=42)

pipeline = [
    SNV(),
    {"sample_augmentation": GaussianAdditiveNoise()},
    KFold(n_splits=5),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"RMSE with augmentation: {result.best_rmse:.4f}")
```

### Recipe 8: Outlier Filtering (tag + exclude)

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from nirs4all.operators.transforms import SNV
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter

dataset = nirs4all.generate.regression(n_samples=500, random_state=42)

pipeline = [
    SNV(),
    {"exclude": [
        YOutlierFilter(method="iqr", threshold=1.5),
        XOutlierFilter(method="mahalanobis", threshold=3.0),
    ], "mode": "any"},
    KFold(n_splits=5),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"RMSE after filtering: {result.best_rmse:.4f}")
```

### Recipe 9: Export and Predict

```python
import nirs4all
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Train
dataset = nirs4all.generate.regression(n_samples=500, random_state=42)
pipeline = [MinMaxScaler(), KFold(n_splits=5), PLSRegression(n_components=10)]
result = nirs4all.run(pipeline, dataset, verbose=1)

# Export best model
result.export("exports/best_model.n4a")

# Later: predict on new data
new_dataset = nirs4all.generate.regression(n_samples=50, random_state=99)
predictions = nirs4all.predict(model="exports/best_model.n4a", data=new_dataset)
print(f"Predicted {len(predictions)} samples")
print(f"Values: {predictions.values[:5]}")
```

### Recipe 10: Session Workflow (train --> predict --> save)

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Create stateful session
session = nirs4all.Session(
    pipeline=[MinMaxScaler(), KFold(n_splits=5), PLSRegression(n_components=10)],
    name="ProductionModel",
    verbose=1,
)

# Train
train_data = nirs4all.generate.regression(n_samples=500, random_state=42)
result = session.run(train_data)
print(f"Training RMSE: {result.best_rmse:.4f}")

# Predict
new_data = nirs4all.generate.regression(n_samples=50, random_state=99)
predictions = session.predict(new_data)
print(f"Predictions: {predictions.values[:5]}")

# Save for deployment
session.save("exports/production_model.n4a")

# Later: reload and predict
loaded = nirs4all.load_session("exports/production_model.n4a")
fresh_preds = loaded.predict(new_data)
```

---

## 18. Import Quick Reference

```python
# Core API
import nirs4all

# Transforms
from nirs4all.operators.transforms import (
    SNV, MSC, EMSC, RNV,
    SavitzkyGolay, Detrend, Gaussian, Baseline,
    FirstDerivative, SecondDerivative, NorrisWilliams,
    WaveletDenoise, Haar, Wavelet,
    OSC, EPO,
    ReflectanceToAbsorbance, LogTransform,
    CARS, MCUVE,
    Resampler, CropTransformer,
    AirPLS, ArPLS, ASLSBaseline, SNIP, RollingBall,
    ToAbsorbance, FromAbsorbance, KubelkaMunk,
    FlexiblePCA, FlexibleSVD,
    AreaNormalization,
    IntegerKBinsDiscretizer, RangeDiscretizer,
)

# Augmentation
from nirs4all.operators.augmentation import (
    GaussianAdditiveNoise, MultiplicativeNoise, SpikeNoise,
    LinearBaselineDrift, PolynomialBaselineDrift,
    WavelengthShift, WavelengthStretch, LocalWavelengthWarp,
    SmoothMagnitudeWarp, BandPerturbation, BandMasking,
    ChannelDropout, MixupAugmenter, LocalMixupAugmenter,
    ScatterSimulationMSC,
    PathLengthAugmenter, BatchEffectAugmenter,
    Spline_Smoothing, Spline_X_Perturbations, Spline_Y_Perturbations,
    TemperatureAugmenter, MoistureAugmenter,
    ParticleSizeAugmenter, EMSCDistortionAugmenter,
)

# Models
from nirs4all.operators.models import (
    PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS,
    SIMPLS, LWPLS, IntervalPLS, RobustPLS, RecursivePLS,
    KOPLS, FCKPLS, OKLMPLS,
    AOMPLSRegressor, AOMPLSClassifier,
    POPPLSRegressor, POPPLSClassifier,
)

# Splitters
from nirs4all.operators.splitters import (
    KennardStoneSplitter, SPXYSplitter, SPXYFold, SPXYGFold,
    KMeansSplitter, KBinsStratifiedSplitter,
    SystematicCircularSplitter, GroupedSplitterWrapper,
)

# Filters
from nirs4all.operators.filters import (
    YOutlierFilter, XOutlierFilter,
    SpectralQualityFilter, HighLeverageFilter,
    MetadataFilter, CompositeFilter,
)

# Common sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.svm import SVR
```

---

## 19. Architecture Overview

```
nirs4all/
  api/              run(), predict(), explain(), retrain(), session(), generate()
    result.py       RunResult, PredictResult, ExplainResult
  pipeline/
    runner.py       PipelineRunner (main orchestration entry)
    execution/      PipelineOrchestrator (parallel via n_jobs), PipelineExecutor
    config/         PipelineConfigs, PipelineGenerator (expands _or_, _range_, etc.)
    storage/        WorkspaceStore (SQLite), ArrayStore (Parquet)
    bundle/         BundleGenerator/BundleLoader (.n4a export/load)
    steps/          StepParser, ControllerRouter, StepRunner
    trace/          ExecutionTrace, TraceBasedExtractor
  controllers/      @register_controller dispatch
    transforms/     TransformerMixinController, YTransformerMixinController
    models/         SklearnModelController, PyTorchModelController, TF, JAX
    data/           BranchController, MergeController, ExcludeController, TagController
    splitters/      CrossValidatorController
  data/
    dataset.py      SpectroDataset (X, y, metadata, folds, multi-source)
    predictions.py  Predictions facade
    loaders/        CSV, Parquet, Excel, NumPy, MATLAB
    parsers/        FolderParser, ConfigNormalizer
  operators/
    transforms/     SNV, MSC, SavitzkyGolay, Detrend, WaveletDenoise, ...
    models/         AOM-PLS, POP-PLS, PLSDA, IKPLS, OPLS, ...
    splitters/      KennardStone, SPXY, SPXYFold, ...
    filters/        YOutlierFilter, XOutlierFilter, ...
    augmentation/   Noise, baseline, wavelength, spectral, mixup, ...
  sklearn/          NIRSPipeline (SHAP-compatible sklearn wrapper)
  visualization/    PredictionAnalyzer, heatmaps, candlestick charts
  synthesis/        SyntheticDatasetBuilder, ProductGenerator
  workspace/        Workspace management utilities
```

**Execution flow**:

```
nirs4all.run(pipeline, dataset)
  -> api/run.py: normalize inputs, create PipelineRunner
    -> PipelineRunner.run(): load dataset, build PipelineConfigs
      -> PipelineOrchestrator.execute(): expand generators, init WorkspaceStore
        -> For each variant (parallel via joblib if n_jobs>1):
          -> PipelineExecutor.execute(): iterate steps
            -> StepRunner -> StepParser.parse() -> ControllerRouter.route() -> controller.execute()
        -> Refit best variant on full data
        -> Save run metadata + predictions
      -> Return RunResult
```

---

## 20. Commands Reference

```bash
# Tests
pytest tests/                     # All tests
pytest tests/unit/                # Unit only
pytest tests/integration/         # Integration only
pytest -m sklearn                 # sklearn-only (fast)
pytest --cov=nirs4all             # With coverage

# Examples
cd examples && ./run.sh           # All examples
./run.sh -c user                  # User category only
./run.sh -n "U01*"               # By pattern

# Verification
nirs4all --test-install
nirs4all --test-integration

# Code quality
ruff check .                      # Linting
mypy .                            # Type checking
```

---

## 21. Controller Pattern (Extension Point)

Custom operators register via decorator. This is the mechanism for adding new step types.

```python
from nirs4all.controllers import register_controller, OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50  # Lower number = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        """Return True if this controller handles the given step."""
        return isinstance(operator, MyOperatorType)

    @classmethod
    def use_multi_source(cls) -> bool:
        """Whether to iterate over sources independently."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Whether this controller runs during predict()."""
        return True

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        """Execute the step. Return (context, StepOutput)."""
        # Transform dataset here
        ...
```

---

## 22. Key Constraints

- **Python 3.11+** required
- Deep learning backends (TensorFlow, PyTorch, JAX) are lazy-loaded -- not needed for PLS/sklearn pipelines
- YAML serialization: tuples may convert to lists
- All sklearn-compatible transformers and estimators work as pipeline steps
- `_or_`, `_range_`, etc. are evaluated at pipeline expansion time, not at fit time
- Refit is enabled by default (`refit=True`); disable with `refit=False`
- Augmentation (`sample_augmentation`) is applied only during training, not during prediction
- Filters used with `exclude` remove samples from training only; test samples are never excluded
