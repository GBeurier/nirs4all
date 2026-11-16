# SHAP Analysis for NIRS Models

## Overview

The SHAP (SHapley Additive exPlanations) module provides explainability for NIRS models by identifying which spectral regions are most important for predictions.

## Key Design Decisions

### ✅ **ALL Visualizations Use Binned Features**

**Every SHAP visualization in this module uses binned wavelengths/features**, not individual points:

- **Spectral importance**: Shows binned regions on the spectrum + bar chart
- **Beeswarm plot**: Bins features before plotting
- **Waterfall plot**: Bins features before showing contributions
- **Summary plot**: Uses raw features (standard SHAP, useful for non-spectral data)

### Why Binning?

Individual wavelengths are prone to:
- **Noise and artifacts** from instrument variability
- **Overfitting** to training set peculiarities
- **Misleading peaks** at single points

**Binning creates robust spectral regions:**
- Aggregates SHAP values over multiple wavelengths
- Smooths out noise while preserving trends
- Provides interpretable regions (e.g., "1600-1620 nm")
- Scientifically meaningful (absorption bands span ranges)

### Binning Configuration

Control binning with these parameters:

```python
shap_params = {
    'bin_size': 20,           # Wavelengths per bin (default: 20)
    'bin_stride': 10,         # Step between bins (default: 10 = 50% overlap)
    'bin_aggregation': 'sum'  # How to combine SHAP values in a bin
}
```

**Per-Visualization Configuration**

You can now specify different binning for each visualization using dictionaries:

```python
shap_params = {
    'bin_size': {
        'spectral': 20,      # Fine detail for overview
        'waterfall': 50,     # Coarser for clarity
        'beeswarm': 30       # Medium detail
    },
    'bin_stride': {
        'spectral': 10,      # 50% overlap
        'waterfall': 50,     # disjoint (required for the waterfall)
        'beeswarm': 15       # 50% overlap
    },
    'bin_aggregation': {
        'spectral': 'sum',      # Total importance
        'waterfall': 'sum',     # Signed sum (required for the waterfall)
        'beeswarm': 'sum_abs'   # Absolute sum
    }
}
```

**Waterfall bins are disjoint.** A waterfall decomposes one prediction from the
base value, so on the general explain lane it always uses disjoint bins
(stride equal to its bin size) with signed sums; bins beyond `max_display` are
shown as one remainder and the plot ends exactly at the prediction. Shared
`bin_stride`/`bin_aggregation` values only configure the spectral and beeswarm
summaries; an explicit `'waterfall'` entry with overlapping bins or a non-sum
aggregation is refused. Waterfall and force figures are also refused for
non-additive explainers (`lime`, `occlusion`) and for the per-row
`interpolate` baseline. Spectral figures label their x axis from the recorded
axis (`Wavelength (nm)`, `Wavenumber (cm-1)`) or as feature indices.

**Aggregation methods:**
- `'sum'` - Sum of SHAP values (emphasizes cumulative effect)
- `'sum_abs'` - Sum of absolute values (ignores direction)
- `'mean'` - Average SHAP value (normalized by bin size)
- `'mean_abs'` - Average absolute value

**Examples:**
- `bin_size=20, bin_stride=10` → 50% overlap between bins
- `bin_size=30, bin_stride=30` → No overlap (independent bins)
- `bin_size=50, bin_stride=25` → 50% overlap with wider regions

## Visualizations

### 1. **Spectral Importance** ⭐ Main Visualization

Shows which spectral regions matter most for predictions.

```{figure} ../../assets/shap_spectral.png
:align: center
:width: 90%
:alt: SHAP Spectral Importance Plot

SHAP spectral importance plot showing important wavelength regions.
```

**Top Panel - Spectrum with Regions:**
- Black line: Mean spectrum of your data
- **Colored bands (Viridis)**: Important regions highlighted
  - Light yellow/green = moderately important
  - Dark blue/purple = highly important
  - No highlight = low importance

**Bottom Panel - Bar Chart:**
- X-axis: Wavelength (nm)
- Y-axis: Aggregated SHAP importance per bin
- **Viridis colormap**: Bars colored by importance
- Blue line: Trend across spectrum

**How it works:**
1. Computes SHAP values for each wavelength
2. Sorts wavelengths to ensure proper ordering
3. Creates overlapping bins (default: 20 wavelengths, 50% overlap)
4. Aggregates SHAP values within each bin
5. Visualizes binned importance

### 2. **Beeswarm Plot (Binned)**

SHAP beeswarm showing feature value vs. SHAP impact, **with binned features**.

```{figure} ../../assets/shap_beeswarm.png
:align: center
:width: 90%
:alt: SHAP Beeswarm Plot

SHAP beeswarm plot showing distribution of SHAP values for binned features.
```

- Each dot = one sample
- X-axis: SHAP value (impact on prediction)
- Color: Feature value (red=high, blue=low)
- Y-axis: Binned wavelength regions (sorted by importance)

**Interpretation:**
- Dense clusters = consistent behavior across samples
- Wide spread = variable impact depending on sample
- Color patterns = how feature values relate to SHAP impact

### 3. **Waterfall Plot (Binned)**

Shows how **binned features** contribute to a single prediction.

```{figure} ../../assets/shap_waterfall.png
:align: center
:width: 90%
:alt: SHAP Waterfall Plot

SHAP waterfall plot showing feature contributions to a single prediction.
```

- Starts from base value (expected value)
- Each bar = contribution from one binned region
- Red bars = push prediction higher
- Blue bars = push prediction lower
- Ends at final prediction

**Use for:**
- Understanding individual predictions
- Debugging specific samples
- Explaining predictions to stakeholders

### 4. **Summary Plot** (Raw Features)

Standard SHAP summary plot showing overall feature importance.

**Note:** This uses **individual features**, not binned. Useful for:
- Comparing different feature types (e.g., spectra + metadata)
- Standard SHAP workflow compatibility
- Non-spectral data analysis

## Usage

### Basic Example

```python
import nirs4all

# Train model
result = nirs4all.run(pipeline=pipeline, dataset="my_data/", verbose=1)
best = result.best

# Explain with SHAP (default binning)
explanation = nirs4all.explain(
    model=best,
    data="my_data/",
    n_samples=200,
    visualizations=['spectral', 'beeswarm', 'waterfall'],
)
```

### Custom Binning (Same for All)

```python
explanation = nirs4all.explain(
    model=best,
    data="my_data/",
    n_samples=200,
    visualizations=['spectral', 'beeswarm', 'waterfall'],
    bin_size=50,              # Wider bins
    bin_stride=25,            # 50% overlap
    bin_aggregation='mean_abs',  # Average absolute SHAP values
)
```

### Custom Binning (Per-Visualization)

```python
shap_params = {
    'n_samples': 200,
    'visualizations': ['spectral', 'waterfall', 'beeswarm'],

    # Different binning for each visualization
    'bin_size': {
        'spectral': 20,      # Fine-grained spectral overview
        'waterfall': 50,     # Coarser - fewer bars for clarity
        'beeswarm': 30       # Medium detail
    },
    'bin_stride': {
        'spectral': 10,      # 50% overlap
        'waterfall': 50,     # disjoint (required for the waterfall)
        'beeswarm': 15       # 50% overlap
    },
    'bin_aggregation': {
        'spectral': 'sum',      # Total importance per region
        'waterfall': 'sum',     # Signed sum (required for the waterfall)
        'beeswarm': 'sum_abs'   # Absolute importance
    }
}

explanation = nirs4all.explain(model=best, data="my_data/", **shap_params)
```

**Why use different binning per visualization?**
- **Spectral**: Fine detail to see all important regions
- **Waterfall**: Coarser disjoint bins → fewer bars → easier to interpret
- **Beeswarm**: Medium bins → balance between detail and readability

### All Parameters

```python
shap_params = {
    # SHAP computation
    'n_samples': 200,              # Background samples (default: 100)
    'explainer_type': 'auto',      # 'auto', 'tree', 'linear', 'deep', 'kernel'
    'explainer_backend': 'shap',   # or 'methods' (see "Methods attribution backend")
    'sample_indices': None,        # rows of the supplied data to explain (default: all)
    'output_index': 0,             # target or class-probability column

    # Visualizations (all support binning)
    'visualizations': ['spectral', 'summary', 'waterfall', 'beeswarm'],

    # Binning configuration - can be int/str OR dict
    'bin_size': 20,                # int: same for all, dict: per-viz
    'bin_stride': 10,              # int: same for all, dict: per-viz
    'bin_aggregation': 'sum'       # str: same for all, dict: per-viz
}
```

## Methods attribution backend

`nirs4all.explain(...)` on the general lane (the default for captured `.n4a`
archives, trained DAG results and `result.best` selections) accepts
`explainer_backend="methods"`. The attributions are then computed by the
nirs4all-methods kernels (`n4m.explain`), the same native implementation used
by the web application through WASM. The default `explainer_backend="shap"`
keeps the `shap` package path unchanged. The explicitly selected
`engine="legacy"` lane refuses the Methods backend and its options.

The backend requires a nirs4all-methods release providing the `n4m.explain`
binding (explain ABI 2.23, nirs4all-methods 1.3.17 or later). An older
installation raises an `ImportError` naming that requirement; there is no
fallback to the `shap` package.

```python
import nirs4all

explanation = nirs4all.explain(
    model="exports/protein.n4a",
    data="my_data/",
    explainer_backend="methods",
    explainer_type="permutation",   # grouped permutation SHAP
    group_width=20,                 # 20 contiguous wavelengths per group
    baseline="interpolate",         # spectral bridge across the occluded band
    n_permutations=10,
    seed=0,
    sample_indices=[3, 8, 15],      # rows of the supplied data to explain
    output_dir="explanations/protein",
)
envelope = explanation.to_explanation_v1()  # nirs4all.explanation.v1
```

### Methods and status

| `explainer_type` | Method | Status | Exactness / additivity |
| --- | --- | --- | --- |
| `"permutation"` | Grouped antithetic permutation SHAP | implemented | sampled, additive by construction, Monte-Carlo standard error |
| `"exact"` | Exact group SHAP over all 2^G coalitions (G ≤ 16) | implemented | exact, additive |
| `"kernel"` | Kernel SHAP (weighted least squares, efficiency enforced) | experimental | sampled, additive |
| `"lime"` | LIME over binary group masks | experimental | local surrogate, not additive |
| `"occlusion"` | Occlusion / ablation per group | implemented | perturbation, not additive |
| `"linear"` | Linear SHAP of an affine estimator (Ridge, LinearRegression, PLSRegression, ...) | implemented | exact, additive |
| `"tree"` | Path-dependent TreeSHAP of scikit-learn trees/forests/gradient boosting, XGBoost and LightGBM regressors, scikit-learn forest class probabilities | implemented | exact, additive; grouped values are sums of feature values |
| `"auto"` | Tree, else linear, else permutation, among the kinds compatible with the options you set | implemented | as selected; recorded in `explainer_type` |
| — | Integrated Gradients, conditional / on-manifold baselines | planned | — |

`n_coalitions` applies to `kernel` and `lime` (`0` lets Methods choose);
`n_permutations` (default 10 antithetic pairs) to `permutation`. Options that do
not apply to the requested method are refused rather than ignored.

### Raw input versus transformed features

* `attribution_space="raw_input"` (default) explains the complete captured
  predictor: preprocessing, estimator and target inverse. Model-agnostic methods
  perturb raw input columns and read predictions in original target units, or
  the selected class probability. `linear` and `tree` explain the estimator
  itself, so in raw space they apply only when the captured predictor has no
  preprocessing step. A target inverse is folded exactly only when it is a
  recognized affine transform (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`,
  an identity `FunctionTransformer`, or a pipeline of these); any other target
  inverse refuses linear and tree SHAP (model-agnostic methods remain available).
* `attribution_space="transformed_features"` explains the terminal estimator
  over the outputs of the fitted preprocessing chain (captured scikit-learn
  pipelines only). Transformed features are labelled `transformed_<j>` with a
  `feature_index` axis: they are not relabelled as wavelengths, even when a
  step preserves the spectral width. The applied steps are recorded in the
  lineage and in `settings.preprocessing`.

Every model-specific export (linear coefficients, flat tree ensembles) and every
transformed-space decomposition must reproduce the captured predictor on the
explained rows; otherwise the request is refused.

### Grouped spectral explanations

Model-agnostic methods treat each **group** of features as one player of the
coalition game (`group_aggregation="group_game"`): the result is the Shapley
value of the band, not a sum of per-wavelength values.

* `group_width=w` builds contiguous groups of `w` features (the last one may be
  shorter).
* `group_ranges=[(start, end), ...]` uses half-open feature-index ranges, for
  example chemically informed absorption bands already mapped to column indices.
  Ranges must be disjoint; each uncovered run becomes a `remainder` group.
* Without either option, model-agnostic methods use one group per feature for
  inputs of at most 20 features and groups of 20 contiguous features otherwise,
  to keep wide spectra tractable. Linear and tree SHAP default to one group per
  feature.

`ExplainResult.shap_values` then has one column per group and
`feature_names` holds the group labels (`"<first>–<last>"`). Figures draw the
groups directly (the default bin size becomes one group).

### Baselines and background

* `baseline="background"` (default): interventional average over the background
  rows.
* `baseline="mean"`: a single reference row, the background column mean.
* `baseline="interpolate"`: each absent band is replaced by the straight line
  joining the explained spectrum at its nearest present neighbours. It keeps
  perturbed spectra continuous for strongly correlated wavelengths, needs no
  background, and gives one expected value per row (`base_value` is then an
  array; waterfall and force figures are refused).

The background is drawn, without replacement and with `seed`, from the supplied
rows: `n_samples` (or `background_samples`, default 100) caps its size. Linear
SHAP uses the background mean as its reference. Path-dependent TreeSHAP uses no
background and refuses `baseline`.

### Results and persisted envelope

Both backends record a `nirs4all.explanation.v1` envelope
(`ExplainResult.to_explanation_v1()`), and write it as `explanation_v1.json`
next to `shap_values.csv`, `provenance.json` and `summary.html` when outputs are
written. The envelope records the engine (`nirs4all-methods` with its version
and ABI, or `python-shap` with the installed `shap` version), exactness,
additivity error, attribution space, explained output (`target` or
`class_probability`), groups, and settings. `rows.sample_indices[i]` is the
index, within the supplied data, of explained row `i`; explained-row positions
and source indices are never interchanged. The axis carries numeric
wavelengths (`nm`) or wavenumbers (`cm-1`) only when a single-source dataset
explicitly declares such headers (`header_unit="nm"` or `"cm-1"`; the implicit
`cm-1` default of headers without a declared unit does not count); otherwise it
is a `feature_index` axis with a null unit.
The Methods backend explains every output column (its `output.index` addresses
the attribution array), while `ExplainResult` keeps the selected `output_index`.
The `shap` backend explains only the selected column, so its envelope holds one
output (`output.index` 0) and records the model column as
`settings.model_output_index`.

### Performance notes

* Model-agnostic cost is dominated by model evaluations: Methods batches the
  perturbed rows (up to 4096 per call) through the captured predictor.
  Permutation SHAP evaluates about `2 * n_permutations * (G + 1) * n_background`
  rows per explained row; exact SHAP `2^G * n_background`. Grouping, the `mean`
  or `interpolate` baselines and a smaller `n_samples` reduce the cost.
* Linear and tree SHAP evaluate no perturbed rows and are fast on wide spectra.

## Interpreting Results

### Spectral Importance

**Example: Protein Prediction Model**

If you see high importance in:
- **1600-1700 nm** (dark purple band): Amide I band → C=O stretch
- **2100-2200 nm** (dark purple band): Amide II band → N-H bend
- **2300-2400 nm** (medium blue band): C-H combinations

✅ **Model learned chemically meaningful features!**

**Troubleshooting:**

- **High importance in unexpected regions** → May indicate artifacts or preprocessing issues
- **Uniform importance everywhere** → Model might be overfitting or data is too noisy
- **Importance at spectrum edges** → Check for edge effects from preprocessing

### Beeswarm/Waterfall (Binned)

Look for:
- **Consistent patterns** → Reliable spectral regions
- **Variable contributions** → Context-dependent regions
- **Strong push in one direction** → Dominant spectral features

## Scientific Interpretation

### Validation Strategy

1. **Run SHAP analysis** on best model
2. **Identify top regions** from spectral importance
3. **Cross-reference with chemistry**:
   - Do important regions match known absorption bands?
   - Are they consistent with the target property?
4. **Compare models**: Do different models rely on similar regions?

### Example Workflow

```python
import nirs4all

# Explain top 3 models
top_models = result.top(n=3)

for model in top_models:
    explanation = nirs4all.explain(model=model, data="my_data/", **shap_params)
    # Compare which regions are consistently important
```

## Advanced Usage

### Experiment with Bin Sizes

```python
import nirs4all

# Try different bin sizes to find optimal resolution
for bin_size in [10, 20, 30, 50]:
    explanation = nirs4all.explain(
        model=best,
        data="my_data/",
        n_samples=200,
        bin_size=bin_size,
        bin_stride=bin_size // 2,  # Always 50% overlap
    )
    # Compare results
```

### Compare Aggregation Methods

```python
import nirs4all

for agg in ['sum', 'sum_abs', 'mean', 'mean_abs']:
    explanation = nirs4all.explain(
        model=best,
        data="my_data/",
        n_samples=200,
        bin_aggregation=agg,
    )
    # See which gives clearest insights
```

## Technical Details

### SHAP Value Calculation

1. Select appropriate explainer (Tree/Linear/Deep/Kernel)
2. Compute SHAP values for each sample × feature
3. Store raw values for later binning

### Binning Process (for spectral/beeswarm/waterfall)

1. Extract wavelengths from feature names (λXXX.X format)
2. Sort features by wavelength
3. Create overlapping bins:
   ```python
   bin_start = 0, bin_stride, 2*bin_stride, ...
   bin_end = bin_start + bin_size
   ```
4. Aggregate SHAP values per bin using selected method
5. Create bin labels (e.g., "1650.0-1670.0 nm")
6. Generate visualization with binned data

### Explainer Selection

- **Tree models** (RF, GBM, XGBoost): TreeExplainer (fast, exact)
- **Linear models** (Ridge, Lasso, PLS): LinearExplainer (fast)
- **Neural networks**: DeepExplainer
- **Others**: KernelExplainer (slower but universal)

## Plotting Behavior

**All plots are blocking** - execution pauses until you close the plot window. This allows you to:
- Examine each visualization carefully
- Save screenshots manually
- Compare visualizations side-by-side

Plots are also **automatically saved** to:
```
results/<dataset>/<config>/explanations/<model_id>/
  ├── spectral_importance.png
  ├── summary.png
  ├── waterfall_binned.png
  └── beeswarm_binned.png
```

## FAQ

**Q: Why aren't individual wavelengths shown in beeswarm/waterfall?**
A: Individual wavelengths are too noisy. Binning creates robust, interpretable regions.

**Q: How do I choose bin_size?**
A: Start with 20 (default). Increase for broader patterns, decrease for finer detail.

**Q: What's the difference between sum and mean aggregation?**
A: Sum emphasizes cumulative importance, mean normalizes by bin size.

**Q: Can I use raw SHAP values without binning?**
A: Yes - use `plot_summary()` which shows individual features. But for spectral data, binning is strongly recommended.

**Q: Why does the spectral plot use Viridis colormap?**
A: Viridis is perceptually uniform, colorblind-friendly, and works well in grayscale.

## References

- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (NIPS)
- SHAP documentation: https://shap.readthedocs.io/
- Viridis colormap: https://matplotlib.org/stable/tutorials/colors/colormaps.html

```{seealso}
**Related Examples:**
- [U01: SHAP Basics](../../../examples/user/07_explainability/U01_shap_basics.py) - SHAP waterfall and spectral plots
- [U02: SHAP with sklearn](../../../examples/user/07_explainability/U02_shap_sklearn.py) - Custom SHAP explainers
- [U03: Feature Selection](../../../examples/user/07_explainability/U03_feature_selection.py) - CARS and MC-UVE selection
```
