# Extended Python API

The config-first workflow is portable. Python also exposes direct objects, controllers, and helper functions for advanced users.

## Public API Map

| Need | Use |
| --- | --- |
| Train/evaluate | `nirs4all.run(pipeline, dataset, ...)` |
| Predict | `nirs4all.predict(model=..., data=...)` or `nirs4all.predict(chain_id=..., data=...)` |
| Explain | `nirs4all.explain(model=..., data=...)` |
| Retrain | `nirs4all.retrain(source=..., data=..., mode="full")` |
| Reuse runner/workspace | `with nirs4all.session(...) as s:` |
| Generate test data | `nirs4all.generate.regression(...)`, `nirs4all.generate.classification(...)` |
| Build config explicitly | `PipelineConfigs`, `DatasetConfigs` |
| Low-level execution | `PipelineRunner` |
| Custom node dispatch | `register_controller`, `OperatorController` |

## Python Object Pipeline

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from nirs4all.operators.transforms import SNV, SavitzkyGolay

pipeline = [
    MinMaxScaler(),
    SNV(),
    SavitzkyGolay(window_length=15, polyorder=2, deriv=1),
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"model": PLSRegression(n_components=10)},
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="dataset.yaml",
    random_state=42,
)
```

## Typical Operator Imports

```python
from nirs4all.operators.transforms import (
    SNV,
    RNV,
    MSC,
    EMSC,
    SavitzkyGolay,
    FirstDerivative,
    SecondDerivative,
    NorrisWilliams,
    Detrend,
    ASLSBaseline,
    ToAbsorbance,
    Resampler,
    CARS,
    MCUVE,
)

from nirs4all.operators.augmentation import (
    GaussianAdditiveNoise,
    WavelengthShift,
    BandMasking,
    MixupAugmenter,
    PathLengthAugmenter,
)

from nirs4all.operators.filters import (
    YOutlierFilter,
    XOutlierFilter,
    SpectralQualityFilter,
    MetadataFilter,
)

from nirs4all.operators.splitters import (
    KennardStoneSplitter,
    SPXYSplitter,
    SPXYFold,
)

from nirs4all.operators.models import (
    AOMPLSRegressor,
    POPPLSRegressor,
    PLSDA,
    OPLS,
    MBPLS,
    DiPLS,
    TabPFNNIRSRegressor,
)
```

## Recipe: Merge Two Sources

```python
from nirs4all.operators.transforms import MSC, SNV
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    {"branch": {
        "by_source": True,
        "steps": {
            "NIR": [SNV()],
            "MIR": [MSC()],
        },
    }},
    {"merge": {"sources": "concat"}},
    {"model": PLSRegression(n_components=12)},
]

result = nirs4all.run(pipeline, "multi_source_dataset.yaml")
```

Portable YAML equivalent: {doc}`/reference/nodes/merge`.

## Recipe: Cartesian Preprocessing Search

```python
pipeline = [
    {"_cartesian_": {
        "scatter": {"_or_": [
            {"class": "nirs4all.operators.transforms.StandardNormalVariate"},
            {"class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"},
            None,
        ]},
        "derivative": {"_or_": [
            None,
            {
                "class": "nirs4all.operators.transforms.SavitzkyGolay",
                "params": {"window_length": 15, "polyorder": 2, "deriv": 1},
            },
        ]},
    }},
    {"model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {"n_components": {"_range_": [5, 15, 5]}},
    }},
]
```

See {doc}`/reference/nodes/generators`.

## Recipe: Session for Multiple Runs

```python
import nirs4all

with nirs4all.session(verbose=1, workspace_path="workspace") as s:
    pls = nirs4all.run("pls_pipeline.yaml", "dataset.yaml", session=s)
    rf = nirs4all.run("rf_pipeline.yaml", "dataset.yaml", session=s)

print(pls.best_score, rf.best_score)
```

## Recipe: Stored Chain Prediction

```python
pred = nirs4all.predict(
    chain_id="stored-chain-id",
    data=X_new,
    workspace_path="workspace",
)
```

For exported bundles:

```python
pred = nirs4all.predict(
    model="exports/best_model.n4a",
    data="new_dataset.yaml",
)
```

## Recipe: Conformal Intervals and Robustness Summary Artifacts

The native conformal/robustness lane has two distinct layers:

- conformal calibration produces prediction intervals and a guarantee-status
  payload for the calibrated cohort;
- robustness reports are audit artifacts. They summarize what happened under
  observed or perturbed scenarios, but they do not recalibrate the model and do
  not renew a conformal guarantee.

```python
import nirs4all

calibrated = nirs4all.calibrate(
    calibration_data={
        "y_true": y_calibration,
        "y_pred": y_calibration_pred,
        "sample_ids": calibration_sample_ids,
    },
    coverage=0.9,
    workspace_path="workspace/",
    workspace_conformal_id="pls-moisture-conformal",
)

pred = nirs4all.predict_calibrated(
    calibrated,
    y_pred=y_new_pred,
    prediction_sample_ids=new_sample_ids,
)

report = pred.robustness(
    y_true=y_new_observed,
    scenarios=[
        {"kind": "observed", "severity": 0.0},
        {"kind": "prediction_noise", "severity": 0.02, "distribution": "normal"},
        {"kind": "spectral_offset", "severity": 0.01},
        {"kind": "spectral_scale", "severity": 0.03},
        {"kind": "spectral_slope", "severity": 0.02},
        {"kind": "spectral_shift", "severity": 0.5},
    ],
    slice_by=["Instrument", "Site"],
    workspace_path="workspace/",
    workspace_robustness_id="pls-moisture-robustness",
)
```

Use the full report for audit/release evidence:

```python
report.save_artifacts("artifacts/robustness/pls-moisture")
```

The artifact directory contains a verified full report plus a lightweight
`summary.json` payload intended for CI, bindings, Studio, dashboards, and release
cards:

```python
summary = report.summary_artifact()
report.save_summary("artifacts/robustness/pls-moisture/summary.json")
schema = nirs4all.get_robustness_summary_schema()
```

`summary.json` is a stable, JSON-compatible contract with:

| Field | Meaning |
| --- | --- |
| `format` | Always `nirs4all.robustness.summary`. |
| `schema_version` | Summary payload schema version, currently `1`. |
| `fingerprint` | Fingerprint of the verified full `RobustnessReport`. |
| `mode` | Audit mode: `clean_frozen`, `matched_recalibration`, or `structural_refit`. |
| `report_version` | Full report contract version. |
| `slice_by` | Metadata columns used for slice diagnostics. |
| `summary` | One compact row per scenario/severity cell. |

Each summary row contains point metrics (`rmse`, `mae`, `bias`,
`max_abs_error`), deltas versus the reference scenario, optional ratios, worst
slice information, and optional conformal diagnostics such as
`conformal_min_observed_coverage`, `conformal_max_abs_coverage_gap`, and
`conformal_mean_width_mean` when the input prediction object carries conformal
intervals.

The key interpretation rule is fail-loud: robustness metrics are diagnostics.
They can show degradation, coverage drift, or slice-specific risk, but they do
not change the fitted predictor, the stored conformal calibrator, or the
declared marginal conformal guarantee. If a training parameter, preprocessing
graph, model, calibration cohort, or physical sample identity changes, create a
new calibration/report pair instead of reusing the old artifacts.

For native tuning plus conformal calibration, use
{doc}`/user_guide/models/native_tuning_conformal` as the operational reference.
Its `Native keyword/effect quick map` links each supported keyword to its
runtime effect, published evidence and fail-closed boundary, including the
workspace/store `Predictions` bridge into native `PredictResult` diagnostics.

## Custom Operators

The easiest extension point is an sklearn-compatible object:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MeanCenter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        return X - self.mean_

pipeline = [
    MeanCenter(),
    {"model": model},
]
```

For portable YAML/JSON, put the class in an importable module:

```yaml
pipeline:
  - class: my_project.transforms.MeanCenter
```

## Custom Controllers

Use a controller when you need a new pipeline keyword rather than a standard sklearn-style operator.

```python
from nirs4all.controllers import OperatorController, register_controller
from nirs4all.pipeline.execution.result import StepOutput

@register_controller
class MyKeywordController(OperatorController):
    priority = 5

    @classmethod
    def matches(cls, step, operator, keyword):
        return keyword == "my_keyword"

    @classmethod
    def use_multi_source(cls):
        return True

    @classmethod
    def supports_prediction_mode(cls):
        return True

    def execute(self, step_info, dataset, context, runtime_context, source=-1, mode="train", loaded_binaries=None, prediction_store=None):
        # Update dataset/context here.
        return context, StepOutput()
```

Then use:

```python
pipeline = [
    {"my_keyword": {"option": 1}},
    {"model": model},
]
```

For internals, see {doc}`/developer/controllers` and {doc}`/developer/pipeline_architecture`.
