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
