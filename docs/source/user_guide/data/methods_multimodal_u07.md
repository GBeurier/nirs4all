# Native multimodal early fusion

`MultimodalRegressor(..., backend="methods")` delegates the complete learned
pipeline to nirs4all-methods. The existing `backend="sklearn"` default and its
constructor, cloning, serialization and nested parameter names stay available
without importing the optional native runtime.

The supported profile has four ordered raw sources named `nir`, `image`,
`series`, and `metadata`. IO aligns rows by explicit sample IDs. The NIR source
uses population StandardScaler; image and series use learned unwhitened
TensorPCA; metadata uses StandardScaler on column 0 and learned dense one-hot
encoding on string column 1. Weighted encoded features are concatenated before
Ridge. Methods owns all encoder arithmetic, category vocabularies and model
state. The Python declarations describe the recipe; their sklearn `fit`
methods do not execute for the Methods backend.

Use the canonical U07 declaration with one additional keyword:

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA

model = MultimodalRegressor(
    transformers={
        "nir": StandardScaler(),
        "image": TensorPCA(2, random_state=17),
        "series": TensorPCA(2, random_state=17),
        "metadata": ColumnTransformer([
            ("numeric", StandardScaler(), [0]),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1]),
        ]),
    },
    model=Ridge(alpha=1.0),
    backend="methods",
)
pipeline = [GroupKFold(3), {"model": model, "_grid_": {
    "model__alpha": [0.1, 1.0],
    "source_weights__image": [0.5, 1.0],
    "transformers__image__n_components": [2, 4],
}}]
# training_cohort is an IO MultimodalDataset with complete y and train/test partitions.
result = nirs4all.run(pipeline, training_cohort, engine="dag-ml",
                      workspace_path="workspace", save_charts=False, random_state=17)
try:
    print(result.cv_best_score)
    archive = result.export("complete_multimodal.n4a")
    prediction = nirs4all.predict(archive, new_cohort, verbose=0)
    assert prediction.metadata["training_performed"] is False
finally:
    result.close()
```

Declare raw sources with IO's `TensorSource`, including representations,
coordinate units, coordinates and feature names, as described in
{doc}`multimodal`. Fixed shapes are declared by the actual inputs rather than
by U07's synthetic dimensions. PCA component counts are positive integers and
must fit both the source width and each fold's training row count. Recorded
PCA seeds are integers from 0 through `2**32 - 1`. NIR, RGB image and series
retain their original rank and declared float32/float64 dtype. A source's
non-sample shape product is at most 1,048,576; native raw/encoded/fused buffers
have a 16,777,216-value budget. Metadata may be an original
Unicode/object table: the binding converts only numeric column 0 and sends
category strings as exact UTF-8. It does not learn host category codes.

DAG-ML owns the same grouped folds, native scores, candidate selection and
full REFIT. Each fold learns its encoder vocabulary, scaling and PCA on that
fold's train rows only. Unknown category strings produce zero one-hot features.
Ridge explicitly centers X/y and disables additional feature scaling, matching
ordinary sklearn Ridge after the declared encoders.

For durable global search, remove `_grid_` from the model step and pass its
mapping as `tuning={"engine": "n4m", "sampler": "random", "seed": 17,
"n_trials": 8, "space": space, "storage": storage_uri,
"study_name": "multimodal"}`. The three nested keys above retain their public
meaning. Checkpoint cancellation and `resume=True` use the existing native
search flow; each completed trial releases its controller. This profile
currently requires serial candidate execution (`n_jobs=1`).

The `.n4a` file is Core Archive V2 with a separate
`methods_multimodal_pipeline` RAW artifact. Its N4MF state contains every fitted
encoder, ordered recipe, source schema and Ridge predictor. Export uses the
captured REFIT state and performs no second fit. It stores no learned sklearn
joblib/pickle state. Replay validates Core storage and DAG recipe/artifact
closure, compares independently supplied new input schemas, imports native
state, predicts, then releases handles. The original training workspace is not
needed. Same-shaped inputs with changed dtype, axis units/coordinates or feature
identity are refused; no implicit conversion or interpolation occurs.

The profile requires complete fixed-shape sources, one complete numeric target
named `y`, early fusion, `GroupKFold(3)`, and one full REFIT. Ragged or missing
modalities, per-target fitting, whitening, category dropping/frequency grouping,
external preprocessing, local HPO, learned stacking, DL and fit-control overrides
are outside this profile and fail explicitly.
The portable export/replay route does not expose legacy sessions or a
`results_path` native-results directory for this profile. Install matching Methods, DAG-ML
and Core builds that provide this profile; an older binding without
`n4m.MultimodalPipeline` fails without a numerical fallback.

From the SDK repository, run the real example after the native stack is built:

```bash
python examples/user/02_data_handling/U07_multimodal.py --backend methods --search grid --output /tmp/u07-methods
python examples/user/02_data_handling/U07_multimodal.py --backend methods --output /tmp/u07-search
python examples/user/02_data_handling/U07_multimodal.py --replay /tmp/u07-methods/multimodal.n4a --output /tmp/u07-replay
python examples/user/02_data_handling/U07_multimodal_qualification.py --backend methods --output /tmp/u07-qualification
```

These deterministic synthetic cohorts qualify software behavior. They do not
establish performance or scientific benefit on a real corpus. Host qualification
requires actual Python, WASM/Node, R and Octave execution against the same raw
inputs and native campaign; an absent runtime leaves that host's qualification
open.
