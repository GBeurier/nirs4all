# Pipeline Keywords Reference

Pipeline steps in nirs4all can be plain operators (class or instance) or **dict-wrapped steps** with special keywords that control how the operator is applied. This page documents the core workflow keywords and the versioned lifecycle vocabulary used for tuning, training, retraining, and planned conformal calibration.

## Quick Reference

| Keyword | Purpose |
|---------|---------|
| `model` | Define model step |
| `y_processing` | Target (y) scaling with automatic inverse during prediction |
| `tag` | Mark samples with a tag without removing them |
| `exclude` | Remove flagged samples from training |
| `branch` | Parallel sub-pipelines (duplication) or sample splitting (separation) |
| `merge` | Combine branch outputs |
| `sample_augmentation` | Data augmentation applied to training samples |
| `feature_augmentation` | Feature-level augmentation with multiple action modes |
| `concat_transform` | Concatenate features from multiple transforms |
| `rep_to_sources` | Convert repetition groups to multi-source format |
| `rep_to_pp` | Convert repetition groups to preprocessing pipelines |
| `name` | Name a pipeline step for display and identification |

---

## Lifecycle keyword and effect registry

The following table is generated from the machine-readable
`nirs4all.pipeline.keyword_registry` module. Documentation and future Studio
forms consume the same descriptive records. The runtime parser and execution
engines do **not** import the registry, so adding this table does not change
pipeline behavior.

Statuses are contractual:

- **supported** means the named surface exists, subject to the per-engine cell;
- **partial** means the public surface exists but its stated behavior is not yet
  proven for every relevant model or engine;
- **planned** means the syntax is reserved for the roadmap and is not executable.

In the generated `Engine support` column, `legacy`, `dag-ml`, and `dual` name
execution backends; `optuna` and `n4m` name optimizer engines. An entry can list
both namespaces when the effect crosses that boundary. `legacy fallback` means
that selecting DAG-ML currently delegates the supported shape to the legacy
runtime instead of executing the feature natively.

```{keyword-effects}
```

(model-local-hpo-finetune-params)=
### Model-local HPO: `finetune_params`

`finetune_params` is the historical spelling for **model-local hyperparameter
optimization**. It does not mean continuing from trained weights.

```python
{
    "model": PLSRegression(),
    "finetune_params": {
        "engine": "n4m",
        "sampler": "tpe",
        "eval_mode": "mean",
        "n_trials": 30,
        "model_params": {
            "n_components": ("int", 2, 30),
        },
    },
}
```

The legacy Python execution path supports the full adaptive form. For
`run(engine="dag-ml")`, only the deterministic subset described below is native;
adaptive n4m/Optuna controls remain a typed boundary and must not be assumed to
run inside DAG-ML.

The W2 native integration now has an internal
`nirs4all.pipeline.dagml.native_client` seam that can call the installed
`dag_ml.execute_training()` and `dag_ml.replay_loaded_predictor_package()`
bindings when they are available. The replay seam forwards the signed
`TrainingReplayRequest` unchanged, so the native DAG-ML package runtime owns the
`phase` semantics: `PREDICT` returns final bound predictions and `EXPLAIN`
returns explanation blocks, optionally with final bound predictions for the
requested output bindings.

The same boundary applies to DAG-ML D10 cache namespace proofs. If a native
training request, replay request, execution bundle or prediction-cache payload
contains `cache_namespace_fingerprints`, nirs4all forwards those fingerprints
unchanged to `dag_ml`; DAG-ML owns validation, handle derivation, persistent
payload naming and manifest exposure. There is no public nirs4all keyword that
edits this proof: changing the candidate, data identity, fold, trial or seed
must happen before the DAG-ML contract is signed.

Public `run(engine="dag-ml")` also lowers the
deterministic model-local subset of `finetune_params` to the existing native
DAG-ML generator path before routing:

```python
run(
    [
        StandardNormalVariate(),
        KFold(n_splits=5, shuffle=True, random_state=42),
        {
            "model": PLSRegression(),
            "finetune_params": {
                "engine": "dag-ml",
                "metric": "rmse",
                "direction": "minimize",
                "model_params": {
                    "n_components": [5, 10, 15],
                    "scale": [True, False],
                },
            },
        },
    ],
    "dataset.csv",
    engine="dag-ml",
)
```

Accepted native `model_params` forms are:

- a plain JSON grid, lowered to step-level `_grid_`;
- a per-parameter `{"_range_": [start, stop, step]}` list form;
- a per-parameter `{"_log_range_": [start, stop, count]}` list form.

`metric` selects the native DAG-ML selection metric. Public
`run(engine="dag-ml")` currently accepts the metrics supported by both native
selection surfaces without implicit fallback: `rmse` for regression and
`accuracy`/`balanced_accuracy` for classification. `direction` must agree with
that metric's native objective (`rmse` minimizes;
`accuracy`/`balanced_accuracy` maximize). Direction overrides that contradict
the metric are rejected rather than silently inverted. Broader core/training
metrics (`mse`, `mae`, `r2`) remain internal training-contract work until CLI
and in-process public selection metric parity is closed. Adaptive keys such as
`n_trials`, `sampler`, `pruner`, phases, and `engine="n4m"`/`"optuna"` are known
model-local HPO controls for the legacy Optuna/n4m paths, but they are still
refused on the deterministic DAG-ML native lowering path until the pipeline
compiler, objective runner, optimizer adapters, and typed tuning result are all
closed.

| `finetune_params` key | Lifecycle/effect | Optuna | n4m | DAG-ML deterministic lowering |
| --- | --- | --- | --- | --- |
| `model_params` | Defines model-local candidate parameters; changing it changes candidates, selection and the final predictor, so existing calibration is stale. | Adaptive DSL supported. | Adaptive DSL supported. | Partial: plain JSON grids and `_range_`/`_log_range_` generator specs only. |
| `metric` | Selects trial ranking/selection metric; invalidates calibration if the selected predictor changes. | Supported. | Supported. | Partial: public `rmse`, `accuracy`, `balanced_accuracy`. |
| `direction` | Selects minimize/maximize objective; invalidates calibration if winner changes. | Supported. | Supported. | Partial: must agree with the native metric objective. |
| `n_trials` | Sets adaptive trial budget and may change the winner. | Supported. | Supported. | Unsupported: deterministic candidate set comes from the grid/range. |
| `sampler` | Selects adaptive trial sequence and may change the winner. | Supported. | Partial: some names remap internally. | Unsupported. |
| `pruner` | Selects adaptive pruning/early stopping and may change the winner. | Supported. | Partial: overlapping but not identical pruner vocabulary. | Unsupported. |
| `approach` | Controls fold search strategy and may change trial ranking/winner. | Supported. | Partial. | Partial: only `grouped`. |
| `eval_mode` | Aggregates trial scores and may change ranking/winner. | Partial. | Partial. | Partial: only `mean` and `best`. |
| `train_params` | Configures trial-fit kwargs, not terminal refit kwargs. | Supported. | Supported. | Unsupported until optimizer adapters preserve trial fit kwargs. |

(execution-engine-versus-optimizer-engine)=
### Execution engine versus optimizer engine

The same token has two independent scopes:

- `run(engine="legacy" | "dag-ml")` selects the **pipeline execution backend**;
- `finetune_params["engine"]` selects the **model-local HPO/generation
  driver**:
  - `"optuna"` or `"n4m"` request adaptive optimization;
  - `"dag-ml"` or `"grid"` request deterministic native DAG-ML generation over
    `model_params`.

The planned `tuning["engine"]` will also select an HPO driver. It must never be
used as an alias for `run.engine`. Likewise, `finetune_params["engine"] =
"dag-ml"` does not switch the whole run to DAG-ML; it only declares that the
model-local search space is deterministic and can be represented as native
DAG-ML generators when the execution backend is already DAG-ML. The accepted
value `run(engine="dual")` is a reserved side-by-side comparison mode and
currently raises `NotImplementedError`; the registry therefore marks that
backend as planned.

(canonical-aliases)=
### Canonical aliases

New code, documentation, manifests, and Studio forms use:

- `finetune_params.engine="dag-ml"`; the old spellings `"dagml"` and `"native"`
  remain readable for the deterministic DAG-ML generation subset;
- `finetune_params.engine="n4m"`; the old spellings `"methods"` and `"libn4m"`
  remain readable during migration;
- `sampler`; the old key `sample` remains readable during migration;
- `eval_mode="mean"`; the old value `"avg"` remains readable during migration.

Aliases are read-only compatibility inputs. Canonical exports must not emit
`dagml`, `native`, `methods`, `libn4m`, `sample`, or `avg`.

The current status is partial: the shared n4m objective adapter now passes
`sampler="grid"` to a native `GRID` enum when the installed bindings expose it,
and otherwise fails closed with an explicit enum error. Other n4m sampler names
such as `binary` and `ternary`, plus `best` versus `robust_best`, are not
behaviorally distinct in all ranking paths. These names must not be advertised
as separate guarantees until capability tests prove them.

For public tooling, `nirs4all.FINETUNE_ENGINES`,
`FINETUNE_ENGINE_ALIASES`, `FINETUNE_SAMPLER_KEY_ALIASES` and
`FINETUNE_EVAL_MODE_ALIASES` expose the registry-level vocabulary and read-only
migration aliases. Engine-specific controls are split intentionally:
`FINETUNE_OPTUNA_SAMPLERS`/`FINETUNE_OPTUNA_PRUNERS` mirror the legacy Optuna
validator, `FINETUNE_N4M_SAMPLERS`/`FINETUNE_N4M_PRUNERS` mirror the native n4m
validator, and `FINETUNE_DAGML_DETERMINISTIC_ENGINES`,
`FINETUNE_DAGML_META_KEYS`, `FINETUNE_DAGML_SELECTION_METRICS`,
`FINETUNE_DAGML_APPROACHES` and `FINETUNE_DAGML_EVAL_MODES` describe only the
deterministic native DAG-ML lowering subset. Studio, bindings and generators
should use these constants rather than assuming one global finetuning capability
matrix.

(three-training-parameter-scopes)=
### Three training-parameter scopes

These similarly named dictionaries act at different lifecycle stages:

1. `finetune_params.train_params` configures or samples fit arguments inside
   HPO trials.
2. Step-level `train_params` configures ordinary legacy training and the
   terminal fit.
3. Step-level `refit_params` overrides `train_params` only for the selected
   legacy winner's refit; missing values inherit from `train_params`.

The current DAG-ML lowering rejects all fit-argument scopes that it cannot
preserve: `finetune_params.train_params`, step-level `train_params`, and
`refit_params`. Their DAG-ML support is explicitly `unsupported`, not partial.
On the legacy path all three can change the deployed predictor. A conformal
calibrator fitted before such a change is stale and must not be reused.

(planned-full-dag-tuning)=
### Full-DAG tuning

`nirs4all.run(..., tuning={...})` is the fixed-topology full-DAG HPO surface.
Its current status is **partial**. The executable public subset currently
requires:

- `engine="dag-ml"`;
- an explicit array dataset, `(X, y)`, `[X, y]`,
  `(X, y, sample_ids, groups, metadata)`,
  `[X, y, sample_ids, groups, metadata]`, or a mapping such as `{"X": X, "y": y}`;
- an explicit `SpectroDataset` mapping
  `{"dataset": ds, "selector": {...}, ...}` for a selected fit cohort;
- either a single estimator or a linear transformer→estimator chain accepted by
  `tune_single_estimator()`, optionally wrapped as `{"steps": [...]}` or the
  legacy public alias `{"pipeline": [...]}`;
- `tuning.score_data` with an explicit score cohort, as a mapping or tuple/list.

All broader forms still fail closed with `DagMLTuningNotImplementedError` or a
specific validation error. In particular, `engine="legacy"` with `tuning`,
splitters, branches, dataset loaders, model aliases, non-linear preprocessing
graphs, and calibration forms that supply their own `calibration_data` inside
`run()` are not public yet.

The typed optimizer shape is:

```python
{
    "engine": "n4m",          # or "optuna"; optimizer driver, not run.engine
    "space": {
        "model.n_components": [2, 3, 4],
        "preprocess.savgol.window_length": {"type": "int", "low": 5, "high": 15, "step": 2},
    },
    "force_params": {"model.n_components": 3},  # optional first optimizer trial
    "metric": "rmse",
    "direction": "minimize",
    "n_trials": 50,
    "sampler": "tpe",
    "pruner": "median",
    "seed": 42,
    "resume": False,
}
```

For public tooling, `nirs4all.TUNING_ENGINES` and
`nirs4all.TUNING_DIRECTIONS` expose the strict `DagMLTuningSpec` optimizer
engine and objective-direction vocabularies. `nirs4all.TUNING_CONTRACT_KEYS`
lists the keys that enter the deterministic tuning fingerprint, while
`nirs4all.TUNING_RUNTIME_KEYS` lists the runtime wrapper keys accepted by
`run(tuning=...)` but excluded from that strict tuning contract.
`force_params` is part of the strict contract: it enqueues a caller-provided
warm-start assignment as the first optimizer trial. Its keys must be a subset of
`space`, and categorical values use the public decoded syntax shown in
`space`/`NativeTuning`, not Optuna or n4m internal labels.
For the supported shared objective subset, Optuna and n4m publish the same
decoded warm-start semantics: the first trial row, winning `best_params` when
applicable, and `search_space_fingerprint` are stable nirs4all values even if a
backend uses private labels or encoded categorical tokens internally.
Before adapter dispatch, `space` is lowered to an ordered search-space contract:
dotted paths and sklearn double-underscore paths are canonicalized to the same
dotted spelling, duplicates after canonicalization are refused, and every
candidate or `force_params` assignment becomes a `ParameterPatch` with a
canonical string path and JSON-native value.
Trial diagnostics expose `search_space_fingerprint` alongside the tuning
fingerprint so the score tape records the exact ordered parameter contract.
`nirs4all.inspect_tuning_space(...)` exposes this same JSON-native
`nirs4all.tuning.ordered_search_space` artifact before execution for CLIs,
bindings, Studio previews and documentation examples. Its static JSON Schema is
available through `get_tuning_space_schema()` and
`tuning_space_schema_json()`, and the CLI can publish the same schema with
`nirs4all tuning-space --schema`.
Direct construction of `SearchSpaceParameter`, `ParameterPatch`,
`OrderedSearchSpaceSpec`, `DagMLTuningSpec`, `TrialResult` and `TuningResult`
is fail-closed: paths are canonicalized, values must stay TCV1 JSON-native,
scores must be finite, integer contract fields reject booleans, trial ids are
unique, and candidate/best params must stay inside `tuning.space` before a
fingerprint or summary artifact can be published. `TuningResult.from_dict()`
and `load_json()` apply the same strict `best_value` boundary, so booleans and
numeric strings fail closed instead of being parsed as floats.

The executable array subset adds runtime-only blocks that are not part of the
deterministic optimizer fingerprint:

```python
result = nirs4all.run(
    pipeline=[{"model": estimator}],
    dataset={
        "X": X_dev,
        "y": y_dev,
        "sample_ids": train_sample_ids,
        "groups": train_groups,
        "metadata": train_metadata,
    },
    engine="dag-ml",
    workspace_path="workspace/",
    tuning={
        "engine": "optuna",
        "space": {"alpha": [0.1, 1.0]},
        "force_params": {"alpha": 0.1},
        "metric": "rmse",
        "direction": "minimize",
        "sampler": "grid",
        "n_trials": 2,
        "resume": False,
        "score_data": {
            "X": X_score,
            "y": y_score,
            "sample_ids": score_sample_ids,
            "groups": score_groups,
            "metadata": score_metadata,
        },
        "winner": {
            "X": X_external_test,
            "y_true": y_external_test,
            "score": 0.42,
            "metric": "rmse",
            "sample_ids": physical_sample_ids,
        },
        "workspace_tuning_id": "single-estimator-hpo",
        "workspace_metadata": {"purpose": "first-public-run-tuning-subset"},
    },
    calibration={
        "y_pred": new_point_predictions,
        "prediction_sample_ids": new_sample_ids,
        "coverage": 0.9,
        "workspace_conformal_id": "single-estimator-conformal",
    },
)
```

The machine-readable keyword registry exposes structured schemas for these
runtime-only blocks. `run.tuning.score_data` advertises explicit `X`/`y`,
`X_score`/`y_score`, dataset-backed `dataset` + `selector`, and tuple forms.
`run.tuning.winner` advertises explicit `X`/`y_true` and dataset-backed forms.
`run.tuning.calibration` advertises `y_pred`, `prediction_sample_ids`, single or
multi-coverage, conformal method/unit constants, workspace metadata, and an
explicit `calibration_data: false` schema entry because calibration evidence is
derived from `winner`. Studio and bindings should use these registry schemas to
build forms, then still rely on runtime validation for Python objects and array
shape checks.

The same public subset can be expressed with typed helper objects. These helpers
only normalize syntax; they do not widen the supported runtime. Their
``to_dict()`` output is the same mapping consumed by ``run(tuning=...)``.

```python
tuning = nirs4all.NativeTuning(
    engine="optuna",
    space={"scale": [nirs4all.TuningPassthrough()], "model.alpha": [0.1, 1.0]},
    force_params={"scale": nirs4all.TuningPassthrough(), "model.alpha": 0.1},
    metric="rmse",
    direction="minimize",
    sampler="grid",
    n_trials=2,
    score_data=nirs4all.TuningScoreData(
        X=X_score,
        y=y_score,
        sample_ids=score_sample_ids,
        groups=score_groups,
        metadata=score_metadata,
    ),
    winner=nirs4all.TuningWinner(
        X=X_external_test,
        y_true=y_external_test,
        score=0.42,
        metric="rmse",
        sample_ids=physical_sample_ids,
    ),
    workspace_tuning_id="single-estimator-hpo",
    workspace_metadata={"purpose": "typed-public-run-tuning-subset"},
)

result = nirs4all.run(
    pipeline=[{"model": estimator}],
    dataset=(X_dev, y_dev, train_sample_ids, train_groups, train_metadata),
    engine="dag-ml",
    workspace_path="workspace/",
    tuning=tuning,
    calibration=nirs4all.TuningCalibration(
        y_pred=new_point_predictions,
        prediction_sample_ids=new_sample_ids,
        coverage=0.9,
        workspace_conformal_id="single-estimator-conformal",
    ),
)
```

`TuningCalibration(...)` validates `coverage` as a scalar in `(0, 1)` or a
non-empty unique list of such values, normalizes `method` and `unit` to the
supported lower-case conformal vocabulary, and rejects unsupported values before
runtime execution; the mapping form follows the same registry constants.
`TuningCalibration.extra={...}` may carry additional calibration options, but it
cannot override typed calibration keys (`coverage`, `method`, `unit`,
`prediction_sample_ids`, `y_pred`, workspace fields) or provide
`calibration_data`. Extra/workspace metadata keys must be canonical non-empty
strings, their values must stay strict JSON-native and finite, and
`as_predict_result` must be a boolean. Non-finite numbers, bytes, tuples, sets
and arbitrary Python objects fail before publication.

A single declarative sklearn model can use the same class-path form without a
`steps` wrapper:

```python
result = nirs4all.run(
    pipeline={
        "name": "ridge",
        "class": "sklearn.linear_model.Ridge",
        "params": {"fit_intercept": False},
    },
    dataset=(X_dev, y_dev),
    engine="dag-ml",
    tuning=nirs4all.NativeTuning(
        engine="optuna",
        space={"ridge.alpha": [0.0, 0.1]},
        sampler="grid",
        n_trials=2,
        score_data=nirs4all.TuningScoreData(X=X_score, y=y_score),
    ),
)
```

`NativeTuning.to_tuning_spec()` returns the normalized deterministic optimizer
contract. Runtime blocks such as `score_data`, `winner`, `calibration`,
`workspace_tuning_id`, and `workspace_metadata` remain outside that fingerprint.
It applies the same public validation as raw `tuning` mappings: Optuna
`storage` must be an explicit URI such as `sqlite:///study.db`, and
`study_name` is trimmed and cannot contain NUL characters.
`TuningPassthrough()` serializes to the structured JSON-native
`{"kind": "passthrough"}` marker for optional non-final preprocessing steps.
That marker is exact: `kind` must be the literal string `"passthrough"` and
objects that merely stringify to that value are rejected before native tuning.
`TuningScoreData` and `TuningWinner` also accept the explicit dataset-backed
form through `dataset=...` and `selector=...`; selector omission is rejected,
and typed dataset-backed helpers reject mixed `dataset` + explicit `X`/`y`
or `X`/`y_true` arrays.

Tuple datasets can also carry fit identities without switching to a mapping:

```python
dataset = (X_dev, y_dev, train_sample_ids, train_groups, train_metadata)
```

The tuple order is fixed. Tuples longer than five fields are rejected as
ambiguous, and `sample_ids`, `groups`, and `metadata` must be row-aligned with
`y` when present.

`score_data` accepts the same explicit identity convention for the scoring
cohort:

```python
"score_data": (X_score, y_score, score_sample_ids, score_groups, score_metadata)
```

Mapping form remains available when the scoring metric must be overridden:
`{"X": X_score, "y": y_score, "metric": "mae", ...}`. The read-only input alias
`score_metric` is also accepted, but canonical exports use `metric` and providing
both names is rejected. Tuple `score_data` longer than five fields is rejected as
ambiguous, tuple/list forms shorter than `(X_score, y_score)` are rejected, and
metadata in the fifth position must use canonical non-empty string keys in either
mapping or row-style sequence form. `NativeTuning.to_dict()` publishes valid
tuple inputs as JSON-native lists, rejects scalar/object `score_data` values that
are neither mapping nor tuple/list, and identity fields must align with
`y_score`.

For conformal-aware development scoring, mapping form can add an explicit
temporary calibration cohort:

```python
"score_data": {
    "X": X_score,
    "y": y_score,
    "metric": "conformal_mean_width",
    "conformal_coverage": 0.9,
    "conformal_calibration": {
        "X": X_dev_calibration,
        "y_true": y_dev_calibration,
        "sample_ids": dev_calibration_ids,
    },
}
```

The same syntax is available through the typed public helpers:

```python
tuning = nirs4all.NativeTuning(
    engine="optuna",
    space={"alpha": [0.2, 0.9]},
    metric="conformal_mean_width",
    direction="minimize",
    score_data=nirs4all.TuningScoreData(
        X=X_score,
        y=y_score,
        conformal_coverage=0.9,
        conformal_calibration=nirs4all.TuningConformalScoreCalibration(
            X=X_dev_calibration,
            y_true=y_dev_calibration,
            physical_sample_ids=dev_calibration_ids,
        ),
    ),
)
```

`NativeTuning` rejects inconsistent typed payloads before runtime execution:
`conformal_calibration` requires one of the supported conformal score metrics,
`conformal_coverage` (read-only mapping alias: `coverage`) requires
`conformal_calibration`, must be a finite numeric scalar in `(0, 1)`, and a
conformal metric requires a mapping `score_data` with a temporary calibration
cohort. Multi-coverage lists belong to final calibration/prediction, not to the
temporary tuning scorer. The keyword/effect registry exposes both
`run.tuning.score_data.conformal_calibration` and
`run.tuning.score_data.conformal_coverage` for generated docs and future Studio
forms. The typed `TuningConformalScoreCalibration(...)` helper accepts
exactly one feature alias among `X`, `X_calibration`, and `features`, plus
exactly one target alias among `y_true`, `y`, `y_calibration`, `target`, and
`targets`; it emits canonical `X`/`y_true`. It also accepts
`calibration_sample_ids` and `physical_sample_ids` as identity aliases and emits
canonical `sample_ids`. The typed `TuningScoreData(...)` helper likewise accepts
either `X`/`y` or `X_score`/`y_score`, plus either `metric` or `score_metric`,
`groups` or `score_groups`, and `metadata` or `score_metadata`, then emits
canonical `X`/`y`, `metric`, `groups`, and `metadata`. Metadata keys must be
canonical non-empty strings for both column-style mappings and row-style
sequences of mappings; raw `NativeTuning.score_data.metadata` follows the same
rule. Metadata values must also stay strict JSON-native and finite. `metric`/`score_metric` must be real non-empty strings, not values that can
only be accepted through Python stringification; blank or NUL-containing strings
fail closed, and the published spelling is lowercase canonical `metric`. The conformal calibration
helper similarly canonicalizes `calibration_groups` and `calibration_metadata` to
`groups` and `metadata`. Metadata keys must be canonical non-empty strings for
both column-style mappings and row-style sequences of mappings; raw
`score_data.conformal_calibration.metadata` follows the same rule before
`NativeTuning.to_dict()` can publish it, including the same strict JSON-native
value boundary. The registry uses separate `oneOf` groups for feature
and target aliases so generated forms should ask for one spelling from each
group, not multiple. Raw `score_data` mapping payloads in `run(tuning=...)` are
checked with the same alias-exclusivity rules before nirs4all builds the standard
or temporary conformal scorer, including nested conformal calibration aliases and
the `conformal_coverage`/`coverage` spelling.

This is an objective-scoring feature, not final calibration. nirs4all predicts
the `conformal_calibration` cohort with each candidate, fits a temporary split
conformal calibrator for that candidate, scores intervals on `score_data`, and
discards the temporary calibrator. The final `run(..., calibration=...)` payload
still derives its calibrator from `tuning.winner`. The current conformal-aware
score metrics are `conformal_mean_width`, `conformal_median_width`,
`conformal_interval_score`, `conformal_mean_interval_score`,
`conformal_abs_coverage_gap`, `conformal_missed_rate`, and
`conformal_observed_coverage`. The same vocabulary is exposed programmatically
as `nirs4all.CONFORMAL_TUNING_SCORE_METRICS`.

Completed tuning results keep this boundary visible in every conformal-aware
trial diagnostic: `score_family="conformal"`,
`score_extractor="conformal_temporary_calibration"`, and
`final_calibration_scope="unmodified_by_score_data"`.

`score_data` can also select an explicit dataset-backed scoring cohort:

```python
"score_data": {
    "dataset": spectro_dataset,  # or DatasetConfigs / config mapping / "dataset.json"
    "selector": {"partition": "val"},
    "sample_id_column": "Sample_ID",
    "group_column": "Batch",
    "metadata_columns": ["Site"],
}
```

As for fit data, the selector is mandatory and the scoring cohort is not guessed.
The dataset source must resolve to exactly one dataset when it is loaded through
`DatasetConfigs` or a path/config mapping. Use `"metric": ...` or the typed
`TuningScoreData(metric=...)` helper when the scoring cohort needs a metric
override; `score_metric` is accepted only as an input alias.
`TuningScoreData(dataset=..., selector=...)` executes the same selected cohort
and forwards selected sample ids, groups and metadata to compatible `predict()`
methods. Dataset-backed column selectors are canonical: `sample_id_column` and
`group_column` must be non-empty strings without surrounding whitespace or NULs,
and `metadata_columns` must be one canonical string or a duplicate-free sequence
of canonical strings. Raw `NativeTuning.score_data` and `NativeTuning.winner`
dataset-backed mappings apply the same rule before publication.

Dataset-backed fit cohorts are supported only through an explicit mapping so the
fit cohort is never guessed:

```python
dataset = {
    "dataset": spectro_dataset,  # or DatasetConfigs / config mapping / "dataset.json"
    "selector": {"partition": "train"},
    "sample_id_column": "Sample_ID",
    "group_column": "Batch",
    "metadata_columns": ["Site", "Instrument"],
}
```

This resolves the dataset source, then extracts `X` with
`dataset.x(selector, layout="2d", concat_source=True)` and `y` with
`dataset.y(selector)`. `include_augmented` defaults to `False` for this native
tuning subset; set it explicitly to `True` only when augmented samples are part
of the intended fit cohort. Dataset-backed tuning selectors reject non-string or
whitespace-padded keys plus non-JSON-native values, and `include_augmented`
must be a boolean before `TuningScoreData` can publish a runtime mapping.
Passing a bare `SpectroDataset` without a selector remains fail-closed, and
path/config sources must resolve to exactly one dataset.

The same explicit form is available for the `winner` projection cohort:

```python
"winner": {
    "dataset": spectro_dataset,  # or DatasetConfigs / config mapping / "dataset.json"
    "selector": {"partition": "test"},
    "sample_id_column": "Sample_ID",
    "group_column": "Batch",
    "metadata_columns": ["Site"],
    "score": 0.42,
    "metric": "rmse",
}
```

`TuningWinner(dataset=..., selector=...)` projects this selected cohort into
`RunResult.best`, preserving physical sample ids, optional groups and selected
metadata columns. It uses the same strict selector and `include_augmented`
boundary as `TuningScoreData`; `score` must be a finite number, not a boolean or
numeric string. Raw `NativeTuning(score_data={...})`, `winner={...}` and
`calibration={...}` mappings are checked before `to_dict()` publishes them:
dataset selectors require canonical string keys, `include_augmented` and
`as_predict_result` are strict booleans, winner scores are finite numbers, and
nested `calibration_data` is rejected. `TuningWinner.metadata` and raw
`NativeTuning.winner.metadata` use the same strict metadata key rule as
`TuningScoreData`. `TuningWinner.metric`, `dataset_name`, `model_name` and
`task_type`, plus the raw `NativeTuning.winner` aliases, must be real non-empty
strings; nirs4all rejects non-string, blank and NUL-containing values instead of
stringifying them. Winner `metric` and `task_type` publish lowercase canonical
values, while dataset and model labels publish trimmed strings. Raw calibration `coverage`, `method`,
`unit`, `workspace_metadata` and `workspace_conformal_id` are validated and
canonicalized like the typed `TuningCalibration(...)` helper. Coverage values,
including every element of a multi-coverage list, must be real numeric scalars;
numeric strings such as `"0.8"` are rejected instead of being coerced with
`float(...)`. Top-level
`NativeTuning` core fields are validated through `DagMLTuningSpec` before
publication: engine, metric and direction are canonicalized,
`n_trials`/`seed`/`resume` reject coercive values, and storage, study and
workspace tuning ids must be canonical strings.

This supplies `winner.X`, `winner.y_true`, and physical sample ids from the
selected dataset rows. It is the supported `SpectroDataset` route when the
winner will feed `tuning.calibration`; mappings that combine `dataset` with
explicit `X`/`y_true` arrays are rejected. For single-target dataset selections
whose `y` extraction returns shape `(n, 1)`, the adapter normalizes
`winner.y_true` to a one-dimensional vector before the conformal contract is
applied; multi-target `winner` calibration remains outside this public subset.

Linear preprocessing uses explicit transform steps followed by one model step.
Generated or named step paths are used in `tuning.space`; the final estimator is
addressed as `model`:

```python
result = nirs4all.run(
    pipeline=[
        {"name": "scale", "transform": StandardScaler()},
        {"model": Ridge()},
    ],
    dataset=(X_dev, y_dev),
    engine="dag-ml",
    tuning={
        "engine": "optuna",
        "space": {"model.alpha": [0.1, 1.0]},
        "metric": "rmse",
        "direction": "minimize",
        "sampler": "grid",
        "n_trials": 2,
        "score_data": {"X": X_score, "y": y_score},
    },
)
```

`score_data` drives optimizer selection through `predict(X_score)` and the
selected metric. If `score_data.sample_ids`, `score_data.groups`, or
`score_data.metadata` are supplied, they must be row-aligned with `y`;
compatible estimators receive them as prediction kwargs, while sklearn-style
estimators that do not declare those kwargs are still scored without them. The
typed `TuningScoreData(...)` helper accepts `score_sample_ids`,
`prediction_sample_ids`, and `physical_sample_ids` as aliases and emits canonical
`sample_ids`.
`winner` is optional; when supplied, it projects the refit winner into
`RunResult.best`. `winner.sample_ids`, `winner.prediction_sample_ids`, and
`winner.physical_sample_ids` are accepted aliases for the physical ids stored in
the prediction metadata; the typed `TuningWinner(...)` helper also accepts those
aliases plus `winner_sample_ids` and emits canonical `sample_ids`. Raw `winner`
mappings are fail-closed when multiple aliases of the same field are provided,
including features, target, score, metric, sample ids, dataset name, model name,
task type, and metadata. The `winner.score` remains caller-provided external
evaluation evidence and is not computed by `run()`. When
`workspace_path` is supplied to `run()`, the completed `TuningResult` is stored
before terminal refit; `workspace_tuning_id` and `workspace_metadata` control
that record and the id can be reloaded with `load_workspace_tuning_result()`.
Caller-provided tuning ids are strict workspace identifiers: they must be
canonical non-empty strings without surrounding whitespace or NULs. Omitting the
id generates one; providing an empty or non-string value fails closed.
Optional tuning workspace link ids, including `run_id`, `pipeline_id` and
`chain_id`, use the same strict identifier boundary when supplied.
`workspace_metadata` is constrained by the published keyword registry schema to
strict JSON-native values with canonical string keys; it is display/storage
metadata only and never changes optimizer behavior.
The raw `tuning_id` alias is accepted for `workspace_tuning_id`, but providing
both spellings is rejected as ambiguous.
Setting `"resume": True` in this public subset requires both `workspace_path`
and `workspace_tuning_id`. nirs4all then loads the completed workspace
`TuningResult`, verifies that the optimizer contract matches with the
operational resume bit cleared, skips optimizer-driving, and only performs the
terminal refit/projection for the current call. This is an idempotent replay of
an already completed tuning result, not yet a bit-exact optimizer checkpoint
resume for interrupted studies or broader pipeline shapes.

`calibration` is also runtime-only for this subset. It can be supplied either as
`tuning.calibration` or as the top-level alias `run(..., calibration={...})`,
but not both in the same call. When supplied, `run()` projects the explicit
`winner` prediction entry first, then calls the same replayed-array conformal path as
`nirs4all.calibrate(calibration_data=result.best, ...)`. The block must not
contain `calibration_data`; calibration evidence is derived from the projected
winner so tuning, terminal refit, and conformal calibration stay ordered and
auditable. The typed `TuningCalibration.extra` escape hatch follows the same
fail-closed rule: it may add non-reserved calibration options, but it cannot
replace typed fields or inject calibration evidence. The returned object is
`TunedSingleEstimatorConformalResult(run, calibrated)` instead of a bare
`RunResult`. The composite result also proxies common conformal operations
(`interval_coverages`, `conformal_guarantee_status`, `interval(...)`,
`metrics(...)`, and `robustness(...)`) to its calibrated prediction container,
so callers can inspect the final chain without reaching into
`result.calibrated` for routine access. When `run(workspace_path=...)` is supplied, that workspace path is
also used as the default `calibration.workspace_path`, so
`calibration.workspace_conformal_id`, `calibration.workspace_name`, and
`calibration.workspace_metadata` persist the calibrated result in the same
nirs4all workspace unless the calibration payload explicitly overrides
`workspace_path`.

A complete smoke-tested script is available at
`examples/user/04_models/U09_native_tuning_conformal.py`. It demonstrates
`run(tuning)` with a linear transformer→estimator chain, typed conformal-aware
objective scoring through `TuningConformalScoreCalibration`, conformal
calibration of the projected winner, workspace reload of both artifacts, and
`predict_calibrated()` on new point predictions.

Workspace artifacts can be inspected from the command line:

```bash
nirs4all workspace tuning list --workspace workspace/ --json
nirs4all workspace tuning show u09-native-tuning --workspace workspace/ --json
nirs4all workspace conformal list --workspace workspace/ --json
nirs4all workspace conformal show u09-conformal --workspace workspace/ --json
nirs4all workspace conformal predict u09-conformal \
  --workspace workspace/ \
  --y-pred "13.0,14.0" \
  --sample-ids "pred-003,pred-004" \
  --json
```

The `show` commands reload and verify the stored `TuningResult` or
`CalibratedRunResult` before emitting JSON, so they are suitable for CI checks
and release audits. `conformal predict` is read-only: it applies the stored
calibrator to already computed point predictions and emits prediction intervals
without modifying the workspace.

The internal DAG-ML native client is the first implementation seam for this
surface. It reports native training/replay capability and forwards already-built
contracts to DAG-ML, but it deliberately does not parse pipeline keywords,
materialize patches, run Optuna/n4m, or choose fallback behavior.

The next internal seam is `nirs4all.pipeline.dagml.estimator.DagMLPipelineEstimator`.
It is sklearn-cloneable and can call native training/replay when supplied with
compiled DAG-ML contracts, but it is not exported as a public user API yet. In
the current state, missing compilers or decoders raise typed coverage errors;
`predict_proba()` never fabricates one-hot pseudo-probabilities.

The shared objective seam is
`nirs4all.pipeline.dagml.pipeline_objective.PipelineObjective`. It evaluates one
explicit candidate by applying fail-closed parameter patches to a cloned
estimator, fitting through the estimator's native compiler/client path, and
extracting a finite score via an injected callback. The standard callback
factory is
`nirs4all.pipeline.dagml.pipeline_objective.make_prediction_score_extractor()`;
it scores a fitted candidate by calling `predict(X_score)` and evaluating the
requested metric against an explicit `y_score` cohort. It never splits data
implicitly and never reuses a development objective value as a test score. The
companion `make_conformal_prediction_score_extractor()` fits only a temporary
development calibrator per candidate and returns conformal diagnostics as
objective scores; it never mutates the terminal conformal artifact produced by
`run(..., calibration=...)`. The
internal Optuna adapter in `nirs4all.pipeline.dagml.tuning_adapters` can already
drive that objective for the portable subset of the ordered `tuning.space`
contract; the internal n4m
adapter drives the same objective through the native ask/tell optimizer for the
matching `random`/`tpe`/`grid` subset when the native sampler enum is exposed.
If an older n4m binding lacks `Sampler.GRID`, the adapter fails closed instead
of remapping to a different trial sequence. Optimizer `pruner` is now threaded
into the selected backend (`none`/`median`/`successive_halving`/`hyperband`, with
n4m aliases such as `asha`/`racing` where the native enum exists), but the shared
objective does not synthesize intermediate pruning evidence; pruning can only use
signals that the backend receives from this single-score objective. If the
shared objective raises a prune exception, the adapter records a distinct
`TrialResult(state="PRUNED", value=None, diagnostics={...})` with
`score_extractor="pruned"` instead of collapsing it into `FAIL`. n4m publishes
that row only when the installed binding can terminalize the native trial
through `optimizer.tell_result(..., TrialStatus.PRUNED)`; older bindings fail
closed instead of rewriting the candidate as `FAIL` or as a worst completed
score. n4m
`sampler` and `pruner` values are normalized case-insensitively by the public
tuning contract before adapter dispatch. Optuna `storage` must be an explicit
URI string such as `sqlite:///study.db`; bare filesystem paths such as
`study.db` are rejected by the public contract to avoid ambiguous persistence
semantics. When `resume=True`, the Optuna adapter also requires both `storage`
and `study_name`; anonymous or in-memory resume is rejected before optimizer
execution. If `force_params` is present during a non-empty Optuna resume, the
assignment must match the existing warm-start trial; the adapter skips duplicate
enqueue and refuses changed warm-start values under the same
`study_name`/`storage`. Existing materialized trial params must also match the
current `tuning.space` keys exactly; changed search-space keys under the same
persisted study fail closed before optimizer execution. Existing categorical
values must also remain present in the current choices for their key; removing
or renaming a choice under the same persisted study fails closed before
optimizer execution. Existing numeric values must also remain inside the
current range for their key; narrowing a range so that a stored trial falls
outside it fails closed before optimizer execution, and a current numeric
`step` also has to contain every restored value. During Optuna storage-backed
resume, `n_trials` is the target total trial count; a one-trial persisted study
with `n_trials=1` runs no extra trial, while `n_trials=2` runs one remaining
trial. New Optuna storage-backed studies persist nirs4all `study.user_attrs`
for format, schema version, optimizer contract fingerprint and search-space
fingerprint; non-empty studies missing those attrs or carrying mismatched
fingerprints fail closed during resume. Optuna storage-backed resume
reconstructs compact diagnostics for rows already present in the study:
completed rows use
`score_extractor="optuna_storage"`, failed rows use `score_extractor="failed"`
and pruned rows use `score_extractor="pruned"` in summary artifacts.
Restored Optuna `COMPLETE` rows must carry a finite numeric value; missing or
non-finite storage values fail closed instead of becoming completed trials with
no usable objective value.
Restored non-`COMPLETE` rows must not carry a final storage value; failed,
pruned or in-flight rows with final values are rejected as corrupted optimizer
history. Restored `RUNNING` rows fail closed during resume because interrupted
active trials cannot be safely recovered into a terminal HPO tape. Restored terminal Optuna rows must keep exactly the current
`tuning.space` parameter keys when the search space is non-empty; rows whose
stored parameter table was removed fail closed instead of becoming completed
trials with empty public params. Restored queued Optuna `WAITING` rows that
already carry materialized params or `fixed_params` must also satisfy the current
`tuning.space`; incompatible values are rejected before Optuna can consume them.
Restored Optuna trial numbers must be canonical unique integers; corrupt storage
that yields non-integer or duplicate trial numbers fails closed before the HPO
tape is projected.
`study_name` is trimmed and must not contain NUL characters. n4m
optimizer-state persistence is local and checkpoint-based in this seam:
`storage` must be `file:///absolute/checkpoint-dir`, `study_name`
must be filename-safe, and nirs4all writes a JSON manifest containing native
N4MOPT checkpoint bytes plus tuning/search-space fingerprints after each
terminal trial. `resume=True` reloads only a matching manifest and treats
`n_trials` as the target total trial count; incompatible optimizer contracts
fail closed. Restored n4m trial rows are decoded back to public
`TrialResult.params`, ordered canonically by numeric trial id, and rejected when
duplicate or non-integer restored trial ids are present, so optimizer-only categorical labels used for JSON-native
values or named `options` do not leak into the HPO tape. The restored checkpoint
row params must still match the current `tuning.space` keys and value domains:
edited or incompatible checkpoint keys, categorical choices, numeric ranges or
numeric steps fail closed before they can contribute optimizer history. Restored
`COMPLETE` rows must carry a finite numeric score; missing, boolean or non-finite
scores fail closed instead of becoming complete trials with no usable value. Restored failed,
pruned and cancelled rows keep `score_extractor="failed"`,
`score_extractor="pruned"` or `score_extractor="cancelled"` in the compact
diagnostics. Non-terminal checkpoint rows such as `RUNNING` fail closed during
resume. Restored n4m non-`COMPLETE` rows must not carry a final score; failed,
pruned or cancelled checkpoint rows with scores are rejected as corrupted
optimizer history. Candidate failures are recorded as `TrialResult(state="FAIL",
value=None, diagnostics={...})` with the exception type/message and do not erase
successful trials from the tape; if every candidate fails, no terminal refit is
performed and the result uses a finite direction-worst `best_value` so the
`TuningResult` fingerprint and `nirs4all.tuning.summary` artifact remain
TCV1-compatible. Optuna- and n4m-pruned candidates stay visible as `PRUNED` in
the full tape, `trial_states` counts and compact summary rows; n4m requires
native `optimizer.tell_result(..., TrialStatus.PRUNED)` support and otherwise
fails closed instead of rewriting a pruned candidate. Public
`run(tuning=...)` now
executes the narrow single-estimator and linear transformer→estimator array
subset above. Broader pipeline execution still refuses until the full
pipeline-to-objective compiler and complete optimizer save/resume semantics are
closed end to end.

The first compiler seam is
`nirs4all.pipeline.dagml.pipeline_objective_compiler.compile_pipeline_objective()`.
It accepts only the currently honest single-estimator and linear forms:

```python
compiled = compile_pipeline_objective(
    [{"model": estimator}],
    X,
    y,
    tuning={"engine": "optuna", "space": {"alpha": [0.1, 1.0]}},
    score_extractor=make_prediction_score_extractor("rmse", X_score, y_score),
    sample_ids=sample_ids,
)
result = compiled.evaluate({"alpha": 0.1})
```

The supported shapes are `estimator`, `[estimator]`, `{"model": estimator}`,
`[{"model": estimator}]`, `{"steps": [...]}` or `{"pipeline": [...]}` around one
of the supported linear sequences, and linear sequences of transformer instances or
`{"transform": transformer, "name": optional_name}` mappings followed by one
model step. Explicit sklearn class-path mappings are accepted for the same
linear subset, for example `{"name": "scale", "class":
"sklearn.preprocessing.StandardScaler", "params": {"with_mean": False}}`.
Direct string steps such as `("scale", "sklearn.preprocessing.StandardScaler")`
are also accepted when no constructor parameters are needed. The same
no-parameter string import can be used as a direct model
`pipeline="sklearn.dummy.DummyRegressor"` or inside explicit mappings such as
`{"name": "scale", "transform": "sklearn.preprocessing.StandardScaler"}` and
`{"name": "dummy", "model": "sklearn.dummy.DummyRegressor"}`. String
`transform` and final `model` mappings can also pass constructor parameters,
for example `{"name": "scale", "transform":
"sklearn.preprocessing.StandardScaler", "params": {"with_mean": false}}` and
`{"name": "ridge", "model": "sklearn.linear_model.Ridge", "params":
{"fit_intercept": false}}`; the latter also works as a direct single-model
pipeline mapping. These forms are intentionally restricted to explicit
`sklearn.*` import paths. Constructor `params` must use canonical string keys
and TCV1-compatible JSON-native values, so host Python objects and tuple/bytes
payloads fail before import or instantiation. Short aliases and arbitrary
imports remain fail-closed.
sklearn-like
`(name, step)` tuples are accepted for the same linear shapes, for example
`[("scale", transformer), ("ridge", estimator)]`. The final model step may also
carry `name`; when present, that name becomes a supported tuning path prefix,
for example `{"name": "ridge", "model": estimator}` with `tuning={"space":
{"ridge.alpha": [...]}}`. The same dotted path syntax also works for sklearn
`Pipeline([("ridge", estimator)])` objects: the compiler maps `ridge.alpha` to
sklearn's `ridge__alpha` parameter name before calling `set_params()`.
Non-final preprocessing steps may be explicit no-ops through `None`,
`"passthrough"` or the structured JSON-native marker
`{"kind": "passthrough"}`; the structured marker requires the literal string
value and does not coerce arbitrary objects through `str(...)`. The final model
step cannot be a passthrough. A
named preprocessing step can also be toggled by the tuning space itself, for
example `tuning={"space": {"scale": [{"kind": "passthrough"}],
"ridge.alpha": [0.1, 1.0]}}`. Optimizer adapters encode/decode structured
categorical choices internally, so `best_params`, trial rows and terminal refit
still see the public JSON-native value; `force_params` uses that same decoded
syntax and never requires callers to pass backend labels. The named form
`{"type": "categorical", "options": {"label": value}}` uses `label` only as the
stable Optuna/n4m optimizer label and returns `value` in `best_params`, trial
rows and terminal refit. The internal categorical codec is fail-closed on direct
construction: choices must be non-empty, unique, optimizer-native without a
decoder, and TCV1 JSON-native; decoder keys must match encoded choices and
decoded public values must be unique before Optuna/n4m can use them. This
replacement is limited to non-final
preprocessing steps, and the `space` contract remains TCV1-compatible
JSON-native: raw Python transformer objects are not valid candidate values
until a registry/canonicalization layer exists. The `{"steps": [...]}` and
`{"pipeline": [...]}` mappings may carry `name`, but no execution keywords; they
are syntax wrappers, not broader DAG compilers. Supplying both `steps` and
`pipeline` in the same wrapper is rejected as ambiguous.
Splitters, branches, dataset loaders, string model aliases, non-linear
preprocessing graphs and step-level keywords such as `finetune_params` remain
fail-closed in this compiler; they belong to later public `run(tuning=...)`
compiler gates.

The typed `TuningResult` already has deterministic `to_dict()`/`to_json()` and
`save_json()`/`load_json()` helpers with fingerprint verification. The same
verified result can now be stored in a nirs4all workspace through
`save_workspace_tuning_result()` and reloaded with
`load_workspace_tuning_result()`:

```python
tuning_id = nirs4all.save_workspace_tuning_result(
    "workspace/",
    tuning_result,
    tuning_id="pls-hpo-main",
    name="PLS HPO",
)
restored = nirs4all.load_workspace_tuning_result("workspace/", tuning_id)
```

This is persistence for already-produced native tuning results. The public
single-estimator/linear `run(tuning=...)` subset can reload such a completed
result via `resume=True`; n4m optimizer-state resume is separately available
through storage-backed N4MOPT checkpoints for the same adapter subset. Broader
pipeline shapes remain a separate gate.

`run_pipeline_objective_tuning()` is the internal orchestration seam that runs
optimizer-driving first and then, only when requested, performs the terminal
winner refit through `PipelineObjective.refit_best()`. This keeps HPO trial
evaluation and deployable predictor training separate. When passed a
`workspace_path`, it now stores the completed `TuningResult` at the
optimizer→refit boundary and returns the `tuning_id`. The storage happens before
terminal refit, so the HPO evidence tape is preserved even if the deployable
winner fit fails later. The public `run(tuning=...)` single-estimator/linear
subset is already an end-user execution path for this seam; full pipeline shapes
still need their broader pipeline-to-objective compiler.

The `RunResult` carrier now has a tuning-evidence projection seam:
`tuning_result`, `tuning_id`, `tuning_best_params`, and `tuning_best_value`.
This makes an optimizer trace visible without fabricating prediction rows,
validation metrics, or a deployable model. The helper
`nirs4all.pipeline.dagml.tuning_projection.project_objective_tuning_to_run_result()`
projects an internal `ObjectiveTuningRunResult` into that carrier.

For CI, release automation, bindings and Studio cards, the public
`TuningResult` also exposes `summary_artifact()`, `to_summary_json()` and
`save_summary(...)`. The lightweight tuning summary payload uses format
`nirs4all.tuning.summary` and carries the result fingerprint, engine, metric,
direction, optimizer, sampler, pruner, seed, best params/value, trial count,
trial state counts and compact trial rows. It is intentionally a
dashboard/indexing artifact; compact diagnostics are scalar-only, so arrays or
mappings under whitelisted diagnostic keys fail closed instead of being
stringified. The full `TuningResult.to_json()` remains the optimizer evidence
tape. The schema is
available at runtime via `get_tuning_summary_schema()` and
`tuning_summary_schema_json()`. The ordered pre-execution tuning-space schema is
separate and available via `get_tuning_space_schema()` and
`tuning_space_schema_json()`.

The same helper can also project an explicit refit winner prediction when the
caller supplies `winner_x`, `winner_score`, and `winner_metric`:

```python
run_result = project_objective_tuning_to_run_result(
    objective_result,
    winner_x=X_external_test,
    winner_y_true=y_external_test,
    winner_score=0.42,
    winner_metric="rmse",
    winner_sample_ids=physical_sample_ids,
)
```

In that mode, `winner_score` is trusted as caller-provided evaluation evidence;
the projection helper only calls `refit_estimator.predict(winner_x)` and never
computes or invents the ranking metric. The public `run(tuning=...)`
single-estimator/linear array subset feeds this projection through its `winner`
block. The remaining public gate is the broader compiler/wiring for splitters,
branches, dataset loaders, non-linear preprocessing graphs, and complete
optimizer save/resume semantics.

For Python users who want this lane without going through `run()`, nirs4all also
exposes the explicit single-estimator helper
`nirs4all.tune_single_estimator(...)`:

`tune_single_estimator()` is intentionally lower-level than `run(tuning=...)`.
It accepts explicit array or tuple scoring cohorts, explicit winner cohorts, and
conformal-aware temporary scoring cohorts, but it does not resolve dataset-backed
`TuningScoreData(dataset=..., selector=...)` or
`TuningWinner(dataset=..., selector=...)`. Dataset loading, selector resolution,
identity-column extraction and dataset-backed winner projection belong to
`run(tuning=...)`, so forms, bindings and Studio should use `run(tuning=...)` for
dataset-backed tuning flows.

```python
run_result = nirs4all.tune_single_estimator(
    [{"model": estimator}],
    X_dev,
    y_dev,
    tuning={"engine": "optuna", "space": {"alpha": [0.1, 1.0]}},
    X_score=X_score,
    y_score=y_score,
    winner_x=X_external_test,
    winner_y_true=y_external_test,
    winner_score=0.42,
    winner_metric="rmse",
    winner_sample_ids=physical_sample_ids,
)
```

This public helper stitches together the current internal seams:
`compile_pipeline_objective()` → optimizer/refit orchestration →
`project_objective_tuning_to_run_result()`. It still accepts only the
single-estimator and linear shapes listed above, including explicit `sklearn.*`
string imports, `{"class": ..., "params": ...}` steps, and the public
`{"steps": [...]}` / `{"pipeline": [...]}` wrappers. Scoring is explicit: provide either
`X_score`/`y_score` (using `tuning["metric"]`, or `score_metric` if supplied) or
a custom `score_extractor`. When `tuning` is a `NativeTuning` or mapping with
`score_data`, the helper can also build the scorer directly from that block for
explicit array/tuple cohorts, including `TuningConformalScoreCalibration`.
Explicit `tuning.winner` and `tuning.calibration` blocks are consumed the same
way as the `winner_*` and `calibration=` arguments; mixing both forms is
rejected. Dataset-backed `score_data` and `winner` remain routed through
`run(tuning=...)`, where dataset loading and selectors are already handled. The
external winner score remains caller-provided evidence; the helper does not turn
a development objective value into a test score. When calibration is requested,
the calibrated result metadata carries `tuning_calibration_source` so Studio,
bindings and audit tools can distinguish final `tuning.winner` evidence from
`tuning.score_data` objective-ranking data.

```python
run_result = nirs4all.tune_single_estimator(
    [{"model": estimator}],
    X_dev,
    y_dev,
    nirs4all.NativeTuning(
        engine="optuna",
        space={"alpha": [0.1, 1.0]},
        metric="conformal_mean_width",
        score_data=nirs4all.TuningScoreData(
            X=X_score,
            y=y_score,
            conformal_coverage=0.9,
            conformal_calibration=nirs4all.TuningConformalScoreCalibration(
                X=X_dev_calibration,
                y_true=y_dev_calibration,
            ),
        ),
        winner=nirs4all.TuningWinner(
            X=X_calibration,
            y_true=y_calibration,
            score=0.42,
            metric="rmse",
            physical_sample_ids=calibration_sample_ids,
        ),
        calibration=nirs4all.TuningCalibration(
            y_pred=new_point_predictions,
            prediction_sample_ids=new_sample_ids,
            coverage=0.9,
        ),
    ),
)
```

Because the winner projection is a normal nirs4all prediction entry with
`y_true`, `y_pred`, and `physical_sample_id` metadata, it can feed the current
replayed-array conformal surface directly:

```python
prediction = nirs4all.calibrate(
    calibration_data=run_result.best,
    y_pred=new_point_predictions,
    prediction_sample_ids=new_sample_ids,
    coverage=0.9,
    as_predict_result=True,
)
interval_90 = prediction.interval(0.9)
```

The same chain can be requested in one call with `calibration={...}` or a typed
`TuningCalibration(...)` helper. In that case `tune_single_estimator()` returns a
`TunedSingleEstimatorConformalResult` with two fields: `run` for the tuning
`RunResult`, and `calibrated` for the conformal `PredictResult` or
`CalibratedRunResult`. The composite proxies the common conformal accessors
directly (`result.interval(...)`, `result.metrics(...)`,
`result.robustness(...)`). The public `run(tuning=..., calibration=...)` alias
and the nested `run(tuning={..., "calibration": ...})` form return the same container.

```python
result = nirs4all.tune_single_estimator(
    [{"model": estimator}],
    X_dev,
    y_dev,
    tuning={"engine": "optuna", "space": {"alpha": [0.1, 1.0]}},
    X_score=X_score,
    y_score=y_score,
    winner_x=X_calibration,
    winner_y_true=y_calibration,
    winner_score=0.42,
    winner_metric="rmse",
    winner_sample_ids=calibration_sample_ids,
    calibration=nirs4all.TuningCalibration(
        y_pred=new_point_predictions,
        prediction_sample_ids=new_sample_ids,
        coverage=0.9,
    ),
)
interval_90 = result.interval(0.9)
```

This is the first supported Python path for
`tune → refit → winner prediction entry → calibrate → PredictResult` on the
single-estimator/linear lane. It is still not a substitute for future
raw-dataset calibration loading or automatic replay from arbitrary
`run(tuning=...)` pipelines.

For native estimator fits, `nirs4all.pipeline.dagml.fit_identity` now normalizes
row-aligned `sample_ids`, `groups`, and `metadata` before any compiler runs.
Explicit `sample_ids` are the conformant path for leakage and exchangeability
claims. If omitted, the estimator can still mint deterministic
content-hash-plus-position compatibility ids, but those ids are not a basis for
future conformal guarantees.

The first compiler seam is
`nirs4all.pipeline.dagml.training_compiler.PreparedDagMLTrainingCompiler`. It
accepts already-prepared DAG-ML training contracts, validates the estimator-side
contract envelope, adds fit-identity diagnostics, and returns a
`DagMLTrainingExecution` for the native client. It is intentionally not a full
pipeline syntax compiler: arbitrary `finetune_params`, branch lowering,
optimizer objectives, and public routing remain planned follow-up work.
`DagMLTrainingContractFactoryCompiler` adds the adjacent hook for future
lowerers: a callable can build prepared contracts from `(estimator, X, y,
sample_ids, groups, metadata, identity_frame)` and still reuse the same
validation and diagnostics path.

The next request-assembly seam is
`nirs4all.pipeline.dagml.training_contracts`. It provides an internal TCV1
signer, `DagMLTrainingRequestSpec`, `assemble_training_request()`, and
`training_data_identity_from_binding()`. `DagMLTrainingRequestCompiler` then
connects those signed request specs to `DagMLPipelineEstimator`. This closes the
contract-shell part of the native path; it still does not lower arbitrary
pipeline syntax or raw array envelopes by itself.
When the installed DAG-ML binding exposes `dag_ml.sign_training_request()`, the
compiler delegates final request canonicalization and signing to DAG-ML before
execution, so dynamically compiled requests use the same typed Rust
serialization as native validation.

`nirs4all.pipeline.dagml.raw_training_lowerer.RawArrayDagMLTrainingCompiler`
adds the first raw-array lowerer. It accepts a linear list pipeline with one
splitter and one model, converts `fit(X, y, sample_ids=...)` into a minimal
`SpectroDataset`, preserves normalized sample ids in the DAG-ML identity map,
builds folds, envelopes, relations and a training influence manifest, and
connects the existing host node runner as the native `op_callback`. With a
recent source-tree DAG-ML binding exposing both `dag_ml.sign_training_request()`
and `dag_ml.sample_relation_set_fingerprint_json()`, this lowerer can execute
the native `dag_ml.execute_training()` path for that minimal raw-array shape.
The second helper is required so the Python-built data envelope uses the same
relation fingerprint as `dag-ml-core`.

For the same internal raw-array seam, the deterministic subset of
`finetune_params` is lowerable natively by the same shared helper used by public
`run(engine="dag-ml")`: `finetune_params.model_params` may be a plain JSON grid
or native `_range_`/`_log_range_` list-form generator, and `metric`/`direction`
feed the DAG-ML selection policy. Adaptive optimizer controls remain
fail-closed: `engine="n4m"`/Optuna, `n_trials`, samplers, pruners and multiphase
tuning are not silently ignored and still belong to the optimizer-adapter lane.
Branches, augmentation, repetition, full objective adapters, conformal
calibration and Studio forms remain planned lanes for native training.

(continuation-training-retrain-finetune)=
### Continuation training: `retrain(mode="finetune")`

This public mode requests continuation from an existing predictor's weights; it
never requests HPO. Its current registry status is **partial**: the legacy API
accepts the mode, but continuation of the original weights is not yet proven
uniformly across model families. DAG-ML retraining does not support it yet.

Do not present a successful prediction as evidence that weights were resumed.
Capability tests must demonstrate that the starting weights were loaded and
updated before this status becomes `supported`.

(planned-conformal-calibration)=
### Conformal calibration

`nirs4all.calibrate()` is available for the replayed-array V1 surface:
calibration targets, replayed calibration predictions, prediction outputs and
explicit physical sample ids must be supplied by the caller. Filesystem,
workspace and `.n4a` persistence are supported for this replayed-array surface;
explicit dataset-backed calibration cohorts are supported when the selector and
either replayed calibration predictions or an in-memory predictor are supplied.
The dataset value can be a `SpectroDataset`, a `DatasetConfigs` object, a
dataset config mapping, or a config/path string resolved by the existing
nirs4all dataset loaders. Predictor replay can also use the bounded public lanes
listed below: an in-memory predictor, saved predictor bundle, `RunResult`-like
prediction entry, or explicit workspace chain. The
calibration cohort must remain isolated from fitting, HPO, and model selection.
Any later change to the predictor invalidates the resulting calibrator.

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "y_true": y_cal,
        "y_pred": y_cal_pred,
        "sample_ids": calibration_sample_ids,
        "groups": calibration_groups,
        "metadata": calibration_metadata,
    },
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=[0.9, 0.95],
)
interval_90 = calibrated.prediction.interval(0.9)
```

Raw replayed-array mappings accept the canonical `y_pred`, `sample_ids`,
`groups`, and `metadata` keys plus the public aliases `y_pred_calibration` /
`calibration_predictions`, `calibration_sample_ids` / `physical_sample_ids`,
`calibration_groups`, and `calibration_metadata`. nirs4all canonicalizes exactly
one alias per field for runtime use; providing two aliases for the same field is
rejected before calibration so bindings and Studio forms cannot publish
ambiguous replay evidence.

For compact array callers, the same replayed calibration evidence can be passed
as `(y_true, y_pred, sample_ids, groups, metadata)`. Optional trailing fields may
be omitted, but physical `sample_ids` must still be provided either in the tuple
or via `calibration_sample_ids=...`:

```python
calibrated = nirs4all.calibrate(
    calibration_data=(y_cal, y_cal_pred, calibration_sample_ids),
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

Typed callers can build the same payload with `nirs4all.ConformalCalibrationData`.
The helper is intended for Python forms, bindings and Studio-style payload
generation; it serializes to the same mapping syntax consumed by
`calibrate()` and does not widen the supported runtime subset:

```python
calibrated = nirs4all.calibrate(
    calibration_data=nirs4all.ConformalCalibrationData(
        y_true=y_cal,
        y_pred=y_cal_pred,
        sample_ids=calibration_sample_ids,
    ),
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

`calibration_data` can also be a nirs4all prediction entry such as
`result.best`, provided it already carries replayed arrays and explicit physical
sample ids. Extra result fields are ignored; missing sample ids remain
fail-closed.

```python
calibrated = nirs4all.calibrate(
    calibration_data=result.best,  # requires y_true, y_pred and physical IDs
    y_pred=test_predictions,
    prediction_sample_ids=test_sample_ids,
    coverage=0.9,
)
```

`calibration_data` can also select an explicit dataset-backed calibration
cohort. In this form, the dataset supplies `y_true`, physical ids, optional
groups and metadata; the already replayed point predictions for exactly those
selected rows can be supplied either as `calibration_data.y_pred` or as the
top-level `y_pred_calibration=` argument. The `dataset` value may be a
`SpectroDataset`, a `DatasetConfigs` object, a dataset config mapping, or a
config/path string accepted by `DatasetConfigs`. As narrow replay lanes,
`calibration_data.predictor` may provide an in-memory sklearn-like object with
`predict(X)`. When its signature accepts `sample_ids`, `groups` or `metadata`,
nirs4all forwards the selected calibration identities. For saved predictors,
`calibration_data.predictor_bundle` uses the public `nirs4all.predict()` replay
path on the extracted `X`; mapping aliases `model_bundle`, `predictor_path` and
`model_path` are read as the same source, and
`ConformalCalibrationData(model_bundle=...)` serializes back to canonical
`predictor_bundle`. For previous nirs4all training outputs,
`calibration_data.predictor_result` accepts a `RunResult.best`-like prediction
entry, and `calibration_data.predictor_chain_id` replays a stored workspace
chain through `nirs4all.predict(chain_id=...)` when
`calibration_data.workspace_path` is also provided. Do not pass more than one of
`y_pred`, `predictor`, `predictor_bundle`, `predictor_result`, and
`predictor_chain_id` in the same calibration mapping. The typed helper also
accepts `workspace_chain_id` as a read alias for `predictor_chain_id` and emits
the canonical key. Both spellings must carry a canonical non-empty workspace
chain id without surrounding whitespace or NULs; invalid ids fail before
`predict(chain_id=...)` replay or provenance publication.

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": spectro_dataset,  # or "calibration_dataset.json"
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "group_column": "Batch",
        "metadata_columns": ["Site"],
        "y_pred": y_cal_pred,
    },
    y_pred=test_predictions,
    prediction_sample_ids=test_sample_ids,
    coverage=0.9,
)
```

Equivalent top-level prediction form:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": spectro_dataset,
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
    },
    y_pred_calibration=y_cal_pred,
    y_pred=test_predictions,
    prediction_sample_ids=test_sample_ids,
    coverage=0.9,
)
```

Equivalent in-memory predictor replay form:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": "calibration_dataset.json",
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "group_column": "Batch",
        "metadata_columns": ["Site"],
        "predictor": fitted_model,
        "predictor_fingerprint": "pls-v1-calibration",
    },
    y_pred=test_predictions,
    prediction_sample_ids=test_sample_ids,
    coverage=0.9,
)
```

Equivalent saved predictor replay form:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": "calibration_dataset.json",
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "predictor_bundle": "model.n4a",
    },
    y_pred=test_predictions,
    prediction_sample_ids=test_sample_ids,
    coverage=0.9,
)
```

Equivalent previous-result replay form:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": "calibration_dataset.json",
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "predictor_result": train_result.best,
        "workspace_path": "workspace",
    },
    y_pred=test_predictions,
    prediction_sample_ids=test_sample_ids,
    coverage=0.9,
)
```

Equivalent workspace-chain replay form:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": "calibration_dataset.json",
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "predictor_chain_id": "chain-abc123",
        "workspace_path": "workspace",
    },
    y_pred=test_predictions,
    prediction_sample_ids=test_sample_ids,
    coverage=0.9,
)
```

The selector is mandatory, `include_augmented` defaults to `False`, and
single-target dataset selections whose `y` extraction returns `(n, 1)` are
normalized to a one-dimensional calibration target vector. Dataset config/path
sources must resolve to exactly one dataset. Mixing the `dataset` form with
explicit `X`/`y_true` arrays is rejected; pass one representation for the
calibration cohort. Physical sample ids may be supplied as `sample_ids`,
`calibration_sample_ids`, `physical_sample_ids`, or `sample_id_column`; their
values for calibration and prediction cohorts must be canonical non-empty strings
without surrounding whitespace or NULs. More than one explicit sample-id alias in
the same dataset-backed mapping is rejected as ambiguous; use `sample_id_column`
only when the ids should be read from dataset metadata. Optional row-aligned
groups and metadata may be supplied as `groups`/`calibration_groups` and
`metadata`/`calibration_metadata`, or derived from `group_column` and
`metadata_columns`. Dataset-backed selectors reject non-string or
whitespace-padded keys plus non-JSON-native values, and `include_augmented`
must be a boolean before the typed helper or raw mapping can publish the
payload. Dataset-backed calibration column selectors are canonical too:
`sample_id_column` and `group_column` must be non-empty strings without
surrounding whitespace or NULs, and `metadata_columns` must be one canonical
string or a duplicate-free sequence of canonical strings. The same rule applies
to `ConformalCalibrationData(...)` and raw `calibration_data={...}` mappings
before cohort extraction.
`workspace_path` inside `calibration_data` is used only to replay a
predictor/result/chain for calibration; it is distinct from the top-level
`workspace_path` that persists the calibrated conformal result.

If downstream code expects the usual prediction container, request a
`PredictResult`:

```python
predict_result = nirs4all.calibrate(
    calibration_data={...},
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
    as_predict_result=True,
)
interval_90 = predict_result.intervals[0.9]
```

To persist the calibrated replayed-array result in the verified filesystem
layout, pass `store_path` and reload it with `load_calibrated_result()`:

```python
calibrated = nirs4all.calibrate(
    calibration_data={...},
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
    store_path="calibrated-result/",
)
restored = nirs4all.load_calibrated_result("calibrated-result/")
```

To persist the same replayed-array result in a nirs4all workspace SQLite store,
pass `workspace_path`. Provide `workspace_conformal_id` when you want a stable
identifier; otherwise use `save_workspace_calibrated_result()` and keep the
returned id. Caller-provided conformal ids must be canonical non-empty strings
without surrounding whitespace or NULs.
Optional conformal workspace link ids, including `run_id`, `pipeline_id`,
`chain_id` and source `prediction_id`, use the same strict identifier boundary
when supplied.

```python
calibrated = nirs4all.calibrate(
    calibration_data={...},
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
    workspace_path="workspace/",
    workspace_conformal_id="wheat-moisture-calibration",
)
restored = nirs4all.load_workspace_calibrated_result(
    "workspace/",
    "wheat-moisture-calibration",
)
```

For a portable archive, use the same replayed-array surface with `bundle_path`
or export an existing calibrated result:

```python
calibrated = nirs4all.calibrate(
    calibration_data={...},
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
    bundle_path="calibrated-result.n4a",
)
restored = nirs4all.load_calibrated_result("calibrated-result.n4a")

nirs4all.export_calibrated_result(calibrated, "calibrated-result.n4a")
```

To apply a saved calibrator to a new vector of already computed point
predictions, use `predict_calibrated()`:

```python
predict_result = nirs4all.predict_calibrated(
    "calibrated-result.n4a",
    y_pred=new_point_predictions,
    prediction_sample_ids=new_sample_ids,
)
interval_90 = predict_result.interval(0.9)
```

The same narrow replayed-array lane is also available through `predict()` when
the model argument is a loaded calibrated result, a conformal result store, or a
conformal `.n4a` bundle. In this form `data` is not raw spectra: it must contain
already computed point predictions and explicit physical sample ids. The
prediction ids must be canonical non-empty strings without surrounding whitespace
or NULs, unique within the prediction cohort, and disjoint from the calibration
cohort embedded in the conformal artifact.

```python
predict_result = nirs4all.predict(
    model="calibrated-result.n4a",
    data={"y_pred": new_point_predictions, "sample_ids": new_sample_ids},
    coverage=0.9,
)
interval_90 = predict_result.interval(0.9)
```

To attach a saved calibrator to an existing model `.n4a` bundle, create a new
bundle with a conformal sidecar:

```python
calibrated_bundle = nirs4all.attach_calibrated_result_to_bundle(
    "wheat_model.n4a",
    "calibrated-result.n4a",
    output_path="wheat_model.calibrated.n4a",
)
```

That attached bundle can then replay raw spectra through the normal model bundle
loader and apply the stored conformal intervals. Physical sample ids remain
mandatory because the V1 guarantee is defined at `unit="physical_sample"`.
Attached sidecar prediction currently applies intervals only to the single
selected prediction returned by `all_predictions=False`; requesting all model
predictions stays fail-closed until each returned prediction entry can carry its
own calibrated identity mapping.
If a model bundle contains a `conformal/` sidecar but the sidecar is incomplete,
duplicated, or contains unexpected members, prediction fails sidecar validation
instead of falling back to an uncalibrated prediction path.

```python
predict_result = nirs4all.predict(
    model=calibrated_bundle,
    data={"X": X_new, "sample_ids": new_sample_ids},
    coverage=0.9,
)
interval_90 = predict_result.interval(0.9)
```

Every conformal result carries explicit guarantee metadata under
`conformal_guarantee_status`. `CalibratedRunResult.conformal_guarantee_status`
and `PredictResult.conformal_guarantee_status` expose the same payload. The
status records the requested engine, effective engine, method, unit, calibrated
coverages, selected coverages, artifact fingerprint, predictor/data
fingerprints when known, `calibration_replay_source`, and invalidation reasons.
When a `CalibratedRunResult` is loaded or converted with
`to_predict_result()`, this status is checked against the embedded conformal
artifact. A stale `artifact_fingerprint`, method, unit, multi-target policy, or
coverage selection fails closed instead of being exposed as a valid guarantee.
For calibrator-application results, a top-level
`source_calibrated_result_fingerprint` must also match the same field inside
`conformal_guarantee_status`; mismatches fail closed on construction or reload.
When `calibration_replay_source` appears both at top level and inside
`conformal_guarantee_status`, the two mappings must also match exactly; this
prevents a stored result from advertising one replay lane to notebooks and a
different lane to Studio or bindings.
Guarantee metadata string fields are strict provenance, not display-only text:
`effective_engine`, `requested_engine`, `source_calibrated_result_fingerprint`,
and each `invalidation_reasons` entry must be non-empty strings without
surrounding whitespace, and persisted `conformal_guarantee_status.version` must
be the strict integer `1`. Boolean, object, empty or whitespace-padded values
fail closed instead of being stringified. Status `predictor_fingerprint`,
`calibration_data_fingerprint`, `guarantee`, and `scope` are also rechecked
against the embedded artifact before exposure. A persisted status must include
the complete generated field set, and `status` must be `active` exactly when
`invalidation_reasons` is empty, otherwise `invalidated`. The generated
`limitations` list must also match the embedded artifact's guarantee mode
exactly; edited, shortened, empty or non-string limitation payloads fail closed.
`calibration_replay_source.kind` distinguishes provided arrays, `PredictResult`,
dataset-backed `y_pred`, `predictor`, `predictor_bundle`, `predictor_result`,
and `predictor_chain_id`; `route` records whether calibration used provided
arrays, `predictor.predict`, or public `nirs4all.predict`. Downstream tools
should show this status rather than inferring a statistical guarantee from
interval columns or empirical metrics.

```python
status = predict_result.conformal_guarantee_status
print(status["status"], status["coverage"], status["effective_engine"])
print(status["calibration_replay_source"]["kind"])
```

When observed targets are available for a prediction cohort, use
`conformal_metrics()` to compute diagnostics. These are empirical diagnostics on
the supplied cohort, not a renewed finite-sample guarantee:

```python
metrics = nirs4all.conformal_metrics(predict_result, y_true=y_observed)
m90 = metrics[0.9]
print(m90.observed_coverage, m90.coverage_gap, m90.mean_width)
```

Each `ConformalMetricSet` contains `observed_coverage`, `coverage_gap`
(`observed_coverage - coverage`), `mean_width`, `median_width`,
`mean_interval_score`, `n_covered`, `n_missed_below`, and `n_missed_above`.
The interval score uses the standard central interval score
`width + 2/alpha * lower_miss + 2/alpha * upper_miss`, with
`alpha = 1 - coverage`. The container validates those diagnostics before
fingerprinting or publication: `observed_coverage` must be finite in `[0, 1]`
and equal `n_covered / n_samples`, `coverage_gap` must equal
`observed_coverage - coverage`, and width/interval-score diagnostics must be
non-negative. Positive infinity is accepted only for unbounded interval metrics.

Changing `multi_target` or `group_by` creates a different calibrator. Adding a
coverage can extend an artifact only when the required calibration scores were
retained; `predict(..., coverage=...)` merely selects an already materialized
coverage and never recalibrates implicitly. Requesting a valid but
non-materialized coverage fails closed and reports the available coverages.

Internally, `nirs4all.pipeline.dagml.conformal_contracts` now provides the
first Python-side contract substrate for V1: finite-sample
`ceil((n + 1) * coverage)` quantiles, `split_absolute_residual` fitting from
already replayed calibration predictions and targets, and multi-coverage
interval application. The same internal substrate also groups point predictions
and materialized intervals in a `CalibratedPredictionBlock`, so downstream
storage, result typing, and UI code can consume one explicit prediction cohort
without reapplying calibration.

`ConformalCalibrationSpec` and `parse_conformal_calibration_spec()` also
normalize the reserved syntax (`method`, `coverage`, `unit`, `group_by`, and
`multi_target`) into a deterministic fingerprint. `group_by` is executable for
the replayed-array substrate when calibration rows and prediction rows both
provide the declared group evidence: `group_by="group"` consumes
`calibration_groups`/`prediction_groups`, while other keys are read from
`calibration_metadata`/`prediction_metadata`. Missing, null or unseen prediction
groups fail closed without global fallback. Group labels are strict strings:
whitespace-padded labels, NUL-containing labels and non-string labels fail before
group quantiles are fitted or selected. `multi_target="joint_max"` is
executable for two-dimensional replayed arrays: the retained score is
`max(abs(y_true - y_pred))` across target columns for each physical sample, and
the resulting `qhat` creates a simultaneous region with the same shape as
`y_pred`.
For public tooling, `nirs4all.CONFORMAL_CALIBRATION_METHODS`,
`nirs4all.CONFORMAL_CALIBRATION_UNITS`,
`nirs4all.CONFORMAL_MULTI_TARGET_POLICIES`, and
`nirs4all.CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES` expose the same
vocabularies without importing internal DAG-ML contract modules.

For storage and packaging work, `ConformalCalibrationArtifact` can now persist
the fitted internal calibrator, retained calibration scores, quantiles and
fingerprints as deterministic verified JSON. `CalibratedRunResult` is the public
Python result container for one calibrated prediction cohort: it keeps sample
ids, point predictions, selected coverage intervals, guarantee metadata and
persistence helpers in one fingerprinted JSON form. `ConformalMetricSet` is the
public empirical diagnostics container for observed prediction cohorts; it
records coverage/width/interval-score evidence without changing the persisted
calibrated-result fingerprint or the guarantee metadata.
`ConformalCalibrationSpec` validates direct construction the same way as
`parse_conformal_calibration_spec()`: coverage values must be real numeric
scalars, not booleans or numeric strings, and method, unit, group keys and
multi-target mode are canonicalized before fingerprinting.
Reloaded cohort manifest `unit` and calibrated prediction `method`/`unit`
fields must already be strings; objects with a valid-looking `__str__` are
rejected instead of being stringified into contract values.
Reload is artifact-derived, not just self-signed: the stored quantiles must
recompute from the retained non-negative residual scores, and every materialized
interval in `CalibratedRunResult` must equal `y_pred ± qhat` for the embedded
artifact coverage. Grouped conformal results store a row-aligned `qhat` vector
derived from each prediction row's calibrated group key; edited group keys,
vector qhat values, intervals, quantiles, or non-integer grouped `n_samples`
summaries are rejected before exposure to Python, CLI, bindings or Studio.
Filesystem stores, workspace
`conformal_results` rows, conformal-only `.n4a` bundles and model `.n4a`
sidecars preserve `group_keys`, `group_calibrators` and row-aligned grouped
`qhat` vectors, then revalidate them against the embedded artifact on reload.
Version fields on conformal cohort manifests, calibration artifacts and
calibrated results are strict integer contract tags: boolean `true`/`false` and
numeric strings are rejected on direct construction and reload instead of being
coerced to schema version `1`. Optional conformal artifact identity strings
(`target_name`, `predictor_fingerprint`, `calibration_data_fingerprint`) must be
either `None` or non-empty strings without surrounding whitespace or NULs; reload
rejects the same invalid values instead of publishing ambiguous provenance.
Conformal numeric arrays (`y_true`, `y_pred`, interval bounds and `qhat`) must
contain real numeric values; boolean payloads fail closed instead of being
coerced to `0.0`/`1.0`, and numeric strings such as `"1.0"` are rejected instead
of being parsed as floats. The same bool rejection applies to Python-side
`from_dict(...)` payloads carrying NumPy boolean scalars in serialized scores or
quantiles, and serialized numeric fields reject NumPy ndarray scalars instead of
coercing them to JSON numbers. Direct `ConformalIntervalBlock` and
`CalibratedPredictionBlock` construction also fails closed for coverage-key
mismatches, interval shape mismatches, inverted bounds, unsupported method/unit
values, invalid group-key lengths, and negative or non-row-aligned `qhat` values.
Direct `SplitConformalCalibrator` construction validates retained residual
scores, coverage keys, recomputed quantiles, method and unit before `apply()`;
negative scores, edited `qhat` values or unsupported vocabulary fail closed
instead of materializing invalid intervals.
`ConformalMetricSet` diagnostics also fail closed when non-finite, out of range,
negative, or inconsistent with their counts and coverage arithmetic; only
positive infinity is accepted for unbounded interval width/score diagnostics.

The PC2 internal cohort seam is
`normalize_conformal_calibration_cohort()`. It requires explicit row-aligned
`sample_ids`, preserves the `calibration` role, optional group labels and
JSON-compatible metadata, and emits a signed
`ConformalCalibrationCohortManifest`. Missing or duplicated physical sample ids
are rejected instead of being silently synthesized for conformal guarantees.
Manifest rows built directly or reloaded from JSON must also carry a strict
non-boolean integer `row_index`, canonical non-whitespace-padded `sample_id`,
`role` and optional `group`, plus strict JSON-native metadata with string keys;
invalid row payloads fail before fingerprinting. Row-aligned calibration
metadata supplied as either column mappings or per-row mappings uses the same
strict key rule: non-string or whitespace-padded metadata keys fail before they
can be coerced into manifest JSON. The optional serialized
`n_samples` summary must be a strict non-boolean integer matching the row count.
A
`CalibratedRunResult` also requires prediction `sample_ids` for every non-empty
prediction cohort, refuses empty, whitespace-padded, non-string or duplicate
ids, rejects prediction ids that overlap the embedded calibration cohort, and
requires result metadata keys and values to be strict JSON-compatible before
fingerprinting or persistence. Nested mapping keys must also be strings without
surrounding whitespace, and tuple values are rejected instead of being silently
coerced into JSON arrays.
Initial calibration, reload, workspace conversion, and later
`predict_calibrated()` application all preserve a required, canonical, unique
and disjoint physical-sample boundary.

`calibrate_replayed_predictions()` is the internal orchestration seam behind the
current public replayed-array surface:
given calibration targets, replayed calibration predictions, prediction outputs
and explicit physical sample ids, it builds the cohort manifest, fits the
artifact, applies the requested coverages, and returns a `CalibratedRunResult`.
`PredictResult` now also carries optional `intervals`, exposes
`interval(coverage)`, and includes interval columns in `to_dataframe()`. Broader
public forms still require dataset calibration loading and automatic predictor
replay for calibration fitting.

For filesystem persistence, the internal
`nirs4all.pipeline.dagml.conformal_store` module writes a calibrated result
directory with `manifest.json`, `artifact.json`, and `calibrated_result.json`.
Loading verifies the manifest fingerprint, the artifact fingerprint, the result
fingerprint, and that the result embeds the same conformal artifact as the
standalone artifact file. Reload also revalidates the `CalibratedRunResult`
identity contract, including required canonical prediction `sample_ids`. The
verified result intervals are re-derived against the embedded artifact quantiles,
so archive/workspace rows cannot publish stale or widened intervals by editing
only `calibrated_result.json`.
The
public replayed-array surface exposes this through `store_path` and
`load_calibrated_result()`. This is still a local store
substrate. The workspace-backed surface stores the same verified JSON in the
`conformal_results` SQLite table via `workspace_path`,
`save_workspace_calibrated_result()`, and
`load_workspace_calibrated_result()`. The table stores both artifact/result JSON
and their fingerprints, so reload verifies the same contracts as filesystem and
archive forms. `bundle_path` and `export_calibrated_result()` wrap the same
verified store in a `.n4a` archive for calibrated replayed-array results; they
do not yet bundle a predictor/model artifact by themselves.
`attach_calibrated_result_to_bundle()` copies an existing model `.n4a` and adds
the verified `conformal/` sidecar.
`predict_calibrated()` applies the stored calibrator to already computed point
predictions. `predict(model=calibrated, data={"y_pred": ..., "sample_ids": ...},
coverage=...)` routes to the same explicit replayed-array lane and only selects
already materialized coverages. The supplied prediction `sample_ids` are
required for non-empty predictions and must be non-empty canonical strings,
unique and disjoint from the calibration cohort embedded in the conformal
artifact.
Applying the calibrator preserves the source result's `calibration_replay_source`
inside the generated `conformal_guarantee_status`; user metadata cannot replace
that provenance or the generated guarantee status.
`calibrate.result_metadata` and `predict_calibrated.result_metadata` are strict
JSON-native metadata mappings. They affect persisted calibrated result metadata
and fingerprint only; they do not refit the calibrator or renew a coverage
guarantee, and generated guarantee/source metadata is appended by nirs4all after
validation.
`predict(model="model.calibrated.n4a",
data={"X": ..., "sample_ids": ...}, coverage=...)` replays the model bundle and
then applies the attached conformal sidecar. All these conformal paths carry
`conformal_guarantee_status`; selecting a subset with `coverage=...` updates
that status to the selected coverages without recalibrating.
`predict.all_predictions` controls the prediction entry selection boundary:
`all_predictions=False` is the only conformal sidecar mode currently supported,
while `all_predictions=True` remains fail-closed until every returned prediction
entry can carry calibrated identity mapping. The coverage selector
accepts one finite coverage or a non-empty list of finite, unique coverages
strictly between 0 and 1; invalid selectors are rejected before interval
selection. Selectors that are valid but absent from the calibrated artifact are
also rejected instead of triggering implicit recalibration.
If a model bundle advertises a `conformal/` sidecar, that sidecar is validated
strictly; incomplete, duplicated, or unexpected sidecar members fail closed
instead of falling back to an uncalibrated prediction path.

(workspace-prediction-bridge)=

`predict.save_to_workspace`, `predict.workspace_metadata` and
`predict.workspace_result_metadata` describe the prediction-time publisher.
When `save_to_workspace=True`, `nirs4all.predict()` writes the returned
`PredictResult` through the workspace prediction store, persists executable
`X`/`spectra` when available from `data`, and returns
`metadata["workspace_prediction_id"]`. Workspace prediction ids are canonical
non-empty strings without surrounding whitespace or NULs; lower-level explicit
`prediction_id` writes fail closed on invalid ids instead of generating or
stringifying a replacement. `workspace_metadata` is sample-level
metadata for the prediction sidecar. `workspace_result_metadata` is result-level
metadata and is the supported place to publish `robustness_evidence` such as a
`predictor_bundle` for later spectral/OOD replay. These keys affect workspace
prediction rows and arrays only; they do not persist conformal artifacts or
renew guarantees. Both metadata mappings are strict JSON-native: keys must be
canonical strings and numeric values must be finite before the prediction
sidecar is written.

The machine-readable registry behind this reference is available from the
top-level Python package:

```python
registry = nirs4all.get_keyword_registry()
registry_json = nirs4all.keyword_registry_json(indent=2)
registry_schema = nirs4all.get_keyword_registry_schema()
registry_schema_json = nirs4all.keyword_registry_schema_json(indent=2)
```

For build systems and release artifacts, the same document can be emitted from
the CLI:

```bash
nirs4all keyword-registry --output keyword-registry.json
nirs4all keyword-registry --schema --output keyword-registry.schema.json
nirs4all robustness-summary-schema --output robustness-summary.schema.json
```

Every successful HTML documentation build also publishes
`_static/keyword-registry.json` and `_static/keyword-registry.schema.json` next
to the generated pages, plus `_static/robustness-summary.schema.json` for
robustness dashboard/card consumers.

Generated docs, Studio, forms and bindings can use it to inspect keyword paths,
value schemas, lifecycle effects, calibration invalidation semantics, engine
support and UI hints without parsing prose.

(planned-robustness-campaigns)=
### Robustness/generalization reports

`nirs4all.robustness()` is available for the first audit-only report surface:
already replayed predictions are compared with observed `y_true`, optionally
with materialized conformal intervals, diagnostic metadata slices, and explicit
frozen-predictor or saved-bundle replay for the bounded `spectral_noise` and
`spectral_offset`/`spectral_scale`/`spectral_slope`/`spectral_shift` cells.

```python
report = nirs4all.robustness(
    predict_result,
    y_true=y_external,
    X=X_external,
    predictor_bundle="model.n4a",
    mode="clean_frozen",
    scenarios=[
        nirs4all.RobustnessScenarioSpec(kind="observed"),
        nirs4all.RobustnessScenarioSpec(kind="prediction_bias", severity=0.2),
        nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity=0.05),
        nirs4all.RobustnessScenarioSpec(kind="spectral_noise", severity=0.001),
        nirs4all.RobustnessScenarioSpec(kind="spectral_offset", severity=0.01),
        nirs4all.RobustnessScenarioSpec(kind="spectral_scale", severity=0.03),
        nirs4all.RobustnessScenarioSpec(kind="spectral_slope", severity=0.02),
        nirs4all.RobustnessScenarioSpec(kind="spectral_shift", severity=0.5),
    ],
    metadata={"instrument": instrument_ids, "batch": batch_ids},
    slice_by=["instrument", "batch"],
    seed=123,
    workspace_path="workspace/",
    workspace_robustness_id="external-audit",
    workspace_name="External robustness audit",
)

report.scenarios[0].metrics.rmse
report.scenarios[0].conformal_metrics[0.9].observed_coverage
report.scenarios[0].slices[0].metrics.mae
report.summary_rows()
report.degradation_rows()
report.worst_slices(metric="rmse", top_k=3)
same_report_from_result = predict_result.robustness(y_true=y_external)
report.save_json("robustness-report.json")
report.save_summary("robustness-summary.json")
report.save_markdown("robustness-report.md")
report.save_html("robustness-report.html")
report.save_parquet("robustness-report.parquet")
report.save_artifacts("robustness-artifacts")
reloaded = nirs4all.RobustnessReport.load_json("robustness-report.json")
reloaded_tables = nirs4all.RobustnessReport.load_parquet("robustness-report.parquet")
reloaded_bundle = nirs4all.RobustnessReport.load_artifacts("robustness-artifacts")
robustness_id = nirs4all.save_workspace_robustness_report(
    "workspace/",
    report,
    robustness_id="external-audit-copy",
    name="External robustness audit",
)
reloaded_workspace = nirs4all.load_workspace_robustness_report("workspace/", "external-audit")
```

Caller-provided robustness ids must also be canonical non-empty strings without
surrounding whitespace or NULs. Omitting the id asks the workspace store to
generate one and does not change the report fingerprint.
Optional robustness workspace link ids, including `run_id`, `pipeline_id`,
`chain_id`, source `conformal_id` and source `prediction_id`, use the same
strict identifier boundary when supplied.

For CI/release jobs that already have the verified JSON artifact, the CLI can
republish the same report without re-running the robustness audit:

```bash
nirs4all robustness-report robustness-report.json --format markdown --output robustness-report.md
nirs4all robustness-report robustness-report.json --format summary --output robustness-summary.json
nirs4all robustness-report robustness-report.json --format html --output robustness-report.html
nirs4all robustness-report robustness-report.json --format parquet --output robustness-report.parquet
nirs4all robustness-report robustness-report.json --format artifacts --output robustness-artifacts/
nirs4all robustness-report robustness-artifacts/ --format markdown --output reviewed-report.md
nirs4all workspace robustness list --workspace workspace --json
nirs4all workspace robustness show external-audit --workspace workspace --json
nirs4all workspace robustness from-prediction --workspace workspace --prediction-id pred-001 --y-true "1.0,2.0,3.0" --scenarios-json '[{"kind":"spectral_offset","severity":0.01}]' --save-to-workspace --workspace-robustness-id pred-001-spectral-audit --format summary --output robustness-summary.json
nirs4all workspace robustness export external-audit --workspace workspace --format summary --output robustness-summary.json
nirs4all workspace robustness export external-audit --workspace workspace --format artifacts --output robustness-artifacts/
nirs4all workspace robustness export external-audit --workspace workspace --format html --output robustness-report.html
```

The current executable scope is intentionally narrow:

- `mode="clean_frozen"` only;
- `nirs4all.ROBUSTNESS_MODES` exposes the reserved mode vocabulary, while
  `nirs4all.ROBUSTNESS_EXECUTABLE_MODES` exposes the currently executable
  subset. The keyword registry mirrors this via
  `robustness.mode.value_schema["x-executable-values"]`;
- already replayed `PredictResult` or `CalibratedRunResult`;
- `{"kind": "observed", "severity": 0.0}` for the unmodified prediction
  audit cell;
- `{"kind": "prediction_bias", "severity": offset}` for a deterministic
  post-prediction stress cell; the offset is added to `y_pred`, and already
  materialized conformal lower/upper bounds are shifted by the same amount;
- `{"kind": "prediction_noise", "severity": sigma}` for a seeded
  post-prediction stress cell; `distribution="normal"` uses `severity` as the
  standard deviation, `distribution="uniform"` samples from
  `[-severity, +severity]`, one noise value is drawn per sample, added to
  `y_pred`, and applied to that sample's materialized conformal lower/upper
  bounds;
- `{"kind": "spectral_noise", "severity": sigma}` for the first bounded
  evaluation-only spectral replay cell. It requires explicit `X=` and
  exactly one frozen replay source, `predictor=` or `predictor_bundle=`, adds
  seeded normal or centered uniform noise to `X`, replays the frozen predictor
  through the in-memory hook or public `nirs4all.predict()` bundle path, and
  recenters already materialized conformal intervals by the prediction delta. It
  does not refit or recalibrate;
- `distribution` is accepted only for `prediction_noise` and `spectral_noise`,
  and currently accepts `"normal"` or `"uniform"`. Supplying it on deterministic
  scenarios is rejected so configs do not carry ignored stochastic parameters;
- `{"kind": "spectral_offset", "severity": offset}` for a deterministic
  evaluation-only spectral replay cell. It requires explicit `X=` and
  exactly one frozen replay source, `predictor=` or `predictor_bundle=`, adds
  the same offset to every feature in `X`, replays the frozen predictor, and
  recenters already materialized conformal intervals by the prediction delta.
  It does not refit or recalibrate;
- `{"kind": "spectral_scale", "severity": relative_factor}` for a deterministic
  evaluation-only multiplicative-scatter replay cell. It requires explicit `X=`
  and exactly one frozen replay source, `predictor=` or `predictor_bundle=`,
  multiplies every feature by `1.0 + severity`, replays the frozen predictor,
  and recenters already materialized conformal intervals by the prediction
  delta. `1.0 + severity` must remain strictly positive. It does not refit or
  recalibrate;
- `{"kind": "spectral_slope", "severity": slope_amplitude}` for a deterministic
  evaluation-only baseline-tilt replay cell. It requires explicit `X=` and
  exactly one frozen replay source, `predictor=` or `predictor_bundle=`, adds a
  centered linear ramp from `-severity/2` to `+severity/2` across the feature
  axis, replays the frozen predictor, and recenters already materialized
  conformal intervals by the prediction delta. It does not refit or
  recalibrate;
- `{"kind": "spectral_shift", "severity": feature_offset}` for a deterministic
  evaluation-only wavelength/feature-shift replay cell. It requires explicit
  `X=` and exactly one frozen replay source, `predictor=` or
  `predictor_bundle=`, shifts spectra along the feature axis by fractional
  feature units using interpolation with edge-value bounds, replays the frozen
  predictor, and recenters already materialized conformal intervals by the
  prediction delta. It does not refit or recalibrate;
- `scenarios` also accepts the typed
  `nirs4all.RobustnessScenarioSpec(...)` helper. The mapping syntax remains the
  lowest-level public form, while the typed helper gives Python, forms, Studio,
  and bindings a stable constructor with fail-closed validation and
  `to_dict()` serialization. `kind` and `distribution` must be real strings,
  not host objects that stringify to supported values; `severity` must be a
  real numeric scalar, not a boolean or numeric string; and `extra` keys must be
  canonical non-empty strings without NULs before `to_dict()` can publish the
  scenario. Raw scenario mappings use the same strict boundary: keys must be
  canonical non-empty strings without NULs, `kind` and `distribution` must be
  real strings, `severity` must be a real finite numeric scalar, and the payload
  must be TCV1 JSON-native and fingerprintable; Python objects, non-finite
  numbers, bytes and other opaque values are rejected before report execution;
- `nirs4all.ROBUSTNESS_SCENARIO_KINDS` exposes the same `kind` vocabulary at
  runtime for Python tooling, bindings and Studio forms; the tuple is aligned
  with report metadata and the keyword registry enum;
- `nirs4all.ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS` exposes the scenario kinds
  that accept `distribution`, and
  `nirs4all.ROBUSTNESS_SCENARIO_DISTRIBUTIONS` exposes the accepted
  distribution vocabulary. Both are aligned with the runtime validator and the
  keyword registry schema;
- `seed` controls `prediction_noise` and `spectral_noise` reproducibility.
  `spectral_offset`, `spectral_scale`, `spectral_slope` and `spectral_shift` metrics are
  deterministic for fixed `X`/predictor/severity, although the seed remains recorded in report
  metadata. If omitted, the report uses `effective_seed=0` so repeated calls
  remain deterministic;
- `robustness.workspace_metadata` is strict JSON-native workspace-row metadata:
  keys must be canonical strings and numeric values must be finite before the
  `robustness_results` row is inserted. It is display/indexing metadata only and
  does not enter the report fingerprint;
- `RobustnessReport.metadata` is stricter report evidence: it enters the report
  fingerprint, so keys must be canonical non-empty strings without NULs and
  values must stay TCV1 JSON-native before JSON, summary or artifact
  publication;
- `RobustnessScenarioResult.scenario` and
  `RobustnessSliceResult.slice_key` are strict published payload mappings:
  keys must be canonical non-empty strings without NULs, values must stay TCV1
  JSON-native, and result `severity` must be a real numeric scalar. These
  mappings feed report JSON, fingerprints, summary/degradation/worst-slice rows
  and tabular exports, so reloads fail closed instead of stringifying corrupted
  Python keys;
- spectral reports record `metadata["spectral_replay"]` with source
  (`predictor` or `predictor_bundle`), the saved bundle path when applicable, the
  replay route (`nirs4all.predict` for bundles), and whether sample ids were
  forwarded;
- point metrics: RMSE, MAE, bias, max absolute error;
- conformal diagnostics when intervals are already materialized;
- diagnostic slices over supplied row metadata;
- compact per-scenario summary rows for CI and Studio cards, using
  `report.summary_rows(reference=0)`;
- derived degradation rows versus a reference scenario, using
  `report.degradation_rows(reference=0)`;
- worst diagnostic slices, using `report.worst_slices(metric="rmse", top_k=3)`;
- deterministic JSON export/reload with fingerprint verification;
- deterministic Markdown and standalone HTML summary exports for CI and release
  artifacts, including conformal guarantee status details (requested/effective
  engine, method, unit, selected coverages, invalidation reasons and
  limitations), compact scenario summary, degradation and worst-slice sections
  when applicable.
- deterministic Parquet-directory export for tabular consumers, using
  `report.save_parquet("robustness-report.parquet")`. The directory contains
  `manifest.json`, `report.json`, and one Parquet file per non-empty table,
  including the compact `summary` table when scenarios are present.
  `manifest.json` keeps table filenames, row counts, and fingerprints.
  `RobustnessReport.load_parquet(...)` reloads and verifies the embedded
  report fingerprint plus the exported table files when table metadata is
  present.
- deterministic publication bundle, using
  `report.save_artifacts("robustness-artifacts")`. The directory manifest lists
  selected `report.json`, `summary.json`, `report.md`, `report.html`, and
  `report.parquet/` outputs so CI, release automation, bindings, and Studio can
  consume one stable artifact root. `summary.json` carries fingerprint, mode,
  report version, slice keys, optional `conformal_guarantee_status`, optional
  `spectral_replay`, and compact `summary_rows()` for cards/dashboards. The
  guarantee and spectral replay blocks are copied from report metadata when
  present so consumers can display engine, coverage, invalidation details, replay
  source, bundle path and sample-id forwarding status without parsing the full
  report.
  The same payload is available through `summary_artifact()`,
  `to_summary_json()`, `save_summary(...)`, and CLI `--format summary`; its
  validation schema is exposed by `get_robustness_summary_schema()` and
  `robustness_summary_schema_json()`, and docs builds publish the same contract
  as `_static/robustness-summary.schema.json`. CLI publication accepts the full
  bundle target through `--format artifacts --output <directory>` on both
  `robustness-report` and `workspace robustness export`.
  `RobustnessReport.load_artifacts(...)` reloads and verifies the
  bundle manifest, fingerprint, deterministic summary JSON, Markdown/HTML
  renderings, and Parquet table manifest when present. The `robustness-report`
  command accepts either a JSON file or a verified artifact directory as input
  before republishing to any supported output format.
- optional workspace persistence for Studio/CI inventory, using
  `save_workspace_robustness_report()` and
  `load_workspace_robustness_report()`.

The canonical future modes remain:

- `clean_frozen`: reuse predictor and calibrator; shifted intervals are
  diagnostic rather than a renewed coverage guarantee;
- `matched_recalibration`: reuse the predictor and fit a new, disjoint
  calibrator matched to the scenario;
- `structural_refit`: fit a new predictor and then recalibrate it, or report
  calibration as invalidated.

The result reports predictor lifecycle, calibrator lifecycle, and statistical
coverage scope separately. `slice_by` only adds diagnostic views. Observing
coverage in an instrument or batch slice does not create a conditional
conformal guarantee.

---

## `model`

Defines the model step in the pipeline.

**Syntax:**
```python
{"model": PLSRegression(n_components=10)}
```

**Behavior:** The operator is treated as the model to be trained and evaluated. During cross-validation, it is fit on training folds and evaluated on test folds. During refit, it is trained on the full dataset.

**Notes:**
- Any sklearn-compatible estimator with `fit()` and `predict()` methods can be used.
- A pipeline can have multiple model steps (e.g., in stacking).
- If a bare estimator is detected (not wrapped in a dict), nirs4all auto-detects it as a model based on whether it has a `predict()` method.

---

## `y_processing`

Applies a transformer to the target variable (y) before model training. The inverse transform is automatically applied to predictions.

**Syntax:**
```python
{"y_processing": MinMaxScaler()}
```

**Behavior:** The transformer is fit on y_train, then transforms y_train for model fitting. During prediction, the model output is inverse-transformed back to the original scale.

**Notes:**
- The transformer must implement `inverse_transform()` for prediction to work correctly.
- Common use: `MinMaxScaler()`, `StandardScaler()` for target normalization.

---

## `tag`

Marks samples with a tag for downstream analysis or branching. Tagged samples are NOT removed from the training set.

**Syntax:**
```python
# Single filter
{"tag": YOutlierFilter(method="iqr")}

# Multiple filters
{"tag": [YOutlierFilter(method="iqr"), XOutlierFilter(method="mahalanobis")]}
```

**Behavior:** The filter is evaluated and sample tags are stored in the dataset. Tags can be used later for separation branching via `{"branch": {"by_tag": ...}}`.

**Notes:**
- Tags are metadata annotations, not data modifications.
- Tag names are auto-generated from the filter class and method (e.g., `y_outlier_iqr`).
- Use `tag_name` parameter on the filter to set a custom tag name.

---

## `exclude`

Removes flagged samples from the training set. Test samples are never excluded.

**Syntax:**
```python
# Single filter
{"exclude": YOutlierFilter(method="iqr", threshold=1.5)}

# Multiple filters with mode
{"exclude": [
    YOutlierFilter(method="iqr"),
    XOutlierFilter(method="mahalanobis", threshold=3.0),
], "mode": "any"}
```

**Parameters:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `mode` | `"any"` (default) | Exclude sample if ANY filter flags it |
| `mode` | `"all"` | Exclude sample only if ALL filters flag it |

**Behavior:** Filters are evaluated and matching samples are removed from the training indexer. The actual data arrays are not modified -- samples are masked out via the indexer.

---

## `branch`

Creates parallel sub-pipelines. Two modes: **duplication** (same samples, different pipelines) and **separation** (disjoint sample subsets).

### Duplication Branches

All branches receive the same samples and process them independently.

**Syntax:**
```python
{"branch": [
    [SNV(), PLSRegression(10)],
    [MSC(), RandomForestRegressor()],
]}
```

### Separation Branches

Branches receive disjoint subsets of samples.

**By metadata column:**
```python
{"branch": {"by_metadata": "site"}}
```

**By tag values:**
```python
{"branch": {"by_tag": "y_outlier_iqr", "values": {
    "clean": False,
    "outliers": True,
}}}
```

**By source (multi-source datasets):**
```python
{"branch": {"by_source": True, "steps": {
    "NIR": [SNV(), PLSRegression(10)],
    "markers": [MinMaxScaler(), Ridge()],
}}}
```

**By filter:**
```python
{"branch": {"by_filter": SampleFilter(...)}}
```

**Notes:**
- Duplication branches are typically followed by `{"merge": "predictions"}` for stacking.
- Separation branches are typically followed by `{"merge": "concat"}` to reassemble samples.

---

## `merge`

Combines outputs from preceding branches.

**Syntax:**
```python
# For duplication branches (stacking)
{"merge": "predictions"}   # Use OOF predictions as features for meta-model
{"merge": "features"}      # Use transformed features as input for next step
{"merge": "all"}           # Merge all available outputs

# For separation branches (reassembly)
{"merge": "concat"}        # Reassemble samples in original order

# For multi-source merging
{"merge": {"sources": "concat"}}
```

| Value | Use Case | Description |
|-------|----------|-------------|
| `"predictions"` | Duplication branches | Out-of-fold predictions become features for a meta-model |
| `"features"` | Duplication branches | Transformed feature matrices are concatenated horizontally |
| `"all"` | Duplication branches | All available outputs (features + predictions) |
| `"concat"` | Separation branches | Reassembles disjoint sample subsets in original order |

---

## `sample_augmentation`

Applies data augmentation to training samples. Augmented samples are added to the training set but not to the test set.

**Syntax:**
```python
# Single augmenter
{"sample_augmentation": GaussianAdditiveNoise(sigma=0.01)}

# Multiple augmenters (applied sequentially)
{"sample_augmentation": [
    GaussianAdditiveNoise(sigma=0.01),
    WavelengthShift(shift_range=(-1.0, 1.0)),
]}
```

**Behavior:** The augmentation operator is applied to training data only. Augmented copies are appended to the training set. During prediction, this step is skipped.

**Notes:**
- Augmentation is only applied during training, never during prediction.
- See {doc}`../reference/augmentations` for all available augmentation operators.

---

## `feature_augmentation`

Creates multiple preprocessing views of the same data. Supports three action modes.

**Syntax:**
```python
# Extend mode (default) - add new views alongside existing
{"feature_augmentation": [SNV(), SavitzkyGolay(deriv=1)]}

# With explicit action mode
{"feature_augmentation": [SNV(), Gaussian()], "action": "extend"}
{"feature_augmentation": [SNV(), Gaussian()], "action": "add"}
{"feature_augmentation": [SNV(), Gaussian()], "action": "replace"}
```

**Action Modes:**

| Action | Growth Pattern | Description |
|--------|---------------|-------------|
| `"extend"` (default) | Linear | Each operator runs independently on the base. New views are added to the set. |
| `"add"` | Multiplicative + originals | Each operator is chained on top of ALL existing views. Originals are kept. |
| `"replace"` | Multiplicative | Each operator is chained on top of ALL existing views. Originals are discarded. |

**Example with `"extend"` (starting from raw_A):**
```
raw_A, SNV, SavitzkyGolay  (3 views)
```

**Example with `"add"` (starting from raw_A):**
```
raw_A, raw_A+SNV, raw_A+SavitzkyGolay  (3 views)
```

**Example with `"replace"` (starting from raw_A):**
```
raw_A+SNV, raw_A+SavitzkyGolay  (2 views, raw_A discarded)
```

---

## `concat_transform`

Applies multiple transforms and concatenates the resulting features horizontally.

**Syntax:**
```python
{"concat_transform": [SNV(), SavitzkyGolay(deriv=1), Detrend()]}
```

**Behavior:** Each transform is applied independently to the same input. The resulting feature matrices are concatenated along the feature axis, producing a wider feature matrix.

**Notes:**
- Different from `feature_augmentation` which creates separate preprocessing views.
- `concat_transform` produces a single 2D matrix with concatenated features.

---

## `rep_to_sources`

Converts repetition groups (multiple spectra per physical sample) into a multi-source format.

**Syntax:**
```python
{"rep_to_sources": "Sample_ID"}
```

**Behavior:** Groups spectra by the specified metadata column (e.g., `"Sample_ID"`). Each repetition becomes a separate source in the multi-source dataset.

---

## `rep_to_pp`

Converts repetition groups into separate preprocessing pipelines.

**Syntax:**
```python
{"rep_to_pp": "Sample_ID"}
```

**Behavior:** Groups spectra by the specified metadata column. Each repetition becomes a separate preprocessing view, enabling per-repetition processing.

---

## `name`

Names a pipeline step for display and identification purposes.

**Syntax:**
```python
{"name": "scatter_correction", "step": SNV()}
```

**Behavior:** The step is executed normally, but uses the provided name in logs, reports, and visualization.

---

## Combining Keywords

Multiple keywords can be combined in a single dict when they apply to the same step:

```python
# Exclude with mode
{"exclude": [YOutlierFilter(), XOutlierFilter()], "mode": "any"}

# Feature augmentation with action
{"feature_augmentation": [SNV(), Detrend()], "action": "extend"}

# Branch with steps
{"branch": {"by_source": True, "steps": {"NIR": [...], "VIS": [...]}}}
```

---

## See Also

- {doc}`../reference/generator_keywords` -- Generator syntax (`_or_`, `_range_`, `_grid_`, etc.)
- {doc}`../reference/transforms` -- Available transforms
- {doc}`../reference/augmentations` -- Available augmentation operators
- {doc}`../reference/filters` -- Available filters for `tag` and `exclude`
- {doc}`../reference/models` -- Available models
- {doc}`../reference/splitters` -- Available splitters
