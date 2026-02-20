# nirs4all Library Backlog

Source: `Roadmap.md`
Goal: keep all roadmap tasks, but split them into executable version milestones.

Task markers:
- `[x]`: done
- `[-]`: partial
- `[ ]`: missing

Effort scale:
- `1`: low
- `5`: high

## Milestones

- `v0.8.0`: webapp `0.1.0` alignment and UI-compliant baseline.
- `v0.8.x`: feature expansion on top of the baseline.
- `v0.9.x`: architecture refactoring and quality hardening.
- `v1.0.0`: stabilization gates and distribution readiness.
- `v1.x`: post-`v1.0` feature set.
- `v2.x+`: long-horizon ecosystem and research backlog.

## v0.8.0 Backlog (Release Baseline)

### Services and release surface

- [-] `[Design]` Define all services. _(effort: 2; evidence: `nirs4all/api/run.py`, `nirs4all/pipeline/storage/artifacts/query_service.py`)_
- [-] `[SERVICE FUNCTIONS]` Provide easy service functions (cf. `Service.md`). _(effort: 2; evidence: `nirs4all/api/session.py`, `nirs4all/api/run.py`, `nirs4all/api/predict.py`)_
- [x] `[Docs]` Updated docs for the release scope. _(evidence: `docs/source/`)_

### Data IO and prediction reliability

- [x] `[Predictions]` Simplify and verify save behavior. _(evidence: `nirs4all/data/predictions.py`, `nirs4all/data/_predictions/result.py`, `tests/unit/data/test_predictions_store.py`)_
- [x] `[Loader]` Read scientific expressions in CSV. _(evidence: `nirs4all/data/loaders/csv_loader_new.py`)_


### Multi-source preprocessing baseline

- [x] `[MB-PLS]` Multi-source and multi-preprocessing support. _(evidence: `nirs4all/operators/models/sklearn/mbpls.py`, `tests/unit/operators/models/test_sklearn_pls.py`)_
- [x] `[Branching]` Clarify preprocessing-to-source vs preprocessing-to-branch behavior (aligned with MB-PLS). _(evidence: `nirs4all/controllers/data/branch.py`, `tests/integration/pipeline/test_branch*.py`)_

## v0.8.x Backlog (Feature Expansion)

### Modeling, optimization, and training

- [ ] `[Optuna]` Modularize god class. _(effort: 4; evidence: `nirs4all/optimization/optuna.py` remains monolithic)_
- [-] `[multivariate]` Define and implement multivariate scope. _(effort: 3; evidence: multivariate pieces exist across dataset/models/docs)_
- [-] `[Transfer]` Partial layers retraining or partial retrain on new data. _(effort: 3; evidence: retrain modes in `nirs4all/pipeline/retrainer.py`, partial freeze-layer path)_
- [-] `[Stacking]` Stacking from prediction files directly (hash for OOF and clean separation); use model path or chain path in pipeline. _(effort: 4; evidence: OOF/stacking flow in `nirs4all/controllers/data/merge.py`, direct prediction-file ingestion not explicit)_
- [-] `[PLS]` Make a pip library with torch/jax/numpy implementations of PLS. _(effort: 4; evidence: strong PLS operators exist, but no separate pip package and incomplete torch scope)_
- [x] `[MB-PLS]` Test on multi-source/block. _(evidence: `tests/unit/operators/models/test_sklearn_pls.py`, `tests/integration/pipeline/test_multisource*.py`)_
- [x] `[PLS]` Implement variable selection methods (CARS and MC-UVE already done). _(evidence: `nirs4all/operators/transforms/feature_selection.py`, `tests/unit/operators/transforms/test_feature_selection.py`)_
- [-] `[FCK-PLS]` Full torch model. _(effort: 4; evidence: torch version mostly in `bench/fck_pls/fckpls_torch.py`)_
- [-] `[transformerMixin]` Implement in PyTorch for full differentiation (learn from data to prediction). _(effort: 3; evidence: torch models exist, transformer-mixin parity incomplete)_
- [-] `[Metrics]` Add custom losses (lambda/functions/classes); manage metrics at global/pipeline/model levels; clarify metrics logic and customization; clean default metrics/loss usage; add negative score minimization behavior; review R2/Q2 and GOF handling. _(effort: 4; evidence: `nirs4all/core/metrics.py` + ranking logic, incomplete cross-level custom loss framework)_
- [ ] `[Fusion]` Mid fusion with multi-head models. _(effort: 4)_
- [-] `[Fusion]` Late fusion with final/avg/w_avg/asymmetric ensembling. _(effort: 2; evidence: avg/w_avg paths in `nirs4all/controllers/models/base_model.py`, asymmetric merge in `nirs4all/controllers/data/merge.py`)_
- [ ] `[Training]` Add prediction cache for stack sweeps retraining identical pipelines. _(effort: 4)_

- [-] `[CSV]` Authorize vertical index (col 1 header / vertical header) in CSV. _(effort: 3; evidence: index-column fixtures in `tests/unit/data/loaders/conftest.py`, no explicit vertical-header feature)_

### Operators, analysis, and visualization

- [-] `[Operators]` Add NorrisWilliams, Whittaker, BandEnergies, FiniteDiffCentral transformer mixin support. _(effort: 3; evidence: NorrisWilliams/whittaker present, BandEnergies/FiniteDiffCentral missing)_
- [x] `[Aggregation]` Outlier dedicated exclusion T² (MAD already implemented; add Hotelling T²). _(evidence: `nirs4all/operators/filters/y_outlier.py`, `nirs4all/operators/filters/x_outlier.py`)_
- [-] `[Analysis]` Add t-SNE. _(effort: 2; evidence: TSNE appears in catalog/reference, dedicated analysis workflow not complete)_
- [ ] `[HuggingFace]` Evaluate/implement Hugging Face controller. _(effort: 3)_
- [ ] `[Chart_Controller]` Migrate individual chart controllers into operators (x, y, folds, 3d, 2d, and more); uniformize operators and analyzers (inside or outside pipeline). _(effort: 5; evidence: chart logic still in `nirs4all/controllers/charts/*.py`)_
- [-] `[Charts]` Aggregate by metadata columns; convert standard indexes (`model_name`, `model_classname`, `pp`, etc.) to enums; keep strings for columns only; add Y grouping value; add variance/mean and related sort scores. _(effort: 3; evidence: aggregation utilities exist, full normalization scope incomplete)_
- [ ] `[Analyses]` Evaluate analysis pipeline using whole run as input; if retained, move visualization classes to analysis operators and define default behavior for `raw_pp` and `XXX_pp` transformation analysis. _(effort: 5)_
- [ ] `[Dummy_Controller]` Remove totally and manage exceptions. _(effort: 2; evidence: still present in `nirs4all/controllers/flow/dummy.py`)_
- [-] `[Transfer]` Automate model transfer across machines. _(effort: 3; evidence: bundle export/import in `nirs4all/pipeline/bundle/*`)_
- [-] `[Clustering Controllers]` Add clustering controllers. _(effort: 3; evidence: `nirs4all/controllers/transforms/cluster.py` exists but empty)_

## v0.9.x Backlog (Refactor + CI/CD Hardening)

### CI/CD, review, and performance

- [-] `[Complete_Review]` Review modules one by one: pipeline, dataset, controllers, core. _(effort: 3; evidence: audits in `docs/_internal/_archives/audits/`)_
- [-] `[CLI]` Update/review CLI (`CLI_EXTENSION_PROPOSAL.md`). _(effort: 2; evidence: active CLI commands/tests, proposal doc not found)_
- [-] `[Tests]` Review and clean tests. _(effort: 2; evidence: broad test suites present, still ongoing)_
- [x] `[Tests]` Clean workspace and run-folder creation during tests. _(evidence: `tests/integration/conftest.py` and test workspace isolation patterns)_
- [x] `[Operators]` Reintroduce operator tests (cf. Pinard for TransformerMixin) and add data augmentation operator tests en masse. _(evidence: `tests/unit/operators/*`, augmentation integration tests)_
- [-] `[Profiling]` Code optimization and performance improvements. _(effort: 3; evidence: caching/perf work exists, no final profiling closure)_

### Core refactoring track

- [-] `[Pipeline]` Rework as GridSearchCV/FineTuner; generation as a choices optimization provider. _(effort: 4; evidence: Optuna fine-tuning exists, full GridSearchCV-style abstraction incomplete)_
- [ ] `[Pipeline]` Rework as a single transformer: pre-instantiate binaries, construct pipeline, `fit()`, `transform()`, `predict()`, `fit_transform()`; support SHAP NN use case; decompose run and pipeline (1 pipeline per config tuple). _(effort: 5; evidence: `NIRSPipeline.fit()` unsupported in `nirs4all/sklearn/pipeline.py`)_
- [x] `[Pipeline]` Bring back parallelization of steps (feature augmentation, sample augmentation). _(evidence: `nirs4all/controllers/data/branch.py`, `nirs4all/controllers/data/sample_augmentation.py`)_
- [ ] `[Pipeline_Bundle]` Change/edit pipeline steps. _(effort: 3; evidence: bundle flow is export/import/replay, no explicit step-edit API)_
- [-] `[SHAP]` Verify SHAP for TF/Torch/JAX; fix imports and numpy compatibility. _(effort: 3; evidence: SHAP pipeline exists, cross-framework verification breadth partial)_
- [-] `[obj_context]` Let controllers compute context (for example PP selection) and reuse it in subsequent controllers. _(effort: 2; evidence: context/state machinery exists, broader scope still open)_
- [-] `[Generator]` Add in-place/internal generation for branches. _(effort: 3; evidence: generator infrastructure exists, explicit in-branch workflow partial)_
- [ ] `[Observers]` Replace copied analysis data (`pp` dataset copies) with observers that aggregate/query analysis data. _(effort: 4)_
- [-] `[Runner]` Verify design logic of execution sequence/history for PP and raw data; use cache by default; generalize default input types (`np.array`, `SpectroDataset`, `DatasetConfig`, ...). _(effort: 2; evidence: trace/cache and flexible inputs exist, broader validation ongoing)_
- [-] `[Models]` Extend custom model fallback beyond sklearn to custom layouts (for example custom NN without framework and 3D data/spectrograms). _(effort: 3; evidence: multi-backend support exists, fallback generalization incomplete)_
- [ ] `[Pipeline + Optuna]` Treat pipeline as Optuna trial; preprocessing becomes choice parameter; stack preprocessing while score improves; keep best by feature augmentation and PP order; stop when score drops. _(effort: 5)_
- [-] `[DAG]` Pipeline as DAG. _(effort: 5; evidence: DAG visualization/topology exists, no general DAG execution scheduler)_

## v1.0.0 Backlog (Stabilization + Deploy Gate)

### Stability and API freeze

- [-] `[GLOBAL REVIEW]` v1.0 signatures freeze (private pattern `_module`); complete tests and production coverage (transformations, controllers, predictions, datasets, runner). _(effort: 4; evidence: high test volume exists, no explicit signature-freeze gate)_

### Packaging and deployment

- [ ] `[Docker]` Provide Docker image and add build/actions. _(effort: 3; evidence: no `Dockerfile` / Docker workflow found)_
- [ ] `[Conda]` Provide Conda package and add build/actions. _(effort: 3; evidence: no Conda recipe/workflow found)_
- [ ] `[HuggingFace]` Deploy. _(effort: 3)_
- [ ] `[SERV/CLIENT]` Clustered computation + productization service doc. _(effort: 5)_

## v1.x Backlog (Post v1 Features)

- [ ] `[onnx]` ONNX export. _(effort: 3)_
- [ ] `[WanDB]` Weights & Biases compatibility. _(effort: 2)_
- [-] `[Customizable_Feature_Source]` Refactor dataset for customizable feature source and customizable layouts to support higher-dimensional datasets (images, lidar). _(effort: 3; evidence: strong architecture in `nirs4all/data/_features/*`, end-to-end images/lidar scope partial)_

## v2.x+ Backlog (Potential Future Features)

### Model families that add diversity

- [ ] `Statsmodels`: classical stats, GLM, mixed models, time-series (ARIMA, etc.) for interpretable/baseline models. _(effort: 2)_
- [ ] `GLM/GAM frameworks`: for example `pyGAM`, as middle ground between linear and black-box models. _(effort: 2)_
- [ ] `Probabilistic programming`: `PyMC` or `NumPyro` (JAX-aligned) for Bayesian regression and uncertainty calibration. _(effort: 4)_

### Time series and sequence-specific

- [ ] `tsfresh` or `Kats` (or equivalent time-series feature extraction library). _(effort: 2)_
- [ ] Optional thin integration with `pytorch-forecasting` or `neuralforecast` for deep TS models. _(effort: 3)_

### Deep learning tooling around the core

- [ ] `Hugging Face ecosystem`: `transformers` for generic sequence models. _(effort: 3)_
- [ ] `Hugging Face ecosystem`: `datasets` for standardized dataset handling and splits. _(effort: 2)_
- [ ] Optional structured training loop framework: Lightning, Keras Tuner, or Ignite. _(effort: 2)_

### Explainability and diagnostics beyond SHAP

- [ ] `Captum` (PyTorch): gradient-based attributions, integrated gradients, etc. _(effort: 2)_
- [ ] `Alibi` / `Alibi-Detect`: drift detection, outlier detection, and local explanations. _(effort: 3)_
- [ ] Optional `Fairlearn` support for fairness metrics/constraints. _(effort: 2)_
- [ ] Conformal learning. _(effort: 2)_

### Optimization, search, and experiment tracking

- [ ] Optional `Ray Tune` or `skopt` for multi-backend HPO/alternative search. _(effort: 3)_
- [ ] Experiment tracking integration: `MLflow`, `Weights & Biases`, or internal tracker. _(effort: 3)_
- [ ] Minimal MLflow integration (params, metrics, artifacts). _(effort: 2)_

### Dimensionality reduction and manifold learning

- [ ] `hdbscan` for density-based clustering in embedded spaces. _(effort: 1)_

### Specialized tabular and AutoML stacks (optional)

- [ ] `TabNet` / `FT-Transformer` integrations. _(effort: 3)_
- [ ] Optional light integration with `FLAML` for fast baseline AutoML passthrough. _(effort: 2)_

### Infra and compute helpers around ML

- [ ] `Dask` / `Ray` for scaling preprocessing and model training (multi-core/multi-node). _(effort: 4)_
- [ ] `numba` for faster custom transforms (spectral transforms, etc.) without full C++. _(effort: 3)_
