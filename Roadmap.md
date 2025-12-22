## ROADMAP

---

**Features**:
> [Test] Speed Up

Creates new datasets from nitrosorgh (binary, classif, regression, multisource (with repetitions), in gz, xls, zip, npy, mat, see. specifications) and dataset configs that cover all cases.
Creates use cases that covers all the diversity of controllers and syntax for pipelines

> [PipelineDAGChart] Display the whole DAG with data shape before each step

> [Runner] Design logic of 'execution sequence' and 'history' > pp and raw data, use cache by defaut, generalize default inputType (np.array, SpectroDataset, DatasetConfig, ...)
> [signature] change signatures to nirs4all.run(pipeline, dataset, config), nirs4all.predict, etc. Add nirs4all config.


**CLI_EXTENSION_PROPOSAL.md**
> [CLI] nirs4all renaming: nirs4all.train(), .predict(), .explain(), .transfer(), .resume(), .stack(), .analyze()
> [CLI]  Reup - run / predict / explain - directly on paths (dataset, pipeline config), json and yaml


> [pytoml] Update imports and configs. Anticipate ui dependencies.

> [Readme] link to all compatible models references and embed models by task_type and backend / link to all possible transformations (embed / compatible) by type (feature processing - smooth, deriv, etc. and smaple augmentation: noises, rotate, etc.)
> [Examples] update, clean and document examples and tutorial notebooks, Add examples with custom classes
> [Examples] Clean and document
> [Examples] Orgzanize and optimize the full run, add verbose global variables, REVIEW the tranformations to ensure pp are still ok and used by models.
> [Example] The full integration example with all features in one, using branching, multidatasets, etc.


**Major Review**:

> Artifact overview and maybe refactoring

> transfer, stacking, branching, multisource, pp, aggregation, pipeline inputs (launch from model, folder, file, yaml, json, etc.)
> review repetitions to source mechanism

**RELEASE** 0.6.0: MVP


> [Design] Define all services

> [WEBAPP] full react version - hidden fastapi / nirs4all

> [DEPLOY] standalone installer, web installer (gpu/no gpu - os) options for backend and CUDA

**RELEASE** 0.6.0: UI

> [Generator] add in-place/internal generation > branches

> [Aggregation] Outlier dedicated exclusion T²

> [PLS] make a pip librairie with torch/jax/numpy implementations of PLS.

> [Transfer] Partial layers retraining

**Bugs**:
>   - [MB-PLS] test on multi-source/block ---
>   - [Predict] Q5_predict is slow as hell
>   - [LightGBM] [Warning] No further splits with positive gain, best gain: -inf   >> look if parameters is passed on cloning (maybe there is a hidden bug)
>   - [Charts] check dataviz. Missing histograms



> [Pipeline_Bundle] Change / edit pipeline step

> [SHAP] verify shap for tf, torch, jax, Fix imports and np compat

> [obj_context] On controller compute something (ie. pp selection), put in in the context, another controller use it (ie. set pp).

> [onnx] onnx export

> [Complete_Review] Review modules one by one: pipeline, dataset, controllers, core,

> [Metrics] add custom losses - lambda / functions / classes; manage metrics per level (global, pipeline, model); clear metrics logic / usage / customization; clean the usage of default metrics and loss. Neg SCORE implementation to minimize, Review R2 computation / Q2 value - GOF (goodness of fit)

> [PLS] Implement variable selection methods (CARS and MC-UVE done)

> [Chart_Controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators. and more. Both operators and analyzers should be uniformized (inside the pipeline or outside)
> [Charts] aggregate based on metadata col, convert std indexes (model_name, model_classname, pp, etc.) to enum, keep string only for columns. Add Y as grouping value, add variance, mean, etc. as sort score.

> [Analyses] Question the idea of Analysis Pipeline that use the whole run as input. If yes, move visualization classes as Analyses operator of this pipeline. Choose a default functionning for raw_pp and XXX_pp dedicated to data transformation analysis

> [Pipeline] as a GridSearchCVGridSearchCV or FineTuner. Generation as a choices optimization provider.
> [Pipeline] as single transformer: pre-instanciate binaries, contruct pipeline, fit(), transform(), predict(), fit_transform(). pour SHAP NN. Decompose run and pipeline (1 pipeline per config tuple)
> [Pipeline] bring back parallelization of steps (feature_aug, sample_aug)

> [Dummy_Controller] remove totally and manage exceptions

> [Models] extend the scope of custom model fallback (sklearn only for now), to include custom layouts (ie. custom NN without framework and 3D data, spectrograms, etc.)

> [Optuna] Integrate complex params setters: Stack (sklearn stacking model), Nested_dict (TABPFN inference params)
> [Optuna] Add pruner (test BOHB ou successive halving pruner). Simplify force params in model to reuse best_params from older runs, review the syntax
> [Optuna] Allows complex scenarios (random X trials then TPE X trials)

> [Operators] Reintroduce operators tests (cf. pinard for TransformerMixin) _ add data aug operators en masse.

> [Tests] clean workspace and run folder creation during tests.

> [Docker] provide a docker, add build and actions

> [Conda] provide a conda, add build and actions

> [Customizable_Feature_Source] Refactor dataset to allow customizable feature source and customizable layouts to allow datasets with more dimensions (images, lidar)


**RELEASE** 0.8: CLI

> [GLOBAL REVIEW] v1.0 signatures freeze (private pattern _module), Complete tests > Prod coverage (transformations, controllers, predictions, datasets, runner)

> [SERVICE FUNCTIONS] provides easy services functions. > cf. Service.md

**RELEASE**  0.9 alpha: Minimum Viable Product. Signatures frozen.

> [Profiling] Code Optimization, Improve performances

> [REVIEW] Complete documentation (RTD, Tutorial, Examples), remove dead code and #TODOs, validate tests coverage

**RELEASE** 0.10 beta: Operators & Controllers rc

> [WEBAPP] full react version - hidden fastapi / nirs4all

> [DEPLOY] standalone installer, web installer

**RELEASE** 0.11 rc - GUI version (cf. nirs4all_ui)



**RELEASE** 1.0: Release



**RELEASE** 1.x.x

> [Pipeline + Optuna] Pipeline as optuna trial. The pp become a choice param. Goal is to stack pp each time score stop progress, select the good ones by feats augmentation and by pp order (1st, 2nd, etc.) and stop once it drops.

> [Transfer] Automate model transfer across machines



> [Mid Fusion] Multi head models

> [Late Fusion] avg / w_avg / asymetric ensembling

> [Clustering Controllers]

> [Analysis] t-sne, umap

> [HugginFace deploy]

> [CLUSTERED COMPUTATION + SERV/CLIENT]

## EXCITERS
- better model naming (with optional pp included) for UX
- add NorrisWilliams, Whittaker, BandEnergies, FiniteDiffCentral transformermixin
- Authorize vertical index (col 1 header - vertical header) in csv



















---
## POSSIBLE FUTURE INTEGRATION


### 1. Core ecosystem / array & data engines

* **pandas / Polars**

  * For tabular data prep, joins, grouping, etc.
  * Polars in particular for performance and lazy pipelines.
* **PyArrow**

  * For zero-copy interchange with Parquet, DuckDB, potentially GPU, etc.
* **DuckDB**

  * For local analytical queries on large datasets (and joining multiple sources before ML).

These three make data management + feature engineering much more powerful and scalable.

---

### 2. Model families that add diversity

You already have gradient boosting and deep nets. I would also cover:

* **Statsmodels**

  * For classical stats, GLM, mixed models, time-series (ARIMA, etc.).
  * Useful when reviewers/colleagues want “statistical” baselines or interpretable models.
* **GLM / GAM frameworks**

  * For example: **pyGAM**.
  * Good middle ground between linear and black-box models.
* **Probabilistic programming**

  * **PyMC** or **NumPyro** (since you already have JAX).
  * For Bayesian regression / uncertainty calibration of NIRS models.

You don’t need to go deep, but having wrappers for “probabilistic regression / calibration” is a big plus.

---

### 3. Time series & sequence-specific

Even if NIRS is not time series, you’re clearly doing climate / longitudinal stuff on the side:

* **tsfresh** or **Kats** (or at least some time series feature extraction lib).
* Optional: a thin integration with **pytorch-forecasting** or **neuralforecast** if you want deep TS models out-of-the-box.

---

### 4. Deep learning tooling around the cores

You have TF / Torch / JAX; I’d add:

* **Hugging Face ecosystem**:

  * `transformers` for generic sequence models (even for 1D spectra, time series, or text annotations).
  * `datasets` for standardized dataset handling and splits.
* **Lightning / Keras Tuner / Ignite** (optional)

  * One structured training loop framework can help standardize training, logging, callbacks.

You might not need them deeply if NIRS4ALL already provides its own training loop abstraction, but basic adapters can ease integration of external models.

---

### 5. Explainability & diagnostics beyond SHAP

You have SHAP; I’d also consider:

* **Captum** (for PyTorch)

  * For gradient-based attributions, integrated gradients, etc.
* **Alibi / Alibi-Detect**

  * For drift detection, outlier detection, and some local explanations.
* **Fairlearn** (optional)

  * If you ever need fairness metrics / constraints (maybe less central for NIRS but good for “coverage”).

---

### 6. Optimization, search, and experiment tracking

You have Optuna, which is excellent. To “round it out”:

* **Ray Tune** or **skopt** (optional)

  * Only if you want multi-backend HPO or alternative search strategies; Optuna alone is usually enough.
* **Experiment tracking**

  * Integrate with something: **MLflow**, **Weights & Biases**, or a simple internal tracker.
  * Even a minimal MLflow integration (params, metrics, artifacts) would greatly help adoption.

---

### 7. Deployment and model serving

Even if NIRS4ALL is mainly research-oriented, having basic deployment support gives good coverage:

* **ONNX / onnxruntime**

  * Export models from sklearn / Torch / TF to a common runtime.
* **FastAPI** bindings or templates

  * For turning a trained pipeline into a microservice.

---

### 8. Dimensionality reduction & manifold learning

Partly in sklearn, but I’d explicitly support:

* **UMAP-learn**

  * For non-linear embedding, very standard now for visualization + structure discovery.
* **hdbscan**

  * For density-based clustering in the embedded spaces.

These are standard tools in modern ML exploratory workflows.

---

### 9. Specialized “tabular” / “auto-ML” stacks (optional but nice)

For completeness of tabular ML:

* **TabNet / FT-Transformer** implementations

  * Either via PyTorch / TF implementations or 3rd-party libs.
* Optional: light integration with **AutoGluon** or **FLAML** if you want a “fast baseline” AutoML passthrough.

---

### 10. Infra/compute helpers around ML

Some are more infra than ML, but directly support ML workflows:

* **Dask / Ray**

  * For scaling up preprocessing and model training to multi-core / multi-node.
* **numba**

  * For fast custom transforms (spectral transforms, etc.) without full C++.

---

### Minimal “coverage” checklist

If I condense this to what I’d really ensure NIRS4ALL supports nativement (i.e. wrappers, configs, good integration):

1. Data backbone:

   * pandas, Polars, PyArrow, DuckDB
2. Models:

   * Statsmodels, pyGAM (or equivalent), PyMC/NumPyro (basic Bayesian regression)
3. Time series:

   * One TS feature extraction lib (tsfresh/Kats) + generic sequence models via Torch/TF
4. Deep ecosystem:

   * transformers (+ datasets) from Hugging Face
5. Explainability:

   * Captum, Alibi/Alibi-Detect (basic integration)
6. Experiment / HPO:

   * MLflow (or W&B) integration on top of Optuna
7. Embedding & clustering:

   * UMAP-learn, hdbscan
8. Deployment:

   * ONNX/onnxruntime + one web serving pattern (FastAPI template)
9. Scaling:

   * Dask or Ray support for parallel preprocessing / training.

With your current stack + the list above, you’d cover almost all modern ML “zones” (classical, gradient boosting, deep, probabilistic, time series, explainability, scaling, deployment) without turning NIRS4ALL into a kitchen sink.
