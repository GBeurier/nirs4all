## ROADMAP

Add documentation for examples

**RELEASE** 0.6.1: Stable MVP


> [Design] Define all services
> [SERVICE FUNCTIONS] provides easy services functions. > cf. Service.md

> [WEBAPP] full react version - hidden fastapi / nirs4all

> [DEPLOY] standalone installer, web installer (gpu/no gpu - os) options for backend and CUDA


**Major Review**:

Creates use cases that covers all the diversity of controllers and syntax for pipelines -
Create the example synth pipelines x synth datasets for testing purpose

> Artifact overview and maybe refactoring
> GLOBAL REVIEW OF WORKFLOW MECHANISM (X,Y,M,Pred - indexed on Models, pp, branches, etc. - with context - What about ?)
> transfer, stacking, branching, multisource, pp, aggregation, pipeline inputs (launch from model, folder, file, yaml, json, etc.)
> [Complete_Review] Review modules one by one: pipeline, dataset, controllers, core,

**RELEASE** 0.7.0: UI
> [HuggingFace] Huggingface controller ?

> [PLS] make a pip librairie with torch/jax/numpy implementations of PLS.
>   - [MB-PLS] test on multi-source/block ---
>   - [PLS] Implement variable selection methods (CARS and MC-UVE done)

> [transformerMixin] implement in pytorch for full differentiation
> [FCK-PLS] full torch model

> [DAG] Pipeline as DAG
> [Pipeline] as a GridSearchCVGridSearchCV or FineTuner. Generation as a choices optimization provider.
> [Pipeline] as single transformer: pre-instanciate binaries, contruct pipeline, fit(), transform(), predict(), fit_transform(). pour SHAP NN. Decompose run and pipeline (1 pipeline per config tuple)
> [Pipeline] bring back parallelization of steps (feature_aug, sample_aug)
> [Pipeline_Bundle] Change / edit pipeline step
> [SHAP] verify shap for tf, torch, jax, Fix imports and np compat
> [obj_context] On controller compute something (ie. pp selection), put in in the context, another controller use it (ie. set pp).
> [Generator] add in-place/internal generation > branches
> [Observers] Replace data copy for analysis (pp dataset copy) by observers. Observers are controllers that can aggregate data and can be queried anytime to get the data and analyze them
> - [Runner] Verify: Design logic of 'execution sequence' and 'history' > pp and raw data, use cache by defaut, generalize default inputType (np.array, SpectroDataset, DatasetConfig, ...)
> [Models] extend the scope of custom model fallback (sklearn only for now), to include custom layouts (ie. custom NN without framework and 3D data, spectrograms, etc.)
> [Pipeline + Optuna] Pipeline as optuna trial. The pp become a choice param. Goal is to stack pp each time score stop progress, select the good ones by feats augmentation and by pp order (1st, 2nd, etc.) and stop once it drops.


**CLI_EXTENSION_PROPOSAL.md**
> [CLI] nirs4all renaming: nirs4all.train(), .predict(), .explain(), .transfer(), .resume(), .stack(), .analyze()
> [CLI]  Reup - run / predict / explain - directly on paths (dataset, pipeline config), json and yaml


> [Aggregation] Outlier dedicated exclusion T²

> [Mid Fusion] Multi head models
> [Late Fusion] avg / w_avg / asymetric ensembling
> [Transfer] Partial layers retraining

**Bugs**:
>   - [LightGBM] [Warning] No further splits with positive gain, best gain: -inf   >> look if parameters is passed on cloning (maybe there is a hidden bug)
>   - [Charts] check dataviz. Missing histograms

> [onnx] onnx export

> [Metrics] add custom losses - lambda / functions / classes; manage metrics per level (global, pipeline, model); clear metrics logic / usage / customization; clean the usage of default metrics and loss. Neg SCORE implementation to minimize, Review R2 computation / Q2 value - GOF (goodness of fit)


> [Chart_Controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators. and more. Both operators and analyzers should be uniformized (inside the pipeline or outside)
> [Charts] aggregate based on metadata col, convert std indexes (model_name, model_classname, pp, etc.) to enum, keep string only for columns. Add Y as grouping value, add variance, mean, etc. as sort score.

> [Analyses] Cf. Observers - Question the idea of Analysis Pipeline that use the whole run as input. If yes, move visualization classes as Analyses operator of this pipeline. Choose a default functionning for raw_pp and XXX_pp dedicated to data transformation analysis

> [Dummy_Controller] remove totally and manage exceptions

> [Optuna] Integrate complex params setters: Stack (sklearn stacking model), Nested_dict (TABPFN inference params)
> [Optuna] Add pruner (test BOHB ou successive halving pruner). Simplify force params in model to reuse best_params from older runs, review the syntax
> [Optuna] Allow complex scenarios (random X trials then TPE X trials)
> [Optuna] Allow sampling on training params not only finetune/model params

> [Operators] Reintroduce operators tests (cf. pinard for TransformerMixin) _ add data aug operators en masse.

> [Tests] clean workspace and run folder creation during tests.

> [Docker] provide a docker, add build and actions

> [Conda] provide a conda, add build and actions

> [Customizable_Feature_Source] Refactor dataset to allow customizable feature source and customizable layouts to allow datasets with more dimensions (images, lidar)


> [GLOBAL REVIEW] v1.0 signatures freeze (private pattern _module), Complete tests > Prod coverage (transformations, controllers, predictions, datasets, runner)
> [Profiling] Code Optimization, Improve performances


**RELEASE** beta


**RELEASE** 1.0: Release


> [Transfer] Automate model transfer across machines

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
