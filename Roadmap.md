## ROADMAP

> [MB-PLS] en multi-source, en multi pp, etc.
> [Branching] Preprocessing to source ou preprocessings to branch cf. MB-PLS
> [Predictions] Simplify save

> [Loader] read scientific expr in csv

**RELEASE** (webapp 0.1.0) - 0.8.0 UI compliant

> [Design] Define all services
> [SERVICE FUNCTIONS] provides easy services functions. > cf. Service.md

> [Parralellization] make a study on that


## FEATURES
> [Docs] Updated

> [Optuma] Modularize god class

> [multivariate]

> [Transfer] Partial layers retraining or partial retrain on new data.

> [Stacking] Stacking from predictions files directly (need hash for oof and clean separation) - use model path or chain path in pipeline

> [PLS] make a pip librairie with torch/jax/numpy implementations of PLS.
>   - [MB-PLS] test on multi-source/block ---
>   - [PLS] Implement variable selection methods (CARS and MC-UVE done)
> [FCK-PLS] full torch model

> [transformerMixin] implement in pytorch for full differentiation > learn from data to pred

> [Metrics] add custom losses - lambda / functions / classes; manage metrics per level (global, pipeline, model); clear metrics logic / usage / customization; clean the usage of default metrics and loss. Neg SCORE implementation to minimize, Review R2 computation / Q2 value - GOF (goodness of fit)

> [Operators] add NorrisWilliams, Whittaker, BandEnergies, FiniteDiffCentral transformermixin

> [Aggregation] Outlier dedicated exclusion T² (MAD already implemented, add Hotelling T²)

> [Analysis] t-sne

> [HuggingFace] Huggingface controller ?

> [CSV] Authorize vertical index (col 1 header - vertical header) in csv

> [Fusion] Mid: Multi head models
> [Fusion] Late: predictions final / avg / w_avg / asymetric ensembling

> [Training] model prediction cache for sweep in stack that retrain exactly same pipelines

> [Chart_Controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators. and more. Both operators and analyzers should be uniformized (inside the pipeline or outside)
> [Charts] aggregate based on metadata col, convert std indexes (model_name, model_classname, pp, etc.) to enum, keep string only for columns. Add Y as grouping value, add variance, mean, etc. as sort score.

> [Analyses] Cf. Observers - Question the idea of Analysis Pipeline that use the whole run as input. If yes, move visualization classes as Analyses operator of this pipeline. Choose a default functionning for raw_pp and XXX_pp dedicated to data transformation analysis

> [Dummy_Controller] remove totally and manage exceptions

> [Transfer] Automate model transfer across machines

> [Clustering Controllers]


## CI/CD

> [Complete_Review] Review modules one by one: pipeline, dataset, controllers, core,

> [CLI] update / review - (CLI_EXTENSION_PROPOSAL.md)

> [Tests] Review and cleaning
> [Tests] clean workspace and run folder creation during tests.

> [Operators] Reintroduce operators tests (cf. pinard for TransformerMixin) _ add data aug operators en masse.

> [GLOBAL REVIEW] v1.0 signatures freeze (private pattern _module), Complete tests > Prod coverage (transformations, controllers, predictions, datasets, runner)

> [Profiling] Code Optimization, Improve performances


## Deploy

> [Docker] provide a docker, add build and actions

> [Conda] provide a conda, add build and actions

> [HuggingFace] Deploy

> [SERV/CLIENT] CLUSTERED COMPUTATION + see productization service doc


## Refactoring

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
> [DAG] Pipeline as DAG


## Post V1 features

> [onnx] onnx export1

> [WanDB] wandb compatibility

> [Customizable_Feature_Source] Refactor dataset to allow customizable feature source and customizable layouts to allow datasets with more dimensions (images, lidar)

## Potential future feature

### Model families that add diversity

* **Statsmodels**
  * For classical stats, GLM, mixed models, time-series (ARIMA, etc.).
  * Useful when reviewers/colleagues want “statistical” baselines or interpretable models.

* **GLM / GAM frameworks**
  * For example: **pyGAM**.
  * Good middle ground between linear and black-box models.

* **Probabilistic programming**
  * **PyMC** or **NumPyro** (since you already have JAX).
  * For Bayesian regression / uncertainty calibration of NIRS models.


### Time series & sequence-specific

* **tsfresh** or **Kats** (or at least some time series feature extraction lib).

* Optional: a thin integration with **pytorch-forecasting** or **neuralforecast** if you want deep TS models out-of-the-box.


### Deep learning tooling around the cores

* **Hugging Face ecosystem**:
  * `transformers` for generic sequence models (even for 1D spectra, time series, or text annotations).
  * `datasets` for standardized dataset handling and splits.

* **Lightning / Keras Tuner / Ignite** (optional)
  * One structured training loop framework can help standardize training, logging, callbacks.


### Explainability & diagnostics beyond SHAP

* **Captum** (for PyTorch)
  * For gradient-based attributions, integrated gradients, etc.

* **Alibi / Alibi-Detect**
  * For drift detection, outlier detection, and some local explanations.

* **Fairlearn** (optional)
  * If you ever need fairness metrics / constraints (maybe less central for NIRS but good for “coverage”).


### Optimization, search, and experiment tracking

* **Ray Tune** or **skopt** (optional)
  * Only if you want multi-backend HPO or alternative search strategies; Optuna alone is usually enough.

* **Experiment tracking**
  * Integrate with something: **MLflow**, **Weights & Biases**, or a simple internal tracker.
  * Even a minimal MLflow integration (params, metrics, artifacts) would greatly help adoption.


### Dimensionality reduction & manifold learning

* **hdbscan**
  * For density-based clustering in the embedded spaces.

### Specialized “tabular” / “auto-ML” stacks (optional but nice)

* **TabNet / FT-Transformer** implementations
  * Either via PyTorch / TF implementations or 3rd-party libs.

* Optional: light integration with **FLAML** if you want a “fast baseline” AutoML passthrough.


### Infra/compute helpers around ML

* **Dask / Ray**
  * For scaling up preprocessing and model training to multi-core / multi-node.

* **numba**
  * For fast custom transforms (spectral transforms, etc.) without full C++.