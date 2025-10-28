## ROADMAP ##

**RELEASE** 0.4

> [Cleaning] Modularize/Clean/Refactor: PipelineRunner, Targets, Predictions, BaseModelController, PredictionAnalyzer, Model_builder, Evaluator

> [Errors] Uniformize exception errors (cf RECOMMANDATIONS DATASET.md)

> [File saving] Fix bad usage of image saving in op_split and op_fold_charts (currently it use directly the saver in runner instead of returning tuple - bad design for custom controllers/operators)

> [Pytorch] controller

> [Imports] import tf and pytorch only when needed, reup backend_utils.

**Bugs**:
>   - [heatmap v2] NA in pred charts + Pred charts missing in multi datasets

> [SEED] review and fix definitive logic

> [ReadsTheDoc] minimal subset of signatures + update and export MD


**RELEASE** 0.5

> [Chart controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators.

> **Enhancement**:
> - Options normalisation in charts (0-1 ou 1-0)

> [Predictions] as a pipeline context.

> [Metrics] uniformize Model_Utils / Evaluator / Predictions and add custom losses

**Bugs**:
>   - Review R2 computation / Q2 value - GOF (goodness of fit)

> [Deprec: Model selection] Tools to select "best" predictions

> [Folds] Operator load fold (csv)

> [PLS] implements all PLS (cf. doc.md)

> [Operators] Reintroduce operators tests (cf. pinard for TransformerMixin)

**RELEASE** 0.6

> [Pipeline] as single transformer: pre-instanciate binaries, contruct pipeline, fit(), transform(), predict(), fit_transform(). pour SHAP NN. Decompose run and pipeline (1 pipeline per config tuple)

> [Runner] Design logic of 'execution sequence' and 'history' > pp and raw data, use cache by defaut, generalize default inputType (np.array, SpectroDataset, DatasetConfig, ...)

> [Logs]

**RELEASE** 0.7

> [CLI] nirs4all renaming: nirs4all.train(), .predict(), .explain(), .transfer(), .resume(), .stack(), .analyze()

> [CLI]  Reup - run / predict / explain - directly on paths (dataset, pipeline config), json and yaml

**RELEASE** 0.8: SIGNATURE FREEZE

> [GLOBAL REVIEW] v1.0 signatures freeze (private pattern _module), Complete tests > Prod coverage (transformations, controllers, predictions, datasets, runner)

> [SERVICE FUNCTIONS] provides easy services functions. > cf. Service.md

**RELEASE**  0.9 apha: MVP (feature complete)

> [Y_pipeline in models]

> [Stacking]

> [Workflow Operators] branch, merge, split_src, scope

> [Transformations] Asymetric processings (PCA in pipelines) > auto/optional padding and cropping

**RELEASE** 0.10 beta: GUI version (wync with nirs4all_ui project)

> [WEBAPP] full react version - hidden fastapi / nirs4all

> [DEPLOY] standalone installer, web installer

**RELEASE** 0.11 rc

> [REVIEW] Documentation complete, Dead code removed,

**RELEASE** 1.0

**RELEASE** 1.x.x

> [Pipeline + Optuna] Pipeline as optuna trial. The pp become a choice param. Goal is to stack pp each time score stop progress, select the good ones by feats augmentation and by pp order (1st, 2nd, etc.) and stop once it drops.

> [Transfer] Automate best transfer model

> [Mid Fusion] Multi head models

> [Late Fusion] avg / w_avg / asymetric ensembling

> [Clustering Controllers]

> [HugginFace deploy]

> [CLUSTERED COMPUTATION + SERV/CLIENT]

> [Classification] averaging

> [PRINT] optional processings tags on models

> [Generator] add in-place/internal generation

> [Analysis] t-sne, umap


#### EXCITERS ####
- feature_augmentation with first item replacement
- add NorrisWilliams, Whittaker, BandEnergies, FiniteDiffCentral transformermixin
- (ui) Clean user interface on datasetConfig/dataset, pipelineConfig/pipeline, predictions
- Charts in 'raw' y for categorical
- More tunable and explicit generation > inner generation, constraints, etc.
- Authorize vertical index (col 1 header - vertical header) in csv
