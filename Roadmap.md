## ROADMAP ##

> [Cleaning] Modularize/Clean/Refactor: Indexer, Predictions, PredictionAnalyzer, Model_builder, PipelineRunner, BaseModelController

> [Pipeline] authorize list of pipelineconfigs instead of pipelines....

**RELEASE** 0.4.1: final structure

**Bugs**:
> [File saving] Fix bad usage of image saving in op_split and op_fold_charts (currently it use directly the saver in runner instead of returning tuple - bad design for custom controllers/operators)
- op_fold_charts:
  lines 140-157 > The op_fold_charts save directly the image with runner saver instead of returning the image to save.
- op_split:
  line 250-260 > same problems as op_fold_charts. Save directly instead of returning tuple

> [Pytorch] controller

> [Imports] import tf and pytorch only when needed, reup backend_utils.

> [SEED] review and fix definitive logic


**RELEASE** 0.4.2: torch up

> [Errors] Uniformize exception errors (cf RECOMMANDATIONS DATASET.md)

> [ReadsTheDoc] minimal subset of signatures + update and export MD

**RELEASE** 0.5: documentation

> [Chart_Controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators.

> **Enhancement**:
> - Options normalisation in charts (0-1 ou 1-0)
> - Chart normalisation per dataset.

**Bugs**:
>   - [_or_] with one element fallback on dummy controller

**RELEASE** 0.5.1: charts operators

> [Predictions] as a pipeline context.

> [Metrics] uniformize Model_Utils / Evaluator / Predictions and add custom losses - lambda / functions / classes

> [Layout] review layouts (tests) and add operators params

> [Deprec: Model selection] Tools to select "best" predictions | verify if already exists

**Bugs**:
>   - Review R2 computation / Q2 value - GOF (goodness of fit)

**RELEASE** 0.5.2: predictions/metrics final

> [Folds] Operator load fold (csv)

> [PLS] implements all PLS (cf. doc.md)

> [Operators] Reintroduce operators tests (cf. pinard for TransformerMixin)

**RELEASE** 0.6: Minimal controller set

> [Pipeline] as single transformer: pre-instanciate binaries, contruct pipeline, fit(), transform(), predict(), fit_transform(). pour SHAP NN. Decompose run and pipeline (1 pipeline per config tuple)

> [Runner] Design logic of 'execution sequence' and 'history' > pp and raw data, use cache by defaut, generalize default inputType (np.array, SpectroDataset, DatasetConfig, ...)

> [Dummy_Controller] remove totally and manage exceptions

**RELEASE** 0.6.1: pipeline logic complete

> [Logs] implement feature and update print/log strategy

> [Examples] update, clean and document examples and tutorial notebooks, Add examples with custom classes

**RELEASE** 0.7: User friendly (log + examples)

> [CLI] nirs4all renaming: nirs4all.train(), .predict(), .explain(), .transfer(), .resume(), .stack(), .analyze()

> [CLI]  Reup - run / predict / explain - directly on paths (dataset, pipeline config), json and yaml

> [Pipeline] verify and clean type for input in pipeline

**RELEASE** 0.8: CLI

> [GLOBAL REVIEW] v1.0 signatures freeze (private pattern _module), Complete tests > Prod coverage (transformations, controllers, predictions, datasets, runner)

> [SERVICE FUNCTIONS] provides easy services functions. > cf. Service.md

**RELEASE**  0.9 alpha: SIGNATURE FREEZE - MVP (feature complete)

> [Y_pipeline in models]

> [Stacking]

> [Workflow Operators] branch, merge, split_src, scope

> [Transformations] Asymetric processings (PCA in pipelines) > auto/optional padding and cropping

**RELEASE** 0.10 beta: operators/controllers complete

> [WEBAPP] full react version - hidden fastapi / nirs4all

> [DEPLOY] standalone installer, web installer

**RELEASE** 0.11 rc - GUI version (sync with nirs4all_ui project)

> [REVIEW] Documentation complete (RTD, Tutorial, Examples), Dead code removed, Tests coverage

**RELEASE** 1.0: Release

**RELEASE** 1.x.x

> [Classification] averaging

> [Pipeline + Optuna] Pipeline as optuna trial. The pp become a choice param. Goal is to stack pp each time score stop progress, select the good ones by feats augmentation and by pp order (1st, 2nd, etc.) and stop once it drops.

> [Transfer] Automate best transfer model

> [Generator] add in-place/internal generation

> [Mid Fusion] Multi head models

> [Late Fusion] avg / w_avg / asymetric ensembling

> [Clustering Controllers]

> [Analysis] t-sne, umap

> [HugginFace deploy]

> [CLUSTERED COMPUTATION + SERV/CLIENT]

#### EXCITERS ####
- better model naming (with optional pp included) for UX
- feature_augmentation with first item replacement
- add NorrisWilliams, Whittaker, BandEnergies, FiniteDiffCentral transformermixin
- (ui) Clean user interface on datasetConfig/dataset, pipelineConfig/pipeline, predictions
- Charts in 'raw' y for categorical
- More tunable and explicit generation > inner generation, constraints, etc.
- Authorize vertical index (col 1 header - vertical header) in csv
