## ROADMAP ##

> [Workspace] structure management

> [WEBAPP] miniature version > keep print and matplotlib, pilot with react.

**RELEASE** 0.3

> [METADATA] Reup load / stratificiation

> [Augmentation + Balanced Augmentation]

> **Bugs**:
>   - Unify task_type usage and detection
>   - Fix train_params in finetuning for Tensorflow
>   - Review R2 computation / Q2 value
> [SEED] review

> **Enhancement**: Options normalisation in charts (0-1 ou 1-0)

**RELEASE** 0.4

> [CLI]  Reup - run / predict / explain - directly on paths (dataset, pipeline config), json and yaml

> [ReadsTheDoc] minimal subset of signatures + export MD

**RELEASE** 0.5

> [Chart controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators.

> [Predictions] refactoring and as a pipeline context and change storage mode (index (Polars/Parquet) + blobs (Zarr/HDF5))

> [Metrics] uniformize Model_Utils / Evaluator / Predictions

**RELEASE** 0.6

> [Pipeline as single transformer]: pre-instanciate binaries, contruct pipeline, fit(), transform(), predict(), fit_transform(). pour SHAP NN.

**RELEASE** 0.7

> [CLI] nirs4all renaming: nirs4all.train(), .predict(), .explain(), .transfer(), .resume(), .stack(), .analyze()

> [Y_pipeline in models]

**RELEASE** 0.8

> [Logs]

> [GLOBAL REVIEW] > signatures freeze

> [TEST] Prod coverage (transformations, controllers, predictions, datasets, runner)

**RELEASE**  0.9

> [Stacking]

> [Workflows: branch, merge, split_src, scope]

> [Transformations] Asymetric processings (PCA in pipelines) > padding or cropping

**RELEASE** 0.10

> [SERVICE FUNCTIONS] provides easy services functions. > cf. Service.md

> [WEBAPP] full react version - hidden fastapi / nirs4all

> [DEPLOY] standalone installer, web installer

**RELEASE** 0.11 rc

> [REVIEW++]

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


## MINORS ##

> [feature_augmentation] with first replacement

#### EXCITERS ####
- Clean user interface on datasetConfig/dataset, pipelineConfig/pipeline, predictions
- Enhanced file savings: better logic, options, enhanced data_types, dynamic loading, caching
- Charts in 'raw' y for categorical
- More tunable and explicit generation
- Visualize effect of a preprocessing in the ui

## REVIEW / POLISH ##
> [cli]
> - run, predict, viz commands
> - packaging exe

> [controller]
> - base_model_controller

> [dataset]
> - predictions > clean redondancy, metrics usage, optimize search and metadata building
> - targets > clean redondancy (task_type), export 'raw' instead of 'numeric', explicit context

## DOCS ##
> Splitter illustration and tutorial
> Transformer mixin illustration and tutorial