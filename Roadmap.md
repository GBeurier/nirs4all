## ROADMAP ##

> [CLI]  ## Reup - run / predict / explain - directly on paths (dataset, pipeline config), json and yaml
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

> [ReadsTheDoc] minimal subset of signatures + export MD

**RELEASE** 0.5

> [Chart controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators.

> [Predictions] refactoring and as a pipeline context

> [Metrics] uniformize Model_Utils / Evaluator / Predictions

**RELEASE** 0.6

> [Pipeline as single transformer]: pre-instanciate binaries, contruct pipeline, fit(), transform(), predict(), fit_transform(). pour SHAP NN.

> [WEBAPP] miniature version

**RELEASE** 0.7

> [Workspace] structure management

> [CLI] nirs4all renaming: nirs4all.train(), .predict(), .explain(), .transfer(), .resume(), .stack(), .analyze()

> [Y_pipeline in models]

**RELEASE** 0.8

> [Logs]

> [GLOBAL REVIEW] > signatures freeze

**RELEASE**  0.9

> [Stacking]

> [Workflows: branch, merge, split_src, scope]

**RELEASE** 0.10

> [WEBAPP] full version

> [DEPLOY] standalone installer, web installer

**RELEASE** 0.11 rc

> [REVIEW++]

**RELEASE** 1.0

**RELEASE** 1.x.x

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