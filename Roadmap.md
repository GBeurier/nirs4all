## ROADMAP ##
> [CLI]  ## Reup - run / predict / explain - directly on paths (dataset, pipeline config), json and yaml

**RELEASE**
> **Enhancement**: Options normalisation in charts (0-1 ou 1-0)

> **Bugs**:
>   - Unify task_type usage and detection
>   - Fix train_params in finetuning for Tensorflow
> - [SEED] review

**RELEASE**
> - [Augmentation + Balanced Augmentation]

**RELEASE**
> [Chart controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators.

**RELEASE**
> [Pipeline as single transformer]: pre-instanciate binaries, contruct pipeline, fit(), transform(), predict(), fit_transform(). pour SHAP NN.

**RELEASE**
> [ReadsTheDoc] update
> [METADATA] Reup load / stratificiation

**RELEASE**
> [Transfer] Automate best transfer model

**RELEASE**
> [Stacking]

**RELEASE**
> [Logs]

**RELEASE**
> GLOBAL REVIEW - refactoring: predictions / predictions analyze / runner (clean) /

**RELEASE**
> [Workflows: branch, merge, split_src, scope]
> [Multi source models] - early/mid fusion
> [Late Fusion - avg / w_avg]

**RELEASE**
> [Y_pipeline in models]
> [Clustering Controllers]

**RELEASE**
> [Stacking]

**RELEASE**
> [Predictions] as a pipeline context
> [Mid Fusion]

**RELEASE**
> [HugginFace deploy]
> [WEBAPP]

**RELEASE**
> [CLUSTERED COMPUTATION + SERV/CLIENT]

**RELEASE**

## REVIEW / POLISH ##
> cli
> - run, predict, viz commands
> - packaging exe
> controller
> - base_model_controller
> dataset
> - predictions
> - targets

## BUGS ##
- link mean in candlesticks

## DOCS ##
> Splitter illustration and tutorial
> Transformer mixin illustration and tutorial
> Update ReadsTheDocs


## FEATURES ##
> optional processings tags on models
> [Generator] add in-place/internal generation
> [Metrics] uniformize Model_Utils / Evaluator / Predictions
> [CLI] nirs4all renaming: nirs4all.train(), .predict(), .explain(), .transfer(), .resume(), .stack(), .analyze()
> [Classification] averaging

## MINORS ##
> feature_augmentation with first replacement

## Review ##
> Predictions > clean redondancy, optimize search and metadata building
> Base_model_controller > disintricate

#### EXCITERS ####
- Clean user interface on datasetConfig/dataset, pipelineConfig/pipeline, predictions
- Enhanced file savings: better logic, options, enhanced data_types, dynamic loading, caching
- Charts in 'raw' y for categorical
- More tunable and explicit generation






---------------------
