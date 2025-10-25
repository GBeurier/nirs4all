## ROADMAP ##

> [Git] merge branch

**RELEASE** 0.4

> [Pytorch] controller

> [Imports] import tf and pytorch only when needed, reup backend_utils.

**Bugs**:
>   - [heatmap v2] NA in pred charts + Pred charts missing in multi datasets

> [SEED] review

> [CLI]  Reup - run / predict / explain - directly on paths (dataset, pipeline config), json and yaml

> [ReadsTheDoc] minimal subset of signatures + update and export MD


**RELEASE** 0.5

> [Chart controller] Migrates individual controller in operators: x, y, folds, 3d, 2d operators.

> **Enhancement**:
> - Options normalisation in charts (0-1 ou 1-0)
>   - Unify task_type usage and detection

> [Predictions] refactoring and as a pipeline context

> [Metrics] uniformize Model_Utils / Evaluator / Predictions
**Bugs**:
>   - Review R2 computation / Q2 value - GOF (goodness of fit)

> [Model selection] Tools to select "best" predictions

> [Folds] Operator load fold (csv)

**RELEASE** 0.6

> [TEST] Improve integration tests before pipeline refactoring

> [Pipeline] as single transformer: pre-instanciate binaries, contruct pipeline, fit(), transform(), predict(), fit_transform(). pour SHAP NN. Decompose run and pipeline (1 pipeline per config tuple)

> [Runner] retrieve raw and pp dataset after run

**RELEASE** 0.7

> [CLI] nirs4all renaming: nirs4all.train(), .predict(), .explain(), .transfer(), .resume(), .stack(), .analyze()

> [Y_pipeline in models]

> [PLS] implements all PLS (cf. doc.md)

> [Logs]

**RELEASE** 0.8

> [SERVICE FUNCTIONS] provides easy services functions. > cf. Service.md

> [GLOBAL REVIEW] v1.0 signatures freeze

> [TEST] Complete tests > Prod coverage (transformations, controllers, predictions, datasets, runner)

**RELEASE**  0.9 apha

> [Stacking]

> [Workflow Operators] branch, merge, split_src, scope

> [Transformations] Asymetric processings (PCA in pipelines) > auto/optional padding and cropping

**RELEASE** 0.10 beta

> [WEBAPP] full react version - hidden fastapi / nirs4all

> [DEPLOY] standalone installer, web installer

**RELEASE** 0.11 rc

> [REVIEW++] cf Review section

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


## MINORS ##

> [feature_augmentation] with first item replacement

#### EXCITERS ####
- add NorrisWilliams, Whittaker, BandEnergies, FiniteDiffCentral
- Clean user interface on datasetConfig/dataset, pipelineConfig/pipeline, predictions
- Enhanced file savings: better logic, options, enhanced data_types, dynamic loading, caching
- Charts in 'raw' y for categorical
- More tunable and explicit generation
- Visualize effect of a preprocessing in the ui
- Authorize vertical index (col 1 header - vertical header) in csv

## REVIEW / POLISH ##
> [cli]
> - run, predict, viz commands
> - packaging exe

> [controller]
> - base_model_controller + model_buiders + op_model + optuna. Rewrite and modularize all this stuff.

> [dataset]
> - predictions > clean redondancy, metrics usage, optimize search and metadata building
> - targets > clean redondancy (task_type), export 'raw' instead of 'numeric', explicit context

## DOCS ##
> Splitter illustration and tutorial
> Transformer mixin illustration and tutorial

## TARGETED DIRECTORIES STRUCTURE (serialization_refactoring branch) ##

workspace/
	export/
		predictions_XXX.csv
		report_XXX.csv

	favorites_pipelines/
		my_pipeline_N.zip # with binaries and/or source data
		my_pipeline_3.json # only pipeline definition for retraining

	runs/
		yyyy-mm-dd_run-N-datasetName/ ### OR Custom session _ datasetname
			metadata.json # generation config for this dataset
			report.json # detailed report and executions
			log.txt # log

			binaries/   #indexed cache
				...

			predictions/
				predictions_1_model.csv
				report_1_model.csv
				...
				predictions_N_model.csv
				report_N_model.csv

			outputs/
				XXX_1.png
				YYY_1.csv
				...
				XXX_N.png
				YYY_N.csv

			pipelines/
				manifest_pipeline_1.json
				manifest_pipeline_2.json
				...
				metadata_pipeline_N.json

	predictions/
		dataset1_name.predictions # parquet + json
		dataset2_name.predictions
		...

## TEMPORARY NOTES ON TRANSFORMATIONS ##

Technique                          |  Main Usage/Effect
-----------------------------------+----------------------------------------------------------------------------------------------------------------------------------
Savitzky-Golay Smoothing           |  Reduces spectral noise; preserves peaks and features .
Derivative Spectroscopy            |  Removes baseline effects; resolves overlapping bands .
Standard Normal Variate (SNV)      |  Corrects scatter; normalizes each spectrum .
Multiplicative Scatter Correction  |  Removes scatter artifacts from particle size/path length .
Local/Robust SNV (LSNV, RNV)       |  Enhanced scatter correction for difficult outliers .
Detrending                         |  Removes global or polynomial trends in spectra .
Baseline Correction                |  Subtracts or fits baseline drift with polynomial or other methods .
Mean Centering/Autoscaling         |  Adjusts spectral features to centered/scaled form .
Normalization (e.g., area)         |  Adjusts all spectra to same overall intensity .
Wavelength Selection               |  Focuses analysis on most relevant regions .
Haar Wavelet Transform             |  Sometimes usedfor noise reduction and feature extraction; less common than above methods but useful in some advanced pipelines .



