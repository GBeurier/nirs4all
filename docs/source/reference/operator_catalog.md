# Operator Catalog

This catalog enumerates every pipeline operator exposed across the `nirs4all` backend and the `nirs4all_ui` component library after the October 2025 restructuring. The component manifest is generated from `scripts/generate_component_library.py`, which populates `public/component-library.json` by introspecting the codebase and the installed scikit-learn distribution.

## Category Overview

| Category | Feather icon | Subcategories | Components | Description |
| --- | --- | --- | --- | --- |
| Augmentation | `GitBranch` | 1 | 8 | Spectral augmentation operators to improve robustness |
| Spectral Preprocessing | `Sliders` | 7 | 18 | Baseline, scatter, smoothing, derivatives, and spectral transforms |
| Feature Engineering | `Layers` | 7 | 45 | scikit-learn TransformerMixin utilities and feature builders |
| Dimension Reduction | `Minimize2` | 3 | 39 | Operators that change feature dimensionality |
| Classical Models | `BarChart2` | 16 | 99 | scikit-learn estimators (regressors, classifiers, wrappers) |
| Deep Learning | `Cpu` | 1 | 29 | TensorFlow models bundled with nirs4all |
| Validation & Splitting | `DivideSquare` | 1 | 17 | Cross-validation and sampling strategies |
| Target Processing | `Crosshair` | 1 | 2 | Transformations applied to the target variable (`y`) |
| Prediction & Outputs | `Target` | 1 | 3 | Prediction helpers and probability calibration |
| Pipeline Utilities | `Box` | 3 | 11 | Containers, generators, and visualization helpers |

## Augmentation

- **Spectral Augmentation** – `Rotate & Translate`, `Random X Operation`, `Spline Smoothing`, `Spline X Perturbations`, `Spline Y Perturbations`, `Spline X Simplification`, `Spline Curve Simplification`, `Identity Augmenter`

## Spectral Preprocessing

- **Baseline Correction** – `Baseline Removal`, `Detrend`
- **Scatter & Normalization** – `MSC`, `Standard Normal Variate`, `Robust Normal Variate`
- **Smoothing** – `Savitzky-Golay`, `Gaussian Filter`
- **Derivatives** – `First Derivative`, `Second Derivative`, `Sample Derivative`
- **Spectral Transforms** – `Wavelet Transform`, `Haar Wavelet`, `Log Transform`
- **NIRS Scaling** – `Normalize Rows`, `Simple Scale`
- **Resampling & Alignment** – `Adaptive Resampler`, `Crop Transformer`, `Resample Transformer`

## Feature Engineering (scikit-learn TransformerMixin)

- **scikit-learn Scalers** – `Binarizer`, `FunctionTransformer`, `KernelCenterer`, `MaxAbsScaler`, `MinMaxScaler`, `Normalizer`, `PolynomialFeatures`, `PowerTransformer`, `QuantileTransformer`, `RobustScaler`, `SplineTransformer`, `StandardScaler`
- **Encoding & Binning** – `KBinsDiscretizer`, `LabelBinarizer`, `LabelEncoder`, `MultiLabelBinarizer`, `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`
- **Imputation** – `KNNImputer`, `MissingIndicator`, `SimpleImputer`
- **Dimensionality Reduction** – `CCA`, `DictionaryLearning`, `FactorAnalysis`, `FastICA`, `IncrementalPCA`, `Isomap`, `KernelPCA`, `LatentDirichletAllocation`, `LocallyLinearEmbedding`, `MiniBatchDictionaryLearning`, `MiniBatchNMF`, `MiniBatchSparsePCA`, `NMF`, `PCA`, `PLSCanonical`, `PLSRegression`, `PLSSVD`, `SparseCoder`, `SparsePCA`, `TSNE`, `TruncatedSVD`
- **Feature Selection** – `GenericUnivariateSelect`, `RFE`, `RFECV`, `SelectFdr`, `SelectFpr`, `SelectFromModel`, `SelectFwe`, `SelectKBest`, `SelectPercentile`, `SequentialFeatureSelector`, `VarianceThreshold`
- **Kernel & Projection** – `AdditiveChi2Sampler`, `GaussianRandomProjection`, `Nystroem`, `PolynomialCountSketch`, `RBFSampler`, `SkewedChi2Sampler`, `SparseRandomProjection`
- **Feature Extraction** – `DictVectorizer`, `FeatureHasher`, `HashingVectorizer`, `PatchExtractor`, `TfidfTransformer`
- **Cluster & Neighbors** – `Birch`, `BisectingKMeans`, `FeatureAgglomeration`, `KMeans`, `KNeighborsTransformer`, `MiniBatchKMeans`, `NeighborhoodComponentsAnalysis`, `RadiusNeighborsTransformer`
- **Meta Transformers** – `ColumnTransformer`, `FeatureUnion`, `RandomTreesEmbedding`, `StackingClassifier`, `StackingRegressor`, `VotingClassifier`, `VotingRegressor`
- **Miscellaneous Transformers** – `BernoulliRBM`, `IsotonicRegression`, `LinearDiscriminantAnalysis`


## Dimension Reduction

- **Dimensionality Reduction** - `CCA`, `DictionaryLearning`, `FactorAnalysis`, `FastICA`, `IncrementalPCA`, `Isomap`, `KernelPCA`, `LatentDirichletAllocation`, `LocallyLinearEmbedding`, `MiniBatchDictionaryLearning`, `MiniBatchNMF`, `MiniBatchSparsePCA`, `NMF`, `PCA`, `PLSCanonical`, `PLSRegression`, `PLSSVD`, `SparseCoder`, `SparsePCA`, `TSNE`, `TruncatedSVD`
- **Feature Selection** - `GenericUnivariateSelect`, `RFE`, `RFECV`, `SelectFdr`, `SelectFpr`, `SelectFromModel`, `SelectFwe`, `SelectKBest`, `SelectPercentile`, `SequentialFeatureSelector`, `VarianceThreshold`
- **Kernel & Projection** - `AdditiveChi2Sampler`, `GaussianRandomProjection`, `Nystroem`, `PolynomialCountSketch`, `RBFSampler`, `SkewedChi2Sampler`, `SparseRandomProjection`

> _Note_: Transformer lists are derived automatically via `sklearn.utils.all_estimators(type_filter="transformer")`, ensuring parity with the installed scikit-learn version.

## Classical Models (scikit-learn estimators)

- **Baseline Models** – `DummyClassifier`, `DummyRegressor`
- **Cross Decomposition** – `CCA`, `PLSCanonical`, `PLSRegression`
- **Decision Trees** – `DecisionTreeClassifier`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `ExtraTreeRegressor`
- **Discriminant Analysis** – `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`
- **Ensemble Methods** – `AdaBoostClassifier`, `AdaBoostRegressor`, `BaggingClassifier`, `BaggingRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `StackingClassifier`, `StackingRegressor`, `VotingClassifier`, `VotingRegressor`
- **Gaussian Process** – `GaussianProcessClassifier`, `GaussianProcessRegressor`
- **Kernel Ridge & Friends** – `KernelRidge`
- **Linear Models** – `ARDRegression`, `BayesianRidge`, `ElasticNet`, `ElasticNetCV`, `GammaRegressor`, `HuberRegressor`, `Lars`, `LarsCV`, `Lasso`, `LassoCV`, `LassoLars`, `LassoLarsCV`, `LassoLarsIC`, `LinearRegression`, `LogisticRegression`, `LogisticRegressionCV`, `MultiTaskElasticNet`, `MultiTaskElasticNetCV`, `MultiTaskLasso`, `MultiTaskLassoCV`, `OrthogonalMatchingPursuit`, `OrthogonalMatchingPursuitCV`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`, `Perceptron`, `PoissonRegressor`, `QuantileRegressor`, `RANSACRegressor`, `Ridge`, `RidgeCV`, `RidgeClassifier`, `RidgeClassifierCV`, `SGDClassifier`, `SGDRegressor`, `TheilSenRegressor`, `TweedieRegressor`
- **Meta Estimators** – `ClassifierChain`, `MultiOutputClassifier`, `MultiOutputRegressor`, `OneVsOneClassifier`, `OneVsRestClassifier`, `OutputCodeClassifier`, `RegressorChain`, `TransformedTargetRegressor`
- **Naive Bayes** – `BernoulliNB`, `CategoricalNB`, `ComplementNB`, `GaussianNB`, `MultinomialNB`
- **Nearest Neighbors** – `KNeighborsClassifier`, `KNeighborsRegressor`, `NearestCentroid`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`
- **Neural Networks (sklearn)** – `MLPClassifier`, `MLPRegressor`
- **Probabilistic & Calibration** – `CalibratedClassifierCV`, `FixedThresholdClassifier`, `IsotonicRegression`, `TunedThresholdClassifierCV`
- **Semi-supervised** – `LabelPropagation`, `LabelSpreading`, `SelfTrainingClassifier`
- **Support Vector Machines** – `LinearSVC`, `LinearSVR`, `NuSVC`, `NuSVR`, `SVC`, `SVR`
- **Miscellaneous Models** – *(currently empty; all estimators are classified above)*

> _Note_: Model listings are produced via `sklearn.utils.all_estimators` for both classifiers and regressors, preserving compatibility with the runtime environment.

## Deep Learning (TensorFlow)

`CONV_LSTM`, `Custom_Residuals`, `Custom_VG_Residuals`, `Custom_VG_Residuals2`, `FFT_Conv`, `MLP`, `ResNetV2_model`, `SEResNet_model`, `UNET`, `UNet_NIRS`, `VGG_1D`, `XCeption1D`, `bard`, `customizable_decon`, `customizable_nicon`, `customizable_nicon_classification`, `decon`, `decon_classification`, `decon_layer_classification`, `inception1D`, `nicon`, `nicon_VG`, `nicon_VG_classification`, `nicon_classification`, `senseen_origin`, `transformer`, `transformer_VG`, `transformer_VG_classification`, `transformer_classification`

These factories are tagged via the `framework("tensorflow")` decorator and are surfaced under the `TensorFlowModelController`.

## Validation & Splitting

- **Splitting Strategies** - `Shuffle Split`, `K-Fold`, `Stratified K-Fold`, `Repeated K-Fold`, `Repeated Stratified K-Fold`, `Group K-Fold`, `Group Shuffle Split`, `Stratified Shuffle Split`, `Time Series Split`, `Leave-One-Out`, `Leave-P-Out`, `Kennard-Stone Splitter`, `SPXY Splitter`, `KMeans Splitter`, `SPlit Splitter`, `Systematic Circular`, `KBins Stratified`

## Target Processing

- **Target Transforms** – `Integer KBins Discretizer`, `Range Discretizer`

## Prediction & Outputs

- **Prediction Utilities** – `Batch Prediction`, `Real-time Prediction`, `Probability Calibration`

## Pipeline Utilities

- **Containers** - `Feature Augmentation`, `Sample Augmentation`, `Sequential`, `Pipeline`, `Y Processing` (augmentation/processing containers accept only preprocessing, augmentation, or feature-engineering nodes; `Y Processing` is limited to target transforms; `Pipeline` remains unrestricted)
- **Generators** – `_OR_`, `_RANGE_` (parameter sweep and branching)
- **Visualization** – `2D Chart`, `Y Distribution Chart`, `Fold Chart`

## Maintenance Notes

- Run `python scripts/generate_component_library.py` from `nirs4all_ui` to regenerate the library after adding new operators. The script introspects nirs4all modules and calls `sklearn.utils.all_estimators` to keep the catalog aligned with the installed version.
- UI components consume `public/component-library.json`; the pipeline editor reads this file via `libraryDataLoader`.
- Container rules use category tokens (e.g. `category:preprocessing`) so augmentation and preprocessing containers reject models or incompatible nodes while `Pipeline` stays unrestricted.
- If new TensorFlow models are added under `nirs4all.operators.models`, ensure they carry the `framework("tensorflow")` decorator so they are picked up automatically.
