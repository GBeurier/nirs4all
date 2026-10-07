# Operator catalog: choose a job, then inspect the method

An **operator** is a numerical operation such as SNV, a learner such as PLS, or a sampling rule. A **node** tells the pipeline how to use it: transform features, transform targets, create training copies, tag observations or fit a model. The same transformer under sample augmentation and ordinary preprocessing has different effects on the dataset.

Start from the job you need. Each link below leads to the parameter table and explanation; the corresponding node links lead to a worked recipe, expected result figure, exercise and common mistake.

| Job | Detailed method list | Workflow example |
|---|---|---|
| Correct spectra or change features | {doc}`transforms` | {doc}`nodes/preprocessing` |
| Scale or discretize the target | {doc}`transforms` | {doc}`nodes/y_processing` |
| Predict a number or category | {doc}`models` | {doc}`nodes/model` |
| Assign independent validation units | {doc}`splitters` | {doc}`nodes/split` |
| Diagnose unusual observations | {doc}`filters` | {doc}`nodes/tag` |
| Exclude eligible training observations | {doc}`filters` | {doc}`nodes/exclude` |
| Generate plausible training variants | {doc}`augmentations` | {doc}`nodes/sample_augmentation` |
| Build multiple representations | {doc}`transforms` | {doc}`nodes/feature_augmentation` and {doc}`nodes/concat_transform` |
| Combine sources or fitted predictions | {doc}`models` | {doc}`nodes/branch` and {doc}`nodes/merge` |
| Correct a base prediction | {doc}`models` | {doc}`nodes/residual` |
| Generate alternative recipes | {doc}`generator_keywords` | {doc}`nodes/generators` |
| Inspect spectra, folds, copies or exclusions | {doc}`nodes/charts` | {doc}`/user_guide/visualization/index` |

## Enumerated spectroscopy methods

These are the Python SDK's maintained spectroscopy families, not UI component counts. A class listed here does not imply native/WASM execution support. For native IDs, supported host/runtime combinations and portable recipes, consult {doc}`/guide/catalog` and {doc}`/guide/languages`.

### Transforms

[StandardNormalVariate](transforms.md#scatter-correction-and-normalization) · [LocalStandardNormalVariate](transforms.md#scatter-correction-and-normalization) · [RobustStandardNormalVariate](transforms.md#scatter-correction-and-normalization) · [MultiplicativeScatterCorrection](transforms.md#scatter-correction-and-normalization) · [ExtendedMultiplicativeScatterCorrection](transforms.md#scatter-correction-and-normalization) · [AreaNormalization](transforms.md#scatter-correction-and-normalization) · [Normalize](transforms.md#scatter-correction-and-normalization) · [SimpleScale](transforms.md#scatter-correction-and-normalization) · [SavitzkyGolay](transforms.md#smoothing) · [Gaussian](transforms.md#smoothing) · [WaveletDenoise](transforms.md#smoothing) · [FirstDerivative](transforms.md#derivatives) · [SecondDerivative](transforms.md#derivatives) · [NorrisWilliams](transforms.md#derivatives) · [Derivate](transforms.md#derivatives) · [Baseline](transforms.md#baseline-correction) · [Detrend](transforms.md#baseline-correction) · [PyBaselineCorrection](transforms.md#baseline-correction) · [ASLSBaseline](transforms.md#baseline-correction) · [AirPLS](transforms.md#baseline-correction) · [ArPLS](transforms.md#baseline-correction) · [IModPoly](transforms.md#baseline-correction) · [ModPoly](transforms.md#baseline-correction) · [SNIP](transforms.md#baseline-correction) · [RollingBall](transforms.md#baseline-correction) · [IASLS](transforms.md#baseline-correction) · [BEADS](transforms.md#baseline-correction) · [OSC](transforms.md#orthogonalization) · [EPO](transforms.md#orthogonalization) · [ReflectanceToAbsorbance](transforms.md#signal-conversion) · [ToAbsorbance](transforms.md#signal-conversion) · [FromAbsorbance](transforms.md#signal-conversion) · [SignalTypeConverter](transforms.md#signal-conversion) · [KubelkaMunk](transforms.md#signal-conversion) · [LogTransform](transforms.md#signal-conversion) · [PercentToFraction](transforms.md#signal-conversion) · [FractionToPercent](transforms.md#signal-conversion) · [Wavelet](transforms.md#wavelet-transforms-and-feature-extraction) · [Haar](transforms.md#wavelet-transforms-and-feature-extraction) · [WaveletFeatures](transforms.md#wavelet-transforms-and-feature-extraction) · [WaveletPCA](transforms.md#wavelet-transforms-and-feature-extraction) · [WaveletSVD](transforms.md#wavelet-transforms-and-feature-extraction) · [CARS](transforms.md#feature-selection) · [MCUVE](transforms.md#feature-selection) · [FlexiblePCA](transforms.md#feature-selection) · [FlexibleSVD](transforms.md#feature-selection) · [Resampler](transforms.md#resampling-and-cropping) · [CropTransformer](transforms.md#resampling-and-cropping) · [ResampleTransformer](transforms.md#resampling-and-cropping) · [FlattenPreprocessing](transforms.md#resampling-and-cropping) · [IntegerKBinsDiscretizer](transforms.md#target-transforms) · [RangeDiscretizer](transforms.md#target-transforms)

### Models

[AOMPLSRegressor](models.md#adaptive-pls-auto-preprocessing) · [AOMPLSClassifier](models.md#adaptive-pls-auto-preprocessing) · [POPPLSRegressor](models.md#adaptive-pls-auto-preprocessing) · [POPPLSClassifier](models.md#adaptive-pls-auto-preprocessing) · [AOMRidgeRegressor](models.md#aom-ridge-and-fastaom) · [AOMRidgeAutoSelector](models.md#aom-ridge-and-fastaom) · [AOMRidgeBlender](models.md#aom-ridge-and-fastaom) · [FastAOMPLSRidge](models.md#aom-ridge-and-fastaom) · [PLSDA](models.md#standard-pls) · [IKPLS](models.md#standard-pls) · [SIMPLS](models.md#standard-pls) · [RobustPLS](models.md#standard-pls) · [RecursivePLS](models.md#standard-pls) · [OPLS](models.md#orthogonal-pls) · [OPLSDA](models.md#orthogonal-pls) · [KOPLS](models.md#orthogonal-pls) · [MBPLS](models.md#multi-block-and-domain-invariant) · [DiPLS](models.md#multi-block-and-domain-invariant) · [SparsePLS](models.md#sparse-and-interval) · [IntervalPLS](models.md#sparse-and-interval) · [KernelPLS](models.md#kernel-pls) · [OKLMPLS](models.md#kernel-pls) · [FCKPLS](models.md#kernel-pls) · [LWPLS](models.md#locally-weighted) · [NLPLS](models.md#nonlinear-pls) · [MetaModel](models.md#meta-model-stacking) · [IdentityOperator](models.md#aom-pls-operator-bank) · [SavitzkyGolayOperator](models.md#aom-pls-operator-bank) · [DetrendProjectionOperator](models.md#aom-pls-operator-bank) · [NorrisWilliamsOperator](models.md#aom-pls-operator-bank) · [FiniteDifferenceOperator](models.md#aom-pls-operator-bank) · [WaveletProjectionOperator](models.md#aom-pls-operator-bank) · [FFTBandpassOperator](models.md#aom-pls-operator-bank) · [LinearOperator](models.md#aom-pls-operator-bank) · [ComposedOperator](models.md#aom-pls-operator-bank)

### Splitters

[KennardStoneSplitter](splitters.md#nirs-specific-splitters) · [SPXYSplitter](splitters.md#nirs-specific-splitters) · [KMeansSplitter](splitters.md#nirs-specific-splitters) · [KBinsStratifiedSplitter](splitters.md#nirs-specific-splitters) · [SystematicCircularSplitter](splitters.md#nirs-specific-splitters) · [SPlitSplitter](splitters.md#nirs-specific-splitters) · [SPXYFold](splitters.md#nirs-specific-splitters) · [SPXYGFold](splitters.md#nirs-specific-splitters) · [BinnedStratifiedGroupKFold](splitters.md#nirs-specific-splitters) · [GroupedSplitterWrapper](splitters.md#nirs-specific-splitters) · [KFold](splitters.md#commonly-used-sklearn-splitters) · [StratifiedKFold](splitters.md#commonly-used-sklearn-splitters) · [ShuffleSplit](splitters.md#commonly-used-sklearn-splitters) · [RepeatedKFold](splitters.md#commonly-used-sklearn-splitters) · [LeaveOneOut](splitters.md#commonly-used-sklearn-splitters) · [GroupKFold](splitters.md#commonly-used-sklearn-splitters) · [StratifiedGroupKFold](splitters.md#commonly-used-sklearn-splitters)

### Filters



### Augmentations

[GaussianAdditiveNoise](augmentations.md#noise) · [MultiplicativeNoise](augmentations.md#noise) · [SpikeNoise](augmentations.md#noise) · [HeteroscedasticNoiseAugmenter](augmentations.md#noise) · [LinearBaselineDrift](augmentations.md#baseline-drift) · [PolynomialBaselineDrift](augmentations.md#baseline-drift) · [WavelengthShift](augmentations.md#wavelength-distortion) · [WavelengthStretch](augmentations.md#wavelength-distortion) · [LocalWavelengthWarp](augmentations.md#wavelength-distortion) · [SmoothMagnitudeWarp](augmentations.md#spectral-distortion) · [BandPerturbation](augmentations.md#spectral-distortion) · [GaussianSmoothingJitter](augmentations.md#spectral-distortion) · [UnsharpSpectralMask](augmentations.md#spectral-distortion) · [BandMasking](augmentations.md#spectral-distortion) · [ChannelDropout](augmentations.md#spectral-distortion) · [LocalClipping](augmentations.md#spectral-distortion) · [MixupAugmenter](augmentations.md#mixup) · [LocalMixupAugmenter](augmentations.md#mixup) · [ScatterSimulationMSC](augmentations.md#mixup) · [PathLengthAugmenter](augmentations.md#physical--instrumental) · [BatchEffectAugmenter](augmentations.md#physical--instrumental) · [InstrumentalBroadeningAugmenter](augmentations.md#physical--instrumental) · [DeadBandAugmenter](augmentations.md#physical--instrumental) · [TemperatureAugmenter](augmentations.md#environmental) · [MoistureAugmenter](augmentations.md#environmental) · [ParticleSizeAugmenter](augmentations.md#scattering) · [EMSCDistortionAugmenter](augmentations.md#scattering) · [DetectorRollOffAugmenter](augmentations.md#edge-artifacts) · [StrayLightAugmenter](augmentations.md#edge-artifacts) · [EdgeCurvatureAugmenter](augmentations.md#edge-artifacts) · [TruncatedPeakAugmenter](augmentations.md#edge-artifacts) · [EdgeArtifactsAugmenter](augmentations.md#edge-artifacts) · [Spline_Smoothing](augmentations.md#spline-based) · [Spline_X_Perturbations](augmentations.md#spline-based) · [Spline_Y_Perturbations](augmentations.md#spline-based) · [Spline_X_Simplification](augmentations.md#spline-based) · [Spline_Curve_Simplification](augmentations.md#spline-based) · [Rotate_Translate](augmentations.md#random-geometric) · [Random_X_Operation](augmentations.md#random-geometric)

## Compatible host estimators beyond this list

The Python SDK also accepts compatible sklearn estimators and installed neural frameworks. For regression, useful comparison families include Ridge/Lasso/ElasticNet, PLS, SVR, random forests, boosting, Gaussian processes and a dummy baseline. Classification counterparts include logistic regression, discriminant analysis, PLSDA, SVC, tree ensembles and calibrated probability models. Their input and target constraints still apply; discovery alone does not establish that an estimator is suitable.

Examples connect these choices to experiments: [multiple models](https://github.com/nirs4all/nirs4all/blob/main/examples/user/04_models/U01_multi_model.py), [PLS variants](https://github.com/nirs4all/nirs4all/blob/main/examples/user/04_models/U04_pls_variants.py), [augmentation](https://github.com/nirs4all/nirs4all/blob/main/examples/user/03_preprocessing/U03_sample_augmentation.py), and [CV strategies](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U01_cv_strategies.py). Exact constructors, fitting methods and learned attributes are under {doc}`/api/modules`.

## How to use a catalog entry

1. Identify the measurement or prediction question.
2. Open the family reference and compare mechanisms and parameters.
3. Open the node page to understand shape, sample count, fitting population and prediction behavior.
4. Choose your language's supported recipe from {doc}`/guide/languages`.
5. Run the matching example and inspect its expected artifacts before composing a larger pipeline.

For typed sources, missing measurements, early/late fusion and target masks, use {doc}`/guide/datasets` and {doc}`multimodal_execution_matrix`. Preserve sample identity and source schema throughout.
