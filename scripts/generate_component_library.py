import json
import re
import inspect
import importlib
import math
from pathlib import Path

from sklearn.utils import all_estimators
from nirs4all.operators.models import generic_tf, cirad_tf

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - numpy is optional for metadata extraction
    np = None


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


CATEGORIES = [
    {"id":"augmentation","label":"Augmentation","featherIcon":"GitBranch","description":"Spectral data augmentation operators for robustness","className":"bg-orange-50 border-orange-200","color":"#fb923c","bgColor":"#fff7ed"},
    {"id":"preprocessing","label":"Spectral Preprocessing","featherIcon":"Sliders","description":"Baseline, scatter, smoothing, and spectral transforms","className":"bg-blue-50 border-blue-200","color":"#60a5fa","bgColor":"#eff6ff"},
    {"id":"feature_engineering","label":"Feature Engineering","featherIcon":"Layers","description":"scikit-learn TransformerMixin utilities and feature extractors","className":"bg-teal-50 border-teal-200","color":"#2dd4bf","bgColor":"#f0fdfa"},
    {"id":"dimension_reduction","label":"Dimensionality Reduction","featherIcon":"Minimize2","description":"Operators that change feature dimensionality","className":"bg-fuchsia-50 border-fuchsia-200","color":"#c084fc","bgColor":"#fdf4ff"},
    {"id":"models_sklearn","label":"Classical Models","featherIcon":"BarChart2","description":"scikit-learn estimators available in the pipeline","className":"bg-purple-50 border-purple-200","color":"#a78bfa","bgColor":"#f5f3ff"},
    {"id":"models_deep","label":"Deep Learning","featherIcon":"Cpu","description":"TensorFlow models bundled with nirs4all","className":"bg-indigo-50 border-indigo-200","color":"#6366f1","bgColor":"#eef2ff"},
    {"id":"validation","label":"Validation & Splitting","featherIcon":"DivideSquare","description":"Data splitting strategies and cross-validation helpers","className":"bg-amber-50 border-amber-200","color":"#facc15","bgColor":"#fefce8"},
    {"id":"targets","label":"Target Processing","featherIcon":"Crosshair","description":"Transformations applied to target variables (y)","className":"bg-rose-50 border-rose-200","color":"#fb7185","bgColor":"#fff1f2"},
    {"id":"prediction","label":"Prediction & Outputs","featherIcon":"Target","description":"Prediction utilities, calibration, and deployment helpers","className":"bg-red-50 border-red-200","color":"#f87171","bgColor":"#fef2f2"},
    {"id":"utilities","label":"Pipeline Utilities","featherIcon":"Box","description":"Containers, generators, visualization, and helper nodes","className":"bg-slate-50 border-slate-200","color":"#64748b","bgColor":"#f8fafc"},
]


SUBCATEGORIES = [
    {"id":"spectral_augmentation","label":"Spectral Augmentation","categoryId":"augmentation","description":"Augmentations that perturb spectra or wavelength axes"},
    {"id":"baseline_correction","label":"Baseline Correction","categoryId":"preprocessing","description":"Remove drift and baseline trends from spectra"},
    {"id":"scatter_correction","label":"Scatter & Normalization","categoryId":"preprocessing","description":"Correct scatter effects and normalize intensity"},
    {"id":"smoothing","label":"Smoothing","categoryId":"preprocessing","description":"Denoise spectra while preserving features"},
    {"id":"derivatives","label":"Derivatives","categoryId":"preprocessing","description":"Spectral derivatives and gradient-based transforms"},
    {"id":"spectral_transforms","label":"Spectral Transforms","categoryId":"preprocessing","description":"Wavelets, log transforms, and other spectral mappings"},
    {"id":"nirs_scalers","label":"NIRS Scaling","categoryId":"preprocessing","description":"Scaling utilities implemented in nirs4all"},
    {"id":"resampling_alignment","label":"Resampling & Alignment","categoryId":"preprocessing","description":"Spectral resampling, cropping, and alignment tools"},
    {"id":"scikit_scalers","label":"scikit-learn Scalers","categoryId":"feature_engineering","description":"Scaling and normalization transformers from scikit-learn"},
    {"id":"scikit_encoding","label":"Encoding & Binning","categoryId":"feature_engineering","description":"Label encoders, one-hot encoders, and discretizers"},
    {"id":"scikit_imputation","label":"Imputation","categoryId":"feature_engineering","description":"Missing value imputers compatible with TransformerMixin"},
    {"id":"scikit_dimensionality","label":"Dimensionality Reduction","categoryId":"dimension_reduction","description":"PCA, manifold learning, and decomposition transformers"},
    {"id":"scikit_feature_selection","label":"Feature Selection","categoryId":"dimension_reduction","description":"Univariate tests, model-based, and sequential selection"},
    {"id":"scikit_kernel_projection","label":"Kernel & Projection","categoryId":"dimension_reduction","description":"Kernel approximations and random projections"},
    {"id":"scikit_feature_extraction","label":"Feature Extraction","categoryId":"feature_engineering","description":"Text, image, and dictionary-based feature builders"},
    {"id":"scikit_cluster_neighbors","label":"Cluster & Neighbors","categoryId":"feature_engineering","description":"Cluster-based and neighborhood graph transformers"},
    {"id":"scikit_meta_transformers","label":"Meta Transformers","categoryId":"feature_engineering","description":"Pipelines, column transformers, and stacking transformers"},
    {"id":"scikit_misc_transformers","label":"Miscellaneous Transformers","categoryId":"feature_engineering","description":"Other scikit-learn transformers compatible with nirs4all"},
    {"id":"sklearn_linear","label":"Linear Models","categoryId":"models_sklearn","description":"Linear regression and classification estimators"},
    {"id":"sklearn_svm","label":"Support Vector Machines","categoryId":"models_sklearn","description":"Kernel-based SVM models"},
    {"id":"sklearn_tree","label":"Decision Trees","categoryId":"models_sklearn","description":"Decision tree based estimators"},
    {"id":"sklearn_ensemble","label":"Ensemble Methods","categoryId":"models_sklearn","description":"Bagging, boosting, stacking, and voting ensembles"},
    {"id":"sklearn_neighbors","label":"Nearest Neighbors","categoryId":"models_sklearn","description":"KNN and neighborhood-based estimators"},
    {"id":"sklearn_naive_bayes","label":"Naive Bayes","categoryId":"models_sklearn","description":"Probabilistic classifiers with Bayes assumptions"},
    {"id":"sklearn_discriminant","label":"Discriminant Analysis","categoryId":"models_sklearn","description":"Linear and quadratic discriminant models"},
    {"id":"sklearn_gaussian_process","label":"Gaussian Process","categoryId":"models_sklearn","description":"Gaussian process regression and classification"},
    {"id":"sklearn_kernel","label":"Kernel Ridge & Friends","categoryId":"models_sklearn","description":"Kernel ridge and related kernelized estimators"},
    {"id":"sklearn_neural","label":"Neural Networks (sklearn)","categoryId":"models_sklearn","description":"Multi-layer perceptrons and RBMs"},
    {"id":"sklearn_probabilistic","label":"Probabilistic & Calibration","categoryId":"models_sklearn","description":"Probability calibration and isotonic regression"},
    {"id":"sklearn_cross_decomposition","label":"Cross Decomposition","categoryId":"models_sklearn","description":"PLS, CCA, and related latent variable models"},
    {"id":"sklearn_meta","label":"Meta Estimators","categoryId":"models_sklearn","description":"Wrapper estimators (multiclass, multioutput, pipelines)"},
    {"id":"sklearn_semi_supervised","label":"Semi-supervised","categoryId":"models_sklearn","description":"Label propagation and self-training models"},
    {"id":"sklearn_baseline","label":"Baseline Models","categoryId":"models_sklearn","description":"Dummy estimators for quick sanity checks"},
    {"id":"sklearn_misc_models","label":"Miscellaneous Models","categoryId":"models_sklearn","description":"Other scikit-learn estimators compatible with nirs4all"},
    {"id":"tensorflow_models","label":"TensorFlow Models","categoryId":"models_deep","description":"TensorFlow architectures bundled with nirs4all"},
    {"id":"splitting_strategies","label":"Splitting Strategies","categoryId":"validation","description":"Cross-validation and sampling strategies"},
    {"id":"target_transforms","label":"Target Transforms","categoryId":"targets","description":"Transforms applied to the target variable y"},
    {"id":"prediction_ops","label":"Prediction Utilities","categoryId":"prediction","description":"Batch, real-time prediction, and calibration helpers"},
    {"id":"containers","label":"Containers","categoryId":"utilities","description":"Pipeline containers that hold other nodes"},
    {"id":"generators","label":"Generators","categoryId":"utilities","description":"Hyperparameter and branching generators"},
    {"id":"visualization","label":"Visualization","categoryId":"utilities","description":"Nodes that produce charts or reports"},
]


MANUAL_COMPONENTS = [
    {"subcategory":"spectral_augmentation","id":"rotate_translate","label":"Rotate & Translate","short":"RotateTranslate","description":"Random affine spectral augmentation (Rotate_Translate)"},
    {"subcategory":"spectral_augmentation","id":"random_x_operation","label":"Random X Operation","short":"RandomX","description":"Random multiplicative/additive perturbations"},
    {"subcategory":"spectral_augmentation","id":"spline_smoothing","label":"Spline Smoothing","short":"SplineSmooth","description":"Spline-based smoothing augmentation"},
    {"subcategory":"spectral_augmentation","id":"spline_x_perturbations","label":"Spline X Perturbations","short":"SplineX","description":"Wavelength warping via splines"},
    {"subcategory":"spectral_augmentation","id":"spline_y_perturbations","label":"Spline Y Perturbations","short":"SplineY","description":"Intensity perturbations via splines"},
    {"subcategory":"spectral_augmentation","id":"spline_x_simplification","label":"Spline X Simplification","short":"SplineXSimpl","description":"Control-point simplification along wavelength axis"},
    {"subcategory":"spectral_augmentation","id":"spline_curve_simplification","label":"Spline Curve Simplification","short":"SplineCurve","description":"Curve simplification for augmentation"},
    {"subcategory":"spectral_augmentation","id":"identity_augmenter","label":"Identity Augmenter","short":"IdentityAug","description":"Pass-through augmenter for pipelines"},
    {"subcategory":"baseline_correction","id":"baseline","label":"Baseline Removal","short":"Baseline","description":"Polynomial baseline correction (Baseline)"},
    {"subcategory":"baseline_correction","id":"detrend","label":"Detrend","short":"Detrend","description":"Remove linear or constant trends (Detrend)"},
    {"subcategory":"scatter_correction","id":"multiplicative_scatter_correction","label":"MSC","short":"MSC","description":"Multiplicative scatter correction (MSC)"},
    {"subcategory":"scatter_correction","id":"standard_normal_variate","label":"Standard Normal Variate","short":"SNV","description":"Row-wise standardization (StandardNormalVariate)"},
    {"subcategory":"scatter_correction","id":"robust_normal_variate","label":"Robust Normal Variate","short":"RobustNV","description":"Robust row-wise scaling (RobustNormalVariate)"},
    {"subcategory":"smoothing","id":"savitzky_golay","label":"Savitzky-Golay","short":"SavGol","description":"Savitzky-Golay smoothing and derivatives"},
    {"subcategory":"smoothing","id":"gaussian_smoothing","label":"Gaussian Filter","short":"Gaussian","description":"Gaussian smoothing (Gaussian)"},
    {"subcategory":"derivatives","id":"first_derivative","label":"First Derivative","short":"Deriv1","description":"First derivative along wavelengths (FirstDerivative)"},
    {"subcategory":"derivatives","id":"second_derivative","label":"Second Derivative","short":"Deriv2","description":"Second derivative along wavelengths (SecondDerivative)"},
    {"subcategory":"derivatives","id":"derivate_samples","label":"Sample Derivative","short":"Derivate","description":"Derivative along sample axis (Derivate)"},
    {"subcategory":"spectral_transforms","id":"wavelet_transform","label":"Wavelet Transform","short":"Wavelet","description":"Continuous wavelet transform (Wavelet)"},
    {"subcategory":"spectral_transforms","id":"haar_wavelet","label":"Haar Wavelet","short":"Haar","description":"Haar wavelet decomposition (Haar)"},
    {"subcategory":"spectral_transforms","id":"log_transform","label":"Log Transform","short":"Log","description":"Logarithmic scaling of spectra (LogTransform)"},
    {"subcategory":"nirs_scalers","id":"normalize_rows","label":"Normalize Rows","short":"Normalize","description":"Row-wise normalization across wavelengths"},
    {"subcategory":"nirs_scalers","id":"simple_scale","label":"Simple Scale","short":"SimpleScale","description":"Scale by constant factors (SimpleScale)"},
    {"subcategory":"resampling_alignment","id":"crop_transformer","label":"Crop Transformer","short":"Crop","description":"Crop spectra to wavelength ranges (CropTransformer)"},
    {"subcategory":"resampling_alignment","id":"resample_transformer","label":"Resample Transformer","short":"Resample","description":"Resample spectra to custom grids (ResampleTransformer)"},
    {"subcategory":"resampling_alignment","id":"resampler","label":"Adaptive Resampler","short":"Resampler","description":"Adaptive resampling controller (Resampler)"},
    {"subcategory":"splitting_strategies","id":"shuffle_split","label":"Shuffle Split","short":"ShuffleSplit","description":"Random permutation cross-validator (ShuffleSplit)","defaults":{"n_splits":3,"test_size":0.25,"random_state":None},"editable":[{"name":"n_splits","type":"integer","description":"Number of re-shuffling iterations","default":3},{"name":"test_size","type":"number","description":"Proportion of dataset for test split","default":0.25},{"name":"random_state","type":"integer","description":"Random seed for reproducibility","default":None}]},
    {"subcategory":"splitting_strategies","id":"kfold","label":"K-Fold","short":"KFold","description":"K-fold cross-validation (KFold)","defaults":{"n_splits":5,"shuffle":True,"random_state":None},"editable":[{"name":"n_splits","type":"integer","description":"Number of folds","default":5},{"name":"shuffle","type":"boolean","description":"Shuffle before splitting","default":True},{"name":"random_state","type":"integer","description":"Random seed","default":None}]},
    {"subcategory":"splitting_strategies","id":"stratified_kfold","label":"Stratified K-Fold","short":"StratKFold","description":"Stratified K-fold cross-validation (StratifiedKFold)","defaults":{"n_splits":5,"shuffle":True,"random_state":None},"editable":[{"name":"n_splits","type":"integer","description":"Number of folds","default":5},{"name":"shuffle","type":"boolean","description":"Shuffle class labels before splitting","default":True},{"name":"random_state","type":"integer","description":"Random seed for reproducible shuffling","default":None}]},
    {"subcategory":"splitting_strategies","id":"group_kfold","label":"Group K-Fold","short":"GroupKFold","description":"Group-aware K-fold cross-validation (GroupKFold)","defaults":{"n_splits":5},"editable":[{"name":"n_splits","type":"integer","description":"Number of folds","default":5}]},
    {"subcategory":"splitting_strategies","id":"repeated_kfold","label":"Repeated K-Fold","short":"RepeatKFold","description":"Repeated K-fold cross-validation (RepeatedKFold)","defaults":{"n_splits":5,"n_repeats":10,"random_state":None},"editable":[{"name":"n_splits","type":"integer","description":"Number of folds","default":5},{"name":"n_repeats","type":"integer","description":"Number of repetitions","default":10},{"name":"random_state","type":"integer","description":"Random seed","default":None}]},
    {"subcategory":"splitting_strategies","id":"repeated_stratified_kfold","label":"Repeated Stratified K-Fold","short":"RepeatStratKFold","description":"Repeated stratified K-fold cross-validation (RepeatedStratifiedKFold)","defaults":{"n_splits":5,"n_repeats":10,"random_state":None},"editable":[{"name":"n_splits","type":"integer","description":"Number of folds","default":5},{"name":"n_repeats","type":"integer","description":"Number of repetitions","default":10},{"name":"random_state","type":"integer","description":"Random seed","default":None}]},
    {"subcategory":"splitting_strategies","id":"stratified_shuffle_split","label":"Stratified Shuffle Split","short":"StratShuffle","description":"Stratified shuffle split (StratifiedShuffleSplit)","defaults":{"n_splits":10,"test_size":0.2,"random_state":None},"editable":[{"name":"n_splits","type":"integer","description":"Number of splits","default":10},{"name":"test_size","type":"number","description":"Test proportion (0.0-1.0)","default":0.2},{"name":"random_state","type":"integer","description":"Random seed","default":None}]},
    {"subcategory":"splitting_strategies","id":"group_shuffle_split","label":"Group Shuffle Split","short":"GroupShuffle","description":"Group-aware shuffle split (GroupShuffleSplit)","defaults":{"n_splits":5,"test_size":0.2,"random_state":None},"editable":[{"name":"n_splits","type":"integer","description":"Number of splits","default":5},{"name":"test_size","type":"number","description":"Test proportion (0.0-1.0)","default":0.2},{"name":"random_state","type":"integer","description":"Random seed","default":None}]},
    {"subcategory":"splitting_strategies","id":"time_series_split","label":"Time Series Split","short":"TimeSeries","description":"Forward-chaining split for temporal data (TimeSeriesSplit)","defaults":{"n_splits":5,"test_size":None,"gap":0},"editable":[{"name":"n_splits","type":"integer","description":"Number of splits","default":5},{"name":"test_size","type":"integer","description":"Size of test sets (None = automatic)","default":None},{"name":"gap","type":"integer","description":"Samples to exclude between train/test","default":0}]},
    {"subcategory":"splitting_strategies","id":"leave_one_out","label":"Leave-One-Out","short":"LOO","description":"Leave-one-out cross-validation (LeaveOneOut)"},
    {"subcategory":"splitting_strategies","id":"leave_p_out","label":"Leave-P-Out","short":"LPO","description":"Leave-P-out cross-validation (LeavePOut)","defaults":{"p":2},"editable":[{"name":"p","type":"integer","description":"Number of samples to leave out","default":2}]},
    {"subcategory":"splitting_strategies","id":"kennard_stone","label":"Kennard-Stone Splitter","short":"KennardStone","description":"D-optimal Kennard-Stone sampling"},
    {"subcategory":"splitting_strategies","id":"spxy_splitter","label":"SPXY Splitter","short":"SPXY","description":"SPXY sampling combining feature and target diversity"},
    {"subcategory":"splitting_strategies","id":"kmeans_splitter","label":"KMeans Splitter","short":"KMeansSplit","description":"Clustering-based sampling with k-means"},
    {"subcategory":"splitting_strategies","id":"split_splitter","label":"SPlit Splitter","short":"SPlit","description":"SPlit sampling based on twinning"},
    {"subcategory":"splitting_strategies","id":"systematic_circular","label":"Systematic Circular","short":"SysCircular","description":"Systematic circular sampling strategy"},
    {"subcategory":"splitting_strategies","id":"kbins_stratified","label":"KBins Stratified","short":"KBins","description":"KBins stratified sampling in feature space"},
    {"subcategory":"target_transforms","id":"integer_kbins_discretizer","label":"Integer KBins Discretizer","short":"KBinsY","description":"Discretize targets into integer KBins"},
    {"subcategory":"target_transforms","id":"range_discretizer","label":"Range Discretizer","short":"RangeDisc","description":"Custom range-based discretization of targets"},
    {"subcategory":"prediction_ops","id":"batch_prediction","label":"Batch Prediction","short":"BatchPredict","description":"Batch inference helper","generationMode":"out"},
    {"subcategory":"prediction_ops","id":"real_time_prediction","label":"Real-time Prediction","short":"Realtime","description":"Streaming inference helper","generationMode":"out"},
    {"subcategory":"prediction_ops","id":"probability_calibration","label":"Probability Calibration","short":"Calibrate","description":"Calibrate predicted probabilities","generationMode":"out"},
    {"subcategory":"visualization","id":"chart_2d","label":"2D Chart","short":"Chart2D","description":"Generate 2D spectra charts"},
    {"subcategory":"visualization","id":"chart_3d","label":"3D Chart","short":"Chart3D","description":"Generate 3D spectra charts"},
    {"subcategory":"visualization","id":"y_chart","label":"Y Distribution Chart","short":"YChart","description":"Visualize target distributions"},
    {"subcategory":"visualization","id":"fold_chart","label":"Fold Chart","short":"FoldChart","description":"Visualize cross-validation folds"},
    {"subcategory":"visualization","id":"confusion_matrix","label":"Confusion Matrix","short":"Confusion","description":"Visualize classification confusion matrices"},
    {"subcategory":"containers","id":"feature_augmentation","label":"Feature Augmentation","short":"FeatureAug","description":"Container executing feature augmentation steps","nodeType":"container","generationMode":"out","allowed":["category:augmentation","category:preprocessing","category:feature_engineering"]},
    {"subcategory":"containers","id":"augmentation_sample","label":"Sample Augmentation","short":"SampleAug","description":"Container generating augmented samples","nodeType":"container","generationMode":"in-place","allowed":["category:augmentation","category:preprocessing"]},
    {"subcategory":"containers","id":"sequential","label":"Sequential","short":"Sequential","description":"Container executing preprocessing steps sequentially","nodeType":"container","generationMode":"in-place","allowed":["category:augmentation","category:preprocessing","category:feature_engineering"]},
    {"subcategory":"containers","id":"pipeline","label":"Pipeline","short":"Pipeline","description":"Container representing a full preprocessing pipeline","nodeType":"container","generationMode":"in-place","allowed":["*"]},
    {"subcategory":"containers","id":"y_processing","label":"Y Processing","short":"YProcess","description":"Container for target-side preprocessing","nodeType":"container","generationMode":"in-place","allowed":["category:targets"]},
    {"subcategory":"generators","id":"_or_","label":"OR Generator","short":"_OR_","description":"Generate alternative branches (OR)","nodeType":"generation","generationMode":"generator","allowed":["*"]},
    {"subcategory":"generators","id":"_range_","label":"Range Generator","short":"_RANGE_","description":"Generate parameter ranges for sweeps","nodeType":"generation","generationMode":"generator","allowed":["*"]},
]

CLASS_PATHS: dict[str, str] = {
    # Augmentation
    "rotate_translate": "nirs4all.operators.augmentation.random.Rotate_Translate",
    "random_x_operation": "nirs4all.operators.augmentation.random.Random_X_Operation",
    "spline_smoothing": "nirs4all.operators.augmentation.splines.Spline_Smoothing",
    "spline_x_perturbations": "nirs4all.operators.augmentation.splines.Spline_X_Perturbations",
    "spline_y_perturbations": "nirs4all.operators.augmentation.splines.Spline_Y_Perturbations",
    "spline_x_simplification": "nirs4all.operators.augmentation.splines.Spline_X_Simplification",
    "spline_curve_simplification": "nirs4all.operators.augmentation.splines.Spline_Curve_Simplification",
    "identity_augmenter": "nirs4all.operators.augmentation.abc_augmenter.IdentityAugmenter",

    # Baseline / preprocessing
    "baseline": "nirs4all.operators.transforms.signal.Baseline",
    "detrend": "nirs4all.operators.transforms.signal.Detrend",
    "multiplicative_scatter_correction": "nirs4all.operators.transforms.nirs.MultiplicativeScatterCorrection",
    "standard_normal_variate": "nirs4all.operators.transforms.scalers.StandardNormalVariate",
    "robust_normal_variate": "nirs4all.operators.transforms.scalers.RobustNormalVariate",
    "savitzky_golay": "nirs4all.operators.transforms.nirs.SavitzkyGolay",
    "gaussian_smoothing": "nirs4all.operators.transforms.signal.Gaussian",
    "first_derivative": "nirs4all.operators.transforms.nirs.FirstDerivative",
    "second_derivative": "nirs4all.operators.transforms.nirs.SecondDerivative",
    "derivate_samples": "nirs4all.operators.transforms.scalers.Derivate",
    "haar_wavelet": "nirs4all.operators.transforms.nirs.Haar",
    "log_transform": "nirs4all.operators.transforms.nirs.LogTransform",
    "normalize_rows": "nirs4all.operators.transforms.scalers.Normalize",
    "simple_scale": "nirs4all.operators.transforms.scalers.SimpleScale",
    "crop_transformer": "nirs4all.operators.transforms.features.CropTransformer",
    "resample_transformer": "nirs4all.operators.transforms.features.ResampleTransformer",
    "resampler": "nirs4all.operators.transforms.resampler.Resampler",

    # Splitters
    "shuffle_split": "sklearn.model_selection._split.ShuffleSplit",
    "kfold": "sklearn.model_selection._split.KFold",
    "stratified_kfold": "sklearn.model_selection._split.StratifiedKFold",
    "group_kfold": "sklearn.model_selection._split.GroupKFold",
    "repeated_kfold": "sklearn.model_selection._split.RepeatedKFold",
    "repeated_stratified_kfold": "sklearn.model_selection._split.RepeatedStratifiedKFold",
    "stratified_shuffle_split": "sklearn.model_selection._split.StratifiedShuffleSplit",
    "group_shuffle_split": "sklearn.model_selection._split.GroupShuffleSplit",
    "time_series_split": "sklearn.model_selection._split.TimeSeriesSplit",
    "leave_one_out": "sklearn.model_selection._split.LeaveOneOut",
    "leave_p_out": "sklearn.model_selection._split.LeavePOut",
    "kennard_stone": "nirs4all.operators.splitters.splitters.KennardStoneSplitter",
    "spxy_splitter": "nirs4all.operators.splitters.splitters.SPXYSplitter",
    "kmeans_splitter": "nirs4all.operators.splitters.splitters.KMeansSplitter",
    "split_splitter": "nirs4all.operators.splitters.splitters.SPlitSplitter",
    "systematic_circular": "nirs4all.operators.splitters.splitters.SystematicCircularSplitter",
    "kbins_stratified": "nirs4all.operators.splitters.splitters.KBinsStratifiedSplitter",

    # Target transforms
    "integer_kbins_discretizer": "nirs4all.operators.transforms.targets.IntegerKBinsDiscretizer",
    "range_discretizer": "nirs4all.operators.transforms.targets.RangeDiscretizer",
}

FUNCTION_PATHS: dict[str, str] = {
    "wavelet_transform": "nirs4all.operators.transforms.nirs.wavelet_transform",
}


def manual_components() -> list[dict]:
    comps = []
    for spec in MANUAL_COMPONENTS:
        comps.append({
            "id": spec["id"],
            "label": spec["label"],
            "shortName": spec["short"],
            "subcategoryId": spec["subcategory"],
            "description": spec["description"],
            "nodeType": spec.get("nodeType", "regular"),
            "allowedChildren": spec.get("allowed", []),
            "defaultParams": spec.get("defaults", {}),
            "editableParams": spec.get("editable", []),
            "generationMode": spec.get("generationMode", "in-place"),
        })
    return comps


def module_prefix(module: str) -> str:
    parts = module.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else module


def scikit_transformer_subcategory(module: str) -> str:
    pref = module_prefix(module)
    if pref == "sklearn.preprocessing":
        if "_encod" in module or "_label" in module or "_target_encoder" in module or "_discretization" in module:
            return "scikit_encoding"
        return "scikit_scalers"
    if pref == "sklearn.impute":
        return "scikit_imputation"
    if pref in {"sklearn.decomposition", "sklearn.manifold", "sklearn.cross_decomposition"}:
        return "scikit_dimensionality"
    if pref == "sklearn.feature_selection":
        return "scikit_feature_selection"
    if pref in {"sklearn.kernel_approximation", "sklearn.random_projection"}:
        return "scikit_kernel_projection"
    if pref == "sklearn.feature_extraction":
        return "scikit_feature_extraction"
    if pref in {"sklearn.cluster", "sklearn.neighbors"}:
        return "scikit_cluster_neighbors"
    if pref in {"sklearn.compose", "sklearn.pipeline", "sklearn.ensemble"}:
        return "scikit_meta_transformers"
    return "scikit_misc_transformers"


def scikit_model_subcategory(module: str) -> str:
    pref = module_prefix(module)
    if pref == "sklearn.linear_model":
        return "sklearn_linear"
    if pref == "sklearn.svm":
        return "sklearn_svm"
    if pref == "sklearn.tree":
        return "sklearn_tree"
    if pref == "sklearn.ensemble":
        return "sklearn_ensemble"
    if pref == "sklearn.neighbors":
        return "sklearn_neighbors"
    if pref == "sklearn.naive_bayes":
        return "sklearn_naive_bayes"
    if pref == "sklearn.discriminant_analysis":
        return "sklearn_discriminant"
    if pref == "sklearn.gaussian_process":
        return "sklearn_gaussian_process"
    if pref == "sklearn.kernel_ridge":
        return "sklearn_kernel"
    if pref == "sklearn.neural_network":
        return "sklearn_neural"
    if pref == "sklearn.calibration" or pref == "sklearn.isotonic" or "classification_threshold" in module:
        return "sklearn_probabilistic"
    if pref == "sklearn.cross_decomposition":
        return "sklearn_cross_decomposition"
    if pref in {"sklearn.multiclass", "sklearn.multioutput", "sklearn.compose"}:
        return "sklearn_meta"
    if pref == "sklearn.semi_supervised":
        return "sklearn_semi_supervised"
    if pref == "sklearn.dummy":
        return "sklearn_baseline"
    return "sklearn_misc_models"


def scikit_transformers() -> list[dict]:
    comps: list[dict] = []
    seen: set[str] = set()
    for name, cls in all_estimators(type_filter="transformer"):
        cid = camel_to_snake(name)
        if cid in seen:
            continue
        seen.add(cid)
        module = cls.__module__
        comps.append({
            "id": cid,
            "label": name,
            "shortName": name,
            "subcategoryId": scikit_transformer_subcategory(module),
            "description": f"scikit-learn transformer {name} ({module})",
            "nodeType": "regular",
            "allowedChildren": [],
            "defaultParams": {"class_path": f"{module}.{name}"},
            "editableParams": [],
            "generationMode": "in-place",
        })
    return comps


def scikit_models() -> list[dict]:
    info: dict[str, dict] = {}
    for est_type in ("classifier", "regressor"):
        for name, cls in all_estimators(type_filter=est_type):
            entry = info.setdefault(name, {"cls": cls, "types": set()})
            entry["types"].add(est_type)
    comps: list[dict] = []
    for name, entry in info.items():
        module = entry["cls"].__module__
        est_types = sorted(entry["types"])
        est_type_label = "classifier_regressor" if len(est_types) == 2 else est_types[0]
        comps.append({
            "id": camel_to_snake(name),
            "label": name,
            "shortName": name,
            "subcategoryId": scikit_model_subcategory(module),
            "description": f"scikit-learn {', '.join(est_types)} {name} ({module})",
            "nodeType": "regular",
            "allowedChildren": [],
            "defaultParams": {"class_path": f"{module}.{name}", "estimator_type": est_type_label},
            "editableParams": [],
            "generationMode": "out",
        })
    return comps


def safe_import_attr(path: str):
    module_name, attr_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def sanitize_default(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str) or value is None:
        return value
    if np is not None:
        if isinstance(value, np.generic):
            return sanitize_default(value.item())
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [sanitize_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): sanitize_default(v) for k, v in value.items()}
    return repr(value)


def infer_param_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "string"
    return "string"


def build_editable_params(defaults: dict[str, object]) -> list[dict]:
    editable = []
    for name, default in defaults.items():
        editable.append({
            "name": name,
            "type": infer_param_type(default),
            "description": "",
            "default": default,
        })
    return editable


def extract_defaults(path: str, is_function: bool = False) -> tuple[dict, list]:
    try:
        target = safe_import_attr(path)
    except Exception:
        return {}, []

    callable_obj = target if is_function or not inspect.isclass(target) else target.__init__

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}, []

    defaults: dict[str, object] = {}
    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        default = None if param.default is inspect._empty else sanitize_default(param.default)
        defaults[name] = default

    editable = build_editable_params(defaults)
    return defaults, editable


def prepare_component_metadata(components: list[dict]) -> None:
    for comp in components:
        defaults = comp.get("defaultParams") or {}
        # Extract metadata stored inside default params
        if "class_path" in defaults:
            comp["classPath"] = defaults.pop("class_path")
        if "function_path" in defaults:
            comp["functionPath"] = defaults.pop("function_path")

        cid = comp["id"]
        if cid in CLASS_PATHS:
            comp["classPath"] = CLASS_PATHS[cid]
        if cid in FUNCTION_PATHS:
            comp["functionPath"] = FUNCTION_PATHS[cid]

        comp["defaultParams"] = defaults


def populate_defaults(components: list[dict]) -> None:
    for comp in components:
        if comp.get("nodeType") in {"container", "generation"}:
            continue

        class_path = comp.get("classPath")
        function_path = comp.get("functionPath")

        defaults: dict[str, object] = {}
        editable: list[dict] = []

        if class_path:
            d, e = extract_defaults(class_path)
            defaults.update(d)
            editable = e
        elif function_path:
            d, e = extract_defaults(function_path, is_function=True)
            defaults.update(d)
            editable = e

        if defaults:
            # merge with existing defaults, preferring explicit ones already set
            existing = comp.get("defaultParams") or {}
            merged = {**defaults, **existing}
            comp["defaultParams"] = merged
            if not comp.get("editableParams"):
                comp["editableParams"] = build_editable_params(merged)
        else:
            comp.setdefault("defaultParams", {})

def tensorflow_models() -> list[dict]:
    funcs: dict[str, str] = {}
    for module in (generic_tf, cirad_tf):
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if getattr(obj, "framework", None) == "tensorflow":
                funcs[name] = obj.__module__
    comps: list[dict] = []
    for name, module in sorted(funcs.items()):
        comps.append({
            "id": camel_to_snake(name),
            "label": name,
            "shortName": name,
            "subcategoryId": "tensorflow_models",
            "description": f"TensorFlow model factory {name} ({module})",
            "nodeType": "regular",
            "allowedChildren": [],
            "defaultParams": {"function_path": f"{module}.{name}"},
            "editableParams": [],
            "generationMode": "out",
        })
    return comps


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    output_path = base_dir / "public" / "component-library.json"
    components = manual_components()
    components.extend(scikit_transformers())
    components.extend(scikit_models())
    components.extend(tensorflow_models())
    prepare_component_metadata(components)
    populate_defaults(components)
    components.sort(key=lambda c: (c["subcategoryId"], c["label"].lower()))
    data = {"categories": CATEGORIES, "subcategories": SUBCATEGORIES, "components": components}
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
