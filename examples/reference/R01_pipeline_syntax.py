"""
R01 - Pipeline Syntax Reference
===============================

Comprehensive reference for ALL nirs4all pipeline declaration formats.

This file serves as documentation of the complete pipeline syntax.
It is designed to be read and studied, not necessarily executed as-is.
Some examples reference files that may not exist.

This reference covers:

* Basic step formats (class, instance, dict, string)
* Step wrapper keywords (preprocessing, y_processing, model, etc.)
* Generator syntax (_or_, _range_, pick, count, etc.)
* Tagging and exclusion (tag, exclude)
* Branching types (duplication branches, separation branches)
* Merging (branch merge, source merge, concat merge)
* Model configuration and finetuning
* Data augmentation (feature_augmentation, sample_augmentation)
* Cross-validation splitters
* Loading saved transformers and models

For generator syntax details, see :ref:`R02_generator_reference`.
For a runnable all-keywords test, see :ref:`R03_all_keywords`.

Duration: Reading reference (~10-15 minutes)
Difficulty: Reference material
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.operators.filters import XOutlierFilter, YOutlierFilter
from nirs4all.operators.transforms import Detrend, FirstDerivative, Gaussian, Haar, SavitzkyGolay, SecondDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# =============================================================================
# SECTION 1: Basic Step Formats
# =============================================================================
#
# A pipeline is a list of steps. Each step can be specified in multiple
# equivalent formats. All formats are normalized during pipeline expansion.
#
# Format 1: Class (not instantiated)
# ----------------------------------
# The class will be instantiated with default parameters.
step_class = MinMaxScaler  # Will become MinMaxScaler()

# Format 2: Instance (instantiated with parameters)
# -------------------------------------------------
# Pre-configured with specific parameters.
step_instance = MinMaxScaler(feature_range=(0.1, 0.8))

# Format 3: String - Controller name or saved file
# ------------------------------------------------
# Special strings trigger specific controllers or load saved objects.
step_string_controller = "chart_2d"           # Triggers a controller by name
step_string_file_pkl = "my_transformer.pkl"   # Load saved sklearn transformer
step_string_file_h5 = "my_model.h5"           # Load saved TensorFlow model
step_string_class_path = "sklearn.linear_model.Ridge"  # Full class path

# Format 4: Dictionary - Explicit configuration
# ---------------------------------------------
# Most flexible format, supports all options.
step_dict_class_path = {
    "class": "sklearn.preprocessing.MinMaxScaler",
    "params": {
        "feature_range": (0, 1)
    }
}

step_dict_custom_code = {
    "source_file": "my_custom_module.py",
    "class": "MyCustomTransformer"
}

# =============================================================================
# SECTION 2: Step Wrapper Keywords
# =============================================================================
#
# Keywords wrap operators to specify their role in the pipeline.
# Without a keyword, transformers apply to features (X).
#

# 2.1 preprocessing - Explicit feature preprocessing
# --------------------------------------------------
# Usually implicit (bare transformers apply to X), but can be explicit.
step_preprocessing = {"preprocessing": StandardScaler()}

# 2.2 y_processing - Target preprocessing
# ---------------------------------------
# Apply transformation to targets (y). Inverses are applied automatically
# during prediction.
step_y_processing = {"y_processing": MinMaxScaler()}

# 2.3 model - Model training step
# -------------------------------
# Identifies a step as a model rather than a transformer.
step_model_basic = {"model": PLSRegression(n_components=10)}

step_model_named = {
    "name": "PLS-10",
    "model": PLSRegression(n_components=10)
}

# 2.4 feature_augmentation - Create multiple feature views
# --------------------------------------------------------
# Generate variants by applying different preprocessing options.
step_feature_augmentation = {
    "feature_augmentation": [SNV, FirstDerivative, MSC],
    "action": "extend"  # "extend" (default) | "add" | "replace"
}

# With generator syntax:
step_feature_augmentation_generator = {
    "feature_augmentation": {
        "_or_": [SNV, FirstDerivative, SecondDerivative, Gaussian, MSC],
        "pick": 2,    # Select combinations of 2
        "count": 5    # Limit to 5 variants
    }
}

# 2.5 sample_augmentation - Data augmentation on samples
# ------------------------------------------------------
# Create additional training samples with perturbations.
from nirs4all.operators.transforms import GaussianAdditiveNoise, MultiplicativeNoise, Rotate_Translate

step_sample_augmentation = {
    "sample_augmentation": {
        "transformers": [
            Rotate_Translate(p_range=1, y_factor=2),
            GaussianAdditiveNoise(sigma=0.005),
        ],
        "count": 2,            # Augmented samples per original
        "selection": "random", # "random" | "all" | "sequential"
        "random_state": 42,
    }
}

# 2.6 concat_transform - Concatenate transformer outputs
# ------------------------------------------------------
# Horizontally concatenate features from multiple transformers.
step_concat_transform = {
    "concat_transform": [
        PCA(n_components=15),
        TruncatedSVD(n_components=10),
    ]
}

# =============================================================================
# SECTION 3: Tagging and Exclusion (v2)
# =============================================================================
#
# Tag samples for analysis without removing them. Exclude samples to remove
# from training. Both keywords work with filter operators.
#

# 3.1 tag - Tag samples without removing
# --------------------------------------
# Tags samples matching the filter. Tagged samples are still used in training.
# Useful for analysis and visualization of outliers without removing them.
step_tag_simple = {"tag": YOutlierFilter(method="iqr", threshold=1.5)}

# Creates a tag column (e.g., "y_outlier_iqr") in predictions
step_tag_named = {"tag": YOutlierFilter(method="zscore", tag_name="zscore_outlier")}

# 3.2 exclude - Remove samples from training
# ------------------------------------------
# Excludes samples matching the filter from training.
# Also creates a tag for analysis. Does NOT apply during prediction.
step_exclude_simple = {"exclude": YOutlierFilter(method="iqr", threshold=1.5)}

# Multiple filters with mode
step_exclude_multiple = {
    "exclude": [
        YOutlierFilter(method="iqr", threshold=1.5),
        XOutlierFilter(method="pca_leverage"),
    ],
    "mode": "any",  # "any" (default) or "all"
}

# =============================================================================
# SECTION 4: Branching and Merging (v2)
# =============================================================================
#
# Two types of branches:
#   - DUPLICATION branches: Same samples, different preprocessing
#   - SEPARATION branches: Different samples, parallel processing
#
# Merging combines branch outputs.
#

# 4.1 branch - Create DUPLICATION branches (default)
# --------------------------------------------------
# All branches process the SAME samples with different preprocessing.

# List syntax: Unnamed branches (accessed by index)
step_branch_list = {
    "branch": [
        [SNV(), PLSRegression(n_components=10)],         # Branch 0
        [MSC(), RandomForestRegressor(n_estimators=50)], # Branch 1
    ]
}

# Dict syntax: Named branches
step_branch_dict = {
    "branch": {
        "pls_path": [SNV(), {"model": PLSRegression(n_components=10)}],
        "rf_path": [MSC(), {"model": RandomForestRegressor(n_estimators=50)}],
    }
}

# 4.2 SEPARATION branches - Different samples per branch
# ------------------------------------------------------
# Samples are routed to branches based on some criterion.
# Use "concat" merge to reassemble samples.

# by_metadata: Branch by metadata column value
step_branch_by_metadata = {
    "branch": {
        "by_metadata": "instrument",
        "steps": {
            "NIR_500": [SNV()],
            "NIR_700": [MSC()],
        }
    }
}

# by_tag: Branch by tag values (from prior tag/exclude step)
step_branch_by_tag = {
    "branch": {
        "by_tag": "y_outlier_iqr",
        "steps": {
            "outlier": [SNV()],    # Samples tagged as outlier
            "normal": [MSC()],     # Samples not tagged
        }
    }
}

# by_filter: Branch by filter result (pass/fail)
step_branch_by_filter = {
    "branch": {
        "by_filter": YOutlierFilter(method="iqr"),
        "steps": {
            "pass": [SNV()],   # Samples passing filter (not outliers)
            "fail": [MSC()],   # Samples failing filter (outliers)
        }
    }
}

# by_source: Branch by feature source (multi-source datasets)
step_branch_by_source = {
    "branch": {
        "by_source": True,
        "steps": {
            "NIR": [SNV(), FirstDerivative()],
            "markers": [StandardScaler()],
        }
    }
}

# 4.3 merge - Combine branch outputs
# ----------------------------------
# Merge ALWAYS exits branch mode (returns to single execution path).

# For DUPLICATION branches:
step_merge_features_all = {"merge": "features"}      # Concatenate X matrices
step_merge_predictions = {"merge": "predictions"}    # Collect OOF predictions

# Selective merge with per-branch model selection
step_merge_selective = {
    "merge": {
        "predictions": [
            {"branch": 0, "select": "best", "metric": "rmse"},
            {"branch": 1, "select": {"top_k": 2}, "metric": "r2"},
        ],
        "features": [0],  # Also include features from branch 0
        "output_as": "features",  # "features" | "sources" | "dict"
    }
}

# For SEPARATION branches:
step_merge_concat = {"merge": "concat"}  # Reassemble samples in original order

# For by_source branches (source merge):
step_merge_sources = {"merge": {"sources": "concat"}}  # Concatenate source features
step_merge_sources_stack = {"merge": {"sources": "stack"}}  # 3D stacking for CNNs

# =============================================================================
# SECTION 5: Cross-Validation Splitters
# =============================================================================
#
# Splitters define how data is partitioned for training and validation.
# They are automatically detected and handled.
#

splitter_shuffle = ShuffleSplit(n_splits=3, test_size=0.25)
splitter_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splitter_group = GroupKFold(n_splits=3)  # Requires group metadata

# Dict format with explicit class path:
splitter_dict = {
    "class": "sklearn.model_selection._split.ShuffleSplit",
    "params": {
        "n_splits": 3,
        "test_size": 0.25
    }
}

# =============================================================================
# SECTION 6: Model Configuration
# =============================================================================
#
# Models can be configured with names, parameters, and finetuning options.
#

# 5.1 Basic model
model_basic = {"model": PLSRegression(n_components=10)}

# 5.2 Named model
model_named = {
    "name": "PLS-10",
    "model": PLSRegression(n_components=10)
}

# 5.3 Model with finetuning (Optuna integration)
model_finetune = {
    "model": PLSRegression(),
    "name": "PLS-Finetuned",
    "finetune_params": {
        "n_trials": 20,
        "verbose": 2,
        "approach": "single",           # "single" | "grouped" | "individual"
        "eval_mode": "best",            # "best" | "mean" | "robust_best"
        "sampler": "grid",              # "auto" | "grid" | "tpe" | "random" | "cmaes"
        "seed": 42,                     # Reproducible optimization
        "metric": "rmse",              # Metric to optimize (auto-infers direction)
        "model_params": {
            'n_components': ('int', 1, 30),
        },
    }
}

# 5.4 Neural network model with train_params
# Note: TensorFlow models require tensorflow to be installed
try:
    from nirs4all.operators.models.tensorflow.nicon import customizable_nicon
    _HAS_TENSORFLOW = True
except ImportError:
    customizable_nicon = None
    _HAS_TENSORFLOW = False

# Only define if TensorFlow is available
if _HAS_TENSORFLOW:
    model_nn = {
        "model": customizable_nicon,
        "name": "CustomNICON",
        "finetune_params": {
            "n_trials": 30,
            "sampler": "tpe",
            "model_params": {
                "filters_1": [8, 16, 32, 64],
                "filters_3": [8, 16, 32, 64]
            },
            "train_params": {
                "epochs": 10,  # During hyperparameter search
                "verbose": 0
            }
        },
        "train_params": {
            "epochs": 250,     # Final training
            "verbose": 0
        }
    }

# 5.5 Load saved model
model_saved_pkl = "my_model.pkl"               # Pickle format
model_saved_h5 = "my_tf_model.h5"              # TensorFlow/Keras format
model_saved_pt = "my_pytorch_model.pt"         # PyTorch format

# =============================================================================
# SECTION 7: Generator Syntax (Overview)
# =============================================================================
#
# Generators create multiple pipeline variants from a single specification.
# See R02_generator_reference.py for complete details.
#

# 6.1 _or_ - Choose between alternatives
generator_or = {"_or_": [SNV, MSC, Detrend]}  # 3 variants

# 6.2 _or_ with selection
generator_pick = {
    "_or_": [SNV, MSC, Detrend, FirstDerivative],
    "pick": 2,   # Combinations of 2: C(4,2) = 6 variants
}

generator_arrange = {
    "_or_": [SNV, MSC, Detrend],
    "arrange": 2,  # Permutations of 2: P(3,2) = 6 variants (order matters)
}

# 6.3 _range_ - Numeric sequences
generator_range = {
    "_range_": [1, 12, 2],  # [1, 3, 5, 7, 9, 11]
    "param": "n_components",
    "model": PLSRegression
}

# 6.4 count - Limit number of variants
generator_count = {
    "_or_": [SNV, MSC, Detrend, FirstDerivative, SecondDerivative],
    "count": 3  # Random sample of 3
}

# =============================================================================
# SECTION 8: Complete Pipeline Examples
# =============================================================================

# 7.1 Simple regression pipeline
pipeline_simple = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

# 7.2 Pipeline with preprocessing exploration
pipeline_preprocessing = [
    MinMaxScaler(feature_range=(0.1, 0.8)),
    {"y_processing": MinMaxScaler()},
    {
        "feature_augmentation": {
            "_or_": [Detrend, FirstDerivative, Gaussian, SNV, MSC],
            "pick": (1, 2),  # 1 or 2 preprocessings
            "count": 5
        }
    },
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

# 7.3 Pipeline with branching for model comparison
pipeline_branching = [
    MinMaxScaler(),
    {"y_processing": StandardScaler()},
    KFold(n_splits=3, shuffle=True, random_state=42),
    {
        "branch": {
            "pls": [SNV(), {"model": PLSRegression(n_components=10)}],
            "rf": [MSC(), {"model": RandomForestRegressor(n_estimators=100)}],
        }
    },
    {"merge": "predictions"},
    {"model": Ridge(alpha=1.0)}  # Meta-learner
]

# 7.4 Pipeline with hyperparameter sweep
pipeline_sweep = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),
    {
        "_range_": [2, 20, 2],
        "param": "n_components",
        "model": {
            "class": "sklearn.cross_decomposition.PLSRegression"
        }
    }
]

# =============================================================================
# SECTION 9: Syntax Equivalences
# =============================================================================
#
# These formats are equivalent - they normalize to the same configuration:
#

# These are all equivalent:
equiv_1 = MinMaxScaler
equiv_2 = MinMaxScaler()
equiv_3 = {"preprocessing": MinMaxScaler}
equiv_4 = {"preprocessing": MinMaxScaler()}
equiv_5 = {"class": "sklearn.preprocessing.MinMaxScaler"}

# These are all equivalent for models:
model_equiv_1 = {"model": PLSRegression(n_components=10)}
model_equiv_2 = {
    "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {"n_components": 10}
    }
}

# =============================================================================
# SUMMARY
# =============================================================================
print("""
R01 - Pipeline Syntax Reference (v2)
====================================

This reference documents all nirs4all pipeline declaration formats:

STEP FORMATS:
  Class               MinMaxScaler
  Instance            MinMaxScaler(feature_range=(0, 1))
  String              "sklearn.preprocessing.MinMaxScaler"
  Dict (class path)   {"class": "...", "params": {...}}
  Dict (file)         {"source_file": "...", "class": "..."}

STEP KEYWORDS:
  preprocessing       {"preprocessing": transformer}
  y_processing        {"y_processing": transformer}
  model               {"model": estimator, "name": "..."}
  feature_augmentation {"feature_augmentation": [...], "action": "extend"}
  sample_augmentation {"sample_augmentation": {...}}
  concat_transform    {"concat_transform": [t1, t2, ...]}

TAGGING & EXCLUSION (v2):
  tag                 {"tag": Filter()}  # Tag without removing
  exclude             {"exclude": Filter()}  # Remove from training

BRANCHING (v2):
  DUPLICATION (same samples, different preprocessing):
    branch (list)     {"branch": [[...], [...]]}
    branch (dict)     {"branch": {"name1": [...], ...}}
    merge features    {"merge": "features"}
    merge predictions {"merge": "predictions"}

  SEPARATION (different samples, parallel processing):
    by_metadata       {"branch": {"by_metadata": "col", "steps": {...}}}
    by_tag            {"branch": {"by_tag": "tag_name", "steps": {...}}}
    by_filter         {"branch": {"by_filter": Filter(), "steps": {...}}}
    by_source         {"branch": {"by_source": True, "steps": {...}}}
    merge concat      {"merge": "concat"}  # Reassemble samples
    merge sources     {"merge": {"sources": "concat"}}

GENERATORS (see R02):
  _or_                {"_or_": [A, B, C]}
  _range_             {"_range_": [start, end, step]}
  pick                {"_or_": [...], "pick": n}
  arrange             {"_or_": [...], "arrange": n}
  count               {"...", "count": n}

MODELS:
  Basic               {"model": estimator}
  Named               {"name": "...", "model": estimator}
  Finetuned           {"model": ..., "finetune_params": {...}}
  Saved               "path/to/model.pkl"

For runnable examples, see:
  - R02_generator_reference.py  (generator syntax)
  - R03_all_keywords.py         (all keywords test)
  - U01_hello_world.py          (getting started)
""")
