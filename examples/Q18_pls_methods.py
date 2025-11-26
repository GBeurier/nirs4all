"""
Q18 PLS Methods Example - Comprehensive PLS Methods for NIRS
=============================================================
Demonstrates various PLS (Partial Least Squares) methods for NIRS analysis.
This example showcases both regression and classification tasks using multiple PLS variants.

PLS Methods Demonstrated:
-------------------------
Tier 1 - sklearn native:
  - PLSRegression (NIPALS)
  - PLS-DA (via PLSRegression + OneHotEncoder)

Tier 2 - External packages (install separately):
  - IKPLS (requires: pip install ikpls)
  - OPLS/OPLS-DA (requires: pip install pyopls)
  - MB-PLS (requires: pip install mbpls)
  - Sparse PLS (requires: pip install py-ddspls)

Tier 3 - Variable selection wrappers:
  - VIP, MCUVE, CARS, SPA (requires: pip install auswahl)

Usage:
------
    python Q18_pls_methods.py --plots --show

    Options:
        --plots   Show pipeline plots interactively
        --show    Show all analysis plots at the end
        --all     Run with all available PLS packages (requires extras)
"""

# Standard library imports
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q18 PLS Methods Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
parser.add_argument('--all', action='store_true', help='Run with all PLS packages')
args = parser.parse_args()
display_pipeline_plots = args.plots
display_analyzer_plots = args.show
use_all_packages = args.all

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


###############
### IMPORTS ###
###############

# Third-party imports - always available
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import (
    FirstDerivative, SecondDerivative,
    StandardNormalVariate, SavitzkyGolay,
    MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer


# ============================================================================
# HELPER: Check package availability
# ============================================================================

def check_package(package_name):
    """Check if a package is available."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# Check available PLS packages
HAS_IKPLS = check_package('ikpls')
HAS_PYOPLS = check_package('pyopls')
HAS_MBPLS = check_package('mbpls')
HAS_DDSPLS = check_package('ddspls')
HAS_AUSWAHL = check_package('auswahl')
HAS_TRENDFITTER = check_package('trendfitter')

print("=" * 60)
print("PLS Package Availability")
print("=" * 60)
print(f"  sklearn PLSRegression: ✅ Always available")
print(f"  ikpls (IKPLS):         {'✅' if HAS_IKPLS else '❌'} pip install ikpls")
print(f"  pyopls (OPLS):         {'✅' if HAS_PYOPLS else '❌'} pip install pyopls")
print(f"  mbpls (Multiblock):    {'✅' if HAS_MBPLS else '❌'} pip install mbpls")
print(f"  ddspls (Sparse PLS):   {'✅' if HAS_DDSPLS else '❌'} pip install py-ddspls")
print(f"  auswahl (VIP/CARS):    {'✅' if HAS_AUSWAHL else '❌'} pip install auswahl")
print(f"  trendfitter (DiPLS):   {'✅' if HAS_TRENDFITTER else '❌'} pip install trendfitter")
print("=" * 60)
print()


# ============================================================================
# SECTION 1: REGRESSION with PLS Methods
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 1: REGRESSION - PLS Methods")
print("=" * 60)

# Data configuration for regression
data_regression = {
    'folder': 'sample_data/regression/',
}

# Build regression pipeline with multiple PLS variants
regression_pipeline = [
    # Preprocessing
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate, SavitzkyGolay]},

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25),
]

# ---- Tier 1: Standard sklearn PLSRegression ----
# Add PLS models with different numbers of components
for n_comp in [5, 10, 15, 20]:
    regression_pipeline.append({
        "name": f"PLS_{n_comp}comp",
        "model": PLSRegression(n_components=n_comp)
    })

# ---- Tier 2: IKPLS (if available) ----
if HAS_IKPLS and use_all_packages:
    from ikpls.numpy_ikpls import PLS as NumpyIKPLS

    # IKPLS uses a slightly different API, so we wrap it
    class IKPLSWrapper:
        """Wrapper to make IKPLS sklearn-compatible."""
        def __init__(self, n_components=10, algorithm=1):
            self.n_components = n_components
            self.algorithm = algorithm
            self._model = NumpyIKPLS(algorithm=algorithm)

        def fit(self, X, y):
            y = np.asarray(y).ravel()
            self._model.fit(X, y, A=self.n_components)
            return self

        def predict(self, X):
            return self._model.predict(X, n_components=self.n_components)

        def get_params(self, deep=True):
            return {'n_components': self.n_components, 'algorithm': self.algorithm}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    for n_comp in [5, 10, 15]:
        regression_pipeline.append({
            "name": f"IKPLS_{n_comp}comp",
            "model": IKPLSWrapper(n_components=n_comp)
        })
    print("  Added: IKPLS models (fast kernel PLS)")

# ---- Tier 2: OPLS (if available) ----
if HAS_PYOPLS and use_all_packages:
    from pyopls import OPLS
    from sklearn.pipeline import Pipeline

    # OPLS is a transformer, chain with PLSRegression
    for n_ortho in [1, 2]:
        opls_pipeline = Pipeline([
            ('opls', OPLS(n_components=n_ortho)),
            ('pls', PLSRegression(n_components=1))
        ])
        regression_pipeline.append({
            "name": f"OPLS_{n_ortho}ortho",
            "model": opls_pipeline
        })
    print("  Added: OPLS models (orthogonal signal correction)")

# ---- Tier 2: Sparse PLS (if available) ----
if HAS_DDSPLS and use_all_packages:
    from ddspls import ddsPLS

    for n_comp in [3, 5]:
        regression_pipeline.append({
            "name": f"SparsePLS_{n_comp}comp",
            "model": ddsPLS(n_components=n_comp)
        })
    print("  Added: Sparse PLS models (variable selection)")

# Run regression pipeline
print("\nRunning regression pipeline...")
pipeline_config_reg = PipelineConfigs(regression_pipeline, "Q18_regression")
dataset_config_reg = DatasetConfigs(data_regression)

runner_reg = PipelineRunner(save_files=False, verbose=1, plots_visible=display_pipeline_plots)
predictions_reg, _ = runner_reg.run(pipeline_config_reg, dataset_config_reg)

# Display regression results
print("\n" + "-" * 40)
print("Top 10 Regression Models (by RMSE)")
print("-" * 40)
top_reg = predictions_reg.top(10, 'rmse')
for idx, pred in enumerate(top_reg):
    print(f"{idx+1}. {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])} - {pred.get('preprocessings', 'None')}")


# ============================================================================
# SECTION 2: CLASSIFICATION with PLS-DA Methods
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: CLASSIFICATION - PLS-DA Methods")
print("=" * 60)

# Data configuration for classification
data_classification = {
    'folder': 'sample_data/binary/',
    'params': {
        'has_header': False,
        'delimiter': ';',
        'decimal_separator': '.'
    }
}

# Build classification pipeline
classification_pipeline = [
    # Preprocessing
    StandardScaler(),
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25),
]


# ---- Custom PLS-DA Wrapper ----
class PLSDA:
    """
    PLS-DA: PLS Discriminant Analysis.

    Uses PLSRegression with one-hot encoded targets for classification.
    Predictions are made by finding the class with highest response.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pls_ = None
        self.classes_ = None
        self.encoder_ = None

    def fit(self, X, y):
        # Store unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            # Binary: use single column
            self.encoder_ = LabelEncoder()
            y_encoded = self.encoder_.fit_transform(y).reshape(-1, 1)
        else:
            # Multiclass: one-hot encode
            self.encoder_ = OneHotEncoder(sparse_output=False)
            y_encoded = self.encoder_.fit_transform(y.reshape(-1, 1))

        # Fit PLS
        self.pls_ = PLSRegression(n_components=min(self.n_components, X.shape[1]))
        self.pls_.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_pred_raw = self.pls_.predict(X)

        if len(self.classes_) == 2:
            # Binary: threshold at 0.5
            y_pred_idx = (y_pred_raw.ravel() > 0.5).astype(int)
            return self.classes_[y_pred_idx]
        else:
            # Multiclass: argmax
            y_pred_idx = np.argmax(y_pred_raw, axis=1)
            return self.classes_[y_pred_idx]

    def predict_proba(self, X):
        """Return pseudo-probabilities (PLS responses)."""
        return self.pls_.predict(X)

    def get_params(self, deep=True):
        return {'n_components': self.n_components}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# Add PLS-DA models
for n_comp in [3, 5, 8, 10, 15]:
    classification_pipeline.append({
        "name": f"PLSDA_{n_comp}comp",
        "model": PLSDA(n_components=n_comp)
    })

# ---- OPLS-DA (if available) ----
if HAS_PYOPLS and use_all_packages:
    from pyopls import OPLS
    from sklearn.pipeline import Pipeline

    class OPLSDA:
        """OPLS-DA: Orthogonal PLS Discriminant Analysis."""
        def __init__(self, n_components=1):
            self.n_components = n_components
            self.opls_ = None
            self.plsda_ = None

        def fit(self, X, y):
            # OPLS for orthogonal filtering
            self.opls_ = OPLS(n_components=self.n_components)
            X_filtered = self.opls_.fit_transform(X, y)

            # PLS-DA on filtered data
            self.plsda_ = PLSDA(n_components=1)
            self.plsda_.fit(X_filtered, y)
            return self

        def predict(self, X):
            X_filtered = self.opls_.transform(X)
            return self.plsda_.predict(X_filtered)

        def get_params(self, deep=True):
            return {'n_components': self.n_components}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    for n_ortho in [1, 2]:
        classification_pipeline.append({
            "name": f"OPLSDA_{n_ortho}ortho",
            "model": OPLSDA(n_components=n_ortho)
        })
    print("  Added: OPLS-DA models")

# Run classification pipeline
print("\nRunning classification pipeline...")
pipeline_config_cls = PipelineConfigs(classification_pipeline, "Q18_classification")
dataset_config_cls = DatasetConfigs([data_classification])

runner_cls = PipelineRunner(save_files=False, verbose=1, plots_visible=display_pipeline_plots)
predictions_cls, _ = runner_cls.run(pipeline_config_cls, dataset_config_cls)

# Display classification results
print("\n" + "-" * 40)
print("Top 10 Classification Models (by Accuracy)")
print("-" * 40)
top_cls = predictions_cls.top(10, 'accuracy')
for idx, pred in enumerate(top_cls):
    print(f"{idx+1}. {Predictions.pred_short_string(pred, metrics=['accuracy', 'balanced_accuracy'])} - {pred.get('preprocessings', 'None')}")


# ============================================================================
# SECTION 3: Variable Selection with PLS (if auswahl available)
# ============================================================================

if HAS_AUSWAHL and use_all_packages:
    print("\n" + "=" * 60)
    print("SECTION 3: VARIABLE SELECTION with PLS")
    print("=" * 60)

    from auswahl import VIP, MCUVE, CARS, SPA

    # Variable selection pipeline
    varsel_pipeline = [
        MinMaxScaler(),
        {"y_processing": MinMaxScaler()},
        ShuffleSplit(n_splits=2, test_size=0.25),
    ]

    # VIP-based selection + PLS
    class VIPPLSRegressor:
        """VIP selection followed by PLS regression."""
        def __init__(self, n_components=10, n_features=50):
            self.n_components = n_components
            self.n_features = n_features
            self.vip_ = None
            self.pls_ = None
            self.selected_idx_ = None

        def fit(self, X, y):
            # Fit VIP selector
            self.vip_ = VIP(pls_kwargs={'n_components': self.n_components})
            self.vip_.fit(X, y.ravel())

            # Get VIP scores and select top features
            vip_scores = self.vip_.scores_
            self.selected_idx_ = np.argsort(vip_scores)[-self.n_features:]

            # Fit PLS on selected features
            X_sel = X[:, self.selected_idx_]
            self.pls_ = PLSRegression(n_components=min(self.n_components, X_sel.shape[1]))
            self.pls_.fit(X_sel, y)
            return self

        def predict(self, X):
            X_sel = X[:, self.selected_idx_]
            return self.pls_.predict(X_sel)

        def get_params(self, deep=True):
            return {'n_components': self.n_components, 'n_features': self.n_features}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    for n_feat in [30, 50, 100]:
        varsel_pipeline.append({
            "name": f"VIP_PLS_{n_feat}feat",
            "model": VIPPLSRegressor(n_components=10, n_features=n_feat)
        })

    print("  Added: VIP + PLS models (variable importance selection)")

    # Run variable selection pipeline
    print("\nRunning variable selection pipeline...")
    pipeline_config_var = PipelineConfigs(varsel_pipeline, "Q18_varsel")
    dataset_config_var = DatasetConfigs(data_regression)

    runner_var = PipelineRunner(save_files=False, verbose=1, plots_visible=display_pipeline_plots)
    predictions_var, _ = runner_var.run(pipeline_config_var, dataset_config_var)

    # Display results
    print("\n" + "-" * 40)
    print("Variable Selection Models (by RMSE)")
    print("-" * 40)
    top_var = predictions_var.top(5, 'rmse')
    for idx, pred in enumerate(top_var):
        print(f"{idx+1}. {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")


# ============================================================================
# SECTION 4: Visualization
# ============================================================================

if display_analyzer_plots:
    print("\n" + "=" * 60)
    print("SECTION 4: VISUALIZATION")
    print("=" * 60)

    # Regression visualization
    print("\nGenerating regression analysis plots...")
    analyzer_reg = PredictionAnalyzer(predictions_reg)

    fig1 = analyzer_reg.plot_top_k(k=5, rank_metric='rmse')
    fig1.suptitle('Top 5 Regression Models by RMSE', fontsize=14)

    fig2 = analyzer_reg.plot_candlestick(variable="model_name", display_metric='rmse')
    fig2.suptitle('Model Performance Distribution (RMSE)', fontsize=14)

    # Classification visualization
    print("Generating classification analysis plots...")
    analyzer_cls = PredictionAnalyzer(predictions_cls)

    fig3 = analyzer_cls.plot_confusion_matrix(k=4, rank_metric='accuracy')

    fig4 = analyzer_cls.plot_candlestick(variable="model_name", display_metric='accuracy')
    fig4.suptitle('Model Performance Distribution (Accuracy)', fontsize=14)

    plt.show()


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Regression models evaluated:     {len(predictions_reg)}")
print(f"Classification models evaluated: {len(predictions_cls)}")
if HAS_AUSWAHL and use_all_packages:
    print(f"Variable selection models:       {len(predictions_var)}")
print()
print("To run with all PLS packages:")
print("  1. Install packages: pip install ikpls pyopls mbpls py-ddspls auswahl trendfitter")
print("  2. Run: python Q18_pls_methods.py --all --show")
print("=" * 60)
