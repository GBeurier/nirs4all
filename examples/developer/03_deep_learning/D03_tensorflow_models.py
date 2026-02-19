"""
D03 - TensorFlow Models: Deep Learning with TensorFlow/Keras
=============================================================

Integrate TensorFlow and Keras neural networks into nirs4all
pipelines using built-in models.

This tutorial covers:

* nicon: Built-in CNN architecture for NIRS
* decon: Depthwise convolution architecture
* Model configuration via train_params

Prerequisites
-------------
- 01_quickstart/U02_basic_regression for pipeline basics
- TensorFlow installation: ``pip install tensorflow``

Next Steps
----------
See D04_framework_comparison for cross-framework benchmarking.

Duration: ~5 minutes
Difficulty: ★★★★☆

Note: Requires TensorFlow. Install with: pip install tensorflow
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.utils.backend import TF_AVAILABLE

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D03 TensorFlow Models Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D03 - TensorFlow Models: Keras Integration")
print("=" * 60)

print("""
nirs4all provides TensorFlow/Keras integration through:

1. nicon: CNN architecture for NIRS
2. decon: Depthwise convolution architecture
3. customizable_nicon: Configurable version

These models use the Keras API and are designed
specifically for spectroscopic data.
""")

# =============================================================================
# Check TensorFlow Availability
# =============================================================================
if not TF_AVAILABLE:
    print("\n✗ TensorFlow not installed. Install with: pip install tensorflow")
    print("  Exiting.")
    import sys
    sys.exit(0)

import tensorflow as tf

print(f"\n✓ TensorFlow {tf.__version__} available")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU available: {gpus[0].name}")
else:
    print("  (Running on CPU)")

# =============================================================================
# Section 1: nicon - Built-in NIRS Architecture
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: nicon - Built-in NIRS Architecture")
print("-" * 60)

print("""
nicon is a specialized CNN for NIRS data. Use it with 'model' + 'train_params':

    from nirs4all.operators.models.tensorflow.nicon import nicon

    pipeline = [
        ...,
        {
            'model': nicon,
            'train_params': {
                'epochs': 30,
                'batch_size': 16,
                'learning_rate': 0.001,
                'verbose': 0
            }
        }
    ]
""")

from nirs4all.operators.models.tensorflow.nicon import nicon

pipeline_nicon = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    SNV(),
    {
        'model': nicon,
        'train_params': {
            'epochs': 20,
            'batch_size': 16,
            'learning_rate': 0.001,
            'verbose': 0
        }
    },
]

result = nirs4all.run(
    pipeline=pipeline_nicon,
    dataset="sample_data/regression",
    name="NiconModel",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nnicon predictions: {result.num_predictions}")
print(f"Best score: {result.best_score:.4f}")

# =============================================================================
# Section 2: decon - Depthwise Convolution Architecture
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: decon - Depthwise Convolution Architecture")
print("-" * 60)

print("""
decon uses depthwise separable convolutions:

    from nirs4all.operators.models.tensorflow.nicon import decon

    pipeline = [
        ...,
        {
            'model': decon,
            'train_params': {
                'epochs': 20,
                'verbose': 0
            }
        }
    ]
""")

from nirs4all.operators.models.tensorflow.nicon import decon

pipeline_decon = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    SNV(),
    {
        'model': decon,
        'train_params': {
            'epochs': 20,
            'batch_size': 16,
            'learning_rate': 0.001,
            'verbose': 0
        }
    },
]

result = nirs4all.run(
    pipeline=pipeline_decon,
    dataset="sample_data/regression",
    name="DeconModel",
    verbose=1,
    plots_visible=args.plots
)

print(f"\ndecon predictions: {result.num_predictions}")
print(f"Best score: {result.best_score:.4f}")

# =============================================================================
# Section 3: Model Comparison - nicon vs PLS
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Model Comparison - nicon vs PLS")
print("-" * 60)

print("""
Compare deep learning with traditional methods via branching:
""")

pipeline_compare = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    SNV(),
    {"branch": {
        "pls": [PLSRegression(n_components=10)],
        "nicon": [{
            'model': nicon,
            'train_params': {'epochs': 15, 'verbose': 0}
        }],
    }},
]

result = nirs4all.run(
    pipeline=pipeline_compare,
    dataset="sample_data/regression",
    name="PLSvsNicon",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nPLS vs nicon comparison: {result.num_predictions} predictions")
print(f"Branches: {result.predictions.get_unique_values('branch_name')}")

# =============================================================================
# Section 4: Model Configuration
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Model Configuration")
print("-" * 60)

print("""
Configure models via model_params and train_params:

    {
        'model': nicon,
        'model_params': {
            'filters1': 8,          # First conv filters
            'filters2': 64,         # Second conv filters
            'filters3': 32,         # Third conv filters
            'dense_units': 16,      # Dense layer units
            'dropout_rate': 0.2,    # Dropout probability
        },
        'train_params': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'verbose': 1
        }
    }
""")

# =============================================================================
# Section 5: Available TensorFlow Models
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Available TensorFlow Models")
print("-" * 60)

print("""
nirs4all provides several pre-built TensorFlow architectures:

    from nirs4all.operators.models.tensorflow.nicon import (
        nicon,                 # NICON architecture for NIR
        decon,                 # Depthwise convolution model
        thin_nicon,            # Smaller NICON variant
        customizable_nicon,    # Configurable NICON
        customizable_decon,    # Configurable decon
    )

    # For classification tasks:
    from nirs4all.operators.models.tensorflow.nicon import (
        nicon_classification,
        decon_classification,
    )
""")

# =============================================================================
# Section 6: GPU Memory Management
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: GPU Memory Management")
print("-" * 60)

print("""
TensorFlow GPU memory configuration:

    import tensorflow as tf

    # Allow memory growth (recommended)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
""")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nGPU devices: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for all GPUs")
    except RuntimeError:
        print("(Memory growth must be set before GPU initialization)")
else:
    print("\nNo GPU devices found (using CPU)")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. nicon: CNN architecture for NIRS
2. decon: Depthwise convolution architecture
3. Model comparison via branching
4. Model configuration via model_params and train_params
5. GPU memory management

Key imports:
  from nirs4all.operators.models.tensorflow.nicon import nicon, decon

Requirements:
  pip install tensorflow

Next: D04_framework_comparison.py - Cross-framework benchmarking
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
