"""
D10 - PyTorch Models: Deep Learning with PyTorch
=================================================

Integrate PyTorch neural networks into nirs4all pipelines using
the ``@framework`` decorator and custom model classes.

This tutorial covers:

* The @framework decorator for PyTorch models
* Built-in models: nicon, decon, transformer
* Custom PyTorch model integration
* Training configuration (epochs, batch size, etc.)
* GPU acceleration

Prerequisites
-------------
- U02_basic_regression for pipeline basics
- PyTorch installation: ``pip install torch``

Next Steps
----------
See D11_jax_models for JAX/Flax integration.

Duration: ~5 minutes
Difficulty: ★★★★☆

Note: Requires PyTorch. Install with: pip install torch
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D10 PyTorch Models Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D10 - PyTorch Models: Deep Learning Integration")
print("=" * 60)

print("""
nirs4all supports PyTorch models through:

1. @framework decorator: Wraps torch.nn.Module for pipeline use
2. Built-in models: nicon, decon, transformer
3. Custom models: Define your own architectures

Key features:
  - Automatic device management (CPU/GPU)
  - Scikit-learn compatible interface
  - Training configuration via train_params
""")


# =============================================================================
# Check PyTorch Availability
# =============================================================================
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print(f"\n✓ PyTorch {torch.__version__} available")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  (Running on CPU)")
except ImportError:
    TORCH_AVAILABLE = False
    print("\n✗ PyTorch not installed. Install with: pip install torch")
    print("  Showing code examples only.")


# =============================================================================
# Section 1: Built-in NICON Model
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Built-in NICON Model")
print("-" * 60)

print("""
nirs4all provides pre-built PyTorch architectures. Use them with
the 'model' keyword and 'train_params' for configuration:

    from nirs4all.operators.models.pytorch.nicon import nicon

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

if TORCH_AVAILABLE:
    from nirs4all.operators.models.pytorch.nicon import nicon

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
        name="NICON",
        verbose=1,
        plots_visible=args.plots
    )

    print(f"\nNICON predictions: {result.num_predictions}")
    print(f"Best score: {result.best_score:.4f}")


# =============================================================================
# Section 2: The @framework Decorator
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: The @framework Decorator")
print("-" * 60)

print("""
The @framework decorator wraps a torch.nn.Module for sklearn compatibility:

    from nirs4all.utils.backend import framework

    @framework('pytorch')
    class MyModel(nn.Module):
        def __init__(self, input_shape=None, params=None, **kwargs):
            super().__init__()
            # input_shape is auto-injected from data
            # Build your architecture here
            ...

        def forward(self, x):
            return self.layers(x)
""")

if TORCH_AVAILABLE:
    from nirs4all.utils.backend import framework

    @framework('pytorch')
    class SimpleRegressor(nn.Module):
        """Simple MLP for regression."""

        def __init__(self, input_shape=None, params=None, **kwargs):
            super().__init__()
            params = params or {}
            params.update(kwargs)
            hidden_dim = params.get('hidden_dim', 64)

            if input_shape is not None:
                # input_shape is (channels, seq_len) for spectral data
                if len(input_shape) == 1:
                    input_dim = input_shape[0]
                else:
                    input_dim = input_shape[0] * input_shape[1]

                self.flatten = nn.Flatten()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
            else:
                self.flatten = None
                self.layers = None

        def forward(self, x):
            if self.layers is None:
                raise RuntimeError("Model not initialized with input_shape")
            x = self.flatten(x)
            return self.layers(x)

    print("Created SimpleRegressor with @framework('pytorch')")
    print("  - input_shape auto-injected from data")
    print("  - Sklearn fit/predict interface")
    print("  - Automatic device management")


# =============================================================================
# Section 3: Using Custom Models in Pipelines
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Using Custom Models in Pipelines")
print("-" * 60)

if TORCH_AVAILABLE:
    print("""
Custom models work like any model, using 'model' + 'train_params':
""")

    pipeline_custom = [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        SNV(),
        {
            'model': SimpleRegressor(hidden_dim=64),
            'train_params': {
                'epochs': 20,
                'batch_size': 16,
                'learning_rate': 0.001,
                'verbose': 0
            }
        },
    ]

    result = nirs4all.run(
        pipeline=pipeline_custom,
        dataset="sample_data/regression",
        name="CustomPyTorch",
        verbose=1,
        plots_visible=args.plots
    )

    print(f"\nCustom model predictions: {result.num_predictions}")
    print(f"Best score: {result.best_score:.4f}")


# =============================================================================
# Section 4: Training Configuration
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Training Configuration")
print("-" * 60)

print("""
Configure training behavior with 'train_params':

    {
        'model': nicon,
        'train_params': {
            'epochs': 100,           # Training epochs
            'batch_size': 32,        # Mini-batch size
            'learning_rate': 0.001,  # Adam learning rate
            'verbose': 1             # Training verbosity
        }
    }

Model-specific parameters go in 'model_params':

    {
        'model': nicon,
        'model_params': {
            'filters1': 16,
            'filters2': 128
        },
        'train_params': {
            'epochs': 50
        }
    }
""")


# =============================================================================
# Section 5: GPU Acceleration
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: GPU Acceleration")
print("-" * 60)

print("""
PyTorch models automatically use GPU when available.
No additional configuration is needed.
""")

if TORCH_AVAILABLE:
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n  Running on CPU (no CUDA device found)")


# =============================================================================
# Section 6: Advanced Architecture - CNN for Spectra
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Advanced Architecture - CNN for Spectra")
print("-" * 60)

print("""
1D CNNs are effective for spectral data:

    @framework('pytorch')
    class SpectralCNN(nn.Module):
        def __init__(self, input_shape=None, params=None, **kwargs):
            super().__init__()
            # Build CNN architecture
            ...

        def forward(self, x):
            x = x.unsqueeze(1)  # Add channel dim
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
""")

if TORCH_AVAILABLE:
    @framework('pytorch')
    class SpectralCNN(nn.Module):
        """1D CNN for spectral regression."""

        def __init__(self, input_shape=None, params=None, **kwargs):
            super().__init__()
            params = params or {}
            params.update(kwargs)
            n_filters = params.get('n_filters', 32)

            if input_shape is not None:
                self.conv = nn.Sequential(
                    nn.Conv1d(1, n_filters, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(16)
                )
                self.fc = nn.Linear(n_filters * 2 * 16, 1)
            else:
                self.conv = None
                self.fc = None

        def forward(self, x):
            if self.conv is None:
                raise RuntimeError("Model not initialized with input_shape")
            x = x.unsqueeze(1)  # (batch, 1, wavelengths)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    print("SpectralCNN defined with 1D convolutions")


# =============================================================================
# Section 7: Model Comparison via Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Model Comparison via Branching")
print("-" * 60)

if TORCH_AVAILABLE:
    from nirs4all.operators.models.pytorch.nicon import decon

    print("""
Compare architectures using branching:
""")

    pipeline_compare = [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        SNV(),
        {"branch": {
            "custom_mlp": [{
                'model': SimpleRegressor(hidden_dim=64),
                'train_params': {'epochs': 15, 'verbose': 0}
            }],
            "nicon": [{
                'model': nicon,
                'train_params': {'epochs': 15, 'verbose': 0}
            }],
        }},
    ]

    result = nirs4all.run(
        pipeline=pipeline_compare,
        dataset="sample_data/regression",
        name="CustomVsNICON",
        verbose=1,
        plots_visible=args.plots
    )

    print(f"\nModel comparison predictions: {result.num_predictions}")
    branches = result.predictions.get_unique_values('branch_name')
    print(f"Branches: {branches}")


# =============================================================================
# Section 8: Available Built-in Models
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Available Built-in Models")
print("-" * 60)

print("""
nirs4all provides several pre-built PyTorch architectures:

    from nirs4all.operators.models.pytorch.nicon import (
        nicon,                 # NICON architecture for NIR
        decon,                 # Depthwise convolution model
        thin_nicon,            # Smaller NICON variant
        transformer,           # Transformer-based model
        customizable_nicon,    # Configurable NICON
        customizable_decon,    # Configurable decon
    )

    # For classification tasks:
    from nirs4all.operators.models.pytorch.nicon import (
        nicon_classification,
        decon_classification,
        transformer_classification,
    )
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Built-in models: nicon, decon, transformer
2. @framework('pytorch'): Wraps nn.Module for pipeline use
3. Custom models with input_shape auto-injection
4. Training config via 'train_params' dict
5. Automatic GPU acceleration
6. 1D CNNs for spectral data
7. Model comparison via branching

Key imports:
  from nirs4all.operators.models.pytorch.nicon import nicon, decon
  from nirs4all.utils.backend import framework

Requirements:
  pip install torch

Next: D11_jax_models.py - JAX/Flax integration
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
