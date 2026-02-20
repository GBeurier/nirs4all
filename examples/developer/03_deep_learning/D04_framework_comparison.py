"""
D04 - Framework Comparison: TensorFlow vs PyTorch vs JAX
=========================================================

Compare deep learning frameworks in nirs4all for performance,
accuracy, and use case suitability.

This tutorial covers:

* Same architecture across frameworks
* Performance benchmarking
* Framework selection guidelines

Prerequisites
-------------
- D01_pytorch_models, D02_jax_models, D03_tensorflow_models

Next Steps
----------
See 04_transfer_learning/D01_transfer_basics for calibration transfer.

Duration: ~5 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse
import time

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.utils.backend import JAX_AVAILABLE, TF_AVAILABLE, TORCH_AVAILABLE

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D04 Framework Comparison Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D04 - Framework Comparison: TF vs PyTorch vs JAX")
print("=" * 60)

print("""
nirs4all supports three deep learning backends:

  TensorFlow/Keras: Production-ready, extensive ecosystem
  PyTorch:          Research-friendly, dynamic graphs
  JAX/Flax:         High-performance, functional style

Let's compare them on the same task.
""")

# =============================================================================
# Check Framework Availability
# =============================================================================
print("\n" + "-" * 60)
print("Framework Availability Check")
print("-" * 60)

if TF_AVAILABLE:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
else:
    print("✗ TensorFlow not available")

if TORCH_AVAILABLE:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
else:
    print("✗ PyTorch not available")

if JAX_AVAILABLE:
    import jax
    print(f"✓ JAX {jax.__version__}")
else:
    print("✗ JAX not available")

# =============================================================================
# Section 1: nicon Across Frameworks
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: nicon Architecture Across Frameworks")
print("-" * 60)

print("""
nirs4all provides nicon in all three frameworks:

    from nirs4all.operators.models.tensorflow.nicon import nicon  # TensorFlow
    from nirs4all.operators.models.pytorch.nicon import nicon     # PyTorch
    from nirs4all.operators.models.jax.nicon import nicon         # JAX

Same architecture, different implementations.
""")

results = {}
timings = {}

# Baseline: PLS
print("\n📊 Running PLS baseline...")
t0 = time.time()
pipeline_pls = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    SNV(),
    PLSRegression(n_components=10),
]
result_pls = nirs4all.run(
    pipeline=pipeline_pls,
    dataset="sample_data/regression",
    name="PLS_Baseline",
    verbose=0,
    plots_visible=False
)
timings['pls'] = time.time() - t0
results['pls'] = result_pls.best_score
print(f"  PLS: Score={results['pls']:.4f}, Time={timings['pls']:.1f}s")

# =============================================================================
# Section 2: TensorFlow nicon
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: TensorFlow nicon")
print("-" * 60)

if TF_AVAILABLE:
    from nirs4all.operators.models.tensorflow.nicon import nicon as nicon_tf

    print("Running TensorFlow nicon...")
    t0 = time.time()
    pipeline_tf = [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        SNV(),
        {
            'model': nicon_tf,
            'train_params': {'epochs': 15, 'verbose': 0}
        },
    ]
    result_tf = nirs4all.run(
        pipeline=pipeline_tf,
        dataset="sample_data/regression",
        name="nicon_TensorFlow",
        verbose=0,
        plots_visible=False
    )
    timings['tensorflow'] = time.time() - t0
    results['tensorflow'] = result_tf.best_score
    print(f"  TensorFlow: Score={results['tensorflow']:.4f}, Time={timings['tensorflow']:.1f}s")
else:
    print("  TensorFlow not available, skipping")

# =============================================================================
# Section 3: PyTorch nicon
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: PyTorch nicon")
print("-" * 60)

if TORCH_AVAILABLE:
    from nirs4all.operators.models.pytorch.nicon import nicon as nicon_pt

    print("Running PyTorch nicon...")
    t0 = time.time()
    pipeline_pt = [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        SNV(),
        {
            'model': nicon_pt,
            'train_params': {'epochs': 15, 'verbose': 0}
        },
    ]
    result_pt = nirs4all.run(
        pipeline=pipeline_pt,
        dataset="sample_data/regression",
        name="nicon_PyTorch",
        verbose=0,
        plots_visible=False
    )
    timings['pytorch'] = time.time() - t0
    results['pytorch'] = result_pt.best_score
    print(f"  PyTorch: Score={results['pytorch']:.4f}, Time={timings['pytorch']:.1f}s")
else:
    print("  PyTorch not available, skipping")

# =============================================================================
# Section 4: JAX nicon
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: JAX nicon")
print("-" * 60)

if JAX_AVAILABLE:
    from nirs4all.operators.models.jax.nicon import nicon as nicon_jax

    print("Running JAX nicon...")
    t0 = time.time()
    pipeline_jax = [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        SNV(),
        {
            'model': nicon_jax,
            'train_params': {'epochs': 15, 'verbose': 0}
        },
    ]
    result_jax = nirs4all.run(
        pipeline=pipeline_jax,
        dataset="sample_data/regression",
        name="nicon_JAX",
        verbose=0,
        plots_visible=False
    )
    timings['jax'] = time.time() - t0
    results['jax'] = result_jax.best_score
    print(f"  JAX: Score={results['jax']:.4f}, Time={timings['jax']:.1f}s")
else:
    print("  JAX not available, skipping")

# =============================================================================
# Section 5: Results Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Results Comparison")
print("-" * 60)

print("\n📊 Performance Summary (lower score = better MSE):")
print("-" * 40)
print(f"{'Framework':<15} {'Score':<10} {'Time (s)':<10}")
print("-" * 40)
for fw in ['pls', 'tensorflow', 'pytorch', 'jax']:
    if fw in results:
        print(f"{fw:<15} {results[fw]:<10.4f} {timings[fw]:<10.1f}")
print("-" * 40)

# Find best
if results:
    best_fw = min(results, key=lambda k: results[k])
    print(f"\n🏆 Best Score: {best_fw} ({results[best_fw]:.4f})")

    fastest_fw = min(timings, key=lambda k: timings[k])
    print(f"⚡ Fastest: {fastest_fw} ({timings[fastest_fw]:.1f}s)")

# =============================================================================
# Section 6: Framework Selection Guidelines
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Framework Selection Guidelines")
print("-" * 60)

print("""
📋 Framework Selection Guide:

┌────────────────┬─────────────────────────────────────────┐
│ TensorFlow     │ • Production deployment                 │
│                │ • TensorFlow Serving / TF Lite          │
│                │ • Large pre-trained model ecosystem     │
│                │ • Keras high-level API                  │
├────────────────┼─────────────────────────────────────────┤
│ PyTorch        │ • Research and prototyping              │
│                │ • Dynamic computation graphs            │
│                │ • Easier debugging (Python-native)      │
│                │ • Strong academic community             │
├────────────────┼─────────────────────────────────────────┤
│ JAX            │ • Maximum performance (JIT)             │
│                │ • TPU access                            │
│                │ • Functional programming style          │
│                │ • Custom research operators             │
└────────────────┴─────────────────────────────────────────┘
""")

# =============================================================================
# Section 7: Practical Recommendations
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Practical Recommendations")
print("-" * 60)

print("""
🎯 For NIRS applications:

1. Start with PLS
   - Fast, interpretable, often competitive
   - Use as baseline for all comparisons

2. Small datasets (< 500 samples)
   - PLS or Ridge usually best
   - Deep learning may overfit

3. Medium datasets (500-5000 samples)
   - nicon can outperform PLS
   - Use early stopping and dropout

4. Large datasets (> 5000 samples)
   - Deep learning shines
   - Consider ensemble of architectures

5. Production deployment
   - TensorFlow for serving
   - Export trained models (.h5, SavedModel)

6. Research / experimentation
   - PyTorch for flexibility
   - JAX for performance-critical work
""")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Same nicon architecture across TF/PyTorch/JAX
2. Performance varies by framework and hardware
3. PLS remains strong baseline for NIRS
4. Framework choice depends on use case
5. Deep learning needs larger datasets

Key imports:
  from nirs4all.operators.models.tensorflow.nicon import nicon  # TensorFlow
  from nirs4all.operators.models.pytorch.nicon import nicon     # PyTorch
  from nirs4all.operators.models.jax.nicon import nicon         # JAX

Next: 04_transfer_learning/D01_transfer_basics.py - Calibration transfer
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
