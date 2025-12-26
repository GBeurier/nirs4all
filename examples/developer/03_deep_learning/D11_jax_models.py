"""
D11 - JAX Models: Deep Learning with JAX/Flax
==============================================

Integrate JAX and Flax neural networks into nirs4all pipelines
for high-performance, JIT-compiled deep learning.

This tutorial covers:

* JAX/Flax model integration
* Built-in JaxMLPRegressor
* JAX nicon architecture
* JIT compilation benefits

Prerequisites
-------------
- U02_basic_regression for pipeline basics
- JAX installation: ``pip install jax jaxlib flax``

Next Steps
----------
See D12_tensorflow_models for TensorFlow/Keras integration.

Duration: ~5 minutes
Difficulty: ★★★★★

Note: Requires JAX and Flax. Install with: pip install jax jaxlib flax
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.utils.backend import JAX_AVAILABLE

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D11 JAX Models Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D11 - JAX Models: High-Performance Deep Learning")
print("=" * 60)

print("""
JAX provides:
  - JIT compilation for fast execution
  - Automatic differentiation (grad, hessian)
  - Vectorization (vmap) for batch operations
  - TPU/GPU support

Flax is the neural network library built on JAX.

nirs4all integrates JAX models with:
  - JaxMLPRegressor: Built-in MLP
  - nicon_jax: JAX implementation of nicon
""")


# =============================================================================
# Check JAX Availability
# =============================================================================
if not JAX_AVAILABLE:
    print("\n✗ JAX/Flax not installed.")
    print("  Install with: pip install jax jaxlib flax")
    print("  Showing code examples only.")
    import sys
    sys.exit(0)

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

print(f"\n✓ JAX {jax.__version__} available")
print(f"✓ Flax {flax.__version__} available")
devices = jax.devices()
print(f"  Devices: {[str(d) for d in devices]}")


# =============================================================================
# Section 1: JaxMLPRegressor - Built-in Architecture
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: JaxMLPRegressor - Built-in Architecture")
print("-" * 60)

print("""
JaxMLPRegressor is a JIT-compiled MLP. Use it with the 'model' + 'train_params' pattern:

    from nirs4all.operators.models.jax import JaxMLPRegressor

    pipeline = [
        ...,
        {
            'model': JaxMLPRegressor(features=[128, 64]),  # features = hidden layers
            'train_params': {
                'epochs': 20,
                'batch_size': 16,
                'learning_rate': 0.001,
                'verbose': 0
            }
        }
    ]
""")

from nirs4all.operators.models.jax import JaxMLPRegressor

pipeline_jax = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    SNV(),
    {
        'model': JaxMLPRegressor(features=[64, 32]),
        'train_params': {
            'epochs': 20,
            'batch_size': 16,
            'learning_rate': 0.001,
            'verbose': 0
        }
    },
]

result = nirs4all.run(
    pipeline=pipeline_jax,
    dataset="sample_data/regression",
    name="JaxMLPRegressor",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nJaxMLPRegressor predictions: {result.num_predictions}")
print(f"Best score: {result.best_score:.4f}")


# =============================================================================
# Section 2: JIT Compilation Benefits
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: JIT Compilation Benefits")
print("-" * 60)

print("""
JAX's JIT (Just-In-Time) compilation:

    @jax.jit
    def forward(params, x):
        return model.apply(params, x)

Benefits:
  - First call: compiles to XLA (takes time)
  - Subsequent calls: extremely fast
  - Automatic optimization (fusion, etc.)
  - TPU/GPU acceleration
""")

import time

# Simple benchmark showing JIT speedup
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (1000, 100))

def slow_fn(x):
    return jnp.sum(jnp.sin(x) * jnp.cos(x), axis=-1)

fast_fn = jax.jit(slow_fn)

# Warm up JIT
_ = fast_fn(x).block_until_ready()

# Time comparison
t0 = time.time()
for _ in range(100):
    _ = slow_fn(x).block_until_ready()
slow_time = time.time() - t0

t0 = time.time()
for _ in range(100):
    _ = fast_fn(x).block_until_ready()
fast_time = time.time() - t0

print(f"\nBenchmark (100 calls):")
print(f"  Without JIT: {slow_time*1000:.1f} ms")
print(f"  With JIT:    {fast_time*1000:.1f} ms")
print(f"  Speedup:     {slow_time/max(fast_time, 0.001):.1f}x")


# =============================================================================
# Section 3: nicon_jax - JAX Implementation
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: nicon_jax - JAX Implementation")
print("-" * 60)

print("""
nicon architecture implemented in JAX:

    from nirs4all.operators.models.jax.nicon import nicon as nicon_jax

    pipeline = [
        ...,
        {
            'model': nicon_jax,
            'train_params': {
                'epochs': 20,
                'batch_size': 16,
                'learning_rate': 0.001,
                'verbose': 0
            }
        }
    ]
""")

from nirs4all.operators.models.jax.nicon import nicon as nicon_jax

pipeline_nicon = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    SNV(),
    {
        'model': nicon_jax,
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
    name="niconJAX",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nnicon_jax predictions: {result.num_predictions}")
print(f"Best score: {result.best_score:.4f}")


# =============================================================================
# Section 4: Model Comparison via Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Model Comparison via Branching")
print("-" * 60)

print("""
Compare JAX architectures:
""")

pipeline_compare = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    SNV(),
    {"branch": {
        "mlp": [{
            'model': JaxMLPRegressor(features=[64, 32]),
            'train_params': {'epochs': 15, 'verbose': 0}
        }],
        "nicon": [{
            'model': nicon_jax,
            'train_params': {'epochs': 15, 'verbose': 0}
        }],
    }},
]

result = nirs4all.run(
    pipeline=pipeline_compare,
    dataset="sample_data/regression",
    name="JAXComparison",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nModel comparison predictions: {result.num_predictions}")
branches = result.predictions.get_unique_values('branch_name')
print(f"Branches: {branches}")


# =============================================================================
# Section 5: Functional Transformations
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Functional Transformations")
print("-" * 60)

print("""
JAX's functional transformations:

    # Vectorize over batch dimension
    batched_fn = jax.vmap(single_fn)

    # Compute gradients
    grad_fn = jax.grad(loss_fn)

    # Parallelize across devices
    parallel_fn = jax.pmap(fn)
""")

# Demonstrate vmap
def single_predict(x):
    return jnp.sum(x ** 2)

batch_predict = jax.vmap(single_predict)

key = jax.random.PRNGKey(42)
batch = jax.random.normal(key, (10, 50))

results = batch_predict(batch)
print(f"\nvmap example: processed batch of {batch.shape[0]} samples")
print(f"  Output shape: {results.shape}")


# =============================================================================
# Section 6: Device Management
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Device Management")
print("-" * 60)

print("""
JAX device management:

    # Check available devices
    print(jax.devices())

    # Move data to device
    x_gpu = jax.device_put(x, jax.devices('gpu')[0])
""")

print(f"\nAvailable devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. JaxMLPRegressor: Built-in JIT-compiled MLP
2. nicon_jax: NIRS-specific CNN in JAX
3. JIT compilation provides significant speedup
4. Functional transformations (vmap, grad)
5. Device management for GPU/TPU

Key imports:
  from nirs4all.operators.models.jax import JaxMLPRegressor
  from nirs4all.operators.models.jax.nicon import nicon as nicon_jax
  import jax
  import flax.linen as nn

Requirements:
  pip install jax jaxlib flax

Next: D12_tensorflow_models.py - TensorFlow/Keras integration
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
