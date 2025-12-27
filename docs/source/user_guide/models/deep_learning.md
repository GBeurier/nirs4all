# Deep Learning Models

Train neural networks using TensorFlow, PyTorch, or JAX backends.

## Overview

nirs4all provides deep learning integration through three major frameworks:
- **TensorFlow/Keras**: Production-ready, extensive ecosystem
- **PyTorch**: Research-friendly, dynamic graphs
- **JAX**: High-performance, functional paradigm

All frameworks use the same pipeline syntax, making it easy to switch backends or compare performance.

## Built-in Architectures

nirs4all includes specialized architectures designed for spectroscopic data:

| Model | Description | Best For |
|-------|-------------|----------|
| `nicon` | NIRS-optimized CNN | General NIRS regression/classification |
| `decon` | Depthwise separable convolutions | Memory-efficient models |
| `thin_nicon` | Smaller NICON variant | Limited data scenarios |
| `transformer` | Transformer architecture | Long-range dependencies |

## Quick Start

### TensorFlow Example

```python
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from nirs4all.operators.transforms import SNV
from nirs4all.operators.models.tensorflow.nicon import nicon
import nirs4all

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    SNV(),
    {
        'model': nicon,
        'train_params': {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.001,
            'verbose': 1
        }
    }
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="DeepLearning"
)
```

### PyTorch Example

```python
from nirs4all.operators.models.pytorch.nicon import nicon

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    SNV(),
    {
        'model': nicon,
        'train_params': {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.001,
            'verbose': 1
        }
    }
]
```

### JAX Example

```python
from nirs4all.operators.models.jax.nicon import nicon

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    SNV(),
    {
        'model': nicon,
        'train_params': {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.001,
            'verbose': 0
        }
    }
]
```

## Model Configuration

### train_params

Control the training process:

```python
{
    'model': nicon,
    'train_params': {
        'epochs': 100,           # Training epochs
        'batch_size': 32,        # Batch size
        'learning_rate': 0.001,  # Learning rate
        'verbose': 1,            # 0=silent, 1=progress, 2=detailed
    }
}
```

### model_params

Customize the architecture:

```python
{
    'model': nicon,
    'model_params': {
        'filters1': 16,          # First conv layer filters
        'filters2': 64,          # Second conv layer filters
        'filters3': 32,          # Third conv layer filters
        'dense_units': 64,       # Dense layer units
        'dropout_rate': 0.2,     # Dropout probability
    },
    'train_params': {
        'epochs': 100,
        'verbose': 1
    }
}
```

## Available Models

### TensorFlow/Keras

```python
from nirs4all.operators.models.tensorflow.nicon import (
    # Regression
    nicon,                 # NICON architecture
    decon,                 # Depthwise convolution
    thin_nicon,            # Smaller NICON
    customizable_nicon,    # Configurable NICON
    customizable_decon,    # Configurable decon
    transformer,           # Transformer model

    # Classification
    nicon_classification,
    decon_classification,
    customizable_nicon_classification,
)
```

### PyTorch

```python
from nirs4all.operators.models.pytorch.nicon import (
    nicon,
    decon,
    thin_nicon,
    customizable_nicon,
    customizable_decon,
    transformer,

    # Classification
    nicon_classification,
    decon_classification,
)
```

### JAX

```python
from nirs4all.operators.models.jax.nicon import (
    nicon,
    decon,
    thin_nicon,
    customizable_nicon,
    customizable_decon,
    transformer,

    # Classification
    nicon_classification,
    decon_classification,
)
```

## Architecture Details

### nicon

CNN architecture optimized for NIRS data:

```
Input (n_features,)
    │
    ├── SpatialDropout1D
    ├── Conv1D (filters=8, kernel=7)
    ├── MaxPooling1D
    ├── BatchNorm
    ├── Conv1D (filters=64, kernel=5)
    ├── MaxPooling1D
    ├── BatchNorm
    ├── Conv1D (filters=32, kernel=3)
    ├── GlobalAveragePooling1D
    ├── Dense (128)
    ├── Dropout
    └── Dense (1)  # Output
```

**Strengths**: Robust to noise, captures local patterns
**Use for**: General NIRS regression and classification

### decon

Uses depthwise separable convolutions:

```
Input (n_features,)
    │
    ├── SpatialDropout1D
    ├── DepthwiseConv1D (kernel=7)
    ├── DepthwiseConv1D (kernel=7)
    ├── MaxPooling1D
    ├── BatchNorm
    ├── DepthwiseConv1D (kernel=5)
    ├── SeparableConv1D (filters=64)
    ├── Conv1D (filters=32)
    └── Dense layers
```

**Strengths**: Fewer parameters, memory-efficient
**Use for**: Limited computational resources, transfer learning

### transformer

Attention-based architecture:

**Strengths**: Captures long-range dependencies
**Use for**: High-resolution spectra, complex patterns

## GPU Configuration

### TensorFlow

```python
import tensorflow as tf

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus}")

# Enable memory growth (recommended)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Limit GPU memory (optional)
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
)
```

### PyTorch

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### JAX

```python
import jax

# Check available devices
print(f"Devices: {jax.devices()}")

# Force CPU if needed
jax.config.update('jax_platform_name', 'cpu')
```

## Comparing Deep Learning with Traditional Models

Use branching to compare approaches:

```python
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.models.tensorflow.nicon import nicon

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    SNV(),

    {"branch": {
        "pls": [PLSRegression(n_components=10)],
        "nicon": [{
            'model': nicon,
            'train_params': {'epochs': 30, 'verbose': 0}
        }],
    }},
]

result = nirs4all.run(pipeline=pipeline, dataset="data/")

# Compare results
for pred in result.top(5):
    print(f"{pred.get('branch_name')}: RMSE={pred.get('rmse'):.4f}")
```

## Framework Comparison

| Aspect | TensorFlow | PyTorch | JAX |
|--------|------------|---------|-----|
| **Ecosystem** | Large, mature | Research-focused | High-performance |
| **Debugging** | Graph-based | Eager by default | Functional |
| **GPU support** | Excellent | Excellent | Excellent |
| **Model export** | SavedModel | TorchScript | ONNX |
| **Best for** | Production | Research | Performance |

### Cross-Framework Benchmark

```python
from nirs4all.utils.backend import TF_AVAILABLE, TORCH_AVAILABLE, JAX_AVAILABLE

results = {}

if TF_AVAILABLE:
    from nirs4all.operators.models.tensorflow.nicon import nicon as nicon_tf
    # Run TensorFlow pipeline...

if TORCH_AVAILABLE:
    from nirs4all.operators.models.pytorch.nicon import nicon as nicon_pt
    # Run PyTorch pipeline...

if JAX_AVAILABLE:
    from nirs4all.operators.models.jax.nicon import nicon as nicon_jax
    # Run JAX pipeline...
```

## Hyperparameter Tuning for Neural Networks

Use Optuna integration for automated architecture search:

```python
{
    'model': nicon,
    'finetune_params': {
        'n_trials': 50,
        'sample': 'hyperband',  # Early stopping for efficiency
        'model_params': {
            'filters1': [8, 16, 32],
            'filters2': [32, 64, 128],
            'dropout_rate': ('float', 0.1, 0.5),
        }
    },
    'train_params': {
        'epochs': 100,
        'verbose': 0
    }
}
```

## Best Practices

1. **Preprocess carefully**: Neural networks are sensitive to scale—always use `MinMaxScaler` or `StandardScaler`
2. **Start simple**: Begin with `nicon` or `decon` before custom architectures
3. **Use early stopping**: Set reasonable epochs and use validation loss monitoring
4. **Batch size matters**: Smaller batches (16-32) often work better for spectral data
5. **Learning rate**: Start with 0.001, reduce if training is unstable
6. **Cross-validate**: Always use multiple folds to assess generalization
7. **Compare with baselines**: Neural networks don't always beat PLS for NIRS

## Troubleshooting

### "TensorFlow not installed"
```bash
pip install tensorflow
# or with GPU
pip install tensorflow[and-cuda]
```

### "CUDA out of memory"
- Reduce batch size
- Enable memory growth (TensorFlow)
- Use `thin_nicon` instead of `nicon`

### "Training is unstable"
- Reduce learning rate
- Add more regularization (dropout)
- Check for data preprocessing issues

### "Model doesn't improve"
- Increase epochs
- Check if data is properly scaled
- Try different architecture

## See Also

- {doc}`training` - Basic model training
- {doc}`hyperparameter_tuning` - Automated hyperparameter optimization
- {doc}`/user_guide/preprocessing/overview` - Preprocessing for deep learning
- [D12_tensorflow_models.py](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/03_deep_learning/D12_tensorflow_models.py) - TensorFlow examples
- [D10_pytorch_models.py](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/03_deep_learning/D10_pytorch_models.py) - PyTorch examples
- [D11_jax_models.py](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/03_deep_learning/D11_jax_models.py) - JAX examples
