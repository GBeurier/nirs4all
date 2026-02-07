# Writing a Pipeline in nirs4all

This guide explains all possible syntaxes for defining pipeline steps in nirs4all, from simple operators to complex model configurations with hyperparameter optimization.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline vs Pipeline Generator](#pipeline-vs-pipeline-generator)
3. [Basic Step Syntaxes](#basic-step-syntaxes)
4. [Model Step Syntaxes](#model-step-syntaxes)
5. [Generator Syntaxes](#generator-syntaxes)
6. [File Formats](#file-formats)
7. [Syntax Reference Table](#syntax-reference-table)
8. [Serialization Rules](#serialization-rules)

---

## Overview

A **pipeline** in nirs4all is a list of processing steps that transform data and train/apply models. Each step can be:
- A **transformer** (preprocessing, feature engineering)
- A **cross-validator** (data splitting strategy)
- A **model** (for training or prediction)
- A **visualization** (charts, reports)
- A **special operator** (resampling, augmentation)

Pipelines are defined in Python as lists, and can be saved/loaded from JSON or YAML files.

### Philosophy

nirs4all accepts **multiple syntaxes** for maximum flexibility:
- **Python objects** (class, instance, function)
- **String references** (module paths, file paths, controller names)
- **Dictionaries** (explicit configuration with parameters)

### Core Principle: Hash-Based Uniqueness

**The fundamental rule**: Different syntaxes that produce the **same object** must serialize to the **same canonical form**, resulting in the **same hash**.

**Example - All these are equivalent**:
```python
# Syntax 1: Class
StandardScaler

# Syntax 2: Instance with defaults
StandardScaler()

# Syntax 3: Instance with explicit default value
MinMaxScaler(feature_range=(0, 1))  # (0, 1) is the default

# Syntax 4: String path
"sklearn.preprocessing.StandardScaler"

# Syntax 5: Dict
{"class": "sklearn.preprocessing.StandardScaler"}
```

**All serialize to**:
```json
"sklearn.preprocessing._data.StandardScaler"
```

**Result**: Same hash → Recognized as identical pipelines → Proper deduplication.

**Counter-example - These are different**:
```python
# Different class
StandardScaler  # vs  MinMaxScaler

# Same class, different non-default params
MinMaxScaler(feature_range=(0, 1))  # default
# vs
MinMaxScaler(feature_range=(0, 2))  # non-default
```

These produce **different serializations** and **different hashes**.

**All syntaxes are normalized** during serialization to a canonical form for hash-based uniqueness:

```python
{
    "class": "module.path.ClassName",
    "params": {"param1": "value1"}
}
```

Or simply (when all parameters are default):

```json
"module.path.ClassName"
```

**Critical principle**: **Same object = Same serialization = Same hash**

This ensures:
- ✅ **Hash-based uniqueness**: Identical configurations produce identical hashes (regardless of input syntax)
- ✅ **Minimalism**: Only non-default parameters are included
- ✅ **Deduplication**: Pipeline variations are properly detected and merged
- ✅ **Reproducibility**: Exact pipeline state can be restored

**Example**: All these syntaxes produce the **same serialization**:
```python
StandardScaler                                  # Class
StandardScaler()                                # Instance with defaults
MinMaxScaler(feature_range=(0, 1))             # Instance with default value
"sklearn.preprocessing.StandardScaler"          # String path
{"class": "sklearn.preprocessing.StandardScaler"}  # Dict
```

All serialize to:
```json
"sklearn.preprocessing._data.StandardScaler"
```

Because they all create the **same object** with **default parameters**.

---

## Pipeline vs Pipeline Generator

### Pipeline Definition

A **pipeline definition** is a concrete list of steps that will be executed sequentially:

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, test_size=0.2),
    PLSRegression(n_components=10)
]
```

This creates **one pipeline** that will be run **once**.

### Pipeline Generator Definition

A **pipeline generator** uses special syntax (`_or_`, `_range_`) to define **multiple pipeline variations**:

```python
pipeline_generator = [
    MinMaxScaler(),
    {"_or_": [Detrend, FirstDerivative, Gaussian]},  # Creates 3 variations
    PLSRegression(n_components=10)
]
```

This creates **three pipelines**:
1. `MinMaxScaler() → Detrend → PLSRegression(...)`
2. `MinMaxScaler() → FirstDerivative → PLSRegression(...)`
3. `MinMaxScaler() → Gaussian → PLSRegression(...)`

**Generator keys**:
- `_or_`: Choose between alternatives (creates N pipelines)
- `_range_`: Sweep parameter values (creates M pipelines)
- `size`: Limit combinations (for feature augmentation)
- `count`: Randomly sample N configurations

Generators are **expanded** by `PipelineConfigs` before execution, producing multiple concrete pipelines.

---

## Basic Step Syntaxes

### 1. Class Reference (Uninstantiated)

**Syntax**: Pass a Python class directly.

```python
from sklearn.preprocessing import StandardScaler

pipeline = [
    StandardScaler  # Class, not instance
]
```

**Behavior**: nirs4all instantiates with **default parameters**.

**Serializes to** (class path only, no params dict):
```json
"sklearn.preprocessing._data.StandardScaler"
```

**Hash**: Based on class path only (all defaults).

---

### 2. Instance with Parameters

**Syntax**: Pass an instantiated object.

```python
from sklearn.preprocessing import MinMaxScaler

pipeline = [
    MinMaxScaler(feature_range=(0, 1))  # Default value
]
```

**Serializes to** (no params if all are defaults):
```json
"sklearn.preprocessing._data.MinMaxScaler"
```

**Example with non-default parameter**:
```python
MinMaxScaler(feature_range=(0, 2))  # Non-default
```

**Serializes to**:
```json
{
    "class": "sklearn.preprocessing._data.MinMaxScaler",
    "params": {
        "feature_range": [0, 2]
    }
}
```

**Note**: Only **non-default** parameters are saved (via `_changed_kwargs()`). This ensures that `MinMaxScaler()`, `MinMaxScaler(feature_range=(0, 1))`, and the class `MinMaxScaler` all produce the **same serialization and hash**.

---

### 3. String - Module Path

**Syntax**: Full module path to a class.

```python
pipeline = [
    "sklearn.preprocessing.StandardScaler"
]
```

**Behavior**: Same as class reference - instantiated with defaults.

**Serializes to** (identical to class reference):
```json
"sklearn.preprocessing._data.StandardScaler"
```

**Hash**: Same as using the class directly or an instance with default params.

*(Note: Internal module path may differ from public API path)*

---

### 4. String - Controller Name

**Syntax**: Short name for built-in nirs4all controllers.

```python
pipeline = [
    "chart_2d"  # Built-in visualization
]
```

**Behavior**: Resolves to a registered controller (e.g., `ChartController2D`).

**Serializes to**:
```json
"chart_2d"
```

---

### 5. String - File Path (Saved Transformer)

**Syntax**: Path to a saved transformer file (`.pkl`, `.joblib`).

```python
pipeline = [
    "my/super/transformer.pkl"
]
```

**Behavior**: Loads the transformer from disk during execution.

**Serializes to**:
```json
"my/super/transformer.pkl"
```

---

### 6. Dictionary - Explicit Configuration

**Syntax**: Dict with `class` and optional `params` keys.

```python
pipeline = [
    {
        "class": "sklearn.preprocessing.StandardScaler"
    }
]
```

**Serializes to** (identical to class reference):
```json
"sklearn.preprocessing._data.StandardScaler"
```

**With non-default parameters**:
```python
pipeline = [
    {
        "class": "sklearn.model_selection.ShuffleSplit",
        "params": {
            "n_splits": 3,
            "test_size": 0.25
        }
    }
]
```

**Serializes to** (same as input, with normalized class path):
```json
{
    "class": "sklearn.model_selection._split.ShuffleSplit",
    "params": {
        "n_splits": 3,
        "test_size": 0.25
    }
}
```

**Hash**: Based on class path + non-default params only.

---

### 7. Dictionary - Special Operators

**Syntax**: Dict with operator-specific keys (e.g., `y_processing`, `feature_augmentation`).

```python
pipeline = [
    {"y_processing": MinMaxScaler},  # Target variable scaling
    {"feature_augmentation": Detrend}  # Feature engineering
]
```

**Serializes to**:
```json
{
    "y_processing": {
        "class": "sklearn.preprocessing._data.MinMaxScaler"
    }
}
```

**Note**: Class wrapped in dict with `class` key during preprocessing (`_preprocess_steps()`).

---

## Model Step Syntaxes

Models have additional complexity due to:
- Custom naming
- Training parameters
- Hyperparameter optimization (finetuning)
- Support for functions (not just classes)

### 1. Model - Instance

**Syntax**: Pass a model instance directly.

```python
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    PLSRegression(n_components=10)
]
```

**Serializes to**:
```json
{
    "class": "sklearn.cross_decomposition._pls.PLSRegression",
    "params": {
        "n_components": 10
    }
}
```

**Note**: If all params are default, serializes to just the class string:
```python
PLSRegression()  # All defaults
```

Serializes to:
```json
"sklearn.cross_decomposition._pls.PLSRegression"
```

**Hash**: Based on class + non-default params.

---

### 2. Model - Dict with Name

**Syntax**: Dict with `model` and `name` keys.

```python
pipeline = [
    {
        "model": PLSRegression(n_components=10),
        "name": "PLS_10_components"
    }
]
```

**Serializes to**:
```json
{
    "name": "PLS_10_components",
    "model": {
        "class": "sklearn.cross_decomposition._pls.PLSRegression",
        "params": {
            "n_components": 10
        }
    }
}
```

**Purpose**: Custom naming for tracking specific models in results.

**Hash behavior**: The `name` field **affects the hash** because it's part of the step configuration. This means:
```python
# These produce DIFFERENT hashes:
{"model": PLSRegression(n_components=10), "name": "Model_A"}
{"model": PLSRegression(n_components=10), "name": "Model_B"}
```

Even though the model is identical, different names create different pipeline variants (useful for comparing the same model with different training strategies).

---

### 3. Model - Dict with Class and Params

**Syntax**: Nested dict structure.

```python
pipeline = [
    {
        "model": {
            "class": "sklearn.cross_decomposition.PLSRegression",
            "params": {
                "n_components": 10
            }
        }
    }
]
```

**Serializes to**:
```json
{
    "model": {
        "class": "sklearn.cross_decomposition._pls.PLSRegression",
        "params": {
            "n_components": 10
        }
    }
}
```

**With defaults only**:
```python
pipeline = [
    {
        "model": {
            "class": "sklearn.cross_decomposition.PLSRegression"
        }
    }
]
```

Serializes to:
```json
{
    "model": "sklearn.cross_decomposition._pls.PLSRegression"
}
```

**Hash**: Based on model class + non-default params.

---

### 4. Model - Function (Neural Network)

**Syntax**: Pass a function that returns a model.

```python
from nirs4all.operators.models.tensorflow.nicon import nicon

pipeline = [
    nicon  # Function that builds a TensorFlow model
]
```

**Serializes to**:
```json
{
    "function": "nirs4all.operators.models.cirad_tf.nicon"
}
```

---

### 5. Model - String File Path

**Syntax**: Path to saved model file.

```python
pipeline = [
    "My_awesome_model.pkl",           # Scikit-learn model
    "My_awesome_tf_model.keras",      # TensorFlow/Keras model
    "my_model.pth"                    # PyTorch model
]
```

**Behavior**: Loads model from disk (framework detected by extension).

**Serializes to**: (preserves file path)

---

### 6. Model - With Training Parameters

**Syntax**: Dict with `model` and `train_params` keys (for neural networks).

```python
from nirs4all.operators.models.tensorflow.nicon import nicon

pipeline = [
    {
        "model": nicon,
        "train_params": {
            "epochs": 250,
            "batch_size": 32,
            "verbose": 0
        }
    }
]
```

**Serializes to**:
```json
{
    "model": {
        "function": "nirs4all.operators.models.cirad_tf.nicon"
    },
    "train_params": {
        "epochs": 250,
        "batch_size": 32,
        "verbose": 0
    }
}
```

---

### 6b. Model - With Architecture Parameters (Customizable NN)

**Syntax**: Dict with `model`, `model_params`, and optional `train_params` keys.

Use `model_params` to customize neural network architecture (filters, kernel sizes, etc.) at training time without finetuning.

```python
from nirs4all.operators.models.pytorch.nicon import customizable_nicon

pipeline = [
    {
        "model": customizable_nicon,
        "name": "CustomNN",
        "model_params": {           # Architecture parameters
            "filters1": 32,
            "filters2": 64,
            "kernel_size1": 9,
            "dropout_rate": 0.3
        },
        "train_params": {           # Training loop parameters
            "epochs": 250,
            "batch_size": 32,
            "lr": 0.001
        }
    }
]
```

**Key distinction**:
- `model_params`: Parameters passed to the **model builder function** (architecture)
- `train_params`: Parameters for the **training loop** (epochs, batch size, optimizer settings)

**Serializes to**:
```json
{
    "name": "CustomNN",
    "model": {
        "function": "nirs4all.operators.models.pytorch.nicon.customizable_nicon"
    },
    "model_params": {
        "filters1": 32,
        "filters2": 64,
        "kernel_size1": 9,
        "dropout_rate": 0.3
    },
    "train_params": {
        "epochs": 250,
        "batch_size": 32,
        "lr": 0.001
    }
}
```

**💡 Tip**: This is useful when you know the optimal architecture and want to train without finetuning overhead.

---

### 7. Model - With Hyperparameter Optimization (Finetuning)

**Syntax**: Dict with `model`, optional `name`, and `finetune_params` keys.

```python
pipeline = [
    {
        "model": PLSRegression(),
        "name": "PLS-Finetuned",
        "finetune_params": {
            "n_trials": 20,
            "verbose": 2,
            "approach": "single",
            "eval_mode": "best",
            "sample": "grid",
            "model_params": {
                'n_components': ('int', 1, 30),  # Tuple: (type, min, max)
            }
        }
    }
]
```

**Finetuning parameters**:
- `n_trials`: Number of optimization trials
- `verbose`: 0=silent, 1=basic, 2=detailed
- `approach`: "single", "grouped", or "individual"
- `eval_mode`: "best" or "avg" (for grouped approach)
- `sample`: Optimizer strategy ("random", "grid", "bayes", "tpe", "hyperband", etc.)
- `model_params`: Dict of parameters to optimize
  - Value format: `(type, min, max)` for ranges
  - Or list `[value1, value2, ...]` for categorical

**Serializes to**:
```json
{
    "name": "PLS-Finetuned",
    "model": {
        "class": "sklearn.cross_decomposition._pls.PLSRegression"
    },
    "finetune_params": {
        "n_trials": 20,
        "verbose": 2,
        "approach": "single",
        "eval_mode": "best",
        "sample": "grid",
        "model_params": {
            "n_components": ["int", 1, 30]
        }
    }
}
```

**⚠️ Important**: Tuples like `('int', 1, 30)` are converted to **lists** `["int", 1, 30]` during serialization for YAML/JSON compatibility.

---

### 8. Model - Neural Network with Finetuning

**Syntax**: Combine function models with hyperparameter optimization.

```python
from nirs4all.operators.models.tensorflow.nicon import customizable_nicon

pipeline = [
    {
        "model": customizable_nicon,
        "name": "NN-Optimized",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,
            "sample": "hyperband",
            "approach": "single",
            "model_params": {
                "filters_1": [8, 16, 32, 64],      # Categorical choices
                "filters_3": [8, 16, 32, 64]
            },
            "train_params": {
                "epochs": 10,
                "verbose": 0
            }
        },
        "train_params": {
            "epochs": 250,  # Final training after optimization
            "verbose": 0
        }
    }
]
```

**Two-stage training**:
1. `finetune_params.train_params`: Used during **hyperparameter search** (fewer epochs)
2. `train_params`: Used for **final training** with best parameters (more epochs)

---

### 9. Model - Custom Code File

**Syntax**: Dict with `source_file` and `class` keys.

```python
pipeline = [
    {
        "source_file": "my_model.py",
        "class": "MyAwesomeModel"
    }
]
```

**Behavior**: Dynamically imports `MyAwesomeModel` from `my_model.py`.

**Serializes to**: (preserves source file and class name)

---

## Generator Syntaxes

Generators allow automatic creation of **multiple pipeline variations** for experimentation.

### 1. `_or_` - Alternative Choices

**Syntax**: Dict with `_or_` key containing a list of choices.

```python
preprocessing_options = [
    Detrend, FirstDerivative, Gaussian, StandardNormalVariate
]

pipeline = [
    {"_or_": preprocessing_options}  # Creates 4 pipelines (one per choice)
]
```

**Expands to**: 4 separate pipelines, each using one preprocessing method.

---

### 2. `_or_` with `count` - Random Sampling

**Syntax**: Add `count` key to limit number of generated pipelines.

```python
pipeline = [
    {"_or_": preprocessing_options, "count": 2}  # Randomly select 2
]
```

**Expands to**: 2 pipelines (randomly sampled from 4 options).

---

### 3. `_or_` with `size` - Combinations

**Syntax**: Add `size` key to select N items at once (creates combinations).

```python
pipeline = [
    {"_or_": preprocessing_options, "size": 2}  # Choose 2 at a time
]
```

**Expands to**: All combinations of 2 items from 4 options = 6 pipelines:
- `[Detrend, FirstDerivative]`
- `[Detrend, Gaussian]`
- `[Detrend, StandardNormalVariate]`
- `[FirstDerivative, Gaussian]`
- `[FirstDerivative, StandardNormalVariate]`
- `[Gaussian, StandardNormalVariate]`

---

### 4. `_or_` with `size` Range - Variable Size

**Syntax**: Use tuple `(from, to)` for size range.

```python
pipeline = [
    {"_or_": preprocessing_options, "size": (1, 2)}  # 1 or 2 items
]
```

**Expands to**: All combinations of size 1 + all combinations of size 2 = 4 + 6 = 10 pipelines.

---

### 5. `_or_` with `size` and `count` - Limited Combinations

**Syntax**: Combine `size` and `count` to randomly sample from combinations.

```python
pipeline = [
    {"feature_augmentation": {
        "_or_": preprocessing_options,
        "size": (1, 2),
        "count": 5  # Randomly pick 5 combinations
    }}
]
```

**Expands to**: 5 randomly sampled pipelines from all possible 1-2 item combinations.

---

### 6. `_or_` with Nested Arrays - Second-Order Combinations

**Syntax**: Use list `[outer_size, inner_size]` for nested combinations (sub-pipelines).

```python
pipeline = [
    {"feature_augmentation": {
        "_or_": preprocessing_options,
        "size": [2, (1, 2)]  # 2 sub-pipelines, each with 1-2 items
    }}
]
```

**Behavior**:
1. Creates all **inner arrangements** (permutations of 1-2 items)
2. Selects **outer combinations** (choose 2 sub-pipelines)

**Example expansion**:
```python
[
    [[Detrend], [FirstDerivative]],
    [[Detrend], [Gaussian]],
    [[FirstDerivative, Detrend], [Gaussian, StandardNormalVariate]],
    ...
]
```

**Note**: Inner uses **permutations** (order matters), outer uses **combinations** (order doesn't matter).

---

### 7. `_range_` - Numeric Parameter Sweep

**Syntax**: Dict with `_range_` key and model configuration.

```python
pipeline = [
    {
        "_range_": [1, 12, 2],  # Start, end, step
        "param": "n_components",
        "model": {
            "class": "sklearn.cross_decomposition.PLSRegression"
        }
    }
]
```

**Expands to**: 6 pipelines with `n_components` = 1, 3, 5, 7, 9, 11.

**Alternative syntax** (dict):
```python
{
    "_range_": {"from": 1, "to": 12, "step": 2},
    "param": "n_components",
    "model": PLSRegression
}
```

---

### 8. `_range_` with `count` - Sampled Range

**Syntax**: Add `count` key to randomly sample from range.

```python
pipeline = [
    {
        "_range_": [1, 30],  # 30 values
        "count": 10,         # Sample 10 randomly
        "param": "n_components",
        "model": PLSRegression
    }
]
```

**Expands to**: 10 pipelines with randomly selected `n_components` values from 1-30.

---

## File Formats

Pipelines can be defined in Python, JSON, or YAML.

### Python Format

```python
from nirs4all.pipeline import PipelineConfigs

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5),
    PLSRegression(n_components=10)
]

config = PipelineConfigs(pipeline, name="my_pipeline")
```

---

### JSON Format

**File**: `pipeline.json`

```json
{
    "pipeline": [
        {
            "class": "sklearn.preprocessing._data.MinMaxScaler"
        },
        {
            "class": "sklearn.model_selection._split.ShuffleSplit",
            "params": {
                "n_splits": 5
            }
        },
        {
            "class": "sklearn.cross_decomposition._pls.PLSRegression",
            "params": {
                "n_components": 10
            }
        }
    ]
}
```

**Load**:
```python
config = PipelineConfigs("pipeline.json", name="my_pipeline")
```

---

### YAML Format

**File**: `pipeline.yaml`

```yaml
pipeline:
  - class: sklearn.preprocessing._data.MinMaxScaler

  - class: sklearn.model_selection._split.ShuffleSplit
    params:
      n_splits: 5

  - class: sklearn.cross_decomposition._pls.PLSRegression
    params:
      n_components: 10
```

**Load**:
```python
config = PipelineConfigs("pipeline.yaml", name="my_pipeline")
```

---

### Generator in YAML

```yaml
pipeline:
  - class: sklearn.preprocessing._data.MinMaxScaler

  - feature_augmentation:
      _or_:
        - class: nirs4all.operators.transforms.Detrend
        - class: nirs4all.operators.transforms.FirstDerivative
        - class: nirs4all.operators.transforms.Gaussian
      size: [1, 2]
      count: 5

  - class: sklearn.model_selection._split.ShuffleSplit
    params:
      n_splits: 5

  - _range_: [1, 12, 2]
    param: n_components
    model:
      class: sklearn.cross_decomposition._pls.PLSRegression
```

**Note**: Tuples in Python become **lists** in JSON/YAML:
- Python: `('int', 1, 30)` → YAML: `["int", 1, 30]`

---

## Syntax Reference Table

| Syntax | Example | Use Case | Serializes To |
|--------|---------|----------|---------------|
| **Class** | `StandardScaler` | Default params | `"sklearn.preprocessing._data.StandardScaler"` |
| **Instance (defaults)** | `StandardScaler()` | Default params | `"sklearn.preprocessing._data.StandardScaler"` (same as class) |
| **Instance (default values)** | `MinMaxScaler(feature_range=(0,1))` | Explicit defaults | `"sklearn.preprocessing._data.MinMaxScaler"` (no params) |
| **Instance (custom)** | `MinMaxScaler(feature_range=(0,2))` | Non-default params | `{"class": "...", "params": {"feature_range": [0, 2]}}` |
| **String (module)** | `"sklearn.preprocessing.StandardScaler"` | Portable reference | `"sklearn.preprocessing._data.StandardScaler"` (same as class) |
| **String (controller)** | `"chart_2d"` | Built-in operator | `"chart_2d"` |
| **String (file)** | `"transformer.pkl"` | Saved object | `"transformer.pkl"` |
| **Dict (explicit, defaults)** | `{"class": "sklearn.preprocessing.StandardScaler"}` | Full control | `"sklearn.preprocessing._data.StandardScaler"` (same as class) |
| **Dict (explicit, params)** | `{"class": "...", "params": {...}}` | Full control | `{"class": "...", "params": {...}}` (non-default only) |
| **Dict (operator)** | `{"y_processing": MinMaxScaler}` | Special operator | `{"y_processing": "sklearn.preprocessing._data.MinMaxScaler"}` |
| **Model + Name** | `{"model": PLSRegression(), "name": "PLS-10"}` | Named model | `{"name": "...", "model": "..."}` or with params |
| **Function** | `nicon` | NN builder | `{"function": "module.nicon"}` |
| **Model + Train** | `{"model": nicon, "train_params": {...}}` | NN with config | `{"model": {...}, "train_params": {...}}` |
| **Model + Finetune** | `{"model": PLSRegression(), "finetune_params": {...}}` | HPO | `{"model": "...", "finetune_params": {...}}` or with params |
| **Generator (_or_)** | `{"_or_": [A, B, C]}` | Alternatives | Expands to N pipelines |
| **Generator (_range_)** | `{"_range_": [1, 10, 2], "param": "n", "model": ...}` | Param sweep | Expands to M pipelines |
| **Generator + size** | `{"_or_": [...], "size": 2}` | Combinations | C(n, k) pipelines |
| **Generator + count** | `{"_or_": [...], "count": 5}` | Random sample | 5 pipelines |
| **Nested generator** | `{"_or_": [...], "size": [2, (1,2)]}` | Sub-pipelines | Complex expansion |

**Key principle**: Different syntaxes producing the same object (same class + same non-default params) → **same serialization** → **same hash**.

---

## Serialization Rules

nirs4all normalizes all syntaxes to a canonical form for storage and reproducibility.

### Rule 1: Classes → String Paths (When Defaults Only)

**Input (all produce same object)**:
```python
from sklearn.preprocessing import StandardScaler

# Syntax 1: Class
StandardScaler

# Syntax 2: Instance with defaults
StandardScaler()

# Syntax 3: String path
"sklearn.preprocessing.StandardScaler"

# Syntax 4: Dict with class only
{"class": "sklearn.preprocessing.StandardScaler"}
```

**All serialize to the same canonical form**:
```json
"sklearn.preprocessing._data.StandardScaler"
```

**Hash**: Based on class path only (all produce identical hash).

**Note**: Uses **internal module path** (may differ from import path).

---

### Rule 2: Instances → Dict with Params (Only Non-Defaults)

**Input with non-default parameter**:
```python
MinMaxScaler(feature_range=(0, 2))  # (0, 2) is NOT default
```

**Serialized**:
```json
{
    "class": "sklearn.preprocessing._data.MinMaxScaler",
    "params": {
        "feature_range": [0, 2]
    }
}
```

**Input with default parameter**:
```python
MinMaxScaler(feature_range=(0, 1))  # (0, 1) IS default
```

**Serialized** (no params, identical to class):
```json
"sklearn.preprocessing._data.MinMaxScaler"
```

**Key behavior**: Only **non-default** parameters are included (via `_changed_kwargs()`). This ensures:
- `MinMaxScaler` (class)
- `MinMaxScaler()` (instance, defaults)
- `MinMaxScaler(feature_range=(0, 1))` (instance, explicit defaults)
- `"sklearn.preprocessing.MinMaxScaler"` (string)

All produce the **same serialization and hash**.

**Hash**: Based on class path + JSON representation of non-default params.

---

### Rule 3: Tuples → Lists

**Reason**: YAML's `safe_load()` cannot deserialize Python-specific tuples.

**Input**:
```python
{
    "feature_range": (0, 1)
}
```

**Serialized**:
```json
{
    "feature_range": [0, 1]
}
```

**Exception**: Hyperparameter range tuples `('int', min, max)` are also converted to lists during YAML serialization but preserved as semantic ranges during JSON serialization phase.

---

### Rule 4: Functions → Dict with Function Key

**Input**:
```python
from nirs4all.operators.models.tensorflow.nicon import nicon
nicon
```

**Serialized**:
```json
{
    "function": "nirs4all.operators.models.cirad_tf.nicon"
}
```

---

### Rule 5: Nested Dicts Recursively Serialized

**Input**:
```python
{
    "model": {
        "class": PLSRegression,
        "params": {"n_components": 10}
    }
}
```

**Serialized**:
```json
{
    "model": {
        "class": "sklearn.cross_decomposition._pls.PLSRegression",
        "params": {
            "n_components": 10
        }
    }
}
```

---

### Rule 6: Special Keys Preserved

Generator keys (`_or_`, `_range_`, `size`, `count`) and operator keys (`y_processing`, `feature_augmentation`, etc.) are **preserved as-is**.

---

### Rule 7: Minimal Serialization (Hash-Based Uniqueness)

The `_changed_kwargs()` function compares current values to defaults:

```python
def _changed_kwargs(obj):
    """Return {param: value} for every __init__ param whose current
    value differs from its default."""
    sig = inspect.signature(obj.__class__.__init__)
    out = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        default = param.default if param.default is not inspect._empty else None
        current = getattr(obj, name, default)

        if current != default:
            out[name] = current  # Only save if different!

    return out
```

**Example comparisons**:

| Input | Default Value | Serialized Params | Serialized Form |
|-------|---------------|-------------------|-----------------|
| `MinMaxScaler` | N/A (class) | None | `"sklearn.preprocessing._data.MinMaxScaler"` |
| `MinMaxScaler()` | All defaults | None | `"sklearn.preprocessing._data.MinMaxScaler"` |
| `MinMaxScaler(feature_range=(0,1))` | `(0, 1)` | None (matches default) | `"sklearn.preprocessing._data.MinMaxScaler"` |
| `MinMaxScaler(feature_range=(0,2))` | `(0, 1)` | `{"feature_range": [0, 2]}` | `{"class": "...", "params": {...}}` |
| `PLSRegression()` | All defaults | None | `"sklearn.cross_decomposition._pls.PLSRegression"` |
| `PLSRegression(n_components=2)` | `2` | None (matches default) | `"sklearn.cross_decomposition._pls.PLSRegression"` |
| `PLSRegression(n_components=10)` | `2` | `{"n_components": 10}` | `{"class": "...", "params": {...}}` |

**This keeps serialized pipelines minimal and ensures hash-based uniqueness**:
- ✅ Same object (same class + same effective params) → Same serialization → Same hash
- ✅ Different objects (different class OR different params) → Different serialization → Different hash
- ✅ Pipeline deduplication works correctly
- ✅ Configuration variations are properly detected

---

## Batch Execution: Multiple Pipelines

You can run **multiple independent pipelines** with a single `nirs4all.run()` call by passing a list of pipelines:

```python
# Define independent pipelines
pipeline_pls = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),
    {"model": PLSRegression(n_components=10)}
]

pipeline_rf = [
    StandardScaler(),
    ShuffleSplit(n_splits=3),
    {"model": RandomForestRegressor()}
]

pipeline_nn = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=3),
    {"model": nicon, "train_params": {"epochs": 100}}
]

# Run all pipelines independently
result = nirs4all.run(
    pipeline=[pipeline_pls, pipeline_rf, pipeline_nn],  # List of pipelines
    dataset="sample_data/regression",
    verbose=1
)
```

**Key difference from sequential steps**:
- A **single pipeline** `[step1, step2, step3]` executes steps sequentially
- A **list of pipelines** `[[step1, step2], [step3, step4]]` executes each pipeline **independently**

### Cartesian Product: Pipelines × Datasets

When you also provide multiple datasets, `nirs4all.run()` executes the **cartesian product**:

```python
result = nirs4all.run(
    pipeline=[pipeline_pls, pipeline_rf],   # 2 pipelines
    dataset=["data/wheat", "data/corn"],    # 2 datasets
    verbose=1
)
# Runs 4 combinations:
# - pipeline_pls on wheat
# - pipeline_pls on corn
# - pipeline_rf on wheat
# - pipeline_rf on corn
```

All results are collected into a single `RunResult` for unified comparison.

### When to Use Multiple Pipelines vs Generators

| Use Case | Approach |
|----------|----------|
| Compare completely different strategies | List of pipelines |
| Explore variations within a strategy | Generator syntax (`_or_`, `_range_`) |
| Benchmark across datasets | List of datasets |
| Full grid search | Combine both approaches |

**Example combining both**:
```python
# Each pipeline uses generators internally
pipeline_pls = [
    MinMaxScaler(),
    {"_or_": [SNV, Detrend]},  # 2 preprocessing variants
    ShuffleSplit(n_splits=3),
    {"model": PLSRegression(n_components=10)}
]  # Expands to 2 configs

pipeline_rf = [
    StandardScaler(),
    ShuffleSplit(n_splits=3),
    {"model": RandomForestRegressor()}
]  # 1 config

result = nirs4all.run(
    pipeline=[pipeline_pls, pipeline_rf],  # 2 pipelines (total 3 configs)
    dataset=["data/wheat", "data/corn"],   # 2 datasets
    verbose=1
)
# Total runs: 3 configs × 2 datasets = 6 runs
```

---

## Best Practices

### ✅ Do

1. **Use any syntax you prefer** - they all normalize correctly:
   - Class: `StandardScaler`
   - Instance with defaults: `StandardScaler()`
   - Instance with custom params: `MinMaxScaler(feature_range=(0, 2))`
   - String: `"sklearn.preprocessing.StandardScaler"`
   - Dict: `{"class": "sklearn.preprocessing.StandardScaler"}`

2. **Trust the serialization** - same object = same hash, regardless of syntax

3. **Use dicts for portability**: `{"class": "...", "params": {...}}` works in JSON/YAML files

4. **Name important models**: `{"model": PLSRegression(), "name": "PLS-Best"}`

5. **Use generators for experimentation**: `{"_or_": [...], "count": 10}`

6. **Don't worry about defaults** - `_changed_kwargs()` handles it automatically:
   - `PLSRegression()` and `PLSRegression(n_components=2)` → same serialization (2 is default)
   - `PLSRegression(n_components=10)` → different serialization (10 is non-default)

### ❌ Avoid

1. **Mixing syntaxes unnecessarily**: Choose one style per project
2. **Hardcoding internal paths**: Use public import paths when possible
3. **Over-specifying generators**: `count` can explode with nested `_or_`
4. **Forgetting YAML limits**: Tuples become lists (usually fine, but be aware)
5. **Custom objects without serialization support**: Extend `serialize_component()` if needed

---

## Examples

### Example 1: Hash Equivalence - Same Object, Different Syntaxes

All these pipeline definitions produce the **same hash**:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression

# Pipeline A: Using classes and instances with defaults
pipeline_a = [
    StandardScaler,  # Class
    PLSRegression()  # Instance with defaults (n_components=2 is default)
]

# Pipeline B: Using instances with explicit defaults
pipeline_b = [
    StandardScaler(),
    PLSRegression(n_components=2)  # Explicit default
]

# Pipeline C: Using string paths
pipeline_c = [
    "sklearn.preprocessing.StandardScaler",
    "sklearn.cross_decomposition.PLSRegression"
]

# Pipeline D: Using dicts
pipeline_d = [
    {"class": "sklearn.preprocessing.StandardScaler"},
    {"class": "sklearn.cross_decomposition.PLSRegression"}
]

# Pipeline E: Mixed syntaxes
pipeline_e = [
    StandardScaler,  # Class
    {"class": "sklearn.cross_decomposition.PLSRegression"}  # Dict
]
```

**All serialize to**:
```json
[
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.cross_decomposition._pls.PLSRegression"
]
```

**Hash**: `get_hash(steps)` produces identical MD5 hash for all 5 pipelines.

**Result**: nirs4all recognizes them as the **same pipeline** and won't run duplicates.

---

### Example 2: Hash Difference - Different Objects

These pipelines produce **different hashes**:

```python
# Pipeline A: PLSRegression with default n_components
pipeline_a = [
    StandardScaler,
    PLSRegression()  # n_components=2 (default)
]

# Pipeline B: PLSRegression with non-default n_components
pipeline_b = [
    StandardScaler,
    PLSRegression(n_components=10)  # n_components=10 (non-default)
]
```

**Pipeline A serializes to**:
```json
[
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.cross_decomposition._pls.PLSRegression"
]
```

**Pipeline B serializes to**:
```json
[
    "sklearn.preprocessing._data.StandardScaler",
    {
        "class": "sklearn.cross_decomposition._pls.PLSRegression",
        "params": {
            "n_components": 10
        }
    }
]
```

**Hash**: Different hashes → Recognized as different pipelines → Both will run.

---

### Example 3: Simple Regression Pipeline

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, test_size=0.25),
    PLSRegression(n_components=10)
]
```

### Example 4: Multi-Model Comparison

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, test_size=0.25),
    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
    {"model": PLSRegression(n_components=15), "name": "PLS-15"}
]
```

### Example 5: Preprocessing Exploration

```python
from nirs4all.operators.transforms import Detrend, FirstDerivative, Gaussian

pipeline = [
    MinMaxScaler(),
    {"_or_": [Detrend, FirstDerivative, Gaussian]},  # 3 variations
    ShuffleSplit(n_splits=5, test_size=0.25),
    PLSRegression(n_components=10)
]
# Expands to 3 pipelines
```

### Example 6: Hyperparameter Optimization

```python
pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=5, test_size=0.25),
    {
        "model": PLSRegression(),
        "name": "PLS-Optimized",
        "finetune_params": {
            "n_trials": 50,
            "verbose": 2,
            "approach": "single",
            "sample": "tpe",
            "model_params": {
                'n_components': ('int', 1, 30)
            }
        }
    }
]
```

### Example 7: Complex Generator

```python
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, Gaussian, StandardNormalVariate
)

pipeline = [
    MinMaxScaler(),
    {"feature_augmentation": {
        "_or_": [Detrend, FirstDerivative, Gaussian, StandardNormalVariate],
        "size": (1, 2),  # 1 or 2 preprocessing steps
        "count": 5        # Random sample 5 combinations
    }},
    ShuffleSplit(n_splits=5, test_size=0.25),
    {
        "_range_": [5, 20, 5],  # 5, 10, 15, 20
        "param": "n_components",
        "model": PLSRegression
    }
]
# Expands to 5 * 4 = 20 pipelines
```

### Example 8: Neural Network with Training Config

```python
from nirs4all.operators.models.tensorflow.nicon import customizable_nicon

pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=5, test_size=0.25),
    {
        "model": customizable_nicon,
        "name": "CustomNN",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,
            "sample": "hyperband",
            "approach": "single",
            "model_params": {
                "filters_1": [8, 16, 32, 64],
                "filters_2": [8, 16, 32, 64]
            },
            "train_params": {
                "epochs": 10,  # Fast training during search
                "verbose": 0
            }
        },
        "train_params": {
            "epochs": 250,  # Full training with best params
            "verbose": 1
        }
    }
]
```

---

## Troubleshooting

### Error: "could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'"

**Cause**: Python tuples in pipeline config (e.g., `('int', 1, 30)`) cannot be deserialized by `yaml.safe_load()`.

**Solution**: This is fixed automatically by `_sanitize_for_yaml()` in the manifest manager. If you see this error, ensure you're using the latest nirs4all version.

**Manual fix** (if needed):
```python
# Change from:
"model_params": {
    'n_components': ('int', 1, 30)
}

# To:
"model_params": {
    'n_components': ['int', 1, 30]
}
```

### Error: "Pipeline configuration expansion would generate X configurations, exceeding the limit"

**Cause**: Generator expansion creates too many pipelines (default limit: 10,000).

**Solutions**:
1. Add `count` parameter: `{"_or_": [...], "count": 100}`
2. Increase limit: `PipelineConfigs(pipeline, max_generation_count=50000)`
3. Simplify generator (reduce options or nesting)

### Error: "Failed to import module.ClassName"

**Cause**: Invalid class path or missing dependency.

**Solutions**:
1. Check import path: `from sklearn.preprocessing import StandardScaler` → class path is `sklearn.preprocessing._data.StandardScaler` (internal)
2. Ensure dependencies installed: `pip install scikit-learn tensorflow pytorch`
3. Use instance instead: `StandardScaler()` instead of `"sklearn.preprocessing.StandardScaler"`

---

## See Also

- **SERIALIZATION_REFACTORING.md**: Details on artifact storage and manifest architecture
- **MANIFEST_ARCHITECTURE.md**: UID-based pipeline management
- **Operator_Catalog.md**: List of all built-in nirs4all operators
- **config_format.md**: Legacy configuration documentation (pre-refactoring)

---

**Last Updated**: October 14, 2025
**Version**: 1.0 (Post-Serialization Refactoring)

```{seealso}
**Related Examples:**
- [U01: Hello World](../../../examples/user/01_getting_started/U01_hello_world.py) - Your first pipeline
- [U02: Basic Regression](../../../examples/user/01_getting_started/U02_basic_regression.py) - Complete pipeline with preprocessing
- [R01: Pipeline Syntax](../../../examples/reference/R01_pipeline_syntax.py) - Comprehensive syntax reference
```
