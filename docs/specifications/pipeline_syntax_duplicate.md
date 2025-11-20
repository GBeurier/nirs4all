# Pipeline Syntax & Serialization - Complete Guide

**Date**: October 14, 2025
**Status**: ✅ Complete - All issues resolved

---

## Summary of Work Completed

### 1. ✅ Tuple Serialization Bug Fixed

**Problem**: Pipeline configurations with hyperparameter tuples `('int', 1, 30)` caused YAML serialization errors:
```
yaml.constructor.ConstructorError: could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'
```

**Solution**:
- Updated `serialize_component()` to convert all tuples to lists
- Applied `_sanitize_for_yaml()` in `save_manifest()`
- Verified backward compatibility (lists are semantically equivalent to tuples for hyperparameter ranges)

**Files Modified**:
- `nirs4all/pipeline/serialization.py` (lines 20-26)
- `nirs4all/pipeline/manifest_manager.py` (line 123)

**Status**: ✅ Tested and working

---

### 2. ✅ Comprehensive Pipeline Documentation Created

**Document**: `WRITING_A_PIPELINE.md` (1000+ lines)

**Contents**:
1. **Overview**: Philosophy of multiple syntaxes and normalization
2. **Pipeline vs Pipeline Generator**: Concrete pipelines vs expansion with `_or_` and `_range_`
3. **Basic Step Syntaxes**: 7 ways to define operators
4. **Model Step Syntaxes**: 9 ways to define models
5. **Generator Syntaxes**: 8 patterns for creating pipeline variations
6. **File Formats**: JSON and YAML examples
7. **Syntax Reference Table**: Quick lookup for all 15+ syntaxes
8. **Serialization Rules**: 7 rules explaining normalization
9. **Best Practices**: Do's and don'ts
10. **Examples**: 6 complete real-world pipelines
11. **Troubleshooting**: Common errors and solutions

**Status**: ✅ Complete and comprehensive

---

### 3. ✅ Serialization Fix Documentation

**Document**: `TUPLE_SERIALIZATION_FIX.md`

**Contents**:
- Problem description
- Root cause analysis
- Solution implementation
- Testing results
- Impact assessment
- Verification checklist
- Lessons learned

**Status**: ✅ Complete with test results

---

## Pipeline Syntax - Quick Reference

### All Supported Step Syntaxes

From `examples/sample.py` and `WRITING_A_PIPELINE.md`:

```python
pipeline = [
    # 1. Class (uninstantiated) - default params
    MinMaxScaler,

    # 2. Instance with params
    MinMaxScaler(feature_range=(0.1, 0.8)),

    # 3. String - controller name
    "chart_2d",

    # 4. Dict - special operator
    {"y_processing": MinMaxScaler},

    # 5. Generator - alternatives (_or_)
    {"feature_augmentation": {
        "_or_": [Detrend, FirstDerivative, Gaussian],
        "size": [1, (1, 2)],
        "count": 5
    }},

    # 6. Instance - cross-validator
    ShuffleSplit(n_splits=3, test_size=0.25),

    # 7. Dict - explicit class and params
    {
        "class": "sklearn.model_selection.ShuffleSplit",
        "params": {
            "n_splits": 3,
            "test_size": 0.25
        }
    },

    # 8. String - file path
    "my/super/transformer.pkl",

    # 9. Generator - parameter range (_range_)
    {
        "_range_": [1, 12, 2],
        "param": "n_components",
        "model": {
            "class": "sklearn.cross_decomposition.PLSRegression"
        }
    },

    # 10. Model - dict with name
    {
        "name": "PLS-3_components",
        "model": {
            "class": "sklearn.cross_decomposition.PLSRegression",
            "params": {
                "n_components": 3
            }
        }
    },

    # 11. Function - neural network builder
    nicon,

    # 12. Model - with finetuning (hyperparameter optimization)
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
                'n_components': ('int', 1, 30),  # Tuple → List in serialization
            },
        }
    },

    # 13. Model - function with finetuning
    {
        "model": customizable_nicon,
        "name": "NN-Optimized",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,
            "sample": "hyperband",
            "approach": "single",
            "model_params": {
                "filters_1": [8, 16, 32, 64],
                "filters_3": [8, 16, 32, 64]
            },
            "train_params": {
                "epochs": 10,
                "verbose": 0
            }
        },
        "train_params": {
            "epochs": 250,
            "verbose": 0
        }
    },

    # 14. String - saved model file
    "My_awesome_model.pkl",
    "My_awesome_tf_model.keras",

    # 15. Model - instance with params
    {"model": PLSRegression(10), "name": "PLS_10_components"},

    # 16. Custom code file
    {"source_file": "my_model.py", "class": "MyAwesomeModel"},

    # 17. String - class path
    "sklearn.linear_model.Ridge"
]
```

---

## Serialization Normalization

### Key Principle

**All syntaxes normalize to a canonical form** during serialization:

```python
{
    "class": "module.path.ClassName",
    "params": {"param1": value1, "param2": value2}  # Only non-default params
}
```

### Example Normalization

**Input (5 different syntaxes)**:
```python
# 1. Class
PLSRegression

# 2. String
"sklearn.cross_decomposition.PLSRegression"

# 3. Dict
{"class": "sklearn.cross_decomposition.PLSRegression"}

# 4. Instance (default params)
PLSRegression()

# 5. Dict with model key
{"model": PLSRegression}
```

**Output (all serialize to)**:
```json
"sklearn.cross_decomposition._pls.PLSRegression"
```

---

**Input (instance with custom params)**:
```python
PLSRegression(n_components=10)
```

**Output**:
```json
{
    "class": "sklearn.cross_decomposition._pls.PLSRegression",
    "params": {
        "n_components": 10
    }
}
```

---

### Tuple Conversion

**Before Fix**:
```python
# Input
{'n_components': ('int', 1, 30)}

# Serialized (WRONG - causes YAML error)
{'n_components': !!python/tuple ['int', 1, 30]}
```

**After Fix**:
```python
# Input
{'n_components': ('int', 1, 30)}

# Serialized (CORRECT)
{'n_components': ['int', 1, 30]}
```

---

## Generator Expansion

### `_or_` - Alternative Choices

```python
# Definition
{"_or_": [Detrend, FirstDerivative, Gaussian]}

# Expands to 3 pipelines:
# Pipeline 1: Detrend
# Pipeline 2: FirstDerivative
# Pipeline 3: Gaussian
```

### `_range_` - Parameter Sweep

```python
# Definition
{
    "_range_": [1, 10, 2],  # Start, end, step
    "param": "n_components",
    "model": PLSRegression
}

# Expands to 5 pipelines:
# Pipeline 1: PLSRegression(n_components=1)
# Pipeline 2: PLSRegression(n_components=3)
# Pipeline 3: PLSRegression(n_components=5)
# Pipeline 4: PLSRegression(n_components=7)
# Pipeline 5: PLSRegression(n_components=9)
```

### `_or_` with `size` - Combinations

```python
# Definition
{"_or_": [A, B, C, D], "size": 2}

# Expands to 6 pipelines (all 2-item combinations):
# Pipeline 1: [A, B]
# Pipeline 2: [A, C]
# Pipeline 3: [A, D]
# Pipeline 4: [B, C]
# Pipeline 5: [B, D]
# Pipeline 6: [C, D]
```

### `_or_` with `size` range

```python
# Definition
{"_or_": [A, B, C], "size": (1, 2)}

# Expands to 6 pipelines (1-item + 2-item combinations):
# Pipeline 1: [A]
# Pipeline 2: [B]
# Pipeline 3: [C]
# Pipeline 4: [A, B]
# Pipeline 5: [A, C]
# Pipeline 6: [B, C]
```

### `_or_` with `count` - Random Sampling

```python
# Definition
{"_or_": [A, B, C, D, E], "size": (1, 2), "count": 5}

# Would normally expand to 10 pipelines (5 singles + 10 pairs)
# But count=5 randomly samples 5 of them
```

---

## File Format Examples

### Python

```python
from nirs4all.pipeline import PipelineConfigs

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5),
    PLSRegression(n_components=10)
]

config = PipelineConfigs(pipeline, name="my_pipeline")
```

### JSON

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

### YAML

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

---
