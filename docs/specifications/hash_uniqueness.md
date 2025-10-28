# Hash-Based Uniqueness in nirs4all Pipeline Serialization

**Date**: October 14, 2025
**Core Principle**: Same object → Same serialization → Same hash

---

## The Fundamental Rule

In nirs4all, **different syntaxes that produce the same object must serialize to the same canonical form**, resulting in the **same hash**.

This ensures:
- ✅ Pipeline deduplication works correctly
- ✅ Configuration variations are properly detected
- ✅ Users can write pipelines in any syntax without affecting hash
- ✅ Same configuration always produces same results (reproducibility)

---

## Examples of Equivalent Syntaxes

All these produce **identical serialization** and **identical hash**:

### Example 1: StandardScaler with Defaults

```python
from sklearn.preprocessing import StandardScaler

# Syntax A: Class
StandardScaler

# Syntax B: Instance (implicit defaults)
StandardScaler()

# Syntax C: String path
"sklearn.preprocessing.StandardScaler"

# Syntax D: Dict
{"class": "sklearn.preprocessing.StandardScaler"}
```

**All serialize to**:
```json
"sklearn.preprocessing._data.StandardScaler"
```

**Hash**: All produce the same MD5 hash (e.g., `"a7b3c9d2"`).

---

### Example 2: MinMaxScaler with Default Values

```python
from sklearn.preprocessing import MinMaxScaler

# The default for feature_range is (0, 1)

# Syntax A: Class
MinMaxScaler

# Syntax B: Instance (implicit defaults)
MinMaxScaler()

# Syntax C: Instance (explicit default value)
MinMaxScaler(feature_range=(0, 1))

# Syntax D: String path
"sklearn.preprocessing.MinMaxScaler"
```

**All serialize to**:
```json
"sklearn.preprocessing._data.MinMaxScaler"
```

**Why?** `_changed_kwargs()` detects that `feature_range=(0, 1)` matches the default, so it's omitted from serialization.

**Hash**: All produce the same MD5 hash.

---

### Example 3: PLSRegression with Default n_components

```python
from sklearn.cross_decomposition import PLSRegression

# The default for n_components is 2

# Syntax A: Class
PLSRegression

# Syntax B: Instance (implicit defaults)
PLSRegression()

# Syntax C: Instance (explicit default value)
PLSRegression(n_components=2)
```

**All serialize to**:
```json
"sklearn.cross_decomposition._pls.PLSRegression"
```

**Hash**: All produce the same MD5 hash.

---

## Examples of Different Serializations

These produce **different serializations** and **different hashes**:

### Example 1: Different Classes

```python
# Object A
StandardScaler

# Object B
MinMaxScaler
```

**Different classes → Different serialization → Different hash**

---

### Example 2: Same Class, Different Non-Default Params

```python
from sklearn.preprocessing import MinMaxScaler

# Object A: Default feature_range
MinMaxScaler()  # feature_range=(0, 1)

# Object B: Non-default feature_range
MinMaxScaler(feature_range=(0, 2))
```

**Object A serializes to**:
```json
"sklearn.preprocessing._data.MinMaxScaler"
```

**Object B serializes to**:
```json
{
    "class": "sklearn.preprocessing._data.MinMaxScaler",
    "params": {
        "feature_range": [0, 2]
    }
}
```

**Different params → Different serialization → Different hash**

---

### Example 3: Same Model, Different Names

```python
from sklearn.cross_decomposition import PLSRegression

# Pipeline A
{
    "model": PLSRegression(n_components=10),
    "name": "Model_A"
}

# Pipeline B
{
    "model": PLSRegression(n_components=10),
    "name": "Model_B"
}
```

**Different names → Different serialization → Different hash**

This is intentional! Different names allow you to track the same model with different training strategies or contexts.

---

## How _changed_kwargs() Works

The `_changed_kwargs()` function in `serialization.py` is responsible for detecting which parameters differ from defaults:

```python
def _changed_kwargs(obj):
    """Return {param: value} for every __init__ param whose current
    value differs from its default."""
    sig = inspect.signature(obj.__class__.__init__)
    out = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Get default from signature
        default = param.default if param.default is not inspect._empty else None

        # Get current value from object
        current = getattr(obj, name, default)

        # Compare (only save if different)
        if current != default:
            out[name] = current

    return out
```

### Decision Table

| Object | Default Value | Current Value | Different? | Serialized? |
|--------|---------------|---------------|------------|-------------|
| `StandardScaler()` | All defaults | All defaults | No | No params |
| `MinMaxScaler()` | `feature_range=(0,1)` | `(0,1)` | No | No params |
| `MinMaxScaler(feature_range=(0,1))` | `(0,1)` | `(0,1)` | No | No params |
| `MinMaxScaler(feature_range=(0,2))` | `(0,1)` | `(0,2)` | Yes | `{"feature_range": [0,2]}` |
| `PLSRegression()` | `n_components=2` | `2` | No | No params |
| `PLSRegression(n_components=2)` | `2` | `2` | No | No params |
| `PLSRegression(n_components=10)` | `2` | `10` | Yes | `{"n_components": 10}` |

---

## Hash Calculation

The hash is calculated in `PipelineConfigs.get_hash()`:

```python
@staticmethod
def get_hash(steps) -> str:
    """
    Generate a hash for the pipeline configuration.
    """
    import hashlib
    serializable = json.dumps(steps, sort_keys=True, default=str).encode('utf-8')
    return hashlib.md5(serializable).hexdigest()[0:8]
```

**Process**:
1. Serialize steps to canonical JSON (with sorted keys)
2. Encode to UTF-8 bytes
3. Compute MD5 hash
4. Take first 8 characters

**Result**: Identical serialization → Identical JSON string → Identical hash

---

## Pipeline Name Generation

In `PipelineConfigs.__init__()`:

```python
self.names = [
    name + "_" + self.get_hash(steps)[0:6]
    for steps in self.steps
]
```

**Example**:
```python
pipeline = [StandardScaler, PLSRegression()]
config = PipelineConfigs(pipeline, name="my_pipeline")
```

**Generated name**: `"my_pipeline_a7b3c9"` (where `a7b3c9` is hash of serialized steps)

**Result**: Pipelines with identical configurations get the **same suffix**, making duplicates obvious.

---

## Practical Implications

### 1. Write Pipelines Flexibly

You can use **any syntax** without worrying about hash collisions:

```python
# All equivalent
pipeline_a = [StandardScaler, PLSRegression(n_components=10)]
pipeline_b = [StandardScaler(), PLSRegression(n_components=10)]
pipeline_c = ["sklearn.preprocessing.StandardScaler", PLSRegression(n_components=10)]
```

All produce the same hash → Recognized as identical.

---

### 2. Automatic Deduplication

If you accidentally define the same pipeline twice:

```python
pipelines = [
    [StandardScaler, PLSRegression(n_components=10)],
    [StandardScaler(), PLSRegression(n_components=10)]  # Duplicate
]
```

nirs4all will detect them as identical (same hash) and skip the duplicate.

---

### 3. Configuration Variations Detection

When using generators:

```python
pipeline = [
    MinMaxScaler,
    {"_or_": [Detrend, FirstDerivative, Gaussian]},
    PLSRegression(n_components=10)
]
```

Expands to 3 pipelines with **different hashes** (different preprocessing step).

---

### 4. Explicit Default Values Don't Affect Hash

This is a common source of confusion:

```python
# User writes
PLSRegression(n_components=2)  # Explicit default

# Expects params in serialization, but gets
"sklearn.cross_decomposition._pls.PLSRegression"  # No params!
```

**Why?** `_changed_kwargs()` detects `n_components=2` matches the default, so it's omitted.

**Result**: Hash is identical to `PLSRegression()` or `PLSRegression` (class).

---

## Testing Hash Equivalence

You can verify hash equivalence programmatically:

```python
from nirs4all.pipeline import PipelineConfigs
from sklearn.preprocessing import StandardScaler

# Define two pipelines with different syntaxes
pipeline_a = [StandardScaler]
pipeline_b = [StandardScaler()]

# Create configs
config_a = PipelineConfigs(pipeline_a, name="test")
config_b = PipelineConfigs(pipeline_b, name="test")

# Compare hashes (extract from name)
hash_a = config_a.names[0].split("_")[-1]
hash_b = config_b.names[0].split("_")[-1]

assert hash_a == hash_b, "Hashes should be identical!"
print(f"✅ Both pipelines have hash: {hash_a}")
```

---

## Common Mistakes

### Mistake 1: Expecting Explicit Defaults to Change Hash

```python
# User expects these to be different
pipeline_a = [PLSRegression()]
pipeline_b = [PLSRegression(n_components=2)]
```

**Reality**: Both produce the same hash (2 is the default).

**Solution**: If you want different hashes, use non-default values:
```python
pipeline_b = [PLSRegression(n_components=10)]
```

---

### Mistake 2: Using Different String Paths

```python
# User expects these to be different
pipeline_a = ["sklearn.preprocessing.StandardScaler"]
pipeline_b = [StandardScaler]
```

**Reality**: Both resolve to internal path and produce the same hash.

**Solution**: This is correct behavior! Different syntaxes should produce the same hash.

---

### Mistake 3: Name Affecting Hash Unexpectedly

```python
# User expects these to have the same hash
pipeline_a = [{"model": PLSRegression(n_components=10)}]
pipeline_b = [{"model": PLSRegression(n_components=10), "name": "PLS-10"}]
```

**Reality**: Different hashes (name is part of the step configuration).

**Solution**: If you want the same hash, use the same name (or no name).

---

## See Also

- **WRITING_A_PIPELINE.md**: Complete guide to all pipeline syntaxes
- **SERIALIZATION_REFACTORING.md**: Details on serialization architecture
- **TUPLE_SERIALIZATION_FIX.md**: Recent fix for tuple handling

---

**Last Updated**: October 14, 2025
**Version**: 1.0
