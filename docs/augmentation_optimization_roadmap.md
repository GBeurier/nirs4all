# Augmentation Pipeline Optimization Roadmap

## Executive Summary

The current augmentation pipeline in nirs4all suffers from severe performance issues due to **sample-by-sample processing** at multiple levels of the architecture. This document provides a comprehensive analysis and a roadmap for optimization.

---

## 🔴 Root Cause Analysis

### 1. Sample-by-Sample Processing in Controller (Critical)

**Location:** `nirs4all/controllers/transforms/transformer.py` - `_execute_for_sample_augmentation()` (lines 186-276)

The controller iterates over each target sample individually:

```python
# Current implementation (simplified)
for sample_id in target_sample_ids:
    origin_data = dataset.x(origin_selector, "3d", ...)  # 1 read per sample

    for source_idx, source_data in enumerate(origin_data):
        for proc_idx in range(source_data.shape[1]):
            transformer = clone(operator)
            transformer.fit(train_data)        # FIT for every sample!
            transformed_data = transformer.transform(proc_data)

    dataset.add_samples(data_to_add, indexes=index_dict)  # 1 write per sample
```

**Impact:**
| Operation | Count for N samples, M sources, P processings |
|-----------|----------------------------------------------|
| Dataset reads | N |
| Transformer clones | N × M × P |
| Transformer fits | N × M × P (redundant!) |
| Dataset writes | N |

For 1000 samples with 16 augmenters, this means **16,000+ individual operations** instead of 16 batched operations.

---

### 2. Python Loops Inside Augmenters (Critical)

Nearly all augmenters iterate sample-by-sample internally:

| File | Augmenters | Loop Pattern |
|------|------------|--------------|
| `augmentation/spectral.py` | WavelengthShift, WavelengthStretch, LocalWavelengthWarp, SmoothMagnitudeWarp, BandPerturbation, GaussianSmoothingJitter, BandMasking, ChannelDropout, SpikeNoise, LocalClipping | `for i in range(n_samples)` |
| `augmentation/splines.py` | Spline_Y_Perturbations, Spline_X_Perturbations, Spline_X_Simplification, Spline_Curve_Simplification | `for x in X` |
| `augmentation/random.py` | Rotate_Translate, Random_X_Operation | list comprehension / `np.vectorize` |

**Example - WavelengthShift (spectral.py):**
```python
def augment(self, X, apply_on="samples"):
    result = np.empty_like(X)
    for i in range(n_samples):  # Python loop!
        shift = self.random_gen.uniform(...)
        result[i] = np.interp(...)  # Per-sample interpolation
    return result
```

---

### 3. Fake Vectorization with `np.vectorize` (Critical)

**Location:** `augmentation/random.py` (line 20)

```python
v_angle_p = np.vectorize(angle_p)  # NOT true vectorization!
```

Despite its name, `np.vectorize` is a Python loop wrapper, not C-level vectorization. It calls the Python function for **every element** of the array.

---

### 4. Repeated Expensive Operations Inside Loops

**Spline fitting per sample (splines.py):**
```python
for x in X:
    y_points = [self.random_gen.uniform(...) for _ in range(nb_spline_points)]
    t, c, k = interpolate.splrep(x_gen, y, s=0, k=3)  # Expensive!
    spline = interpolate.BSpline(t, c, k)
    distor = spline(x_range)
```

**Redundant linspace (random.py):**
```python
def deformation(x):
    x_range = np.linspace(0, 1, x.shape[-1])  # Computed for EVERY sample!
    # ...
```

---

### 5. Inefficient Convolution (spectral.py)

```python
for i in range(n_samples):
    noise[i] = _convolve_1d(noise[i], kernel)  # Row-by-row instead of batch
```

Should use `scipy.ndimage.convolve1d(noise, kernel, axis=1)` for batch processing.

---

## 📊 Augmenter Classification by Vectorization Status

### ✅ Already Vectorized (No Changes Needed)
- `GaussianAdditiveNoise` - Uses `random_gen.normal()` on full array
- `MultiplicativeNoise` - Uses `random_gen.normal()` on full array
- `LinearBaselineDrift` - Vectorized slope/intercept application
- `PolynomialBaselineDrift` - Vectorized polynomial application
- `UnsharpSpectralMask` - Uses `scipy.ndimage.gaussian_filter1d` with axis

### 🟡 Partially Vectorized (Minor Fixes Needed)
- `GaussianSmoothingJitter` - Noise generation is vectorized, but smoothing loop needs fixing
- `MixupAugmenter` - Core logic vectorized, but has some per-sample index operations

### 🔴 Not Vectorized (Major Refactoring Needed)
- `WavelengthShift` - Per-sample `np.interp`
- `WavelengthStretch` - Per-sample `np.interp`
- `LocalWavelengthWarp` - Per-sample spline + interpolation
- `SmoothMagnitudeWarp` - Per-sample spline + interpolation
- `BandPerturbation` - Nested loops (samples × bands)
- `BandMasking` - Per-sample random band selection
- `ChannelDropout` - Per-sample channel selection + interpolation
- `SpikeNoise` - Per-sample spike insertion
- `LocalClipping` - Per-sample clipping regions
- `Rotate_Translate` - `np.vectorize` + list comprehension
- `Random_X_Operation` - List comprehension
- `Spline_Y_Perturbations` - Per-sample spline fitting
- `Spline_X_Perturbations` - Per-sample spline fitting
- `Spline_X_Simplification` - Per-sample random selection + spline
- `Spline_Curve_Simplification` - Per-sample spline fitting

---

## 🛠️ Optimization Roadmap

### Phase 1: Controller-Level Batch Processing (Highest Impact)

**Goal:** Eliminate sample-by-sample dataset I/O and transformer cloning.

**Changes to `TransformerMixinController._execute_for_sample_augmentation()`:**

1. **Batch data fetching:**
   - Fetch all target samples in one `dataset.x()` call
   - Reshape to `(n_samples, n_sources, n_processings, n_features)`

2. **Single transformer fit:**
   - Fit transformer once on train data
   - Reuse fitted transformer for all samples

3. **Batch transform:**
   - Call `transformer.transform(all_samples)` once per augmenter

4. **Bulk insert:**
   - Create new method `dataset.add_samples_batch()`
   - Pre-allocate storage for all new samples
   - Update indices in bulk

**Expected Speedup:** 5-10x for the controller overhead alone.

---

### Phase 2: Vectorize Interpolation-Based Augmenters

**Target:** WavelengthShift, WavelengthStretch, ChannelDropout

**Strategy:** Use `scipy.ndimage.map_coordinates` for batch interpolation.

```python
# Before (per-sample)
for i in range(n_samples):
    result[i] = np.interp(new_x, old_x, X[i])

# After (batch)
from scipy.ndimage import map_coordinates
# Generate all coordinates at once
coords = compute_shifted_coords(n_samples, n_features, shifts)  # (2, n_samples, n_features)
result = map_coordinates(X, coords, order=1, mode='nearest')
```

**Expected Speedup:** 50-100x for these augmenters.

---

### Phase 3: Vectorize Spline-Based Augmenters

**Target:** LocalWavelengthWarp, SmoothMagnitudeWarp, all Spline_* augmenters

**Strategy A - Batch control point generation:**
```python
# Generate all random control points at once
control_points = self.random_gen.uniform(low, high, size=(n_samples, n_control_points))

# Vectorized spline evaluation using scipy.interpolate.make_interp_spline with axis
```

**Strategy B - Numba JIT compilation:**
```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def batch_spline_warp(X, control_points):
    result = np.empty_like(X)
    for i in prange(X.shape[0]):  # Parallel loop!
        result[i] = apply_spline(X[i], control_points[i])
    return result
```

**Strategy C - JAX vmap (if JAX already in project):**
```python
import jax.numpy as jnp
from jax import vmap

def single_sample_warp(x, control_points):
    # Single-sample logic
    return warped

batch_warp = vmap(single_sample_warp, in_axes=(0, 0))
result = batch_warp(X, all_control_points)
```

**Expected Speedup:** 20-50x depending on approach.

---

### Phase 4: Fix Rotate_Translate

**Target:** `augmentation/random.py`

**Changes:**
1. Remove `np.vectorize` - replace with proper NumPy broadcasting
2. Pre-compute `linspace` once
3. Generate all random parameters upfront

```python
# Before
v_angle_p = np.vectorize(angle_p)
for x in X:
    x_range = np.linspace(0, 1, x.shape[-1])
    # ...

# After
def augment(self, X, apply_on="samples"):
    n_samples, n_features = X.shape
    x_range = np.linspace(0, 1, n_features)  # Once!

    # Generate all random params at once
    p1 = self.random_gen.uniform(-self.p_range, self.p_range, n_samples)
    p2 = self.random_gen.uniform(-self.p_range, self.p_range, n_samples)
    xI = self.random_gen.uniform(0, 1, n_samples)
    yI = self.random_gen.uniform(-self.y_factor, self.y_factor, n_samples)

    # Vectorized angle computation using np.where
    # ... broadcast across samples
```

**Expected Speedup:** 100x+

---

### Phase 5: Fix Convolution Loops

**Target:** `GaussianSmoothingJitter`, any `_convolve_1d` usage

```python
# Before
for i in range(n_samples):
    noise[i] = _convolve_1d(noise[i], kernel)

# After
from scipy.ndimage import convolve1d
noise = convolve1d(noise, kernel, axis=1, mode='reflect')
```

**Expected Speedup:** 10-20x

---

### Phase 6: Optimize Dataset Operations

**Target:** `nirs4all/data/dataset.py`

1. **Add `add_samples_batch()` method:**
   - Pre-allocate array space for N new samples
   - Bulk index update
   - Single metadata operation

2. **Optimize `x()` method for batch access:**
   - Cache frequently accessed slices
   - Reduce index computation overhead

---

## 📈 Expected Overall Impact

| Phase | Speedup Factor | Effort |
|-------|---------------|--------|
| Phase 1: Controller batching | 5-10x | Medium |
| Phase 2: Interpolation vectorization | 50-100x (for affected augmenters) | Medium |
| Phase 3: Spline vectorization | 20-50x (for affected augmenters) | High |
| Phase 4: Rotate_Translate fix | 100x+ | Low |
| Phase 5: Convolution fix | 10-20x | Low |
| Phase 6: Dataset optimization | 2-3x | Medium |

**Combined effect for augmentation-heavy pipelines: 20-100x faster**

---

## 🎯 Recommended Implementation Order

1. **Quick wins (1-2 days):**
   - Phase 4: Fix Rotate_Translate
   - Phase 5: Fix convolution loops

2. **Medium effort, high impact (1 week):**
   - Phase 1: Controller batching
   - Phase 2: Interpolation vectorization

3. **Larger refactoring (2+ weeks):**
   - Phase 3: Spline vectorization
   - Phase 6: Dataset optimization

---

## 🔧 Testing Strategy

1. **Correctness tests:**
   - Compare output distributions before/after optimization
   - Use fixed random seeds for reproducibility

2. **Performance benchmarks:**
   - Create benchmark script with varying dataset sizes (100, 1000, 10000 samples)
   - Measure time per augmenter
   - Track memory usage

3. **Integration tests:**
   - Run existing examples (`Q12_sample_augmentation.py`, `Q20_analysis.py`)
   - Verify prediction metrics remain stable

---

## 📚 References

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [SciPy ndimage.map_coordinates](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html)
- [Numba JIT Compilation](https://numba.pydata.org/numba-doc/latest/user/jit.html)
- [JAX vmap](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html)
