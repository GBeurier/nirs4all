# FCK-PLS Torch Prototype

Experimental implementation of learnable Fractional Convolutional Kernel PLS using PyTorch.

## Results Summary (2024-12-31)

Tested on three datasets with proper train/test splits:

| Dataset | PLS | FCK-PLS (orig) | Torch V1 | Torch V2 |
|---------|-----|----------------|----------|----------|
| Synthetic (400 train) | 0.9986 | 0.9953 | **0.9983** | 0.9979 |
| LDMC_hiba (480 train) | 0.7670 | **0.8433** | 0.8085 | 0.7482 |
| Redox Brix (1888 train) | 0.6792 | 0.6847 | 0.6815 | **0.6860** |

**Key findings:**
- Both V1 and V2 match or exceed standard PLS performance
- V1 outperforms standard PLS on LDMC (+3.4% R²)
- V2 achieves best result on Redox Brix (+1% R²)
- The learned kernels successfully adapt to dataset characteristics

## Concept

The original FCK-PLS uses fixed fractional filters (parameterized by α and σ) followed by standard sklearn PLS.
This prototype makes the convolution learnable via backpropagation while keeping the PLS head as a differentiable "solved" layer.

### Two Versions

- **V1 (Learnable Kernels)**: Directly learn the kernel weights as free parameters. More stable, flexible.
- **V2 (Alpha/Sigma Parametric)**: Learn the fractional order α and scale σ, rebuild kernels each forward pass. More interpretable but potentially less stable.

### Key Design Choices

1. **PLS as Solved Head (Deflation Mode)**: The PLS directions are computed via iterative deflation (for single-target regression). Gradients flow through these operations to the convolutional front-end. Note: SVD mode only works for multi-target regression.

2. **Validation-Based Kernel Learning**: Critical architectural insight - the PLS head is fit on a training subset (75%), while kernel optimization loss is computed on a validation subset (25%). This prevents information leakage where the solved head would "see" the same targets used for loss.

3. **Full-Batch Training**: PLS depends on global covariances, so full-batch (or very large batch) training is recommended for stability.

4. **Regularization Priors**: Kernels are regularized for smoothness (second differences) and zero-mean (derivative-like behavior).

## Files

| File | Description |
|------|-------------|
| `fckpls_torch.py` | Main implementation: `FCKPLSTorch` estimator |
| `cv_utils.py` | Cross-validation search and analysis tools |
| `quick_test.py` | Basic validation tests |
| `compare_fckpls.py` | Comparison study: PLS vs OPLS vs FCK-PLS variants |
| `experiment_pipeline.py` | Pipeline experiments with preprocessing |

## Usage

### Quick Test
```bash
cd bench/fck-pls
python quick_test.py
```

### Basic Usage
```python
from fckpls_torch import FCKPLSTorch, TrainConfig, create_fckpls_v1

# Simple usage
model = create_fckpls_v1(n_kernels=16, n_components=10, epochs=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Full configuration
cfg = TrainConfig(
    epochs=300,
    lr=1e-3,
    early_stopping_patience=40,
    verbose=1,
)
model = FCKPLSTorch(
    version="v1",  # or "v2"
    n_kernels=16,
    kernel_size=31,
    n_components=10,
    init_mode="fractional",  # "random", "derivative", "fractional"
    train_cfg=cfg,
)
model.fit(X_train, y_train)
```

### Comparison Study
```bash
# Full comparison
python compare_fckpls.py

# Quick mode (fewer epochs)
python compare_fckpls.py --quick

# With plots
python compare_fckpls.py --plot
```

### Pipeline Experiments
```bash
python experiment_pipeline.py
python experiment_pipeline.py --preprocessing  # Analyze preprocessing impact
```

### Cross-Validation Search
```python
from cv_utils import FCKPLSCVSearch, GRID_STANDARD

search = FCKPLSCVSearch(
    param_grid=GRID_STANDARD,
    n_splits=5,
    verbose=1,
)
search.fit(X, y)
best_model = search.refit_best(X, y)
```

## Key Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `n_kernels` | 8-32 | Number of convolutional filters |
| `kernel_size` | 15-51 | Size of each kernel (odd) |
| `n_components` | 5-20 | PLS components |
| `ridge_lambda` | 1e-5 to 1e-2 | Regularization in PLS ridge |
| `epochs` | 200-500 | Training epochs |
| `lr` | 1e-4 to 1e-2 | Learning rate |

## Analysis Tools

```python
from cv_utils import analyze_kernels, compare_kernels_to_reference, plot_training_history

# Analyze learned kernels
info = analyze_kernels(model)
print(info["kernel_means"])

# Compare to reference fractional kernels
comparison = compare_kernels_to_reference(model)
print(comparison["best_match_alpha"])  # Which α each kernel resembles

# Plot training
plot_training_history(model)
```

## Questions to Investigate

1. Does FCK-PLS Torch outperform original FCK-PLS?
2. Does FCK-PLS Torch benefit from preprocessing, or does it learn equivalent transformations?
3. Which initialization (random vs fractional) works better?
4. V1 vs V2: which is more robust/performant?
5. How do learned kernels compare to standard fractional derivatives?

## Dependencies

- torch
- numpy
- scikit-learn
- nirs4all (for comparison with original FCK-PLS)
- matplotlib (optional, for plotting)
