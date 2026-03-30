# Core Concepts

NIRS4ALL is built around a few core ideas. This page gives you a quick overview — for in-depth explanations, see the dedicated {doc}`/concepts/index` section.

## The Big Picture

```
Dataset  +  Pipeline  →  nirs4all.run()  →  RunResult
```

1. **Dataset** — Your spectral data (X), targets (y), and metadata, loaded into a `SpectroDataset`
2. **Pipeline** — A list of processing steps: preprocessing, cross-validation, model
3. **Result** — Predictions, scores (RMSE, R²), and exportable models

## Pipelines

A pipeline is a list of steps applied sequentially:

```python
pipeline = [
    MinMaxScaler(),                              # Preprocessing (modifies X)
    {"y_processing": MinMaxScaler()},            # Target scaling (modifies y)
    ShuffleSplit(n_splits=5, test_size=0.25),    # Cross-validation splitter
    {"model": PLSRegression(n_components=10)}    # Model
]
```

Steps are dispatched to **controllers** that know how to execute each type. You rarely interact with controllers directly.

See {doc}`/concepts/pipelines` for details on step types, keywords, and execution order.

## Datasets

NIRS4ALL accepts data from many sources:

```python
# Folder path (auto-detects files)
result = nirs4all.run(pipeline, dataset="data/wheat/")

# Numpy arrays
result = nirs4all.run(pipeline, dataset=(X, y))

# Synthetic data for testing
dataset = nirs4all.generate(n_samples=500, random_state=42)
```

See {doc}`/concepts/datasets` for partitions, multi-source data, signal types, and repetitions.

## Results

```python
result = nirs4all.run(pipeline, dataset)

result.best_rmse         # Best model's RMSE
result.best_r2           # Best model's R²
result.top(n=5)          # Top 5 configurations
result.export("model.n4a")  # Export for deployment
```

See {doc}`/concepts/predictions_and_deployment` for the full result API and deployment workflow.

## Key Concepts (In-Depth)

The {doc}`/concepts/index` section covers these topics in detail:

| Concept | What You'll Learn |
|---------|------------------|
| {doc}`/concepts/pipelines` | Step types, keywords, execution model |
| {doc}`/concepts/datasets` | SpectroDataset, partitions, multi-source, signal types |
| {doc}`/concepts/cross_validation` | CV, OOF predictions, refit, scoring |
| {doc}`/concepts/branching_and_merging` | Duplication vs separation branches, merge strategies, stacking |
| {doc}`/concepts/generators` | `_or_`, `_range_`, `_grid_` — automated pipeline expansion |
| {doc}`/concepts/augmentation` | Sample and feature augmentation |
| {doc}`/concepts/predictions_and_deployment` | Results, export, predict, explain, retrain, sessions |

## Next Steps

- {doc}`/getting_started/tutorial` — Progressive tutorial from hello world to stacking
- {doc}`/concepts/index` — In-depth concept explanations
- {doc}`/reference/pipeline_keywords` — Complete keyword reference
