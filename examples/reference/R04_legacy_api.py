"""
R04 - Legacy API Reference
==========================

Reference for the original PipelineRunner/PipelineConfigs API.

This example demonstrates the legacy API that was the standard before v0.6.
The legacy API still works and is fully supported, but the new module-level
API (nirs4all.run, nirs4all.predict) is recommended for new projects.

Use cases for legacy API:

* Existing codebases that use PipelineRunner
* When you need fine-grained control over execution
* When integrating with custom orchestration logic
* For compatibility with older nirs4all versions

For the new recommended API, see any U* example (e.g., U01_hello_world.py).
This reference shows equivalent operations in both APIs for easy migration.

Duration: ~1 minute
Difficulty: Reference material
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports - Legacy API
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# New API (for comparison)
import nirs4all

# Parse command-line arguments
parser = argparse.ArgumentParser(description='R04 Legacy API Reference')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Legacy API - Basic Training
# =============================================================================
print("\n" + "=" * 70)
print("Section 1: Legacy API - Basic Training")
print("=" * 70)

# Define pipeline (same in both APIs)
pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

# LEGACY API:
# 1. Create configuration objects
# 2. Create a runner
# 3. Call runner.run()
# 4. Access results from Predictions object

pipeline_config = PipelineConfigs(
    definition=pipeline,
    name="LegacyExample"
)

dataset_config = DatasetConfigs("sample_data/regression")

runner = PipelineRunner(
    verbose=1,
    save_artifacts=True,
    workspace_path="workspace/legacy_api",
    plots_visible=args.plots,
)

# Run returns (predictions, per_dataset_results)
predictions, per_dataset = runner.run(pipeline_config, dataset_config)

# Access results
print(f"\nLegacy API Results:")
print(f"  Number of predictions: {predictions.num_predictions}")

# Get best by metric
top = predictions.top(n=3, rank_metric='rmse')
print(f"  Top 3 by RMSE:")
for i, p in enumerate(top, 1):
    rmse = p.get('test_rmse', p.get('val_score', 0))
    r2 = p.get('test_r2', 0)
    print(f"    {i}. RMSE: {rmse:.4f}, R²: {r2:.4f}")


# =============================================================================
# Section 2: New API Equivalent
# =============================================================================
print("\n" + "-" * 70)
print("Section 2: New API Equivalent")
print("-" * 70)

# 1. Call nirs4all.run() with all parameters
# 2. Access results via RunResult convenience methods

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="NewApiExample",
    verbose=1,
    save_artifacts=True,
    workspace_path="workspace/new_api",
    plots_visible=args.plots
)

print(f"\nNew API Results:")
print(f"  Number of predictions: {result.num_predictions}")
print(f"  Best RMSE: {result.best_rmse:.4f}")
r2_val = result.best_r2
print(f"  Best R²: {r2_val:.4f}" if not np.isnan(r2_val) else "  Best R²: (see test metrics)")


# =============================================================================
# Section 3: API Comparison Table
# =============================================================================
print("\n" + "-" * 70)
print("Section 3: API Comparison")
print("-" * 70)

print("""
TASK                          LEGACY API                              PUBLIC API
─────────────────────────────────────────────────────────────────────────────────
Configuration objects         PipelineConfigs, DatasetConfigs         (inline)
Runner creation               PipelineRunner(...)                     nirs4all.run(...)
Execute training              runner.run(configs...)                  nirs4all.run(...)
Return type                   (Predictions, dict)                     RunResult
Best score                    preds.top(1)[0]['val_score']            result.best_score
Best RMSE                     (compute manually)                      result.best_rmse
Best R²                       (compute manually)                      result.best_r2
Top N models                  preds.top(n=N)                          result.top(n=N)
Best entry                    preds.top(1)[0]                         result.best
Filter by model               preds.filter(model_name=...)            result.filter(...)
Get model names               preds.get_unique_values('model_name')   result.get_models()
Predictions (for later use)   nirs4all.predict(manifest_path, ...)    nirs4all.predict(...)
Export bundle                 (manual)                                result.export(...)
""")


# =============================================================================
# Section 4: Legacy API - Advanced Usage
# =============================================================================
print("\n" + "-" * 70)
print("Section 4: Legacy API - Advanced Usage")
print("-" * 70)

print("""
Advanced patterns with legacy API:

1. Reusing runner for multiple configurations:

   runner = PipelineRunner(verbose=1)

   for config in configs:
       predictions, _ = runner.run(config, dataset_config)
       # process results...

2. Custom workspace per experiment:

   for exp_name, pipeline in experiments.items():
       runner = PipelineRunner(
           workspace_path=f"workspace/{exp_name}",
           save_artifacts=True
       )
       predictions, _ = runner.run(PipelineConfigs(pipeline, exp_name), dataset_config)

3. Accessing per-dataset results:

   predictions, per_dataset = runner.run(pipeline_config, multi_dataset_config)
   for dataset_name, dataset_preds in per_dataset.items():
       print(f"{dataset_name}: {dataset_preds.num_predictions} predictions")

4. Working with execution trace:

   predictions, _ = runner.run(pipeline_config, dataset_config)
   trace = runner.last_execution_trace
   for step in trace.steps:
       print(f"Step: {step.name}, Shape: {step.output_shape}")
""")


# =============================================================================
# Section 5: Migration Guide
# =============================================================================
print("\n" + "-" * 70)
print("Section 5: Migration Guide")
print("-" * 70)

print("""
Migrating from Legacy API to New API:

BEFORE (Legacy):
    from nirs4all.pipeline import PipelineRunner, PipelineConfigs
    from nirs4all.data import DatasetConfigs

    runner = PipelineRunner(verbose=1, save_artifacts=True)
    pipeline_config = PipelineConfigs(pipeline, "MyPipeline")
    dataset_config = DatasetConfigs("path/to/data")
    predictions, per_dataset = runner.run(pipeline_config, dataset_config)
    best = predictions.top(n=1)[0]

AFTER (New API):
    import nirs4all

    result = nirs4all.run(
        pipeline=pipeline,
        dataset="path/to/data",
        name="MyPipeline",
        verbose=1,
        save_artifacts=True
    )
    best = result.best

Key differences:
- New API uses module-level functions (nirs4all.run, nirs4all.predict)
- RunResult provides convenience accessors (best_rmse, best_r2, best)
- DatasetConfigs can be replaced with string path
- PipelineConfigs is created implicitly

Both APIs are fully supported and can be mixed in the same codebase.
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
R04 - Legacy API Reference
==========================

The legacy API (PipelineRunner, PipelineConfigs, DatasetConfigs) is:
  ✓ Fully supported
  ✓ Useful for fine-grained control
  ✓ Compatible with older codebases

For new projects, the new API is recommended:
  result = nirs4all.run(pipeline, dataset, name, ...)
  result.best_rmse
  result.best_r2
  result.top(n=5)
  result.export("path.n4a")

See U01_hello_world.py for the recommended approach.
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()

print("\n✓ R04_legacy_api.py completed!")
