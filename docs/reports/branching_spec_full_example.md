## Appendix A: Complete Example - Outlier Comparison with Nested Branches

```python
"""
Complete example: Compare outlier exclusion strategies with different preprocessings.
Train, save, reload, and predict - verifying reproducibility.
"""
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression

from nirs4all.operators.transforms import SNV, MSC, FirstDerivative
from nirs4all.pipeline import PipelineRunner, PipelineConfigs, DatasetConfigs

# Set seed for reproducibility
np.random.seed(42)

# Define pipeline with nested branches
pipeline = [
    ShuffleSplit(n_splits=5, random_state=42),
    MinMaxScaler(),  # Shared preprocessing

    # Level 1: Outlier exclusion strategies
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [
            None,  # Baseline - no exclusion
            {"method": "isolation_forest", "contamination": 0.05, "random_state": 42},
            {"method": "mahalanobis", "threshold": 3.0},
        ],
    }},

    # Level 2: Spectral preprocessing
    {"branch": {
        "snv_pca": [SNV(), {"y_processing": StandardScaler()}],
        "msc_d1": [MSC(), FirstDerivative()],
    }},

    # Model applied to all 6 branch combinations
    PLSRegression(n_components=5),
]
# Result: 3 outlier × 2 preprocessing × 5 folds = 30 predictions

# Train and save
runner = PipelineRunner(save_files=True, workspace="./branching_demo")
predictions, _ = runner.run(
    PipelineConfigs(pipeline),
    DatasetConfigs("data/sample_data")
)

print(f"Total predictions: {len(predictions)}")
print(f"Unique branches: {len(set(p.branch_path for p in predictions))}")

# Summary by branch
from nirs4all.analysis import PredictionAnalyzer
analyzer = PredictionAnalyzer(predictions)

summary = analyzer.branch_summary(metrics=["rmse", "r2"])
print(summary.to_markdown())

# Visualize
analyzer.plot_nested_branches(
    level1_var="outlier_strategy",
    level2_var="preprocessing",
    metric="rmse"
)
analyzer.plot_branch_diagram()

# Save report
analyzer.generate_report("reports/outlier_comparison.html", branch_comparison=True)

# Reproducibility test: reload and predict
print("\n--- Reproducibility Test ---")
reloaded_runner = PipelineRunner(workspace="./branching_demo")

for branch_id in range(6):  # 6 branch combinations
    best_pred = predictions.top(1, branch_id=branch_id)[0]

    # Predict on validation fold
    y_pred_reload, _ = reloaded_runner.predict(
        best_pred,
        DatasetConfigs("data/sample_data"),
        fold_id=best_pred.fold_id
    )

    # Compare with original
    original_y_pred = np.array(best_pred.y_pred)
    max_diff = np.max(np.abs(y_pred_reload - original_y_pred))

    print(f"Branch {branch_id} ({best_pred.branch_name}): max diff = {max_diff:.2e}")
    assert max_diff < 1e-10, f"Reproducibility failed for branch {branch_id}"

print("\n✅ All branches passed reproducibility test!")
```

---

**Author:** Senior Python/ML Developer
**Date:** December 2025
**Status:** Specification - Pending Review
