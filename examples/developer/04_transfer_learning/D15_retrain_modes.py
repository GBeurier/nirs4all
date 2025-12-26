"""
D15 - Retrain Modes: Model Retraining and Transfer Strategies
==============================================================

nirs4all provides multiple retraining modes for adapting trained
models to new data.

This tutorial covers:

* Retrain modes: full, transfer, finetune
* Practical retraining workflows
* When to use each mode

Prerequisites
-------------
- U02_basic_regression for pipeline basics
- D14_transfer_analysis for transfer concepts

Next Steps
----------
See D16_pca_geometry for preprocessing quality analysis.

Duration: ~5 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D15 Retrain Modes Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D15 - Retrain Modes: Model Adaptation Strategies")
print("=" * 60)

print("""
nirs4all.retrain() provides different modes for model adaptation:

  Mode       │ Description
  ───────────┼─────────────────────────────────────────
  'full'     │ Retrain entire pipeline from scratch
  'transfer' │ Freeze preprocessing, retrain model
  'finetune' │ Fine-tune model with lower learning rate

Each mode suits different scenarios.
""")


# =============================================================================
# Section 1: Initial Training
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Initial Training (Create Base Model)")
print("-" * 60)

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    SNV(),
    {"y_processing": StandardScaler()},
    PLSRegression(n_components=10),
]

# Train and save
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="BaseModel",
    verbose=1,
    save_artifacts=True,
    plots_visible=args.plots
)

print(f"\nBase model trained: RMSE = {result.best_rmse:.4f}")

# Export the model for retraining
export_path = Path("exports/D15_base_model.n4a")
export_path.parent.mkdir(parents=True, exist_ok=True)
result.export(export_path)
print(f"Model exported to: {export_path}")


# =============================================================================
# Section 2: Full Retrain Mode
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Full Retrain Mode")
print("-" * 60)

print("""
mode='full' retrains everything from scratch.
Use when: Completely new dataset with different distribution.

    nirs4all.retrain(
        source=result.best,
        data="new_dataset/",
        mode='full'
    )
""")

# Retrain on same data (demonstration)
result_full = nirs4all.retrain(
    source=result.best,
    data="sample_data/regression",
    mode='full',
    verbose=1
)

print(f"\nFull retrain: RMSE = {result_full.best_rmse:.4f}")
print("  All steps retrained from scratch")


# =============================================================================
# Section 3: Transfer Mode
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Transfer Mode")
print("-" * 60)

print("""
mode='transfer' keeps preprocessing frozen, retrains model only.
Use when: New instrument, similar samples.

    Preprocessing   │ Model
    ────────────────┼──────────────
    FROZEN          │ RETRAINED
    (MinMaxScaler,  │ (PLSRegression
     SNV, y_proc)   │  learns new)
""")

result_transfer = nirs4all.retrain(
    source=result.best,
    data="sample_data/regression",
    mode='transfer',
    verbose=1
)

print(f"\nTransfer retrain: RMSE = {result_transfer.best_rmse:.4f}")
print("  Preprocessing frozen, only model retrained")


# =============================================================================
# Section 4: Finetune Mode
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Finetune Mode")
print("-" * 60)

print("""
mode='finetune' uses trained model as starting point.
Use when: Similar domain, want to adapt slightly.

    Preprocessing   │ Model
    ────────────────┼──────────────
    FROZEN          │ FINE-TUNED
    (keep learned)  │ (start from
                    │  trained weights)

For neural networks, uses lower learning rate.
""")

result_finetune = nirs4all.retrain(
    source=result.best,
    data="sample_data/regression",
    mode='finetune',
    verbose=1
)

print(f"\nFinetune: RMSE = {result_finetune.best_rmse:.4f}")
print("  Model fine-tuned from trained weights")


# =============================================================================
# Section 5: Retrain from Exported Bundle
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Retrain from Exported Bundle")
print("-" * 60)

print("""
You can retrain from an exported .n4a bundle file:

    nirs4all.retrain(
        source="exports/model.n4a",
        data="new_data/",
        mode='transfer'
    )

This is useful for sharing models between teams.
""")

# Retrain from the exported bundle
result_from_bundle = nirs4all.retrain(
    source=str(export_path),
    data="sample_data/regression",
    mode='transfer',
    verbose=1
)

print(f"\nRetrain from bundle: RMSE = {result_from_bundle.best_rmse:.4f}")
print("  Loaded pipeline from .n4a bundle and retrained model")


# =============================================================================
# Section 6: Retraining Workflow
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Practical Retraining Workflow")
print("-" * 60)

print("""
📋 Retraining Decision Tree:

  New Dataset
      │
      ▼
  Same domain? ─────No────▶ mode='full'
      │
     Yes
      │
      ▼
  Same instrument? ───No───▶ mode='transfer'
      │
     Yes
      │
      ▼
  Minor updates? ────Yes───▶ mode='finetune'
      │
      No
      │
      ▼
  mode='full'
""")


# =============================================================================
# Section 7: Retraining with Different Models
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Retraining with Different Models")
print("-" * 60)

print("""
Replace the model during retrain:

    from sklearn.ensemble import RandomForestRegressor

    nirs4all.retrain(
        source=result.best,
        data="new_data/",
        mode='transfer',
        new_model=RandomForestRegressor(n_estimators=100)
    )

Keeps preprocessing, uses different model.
""")

from sklearn.ensemble import RandomForestRegressor

result_new_model = nirs4all.retrain(
    source=result.best,
    data="sample_data/regression",
    mode='transfer',
    new_model=RandomForestRegressor(n_estimators=50, random_state=42),
    verbose=1
)

print(f"\nTransfer with RF: RMSE = {result_new_model.best_rmse:.4f}")
print("  Same preprocessing, different model")


# =============================================================================
# Section 8: Mode Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Mode Comparison Summary")
print("-" * 60)

print("""
📊 Retrain Mode Comparison:

┌────────────┬─────────────┬─────────────┬────────────────────────┐
│ Mode       │ Preprocessing│ Model       │ When to Use            │
├────────────┼─────────────┼─────────────┼────────────────────────┤
│ full       │ Retrain     │ Retrain     │ New domain/instrument  │
│ transfer   │ Frozen      │ Retrain     │ Same preproc works     │
│ finetune   │ Frozen      │ Fine-tune   │ Minor adaptation       │
└────────────┴─────────────┴─────────────┴────────────────────────┘
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. 'full' mode: Complete retrain from scratch
2. 'transfer' mode: Keep preprocessing, retrain model
3. 'finetune' mode: Start from trained weights
4. new_model: Replace model during retrain

Key function:
    nirs4all.retrain(
        source=result.best,  # or "exports/model.n4a"
        data="new_data/",
        mode='transfer'|'full'|'finetune',
        new_model=...,  # Optional model replacement
    )

Next: D16_pca_geometry.py - PCA geometry for preprocessing quality
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
