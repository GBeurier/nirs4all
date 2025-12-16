"""
Q_feature_augmentation_modes: Demonstration of Feature Augmentation Action Modes
==================================================================================
This example demonstrates the three action modes for feature_augmentation:

1. **extend** (default): Add new processings to the set (linear growth)
   - Each operation runs independently on the base processing
   - If a processing already exists, it is not duplicated
   - Use case: Exploring independent preprocessing options

2. **add**: Chain operations on all existing + keep originals (multiplicative with originals)
   - Each operation is chained on ALL existing processings
   - Original processings are kept alongside new chained versions
   - Use case: Ablation studies where you need base processings as reference

3. **replace**: Chain operations on all existing + discard originals (multiplicative without originals)
   - Each operation is chained on ALL existing processings
   - Original processings are discarded, only chained versions remain
   - Use case: Multi-stage preprocessing pipelines without intermediate processings

Examples:
---------
Given initial processing: [raw_A]
After feature_augmentation: [B, C]

- extend:  [raw_A, raw_B, raw_C]        (2+1=3, but raw_A already existed → 3 total)
- add:     [raw_A, raw_A_B, raw_A_C]    (1 + 1×2 = 3 total)
- replace: [raw_A_B, raw_A_C]           (1×2 = 2 total, raw_A discarded)
"""

import os
os.environ['DISABLE_EMOJIS'] = '1'

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Feature Augmentation Action Modes Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()
display_pipeline_plots = args.plots
display_analyzer_plots = args.show


from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import (
    FirstDerivative,
    SecondDerivative,
    StandardNormalVariate,
    Detrend,
    SavitzkyGolay,
    Gaussian,
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def run_pipeline(name: str, pipeline_steps: list, description: str):
    """Run a pipeline and display results."""
    print(f"\n--- {name} ---")
    print(f"Description: {description}")
    print(f"Pipeline: {pipeline_steps}")

    pipeline_config = PipelineConfigs(pipeline_steps, name.lower().replace(" ", "_"))
    dataset_config = DatasetConfigs("sample_data/regression_2")

    runner = PipelineRunner(save_files=False, verbose=0, plots_visible=display_pipeline_plots)
    predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

    # Show resulting processings
    if predictions.num_predictions > 0:
        sample_pred = predictions.top(1)[0]
        print(f"Result: {predictions.num_predictions} prediction(s)")
        print(f"Sample preprocessing chain: {sample_pred.get('preprocessings', 'N/A')}")
    else:
        print("No predictions generated")

    return predictions


# =============================================================================
# DEMO 1: EXTEND MODE (Default - Linear Growth)
# =============================================================================
print_section("EXTEND MODE: Add new processings to set (linear growth)")

print("""
The 'extend' mode adds new preprocessing options independently.
Each operation runs on the base (raw) processing, creating parallel options.
If a processing already exists, it is NOT duplicated.

Example pipeline:
  [MinMaxScaler, {feature_augmentation: [SNV, FirstDerivative], action: "extend"}]

Starting: raw
After MinMaxScaler: raw_MinMaxScaler
After feature_augmentation:
  - SNV applied to raw → raw_StandardNormalVariate
  - FirstDerivative applied to raw → raw_FirstDerivative
Result: [raw_MinMaxScaler, raw_StandardNormalVariate, raw_FirstDerivative]
""")

extend_pipeline = [
    MinMaxScaler(),
    {"feature_augmentation": [StandardNormalVariate, FirstDerivative], "action": "extend"},
    ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

run_pipeline(
    "Extend Mode Demo",
    extend_pipeline,
    "Add SNV and FirstDerivative as independent preprocessing options"
)


# =============================================================================
# DEMO 2: ADD MODE (Multiplicative with Originals - Legacy Behavior)
# =============================================================================
print_section("ADD MODE: Chain on all + keep originals (multiplicative)")

print("""
The 'add' mode chains each operation on ALL existing processings.
Original processings are KEPT alongside new chained versions.
This is the default/legacy behavior for backward compatibility.

Example pipeline:
  [MinMaxScaler, {feature_augmentation: [SNV, FirstDerivative], action: "add"}]

Starting: raw
After MinMaxScaler: raw_MinMaxScaler
After feature_augmentation:
  - Keep original: raw_MinMaxScaler
  - SNV chained on raw_MinMaxScaler → raw_MinMaxScaler_StandardNormalVariate
  - FirstDerivative chained on raw_MinMaxScaler → raw_MinMaxScaler_FirstDerivative
Result: [raw_MinMaxScaler, raw_MinMaxScaler_StandardNormalVariate,
         raw_MinMaxScaler_FirstDerivative]
""")

add_pipeline = [
    MinMaxScaler(),
    {"feature_augmentation": [StandardNormalVariate, FirstDerivative], "action": "add"},
    ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

run_pipeline(
    "Add Mode Demo",
    add_pipeline,
    "Chain SNV and FirstDerivative, keeping MinMaxScaler as reference"
)


# =============================================================================
# DEMO 3: REPLACE MODE (Multiplicative without Originals)
# =============================================================================
print_section("REPLACE MODE: Chain on all + discard originals (multiplicative)")

print("""
The 'replace' mode chains each operation on ALL existing processings.
Original processings are DISCARDED - only the chained versions remain.

Example pipeline:
  [MinMaxScaler, {feature_augmentation: [SNV, FirstDerivative], action: "replace"}]

Starting: raw
After MinMaxScaler: raw_MinMaxScaler
After feature_augmentation:
  - SNV chained on raw_MinMaxScaler → raw_MinMaxScaler_StandardNormalVariate
  - FirstDerivative chained on raw_MinMaxScaler → raw_MinMaxScaler_FirstDerivative
  - raw_MinMaxScaler DISCARDED from context
Result: [raw_MinMaxScaler_StandardNormalVariate, raw_MinMaxScaler_FirstDerivative]
""")

replace_pipeline = [
    MinMaxScaler(),
    {"feature_augmentation": [StandardNormalVariate, FirstDerivative], "action": "replace"},
    ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

run_pipeline(
    "Replace Mode Demo",
    replace_pipeline,
    "Chain SNV and FirstDerivative, discarding MinMaxScaler from final selection"
)


# =============================================================================
# DEMO 4: SEQUENTIAL FEATURE AUGMENTATIONS WITH MIXED MODES
# =============================================================================
print_section("SEQUENTIAL AUGMENTATIONS: Mixed modes in sequence")

print("""
You can combine different action modes in sequence for complex pipelines.

Example pipeline:
  [{feature_augmentation: [SNV, Detrend], action: "extend"},
   {feature_augmentation: [FirstDerivative], action: "replace"}]

Step 1 (extend): raw_SNV, raw_Detrend
Step 2 (replace on each):
  - FirstDerivative on raw_SNV → raw_SNV_FirstDerivative
  - FirstDerivative on raw_Detrend → raw_Detrend_FirstDerivative
  - raw_SNV and raw_Detrend DISCARDED
Result: [raw_SNV_FirstDerivative, raw_Detrend_FirstDerivative]
""")

sequential_pipeline = [
    {"feature_augmentation": [StandardNormalVariate, Detrend], "action": "extend"},
    {"feature_augmentation": [FirstDerivative], "action": "replace"},
    ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

run_pipeline(
    "Sequential Mixed Modes",
    sequential_pipeline,
    "Extend with SNV/Detrend, then replace with FirstDerivative"
)


# =============================================================================
# DEMO 5: MULTI-STAGE PIPELINE WITH REPLACE
# =============================================================================
print_section("MULTI-STAGE PIPELINE: Build clean preprocessing chains")

print("""
Use 'replace' for multi-stage preprocessing without accumulating intermediates.

Example: Apply smoothing, then derivative, then normalization
  [{feature_augmentation: [SavitzkyGolay], action: "replace"},
   {feature_augmentation: [FirstDerivative], action: "replace"},
   {feature_augmentation: [StandardNormalVariate], action: "replace"}]

raw → raw_SavitzkyGolay → raw_SavitzkyGolay_FirstDerivative
    → raw_SavitzkyGolay_FirstDerivative_StandardNormalVariate

Only the final chain is kept at each stage (no intermediate bloat).
""")

multistage_pipeline = [
    {"feature_augmentation": [SavitzkyGolay], "action": "replace"},
    {"feature_augmentation": [FirstDerivative], "action": "replace"},
    {"feature_augmentation": [StandardNormalVariate], "action": "replace"},
    ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

run_pipeline(
    "Multi-Stage Replace Chain",
    multistage_pipeline,
    "SavitzkyGolay → FirstDerivative → SNV with no intermediates"
)


# =============================================================================
# SUMMARY
# =============================================================================
print_section("SUMMARY: Choosing the Right Action Mode")

print("""
┌──────────┬─────────────────────────────────────┬────────────────────────┐
│ Action   │ Behavior                            │ Use Case               │
├──────────┼─────────────────────────────────────┼────────────────────────┤
│ extend   │ Add new processings to set          │ Explore independent    │
│          │ No chaining, linear growth          │ preprocessing options  │
├──────────┼─────────────────────────────────────┼────────────────────────┤
│ add      │ Chain on all existing processings   │ Ablation studies with  │
│ (default)│ Keep originals + new chained        │ baselines as reference │
├──────────┼─────────────────────────────────────┼────────────────────────┤
│ replace  │ Chain on all existing processings   │ Multi-stage pipelines  │
│          │ Discard originals, keep chained     │ Clean chain building   │
└──────────┴─────────────────────────────────────┴────────────────────────┘

Quick Reference:
  # Extend: Build a flat set of preprocessing options
  {"feature_augmentation": [SNV, Gaussian, Detrend], "action": "extend"}

  # Add: Chain while keeping originals for comparison (default)
  {"feature_augmentation": [FirstDerivative], "action": "add"}

  # Replace: Pure chaining, clean pipeline stages
  {"feature_augmentation": [PCA(50)], "action": "replace"}
""")

print("\nDemo completed successfully!")
