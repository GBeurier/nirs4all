# Sample Augmentation Guide

## Overview

The sample augmentation feature in nirs4all allows you to create synthetic variations of your training samples by applying transformations. This is particularly useful for:

- **Increasing dataset size** when you have limited training samples
- **Balancing class distributions** in imbalanced datasets
- **Improving model generalization** through data diversity
- **Preventing overfitting** by introducing controlled variations

**Key Feature:** Built-in leak prevention ensures augmented samples never appear in validation folds during cross-validation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Usage Modes](#usage-modes)
4. [API Reference](#api-reference)
5. [Best Practices](#best-practices)
6. [Examples](#examples)
7. [Technical Details](#technical-details)

## Quick Start

### Basic Augmentation

```json
{
  "steps": [
    {
      "sample_augmentation": {
        "transformers": [
          { "StandardScaler": {} }
        ],
        "count": 2
      }
    }
  ]
}
```

```yaml
steps:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
      count: 2  # Create 2 augmented samples per base sample
```

### Balanced Augmentation

```json
{
  "steps": [
    {
      "sample_augmentation": {
        "transformers": [
          { "StandardScaler": {} },
          { "MinMaxScaler": {} }
        ],
        "balance": "y",
        "target_size": 100
      }
    }
  ]
}
```

```yaml
steps:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
        - MinMaxScaler: {}
      balance: "y"  # Balance based on target variable
      target_size: 100  # Target samples per class
```

## Architecture Overview

### Components

1. **SampleAugmentationController**: Orchestrates augmentation strategy
2. **TransformerMixinController**: Executes transformations on samples
3. **Dataset API**: Manages augmented samples with origin tracking
4. **Indexer**: Tracks relationships between base and augmented samples
5. **Split Controllers**: Ensure leak prevention in CV splits

### Data Flow

```
Base Samples (n samples)
    ↓
SampleAugmentationController
    ↓ (delegates to)
TransformerMixinController (applies transformations)
    ↓
Dataset.augment_samples() (stores with origin tracking)
    ↓
Augmented Dataset (n base + m augmented samples)
    ↓
Split Controller (uses only base samples for splitting)
    ↓
Training Folds (can access augmented samples)
Validation Folds (only base samples, leak-free)
```

## Usage Modes

### Standard Mode (Count-Based)

Creates a fixed number of augmented samples per base sample.

```json
{
  "sample_augmentation": {
    "transformers": [
      { "StandardScaler": {} },
      { "MinMaxScaler": {} }
    ],
    "count": 3,
    "selection": "random",
    "random_state": 42
  }
}
```

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
    - MinMaxScaler: {}
  count: 3  # 3 augmentations per sample
  selection: "random"  # or "all"
  random_state: 42
```

**Parameters:**
- `count` (int): Number of augmentations per sample
- `selection` (str):
  - `"random"`: Randomly assign transformers (default)
  - `"all"`: Cycle through transformers systematically
- `random_state` (int): Random seed for reproducibility

**Behavior:**
- With `count=3` and 2 transformers + `selection="random"`:
  - Each sample gets 3 random transformer applications
  - Example: Sample 1 → `[T1, T2, T1]`, Sample 2 → `[T2, T1, T2]`

- With `count=3` and 2 transformers + `selection="all"`:
  - Transformers cycle through augmentations
  - Example: Sample 1 → `[T1, T2, T1]`, Sample 2 → `[T2, T1, T2]`

### Balanced Mode (Class-Aware)

Balances class distribution by augmenting minority classes. Three balancing strategies are available:

```json
{
  "sample_augmentation": {
    "transformers": [ { "StandardScaler": {} } ],
    "balance": "y",
    "target_size": 100,
    "selection": "random",
    "random_state": 42
  }
}
```

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"  # Balance on target variable
  target_size: 100  # Target samples per class (Strategy 1)
  selection: "random"
  random_state: 42
```

**Parameters:**
- `balance` (str): Column to balance on (`"y"` for targets or metadata column name)
- **Choose ONE balancing strategy:**
  - `target_size` (int): Fixed number of samples per class - each class augmented to exactly this size (no cap)
  - `max_factor` (float): Multiplier factor - each class augmented to `current_size * max_factor`, capped at majority class size
  - `ref_percentage` (float): Target as multiple of majority class (any positive value: 0.5, 0.8, 1.0, 1.5, etc.)
- `selection` (str): Same as standard mode
- `random_state` (int): Random seed

**Strategy 1 - Fixed Target Size:**
```json
{
  "sample_augmentation": {
    "transformers": [ { "StandardScaler": {} } ],
    "balance": "y",
    "target_size": 100
  }
}
```

Each class is augmented to exactly 100 samples (no cap at majority):
```
Initial: Class 0: 150 samples, Class 1: 30 samples, Class 2: 50 samples
Result:  Class 0: 150 samples (no change - already >= 100)
         Class 1: 100 samples (70 augmented)
         Class 2: 100 samples (50 augmented)
```

**Strategy 2 - Multiplier Factor (capped at majority):**
```json
{
  "sample_augmentation": {
    "transformers": [ { "StandardScaler": {} } ],
    "balance": "y",
    "max_factor": 3.0
  }
}
```

Each class is multiplied by the factor, but capped at majority class size:
```
Initial: Class 0: 100 samples (majority), Class 1: 20 samples, Class 2: 50 samples
Result:  Class 0: 100 samples (no augmentation - it's majority)
         Class 1: 60 samples (40 augmented, min(20 × 3, 100) = 60)
         Class 2: 100 samples (50 augmented, min(50 × 3, 100) = 100, capped)
```

**Strategy 3 - Reference Percentage:**
```json
{
  "sample_augmentation": {
    "transformers": [ { "StandardScaler": {} } ],
    "balance": "y",
    "ref_percentage": 0.8
  }
}
```

Target each class to be a multiple of the majority class:
```
Initial: Class 0: 100 samples (majority), Class 1: 30 samples, Class 2: 20 samples
Result:  Class 0: 100 samples (no change - it's the majority)
         Class 1: 80 samples (50 augmented, targeting 0.8 × 100)
         Class 2: 80 samples (60 augmented, targeting 0.8 × 100)
```

Reference percentage > 1.0 example:
```
Initial: Class 0: 100 samples (majority), Class 1: 30 samples, Class 2: 20 samples
ref_percentage: 1.5
Result:  Class 0: 150 samples (50 augmented, targeting 1.5 × 100)
         Class 1: 150 samples (120 augmented, targeting 1.5 × 100)
         Class 2: 150 samples (130 augmented, targeting 1.5 × 100)
```

### Binning for Regression (Automatic)

When balancing regression targets (`balance: "y"` with continuous target values), the system automatically converts continuous values into discrete bins to enable balanced augmentation. This is useful for regression datasets where you want to ensure augmentation is distributed across the full target range.

**Automatic Activation:**
- Triggers when `balance: "y"` AND dataset is detected as regression
- Bins are created only if `bins` or `binning_strategy` parameter is provided
- For classification tasks, binning is automatically skipped

```json
{
  "sample_augmentation": {
    "transformers": [ { "StandardScaler": {} } ],
    "balance": "y",
    "bins": 10,
    "binning_strategy": "equal_width",
    "max_factor": 2.0
  }
}
```

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"  # Regression targets
  bins: 10  # Create 10 virtual classes from continuous y
  binning_strategy: "equal_width"  # or "quantile" (default: "equal_width")
  max_factor: 2.0  # Balance strategy applies to bins
```

**Binning Strategies:**

1. **Equal Width (default)** - Uniform spacing
   ```
   Target range: [0, 100], bins=4
   Bin edges: [0, 25, 50, 75, 100]
   Bin 0: [0-25)
   Bin 1: [25-50)
   Bin 2: [50-75)
   Bin 3: [75-100]

   Best for: Uniform distributions
   ```

2. **Quantile** - Equal probability per bin
   ```
   Target values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], bins=3
   Bin edges: [1, 4, 8, 100]
   Bin 0: [1-4) - contains ~3 samples
   Bin 1: [4-8) - contains ~3 samples
   Bin 2: [8-100] - contains ~3 samples

   Best for: Skewed distributions
   ```

**Example:**
```yaml
# Regression dataset: continuous y values
# Initial distribution: 20 samples spread across range [0, 100]
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  bins: 5  # Create 5 virtual classes
  binning_strategy: "equal_width"
  max_factor: 1.0  # Equal samples per bin

# Result:
# - y binned into 5 ranges: [0-20), [20-40), [40-60), [60-80), [80-100]
# - Each bin augmented to have equal samples
# - Ensures augmentation covers full target range
```

**Parameters:**
- `bins` (int): Number of virtual classes (default: 10, range: 1-1000)
- `binning_strategy` (str):
  - `"equal_width"`: Uniform width bins (default)
  - `"quantile"`: Equal probability bins
- Other balancing parameters (`max_factor`, `target_size`, `ref_percentage`) apply normally after binning

## API Reference

### Pipeline Syntax

#### sample_augmentation Step

```json
{
  "sample_augmentation": {
    "transformers": [ { "StandardScaler": {} } ],
    "count": 1,
    "balance": "y",
    "target_size": 100,
    "max_factor": 3.0,
    "ref_percentage": 0.8,
    "bins": 10,
    "binning_strategy": "equal_width",
    "selection": "random",
    "random_state": 42
  }
}
```

```yaml
sample_augmentation:
  transformers: List[transformer_spec]  # Required

  # Standard mode
  count: int  # Number of augmentations per sample

  # OR Balanced mode (choose ONE strategy)
  balance: str  # "y" or metadata column name
  target_size: int  # Strategy 1: Fixed target samples per class
  max_factor: float  # Strategy 2: Multiplier factor (e.g., 3 means 3x size)
  ref_percentage: float  # Strategy 3: % of majority class (0.0-1.0)

  # Binning for regression (automatic when balance="y" and task is regression)
  bins: int  # Number of virtual classes (default: 10)
  binning_strategy: str  # "equal_width" (default) or "quantile"

  # Common options
  selection: str  # "random" (default) or "all"
  random_state: int  # Optional seed
````

### Dataset API

#### augment_samples()

```python
dataset.augment_samples(
    data: np.ndarray,  # Augmented feature data
    processings: List[str],  # Processing names
    augmentation_id: str,  # Unique augmentation ID
    selector: Optional[Dict] = None,  # Select samples to augment
    count: Union[int, List[int]] = 1  # Augmentation count
) -> List[int]
```

**Returns:** List of augmented sample IDs

**Example:**
```python
# Augment training samples
aug_data = transform(base_data)
dataset.augment_samples(
    data=aug_data,
    processings=["standardized"],
    augmentation_id="aug_001",
    selector={"partition": "train"},
    count=2
)
```

#### include_augmented Parameter

All dataset retrieval methods support `include_augmented` parameter:

```python
# Get all samples (base + augmented)
X_all = dataset.x({}, include_augmented=True)

# Get only base samples
X_base = dataset.x({}, include_augmented=False)

# Also available for:
y = dataset.y({}, include_augmented=False)
metadata = dataset.metadata(include_augmented=False)
col = dataset.metadata_column("group", {}, include_augmented=False)
```

### Indexer API

```python
# Get augmented samples for origin sample(s)
aug_indices = indexer.get_augmented_for_origins([0, 1, 2])

# Get origin sample for augmented sample
origin_idx = indexer.get_origin_for_sample(10)

# Filter indices by augmentation status
base_indices = indexer.x_indices({}, include_augmented=False)
all_indices = indexer.x_indices({}, include_augmented=True)
```

## Best Practices

### 1. Choose the Right Mode

- **Use Standard Mode when:**
  - You want uniform augmentation across all samples
  - Dataset is already balanced
  - You have specific augmentation requirements

- **Use Balanced Mode when:**
  - You have class imbalance
  - Minority classes need more representation
  - You want automatic balancing

### 2. Select Appropriate Transformers

Good choices for spectroscopy data:
- `StandardScaler`: Z-score normalization
- `MinMaxScaler`: Scale to `[0, 1]` range
- `RobustScaler`: Robust to outliers
- `MaxAbsScaler`: Scale by maximum absolute value

**Avoid:**
- Transformers that drastically change signal characteristics
- Too many transformers (increases computation)
- Incompatible transformers (check preprocessing chain)

### 3. Augmentation Count Guidelines

- **Small datasets (<100 samples)**: count=2-5
- **Medium datasets (100-1000 samples)**: count=1-3
- **Large datasets (>1000 samples)**: count=1-2 or use balanced mode only

**Warning:** Over-augmentation can lead to:
- Overfitting to augmented patterns
- Loss of original data characteristics
- Increased computation time

### 4. Balanced Mode Settings

Three balancing strategies are available. Choose the one that best fits your use case:

**Strategy 1 - Fixed Target Size:**
```json
{
  "balance": "y",
  "target_size": 100
}
```
Use when you want each class to have exactly a specific number of samples. Best for:
- Setting a fixed validation set size
- Ensuring consistent class sizes across experiments
- Specific domain requirements

**Strategy 2 - Multiplier Factor:**
```json
{
  "balance": "y",
  "max_factor": 3.0
}
```
Use when you want each class multiplied by a consistent factor. Best for:
- Conservative augmentation (e.g., 2x to 3x)
- Proportional growth (maintains relative class size differences)
- Aggressive augmentation with control

**Strategy 3 - Percentage of Majority:**
```json
{
  "balance": "y",
  "ref_percentage": 0.8
}
```
Use when you want minority classes to approach majority size. Best for:
- Balancing imbalanced datasets toward majority class
- Progressive augmentation strategies
- Legacy or familiar approaches

**Comparison Example:**

```
Initial distribution: Class A: 1000, Class B: 100, Class C: 50
```

Strategy 1 (`target_size: 500`):
- Class A: 1000 → 1000 (no augmentation needed)
- Class B: 100 → 500 (400 augmented)
- Class C: 50 → 500 (450 augmented)

Strategy 2 (`max_factor: 2.0`, capped at majority 1000):
- Class A: 1000 → 1000 (no augmentation - it's majority)
- Class B: 100 → min(200, 1000) = 200 (100 augmented)
- Class C: 50 → min(100, 1000) = 100 (50 augmented)

Strategy 3 (`ref_percentage: 0.5`, with majority 1000):
- Class A: 1000 → 1000 (no change - reference class)
- Class B: 100 → 500 (400 augmented, targeting 0.5 × 1000)
- Class C: 50 → 500 (450 augmented, targeting 0.5 × 1000)

Strategy 3 with value > 1.0 (`ref_percentage: 1.2`, with majority 1000):
- Class A: 1000 → 1200 (200 augmented, targeting 1.2 × 1000)
- Class B: 100 → 1200 (1100 augmented, targeting 1.2 × 1000)
- Class C: 50 → 1200 (1150 augmented, targeting 1.2 × 1000)

### 5. Random State for Reproducibility

Always set `random_state` for reproducible experiments:

```json
{
  "sample_augmentation": {
    "transformers": [ "..." ],
    "count": 2,
    "random_state": 42
  }
}
```

```yaml
sample_augmentation:
  transformers: [...]
  count: 2
  random_state: 42  # Reproducible results
```

## Examples

### Example 1: Basic Augmentation with Single Transformer

```yaml
pipeline:
  - preprocessing:
      - SNV: {}

  - sample_augmentation:
      transformers:
        - StandardScaler: {}
      count: 2
      selection: "all"

  - split:
      - StratifiedKFold:
          n_splits: 5
          shuffle: true
          random_state: 42

  - model:
      - PLSRegression:
          n_components: 10
```

**Result:**
- Original: 100 samples
- After augmentation: 300 samples (100 base + 200 augmented)
- CV splits use only 100 base samples
- Training folds access all 300 samples

### Example 2: Multiple Transformers with Random Selection

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
        - MinMaxScaler: {}
        - RobustScaler: {}
      count: 3
      selection: "random"
      random_state: 42
```

**Result:**
- Each base sample gets 3 augmentations
- Transformers randomly selected for each augmentation
- Different runs produce different augmentations (deterministic with same seed)

### Example 3: Balanced Augmentation with Fixed Target Size

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
        - MinMaxScaler: {}
      balance: "y"
      target_size: 100  # Each class will have 100 samples
      selection: "random"
      random_state: 42
```

**Scenario:**
- Class 0: 100 samples → 100 samples (unchanged, already at target)
- Class 1: 30 samples → 100 samples (70 augmented)
- Class 2: 20 samples → 100 samples (80 augmented)

### Example 4: Balanced Augmentation with Multiplier Factor

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
        - MinMaxScaler: {}
      balance: "y"
      max_factor: 3.0  # Each class multiplied by 3, capped at majority size
      selection: "random"
      random_state: 42
```

**Scenario (majority class = 100):**
- Class 0: 100 samples (majority) → 100 samples (no augmentation)
- Class 1: 30 samples → min(90, 100) = 90 samples (60 augmented)
- Class 2: 20 samples → min(60, 100) = 60 samples (40 augmented)

### Example 5: Balanced Augmentation with Reference Percentage

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
        - MinMaxScaler: {}
      balance: "y"
      ref_percentage: 0.8  # Target 80% of majority class
      selection: "random"
      random_state: 42
```

**Scenario:**
- Class 0: 100 samples (majority) → 100 samples (unchanged)
- Class 1: 30 samples → 80 samples (50 augmented, targeting 0.8 × 100)
- Class 2: 20 samples → 80 samples (60 augmented, targeting 0.8 × 100)

### Example 6: Sequential Augmentation Rounds

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
      count: 1
      selection: "all"

  - sample_augmentation:
      transformers:
        - MinMaxScaler: {}
      count: 1
      selection: "all"
```

**Result:**
- First round: 100 base → 200 total (100 base + 100 augmented)
- Second round: Augments same 100 base → 300 total (100 base + 200 augmented)
- Each augmentation round only targets original base samples

### Example 7: Augmentation with Metadata Filtering

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
      count: 2
      context:
        partition: "train"  # Only augment training samples
```

## Technical Details

### Leak Prevention Mechanism

**Problem:** Without leak prevention, augmented samples could appear in both training and validation folds, leading to overoptimistic performance estimates.

**Solution:** nirs4all implements automatic leak prevention:

1. **Origin Tracking:** Every augmented sample stores its origin sample index
2. **Two-Phase Selection:**
   - CV splits use `include_augmented=False` → only base samples for splitting
   - Training uses `include_augmented=True` → access base + augmented samples
3. **Metadata Inheritance:** Augmented samples inherit all metadata from origins

**Verification:**
```python
# After augmentation and splitting
assert all(idx < num_base_samples for idx in validation_indices)
# Validation never sees augmented samples
```

### Multi-Round Augmentation

Sequential augmentation operations always target only base samples:

```python
# Round 1
dataset.augment_samples(...)  # Uses include_augmented=False internally

# Round 2
dataset.augment_samples(...)  # Still only augments original base samples
```

This prevents "augmentation of augmentations" which would drift from original data.

### Memory Considerations

- **Augmented samples stored separately:** No duplication of base data
- **Metadata inherited by reference:** Minimal overhead
- **Processing data:** Each augmentation stores transformed features

**Memory formula:**
```
Memory ≈ base_samples_size + (augmentation_count × features_size)
```

### Performance Tips

1. **Limit augmentation count:** More augmentations = longer training
2. **Use balanced mode judiciously:** Calculates augmentation needs automatically
3. **Consider computational cost:** Each transformer adds processing time
4. **Batch augmentation:** Pipeline processes augmentation once, reuses for all folds

## Troubleshooting

### Common Issues

#### 1. "Processing 'X' not found"
**Cause:** Transformer produces different processing name than expected

**Solution:** Check transformer output processing names match pipeline expectations

#### 2. High memory usage
**Cause:** Too many augmented samples

**Solution:**
- Reduce `count` parameter
- Use `max_factor` in balanced mode
- Process in batches if needed

#### 3. Poor model performance
**Cause:** Over-augmentation or inappropriate transformers

**Solution:**
- Reduce augmentation count
- Try different transformers
- Use validation curve to find optimal augmentation level

#### 4. Inconsistent results across runs
**Cause:** Missing `random_state`

**Solution:** Always set `random_state` for reproducibility

## Migration Guide

### From Previous Versions

If you were using custom augmentation:

**Before:**
```python
# Manual augmentation (error-prone)
for fold in cv_splits:
    train_data = X[fold[0]]
    aug_data = transform(train_data)
    combined = np.vstack([train_data, aug_data])
    model.fit(combined, y)
```

**After:**
```json
{
  "pipeline": [
    {
      "sample_augmentation": {
        "transformers": [ { "StandardScaler": {} } ],
        "count": 1
      }
    },
    { "split": [ "..." ] },
    { "model": [ "..." ] }
  ]
}
```

```yaml
# Automatic with leak prevention
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
      count: 1
  - split: [...]
  - model: [...]
```

## Implementation Details

### Architecture Overview (Developer Reference)

The sample augmentation feature uses a **delegation pattern** where the SampleAugmentationController orchestrates the distribution strategy, and TransformerMixinController executes the actual transformations.

**Data Flow:**
```
Pipeline → SampleAugmentationController
  ├─ Calculates augmentation plan (standard or balanced)
  ├─ Builds transformer→sample_indices distribution map
  └─ Emits ONE run_step per transformer with target_samples list
       ↓
     TransformerMixinController
       ├─ Detects augment_sample action in context
       ├─ For each sample in target_samples:
       │  ├─ Retrieves origin sample data (all sources)
       │  ├─ Applies transformer to all processings
       │  └─ Calls dataset.augment_samples() to store result
       └─ Returns fitted transformers for serialization
            ↓
          Dataset with augmented samples + origin tracking
            ↓
          CrossValidator uses include_augmented=False
            ├─ Creates folds from base samples only
            └─ Training accesses base + augmented (leak-free!)
```

### Components

**BalancingCalculator** (`nirs4all/utils/balancing.py`)
- `calculate_balanced_counts()`: Computes augmentation needs per sample
  - Strategies: target_size, max_factor, ref_percentage
  - Multi-class support
  - Edge case handling (already balanced, impossible targets)

- `calculate_balanced_counts_value_aware()`: Two-level balancing for binned regression
  - Level 1: Fair distribution across unique y-values within a bin
  - Level 2: Fair distribution within each value group across samples
  - Prevents value imbalance when multiple samples share same y-value

**Indexer Enhancements** (`nirs4all/dataset/indexer.py`)
- `get_augmented_for_origins()`: Find augmented samples by origin
- `get_origin_for_sample()`: Find origin of augmented sample
- `x_indices()`: Two-phase selection with `include_augmented` parameter
  - Phase 1: Filter base samples (origin=null)
  - Phase 2: Aggregate augmented versions if requested

**Dataset API** (`nirs4all/dataset/dataset.py`)
- All methods support `include_augmented` parameter (default=True)
- `augment_samples()`: Add augmented samples with origin tracking
- Automatic metadata inheritance from origin samples

**SampleAugmentationController** (`nirs4all/controllers/dataset/op_sample_augmentation.py`)
- Detects standard vs. balanced mode
- Orchestrates distribution calculation
- Delegates to TransformerMixinController via run_step

**TransformerMixinController** Enhancement (`nirs4all/controllers/sklearn/op_transformermixin.py`)
- Detects `augment_sample=True` in context (like add_feature)
- Applies transformers to origin samples
- Calls dataset.augment_samples() to persist results

**Split Controller** (`nirs4all/controllers/sklearn/op_split.py`)
- Uses `include_augmented=False` for all data retrieval
- Creates folds from base samples only
- Prevents data leakage

### Advanced Features

#### Value-Aware Bin Balancing

When balancing regression data with binning, multiple samples can share the same y-value. Standard sample-aware balancing may create imbalance:

**Example:**
```
Bin 5: 3 samples
  Sample A: y=18.3999 (value 1)
  Sample B: y=18.5    (value 2)
  Sample C: y=18.5    (value 2)

Target: 10 total samples (need 7 augmentations)

Sample-aware result (imbalanced):
  Sample A: 6 total
  Sample B: 14 total  ← y=18.5 overrepresented
  Sample C: 0 total

Value-aware result (fair):
  Sample A: 5 total
  Sample B: 3 total
  Sample C: 2 total   ← y=18.5 more balanced
```

**Enable with:**
```json
{
  "sample_augmentation": {
    "transformers": [{"Rotate_Translate": {}}],
    "balance": "y",
    "bins": 5,
    "bin_balancing": "value"
  }
}
```

#### Random Remainder Selection

When augmentation counts don't divide evenly, remainders are randomly distributed instead of always going to first samples:

**Example:**
```
5 samples need 12 augmentations total
12 ÷ 5 = 2 base + 2 remainder

Random selection (reproducible with seed):
  Seed 42: Samples [100, 101] get 3 augmentations (bonus)
  Seed 99: Samples [102, 104] get 3 augmentations (bonus)

Benefits:
  - Fair selection across all samples
  - No systematic bias toward first samples
  - Reproducible with random_state
```

### Testing

**Test Coverage:**
- BalancingCalculator: 27+ tests covering all strategies
- Indexer: 23+ tests for augmented sample management
- Dataset API: 18+ tests for include_augmented behavior
- SampleAugmentationController: 19+ tests for distribution logic
- TransformerMixinController: 12+ tests for augmentation execution
- Split Controllers: 11+ tests for leak prevention
- Integration: 7+ end-to-end tests
- **Total: 120+ tests, all passing**

**Leak Prevention Validation:**
- Augmented samples never appear in opposite CV fold
- Validation fold isolation verified
- Multi-round augmentation safety

### Performance

**Memory Usage Formula:**
```
Memory ≈ base_samples_size + (augmentation_count × features_size)
```

**Computational Complexity:**
- Standard mode: O(n × count × transformers)
- Balanced mode: O(minority_samples × factor × transformers)
- Splitting: O(n_base) - augmented samples excluded

**Recommendations:**
- **Small datasets (<100 samples)**: count=2-5
- **Medium datasets (100-1000 samples)**: count=1-3
- **Large datasets (>1000 samples)**: count=1-2 or balanced mode only
- Start with 1-2 transformers to assess computational cost

### Common Pitfalls

1. **Over-augmentation**: count > 5 usually unnecessary and increases training time
2. **Wrong transformer**: Some transformers drastically change signal, test on validation set
3. **Missing random_state**: Always set for reproducible experiments
4. **Leakage**: Verify CV splits use `include_augmented=False` (automatic in framework)
5. **Memory overflow**: Large augmentation counts consume significant memory

### Serialization

Augmented samples persist across save/load cycles:
- Indexer DataFrame stores origin/augmentation metadata
- Features stored separately (efficient)
- Metadata inherited automatically
- Transformers serialized as pickles

## References

- Full Guide: This document
- Quick Reference: `docs/SAMPLE_AUGMENTATION_QUICK_REFERENCE.md`
- Examples: `examples/Q12_sample_augmentation.py`
- Tests: `tests/unit/test_balancing.py`, `tests/unit/test_sample_augmentation_controller.py`

## Support

For questions or issues:
1. Check examples in `examples/` directory
2. Review test cases for usage patterns
3. Check implementation files for detailed docstrings
4. Open an issue on GitHub
