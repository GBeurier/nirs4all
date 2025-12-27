# Prediction Ranking System - Analysis and Roadmap

## Part 1: Desired Logic

### 1.1 Overview

The nirs4all prediction ranking system retrieves, ranks, and displays predictions from trained models. It supports:

1. **Multiple training scenarios**: Models trained on train partition, fold-based cross-validation, or ensemble predictions (averaging/weighted-averaging fold predictions)
2. **Multiple partitions**: Predictions are stored for `train`, `val`, and `test` partitions
3. **Metadata preservation**: Each prediction retains sample IDs and other metadata properties

### 1.2 Core Ranking Flow

The ranking flow should follow this **strict order**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: FILTER                                                              │
│  Apply user-provided filters (dataset_name, model_name, config_name, etc.)  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: SELECT PARTITION FOR RANKING                                       │
│  Filter to rank_partition (e.g., 'val') to get ranking scores               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: OPTIONAL SAMPLE AGGREGATION                                        │
│  If aggregate='sample_id', aggregate predictions by metadata column:        │
│  - Regression: average y_pred values for same sample_id                     │
│  - Classification: average y_proba, then argmax for y_pred                  │
│  - y_true: take first (or average if needed)                               │
│  → Recalculate rank_metric on aggregated data                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: GLOBAL SORT                                                        │
│  Sort ALL predictions by rank_score (ascending or descending per metric)   │
│  This establishes the GLOBAL ranking order                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: OPTIONAL GROUP-BY FILTERING                                        │
│  If grouping by variables (model_name, model_classname, preprocessings):   │
│  - Walk through SORTED list                                                 │
│  - For each unique combination of group variables, keep ONLY THE FIRST     │
│  - This preserves the global rank order while deduplicating                │
│                                                                             │
│  IMPORTANT: Do NOT group first then sort within groups!                    │
│  The correct approach is: SORT GLOBALLY → TAKE FIRST PER GROUP             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: APPLY TOP-N LIMIT                                                  │
│  Take the first N results from the filtered+sorted list                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: FETCH DISPLAY DATA                                                 │
│  For each selected prediction, fetch data from display_partition            │
│  Apply same aggregation if requested                                        │
│  Calculate display_metrics on display_partition data                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Detailed Requirements

#### 1.3.1 Basic Ranking (`top()`)

**Input Parameters:**
- `n`: Number of top predictions to return
- `rank_metric`: Metric for ranking (e.g., 'rmse', 'balanced_accuracy')
- `rank_partition`: Partition to rank on ('train', 'val', 'test')
- `display_partition`: Partition to display results from
- `display_metrics`: List of metrics to compute for display
- `ascending`: Sort order (inferred from metric if None)
- `**filters`: Additional filter criteria

**Expected Behavior:**
```python
# Example: Get top 5 models ranked by val RMSE, show test results
top_5 = predictions.top(
    n=5,
    rank_metric='rmse',
    rank_partition='val',
    display_partition='test'
)
# Returns 5 predictions, sorted by val_rmse (ascending),
# but showing y_true/y_pred from test partition
```

#### 1.3.2 Sample Aggregation (`aggregate` parameter)

When the dataset has multiple scans per sample (e.g., 4 measurements for each sample ID), predictions should be aggregated before metric calculation.

**Input:** `aggregate='sample_id'` (or any metadata column)

**Expected Behavior:**
1. For each unique sample_id, collect all y_pred values
2. **Regression**: `aggregated_y_pred = mean(y_pred_values)`
3. **Classification**:
   - If y_proba available: `aggregated_y_proba = mean(y_proba_values)`, then `aggregated_y_pred = argmax(aggregated_y_proba)`
   - If no y_proba: `aggregated_y_pred = majority_vote(y_pred_values)`
4. **y_true**: Take the value (should be identical for same sample)
5. **Recalculate metric** on aggregated arrays

```python
# Example: Aggregate 4 scans per sample before ranking
top_5 = predictions.top(
    n=5,
    rank_metric='rmse',
    rank_partition='val',
    aggregate='sample_id'  # Aggregate by metadata column 'sample_id'
)
```

#### 1.3.3 Group-By Filtering (`best_per_model`, heatmap grouping)

When charts need one result per model_name (or other grouping variable):

**CORRECT Approach:**
1. Get ALL predictions
2. Sort by rank_metric globally
3. Walk through sorted list, keeping ONLY the first occurrence of each group
4. Apply top-N limit

**INCORRECT Approach (current bug):**
1. Group predictions by variable ❌
2. Sort within each group ❌
3. Take best from each group ❌

This incorrect approach loses the global ranking and can show a model that should be ranked #15 as appearing before one ranked #3.

```python
# Example: Heatmap with one cell per (model_name, preprocessings) combination
# For each cell, show the BEST prediction from that combination
# Rankings should reflect global order, not per-group order
```

#### 1.3.4 Chart-Specific Requirements

| Chart Type | Group-By Behavior | Aggregation Behavior |
|------------|-------------------|----------------------|
| **top_k** | `best_per_model=True`: One result per model_name | Apply aggregation before ranking |
| **confusion_matrix** | `best_per_model=True`: One result per model_name | Apply aggregation before ranking |
| **heatmap** | Group by (x_var, y_var), show best per cell | Apply aggregation before ranking |
| **candlestick** | Group by variable, show distribution | Keep ALL predictions (distribution) |
| **histogram** | No grouping, show all scores | Keep ALL predictions (distribution) |

### 1.4 Data Flow Diagram

```
                    ┌───────────────────────────┐
                    │    Predictions Storage    │
                    │  (Polars DataFrame + ID)  │
                    └───────────────────────────┘
                               │
                               ▼
                    ┌───────────────────────────┐
                    │      PredictionRanker     │
                    │        .top() method      │
                    └───────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
    │ TopKChart   │    │  HeatmapChart   │    │HistogramChart│
    │ ConfMatrix  │    │   Candlestick   │    │              │
    └─────────────┘    └─────────────────┘    └──────────────┘
    best_per_model=T   Group by (x,y)         No grouping
```

### 1.5 Key Invariants

1. **Global Sort First**: Rankings are ALWAYS computed globally before any grouping
2. **Aggregation Before Ranking**: If `aggregate` is provided, apply it BEFORE calculating rank_metric
3. **Preserve Order**: Group-by filtering walks through sorted list, never re-sorts
4. **Consistent Results**: Same parameters → same results across all charts
5. **Partition Independence**: Ranking uses rank_partition, display uses display_partition
6. **Metric Direction**: ascending=True for RMSE/MAE (lower=better), False for R²/accuracy (higher=better)

---

## Part 2: Current State of the Code

### 2.1 Architecture Overview

The prediction ranking system is spread across multiple modules:

```
nirs4all/
├── data/
│   ├── predictions.py              # Main Predictions facade class
│   └── _predictions/
│       ├── ranker.py               # PredictionRanker - core ranking logic
│       ├── aggregator.py           # PartitionAggregator - partition data combining
│       ├── storage.py              # PredictionStorage - DataFrame backend
│       ├── indexer.py              # PredictionIndexer - filtering
│       ├── result.py               # PredictionResult, PredictionResultsList
│       └── serializer.py           # JSON/Parquet serialization
└── visualization/
    ├── predictions.py              # PredictionAnalyzer - orchestrator
    └── charts/
        ├── base.py                 # BaseChart - common functionality
        ├── heatmap.py              # HeatmapChart
        ├── top_k_comparison.py     # TopKComparisonChart
        ├── confusion_matrix.py     # ConfusionMatrixChart
        ├── candlestick.py          # CandlestickChart
        └── histogram.py            # ScoreHistogramChart
```

### 2.2 Current Implementation Analysis

#### 2.2.1 PredictionRanker.top() - The Core ([ranker.py](../../nirs4all/data/_predictions/ranker.py))

**What it does correctly:**
- ✅ Filters by rank_partition to get ranking data
- ✅ Computes rank_score from scores JSON or y_true/y_pred
- ✅ Applies ascending/descending based on metric type
- ✅ Uses tiebreaker scores for equal rank_scores
- ✅ Has `aggregate` parameter for sample aggregation
- ✅ Has `best_per_model` parameter for deduplication

**Issues identified:**

1. **Aggregation applied AFTER initial score extraction for non-aggregate path:**
   ```python
   # Lines 340-375: Non-aggregate path extracts pre-computed scores
   # Aggregation is only applied in the "if aggregate:" block
   # This means ranking uses NON-aggregated scores even when aggregate is requested!
   ```

2. **Group-by logic in `best_per_model` is simplistic:**
   ```python
   # Lines 419-426
   if best_per_model:
       seen_models = set()
       filtered_scores = []
       for rs in rank_scores:
           model_name = rs.get("model_name", "").lower()
           if model_name not in seen_models:
               seen_models.add(model_name)
               filtered_scores.append(rs)
       rank_scores = filtered_scores
   ```
   - Only groups by `model_name`, not arbitrary variables
   - Comes AFTER sorting ✅, so order is correct
   - But doesn't support multi-variable grouping (e.g., model_name + preprocessings)

3. **Aggregation applied inconsistently for rank vs display:**
   - Ranking: Aggregation applied if `aggregate` is set (lines 340-370)
   - Display: Aggregation applied again for display partitions (lines 450-500)
   - BUT: The rank_score used for sorting might not reflect aggregated values

4. **Score extraction relies on JSON parsing regex or precomputed fields:**
   ```python
   # Fast path uses pre-computed scores from 'scores' JSON column
   # This doesn't account for aggregation which requires recalculation
   ```

#### 2.2.2 HeatmapChart - The Most Complex ([heatmap.py](../../nirs4all/visualization/charts/heatmap.py))

**What it does:**
- Has two render paths: fast (Polars-optimized) and slow (with aggregation)
- Uses `_render_with_aggregation()` when `aggregate` is provided

**Issues identified:**

1. **Fast path (`render()`) doesn't call `predictions.top()`:**
   ```python
   # Lines 423-580: Uses raw Polars operations
   # Builds its own ranking logic with group_by + agg
   # Doesn't benefit from consistent ranking in PredictionRanker
   ```

2. **Slow path (`_render_with_aggregation()`) has ranking bugs:**
   ```python
   # Line 987-1010: Gets ALL predictions with large n
   all_top_preds = self.predictions.top(
       n=10000,
       ...
       best_per_model=True,  # ← PROBLEM: This deduplicates to one per model_name
   )
   ```
   - Uses `best_per_model=True` which keeps only one per `model_name`
   - But heatmap might need grouping by OTHER variables (e.g., `model_classname`, `preprocessings`)
   - The grouping logic after this (lines 1020-1040) groups by `y_var` which might differ from `model_name`

3. **Matrix building uses wrong group-by approach:**
   ```python
   # Lines 1020-1040: Groups by y_var value
   y_var_to_best_pred = {}
   for pred in all_top_preds:
       y_val = pred.get(y_var, 'Unknown')
       y_val_str = str(y_val).lower() if y_var in ['model_name', 'model_classname'] else str(y_val)
       if y_val_str not in y_var_to_best_pred:
           y_var_to_best_pred[y_val_str] = pred
   ```
   - This only groups by ONE variable (y_var)
   - Doesn't handle (x_var, y_var) combinations properly
   - If x_var='partition', it puts ALL partitions in one row incorrectly

4. **Column filling logic is broken for non-partition x_var:**
   ```python
   # Lines 1100-1120: Fills matrix for partition-grouped case
   # But the else branch (non-partition x_var) doesn't work:
   if x_labels:
       x_val = pred.get(x_var, x_labels[0])  # ← Gets x_var from pred
       # But we already grouped by y_var, so we lose x_var variation!
   ```

#### 2.2.3 TopKComparisonChart ([top_k_comparison.py](../../nirs4all/visualization/charts/top_k_comparison.py))

**What it does correctly:**
- ✅ Calls `predictions.top()` with proper parameters
- ✅ Uses `best_per_model=True` (appropriate for TopK)
- ✅ Uses `aggregate_partitions=True` to get all partition data

**Minor issues:**
- Relies on PredictionRanker's correct behavior (which has bugs)

#### 2.2.4 ConfusionMatrixChart ([confusion_matrix.py](../../nirs4all/visualization/charts/confusion_matrix.py))

**What it does correctly:**
- ✅ Calls `predictions.top()` with proper parameters
- ✅ Uses `best_per_model=True`
- ✅ Handles aggregation via `aggregate` parameter

**Same dependency issue:**
- Relies on PredictionRanker working correctly

#### 2.2.5 CandlestickChart and HistogramChart

**What they do correctly:**
- ✅ Use `best_per_model=False` (correct for distribution charts)
- ✅ Get ALL predictions when aggregation is needed

**Issues:**
- Still call `predictions.top()` with aggregation, which has the ranking bugs
- The ranking itself isn't critical for these (they show distributions)

### 2.3 Specific Bugs Identified

#### Bug 1: Aggregation Not Applied to Ranking Scores

**Location:** [ranker.py](../../nirs4all/data/_predictions/ranker.py) lines 320-380

**Description:** When `aggregate` is provided, the ranking SHOULD use aggregated scores. Currently:
- The fast path extracts pre-computed scores from JSON (line 350-360)
- These pre-computed scores are NOT aggregated
- Aggregation is applied later for display, but ranking uses wrong values

**Impact:** Models are ranked by NON-aggregated performance even when user requests aggregation.

**Example:**
```python
# User expects: Rank by aggregated RMSE (4 scans → 1 sample)
top = predictions.top(n=5, rank_metric='rmse', aggregate='sample_id')
# Actually gets: Rank by non-aggregated RMSE, then aggregation applied after
```

#### Bug 2: Heatmap Uses Wrong Grouping Strategy

**Location:** [heatmap.py](../../nirs4all/visualization/charts/heatmap.py) lines 987-1120

**Description:** The `_render_with_aggregation()` method:
1. Gets predictions with `best_per_model=True` (groups by model_name)
2. Then tries to re-group by y_var (which might be different, e.g., model_classname)
3. This double-grouping loses data

**Example:**
```python
# x_var='partition', y_var='model_classname'
# Expected: One row per model_classname, columns for train/val/test
# Actual:
#   - best_per_model=True keeps one per model_name (e.g., 'pls_5_snv')
#   - Then groups by model_classname (e.g., 'PLSRegression')
#   - If two model_names have same model_classname, second one is lost
```

#### Bug 3: Fast Path vs Slow Path Inconsistency

**Location:** [heatmap.py](../../nirs4all/visualization/charts/heatmap.py)

**Description:** Two different code paths produce different results:
- Fast path (no aggregate): Uses Polars group_by operations
- Slow path (with aggregate): Uses predictions.top() + manual grouping

**Impact:** Same visual request gives different rankings depending on whether aggregate is used.

#### Bug 4: Missing Multi-Variable Grouping Support

**Location:** [ranker.py](../../nirs4all/data/_predictions/ranker.py)

**Description:** `best_per_model` only supports grouping by `model_name`. Heatmaps need:
- Grouping by arbitrary y_var (model_name, model_classname, preprocessings, etc.)
- Optionally grouping by (x_var, y_var) combinations

**Current workaround:** Heatmap tries to do this post-hoc, but loses global ranking order.

#### Bug 5: Sort-Then-Group vs Group-Then-Sort Confusion

**Location:** [heatmap.py](../../nirs4all/visualization/charts/heatmap.py) lines 1020-1040

**Description:** The current logic:
```python
# predictions are already sorted by rank (from top())
y_var_to_best_pred = {}
for pred in all_top_preds:
    y_val = pred.get(y_var, 'Unknown')
    if y_val_str not in y_var_to_best_pred:
        y_var_to_best_pred[y_val_str] = pred  # Takes first (correct!)
```

This is actually correct in principle (takes first per group from sorted list), BUT:
- It only groups by ONE dimension (y_var)
- It doesn't populate the matrix correctly for (x_var, y_var) cells

### 2.4 Test Case Failures

Based on the notebook usage, here are the failing scenarios:

1. **Top models with aggregation:**
   ```python
   top_models = predictions.top(n=5, rank_metric='rmse', rank_partition='val', aggregate="ID")
   ```
   - Expected: Top 5 models ranked by aggregated RMSE
   - Actual: Ranking might use non-aggregated scores

2. **Heatmap with partition on x-axis:**
   ```python
   analyzer.plot_heatmap(
       x_var="partition",
       y_var="model_name",
       rank_metric=rank_metric,
       aggregate="ID"
   )
   ```
   - Expected: Each row = model, each column = partition, cell = aggregated score
   - Actual: Missing models, wrong scores, inconsistent with top()

3. **Heatmap grouped by model_classname:**
   - Expected: One row per model_classname, showing best model of that class
   - Actual: Might show wrong model due to grouping bugs

### 2.5 Code Quality Issues

1. **Duplicated aggregation logic:**
   - `_apply_aggregation()` in ranker.py
   - `Predictions.aggregate()` static method in predictions.py
   - Different code paths with potential inconsistencies

2. **Complex conditional logic:**
   - Multiple if/else branches for aggregate vs non-aggregate
   - Hard to follow and maintain

3. **Inconsistent parameter naming:**
   - `best_per_model` in ranker.py
   - `group_by` concept not explicit in API
   - Heatmap uses `y_var` for grouping which is confusing

4. **Missing abstraction:**
   - No dedicated "GroupBy" concept for ranking
   - Each chart implements its own grouping logic

---

## Part 3: Roadmap for Production-Ready Ranking System

### 3.1 Phased Approach

| Phase | Focus | Risk | Effort |
|-------|-------|------|--------|
| **Phase 1** | Fix core ranking bugs | Low | Medium |
| **Phase 2** | Unify chart data retrieval | Medium | High |
| **Phase 3** | Add multi-variable grouping | Low | Medium |
| **Phase 4** | Clean up & documentation | Low | Low |

---

### 3.2 Phase 1: Fix Core Ranking Bugs (Priority: CRITICAL)

**Goal:** Make `PredictionRanker.top()` work correctly with aggregation.

#### Task 1.1: Fix Aggregation in Ranking Path

**File:** [ranker.py](../../nirs4all/data/_predictions/ranker.py)

**Current Problem:** Ranking uses pre-computed scores even when aggregation is requested.

**Solution:**
```python
def top(self, ...):
    # ...existing filter code...

    # CRITICAL FIX: When aggregate is provided, ALWAYS recalculate
    # scores from arrays, not from pre-computed values

    rank_scores = []
    for row in rank_data.to_dicts():
        score = None

        # Load arrays for aggregation OR metric recalculation
        y_true = self._get_array(row, "y_true")
        y_pred = self._get_array(row, "y_pred")
        y_proba = self._get_array(row, "y_proba")
        metadata = json.loads(row.get("metadata", "{}"))

        if aggregate and y_true is not None and y_pred is not None:
            # Apply aggregation FIRST
            agg_y_true, agg_y_pred, agg_y_proba, was_aggregated = self._apply_aggregation(
                y_true, y_pred, y_proba, metadata, aggregate, row.get("model_name", "")
            )
            if was_aggregated:
                y_true, y_pred, y_proba = agg_y_true, agg_y_pred, agg_y_proba
            # Calculate metric on (potentially aggregated) arrays
            score = evaluator.eval(y_true, y_pred, rank_metric)
        else:
            # Fast path: use pre-computed scores
            score = self._get_precomputed_score(row, rank_metric, rank_partition)
            if score is None and y_true is not None and y_pred is not None:
                score = evaluator.eval(y_true, y_pred, rank_metric)

        rank_scores.append({...})
```

**Acceptance Criteria:**
- [ ] With `aggregate='ID'`, ranking uses aggregated scores
- [ ] Without `aggregate`, ranking uses fast path (pre-computed scores)
- [ ] Same results for same parameters across multiple calls

#### Task 1.2: Add `group_by` Parameter to `top()`

**File:** [ranker.py](../../nirs4all/data/_predictions/ranker.py)

**Current Problem:** `best_per_model` only groups by `model_name`. Charts need arbitrary grouping.

**Solution:** Replace `best_per_model` with more flexible `group_by` parameter:

```python
def top(
    self,
    n: int,
    rank_metric: str = "",
    rank_partition: str = "val",
    ...
    group_by: Optional[Union[str, List[str]]] = None,  # NEW: Replaces best_per_model
    # best_per_model: bool = False,  # DEPRECATED: Use group_by=['model_name']
    ...
) -> PredictionResultsList:
    """
    ...
    Args:
        group_by: Group predictions and keep only the best per group.
                 Can be a single column name or list of column names.
                 - group_by='model_name': One result per model_name (same as best_per_model=True)
                 - group_by=['model_name', 'preprocessings']: One per combination
                 - group_by=None: No grouping, return all (default)
    ...
    """
    # ... sorting code ...

    # Apply group-by filtering
    if group_by is not None:
        if isinstance(group_by, str):
            group_by = [group_by]

        seen_groups = set()
        filtered_scores = []
        for rs in rank_scores:  # Already sorted!
            group_key = tuple(str(rs.get(col, '')).lower() for col in group_by)
            if group_key not in seen_groups:
                seen_groups.add(group_key)
                filtered_scores.append(rs)
        rank_scores = filtered_scores

    # Backward compatibility
    elif best_per_model:  # Deprecated
        group_by = ['model_name']
        # ... same logic ...
```

**Acceptance Criteria:**
- [ ] `group_by='model_name'` gives same result as `best_per_model=True`
- [ ] `group_by=['model_classname']` groups by model class
- [ ] `group_by=['x_var', 'y_var']` works for heatmap use case
- [ ] Group-by respects global sort order (takes first per group)

#### Task 1.3: Add Helper Method for Score Extraction

**File:** [ranker.py](../../nirs4all/data/_predictions/ranker.py)

**Current Problem:** Score extraction logic is duplicated and complex.

**Solution:**
```python
def _get_precomputed_score(
    self,
    row: Dict[str, Any],
    metric: str,
    partition: str
) -> Optional[float]:
    """
    Get pre-computed score from row, using scores JSON or legacy fields.

    Priority:
    1. scores JSON: {"val": {"rmse": 0.5}}
    2. Legacy field: val_score (if metric matches row's metric)
    """
    # Try scores JSON first
    scores_json = row.get("scores")
    if scores_json:
        try:
            scores_dict = json.loads(scores_json)
            if partition in scores_dict and metric in scores_dict[partition]:
                return float(scores_dict[partition][metric])
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # Fallback to legacy field
    if metric == row.get("metric"):
        return row.get(f"{partition}_score")

    return None
```

---

### 3.3 Phase 2: Unify Chart Data Retrieval (Priority: HIGH)

**Goal:** All charts use the same data retrieval path through `predictions.top()`.

#### Task 2.1: Simplify HeatmapChart

**File:** [heatmap.py](../../nirs4all/visualization/charts/heatmap.py)

**Current Problem:** Two code paths (fast/slow) with different behavior.

**Solution:** Use `predictions.top()` for ALL cases, with proper `group_by`:

```python
def render(
    self,
    x_var: str,
    y_var: str,
    rank_metric: str,
    ...
) -> Figure:
    # Determine grouping for this heatmap
    # For heatmap, we need one value per (x_var, y_var) cell
    # BUT we can't group by partition (it's a column in the output)

    if x_var == 'partition':
        # Special case: x-axis is partition
        # Group by y_var only, get all partitions separately
        group_by = [y_var]
        fetch_all_partitions = True
    elif y_var == 'partition':
        # Special case: y-axis is partition
        group_by = [x_var]
        fetch_all_partitions = True
    else:
        # Normal case: group by both
        group_by = [x_var, y_var]
        fetch_all_partitions = False

    # Get ranked predictions with proper grouping
    all_preds = self.predictions.top(
        n=10000,  # Large n to get all
        rank_metric=rank_metric,
        rank_partition=rank_partition,
        display_partition=display_partition,
        aggregate=aggregate,
        aggregate_partitions=fetch_all_partitions,
        group_by=group_by,  # NEW: Use group_by instead of best_per_model
        **filters
    )

    # Build matrix from predictions (no re-grouping needed!)
    matrix = self._build_matrix_from_predictions(
        all_preds, x_var, y_var, display_metric
    )
```

**Remove:** The `_render_with_aggregation()` method entirely. Merge logic into main `render()`.

#### Task 2.2: Standardize Chart Base Class

**File:** [base.py](../../nirs4all/visualization/charts/base.py)

**Add helper methods for common operations:**

```python
class BaseChart(ABC):

    def _get_ranked_predictions(
        self,
        n: int = 10000,
        rank_metric: Optional[str] = None,
        rank_partition: str = 'val',
        display_partition: str = 'test',
        aggregate: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
        **filters
    ) -> PredictionResultsList:
        """
        Common method for getting ranked predictions.

        Centralizes the call to predictions.top() with proper defaults.
        """
        if rank_metric is None:
            rank_metric = self._get_default_metric()

        return self.predictions.top(
            n=n,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            display_metrics=[rank_metric],
            aggregate=aggregate,
            aggregate_partitions=True,
            group_by=group_by,
            **filters
        )
```

#### Task 2.3: Update All Charts to Use Common Method

**Files:** All chart classes

**TopKComparisonChart:**
```python
def render(self, ...):
    top_predictions = self._get_ranked_predictions(
        n=k,
        rank_metric=rank_metric,
        rank_partition=rank_partition,
        aggregate=aggregate,
        group_by=['model_name'],  # One per model
        **filters
    )
```

**ConfusionMatrixChart:**
```python
def render(self, ...):
    top_predictions = self._get_ranked_predictions(
        n=k,
        rank_metric=rank_metric,
        rank_partition=rank_partition,
        aggregate=aggregate,
        group_by=['model_name'],  # One per model
        **ds_filters
    )
```

**CandlestickChart / HistogramChart:**
```python
def render(self, ...):
    all_predictions = self._get_ranked_predictions(
        n=10000,  # All predictions
        rank_metric=display_metric,
        rank_partition=display_partition,
        aggregate=aggregate,
        group_by=None,  # No grouping - show distribution
        **filters
    )
```

---

### 3.4 Phase 3: Add Multi-Variable Grouping (Priority: MEDIUM)

#### Task 3.1: Support Tuple Group Keys

**File:** [ranker.py](../../nirs4all/data/_predictions/ranker.py)

Already covered in Task 1.2, but ensure it handles:
- Case-insensitive comparison for string columns
- None/missing values gracefully
- Numeric columns (don't lowercase)

```python
def _make_group_key(
    self,
    row: Dict[str, Any],
    group_by: List[str]
) -> Tuple:
    """Create hashable group key from row values."""
    key_parts = []
    for col in group_by:
        val = row.get(col)
        if val is None:
            key_parts.append('')
        elif isinstance(val, str):
            key_parts.append(val.lower())
        else:
            key_parts.append(str(val))
    return tuple(key_parts)
```

#### Task 3.2: Support rank_agg Parameter (Best/Mean/Median)

**File:** [ranker.py](../../nirs4all/data/_predictions/ranker.py)

For some use cases, users want mean score per group, not just best:

```python
def top(
    self,
    ...
    group_by: Optional[Union[str, List[str]]] = None,
    group_agg: str = 'best',  # NEW: 'best', 'mean', 'median', 'worst'
    ...
):
    if group_by is not None:
        if group_agg == 'best':
            # Current behavior: take first (best) per group
            ...
        elif group_agg == 'mean':
            # Calculate mean score per group, rank groups by mean
            ...
```

**Note:** This is lower priority and can be deferred.

---

### 3.5 Phase 4: Cleanup & Documentation (Priority: LOW)

#### Task 4.1: Deprecate Old Parameters

**File:** [ranker.py](../../nirs4all/data/_predictions/ranker.py)

```python
import warnings

def top(
    self,
    ...
    best_per_model: bool = False,  # DEPRECATED
    group_by: Optional[Union[str, List[str]]] = None,
    ...
):
    # Handle deprecation
    if best_per_model:
        warnings.warn(
            "best_per_model is deprecated, use group_by=['model_name'] instead",
            DeprecationWarning,
            stacklevel=2
        )
        if group_by is None:
            group_by = ['model_name']
```

#### Task 4.2: Remove Dead Code

- Remove `_render_with_aggregation()` from heatmap.py once unified
- Remove duplicate aggregation logic (consolidate in one place)
- Remove unused parameters passed through filters

#### Task 4.3: Add Tests

**New file:** `tests/test_prediction_ranking.py`

```python
def test_top_with_aggregation():
    """Verify ranking uses aggregated scores when aggregate is provided."""
    ...

def test_top_group_by_single_column():
    """Verify group_by=['model_name'] keeps one per model."""
    ...

def test_top_group_by_multiple_columns():
    """Verify group_by=['model_name', 'preprocessings'] works."""
    ...

def test_heatmap_uses_correct_ranking():
    """Verify heatmap cells show correctly ranked predictions."""
    ...

def test_consistent_results_across_charts():
    """Verify same model appears in same rank across TopK, Heatmap, etc."""
    ...
```

#### Task 4.4: Update Documentation

**Files:** docstrings in all modified files

- Add examples for new `group_by` parameter
- Document the ranking flow
- Add troubleshooting section for common issues

---

### 3.6 Implementation Order

```
Week 1: Phase 1 (Core Fixes)
├── Task 1.1: Fix aggregation in ranking path
├── Task 1.2: Add group_by parameter
└── Task 1.3: Add score extraction helper

Week 2: Phase 2 (Chart Unification)
├── Task 2.1: Simplify HeatmapChart
├── Task 2.2: Standardize BaseChart
└── Task 2.3: Update all charts

Week 3: Phase 3 + 4 (Polish)
├── Task 3.1: Support tuple group keys (already done in heatmap, verify.)
├── Task 4.1: Deprecate old parameters
├── Task 4.2: Remove dead code
├── Task 4.3: Add tests
└── Task 4.4: Update documentation
```

---

### 3.7 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing notebooks | Keep `best_per_model` as deprecated alias |
| Performance regression | Benchmark before/after, keep fast path for non-aggregate |
| Inconsistent behavior during transition | Merge all changes together, test comprehensively |
| Missing edge cases | Add tests for each identified bug scenario |

---

### 3.8 Success Criteria

1. **Functional:**
   - [ ] `predictions.top(aggregate='ID')` ranks by aggregated scores
   - [ ] Heatmap shows same rankings as TopK for same parameters
   - [ ] ConfusionMatrix shows same models as TopK
   - [ ] `group_by=['model_classname']` works correctly

2. **Performance:**
   - [ ] Non-aggregated path remains fast (uses pre-computed scores)
   - [ ] Aggregated path is acceptable (<5s for 10k predictions)

3. **Code Quality:**
   - [ ] Single code path for ranking (no fast/slow split in charts)
   - [ ] All charts call same underlying method
   - [ ] Tests cover identified bug scenarios

---

*End of Part 3 - Roadmap*

---

## Summary

The prediction ranking system has several interconnected bugs stemming from:

1. **Aggregation applied too late** in the ranking flow
2. **Inconsistent group-by logic** between ranker and charts
3. **Two code paths** in heatmap with different behaviors
4. **Limited grouping** (only model_name, not arbitrary columns)

The fix requires:
1. Move aggregation BEFORE ranking score calculation
2. Add flexible `group_by` parameter to `top()` method
3. Unify all charts to use `predictions.top()` consistently
4. Remove duplicated/divergent code paths

The changes are surgical and can be made incrementally with backward compatibility maintained through deprecated aliases.

---

*Document created: 2025-12-07*
*Author: GitHub Copilot*
*Status: Ready for review*
