# Visualization Module: Impact of Refit & Final Scores

Date: 2026-02-08 (updated)
Scope: How the refit/final score feature should reshape default behavior across the ranking API, result API, and visualization module.

## Prerequisite: Refit Must Run on ALL Models

### Current behavior (broken)

The refit phase currently only retrains **the single global winner**:

1. `extract_winning_config()` ([config_extractor.py:103](nirs4all/pipeline/execution/refit/config_extractor.py#L103)) selects the ONE pipeline with the best `best_val` across ALL completed pipelines.
2. `execute_simple_refit()` retrains that ONE variant → produces 1 `fold_id="final"` entry.
3. All other models (from `_or_`, generators, competing branches) are never refit.

For example, with `{"_or_": [PLS(10), RF(), Ridge()]}`:
- Generator expands into 3 variants, each with a different model
- CV runs all 3 variants across folds
- Refit picks the single best (e.g., PLS with variant #2) and only retrains that one
- RF and Ridge never get a final model

Similarly, `execute_competing_branches_refit()` ([stacking_refit.py:1026](nirs4all/pipeline/execution/refit/stacking_refit.py#L1026)) selects only the winning branch and discards the rest.

The only exception is **stacking** (branches with `merge: "predictions"`), where all base models ARE refit because the ensemble requires them.

### What exists but is NOT wired up

The per-model selection infrastructure IS implemented but dead:

| Component | File | Status |
|-----------|------|--------|
| `select_best_per_model()` | [model_selector.py:55](nirs4all/pipeline/execution/refit/model_selector.py#L55) | Exists, never called by orchestrator |
| `_aggregate_scores_per_variant()` | [model_selector.py:241](nirs4all/pipeline/execution/refit/model_selector.py#L241) | Correctly averages val_score across folds |
| `PerModelSelection` dataclass | [model_selector.py:32](nirs4all/pipeline/execution/refit/model_selector.py#L32) | Exists |
| `LazyModelRefitResult` | [result.py:63](nirs4all/api/result.py#L63) | Exists, can trigger per-model refit on access |
| `RunResult._per_model_selections` | [result.py:264](nirs4all/api/result.py#L264) | Declared, never set by orchestrator |
| `RunResult.models` | [result.py:484](nirs4all/api/result.py#L484) | Uses `_per_model_selections` if set, falls back to single final |

### Required behavior

Refit must run on **every unique model** in the pipeline, each on its own best chain:

1. After CV completes, identify all unique model classes that were evaluated
2. For each model, find its best chain (variant) by **average of folds val_score** (already implemented in `_aggregate_scores_per_variant`)
3. Refit each model independently on its best chain → produces N `fold_id="final"` entries
4. Populate `_per_model_selections` so `result.models` works

Result: `predictions` buffer contains N final entries (one per model), each with:
- `fold_id="final"`
- `model_name` identifying which model
- `test_score` from full-training refit
- `refit_context="standalone"`

## The Core Problem (Visualization)

Today, every chart and analyzer method defaults to:

```
rank_partition='val'    →  "select the configuration with best validation score"
display_partition='test' →  "show that configuration's test score"
```

This was the only option pre-refit. With refit producing final models for ALL models, the primary question becomes: *"how do the final (production) models perform?"*

## Current State: Where Each Component Stands

### RunResult API (result.py)

| Accessor | What it returns | Refit-aware? |
|----------|----------------|--------------|
| `result.best` | `predictions.top(n=1)` — ranks by val, returns CV entry | No — ignores refit |
| `result.best_score` | `best.get('test_score')` — test score of CV-best | No |
| `result.best_rmse` | RMSE from CV-best entry | No |
| `result.best_r2` | R² from CV-best entry | No |
| `result.final` | Searches for `fold_id="final"` in per-dataset stores | Yes (single entry only) |
| `result.final_score` | `final.get('test_score')` | Yes (single entry only) |
| `result.cv_best` | Best CV entry excluding refit | Yes (explicit CV) |
| `result.cv_best_score` | val_score of cv_best | Yes (explicit CV) |
| `result.models` | Per-model refit results (lazy) | Plumbing exists, never populated |

### Predictions.top() (data/predictions.py:482)

```python
def top(self, n, rank_metric="", rank_partition="val", display_partition="test", ...):
```

- Defaults to `rank_partition="val"` — ranks by validation score
- Filters candidates by `partition == rank_partition` before sorting
- **No awareness of `fold_id="final"`** — refit entries are in the buffer but have no special treatment
- Refit entries have `partition="test"` (from refit execution), so they're invisible when `rank_partition="val"`
- To get refit results: `top(n=1, fold_id="final", rank_partition="test")` — non-obvious

### Predictions.get_best() (data/predictions.py:769)

- Tries `val` first, falls back to `test`
- Never specifically looks for refit entries
- Used by orchestrator for dataset-end reporting

### Chart Defaults Summary

| Chart | rank_partition | display_partition | Refit handling |
|-------|---------------|-------------------|----------------|
| `plot_top_k()` | `'val'` | `'all'` | None |
| `plot_confusion_matrix()` | `'val'` | `'test'` | None |
| `plot_histogram()` | N/A (distribution) | `'test'` | None |
| `plot_heatmap()` | `'val'` | `'test'` | None |
| `plot_candlestick()` | N/A (distribution) | `'test'` | None |
| `plot_branch_comparison()` | (inferred) | `'test'` | None |
| `plot_branch_boxplot()` | (inferred) | `'test'` | None |
| `plot_branch_heatmap()` | (inferred) | `'test'` | None |
| `branch_summary()` | N/A | `'test'` | None |

**Zero charts** are aware of refit/final scores. The visualization module is entirely CV-centric.

### Store Aggregation (store_schema.py)

- Aggregated prediction views explicitly exclude refit: `WHERE p.refit_context IS NULL`
- This means any store-backed queries used by future visualization will also be CV-only

## Score Scope Design

### The `score_scope` parameter

Add to `Predictions.top()`, `PredictionAnalyzer.get_cached_predictions()`, and all chart methods:

```python
score_scope: str = 'mix'  # 'final' | 'cv' | 'mix' | 'flat'
```

Four modes:

- **`'final'`**: Sort only final entries (`fold_id="final"`). Sorted by their originating chain's average folds val_score, but the **final test score is displayed**. Users can override display with args. When refit is disabled, returns empty results.

- **`'cv'`**: Current behavior. Only CV entries (exclude `fold_id="final"`). Ranked by val_score, display test_score.

- **`'mix'`**: Final entries always sort on top, then CV entries below. Two-level ordering: `final > cv`, then `score > score` within each group. Final entries show their test score; CV entries show their test score. This gives a complete picture: the production models first, then the exploration results.

- **`'flat'`**: All predictions (CV + final) sorted equally by the same ranking criteria, regardless of source. No special treatment for final entries.

**Default: `'mix'`** (aliased as `'auto'`).

### Implementation in `Predictions.top()`

```python
def top(self, n, rank_metric="", rank_partition="val", score_scope="mix", ...):
    candidates = [dict(r) for r in self._buffer]

    if score_scope == 'final':
        candidates = [r for r in candidates if str(r.get("fold_id")) == "final"]
        # Rank by originating chain's avg folds val_score, display test_score
        # (final entries carry their cv_rank_score from selection phase)
    elif score_scope == 'cv':
        candidates = [r for r in candidates if r.get("refit_context") is None]
        # Current behavior
    elif score_scope == 'mix':
        # Split into final and cv, rank each group, concatenate final-first
        finals = [r for r in candidates if str(r.get("fold_id")) == "final"]
        cvs = [r for r in candidates if r.get("refit_context") is None]
        # Rank finals by their originating avg folds val_score
        # Rank cvs by rank_partition score
        # Result = ranked_finals + ranked_cvs
    elif score_scope == 'flat':
        # No filtering, rank all together
        pass
    # ... ranking logic per group
```

### Key detail for `final` and `mix` modes

Final entries are ranked by **the average of folds val_score of their best chain** (the score that caused them to be selected for refit). This is NOT the final entry's own val_score (which doesn't exist — refit has no validation set). The selection score must be carried forward from `PerModelSelection.best_score` into the prediction entry metadata during refit relabeling.

The **displayed** score is the final test_score (the actual performance of the production model). Users can override with `display_partition` and `display_metrics` args.

## Impact on RunResult API

### `result.best` → returns best final entry

```python
@property
def best(self) -> dict[str, Any]:
    """Best final entry, falling back to cv_best if no refit."""
    final = self.best_final
    if final:
        return final
    return self.cv_best
```

### New: `result.best_final`

```python
@property
def best_final(self) -> dict[str, Any]:
    """Best refit entry across all models (by originating cv score)."""
    final_entries = [e for e in self.predictions._buffer
                     if str(e.get("fold_id")) == "final"]
    if not final_entries:
        return {}
    # Rank by originating chain's avg folds val_score
    ...
    return best
```

### Keep: `result.cv_best` (already exists)

No change. Explicit CV-only accessor.

### `result.best_score`, `best_rmse`, `best_r2`, `best_accuracy`

These follow `result.best` — will now return final model scores when available.

### `result.final` and `result.final_score`

Keep for single-entry access (returns first final entry found). For multi-model access, use `result.models`.

### `result.models` (per-model access)

Once the orchestrator populates `_per_model_selections`, this returns a dict of `{model_name: LazyModelRefitResult}` with lazy refit per model. Each entry exposes:
- `.final_score` — test score from refit
- `.cv_score` — average folds val_score that won selection
- `.final_entry` — full prediction dict
- `.cv_entry` — best CV prediction dict

## Chart-by-Chart Impact

With `score_scope='mix'` (default) and all models having final entries:

| Chart | New default behavior |
|-------|---------------------|
| `plot_top_k(k=5)` | Shows all 5 models. Those with final entries show final test scores and are ranked first. Those without (if fewer models were refit) show CV scores below. Titles annotate "(final)" vs "(cv)". |
| `plot_confusion_matrix(k=3)` | Top 3 models. Final entries' confusion matrices first, then CV. |
| `plot_histogram()` | Distribution includes both final and CV entries. Final entries highlighted or annotated differently. |
| `plot_heatmap()` | Cells with final entries use final test score. Other cells use CV score. Annotation distinguishes them. |
| `plot_candlestick()` | Final entries included in distributions per model. Distinguishable from CV fold distributions. |
| `branch_summary()` | Each branch reports its final score if refit; CV mean otherwise. |
| `plot_branch_comparison()` | Each branch bar uses final score if available. |

## Specific Discrepancies and Required Changes

### Issue 1: Refit only retrains the global winner (BLOCKING)

**Current**: `extract_winning_config()` picks one winner globally. Only that model gets refit.
**Required**: All unique models must be refit on their best chain. Use `select_best_per_model()` (already exists, needs wiring).
**Impact**: Prerequisite for everything else. Without multiple final entries, the visualization changes are limited.

### Issue 2: `result.best` ignores refit

**Current**: Returns CV-best.
**Required**: Returns best final entry (falling back to cv_best). Add `result.best_final` and keep `result.cv_best`.

### Issue 3: Charts have no score_scope parameter

**Current**: All charts hardcode `rank_partition='val'`.
**Required**: Add `score_scope` parameter with default `'mix'`. Thread through the full stack: `plot_*()` → `get_cached_predictions()` → `top()`.

### Issue 4: Final entries lack ranking metadata

**Current**: Final entries have `test_score` but no `val_score` (refit has no validation set). They can't participate in val-based ranking.
**Required**: During refit relabeling, inject the originating chain's average folds val_score as `cv_rank_score` metadata so that final entries can be ranked in `mix` mode.

### Issue 5: Tab reports headline CV scores

**Current**: `_print_best_predictions()` uses `get_best()` which is CV-centric.
**Required**: When final entries exist, headline with final scores. CV scores become supporting detail.

### Issue 6: Store aggregation excludes refit

**Current**: `WHERE p.refit_context IS NULL`.
**Required**: Add query mode or separate view that includes refit entries for store-backed visualization.

### Issue 7: Cache keys don't include score_scope

**Current**: Cache keys include `rank_partition`, `display_partition`.
**Required**: Add `score_scope` to cache key generation.

## What NOT to Change

- **`result.cv_best` and `result.cv_best_score`**: Keep as-is. Explicit CV accessors.
- **`result.final` and `result.final_score`**: Keep as-is. Single-entry refit accessor.
- **Store aggregation CV-only view**: Keep for backward compatibility.
- **`score_scope='cv'`**: Always available for users who want the old behavior.

## Decisions (Resolved)

### D1: Can `plot_top_k(k=5)` show 5 models in `score_scope='final'`?

Yes. With all models refit, there are N final entries (one per model). `score_scope='final'` returns all of them, ranked by originating avg folds val_score. If k > N, only N results are returned.

### D2: What is the default score_scope?

`'mix'` (aliased as `'auto'`). Final entries on top, CV entries below. This gives the most complete picture.

### D3: What happens when refit is disabled?

`score_scope='mix'` degrades to `'cv'` (no final entries exist). No change in user experience.

### D4: Multi-model refit — how many final entries?

One per unique model class. If pipeline has PLS, RF, Ridge → 3 final entries. Each model gets its best chain refit.

### D5: Should `result.best` change?

Yes. `result.best` returns best final entry (by originating cv score). `result.best_final` is the explicit accessor. `result.cv_best` is the explicit CV accessor.

## Comprehensive Roadmap

### Phase 0: Refit All Models (prerequisite — orchestrator + refit)

**Goal**: Every unique model in the pipeline gets refit on its best chain.

#### 0.1: Wire `select_best_per_model()` into the orchestrator

- **File**: `nirs4all/pipeline/execution/orchestrator.py`
- After CV completes (line 325), instead of calling `extract_winning_config()` which picks one global winner:
  1. Collect all CV predictions from `run_dataset_predictions`
  2. Analyze topology to get model nodes
  3. Call `select_best_per_model()` for each model to find its best variant
  4. Store result as `per_model_selections: dict[str, PerModelSelection]`
- `extract_winning_config()` can still be used internally for the global winner (needed for topology dispatch), but refit must iterate over all models.

#### 0.2: Refit loop over all models

- **File**: `nirs4all/pipeline/execution/orchestrator.py` (`_execute_refit_pass`)
- For non-stacking topologies:
  1. Get `per_model_selections` from step 0.1
  2. For each `(model_name, selection)` in selections:
     - Build a `RefitConfig` from that selection's `expanded_steps` and `best_params`
     - Call `execute_simple_refit()` with that config
     - Each produces a `fold_id="final"` entry with the correct `model_name`
  3. Merge all refit predictions into `run_dataset_predictions`
- For stacking topologies: keep existing stacking refit (all models already refit).
- For competing branches: refit ALL branches (not just the winner).

#### 0.3: Inject ranking metadata into final entries

- **File**: `nirs4all/pipeline/execution/refit/executor.py` (`_relabel_refit_predictions`)
- During relabeling, inject `cv_rank_score = selection.best_score` (the average folds val_score that selected this model's best chain) into each final entry.
- This enables ranking final entries by their CV selection quality without needing access to CV predictions.

#### 0.4: Populate `_per_model_selections` on RunResult

- **File**: `nirs4all/api/run.py` or `nirs4all/pipeline/execution/orchestrator.py`
- After refit, set `result._per_model_selections = per_model_selections` so that `result.models` works.

#### 0.5: Tests

- Unit test: `select_best_per_model()` is called for simple pipeline with 3 models → 3 selections
- Unit test: `execute_simple_refit()` called 3 times → 3 final entries in buffer
- Integration test: `nirs4all.run()` with `_or_` models → `result.models` has entries for each model
- Integration test: `result.final_score` returns best final model's test score

### Phase 1: Ranking API (`Predictions.top()`, `get_best()`)

**Goal**: Add `score_scope` to the ranking API.

#### 1.1: Add `score_scope` parameter to `Predictions.top()`

- **File**: `nirs4all/data/predictions.py`
- New parameter: `score_scope: str = "mix"` (accepts `"final"`, `"cv"`, `"mix"`, `"flat"`)
- Apply scope filtering before partition filtering (existing line 564)
- For `"final"`: filter to `fold_id="final"`, rank by `cv_rank_score` metadata
- For `"cv"`: filter to `refit_context is None` (exclude final entries)
- For `"mix"`: split into final/cv groups, rank each, concatenate final-first, apply n limit
- For `"flat"`: no filtering (both final and CV participate equally)
- Tag each result with `is_final: bool` for downstream consumers

#### 1.2: Update `Predictions.get_best()`

- **File**: `nirs4all/data/predictions.py`
- Add `score_scope` parameter, default `"mix"`
- With `"mix"` default, `get_best()` now returns the best final entry if one exists

#### 1.3: Tests

- Unit test: `top(score_scope="final")` returns only final entries
- Unit test: `top(score_scope="cv")` excludes final entries
- Unit test: `top(score_scope="mix")` returns finals first, then CVs
- Unit test: `top(score_scope="flat")` mixes all equally
- Unit test: `top(score_scope="mix")` with no final entries behaves like `"cv"`
- Unit test: results tagged with `is_final` correctly

### Phase 2: Result API (`RunResult`)

**Goal**: Make `result.best` final-aware.

#### 2.1: Add `result.best_final` property

- **File**: `nirs4all/api/result.py`
- Returns the best final entry (ranked by `cv_rank_score` or originating val_score)
- Returns empty dict if no final entries exist

#### 2.2: Change `result.best` to prefer final

- **File**: `nirs4all/api/result.py`
- `result.best` returns `best_final` if non-empty, else `cv_best`
- `result.best_score`, `best_rmse`, `best_r2`, `best_accuracy` follow automatically

#### 2.3: Update `result.final` for multi-model

- **File**: `nirs4all/api/result.py`
- Currently returns the first `fold_id="final"` entry found. Keep for backward compatibility.
- `result.models` is the proper multi-model accessor (now populated from Phase 0.4).

#### 2.4: Tests

- Unit test: `result.best` returns final entry when available
- Unit test: `result.best` returns cv_best when no final entries
- Unit test: `result.best_final` returns best final entry by cv_rank_score
- Unit test: `result.best_rmse` reflects final model when available
- Regression test: `result.cv_best` unchanged

### Phase 3: Visualization Module

**Goal**: Add `score_scope` to all charts and make `'mix'` the default.

#### 3.1: Add `score_scope` to `PredictionAnalyzer.get_cached_predictions()`

- **File**: `nirs4all/visualization/predictions.py`
- New parameter: `score_scope: str = "mix"`
- Pass through to `predictions.top()`
- Include `score_scope` in cache key

#### 3.2: Add `score_scope` to `BaseChart._get_ranked_predictions()`

- **File**: `nirs4all/visualization/charts/base.py`
- New parameter: `score_scope: str = "mix"`
- Pass through to analyzer

#### 3.3: Thread `score_scope` through all `plot_*()` methods

- **Files**: `nirs4all/visualization/predictions.py` (all `plot_*` methods)
- Add `score_scope` parameter to: `plot_top_k()`, `plot_confusion_matrix()`, `plot_histogram()`, `plot_heatmap()`, `plot_candlestick()`, `plot_branch_comparison()`, `plot_branch_boxplot()`, `plot_branch_heatmap()`, `branch_summary()`
- Default: `'mix'`

#### 3.4: Chart rendering updates

For each chart type, handle `is_final` tagging in rendering:

- **TopKComparisonChart**: Annotate titles with "(final)" vs "(cv)" per model subplot
- **HeatmapChart**: Annotate cells with final entries (e.g., bold or marker)
- **CandlestickChart**: Optionally distinguish final entries in distributions
- **HistogramChart**: Optionally distinguish final entries
- **ConfusionMatrixChart**: Label final vs cv matrices
- **Branch charts**: Label final vs cv per branch

#### 3.5: Update chart title generation

- All charts should indicate active scope in figure title/subtitle
- Example: "Top 5 Models (mix: final + cv)" or "Top 5 Models (cv only)"

#### 3.6: Update `prediction_cache.py`

- **File**: `nirs4all/visualization/prediction_cache.py`
- Include `score_scope` in `make_key()`

#### 3.7: Tests

- Unit test: each chart renders without error with all score_scope values
- Visual regression tests for chart output
- Test cache key includes score_scope (different scopes don't collide)

### Phase 4: Reporting and Logging

**Goal**: Final scores become the headline in reports and logs.

#### 4.1: Update `_print_best_predictions()`

- **File**: `nirs4all/pipeline/execution/orchestrator.py`
- When final entries exist:
  - Headline: "Final model performance" with best final score
  - Detail: "CV selection summary" with cv_best score
- When no final entries: keep current behavior

#### 4.2: Update `TabReportManager`

- **File**: `nirs4all/visualization/reports.py`
- Restructure to show both final and CV sections
- Final section first when available
- Show per-model final scores table when multiple models refit

#### 4.3: Tests

- Unit test: orchestrator logs include final model headline
- Unit test: tab report includes final section

### Phase 5: Store Integration

**Goal**: Store-backed queries support refit entries.

#### 5.1: Add refit-inclusive aggregated view

- **File**: `nirs4all/pipeline/storage/store_schema.py`
- Add `v_aggregated_predictions_all` or query parameter `include_refit=True`
- Keep existing CV-only view unchanged

#### 5.2: Extend query builders

- **File**: `nirs4all/pipeline/storage/store_queries.py`
- Add `score_scope` parameter to `query_top_aggregated_predictions`
- Default: `'cv'` for backward compatibility in store queries

#### 5.3: Tests

- Unit test: store query with `include_refit=True` returns final entries
- Unit test: existing CV-only queries unchanged

## Files to Modify (Complete List)

| File | Phase | Change |
|------|-------|--------|
| `nirs4all/pipeline/execution/orchestrator.py` | 0, 4 | Wire per-model refit loop; update reporting |
| `nirs4all/pipeline/execution/refit/executor.py` | 0 | Inject `cv_rank_score` during relabeling |
| `nirs4all/pipeline/execution/refit/stacking_refit.py` | 0 | Update competing branches to refit all branches |
| `nirs4all/api/run.py` | 0 | Populate `_per_model_selections` on RunResult |
| `nirs4all/data/predictions.py` | 1 | Add `score_scope` to `top()`, `get_best()` |
| `nirs4all/api/result.py` | 2 | Add `best_final`; change `best` semantics |
| `nirs4all/visualization/predictions.py` | 3 | Add `score_scope` to all `plot_*()` methods |
| `nirs4all/visualization/charts/base.py` | 3 | Add `score_scope` to `_get_ranked_predictions()` |
| `nirs4all/visualization/charts/top_k_comparison.py` | 3 | Render final vs cv annotations |
| `nirs4all/visualization/charts/heatmap.py` | 3 | Annotate final cells |
| `nirs4all/visualization/charts/candlestick.py` | 3 | Handle final entries in distributions |
| `nirs4all/visualization/charts/histogram.py` | 3 | Handle final entries in distributions |
| `nirs4all/visualization/charts/confusion_matrix.py` | 3 | Label final vs cv matrices |
| `nirs4all/visualization/prediction_cache.py` | 3 | Include `score_scope` in cache key |
| `nirs4all/visualization/reports.py` | 4 | Restructure to headline final scores |
| `nirs4all/pipeline/storage/store_schema.py` | 5 | Add refit-inclusive aggregated view |
| `nirs4all/pipeline/storage/store_queries.py` | 5 | Add `score_scope` parameter |

## Conclusion

There are two layers of work:

1. **Phase 0 (prerequisite)**: Refit must run on ALL models, not just the global winner. The infrastructure exists (`select_best_per_model`, `PerModelSelection`, `LazyModelRefitResult`) but is not wired into the orchestrator. Each model gets its best chain (selected by average folds val_score) and produces its own `fold_id="final"` entry.

2. **Phases 1-5 (visualization and API)**: With multiple final entries available, the `score_scope` parameter (`'final'` | `'cv'` | `'mix'` | `'flat'`) threads through `top()` → `get_cached_predictions()` → all charts. Default is `'mix'` (finals on top, CVs below). `result.best` returns the best final entry. `result.best_final` and `result.cv_best` provide explicit access.

---

## Implementation Status

**All phases implemented** (2026-02-08).

### Phase 0: Refit All Models
- Orchestrator now calls `_execute_per_model_refit()` to refit ALL unique model classes, not just the global winner
- Each model's best chain (selected by average folds val_score) produces its own `fold_id="final"` entry with `cv_rank_score` metadata
- Stacking competing branches refit all branches, not just the winner

### Phase 1: Ranking API
- `Predictions.top()` and `get_best()` accept `score_scope` parameter (`'final'`|`'cv'`|`'mix'`|`'flat'`, default `'mix'`)
- Final entries ranked by `cv_rank_score`, bypass `rank_partition` filtering
- Mix mode: two-level sort — finals first, then CV entries

### Phase 2: Result API
- `RunResult.best_final` — best refit entry across all models
- `RunResult.best` — prefers `best_final`, falls back to `cv_best`
- `RunResult.cv_best` — uses `top(n=1, score_scope="cv")`

### Phase 3: Visualization Module
- `score_scope` threaded through `get_cached_predictions()` → `_get_ranked_predictions()` → all chart render methods
- Cache key includes `score_scope` for proper cache differentiation

### Phase 4: Reporting and Logging
- When final entries exist: headline "Final model performance", tab report, per-model table, then "CV selection summary"
- When no final entries: preserves existing CV-only behavior
- `TabReportManager.generate_per_model_summary()` for multi-model table formatting

### Phase 5: Store Integration
- New `v_aggregated_predictions_all` view includes `refit_context` in GROUP BY (separates CV and refit aggregations)
- `score_scope` parameter added to `build_aggregated_query()`, `build_top_aggregated_query()`, and workspace store methods
- Default `'cv'` for backward compatibility
