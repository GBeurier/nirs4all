# Generator Variant Redundancy: Redundant Model Training Across Variants

## Problematic

When a pipeline contains multiple models and a generator keyword (`_range_`, `_or_`, `_grid_`, ...) is attached to only one of them, the generator expands into N full pipeline copies. Every model in the pipeline -- including those whose configuration is identical across all variants -- is re-trained from scratch for each variant. The cost of redundant model training (especially with finetuning) dwarfs preprocessing, which is negligible by comparison.

**Simple case** (common prefix exists):

```json
[
  "SNV",
  {"SavitzkyGolay": {"window_length": 11, "polyorder": 2, "deriv": 1}},
  {"KennardStone": {"test_size": 0.2}},
  {"model": {"PLSRegression": {"n_components": 10}},
   "finetune_params": {"n_trials": 50, "approach": "grouped"}},
  {"model": {"RandomForestRegressor": {"n_estimators": 100}},
   "_range_": [50, 150, 10], "param": "n_estimators"}
]
```

11 variants. PLS finetuning (50 Optuna trials × CV) runs 11 times identically. Only the RF `n_estimators` actually changes.

**Hard case** (no common prefix -- generators on multiple steps):

```json
[
  "SNV",
  {"model": {"PLSRegression": {"n_components": {"_range_": [5, 15, 5]}}}},
  {"model": {"RandomForestRegressor": {"n_estimators": {"_range_": [50, 150, 10]}}}}
]
```

3 × 11 = 33 variants. Each PLS config should train once, then fork into 11 RF configs. Instead, PLS trains 33 times (11 redundant trainings per PLS config).

The fundamental problem: the generator produces a **flat list of complete pipelines** via Cartesian product. The orchestrator executes each variant independently. There is no concept of "which steps actually changed between variants."

---

## Status in the Code

### Generator expansion (flat Cartesian product)

`PipelineConfigs.__init__` (`pipeline/config/pipeline_config.py:73-91`) calls `expand_spec_with_choices(self.steps)` which recursively expands via `_expand_list_with_choices` (`pipeline/config/_generator/core.py`). This function takes the **Cartesian product** of all expanded list elements. A `_range_` on step 5 multiplies with everything else, producing N complete pipeline lists where steps 1-4 are duplicated verbatim.

### Variant execution (independent, no state sharing)

The orchestrator (`pipeline/execution/orchestrator.py:507-565`) iterates over all variants. Each variant loads a fresh dataset, creates a fresh context, and executes all steps from scratch. No fitted models, predictions, or dataset state is shared between variants.

### Step cache: covers preprocessing only, not models

The step cache (`CacheConfig(step_cache_enabled=True)`) only caches controllers returning `supports_step_cache() = True`:
- `TransformerMixinController` returns `True` (SNV, SG, etc.)
- `OperatorController` base returns `False` -- all model controllers, splitters, branch/merge inherit this default

So with step cache enabled, preprocessing transforms are deduplicated, but **model training (the expensive part) always re-runs**.

Additionally, the step cache is **disabled in parallel mode** (`step_cache=None` in workers, `orchestrator.py:356`) because `StepCache._lock` cannot be pickled across loky processes.

### What happens concretely in the simple case (11 variants, sequential):

| Step | Cache enabled | Runs |
|------|--------------|------|
| SNV | cached after variant 1 | 1 |
| SavitzkyGolay | cached after variant 1 | 1 |
| KennardStone | not cacheable | 11 |
| PLS + 50 Optuna trials | not cacheable | **11** |
| RF (n_estimators varies) | different config each time | 11 (intended) |

With `n_jobs=-1` (parallel): everything runs 11 times independently.

---

## Proposed Solution: Tree-Structured Variant Execution

Instead of a flat list of complete pipelines, build an **execution tree** where shared step prefixes are executed once and the tree forks only at points of divergence.

### Concept

After generator expansion, analyze the N variant step-lists to build a tree:

```
Root
├── SNV (shared)
├── SavitzkyGolay (shared)
├── KennardStone (shared)
├── PLS + finetune (shared)       ← trained ONCE
└── Fork:
    ├── RF(n_estimators=50)
    ├── RF(n_estimators=60)
    ├── ...
    └── RF(n_estimators=150)
```

For the hard case (generators on both models):

```
Root
├── SNV (shared)
└── Fork on PLS:
    ├── PLS(n_components=5)       ← trained ONCE
    │   └── Fork on RF:
    │       ├── RF(50) ... RF(150)   ← 11 variants
    ├── PLS(n_components=10)      ← trained ONCE
    │   └── Fork on RF:
    │       ├── RF(50) ... RF(150)
    └── PLS(n_components=15)      ← trained ONCE
        └── Fork on RF:
            ├── RF(50) ... RF(150)
```

PLS trains 3 times instead of 33. RF trains 33 times (intended).

### Algorithm

1. **Build a trie from variant step-lists.** Each node is a step config (compared by content hash). Children represent divergent next-steps.
2. **Execute the trie depth-first.** At each node:
   - Execute the step on the current dataset.
   - If the node has a single child: continue (no fork needed).
   - If the node has multiple children: snapshot the dataset (CoW), then for each child branch, restore the snapshot and recurse.
3. **Leaf nodes** correspond to complete variant executions. Record results as usual.

### Implementation location

The logic belongs in the **orchestrator** (`PipelineOrchestrator`), between variant expansion and the per-variant execution loop. The executor, controllers, and generator remain unchanged.

### Key considerations

- **Dataset snapshot**: `SpectroDataset` already supports CoW snapshots via `SharedBlocks` (used by branch controller and step cache). The infrastructure exists.
- **ExecutionContext**: Must also be snapshotted/cloned at fork points, since it carries processing history, selector state, etc.
- **Prediction storage**: Each leaf variant must have its own `Predictions` store. Predictions from shared steps (like the PLS model in the trunk) need to be duplicated or referenced by all leaf variants that share them.
- **Parallel mode**: Fork points are natural parallelization boundaries. Instead of parallelizing across flat variants, parallelize across children at each fork node. This is more efficient (fewer total step executions) and sidesteps the step cache pickling issue entirely.
- **Refit**: The "refit best variant on full data" logic currently works on flat variant indices. It would need to reconstruct the full step list from the trie path for the winning leaf.

### Fallback: prefix-only (simpler, covers the simple case)

If the full trie is too complex initially, a simpler version handles the common case:

1. Compute the **longest common prefix** across all variant step-lists (by config hash).
2. Execute the prefix once (including model steps + finetuning).
3. Snapshot dataset + context.
4. For each variant, restore snapshot and execute only the suffix.

This solves the simple case (PLS in prefix, RF varies) but not the hard case (generators on multiple steps). It's a pragmatic first step.

---

## Insights

1. **The real waste is model training, not preprocessing.** Preprocessing transforms are cheap and already handled by the step cache (in sequential mode). The 50× Optuna finetuning cost is the problem. Any solution must target model deduplication specifically.

2. **The generator's flat Cartesian product is the root cause.** It's the right design for the generator (simple, generic), but the orchestrator should not naively execute the flat list when structural redundancy exists. The generator could optionally output a tree structure, or the orchestrator could reconstruct it.

3. **Parallel mode needs the tree approach.** The current step cache cannot work across processes. Tree-structured execution naturally parallelizes at fork points without needing cross-process caching.

4. **Two sequential `{"model": ...}` steps are independent.** Both train on the same preprocessed X. The second model does NOT receive the first model's predictions as input (no `{"merge": "predictions"}` between them). Users should be aware that this is different from stacking. Each model produces its own predictions stored independently.

5. **CoW snapshot infrastructure already exists.** `SharedBlocks` in `SpectroDataset`, used by branch controllers and step cache, provides near-free dataset forking. The main gap is snapshotting `ExecutionContext` and `Predictions` state at fork points.

6. **Incremental path:** prefix-only (simple case) → full trie (hard case). The prefix approach is low-risk, high-impact for the most common scenario (single generator keyword in a multi-step pipeline). The trie generalizes it for multiple generators.
