# Python SDK agent workflow

## Commands

```bash
# Tests
pytest tests/                     # All tests
pytest tests/unit/                # Unit only
pytest tests/integration/         # Integration only
pytest -m sklearn                 # sklearn-only
pytest --cov=nirs4all             # With coverage

# Examples (from examples/ directory)
./run.sh                          # All examples
./run.sh -c user                  # User examples only
./run_ci_examples.sh -c all -j 4  # CI runner with 4 parallel workers

# Code quality
ruff check .                      # Lint
mypy .                            # Type check

# Installed CLI (entry point: nirs4all.cli.main:main)
nirs4all --version
nirs4all --test-install           # Verify dependencies
nirs4all --test-integration       # Run integration smoke tests
nirs4all workspace ...            # Workspace inspection / migration
nirs4all dataset ...              # Dataset commands
nirs4all artifacts ...            # Artifact inspection
nirs4all config ...               # Config validation

# Pre-publish validation (mirrors CI locally, runs all steps with -j nproc parallelism)
./scripts/pre-publish.sh                    # Full: ruff, mypy, tests, docs, examples, build
./scripts/pre-publish.sh -j 4               # Limit to 4 parallel workers
./scripts/pre-publish.sh --only tests       # Single step
./scripts/pre-publish.sh --skip-docs        # Skip a step
./scripts/pre-publish.sh --docker           # Run in clean ubuntu:24.04 container
```

## Pipeline syntax and canonical references

The following retained syntax describes authoring semantics, not a claim that every backend executes every request. Runtime adapters preserve complete recipes or explicitly refuse. Canonical references are `source/reference/pipeline_keywords.md` and its linked `generator_keywords.md`; these tables are not a complete token inventory.

Steps are classes, instances, or dicts with keywords:

```python
pipeline = [
    MinMaxScaler(),                            # Transformer
    {"y_processing": MinMaxScaler()},          # Target scaling
    ShuffleSplit(n_splits=3),                  # Cross-validator
    {"model": PLSRegression(n_components=10)}, # Model
]
```

### Keywords

| Keyword | Purpose |
|---------|---------|
| `model` | Define model step |
| `y_processing` | Target scaling |
| `tag` | Mark samples (non-removal) |
| `exclude` | Remove samples from training (`mode`: "any"/"all" for multiple) |
| `branch` | Duplication branches (parallel pipelines) or separation branches (by_metadata/by_tag/by_filter/by_source) |
| `merge` | Combine branches: `"predictions"` (stacking), `"features"`, `"all"`, `"concat"` (reassembly) |
| `sample_augmentation` | Data augmentation applied to training samples |
| `feature_augmentation` | Feature-level augmentation |
| `concat_transform` | Concatenate transformed features |
| `rep_to_sources` | Convert repetition groups to multi-source format |
| `rep_to_pp` | Convert repetition groups to preprocessing pipelines |
| `finetune_params` | Optuna hyperparameter search alongside `model` (driven by `nirs4all.optimization.optuna`) |
| `train_params` | Per-model training kwargs (epochs, patience, batch size, etc.) |
| `name` | Human-readable label for the step (appears in results) |

### Generator Syntax (Hyperparameter Sweeps)

| Keyword | Effect | Example |
|---------|--------|---------|
| `_or_` | Try alternatives | `[SNV, MSC, Detrend]` |
| `_range_` | Linear sweep | `[1, 30, 5]` → `[1, 6, 11, 16, 21, 26]` |
| `_log_range_` | Log sweep | `[1e-3, 1e0, 4]` → `[0.001, 0.01, 0.1, 1.0]` |
| `_grid_` | Cartesian product of params | `{"n_components": [5,10], "alpha": [0.1,1.0]}` → 4 combos |
| `_cartesian_` | Pipeline stage combinations | `[{"_or_": [A,B]}, {"_or_": [X,Y]}]` → 4 pipelines |
| `_zip_` | Paired iteration | `{"x": [1,2,3], "y": ["a","b","c"]}` → 3 pairs |
| `_chain_` | Sequential ordered | `[config1, config2, config3]` |
| `_sample_` | Random sampling | `{"distribution": "log_uniform", "from": 1e-4, "to": 1e-1, "num": 20}` |

## Code Conventions

- Google-style docstrings
- Line length: 220 (`pyproject.toml`)
- Ruff for linting, mypy for types
- Type hints required for public APIs
- Use `.venv` for running scripts/tests
- Deep learning backends (TF, PyTorch, JAX) are lazy-loaded
- YAML: tuples may convert to lists during serialization
- No dead code, no deprecated code, no backward compatibility shims

## Constraints

- Read and understand files before proposing edits. Never speculate about unread code.
- No over-engineering: minimum complexity for the current task. No hypothetical features.
- No unnecessary error handling for impossible scenarios. Validate only at system boundaries.
- No helpers/utilities/abstractions for one-time operations. Reuse existing abstractions (DRY).
- Write general-purpose solutions, not hard-coded workarounds for specific test inputs.
- After API changes, run: `cd examples && ./run.sh`

## Scoped local agent mechanics

These local mechanics are subordinate to current user authorization and owner-first fixes. They grant no extra approval gate, commit, delegation or frontend scope.

### Pre-work

1. THE "STEP 0" RULE: Dead code accelerates context compaction. For an authorized structural refactor on a file >300 LOC, identify dead props/exports/imports/debug logs and perform the necessary cleanup within scope. Keep unrelated cleanup out; a separate cleanup commit applies only when commits are authorized, never as an integration prerequisite here.

2. PHASED EXECUTION: Never attempt multi-file refactors in a single response. Break work into explicit phases. Complete and verify each phase; seek direction only for scope outside existing authorization. Each phase must touch no more than 5 files.

### Code quality

3. THE SENIOR DEV OVERRIDE: Trace demonstrated architectural defects/duplicate state/inconsistent patterns to their owner and fix them within authorized scope; do not introduce unrelated speculative structural changes. Ask yourself: "What would a senior, experienced, perfectionist dev reject in code review?" Report remaining out-of-scope issues explicitly.

4. FORCED VERIFICATION: Your internal tools mark file writes as successful even if the code does not compile. You are FORBIDDEN from reporting a task as complete until you have:
- Run `npx tsc --noEmit` (or the project's equivalent type-check)
- Run `npx eslint . --quiet` (if configured)
- Fixed ALL resulting errors
- In particular ensure ruff and mypy checks pass !


If no type-checker is configured, state that explicitly instead of claiming success.

### Context management

5. SUB-AGENT SWARMING: For tasks touching >5 independent files, split coherent reviewable packets; use parallel subagents only when the user explicitly authorizes delegation. Never infer agent permission from file count.

6. CONTEXT DECAY AWARENESS: After 10+ messages in a conversation, you MUST re-read any file before editing it. Do not trust your memory of file contents. Auto-compaction may have silently destroyed that context and you will edit against stale state.

7. FILE READ BUDGET: Each file read is capped at 2,000 lines. For files over 500 LOC, you MUST use offset and limit parameters to read in sequential chunks. Never assume you have seen a complete file from a single read.

8. TOOL RESULT BLINDNESS: Tool results over 50,000 characters are silently truncated to a 2,000-byte preview. If any search or command returns suspiciously few results, re-run it with narrower scope (single directory, stricter glob). State when you suspect truncation occurred.

### Edit safety

9.  EDIT INTEGRITY: Before EVERY file edit, re-read the file. After editing, read it again to confirm the change applied correctly. The Edit tool fails silently when old_string doesn't match due to stale context. Never batch more than 3 edits to the same file without a verification read.

10. COMPLETE RENAME COVERAGE: Use CodeGraph first only with this checkout's existing index, then verify source references directly. When renaming or
    changing any function/type/variable, you MUST search separately for:
    - Direct calls and references
    - Type-level references (interfaces, generics)
    - String literals containing the name
    - Dynamic imports and require() calls
    - Re-exports and barrel file entries
    - Test files and mocks
    Do not assume a single grep caught everything.

## Applicability and tools

- Preserve supported published API/artifact semantics; clean-V1 removals require migrated active callers and parity, not an old rollback example. Keep identified scientific references/frozen provenance. Discover current runtime routing from the supported public contracts.
- Shared managed RTK catalogues are not reinjected here. Respect current scoped tool policy, raw exit evidence and secrecy; no tool description grants installs, network, Git or publishing.

## SDK host storage

- The SDK's host workspace uses SQLite metadata (`store.sqlite`) plus Zstd Parquet prediction arrays (`arrays/<dataset_name>.parquet`). Preserve WorkspaceStore WAL/thread/retry-on-lock behavior, ArrayStore tombstone deletion and the Predictions facade over metadata/arrays.
- Content-addressed host model artifacts use `artifacts/<hash>.joblib`; run manifests live in `runs/<dataset>/.../manifest.yaml`, exports in `exports/<dataset>/` and pipeline templates in `library/templates/`. These host locations do not transfer native lineage/scoring or recipe semantics away from DAG.
- BundleGenerator/BundleLoader remain the interim SDK bundle export/load/predict seam pending V1 cutover; Core/container and payload owners retain their contracts. Host deserialization and archive readability do not establish supported execution or permission to load untrusted artifacts.
- The old controller flow is not the default execution path. `engine="legacy"` remains an explicit direct-Python rollback lane while [public interfaces](source/reference/public_interfaces.md) documents it; follow that document for current lanes. Use the [controller contract](source/developer/controllers.md) and [architecture map](source/developer/architecture.md) for that lane; preserve supported owner APIs and named scientific references/frozen reproductions.
