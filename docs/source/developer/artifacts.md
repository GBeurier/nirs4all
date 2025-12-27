# Artifacts & Storage

This guide covers the artifact storage system and workspace structure in nirs4all.

## Overview

The artifacts system (V3) provides:

- **Deterministic artifact IDs** based on operator chains for complete execution path tracking
- **Content-addressed storage** with deduplication across pipelines
- **Dependency tracking** for stacking and transfer learning
- **LRU caching** for efficient artifact loading

## Workspace Structure

```
workspace/
├── runs/                          # Experimental runs
│   └── {dataset}/                 # Dataset-centric organization
│       ├── _binaries/             # Shared artifacts (deduplicated)
│       ├── 0001_hash/             # Pipeline 1
│       └── 0002_name_hash/        # Pipeline 2 (with custom name)
│
├── binaries/                      # Centralized artifact storage (V3)
│   └── {dataset}/                 # Per-dataset binaries
│
├── exports/                       # Best results (fast access)
│   ├── {dataset}/                 # Full exports
│   └── best_predictions/          # Predictions only
│
├── library/                       # Reusable pipelines
│   ├── templates/                 # Config only
│   └── trained/                   # With binaries
│       ├── filtered/              # Config + metrics
│       ├── pipeline/              # Full pipeline
│       └── fullrun/               # Everything + data
│
└── catalog/                       # Prediction index
    ├── predictions_meta.parquet   # Metadata (fast queries)
    └── predictions_data.parquet   # Arrays (on-demand)
```

## Artifact Types

| Type | Description |
|------|-------------|
| `MODEL` | Trained ML models (sklearn, TensorFlow, PyTorch, JAX) |
| `TRANSFORMER` | Fitted preprocessors (scalers, feature extractors) |
| `SPLITTER` | Train/test split configuration |
| `ENCODER` | Label encoders, y-scalers |
| `META_MODEL` | Stacking meta-models with source dependencies |

## V3 Artifact ID Format

Format: `{pipeline_id}${chain_hash}:{fold_id}`

Examples:
- `0001_pls$a1b2c3d4e5f6:all` - Shared artifact
- `0001_pls$7f8e9d0c1b2a:0` - Fold 0 artifact
- `0001_pls$3c4d5e6f7a8b:1` - Fold 1 artifact

The chain hash is computed from the operator chain path (e.g., `s1.MinMaxScaler>s3.PLS[br=0]`), ensuring deterministic identification across branching, multi-source, and stacking scenarios.

## Using the ArtifactRegistry

The `ArtifactRegistry` is the central class for artifact management:

```python
from pathlib import Path
from nirs4all.pipeline.storage.artifacts import ArtifactRegistry, ArtifactType

# Initialize registry
registry = ArtifactRegistry(
    workspace=Path("./workspace"),
    dataset="wheat_sample1",
    pipeline_id="0001_pls_abc123"
)

# Register an artifact with V3 chain-based ID
record = registry.register_with_chain(
    obj=trained_model,
    chain="s1.MinMaxScaler>s3.PLSRegression",
    artifact_type=ArtifactType.MODEL,
    step_index=3,
    fold_id=0,
    params={"n_components": 10}
)

print(f"Saved: {record.artifact_id}")
# Output: 0001_pls_abc123$a1b2c3d4e5f6:0
```

### Key Methods

```python
# Generate ID from chain
artifact_id = registry.generate_id(chain, fold_id=0)

# Resolve ID to record
record = registry.resolve(artifact_id)

# Get by chain path (V3)
record = registry.get_by_chain("s1.MinMaxScaler>s3.PLS", fold_id=0)

# Get artifacts for a step
records = registry.get_artifacts_for_step(
    pipeline_id="0001",
    step_index=3,
    branch_path=[0],
    fold_id=None
)

# Get fold models for CV averaging
fold_records = registry.get_fold_models(
    pipeline_id="0001",
    step_index=3
)
```

## Using the ArtifactLoader

The `ArtifactLoader` provides efficient loading with caching:

```python
from nirs4all.pipeline.storage.artifacts import ArtifactLoader

# Create from manifest
loader = ArtifactLoader.from_manifest(manifest, results_dir)

# Load by ID (uses LRU cache)
model = loader.load_by_id("0001_pls$abc123:0")

# Load by chain path
model = loader.load_by_chain("s1.MinMaxScaler>s3.PLS", fold_id=0)

# Load all artifacts for a step
artifacts = loader.load_for_step(
    step_index=3,
    branch_path=[0],
    fold_id=0
)
for artifact_id, obj in artifacts:
    print(f"Loaded: {artifact_id}")

# Load fold models for ensemble
fold_models = loader.load_fold_models(step_index=3)
for fold_id, model in fold_models:
    print(f"Fold {fold_id}: {model}")
```

### Meta-Model Loading (Stacking)

```python
# Load meta-model with all source models
meta_model, sources, feature_cols = loader.load_meta_model_with_sources(
    artifact_id="0001_pls$abc123:all",
    validate_branch=True
)

# sources is [(source_id, source_model), ...]
# feature_cols is ["PLSRegression_pred", "RandomForest_pred", ...]
```

### Cache Management

```python
# Get cache statistics
info = loader.get_cache_info()
print(f"Cache hit rate: {info['hit_rate']:.2%}")

# Preload artifacts
loader.preload_artifacts(artifact_ids=["0001:3:0", "0001:3:1"])

# Clear cache
loader.clear_cache()

# Resize cache
loader.set_cache_size(200)
```

## Library Management

### Save Pipeline Templates

```python
from nirs4all.workspace import LibraryManager

library = LibraryManager(workspace / "library")

# Save config-only template
library.save_template(
    pipeline_config=pipeline_dict,
    name="baseline_pls",
    description="PLS baseline with SNV preprocessing"
)

# Save full trained pipeline
library.save_pipeline_full(
    run_dir=runs_dir / "wheat_sample1",
    pipeline_dir=runs_dir / "wheat_sample1" / "0042_x9y8z7",
    name="wheat_quality_v1"
)
```

### Load and Reuse

```python
# List templates
templates = library.list_templates()
for t in templates:
    print(f"{t['name']}: {t['description']}")

# Load template
config = library.load_template("baseline_pls")

# Use in pipeline
runner = PipelineRunner(workspace="./workspace")
predictions = runner.run(config, new_dataset)
```

## Cleanup Utilities

```python
# Find orphaned artifacts
orphans = registry.find_orphaned_artifacts()
print(f"Found {len(orphans)} orphaned files")

# Delete orphans (dry run first)
deleted, freed = registry.delete_orphaned_artifacts(dry_run=True)
print(f"Would delete {len(deleted)} files, freeing {freed / 1024:.1f} KB")

# Actually delete
deleted, freed = registry.delete_orphaned_artifacts(dry_run=False)

# Get storage statistics
stats = registry.get_stats()
print(f"Total artifacts: {stats['total_artifacts']}")
print(f"Unique files: {stats['unique_files']}")
print(f"Deduplication ratio: {stats['deduplication_ratio']:.1%}")
```

## Best Practices

1. **Use chain-based registration** (`register_with_chain`) for new code to ensure deterministic artifact IDs.

2. **Let the registry handle deduplication** - identical artifacts (same content hash) automatically share the same file.

3. **Use the loader's cache** - the LRU cache significantly improves performance when loading the same artifacts multiple times.

4. **Track dependencies for meta-models** - always register source models before the meta-model to enable proper dependency resolution.

5. **Clean up periodically** - use `find_orphaned_artifacts()` and `delete_orphaned_artifacts()` to reclaim disk space.

## See Also

- {doc}`/reference/storage` - Storage API reference
- {doc}`/reference/workspace` - Workspace architecture
- {doc}`architecture` - Pipeline architecture overview
- {doc}`/reference/pipeline_syntax` - Pipeline configuration syntax
