# Run-Based Architecture with Per-Run Artifact Cache

**Version**: 2.0
**Date**: October 14, 2025
**Status**: Recommended Architecture

---

## Executive Summary

This architecture organizes all results by **run** (date + run ID), with each run containing multiple pipelines and a shared artifact cache. Symlinks enable deduplication within a run while keeping the file structure simple and portable.

### Key Features

✅ **Date-first organization** - Chronological sorting by default
✅ **Self-contained runs** - Each run includes its own artifact cache
✅ **Symlink-based deduplication** - Identical artifacts stored once per run
✅ **Export function** - Creates portable packages by resolving symlinks
✅ **Human-readable names** - Artifact names like `StandardScaler_abc123.pkl`
✅ **Simple cleanup** - `rm -rf 2024-10-14_*` removes everything
✅ **No global state** - No databases, no global indexes

---

## Complete Structure

```
results/
├── 2024-10-14_wheat_quality/              # Date_runid (chronological)
│   │
│   ├── .artifacts/                        # Per-run cache (hidden)
│   │   ├── StandardScaler_abc123.pkl     # Human-readable + short hash
│   │   ├── PLS_model_def456.pkl
│   │   ├── SVC_model_ghi789.pkl
│   │   └── RandomForest_jkl012.pkl
│   │
│   ├── regression_Q1_c20f9b/              # dataset_pipelineid
│   │   ├── pipeline.yaml                  # Portable configuration
│   │   ├── metadata.yaml                  # Training metadata
│   │   ├── scores.yaml                    # All metrics consolidated
│   │   ├── outputs/                       # Charts & visualizations
│   │   │   ├── confusion_matrix.png
│   │   │   ├── predictions_plot.png
│   │   │   ├── feature_importance.png
│   │   │   └── learning_curve.png
│   │   ├── predictions/                   # Prediction CSVs
│   │   │   ├── train_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   └── validation_predictions.csv
│   │   └── binaries/                      # Symlinks to cached artifacts
│   │       ├── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl
│   │       └── model_1.pkl -> ../../.artifacts/PLS_model_def456.pkl
│   │
│   ├── regression_Q2_xyz789/              # Another pipeline
│   │   ├── pipeline.yaml
│   │   ├── metadata.yaml
│   │   ├── scores.yaml
│   │   ├── outputs/
│   │   ├── predictions/
│   │   └── binaries/
│   │       ├── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl  # REUSED!
│   │       └── model_1.pkl -> ../../.artifacts/SVC_model_ghi789.pkl
│   │
│   └── classification_Q3_abc123/          # Different dataset, same run
│       ├── pipeline.yaml
│       ├── metadata.yaml
│       ├── scores.yaml
│       ├── outputs/
│       ├── predictions/
│       └── binaries/
│           ├── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl  # REUSED AGAIN!
│           └── model_1.pkl -> ../../.artifacts/RandomForest_jkl012.pkl
│
└── 2024-10-15_corn_analysis/              # Another run (different date)
    ├── .artifacts/                        # Separate cache per run
    │   └── ...
    └── regression_yield_def456/
        └── ...
```

---

## File Structure Details

### Run Folder: `Date_runid/`

**Naming Convention**: `YYYY-MM-DD_<run_name>`

Examples:
- `2024-10-14_wheat_quality/`
- `2024-10-15_corn_analysis/`
- `2024-10-16_exploratory_tests/`

**Benefits:**
- Chronological sorting in file browser
- Easy to find recent work
- Clear temporal organization
- Run name provides context

### Pipeline Folder: `dataset_pipelineid/`

**Naming Convention**: `<dataset_name>_<pipeline_id>`

Examples:
- `regression_Q1_c20f9b/` - Regression dataset, pipeline Q1
- `classification_Q3_abc123/` - Classification dataset, pipeline Q3
- `multioutput_experiment_xyz789/`

**Pipeline ID:** 6-character hash or user-provided name

**Benefits:**
- Dataset name immediately visible
- Pipeline ID for uniqueness
- Human-readable folder names

### Artifact Cache: `.artifacts/`

**Location**: Hidden folder at run level

**Naming Convention**: `<operator_name>_<short_hash>.<ext>`

Examples:
- `StandardScaler_abc123.pkl`
- `PLS_model_def456.pkl`
- `SVC_model_ghi789.keras`
- `RandomForest_jkl012.pkl`

**Short Hash**: First 6 characters of SHA256 content hash

**Benefits:**
- Human-readable (can identify "StandardScaler")
- Unique (hash prevents collisions)
- Content-addressed (deterministic)
- Hidden from users (`.` prefix)

### Pipeline Contents

#### `pipeline.yaml`
Portable pipeline configuration (can copy to reuse):

```yaml
name: "Q1_baseline"
dataset: "regression"
created_at: "2024-10-14T10:30:00Z"

steps:
  - class: "sklearn.preprocessing.StandardScaler"
    params: {}

  - class: "sklearn.decomposition.PCA"
    params:
      n_components: 15

  - class: "sklearn.cross_decomposition.PLSRegression"
    params:
      n_components: 5
```

#### `metadata.yaml`
Training metadata and environment:

```yaml
training:
  started_at: "2024-10-14T10:30:00Z"
  finished_at: "2024-10-14T10:35:22Z"
  duration_seconds: 322

dataset:
  name: "regression"
  n_samples_train: 80
  n_samples_test: 20
  n_features: 100
  target_type: "continuous"

environment:
  python_version: "3.11.5"
  sklearn_version: "1.3.2"
  nirs4all_version: "0.6.0"

artifacts:
  total_count: 2
  total_size_bytes: 25600
  deduplication_ratio: 1.0
```

#### `scores.yaml`
All metrics in one place:

```yaml
train:
  rmse: 0.45
  r2: 0.92
  mae: 0.32

test:
  rmse: 0.52
  r2: 0.89
  mae: 0.38

cross_validation:
  mean_rmse: 0.48
  std_rmse: 0.03
  folds: 5

best_params:
  n_components: 5

feature_importance:
  top_10: [2, 45, 67, 89, 12, 34, 56, 78, 90, 23]
```

#### `outputs/` Folder
All visualizations and charts:

```
outputs/
├── confusion_matrix.png
├── predictions_plot.png
├── feature_importance.png
├── learning_curve.png
├── residuals_plot.png
├── cross_validation_scores.png
└── shap_summary.png
```

#### `predictions/` Folder
All prediction CSVs:

```
predictions/
├── train_predictions.csv
├── test_predictions.csv
├── validation_predictions.csv
└── cross_val_predictions.csv
```

#### `binaries/` Folder
Symlinks to cached artifacts:

```
binaries/
├── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl
├── pca_1.pkl -> ../../.artifacts/PCA_def456.pkl
└── model_2.pkl -> ../../.artifacts/PLS_model_ghi789.pkl
```

**Why symlinks?**
- Multiple pipelines can share artifacts (e.g., same scaler)
- Saves disk space within a run
- User can browse binaries/ and see model files
- Export function resolves to physical files for portability

---

## Workflows

### Training Workflow

```python
from nirs4all.pipeline.runner import PipelineRunner

# 1. Create runner with run ID
runner = PipelineRunner(
    config=pipeline_config,
    results_dir="results/",
    run_id="2024-10-14_wheat_quality"
)

# 2. Train pipeline (runner handles artifacts)
runner.run(
    dataset_name="regression",
    pipeline_name="Q1_baseline",
    mode="train"
)

# Result:
# - Creates: results/2024-10-14_wheat_quality/regression_Q1_c20f9b/
# - Saves artifacts to: results/2024-10-14_wheat_quality/.artifacts/
# - Creates symlinks in: regression_Q1_c20f9b/binaries/
# - Writes: pipeline.yaml, metadata.yaml, scores.yaml
# - Saves charts to: outputs/
# - Saves predictions to: predictions/
```

**How artifact caching works:**

```python
# Inside controller.execute():
def execute(self, context, runner):
    # 1. Train transformer
    scaler = StandardScaler()
    scaler.fit(X)

    # 2. Serialize and compute hash
    data = cloudpickle.dumps(scaler)
    hash_value = hashlib.sha256(data).hexdigest()[:6]

    # 3. Save to run-level cache
    artifact_name = f"StandardScaler_{hash_value}.pkl"
    artifact_path = runner.run_dir / ".artifacts" / artifact_name

    if not artifact_path.exists():
        artifact_path.write_bytes(data)  # Save only if new

    # 4. Create symlink in pipeline binaries/
    symlink_path = runner.pipeline_dir / "binaries" / "scaler_0.pkl"
    symlink_path.symlink_to(f"../../.artifacts/{artifact_name}")

    # 5. Update metadata
    return context, artifact_metadata
```

### Prediction Workflow

```python
# 1. Load pipeline
runner = PipelineRunner.from_pipeline(
    run_id="2024-10-14_wheat_quality",
    dataset_name="regression",
    pipeline_name="Q1_baseline"
)

# 2. Load artifacts via symlinks
runner.load_artifacts()  # Resolves symlinks → loads from .artifacts/

# 3. Predict
predictions = runner.predict(new_data)

# 4. Save predictions
runner.save_predictions(predictions, "new_batch_predictions.csv")
```

**How artifact loading works:**

```python
def load_artifacts(self):
    binaries_dir = self.pipeline_dir / "binaries"

    for symlink in binaries_dir.glob("*.pkl"):
        # 1. Resolve symlink
        target = symlink.resolve()  # Points to .artifacts/StandardScaler_abc123.pkl

        # 2. Load artifact
        with open(target, "rb") as f:
            obj = cloudpickle.load(f)

        # 3. Store in context
        self.artifacts[symlink.stem] = obj
```

### Export Workflow

**Problem:** Symlinks break when copying folder or sharing with colleagues.

**Solution:** Export function that resolves symlinks and creates self-contained package.

```python
from nirs4all.pipeline.export import export_pipeline

# Export single pipeline
export_pipeline(
    run_id="2024-10-14_wheat_quality",
    pipeline_name="regression_Q1_c20f9b",
    output="wheat_pls_model.zip"
)

# Creates zip with:
# - pipeline.yaml
# - metadata.yaml
# - scores.yaml
# - outputs/ (all charts)
# - predictions/ (all CSVs)
# - binaries/ (PHYSICAL FILES, not symlinks!)
# - EXPORT_INFO.yaml (provenance)
```

**Export function implementation:**

```python
def export_pipeline(run_id: str, pipeline_name: str, output: str):
    """Create self-contained portable package."""
    import zipfile
    import shutil
    from pathlib import Path

    # 1. Find pipeline folder
    pipeline_dir = Path("results") / run_id / pipeline_name

    # 2. Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / pipeline_name

        # 3. Copy all files except binaries/
        for item in pipeline_dir.iterdir():
            if item.name != "binaries":
                if item.is_dir():
                    shutil.copytree(item, export_dir / item.name)
                else:
                    shutil.copy2(item, export_dir / item.name)

        # 4. Resolve symlinks and copy physical files
        binaries_dir = export_dir / "binaries"
        binaries_dir.mkdir()

        for symlink in (pipeline_dir / "binaries").glob("*"):
            target = symlink.resolve()  # Get actual file
            shutil.copy2(target, binaries_dir / symlink.name)

        # 5. Add export info
        export_info = {
            "exported_at": datetime.now().isoformat(),
            "original_run": run_id,
            "original_pipeline": pipeline_name,
            "artifacts_resolved": True,
            "artifact_hashes": {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in binaries_dir.glob("*")
            }
        }

        with open(export_dir / "EXPORT_INFO.yaml", "w") as f:
            yaml.dump(export_info, f)

        # 6. Create zip
        shutil.make_archive(output.replace(".zip", ""), "zip", tmpdir)

    print(f"✓ Exported to {output}")
    print(f"  - All symlinks resolved to physical files")
    print(f"  - Package is self-contained and portable")
```

**Exported package structure:**

```
wheat_pls_model.zip
└── regression_Q1_c20f9b/
    ├── pipeline.yaml
    ├── metadata.yaml
    ├── scores.yaml
    ├── outputs/
    ├── predictions/
    ├── binaries/
    │   ├── scaler_0.pkl          # PHYSICAL FILE (not symlink!)
    │   └── model_1.pkl            # PHYSICAL FILE (not symlink!)
    └── EXPORT_INFO.yaml           # Provenance
```

**Friend receives:**
- ✅ Complete pipeline package
- ✅ No broken symlinks
- ✅ No dependency on your run folder
- ✅ Ready to use immediately

### Cleanup Workflows

#### Delete Entire Run

```bash
# Everything is self-contained
rm -rf results/2024-10-14_wheat_quality/

# Or multiple runs
rm -rf results/2024-09-*
rm -rf results/2024-10-0[1-9]_*
```

**Effect:**
- ✅ Removes all pipelines in run
- ✅ Removes artifact cache
- ✅ No orphaned files
- ✅ No database to update

#### Delete Specific Pipeline

```bash
rm -rf results/2024-10-14_wheat_quality/regression_Q1_c20f9b/
```

**Effect:**
- ✅ Removes pipeline folder
- ⚠️ Artifacts remain in `.artifacts/` (may be used by other pipelines)

#### Clean Unused Artifacts

Artifacts not referenced by any symlink can be safely deleted:

```python
# CLI tool
nirs4all clean-artifacts --run 2024-10-14_wheat_quality --dry-run
nirs4all clean-artifacts --run 2024-10-14_wheat_quality --force

# Or for all runs
nirs4all clean-artifacts --all --older-than 30d
```

**Implementation:**

```python
def clean_unused_artifacts(run_dir: Path, dry_run: bool = True):
    """Remove artifacts not referenced by any symlink."""
    artifacts_dir = run_dir / ".artifacts"

    # 1. Collect all artifacts
    all_artifacts = set()
    for artifact in artifacts_dir.glob("*"):
        if artifact.is_file():
            all_artifacts.add(artifact.name)

    # 2. Collect all referenced artifacts (via symlinks)
    referenced = set()
    for pipeline_dir in run_dir.iterdir():
        if pipeline_dir.is_dir() and not pipeline_dir.name.startswith("."):
            binaries_dir = pipeline_dir / "binaries"
            if binaries_dir.exists():
                for symlink in binaries_dir.glob("*"):
                    if symlink.is_symlink():
                        target = symlink.resolve()
                        referenced.add(target.name)

    # 3. Find orphans
    orphaned = all_artifacts - referenced

    # 4. Report and optionally delete
    if orphaned:
        total_size = sum((artifacts_dir / name).stat().st_size for name in orphaned)
        print(f"Found {len(orphaned)} orphaned artifacts ({total_size / 1024:.1f} KB)")

        if not dry_run:
            for name in orphaned:
                (artifacts_dir / name).unlink()
            print(f"✓ Deleted {len(orphaned)} artifacts")
    else:
        print("✓ No orphaned artifacts found")
```

---

## Advantages

### 1. Chronological Organization

**Date-first sorting:**
```
results/
├── 2024-09-15_initial_tests/
├── 2024-09-20_feature_engineering/
├── 2024-10-01_model_tuning/
├── 2024-10-14_wheat_quality/        ← Easy to find recent work
└── 2024-10-15_corn_analysis/
```

**Benefits:**
- File browser shows newest work first (if sorted descending)
- Easy to find work from specific date
- Natural temporal grouping

### 2. Self-Contained Runs

**Each run is independent:**
- Has its own artifact cache
- No dependency on global state
- Can move/copy entire run folder
- Can delete without affecting other runs

**Benefits:**
- Easy backup (copy one folder)
- Easy archiving (zip one folder)
- Easy sharing (with export function)
- Easy cleanup (delete one folder)

### 3. Symlink-Based Deduplication

**Within a run, identical artifacts shared:**

Example: 3 pipelines use same StandardScaler
```
.artifacts/
└── StandardScaler_abc123.pkl      # Stored once (10 KB)

regression_Q1_c20f9b/binaries/
└── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl

regression_Q2_xyz789/binaries/
└── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl  # Same file!

classification_Q3_abc123/binaries/
└── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl  # Same file!
```

**Storage saved:** 10 KB × 3 = 30 KB → 10 KB (saved 20 KB)

**Benefits:**
- Deduplication where it matters (within run)
- No global cache complexity
- Symlinks are transparent to user
- Export function resolves symlinks for portability

### 4. Human-Readable Names

**Artifact names include operator type:**
- `StandardScaler_abc123.pkl` - Know it's a scaler
- `PLS_model_def456.pkl` - Know it's a PLS model
- `RandomForest_ghi789.pkl` - Know it's a Random Forest

**Benefits:**
- Can identify artifacts without loading
- Easier debugging
- Clearer than pure hashes (`abc123...pkl`)

### 5. Simple Cleanup

**No cascading deletes, no orphan tracking:**
```bash
# Delete run → everything gone
rm -rf results/2024-10-14_wheat_quality/

# Delete pipeline → artifacts remain for other pipelines
rm -rf results/2024-10-14_wheat_quality/regression_Q1_c20f9b/

# Clean orphans → simple script
nirs4all clean-artifacts --run 2024-10-14_wheat_quality
```

**Benefits:**
- No database to update
- No global index to maintain
- Simple filesystem operations
- Clear what gets deleted

### 6. Export for Portability

**Symlinks work locally, export works remotely:**

Local work:
```
binaries/
└── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl
```

Exported package:
```
binaries/
└── scaler_0.pkl  # Physical file (not symlink)
```

**Benefits:**
- Best of both worlds
- Efficient local storage
- Portable shared packages
- No broken symlinks when sharing

---

## Comparison to Previous Proposals

### vs. Global Artifacts (Proposal 3)

| Aspect | Global Artifacts | Run-Based (This) |
|--------|-----------------|------------------|
| Deduplication | Across all runs | Within each run |
| Cleanup | Complex (orphan tracking) | Simple (delete folder) |
| Portability | Needs export + global cache | Needs export only |
| Organization | UID-based (opaque) | Date-based (clear) |
| State management | Dataset indexes required | No indexes needed |
| Sharing | Export resolves global refs | Export resolves local refs |

**Winner:** Run-based for simplicity

### vs. Flat Timeline (Proposal 2)

| Aspect | Flat Timeline | Run-Based (This) |
|--------|--------------|------------------|
| Grouping | All pipelines flat | Grouped by run |
| Deduplication | None | Within run (symlinks) |
| Organization | Date per pipeline | Date per run |
| Cleanup | Delete individual pipelines | Delete run or pipeline |

**Winner:** Run-based for grouping and deduplication

### vs. Project-Centric (Proposal 1)

| Aspect | Project-Centric | Run-Based (This) |
|--------|----------------|------------------|
| Organization | By dataset/project | By run (temporal) |
| Chronology | Not primary | Primary (date-first) |
| Deduplication | None | Within run (symlinks) |
| Run grouping | Not explicit | Explicit (run folder) |

**Winner:** Run-based for temporal organization

---

## Implementation Plan

### Phase 1: Core Structure (Week 1)

**Tasks:**
- [x] Update `PipelineRunner` to accept `run_id` parameter
- [ ] Create run directory structure: `Date_runid/`
- [ ] Create pipeline subdirectories: `dataset_pipelineid/`
- [ ] Create `.artifacts/` folder at run level
- [ ] Update file paths in all save operations

**Deliverable:** Basic structure working

### Phase 2: Artifact Caching (Week 2)

**Tasks:**
- [ ] Implement content hashing for artifacts
- [ ] Generate human-readable names: `OperatorName_hash.pkl`
- [ ] Save artifacts to `.artifacts/` folder
- [ ] Create symlinks in `binaries/` folder
- [ ] Update metadata to track artifact info

**Deliverable:** Symlink-based caching working

### Phase 3: Export Function (Week 3)

**Tasks:**
- [ ] Implement `export_pipeline()` function
- [ ] Resolve symlinks to physical files
- [ ] Create self-contained zip packages
- [ ] Add `EXPORT_INFO.yaml` with provenance
- [ ] Add CLI command: `nirs4all export`

**Deliverable:** Export creates portable packages

### Phase 4: Cleanup Tools (Week 3-4)

**Tasks:**
- [ ] Implement `clean_unused_artifacts()` function
- [ ] Add CLI command: `nirs4all clean-artifacts`
- [ ] Add `--dry-run` and `--force` flags
- [ ] Add statistics reporting
- [ ] Test with large runs

**Deliverable:** Cleanup tools working

### Phase 5: Testing & Documentation (Week 4)

**Tasks:**
- [ ] Integration tests for full workflow
- [ ] Test export function with symlinks
- [ ] Test cleanup with orphaned artifacts
- [ ] Update all documentation
- [ ] Create migration guide
- [ ] Performance testing

**Deliverable:** Production-ready

---

## Migration from Current Structure

### Current Structure (Before)

```
results/
└── corn_m5/
    └── svm_baseline/
        ├── pipeline.json
        ├── metadata.json
        └── binaries/
            ├── 0_StandardScaler.pkl
            └── 1_SVC_model.pkl
```

### New Structure (After)

```
results/
└── 2024-10-14_migration/
    ├── .artifacts/
    │   ├── StandardScaler_abc123.pkl
    │   └── SVC_model_def456.pkl
    └── corn_m5_svm_baseline_old001/
        ├── pipeline.yaml
        ├── metadata.yaml
        ├── scores.yaml
        ├── outputs/
        ├── predictions/
        └── binaries/
            ├── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl
            └── model_1.pkl -> ../../.artifacts/SVC_model_def456.pkl
```

### Migration Script

**Recommended:** Just re-run training with new code.

**If needed:**

```python
def migrate_old_pipeline(old_path: Path, run_id: str):
    """Migrate old pipeline to new structure."""

    # 1. Create run directory
    run_dir = Path("results") / run_id
    run_dir.mkdir(exist_ok=True)

    artifacts_dir = run_dir / ".artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # 2. Load old pipeline.json
    with open(old_path / "pipeline.json") as f:
        old_config = json.load(f)

    # 3. Create new pipeline folder
    dataset_name = old_path.parent.name
    pipeline_name = old_path.name
    new_pipeline_dir = run_dir / f"{dataset_name}_{pipeline_name}_migrated"
    new_pipeline_dir.mkdir()

    # 4. Convert pipeline.json → pipeline.yaml
    with open(new_pipeline_dir / "pipeline.yaml", "w") as f:
        yaml.dump(old_config, f)

    # 5. Migrate binaries
    binaries_dir = new_pipeline_dir / "binaries"
    binaries_dir.mkdir()

    for binary_file in (old_path / "binaries").glob("*.pkl"):
        # Compute hash
        data = binary_file.read_bytes()
        hash_value = hashlib.sha256(data).hexdigest()[:6]

        # Generate name from filename
        operator_name = binary_file.stem.split("_", 1)[-1]
        artifact_name = f"{operator_name}_{hash_value}.pkl"

        # Save to artifacts/
        artifact_path = artifacts_dir / artifact_name
        if not artifact_path.exists():
            artifact_path.write_bytes(data)

        # Create symlink
        symlink_path = binaries_dir / binary_file.name
        symlink_path.symlink_to(f"../../.artifacts/{artifact_name}")

    print(f"✓ Migrated: {old_path} → {new_pipeline_dir}")
```

---

## FAQ

### Q: Why per-run cache instead of global cache?

**A:** Simpler and more practical:
- Most duplicate artifacts happen within a run (same scaler for multiple models)
- Self-contained runs are easier to manage
- No global state to maintain
- Cleanup is trivial (delete folder)
- Cross-run deduplication rarely worth the complexity

### Q: What if symlinks don't work on my system?

**A:** Fallback options:
1. **Windows**: Requires admin rights or Developer Mode for symlinks
   - Fallback: Copy files instead of symlinking
   - Check with: `os.symlink` availability
2. **Export always works**: Creates physical copies
3. **Configuration**: `use_symlinks: false` → copy instead

### Q: How much space does deduplication save?

**A:** Depends on workflow:
- **Grid search** (same preprocessing, different models): 50-80% savings
- **Ensemble** (same models, different data splits): 30-50% savings
- **Single pipeline**: 0% (nothing to deduplicate)

Example: 10 pipelines with same StandardScaler (10 KB each)
- Without deduplication: 100 KB
- With deduplication: 10 KB (90% savings)

### Q: Can I manually add artifacts to cache?

**A:** Not recommended, but possible:
```python
import cloudpickle
import hashlib

# 1. Serialize object
data = cloudpickle.dumps(my_scaler)

# 2. Compute hash
hash_value = hashlib.sha256(data).hexdigest()[:6]

# 3. Save to cache
artifact_name = f"StandardScaler_{hash_value}.pkl"
cache_path = Path("results/2024-10-14_myrun/.artifacts") / artifact_name
cache_path.write_bytes(data)

# 4. Create symlink
symlink = Path("results/2024-10-14_myrun/regression_Q1/binaries/scaler_0.pkl")
symlink.symlink_to(f"../../.artifacts/{artifact_name}")
```

### Q: How do I list all pipelines in a run?

**A:**
```python
from pathlib import Path

run_dir = Path("results/2024-10-14_wheat_quality")

pipelines = [
    d for d in run_dir.iterdir()
    if d.is_dir() and not d.name.startswith(".")
]

for pipeline in pipelines:
    print(f"- {pipeline.name}")

    # Load scores
    with open(pipeline / "scores.yaml") as f:
        scores = yaml.safe_load(f)

    print(f"  Test RMSE: {scores['test']['rmse']:.3f}")
```

### Q: Can I rename a run after creation?

**A:** Yes, just rename the folder:
```bash
mv results/2024-10-14_wheat_quality results/2024-10-14_wheat_final
```

All symlinks are relative, so they still work!

### Q: How do I compare pipelines across runs?

**A:**
```python
def compare_pipelines(run1, pipeline1, run2, pipeline2):
    # Load scores
    scores1 = yaml.safe_load(
        open(f"results/{run1}/{pipeline1}/scores.yaml")
    )
    scores2 = yaml.safe_load(
        open(f"results/{run2}/{pipeline2}/scores.yaml")
    )

    # Compare
    print(f"{run1}/{pipeline1}: RMSE = {scores1['test']['rmse']}")
    print(f"{run2}/{pipeline2}: RMSE = {scores2['test']['rmse']}")
```

---

## Success Criteria

✅ **Complete when:**
1. Run-based directory structure implemented
2. Per-run artifact cache working
3. Symlink-based deduplication functional
4. Export function creates portable packages
5. Cleanup tools remove orphaned artifacts
6. Human-readable artifact names generated
7. All tests passing (unit + integration)
8. Documentation complete
9. Migration guide available
10. Performance acceptable (< 5% overhead)

---

## Summary

This architecture provides:

✅ **Simplicity** - Date-first organization, no global state
✅ **Efficiency** - Symlink deduplication within runs
✅ **Portability** - Export function resolves symlinks
✅ **Usability** - Human-readable names, easy cleanup
✅ **Self-contained** - Each run is independent
✅ **Temporal** - Chronological sorting by date

**Recommendation:** Implement this architecture for nirs4all v6.0.

---

**Document Status:** Complete and ready for implementation
**Last Updated:** October 14, 2025
**Next Steps:** Begin Phase 1 implementation
