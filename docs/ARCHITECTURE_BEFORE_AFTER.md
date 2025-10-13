# Architecture Comparison: Before → After

**Visual guide to the new run-based architecture**

---

## Structure Comparison

### BEFORE (Current/Proposal 3)

```
results/
├── artifacts/
│   └── objects/
│       ├── ab/
│       │   └── abc123...pkl          # Global cache
│       └── de/
│           └── def456...pkl
│
├── pipelines/
│   ├── a1b2c3d4-...-uuid1/          # Opaque UUID
│   │   └── manifest.yaml
│   └── b2c3d4e5-...-uuid2/
│       └── manifest.yaml
│
└── datasets/
    └── corn_m5/
        └── index.yaml                 # Name → UUID mapping
```

**Problems:**
- ❌ UUIDs are opaque (can't identify pipelines)
- ❌ Global cache complex to manage
- ❌ No chronological organization
- ❌ Cleanup requires orphan tracking
- ❌ Need dataset indexes for lookup

### AFTER (Run-Based)

```
results/
├── 2024-10-14_wheat_quality/         # Clear date + name
│   ├── .artifacts/                   # Hidden per-run cache
│   │   ├── StandardScaler_abc123.pkl # Human-readable!
│   │   └── PLS_model_def456.pkl
│   │
│   ├── regression_Q1_c20f9b/         # Dataset + pipeline visible
│   │   ├── pipeline.yaml
│   │   ├── metadata.yaml
│   │   ├── scores.yaml
│   │   ├── outputs/
│   │   ├── predictions/
│   │   └── binaries/
│   │       └── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl
│   │
│   └── regression_Q2_xyz789/
│       └── binaries/
│           └── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl  # Dedupe!
│
└── 2024-10-15_corn_analysis/
    └── ...
```

**Benefits:**
- ✅ Date-first: Chronological sorting
- ✅ Human-readable: Can identify everything
- ✅ Self-contained: Each run independent
- ✅ Simple cleanup: Delete folder
- ✅ No indexes needed: Direct lookup

---

## Workflow Comparison

### Training Pipeline

#### BEFORE
```python
# 1. Create pipeline (generates UUID)
uid = manifest_manager.create_pipeline("svm_baseline", "corn_m5", config)
# uid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# 2. Train
runner.run(...)

# 3. Artifacts saved globally
# → results/artifacts/objects/ab/abc123...pkl

# 4. Manifest references global artifacts
# → results/pipelines/a1b2c3d4-.../manifest.yaml

# 5. Register in dataset index
# → results/datasets/corn_m5/index.yaml
#    {"svm_baseline": "a1b2c3d4-..."}
```

**Problem**: 5 locations to track!

#### AFTER
```python
# 1. Create runner with run ID
runner = PipelineRunner(
    config=config,
    run_id="2024-10-14_wheat_quality"
)

# 2. Train
runner.run(
    dataset_name="regression",
    pipeline_name="Q1_baseline"
)

# 3. Artifacts saved to run cache
# → results/2024-10-14_wheat_quality/.artifacts/StandardScaler_abc123.pkl

# 4. Symlinks created
# → results/2024-10-14_wheat_quality/regression_Q1_c20f9b/binaries/
#    scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl

# 5. Done! Everything in one place
```

**Benefit**: 1 location - the run folder!

---

### Finding Pipelines

#### BEFORE
```python
# 1. Look up in dataset index
uid = manifest_manager.get_pipeline_uid("corn_m5", "svm_baseline")
# uid = "a1b2c3d4-..."

# 2. Load manifest
manifest = manifest_manager.load_manifest(uid)

# 3. Can't browse - UUIDs everywhere
# results/pipelines/a1b2c3d4-.../
#                   b2c3d4e5-.../
#                   c3d4e5f6-.../  # Which one is "svm_baseline"?
```

#### AFTER
```python
# Just browse folders!
results/
└── 2024-10-14_wheat_quality/
    ├── regression_Q1_c20f9b/     # Clear name!
    ├── regression_Q2_xyz789/
    └── classification_Q3_abc123/

# Or search by date
results/2024-10-14_*/regression_*/scores.yaml
```

**Benefit**: No lookup needed, just browse!

---

### Sharing Pipelines

#### BEFORE
```python
# Export must resolve global artifacts
export_pipeline(
    uid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    output="model.zip"
)

# Function must:
# 1. Find pipeline by UUID
# 2. Find all artifact references in manifest
# 3. Resolve paths: artifacts/objects/ab/abc123...pkl
# 4. Copy artifacts from global cache
# 5. Create zip
```

#### AFTER
```python
# Export resolves local symlinks
export_pipeline(
    run_id="2024-10-14_wheat_quality",
    pipeline_name="regression_Q1_c20f9b",
    output="model.zip"
)

# Function just:
# 1. Find pipeline folder (simple path)
# 2. Follow symlinks in binaries/
# 3. Copy files
# 4. Create zip
```

**Benefit**: Simpler export, no global state!

---

### Cleanup

#### BEFORE
```bash
# Delete pipeline
rm -rf results/pipelines/a1b2c3d4-.../

# Artifacts remain in global cache
# → Need garbage collection!

# Find orphaned artifacts
nirs4all gc-artifacts --scan-all
# Scans ALL manifests to find unused artifacts
# Complex and slow!

# Clean orphans
nirs4all gc-artifacts --force
```

#### AFTER
```bash
# Delete entire run (self-contained!)
rm -rf results/2024-10-14_wheat_quality/

# Done! Cache deleted with run.

# Or delete single pipeline
rm -rf results/2024-10-14_wheat_quality/regression_Q1_c20f9b/

# Clean orphans in run (if needed)
nirs4all clean-artifacts --run 2024-10-14_wheat_quality
# Only scans ONE run, not all runs!
```

**Benefit**: Simple, fast, no global scan!

---

## Deduplication Comparison

### Scenario: 3 pipelines with same StandardScaler

#### BEFORE (Global Cache)

```
artifacts/objects/ab/abc123...pkl  # StandardScaler (10 KB)

pipelines/
├── uuid1/manifest.yaml  # References: sha256:abc123...
├── uuid2/manifest.yaml  # References: sha256:abc123...
└── uuid3/manifest.yaml  # References: sha256:abc123...

Storage: 10 KB (deduplicated)
```

**Problem**: Delete pipeline → artifact remains → need GC

#### AFTER (Run Cache + Symlinks)

```
2024-10-14_wheat_quality/
├── .artifacts/
│   └── StandardScaler_abc123.pkl  # 10 KB (stored once)
│
├── regression_Q1_c20f9b/binaries/
│   └── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl
│
├── regression_Q2_xyz789/binaries/
│   └── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl
│
└── classification_Q3_abc123/binaries/
    └── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl

Storage: 10 KB (deduplicated via symlinks)
```

**Benefit**: Delete run → cache deleted → no GC needed!

---

## File Browser View

### BEFORE

```
📁 results/
├── 📁 artifacts/
│   └── 📁 objects/
│       ├── 📁 ab/
│       │   └── 📄 abc123def456ghi789...pkl  # What is this?
│       ├── 📁 cd/
│       ├── 📁 ef/
│       └── ...  (hundreds of hash folders)
│
├── 📁 pipelines/
│   ├── 📁 a1b2c3d4-e5f6-7890-abcd-ef1234567890/  # Which pipeline?
│   ├── 📁 b2c3d4e5-f6a7-8901-bcde-f12345678901/
│   ├── 📁 c3d4e5f6-a7b8-9012-cdef-123456789012/
│   └── ...  (can't identify without opening manifests)
│
└── 📁 datasets/
    └── 📁 corn_m5/
        └── 📄 index.yaml  # Need to read this to find pipelines
```

**User experience**: 😵 "Which folder is my wheat model?"

### AFTER

```
📁 results/
├── 📁 2024-09-15_initial_tests/         # Old work (archive?)
├── 📁 2024-09-20_feature_engineering/
├── 📁 2024-10-01_model_tuning/
├── 📁 2024-10-14_wheat_quality/         # Recent work!
│   ├── 📁 .artifacts/  (hidden)
│   ├── 📁 regression_Q1_c20f9b/         # Clear dataset name!
│   │   ├── 📄 pipeline.yaml
│   │   ├── 📄 scores.yaml               # Quick check metrics
│   │   ├── 📁 outputs/
│   │   │   └── 🖼️ predictions_plot.png  # Visual results
│   │   ├── 📁 predictions/
│   │   └── 📁 binaries/
│   │       └── 📄 model_1.pkl          # The model!
│   └── 📁 regression_Q2_xyz789/
│
└── 📁 2024-10-15_corn_analysis/         # Today's work!
```

**User experience**: 😊 "Easy! My wheat model is in 2024-10-14!"

---

## Real-World Example

### Scenario: Grid Search (10 models, same preprocessing)

#### BEFORE
```
artifacts/objects/
├── ab/abc123...pkl  # StandardScaler (10 KB)
├── cd/cde234...pkl  # Model 1 (50 KB)
├── ef/efg345...pkl  # Model 2 (50 KB)
├── ...              # Models 3-10 (50 KB each)

pipelines/
├── uuid_model1/manifest.yaml
├── uuid_model2/manifest.yaml
├── ...
└── uuid_model10/manifest.yaml

Total storage: 10 KB + (50 KB × 10) = 510 KB
Deduplication: 1 scaler shared
```

#### AFTER
```
2024-10-14_grid_search/
├── .artifacts/
│   ├── StandardScaler_abc123.pkl  # 10 KB (shared!)
│   ├── Model_C1_cde234.pkl       # 50 KB
│   ├── Model_C10_efg345.pkl      # 50 KB
│   └── ...                        # Models C2-C9
│
├── regression_C1_model1/binaries/
│   ├── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl  # Symlink!
│   └── model_1.pkl -> ../../.artifacts/Model_C1_cde234.pkl
│
├── regression_C2_model2/binaries/
│   ├── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl  # Same symlink!
│   └── model_1.pkl -> ../../.artifacts/Model_C2_efg345.pkl
│
└── ...  (models 3-10)

Total storage: 10 KB + (50 KB × 10) = 510 KB (same)
Deduplication: 1 scaler, 9 symlinks
```

**Same efficiency, better organization!**

---

## Migration Visualization

### From Current Structure

```
results/corn_m5/svm_baseline/
├── pipeline.json
├── metadata.json
└── binaries/
    ├── 0_StandardScaler.pkl
    └── 1_SVC_model.pkl
```

### To Run-Based Structure

```
results/2024-10-14_migration/
├── .artifacts/
│   ├── StandardScaler_abc123.pkl  # Hashed + cached
│   └── SVC_model_def456.pkl
│
└── corn_m5_svm_baseline_migrated/
    ├── pipeline.yaml               # Converted from .json
    ├── metadata.yaml
    ├── scores.yaml
    ├── outputs/
    ├── predictions/
    └── binaries/                   # Symlinks created!
        ├── scaler_0.pkl -> ../../.artifacts/StandardScaler_abc123.pkl
        └── model_1.pkl -> ../../.artifacts/SVC_model_def456.pkl
```

---

## Summary Table

| Aspect | BEFORE (Global) | AFTER (Run-Based) |
|--------|----------------|-------------------|
| **Organization** | UUID-based (opaque) | Date-based (chronological) |
| **Deduplication** | Global cache | Per-run symlinks |
| **Cleanup** | Complex (GC needed) | Simple (delete folder) |
| **Browsing** | Need index lookup | Direct filesystem browse |
| **Sharing** | Export resolves global | Export resolves local |
| **State** | Global indexes | Self-contained runs |
| **Human-readable** | UUIDs everywhere | Dates, names, descriptions |
| **Artifacts** | `abc123...pkl` | `StandardScaler_abc123.pkl` |
| **Portability** | Export creates package | Export creates package |
| **Efficiency** | Dedupe across all runs | Dedupe within runs |

**Winner**: Run-based for simplicity and usability! ✅

---

## Key Insight

> **"Most deduplication happens within a run, not across runs."**

When doing a grid search or ensemble, you run multiple pipelines with:
- ✅ Same preprocessing (StandardScaler, PCA, etc.)
- ✅ Different models or hyperparameters
- ✅ Same dataset

This is where deduplication matters! And run-based caching captures exactly this pattern.

Cross-run deduplication is rare because:
- Different runs often use different data
- Different preprocessing strategies
- Retraining changes artifacts

**Conclusion**: Per-run cache is the sweet spot! 🎯

---

**See also:**
- `docs/RUN_BASED_ARCHITECTURE.md` - Complete specification
- `docs/RUN_BASED_ARCHITECTURE_SUMMARY.md` - Quick summary
