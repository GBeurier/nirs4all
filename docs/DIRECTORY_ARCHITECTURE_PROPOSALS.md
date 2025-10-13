# Directory Architecture Proposals for nirs4all

## Current Situation Analysis

**Problem Statement:**
The current system has multiple file storage patterns that are confusing for users:
- Content-addressed artifacts in `results/artifacts/objects/XX/hash.ext` (unreadable hashes)
- Pipeline configs in `results/pipelines/<UUID>/pipeline.json` + `manifest.yaml` (UUID is unreadable)
- Dataset configs in `results/datasets/<name>/config_name/` folders
- Human-readable outputs in `results/outputs/<dataset>_<pipeline>/`
- Prediction CSVs scattered in various locations
- Full UUID identifiers everywhere (c40ca32c-43d7-4e1e-80c4-7f52de42b7a5) making navigation impossible

**User Pain Points:**
1. ❌ UUIDs are completely unreadable - users can't find their work
2. ❌ Hash-based artifact names prevent browsing
3. ❌ Multiple metadata files (pipeline.json, metadata.json, manifest.yaml) create confusion
4. ❌ No clear hierarchy between datasets, pipelines, and results
5. ❌ Files scattered across many directories
6. ⚠️ However, users DO appreciate knowing where their files are

**What Works:**
- ✅ Content-addressed storage prevents duplication (technical efficiency)
- ✅ Metadata tracking is comprehensive
- ✅ Recent outputs directory is user-friendly
- ✅ Users like having organized folders, not a flat structure

---

## Proposal 1: Project-Centric with Human-Readable IDs

### Philosophy
**"Projects are the unit of work"** - Users think in terms of experiments/projects, not individual pipeline runs. Use shortened readable IDs (6-8 chars) instead of full UUIDs. **All artifacts stay WITH the run for easy sharing.**

### Complete Directory Structure
```
results/
├── projects/
│   ├── wheat_quality_2024/                    # User-named project folder
│   │   ├── project.yaml                       # Project metadata, notes, objectives
│   │   ├── runs/
│   │   │   ├── run_pb8yky/                    # PLS-17_components run [pb8yky]
│   │   │   │   ├── summary.yaml               # ✨ ALL metadata in ONE file
│   │   │   │   │                              #    - Pipeline config
│   │   │   │   │                              #    - Dataset info
│   │   │   │   │                              #    - Scores (RMSE, R², etc.)
│   │   │   │   │                              #    - Timestamps
│   │   │   │   │                              #    - Fold results
│   │   │   │   ├── pipeline.yaml              # 📋 Standalone pipeline definition
│   │   │   │   │                              #    (can copy to reuse elsewhere)
│   │   │   │   ├── predictions/               # 📊 All prediction files
│   │   │   │   │   ├── best_prediction.csv    #    Main: y_true, y_pred, residuals
│   │   │   │   │   ├── fold_0_predictions.csv #    Individual fold results
│   │   │   │   │   ├── fold_1_predictions.csv
│   │   │   │   │   ├── fold_2_predictions.csv
│   │   │   │   │   └── train_predictions.csv  #    Training set predictions
│   │   │   │   ├── charts/                    # 📈 All visualizations
│   │   │   │   │   ├── 2D_Chart.png           #    Spectra visualization (2D)
│   │   │   │   │   ├── 3D_Chart_src0.png      #    Spectra 3D (if applicable)
│   │   │   │   │   ├── Y_distribution_train_test.png  # Y distribution
│   │   │   │   │   ├── fold_visualization_3folds_train.png  # CV folds
│   │   │   │   │   ├── fold_visualization_3folds_test.png
│   │   │   │   │   ├── predictions_scatter.png         # Pred vs True
│   │   │   │   │   ├── residuals_plot.png              # Residuals analysis
│   │   │   │   │   └── feature_importance.png          # If model supports it
│   │   │   │   ├── scores/                    # 📊 Detailed scoring files
│   │   │   │   │   ├── cross_validation_scores.csv     # All folds scores
│   │   │   │   │   ├── train_scores.yaml      #    Train metrics
│   │   │   │   │   ├── test_scores.yaml       #    Test metrics
│   │   │   │   │   └── summary_table.txt      #    Pretty table (console output)
│   │   │   │   ├── models/                    # 🔧 All artifacts (SELF-CONTAINED)
│   │   │   │   │   ├── pls_17comp_pb8yky.pkl  #    Trained model
│   │   │   │   │   ├── scaler_minmax_pb8yky.pkl     # Preprocessing artifacts
│   │   │   │   │   ├── transformer_1stder_pb8yky.pkl
│   │   │   │   │   ├── transformer_haar_pb8yky.pkl
│   │   │   │   │   └── MANIFEST.yaml          #    List of all artifacts with metadata
│   │   │   │   └── logs/                      # 📝 Execution logs
│   │   │   │       ├── console_output.log     #    Full console output
│   │   │   │       ├── warnings.log           #    Warnings/errors
│   │   │   │       └── timing.yaml            #    Step execution times
│   │   │   │
│   │   │   ├── run_5grd70/                    # PLS-15_components run [5grd70]
│   │   │   │   ├── summary.yaml               # ✨ Best model: RMSE 6.8466
│   │   │   │   ├── pipeline.yaml              # 📋 Slightly different config
│   │   │   │   ├── predictions/
│   │   │   │   │   ├── best_prediction.csv
│   │   │   │   │   └── ...
│   │   │   │   ├── charts/
│   │   │   │   │   ├── 2D_Chart.png
│   │   │   │   │   └── ...
│   │   │   │   ├── scores/
│   │   │   │   │   └── ...
│   │   │   │   ├── models/                    # 🔧 Complete artifact set
│   │   │   │   │   ├── pls_15comp_5grd70.pkl
│   │   │   │   │   ├── scaler_minmax_5grd70.pkl
│   │   │   │   │   └── ...
│   │   │   │   └── logs/
│   │   │   │       └── ...
│   │   │   │
│   │   │   └── run_2e3nwn/                    # Classification run
│   │   │       ├── summary.yaml               # Different metrics (accuracy, f1, etc.)
│   │   │       ├── pipeline.yaml
│   │   │       ├── predictions/
│   │   │       │   ├── best_prediction.csv    # class labels, probabilities
│   │   │       │   └── confusion_matrix.csv
│   │   │       ├── charts/
│   │   │       │   ├── confusion_matrix.png
│   │   │       │   ├── roc_curve.png
│   │   │       │   └── class_distribution.png
│   │   │       ├── scores/
│   │   │       │   └── classification_report.txt
│   │   │       └── models/
│   │   │           └── random_forest_2e3nwn.pkl
│   │   │
│   │   ├── datasets/                          # 📦 Dataset configs
│   │   │   ├── regression/
│   │   │   │   ├── config.yaml                # Dataset configuration
│   │   │   │   └── metadata.yaml              # Features, samples, statistics
│   │   │   └── classification_Xtrain/
│   │   │       └── config.yaml
│   │   │
│   │   └── comparisons/                       # 📊 Cross-run comparisons
│   │       ├── all_runs_comparison.csv        # Compare all runs in project
│   │       └── best_models_report.html        # Visual comparison
│   │
│   ├── protein_prediction/                    # Another project
│   │   └── ...
│   │
│   └── default/                               # Auto-created if no project specified
│       └── runs/
│           └── run_xxxxxx/
│               └── ...
│
└── archive/                                   # Old runs (auto-archived after 90 days)
    └── 2024-07_wheat_quality/
        └── runs/
            └── run_old123/
                └── ... (complete structure preserved)
```

### Real Example - Complete Run Folder

**Example: `results/projects/wheat_quality_2024/runs/run_pb8yky/`**

This is the **winning model** from Q1 regression example:
- Model: PLS-17_components
- Test RMSE: 13.3989
- Val RMSE: 6.9658

```
run_pb8yky/
├── summary.yaml                    # 4.2 KB - All metadata
│   ├── run_id: "pb8yky"
│   ├── pipeline_name: "Q1_c20f9b"
│   ├── dataset: "regression"
│   ├── model: "PLS-17_components"
│   ├── created_at: "2024-10-14T00:22:31"
│   ├── scores:
│   │   ├── test: {rmse: 13.3989, r2: 0.546, mae: 9.763}
│   │   ├── val: {rmse: 6.9658, r2: 0.878, mae: 5.389}
│   │   └── train: {rmse: 4.904, r2: 0.957, mae: 3.129}
│   ├── fold_results: [...]
│   └── processing_steps: [...]
│
├── pipeline.yaml                   # 2.1 KB - Portable pipeline config
│   ├── steps:
│   │   ├── MinMax scaler
│   │   ├── 1st Derivative
│   │   ├── Haar wavelet
│   │   ├── MinMax scaler (again)
│   │   └── PLS (n_components: 17)
│   └── ... (complete, can copy/paste to new project)
│
├── predictions/
│   ├── best_prediction.csv         # 4.8 KB - Main results
│   │   # Columns: index, y_true, y_pred, residual, fold
│   │   # 59 rows (test set)
│   │
│   ├── fold_0_predictions.csv      # Individual fold predictions
│   ├── fold_1_predictions.csv
│   ├── fold_2_predictions.csv
│   ├── avg_predictions.csv         # Average across folds
│   ├── w_avg_predictions.csv       # Weighted average
│   └── train_predictions.csv       # Training set (130 samples)
│
├── charts/
│   ├── 2D_Chart.png                # 156 KB - All spectra overlaid
│   ├── Y_distribution_train_test.png  # 89 KB - Y value distributions
│   ├── fold_visualization_3folds_train.png  # 124 KB - CV visualization
│   ├── predictions_scatter.png     # 142 KB - Predicted vs True
│   └── residuals_plot.png          # 98 KB - Residual analysis
│
├── scores/
│   ├── cross_validation_scores.csv # All metrics for each fold
│   │   # Columns: fold, rmse, r2, mae, sep, rpd, bias, consistency
│   │
│   ├── train_scores.yaml           # Detailed train metrics
│   ├── test_scores.yaml            # Detailed test metrics
│   └── summary_table.txt           # Pretty ASCII table (console output)
│       # The table you see in console with all metrics
│
├── models/                         # 🎯 COMPLETE ARTIFACT SET
│   ├── pls_model_pb8yky.pkl        # 2.4 MB - Trained PLS model
│   ├── scaler_minmax_1_pb8yky.pkl  # 12 KB - First MinMax scaler
│   ├── scaler_minmax_2_pb8yky.pkl  # 12 KB - Second MinMax scaler
│   ├── transformer_1stder_pb8yky.pkl  # 8 KB - 1st Derivative transformer
│   ├── transformer_haar_pb8yky.pkl    # 24 KB - Haar wavelet transformer
│   └── MANIFEST.yaml               # Index of all artifacts
│       # Lists each artifact with:
│       # - filename
│       # - type (model/scaler/transformer)
│       # - step_number
│       # - sha256 hash (for verification)
│       # - size
│
└── logs/
    ├── console_output.log          # Full console output (ANSI colors stripped)
    ├── warnings.log                # Any warnings during execution
    └── timing.yaml                 # Execution time per step
```

### Sharing the Pipeline with a Friend 🎁

**Problem Addressed:** How to share a working pipeline including all artifacts?

**Solution: Self-Contained Run Folder**

```bash
# Option 1: Share the entire run folder
zip -r wheat_model_pb8yky.zip results/projects/wheat_quality_2024/runs/run_pb8yky/
# Send to friend: wheat_model_pb8yky.zip (contains EVERYTHING)

# Friend extracts and can:
# 1. View summary.yaml to understand the model
# 2. Copy pipeline.yaml to create new runs
# 3. Use models/*.pkl for predictions
# 4. See all charts and scores
```

**What the friend gets:**
- ✅ Complete pipeline definition (`pipeline.yaml`)
- ✅ All trained models and transformers (`models/`)
- ✅ Performance metrics (`scores/`, `summary.yaml`)
- ✅ Visual validation (`charts/`)
- ✅ Example predictions (`predictions/`)
- ✅ **NO dependency on a shared artifact cache!**

**Using the shared pipeline:**

```python
# Friend loads your pipeline
from nirs4all import PipelineRunner

# Option 1: Use your exact pipeline config
runner = PipelineRunner()
predictions = runner.predict_from_pipeline(
    pipeline_path="run_pb8yky/pipeline.yaml",
    model_artifacts="run_pb8yky/models/",
    new_data="my_samples.csv"
)

# Option 2: Copy pipeline.yaml to their project and run
# results/projects/my_own_study/pipelines/wheat_method.yaml
runner.run(
    pipeline="wheat_method.yaml",  # Your pipeline
    dataset="my_dataset"           # Their data
)
```

### Naming Convention
- **Projects**: User-defined names (e.g., `wheat_quality_2024`)
- **Runs**: `run_<6-char-id>` (e.g., `run_pb8yky` from console output `[pb8yky]`)
- **Artifacts**: `<type>_<descriptor>_<id>.pkl` (e.g., `pls_17comp_pb8yky.pkl`)
- **Predictions**: Descriptive (e.g., `best_prediction.csv`, `fold_0_predictions.csv`)
- **Charts**: Descriptive (e.g., `2D_Chart.png`, `Y_distribution_train_test.png`)

### Rationale

**Pros:**
- ✅ **Complete isolation** - Each run is self-contained
- ✅ **Easy sharing** - Zip one folder, includes everything
- ✅ **No broken links** - All artifacts physically present
- ✅ **Project organization** - Related runs grouped together
- ✅ **Short memorable IDs** - `pb8yky` visible in console and folder name
- ✅ **Portable pipelines** - `pipeline.yaml` can be copied anywhere
- ✅ **Clear structure** - predictions/, charts/, scores/, models/ separate
- ✅ **One metadata file** - `summary.yaml` instead of 3 JSONs

**Cons:**
- ⚠️ **Disk space** - Models duplicated across runs (but realistic for ML)
- ⚠️ **Project naming** - Requires user to think about organization
- ⚠️ **More files** - But organized into clear subdirectories

**Why No Shared Artifacts Cache?**

In Proposal 1, we **prioritize portability over deduplication**:

1. **ML Reality:** Trained models are unique to each run's data/params
2. **Sharing:** Users want to share complete working pipelines
3. **Backup:** Each run is independently archivable
4. **Simplicity:** No symlinks or references to resolve
5. **Typical Size:** A complete run with all artifacts is ~5-20 MB (reasonable)

**When deduplication matters:** If you run 1000s of experiments and disk space is critical, see Proposal 3's artifact library approach.

### Implementation Notes
- Projects created explicitly: `runner.run(..., project="wheat_quality")`
- If no project specified, use `projects/default/`
- Short hash: Use existing console output ID (`[pb8yky]`)
- All artifacts saved directly to `models/` subfolder
- `MANIFEST.yaml` tracks artifact relationships
- Auto-archive moves runs to `archive/<date>_<project>/` after 90 days

---

## Proposal 2: Flat Timeline with Smart Naming

### Philosophy
**"Chronological browsing"** - Users want to see recent work first. Use timestamp + descriptive names + short IDs.

### Directory Structure
```
results/
├── runs/
│   ├── 2024-10-14_00h22_regression_PLS-17comp_pb8yky/    # Date + Dataset + Model + ShortID
│   │   ├── run.yaml                     # All metadata in ONE file
│   │   ├── predictions.csv
│   │   ├── 2D_Chart.png
│   │   ├── 3D_Chart.png
│   │   ├── Y_distribution.png
│   │   ├── fold_visualization.png
│   │   └── models/
│   │       ├── pls_model.pkl
│   │       └── minmax_scaler.pkl
│   ├── 2024-10-14_00h23_classification_RFC-30depth_2e3nwn/
│   │   └── ...
│   └── 2024-10-14_00h24_regression_PLS-15comp_5grd70/    # Best from Q1
│       └── ...
├── datasets/
│   ├── regression.yaml                  # Dataset config (NOT in separate folder)
│   ├── regression.nirs4all              # Dataset data archive
│   ├── regression_2.yaml
│   └── classification_Xtrain.yaml
└── cache/                               # Hidden cache for deduplication
    └── objects/
        └── ab/abc123.pkl
```

### Naming Convention
- **Runs**: `YYYY-MM-DD_HHhMMmSSs_<dataset>_<model>_<shortid>`
  - Example: `2024-10-14_00h22_regression_PLS-17comp_pb8yky`
- **Short ID**: Last 6 chars from pipeline UID
- **Model Name**: Simplified (e.g., `PLS-17comp` instead of full class name)

### Rationale
**Pros:**
- ✅ Chronological sorting - latest runs at top when sorted by name
- ✅ ALL information in folder name (dataset, model, ID)
- ✅ Zero navigation depth - all runs at same level
- ✅ No sub-folders to dig through
- ✅ Single `run.yaml` file instead of multiple JSONs
- ✅ Very fast browsing (ls/dir shows everything)
- ✅ Short IDs for reference (`pb8yky`)

**Cons:**
- ⚠️ Long folder names (but descriptive)
- ⚠️ Flat structure can become crowded (100+ runs)
- ⚠️ Need to clean old runs manually
- ⚠️ Models duplicated (no deduplication)

### Implementation Notes
- Generate folder name from: `f"{timestamp}_{dataset.name}_{model_name}_{short_id}"`
- Model name extracted from pipeline step config
- All outputs directly in run folder (no subdirectories)
- Optional: Add `archive/` command to move old runs

---

## Proposal 3: Hybrid - Date-Organized with Named Experiments

### Philosophy
**"Best of both worlds"** - Organize by date for freshness, but allow grouping by experiment name. Use semantic IDs.

### Directory Structure
```
results/
├── 2024-10/                             # Month folders
│   ├── wheat_study/                     # User experiment name (optional)
│   │   ├── 14_regression_PLS17_pb8yky/  # Day + Dataset + Model + ID
│   │   │   ├── summary.yaml             # Single metadata file
│   │   │   ├── predictions.csv
│   │   │   ├── charts/
│   │   │   │   ├── 2D_Chart.png
│   │   │   │   └── Y_distribution.png
│   │   │   └── artifacts/
│   │   │       └── pls_model_pb8yky.pkl # ID in filename
│   │   ├── 14_regression_PLS15_5grd70/
│   │   │   └── ...
│   │   └── experiment.yaml              # Experiment-level notes/config
│   ├── protein_pred/
│   │   └── ...
│   └── quick_tests/                     # Default experiment name
│       └── 14_classification_RFC_2e3nwn/
│           └── ...
├── datasets/
│   ├── regression/
│   │   ├── config.yaml                  # Single config file
│   │   └── data.nirs4all                # Optional: bundled data
│   └── classification_Xtrain/
│       └── config.yaml
└── library/                             # Reusable artifacts catalog
    ├── models/
    │   ├── pls_17comp_pb8yky.pkl        # Named with ID
    │   └── pls_15comp_5grd70.pkl
    ├── scalers/
    │   └── minmax_regression_pb8yky.pkl
    └── catalog.yaml                     # Index of all artifacts
```

### Naming Convention
- **Months**: `YYYY-MM/` (e.g., `2024-10/`)
- **Experiments**: User-defined or `quick_tests` default
- **Runs**: `DD_<dataset>_<model>_<shortid>` (e.g., `14_regression_PLS17_pb8yky`)
- **Artifacts**: `<type>_<context>_<id>.pkl` (e.g., `pls_17comp_pb8yky.pkl`)

### Rationale
**Pros:**
- ✅ Natural organization by date (recent work easy to find)
- ✅ Optional experiment grouping (flexibility)
- ✅ Artifact library with semantic names + IDs
- ✅ Shorter folder names (day only, not full timestamp)
- ✅ Can browse by month when cleaning old runs
- ✅ Single metadata file per run
- ✅ Clear separation of outputs (charts) and artifacts (models)
- ✅ Catalog system makes artifacts discoverable

**Cons:**
- ⚠️ Two-level hierarchy (month + experiment)
- ⚠️ Need to specify experiment name (or use default)
- ⚠️ Artifact library needs maintenance
- ⚠️ More complex than flat structure

### Implementation Notes
- Experiment name: `runner.run(..., experiment="wheat_study")` or auto-default
- Month folder auto-created from current date
- Short ID: Use existing console output ID (`[pb8yky]`)
- Catalog updated automatically when artifacts saved
- `library/` allows browsing all saved models across experiments

---

## Summary Comparison Table

| Feature | Proposal 1 (Project-Centric) | Proposal 2 (Flat Timeline) | Proposal 3 (Hybrid Date-Experiment) |
|---------|------------------------------|----------------------------|-------------------------------------|
| **Readability** | ⭐⭐⭐⭐ Project names | ⭐⭐⭐⭐⭐ Full context in name | ⭐⭐⭐⭐ Month + experiment |
| **Simplicity** | ⭐⭐⭐ Two levels | ⭐⭐⭐⭐⭐ One level | ⭐⭐⭐ Two-three levels |
| **Latest First** | ⭐⭐ Need to check project | ⭐⭐⭐⭐⭐ Chronological names | ⭐⭐⭐⭐ Month folder sorting |
| **Deduplication** | ⭐⭐⭐⭐⭐ Hidden shared artifacts | ⭐⭐ No dedup | ⭐⭐⭐⭐ Artifact library |
| **Discoverability** | ⭐⭐⭐⭐ By project | ⭐⭐⭐⭐ By folder name | ⭐⭐⭐⭐⭐ By month + catalog |
| **Cleaning Old Runs** | ⭐⭐⭐⭐⭐ Auto-archive | ⭐⭐ Manual | ⭐⭐⭐⭐ By month |
| **File Count** | ⭐⭐⭐ Multiple per run | ⭐⭐⭐⭐ All in one folder | ⭐⭐⭐ Balanced |
| **ID Length** | ⭐⭐⭐⭐⭐ 6 chars | ⭐⭐⭐⭐⭐ 6 chars | ⭐⭐⭐⭐⭐ 6 chars |
| **Flexibility** | ⭐⭐⭐ Need project | ⭐⭐⭐⭐ No structure needed | ⭐⭐⭐⭐⭐ Optional experiment |

---

## Recommended Choice

### 🏆 **Proposal 3 (Hybrid Date-Experiment)** is recommended because:

1. **Best for users at all skill levels:**
   - Beginners: Default `quick_tests/` experiment, easy chronological browsing
   - Advanced: Custom experiment names for organization
   - Administrators: Month folders make cleanup natural

2. **Balances technical needs:**
   - Artifact library maintains deduplication benefits
   - Short IDs (6 chars) are user-friendly and console-visible
   - Single `summary.yaml` per run reduces clutter

3. **Future-proof:**
   - Can add more experiments without restructuring
   - Month archiving prevents results/ from growing unbounded
   - Catalog system enables advanced queries later

4. **Migration path:**
   - Current UUID-based structure can be converted
   - Generate short IDs from existing UUIDs (first 6 chars or hash)
   - Build catalog from existing artifacts directory

### Next Steps:
1. Confirm chosen proposal
2. Design migration strategy from current structure
3. Implement short ID generation (use existing console IDs like `[pb8yky]`)
4. Create `summary.yaml` schema (merge pipeline.json + metadata.json)
5. Add experiment name parameter to runner API
