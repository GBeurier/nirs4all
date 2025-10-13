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

## Proposal 3: Multi-Level Hierarchy with Smart Artifact Strategy

### Philosophy
**"Runs contain pipelines, pipelines contain artifacts"** - Clear 4-level hierarchy: Run → Dataset → Pipeline → Models. Balance between content-addressed efficiency and user-friendly browsing with a **hybrid artifact approach**.

### Key Insight: The Artifact Dilemma - SOLVED

**The Problem:**
- ❌ Pure content-addressed storage: Efficient but unreadable hashes
- ❌ Pure duplication: User-friendly but wastes disk space
- ❌ Symlinks: Complex, breaks when moving folders

**The Solution: HYBRID APPROACH**
1. **Primary artifacts** live IN the pipeline folder (user-browsable)
2. **Catalog links** them to content-addressed cache (optional deduplication)
3. **On export/share**: Folder is self-contained (includes all artifacts)
4. **On cleanup**: Can remove duplicates using catalog

**Mental Model:** Like Git - you see files in your working directory, but Git tracks them efficiently behind the scenes.

---

### Complete Directory Structure

```
results/
│
├── 2024-10-14_wheat_quality/              # 📅 DATE + RUN_NAME (user-provided or auto)
│   │                                       # Date: YYYY-MM-DD (sortable, no hours)
│   │                                       # Run Name: User-defined context
│   ├── RUN.yaml                           # ✨ Run-level metadata
│   │   ├── run_id: "wheat_quality"        #    User-friendly name
│   │   ├── created: "2024-10-14T00:22:00"
│   │   ├── description: "Testing PLS variants on wheat protein"
│   │   ├── datasets: ["regression"]
│   │   ├── pipelines: ["Q1_c20f9b", "Q1_eb2552", "Q1_3e8bec"]
│   │   ├── best_pipeline: "Q1_c20f9b"     #    Pointer to winner
│   │   └── notes: "Comparing 1-29 components"
│   │
│   ├── regression/                        # 📊 DATASET folder (1 per dataset)
│   │   │
│   │   ├── DATASET.yaml                   # Dataset metadata
│   │   │   ├── name: "regression"
│   │   │   ├── samples: {train: 130, test: 59}
│   │   │   ├── features: 6453
│   │   │   ├── target_stats: {mean: 30.33, std: 23.57}
│   │   │   └── source_files: ["train_x.csv", "test_x.csv"]
│   │   │
│   │   ├── Q1_c20f9b/                     # 🔧 PIPELINE folder (pipeline_name or pipeline_id)
│   │   │   │                              # This is the BEST pipeline (RMSE: 6.8466)
│   │   │   ├── PIPELINE.yaml              # ✨ Complete pipeline definition
│   │   │   │   ├── pipeline_id: "Q1_c20f9b"
│   │   │   │   ├── name: "PLS-15_components"
│   │   │   │   ├── steps:
│   │   │   │   │   - {step: 1, op: "MinMax", params: {...}}
│   │   │   │   │   - {step: 2, op: "FirstDerivative"}
│   │   │   │   │   - {step: 3, op: "Haar"}
│   │   │   │   │   - {step: 4, op: "MinMax"}
│   │   │   │   │   - {step: 5, op: "PLS", params: {n_components: 15}}
│   │   │   │   ├── created: "2024-10-14T00:22:31"
│   │   │   │   └── ... (full config - PORTABLE)
│   │   │   │
│   │   │   ├── SCORES.yaml                # 📊 All metrics in ONE file
│   │   │   │   ├── summary:
│   │   │   │   │   ├── best_model_id: "5grd70"  # Short ID
│   │   │   │   │   ├── test: {rmse: 13.3376, r2: 0.546, ...}
│   │   │   │   │   ├── val: {rmse: 6.8466, r2: 0.878, ...}
│   │   │   │   │   └── train: {rmse: 4.904, r2: 0.957, ...}
│   │   │   │   ├── fold_results:
│   │   │   │   │   - {fold: 0, id: "o2tlmc", rmse: 12.4681, ...}
│   │   │   │   │   - {fold: 1, id: "e1k1rl", rmse: 11.1498, ...}
│   │   │   │   │   - {fold: 2, id: "a3817y", rmse: 34.9569, ...}
│   │   │   │   │   - {fold: "avg", id: "poc6ku", ...}
│   │   │   │   │   - {fold: "w_avg", id: "5grd70", ...} ⭐ BEST
│   │   │   │   └── comparison_table: "..." # ASCII table
│   │   │   │
│   │   │   ├── predictions/               # 📈 All prediction files
│   │   │   │   ├── best_5grd70.csv        # Best model (weighted avg)
│   │   │   │   │   # y_true, y_pred, residual, fold
│   │   │   │   ├── fold_0_o2tlmc.csv      # Individual folds
│   │   │   │   ├── fold_1_e1k1rl.csv
│   │   │   │   ├── fold_2_a3817y.csv
│   │   │   │   ├── avg_poc6ku.csv
│   │   │   │   └── train_predictions.csv
│   │   │   │
│   │   │   ├── charts/                    # 📊 All visualizations
│   │   │   │   ├── 2D_Chart.png
│   │   │   │   ├── Y_distribution_train_test.png
│   │   │   │   ├── fold_visualization_3folds_train.png
│   │   │   │   ├── fold_visualization_3folds_test.png
│   │   │   │   ├── predictions_scatter_5grd70.png  # Best model viz
│   │   │   │   └── residuals_5grd70.png
│   │   │   │
│   │   │   ├── models/                    # 🎯 ARTIFACTS (physical files here!)
│   │   │   │   │                          # ✨ This is the KEY solution
│   │   │   │   ├── 5grd70/                # Best model subfolder
│   │   │   │   │   ├── pls_model.pkl      # 2.4 MB
│   │   │   │   │   ├── scaler_minmax_1.pkl  # 12 KB
│   │   │   │   │   ├── scaler_minmax_2.pkl  # 12 KB
│   │   │   │   │   ├── transform_1stder.pkl # 8 KB
│   │   │   │   │   ├── transform_haar.pkl   # 24 KB
│   │   │   │   │   └── MANIFEST.yaml        # Artifact inventory
│   │   │   │   │       ├── artifacts:
│   │   │   │   │       │   - {name: "pls_model.pkl", type: "model",
│   │   │   │   │       │      step: 5, sha256: "abc...", size: 2.4MB}
│   │   │   │   │       │   - {name: "scaler_minmax_1.pkl", ...}
│   │   │   │   │       └── total_size: "2.47 MB"
│   │   │   │   │
│   │   │   │   ├── o2tlmc/                # Fold 0 artifacts
│   │   │   │   │   └── ... (similar structure)
│   │   │   │   ├── e1k1rl/                # Fold 1 artifacts
│   │   │   │   │   └── ...
│   │   │   │   └── shared/                # Shared artifacts (transformers)
│   │   │   │       └── ... (if any preprocessing shared)
│   │   │   │
│   │   │   └── logs/
│   │   │       ├── console.log
│   │   │       ├── warnings.log
│   │   │       └── timing.yaml
│   │   │
│   │   ├── Q1_eb2552/                     # Another pipeline (RMSE: 8.4445)
│   │   │   ├── PIPELINE.yaml              # Different config (21 components)
│   │   │   ├── SCORES.yaml
│   │   │   ├── predictions/
│   │   │   │   └── best_7qc4iw.csv
│   │   │   ├── charts/
│   │   │   │   └── ...
│   │   │   └── models/
│   │   │       └── 7qc4iw/                # Different model artifacts
│   │   │           └── ...
│   │   │
│   │   ├── Q1_3e8bec/                     # Third pipeline
│   │   │   └── ...
│   │   │
│   │   └── COMPARISON.yaml                # Compare all pipelines on this dataset
│   │       ├── best_pipeline: "Q1_c20f9b"
│   │       ├── rankings:
│   │       │   - {rank: 1, pipeline: "Q1_c20f9b", model_id: "5grd70", rmse: 6.8466}
│   │       │   - {rank: 2, pipeline: "Q1_3e8bec", model_id: "3tq71m", rmse: 8.2408}
│   │       │   - {rank: 3, pipeline: "Q1_eb2552", model_id: "7qc4iw", rmse: 8.4445}
│   │       └── comparison_chart: "all_pipelines_comparison.png"
│   │
│   └── classification_Xtrain/             # Another dataset in same run
│       ├── DATASET.yaml
│       ├── Q1_classification_c3abeb/      # Pipeline for classification
│       │   ├── PIPELINE.yaml
│       │   ├── SCORES.yaml                # Different metrics (accuracy, f1)
│       │   ├── predictions/
│       │   │   └── best_2e3nwn.csv        # class, probabilities
│       │   ├── charts/
│       │   │   ├── confusion_matrix.png
│       │   │   └── roc_curve.png
│       │   └── models/
│       │       └── 2e3nwn/
│       │           └── random_forest.pkl
│       └── COMPARISON.yaml
│
├── 2024-10-14_multimodel/                 # Another RUN (different experiment)
│   ├── RUN.yaml
│   └── regression/
│       ├── Q2_b9f52b/                     # Multi-model pipeline
│       │   ├── PIPELINE.yaml
│       │   ├── SCORES.yaml
│       │   ├── predictions/
│       │   ├── charts/
│       │   └── models/
│       │       ├── gahxeq/                # GradientBoosting (winner)
│       │       ├── tx5inn/                # PLS
│       │       ├── ny2373/                # SVR
│       │       └── ...
│       └── COMPARISON.yaml
│
├── 2024-10-14_multidatasets/              # Multi-dataset run
│   ├── RUN.yaml
│   ├── regression/
│   │   └── Q3_0f8b1a/
│   │       └── ...
│   ├── regression_2/
│   │   └── Q3_0f8b1a/                     # Same pipeline, different dataset
│   │       └── ...
│   └── regression_3/
│       └── Q3_0f8b1a/
│           └── ...
│
├── 2024-10-14_finetune/                   # Run name describes purpose
│   └── ...
│
├── datasets/                              # 📦 Dataset library (reusable)
│   ├── regression/
│   │   ├── config.yaml
│   │   └── metadata.yaml
│   ├── regression_2/
│   │   └── ...
│   └── classification_Xtrain/
│       └── ...
│
└── .cache/                                # 🔒 HIDDEN - Content-addressed cache
    ├── objects/                           # (Optional deduplication)
    │   └── ab/
    │       └── abc123def456.pkl
    └── catalog.db                         # SQLite database
        # Maps content hash → file locations
        # Enables "find duplicates" command
        # sha256 → [list of paths where this artifact lives]
```

---

### Real Example - Complete Run Walkthrough

**Example Run:** Testing 3 PLS configurations on wheat protein dataset

**Console command:**
```python
runner.run(
    pipeline_configs=["Q1_c20f9b", "Q1_eb2552", "Q1_3e8bec"],
    dataset="regression",
    run_name="wheat_quality"  # User-provided name
)
```

**Result:** `results/2024-10-14_wheat_quality/`

```
2024-10-14_wheat_quality/
│
├── RUN.yaml                               # 1.2 KB
│   run_id: "wheat_quality"
│   created: "2024-10-14T00:22:00"
│   description: "Auto-generated: 3 pipelines on 1 dataset"
│   datasets: ["regression"]
│   pipelines: ["Q1_c20f9b", "Q1_eb2552", "Q1_3e8bec"]
│   best_overall:
│     dataset: "regression"
│     pipeline: "Q1_c20f9b"
│     model_id: "5grd70"
│     rmse: 6.8466
│   total_size: "47.3 MB"
│
└── regression/
    │
    ├── DATASET.yaml                       # 2.4 KB
    │   name: "regression"
    │   samples: {train: 130, test: 59, total: 189}
    │   features: 6453
    │   target: "protein"
    │   target_stats: {mean: 30.33, min: 2.05, max: 128.31}
    │
    ├── Q1_c20f9b/                         # ⭐ WINNER: RMSE 6.8466
    │   │
    │   ├── PIPELINE.yaml                  # 3.1 KB - PORTABLE config
    │   │   pipeline_id: "Q1_c20f9b"
    │   │   name: "PLS-15_components"
    │   │   steps:
    │   │     1. MinMax scaler
    │   │     2. FirstDerivative (window=7)
    │   │     3. Haar wavelet
    │   │     4. MinMax scaler (again)
    │   │     5. PLS (n_components=15)
    │   │   # Can copy this file to reuse pipeline!
    │   │
    │   ├── SCORES.yaml                    # 8.7 KB - All metrics
    │   │   best_model_id: "5grd70"
    │   │   test: {rmse: 13.3376, r2: 0.546, mae: 9.763}
    │   │   val: {rmse: 6.8466, r2: 0.878, mae: 5.389}
    │   │   train: {rmse: 4.904, r2: 0.957, mae: 3.129}
    │   │   folds: [15 component variations tested]
    │   │
    │   ├── predictions/
    │   │   ├── best_5grd70.csv            # 3.8 KB - THE prediction file
    │   │   ├── fold_0_o2tlmc.csv
    │   │   ├── fold_1_e1k1rl.csv
    │   │   ├── fold_2_a3817y.csv
    │   │   └── ... (13 total prediction files)
    │   │
    │   ├── charts/
    │   │   ├── 2D_Chart.png               # 156 KB
    │   │   ├── Y_distribution_train_test.png  # 89 KB
    │   │   ├── fold_visualization_3folds_train.png  # 124 KB
    │   │   ├── predictions_scatter_5grd70.png  # 142 KB
    │   │   └── residuals_5grd70.png       # 98 KB
    │   │
    │   ├── models/                        # 🎯 Physical artifacts here!
    │   │   │
    │   │   ├── 5grd70/                    # ⭐ BEST model (w_avg fold)
    │   │   │   ├── pls_model.pkl          # 2.4 MB - Main model
    │   │   │   ├── scaler_minmax_1.pkl    # 12 KB - Step 1
    │   │   │   ├── scaler_minmax_2.pkl    # 12 KB - Step 4
    │   │   │   ├── transform_1stder.pkl   # 8 KB - Step 2
    │   │   │   ├── transform_haar.pkl     # 24 KB - Step 3
    │   │   │   └── MANIFEST.yaml          # 1.1 KB
    │   │   │       artifacts:
    │   │   │         - name: "pls_model.pkl"
    │   │   │           type: "model"
    │   │   │           step: 5
    │   │   │           operator: "PLSRegression"
    │   │   │           sha256: "abc123..."
    │   │   │           size: 2457600
    │   │   │           cached: ".cache/objects/ab/abc123.pkl"
    │   │   │         - name: "scaler_minmax_1.pkl"
    │   │   │           ...
    │   │   │       total_size: 2.47 MB
    │   │   │
    │   │   ├── o2tlmc/                    # Fold 0 artifacts
    │   │   ├── e1k1rl/                    # Fold 1 artifacts
    │   │   ├── a3817y/                    # Fold 2 artifacts
    │   │   └── poc6ku/                    # Avg fold artifacts
    │   │
    │   └── logs/
    │       ├── console.log                # Full output
    │       └── timing.yaml                # Step timings
    │
    ├── Q1_eb2552/                         # Runner-up: RMSE 8.4445
    │   ├── PIPELINE.yaml                  # Different config (21 components)
    │   ├── SCORES.yaml
    │   ├── predictions/
    │   │   └── best_7qc4iw.csv
    │   ├── charts/
    │   └── models/
    │       └── 7qc4iw/                    # Different artifacts
    │           └── ...
    │
    ├── Q1_3e8bec/                         # Third place: RMSE 8.2408
    │   └── ...
    │
    └── COMPARISON.yaml                    # 4.2 KB
        best_pipeline: "Q1_c20f9b"
        rankings:
          - rank: 1
            pipeline: "Q1_c20f9b"
            pipeline_name: "PLS-15_components"
            model_id: "5grd70"
            val_rmse: 6.8466
            test_rmse: 13.3376
          - rank: 2
            pipeline: "Q1_3e8bec"
            model_id: "3tq71m"
            val_rmse: 8.2408
          - rank: 3
            pipeline: "Q1_eb2552"
            model_id: "7qc4iw"
            val_rmse: 8.4445
        comparison_chart: "pipelines_comparison.png"
```

**Total size:** ~47 MB (includes all 3 pipelines with all folds)

---

### Artifact Storage Strategies - Deep Dive 🔧

Now that we have the export method, let's compare THREE strategies for artifact storage:

---

#### Strategy A: Local Per-Run Storage (Simplest)

**Structure:**
```
2024-10-14_wheat_quality/
└── regression/
    └── Q1_c20f9b/
        └── models/
            └── 5grd70/
                ├── pls_model.pkl        # ✅ Physical file here
                ├── scaler_minmax_1.pkl  # ✅ Physical file here
                └── ...                  # All artifacts physically present
```

**How it works:**
- Artifacts saved directly to `models/5grd70/`
- NO cache, NO symlinks, NO references
- Each run has its own complete copy

**Pros:**
- ✅ **Simplest possible** - what you see is what you get
- ✅ **Zero dependencies** - delete folder = delete everything
- ✅ **Fast sharing** - just zip the folder
- ✅ **No broken links** - always works
- ✅ **Easy cleanup** - `rm -rf 2024-10-14_*` and done
- ✅ **Browsable** - can open any .pkl file directly

**Cons:**
- ❌ **Disk space** - Models duplicated if you run similar experiments
- ❌ **No deduplication** - 100 runs with same preprocessing = 100 copies

**Best for:**
- Users who run 10-50 experiments per project
- Sharing pipelines frequently
- Simple mental model
- Storage is not a constraint

**Disk usage example:**
- 10 runs, each 15 MB = 150 MB (totally reasonable)
- Even 100 runs = 1.5 GB (acceptable on modern systems)

---

#### Strategy B: Shared Cache with Smart References (Efficient)

**Structure:**
```
2024-10-14_wheat_quality/
└── regression/
    └── Q1_c20f9b/
        └── models/
            └── 5grd70/
                ├── pls_model.pkl → .cache/objects/ab/abc123.pkl  # Symlink
                ├── scaler_minmax_1.pkl → .cache/objects/cd/cde456.pkl
                └── MANIFEST.yaml          # Metadata + references

.cache/
├── objects/
│   ├── ab/
│   │   └── abc123def456.pkl  # ✅ Physical file (2.4 MB)
│   └── cd/
│       └── cde456ghi789.pkl  # ✅ Physical file (12 KB)
└── catalog.db
    # Maps: sha256 → [list of symlink paths]
```

**How it works:**
- Artifacts saved to content-addressed cache
- Symlinks created in `models/5grd70/` pointing to cache
- Catalog tracks which pipelines use which artifacts

**Pros:**
- ✅ **Maximum efficiency** - Identical artifacts stored once
- ✅ **Space savings** - Can save 30-80% with similar experiments
- ✅ **Fast saving** - Check hash, reuse if exists
- ✅ **Orphan detection** - Know which cache files are unused

**Cons:**
- ❌ **Symlinks break when moving** - Zip breaks, copy breaks
- ❌ **Requires export command** - Can't just zip folder
- ❌ **Complex cleanup** - Must update catalog and check references
- ❌ **Browsing confusion** - Symlinks look like files but aren't
- ❌ **Cache corruption** - If cache deleted, all symlinks break

**Best for:**
- Power users running 1000s of experiments
- Disk space is critical constraint
- Rarely share pipelines externally
- Computational clusters with limited storage

**Disk usage example:**
- 100 runs with 50% shared artifacts = 750 MB (50% savings)

---

#### Strategy C: Hybrid - Local Storage + Optional Cache (RECOMMENDED)

**Structure:**
```
2024-10-14_wheat_quality/
└── regression/
    └── Q1_c20f9b/
        └── models/
            └── 5grd70/
                ├── pls_model.pkl        # ✅ Physical file (2.4 MB)
                ├── scaler_minmax_1.pkl  # ✅ Physical file (12 KB)
                └── MANIFEST.yaml
                    # Contains hash: sha256_abc123...
                    # Links to cache: .cache/objects/ab/abc123.pkl

.cache/
├── objects/
│   ├── ab/
│   │   └── abc123def456.pkl  # ✅ ALSO stored here (redundant)
│   └── cd/
│       └── cde456ghi789.pkl
└── catalog.db
    # Maps: sha256 → [list of run/pipeline/model paths]
    # Used for: deduplication analysis, not storage
```

**How it works:**
1. **On save:** Artifacts written to BOTH locations:
   - Primary: `models/5grd70/pls_model.pkl` (browsable)
   - Cache: `.cache/objects/ab/abc123.pkl` (tracking)

2. **On export:** Uses primary location (always works)

3. **On deduplicate (optional command):**
   ```bash
   nirs4all deduplicate --dry-run
   # Output:
   # Found 15 duplicate artifacts (36 MB potential savings)
   #
   # Duplicate Set 1: scaler_minmax_1.pkl (12 KB)
   #   - 2024-10-14_wheat/regression/Q1_c20f9b/models/5grd70/
   #   - 2024-10-14_wheat/regression/Q1_eb2552/models/7qc4iw/
   #   - 2024-10-15_wheat_v2/regression/Q2_xyz/models/abc123/
   #
   # Action: Keep first, replace others with symlinks? [y/N]

   nirs4all deduplicate --auto --keep-newest
   # Keeps newest, replaces older with symlinks
   # Savings: 36 MB
   ```

4. **On cleanup:**
   ```bash
   rm -rf 2024-10-14_wheat_quality/
   # Just works! No orphaned cache files to worry about

   nirs4all cache clean --orphaned
   # Optional: Remove cache entries no longer referenced
   ```

**Pros:**
- ✅ **Best of both worlds** - Simple AND efficient
- ✅ **Always browsable** - Physical files in models/
- ✅ **Easy sharing** - Export uses primary location
- ✅ **Optional optimization** - Deduplicate when you want
- ✅ **Safe cleanup** - Delete run = delete artifacts
- ✅ **Space analysis** - Cache shows potential savings
- ✅ **No forced deduplication** - User decides

**Cons:**
- ⚠️ **Initial redundancy** - Artifacts stored twice initially
- ⚠️ **Cache maintenance** - Need occasional `cache clean`
- ⚠️ **Slightly more complex** - Two storage locations

**Best for:**
- Most users (balance simplicity and efficiency)
- Projects with 50-500 experiments
- Occasional sharing but space-conscious
- Want flexibility to optimize later

**Disk usage example:**
- 100 runs, 15 MB each = 1.5 GB initially
- After `deduplicate`: 750 MB (if 50% duplicates)
- User chooses when to optimize

---

### Strategy Comparison Table

| Feature | A: Local Only | B: Shared Cache | C: Hybrid (Recommended) |
|---------|--------------|-----------------|------------------------|
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Disk Efficiency** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Easy Sharing** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Easy Cleanup** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Browsability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **No Broken Links** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Space Analysis** | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **User Control** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

### Recommendation: Hybrid Strategy (C) with Export

**Implementation:**

```python
# In SimulationSaver (io.py)

def persist_artifact(self, step_number, name, obj, format_hint=None):
    """Save artifact with hybrid storage strategy."""

    # 1. Serialize the artifact
    artifact_binary = serialize(obj, format_hint)

    # 2. Compute hash
    sha256 = hashlib.sha256(artifact_binary).hexdigest()

    # 3. Save to PRIMARY location (models/)
    primary_path = self.models_path / f"{self.model_id}" / f"{name}.pkl"
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_bytes(artifact_binary)

    # 4. Save to CACHE (optional, async)
    if self.enable_cache:  # Default: True
        cache_path = self.cache_path / sha256[:2] / f"{sha256}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            cache_path.write_bytes(artifact_binary)

        # Update catalog
        self.catalog.register(
            sha256=sha256,
            primary_path=primary_path,
            cache_path=cache_path,
            metadata={
                "name": name,
                "type": format_hint,
                "step": step_number,
                "size": len(artifact_binary)
            }
        )

    # 5. Update MANIFEST.yaml in models folder
    manifest = {
        "name": name,
        "type": format_hint,
        "step": step_number,
        "sha256": sha256,
        "size": len(artifact_binary),
        "cached": f".cache/objects/{sha256[:2]}/{sha256}.pkl"
    }

    return primary_path, manifest
```

**Export command:**
```python
def export_pipeline(self, run, dataset, pipeline, model_id="best",
                   output=None, include_predictions=True, include_charts=True):
    """Export a pipeline as standalone package."""

    # 1. Find the pipeline folder
    pipeline_path = self.results_path / run / dataset / pipeline

    # 2. Select model (best or specific)
    if model_id == "best":
        scores = yaml.load(pipeline_path / "SCORES.yaml")
        model_id = scores["summary"]["best_model_id"]

    # 3. Create temp export folder
    export_folder = tempfile.mkdtemp()

    # 4. Copy all files (artifacts are ALREADY physical files!)
    shutil.copytree(pipeline_path, export_folder / pipeline)

    # 5. Add EXPORT.yaml metadata
    export_meta = {
        "exported_from": run,
        "dataset": dataset,
        "pipeline": pipeline,
        "model_id": model_id,
        "export_date": datetime.now().isoformat(),
        "nirs4all_version": __version__
    }

    # 6. Create zip
    output_path = output or f"{pipeline}_{model_id}_export.zip"
    shutil.make_archive(output_path.replace('.zip', ''), 'zip', export_folder)

    # 7. Cleanup
    shutil.rmtree(export_folder)

    return output_path
```

**Deduplication command:**
```python
def deduplicate(self, dry_run=True, keep_strategy="first"):
    """Find and optionally remove duplicate artifacts."""

    # 1. Query catalog for duplicates
    duplicates = self.catalog.find_duplicates()

    # 2. Group by hash
    for sha256, paths in duplicates.items():
        if len(paths) <= 1:
            continue  # Not a duplicate

        print(f"\nDuplicate artifact (sha256: {sha256[:8]}...)")
        print(f"  Size: {self.catalog.get_size(sha256)} bytes")
        print(f"  Found in {len(paths)} locations:")
        for path in paths:
            print(f"    - {path}")

        if not dry_run:
            # Keep one, replace others with symlinks
            keep_path = self._select_keep_path(paths, keep_strategy)
            for path in paths:
                if path != keep_path:
                    path.unlink()
                    path.symlink_to(keep_path)
                    print(f"  ✓ Replaced with symlink: {path}")
```

**Why Hybrid is Best:**

1. **Scientists benefit:**
   - Browse models like normal files
   - Share by zipping folders
   - Delete runs without fear

2. **Admins benefit:**
   - See potential space savings
   - Optimize when needed
   - Track artifact reuse

3. **Workflow:**
   ```bash
   # Normal work - no thinking about cache
   nirs4all run pipeline.yaml

   # Share specific pipeline
   nirs4all export Q1_c20f9b --model best

   # Check space usage (optional)
   nirs4all analyze storage
   # Output: 1.5 GB used, 450 MB duplicates found

   # Optimize if wanted (optional)
   nirs4all deduplicate --auto
   # Saved: 450 MB

   # Clean old work
   rm -rf 2024-10-*  # Just works!
   nirs4all cache clean --orphaned  # Optional cleanup
   ```

This gives you:
- ✅ Artifact efficiency (30-50% savings with deduplicate)
- ✅ Good UX (browsable files, easy cleanup)
- ✅ Safe sharing (export command resolves everything)
- ✅ User control (deduplicate when you want, not forced)

---

### The 4-Level ID Hierarchy

Every file and folder encodes its place in the hierarchy:

```
📍 LEVEL 1: RUN
   ├── ID: User-provided name (e.g., "wheat_quality")
   ├── Prefix: Date (YYYY-MM-DD)
   └── Full: "2024-10-14_wheat_quality"

📍 LEVEL 2: DATASET
   ├── ID: Dataset name (e.g., "regression")
   └── Folder: "regression/"

📍 LEVEL 3: PIPELINE
   ├── ID: Pipeline hash (e.g., "Q1_c20f9b")
   ├── Name: Pipeline name (e.g., "PLS-15_components")
   └── Folder: "Q1_c20f9b/"

📍 LEVEL 4: MODEL
   ├── ID: Model short ID (e.g., "5grd70")
   ├── Context: Fold identifier (fold_0, avg, w_avg)
   └── Folder: "models/5grd70/"
```

**In File Names:**
- Predictions: `best_5grd70.csv` → Pipeline implicit, model explicit
- Charts: `predictions_scatter_5grd70.png` → Model explicit
- Artifacts: `models/5grd70/pls_model.pkl` → Model in folder path

**Navigation:**
1. Browse by date → Find recent runs
2. Open run folder → See all datasets
3. Open dataset → See all pipelines compared
4. Open pipeline → See best model results
5. Open models/best_id/ → Get artifacts to reuse

---

### Sharing & Portability with Export Method 🎁

**Problem:** Friend wants your winning wheat protein model

#### Built-in Export Command

```python
from nirs4all import PipelineRunner

runner = PipelineRunner()

# Export a specific pipeline (recommended!)
runner.export_pipeline(
    run="2024-10-14_wheat_quality",
    dataset="regression",
    pipeline="Q1_c20f9b",
    model_id="5grd70",  # Optional: specific model, or "best" for auto-select
    output="wheat_pls15_model.zip",
    include_predictions=True,
    include_charts=True
)
```

**What this does:**
1. ✅ Copies all artifacts from shared cache to temporary folder
2. ✅ Bundles: pipeline, scores, predictions, charts, models
3. ✅ Creates standalone package (no dependencies on your results/)
4. ✅ Generates `EXPORT.yaml` with provenance info
5. ✅ Creates zip: `wheat_pls15_model.zip` (18.2 MB)

**Export package structure:**
```
wheat_pls15_model.zip
├── EXPORT.yaml                    # Provenance metadata
│   ├── exported_from: "2024-10-14_wheat_quality"
│   ├── original_path: "results/2024-10-14_wheat_quality/regression/Q1_c20f9b"
│   ├── exported_by: "username"
│   ├── export_date: "2024-10-14T15:30:00"
│   └── nirs4all_version: "6.0.0"
├── PIPELINE.yaml                  # Reusable config
├── SCORES.yaml                    # Performance proof
├── predictions/
│   └── best_5grd70.csv
├── charts/
│   ├── 2D_Chart.png
│   └── ...
└── models/
    └── 5grd70/
        ├── pls_model.pkl          # ✅ Physical copies (not symlinks!)
        ├── scaler_minmax_1.pkl
        └── ... (all artifacts)
```

**Friend receives:** Self-contained zip with ZERO dependencies

**Friend uses it:**
```python
# Option 1: Import the exported pipeline
runner.import_pipeline("wheat_pls15_model.zip", to_project="my_wheat_study")
# Creates: results/my_project/imported/wheat_pls15_model/

# Option 2: Extract and use directly
unzip wheat_pls15_model.zip
runner.predict_from_artifacts(
    model_dir="models/5grd70/",
    new_data="my_samples.csv"
)
```

---

### Manual Sharing Options (Alternative)

**Solution 1: Share the pipeline folder directly**
```bash
# Zip the pipeline folder - includes EVERYTHING if using local storage
cd results/2024-10-14_wheat_quality/regression/
zip -r wheat_pls15_pipeline.zip Q1_c20f9b/

# If using shared artifacts, must resolve references first:
nirs4all export Q1_c20f9b --resolve-artifacts
# This copies artifacts from cache to pipeline folder before zipping
```

**Solution 2: Share just the best model**
```bash
# Even smaller - just the winner
nirs4all export Q1_c20f9b --model 5grd70 --minimal
# Output: Q1_c20f9b_5grd70_minimal.zip (2.5 MB)
```

**Solution 3: Share entire run** (for reproducibility)
```bash
nirs4all export 2024-10-14_wheat_quality --complete
# Output: 2024-10-14_wheat_quality_complete.zip (47 MB)
# Everything: All pipelines, all folds, all results
```

---

### The Hybrid Artifact Strategy Explained

**Why artifacts live IN the pipeline folder:**

1. **✅ User-Browsable:** Users can navigate and find their models
2. **✅ Self-Contained:** Each pipeline folder is independently shareable
3. **✅ No Broken Links:** Moving/copying folders just works
4. **✅ Cleanup-Friendly:** Delete a pipeline folder = delete all its artifacts
5. **✅ Semantic Names:** `pls_model.pkl` not `abc123.pkl`

**Why we ALSO have `.cache/`:**

1. **Deduplication Detection:** Catalog tracks which artifacts are identical
2. **Optional Cleanup:** Run `nirs4all deduplicate` to remove duplicates
3. **Space Analysis:** See which artifacts are reused across runs
4. **Not Required:** Cache is built automatically, user can ignore it

**How it works:**

```python
# When saving an artifact:
1. Save to: models/5grd70/pls_model.pkl (primary location)
2. Compute: sha256 hash of the file
3. Update: .cache/catalog.db
   INSERT (sha256, path: "2024-10-14_wheat/.../5grd70/pls_model.pkl")
4. Check: Is this hash already in catalog?
   - If NO: This is a new unique model
   - If YES: Log potential duplicate (but don't auto-delete!)

# When user runs: nirs4all deduplicate (optional)
1. Query catalog for duplicate hashes
2. Show user: "These models are identical:"
   - 2024-10-14_wheat/.../5grd70/pls_model.pkl
   - 2024-10-15_wheat/.../8xkf12/pls_model.pkl
3. Ask: "Keep which one? (1) First (2) Second (3) Both"
4. If user chooses (1):
   - Keep first, replace second with symlink
   - Save: ~2.4 MB

# When user runs: nirs4all clean --older-than 30d
1. Delete run folders older than 30 days
2. Update catalog to remove deleted paths
3. Optionally remove orphaned cache files
```

**Key Principle:**
- Primary storage: IN the pipeline folder (visible, browsable)
- Cache: TRACKING system (optional, invisible)
- Deduplication: USER CHOICE (manual command, not automatic)

This way:
- Scientists get simple folder structures
- Developers get efficiency when needed
- Everyone gets portability

---

### Naming Conventions

| Level | Format | Example | Notes |
|-------|--------|---------|-------|
| **Run Folder** | `YYYY-MM-DD_<run_name>` | `2024-10-14_wheat_quality` | Date-prefixed for sorting |
| **Run Name** | User-provided or auto | `wheat_quality` or `run_001` | Descriptive context |
| **Dataset Folder** | `<dataset_name>/` | `regression/` | From dataset config |
| **Pipeline Folder** | `<pipeline_id>/` | `Q1_c20f9b/` | 6-8 char hash |
| **Pipeline ID** | Config hash | `Q1_c20f9b` | Unique to config |
| **Model Folder** | `models/<model_id>/` | `models/5grd70/` | Short ID (6 chars) |
| **Model ID** | Short hash | `5grd70` | From console output `[5grd70]` |
| **Prediction Files** | `<type>_<model_id>.csv` | `best_5grd70.csv` | Model ID embedded |
| **Chart Files** | `<description>_<model_id>.png` | `scatter_5grd70.png` | Model ID if model-specific |
| **Artifact Files** | `<semantic_name>.pkl` | `pls_model.pkl` | Semantic, not hash |

**Philosophy:**
- Dates first → Chronological sorting
- IDs short → Memorable and findable
- Names semantic → Understandable without docs

---

### Rationale & Trade-offs

**Pros:**
- ✅ **Clear 4-level hierarchy:** Run → Dataset → Pipeline → Model
- ✅ **Date-first sorting:** Recent work always at top
- ✅ **Self-contained pipelines:** Share one folder, get everything
- ✅ **Browsable artifacts:** Real files, not symlinks or hashes
- ✅ **Optional deduplication:** Power users can optimize space
- ✅ **Portable configs:** `PIPELINE.yaml` is standalone
- ✅ **Semantic file names:** `pls_model.pkl` not `abc123.pkl`
- ✅ **Comparison built-in:** `COMPARISON.yaml` in each dataset folder
- ✅ **Single metadata files:** `RUN.yaml`, `PIPELINE.yaml`, `SCORES.yaml` (not 5 JSONs)
- ✅ **Short memorable IDs:** `5grd70` visible in console and file system

**Cons:**
- ⚠️ **Disk space:** Artifacts duplicated across similar runs (but realistic)
- ⚠️ **Deeper nesting:** 4 levels (but logical and navigable)
- ⚠️ **Run naming:** User should provide meaningful names (or accept auto-generated)

**Performance Considerations:**

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Save artifacts | Fast | Direct write to models/ folder |
| Load artifacts | Fast | Direct read from models/ folder |
| Find best model | Fast | Read SCORES.yaml (single file) |
| Compare pipelines | Fast | Read COMPARISON.yaml |
| Share pipeline | Fast | Zip one folder |
| Deduplicate | Slow (optional) | Only when user runs command |
| Catalog update | Fast | Async background task |

**Space Analysis (realistic example):**

Single pipeline with 5 folds:
- 5 trained models × 2.4 MB = 12 MB
- Shared transformers = 2 MB (saved once in shared/)
- Predictions = 0.5 MB
- Charts = 1 MB
- **Total: ~15.5 MB per pipeline**

Run with 3 pipelines:
- **Total: ~47 MB** (acceptable for scientific workflows)

With deduplication (if transformers identical):
- Save: ~6 MB (30% savings)

---

### Implementation Notes

**API Changes:**

```python
# User specifies run name (optional)
runner = PipelineRunner()

runner.run(
    pipeline_configs=["config1", "config2"],
    dataset="regression",
    run_name="wheat_quality",  # NEW: User-provided name
    save_files=True
)

# If run_name not provided:
# Auto-generate: "run_001", "run_002", etc.
```

**Folder Creation:**

```python
# Runner creates structure:
1. Generate run folder name: f"{date.today()}_{run_name}"
2. Create: results/{run_folder}/RUN.yaml
3. For each dataset:
   - Create: results/{run_folder}/{dataset}/DATASET.yaml
4. For each pipeline:
   - Create: results/{run_folder}/{dataset}/{pipeline_id}/
   - Save: PIPELINE.yaml, SCORES.yaml
5. For each model:
   - Create: models/{model_id}/
   - Save artifacts with semantic names
   - Generate: MANIFEST.yaml
6. Save outputs: predictions/, charts/
7. Background: Update .cache/catalog.db
```

**Cleanup Commands:**

```bash
# ===== SIMPLE CLEANUPS (work with all strategies) =====

# Remove old runs (filesystem level)
rm -rf 2024-09-*  # Delete all September runs
rm -rf 2024-10-14_wheat_quality/  # Delete specific run

# ===== SMART CLEANUPS (use nirs4all CLI) =====

# 1. Clean by age
nirs4all clean --older-than 30d
# Output:
#   Found 3 runs older than 30 days (127 MB)
#   - 2024-09-10_initial_tests (45 MB)
#   - 2024-09-15_wheat_v1 (52 MB)
#   - 2024-09-20_protein_test (30 MB)
#   Delete these runs? [y/N]

# 2. Clean by size (keep only top performers)
nirs4all clean --keep-best 5 --by rmse
# Output:
#   Found 23 runs, keeping best 5 by RMSE
#   Will delete 18 runs (723 MB)
#   Keep:
#     1. 2024-10-14_wheat_quality (RMSE: 6.84)
#     2. 2024-10-12_wheat_v3 (RMSE: 7.12)
#     ...

# 3. Clean specific dataset runs
nirs4all clean --dataset regression --older-than 15d

# 4. Interactive cleanup
nirs4all clean --interactive
# Shows each run with:
#   - Date, name, size
#   - Best score
#   - Number of pipelines
#   - Asks: Keep? [y/N/skip-all]

# ===== CACHE MANAGEMENT (for Hybrid strategy) =====

# Show cache statistics
nirs4all cache stats
# Output:
#   Cache location: .cache/
#   Total size: 456 MB
#   Unique artifacts: 1,247
#   Duplicate references: 3,891
#   Orphaned entries: 23 (12 MB)
#   Space savings potential: 234 MB

# Clean orphaned cache (no longer referenced)
nirs4all cache clean --orphaned
# Removes cache entries where primary files deleted

# Rebuild cache from scratch
nirs4all cache rebuild
# Scans all runs, rebuilds catalog.db

# ===== DEDUPLICATION (for Hybrid strategy) =====

# Find duplicates (dry run)
nirs4all deduplicate --dry-run
# Shows what WOULD be deduplicated

# Deduplicate automatically
nirs4all deduplicate --auto --keep-newest
# Replaces old duplicates with symlinks

# Undo deduplication (restore physical files)
nirs4all deduplicate --restore
# Copies files back from cache, removes symlinks
```

**Cleanup Workflow Example:**

```bash
# Monthly maintenance routine
# 1. Review what you have
nirs4all analyze storage
# Output:
#   Total runs: 47 (2.3 GB)
#   Runs by month:
#     2024-10: 12 runs (756 MB)
#     2024-09: 23 runs (1.1 GB)  ← old!
#     2024-08: 12 runs (447 MB)  ← very old!
#
#   Duplicates found: 234 MB
#   Oldest run: 2024-08-05 (82 days old)

# 2. Clean old work
nirs4all clean --older-than 60d
# Removes August runs

# 3. Optimize space
nirs4all deduplicate --auto
# Saves 234 MB

# 4. Final check
nirs4all cache clean --orphaned
# Removes unreferenced cache entries

# Result: 2.3 GB → 1.2 GB (saved 47%)
```

---

### Cleanup Strategies Summary

| Action | Command | Safe? | Recoverable? |
|--------|---------|-------|--------------|
| Delete old runs | `rm -rf 2024-09-*` | ✅ Yes | ❌ No |
| Delete old runs (CLI) | `nirs4all clean --older-than 30d` | ✅ Yes | ⚠️ If backup made |
| Deduplicate | `nirs4all deduplicate --auto` | ✅ Yes | ✅ Yes (`--restore`) |
| Clean orphaned cache | `nirs4all cache clean --orphaned` | ✅ Yes | ❌ No (but safe) |
| Keep best only | `nirs4all clean --keep-best 5` | ⚠️ Check first | ❌ No |

**Best Practices:**
1. ✅ Export important pipelines before cleaning
2. ✅ Use `--dry-run` flags to preview
3. ✅ Run `cache stats` to see savings potential
4. ✅ Clean by age, not by score (score may change with new data)
5. ✅ Keep at least 1 month of recent work

---

### Cleanup Commands:**

```bash
# Remove old runs
nirs4all clean --older-than 30d

# Find duplicates (optional)
nirs4all deduplicate --dry-run  # Show duplicates
nirs4all deduplicate --auto     # Replace with symlinks

# Analyze space
nirs4all analyze storage
# Output:
#   Total runs: 15
#   Total size: 723 MB
#   Potential savings: 89 MB (12%)
#   Runs older than 30d: 3 (47 MB)
```

---

### Migration Path

From current structure → Proposal 3:

1. **Phase 1:** Add run_name parameter (optional)
2. **Phase 2:** Generate date-prefixed folders
3. **Phase 3:** Consolidate metadata (merge JSONs → YAMLs)
4. **Phase 4:** Move artifacts from cache to pipeline folders
5. **Phase 5:** Build catalog.db from existing artifacts
6. **Phase 6:** Generate COMPARISON.yaml for existing runs

**Backward Compatibility:**

- Old UUID-based folders still loadable
- Conversion script: `nirs4all migrate --from v5 --to v6`
- Keeps old structure in `results/legacy/`

---

This structure achieves your goals:
✅ Clear hierarchy (Run/Dataset/Pipeline/Model)
✅ User-friendly (browsable, semantic names)
✅ Performance (optional deduplication, not mandatory)
✅ Portability (self-contained folders)
✅ Logic (4-level ID system traceable in paths)

---

---

## 🎯 FINAL RECOMMENDATION: Hybrid Strategy with Export

### Executive Summary

**Chosen Approach:** Proposal 3 with Hybrid Artifact Storage (Strategy C)

**Why this combination wins:**

1. **Four-Level Hierarchy** (Run → Dataset → Pipeline → Model)
   - Clear organization: `2024-10-14_wheat_quality/regression/Q1_c20f9b/models/5grd70/`
   - Every file shows its context
   - Date-first sorting for recency

2. **Hybrid Artifact Storage**
   - Artifacts physically present in pipeline folders (browsable)
   - Optional cache tracks duplicates (efficient)
   - User decides when to deduplicate (control)

3. **Export Command**
   - One command creates standalone package
   - Friend gets self-contained zip
   - No dependency on your cache or structure

4. **Easy Cleanup**
   - Simple: `rm -rf 2024-09-*`
   - Smart: `nirs4all clean --older-than 30d`
   - Safe: Physical files, not symlinks

### What Users Experience

**Day-to-Day Work:**
```python
# Run pipeline - artifacts saved locally
runner.run(pipeline="wheat_pls.yaml", dataset="regression", run_name="wheat_quality")

# Browse results - normal files
ls results/2024-10-14_wheat_quality/regression/Q1_c20f9b/models/5grd70/
# pls_model.pkl  scaler_minmax_1.pkl  ...

# Share with colleague
runner.export_pipeline("2024-10-14_wheat_quality", "regression", "Q1_c20f9b", output="wheat_model.zip")
# wheat_model.zip created (18 MB) - self-contained!
```

**Monthly Maintenance:**
```bash
# Check storage
nirs4all analyze storage
# 2.3 GB used, 234 MB duplicates

# Clean old work
nirs4all clean --older-than 60d
# Deleted 3 runs (500 MB)

# Optimize space (optional)
nirs4all deduplicate --auto
# Saved 234 MB
```

**Result:**
- ✅ Scientists work with normal files and folders
- ✅ Admins can optimize when needed
- ✅ Sharing is one command
- ✅ Cleanup is straightforward

### Implementation Roadmap

**Phase 1: Core Structure** (Week 1-2)
- [ ] Date-prefixed run folders: `YYYY-MM-DD_<run_name>`
- [ ] Four-level hierarchy: Run/Dataset/Pipeline/Model
- [ ] Single metadata files: `RUN.yaml`, `PIPELINE.yaml`, `SCORES.yaml`
- [ ] Consolidated outputs: `predictions/`, `charts/`, `models/`
- [ ] Short IDs: Use existing console IDs (`[5grd70]`)

**Phase 2: Hybrid Storage** (Week 3-4)
- [ ] Save artifacts to `models/<model_id>/` (primary)
- [ ] Parallel save to `.cache/objects/` (tracking)
- [ ] Generate `MANIFEST.yaml` per model
- [ ] Create `catalog.db` (SQLite) for tracking
- [ ] Hash computation and deduplication detection

**Phase 3: Export Command** (Week 5)
- [ ] `runner.export_pipeline()` method
- [ ] Create self-contained zip packages
- [ ] Include `EXPORT.yaml` metadata
- [ ] `runner.import_pipeline()` method

**Phase 4: Cleanup Tools** (Week 6)
- [ ] `nirs4all clean` command with various filters
- [ ] `nirs4all cache` commands (stats, clean, rebuild)
- [ ] `nirs4all deduplicate` command
- [ ] `nirs4all analyze storage` command

**Phase 5: Migration** (Week 7)
- [ ] Conversion script from current UUID structure
- [ ] Generate short IDs from existing pipeline UIDs
- [ ] Rebuild catalog from existing artifacts
- [ ] Backward compatibility loader

### API Examples

**Creating Runs:**
```python
from nirs4all import PipelineRunner

runner = PipelineRunner()

# Simple run (auto-generated name: run_001)
runner.run("pipeline.yaml", "regression")

# Named run
runner.run("pipeline.yaml", "regression", run_name="wheat_quality")

# Multi-pipeline run
runner.run(
    pipeline_configs=["config1", "config2", "config3"],
    dataset="regression",
    run_name="wheat_comparison"
)

# Multi-dataset run
runner.run(
    pipeline_configs="universal_pipeline.yaml",
    datasets=["regression", "regression_2", "regression_3"],
    run_name="cross_dataset"
)
```

**Exporting Pipelines:**
```python
# Export best model from a pipeline
runner.export_pipeline(
    run="2024-10-14_wheat_quality",
    dataset="regression",
    pipeline="Q1_c20f9b",
    model_id="best",  # or specific: "5grd70"
    output="wheat_model.zip"
)

# Export entire pipeline (all models)
runner.export_pipeline(
    run="2024-10-14_wheat_quality",
    dataset="regression",
    pipeline="Q1_c20f9b",
    model_id="all",
    output="wheat_pipeline_complete.zip"
)

# Export entire run
runner.export_run(
    run="2024-10-14_wheat_quality",
    output="wheat_study_complete.zip"
)
```

**Using Exported Pipelines:**
```python
# Import to your workspace
runner.import_pipeline(
    "wheat_model.zip",
    to_run="my_wheat_study"  # Creates: 2024-10-15_my_wheat_study/
)

# Or extract and predict directly
import zipfile
zipfile.extract("wheat_model.zip", "temp/")

runner.predict_from_artifacts(
    model_dir="temp/models/5grd70/",
    data="my_new_samples.csv"
)
```

**Cleanup:**
```python
from nirs4all.utils import cleanup

# Programmatic cleanup
cleanup.remove_old_runs(older_than_days=30)

# Find duplicates
duplicates = cleanup.find_duplicates()
print(f"Potential savings: {duplicates.total_size_mb} MB")

# Deduplicate
cleanup.deduplicate(keep_strategy="newest", dry_run=False)
```

### File Size Examples (Realistic)

**Small Run** (1 pipeline, 1 dataset, 3 folds):
```
2024-10-14_quick_test/        Total: 18 MB
├── RUN.yaml                  2 KB
└── regression/
    └── Q1_simple/
        ├── PIPELINE.yaml     3 KB
        ├── SCORES.yaml       8 KB
        ├── predictions/      500 KB (5 CSV files)
        ├── charts/           1.2 MB (5 PNG files)
        ├── models/5grd70/    15 MB (5 pkl files)
        └── logs/             1 MB
```

**Medium Run** (3 pipelines, 1 dataset, 3 folds each):
```
2024-10-14_wheat_quality/     Total: 47 MB
├── RUN.yaml                  2 KB
└── regression/
    ├── DATASET.yaml          3 KB
    ├── Q1_c20f9b/            15.5 MB
    ├── Q1_eb2552/            15.8 MB
    ├── Q1_3e8bec/            15.2 MB
    └── COMPARISON.yaml       4 KB
```

**Large Run** (1 pipeline, 3 datasets, 6 folds):
```
2024-10-14_multidatasets/     Total: 89 MB
├── RUN.yaml                  2 KB
├── regression/
│   └── Q3_0f8b1a/            28 MB
├── regression_2/
│   └── Q3_0f8b1a/            29 MB
└── regression_3/
    └── Q3_0f8b1a/            32 MB
```

**After Deduplication** (if transformers shared):
- Medium Run: 47 MB → 35 MB (25% savings)
- Large Run: 89 MB → 61 MB (31% savings)

### Configuration Options

**In `~/.nirs4all/config.yaml`:**
```yaml
# Storage settings
storage:
  results_path: "results/"
  enable_cache: true              # Enable hybrid storage
  cache_path: ".cache/"
  auto_deduplicate: false         # Manual control

# Cleanup settings
cleanup:
  auto_archive_days: 90           # Auto-move to archive/
  warn_old_runs_days: 60          # Warning threshold
  max_storage_gb: 10              # Warning if exceeded

# Export settings
export:
  default_include_predictions: true
  default_include_charts: true
  default_include_logs: false
  compression_level: 6            # ZIP compression (0-9)

# Naming settings
naming:
  date_format: "YYYY-MM-DD"       # Run folder prefix
  auto_run_name_prefix: "run_"    # If no name provided
  auto_run_name_counter: true     # run_001, run_002, etc.
```

### Migration from Current Structure

**Current:**
```
results/
├── pipelines/
│   └── c40ca32c-43d7-4e1e-80c4-7f52de42b7a5/  # UUID
│       ├── pipeline.json
│       └── manifest.yaml
├── artifacts/objects/
│   └── ab/abc123.pkl  # Content-addressed
└── outputs/
    └── regression_Q1_c20f9b/
        └── 2D_Chart.png
```

**New:**
```
results/
├── 2024-10-14_wheat_quality/
│   └── regression/
│       └── Q1_c20f9b/
│           ├── PIPELINE.yaml      # Merged from pipeline.json
│           ├── SCORES.yaml        # Merged from manifest.yaml
│           ├── charts/
│           │   └── 2D_Chart.png   # Moved from outputs/
│           └── models/5grd70/
│               └── *.pkl          # Moved from artifacts/
└── .cache/
    └── objects/                   # Optional tracking
```

**Migration command:**
```bash
nirs4all migrate --from-version 5 --to-version 6

# Output:
#   Found 15 runs in old format
#   Migrating...
#   [1/15] UUID c40ca32c... → 2024-10-14_run_001
#   [2/15] UUID a7f3e9d1... → 2024-10-13_run_002
#   ...
#   Migration complete!
#   Old structure preserved in: results/legacy/
```

---

## Summary

**This architecture provides:**

✅ **Clear Organization:** Date → Run → Dataset → Pipeline → Model
✅ **Easy Browsing:** Physical files with semantic names
✅ **Simple Sharing:** Export command creates self-contained packages
✅ **Efficient Storage:** Optional deduplication saves 30-50% space
✅ **Easy Cleanup:** Delete folders or use smart CLI commands
✅ **User Control:** Deduplicate when you want, not automatic
✅ **Portability:** Each pipeline folder is independently movable
✅ **Short IDs:** 6-char memorable IDs everywhere (`5grd70`)

**Best for:**
- Individual researchers
- Small to medium teams
- 10-1000 experiments per project
- Frequent sharing of models
- Balance between simplicity and efficiency

**The key innovation:** Hybrid storage gives you efficiency benefits without sacrificing UX. Export command makes sharing trivial. You get the best of both worlds!

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
