# Workspace Architecture Implementation Roadmap

**Document**: Implementation Plan for User-Friendly Workspace Architecture
**Version**: 3.2 (Updated for Final Design)
**Timeline**: 5 weeks (5 phases × 1 week each)
**Target Branch**: `serialization_refactoring`

---

## Overview

This roadmap outlines the step-by-step implementation of the **user-friendly workspace architecture** (v3.2) described in `WORKSPACE_ARCHITECTURE.md`. The implementation prioritizes shallow folder structures, sequential numbering, and fast access to results.

### Key Changes from Previous Version

✅ **Shallow structure** - Max 3 levels (removed dataset subfolder)
✅ **Sequential numbering** - Pipelines named `0001_hash/` not `name_hash/`
✅ **Hidden binaries** - `_binaries/` with underscore prefix
✅ **Fast access** - `best_predictions/` folder for quick CSV access
✅ **Library types** - Three subdirectories: `filtered/`, `pipeline/`, `fullrun/`
✅ **Catalog archives** - Copies (not links) with `best_predictions/` folder

### Goals

✅ Implement user-friendly run-based organization
✅ Maximum 3-level folder depth
✅ Sequential numbering for easy browsing
✅ Fast access to best predictions
✅ Permanent catalog (survives run deletion)
✅ Three library types (filtered/pipeline/fullrun)
✅ Parquet database for fast queries
✅ Comprehensive testing at each phase

### Non-Goals

❌ Deep hierarchical structures (v3.0 design)
❌ Hash-only pipeline naming
❌ Link-based catalog (must be copies)
❌ SQL database (use Parquet only)

### Integration with Existing Code

This roadmap **extends existing nirs4all code** to minimize refactoring:

**Existing Classes to Extend:**
- `ManifestManager` (`pipeline/manifest_manager.py`) → Add sequential numbering per run
- `SimulationSaver` (`pipeline/io.py`) → Update to `runs/date_dataset/NNNN_hash/` structure, add export methods
- `Predictions` (`dataset/predictions.py`) → Add split Parquet storage (metadata + arrays), catalog/query methods

**New Coordination Layer:**
- `WorkspaceManager` → Initialize workspace structure
- `RunManager` → Coordinate run creation using existing classes
- `LibraryManager` → Manage filtered/pipeline/fullrun saves

**No Redundant Managers:**
- ❌ No separate ArtifactManager (use `ManifestManager`)
- ❌ No separate ExportManager (extend `SimulationSaver`)
- ❌ No separate CatalogManager (extend `Predictions`)
- ❌ No separate WorkspaceQuery (extend `Predictions`)

---

## Phase 1: Foundation Layer (Week 1)

### Goal
Create the core workspace structure with shallow hierarchy and sequential numbering.

### Tasks

#### 1.1 Create Workspace Manager (`nirs4all/workspace/workspace_manager.py`)

```python
class WorkspaceManager:
    """Manages workspace-level operations."""

    def __init__(self, workspace_root: Path):
        self.root = workspace_root
        self.runs_dir = workspace_root / "runs"
        self.exports_dir = workspace_root / "exports"
        self.library_dir = workspace_root / "library"
        self.catalog_dir = workspace_root / "catalog"

    def initialize_workspace(self) -> None:
        """Create workspace directory structure."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        # Library with 3 types
        (self.library_dir / "templates").mkdir(parents=True, exist_ok=True)
        (self.library_dir / "trained" / "filtered").mkdir(parents=True, exist_ok=True)
        (self.library_dir / "trained" / "pipeline").mkdir(parents=True, exist_ok=True)
        (self.library_dir / "trained" / "fullrun").mkdir(parents=True, exist_ok=True)

        # Catalog with archives
        (self.catalog_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.catalog_dir / "archives" / "filtered").mkdir(parents=True, exist_ok=True)
        (self.catalog_dir / "archives" / "pipeline").mkdir(parents=True, exist_ok=True)
        (self.catalog_dir / "archives" / "best_predictions").mkdir(parents=True, exist_ok=True)

        # Exports with best_predictions
        (self.exports_dir / "best_predictions").mkdir(parents=True, exist_ok=True)
        (self.exports_dir / "session_reports").mkdir(parents=True, exist_ok=True)

    def create_run(self, dataset_name: str, run_name: str = None) -> 'RunManager':
        """Create a new run for a dataset with optional custom name.

        Creates run directory:
        - Without custom name: YYYY-MM-DD_dataset/
        - With custom name: YYYY-MM-DD_dataset_runname/
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        if run_name:
            run_dir = self.runs_dir / f"{date_str}_{dataset_name}_{run_name}"
        else:
            run_dir = self.runs_dir / f"{date_str}_{dataset_name}"
        return RunManager(run_dir, dataset_name)

    def list_runs(self) -> List[Dict]:
        """List all runs with metadata."""
        # Implementation
```

#### 1.2 Extend SimulationSaver for Workspace Structure (`nirs4all/pipeline/io.py`)

**Rationale**: `SimulationSaver` already handles file I/O with `persist_artifact()`, `save_output()`, and directory management. Extend it with `register()` method to support workspace structure.

```python
# Extension to existing SimulationSaver class
class SimulationSaver:
    """Handles pipeline execution I/O (EXISTING CODE - extend with new methods)."""

    # ... existing __init__, persist_artifact, save_output, etc. ...

    # NEW: Workspace registration method
    def register(self, workspace_root: Path, dataset_name: str, pipeline_hash: str,
                 run_name: str = None, pipeline_name: str = None) -> Path:
        """Register pipeline in workspace structure with optional custom names.

        Creates:
        - Without custom names: workspace_root/runs/{date}_{dataset}/NNNN_{hash}/
        - With run_name: workspace_root/runs/{date}_{dataset}_{runname}/NNNN_{hash}/
        - With pipeline_name: workspace_root/runs/{date}_{dataset}/NNNN_{pipelinename}_{hash}/
        - With both: workspace_root/runs/{date}_{dataset}_{runname}/NNNN_{pipelinename}_{hash}/

        Returns: Full path to pipeline directory
        """
        from datetime import datetime
        from nirs4all.pipeline.manifest_manager import ManifestManager

        run_date = datetime.now().strftime("%Y-%m-%d")

        # Build run_id with optional custom name
        if run_name:
            run_id = f"{run_date}_{dataset_name}_{run_name}"
        else:
            run_id = f"{run_date}_{dataset_name}"
        run_dir = workspace_root / "runs" / run_id

        # Use ManifestManager for sequential numbering
        manifest = ManifestManager(str(run_dir))
        pipeline_num = manifest.get_next_pipeline_number()

        # Build pipeline_id with optional custom name
        if pipeline_name:
            pipeline_id = f"{pipeline_num:04d}_{pipeline_name}_{pipeline_hash}"
        else:
            pipeline_id = f"{pipeline_num:04d}_{pipeline_hash}"
        pipeline_dir = run_dir / pipeline_id

        # Create structure
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "_binaries").mkdir(exist_ok=True)

        return pipeline_dir
```

#### 1.3 Extend ManifestManager for Sequential Numbering (`nirs4all/pipeline/manifest_manager.py`)

**Rationale**: `ManifestManager` already generates UIDs and manages manifests. Add simple counter for sequential numbering.

```python
# Extension to existing ManifestManager class
class ManifestManager:
    """Manages pipeline manifests, UIDs, and datasets (EXISTING CODE - extend with new method)."""

    # ... existing methods: generate_uid, create_manifest, etc. ...

    # NEW: Sequential numbering for pipelines
    def get_next_pipeline_number(self) -> int:
        """Get next sequential pipeline number in run directory.

        Counts existing pipeline directories (excludes _binaries).
        Returns: Next number (e.g., 1, 2, 3...)
        """
        run_dir = Path(self.manifest_dir).parent  # manifest_dir points to run
        existing = [d for d in run_dir.iterdir()
                    if d.is_dir() and not d.name.startswith("_")]
        return len(existing) + 1
```

#### 1.4 Create Configuration Schema (`nirs4all/workspace/schemas.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class PipelineEntry(BaseModel):
    """Single pipeline in run (NNNN_hash format)."""
    pipeline_id: str  # e.g., "0001_a1b2c3"
    config_file: str
    status: str = "pending"

class RunConfig(BaseModel):
    """Configuration for a run (date_dataset)."""
    dataset_name: str
    created_at: datetime
    created_by: Optional[str] = None
    nirs4all_version: str
    python_version: str
    dataset_source: str
    task_type: str  # 'regression' or 'classification'
    pipelines: List[PipelineEntry]
    global_params: Dict = {}
    description: Optional[str] = None

class RunSummary(BaseModel):
    """Summary generated after run completes."""
    dataset_name: str
    run_date: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    total_pipelines: int
    successful_pipelines: int
    failed_pipelines: int
    best_pipeline: Optional[Dict] = None  # {pipeline_id, metric_name, metric_value}
    statistics: Dict
    errors: List[Dict] = []
```

### Deliverables

- [ ] `SimulationSaver.register()` method implemented (workspace structure support)
- [ ] `ManifestManager.get_next_pipeline_number()` method implemented (sequential numbering)
- [ ] `WorkspaceManager` class implemented (top-level workspace coordination)
- [ ] Configuration schemas defined (Pydantic models for RunConfig/RunSummary)
- [ ] Sequential numbering logic (0001, 0002, ...)
- [ ] `_binaries/` support (underscore prefix)
- [ ] Unit tests for extensions (30+ tests for new methods)
- [ ] Integration tests for workspace structure

### Testing

```bash
# Unit tests for new methods
pytest tests/pipeline/test_simulation_saver_workspace.py
pytest tests/pipeline/test_manifest_manager_numbering.py
pytest tests/workspace/test_workspace_manager.py

# Integration test - Create full run using extended classes
pytest tests/workspace/test_workspace_integration.py

# Test sequential numbering with ManifestManager
pytest tests/workspace/test_sequential_numbering.py
```

### Success Criteria

✅ Can create workspace structure using `SimulationSaver.register()`
✅ `ManifestManager` provides sequential numbering
✅ Sequential numbering works (0001, 0002, ...)
✅ `_binaries/` folder created with underscore
✅ Extends existing classes without duplication
✅ `_binaries/` folder created with underscore
✅ Can save artifacts with content-addressed naming
✅ All unit tests pass (50+)
✅ Integration test creates full structure



---

## Phase 2: Catalog & Export (Week 2)

### Goal
Extend `Predictions` class with Parquet storage (split metadata/arrays) and extend `SimulationSaver` with export capabilities. No redundant managers.

### Tasks

#### 2.1 Extend Predictions for Split Parquet Storage (`nirs4all/dataset/predictions.py`)

**Rationale**: `Predictions` already uses Polars DataFrames. Extend with split Parquet save/load (metadata separate from arrays).

```python
# Extension to existing Predictions class
class Predictions:
    """Prediction storage with Polars backend (EXISTING CODE - extend with Parquet methods)."""

    # ... existing __init__, from_csv, to_csv, filter methods ...

    # NEW: Split Parquet storage methods
    def save_to_parquet(self, catalog_dir: Path, prediction_id: str = None) -> tuple[Path, Path]:
        """Save predictions as split Parquet (metadata + arrays separate).

        Creates:
        - predictions_meta.parquet: lightweight metadata (prediction_id, dataset, metrics)
        - predictions_data.parquet: heavy arrays (prediction_id, y_true, y_pred, indices)

        Returns: (meta_path, data_path)
        """
        import polars as pl
        from uuid import uuid4

        pred_id = prediction_id or str(uuid4())

        # Split dataframe into metadata and arrays
        meta_cols = ["prediction_id", "dataset_name", "config_name", "test_score",
                     "train_score", "model_type", "created_at", "pipeline_hash"]
        array_cols = ["prediction_id", "y_true", "y_pred", "sample_indices",
                      "fold_id", "train_indices", "test_indices"]

        # Create metadata dataframe
        meta_df = self.data.select(meta_cols).unique(subset=["prediction_id"])

        # Create arrays dataframe
        data_df = self.data.select(array_cols)

        # Save to Parquet
        meta_path = catalog_dir / "predictions_meta.parquet"
        data_path = catalog_dir / "predictions_data.parquet"

        # Append mode - accumulate predictions over time
        if meta_path.exists():
            existing_meta = pl.read_parquet(meta_path)
            meta_df = pl.concat([existing_meta, meta_df])

        if data_path.exists():
            existing_data = pl.read_parquet(data_path)
            data_df = pl.concat([existing_data, data_df])

        meta_df.write_parquet(meta_path)
        data_df.write_parquet(data_path)

        return (meta_path, data_path)

    @classmethod
    def load_from_parquet(cls, catalog_dir: Path, prediction_ids: list[str] = None) -> 'Predictions':
        """Load predictions from split Parquet storage.

        Args:
            catalog_dir: Path to catalog directory
            prediction_ids: Optional list of prediction IDs to load (None = all)

        Returns: Predictions object with joined metadata + arrays
        """
        import polars as pl

        meta_path = catalog_dir / "predictions_meta.parquet"
        data_path = catalog_dir / "predictions_data.parquet"

        # Load metadata (always fast)
        meta_df = pl.read_parquet(meta_path)

        # Filter if needed
        if prediction_ids:
            meta_df = meta_df.filter(pl.col("prediction_id").is_in(prediction_ids))
            data_df = pl.read_parquet(data_path).filter(
                pl.col("prediction_id").is_in(prediction_ids)
            )
        else:
            data_df = pl.read_parquet(data_path)

        # Join metadata + arrays on prediction_id
        full_df = meta_df.join(data_df, on="prediction_id", how="inner")

        return cls(full_df)

    def archive_to_catalog(self, catalog_dir: Path, pipeline_dir: Path, metrics: dict) -> str:
        """Archive pipeline results to catalog with split Parquet.

        Reads predictions.csv from pipeline_dir, adds metadata, saves to catalog.
        Returns: prediction_id (UUID)
        """
        from uuid import uuid4
        import polars as pl

        # Generate prediction ID
        pred_id = str(uuid4())

        # Load predictions from pipeline
        pred_df = pl.read_csv(pipeline_dir / "predictions.csv")

        # Add metadata columns
        pred_df = pred_df.with_columns([
            pl.lit(pred_id).alias("prediction_id"),
            pl.lit(metrics.get("dataset_name")).alias("dataset_name"),
            pl.lit(metrics.get("config_name")).alias("config_name"),
            pl.lit(metrics.get("test_score")).alias("test_score"),
            pl.lit(metrics.get("train_score")).alias("train_score"),
            pl.lit(metrics.get("model_type")).alias("model_type"),
            pl.lit(pipeline_dir.name).alias("pipeline_hash"),
        ])

        # Update internal dataframe
        self.data = pred_df

        # Save to split Parquet
        self.save_to_parquet(catalog_dir, pred_id)

        return pred_id
```

#### 2.2 Extend SimulationSaver with Export Methods (`nirs4all/pipeline/io.py`)

#### 2.2 Extend SimulationSaver with Export Methods (`nirs4all/pipeline/io.py`)

**Rationale**: `SimulationSaver` already handles file I/O. Add export methods for best pipelines.

```python
# Extension to existing SimulationSaver class
class SimulationSaver:
    """Handles pipeline execution I/O (EXISTING CODE - extend with export methods)."""

    # ... existing methods: persist_artifact, save_output, register ...

    # NEW: Export methods
    def export_pipeline_full(self, pipeline_dir: Path, exports_dir: Path,
                            dataset_name: str, run_date: str, custom_name: str = None) -> Path:
        """Export full pipeline results to flat structure with optional custom name.

        Args:
            pipeline_dir: Path to pipeline (NNNN_hash/ or NNNN_pipelinename_hash/)
            exports_dir: Workspace exports directory
            dataset_name: Dataset name
            run_date: Run date (YYYYMMDD)
            custom_name: Optional custom name for export

        Creates export directory:
        - Without custom_name: dataset_run_pipelineid/
        - With custom_name: customname_pipelineid/

        Returns: Path to exported directory
        """
        import shutil

        pipeline_id = pipeline_dir.name  # e.g., "0001_a1b2c3" or "0001_baseline_a1b2c3"

        if custom_name:
            export_name = f"{custom_name}_{pipeline_id}"
        else:
            export_name = f"{dataset_name}_{run_date}_{pipeline_id}"
        export_path = exports_dir / export_name

        # Copy entire pipeline folder
        shutil.copytree(pipeline_dir, export_path, dirs_exist_ok=True)

        return export_path

    def export_best_prediction(self, predictions_file: Path, exports_dir: Path,
                              dataset_name: str, run_date: str, pipeline_id: str,
                              custom_name: str = None) -> Path:
        """Export predictions CSV to best_predictions/ folder with optional custom name.

        Args:
            predictions_file: Path to predictions.csv
            exports_dir: Workspace exports directory
            dataset_name, run_date, pipeline_id: Metadata for naming
            custom_name: Optional custom name for export

        Creates CSV filename:
        - Without custom_name: dataset_run_pipelineid.csv
        - With custom_name: customname_pipelineid.csv

        Returns: Path to exported CSV
        """
        import shutil

        best_dir = exports_dir / "best_predictions"
        best_dir.mkdir(parents=True, exist_ok=True)

        if custom_name:
            csv_name = f"{custom_name}_{pipeline_id}.csv"
        else:
            csv_name = f"{dataset_name}_{run_date}_{pipeline_id}.csv"
        dest_path = best_dir / csv_name

        shutil.copy2(predictions_file, dest_path)

        return dest_path
```

#### 2.3 Update PipelineRunner Integration (`nirs4all/pipeline/runner.py`)
        }

        # Append to parquet
        df = pd.DataFrame([row])

        if self.parquet_file.exists():
            existing = pd.read_parquet(self.parquet_file)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(self.parquet_file, index=False)

    def query_best(
        self,
        dataset: str = None,
        metric: str = "test_rmse",
        n: int = 10
    ) -> pd.DataFrame:
        """Query best predictions from Parquet database."""
        df = pd.read_parquet(self.parquet_file)

        if dataset:
            df = df[df["dataset"] == dataset]

        # Sort by metric (assumes lower is better for RMSE)
        df_sorted = df.sort_values(metric).head(n)

        return df_sorted

    def archive_filtered(self, pipeline_dir: Path, archive_name: str) -> Path:
        """Archive filtered pipeline (config + metrics only)."""
        dest = self.archives_dir / "filtered" / archive_name
        dest.mkdir(parents=True, exist_ok=True)

        # Copy only JSON files
        shutil.copy2(pipeline_dir / "pipeline.json", dest)
        shutil.copy2(pipeline_dir / "metrics.json", dest)

        return dest

    def archive_pipeline(self, pipeline_dir: Path, archive_name: str) -> Path:
        """Archive full pipeline (with binaries)."""
        dest = self.archives_dir / "pipeline" / archive_name

        # Copy entire pipeline folder
        shutil.copytree(pipeline_dir, dest, dirs_exist_ok=True)

        return dest

    def archive_best_prediction(
        self,
        predictions_file: Path,
        archive_name: str
    ) -> Path:
        """Archive predictions CSV to best_predictions/ folder."""
        dest = self.archives_dir / "best_predictions" / f"{archive_name}.csv"
        shutil.copy2(predictions_file, dest)
        return dest
```

### Deliverables

- [ ] `Predictions.save_to_parquet()` implemented (split metadata/arrays)
- [ ] `Predictions.load_from_parquet()` implemented
- [ ] `Predictions.archive_to_catalog()` implemented
- [ ] `SimulationSaver.export_pipeline_full()` implemented
- [ ] `SimulationSaver.export_best_prediction()` implemented
- [ ] `PipelineRunner` updated to use extended classes
- [ ] Unit tests for Predictions Parquet methods (20+ tests)
- [ ] Unit tests for SimulationSaver export methods (10+ tests)
- [ ] Integration test (run pipeline, verify catalog)

### Testing

```bash
# Unit tests for Predictions extensions
pytest tests/dataset/test_predictions_parquet.py

# Unit tests for SimulationSaver extensions
pytest tests/pipeline/test_simulation_saver_export.py

# Integration test - Full pipeline with catalog
pytest tests/workspace/test_catalog_integration.py
```

### Success Criteria

✅ Can save predictions as split Parquet (metadata + arrays separate)
✅ Can load predictions from Parquet efficiently (metadata-only queries fast)
✅ Can archive pipeline results to catalog automatically
✅ Can export best pipelines to exports/
✅ Extends existing classes without redundancy


---

## Phase 3: Library Management (Week 3)

### Goal
Implement library system with 3 types: filtered/, pipeline/, fullrun/.

### Tasks

#### 3.1 Create Library Manager (`nirs4all/workspace/library_manager.py`)

```python
class LibraryManager:
    """Manage library of saved pipelines."""

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.templates_dir = library_dir / "templates"
        self.trained_dir = library_dir / "trained"

        # Three types of trained pipelines
        self.filtered_dir = self.trained_dir / "filtered"
        self.pipeline_dir = self.trained_dir / "pipeline"
        self.fullrun_dir = self.trained_dir / "fullrun"

        # Initialize
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.filtered_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        self.fullrun_dir.mkdir(parents=True, exist_ok=True)

    def save_template(
        self,
        pipeline_config: Dict,
        name: str,
        description: str = ""
    ) -> Path:
        """Save pipeline template (config only, no trained artifacts)."""

        template_file = self.templates_dir / f"{name}.json"

        template = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "config": pipeline_config,
            "type": "template"
        }

        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)

        return template_file

    def save_filtered(
        self,
        pipeline_dir: Path,
        name: str,
        description: str = ""
    ) -> Path:
        """Save filtered pipeline (config + metrics only).

        Useful for: Tracking experiments, comparing configurations
        """
        dest_dir = self.filtered_dir / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy only JSON files
        shutil.copy2(pipeline_dir / "pipeline.json", dest_dir)
        shutil.copy2(pipeline_dir / "metrics.json", dest_dir)

        # Add metadata
        metadata = {
            "name": name,
            "description": description,
            "saved_at": datetime.now().isoformat(),
            "type": "filtered",
            "source": str(pipeline_dir)
        }
        with open(dest_dir / "library_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return dest_dir

    def save_pipeline_full(
        self,
        run_dir: Path,
        pipeline_dir: Path,
        name: str,
        description: str = ""
    ) -> Path:
        """Save full pipeline (all files + binaries).

        Useful for: Deployment, retraining, full reproducibility
        """
        dest_dir = self.pipeline_dir / name

        # Copy entire pipeline folder
        shutil.copytree(pipeline_dir, dest_dir, dirs_exist_ok=True)

        # Copy referenced binaries from run's _binaries/
        binaries_src = run_dir / "_binaries"
        binaries_dest = dest_dir / "_binaries"
        binaries_dest.mkdir(exist_ok=True)

        # Parse pipeline.json to find referenced artifacts
        with open(pipeline_dir / "pipeline.json") as f:
            pipeline_config = json.load(f)

        if "artifacts" in pipeline_config:
            for artifact_ref in pipeline_config["artifacts"]:
                artifact_path = artifact_ref["path"]  # e.g., "../_binaries/scaler_a1b2c3.pkl"
                artifact_filename = Path(artifact_path).name
                src_file = binaries_src / artifact_filename
                if src_file.exists():
                    shutil.copy2(src_file, binaries_dest)

        # Add metadata
        metadata = {
            "name": name,
            "description": description,
            "saved_at": datetime.now().isoformat(),
            "type": "pipeline",
            "source": str(pipeline_dir)
        }
        with open(dest_dir / "library_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return dest_dir

    def save_fullrun(
        self,
        run_dir: Path,
        name: str,
        description: str = ""
    ) -> Path:
        """Save entire run (all pipelines + binaries + data).

        Useful for: Complete experiment archiving, cross-dataset comparison
        """
        dest_dir = self.fullrun_dir / name

        # Copy entire run folder
        shutil.copytree(run_dir, dest_dir, dirs_exist_ok=True)

        # Add metadata
        metadata = {
            "name": name,
            "description": description,
            "saved_at": datetime.now().isoformat(),
            "type": "fullrun",
            "source": str(run_dir)
        }
        with open(dest_dir / "library_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return dest_dir

    def list_templates(self) -> List[Dict]:
        """List all available templates."""
        templates = []
        for file in self.templates_dir.glob("*.json"):
            with open(file) as f:
                templates.append(json.load(f))
        return templates

    def load_template(self, name: str) -> Dict:
        """Load a template by name."""
        template_file = self.templates_dir / f"{name}.json"
        with open(template_file) as f:
            return json.load(f)
```

### Deliverables

- [ ] `LibraryManager` with 3 save types (filtered/pipeline/fullrun)
- [ ] Template management (config-only pipelines)
- [ ] Metadata tracking for all library entries
- [ ] Binary copying for pipeline type
- [ ] CLI commands for library operations

### Testing

```bash
# Unit tests
pytest tests/workspace/test_library_manager.py

# Integration tests
pytest tests/workspace/test_library_save_load.py
```

### Success Criteria

✅ Can save filtered pipelines (config + metrics)
✅ Can save full pipelines (with binaries)
✅ Can save complete runs (fullrun)
✅ Can list and load templates
✅ Metadata tracked for all library entries
✅ All tests pass


---

## Phase 4: Query and Reporting (Week 4)

### Goal
Extend `Predictions` class with query methods for Parquet catalog. No new WorkspaceQuery class.

### Tasks

#### 4.1 Extend Predictions with Query Methods (`nirs4all/dataset/predictions.py`)

**Rationale**: `Predictions` already has Polars backend and Parquet storage. Add query methods for catalog analysis.

```python
# Extension to existing Predictions class
class Predictions:
    """Prediction storage with Polars backend (EXISTING CODE - extend with query methods)."""

    # ... existing methods: save_to_parquet, load_from_parquet, archive_to_catalog ...

    # NEW: Query methods for catalog
    def query_best(self, dataset_name: str = None, metric: str = "test_score",
                   n: int = 10, ascending: bool = False) -> pl.DataFrame:
        """Find best pipelines by metric.

        Args:
            dataset_name: Optional filter by dataset
            metric: Metric column to sort by
            n: Number of results
            ascending: Sort order (False = higher is better)

        Returns: Top N predictions by metric
        """
        import polars as pl

        df = self.data  # Use existing Polars DataFrame

        if dataset_name:
            df = df.filter(pl.col("dataset_name") == dataset_name)

        df_sorted = df.sort(metric, descending=not ascending).head(n)

        return df_sorted.select([
            "prediction_id", "dataset_name", "config_name",
            metric, "pipeline_hash", "created_at"
        ])

    def filter_by_criteria(self, dataset_name: str = None,
                           date_range: tuple[str, str] = None,
                           metric_thresholds: dict[str, float] = None) -> pl.DataFrame:
        """Filter predictions by multiple criteria.

        Args:
            dataset_name: Optional dataset filter
            date_range: Optional (start_date, end_date) tuple
            metric_thresholds: Optional dict of {metric: min_value}

        Returns: Filtered predictions
        """
        import polars as pl

        df = self.data

        if dataset_name:
            df = df.filter(pl.col("dataset_name") == dataset_name)

        if date_range:
            start, end = date_range
            df = df.filter(
                (pl.col("created_at") >= start) & (pl.col("created_at") <= end)
            )

        if metric_thresholds:
            for metric, threshold in metric_thresholds.items():
                df = df.filter(pl.col(metric) >= threshold)

        return df

    def compare_across_datasets(self, pipeline_hash: str,
                                metric: str = "test_score") -> pl.DataFrame:
        """Compare same pipeline configuration across datasets.

        Args:
            pipeline_hash: Pipeline configuration hash
            metric: Metric to compare

        Returns: Comparison table with aggregated stats
        """
        import polars as pl

        df = self.data.filter(pl.col("pipeline_hash") == pipeline_hash)

        comparison = df.group_by("dataset_name").agg([
            pl.col(metric).min().alias(f"{metric}_min"),
            pl.col(metric).max().alias(f"{metric}_max"),
            pl.col(metric).mean().alias(f"{metric}_mean"),
            pl.count().alias("num_predictions")
        ])

        return comparison

    def list_runs(self, dataset_name: str = None) -> pl.DataFrame:
        """List all runs with summary statistics.

        Args:
            dataset_name: Optional dataset filter

        Returns: Run summary with count and best score
        """
        import polars as pl

        df = self.data

        if dataset_name:
            df = df.filter(pl.col("dataset_name") == dataset_name)

        runs = df.group_by(["dataset_name", "created_at"]).agg([
            pl.count().alias("num_pipelines"),
            pl.col("test_score").max().alias("best_score")
        ])

        return runs.sort("created_at", descending=True)
```

#### 4.2 Create Report Generator (`nirs4all/workspace/reporter.py`)

```python
class ReportGenerator:
    """Generate reports from workspace data."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.catalog_dir = workspace_root / "catalog"
        self.reports_dir = self.catalog_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Use Predictions class for catalog queries
        from nirs4all.data.predictions import Predictions
        self.predictions = Predictions.load_from_parquet(self.catalog_dir)

    def generate_run_summary(
        self,
        run_dir: Path
    ) -> Path:
        """Generate summary report for a run."""

        # Read run_summary.json
        with open(run_dir / "run_summary.json") as f:
            summary = json.load(f)

        # Create markdown report
        report_name = f"{run_dir.name}_summary.md"
        report_path = self.reports_dir / report_name

        with open(report_path, 'w') as f:
            f.write(f"# Run Summary: {run_dir.name}\n\n")
            f.write(f"**Dataset**: {summary['dataset_name']}\n")
            f.write(f"**Date**: {summary['run_date']}\n")
            f.write(f"**Status**: {summary['status']}\n")
            f.write(f"**Duration**: {summary['duration_seconds']}s\n\n")

            f.write(f"## Statistics\n\n")
            f.write(f"- Total pipelines: {summary['total_pipelines']}\n")
            f.write(f"- Successful: {summary['successful_pipelines']}\n")
            f.write(f"- Failed: {summary['failed_pipelines']}\n\n")

            if summary.get("best_pipeline"):
                best = summary["best_pipeline"]
                f.write(f"## Best Pipeline\n\n")
                f.write(f"- ID: {best['pipeline_id']}\n")
                f.write(f"- Metric: {best['metric_name']} = {best['metric_value']:.4f}\n\n")

            if summary.get("errors"):
                f.write(f"## Errors\n\n")
                for error in summary["errors"]:
                    f.write(f"- {error['pipeline_id']}: {error['message']}\n")

        return report_path

    def generate_dataset_report(
        self,
        dataset_name: str
    ) -> Path:
        """Generate comprehensive report for dataset."""

        # Query all runs for dataset
        df = self.query.filter_pipelines(dataset=dataset_name)

        report_name = f"{dataset_name}_report.md"
        report_path = self.reports_dir / report_name

        with open(report_path, 'w') as f:
            f.write(f"# Dataset Report: {dataset_name}\n\n")
            f.write(f"**Total Predictions**: {len(df)}\n\n")

            # Best models
            f.write(f"## Top 10 Models\n\n")
            best = self.query.best_by_metric(dataset=dataset_name, n=10)
            f.write(best.to_markdown(index=False))
            f.write("\n\n")

            # Statistics by date
            f.write(f"## Runs by Date\n\n")
            runs = self.query.list_runs(dataset=dataset_name)
            f.write(runs.to_markdown(index=False))

        return report_path
```

### Deliverables

- [ ] `Predictions.query_best()` implemented (top N by metric)
- [ ] `Predictions.filter_by_criteria()` implemented (dataset/date/threshold filters)
- [ ] `Predictions.compare_across_datasets()` implemented (cross-dataset analysis)
- [ ] `Predictions.list_runs()` implemented (run summaries)
- [ ] `ReportGenerator` for markdown reports
- [ ] CLI commands for querying (`nirs4all query best --dataset X`)
- [ ] Unit tests for query methods (25+ tests)

### Testing

```bash
# Unit tests for Predictions query extensions
pytest tests/dataset/test_predictions_query.py

# Unit tests for report generator
pytest tests/workspace/test_reporter.py

# Integration tests (requires existing catalog)
pytest tests/workspace/test_query_integration.py
```

### Success Criteria

✅ Can query best pipelines by metric using `Predictions.query_best()`
✅ Can filter predictions by multiple criteria
✅ Can compare same pipeline across datasets
✅ Can list runs with summary statistics
✅ Can generate markdown reports from catalog
✅ Extends existing Predictions class without redundancy

---

## Phase 5: UI Integration (Week 5)

### Goal
Update CLI and UI to use workspace structure with extended classes.

### Tasks

#### 5.1 Update CLI Commands (`nirs4all/cli/workspace_commands.py`)

```python
import click
from pathlib import Path
from nirs4all.workspace import WorkspaceManager
from nirs4all.data.predictions import Predictions

@click.group()
def workspace():
    """Workspace management commands."""
    pass

@workspace.command()
@click.argument("path", type=click.Path())
def init(path):
    """Initialize a new workspace."""
    ws = WorkspaceManager(Path(path))
    ws.initialize_workspace()
    click.echo(f"✓ Workspace initialized at {path}")

@workspace.command()
@click.option("--workspace", default="workspace", help="Workspace root")
def list_runs(workspace):
    """List all runs in workspace."""
    ws_path = Path(workspace)
    runs_dir = ws_path / "runs"

    if not runs_dir.exists():
        click.echo("No runs found")
        return

    for run in runs_dir.iterdir():
        if run.is_dir() and not run.name.startswith("_"):
            click.echo(f"  {run.name}")

@workspace.command()
@click.option("--workspace", default="workspace", help="Workspace root")
@click.option("--dataset", help="Filter by dataset")
@click.option("--metric", default="test_score", help="Metric to sort by")
@click.option("-n", default=10, help="Number of results")
def query_best(workspace, dataset, metric, n):
    """Query best pipelines from catalog."""
    catalog_dir = Path(workspace) / "catalog"

    # Load predictions from catalog
    preds = Predictions.load_from_parquet(catalog_dir)

    # Query best
    best = preds.query_best(dataset_name=dataset, metric=metric, n=n)

    # Display results
    click.echo(best.to_pandas().to_string(index=False))
```

### Deliverables

- [ ] CLI commands updated to use `WorkspaceManager`, `Predictions`
- [ ] Commands: `init`, `list-runs`, `query-best`, `export`
- [ ] UI integration with workspace structure
- [ ] Documentation for CLI commands

### Testing

```bash
# Unit tests for CLI
pytest tests/cli/test_workspace_commands.py

# Manual testing
nirs4all workspace init ./my_workspace
nirs4all workspace query-best --dataset corn --metric test_rmse -n 5
```

### Success Criteria

✅ CLI commands work with workspace structure
✅ Can initialize, query, and export via CLI
✅ UI correctly displays workspace data
✅ Extends existing classes without redundancy

---

## Summary

This roadmap implements a clean, modern workspace architecture by **extending existing classes** instead of creating redundant managers.

### Key Design Decisions

**Extend Existing Code**
- `SimulationSaver` → add `register()` and `export_*()` methods
- `ManifestManager` → add `get_next_pipeline_number()` method
- `Predictions` → add Parquet storage (`save_to_parquet()`, `load_from_parquet()`, query methods)
- No redundant `PipelineWorkspace`, `ExportManager`, `CatalogManager`, `WorkspaceQuery`

**Split Parquet Storage**
- `predictions_meta.parquet` - Lightweight metadata (fast filtering)
- `predictions_data.parquet` - Heavy arrays (loaded only when needed)
- Linked via `prediction_id` UUID
- Optimized for common use case: metadata queries (90%) vs array loading (10%)

### Architecture Principles

1. **Shallow structure** (max 3 levels)
2. **Sequential numbering** (0001_hash, 0002_hash...)
3. **Content-addressed artifacts** (shared `_binaries/` with deduplication)
4. **Split storage** (metadata separate from arrays)
5. **Extend, don't replace** (use existing SimulationSaver, Predictions, ManifestManager)

### Phase Summary

| Phase | Week | Focus | Key Deliverables |
|-------|------|-------|------------------|
| 1 | Week 1 | Foundation | `SimulationSaver.register()`, `ManifestManager.get_next_pipeline_number()` |
| 2 | Week 2 | Catalog & Export | `Predictions` Parquet methods, `SimulationSaver.export_*()` |
| 3 | Week 3 | Library | Library management for reusable artifacts |
| 4 | Week 4 | Query | `Predictions.query_best()`, `filter_by_criteria()`, reporting |
| 5 | Week 5 | UI/CLI | CLI commands, UI integration |

### Testing Strategy

**Unit Testing**
- Test each extension method individually (30-50 tests per phase)
- Mock dependencies for isolated testing
- Test edge cases and error handling

**Integration Testing**
- Full pipeline execution with workspace registration
- Catalog save/load round-trips
- Cross-dataset query scenarios

**Performance Testing**
- Metadata-only queries (<100ms for 10k predictions)
- Array loading (acceptable for 90th percentile)
- Parquet file size growth

### Success Metrics

✅ All extension methods tested (150+ tests total)
✅ Split Parquet storage working (metadata fast, arrays on-demand)
✅ No redundant managers (extend existing classes only)
✅ Sequential numbering functional (0001, 0002...)
✅ Documentation complete for all extensions
✅ CLI commands working with extended classes

        zip_file = self.trained_dir / f"{library_name}.zip"

        with zipfile.ZipFile(zip_file, 'w') as zf:
            # Add pipeline.json
            zf.write(
                pipeline_workspace.pipeline_file,
                arcname="pipeline.json"
            )

            # Add metrics.json
            zf.write(
                pipeline_workspace.metrics_file,
                arcname="metrics.json"
            )

            # Add all referenced binaries
            pipeline_config = pipeline_workspace._load_config()
            for artifact in pipeline_config.get("artifacts", []):
                artifact_path = pipeline_workspace.pipeline_dir.parent.parent / artifact["path"]
                zf.write(artifact_path, arcname=f"binaries/{artifact_path.name}")

            # Add metadata
            metadata = {
                "library_name": library_name,
                "description": description,
                "saved_at": datetime.now().isoformat(),
                "original_path": str(pipeline_workspace.pipeline_dir),
                "nirs4all_version": __version__
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        return zip_file

    def load_trained_pipeline(self, library_name: str) -> Tuple[Dict, Dict[str, Any]]:
        """Load trained pipeline from library.

        Returns:
            (pipeline_config, {artifact_name: artifact_object})
        """
        zip_file = self.trained_dir / f"{library_name}.zip"

        if not zip_file.exists():
            raise FileNotFoundError(f"Library item not found: {library_name}")

        with zipfile.ZipFile(zip_file, 'r') as zf:
            # Load pipeline config
            with zf.open("pipeline.json") as f:
                pipeline_config = json.load(f)

            # Load artifacts
            artifacts = {}
            for artifact_info in pipeline_config.get("artifacts", []):
                artifact_name = Path(artifact_info["path"]).name
                with zf.open(f"binaries/{artifact_name}") as f:
                    artifacts[artifact_info["name"]] = joblib.load(f)

        return pipeline_config, artifacts

    def list_templates(self) -> List[Dict]:
        """List all templates in library."""
        templates = []
        for template_file in self.templates_dir.glob("*.json"):
            with open(template_file, 'r') as f:
                templates.append(json.load(f))
        return templates

    def list_trained(self) -> List[Dict]:
        """List all trained pipelines in library."""
        trained = []
        for zip_file in self.trained_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_file, 'r') as zf:
                with zf.open("metadata.json") as f:
                    metadata = json.load(f)
                trained.append(metadata)
        return trained
```

#### 4.2 Extend Predictions with Search Methods (`nirs4all/dataset/predictions.py`)

**Rationale**: `Predictions` already stores catalog data in Parquet. Extend with search/query methods instead of creating separate CatalogManager.

```python
# Extension to existing Predictions class
class Predictions:
    # ... existing __init__, save_to_parquet, load_from_parquet methods ...

    def search_predictions(
        self,
        dataset_name: str = None,
        pipeline_name: str = None,
        metric_filter: Dict = None,
        created_after: str = None
    ) -> 'Predictions':
        """Search predictions with filters using Polars queries.

        Args:
            dataset_name: Filter by dataset (None = all)
            pipeline_name: Filter by pipeline name (None = all)
            metric_filter: e.g., {"test_rmse_max": 0.5, "train_score_min": 0.8}
            created_after: ISO date string (e.g., "2024-10-01")

        Returns:
            New Predictions object with filtered results
        """
        import polars as pl

        df = self.data

        # Apply filters using Polars expressions
        if dataset_name:
            df = df.filter(pl.col("dataset_name") == dataset_name)

        if pipeline_name:
            df = df.filter(pl.col("config_name").str.contains(pipeline_name))

        if created_after:
            df = df.filter(pl.col("created_at") >= created_after)

        if metric_filter:
            for metric_spec, threshold in metric_filter.items():
                # Parse metric_spec like "test_rmse_max" or "train_score_min"
                if "_max" in metric_spec:
                    metric_col = metric_spec.replace("_max", "")
                    df = df.filter(pl.col(metric_col) <= threshold)
                elif "_min" in metric_spec:
                    metric_col = metric_spec.replace("_min", "")
                    df = df.filter(pl.col(metric_col) >= threshold)

        return Predictions(df)

    def get_global_best(self, metric: str = "test_score", mode: str = "max") -> Dict:
        """Get best model across ALL predictions in catalog.

        Args:
            metric: Metric column name (e.g., "test_score", "test_rmse")
            mode: "max" (higher is better) or "min" (lower is better)

        Returns: Dictionary with best prediction metadata
        """
        import polars as pl

        if self.data.height == 0:
            raise ValueError("No predictions in catalog")

        # Sort and get best
        if mode == "min":
            best_row = self.data.sort(metric).head(1)
        else:
            best_row = self.data.sort(metric, descending=True).head(1)

        # Convert to dict
        return best_row.to_dicts()[0]

    def get_summary_stats(self, group_by: str = "dataset_name") -> pl.DataFrame:
        """Get summary statistics grouped by column.

        Args:
            group_by: Column to group by ("dataset_name", "config_name", "model_type")

        Returns: Polars DataFrame with aggregated statistics
        """
        import polars as pl

        return self.data.group_by(group_by).agg([
            pl.col("test_score").mean().alias("avg_test_score"),
            pl.col("test_score").max().alias("best_test_score"),
            pl.col("test_score").min().alias("worst_test_score"),
            pl.col("prediction_id").count().alias("num_predictions")
        ])
```

### Deliverables

- [ ] `LibraryManager` for template and trained pipeline management
- [ ] `Predictions.search_predictions()` method with Polars filtering
- [ ] `Predictions.get_global_best()` method for best model tracking
- [ ] `Predictions.get_summary_stats()` method for catalog analytics
- [ ] Pipeline packaging (zip with config + binaries)
- [ ] Pipeline loading from library

### Testing

```bash
# Library tests
pytest tests/workspace/test_library_manager.py
pytest tests/workspace/test_pipeline_packaging.py

# Predictions search tests
pytest tests/dataset/test_predictions_search.py
pytest tests/dataset/test_predictions_global_best.py
pytest tests/dataset/test_predictions_summary.py
```

### Success Criteria

✅ Can save pipelines to library (with and without binaries)
✅ Can load pipelines from library
✅ Can search predictions with complex Polars filters
✅ Can find global best model across all predictions
✅ Can generate summary statistics by dataset/pipeline/model
✅ Library and predictions tests pass

---

## Phase 5: CLI & Documentation (Week 5)

### Goal
Update CLI, create comprehensive documentation, and finalize user-facing interfaces.

### Tasks

#### 6.1 Update CLI Commands (`nirs4all/cli/`)

```bash
# New CLI structure with custom naming support

# Run management with custom names
nirs4all run \
  --dataset wheat_sample1 \
  --pipelines configs/baseline_pls.json configs/optimized_svm.json \
  --run-name "wheat-quality-study" \
  --pipeline-names "baseline" "optimized" \
  --description "Testing quality prediction models"

# Creates: runs/2024-10-23_wheat_sample1_wheat-quality-study/
# With pipelines: 0001_baseline_a1b2c3/, 0002_optimized_d4e5f6/

# Without custom names (default sequential naming)
nirs4all run \
  --dataset wheat_sample1 \
  --pipelines configs/baseline_pls.json \
  --description "Quick test"

# Creates: runs/2024-10-23_wheat_sample1/
# With pipeline: 0001_a1b2c3/

nirs4all sessions list
nirs4all sessions show 2024-10-23_wheat_sample1_wheat-quality-study
nirs4all sessions delete 2024-09-15_old-experiments

# Library management
nirs4all library save \
  --from runs/2024-10-23_wheat_sample1_wheat-quality-study/0001_baseline_a1b2c3 \
  --name wheat_quality_baseline \
  --include-binaries

nirs4all library list
nirs4all library load wheat_quality_baseline

# Catalog queries
nirs4all catalog list wheat_sample1
nirs4all catalog best wheat_sample1 --metric test_rmse
nirs4all catalog search --pipeline baseline_pls --rmse-max 0.5

# Export with custom names
nirs4all export \
  --pipeline runs/2024-10-23_wheat_sample1_wheat-quality-study/0001_baseline_a1b2c3 \
  --custom-name "production_model_v1" \
  --output exports/

# Creates: exports/production_model_v1_0001_baseline_a1b2c3/
```

#### 6.2 Create User Documentation

**File**: `docs/WORKSPACE_GUIDE.md`

```markdown
# Workspace Guide

## Introduction
The workspace organizes all your NIR spectroscopy work by experimental sessions...

## Quick Start
...

## Workflows
### Training Models
### Reusing Pipelines
### Finding Best Models
...

## CLI Reference
...

## File Structure Explained
...

## Best Practices
...

## Troubleshooting
...
```

#### 6.3 Create Developer Documentation

**File**: `docs/WORKSPACE_ARCHITECTURE_DEV.md`

```markdown
# Workspace Architecture - Developer Guide

## Design Principles
...

## Class Hierarchy
...

## Extending the System
### Adding new workspace managers
### Custom catalog backends
### Alternative storage formats
...

## Testing Strategy
...

## Performance Considerations
...
```

#### 6.4 Update Examples

Update all examples in `examples/` to use new workspace structure:

```python
# examples/Q14_workspace_example.py

from nirs4all.workspace import WorkspaceManager, SessionExecutor

# Initialize workspace
workspace = WorkspaceManager("my_workspace")

# Execute session
executor = SessionExecutor(workspace.root)
results = executor.execute_session(
    session_name="wheat-quality-study",
    datasets=["data/wheat_sample1.csv"],
    pipelines=["configs/baseline_pls.json"],
    description="Testing PLS regression"
)

# Check results
print(f"Session: {results.session_dir}")
print(f"Best model: {results.best_model}")

# Save to library
library = workspace.get_library_manager()
library.save_trained_pipeline(
    results.best_model_workspace,
    library_name="wheat_quality_baseline",
    description="Baseline PLS model for wheat quality"
)
```

### Deliverables

- [ ] Complete CLI implementation for workspace operations with custom naming
- [ ] User guide (`WORKSPACE_GUIDE.md`) with custom naming examples
- [ ] Developer guide (`WORKSPACE_ARCHITECTURE_DEV.md`)
- [ ] Updated examples (Q14, Q15, etc.) using new workspace structure
- [ ] API reference documentation for extended classes

### Testing

```bash
# CLI tests
pytest tests/cli/test_workspace_commands.py
pytest tests/cli/test_library_commands.py
pytest tests/cli/test_catalog_commands.py

# Documentation tests
pytest tests/docs/test_examples.py  # Run all examples
```

### Success Criteria

✅ All CLI commands implemented and working
✅ Comprehensive user documentation
✅ Developer documentation complete
✅ All examples updated and tested
✅ CLI tests pass

---

## Testing Strategy

### Unit Tests (Ongoing)

```
tests/
├── workspace/
│   ├── test_workspace_manager.py         # 15 tests
│   ├── test_session_manager.py           # 12 tests
│   ├── test_dataset_workspace.py         # 10 tests
│   ├── test_pipeline_workspace.py        # 15 tests
│   ├── test_workspace_manager.py         # 12 tests (workspace structure)
│   ├── test_predictions_parquet.py       # 18 tests (split Parquet storage)
│   └── test_library_manager.py           # 15 tests
├── execution/
│   ├── test_pipeline_execution.py        # 10 tests
│   ├── test_progress_tracker.py          # 8 tests
│   └── test_error_handler.py             # 10 tests
├── integration/
│   ├── test_workspace_execution.py       # 5 tests
│   ├── test_multi_dataset.py             # 5 tests
│   ├── test_library_workflow.py          # 5 tests
│   ├── test_catalog_workflow.py          # 5 tests
│   └── test_custom_naming.py             # 5 tests
└── cli/
    ├── test_workspace_commands.py        # 15 tests
    ├── test_library_commands.py          # 10 tests
    └── test_catalog_commands.py          # 10 tests

Total: ~150 tests
```

### Integration Tests (End of Each Phase)

Phase-specific integration tests ensure components work together:

- **Phase 1**: Create full workspace structure with custom naming
- **Phase 2**: Execute pipeline end-to-end with split Parquet catalog
- **Phase 3**: Library save/load workflow
- **Phase 4**: Catalog search and global best queries
- **Phase 5**: CLI commands produce expected results with custom naming

### Manual Testing Checklist

After each phase:

- [ ] Create workspace structure manually (verify file paths)
- [ ] Run example pipelines (verify results)
- [ ] Test custom naming (runs, pipelines, exports)
- [ ] Check generated JSON/Parquet files (verify format)
- [ ] Test error conditions (verify error handling)
- [ ] Review logs (verify logging quality)
- [ ] Test cleanup (verify safe deletion)
- [ ] Verify no redundant managers created

---

## Performance Considerations

### Deduplication Efficiency

**Goal**: Minimize storage with content-addressed artifacts

**Metrics**:
- Storage saved: 30-50% for grid search experiments
- Hash computation time: <50ms per artifact
- Load time: No overhead (same as before)

### Catalog Indexing

**Goal**: Fast queries across large numbers of predictions

**Options**:
1. **JSON files** (current): Simple, good for <1000 predictions
2. **SQLite** (future): Better for >1000 predictions, enables complex queries
3. **Parquet** (future): Best for analytics, DataFrame-friendly

### Session Cleanup

**Goal**: Easy deletion of old sessions without breaking references

**Strategy**:
- Per-session binaries (no global references)
- Catalog updated automatically on deletion
- Library pipelines never deleted by session cleanup

---

## Rollout Strategy

### Week 1-2: Foundation & Catalog

- Phases 1-2 completed
- Internal testing
- Extension methods validated

### Week 3: Library & Query

- Phases 3-4 completed
- Integration testing
- Performance benchmarking

### Week 4-5: CLI & Documentation

- Phase 5 completed
- User documentation complete
- Examples updated
- Ready for release

---

## Risk Assessment

### High Priority Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking API changes | HIGH | Careful extension design, thorough testing |
| Performance regression | MEDIUM | Benchmarking, profiling, Parquet optimization |
| Catalog scaling | MEDIUM | Split Parquet design, indexed queries |

### Medium Priority Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| User confusion | MEDIUM | Clear documentation, examples, CLI help |
| Integration complexity | MEDIUM | Extend existing classes, minimal new code |
| File format changes | LOW | Version metadata in manifests |

---

## Success Metrics

### Technical Metrics

- [ ] All tests passing (150+ tests across 5 phases)
- [ ] Storage deduplication working (`_binaries/` shared across pipelines)
- [ ] No performance regression (<5% overhead)
- [ ] Split Parquet queries <100ms for metadata (10k predictions)
- [ ] Custom naming support functional (runs, pipelines, exports)

### User Experience Metrics

- [ ] Run creation: <10 seconds for typical setup
- [ ] Catalog queries: <1 second for 1000 predictions
- [ ] CLI commands: Intuitive, self-documenting
- [ ] Documentation: Complete with custom naming examples
- [ ] Shallow structure: Max 3 levels maintained

### Code Quality Metrics

- [ ] No redundant managers created (use existing SimulationSaver, Predictions, ManifestManager)
- [ ] Extension pattern successful (minimal new classes)
- [ ] Split Parquet storage validated
- [ ] Sequential numbering working (0001, 0002, ...)
- [ ] Content-addressed artifacts deduplicated

---

## Post-Implementation

### Future Enhancements

1. **Web UI** (v0.7): Visual workspace browser, interactive catalog
2. **Remote Storage** (v0.8): S3/Azure blob support for artifacts
3. **Collaboration** (v0.9): Shared workspaces, permissions
4. **Versioning** (v1.0): Git-like pipeline versioning

### Maintenance

- Weekly bug fixes
- Monthly performance reviews
- Quarterly feature additions
- Yearly major updates

---

## Contact & Support

**Questions?** Open an issue on GitHub
**Feedback?** Email maintainers
**Contributions?** See CONTRIBUTING.md

---

**Status**: Ready for implementation ✅
**Next Action**: Begin Phase 1 - Foundation Layer
