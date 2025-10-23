# Workspace Architecture Implementation Roadmap

**Document**: Implementation Plan for User-Friendly Workspace Architecture
**Version**: 3.2 (Updated for Final Design)
**Timeline**: 6 weeks (6 phases × 1 week each)
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
❌ SQL database (parquet only)

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

    def create_run(self, dataset_name: str) -> 'RunManager':
        """Create a new run for a dataset."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        run_dir = self.runs_dir / f"{date_str}_{dataset_name}"
        return RunManager(run_dir, dataset_name)

    def list_runs(self) -> List[Dict]:
        """List all runs with metadata."""
        # Implementation
```

#### 1.2 Create Run Manager (`nirs4all/workspace/run_manager.py`)

**Note**: Simplified from Session/Dataset/Pipeline hierarchy to Run/Pipeline.

```python
class RunManager:
    """Manages a single run (date + dataset)."""

    def __init__(self, run_dir: Path, dataset_name: str):
        self.run_dir = run_dir
        self.dataset_name = dataset_name
        self.binaries_dir = run_dir / "_binaries"       # Underscore prefix!
        self.pipelines_dir = run_dir                     # Pipelines at run level
        self.config_file = run_dir / "run_config.json"
        self.summary_file = run_dir / "run_summary.json"
        self.log_file = run_dir / "run.log"

    def initialize(self, config: Dict) -> None:
        """Initialize run structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.binaries_dir.mkdir(exist_ok=True)
        self._write_config(config)
        self._init_logger()

    def get_next_pipeline_number(self) -> int:
        """Get next pipeline number (simple counter)."""
        existing = [d for d in self.run_dir.iterdir()
                    if d.is_dir() and not d.name.startswith("_")]
        return len(existing) + 1

    def create_pipeline(self, pipeline_hash: str) -> 'PipelineWorkspace':
        """Create pipeline with sequential numbering."""
        pipeline_num = self.get_next_pipeline_number()
        pipeline_id = f"{pipeline_num:04d}_{pipeline_hash}"  # 0001_a1b2c3
        pipeline_dir = self.run_dir / pipeline_id
        return PipelineWorkspace(pipeline_dir, self.binaries_dir)

    def finalize(self) -> None:
        """Generate run summary and close."""
        summary = self._compute_summary()
        self._write_summary(summary)
        self.logger.info(f"Run finalized: {self.run_dir.name}")
```

#### 1.3 Create Pipeline Workspace (`nirs4all/workspace/pipeline_workspace.py`)

```python
class PipelineWorkspace:
    """Manages a single pipeline (0001_hash/)."""

    def __init__(self, pipeline_dir: Path, shared_binaries_dir: Path):
        self.pipeline_dir = pipeline_dir
        self.shared_binaries_dir = shared_binaries_dir
        self.pipeline_file = pipeline_dir / "pipeline.json"
        self.metrics_file = pipeline_dir / "metrics.json"
        self.predictions_file = pipeline_dir / "predictions.csv"
        # Charts directly in pipeline folder (no outputs/ subfolder)

    def initialize(self, pipeline_config: Dict) -> None:
        """Initialize pipeline workspace."""
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        self._write_pipeline_config(pipeline_config)

    def save_artifact(self, operator_name: str, artifact: Any) -> str:
        """Save artifact to shared binaries with content-addressed naming."""
        artifact_hash = self._compute_hash(artifact)
        filename = f"{operator_name}_{artifact_hash[:6]}.pkl"
        filepath = self.shared_binaries_dir / filename

        if not filepath.exists():  # Only save if not already present
            joblib.dump(artifact, filepath)

        # Return relative path for pipeline.json
        return f"../_binaries/{filename}"  # Note: _binaries with underscore!

    def save_metrics(self, metrics: Dict) -> None:
        """Save metrics to metrics.json."""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_predictions(self, predictions: pd.DataFrame) -> None:
        """Save predictions to predictions.csv."""
        predictions.to_csv(self.predictions_file, index=False)

    def save_chart(self, chart_name: str, figure: Any) -> None:
        """Save chart directly in pipeline folder."""
        chart_path = self.pipeline_dir / f"{chart_name}.png"
        figure.savefig(chart_path, dpi=150, bbox_inches='tight')
```

#### 1.5 Create Configuration Schema (`nirs4all/workspace/schemas.py`)

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

- [ ] `nirs4all/workspace/` package created
- [ ] `WorkspaceManager` class implemented (with library/catalog structure)
- [ ] `RunManager` class implemented (simplified from SessionManager)
- [ ] `PipelineWorkspace` class implemented
- [ ] Configuration schemas defined (Pydantic models)
- [ ] Sequential numbering logic (0001, 0002, ...)
- [ ] `_binaries/` support (underscore prefix)
- [ ] Unit tests for all managers (50+ tests)

### Testing

```bash
# Unit tests
pytest tests/workspace/test_workspace_manager.py
pytest tests/workspace/test_run_manager.py
pytest tests/workspace/test_pipeline_workspace.py

# Integration test - Create full run
pytest tests/workspace/test_workspace_integration.py

# Test sequential numbering
pytest tests/workspace/test_sequential_numbering.py
```

### Success Criteria

✅ Can create workspace structure programmatically
✅ Can initialize runs with proper metadata
✅ Sequential numbering works (0001, 0002, ...)
✅ `_binaries/` folder created with underscore
✅ Can save artifacts with content-addressed naming
✅ All unit tests pass (50+)
✅ Integration test creates full structure



---

## Phase 2: Storage Integration (Week 2)

### Goal
Integrate workspace managers with existing pipeline execution logic. Update to shallow structure.

### Tasks

#### 2.1 Update Pipeline Runner (`nirs4all/pipeline_runner.py`)

```python
class PipelineRunner:
    """Execute pipelines with new workspace structure."""

    def __init__(self, workspace_root: Path = None):
        self.workspace_root = workspace_root or Path("workspace")
        self.workspace_manager = WorkspaceManager(self.workspace_root)
        # Legacy support
        self.results_root = Path("results")
        self.use_legacy = False  # Flag for backward compatibility

    def run_batch(
        self,
        dataset: Dataset,
        pipelines: List[Pipeline],
        **kwargs
    ) -> RunResults:
        """Run multiple pipelines for a single dataset."""

        # Create run (date_dataset)
        run_mgr = self.workspace_manager.create_run(dataset.name)
        run_mgr.initialize({
            "dataset_name": dataset.name,
            "dataset_source": dataset.source,
            "task_type": dataset.task_type,
            "total_pipelines": len(pipelines),
            ...
        })

        # Run all pipelines with sequential numbering
        results = []
        for pipeline in pipelines:
            pipeline_hash = self._compute_hash(pipeline)
            pipeline_ws = run_mgr.create_pipeline(pipeline_hash[:6])

            result = self._run_pipeline(pipeline, dataset, pipeline_ws)
            results.append(result)

        # Finalize
        run_mgr.finalize()
        return RunResults(run_mgr.run_dir, results)

    def _run_pipeline(
        self,
        pipeline: Pipeline,
        dataset: Dataset,
        pipeline_ws: PipelineWorkspace
    ) -> PipelineResult:
        """Execute single pipeline and save to workspace."""

        # Initialize workspace
        pipeline_ws.initialize({
            "id": pipeline_ws.pipeline_dir.name,  # e.g., "0001_a1b2c3"
            "config": pipeline.to_dict(),
            "created_at": datetime.now().isoformat(),
            "dataset": dataset.name,
            ...
        })

        # Train pipeline
        trained_pipeline = pipeline.fit(dataset.X_train, dataset.y_train)

        # Save artifacts (content-addressed to _binaries/)
        artifact_refs = []
        for step_idx, step in enumerate(trained_pipeline.steps):
            operator_name = step.__class__.__name__
            artifact_path = pipeline_ws.save_artifact(operator_name, step)
            artifact_refs.append({
                "step": step_idx,
                "name": operator_name,
                "path": artifact_path  # e.g., "../_binaries/scaler_a1b2c3.pkl"
            })

        # Update pipeline.json with artifact references
        pipeline_config = pipeline_ws.read_pipeline_config()
        pipeline_config["artifacts"] = artifact_refs
        pipeline_ws.write_pipeline_config(pipeline_config)

        # Evaluate
        metrics = self._compute_metrics(trained_pipeline, dataset)
        pipeline_ws.save_metrics(metrics)

        # Predictions
        predictions_df = self._create_predictions_df(
            trained_pipeline, dataset
        )
        )
        pipeline_ws.save_predictions(predictions_df)

        # Save charts
        figures = self._generate_charts(trained_pipeline, dataset, metrics)
        for chart_name, fig in figures.items():
            pipeline_ws.save_chart(chart_name, fig)

        # Update catalog
        self._update_catalog(dataset.name, pipeline_ws, metrics)

        return PipelineResult(pipeline_ws.pipeline_dir, metrics)

    def _compute_hash(self, pipeline: Pipeline) -> str:
        """Compute hash of pipeline configuration."""
        config_hash = hashlib.sha256(
            json.dumps(pipeline.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        return config_hash
```

#### 2.2 Update Artifact Manager (`nirs4all/storage/artifact_manager.py`)

**No changes needed** - Already handles content-addressing. Works with `_binaries/` folder.

#### 2.3 Add Export Manager (`nirs4all/workspace/export_manager.py`)

```python
class ExportManager:
    """Manage exports of best results."""

    def __init__(self, exports_dir: Path):
        self.exports_dir = exports_dir
        self.best_predictions_dir = exports_dir / "best_predictions"
        self.best_predictions_dir.mkdir(parents=True, exist_ok=True)

    def export_pipeline_full(
        self,
        run_dir: Path,
        pipeline_dir: Path,
        dataset_name: str
    ) -> Path:
        """Export full pipeline results to flat structure.

        Args:
            run_dir: Path to run (YYYY-MM-DD_dataset/)
            pipeline_dir: Path to pipeline (NNNN_hash/)
            dataset_name: Name of dataset

        Returns:
            Path to export directory
        """
        # Create export name: dataset_date_pipeline
        run_date = run_dir.name.split("_")[0]  # Extract YYYY-MM-DD
        pipeline_id = pipeline_dir.name  # e.g., "0001_a1b2c3"

        export_name = f"{dataset_name}_{run_date}_{pipeline_id}"
        export_path = self.exports_dir / export_name

        # Copy entire pipeline folder
        shutil.copytree(pipeline_dir, export_path, dirs_exist_ok=True)

        return export_path

    def export_best_prediction(
        self,
        predictions_file: Path,
        dataset_name: str,
        run_date: str,
        pipeline_id: str
    ) -> Path:
        """Export predictions CSV to best_predictions/ folder.

        Args:
            predictions_file: Path to predictions.csv
            dataset_name: Name of dataset
            run_date: Date string (YYYY-MM-DD)
            pipeline_id: Pipeline ID (NNNN_hash)

        Returns:
            Path to exported CSV
        """
        # Create unique name for CSV
        csv_name = f"{dataset_name}_{run_date}_{pipeline_id}.csv"
        dest_path = self.best_predictions_dir / csv_name

        # Copy CSV
        shutil.copy2(predictions_file, dest_path)

        return dest_path
```

#### 2.4 Update Catalog Manager (`nirs4all/workspace/catalog_manager.py`)

```python
class CatalogManager:
    """Manage global prediction catalog with Parquet database."""

    def __init__(self, catalog_dir: Path):
        self.catalog_dir = catalog_dir
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_file = catalog_dir / "predictions.parquet"
        self.reports_dir = catalog_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Archives structure
        self.archives_dir = catalog_dir / "archives"
        (self.archives_dir / "filtered").mkdir(parents=True, exist_ok=True)
        (self.archives_dir / "pipeline").mkdir(parents=True, exist_ok=True)
        (self.archives_dir / "best_predictions").mkdir(parents=True, exist_ok=True)

    def add_prediction(
        self,
        dataset_name: str,
        run_date: str,
        pipeline_id: str,
        pipeline_config: Dict,
        metrics: Dict
    ) -> None:
        """Add prediction entry to Parquet database."""

        # Create row
        row = {
            "dataset": dataset_name,
            "run_date": run_date,
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now().isoformat(),
            **metrics,  # Flatten metrics
            "config_hash": self._hash_config(pipeline_config)
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

- [ ] `PipelineRunner` updated to use `RunManager` (not Session/Dataset hierarchy)
- [ ] Sequential numbering implemented (0001, 0002, ...)
- [ ] `ExportManager` created with `best_predictions/` support
- [ ] `CatalogManager` updated with Parquet database and archives structure
- [ ] Artifact manager works with `_binaries/` folder
- [ ] Integration tests (run 10 pipelines, verify structure)

### Testing

```bash
# Unit tests
pytest tests/workspace/test_pipeline_runner.py
pytest tests/workspace/test_export_manager.py
pytest tests/workspace/test_catalog_manager.py

# Integration test - Run 10 pipelines
pytest tests/workspace/test_full_batch_run.py
```

### Success Criteria

✅ Can run batch of pipelines with sequential numbering
✅ Artifacts saved to `_binaries/` with deduplication
✅ Predictions exported to `exports/best_predictions/`
✅ Catalog updated with Parquet entries
✅ Charts saved directly in pipeline folders
✅ All integration tests pass


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
Implement Parquet-based querying and reporting tools.

### Tasks

#### 4.1 Create Query Interface (`nirs4all/workspace/query.py`)

```python
class WorkspaceQuery:
    """Query workspace using Parquet database."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.catalog_dir = workspace_root / "catalog"
        self.parquet_file = self.catalog_dir / "predictions.parquet"

    def best_by_metric(
        self,
        dataset: str = None,
        metric: str = "test_rmse",
        n: int = 10,
        ascending: bool = True
    ) -> pd.DataFrame:
        """Find best pipelines by metric."""
        df = pd.read_parquet(self.parquet_file)

        if dataset:
            df = df[df["dataset"] == dataset]

        df_sorted = df.sort_values(metric, ascending=ascending).head(n)

        return df_sorted[["dataset", "run_date", "pipeline_id", metric, "config_hash"]]

    def filter_pipelines(
        self,
        dataset: str = None,
        date_range: Tuple[str, str] = None,
        metric_threshold: Dict[str, float] = None
    ) -> pd.DataFrame:
        """Filter pipelines by criteria."""
        df = pd.read_parquet(self.parquet_file)

        if dataset:
            df = df[df["dataset"] == dataset]

        if date_range:
            start, end = date_range
            df = df[(df["run_date"] >= start) & (df["run_date"] <= end)]

        if metric_threshold:
            for metric, threshold in metric_threshold.items():
                df = df[df[metric] <= threshold]

        return df

    def compare_datasets(
        self,
        pipeline_config_hash: str,
        metric: str = "test_rmse"
    ) -> pd.DataFrame:
        """Compare same pipeline across datasets."""
        df = pd.read_parquet(self.parquet_file)
        df_filtered = df[df["config_hash"] == pipeline_config_hash]

        return df_filtered.pivot_table(
            index="dataset",
            values=metric,
            aggfunc=["min", "max", "mean"]
        )

    def list_runs(self, dataset: str = None) -> pd.DataFrame:
        """List all runs."""
        df = pd.read_parquet(self.parquet_file)

        if dataset:
            df = df[df["dataset"] == dataset]

        # Group by run
        runs = df.groupby(["run_date", "dataset"]).agg({
            "pipeline_id": "count",
            "test_rmse": "min"
        }).rename(columns={"pipeline_id": "num_pipelines", "test_rmse": "best_rmse"})

        return runs.reset_index()
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
        self.query = WorkspaceQuery(workspace_root)

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

- [ ] `WorkspaceQuery` for Parquet-based queries
- [ ] `ReportGenerator` for markdown reports
- [ ] Support for cross-dataset comparisons
- [ ] CLI commands for querying (`nirs4all query best --dataset X`)

### Testing

```bash
# Unit tests
pytest tests/workspace/test_query.py
pytest tests/workspace/test_reporter.py

# Integration tests (requires existing runs)
pytest tests/workspace/test_query_integration.py
```

### Success Criteria

✅ Can query best pipelines by metric
✅ Can filter pipelines by dataset/date/threshold
✅ Can compare pipelines across datasets
✅ Can generate run summaries
✅ Can generate dataset reports
✅ All tests pass

---

## Phase 5: Migration Tools (Week 5)

### Goal
Provide tools to migrate from old `results/` structure to new workspace.

### Tasks

#### 5.1 Create Migration Tool (`nirs4all/workspace/migration.py`)

```python
class WorkspaceMigration:
    """Migrate data from old structure to new workspace."""

    def __init__(
        self,
        old_results_dir: Path,
        workspace_root: Path
    ):
        self.old_results_dir = old_results_dir
        self.workspace_root = workspace_root
        self.workspace_manager = WorkspaceManager(workspace_root)

    def migrate_all(self) -> Dict:
        """Migrate all results to workspace."""

        stats = {
            "sessions_migrated": 0,
            "pipelines_migrated": 0,
            "errors": []
        }

        # Find all result folders
        for result_folder in self.old_results_dir.glob("*"):
            if not result_folder.is_dir():
                continue

            try:
                self._migrate_result_folder(result_folder)
                stats["sessions_migrated"] += 1
            except Exception as e:
                stats["errors"].append({
                    "folder": str(result_folder),
                    "error": str(e)
                })

        return stats

    def _migrate_result_folder(self, result_folder: Path) -> None:
        """Migrate a single result folder."""

        # Parse old structure
        # Example: results/2024-01-15_experiment1/
        folder_name = result_folder.name
        parts = folder_name.split("_", 1)
        date_str = parts[0] if len(parts) > 0 else "unknown"
        session_name = parts[1] if len(parts) > 1 else folder_name

        # Create run in new structure
        # Assumes single dataset per old result folder
        dataset_name = self._extract_dataset_name(result_folder)
        run_mgr = self.workspace_manager.create_run(dataset_name)

        # Initialize
        run_mgr.initialize({
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "migration_source": str(result_folder)
        })

        # Migrate pipelines
        for pipeline_folder in result_folder.glob("*"):
            if pipeline_folder.is_dir():
                self._migrate_pipeline(pipeline_folder, run_mgr)

        run_mgr.finalize()

    def _migrate_pipeline(
        self,
        old_pipeline_dir: Path,
        run_mgr: 'RunManager'
    ) -> None:
        """Migrate a single pipeline."""

        # Generate hash from config
        pipeline_hash = self._compute_hash(old_pipeline_dir)

        # Create new pipeline workspace
        pipeline_ws = run_mgr.create_pipeline(pipeline_hash[:6])
        pipeline_ws.initialize({})

        # Copy files
        for file in old_pipeline_dir.glob("*"):
            if file.is_file():
                shutil.copy2(file, pipeline_ws.pipeline_dir)

    def _extract_dataset_name(self, result_folder: Path) -> str:
        """Extract dataset name from old structure."""
        # Implementation depends on old structure
        # Try to find dataset info in config files
        pass

    def _compute_hash(self, pipeline_dir: Path) -> str:
        """Compute hash of pipeline."""
        # Implementation
        pass
```

### Deliverables

- [ ] Migration tool for old `results/` structure
- [ ] CLI command (`nirs4all migrate results/`)
- [ ] Validation of migrated data
- [ ] Migration report with stats

### Testing

```bash
# Unit tests
pytest tests/workspace/test_migration.py

# Integration tests (with sample old data)
pytest tests/workspace/test_migration_integration.py
```

### Success Criteria

✅ Can migrate old results to new workspace
✅ Preserves all files and metadata
✅ Generates migration report
✅ All tests pass


---

## Phase 6: CLI and API (Week 6)

### Goal
Update CLI and API to use new workspace structure.

### Tasks

#### 6.1 Update CLI Commands (`nirs4all/cli/workspace_commands.py`)

```python
import click
from pathlib import Path
from nirs4all.workspace import WorkspaceManager, WorkspaceQuery, LibraryManager

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
@click.option("--dataset", "-d", required=True, help="Dataset name")
@click.option("--pipelines", "-p", multiple=True, help="Pipeline config files")
def run(dataset, pipelines):
    """Run batch of pipelines on a dataset."""
    from nirs4all.pipeline_runner import PipelineRunner

    runner = PipelineRunner()
    # Load dataset
    # Load pipelines
    # Execute batch
    click.echo(f"✓ Running {len(pipelines)} pipelines on {dataset}")

@workspace.command()
@click.option("--dataset", "-d", help="Filter by dataset")
@click.option("--metric", "-m", default="test_rmse", help="Metric to sort by")
@click.option("--top", "-n", default=10, type=int, help="Number of results")
def query(dataset, metric, top):
    """Query best pipelines."""
    query_engine = WorkspaceQuery(Path("workspace"))
    results = query_engine.best_by_metric(dataset=dataset, metric=metric, n=top)

    click.echo(results.to_string(index=False))

@workspace.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.argument("pipeline_id")
@click.option("--name", "-n", required=True, help="Library name")
@click.option("--type", "-t",
              type=click.Choice(["filtered", "pipeline", "fullrun"]),
              default="pipeline",
              help="Save type")
def save(run_dir, pipeline_id, name, type):
    """Save pipeline to library."""
    lib = LibraryManager(Path("workspace/library"))
    run_path = Path(run_dir)
    pipeline_path = run_path / pipeline_id

    if type == "filtered":
        dest = lib.save_filtered(pipeline_path, name)
    elif type == "pipeline":
        dest = lib.save_pipeline_full(run_path, pipeline_path, name)
    elif type == "fullrun":
        dest = lib.save_fullrun(run_path, name)

    click.echo(f"✓ Saved to library: {dest}")

@workspace.command()
@click.option("--type", "-t",
              type=click.Choice(["templates", "filtered", "pipeline", "fullrun"]),
              default="templates",
              help="Library type to list")
def list_library(type):
    """List library contents."""
    lib = LibraryManager(Path("workspace/library"))

    if type == "templates":
        templates = lib.list_templates()
        for t in templates:
            click.echo(f"- {t['name']}: {t.get('description', 'N/A')}")

@workspace.command()
@click.argument("run_dir", type=click.Path(exists=True))
def report(run_dir):
    """Generate report for a run."""
    from nirs4all.workspace.reporter import ReportGenerator

    reporter = ReportGenerator(Path("workspace"))
    report_path = reporter.generate_run_summary(Path(run_dir))

    click.echo(f"✓ Report generated: {report_path}")

@workspace.command()
@click.argument("old_results_dir", type=click.Path(exists=True))
def migrate(old_results_dir):
    """Migrate old results to new workspace."""
    from nirs4all.workspace.migration import WorkspaceMigration

    migration = WorkspaceMigration(
        Path(old_results_dir),
        Path("workspace")
    )
    stats = migration.migrate_all()

    click.echo(f"✓ Migrated {stats['sessions_migrated']} runs")
    if stats["errors"]:
        click.echo(f"⚠ {len(stats['errors'])} errors")
```

#### 6.2 Update API Endpoints (for UI integration)

```python
# In nirs4all_ui/api/workspace.py

from fastapi import APIRouter
from nirs4all.workspace import WorkspaceManager, WorkspaceQuery

router = APIRouter(prefix="/workspace")

@router.get("/runs")
def list_runs(dataset: str = None):
    """List all runs."""
    query = WorkspaceQuery(Path("workspace"))
    runs_df = query.list_runs(dataset=dataset)
    return runs_df.to_dict(orient="records")

@router.get("/best")
def get_best_pipelines(
    dataset: str = None,
    metric: str = "test_rmse",
    top: int = 10
):
    """Get best pipelines."""
    query = WorkspaceQuery(Path("workspace"))
    results = query.best_by_metric(dataset=dataset, metric=metric, n=top)
    return results.to_dict(orient="records")

@router.get("/library/templates")
def list_templates():
    """List library templates."""
    lib = LibraryManager(Path("workspace/library"))
    return lib.list_templates()

@router.post("/export")
def export_pipeline(
    run_dir: str,
    pipeline_id: str
):
    """Export pipeline to exports/ folder."""
    from nirs4all.workspace.export_manager import ExportManager

    export_mgr = ExportManager(Path("workspace/exports"))
    export_path = export_mgr.export_pipeline_full(
        Path(run_dir),
        Path(run_dir) / pipeline_id,
        dataset_name="..."  # Extract from run_dir
    )
    return {"export_path": str(export_path)}
```

### Deliverables

- [ ] CLI commands: `init`, `run`, `query`, `save`, `list-library`, `report`, `migrate`
- [ ] API endpoints for UI integration
- [ ] Documentation for all commands
- [ ] Examples and usage guides

### Testing

```bash
# CLI tests
pytest tests/cli/test_workspace_commands.py

# API tests
pytest tests/api/test_workspace_endpoints.py

# Manual testing
nirs4all workspace init ./workspace
nirs4all workspace query --dataset corn --metric test_rmse --top 5
nirs4all workspace save runs/2024-01-15_corn/0001_a1b2c3 -n "best_corn_model" -t pipeline
```

### Success Criteria

✅ All CLI commands work
✅ API endpoints functional
✅ Documentation complete
✅ Examples tested
✅ All tests pass

---

## Summary

### Timeline Overview

| Phase | Week | Focus | Key Deliverables |
|-------|------|-------|------------------|
| 1 | Week 1 | Foundation | WorkspaceManager, RunManager, PipelineWorkspace |
| 2 | Week 2 | Storage | PipelineRunner, ExportManager, CatalogManager with Parquet |
| 3 | Week 3 | Library | LibraryManager with 3 types (filtered/pipeline/fullrun) |
| 4 | Week 4 | Query & Reports | WorkspaceQuery, ReportGenerator |
| 5 | Week 5 | Migration | Migration tools for old structure |
| 6 | Week 6 | CLI & API | Commands, endpoints, documentation |

### Key Architectural Changes from v3.0

✅ **Removed DatasetWorkspace** - Pipelines directly in run folder (shallow structure)
✅ **Sequential numbering** - 0001_hash, 0002_hash... (user-friendly)
✅ **Hidden binaries** - `_binaries/` with underscore prefix
✅ **Fast access** - `best_predictions/` folder for quick CSV access
✅ **Library types** - Three subdirectories: filtered/, pipeline/, fullrun/
✅ **Parquet database** - Single file for all predictions (fast queries)
✅ **Permanent archives** - Catalog stores copies (not links)

### Testing Strategy

- Unit tests for each class (50+ tests per phase)
- Integration tests for full workflows
- Manual testing of CLI commands
- Backward compatibility tests
- Performance tests for Parquet queries

### Documentation Requirements

- API documentation (docstrings)
- User guide for workspace structure
- CLI command reference
- Migration guide
- Examples and tutorials

### Success Metrics

✅ All tests pass (300+ tests total)
✅ Documentation complete
✅ CLI commands functional
✅ API endpoints working
✅ Migration successful
✅ Performance acceptable (query <1s)
✅ User-friendly structure (max 3 levels deep)
        return template_file

    def save_trained_pipeline(
        self,
        pipeline_workspace: PipelineWorkspace,
        library_name: str,
        description: str = None
    ) -> Path:
        """Package trained pipeline (config + binaries) into library."""

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

#### 4.2 Enhance Catalog with Search (`nirs4all/workspace/catalog_manager.py`)

```python
class CatalogManager:
    # ... existing methods ...

    def search_predictions(
        self,
        dataset_name: str = None,
        pipeline_name: str = None,
        metric_filter: Dict = None,
        session_filter: str = None
    ) -> List[Dict]:
        """Search predictions with filters.

        Args:
            dataset_name: Filter by dataset (None = all)
            pipeline_name: Filter by pipeline name (None = all)
            metric_filter: e.g., {"test_rmse": {"max": 0.5}}
            session_filter: Filter by session name pattern

        Returns:
            List of matching predictions
        """
        results = []

        # Iterate over all datasets
        dataset_dirs = (self.catalog_dir / "datasets").glob("*")
        for dataset_dir in dataset_dirs:
            if dataset_name and dataset_dir.name != dataset_name:
                continue

            index_file = dataset_dir / "index.json"
            if not index_file.exists():
                continue

            with open(index_file, 'r') as f:
                catalog = json.load(f)

            for pred in catalog["predictions"]:
                # Apply filters
                if pipeline_name and pred["pipeline_name"] != pipeline_name:
                    continue

                if session_filter and session_filter not in pred["session"]:
                    continue

                if metric_filter:
                    if not self._matches_metric_filter(pred["metrics"], metric_filter):
                        continue

                results.append(pred)

        return results

    def _matches_metric_filter(self, metrics: Dict, filter_spec: Dict) -> bool:
        """Check if metrics match filter specification."""
        for metric_name, constraints in filter_spec.items():
            if metric_name not in metrics:
                return False

            value = metrics[metric_name]

            if "min" in constraints and value < constraints["min"]:
                return False
            if "max" in constraints and value > constraints["max"]:
                return False

        return True

    def get_global_best(self, metric: str = "test_rmse") -> Dict:
        """Get best model across ALL datasets."""
        all_predictions = self.search_predictions()

        if not all_predictions:
            raise ValueError("No predictions in catalog")

        if metric.endswith("rmse") or metric.endswith("mae"):
            # Lower is better
            best = min(all_predictions, key=lambda p: p["metrics"].get(metric, float('inf')))
        else:
            # Higher is better
            best = max(all_predictions, key=lambda p: p["metrics"].get(metric, float('-inf')))

        return best
```

### Deliverables

- [ ] `LibraryManager` for template and trained pipeline management
- [ ] Enhanced `CatalogManager` with search functionality
- [ ] Pipeline packaging (zip with config + binaries)
- [ ] Pipeline loading from library
- [ ] Global best model tracking

### Testing

```bash
# Library tests
pytest tests/workspace/test_library_manager.py
pytest tests/workspace/test_pipeline_packaging.py

# Catalog search tests
pytest tests/workspace/test_catalog_search.py
pytest tests/workspace/test_global_best.py
```

### Success Criteria

✅ Can save pipelines to library (with and without binaries)
✅ Can load pipelines from library
✅ Can search catalog with complex filters
✅ Can find global best model across all datasets
✅ Library and catalog tests pass

---

## Phase 5: Migration Tools (Week 5)

### Goal
Create tools to migrate existing `results/` data to new `workspace/` structure.

### Tasks

#### 5.1 Create Migration Manager (`nirs4all/migration/migration_manager.py`)

```python
class MigrationManager:
    """Migrate from old results/ structure to new workspace/ structure."""

    def __init__(self, results_dir: Path, workspace_dir: Path):
        self.results_dir = results_dir
        self.workspace_dir = workspace_dir
        self.workspace_manager = WorkspaceManager(workspace_dir)

    def migrate_all(self, session_name: str = "migrated_results") -> Dict:
        """Migrate all existing results to new structure.

        Args:
            session_name: Name for migrated session

        Returns:
            Migration report (success/failure counts)
        """
        logger.info("Starting migration from results/ to workspace/")

        # Analyze existing structure
        analysis = self._analyze_results_structure()
        logger.info(f"Found {analysis['pipeline_count']} pipelines")

        # Create migration session
        session_mgr = self.workspace_manager.create_session(session_name)
        session_mgr.initialize({
            "session_name": session_name,
            "description": "Migrated from legacy results/ structure",
            "migration_date": datetime.now().isoformat()
        })

        # Migrate pipelines
        report = {
            "total_pipelines": analysis["pipeline_count"],
            "migrated": 0,
            "failed": 0,
            "errors": []
        }

        for pipeline_dir in analysis["pipeline_dirs"]:
            try:
                self._migrate_pipeline(pipeline_dir, session_mgr)
                report["migrated"] += 1
            except Exception as e:
                logger.error(f"Failed to migrate {pipeline_dir}: {e}")
                report["failed"] += 1
                report["errors"].append({"pipeline": str(pipeline_dir), "error": str(e)})

        # Finalize
        session_mgr.finalize()
        logger.info(f"Migration complete: {report['migrated']} migrated, {report['failed']} failed")

        return report

    def _analyze_results_structure(self) -> Dict:
        """Analyze existing results/ structure."""
        pipeline_dirs = list((self.results_dir / "pipelines").glob("*"))

        return {
            "pipeline_count": len(pipeline_dirs),
            "pipeline_dirs": pipeline_dirs,
            "has_artifacts": (self.results_dir / "artifacts").exists(),
            "has_datasets": (self.results_dir / "datasets").exists()
        }

    def _migrate_pipeline(self, old_pipeline_dir: Path, session_mgr: SessionManager):
        """Migrate single pipeline."""

        # Load old manifest
        manifest_file = old_pipeline_dir / "manifest.yaml"
        with open(manifest_file, 'r') as f:
            manifest = yaml.safe_load(f)

        # Determine dataset name (from manifest or directory name)
        dataset_name = manifest.get("dataset", "unknown_dataset")

        # Create dataset workspace
        dataset_ws = session_mgr.create_dataset_workspace(dataset_name)
        if not dataset_ws.dataset_info_file.exists():
            dataset_ws.initialize({
                "name": dataset_name,
                "migrated": True,
                "original_path": str(old_pipeline_dir)
            })

        # Create pipeline workspace
        pipeline_id = old_pipeline_dir.name  # Use old UID as pipeline ID
        pipeline_ws = dataset_ws.create_pipeline_workspace(pipeline_id)

        # Migrate pipeline config
        pipeline_config = self._convert_manifest_to_config(manifest)
        pipeline_ws.initialize(pipeline_config)

        # Migrate artifacts
        artifact_refs = []
        for artifact_info in manifest.get("artifacts", []):
            old_artifact_path = self.results_dir / artifact_info["path"]
            if old_artifact_path.exists():
                # Load artifact
                artifact = joblib.load(old_artifact_path)

                # Save to new location (content-addressed)
                new_path = pipeline_ws.save_artifact(
                    artifact_info["operator"],
                    artifact
                )
                artifact_refs.append({
                    "step": artifact_info["step"],
                    "name": artifact_info["operator"],
                    "path": new_path
                })

        # Update pipeline config with artifact references
        pipeline_config["artifacts"] = artifact_refs
        with open(pipeline_ws.pipeline_file, 'w') as f:
            json.dump(pipeline_config, f, indent=2)

        # Migrate predictions (if exist)
        old_predictions = old_pipeline_dir / "predictions.csv"
        if old_predictions.exists():
            shutil.copy(old_predictions, pipeline_ws.predictions_file)

        # Migrate metrics (if exist)
        old_metrics = old_pipeline_dir / "metrics.json"
        if old_metrics.exists():
            shutil.copy(old_metrics, pipeline_ws.metrics_file)

        logger.info(f"✓ Migrated {old_pipeline_dir.name} → {pipeline_ws.pipeline_dir}")

    def _convert_manifest_to_config(self, manifest: Dict) -> Dict:
        """Convert old manifest.yaml to new pipeline.json format."""
        return {
            "id": manifest["pipeline_uid"],
            "name": manifest.get("pipeline_name", "unknown"),
            "created_at": manifest.get("created_at", datetime.now().isoformat()),
            "status": "migrated",
            "steps": manifest.get("steps", []),
            "dataset": manifest.get("dataset", "unknown"),
            "migrated_from": "results/"
        }
```

#### 5.2 Create Migration CLI (`nirs4all/cli/migrate_command.py`)

```bash
# CLI command
nirs4all migrate \
  --from results/ \
  --to workspace/ \
  --session migrated_results \
  --dry-run  # Preview what will be migrated

# Output:
# Analyzing results/ structure...
# Found 42 pipelines across 5 datasets
#
# Migration plan:
#   - 42 pipelines → workspace/runs/2024-10-23_migrated_results/
#   - 5 datasets will be created
#   - 157 artifacts will be migrated (3.2 GB)
#
# Proceed with migration? [y/N]: y
#
# Migrating...
# ✓ Pipeline abc123 → wheat_sample1
# ✓ Pipeline def456 → wheat_sample2
# ...
#
# Migration complete:
#   - 40 pipelines migrated successfully
#   - 2 pipelines failed (see migration_errors.log)
#   - Total duration: 2m 15s
```

### Deliverables

- [ ] `MigrationManager` for automatic migration
- [ ] CLI command: `nirs4all migrate`
- [ ] Dry-run mode for preview
- [ ] Error handling and reporting
- [ ] Migration validation

### Testing

```bash
# Migration tests
pytest tests/migration/test_migration_manager.py
pytest tests/migration/test_manifest_conversion.py
pytest tests/migration/test_artifact_migration.py

# End-to-end migration test
pytest tests/migration/test_full_migration.py
```

### Success Criteria

✅ Can migrate all existing pipelines from `results/`
✅ Artifacts correctly migrated and deduplicated
✅ Metrics and predictions preserved
✅ Migration report generated
✅ Failed migrations logged for manual review
✅ Migration tests pass

---

## Phase 6: CLI & Documentation (Week 6)

### Goal
Update CLI, create comprehensive documentation, and finalize user-facing interfaces.

### Tasks

#### 6.1 Update CLI Commands (`nirs4all/cli/`)

```bash
# New CLI structure

# Session management
nirs4all run \
  --session wheat-quality-study \
  --datasets data/wheat_sample1.csv data/wheat_sample2.csv \
  --pipelines configs/baseline_pls.json configs/optimized_svm.json \
  --description "Testing quality prediction models"

nirs4all sessions list
nirs4all sessions show 2024-10-23_wheat-quality-study
nirs4all sessions delete 2024-09-15_old-experiments

# Library management
nirs4all library save \
  --from runs/2024-10-23_wheat-quality-study/datasets/wheat_sample1/pipelines/baseline_pls_a1b2c3 \
  --name wheat_quality_baseline \
  --include-binaries

nirs4all library list
nirs4all library load wheat_quality_baseline

# Catalog queries
nirs4all catalog list wheat_sample1
nirs4all catalog best wheat_sample1 --metric test_rmse
nirs4all catalog search --pipeline baseline_pls --rmse-max 0.5

# Export
nirs4all export \
  --session 2024-10-23_wheat-quality-study \
  --dataset wheat_sample1 \
  --pipeline baseline_pls_a1b2c3 \
  --output best_model.zip
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

- [ ] Complete CLI implementation for workspace operations
- [ ] User guide (`WORKSPACE_GUIDE.md`)
- [ ] Developer guide (`WORKSPACE_ARCHITECTURE_DEV.md`)
- [ ] Updated examples (Q14, Q15, etc.)
- [ ] API reference documentation
- [ ] Migration guide

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
│   ├── test_artifact_manager.py          # 12 tests
│   ├── test_catalog_manager.py           # 18 tests
│   └── test_library_manager.py           # 15 tests
├── execution/
│   ├── test_session_executor.py          # 10 tests
│   ├── test_progress_tracker.py          # 8 tests
│   └── test_error_handler.py             # 10 tests
├── migration/
│   ├── test_migration_manager.py         # 12 tests
│   ├── test_manifest_conversion.py       # 8 tests
│   └── test_artifact_migration.py        # 10 tests
├── integration/
│   ├── test_workspace_execution.py       # 5 tests
│   ├── test_multi_dataset.py             # 5 tests
│   ├── test_library_workflow.py          # 5 tests
│   └── test_catalog_workflow.py          # 5 tests
└── cli/
    ├── test_workspace_commands.py        # 15 tests
    ├── test_library_commands.py          # 10 tests
    └── test_catalog_commands.py          # 10 tests

Total: ~200 tests
```

### Integration Tests (End of Each Phase)

Phase-specific integration tests ensure components work together:

- **Phase 1**: Create full workspace structure programmatically
- **Phase 2**: Execute pipeline end-to-end with new structure
- **Phase 3**: Run multi-dataset session
- **Phase 4**: Library save/load workflow
- **Phase 5**: Migrate and validate existing data
- **Phase 6**: CLI commands produce expected results

### Manual Testing Checklist

After each phase:

- [ ] Create workspace structure manually (verify file paths)
- [ ] Run example pipelines (verify results)
- [ ] Check generated JSON files (verify format)
- [ ] Test error conditions (verify error handling)
- [ ] Review logs (verify logging quality)
- [ ] Test cleanup (verify safe deletion)

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

## Backward Compatibility

### Phase 1-2: Parallel Systems

Both `results/` and `workspace/` coexist:

```python
# Legacy mode (default for now)
runner = PipelineRunner(use_legacy=True)
results = runner.run(...)  # Uses results/ structure

# New mode (opt-in)
runner = PipelineRunner(use_legacy=False)
results = runner.run(...)  # Uses workspace/ structure
```

### Phase 3-5: Transition Period

- New runs use `workspace/` by default
- `results/` still supported but deprecated warning
- Migration tool available

### Phase 6+: Full Migration

- `workspace/` is the only structure
- `results/` support removed (breaking change in v0.7.0)
- Clear migration guide provided

---

## Rollout Strategy

### Week 1-2: Internal Development

- Phases 1-2 completed
- Internal testing only
- No public release

### Week 3-4: Alpha Release

- Phases 3-4 completed
- Release as `0.6.0-alpha`
- Invite early adopters
- Gather feedback

### Week 5: Beta Release

- Phase 5 completed
- Release as `0.6.0-beta`
- Migration tool available
- Broader testing

### Week 6: Stable Release

- Phase 6 completed
- Release as `0.6.0`
- Full documentation
- Announcement

---

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Migration data loss | HIGH | Thorough testing, dry-run mode, backups |
| Breaking API changes | HIGH | Deprecation warnings, parallel systems |
| Performance regression | MEDIUM | Benchmarking, profiling |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| User confusion | MEDIUM | Clear documentation, examples, migration guide |
| Incomplete migration | MEDIUM | Validation checks, detailed error messages |
| Catalog scaling | MEDIUM | SQLite backend option for large deployments |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| CLI command conflicts | LOW | Careful command design, user feedback |
| Library format changes | LOW | Version metadata in zip files |

---

## Success Metrics

### Technical Metrics

- [ ] All 200+ tests passing
- [ ] Storage reduction: 30-50% (grid search experiments)
- [ ] No performance regression (<5% overhead)
- [ ] Migration success rate: >95%

### User Experience Metrics

- [ ] Session creation: <10 seconds for typical setup
- [ ] Catalog queries: <1 second for 1000 predictions
- [ ] CLI commands: Intuitive, self-documenting
- [ ] Documentation: Complete, with examples

### Adoption Metrics

- [ ] 10+ early adopters test alpha
- [ ] No critical bugs in beta
- [ ] Positive user feedback
- [ ] Migration completed by >80% of users within 1 month

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
