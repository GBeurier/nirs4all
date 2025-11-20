# External API (for webapps controlling nirs4all)

Purpose
-------
This document collects the external signatures a web application would call to manage nirs4all: load datasets, inspect samples/features/metadata, create/load pipelines, run pipelines, predict, retrieve and browse predictions, and launch charts. Refer to the code and examples in the repository for concrete implementations: [nirs4all/__init__.py](nirs4all/__init__.py) and [examples](examples). This file is [docs/External_api.MD](docs/External_api.MD).

Guiding principles
------------------
- Keep all operations non-blocking on the webserver (run long tasks in background workers).
- Use file paths or in-memory blobs for dataset/pipeline transfer.
- Return lightweight metadata for listing and detailed payloads only on demand.

Core concepts (types)
---------------------
- Dataset — container with samples, features, and metadata.
- Pipeline — processing pipeline (preprocessing + model/estimator).
- PipelineRun / RunResult — result of executing a pipeline on a dataset.
- Predictions — structured predictions (rows indexed by sample, columns per target).
- ChartSpec — description of a visualization request.

Suggested external signatures
-----------------------------
Below are suggested function signatures and brief descriptions. Map these to the actual functions/classes in your workspace (see [nirs4all/__init__.py](nirs4all/__init__.py) and [examples](examples)).

1) Dataset management
- load_dataset(path: str) -> Dataset
  - Load dataset from disk (supported formats: raw files, CSVs, nirs4all archive). Returns a Dataset object or descriptor.
- save_dataset(dataset: Dataset, path: str) -> None
  - Save dataset to disk.
- list_datasets() -> List[str]
  - Return available dataset identifiers/paths.
- get_dataset_info(dataset: Dataset) -> dict
  - Return basic metadata: number of samples, features, channels, sampling rate, subject info.
- get_samples(dataset: Dataset, start: int = 0, count: int = 100) -> DataFrame
  - Return a paginated view of samples (rows).
- get_features(dataset: Dataset) -> List[str]
  - Return list of feature/channel names.

Usage example (webapp):
- Call get_dataset_info for dataset listing pages.
- Use get_samples to populate table views with pagination.

2) Pipeline creation / loading / saving
- create_pipeline(spec: dict) -> Pipeline
  - Build a pipeline from a JSON spec (sequence of steps, parameters).
- load_pipeline(path: str) -> Pipeline
  - Load an existing pipeline definition from disk.
- save_pipeline(pipeline: Pipeline, path: str) -> None
- list_pipelines() -> List[str]

3) Running pipelines (long-running)
- start_pipeline_run(pipeline: Pipeline, dataset: Dataset, run_id: Optional[str] = None, options: dict = {}) -> str
  - Start a background run. Returns run_id immediately; run executes asynchronously.
- get_run_status(run_id: str) -> dict
  - Status: queued/running/succeeded/failed, progress, logs summary.
- fetch_run_logs(run_id: str, offset: int = 0) -> str
  - Tail-like access to logs.
- get_run_result(run_id: str) -> PipelineRun
  - Final run result object (when completed).

Note: Implement the runner with a worker system (Celery, RQ, or local threads) so the webapp can poll status.

4) Predict / inference
- predict(pipeline: Pipeline, dataset: Dataset, options: dict = {}) -> Predictions
  - Run prediction (can be synchronous for small data or via start_pipeline_run for large).
- list_predictions(dataset_id: str | None = None) -> List[str]
  - List stored prediction objects (ids).
- get_predictions(pred_id: str, start: int = 0, count: int = 100) -> DataFrame
  - Paginated access to predictions.

5) Browsing & visualization
- list_available_charts() -> List[ChartSpec]
  - Return supported chart types (timeseries, topographic, histogram, scatter).
- generate_chart(spec: ChartSpec) -> ChartBlob
  - Generate chart assets (SVG/PNG/JSON) for immediate display.
- stream_chart_data(pred_id: str, chart_spec: ChartSpec, downsample: int = 1) -> Generator
  - Stream chart-able data for client-side libraries (e.g., plotly).

6) Utility & metadata
- export_result(run_id: str, format: str = "csv") -> str
  - Export predictions or processed data; return export path.
- compute_summary_statistics(dataset: Dataset | Predictions) -> dict
  - Mean, std, ranges per channel/target.

Integration notes for webapps
-----------------------------
- Authentication/authorization: expose file access and run control via authenticated endpoints.
- Use lightweight metadata endpoints (get_dataset_info, list_pipelines) to populate UI lists without loading full data.
- For large arrays, prefer streaming or returning URLs to stored files rather than embedding raw arrays in JSON.
- Visualizations: return either rendered images or JSON series suitable for client-side plotting.
- Error handling: standardize error objects with code, message, and optional traceback for logs endpoint.

Mapping to repository
---------------------
- Examples showing common workflows are under [examples](examples).
- Core package entry point is at [nirs4all/__init__.py](nirs4all/__init__.py).
- Use the repository README for installation and runtime notes: [README.md](README.md).

Quick checklist for implementing a web UI
-----------------------------------------
- Dataset browser: list_datasets + get_dataset_info + get_samples
- Pipeline editor: list_pipelines + create_pipeline + save_pipeline + load_pipeline
- Execution panel: start_pipeline_run + get_run_status + fetch_run_logs + get_run_result
- Predictions explorer: list_predictions + get_predictions + export_result
- Visualization: list_available_charts + generate_chart OR stream_chart_data

If you prefer, I can generate a concrete mapping from these signatures to functions present in the codebase (examples) and produce ready-to-use Flask/FastAPI endpoint stubs. Refer to the example scripts in [examples](examples) to see how the library is currently used.