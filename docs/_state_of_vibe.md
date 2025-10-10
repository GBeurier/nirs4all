# nirs4all — Web UI: current state, goals, and roadmap

This document summarizes the work done so far to build a local web UI for nirs4all, the decisions taken, the current implementation status, and an actionable roadmap for the next steps. It is intended as a hand-off guide that lets another developer or an AI pick up where we left off.

## High-level objective

Enable a local-first web UI for nirs4all that lets users:
- Manage/select a workspace and workspace-scoped dataset configs
- Browse prediction results (Predictions DB) with server-side search, filters and pagination
- Inspect and re-open pipelines used to produce predictions (pipeline editor)
- Launch, edit and save pipeline configurations locally

The UI is intentionally file-system centric (server has read/write access to local files) and is meant to be run locally (CLI: `nirs4all --ui`).

## What is implemented (status)

- FastAPI-based minimal UI server: `nirs4all/ui/server.py` — serves API endpoints and static files from `webapp/`.
- CLI integration: `nirs4all --ui` starts the local web UI (already wired in the CLI scaffolding).
- Predictions management backend using the existing Predictions API: `nirs4all/dataset/predictions.py` (loads `results/<dataset>/predictions.json`).
- Server endpoints (non-exhaustive):
	- `/api/predictions/datasets` — lists dataset folders and counts
	- `/api/predictions/list` — returns all predictions for a dataset (serialized)
	- `/api/predictions/search` — paginated, sortable, filterable search endpoint
	- `/api/predictions/meta` — models/configs/partitions metadata
	- `/api/pipeline/from-prediction` — fetch pipeline.json associated with a prediction ID
	- `/api/workspace/*` — workspace selection, dataset linking, pipeline repo, results folder and dataset config save/load/delete
	- `/api/files/*` — server-side file browser helpers
- Frontend (static SPA under `webapp/`):
	- `predictions.html` — dynamic Predictions table, filters and actions (Open Pipeline navigates to pipeline editor and passes prediction id)
	- `pipeline.html` — pipeline editor canvas, "Load from Prediction" modal, and `renderPipelineFromJSON(pipeline)` which renders steps as cards
	- `workspace.html` — workspace and dataset config management UI (file browser/modal flows implemented)

## Notable design choices / constraints

- Local-first: UI uses server-side filesystem access (file browser, saving dataset configs, reading `results/`) instead of a remote storage API. This is fast and simple for local workflows but unsuitable for untrusted remote hosting.
- Server-side search & pagination for the Predictions DB: avoids transferring very large JSON payloads to the browser and scales to many predictions.
- Client-side hiding of heavy payload columns: `y_true`, `y_pred`, and `sample_indices` are excluded from table rendering (they are still returned server-side unless explicitly removed).
- Pipeline retrieval: predictions must include a `config_path` (or similar) that points to the run folder. The server resolves that path relative to the workspace results folder and loads `pipeline.json`.

## Key files & functions to inspect

- Backend
	- `nirs4all/ui/server.py` — main API server implementation and workspace persistence
		- helpers: `_resolve_results_path()`, `_to_json_basic()`, `_apply_filters_to_rows()`
	- `nirs4all/dataset/predictions.py` — Predictions loader, filter and utilities (authoritative source for predictions format)
- Frontend
	- `webapp/predictions.html` — client code that calls `/api/predictions/search`, renders table and issues navigation to pipeline editor
	- `webapp/pipeline.html` — `renderPipelineFromJSON()` and modal logic; auto-load logic reads ?prediction_id and fetches `/api/pipeline/from-prediction`
	- `webapp/workspace.html` — file-browser & dataset config modal (UI state persisted in workspace config)

## How to run and debug locally

1. Activate the virtualenv used for the project: `& .\.venv\Scripts\Activate.ps1` (PowerShell on Windows).
2. Start the web UI: `nirs4all --ui` (this launches the FastAPI app and serves `webapp/` under `/webapp`).
3. Open a browser at the root (`/`) — the server redirects to `/webapp/index.html`.
4. Use the Predictions page and Pipeline editor; open DevTools to monitor network requests, console logs and inspect any fetch failures. Look specifically for requests to `/api/pipeline/from-prediction` when opening the pipeline editor from a prediction.

## Current development state (what's done vs pending)

Done:
- Core UI server, SPA skeleton, workspace management, dataset config save/load/delete
- Predictions DB backend with server-side pagination, search, sorting, and meta endpoint
- Pipeline-from-prediction endpoint and frontend flow (modal + auto-load using `?prediction_id=`)

In-progress / partially done:
- Server still returns heavy array columns in prediction rows; the UI hides them but they remain in API responses (optimize server-side to reduce payload size).
- Several backend functions contain broad except blocks and have lint warnings for being too complex — refactor recommended.

Missing / recommended next work (prioritized)

1. Critical (quick win)
	 - Ensure end-to-end visible failure messages when pipeline auto-load fails (display a toast/modal on error instead of console warnings) — file: `webapp/pipeline.html`.
	 - Server-side filtering to avoid returning heavy arrays at all (strip `y_true`, `y_pred`, `sample_indices` from list/search endpoints). Modify `_to_json_basic()` or strip keys before returning rows in `predictions_search`/`predictions_list`.

2. High value
	 - Add server-side delete/download endpoints for predictions and wire them to the UI actions (the buttons exist but actions are placeholders).
	 - Add unit/integration tests for `/api/pipeline/from-prediction` and `/api/predictions/search` (mock a small `results/` layout and `predictions.json`).
	 - Add E2E or integration test that boots `nirs4all --ui` and asserts that predictions → pipeline editor flow works.

3. Medium term
	 - Refactor `nirs4all/ui/server.py` to reduce complexity (split large functions such as `save_dataset_config`, `workspace_datasets`, and `predictions_list` into smaller helpers). Remove broad `except Exception:` catches and add explicit error logging.
	 - Add authentication / access control if the server may be exposed beyond a single-user local machine in future.

4. UX polish
	 - Add toasts/alerts for success and error states (e.g., pipeline auto-load failure, dataset save success).
	 - Provide a visual indicator when arrays are truncated in parameter previews (renderPipelineFromJSON already truncates but add explicit "truncated" flag).
	 - Add a detail view / report page for a prediction that can show plots and allow re-run/replay.

## Data model & conventions

- Predictions live under workspace results (default `results/` under the workspace). Each dataset folder may contain `predictions.json`.
- A prediction row is a dict that may include keys such as: `id`, `dataset_name`, `dataset_path`, `config_name`, `config_path`, `model_name`, `model_classname`, `metric`, `test_score`, `y_true`, `y_pred`, `sample_indices`, `metadata`, etc.
- The pipeline runner writes a `pipeline.json` in the same folder as the config used for the run. `config_path` in the prediction record should point to that folder; the server resolves relative paths using the workspace `results_folder`.

## Troubleshooting notes

- If pipeline editor shows nothing after navigation:
	1. Open DevTools → Network and verify `/api/pipeline/from-prediction?prediction_id=<id>` was requested.
 2. If request returns 404: check that prediction record has a `config_path` and that `pipeline.json` exists in the resolved folder.
 3. If request returns 500: check server logs for JSON parse/IO errors.

- To inspect predictions file format, open: `results/<dataset>/predictions.json` (small sample records are useful for testing).

## Immediate next steps I can take for you

- Implement server-side removal/truncation of heavy array fields in search/list endpoints to reduce payload sizes.
- Add visible UI error messages for pipeline auto-load failures and a small retry button.
- Create unit tests for the pipeline-from-prediction flow.
- Refactor and reduce complexity in `nirs4all/ui/server.py` (break into smaller helpers and narrow exceptions).

If you want me to continue, tell me which of the immediate next steps you want prioritized (server-side trimming, visible UI errors, tests, or refactor) and I will implement it next.

---
Last updated: 2025-10-10

