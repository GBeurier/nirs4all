"""
Workspace management CLI commands for nirs4all.

Provides commands for workspace initialization, run management, and
library operations.
"""

import json
import sys
from pathlib import Path

import numpy as np

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


def _validate_workspace_exists(workspace_path: Path) -> None:
    """Validate that a workspace path exists, exit with code 1 if not."""
    if not workspace_path.exists():
        logger.error(f"Workspace path does not exist: {workspace_path}")
        sys.exit(1)


def workspace_init(args):
    """Initialize a new workspace."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.path)

    # Validate parent directory exists and path is not a file
    if not workspace_path.parent.exists():
        logger.error(f"Parent directory does not exist: {workspace_path.parent}")
        sys.exit(1)
    if workspace_path.exists() and workspace_path.is_file():
        logger.error(f"Path exists and is a file, not a directory: {workspace_path}")
        sys.exit(1)

    # WorkspaceStore creates the SQLite database and workspace directories
    store = WorkspaceStore(workspace_path)
    store.close()

    # Also create standard directories
    (workspace_path / "exports").mkdir(parents=True, exist_ok=True)
    (workspace_path / "library").mkdir(parents=True, exist_ok=True)

    logger.success(f"Workspace initialized at: {workspace_path}")
    logger.info("  Created:")
    logger.info("    - store.sqlite (workspace database)")
    logger.info("    - exports/")
    logger.info("    - library/")


def workspace_list_runs(args):
    """List all runs in workspace."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        runs = store.list_runs()

    if runs.height == 0:
        logger.info("No runs found in workspace.")
        return

    logger.info(f"Found {runs.height} run(s):\n")
    for row in runs.to_dicts():
        logger.info(f"  {row.get('name', 'unknown')}")
        logger.info(f"    Status: {row.get('status', 'unknown')}")
        logger.info(f"    Created: {row.get('created_at', 'unknown')}")
        logger.info("")


def workspace_query_best(args):
    """Query best predictions from workspace store."""
    from nirs4all.core.metrics import infer_ascending
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)

    try:
        ascending = True if args.ascending else infer_ascending(args.metric)
        with WorkspaceStore(workspace_path) as store:
            top_df = store.top_predictions(
                n=args.n,
                dataset_name=args.dataset,
                metric=args.metric,
                ascending=ascending,
            )
        if top_df.height == 0:
            logger.info("No predictions found matching criteria.")
            return

        logger.info(f"Top {args.n} predictions by {args.metric}:")
        logger.info(f"{'=' * 80}\n")

        df = top_df.to_pandas()
        logger.info(df.to_string(index=False))
    except Exception as e:
        logger.error(f"Error querying predictions: {e}")
        sys.exit(1)


def workspace_query_filter(args):
    """Filter predictions by criteria."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)

    try:
        with WorkspaceStore(workspace_path) as store:
            filtered = store.query_predictions(
                dataset_name=args.dataset,
            )

        logger.info(f"Found {filtered.height} predictions matching criteria\n")

        if filtered.height > 0:
            df = filtered.to_pandas()
            logger.info(df.to_string(index=False))
    except Exception as e:
        logger.error(f"Error filtering predictions: {e}")
        sys.exit(1)


def workspace_stats(args):
    """Show workspace statistics."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)

    logger.info("Workspace Statistics")
    logger.info(f"{'=' * 60}\n")

    try:
        with WorkspaceStore(workspace_path) as store:
            all_preds = store.query_predictions()
            runs = store.list_runs()

        logger.info(f"Total predictions: {all_preds.height}")

        if all_preds.height > 0 and "dataset_name" in all_preds.columns:
            datasets = all_preds["dataset_name"].unique().to_list()
            logger.info(f"Datasets: {len(datasets)}")
            for ds in datasets:
                count = all_preds.filter(all_preds["dataset_name"] == ds).height
                logger.info(f"  - {ds}: {count} predictions")

        logger.info(f"\nRuns: {runs.height}")
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        sys.exit(1)


def workspace_list_library(args):
    """List items in library."""
    from nirs4all.pipeline.storage.library import PipelineLibrary

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    library = PipelineLibrary(workspace_path)

    templates = library.list_templates()
    logger.info(f"Templates: {len(templates)}")
    for t in templates:
        logger.info(f"  - {t['name']}: {t.get('description', 'No description')}")


def workspace_inspect(args):
    """Inspect workspace format without opening it writable."""
    from nirs4all.workspace.compat import inspect_workspace_format

    info = inspect_workspace_format(Path(args.path))
    logger.info(f"Path: {info.path}")
    logger.info(f"Format: {info.format}")
    logger.info(f"Conversion required: {info.conversion_required}")
    logger.info(info.message)
    if info.conversion_command:
        logger.info(f"Conversion command: {info.conversion_command}")


def workspace_convert(args):
    """Convert a legacy workspace by delegating to nirs4all-tools."""
    try:
        from nirs4all_tools.cli import main as tools_main
    except Exception as exc:
        logger.error("nirs4all-tools is required for conversion. Install it with: pip install nirs4all[transition] or pip install nirs4all-tools")
        raise SystemExit(1) from exc

    argv = [
        "legacy",
        "migrate",
        str(args.input),
        "--output",
        str(args.output),
        "--target",
        "nirs4all-workspace-v2",
    ]
    if args.verify:
        argv.append("--verify")
    if args.dry_run:
        argv.append("--dry-run")
    if args.strict:
        argv.append("--strict")
    raise SystemExit(tools_main(argv))


def _json_default(value):
    """JSON fallback for timestamps and Path-like values."""

    return str(value)


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _parse_csv_floats(value: str, *, name: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one comma-separated value")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"{name} must contain only numeric comma-separated values") from exc


def _parse_csv_strings(value: str, *, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one comma-separated value")
    return items


def _parse_json_cli_value(value: str, *, name: str):
    """Parse a JSON CLI payload with a clear option-specific error."""

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc


def _json_safe_value(value):
    """Return a JSON-native representation for NumPy/path-like diagnostic values."""

    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _predict_result_payload(result) -> dict:
    intervals = {}
    for coverage in result.interval_coverages:
        interval = result.interval(coverage)
        intervals[str(coverage)] = {
            "lower": np.asarray(interval.lower).tolist(),
            "upper": np.asarray(interval.upper).tolist(),
        }
    return {
        "y_pred": np.asarray(result.y_pred).tolist(),
        "sample_ids": [] if result.sample_indices is None else np.asarray(result.sample_indices, dtype=object).tolist(),
        "model_name": getattr(result, "model_name", ""),
        "intervals": intervals,
        "calibrated_result_fingerprint": result.metadata.get("calibrated_result_fingerprint") if isinstance(result.metadata, dict) else None,
        "conformal_guarantee_status": result.conformal_guarantee_status,
        "calibration_replay_source": result.calibration_replay_source,
        "tuning_calibration_source": result.tuning_calibration_source,
    }


def workspace_tuning_list(args):
    """List native tuning results persisted in a workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        rows = store.list_tuning_results(limit=args.limit, offset=args.offset).to_dicts()

    if args.json:
        _print_json(rows)
        return
    if not rows:
        logger.info("No tuning results found in workspace.")
        return
    logger.info(f"Found {len(rows)} tuning result(s):\n")
    for row in rows:
        logger.info(f"  {row.get('tuning_id')}")
        logger.info(f"    Name: {row.get('name') or '-'}")
        logger.info(f"    Engine: {row.get('engine')}  Metric: {row.get('metric')}  Direction: {row.get('direction')}")
        logger.info(f"    Best value: {row.get('best_value')}  Trials: {row.get('n_trials')}")
        logger.info(f"    Result fingerprint: {row.get('result_fingerprint')}")
        logger.info("")


def workspace_tuning_show(args):
    """Show one native tuning result from a workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        result = store.load_tuning_result(args.id)
    payload = result.to_dict()
    if args.json:
        _print_json(payload)
        return
    logger.info(f"Tuning result: {args.id}")
    logger.info(f"  Optimizer: {result.optimizer}")
    logger.info(f"  Engine: {result.tuning.engine}")
    logger.info(f"  Metric: {result.tuning.metric}")
    logger.info(f"  Direction: {result.tuning.direction}")
    logger.info(f"  Best value: {result.best_value}")
    logger.info(f"  Best params: {dict(result.best_params)}")
    logger.info(f"  Trials: {result.n_trials}")
    logger.info(f"  Result fingerprint: {result.fingerprint}")
    logger.info(f"  Tuning fingerprint: {result.tuning.fingerprint}")


def workspace_tuning_export(args):
    """Export one persisted native tuning result from a workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        result = store.load_tuning_result(args.id)

    output_format = str(args.format)
    if output_format == "json":
        payload = result.to_json(indent=None if args.compact else args.indent)
    else:
        payload = result.to_summary_json(indent=None if args.compact else args.indent)

    if args.output is None:
        print(payload, end="")
        return
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")


def workspace_conformal_list(args):
    """List conformal calibration results persisted in a workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        rows = store.list_conformal_results(limit=args.limit, offset=args.offset).to_dicts()

    if args.json:
        _print_json(rows)
        return
    if not rows:
        logger.info("No conformal results found in workspace.")
        return
    logger.info(f"Found {len(rows)} conformal result(s):\n")
    for row in rows:
        logger.info(f"  {row.get('conformal_id')}")
        logger.info(f"    Name: {row.get('name') or '-'}")
        logger.info(f"    Target: {row.get('target_name') or '-'}  Coverages: {row.get('coverages')}")
        logger.info(f"    Result fingerprint: {row.get('result_fingerprint')}")
        logger.info(f"    Artifact fingerprint: {row.get('artifact_fingerprint')}")
        logger.info("")


def workspace_conformal_show(args):
    """Show one conformal calibration result from a workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        result = store.load_conformal_result(args.id)
    if args.as_predict_result:
        prediction = result.to_predict_result()
        payload = _predict_result_payload(prediction)
        if args.json:
            _print_json(payload)
            return
        logger.info(f"Conformal prediction result: {args.id}")
        logger.info(f"  Target: {prediction.model_name or '-'}")
        logger.info(f"  Coverages: {list(prediction.interval_coverages)}")
        logger.info(f"  Samples: {len(prediction.y_pred)}")
        logger.info(f"  Result fingerprint: {payload.get('calibrated_result_fingerprint')}")
        if prediction.conformal_guarantee_status is not None:
            logger.info(f"  Guarantee: {prediction.conformal_guarantee_status.get('status')}")
        if prediction.tuning_calibration_source is not None:
            logger.info(f"  Tuning calibration source: {prediction.tuning_calibration_source.get('source')}")
        return
    payload = result.to_dict()
    if args.json:
        _print_json(payload)
        return
    logger.info(f"Conformal result: {args.id}")
    logger.info(f"  Target: {result.artifact.target_name or '-'}")
    logger.info(f"  Coverages: {list(result.prediction.coverages)}")
    logger.info(f"  Samples: {len(result.sample_ids)}")
    logger.info(f"  Result fingerprint: {result.fingerprint}")
    logger.info(f"  Artifact fingerprint: {result.artifact.fingerprint}")
    if result.conformal_guarantee_status is not None:
        logger.info(f"  Guarantee: {result.conformal_guarantee_status.get('status')}")
    if result.tuning_calibration_source is not None:
        logger.info(f"  Tuning calibration source: {result.tuning_calibration_source.get('source')}")


def workspace_conformal_predict(args):
    """Apply a persisted conformal calibrator to point predictions."""

    import nirs4all

    y_pred = _parse_csv_floats(args.y_pred, name="--y-pred")
    sample_ids = _parse_csv_strings(args.sample_ids, name="--sample-ids")
    if len(y_pred) != len(sample_ids):
        raise ValueError("--y-pred and --sample-ids must contain the same number of values")

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    calibrated = nirs4all.load_workspace_calibrated_result(workspace_path, args.id)
    result = nirs4all.predict_calibrated(
        calibrated,
        y_pred=y_pred,
        prediction_sample_ids=sample_ids,
    )
    payload = _predict_result_payload(result)
    if args.json:
        _print_json(payload)
        return
    logger.info(f"Applied conformal result: {args.id}")
    logger.info(f"  Samples: {len(sample_ids)}")
    logger.info(f"  Coverages: {list(result.interval_coverages)}")
    if result.conformal_guarantee_status is not None:
        logger.info(f"  Guarantee: {result.conformal_guarantee_status.get('status')}")
    for coverage, interval in payload["intervals"].items():
        logger.info(f"  Coverage {coverage}: lower={interval['lower']} upper={interval['upper']}")


def workspace_robustness_list(args):
    """List robustness reports persisted in a workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        rows = store.list_robustness_results(limit=args.limit, offset=args.offset).to_dicts()

    if args.json:
        _print_json(rows)
        return
    if not rows:
        logger.info("No robustness reports found in workspace.")
        return
    logger.info(f"Found {len(rows)} robustness report(s):\n")
    for row in rows:
        logger.info(f"  {row.get('robustness_id')}")
        logger.info(f"    Name: {row.get('name') or '-'}")
        logger.info(f"    Mode: {row.get('mode')}  Scenarios: {row.get('scenario_count')}  Slice keys: {row.get('slice_by')}")
        logger.info(f"    Result fingerprint: {row.get('result_fingerprint')}")
        logger.info("")


def workspace_robustness_show(args):
    """Show one robustness report from a workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        report = store.load_robustness_result(args.id)
    payload = report.to_dict()
    if args.json:
        _print_json(payload)
        return
    logger.info(f"Robustness report: {args.id}")
    logger.info(f"  Mode: {report.mode}")
    logger.info(f"  Scenarios: {len(report.scenarios)}")
    logger.info(f"  Slice keys: {list(report.slice_by)}")
    logger.info(f"  Audit-only: {report.metadata.get('audit_only')}")
    logger.info(f"  Result fingerprint: {report.fingerprint}")


def workspace_robustness_export(args):
    """Export one persisted robustness report from a workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        report = store.load_robustness_result(args.id)

    output_format = str(args.format)
    if output_format == "json":
        payload = report.to_json(indent=None if args.compact else args.indent)
        if args.output is None:
            print(payload, end="")
            return
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        return

    if output_format == "summary":
        payload = report.to_summary_json(indent=None if args.compact else args.indent)
        if args.output is None:
            print(payload, end="")
            return
        report.save_summary(args.output)
        return

    if output_format == "markdown":
        payload = report.to_markdown()
        if args.output is None:
            print(payload, end="")
            return
        report.save_markdown(args.output)
        return

    if output_format == "html":
        payload = report.to_html()
        if args.output is None:
            print(payload, end="")
            return
        report.save_html(args.output)
        return

    if output_format == "artifacts":
        if args.output is None:
            raise ValueError("--output is required when --format=artifacts")
        report.save_artifacts(args.output)
        return

    if args.output is None:
        raise ValueError("--output is required when --format=parquet")
    report.save_parquet(args.output)


def _write_robustness_report_cli_output(report, args) -> None:
    """Write a robustness report in the CLI's shared export formats."""

    output_format = str(args.format)
    if output_format == "json":
        payload = report.to_json(indent=None if args.compact else args.indent)
        if args.output is None:
            print(payload, end="")
            return
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        return

    if output_format == "summary":
        payload = report.to_summary_json(indent=None if args.compact else args.indent)
        if args.output is None:
            print(payload, end="")
            return
        report.save_summary(args.output)
        return

    if output_format == "markdown":
        payload = report.to_markdown()
        if args.output is None:
            print(payload, end="")
            return
        report.save_markdown(args.output)
        return

    if output_format == "html":
        payload = report.to_html()
        if args.output is None:
            print(payload, end="")
            return
        report.save_html(args.output)
        return

    if output_format == "artifacts":
        if args.output is None:
            raise ValueError("--output is required when --format=artifacts")
        report.save_artifacts(args.output)
        return

    if args.output is None:
        raise ValueError("--output is required when --format=parquet")
    report.save_parquet(args.output)


def workspace_robustness_from_prediction(args):
    """Compute a robustness report from one persisted workspace prediction."""

    import nirs4all

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    y_true = _parse_csv_floats(args.y_true, name="--y-true") if args.y_true is not None else _parse_json_cli_value(args.y_true_json, name="--y-true-json")
    scenarios = None
    if args.scenarios_json is not None:
        scenarios = _parse_json_cli_value(args.scenarios_json, name="--scenarios-json")
        if not isinstance(scenarios, list):
            raise ValueError("--scenarios-json must decode to a JSON array")
    metadata = None
    if args.metadata_json is not None:
        metadata = _parse_json_cli_value(args.metadata_json, name="--metadata-json")
        if not isinstance(metadata, (dict, list)):
            raise ValueError("--metadata-json must decode to a JSON object or array")
    workspace_metadata = None
    if args.workspace_metadata_json is not None:
        workspace_metadata = _parse_json_cli_value(args.workspace_metadata_json, name="--workspace-metadata-json")
        if not isinstance(workspace_metadata, dict):
            raise ValueError("--workspace-metadata-json must decode to a JSON object")
    slice_by = None if args.slice_by is None else _parse_csv_strings(args.slice_by, name="--slice-by")

    report = nirs4all.robustness_from_workspace_prediction(
        workspace_path,
        args.prediction_id,
        y_true=y_true,
        scenarios=scenarios,
        slice_by=slice_by,
        metadata=metadata,
        seed=args.seed,
        save_to_workspace=args.save_to_workspace,
        workspace_name=args.workspace_name,
        workspace_robustness_id=args.workspace_robustness_id,
        workspace_metadata=workspace_metadata,
    )
    _write_robustness_report_cli_output(report, args)


def _prediction_robustness_evidence_payload(record, *, error: Exception | None = None) -> dict:
    """Build a JSON-safe spectral/OOD replay evidence diagnostic for one prediction row."""

    prediction_id = str(record.get("id") or record.get("prediction_id") or "")
    payload = {
        "prediction_id": prediction_id,
        "dataset_name": record.get("dataset_name") or "",
        "model_name": record.get("model_name") or "",
        "partition": record.get("partition") or "",
        "ready": False,
        "spectral_replay_evidence_status": {
            "status": "conversion_failed",
            "missing": ["load_arrays_true_y_pred"],
            "source": "prediction_record",
        },
    }
    if error is not None:
        payload["error"] = str(error)
        return payload

    from nirs4all.api.result import PredictResult

    result = PredictResult.from_prediction_record(record)
    status = _json_safe_value(result.spectral_replay_evidence_status)
    payload["ready"] = status.get("status") == "ready_for_spectral_replay"
    payload["spectral_replay_evidence_status"] = status
    evidence = result.robustness_evidence or {}
    payload["has_robustness_evidence_metadata"] = bool(evidence)
    if result.calibration_replay_source is not None:
        payload["calibration_replay_source"] = _json_safe_value(result.calibration_replay_source)
    if result.tuning_calibration_source is not None:
        payload["tuning_calibration_source"] = _json_safe_value(result.tuning_calibration_source)
    publisher = evidence.get("publisher") if isinstance(evidence, dict) else None
    if publisher is not None:
        payload["publisher"] = str(publisher)
    return payload


def workspace_robustness_evidence(args):
    """Inspect prediction-level spectral/OOD replay evidence in a workspace."""

    from nirs4all.data.predictions import Predictions

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    predictions = Predictions.from_workspace(workspace_path, dataset_name=args.dataset, load_arrays=True)
    rows = predictions.to_dicts(load_arrays=True)
    if args.id:
        wanted = str(args.id)
        rows = [row for row in rows if str(row.get("id") or row.get("prediction_id") or "") == wanted]

    diagnostics = []
    for row in rows:
        try:
            diagnostics.append(_prediction_robustness_evidence_payload(row))
        except Exception as exc:
            diagnostics.append(_prediction_robustness_evidence_payload(row, error=exc))

    if args.ready_only:
        diagnostics = [item for item in diagnostics if item["ready"]]
    if args.limit is not None:
        diagnostics = diagnostics[: max(0, int(args.limit))]

    payload = {
        "workspace": str(workspace_path),
        "dataset_name": args.dataset,
        "count": len(diagnostics),
        "ready_count": sum(1 for item in diagnostics if item["ready"]),
        "predictions": diagnostics,
    }
    if args.json:
        _print_json(payload)
        return
    if not diagnostics:
        logger.info("No prediction spectral/OOD replay evidence found.")
        return
    logger.info(f"Found {len(diagnostics)} prediction evidence diagnostic(s):\n")
    for item in diagnostics:
        status = item["spectral_replay_evidence_status"]
        logger.info(f"  {item['prediction_id']}")
        logger.info(f"    Dataset: {item.get('dataset_name') or '-'}  Model: {item.get('model_name') or '-'}  Partition: {item.get('partition') or '-'}")
        logger.info(f"    Status: {status.get('status')}")
        if status.get("predictor_bundle"):
            logger.info(f"    Predictor bundle: {status.get('predictor_bundle')}")
        missing = status.get("missing") or []
        if missing:
            logger.info(f"    Missing: {', '.join(str(value) for value in missing)}")
        if item.get("error"):
            logger.info(f"    Error: {item['error']}")
        logger.info("")


def add_workspace_commands(subparsers):
    """Add workspace commands to CLI."""

    # Workspace command group
    workspace = subparsers.add_parser("workspace", help="Workspace management commands")
    workspace_subparsers = workspace.add_subparsers(dest="workspace_command")

    # workspace init
    init_parser = workspace_subparsers.add_parser("init", help="Initialize a new workspace")
    init_parser.add_argument("path", type=str, help="Path to workspace directory")
    init_parser.set_defaults(func=workspace_init)

    # workspace list-runs
    list_runs_parser = workspace_subparsers.add_parser("list-runs", help="List all runs in workspace")
    list_runs_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    list_runs_parser.set_defaults(func=workspace_list_runs)

    # workspace query-best
    query_best_parser = workspace_subparsers.add_parser("query-best", help="Query best predictions from workspace")
    query_best_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    query_best_parser.add_argument("--dataset", type=str, help="Filter by dataset name")
    query_best_parser.add_argument("--metric", type=str, default="test_score", help="Metric to sort by (default: test_score)")
    query_best_parser.add_argument("-n", type=int, default=10, help="Number of results (default: 10)")
    query_best_parser.add_argument("--ascending", action="store_true", help="Sort ascending (lower is better)")
    query_best_parser.set_defaults(func=workspace_query_best)

    # workspace filter
    filter_parser = workspace_subparsers.add_parser("filter", help="Filter predictions by criteria")
    filter_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    filter_parser.add_argument("--dataset", type=str, help="Filter by dataset name")
    filter_parser.add_argument("--test-score", type=float, help="Minimum test score")
    filter_parser.add_argument("--train-score", type=float, help="Minimum train score")
    filter_parser.add_argument("--val-score", type=float, help="Minimum validation score")
    filter_parser.set_defaults(func=workspace_query_filter)

    # workspace stats
    stats_parser = workspace_subparsers.add_parser("stats", help="Show workspace statistics")
    stats_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    stats_parser.add_argument("--metric", type=str, default="test_score", help="Metric for statistics (default: test_score)")
    stats_parser.set_defaults(func=workspace_stats)

    # workspace list-library
    list_library_parser = workspace_subparsers.add_parser("list-library", help="List items in library")
    list_library_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    list_library_parser.set_defaults(func=workspace_list_library)

    # workspace inspect
    inspect_parser = workspace_subparsers.add_parser("inspect", help="Inspect workspace format without opening it writable")
    inspect_parser.add_argument("path", type=str, help="Workspace directory or legacy artifact to inspect")
    inspect_parser.set_defaults(func=workspace_inspect)

    # workspace convert
    convert_parser = workspace_subparsers.add_parser("convert", help="Convert a legacy workspace/artifact into a fresh V1 workspace")
    convert_parser.add_argument("input", type=str, help="Legacy workspace directory or artifact (read-only input)")
    convert_parser.add_argument("--output", required=True, type=str, help="Fresh output directory for the converted workspace")
    convert_mode = convert_parser.add_mutually_exclusive_group()
    convert_mode.add_argument("--verify", action="store_true", help="Verify the converted output after migration")
    convert_mode.add_argument("--dry-run", action="store_true", help="Inspect and simulate conversion without writing an output store")
    convert_parser.add_argument("--strict", action="store_true", help="Abort on the first unsupported legacy item")
    convert_parser.set_defaults(func=workspace_convert)

    # workspace tuning
    tuning_parser = workspace_subparsers.add_parser("tuning", help="Inspect native tuning results persisted in a workspace")
    tuning_subparsers = tuning_parser.add_subparsers(dest="tuning_command")

    tuning_list_parser = tuning_subparsers.add_parser("list", help="List persisted native tuning results")
    tuning_list_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    tuning_list_parser.add_argument("--limit", type=int, default=100, help="Maximum rows to show (default: 100)")
    tuning_list_parser.add_argument("--offset", type=int, default=0, help="Rows to skip (default: 0)")
    tuning_list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    tuning_list_parser.set_defaults(func=workspace_tuning_list)

    tuning_show_parser = tuning_subparsers.add_parser("show", help="Show one persisted native tuning result by id or fingerprint")
    tuning_show_parser.add_argument("id", type=str, help="Tuning result id or result fingerprint")
    tuning_show_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    tuning_show_parser.add_argument("--json", action="store_true", help="Emit the verified TuningResult JSON")
    tuning_show_parser.set_defaults(func=workspace_tuning_show)

    tuning_export_parser = tuning_subparsers.add_parser("export", help="Export one persisted native tuning result by id or fingerprint")
    tuning_export_parser.add_argument("id", type=str, help="Tuning result id or result fingerprint")
    tuning_export_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    tuning_export_parser.add_argument(
        "--format",
        choices=("json", "summary"),
        default="summary",
        help="Output artifact format (default: summary)",
    )
    tuning_export_parser.add_argument("--output", "-o", help="Write the artifact to this file instead of stdout")
    tuning_export_parser.add_argument("--indent", type=int, default=2, help="JSON indentation when not compact (default: 2)")
    tuning_export_parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    tuning_export_parser.set_defaults(func=workspace_tuning_export)

    # workspace conformal
    conformal_parser = workspace_subparsers.add_parser("conformal", help="Inspect conformal calibration results persisted in a workspace")
    conformal_subparsers = conformal_parser.add_subparsers(dest="conformal_command")

    conformal_list_parser = conformal_subparsers.add_parser("list", help="List persisted conformal calibration results")
    conformal_list_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    conformal_list_parser.add_argument("--limit", type=int, default=100, help="Maximum rows to show (default: 100)")
    conformal_list_parser.add_argument("--offset", type=int, default=0, help="Rows to skip (default: 0)")
    conformal_list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    conformal_list_parser.set_defaults(func=workspace_conformal_list)

    conformal_show_parser = conformal_subparsers.add_parser("show", help="Show one persisted conformal calibration result by id or fingerprint")
    conformal_show_parser.add_argument("id", type=str, help="Conformal result id or result fingerprint")
    conformal_show_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    conformal_show_parser.add_argument("--json", action="store_true", help="Emit the verified CalibratedRunResult JSON")
    conformal_show_parser.add_argument(
        "--as-predict-result",
        action="store_true",
        help="Emit the restored calibrated prediction through the public PredictResult-compatible payload",
    )
    conformal_show_parser.set_defaults(func=workspace_conformal_show)

    conformal_predict_parser = conformal_subparsers.add_parser("predict", help="Apply a persisted conformal calibrator to comma-separated point predictions")
    conformal_predict_parser.add_argument("id", type=str, help="Conformal result id or result fingerprint")
    conformal_predict_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    conformal_predict_parser.add_argument("--y-pred", required=True, type=str, help="Comma-separated point predictions, e.g. '1.2,1.5'")
    conformal_predict_parser.add_argument("--sample-ids", required=True, type=str, help="Comma-separated physical sample ids, e.g. 's1,s2'")
    conformal_predict_parser.add_argument("--json", action="store_true", help="Emit machine-readable prediction intervals")
    conformal_predict_parser.set_defaults(func=workspace_conformal_predict)

    # workspace robustness
    robustness_parser = workspace_subparsers.add_parser("robustness", help="Inspect robustness reports persisted in a workspace")
    robustness_subparsers = robustness_parser.add_subparsers(dest="robustness_command")

    robustness_list_parser = robustness_subparsers.add_parser("list", help="List persisted robustness reports")
    robustness_list_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    robustness_list_parser.add_argument("--limit", type=int, default=100, help="Maximum rows to show (default: 100)")
    robustness_list_parser.add_argument("--offset", type=int, default=0, help="Rows to skip (default: 0)")
    robustness_list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    robustness_list_parser.set_defaults(func=workspace_robustness_list)

    robustness_show_parser = robustness_subparsers.add_parser("show", help="Show one persisted robustness report by id or fingerprint")
    robustness_show_parser.add_argument("id", type=str, help="Robustness report id or result fingerprint")
    robustness_show_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    robustness_show_parser.add_argument("--json", action="store_true", help="Emit the verified RobustnessReport JSON")
    robustness_show_parser.set_defaults(func=workspace_robustness_show)

    robustness_export_parser = robustness_subparsers.add_parser("export", help="Export one persisted robustness report by id or fingerprint")
    robustness_export_parser.add_argument("id", type=str, help="Robustness report id or result fingerprint")
    robustness_export_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    robustness_export_parser.add_argument(
        "--format",
        choices=("json", "summary", "markdown", "html", "parquet", "artifacts"),
        default="markdown",
        help="Output artifact format (default: markdown)",
    )
    robustness_export_parser.add_argument("--output", "-o", help="Write the artifact to this file or directory instead of stdout")
    robustness_export_parser.add_argument("--indent", type=int, default=2, help="JSON indentation when --format=json/summary and not compact (default: 2)")
    robustness_export_parser.add_argument("--compact", action="store_true", help="Emit compact JSON when --format=json")
    robustness_export_parser.set_defaults(func=workspace_robustness_export)

    robustness_from_prediction_parser = robustness_subparsers.add_parser(
        "from-prediction",
        help="Compute a robustness report from one persisted prediction row",
    )
    robustness_from_prediction_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    robustness_from_prediction_parser.add_argument("--prediction-id", required=True, type=str, help="Prediction id to load from the workspace")
    y_true_group = robustness_from_prediction_parser.add_mutually_exclusive_group(required=True)
    y_true_group.add_argument("--y-true", type=str, help="Comma-separated observed target values, e.g. '1.2,1.5'")
    y_true_group.add_argument("--y-true-json", type=str, help="Observed target values as JSON, e.g. '[1.2, 1.5]'")
    robustness_from_prediction_parser.add_argument("--scenarios-json", type=str, help="Robustness scenarios as a JSON array")
    robustness_from_prediction_parser.add_argument("--slice-by", type=str, help="Comma-separated metadata keys used for slices")
    robustness_from_prediction_parser.add_argument("--metadata-json", type=str, help="Row metadata as a JSON object of arrays or JSON array of objects")
    robustness_from_prediction_parser.add_argument("--seed", type=int, help="Non-negative robustness scenario seed")
    robustness_from_prediction_parser.add_argument("--save-to-workspace", action="store_true", help="Persist the generated report back to the same workspace")
    robustness_from_prediction_parser.add_argument("--workspace-name", type=str, default="", help="Name used when --save-to-workspace is set")
    robustness_from_prediction_parser.add_argument("--workspace-robustness-id", type=str, help="Robustness id used when --save-to-workspace is set")
    robustness_from_prediction_parser.add_argument("--workspace-metadata-json", type=str, help="Workspace report metadata as a JSON object")
    robustness_from_prediction_parser.add_argument(
        "--format",
        choices=("json", "summary", "markdown", "html", "parquet", "artifacts"),
        default="markdown",
        help="Output artifact format (default: markdown)",
    )
    robustness_from_prediction_parser.add_argument("--output", "-o", help="Write the artifact to this file or directory instead of stdout")
    robustness_from_prediction_parser.add_argument("--indent", type=int, default=2, help="JSON indentation when --format=json/summary and not compact (default: 2)")
    robustness_from_prediction_parser.add_argument("--compact", action="store_true", help="Emit compact JSON when --format=json")
    robustness_from_prediction_parser.set_defaults(func=workspace_robustness_from_prediction)

    robustness_evidence_parser = robustness_subparsers.add_parser("evidence", help="Inspect prediction-level spectral/OOD replay evidence")
    robustness_evidence_parser.add_argument("--workspace", type=str, default="workspace", help="Workspace root directory (default: workspace)")
    robustness_evidence_parser.add_argument("--dataset", type=str, help="Filter by dataset name")
    robustness_evidence_parser.add_argument("--id", type=str, help="Filter by prediction id")
    robustness_evidence_parser.add_argument("--limit", type=int, default=100, help="Maximum prediction diagnostics to show (default: 100)")
    robustness_evidence_parser.add_argument("--ready-only", action="store_true", help="Show only predictions ready for spectral/OOD replay")
    robustness_evidence_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    robustness_evidence_parser.set_defaults(func=workspace_robustness_evidence)
