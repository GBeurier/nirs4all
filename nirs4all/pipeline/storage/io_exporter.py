"""Workspace exporter - exports best results to exports/ folder."""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class WorkspaceExporter:
    """Exports best results to workspace exports/ folder.

    Focused responsibility: Export functionality for best pipelines.
    """

    def __init__(self, workspace_path: Path):
        """Initialize exporter for a workspace.

        Args:
            workspace_path: Root workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self.exports_dir = self.workspace_path / "exports"

    def export_pipeline_full(
        self,
        pipeline_dir: Path,
        dataset_name: str,
        run_date: str = None,
        custom_name: Optional[str] = None
    ) -> Path:
        """Export full pipeline results to flat structure.

        Args:
            pipeline_dir: Path to pipeline (NNNN_hash/)
            dataset_name: Dataset name
            run_date: Run date (YYYYMMDD) - deprecated, kept for compatibility
            custom_name: Optional custom name for export

        Returns:
            Path to exported directory
        """
        pipeline_dir = Path(pipeline_dir)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        pipeline_id = pipeline_dir.name

        if custom_name:
            export_name = f"{custom_name}_{pipeline_id}"
        else:
            export_name = f"{dataset_name}_{pipeline_id}"

        export_path = self.exports_dir / export_name

        # Copy entire pipeline folder
        shutil.copytree(pipeline_dir, export_path, dirs_exist_ok=True)

        return export_path

    def export_best_prediction(
        self,
        predictions_file: Path,
        dataset_name: str,
        run_date: str = None,
        pipeline_id: str = None,
        custom_name: Optional[str] = None
    ) -> Path:
        """Export predictions CSV to best_predictions/ folder.

        Args:
            predictions_file: Path to predictions.csv
            dataset_name: Dataset name
            run_date: Run date - deprecated, kept for compatibility
            pipeline_id: Pipeline identifier
            custom_name: Optional custom name for export

        Returns:
            Path to exported CSV
        """
        predictions_file = Path(predictions_file)

        best_dir = self.exports_dir / "best_predictions"
        best_dir.mkdir(parents=True, exist_ok=True)

        if custom_name:
            csv_name = f"{custom_name}_{pipeline_id}.csv" if pipeline_id else f"{custom_name}.csv"
        else:
            csv_name = f"{dataset_name}_{pipeline_id}.csv" if pipeline_id else f"{dataset_name}.csv"

        dest_path = best_dir / csv_name

        shutil.copy2(predictions_file, dest_path)

        return dest_path

    def export_best_for_dataset(
        self,
        dataset_name: str,
        runs_dir: Path,
        mode: str = "predictions"
    ) -> Optional[Path]:
        """Export best results for a dataset to exports/ folder.

        Args:
            dataset_name: Dataset name
            runs_dir: Runs directory path
            mode: Export mode - "predictions", "template", "trained", or "full"

        Returns:
            Path to export directory, or None if no predictions found
        """
        runs_dir = Path(runs_dir)

        # Load global predictions for this dataset
        # Try .meta.parquet first (new format), then .json (legacy)
        predictions_file = self.workspace_path / f"{dataset_name}.meta.parquet"
        if not predictions_file.exists():
            predictions_file = self.workspace_path / f"{dataset_name}.json"
            if not predictions_file.exists():
                logger.warning(f"No predictions found for dataset '{dataset_name}'")
                return None

        from nirs4all.data.predictions import Predictions

        predictions = Predictions.load_from_file_cls(str(predictions_file))
        if predictions.num_predictions == 0:
            logger.warning(f"No predictions in database for '{dataset_name}'")
            return None

        # Get best prediction
        best = predictions.get_best(ascending=True)

        # Create export directory structure
        exports_dir = self.exports_dir / dataset_name
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Find run directory - now just the dataset name (no date prefix)
        run_dir = runs_dir / dataset_name
        if not run_dir.exists():
            # Fallback: try legacy format with date prefix (YYYY-MM-DD_dataset)
            legacy_dirs = list(runs_dir.glob(f"*_{dataset_name}"))
            if legacy_dirs:
                run_dir = legacy_dirs[-1]  # Get most recent
            else:
                logger.warning(f"No run directory found for dataset '{dataset_name}'")
                return None

        # Find the pipeline directory
        config_name = best['config_name']
        pipeline_dir = None
        for pd in run_dir.iterdir():
            if pd.is_dir() and config_name in pd.name and not pd.name.startswith('_'):
                pipeline_dir = pd
                break

        if not pipeline_dir:
            logger.warning(f"Pipeline directory not found for config '{config_name}'")
            return None

        # Export predictions
        pred_filename = f"{best['model_name']}_predictions.csv"
        pred_path = exports_dir / pred_filename
        Predictions.save_predictions_to_csv(best["y_true"], best["y_pred"], pred_path)
        logger.success(f"Exported predictions: {pred_path}")

        # Export pipeline config
        pipeline_json = pipeline_dir / "pipeline.json"
        if pipeline_json.exists():
            config_filename = f"{best['model_name']}_pipeline.json"
            config_path = exports_dir / config_filename
            shutil.copy(pipeline_json, config_path)
            logger.success(f"Exported pipeline config: {config_path}")

        # Export charts if they exist
        for chart_file in pipeline_dir.glob("*.png"):
            chart_filename = f"{best['model_name']}_{chart_file.name}"
            chart_path = exports_dir / chart_filename
            shutil.copy(chart_file, chart_path)
            logger.success(f"Exported chart: {chart_path}")

        # Handle different export modes for binaries
        if mode in ["trained", "full"]:
            binaries_dir = run_dir / "_binaries"
            if binaries_dir.exists():
                export_binaries_dir = exports_dir / "_binaries"
                export_binaries_dir.mkdir(exist_ok=True)

                # Copy referenced binaries from manifest
                manifest_file = pipeline_dir / "manifest.yaml"
                if manifest_file.exists():
                    import yaml

                    with open(manifest_file, 'r') as f:
                        manifest = yaml.safe_load(f)

                    for artifact in manifest.get('artifacts', []):
                        binary_name = Path(artifact['path']).name
                        src = binaries_dir / binary_name
                        if src.exists():
                            shutil.copy(src, export_binaries_dir / binary_name)

                    logger.success(f"Exported {len(list(export_binaries_dir.iterdir()))} binaries")

        # Create summary metadata
        summary = {
            "dataset": dataset_name,
            "model_name": best['model_name'],
            "pipeline_id": config_name,
            "prediction_id": best['id'],
            "test_score": best.get('test_score'),
            "val_score": best.get('val_score'),
            "export_date": datetime.now().isoformat(),
            "export_mode": mode
        }
        summary_path = exports_dir / f"{best['model_name']}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.success(f"Exported summary: {summary_path}")

        return exports_dir
