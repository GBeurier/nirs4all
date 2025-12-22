"""Prediction target resolver - resolves prediction targets for predict mode.

Note: This is the legacy target resolver for finding predictions by ID.
For the comprehensive Phase 3 prediction source resolver that normalizes
sources to all components needed for replay, see:
    nirs4all.pipeline.resolver.PredictionResolver
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

from nirs4all.data.predictions import Predictions


class TargetResolver:
    """Resolves prediction targets for predict mode.

    Focused responsibility: Finding and resolving prediction targets by ID.

    Note: For the comprehensive Phase 3 resolver that handles multiple
    source types (prediction dict, folder, Run, artifact_id, bundle),
    see nirs4all.pipeline.resolver.PredictionResolver.
    """

    def __init__(self, workspace_path: Path):
        """Initialize resolver for a workspace.

        Args:
            workspace_path: Root workspace directory
        """
        self.workspace_path = Path(workspace_path)

    def resolve_target(
        self,
        prediction_obj: Union[Dict[str, Any], str]
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Resolve prediction object to config path and model metadata.

        Args:
            prediction_obj: Either:
                - Dict with 'config_path' and optional model metadata
                - String: config path or prediction ID

        Returns:
            Tuple of (config_path, target_model_metadata)

        Raises:
            ValueError: If prediction ID not found or invalid input
        """
        if isinstance(prediction_obj, dict):
            config_path = prediction_obj['config_path']
            target_model = prediction_obj if 'model_name' in prediction_obj else None
            return config_path, target_model

        elif isinstance(prediction_obj, str):
            # Check if it's a file path
            if Path(prediction_obj).exists():
                config_path = prediction_obj
                target_model = None
                return config_path, target_model

            # Otherwise, treat as prediction ID
            target_model = self.find_prediction_by_id(prediction_obj)
            if not target_model:
                raise ValueError(f"Prediction ID not found: {prediction_obj}")

            config_path = target_model['config_path']
            return config_path, target_model

        else:
            raise ValueError(f"Invalid prediction_obj type: {type(prediction_obj)}")

    def find_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Search for a prediction by ID in global predictions databases.

        Uses direct ID filtering for O(1) lookup per file instead of O(N) iteration.

        Args:
            prediction_id: Unique prediction identifier

        Returns:
            Prediction metadata dict, or None if not found
        """
        if not self.workspace_path.exists():
            return None

        # Define search paths (workspace root and runs directory)
        search_paths = [self.workspace_path]
        if (self.workspace_path / "runs").exists():
            search_paths.append(self.workspace_path / "runs")

        # Search in global prediction databases
        for path in search_paths:
            # Try Parquet files first (new format)
            for predictions_file in path.glob("*.meta.parquet"):
                if not predictions_file.is_file():
                    continue

                try:
                    predictions = Predictions.load_from_file_cls(str(predictions_file))
                    # Use direct ID filter instead of iterating all predictions
                    pred = predictions.get_prediction_by_id(prediction_id)
                    if pred is not None:
                        return pred
                except Exception:
                    continue

            # Fall back to JSON files (legacy format)
            for predictions_file in path.glob("*.json"):
                if not predictions_file.is_file():
                    continue

                try:
                    predictions = Predictions.load_from_file_cls(str(predictions_file))
                    # Use direct ID filter instead of iterating all predictions
                    pred = predictions.get_prediction_by_id(prediction_id)
                    if pred is not None:
                        return pred
                except Exception:
                    continue

        return None

    def find_best_for_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Find the best prediction for a given config path.

        Args:
            config_path: Path to pipeline configuration

        Returns:
            Best prediction metadata, or None if not found
        """
        # Extract dataset name from path structure
        # config_path typically: "runs/YYYY-MM-DD_dataset/0001_hash/pipeline.json"
        path_parts = Path(config_path).parts

        for part in path_parts:
            if '_' in part and not part.startswith('_'):
                # Likely the dataset directory
                dataset_name = part.split('_', 1)[1]  # Remove date prefix
                break
        else:
            return None

        # Load predictions for this dataset
        predictions_file = self.workspace_path / f"{dataset_name}.json"
        if not predictions_file.exists():
            # Try parquet format
            predictions_file = self.workspace_path / f"{dataset_name}.meta.parquet"
            if not predictions_file.exists():
                return None

        try:
            predictions = Predictions.load_from_file_cls(str(predictions_file))

            # Filter by config path - use filter_predictions with load_arrays=False first
            # to avoid loading all arrays when we just need to find the best score
            matching = predictions.filter_predictions(config_path=config_path, load_arrays=False)

            if not matching:
                return None

            # Return best by score (assumes lower is better)
            best = min(matching, key=lambda p: p.get('test_score', float('inf')))

            # Load arrays for the best prediction only
            best_id = best.get('id')
            if best_id:
                return predictions.get_prediction_by_id(best_id, load_arrays=True)
            return best

        except Exception:
            return None


# Backward compatibility alias
# Deprecated: Use nirs4all.pipeline.resolver.PredictionResolver for Phase 3+ features
PredictionResolver = TargetResolver
