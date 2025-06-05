"""
FittedPipeline - Reusable fitted pipeline for prediction and re-execution

Loads a saved PipelineTree and provides methods for:
- Prediction on new data
- Full re-execution with new training
- Partial execution (transform only)
"""
from typing import Any, Dict, List, Union
from pathlib import Path
import numpy as np

from nirs4all.spectra.SpectraDataset import SpectraDataset
from .PipelineTree import PipelineTree


class FittedPipeline:
    """Reusable fitted pipeline that preserves structure"""

    def __init__(self, fitted_tree: PipelineTree):
        self.tree = fitted_tree
        self.structure = fitted_tree.structure
        self.metadata = fitted_tree.metadata

    def predict(self, dataset: SpectraDataset) -> Dict[str, Any]:
        """Apply fitted pipeline for prediction"""
        predictions = {}

        # Apply transformations and collect predictions
        dataset_copy = dataset.copy() if hasattr(dataset, 'copy') else dataset

        self._apply_structure_for_prediction(self.structure, dataset_copy, predictions, "")

        return predictions

    def _apply_structure_for_prediction(self, structure, dataset, predictions, path=""):
        """Recursively apply fitted objects for prediction"""

        if isinstance(structure, list):
            for i, item in enumerate(structure):
                self._apply_structure_for_prediction(item, dataset, predictions, f"{path}[{i}]")

        elif isinstance(structure, dict):
            if "dispatch" in structure:
                # For dispatch, we'll use the first branch for simplicity
                # In practice, you might want to aggregate all branches
                for i, branch in enumerate(structure["dispatch"]):
                    branch_predictions = {}
                    dataset_branch = dataset.copy() if hasattr(dataset, 'copy') else dataset
                    self._apply_structure_for_prediction(branch, dataset_branch, branch_predictions, f"{path}.dispatch[{i}]")
                    predictions.update(branch_predictions)
            else:
                for k, v in structure.items():
                    self._apply_structure_for_prediction(v, dataset, predictions, f"{path}.{k}")

        else:
            # This is a fitted object - use it for prediction/transformation
            if structure is not None:
                self._apply_fitted_object(structure, dataset, predictions, path)

    def _apply_fitted_object(self, fitted_obj, dataset, predictions, path):
        """Apply a single fitted object to the dataset"""
        try:
            if hasattr(fitted_obj, 'predict'):
                # Model - make predictions
                features = self._extract_features(dataset)
                if features is not None:
                    pred = fitted_obj.predict(features)
                    predictions[f"model_{path}"] = pred

            elif hasattr(fitted_obj, 'transform'):
                # Transformer - apply transformation
                features = self._extract_features(dataset)
                if features is not None:
                    transformed = fitted_obj.transform(features)
                    self._update_dataset_features(dataset, transformed)

            elif hasattr(fitted_obj, 'predict_proba'):
                # Classifier with probabilities
                features = self._extract_features(dataset)
                if features is not None:
                    proba = fitted_obj.predict_proba(features)
                    predictions[f"proba_{path}"] = proba

        except Exception as e:
            print(f"⚠️ Failed to apply fitted object at {path}: {e}")

    def _extract_features(self, dataset):
        """Extract features from dataset"""
        try:
            if hasattr(dataset, 'get_features'):
                return dataset.get_features()
            elif hasattr(dataset, 'features'):
                return dataset.features.get_features() if hasattr(dataset.features, 'get_features') else dataset.features
            else:
                return None
        except:
            return None

    def _update_dataset_features(self, dataset, new_features):
        """Update dataset with transformed features"""
        try:
            if hasattr(dataset, 'set_features'):
                dataset.set_features(new_features)
            elif hasattr(dataset, 'features') and hasattr(dataset.features, 'set_features'):
                dataset.features.set_features(new_features)
        except Exception as e:
            print(f"⚠️ Failed to update features: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            "creation_date": self.metadata.get("creation_timestamp", "unknown"),
            "feature_count": len(self.metadata.get("feature_names", [])),
            "config_summary": str(self.metadata.get("original_config", {}))[:200] + "...",
            "execution_summary": self.metadata.get("execution_summary", {}),
        }

    def get_model_names(self) -> List[str]:
        """Get names of all models in the pipeline"""
        models = []
        self._collect_models(self.structure, models, "")
        return models

    def _collect_models(self, structure, models, path=""):
        """Recursively collect model names"""
        if isinstance(structure, list):
            for i, item in enumerate(structure):
                self._collect_models(item, models, f"{path}[{i}]")

        elif isinstance(structure, dict):
            if "dispatch" in structure:
                for i, branch in enumerate(structure["dispatch"]):
                    self._collect_models(branch, models, f"{path}.dispatch[{i}]")
            else:
                for k, v in structure.items():
                    self._collect_models(v, models, f"{path}.{k}")

        else:
            if structure is not None and hasattr(structure, 'predict'):
                models.append(f"model_{path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FittedPipeline':
        """Load fitted pipeline from saved tree"""
        tree = PipelineTree.load(filepath)
        return cls(tree)


def load_pipeline(filepath: Union[str, Path]) -> FittedPipeline:
    """Convenience function to load a fitted pipeline"""
    return FittedPipeline.load(filepath)
