# manager.py

import os
import inspect
import json
import hashlib
import logging
import shutil
import dataclasses
from typing import Any, Dict, Optional, Union, List

import numpy as np
import pandas as pd

from ..model.model_builder_factory import ModelBuilderFactory
from ..model.model_manager import ModelManagerFactory
from ..utils import TF_AVAILABLE, TORCH_AVAILABLE


def sanitize_folder_name(name: str) -> str:
    """Sanitize folder name by removing invalid characters."""
    return ''.join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ExperimentManager:
    """Manages experiments by handling paths, logging, and results."""

    def __init__(self, results_dir: str, resume_mode: str = 'skip', verbose: int = 1):
        self.results_dir = results_dir
        self.resume_mode = resume_mode
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.experiment_info: Dict[str, Any] = {}
        self.experiment_path: Optional[str] = None
        self.key_metric = 'mean_squared_error'
        os.makedirs(self.results_dir, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Set up the logger for the experiment."""
        logger = logging.getLogger('ExperimentManager')
        logger.setLevel(logging.DEBUG)# if self.verbose >= 2 else logging.INFO)

        # Avoid adding multiple handlers if logger already has handlers
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        logger.propagate = False
        return logger

    def prepare_experiment(self, config: Any) -> None:
        """Prepare experiment by setting up paths and handling previous runs."""
        dataset_config = config.dataset
        model_config = config.model
        seed = config.seed
        dataset_name = self._extract_dataset_name(dataset_config)
        model_name = self._extract_model_name(model_config)

        # Create unique experiment identifier based on config hash
        config_serializable = self.make_config_serializable(config)
        config_str = json.dumps(config_serializable, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        experiment_id = f"experiment_{config_hash}"

        # Construct experiment path as per requirement
        experiment_path = os.path.join(self.results_dir, dataset_name, model_name, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)

        self._handle_existing_experiment(experiment_path, experiment_id)

        # Save the config to the experiment folder
        config_save_path = os.path.join(experiment_path, 'config.json')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(config_serializable, f, indent=4)

        # Update experiment information
        self.experiment_info = {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'seed': seed,
            'experiment_id': experiment_id,
            'experiment_path': experiment_path,
            'config': config
        }
        self.experiment_path = experiment_path
        self.logger.info("Experiment prepared at %s", self.experiment_path)

        # Set up per-experiment log file
        self._add_file_handler_to_logger()

    def _add_file_handler_to_logger(self) -> None:
        """Add a file handler to the logger to write logs to experiment.log in the experiment folder."""
        log_file_path = os.path.join(self.experiment_path, 'experiment.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        # Remove existing file handlers to avoid duplicate logs
        self.logger.handlers = [h for h in self.logger.handlers if not isinstance(h, logging.FileHandler)]
        self.logger.addHandler(file_handler)

    def _extract_dataset_name(self, dataset_config: Any) -> str:
        """Extract and sanitize dataset name from the dataset configuration."""
        if isinstance(dataset_config, str):
            dataset_name = dataset_config
        elif hasattr(dataset_config, 'name'):
            dataset_name = dataset_config.name
        elif isinstance(dataset_config, dict):
            if 'name' in dataset_config:
                dataset_name = dataset_config['name']
            elif 'path' in dataset_config:
                # dataset_name = os.path.splitext(os.path.basename(dataset_config['path']))[0]
                dataset_name = dataset_config['path']
            else:
                dataset_name = 'unknown_dataset'
        else:
            dataset_name = 'unknown_dataset'
        return sanitize_folder_name(dataset_name)

    def _extract_model_name(self, model_config: Any) -> str:
        """Extract and sanitize model name from the model configuration."""
        if isinstance(model_config, dict):
            if 'class' in model_config:
                model_name = model_config['class'].split('.')[-1]
            elif 'path' in model_config:
                model_name = os.path.splitext(os.path.basename(model_config['path']))[0]
            else:
                model_name = 'unknown_model'
        elif callable(model_config):
            model_name = model_config.__name__
        elif isinstance(model_config, str):
            model_name = os.path.splitext(os.path.basename(model_config))[0]
        else:
            model_name = 'unknown_model'
        return sanitize_folder_name(model_name)

    def _handle_existing_experiment(self, experiment_path: str, experiment_id: str) -> None:
        """Handle existing experiment based on the resume mode."""
        if self.is_experiment_completed(experiment_path):
            if self.resume_mode == 'skip':
                self.logger.info("Experiment %s already completed. Skipping.", experiment_id)
                raise RuntimeError(f"Experiment {experiment_id} already completed.")
            elif self.resume_mode == 'resume':
                self.logger.info("Resuming experiment %s.", experiment_id)
            elif self.resume_mode == 'restart':
                self.logger.info("Restarting experiment %s.", experiment_id)
                self._archive_experiment(experiment_path, experiment_id)
                os.makedirs(experiment_path, exist_ok=True)
            else:
                self.logger.error("Unknown resume mode: %s", self.resume_mode)
                raise ValueError(f"Unknown resume mode: {self.resume_mode}")
        else:
            self.logger.info("Starting new experiment %s.", experiment_id)

    def _archive_experiment(self, experiment_path: str, experiment_id: str) -> None:
        """Archive existing experiment by moving it to an archives directory."""
        archives_dir = os.path.join(self.results_dir, 'archives')
        os.makedirs(archives_dir, exist_ok=True)
        archive_count = 0
        while True:
            archived_experiment_path = os.path.join(archives_dir, f"{experiment_id}_{archive_count}")
            if not os.path.exists(archived_experiment_path):
                break
            archive_count += 1
        shutil.move(experiment_path, archived_experiment_path)
        self.logger.info("Archived experiment to %s", archived_experiment_path)

    def is_experiment_completed(self, experiment_path: str) -> bool:
        """Check if the experiment is completed by verifying the presence and validity of model and metrics."""
        model_path = os.path.join(experiment_path, 'model')
        metrics_path = os.path.join(experiment_path, 'metrics.json')

        if not os.path.exists(model_path) or not os.path.exists(metrics_path):
            return False

        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            if not metrics:  # Ensure metrics file is not empty
                return False
        except (json.JSONDecodeError, IOError):
            return False

        return True

    def save_results(self, model_manager: Any, y_pred: Union[np.ndarray, List[np.ndarray]], y_true: np.ndarray, metrics: list, best_params: dict = None, fold_scores: List[dict] = None) -> None:
        if not self.experiment_path:
            self.logger.error("Experiment path not set. Cannot save results.")
            return

        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        if fold_scores is None:
            fold_scores = []

        results_df = pd.DataFrame({'y_true': y_true.flatten()})
        scores_dict = {}
        total_preds = len(y_pred)

        for i, y_pred_i in enumerate(y_pred):
            # Determine the key for saving metrics and predictions
            if i < total_preds - 3:
                key = f'fold_{i}'
                pred_column = f'y_pred_fold_{i}'
            elif i == total_preds - 3:
                key = 'mean'
                pred_column = 'y_pred_mean'
            elif i == total_preds - 2:
                key = 'best'
                pred_column = 'y_pred_best'
            elif i == total_preds - 1:
                key = 'weighted'
                pred_column = 'y_pred_weighted'
            else:
                key = f'prediction_{i}'
                pred_column = f'y_pred_{i}'

            # Retrieve scores
            scores = fold_scores[i] if i < len(fold_scores) else model_manager.evaluate(y_true, y_pred_i, metrics)
            scores_dict[key] = scores

            # Save metrics to log
            self.logger.info("Evaluation Metrics %s: %s", key, scores)

            # Add predictions to DataFrame
            results_df[pred_column] = y_pred_i.flatten()

        # Save all metrics to a single JSON file
        scores_path = os.path.join(self.experiment_path, "metrics.json")
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(scores_dict, f, cls=NumpyEncoder, indent=4)
        self.logger.info("Metrics saved to %s", scores_path)

        # Save best parameters if available
        if best_params:
            best_params_path = os.path.join(self.experiment_path, "best_params.json")
            with open(best_params_path, "w", encoding="utf-8") as f:
                json.dump(best_params, f, cls=NumpyEncoder, indent=4)
            self.logger.info("Best parameters %s saved to %s", best_params, best_params_path)

        # Save predictions to CSV
        results_csv_path = os.path.join(self.experiment_path, 'predictions.csv')
        results_df.to_csv(results_csv_path, index=False)
        self.logger.info("Predictions saved to %s", results_csv_path)

        # Update centralized results if necessary
        self.update_centralized_results(scores_dict, best_params)



    def make_config_serializable(self, config: Any) -> Dict[str, Any]:
        """
        Convert config to a JSON-serializable dictionary. Objects are replaced with their import path,
        class names, and parameters. This allows reloading the pipeline from the config.
        """

        def sanitize_path(path):
            """Remove leading underscore in module paths."""
            if '.' in path:
                return '.'.join(part for part in path.split('.') if not part.startswith('_'))
            return path

        def obj_module(obj):
            """Get the module path of an object."""
            return sanitize_path(obj.__module__)

        def serialize_object(obj):
            # Handle primitive types (str, int, float, bool) directly
            if isinstance(obj, (str, int, float, bool)):
                return obj
            # Handle None
            elif obj is None:
                return None
            # Handle lists and tuples
            elif isinstance(obj, (list, tuple)):
                # Recursively serialize lists/tuples by calling serialize_pipeline on each element
                return [serialize_pipeline(item) for item in obj]
            # Handle dictionaries (to recursively serialize their contents)
            elif isinstance(obj, dict):
                # Recursively serialize dictionary values
                return {key: serialize_pipeline(value) for key, value in obj.items()}
            # Handle class objects (like sklearn.model_selection._split.KFold)
            elif inspect.isclass(obj):  # This catches class types like KFold
                return {
                    'class': f"{obj_module(obj)}.{obj.__name__}",
                    'params': None  # Classes do not have params, so set to None
                }
            # Handle scikit-learn objects and objects with get_params()
            elif hasattr(obj, 'get_params'):
                return {
                    'class': f"{obj_module(obj)}.{obj.__class__.__name__}",
                    'params': serialize_pipeline(obj.get_params())  # Recursively serialize params
                }
            # Handle functions and callables
            elif callable(obj):
                return {
                    'function': f"{obj_module(obj)}.{obj.__name__}"
                }
            # Handle objects with __dict__ (custom objects or others)
            elif hasattr(obj, '__dict__'):
                return {
                    'class': f"{obj_module(obj)}.{obj.__class__.__name__}",
                    'params': serialize_pipeline(vars(obj))  # Recursively serialize vars
                }
            # Default case for unsupported types
            return str(obj)  # Convert to a string if none of the above matches

        def serialize_pipeline(pipeline):
            """Recursively serialize pipeline objects."""
            # Recursively serialize lists and dictionaries
            if isinstance(pipeline, list):
                return [serialize_pipeline(step) for step in pipeline]
            if isinstance(pipeline, dict):
                return {key: serialize_pipeline(value) for key, value in pipeline.items()}
            # If it's not a list or dict, serialize the object
            return serialize_object(pipeline)

        try:
            # If the config is a dataclass, convert it to a dictionary
            serializable_config = dataclasses.asdict(config)
        except TypeError:
            # Otherwise, just use its vars
            serializable_config = vars(config)

        # Serialize all objects in the configuration, including pipelines
        for key, value in serializable_config.items():
            serializable_config[key] = serialize_pipeline(value)

        return serializable_config

    def update_centralized_results(self, metrics: Dict[str, Any], best_params: Optional[Dict[str, Any]] = None) -> None:
        """Update centralized JSON files with experiment results."""
        experiment_info = self.experiment_info
        dataset_name = experiment_info['dataset_name']
        model_name = experiment_info['model_name']
        experiment_entry = {
            'model_params': self._extract_model_params(experiment_info['config']),
            'scores': metrics,
            'path': os.path.relpath(experiment_info['experiment_path'], self.results_dir)
        }
        if best_params:
            experiment_entry['best_params'] = best_params

        # Update experiments at various levels
        self._update_experiments_json(dataset_name, model_name, experiment_entry)

    def _extract_model_params(self, config: Any) -> Dict[str, Any]:
        """Extract model parameters from the config."""
        model_config = config.model
        if isinstance(model_config, dict):
            return model_config.get('model_params', {})
        else:
            return {}

    def _update_experiments_json(self, dataset_name: str, model_name: str, experiment_entry: Dict[str, Any]) -> None:
        """Update experiments.json at dataset and model levels."""
        # Update model-level experiments.json
        model_experiments_path = os.path.join(self.results_dir, dataset_name, model_name, 'experiments.json')
        self._append_and_sort_experiments(model_experiments_path, experiment_entry)

        # Update dataset-level experiments.json
        dataset_experiments_path = os.path.join(self.results_dir, dataset_name, 'experiments.json')
        self._append_and_sort_experiments(dataset_experiments_path, experiment_entry)
        self.logger.info("Updated experiments at %s and %s", model_experiments_path, dataset_experiments_path)

    def _append_and_sort_experiments(self, json_path: str, experiment_entry: Dict[str, Any]) -> None:
        """Append an experiment entry to a JSON file and sort by key metric."""
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                experiments = json.load(f)
        else:
            experiments = []
        experiments.append(experiment_entry)
        experiments.sort(key=lambda x: x['scores'].get(self.key_metric, float('inf')))
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(experiments, f, indent=4, cls=NumpyEncoder)
        self.logger.info("Updated experiments at %s", json_path)
