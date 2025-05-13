import itertools
import dataclasses
from typing import List, Optional, Union, Any, Dict
import json
import inspect

# Helper function to serialize components
def _serialize_component(component: Any) -> Any:
    if component is None:
        return None
    if isinstance(component, (str, int, float, bool)):
        return component
    if isinstance(component, dict):
        # If it's a dict, assume it might be already in a serializable form
        # or a representation of a class with params.
        # We'll try to ensure keys like 'class' or 'method' are present if it's a custom object.
        if 'class' in component or 'method' in component:
            # Recursively serialize params if they exist
            if 'params' in component and isinstance(component['params'], dict):
                component['params'] = {k: _serialize_component(v) for k, v in component['params'].items()}
            return component
        # For general dicts, serialize their values
        return {k: _serialize_component(v) for k, v in component.items()}
    if isinstance(component, list):
        return [_serialize_component(c) for c in component]
    if inspect.isclass(component):
        return {"class": f"{component.__module__}.{component.__name__}"}
    # For instances (e.g., scikit-learn transformers, models)
    if hasattr(component, 'get_params') and callable(component.get_params):
        try:
            params = component.get_params(deep=True)
            # Recursively serialize parameter values
            serialized_params = {k: _serialize_component(v) for k, v in params.items()}
            return {
                "class": f"{component.__class__.__module__}.{component.__class__.__name__}",
                "params": serialized_params
            }
        except Exception:  # Fallback if get_params fails or params are not easily serializable
            return {"class": f"{component.__class__.__module__}.{component.__class__.__name__}"}
    # Fallback for other object types: just represent them by their class name
    return {"class": f"{component.__class__.__module__}.{component.__class__.__name__}"}


@dataclasses.dataclass
class Config:
    dataset: Union[str, dict]  # For JSON, dataset path or a data config dict
    x_pipeline: Optional[Any] = None
    y_pipeline: Optional[Any] = None
    model: Optional[Any] = None
    experiment: Optional[dict] = None
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Converts the Config object to a dictionary suitable for JSON serialization."""
        return {
            "dataset": _serialize_component(self.dataset),
            "x_pipeline": _serialize_component(self.x_pipeline),
            "y_pipeline": _serialize_component(self.y_pipeline),
            "model": _serialize_component(self.model),
            "experiment": _serialize_component(self.experiment),  # Experiments can have nested serializable components
            "seed": self.seed
        }

    def to_json_file(self, filepath: str, indent: int = 4) -> None:
        """Saves the Config object to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Creates a Config object from a dictionary."""
        # The downstream components (get_transformer, model_builder, etc.)
        # are expected to handle the string/dict representations.
        return cls(
            dataset=data.get("dataset"),
            x_pipeline=data.get("x_pipeline"),
            y_pipeline=data.get("y_pipeline"),
            model=data.get("model"),
            experiment=data.get("experiment"),
            seed=data.get("seed")
        )

    @classmethod
    def from_json_file(cls, filepath: str) -> "Config":
        """Loads a Config object from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self, dataset_instance):
        experiment_config = self.experiment or {}
        
        finetune_params = experiment_config.get('finetune_params', {})
        training_params = experiment_config.get('training_params', {})
        action = experiment_config.get('action', 'train')
        
        task = experiment_config.get('task', None)
        metrics = experiment_config.get('metrics', None)
        loss = training_params.get('loss', None)
        classification_losses = {'binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy'}
        classification_metrics = {'accuracy', 'acc', 'precision', 'recall', 'f1', 'auc'}
        
        if task is None:
            if loss is not None:
                if loss in classification_losses:
                    task = 'classification'
                    if metrics is None:
                        metrics = ['accuracy']
                else:
                    task = 'regression'
                    if metrics is None:
                        metrics = ['mse', 'mae']
            elif metrics is not None:
                if any(metric in classification_metrics for metric in metrics):  # Corrected syntax for any() and comment spacing
                    task = 'classification'
                    if dataset_instance.num_classes is None:
                        raise ValueError("Number of classes is not defined in dataset. Please specify the number of classes in the dataset config.")
                    elif dataset_instance.num_classes == 2:
                        training_params['loss'] = 'binary_crossentropy'
                    else:
                        training_params['loss'] = 'sparse_categorical_crossentropy'
                else:
                    task = 'regression'
                    training_params['loss'] = 'mse'
            else:
                task = 'regression'
                if metrics is None:
                    metrics = ['mse', 'mae']
        else:
            if task == 'classification' and metrics is None:
                metrics = ['accuracy']
            elif task == 'regression' and metrics is None:
                metrics = ['mse', 'mae']
                
            if task == 'classification' and 'loss' not in training_params:
                if dataset_instance.num_classes == 2:
                    training_params['loss'] = 'binary_crossentropy'
                else:
                    training_params['loss'] = 'sparse_categorical_crossentropy'
            elif task == 'regression' and 'loss' not in training_params:
                training_params['loss'] = 'mse'
            
        experiment_config['metrics'] = metrics
        experiment_config['task'] = task
        if task == 'classification':
            experiment_config['num_classes'] = dataset_instance.num_classes
        
        # Ensure self.experiment is updated with the processed experiment_config
        # This is important if defaults were added or task was inferred.
        self.experiment = experiment_config 
        
        return action, metrics, training_params, finetune_params, task

# @dataclasses.dataclass
# class Configs_Generator:
#     datasets: List[str]
#     model_experiments: List[dict]  # List of tuples (model_config, experiment)
#     preparations: Optional[List[str]] = None
#     scalers: Optional[List[str]] = None
#     augmenters: Optional[List[str]] = None
#     preprocessings: Optional[List[str]] = None
#     reporting: Optional[dict] = None
#     seeds: Optional[List[int]] = None

#     def generate_configs(self):
#         self.preparations = self.preparations or [None]
#         self.scalers = self.scalers or [None]
#         self.preprocessings = self.preprocessings or [None]

#         for dataset, (model_config, experiment), preparation, scaler, preprocessing, seed in itertools.product(
#             self.datasets, self.model_experiments, self.preparations, self.scalers, self.preprocessings, self.seeds
#         ):
#             yield Config(dataset, model_config, preparation, scaler, preprocessing, experiment, seed, self.reporting)
