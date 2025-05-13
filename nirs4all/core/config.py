# config.py
import itertools
import dataclasses
from typing import List, Optional, Union

@dataclasses.dataclass
class Config:
    dataset: Union[str, object]
    x_pipeline: Optional[Union[str, object]] = None
    y_pipeline: Optional[Union[str, object]] = None
    model: Optional[Union[str, object]] = None
    experiment: Optional[dict] = None
    seed: Optional[int] = None


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
                if any(metric in classification_metrics for metric in metrics):
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
