# base_finetuner.py
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
from abc import ABC, abstractmethod
from ..model.model_builder_factory import ModelBuilderFactory
from ..model.model_manager import ModelManagerFactory
from ..utils import TF_AVAILABLE, TORCH_AVAILABLE
import copy
import numpy as np


class BaseFineTuner(ABC):
    def __init__(self, model_manager):
        self.model_manager = model_manager

    @abstractmethod
    def finetune(self, dataset, finetune_params, src_training_params=None, metrics=None, task=None):
        pass


class OptunaFineTuner(BaseFineTuner):
    def __init__(self, model_manager):
        super().__init__(model_manager)
        self.model_config = copy.deepcopy(model_manager.model_config)  # Access initial model_config

    def finetune(self, dataset, finetune_params, src_training_params=None, metrics=None, task=None):
        union = 'concat' if self.model_manager.framework == 'sklearn' else 'union'
        model_config = copy.deepcopy(self.model_config)
        model = ModelBuilderFactory.build_single_model(model_config, dataset, task)
        finetune_model_manager = ModelManagerFactory.get_model_manager(model, dataset, task)  # TODO avoid multiple models if not necessary
        
        def objective(trial):
            # Deep copy to avoid mutations
            model_config = copy.deepcopy(self.model_config)
            model_params = {}
            training_params = {}

            # Extract model parameters
            for key, values in finetune_params.get('model_params', {}).items():
                if isinstance(values, list):
                    model_params[key] = trial.suggest_categorical(key, values)
                elif isinstance(values, tuple):
                    if values[0] == 'int':
                        model_params[key] = trial.suggest_int(key, values[1], values[2])
                    elif values[0] == 'float':
                        model_params[key] = trial.suggest_float(key, values[1], values[2])

            # Update the model_config with new parameters
            # if 'model_params' not in model_config:
                # model_config['model_params'] = {}
            # model_config['model_params'].update(model_params)

            # Similarly extract training parameters
            for key, values in finetune_params.get('training_params', {}).items():
                if isinstance(values, list):
                    training_params[key] = trial.suggest_categorical(key, values)
                elif isinstance(values, tuple):
                    if values[0] == 'int':
                        training_params[key] = trial.suggest_int(key, values[1], values[2])
                    elif values[0] == 'float':
                        training_params[key] = trial.suggest_float(key, values[1], values[2])

            # Build the model with updated parameters
            model = ModelBuilderFactory.build_single_model(model_config, dataset, task, model_params)
            finetune_model_manager.models = [model]
            
            # merge training params and src_training_params, keeping the training_params
            if src_training_params is not None:
                for key, value in src_training_params.items():
                    if key not in training_params:
                        training_params[key] = value
            
            # Train the model
            finetune_model_manager.train(dataset, training_params=training_params, metrics=metrics, no_folds=True)

            # Predict and evaluate
            y_pred = finetune_model_manager.predict(dataset, task, no_folds=True, raw_class_output=(task == 'classification'))
            y_true = dataset.y_test
            if task == 'classification':
                if y_pred.ndim > 1:
                    if y_pred.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                        y_pred = np.argmax(y_pred, axis=-1)
                    elif y_pred.shape[-1] == 1 or dataset.num_classes == 1:
                        y_pred = (y_pred >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected output shape for classification task")
                else:
                    y_pred = (y_pred >= 0.5).astype(int)
            
            scores = finetune_model_manager.evaluate(y_true, y_pred, metrics)
            
            return scores[metrics[0]]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=finetune_params.get('n_trials', 100), n_jobs=-1)

        best_params = study.best_params
        # print(f"Best hyperparameters: {best_params}")

        # Update the model_manager with the best model
        # best_trial = study.best_trial
        # Rebuild the model with best parameters
        best_model_params = {}
        best_training_params = {}
        for key, value in best_params.items():
            if key in finetune_params.get('model_params', {}):
                best_model_params[key] = value
            elif key in finetune_params.get('training_params', {}):
                best_training_params[key] = value
                
        print(f"Best model params: {best_model_params}")

        # self.model_manager.model_config['model_params'].update(best_model_params)
        # Build the best model
        best_models, _ = ModelBuilderFactory.build_models(self.model_manager.model_config, dataset, task, best_model_params)
        self.model_manager.models = best_models

        if src_training_params is not None:
            for key, value in src_training_params.items():
                if key not in best_training_params:
                    best_training_params[key] = value

        # Retrain best model using original metrics
        self.model_manager.train(dataset, training_params=best_training_params, metrics=metrics)

        return best_params


class SklearnFineTuner(BaseFineTuner):
    def __init__(self, model_manager):
        super().__init__(model_manager)
        self.model_config = copy.deepcopy(model_manager.model_config)

    def finetune(self, dataset, finetune_params, src_training_params=None, metrics=None, task=None):
        union = 'concat' if self.model_manager.framework == 'sklearn' else 'union'
        x_train, y_train = dataset.x_train_(union), dataset.y_train

        approach = finetune_params.get('approach', 'grid')
        param_dict = {}

        # Convert model_params for sklearn GridSearchCV or RandomizedSearchCV
        for key, values in finetune_params.get('model_params', {}).items():
            if isinstance(values, list):
                param_dict[key] = values  # Discrete choices
            elif isinstance(values, tuple):
                if values[0] == 'int':
                    # print("int", values[1], values[2], key)
                    param_dict[key] = list(range(values[1], values[2] + 1))
                elif values[0] == 'float':
                    param_dict[key] = np.linspace(values[1], values[2], num=5)

        n_trials = finetune_params.get('n_trials', 30)
        n_jobs = finetune_params.get('n_jobs', -1)

        # Build base estimator
        estimator = ModelBuilderFactory.build_single_model(self.model_config, dataset=dataset, task=task)

        # cv = finetune_params.get('cv', 5)
        
        # print(param_dict)
        if approach == 'grid':
            search = GridSearchCV(estimator, param_dict, cv=None, n_jobs=n_jobs)
        elif approach == 'random':
            search = RandomizedSearchCV(estimator, param_dict, n_iter=n_trials, cv=None, n_jobs=n_jobs)
        else:
            search = GridSearchCV(estimator, param_dict, cv=None, n_jobs=n_jobs)

        search.fit(x_train, y_train)
        best_params = search.best_params_
        best_models, _ = ModelBuilderFactory.build_models(search.best_estimator_, dataset, task, best_params)
        self.model_manager.models = best_models

        # Update the model_config
        print(f"Best model params: {best_params}")
        # self.model_manager.model_config['model_params'].update(best_params)

        # Optional training with new parameters
        training_params = finetune_params.get('training_params', {})
        
        if src_training_params is not None:
            for key, value in src_training_params.items():
                if key not in training_params:
                    training_params[key] = value
        self.model_manager.train(dataset, training_params=training_params, metrics=metrics)

        return best_params


class FineTunerFactory:
    @staticmethod
    def get_fine_tuner(tuner_type, model_manager):
        if tuner_type == 'optuna':
            return OptunaFineTuner(model_manager)
        # elif tuner_type == 'hyperband' and TF_AVAILABLE:
            # return HyperbandFineTuner(model_manager)
        elif tuner_type == 'sklearn':
            return SklearnFineTuner(model_manager)
        # elif tuner_type == 'torch' and TORCH_AVAILABLE:
            # return TorchFineTuner(model_manager)
        else:
            raise ValueError(f"Unsupported fine-tuner type or framework not available: {tuner_type}")


# if TF_AVAILABLE:
#     from keras_tuner import Hyperband

#     class HyperbandFineTuner(BaseFineTuner):
#         def __init__(self, model_manager):
#             super().__init__(model_manager)
#             self.model_config = copy.deepcopy(model_manager.model_config)

#         def finetune(self, dataset, finetune_params):
#             aggregation_type = self.model_manager.get_aggregation_type(finetune_params)
#             X_train, y_train = dataset.processed().train_data(aggregation_type=aggregation_type)
#             X_val, y_val = dataset.processed().test_data(aggregation_type=aggregation_type)

#             def build_model(hp):
#                 model_params = {}
#                 for key, values in finetune_params.get('model_params', {}).items():
#                     if isinstance(values, list):
#                         model_params[key] = hp.Choice(key, values)
#                     elif isinstance(values, tuple):
#                         if values[0] == 'int':
#                             model_params[key] = hp.Int(key, values[1], values[2])
#                         elif values[0] == 'float':
#                             model_params[key] = hp.Float(key, values[1], values[2])

#                 # Update the model_config with new parameters
#                 model_config = copy.deepcopy(self.model_config)
#                 if 'model_params' not in model_config:
#                     model_config['model_params'] = {}
#                 model_config['model_params'].update(model_params)

#                 model, _ = ModelBuilderFactory.build_model(model_config)
#                 model.compile(optimizer=finetune_params.get('optimizer', 'adam'),
#                               loss=finetune_params.get('loss', 'mse'),
#                               metrics=finetune_params.get('metrics', ['mse']))
#                 return model

#             tuner = Hyperband(
#                 build_model,
#                 objective="val_loss",
#                 max_epochs=finetune_params.get('max_epochs', 50),
#                 factor=finetune_params.get('factor', 3),
#                 directory=finetune_params.get('directory', 'hyperband_results'),
#                 project_name="hyperband_tuning"
#             )
#             tuner.search(X_train, y_train, validation_data=(X_val, y_val))
#             best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#             # print(f"Best hyperparameters: {best_hps.values}")

#             # Update the model_manager with the best model
#             self.model_manager.model_config['model_params'].update(best_hps.values)
#             best_model = tuner.get_best_models(num_models=1)[0]
#             self.model_manager.model = best_model

#             # Optionally, retrain the best model on the full training data
#             training_params = finetune_params.get('training_params', {})
#             self.model_manager.train(dataset, training_params=training_params)

#             return best_hps.values


# if TORCH_AVAILABLE:
#     import torch
#     import torch.optim as optim
#     from torch.utils.data import DataLoader, TensorDataset

#     class TorchFineTuner(BaseFineTuner):
#         def __init__(self, model_manager):
#             super().__init__(model_manager)
#             self.model_config = copy.deepcopy(model_manager.model_config)

#         def finetune(self, dataset, finetune_params):
#             aggregation_type = self.model_manager.get_aggregation_type(finetune_params)
#             X_train, y_train = dataset.processed().train_data(aggregation_type=aggregation_type)
#             X_val, y_val = dataset.processed().test_data(aggregation_type=aggregation_type)

#             model_params_list = []
#             training_params_list = []

#             # Prepare grid of parameters
#             model_params_grid = {}
#             for key, values in finetune_params.get('model_params', {}).items():
#                 if isinstance(values, list):
#                     model_params_grid[key] = values
#                 elif isinstance(values, tuple):
#                     if values[0] == 'int':
#                         model_params_grid[key] = list(range(values[1], values[2] + 1))
#                     elif values[0] == 'float':
#                         model_params_grid[key] = np.linspace(values[1], values[2], num=5)

#             from itertools import product
#             param_keys = list(model_params_grid.keys())
#             param_values = list(model_params_grid.values())
#             param_combinations = [dict(zip(param_keys, v)) for v in product(*param_values)]

#             best_val_loss = float('inf')
#             best_params = None
#             best_model_state = None

#             for params in param_combinations:
#                 # Update model_config with current params
#                 model_config = copy.deepcopy(self.model_config)
#                 if 'model_params' not in model_config:
#                     model_config['model_params'] = {}
#                 model_config['model_params'].update(params)

#                 # Build the model
#                 model, _ = ModelBuilderFactory.build_model(model_config)
#                 # Create a new ModelManager instance with the updated model
#                 model_manager_class = type(self.model_manager)
#                 temp_model_manager = model_manager_class(model=model, model_config=model_config)
#                 temp_model_manager.framework = 'pytorch'

#                 # Train the model
#                 training_params = finetune_params.get('training_params', {})
#                 temp_model_manager.train(dataset, training_params=training_params)

#                 # Validate the model
#                 y_pred = temp_model_manager.predict(dataset)
#                 y_true = y_val
#                 metrics = temp_model_manager.evaluate(y_true, y_pred)
#                 val_loss = metrics['mse']

#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     best_params = params
#                     best_model_state = temp_model_manager.model.state_dict()

#             # Update the model_manager with the best model
#             self.model_manager.model_config['model_params'].update(best_params)
#             best_model, _ = ModelBuilderFactory.build_model(self.model_manager.model_config)
#             best_model.load_state_dict(best_model_state)
#             self.model_manager.model = best_model

#             # Optionally, retrain the best model on the full training data
#             self.model_manager.train(dataset, training_params=finetune_params.get('training_params', {}))

#             return best_params

# Example Configurations for Fine-Tuners
#
# Optuna Fine-Tuner Config
# finetune_params = {
#     'model_params': {
#         'n_components': ('int', 1, 10),
#         'hidden_layers': [1, 2, 3, 4],
#         'activation': ['relu', 'tanh', 'sigmoid']
#     },
#     'training_params': {
#         'epochs': ('int', 10, 100),
#         'batch_size': [16, 32, 64],
#         'learning_rate': ('float', 1e-5, 1e-2),
#         'optimizer': ['adam', 'sgd']
#     },
#     'n_trials': 20
# }
#
# Hyperband Fine-Tuner Config
# finetune_params = {
#     'model_params': {
#         'n_components': ('int', 2, 15),
#         'dropout_rate': ('float', 0.1, 0.5)
#     },
#     'training_params': {
#         'epochs': ('int', 5, 50),
#         'batch_size': [32, 64, 128],
#         'learning_rate': ('float', 1e-4, 1e-1)
#     },
#     'max_epochs': 50,
#     'factor': 3,
#     'directory': 'hyperband_results',
#     'project_name': 'hyperband_tuning'
# }
#
# Sklearn Fine-Tuner Config
# finetune_params = {
#     'model_params': {
#         'n_estimators': ('int', 50, 150),
#         'max_depth': [3, 5, 10],
#         'learning_rate': ('float', 0.01, 0.2)
#     },
#     'approach': 'grid',
#     'cv': 5,
#     'n_trials': 10,
#     'n_jobs': -1,
#     'training_params': {
#         'epochs': 100,
#         'batch_size': 32
#     }
# }
#
# Torch Fine-Tuner Config
# finetune_params = {
#     'model_params': {
#         'num_layers': [1, 2, 3],
#         'units_per_layer': ('int', 16, 128)
#     },
#     'training_params': {
#         'learning_rate': ('float', 1e-5, 1e-2),
#         'batch_size': [16, 32, 64],
#         'epochs': 20
#     }
# }
