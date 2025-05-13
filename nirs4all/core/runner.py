# runner.py

import os
import numpy as np

from ..data.dataset_loader import get_dataset
from .processor import run_pipeline
from nirs4all.core.finetuner.base_finetuner import FineTunerFactory
from .model.model_manager import ModelManagerFactory
from .manager.experiment_manager import ExperimentManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ExperimentRunner:
    def __init__(self, configs, results_dir='results', resume_mode='skip', verbose=0):
        self.configs = configs
        self.manager = ExperimentManager(results_dir, resume_mode, verbose=3)
        self.logger = self.manager.logger
        self.results_dir = results_dir
        self.cache = {}

    def _run_config(self, config):
        self.cache = {}
        self.logger.info("=" * 80)
        self.logger.info("### LOADING DATASET ###")
        dataset = get_dataset(config.dataset)
        self.logger.info(dataset)

        self.logger.info("### PROCESSING DATASET ###")
        dataset = run_pipeline(dataset, config.x_pipeline, config.y_pipeline, self.logger, self.cache)
        self.logger.info(dataset)
        
        # len of unique classes for y_train merged with y_test
        dataset.num_classes = len(np.unique(np.concatenate([dataset.y_train_init, dataset.y_test_init])))
        
        action, metrics, training_params, finetune_params, task = config.validate(dataset)
        
        self.logger.info("### PREPARING MODEL ###")
        model_manager = None
        model_config = config.model
        if model_config is not None:
            model_manager = ModelManagerFactory.get_model_manager(model_config, dataset, task)
        
        self.logger.info("Running config > %s", self.manager.make_config_serializable(config))
        self.manager.prepare_experiment(config)
        
        preds, scores, best_params = None, None, None
        if model_manager is None:
            self.logger.warning("No model manager found. Skipping training/prediction.")
            return dataset, preds, scores, best_params

        if action == 'predict':
            preds, scores, best_params = self._predict(model_manager, dataset, metrics, task)
        elif action == 'train':
            preds, scores, best_params = self._train(model_manager, dataset, training_params, metrics, task)
        elif action == 'finetune':
            preds, scores, best_params = self._fine_tune(model_manager, dataset, finetune_params, training_params, metrics, task)
    
        return dataset, preds, scores, best_params

    def run(self):
        if not isinstance(self.configs, list):
            self.configs = [self.configs]

        datasets, predictions, scores, best_params = [], [], [], []

        for config in self.configs:
            self.logger.info("=" * 80)
            self.logger.info("Running config: %s", config)
            dataset_, preds_, scores_, best_params_ = self._run_config(config)

            datasets.append(dataset_)
            predictions.append(preds_)
            scores.append(scores_)
            best_params.append(best_params_)

        self.logger.info("All experiments completed.")
        return datasets, predictions, scores, best_params
    
    def _evaluate_and_save_results(self, model_manager, dataset, metrics, best_params=None, task=None):
        # Request raw outputs for classification if averaging across folds
        y_pred = model_manager.predict(dataset, task, return_all=True, raw_class_output=(task == 'classification'))
        y_true = dataset.y_test_init

        if isinstance(y_pred, list):
            raw_preds = []          # To store raw model outputs (probabilities or logits)
            fold_scores = []        # To store per-fold evaluation scores

            for y_pred_i in y_pred:
                raw_preds.append(y_pred_i)  # Store raw predictions for averaging later

                # Convert to class predictions if classification
                if task == 'classification':
                    if y_pred_i.ndim > 1:
                        if y_pred_i.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                            # Multi-class classification
                            y_pred_class = np.argmax(y_pred_i, axis=-1)
                        elif y_pred_i.shape[-1] == 1 or dataset.num_classes == 1:
                            # Binary classification with outputs of shape (num_samples, 1)
                            y_pred_class = (y_pred_i >= 0.5).astype(int).flatten()
                        else:
                            raise ValueError("Unexpected output shape for classification task")
                    else:
                        # y_pred_i.ndim == 1, binary classification
                        y_pred_class = (y_pred_i >= 0.5).astype(int)
                else:
                    y_pred_class = y_pred_i

                # Apply inverse transform if necessary (after obtaining class labels)
                y_pred_inverse = dataset.inverse_transform(y_pred_class)

                # Evaluate scores for this fold
                scores = model_manager.evaluate(y_true, y_pred_inverse, metrics)
                fold_scores.append(scores)

            # Compute mean prediction across all folds (probabilities or logits)
            mean_raw_pred = np.mean(np.array(raw_preds), axis=0)
            if task == 'classification':
                if mean_raw_pred.ndim > 1:
                    if mean_raw_pred.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                        # Multi-class classification
                        mean_pred_class = np.argmax(mean_raw_pred, axis=-1)
                    elif mean_raw_pred.shape[-1] == 1 or dataset.num_classes == 1:
                        # Binary classification
                        mean_pred_class = (mean_raw_pred >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected output shape for classification task")
                else:
                    # mean_raw_pred.ndim == 1, binary classification
                    mean_pred_class = (mean_raw_pred >= 0.5).astype(int)
                mean_pred_inverse = dataset.inverse_transform(mean_pred_class)
            else:
                mean_pred_inverse = dataset.inverse_transform(mean_raw_pred)

            # Identify the best fold based on the first metric
            metric_to_use = metrics[0]  # Adjust if needed

            # Ensure fold_scores_values contains numeric values
            fold_scores_values = np.array([
                fold_score.get(metric_to_use, 0)
                for fold_score in fold_scores
            ])

            if task == 'classification':
                best_fold_index = np.argmax(fold_scores_values)  # Higher is better
            else:
                best_fold_index = np.argmin(fold_scores_values)  # Lower is better

            # Get the best fold's predictions
            best_raw_pred = raw_preds[best_fold_index]
            if task == 'classification':
                if best_raw_pred.ndim > 1:
                    if best_raw_pred.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                        best_pred_class = np.argmax(best_raw_pred, axis=-1)
                    elif best_raw_pred.shape[-1] == 1 or dataset.num_classes == 1:
                        best_pred_class = (best_raw_pred >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected output shape for classification task")
                else:
                    best_pred_class = (best_raw_pred >= 0.5).astype(int)
                best_pred_inverse = dataset.inverse_transform(best_pred_class)
            else:
                best_pred_inverse = dataset.inverse_transform(best_raw_pred)
            best_scores = fold_scores[best_fold_index]

            # Compute weighted mean prediction using fold scores as weights
            if task == 'classification':
                # For classification, higher scores are better
                min_score = np.min(fold_scores_values)
                if min_score < 0:
                    fold_scores_values -= min_score  # Adjust to be non-negative
                total_score = np.sum(fold_scores_values)
                weights = fold_scores_values / total_score if total_score > 0 else np.ones_like(fold_scores_values) / len(fold_scores_values)
            else:
                mse_array = np.array(fold_scores_values, dtype=float)
                print("MSE array:", mse_array)
                # Handle potential division by zero by adding a small epsilon if necessary
                epsilon = 1e-8
                mse_array = mse_array + epsilon  # Avoid division by zero

                # Compute inverse of MSEs
                inverse_mse = 1.0 / mse_array

                # Normalize the inverses to sum to 1
                weights = inverse_mse / np.sum(inverse_mse)

                # total_inverted_score = np.sum(inverted_scores)
                # weights = inverted_scores / total_inverted_score if total_inverted_score > 0 else np.ones_like(fold_scores_values) / len(fold_scores_values)

            print("Weights:", weights)

            # Compute weighted average of raw predictions
            weighted_raw_pred = np.average(np.array(raw_preds), axis=0, weights=weights)
            if task == 'classification':
                if weighted_raw_pred.ndim > 1:
                    if weighted_raw_pred.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                        weighted_pred_class = np.argmax(weighted_raw_pred, axis=-1)
                    elif weighted_raw_pred.shape[-1] == 1 or dataset.num_classes == 1:
                        weighted_pred_class = (weighted_raw_pred >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected output shape for classification task")
                else:
                    weighted_pred_class = (weighted_raw_pred >= 0.5).astype(int)
                weighted_pred_inverse = dataset.inverse_transform(weighted_pred_class)
            else:
                weighted_pred_inverse = dataset.inverse_transform(weighted_raw_pred)

            # Evaluate mean and weighted predictions
            mean_scores = model_manager.evaluate(y_true, mean_pred_inverse, metrics)
            weighted_scores = model_manager.evaluate(y_true, weighted_pred_inverse, metrics)

            # Collect all predictions and scores
            all_preds = []
            for y in raw_preds:
                if task == 'classification':
                    if y.ndim > 1:
                        if y.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                            y_pred_class = np.argmax(y, axis=-1)
                        elif y.shape[-1] == 1 or dataset.num_classes == 1:
                            y_pred_class = (y >= 0.5).astype(int).flatten()
                        else:
                            raise ValueError("Unexpected output shape for classification task")
                    else:
                        y_pred_class = (y >= 0.5).astype(int)
                    y_pred_inverse = dataset.inverse_transform(y_pred_class)
                else:
                    y_pred_inverse = dataset.inverse_transform(y)
                all_preds.append(y_pred_inverse)

            all_preds.extend([mean_pred_inverse, best_pred_inverse, weighted_pred_inverse])
            all_scores = fold_scores + [mean_scores, best_scores, weighted_scores]

            # Save all results
            self.manager.save_results(model_manager, all_preds, y_true, metrics, best_params, all_scores)

            return all_preds, all_scores, best_params
        else:
            # Handle single prediction case
            if task == 'classification':
                if y_pred.ndim > 1:
                    if y_pred.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                        y_pred_class = np.argmax(y_pred, axis=-1)
                    elif y_pred.shape[-1] == 1 or dataset.num_classes == 1:
                        y_pred_class = (y_pred >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected output shape for classification task")
                else:
                    y_pred_class = (y_pred >= 0.5).astype(int)
                y_pred_inverse = dataset.inverse_transform(y_pred_class)
            else:
                y_pred_inverse = dataset.inverse_transform(y_pred)
            scores = model_manager.evaluate(y_true, y_pred_inverse, metrics)
            self.manager.save_results(model_manager, y_pred_inverse, y_true, metrics, best_params, [scores])
            return y_pred_inverse, [scores], best_params

    def _train(self, model_manager, dataset, training_params, metrics, task):
        self.logger.info("Training the model")
        model_manager.train(dataset, training_params=training_params, metrics=metrics)
        model_manager.save_model(os.path.join(self.manager.experiment_path, "model"))
        self.logger.info("Saved model to %s", self.manager.experiment_path)
        return self._evaluate_and_save_results(model_manager, dataset, metrics, task=task)

    def _predict(self, model_manager, dataset, metrics, task):
        self.logger.info("Predicting using the model")
        return self._evaluate_and_save_results(model_manager, dataset, metrics, task=task)

    def _fine_tune(self, model_manager, dataset, finetune_params, training_params, metrics, task):
        self.logger.info("Finetuning the model")
        finetuner_type = finetune_params.get('tuner', 'optuna')
        finetuner = FineTunerFactory.get_fine_tuner(finetuner_type, model_manager)
        best_params = finetuner.finetune(dataset, finetune_params, training_params, metrics=metrics, task=task)
        model_manager.save_model(os.path.join(self.manager.experiment_path, "model"))
        return self._evaluate_and_save_results(model_manager, dataset, metrics, best_params, task=task)
