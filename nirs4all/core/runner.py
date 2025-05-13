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
    def __init__(self, configs, results_dir='results', resume_mode='restart', verbose=0):
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
        # Request raw outputs from the model
        y_pred_raw_outputs = model_manager.predict(dataset, task, return_all=True, raw_class_output=(task == 'classification'))
        
        # Get the initial true labels (these might be encoded)
        y_true_initial = dataset.y_test_init.ravel()  # Ensure y_true is 1D

        # Prepare y_true for evaluation and saving: Inverse transform if classification
        y_true_for_eval_and_save = y_true_initial
        if task == 'classification':
            if hasattr(dataset, 'inverse_transform') and callable(dataset.inverse_transform):
                y_true_for_eval_and_save = dataset.inverse_transform(y_true_initial)
            else:
                print(("[WARNING] Dataset does not have a callable 'inverse_transform' method. "
                       "y_true for classification might remain in its initial (potentially encoded) form, "
                       "which could affect metrics and saved results if predictions are in original label format."))
        
        # Ensure y_true_for_eval_and_save is consistently 1D array after potential inverse_transform
        if not isinstance(y_true_for_eval_and_save, np.ndarray):
            y_true_for_eval_and_save = np.array(y_true_for_eval_and_save)
        if y_true_for_eval_and_save.ndim > 1 and y_true_for_eval_and_save.shape[1] == 1:
            y_true_for_eval_and_save = y_true_for_eval_and_save.ravel()

        if isinstance(y_pred_raw_outputs, list):  # Handling multiple folds
            raw_preds_model_output_folds = []
            fold_scores = []
            all_preds_inverse_transformed_for_saving = []

            for y_pred_i_raw_output_fold in y_pred_raw_outputs:
                raw_preds_model_output_folds.append(y_pred_i_raw_output_fold)

                y_pred_class_fold = None
                if task == 'classification':
                    if y_pred_i_raw_output_fold.ndim > 1:
                        if y_pred_i_raw_output_fold.shape[-1] == dataset.num_classes and dataset.num_classes > 1:  # Multi-class
                            y_pred_class_fold = np.argmax(y_pred_i_raw_output_fold, axis=-1)
                        elif y_pred_i_raw_output_fold.shape[-1] == 1 or dataset.num_classes == 1:  # Binary
                            y_pred_class_fold = (y_pred_i_raw_output_fold >= 0.5).astype(int).flatten()
                        else:
                            raise ValueError("Unexpected raw prediction output shape for classification task during fold processing.")
                    else:  # Binary, 1D raw output
                        y_pred_class_fold = (y_pred_i_raw_output_fold >= 0.5).astype(int)
                    
                    y_pred_inverse_transformed_fold = dataset.inverse_transform(y_pred_class_fold)
                else:  # Regression
                    # For regression, raw output is the prediction; inverse_transform handles any scaling
                    y_pred_inverse_transformed_fold = dataset.inverse_transform(y_pred_i_raw_output_fold)
                
                all_preds_inverse_transformed_for_saving.append(y_pred_inverse_transformed_fold)
                # Evaluate scores for this fold using original y_true and original y_pred
                current_fold_scores = model_manager.evaluate(y_true_for_eval_and_save, y_pred_inverse_transformed_fold, metrics)
                fold_scores.append(current_fold_scores)

            # Compute mean prediction across all folds (from raw model outputs before class conversion)
            mean_raw_pred = np.mean(np.array(raw_preds_model_output_folds), axis=0)
            mean_pred_class = None
            if task == 'classification':
                if mean_raw_pred.ndim > 1:
                    if mean_raw_pred.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                        mean_pred_class = np.argmax(mean_raw_pred, axis=-1)
                    elif mean_raw_pred.shape[-1] == 1 or dataset.num_classes == 1:
                        mean_pred_class = (mean_raw_pred >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected mean raw prediction output shape for classification task.")
                else:
                    mean_pred_class = (mean_raw_pred >= 0.5).astype(int)
                mean_pred_inverse_transformed = dataset.inverse_transform(mean_pred_class)
            else:  # Regression
                mean_pred_inverse_transformed = dataset.inverse_transform(mean_raw_pred)

            # Identify the best fold based on the first metric
            metric_to_use_for_best_fold = metrics[0]
            # Ensure scores for metric_to_use_for_best_fold are numeric, default if not found
            fold_metric_values = np.array([
                s.get(metric_to_use_for_best_fold, 0 if task == 'classification' else np.inf)
                for s in fold_scores if isinstance(s.get(metric_to_use_for_best_fold), (int, float))
            ])
            if len(fold_metric_values) == 0:  # Fallback if metric not found or not numeric
                fold_metric_values = np.array([0 if task == 'classification' else np.inf] * len(fold_scores))

            best_fold_index = np.argmax(fold_metric_values) if task == 'classification' else np.argmin(fold_metric_values)
            
            best_raw_pred_fold_output = raw_preds_model_output_folds[best_fold_index]
            best_pred_class_fold = None
            if task == 'classification':
                if best_raw_pred_fold_output.ndim > 1:
                    if best_raw_pred_fold_output.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                        best_pred_class_fold = np.argmax(best_raw_pred_fold_output, axis=-1)
                    elif best_raw_pred_fold_output.shape[-1] == 1 or dataset.num_classes == 1:
                        best_pred_class_fold = (best_raw_pred_fold_output >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected best raw prediction output shape for classification task.")
                else:
                    best_pred_class_fold = (best_raw_pred_fold_output >= 0.5).astype(int)
                best_pred_inverse_transformed = dataset.inverse_transform(best_pred_class_fold)
            else:  # Regression
                best_pred_inverse_transformed = dataset.inverse_transform(best_raw_pred_fold_output)
            best_fold_overall_scores = fold_scores[best_fold_index]

            # Weighting logic for weighted average prediction
            weights_for_avg = None
            if task == 'classification':  # Higher score is better
                min_metric_val = np.min(fold_metric_values)
                adjusted_scores_for_weights = fold_metric_values - min_metric_val if min_metric_val < 0 else fold_metric_values
                total_score_for_weights = np.sum(adjusted_scores_for_weights)
                weights_for_avg = adjusted_scores_for_weights / total_score_for_weights if total_score_for_weights > 0 else np.ones_like(fold_metric_values) / len(fold_metric_values)
            else:  # Regression, lower score is better, so invert for weighting
                epsilon = 1e-8
                safe_scores_for_weights = np.maximum(fold_metric_values, epsilon)  # Avoid division by zero or issues with non-positive scores
                inverse_scores_for_weights = 1.0 / safe_scores_for_weights
                sum_inverse_scores = np.sum(inverse_scores_for_weights)
                weights_for_avg = inverse_scores_for_weights / sum_inverse_scores if sum_inverse_scores > 0 else np.ones_like(fold_metric_values) / len(fold_metric_values)
            
            weighted_avg_raw_pred = np.average(np.array(raw_preds_model_output_folds), axis=0, weights=weights_for_avg)
            weighted_avg_pred_class = None
            if task == 'classification':
                if weighted_avg_raw_pred.ndim > 1:
                    if weighted_avg_raw_pred.shape[-1] == dataset.num_classes and dataset.num_classes > 1:
                        weighted_avg_pred_class = np.argmax(weighted_avg_raw_pred, axis=-1)
                    elif weighted_avg_raw_pred.shape[-1] == 1 or dataset.num_classes == 1:
                        weighted_avg_pred_class = (weighted_avg_raw_pred >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected weighted average raw prediction output shape for classification task.")
                else:
                    weighted_avg_pred_class = (weighted_avg_raw_pred >= 0.5).astype(int)
                weighted_avg_pred_inverse_transformed = dataset.inverse_transform(weighted_avg_pred_class)
            else:  # Regression
                weighted_avg_pred_inverse_transformed = dataset.inverse_transform(weighted_avg_raw_pred)

            # Evaluate mean, best, and weighted predictions
            mean_overall_scores = model_manager.evaluate(y_true_for_eval_and_save, mean_pred_inverse_transformed, metrics)
            # best_scores are already calculated as best_fold_overall_scores
            weighted_overall_scores = model_manager.evaluate(y_true_for_eval_and_save, weighted_avg_pred_inverse_transformed, metrics)
            
            all_preds_inverse_transformed_for_saving.extend([mean_pred_inverse_transformed, best_pred_inverse_transformed, weighted_avg_pred_inverse_transformed])
            # The fold_scores list already contains dicts of scores for each fold.
            # best_fold_overall_scores is the dict for the best fold.
            all_scores_to_save = fold_scores + [mean_overall_scores, best_fold_overall_scores, weighted_overall_scores]

            self.manager.save_results(model_manager, all_preds_inverse_transformed_for_saving, y_true_for_eval_and_save, metrics, best_params, all_scores_to_save)
            return all_preds_inverse_transformed_for_saving, all_scores_to_save, best_params

        else:  # Handling single prediction (no folds)
            y_pred_single_raw_output = y_pred_raw_outputs  # This is the direct model output
            
            y_pred_class_single = None
            if task == 'classification':
                if y_pred_single_raw_output.ndim > 1:
                    if y_pred_single_raw_output.shape[-1] == dataset.num_classes and dataset.num_classes > 1:  # Multi-class
                        y_pred_class_single = np.argmax(y_pred_single_raw_output, axis=-1)
                    elif y_pred_single_raw_output.shape[-1] == 1 or dataset.num_classes == 1:  # Binary
                        y_pred_class_single = (y_pred_single_raw_output >= 0.5).astype(int).flatten()
                    else:
                        raise ValueError("Unexpected raw prediction output shape for single classification task.")
                else:  # Binary, 1D raw output
                    y_pred_class_single = (y_pred_single_raw_output >= 0.5).astype(int)
                
                y_pred_inverse_transformed_single = dataset.inverse_transform(y_pred_class_single)
            else:  # Regression
                y_pred_inverse_transformed_single = dataset.inverse_transform(y_pred_single_raw_output)
            
            # Evaluate scores using original y_true and original y_pred
            final_scores = model_manager.evaluate(y_true_for_eval_and_save, y_pred_inverse_transformed_single, metrics)
            self.manager.save_results(model_manager, y_pred_inverse_transformed_single, y_true_for_eval_and_save, metrics, best_params, [final_scores])
            return y_pred_inverse_transformed_single, [final_scores], best_params

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
