# model_manager.py
from abc import ABC, abstractmethod
import joblib
from sklearn.metrics import get_scorer, mean_gamma_deviance
import sys
import os
import numpy as np
from .model_builder_factory import ModelBuilderFactory
from ..utils import TF_AVAILABLE, TORCH_AVAILABLE


METRIC_ABBREVIATIONS = {
    # Regression metrics
    "ev": "explained_variance",                       # explained_variance_scorer
    "r2": "r2",                                       # r2_scorer
    "max_err": "max_error",                           # max_error_scorer
    "mae": "neg_mean_absolute_error",                 # neg_mean_absolute_error_scorer
    "mape": "neg_mean_absolute_percentage_error",     # neg_mean_absolute_percentage_error_scorer
    "mse": "neg_mean_squared_error",                  # neg_mean_squared_error_scorer
    "msle": "neg_mean_squared_log_error",             # neg_mean_squared_log_error_scorer
    "rmse": "neg_root_mean_squared_error",            # neg_root_mean_squared_error_scorer
    "rmsle": "neg_root_mean_squared_log_error",       # neg_root_mean_squared_log_error_scorer
    "median_ae": "neg_median_absolute_error",         # neg_median_absolute_error_scorer
    "poisson_dev": "neg_mean_poisson_deviance",       # neg_mean_poisson_deviance_scorer
    "gamma_dev": "neg_mean_gamma_deviance",           # neg_mean_gamma_deviance_scorer
    "d2_ae": "d2_absolute_error_score",               # d2_absolute_error_scorer

    # Classification metrics
    "acc": "accuracy",                                # accuracy_scorer
    "top_k_acc": "top_k_accuracy",                    # top_k_accuracy_scorer
    "roc_auc": "roc_auc",                             # roc_auc_scorer
    "roc_auc_ovr": "roc_auc_ovr",                     # roc_auc_ovr_scorer
    "roc_auc_ovo": "roc_auc_ovo",                     # roc_auc_ovo_scorer
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",   # roc_auc_ovr_weighted_scorer
    "roc_auc_ovo_weighted": "roc_auc_ovo_weighted",   # roc_auc_ovo_weighted_scorer
    "balanced_acc": "balanced_accuracy",              # balanced_accuracy_scorer
    "ap": "average_precision",                        # average_precision_scorer
    "log_loss": "neg_log_loss",                       # neg_log_loss_scorer
    "brier": "neg_brier_score",                       # neg_brier_score_scorer
    "plr": "positive_likelihood_ratio",               # positive_likelihood_ratio_scorer
    "nlr": "neg_negative_likelihood_ratio",           # neg_negative_likelihood_ratio_scorer

    # Cluster metrics (supervised)
    "adj_rand": "adjusted_rand_score",                # adjusted_rand_scorer
    "rand": "rand_score",                             # rand_scorer
    "homogeneity": "homogeneity_score",               # homogeneity_scorer
    "completeness": "completeness_score",             # completeness_scorer
    "v_measure": "v_measure_score",                   # v_measure_scorer
    "mutual_info": "mutual_info_score",               # mutual_info_scorer
    "adj_mutual_info": "adjusted_mutual_info_score",  # adjusted_mutual_info_scorer
    "nmi": "normalized_mutual_info_score",            # normalized_mutual_info_scorer
    "fmi": "fowlkes_mallows_score",                   # fowlkes_mallows_scorer
}


def detect_task_type(loss, metrics):
    classification_losses = {'binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy'}
    classification_metrics = {'accuracy', 'acc', 'precision', 'recall', 'f1', 'auc'}

    # Check if loss is for classification
    if loss in classification_losses:
        return 'classification'

    # Check if any of the metrics are for classification
    if any(metric in classification_metrics for metric in metrics):
        return 'classification'

    # Default to regression
    return 'regression'


def prepare_y(y_train, y_val, model, framework, loss, task):
    """
    Prepares y_train and y_val based on the loss function, model type (Keras or scikit-learn),
    and the task (classification, regression, etc.). Adjusts the model if necessary and counts num_classes.

    Parameters:
        y_train (array-like): Training labels (n_samples,) or (n_samples, 1) shape.
        y_val (array-like): Validation labels (n_samples,) or (n_samples, 1) shape.
        model (object): The model instance (Keras or scikit-learn).
        framework (str): The framework of the model ('tensorflow' or 'sklearn').
        loss (str): The loss function being used (e.g., 'categorical_crossentropy', 'poisson', 'kl_divergence', etc.).
        task (str): The task type ('classification', 'regression', 'clustering', etc.).

    Returns:
        y_train (array-like): Modified training labels.
        y_val (array-like): Modified validation labels.
        model (object): Updated model if necessary.
        num_classes (int or None): Number of classes (if applicable).

    Raises:
        ValueError: If labels are inappropriate for the specified loss or task.
    """
    # Ensure labels are numpy arrays and flatten them
    y_train = np.array(y_train).reshape(-1)
    y_val = np.array(y_val).reshape(-1)
    num_classes = None  # Initialize num_classes to None

    if task == 'classification':
        # Count the number of classes
        classes = np.unique(y_train)
        num_classes = len(classes)

        # Validate labels
        if num_classes < 2:
            raise ValueError("At least two classes are required for classification.")

        # Framework-specific processing
        if framework == 'tensorflow':
            import tensorflow
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.utils import to_categorical

            if loss == 'sparse_categorical_crossentropy':
                # Labels must be integer-encoded
                if not np.issubdtype(y_train.dtype, np.integer):
                    raise ValueError("Labels must be integer-encoded for sparse_categorical_crossentropy.")

                # Check label range
                if np.min(y_train) < 0 or np.max(y_train) >= num_classes:
                    raise ValueError(f"Labels must be in the range [0, {num_classes - 1}] for sparse_categorical_crossentropy.")

                # Adjust model's last layer
                from tensorflow.keras.layers import Dense
                last_layer = model.layers[-1]
                if last_layer.units != num_classes or last_layer.activation.__name__ != 'softmax':
                    model.pop()
                    model.add(Dense(num_classes, activation='softmax'))

            elif loss == 'categorical_crossentropy':
                # Convert labels to one-hot encoding
                y_train = to_categorical(y_train, num_classes=num_classes)
                y_val = to_categorical(y_val, num_classes=num_classes)

                # Adjust model's last layer
                last_layer = model.layers[-1]
                if last_layer.units != num_classes or last_layer.activation.__name__ != 'softmax':
                    model.pop()
                    model.add(Dense(num_classes, activation='softmax'))

            elif loss == 'binary_crossentropy':
                # Ensure labels are binary (0 or 1)
                unique_labels = np.unique(y_train)
                if not set(unique_labels).issubset({0, 1}):
                    raise ValueError("Labels must be binary (0 or 1) for binary_crossentropy.")

                # Adjust model's last layer
                last_layer = model.layers[-1]
                if last_layer.units != 1 or last_layer.activation.__name__ != 'sigmoid':
                    model.pop()
                    model.add(Dense(1, activation='sigmoid'))

            else:
                raise ValueError(f"Unsupported loss '{loss}' for classification with TensorFlow.")

        elif framework == 'sklearn':
            # Encode labels as integers (if not already)
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val = le.transform(y_val)
            num_classes = len(le.classes_)
            # No model adjustments needed for scikit-learn classifiers

        else:
            raise ValueError("Unsupported framework. Choose 'tensorflow' or 'sklearn'.")

    elif task == 'regression':
        # Labels should be continuous values
        if framework == 'tensorflow':
            from tensorflow.keras.layers import Dense

            # Convert labels to float32
            y_train = y_train.astype(np.float32)
            y_val = y_val.astype(np.float32)

            # Adjust model's last layer based on loss function
            if loss in ['mean_squared_error', 'mse', 'mean_absolute_error', 'mae']:
                # Standard regression
                last_layer = model.layers[-1]
                if last_layer.units != 1 or last_layer.activation.__name__ != 'linear':
                    model.pop()
                    model.add(Dense(1, activation='linear'))

            elif loss == 'poisson':
                # Poisson regression (counts)
                last_layer = model.layers[-1]
                if last_layer.units != 1 or last_layer.activation.__name__ != 'exponential':
                    model.pop()
                    model.add(Dense(1, activation='exponential'))

            else:
                raise ValueError(f"Unsupported loss '{loss}' for regression with TensorFlow.")

        elif framework == 'sklearn':
            # No changes needed for scikit-learn regressors
            pass

        else:
            raise ValueError("Unsupported framework. Choose 'tensorflow' or 'sklearn'.")

    elif task == 'clustering':
        # Clustering tasks typically do not involve supervised labels in the same way
        # For unsupervised clustering, labels may not be provided or used
        # Here we can raise a warning or handle accordingly
        print("Warning: Clustering tasks are unsupervised. Labels may not be used.")

    elif task == 'density_estimation':
        # For tasks like density estimation using KL divergence
        if framework == 'tensorflow':
            from tensorflow.keras.layers import Dense

            # For KL divergence, the labels should represent probability distributions
            # Ensure labels are valid probability distributions
            if not np.allclose(y_train.sum(axis=1), 1):
                raise ValueError("Labels must be valid probability distributions (sum to 1) for KL divergence.")

            # Adjust model's last layer
            last_layer = model.layers[-1]
            if last_layer.units != y_train.shape[1] or last_layer.activation.__name__ != 'softmax':
                model.pop()
                model.add(Dense(y_train.shape[1], activation='softmax'))

        else:
            raise ValueError("KL divergence is typically used with TensorFlow models.")

    else:
        raise ValueError(f"Unsupported task '{task}'. Supported tasks are 'classification', 'regression', 'clustering', 'density_estimation'.")

    return y_train, y_val, model, num_classes


class BaseModelManager(ABC):

    def __init__(self, models=None, model_config=None):
        self.models = models
        self.model_config = model_config
        self.framework = 'framework undefined'

    @staticmethod
    def evaluate(y_true, y_pred, metrics):
        """
        Evaluate a list of metrics on given true and predicted values.

        Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        metrics (list): A list of metrics to evaluate. Each metric can be:
                        - a string (sklearn metric name or abbreviation),
                        - a class instance with an `evaluate` method,
                        - a function that takes (y_true, y_pred).

        Returns:
        dict: A dictionary with metric names as keys and their computed scores as values.
        
        """
        scores = {}
        
        for metric in metrics:
            # If the metric is a string, map abbreviation to sklearn's full metric name
            if isinstance(metric, str):
                full_metric = METRIC_ABBREVIATIONS.get(metric, metric)  # Use full name if found, else the original string
                try:
                    scorer = get_scorer(full_metric)
                    score = scorer._score_func(y_true, y_pred)
                    scores[metric] = score  # Keep the original metric abbreviation or name
                except Exception as e:
                    scores[metric] = f"Error: {e}"

            # If the metric is a class instance with an `evaluate` method
            elif hasattr(metric, 'evaluate') and callable(metric.evaluate):
                try:
                    score = metric.evaluate(y_true, y_pred)
                    scores[metric.__class__.__name__] = score
                except Exception as e:
                    scores[metric.__class__.__name__] = f"Error: {e}"

            # If the metric is a callable function
            elif callable(metric):
                try:
                    score = metric(y_true, y_pred)
                    scores[metric.__name__] = score
                except Exception as e:
                    scores[metric.__name__] = f"Error: {e}"

            # If the metric is unrecognized, log an error
            else:
                scores[str(metric)] = "Unsupported metric type"

        return scores

    # def evaluate(self, y_true, y_pred, metrics=['mse', 'r2']):
    #     for metric in metrics:
    #         if isinstance(metric, str):
    #             # test if is a sklearn metric in the module sklearn.metrics
    #             if hasattr(sys.modules[__name__], metric):
    #                 metric = getattr(sys.modules[__name__], metric)
    #             else:
    #                 raise ValueError(f"Metric {metric} not found in sklearn.metrics module")
    #         elif inspect.isfunction(metric):
    #     return {"mse": float(mse), "r2": float(r2)}

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for i, model in enumerate(self.models):
            joblib.dump(model, path + f'_{i}.pkl')
        
    def load_model(self, path):
        if os.path.exists(path):
            self.models = []
            for i in range(len(os.listdir(path))):
                if os.path.exists(path + f'model_{i}.pkl'):
                    self.models.append(joblib.load(path + f'model_{i}.pkl'))
        else:
            raise FileNotFoundError(f"Model file not found at: {path}")

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, training_params, input_dim, metrics, no_folds):
        pass

    @abstractmethod
    def predict(self, X, task, return_all, no_folds, raw_class_output):
        pass


if TF_AVAILABLE:
    import tensorflow as tf
    from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler
    from tensorflow.keras.models import load_model
    from tensorflow.keras import metrics as tf_metrics
    from tensorflow.keras.utils import to_categorical

    # Abbreviations mapping for TensorFlow/Keras metrics
    TF_METRIC_ABBREVIATIONS = {
        # Regression metrics
        "mae": tf_metrics.MeanAbsoluteError(),           # Mean Absolute Error
        "mse": tf_metrics.MeanSquaredError(),            # Mean Squared Error
        "msle": tf_metrics.MeanSquaredLogarithmicError(),  # Mean Squared Logarithmic Error
        "rmse": tf_metrics.RootMeanSquaredError(),       # Root Mean Squared Error
        "mape": tf_metrics.MeanAbsolutePercentageError(),  # Mean Absolute Percentage Error
        # "r2": tf_metrics.R2Score(),                      # R-squared

        # Classification metrics
        "acc": tf_metrics.Accuracy(),                    # Accuracy
        "accuracy": tf_metrics.Accuracy(),
        "auc": tf_metrics.AUC(),                         # AUC (ROC)
        # "f1": tf_metrics.F1Score(),                      # F1 Score
        "precision": tf_metrics.Precision(),             # Precision
        "recall": tf_metrics.Recall(),                   # Recall
        "log_loss": tf_metrics.BinaryCrossentropy(),     # Binary Crossentropy
    }
    
    def get_keras_metric(metric_names):
        res_metrics = []
        for metric_name in metric_names:
            if isinstance(metric_name, str):
                if metric_name in TF_METRIC_ABBREVIATIONS:
                    res_metrics.append(TF_METRIC_ABBREVIATIONS[metric_name])
                # else:
                #     try:
                #         res_metrics.append(tf_metrics.get(metric_name))
                #     except ValueError:
                #         raise ValueError(f"Unknown metric '{metric_name}'")

            elif callable(metric_name):
                res_metrics.append(metric_name)
            else:
                raise ValueError(f"Unsupported metric type: {metric_name}")
        return res_metrics

    class BestModelMemory(Callback):
        def __init__(self):
            super(BestModelMemory, self).__init__()
            self.best_weights = None
            self.best_val_loss = np.inf

        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = self.model.get_weights()

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    class TFModelManager(BaseModelManager):
        def __init__(self, models=None, model_config=None):
            super(TFModelManager, self).__init__(models, model_config)
            self.framework = 'tensorflow'
        
        def train(self, dataset, training_params, input_dim=None, metrics=None, no_folds=False):
            # models = [self.model]
            models = [self.models[0]] if no_folds else self.models
            for (x_train, y_train, x_val, y_val), model in zip(dataset.fold_data('union', no_folds), models):
                print(f"Training fold with shapes: {x_train.shape}, {y_train.shape}, {x_val.shape}, {y_val.shape}")
                loss = training_params.get('loss', 'mse')

                # --- ENFORCE CORRECT LOSS FOR TENSORFLOW CLASSIFICATION ---
                # Detect task and num_classes using prepare_y logic
                # (We need to know if this is classification and how many classes)
                task = training_params.get('task', None)
                # If not explicitly set, try to infer from loss/metrics
                if task is None:
                    task = detect_task_type(loss, metrics or [])
                # Use the same logic as prepare_y to count classes
                num_classes = len(np.unique(y_train)) if task == 'classification' else None
                # Only override for TensorFlow classification
                if task == 'classification':
                    if num_classes == 2:
                        enforced_loss = 'binary_crossentropy'
                    else:
                        enforced_loss = 'sparse_categorical_crossentropy'
                    if loss != enforced_loss:
                        print(f"[INFO] Overriding loss '{loss}' with '{enforced_loss}' for TensorFlow classification (num_classes={num_classes})")
                    loss = enforced_loss

                # Reshape labels for sparse categorical cross-entropy
                if loss == 'sparse_categorical_crossentropy':
                    y_train = y_train.reshape(-1)  # Flatten to (n_samples,)
                    y_val = y_val.reshape(-1)
                    y_train, y_val = np.array(y_train, dtype=np.int32), np.array(y_val, dtype=np.int32)

                print(loss, metrics)

                x_train = tf.convert_to_tensor(x_train)
                y_train = tf.convert_to_tensor(y_train)
                x_val = tf.convert_to_tensor(x_val)
                y_val = tf.convert_to_tensor(y_val)

                print("Training with shapes:", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
                model.compile(optimizer=training_params.get('optimizer', 'adam'),
                              loss=loss,
                              metrics=metrics)

                callbacks = []
                if training_params.get('early_stopping', True):
                    callbacks.append(EarlyStopping(monitor='val_loss', patience=training_params.get('patience', 50)))

                if training_params.get('cyclic_lr', False):
                    base_lr = training_params.get('base_lr', 1e-4)
                    max_lr = training_params.get('max_lr', 1e-2)
                    step_size = training_params.get('step_size', 2000)

                    def cyclic_lr(epoch):
                        cycle = np.floor(1 + epoch / (2 * step_size))
                        x = np.abs(epoch / step_size - 2 * cycle + 1)
                        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
                        return lr

                    callbacks.append(LearningRateScheduler(cyclic_lr))

                # Add BestModelMemory callback to keep best weights in memory
                callbacks.append(BestModelMemory())

                # model.summary()

                model.fit(x_train, y_train,
                          validation_data=(x_val, y_val),
                          epochs=training_params.get('epochs', 100),
                          batch_size=training_params.get('batch_size', 32),
                          callbacks=callbacks,
                          verbose=training_params.get('verbose', 1))

        def predict(self, dataset, task, return_all=False, no_folds=False, raw_class_output=False):
            y_pred = None
            if len(self.models) == 1 or no_folds:
                y_pred = self.models[0].predict(dataset.x_test_('union'))
                if task == 'classification' and not raw_class_output:
                    return np.argmax(y_pred, axis=1)
                else:
                    return y_pred
            else:
                y_preds = [model.predict(dataset.x_test_('union')) for model in self.models]
                if task == 'classification':
                    if return_all:
                        if raw_class_output:
                            return y_preds  # Return raw probabilities/logits for each fold
                        y_preds = [np.argmax(y_pred, axis=1) for y_pred in y_preds]
                        return y_preds
                    else:
                        y_preds = np.mean(y_preds, axis=0)
                        return np.argmax(y_preds, axis=1) if not raw_class_output else y_preds
                else:
                    return y_preds if return_all else np.mean(y_preds, axis=0)
            
        def save_model(self, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            for i, model in enumerate(self.models):
                model.save(path + f'_{i}.keras')

        def load_model(self, path):
            if os.path.exists(path):
                self.models = []
                for i in range(len(os.listdir(path))):
                    if os.path.exists(path + f'model_{i}.keras'):
                        self.models.append(load_model(path + f'model_{i}.keras'))
            else:
                raise FileNotFoundError(f"Model file not found at: {path}")


class SklearnModelManager(BaseModelManager):
    def __init__(self, models, model_config):
        super(SklearnModelManager, self).__init__(models, model_config)
        self.framework = 'sklearn'
        
    def train(self, dataset, training_params, metrics=None, no_folds=False):
        i = 0
        models = [self.models[0]] if no_folds else self.models
        for (x_train, y_train, x_val, y_val), model in zip(dataset.fold_data('concat', no_folds), models):
            print(f"Training fold {i + 1}, with shapes:", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
            i += 1
            if 'accuracy' in metrics:
                if y_train.ndim > 1:
                    y_train = y_train.ravel()
                    y_val = y_val.ravel()
            model.fit(x_train, y_train)
            # if x_val is not None and y_val is not None:
            #     y_pred = model.predict(x_val)
            #     metrics = BaseModelManager.evaluate(y_val, y_pred, training_params.get('metrics', ['mse', 'r2']))
            #     print(f"Validation Metrics: {metrics}")

    def predict(self, dataset, task=None, return_all=False, no_folds=False, raw_class_output=False):
        if len(self.models) == 1 or no_folds:
            return self.models[0].predict(dataset.x_test_('concat'))
        else:
            y_preds = [model.predict(dataset.x_test_('concat')) for model in self.models]
            if return_all:
                return y_preds
            
            return np.mean(y_preds, axis=0)

class ModelManagerFactory:
    
    @staticmethod
    def get_model_manager(model_config, dataset, task):
        
        models, framework = ModelBuilderFactory.build_models(model_config, dataset, task)
        print("Using framework:", framework)

        if framework == 'tensorflow' and TF_AVAILABLE:
            return TFModelManager(models, model_config)
        elif framework == 'sklearn':
            return SklearnModelManager(models, model_config)
        # elif framework == 'pytorch' and TORCH_AVAILABLE:
            # return TorchModelManager(model, callable_model, model_params, model_config)
        else:
            raise ValueError(f"Unsupported framework or framework not available in the environment: {framework}")


# if TORCH_AVAILABLE:
    # import torch
    # import torch.optim as optim
    # from torch.utils.data import DataLoader, TensorDataset

    # class TorchModelManager(BaseModelManager):
    #     def __init__(self, model=None, callable_model=None, model_params=None, model_config=None):
    #         super(TorchModelManager, self).__init__(model, callable_model, model_params, model_config)
    #         self.framework = 'pytorch'

    #     def train(self, dataset, training_params):
    #         aggregation_type = self.get_aggregation_type(training_params)
    #         X_train, y_train = dataset.processed().train_data(aggregation_type=aggregation_type)
    #         X_val, y_val = dataset.processed().test_data(aggregation_type=aggregation_type)
    #         input_dim = X_train.shape[1]
    #         if self.model is None:
    #             self.instantiate_model(input_dim)
    #         # PyTorch training code
    #         self.model.train()
    #         optimizer = optim.Adam(self.model.parameters(), lr=training_params.get('base_lr', 1e-4))
    #         criterion = torch.nn.MSELoss()
    #         batch_size = training_params.get('batch_size', 32)
    #         epochs = training_params.get('epochs', 100)

    #         print("Training with shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    #         train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    #         val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    #         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #         best_val_loss = float('inf')
    #         best_weights = None

    #         for epoch in range(epochs):
    #             self.model.train()
    #             for X_batch, y_batch in train_loader:
    #                 optimizer.zero_grad()
    #                 outputs = self.model(X_batch)
    #                 loss = criterion(outputs, y_batch)
    #                 loss.backward()
    #                 optimizer.step()

    #             # Validation phase
    #             self.model.eval()
    #             val_loss = 0.0
    #             with torch.no_grad():
    #                 for X_batch, y_batch in val_loader:
    #                     outputs = self.model(X_batch)
    #                     loss = criterion(outputs, y_batch)
    #                     val_loss += loss.item()
    #             val_loss /= len(val_loader)

    #             # Update best weights
    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss
    #                 best_weights = self.model.state_dict()

    #             print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

    #         # Load best weights
    #         if best_weights is not None:
    #             self.model.load_state_dict(best_weights)

    #     def predict(self, dataset):
    #         aggregation_type = self.get_aggregation_type()
    #         X = dataset.processed().x_test(aggregation_type=aggregation_type)
    #         self.model.eval()
    #         with torch.no_grad():
    #             return self.model(torch.Tensor(X)).numpy()
