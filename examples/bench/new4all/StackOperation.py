import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Dict, Any, Union
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext
from ModelOperation import SklearnModelOperation, TensorFlowModelOperation, TorchModelOperation


class StackOperation(PipelineOperation):
    """
    Generic stacking operation that supports sklearn, TensorFlow, and PyTorch models.

    This implements proper stacking where:    1. Base learners are trained on K-1 folds and predict on the held-out fold
    2. Meta-features are constructed from out-of-fold predictions
    3. Meta-learner is trained on these meta-features
    4. Final predictions use base learners trained on full training set + meta-learner
    """

    def __init__(self,
                 base_learners: List[Union[BaseEstimator, Any]],
                 meta_learner: Optional[Union[BaseEstimator, Any]] = None,
                 cv_folds: int = 5,
                 stratified: bool = True,
                 random_state: int = 42,
                 base_learner_names: Optional[List[str]] = None,
                 base_learner_types: Optional[List[str]] = None):
        """
        Initialize stacking operation.

        Args:
            base_learners: List of models (sklearn, TensorFlow, or PyTorch)
            meta_learner: Meta-learner model (default: LogisticRegression)
            cv_folds: Number of CV folds for generating meta-features
            stratified: Whether to use stratified CV
            random_state: Random state for reproducibility
            base_learner_names: Optional names for base learners
            base_learner_types: Types of base learners ('sklearn', 'tensorflow', 'torch')
        """
        self.base_learners = base_learners
        self.meta_learner = meta_learner if meta_learner is not None else LogisticRegression(random_state=random_state)
        self.cv_folds = cv_folds
        self.stratified = stratified
        self.random_state = random_state
        self.base_learner_names = base_learner_names or [f"base_{i}" for i in range(len(base_learners))]

        # Auto-detect model types if not provided
        if base_learner_types is None:
            self.base_learner_types = []
            for learner in base_learners:
                if hasattr(learner, 'fit') and hasattr(learner, 'predict'):
                    # Check if it's a TensorFlow model
                    if hasattr(learner, 'compile') and hasattr(learner, 'layers'):
                        self.base_learner_types.append('tensorflow')
                    # Check if it's a PyTorch model
                    elif hasattr(learner, 'parameters') and hasattr(learner, 'forward'):
                        self.base_learner_types.append('torch')
                    else:
                        self.base_learner_types.append('sklearn')
                else:
                    self.base_learner_types.append('sklearn')  # Default
        else:
            self.base_learner_types = base_learner_types

        # Fitted models and encoders
        self.fitted_base_learners = []
        self.fitted_meta_learner = None
        self.label_encoders = []  # For TensorFlow models
        self.is_fitted = False

    def execute(self, dataset: SpectraDataset, context: PipelineContext):
        """Execute stacking operation."""

        print(f"Executing stacking with {len(self.base_learners)} base learners and meta-learner")

        # Get training data
        train_view = dataset.select(partition="train", **context.current_filters)
        if len(train_view) == 0:
            raise ValueError("No training data found for stacking")

        X_train = train_view.get_features(concatenate=True)
        y_train = train_view.get_targets("auto")

        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

        # Step 1: Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X_train, y_train, dataset.task_type)
        print(f"Generated meta-features: {meta_features.shape}")

        # Step 2: Train base learners on full training set
        self.fitted_base_learners = []
        self.label_encoders = []

        for i, (base_learner, learner_type) in enumerate(zip(self.base_learners, self.base_learner_types)):
            if learner_type == 'sklearn':
                fitted_learner = clone(base_learner)
                fitted_learner.fit(X_train, y_train)
                self.fitted_base_learners.append(fitted_learner)
                self.label_encoders.append(None)

            elif learner_type == 'tensorflow':
                # Handle TensorFlow model
                fitted_learner = self._train_tensorflow_model(base_learner, X_train, y_train, dataset.task_type)
                self.fitted_base_learners.append(fitted_learner)

            elif learner_type == 'torch':
                # Handle PyTorch model
                fitted_learner = self._train_torch_model(base_learner, X_train, y_train, dataset.task_type)
                self.fitted_base_learners.append(fitted_learner)
                self.label_encoders.append(None)

            print(f"Trained base learner {i+1}/{len(self.base_learners)}: {learner_type} {type(base_learner).__name__}")

        # Step 3: Train meta-learner on meta-features
        self.fitted_meta_learner = clone(self.meta_learner)
        self.fitted_meta_learner.fit(meta_features, y_train)
        print(f"Trained meta-learner: {type(self.meta_learner).__name__}")

        # Step 4: Generate predictions for all partitions
        self._generate_stacking_predictions(dataset, context)

        self.is_fitted = True

    def _generate_meta_features(self, X_train: np.ndarray, y_train: np.ndarray, task_type: str) -> np.ndarray:
        """Generate meta-features using cross-validation predictions for mixed model types."""

        # Set up cross-validation
        if self.stratified and task_type == "classification":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Generate out-of-fold predictions for each base learner
        meta_features = np.zeros((X_train.shape[0], len(self.base_learners)))

        for i, (base_learner, learner_type) in enumerate(zip(self.base_learners, self.base_learner_types)):
            if learner_type == 'sklearn':
                # Use sklearn's cross_val_predict for sklearn models
                if hasattr(base_learner, 'predict_proba') and task_type == "classification":
                    oof_preds = cross_val_predict(
                        clone(base_learner), X_train, y_train,
                        cv=cv, method='predict_proba'
                    )
                    if oof_preds.ndim > 1 and oof_preds.shape[1] > 1:
                        meta_features[:, i] = oof_preds[:, 1] if oof_preds.shape[1] == 2 else np.max(oof_preds, axis=1)
                    else:
                        meta_features[:, i] = oof_preds.ravel()
                else:
                    oof_preds = cross_val_predict(
                        clone(base_learner), X_train, y_train, cv=cv
                    )
                    meta_features[:, i] = oof_preds
            else:
                # Manual cross-validation for non-sklearn models
                oof_preds = np.zeros(X_train.shape[0])

                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train = y_train[train_idx]

                    # Train model on fold
                    if learner_type == 'tensorflow':
                        fold_model = self._train_tensorflow_model(
                            base_learner, X_fold_train, y_fold_train, task_type
                        )
                        fold_preds = self._predict_with_model(
                            fold_model, 'tensorflow', X_fold_val, task_type,
                            self.label_encoders[-1] if self.label_encoders else None
                        )
                    elif learner_type == 'torch':
                        fold_model = self._train_torch_model(
                            base_learner, X_fold_train, y_fold_train, task_type
                        )
                        fold_preds = self._predict_with_model(
                            fold_model, 'torch', X_fold_val, task_type
                        )
                    else:
                        fold_preds = np.zeros(len(val_idx))

                    oof_preds[val_idx] = fold_preds

                meta_features[:, i] = oof_preds

        return meta_features

    def _generate_stacking_predictions(self, dataset: SpectraDataset, context: PipelineContext):
        """Generate stacking predictions for all partitions."""

        partitions = ["train", "val", "test"]
        all_predictions = {}

        for partition in partitions:
            partition_view = dataset.select(partition=partition, **context.current_filters)
            if len(partition_view) == 0:
                continue

            X_partition = partition_view.get_features(concatenate=True)

            # Get base learner predictions
            base_predictions = np.zeros((X_partition.shape[0], len(self.fitted_base_learners)))

            for i, (fitted_learner, learner_type) in enumerate(zip(self.fitted_base_learners, self.base_learner_types)):
                task_type = dataset.task_type if hasattr(dataset, 'task_type') else "classification"
                label_encoder = self.label_encoders[i] if i < len(self.label_encoders) else None

                base_predictions[:, i] = self._predict_with_model(
                    fitted_learner, learner_type, X_partition, task_type, label_encoder
                )

            # Generate meta-learner predictions
            if self.fitted_meta_learner is not None:
                stacking_predictions = self.fitted_meta_learner.predict(base_predictions)
            else:
                stacking_predictions = np.zeros(X_partition.shape[0])

            # Store predictions in context
            model_name = f"stacking_ensemble"
            if model_name not in all_predictions:
                all_predictions[model_name] = {}

            all_predictions[model_name][partition] = {
                'predictions': stacking_predictions,
                'base_predictions': base_predictions,
                'sample_ids': partition_view.sample_ids
            }

        # Store all predictions in context
        for model_name, partition_preds in all_predictions.items():
            context.add_predictions(model_name, partition_preds)

        print(f"Stored stacking predictions for partitions: {list(all_predictions[model_name].keys())}")

    def get_name(self) -> str:
        """Get operation name."""
        return "stacking_ensemble"

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if stacking operation can be executed."""
        # Check if we have training data
        train_view = dataset.select(partition="train", **context.current_filters)
        return len(train_view) > 0

    def _train_tensorflow_model(self, model, X_train, y_train, task_type):
        """Train a TensorFlow model and return fitted model with label encoder."""
        # Create a copy of the model
        import tensorflow as tf
        try:
            from tensorflow.keras.utils import to_categorical
        except ImportError:
            from tensorflow.python.keras.utils import to_categorical

        # Clone the model architecture
        try:
            # Try modern TensorFlow approach
            fitted_model = tf.keras.models.clone_model(model)
            fitted_model.set_weights(model.get_weights())
        except:
            try:
                # Fallback for older TensorFlow versions
                fitted_model = tf.keras.models.model_from_json(model.to_json())
                fitted_model.set_weights(model.get_weights())
            except:
                # Final fallback - create new model with same architecture
                fitted_model = tf.keras.models.Sequential()
                for layer in model.layers:
                    try:
                        fitted_model.add(layer.__class__.from_config(layer.get_config()))
                    except:
                        # If layer cloning fails, use a simple dense layer
                        fitted_model.add(tf.keras.layers.Dense(64, activation='relu'))

        # Compile model with appropriate loss
        if task_type == "classification":
            fitted_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)

            # Train model
            fitted_model.fit(X_train, y_train_encoded, epochs=10, verbose=0, validation_split=0.1)

            # Store label encoder
            self.label_encoders.append(le)
            return fitted_model
        else:
            fitted_model.compile(optimizer='adam', loss='mse')
            y_train_reshaped = y_train.reshape(-1, 1)
            fitted_model.fit(X_train, y_train_reshaped, epochs=10, verbose=0, validation_split=0.1)
            self.label_encoders.append(None)
            return fitted_model

    def _train_torch_model(self, model, X_train, y_train, task_type):
        """Train a PyTorch model and return fitted model."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # Clone the model
        try:
            fitted_model = type(model)()
            fitted_model.load_state_dict(model.state_dict())
        except:
            # Fallback if model cloning fails
            fitted_model = model

        # Prepare data
        X_tensor = torch.FloatTensor(X_train)

        if task_type == "classification":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_train)
            y_tensor = torch.LongTensor(y_encoded)
            loss_fn = nn.CrossEntropyLoss()
        else:
            y_tensor = torch.FloatTensor(y_train)
            loss_fn = nn.MSELoss()

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Train model
        optimizer = optim.Adam(fitted_model.parameters(), lr=0.001)
        fitted_model.train()

        for epoch in range(10):  # Reduced epochs for faster training
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = fitted_model(batch_X)

                if task_type == "classification":
                    loss = loss_fn(outputs, batch_y)
                else:
                    loss = loss_fn(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()

        fitted_model.eval()
        return fitted_model

    def _predict_with_model(self, model, model_type, X, task_type, label_encoder=None):
        """Make predictions with a model of any type."""
        if model_type == 'sklearn':
            if hasattr(model, 'predict_proba') and task_type == "classification":
                preds = model.predict_proba(X)
                return preds[:, 1] if preds.shape[1] == 2 else np.max(preds, axis=1)
            else:
                return model.predict(X)

        elif model_type == 'tensorflow':
            preds = model.predict(X, verbose=0)
            if task_type == "classification":
                if label_encoder is not None:
                    # Return probabilities for the positive class or max probability
                    if preds.shape[1] == 2:
                        return preds[:, 1]
                    else:
                        return np.max(preds, axis=1)
                else:
                    return np.argmax(preds, axis=1)
            else:
                return preds.flatten()

        elif model_type == 'torch':
            import torch
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = model(X_tensor)

                if task_type == "classification":
                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        return probs[:, 1].numpy() if probs.shape[1] == 2 else torch.max(probs, dim=1)[0].numpy()
                    else:
                        return torch.argmax(outputs, dim=1).numpy()
                else:
                    return outputs.squeeze().numpy()

        return np.zeros(X.shape[0])  # Fallback
