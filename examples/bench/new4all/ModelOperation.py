"""
Model operations for different ML frameworks
"""
from abc import ABC, abstractmethod
import numpy as np
from PipelineOperation import PipelineOperation

class ModelOperation(PipelineOperation, ABC):
    """Abstract base class for model operations."""

    def __init__(self, model, model_name=None, target_representation="auto",
                 train_on="train", predict_on=None):
        self.model = model
        self.model_name = model_name
        self.target_representation = target_representation
        self.train_on = train_on
        self.predict_on = predict_on or ["all"]

    @abstractmethod
    def execute(self, dataset, context):
        """Execute model training and prediction."""
        pass

    def get_name(self):
        if self.model_name:
            return self.model_name
        return f"Model_{self.model.__class__.__name__}"

    def can_execute(self, dataset, context):
        """Check if model can be executed."""
        return len(dataset.select(partition="train")) > 0

    def _get_training_data(self, dataset):
        """Get training data from dataset."""
        train_view = dataset.select(partition=self.train_on)
        X_train = train_view.get_features(concatenate=True)
        y_train = train_view.get_targets("auto")
        return X_train, y_train

    def _store_predictions(self, dataset, context, partition, sample_ids, predictions):
        """Store predictions in dataset and context."""
        dataset.add_predictions(
            sample_ids=sample_ids,
            predictions=predictions,
            model_name=self.get_name(),
            partition=partition,
            fold=-1,
            prediction_type="raw"
        )

        context.add_predictions(
            model_name=self.get_name(),
            predictions={
                partition: {
                    "predictions": predictions,
                    "sample_ids": sample_ids,
                    "fold": -1
                }
            }
        )


class SklearnModelOperation(ModelOperation):
    """Sklearn-compatible model operation."""

    def execute(self, dataset, context):
        """Execute sklearn model training and prediction."""
        X_train, y_train = self._get_training_data(dataset)

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions on test data
        test_view = dataset.select(partition="test")
        if len(test_view) > 0:
            X_test = test_view.get_features(concatenate=True)
            y_pred = self.model.predict(X_test)

            self._store_predictions(dataset, context, "test", test_view.sample_ids, y_pred)


class TensorFlowModelOperation(ModelOperation):
    """TensorFlow/Keras-compatible model operation."""

    def __init__(self, model, model_name=None, target_representation="auto",
                 train_on="train", predict_on=None, epochs=50, batch_size=32,
                 validation_split=0.2, verbose=0):
        super().__init__(model, model_name, target_representation, train_on, predict_on)
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose

    def execute(self, dataset, context):
        """Execute TensorFlow model training and prediction."""
        X_train, y_train = self._get_training_data(dataset)

        # Prepare targets for classification/regression
        if dataset.task_type == "classification":
            from tensorflow.keras.utils import to_categorical
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_train_categorical = to_categorical(y_train_encoded)

            # Store label encoder for predictions
            self.label_encoder = le
        else:
            y_train_categorical = y_train.reshape(-1, 1)

        # Train model
        self.model.fit(
            X_train, y_train_categorical,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose
        )

        # Make predictions on test data
        test_view = dataset.select(partition="test")
        if len(test_view) > 0:
            X_test = test_view.get_features(concatenate=True)
            y_pred_proba = self.model.predict(X_test)

            if dataset.task_type == "classification":
                y_pred = self.label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
            else:
                y_pred = y_pred_proba.flatten()

            self._store_predictions(dataset, context, "test", test_view.sample_ids, y_pred)


class TorchModelOperation(ModelOperation):
    """PyTorch-compatible model operation."""

    def __init__(self, model, model_name=None, target_representation="auto",
                 train_on="train", predict_on=None, epochs=50, batch_size=32,
                 learning_rate=0.001, device="cpu"):
        super().__init__(model, model_name, target_representation, train_on, predict_on)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

    def execute(self, dataset, context):
        """Execute PyTorch model training and prediction."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import LabelEncoder

        X_train, y_train = self._get_training_data(dataset)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)

        if dataset.task_type == "classification":
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)

            criterion = nn.CrossEntropyLoss()
            self.label_encoder = le
        else:
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            criterion = nn.MSELoss()

        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Train model
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                if dataset.task_type == "classification":
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()

        # Make predictions on test data
        test_view = dataset.select(partition="test")
        if len(test_view) > 0:
            X_test = test_view.get_features(concatenate=True)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_test_tensor)

                if dataset.task_type == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred = self.label_encoder.inverse_transform(predicted.cpu().numpy())
                else:
                    y_pred = outputs.squeeze().cpu().numpy()

            self._store_predictions(dataset, context, "test", test_view.sample_ids, y_pred)
