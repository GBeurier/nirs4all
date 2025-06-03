"""
Simplified ModelOperation for testing the fold-based architecture
"""

class ModelOperation:
    """Simplified model operation for testing."""

    def __init__(self, model, model_name=None):
        self.model = model
        self.model_name = model_name

    def execute(self, dataset, context):
        """Execute model training and prediction."""
        # Get training data
        train_view = dataset.select(partition="train")
        X_train = train_view.get_features(concatenate=True)
        y_train = train_view.get_targets("auto")

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions on test data
        test_view = dataset.select(partition="test")
        if len(test_view) > 0:
            X_test = test_view.get_features(concatenate=True)
            y_pred = self.model.predict(X_test)

            # Store predictions in dataset
            dataset.add_predictions(
                sample_ids=test_view.sample_ids,
                predictions=y_pred,
                model_name=self.get_name(),
                partition="test",
                fold=-1,
                prediction_type="raw"
            )

    def get_name(self):
        if self.model_name:
            return self.model_name
        return f"Model_{self.model.__class__.__name__}"

    def can_execute(self, dataset, context):
        """Check if model can be executed."""
        return len(dataset.select(partition="train")) > 0
