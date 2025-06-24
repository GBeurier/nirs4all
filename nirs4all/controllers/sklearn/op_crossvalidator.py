from typing import Any, Dict, TYPE_CHECKING
from sklearn.model_selection import BaseCrossValidator

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


@register_controller
class CrossValidatorController(OperatorController):
    """Controller for sklearn cross-validation splitters that creates train/validation folds."""

    priority = 20

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the operator is a sklearn cross-validator."""
        return isinstance(operator, BaseCrossValidator) or issubclass(operator.__class__, BaseCrossValidator)

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return False

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1
    ):
        """Execute the cross-validator to create train/validation folds in the dataset."""
        print(f"🔄 Executing cross-validation with {operator.__class__.__name__}")

        try:
            # Get the data for splitting
            X = dataset.x(context, layout="2d", source=source)
            y = dataset.y(context)

            # Handle case where X might be a tuple for multi-source
            if isinstance(X, tuple):
                X = X[0]  # Use first source for splitting

            print(f"🔄 Creating folds for {X.shape[0]} samples using {operator.__class__.__name__}")

            # Get current sample indices from the context
            current_indices, _ = dataset.features.index.get_indices(context)

            # Generate the cross-validation splits and collect them
            fold_splits = []
            fold_idx = 0
            for train_idx, val_idx in operator.split(X, y):
                # Map back to original sample indices
                train_samples = [current_indices[i] for i in train_idx]
                val_samples = [current_indices[i] for i in val_idx]

                # Store the fold split
                fold_splits.append((train_samples, val_samples))

                print(f"   📊 Fold {fold_idx}: {len(train_samples)} train, {len(val_samples)} val samples")
                fold_idx += 1

            # Set the folds in the dataset
            dataset.set_folds(fold_splits)

            print(f"✅ Successfully created {fold_idx} cross-validation folds")

            return context

        except Exception as e:
            print(f"❌ Error in cross-validation: {e}")
            raise