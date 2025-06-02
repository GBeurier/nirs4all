"""
AugmentationOperation - Data augmentation operations for spectroscopic data
"""
import numpy as np
from typing import Dict, List, Optional, Callable
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class AugmentationOperation(PipelineOperation):
    """Operation for data augmentation"""

    def __init__(self, augmentation_type: str = "noise",
                 augmentation_factor: float = 1.0,
                 noise_level: float = 0.01,
                 preserve_original: bool = True):
        """
        Initialize augmentation operation

        Parameters:
        -----------
        augmentation_type : str
            Type of augmentation: "noise", "shift", "scale", "multiply", "baseline"
        augmentation_factor : float
            Factor controlling augmentation intensity
        noise_level : float
            Noise level for noise augmentation
        preserve_original : bool
            Whether to keep original data alongside augmented
        """
        super().__init__()
        self.augmentation_type = augmentation_type
        self.augmentation_factor = augmentation_factor
        self.noise_level = noise_level
        self.preserve_original = preserve_original

        # Available augmentation functions
        self.augmentation_functions = {
            "noise": self.add_noise,
            "shift": self.baseline_shift,
            "scale": self.intensity_scale,            "multiply": self.multiplicative_scatter,
            "baseline": self.baseline_correction
        }

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute the augmentation operation"""
        if not self.can_execute(dataset, context):
            raise ValueError("Cannot execute augmentation - no data available")

        augmentation_func = self.augmentation_functions.get(self.augmentation_type)
        if augmentation_func is None:
            raise ValueError(f"Unknown augmentation type: {self.augmentation_type}")

        if not self.preserve_original:
            # Replace original data with augmented data (in-place)
            n_sources = len(dataset.features.sources)
            for source_idx in range(n_sources):
                X_original = dataset.features.get_source(source_idx)
                X_augmented = augmentation_func(X_original)
                dataset.features.update_source(source_idx, np.arange(len(X_original)), X_augmented)

            print(f"Applied {self.augmentation_type} augmentation to {n_sources} sources (in-place)")
            return

        # If preserve_original=True, add augmented samples as new samples
        n_sources = len(dataset.features.sources)

        # Get all current samples
        all_row_indices = np.arange(len(dataset))
        current_sources = []
        for source_idx in range(n_sources):
            current_sources.append(dataset.features.get_source(source_idx))

        # Apply augmentation to create new samples
        augmented_sources = []
        for source_data in current_sources:
            augmented_data = augmentation_func(source_data)
            augmented_sources.append(augmented_data)

        # Add augmented data as new samples to the dataset
        # First, get current targets if they exist
        current_targets = None
        has_targets = False
        if dataset.target_manager:
            try:
                # Get targets corresponding to the samples that were augmented
                # In this version, all original samples are augmented
                original_sample_ids = dataset.indices["sample"].head(len(current_sources[0])).to_list()
                if not original_sample_ids:
                    print("Warning: No original sample IDs found to fetch targets for augmentation.")
                else:
                    current_targets = dataset.get_targets(original_sample_ids)
                    if current_targets is not None and len(current_targets) == len(original_sample_ids):
                        has_targets = True
                        # print(f"Debug: Successfully fetched {len(current_targets)} targets for augmentation.")
                    else:
                        print(f"Warning: Fetched {len(current_targets) if current_targets is not None else 0} targets, but expected {len(original_sample_ids)}.")
            except Exception as e:
                print(f"Error fetching targets in AugmentationOperation: {e}")
                # Fall through with has_targets = False, current_targets = None

        # Add the augmented samples
        new_sample_ids = dataset.add_data(
            augmented_sources,
            targets=current_targets if has_targets else None,
            partition="train",  # Default partition for augmented data
            processing=f"augmented_{self.augmentation_type}"
        )

        print(f"Applied {self.augmentation_type} augmentation to {n_sources} sources")
        print(f"Added {len(new_sample_ids)} augmented samples to dataset")

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if augmentation can be executed"""
        return len(dataset) > 0

    def get_name(self) -> str:
        """Get operation name"""
        return f"AugmentationOperation({self.augmentation_type})"

    def add_noise(self, X: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to spectra"""
        noise = np.random.normal(0, self.noise_level, X.shape)
        return X + noise * self.augmentation_factor

    def baseline_shift(self, X: np.ndarray) -> np.ndarray:
        """Apply random baseline shift"""
        n_samples = X.shape[0]
        shifts = np.random.normal(0, self.augmentation_factor, (n_samples, 1))
        return X + shifts

    def intensity_scale(self, X: np.ndarray) -> np.ndarray:
        """Apply random intensity scaling"""
        n_samples = X.shape[0]
        scales = np.random.normal(1.0, self.augmentation_factor, (n_samples, 1))
        # Ensure scales are positive
        scales = np.abs(scales)
        return X * scales

    def multiplicative_scatter(self, X: np.ndarray) -> np.ndarray:
        """Apply multiplicative scatter correction simulation"""
        n_samples, n_features = X.shape

        # Random slope and offset for each sample
        slopes = np.random.normal(1.0, self.augmentation_factor * 0.1, (n_samples, 1))
        offsets = np.random.normal(0.0, self.augmentation_factor * 0.01, (n_samples, 1))

        # Apply linear transformation
        return X * slopes + offsets

    def baseline_correction(self, X: np.ndarray) -> np.ndarray:
        """Simulate baseline drift and correction"""
        n_samples, n_features = X.shape

        # Create random polynomial baseline for each sample
        baselines = np.zeros_like(X)
        for i in range(n_samples):
            # Random polynomial coefficients
            poly_order = 3
            coeffs = np.random.normal(0, self.augmentation_factor * 0.001, poly_order + 1)

            # Generate polynomial baseline
            x_vals = np.linspace(-1, 1, n_features)
            baseline = np.polyval(coeffs, x_vals)
            baselines[i] = baseline

        return X + baselines

    def duplicate_targets(self, dataset: SpectraDataset, n_sources: int) -> None:
        """Duplicate targets for augmented samples"""
        if hasattr(dataset, 'target_manager') and dataset.target_manager is not None:
            # Get current targets
            y_current = dataset.target_manager.get_targets()

            if y_current is not None:
                # Duplicate targets for each augmented source
                if len(y_current.shape) == 1:
                    y_duplicated = np.tile(y_current, n_sources)
                else:
                    y_duplicated = np.tile(y_current, (n_sources, 1))

                # Update targets in manager
                # Note: This is a simplified approach - in practice we'd need
                # to handle partitions and other metadata properly
                print(f"Duplicated targets for {n_sources} augmented sources")


class AugmentationStrategy:
    """Strategy pattern for different augmentation approaches"""

    def __init__(self):
        self.operations = []

    def add_operation(self, operation: AugmentationOperation) -> 'AugmentationStrategy':
        """Add augmentation operation to strategy"""
        self.operations.append(operation)
        return self

    def execute_all(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute all augmentation operations in sequence"""
        for operation in self.operations:
            operation.execute(dataset, context)

    @classmethod
    def noise_augmentation(cls, noise_levels: List[float]) -> 'AugmentationStrategy':
        """Create noise augmentation strategy with multiple noise levels"""
        strategy = cls()
        for level in noise_levels:
            strategy.add_operation(AugmentationOperation(
                augmentation_type="noise",
                noise_level=level,
                preserve_original=True
            ))
        return strategy

    @classmethod
    def spectroscopic_augmentation(cls) -> 'AugmentationStrategy':
        """Create comprehensive spectroscopic augmentation strategy"""
        strategy = cls()

        # Add various spectroscopic augmentations
        strategy.add_operation(AugmentationOperation(
            augmentation_type="noise",
            noise_level=0.005,
            preserve_original=True
        ))

        strategy.add_operation(AugmentationOperation(
            augmentation_type="shift",
            augmentation_factor=0.01,
            preserve_original=True
        ))

        strategy.add_operation(AugmentationOperation(
            augmentation_type="scale",
            augmentation_factor=0.05,
            preserve_original=True
        ))

        return strategy
