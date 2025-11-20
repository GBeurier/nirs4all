"""
Model Loader - Load models from serialized binaries

This component handles loading models from binary storage for prediction
and explanation modes. Supports multiple naming patterns for backward compatibility.

Extracted from launch_training() lines 359-390 to centralize model loading logic.
"""

from typing import List, Tuple, Optional, Any


class ModelLoader:
    """Loads models from serialized binaries.

    Handles multiple naming patterns for backward compatibility:
        1. Exact match: "MyModel_10"
        2. With .pkl extension: "MyModel_10.pkl"
        3. With .joblib extension: "MyModel_10.joblib"
        4. With fold suffix: "MyModel_10_fold0"

    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load(
        ...     model_id="MyModel_10",
        ...     loaded_binaries=binaries,
        ...     fold_idx=0
        ... )
    """

    def load(
        self,
        model_id: str,
        loaded_binaries: List[Tuple[str, Any]],
        fold_idx: Optional[int] = None
    ) -> Any:
        """Load model from binaries with fallback patterns.

        Args:
            model_id: Base model identifier (e.g., "MyModel_10")
            loaded_binaries: List of (name, binary) tuples
            fold_idx: Optional fold index for fold-specific models

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model not found in binaries
        """
        binaries_dict = dict(loaded_binaries)

        # Build search patterns in order of preference
        search_patterns = self._build_search_patterns(model_id, fold_idx)

        # Try each pattern
        for pattern in search_patterns:
            if pattern in binaries_dict:
                return binaries_dict[pattern]

        # Model not found - raise helpful error
        available_keys = list(binaries_dict.keys())
        raise ValueError(
            f"Model binary for '{model_id}' not found in loaded_binaries. "
            f"Searched patterns: {search_patterns}. "
            f"Available keys: {available_keys}"
        )

    def _build_search_patterns(
        self,
        model_id: str,
        fold_idx: Optional[int] = None
    ) -> List[str]:
        """Build list of patterns to search for model binary.

        Args:
            model_id: Base model identifier
            fold_idx: Optional fold index

        Returns:
            List of patterns to try, in order of preference
        """
        patterns = []

        # If fold specified, try fold-specific patterns first
        if fold_idx is not None:
            fold_suffix = f"_fold{fold_idx}"
            patterns.extend([
                f"{model_id}{fold_suffix}",
                f"{model_id}{fold_suffix}.pkl",
                f"{model_id}{fold_suffix}.joblib"
            ])

        # Try base patterns
        patterns.extend([
            model_id,
            f"{model_id}.pkl",
            f"{model_id}.joblib"
        ])

        return patterns

    def check_availability(
        self,
        model_id: str,
        loaded_binaries: List[Tuple[str, Any]],
        fold_idx: Optional[int] = None
    ) -> bool:
        """Check if model is available in binaries without loading.

        Args:
            model_id: Base model identifier
            loaded_binaries: List of (name, binary) tuples
            fold_idx: Optional fold index

        Returns:
            True if model found, False otherwise
        """
        try:
            self.load(model_id, loaded_binaries, fold_idx)
            return True
        except ValueError:
            return False
