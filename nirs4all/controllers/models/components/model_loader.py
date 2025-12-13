"""
Model Loader - Load models from serialized binaries

This component handles loading models from binary storage for prediction
and explanation modes. Supports multiple naming patterns for backward compatibility.

Extracted from launch_training() lines 359-390 to centralize model loading logic.

For branch-aware pipelines, the model_id generated in predict mode may not match
the training mode due to different operation counters. The loader handles this by:
1. First trying exact match patterns
2. Falling back to class-name based search within branch-specific binaries
"""

import re
from typing import List, Tuple, Optional, Any


# Mapping of nirs4all model aliases to their underlying class names
# These aliases are common in pipeline configs but the artifacts store
# the actual framework class names
MODEL_ALIAS_MAPPINGS = {
    # TensorFlow/Keras models - nicon and variants are common aliases
    "nicon": ["Sequential", "Functional", "Model"],
    "customizable_nicon": ["Sequential", "Functional", "Model"],
    "keras_model": ["Sequential", "Functional", "Model"],
    # Sklearn aliases
    "gbr": ["GradientBoostingRegressor"],
    "gbc": ["GradientBoostingClassifier"],
    "rfr": ["RandomForestRegressor"],
    "rfc": ["RandomForestClassifier"],
    "pls": ["PLSRegression"],
    "svr": ["SVR"],
    "svc": ["SVC"],
    "dtr": ["DecisionTreeRegressor"],
    "dtc": ["DecisionTreeClassifier"],
    "knn": ["KNeighborsRegressor", "KNeighborsClassifier"],
    "lr": ["LinearRegression"],
    "ridge": ["Ridge"],
    "lasso": ["Lasso"],
    "elasticnet": ["ElasticNet"],
    "logistic": ["LogisticRegression"],
    "xgb": ["XGBRegressor", "XGBClassifier"],
    "lgbm": ["LGBMRegressor", "LGBMClassifier"],
    "catboost": ["CatBoostRegressor", "CatBoostClassifier"],
    # Other potential mappings can be added here
}


class ModelLoader:
    """Loads models from serialized binaries.

    Handles multiple naming patterns for backward compatibility:
        1. Exact match: "MyModel_10"
        2. With .pkl extension: "MyModel_10.pkl"
        3. With .joblib extension: "MyModel_10.joblib"
        4. With fold suffix: "MyModel_10_fold0"
        5. Class-name based search for branch-aware loading

    For branched pipelines, binaries are loaded per-branch, so when the exact
    model_id doesn't match, we search for any binary matching the class name
    and fold index.

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

        For branched pipelines, the operation counter in model_id may not match
        the training. Since binaries are loaded per-branch, we can safely search
        by class name + fold when exact match fails.

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

        # Try each exact pattern first
        for pattern in search_patterns:
            if pattern in binaries_dict:
                return binaries_dict[pattern]

        # Fallback: search by class name for branch-aware loading
        # This handles cases where operation counters differ between train/predict
        class_name = self._extract_class_name(model_id)
        if class_name:
            model = self._search_by_class_name(
                class_name, fold_idx, binaries_dict
            )
            if model is not None:
                return model

        # Model not found - raise helpful error
        available_keys = list(binaries_dict.keys())
        raise ValueError(
            f"Model binary for '{model_id}' not found in loaded_binaries. "
            f"Searched patterns: {search_patterns}. "
            f"Available keys: {available_keys}"
        )

    def _extract_class_name(self, model_id: str) -> Optional[str]:
        """Extract the class name from a model_id.

        Model IDs have format: "ClassName_N" or "ClassName_N_foldM"
        For MetaModel, format is: "MetaModel_InnerClassName_N"

        Args:
            model_id: Model identifier like "Ridge_5", "PLSRegression_10",
                     or "MetaModel_Ridge_1"

        Returns:
            Class name or None if cannot be extracted
        """
        # Handle MetaModel case: "MetaModel_InnerClassName_N"
        # The artifact is stored with the inner model's class name, not "MetaModel"
        metamodel_match = re.match(r'^MetaModel_([A-Za-z][A-Za-z0-9]*)_\d+', model_id)
        if metamodel_match:
            return metamodel_match.group(1)

        # Standard pattern: ClassName_Number or ClassName_Number_foldN
        match = re.match(r'^([A-Za-z][A-Za-z0-9]*)_\d+', model_id)
        if match:
            return match.group(1)
        return None

    def _search_by_class_name(
        self,
        class_name: str,
        fold_idx: Optional[int],
        binaries_dict: dict
    ) -> Optional[Any]:
        """Search for a model binary by class name and fold.

        When binaries are loaded per-branch, there should be exactly one model
        of each class per fold. This method finds that model.

        Handles alias matching:
        - If searching for "PLS", will also match "PLSRegression".
        - If searching for "nicon", will also match "Sequential", "Functional".

        This handles cases where model config uses short aliases but the artifact
        stores the full class name.

        Note: Models are stored with names like "Ridge_5.pkl" where each fold
        gets a different operation counter. So fold_idx determines which one
        to pick from multiple matches, not the filename pattern.

        Args:
            class_name: Model class name or alias (e.g., "Ridge", "PLS", "nicon")
            fold_idx: Optional fold index to match (used to select from matches)
            binaries_dict: Dict of binary name -> model

        Returns:
            Model instance or None if not found
        """
        # Build list of class names to search for (original + aliases)
        class_names_to_try = [class_name]

        # Check if this is an alias that maps to multiple class names
        if class_name.lower() in MODEL_ALIAS_MAPPINGS:
            class_names_to_try.extend(MODEL_ALIAS_MAPPINGS[class_name.lower()])

        # Find matching binaries with their operation numbers
        matches = []

        for search_name in class_names_to_try:
            # Try exact pattern first
            exact_pattern = re.compile(
                rf'^{re.escape(search_name)}_(\d+)(\.pkl|\.joblib)?$'
            )

            # Also try pattern that starts with the class_name (for alias matching)
            # e.g., "PLS" should match "PLSRegression_400"
            prefix_pattern = re.compile(
                rf'^{re.escape(search_name)}[A-Za-z]*_(\d+)(\.pkl|\.joblib)?$'
            )

            for name, obj in binaries_dict.items():
                # Skip if already matched
                if any(m[1] == name for m in matches):
                    continue

                match = exact_pattern.match(name)
                if not match:
                    match = prefix_pattern.match(name)
                if match:
                    op_num = int(match.group(1))
                    matches.append((op_num, name, obj))

        if not matches:
            return None

        # Sort by operation number (ascending)
        matches.sort(key=lambda x: x[0])

        if fold_idx is not None and fold_idx < len(matches):
            # Return the nth model for fold n
            return matches[fold_idx][2]
        elif len(matches) == 1:
            # Single match - return it
            return matches[0][2]
        elif matches:
            # Multiple matches but no fold specified - return first
            return matches[0][2]

        return None

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
