"""
Model Naming Manager - Centralized model naming and ID management

This module provides a centralized system for generating consistent model names
and IDs throughout the pipeline, ensuring continuity across all components.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelIdentifiers:
    """Container for all model identifiers."""
    classname: str          # e.g., "PLSRegression"
    name: str              # Custom name or classname
    model_id: str          # Unique ID for this run (name + operation_counter)
    model_uuid: str        # Global unique ID (model_id + fold + step + config)
    custom_name: Optional[str] = None  # Custom name if provided


class ModelNamingManager:
    """
    Centralized manager for model naming and ID generation.

    This ensures consistency across the entire pipeline and eliminates
    the need to reconstruct model names in different places.
    """

    def __init__(self):
        """Initialize the naming manager."""
        self._model_cache: Dict[str, ModelIdentifiers] = {}

    def create_model_identifiers(
        self,
        model_config: Dict[str, Any],
        runner,
        fold_idx: Optional[int] = None,
        is_avg: bool = False,
        is_weighted_avg: bool = False
    ) -> ModelIdentifiers:
        """
        Create complete model identifiers for a model.

        Args:
            model_config: Model configuration dictionary
            runner: Pipeline runner instance
            fold_idx: Fold index if applicable
            is_avg: True if this is an average model
            is_weighted_avg: True if this is a weighted average model

        Returns:
            ModelIdentifiers: Complete set of model identifiers
        """
        # Extract classname from model or config
        classname = self._extract_classname(model_config)

        # Extract custom name if provided
        custom_name = model_config.get('name') if isinstance(model_config, dict) else None

        # Determine the base name (custom name takes priority)
        name = custom_name if custom_name else classname

        # Generate unique model_id using operation counter
        operation_counter = runner.next_op()
        model_id = f"{name}_{operation_counter}"

        # Handle average models
        if is_avg or is_weighted_avg:
            if is_weighted_avg:
                model_id = f"{name}_w_avg"
            else:
                model_id = f"{name}_avg"

        # Generate model_uuid (globally unique identifier)
        model_uuid = self._generate_model_uuid(
            model_id, fold_idx, runner.step_number,
            getattr(runner.saver, 'pipeline_name', 'unknown')
        )

        identifiers = ModelIdentifiers(
            classname=classname,
            name=name,
            model_id=model_id,
            model_uuid=model_uuid,
            custom_name=custom_name
        )

        # Cache for consistency
        self._model_cache[model_uuid] = identifiers

        return identifiers

    def get_cached_identifiers(self, model_uuid: str) -> Optional[ModelIdentifiers]:
        """Get cached model identifiers by UUID."""
        return self._model_cache.get(model_uuid)

    def _extract_classname(self, model_config: Dict[str, Any]) -> str:
        """Extract the model class name from configuration."""
        if isinstance(model_config, dict):
            if 'model_instance' in model_config:
                model = model_config['model_instance']
                if model is not None:
                    return model.__class__.__name__
            elif 'model' in model_config:
                model = model_config['model']
                if model is not None:
                    return model.__class__.__name__
        elif hasattr(model_config, '__class__'):
            return model_config.__class__.__name__
        elif callable(model_config):
            return model_config.__name__
        else:
            return str(model_config)

        return "UnknownModel"

    def _generate_model_uuid(
        self,
        model_id: str,
        fold_idx: Optional[int],
        step_number: int,
        config_id: str
    ) -> str:
        """Generate a globally unique model UUID."""
        uuid_parts = [model_id]

        if fold_idx is not None:
            uuid_parts.append(f"fold{fold_idx}")

        uuid_parts.extend([f"step{step_number}", config_id])

        return "_".join(uuid_parts)
