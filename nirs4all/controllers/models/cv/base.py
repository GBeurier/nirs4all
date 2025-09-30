"""
Cross-Validation Strategies - Pluggable CV implementations

This module provides a strategy pattern for different cross-validation approaches,
replacing the monolithic CV handling in the original BaseModelController.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

from nirs4all.controllers.models.config import CVConfig, DataSplit

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


@dataclass
class CVExecutionContext:
    """Context for CV execution."""
    model_config: Dict[str, Any]
    data_splits: List[DataSplit]
    train_params: Dict[str, Any]
    cv_config: CVConfig
    runner: 'PipelineRunner'
    dataset: 'SpectroDataset'
    controller: Any  # Reference to the controller for method calls
    finetune_config: Optional[Any] = None  # Finetune configuration if applicable


@dataclass
class CVResult:
    """Result of CV execution."""
    context: Dict[str, Any]
    binaries: List[Tuple[str, bytes]]
    best_params: Optional[Dict[str, Any]] = None


class CVStrategy(ABC):
    """Base class for cross-validation strategies."""

    @abstractmethod
    def execute(self, context: CVExecutionContext) -> CVResult:
        """
        Execute the cross-validation strategy.

        Args:
            context: Execution context with all necessary data

        Returns:
            CVResult: Results of the CV execution
        """

    def _get_model_instance(self, context: CVExecutionContext) -> Any:
        """Get model instance from context."""
        return context.controller._get_model_instance(context.model_config)

    def _execute_finetune(self, context: CVExecutionContext, fold_idx: Optional[int] = None) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute finetuning - delegates to controller."""
        return context.controller._execute_finetune_modular(  # type: ignore
            context.model_config,
            context.data_splits[fold_idx].X_train if fold_idx is not None else None,
            context.data_splits[fold_idx].y_train if fold_idx is not None else None,
            context.data_splits[fold_idx].X_val if fold_idx is not None else None,
            context.data_splits[fold_idx].y_val if fold_idx is not None else None,
            context.data_splits[fold_idx].X_test if fold_idx is not None else None,
            context.data_splits[fold_idx].y_test if fold_idx is not None else None,
            context.train_params,
            context.finetune_config,
            {},  # context
            context.runner,
            context.dataset,
            fold_idx
        )

    def _execute_train(self, context: CVExecutionContext, fold_idx: Optional[int] = None) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute training - delegates to controller."""
        return context.controller._execute_train_modular(
            context.model_config,
            context.data_splits[fold_idx].X_train if fold_idx is not None else None,
            context.data_splits[fold_idx].y_train if fold_idx is not None else None,
            context.data_splits[fold_idx].X_val if fold_idx is not None else None,
            context.data_splits[fold_idx].y_val if fold_idx is not None else None,
            context.data_splits[fold_idx].X_test if fold_idx is not None else None,
            context.data_splits[fold_idx].y_test if fold_idx is not None else None,
            context.train_params,
            {},  # context
            context.runner,
            context.dataset,
            fold_idx
        )
