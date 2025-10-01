"""
CV Strategy Factory - Creates CV strategy instances

This module provides a factory for creating cross-validation strategy instances
based on configuration.
"""

from typing import TYPE_CHECKING

from nirs4all.controllers.models.config import CVMode
from nirs4all.controllers.models.cv.base import CVStrategy

if TYPE_CHECKING:
    pass  # We'll import concrete strategies here when we create them


class CVStrategyFactory:
    """
    Factory for creating cross-validation strategies.

    This factory centralizes the creation of CV strategies and makes it easy
    to add new strategies without modifying the controller logic.
    """

    @staticmethod
    def create_strategy(cv_mode: CVMode) -> CVStrategy:
        """
        Create a CV strategy instance based on the mode.

        Args:
            cv_mode: The cross-validation mode to use

        Returns:
            CVStrategy: The appropriate strategy instance

        Raises:
            ValueError: If the CV mode is not supported
        """
        if cv_mode == CVMode.SIMPLE:
            from nirs4all.controllers.models.cv.strategies.simple_cv import SimpleCVStrategy
            return SimpleCVStrategy()
        elif cv_mode == CVMode.PER_FOLD:
            from nirs4all.controllers.models.cv.strategies.per_fold_cv import PerFoldCVStrategy
            return PerFoldCVStrategy()
        elif cv_mode == CVMode.NESTED:
            from nirs4all.controllers.models.cv.strategies.nested_cv import NestedCVStrategy
            return NestedCVStrategy()
        elif cv_mode == CVMode.GLOBAL_AVERAGE:
            from nirs4all.controllers.models.cv.strategies.global_average_cv import GlobalAverageCVStrategy
            return GlobalAverageCVStrategy()
        else:
            raise ValueError(f"Unsupported CV mode: {cv_mode}")

    @staticmethod
    def get_available_strategies() -> list[CVMode]:
        """
        Get list of available CV strategies.

        Returns:
            list[CVMode]: Available CV modes
        """
        return [CVMode.SIMPLE, CVMode.PER_FOLD, CVMode.NESTED, CVMode.GLOBAL_AVERAGE]