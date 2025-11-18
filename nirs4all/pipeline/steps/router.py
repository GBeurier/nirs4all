"""Controller router for selecting appropriate controllers."""
from typing import Any, Optional, Type

from nirs4all.controllers.base import BaseController
from nirs4all.controllers.registry import CONTROLLER_REGISTRY
from nirs4all.pipeline.steps.parser import ParsedStep


class ControllerRouter:
    """Routes parsed steps to appropriate controllers.

    Uses registry pattern with controller priorities.
    Extensible - new controllers automatically discovered via registry.

    Attributes:
        registry: Controller registry (from nirs4all.controllers.registry)
    """

    def __init__(self):
        """Initialize router with controller registry."""
        self.registry = CONTROLLER_REGISTRY

    def route(self, parsed_step: ParsedStep, step: Any = None) -> BaseController:
        """Select the appropriate controller for a parsed step.

        Args:
            parsed_step: Parsed step configuration
            step: Original step (for backward compatibility with matches())

        Returns:
            Instantiated controller instance

        Raises:
            TypeError: If no matching controller found
        """
        # Use original step for matching if available
        match_step = step if step is not None else parsed_step.original_step
        operator = parsed_step.operator
        keyword = parsed_step.keyword

        # Find all matching controllers
        matches = [
            cls for cls in self.registry
            if cls.matches(match_step, operator, keyword)
        ]

        if not matches:
            raise TypeError(
                f"No matching controller found for step: {match_step}. "
                f"Operator: {operator}, Keyword: {keyword}. "
                f"Available controllers: {[cls.__name__ for cls in self.registry]}"
            )

        # Sort by priority (lower number = higher priority)
        matches.sort(key=lambda c: c.priority)

        # Return instantiated controller with highest priority
        return matches[0]()

    def route_from_raw(
        self,
        step: Any,
        operator: Any = None,
        keyword: str = ""
    ) -> BaseController:
        """Route from raw step parameters (backward compatibility).

        Args:
            step: Raw step configuration
            operator: Optional operator instance
            keyword: Optional keyword hint

        Returns:
            Instantiated controller instance

        Raises:
            TypeError: If no matching controller found
        """
        matches = [
            cls for cls in self.registry
            if cls.matches(step, operator, keyword)
        ]

        if not matches:
            raise TypeError(
                f"No matching controller found for {step}. "
                f"Available controllers: {[cls.__name__ for cls in self.registry]}"
            )

        matches.sort(key=lambda c: c.priority)
        return matches[0]()
