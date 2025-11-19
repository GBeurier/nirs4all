"""Base classes for pipeline controllers.

Controllers handle the execution logic for operators in nirs4all pipelines.
Each controller type knows how to execute specific operator types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nirs4all.pipeline.config.context import ExecutionContext


class BaseController(ABC):
    """Abstract base class for all controllers.

    Controllers are responsible for executing operators within a pipeline context.
    They handle framework-specific logic, state management, and validation.
    """

    @abstractmethod
    def can_handle(self, operator: Any) -> bool:
        """Check if this controller can handle the given operator.

        Parameters
        ----------
        operator : Any
            The operator to check.

        Returns
        -------
        bool
            True if this controller can handle the operator.
        """
        pass

    @abstractmethod
    def execute(self, operator: Any, context: 'ExecutionContext') -> Any:
        """Execute the operator within the pipeline context.

        Parameters
        ----------
        operator : Any
            The operator to execute.
        context : ExecutionContext
            Pipeline execution context including data, state, etc.

        Returns
        -------
        Any
            Result of operator execution.
        """
        pass

    def validate(self, operator: Any) -> None:
        """Validate the operator before execution.

        Parameters
        ----------
        operator : Any
            The operator to validate.

        Raises
        ------
        ValueError
            If operator is invalid.
        """
        pass

    def prepare(self, operator: Any, context: 'ExecutionContext') -> None:
        """Prepare the operator for execution.

        This method can be overridden to perform setup tasks before execution.

        Parameters
        ----------
        operator : Any
            The operator to prepare.
        context : ExecutionContext
            Pipeline execution context.
        """
        pass

    def cleanup(self, operator: Any, context: 'ExecutionContext') -> None:
        """Clean up after operator execution.

        This method can be overridden to perform cleanup tasks after execution.

        Parameters
        ----------
        operator : Any
            The operator that was executed.
        context : ExecutionContext
            Pipeline execution context.
        """
        pass
