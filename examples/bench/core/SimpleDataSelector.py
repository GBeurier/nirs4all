"""
SimpleDataSelector - Basic data selection without complex scoping rules

Operations handle their own data selection using simple branch filtering.
"""
from typing import Dict, Any


class SimpleDataSelector:
    """Simple data selector that just passes through basic filters"""

    def get_filters_for_operation(self, operation: Any, context: Dict[str, Any], phase: str = "execute") -> Dict[str, Any]:
        """
        Get basic filters for an operation based on simple context

        Args:
            operation: The operation to get filters for
            context: Simple context dict with branch info
            phase: Execution phase (ignored in simple version)

        Returns:
            Basic filter dict
        """
        # Just return the context as filters (contains branch info)
        return context.copy()
