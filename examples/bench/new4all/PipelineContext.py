import numpy as np
from typing import Any, Dict, List, Union
from datetime import datetime


class PipelineContext:
    """Context object to track pipeline state and results."""

    def __init__(self):
        self.current_filters = {}
        self.predictions = {}
        self.branch_stack = [0]
        self.processing_history = []

    def push_filters(self, **filters):
        """Add temporary filters."""
        old_filters = self.current_filters.copy()
        self.current_filters.update(filters)
        return old_filters

    def pop_filters(self, old_filters: Dict[str, Any]):
        """Restore previous filters."""
        self.current_filters = old_filters

    def add_predictions(self, model_name: str, predictions: Dict[str, Any]):
        """Store model predictions."""
        if model_name not in self.predictions:
            self.predictions[model_name] = []
        self.predictions[model_name].append(predictions)

    @property
    def current_branch(self) -> int:
        return self.branch_stack[-1]

    def push_branch(self, branch: int):
        self.branch_stack.append(branch)

    def pop_branch(self):
        if len(self.branch_stack) > 1:
            self.branch_stack.pop()