"""
Core functionality for nirs4all.

This module contains fundamental types, metrics, and task detection logic
that are used throughout the library.
"""

from . import metrics
from .task_detection import detect_task_type
from .task_type import TaskType

__all__ = [
    'TaskType',
    'detect_task_type',
    'metrics',
]
