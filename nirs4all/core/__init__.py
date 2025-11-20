"""
Core functionality for nirs4all.

This module contains fundamental types, metrics, and task detection logic
that are used throughout the library.
"""

from .task_type import TaskType
from .task_detection import detect_task_type
from . import metrics

__all__ = [
    'TaskType',
    'detect_task_type',
    'metrics',
]