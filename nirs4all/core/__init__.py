# nirs4all/core/__init__.py"""
from typing import List, Type

from .PipelineOperator import PipelineOperator

RUNNER_REGISTRY: List[Type[PipelineOperator]] = []

def register_runner(cls: Type[PipelineOperator]):
    """ Register a new pipeline operator runner class."""
    RUNNER_REGISTRY.append(cls)
    RUNNER_REGISTRY.sort(key=lambda c: c.priority)
    return cls
