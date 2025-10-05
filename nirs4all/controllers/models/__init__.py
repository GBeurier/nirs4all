"""
Model controllers module for nirs4all.

This module contains model controllers for different machine learning frameworks:
- BaseModelController: Simplified base class with common functionality
- Framework-specific controllers in their respective framework directories

All model controllers support training, fine-tuning with Optuna, and prediction modes.
"""

from .base_model_controller import BaseModelController

__all__ = [
    'BaseModelController'
]
