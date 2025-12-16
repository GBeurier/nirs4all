"""
Model Controller Components - Modular components for base_model refactoring

This package contains focused, testable components that replace the monolithic
logic in the original launch_training() method.

Components:
    - identifier_generator: Generate model identifiers and names
    - prediction_transformer: Handle scaling/unscaling of predictions
    - prediction_assembler: Assemble prediction data for storage
    - score_calculator: Calculate evaluation scores
    - index_normalizer: Normalize and validate sample indices
"""

from .identifier_generator import ModelIdentifierGenerator, ModelIdentifiers
from .prediction_transformer import PredictionTransformer
from .prediction_assembler import (
    PredictionDataAssembler,
    PartitionPrediction,
    PredictionRecord
)
from .score_calculator import ScoreCalculator, PartitionScores
from .index_normalizer import IndexNormalizer

__all__ = [
    'ModelIdentifierGenerator',
    'ModelIdentifiers',
    'PredictionTransformer',
    'PredictionDataAssembler',
    'PartitionPrediction',
    'PredictionRecord',
    'ScoreCalculator',
    'PartitionScores',
    'IndexNormalizer',
]
