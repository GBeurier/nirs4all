"""
Schema definitions for predictions storage.

This module centralizes all DataFrame schema definitions used by the
predictions storage system. Uses array registry for efficient external
array storage.
"""

import polars as pl

# Prediction schema with array references for ArrayRegistry
PREDICTION_SCHEMA = {
    "id": pl.Utf8,
    "dataset_name": pl.Utf8,
    "dataset_path": pl.Utf8,
    "config_name": pl.Utf8,
    "config_path": pl.Utf8,
    "pipeline_uid": pl.Utf8,
    "step_idx": pl.Int64,
    "op_counter": pl.Int64,
    "model_name": pl.Utf8,
    "model_classname": pl.Utf8,
    "model_path": pl.Utf8,
    "model_artifact_id": pl.Utf8,  # Deterministic artifact ID for model loading (v2)
    "trace_id": pl.Utf8,  # Execution trace ID for deterministic prediction replay (v2)
    "fold_id": pl.Utf8,
    "partition": pl.Utf8,
    "val_score": pl.Float64,
    "test_score": pl.Float64,
    "train_score": pl.Float64,
    "metric": pl.Utf8,
    "task_type": pl.Utf8,
    "n_samples": pl.Int64,
    "n_features": pl.Int64,
    "preprocessings": pl.Utf8,
    "best_params": pl.Utf8,  # JSON serialized
    "metadata": pl.Utf8,  # JSON serialized
    "scores": pl.Utf8,  # JSON serialized: {"train": {"rmse": 0.1}, "val": ...}
    # Branch tracking for pipeline branching
    "branch_id": pl.Int64,  # Branch identifier (0-indexed, None if not branching)
    "branch_name": pl.Utf8,  # Human-readable branch name
    # Outlier exclusion tracking for outlier_excluder branches
    "exclusion_count": pl.Int64,  # Number of samples excluded during training
    "exclusion_rate": pl.Float64,  # Rate of samples excluded (0.0-1.0)
    # Array references (IDs pointing to ArrayRegistry)
    "y_true_id": pl.Utf8,
    "y_pred_id": pl.Utf8,
    "y_proba_id": pl.Utf8,  # Class probabilities for classification tasks
    "sample_indices_id": pl.Utf8,
    "weights_id": pl.Utf8,
}
