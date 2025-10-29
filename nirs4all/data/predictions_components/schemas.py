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
    # Array references (IDs pointing to ArrayRegistry)
    "y_true_id": pl.Utf8,
    "y_pred_id": pl.Utf8,
    "sample_indices_id": pl.Utf8,
    "weights_id": pl.Utf8,
}
