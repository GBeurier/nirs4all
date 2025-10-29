"""
Schema definitions for predictions storage.

This module centralizes all DataFrame schema definitions used by the
predictions storage system.
"""

import polars as pl

# Full prediction schema with all columns
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
    "sample_indices": pl.Utf8,  # JSON serialized
    "weights": pl.Utf8,  # JSON serialized
    "metadata": pl.Utf8,  # JSON serialized
    "partition": pl.Utf8,
    "y_true": pl.Utf8,  # JSON serialized (or Parquet for arrays)
    "y_pred": pl.Utf8,  # JSON serialized (or Parquet for arrays)
    "val_score": pl.Float64,
    "test_score": pl.Float64,
    "train_score": pl.Float64,
    "metric": pl.Utf8,
    "task_type": pl.Utf8,
    "n_samples": pl.Int64,
    "n_features": pl.Int64,
    "preprocessings": pl.Utf8,
    "best_params": pl.Utf8,  # JSON serialized
}

# Metadata-only schema for catalog queries (excludes array data)
METADATA_SCHEMA = {
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
}

# Array-only schema for efficient Parquet storage
ARRAY_SCHEMA = {
    "id": pl.Utf8,
    "y_true": pl.Utf8,  # JSON serialized arrays
    "y_pred": pl.Utf8,
    "sample_indices": pl.Utf8,
    "weights": pl.Utf8,
}
