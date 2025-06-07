"""
Input/Output functionality for SpectroDataset persistence.

This module provides save/load functionality for complete datasets,
storing features as numpy files and metadata as parquet files.
"""

import os
import json
import numpy as np
import polars as pl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import SpectroDataset


def save(ds: "SpectroDataset", path: str) -> None:
    """
    Save a SpectroDataset to disk.

    Creates a folder with:
    - features_src#.npy files (one per source)
    - targets.parquet
    - metadata.parquet
    - predictions.parquet
    - index.parquet
    - folds.json

    Args:
        ds: SpectroDataset instance to save
        path: Directory path where to save the dataset
    """
    # Create directory
    os.makedirs(path, exist_ok=True)

    # Save features as numpy files
    if ds.features.sources:
        features_base = os.path.join(path, "features")
        ds.features.dump_numpy(features_base)

    # Save index if available
    if ds.features.index_df is not None:
        index_path = os.path.join(path, "index.parquet")
        ds.features.index_df.write_parquet(index_path)

    # Save targets if available
    if ds.targets.table is not None:
        targets_path = os.path.join(path, "targets.parquet")
        ds.targets.table.write_parquet(targets_path)

    # Save metadata if available
    if ds.metadata.table is not None:
        metadata_path = os.path.join(path, "metadata.parquet")
        ds.metadata.table.write_parquet(metadata_path)

    # Save predictions if available
    if ds.predictions.table is not None:
        predictions_path = os.path.join(path, "predictions.parquet")
        ds.predictions.table.write_parquet(predictions_path)

    # Save folds if available
    if ds.folds.folds:
        folds_path = os.path.join(path, "folds.json")
        with open(folds_path, 'w') as f:
            json.dump(ds.folds.folds, f, indent=2)


def load(path: str) -> "SpectroDataset":
    """
    Load a SpectroDataset from disk using zero-copy memory mapping.

    Args:
        path: Directory path containing the saved dataset

    Returns:
        Reconstructed SpectroDataset instance
    """
    from .dataset import SpectroDataset

    ds = SpectroDataset()

    # Load features with memory mapping
    features_base = os.path.join(path, "features")
    if os.path.exists(f"{features_base}_src0.npy"):
        ds.features.load_numpy(features_base, mmap_mode="r")

    # Load index if available
    index_path = os.path.join(path, "index.parquet")
    if os.path.exists(index_path):
        ds.features.index_df = pl.read_parquet(index_path)

    # Load targets if available
    targets_path = os.path.join(path, "targets.parquet")
    if os.path.exists(targets_path):
        ds.targets.table = pl.read_parquet(targets_path)

    # Load metadata if available
    metadata_path = os.path.join(path, "metadata.parquet")
    if os.path.exists(metadata_path):
        ds.metadata.table = pl.read_parquet(metadata_path)

    # Load predictions if available
    predictions_path = os.path.join(path, "predictions.parquet")
    if os.path.exists(predictions_path):
        ds.predictions.table = pl.read_parquet(predictions_path)

    # Load folds if available
    folds_path = os.path.join(path, "folds.json")
    if os.path.exists(folds_path):
        with open(folds_path, 'r') as f:
            ds.folds.folds = json.load(f)

    return ds
