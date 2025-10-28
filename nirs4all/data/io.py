# """
# Input/Output functionality for SpectroDataset persistence.

# This module provides save/load functionality for complete datasets,
# storing features as numpy files and metadata as parquet files.
# """

# import os
# import json
# from typing import TYPE_CHECKING

# import numpy as np
# import polars as pl

# if TYPE_CHECKING:
#     from .dataset import SpectroDataset


# def save(ds: "SpectroDataset", path: str) -> None:
#     """
#     Save a SpectroDataset to disk.

#     Creates a folder with:
#     - features_src#.npy files (one per source)
#     - targets.parquet
#     - metadata.parquet
#     - predictions.parquet
#     - index.parquet
#     - folds.json

#     Args:
#         ds: SpectroDataset instance to save
#         path: Directory path where to save the dataset
#     """
#     # Create directory
#     os.makedirs(path, exist_ok=True)

#     # Save features as numpy files
#     if ds.features.sources:
#         features_base = os.path.join(path, "features")
#         ds.features.dump_numpy(features_base)

#     # Save index if available
#     if ds.features.index_df is not None:
#         index_path = os.path.join(path, "index.parquet")
#         ds.features.index_df.write_parquet(index_path)    # Save targets if available
#     if ds.targets.sources:
#         # For the new TargetBlock, we need to serialize the sources
#         # For now, let's save the raw target data - this is a simplified implementation
#         # A more complete implementation would serialize all the source metadata
#         import polars as pl
#         target_rows = []
#         for key, source in ds.targets.sources.items():
#             # Extract name and processing from key
#             parts = key.rsplit('_', 1)
#             name = parts[0] if len(parts) > 1 else key
#             processing = parts[1] if len(parts) > 1 else "raw"

#             # Convert source data to table format
#             data = source.get_raw_data()
#             if data.ndim == 1:
#                 data = data.reshape(-1, 1)

#             for i, sample_id in enumerate(source.samples):
#                 target_rows.append({
#                     'sample': int(sample_id),
#                     'targets': data[i].tolist(),
#                     'processing': processing,
#                     'target_type': source.target_type.value,
#                     'name': name
#                 })

#         if target_rows:
#             targets_df = pl.DataFrame(target_rows)
#             targets_path = os.path.join(path, "targets.parquet")
#             targets_df.write_parquet(targets_path)

#     # Save metadata if available
#     if ds.metadata.table is not None:
#         metadata_path = os.path.join(path, "metadata.parquet")
#         ds.metadata.table.write_parquet(metadata_path)

#     # Save predictions if available
#     if ds.predictions.table is not None:
#         predictions_path = os.path.join(path, "predictions.parquet")
#         ds.predictions.table.write_parquet(predictions_path)

#     # Save folds if available
#     if ds.folds.folds:
#         folds_path = os.path.join(path, "folds.json")
#         with open(folds_path, 'w', encoding='utf-8') as f:
#             json.dump(ds.folds.folds, f, indent=2)


# def load(path: str) -> "SpectroDataset":
#     """
#     Load a SpectroDataset from disk using zero-copy memory mapping.

#     Args:
#         path: Directory path containing the saved dataset

#     Returns:
#         Reconstructed SpectroDataset instance

#     Raises:
#         FileNotFoundError: If the directory does not exist
#     """
#     from .dataset import SpectroDataset

#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset directory not found: {path}")

#     if not os.path.isdir(path):
#         raise NotADirectoryError(f"Path is not a directory: {path}")

#     ds = SpectroDataset()

#     # Load features with memory mapping
#     features_base = os.path.join(path, "features")
#     if os.path.exists(f"{features_base}_src0.npy"):
#         ds.features.load_numpy(features_base, mmap_mode="r")    # Load index if available
#     index_path = os.path.join(path, "index.parquet")
#     if os.path.exists(index_path):
#         ds.features.index_df = pl.read_parquet(index_path)

#     # Load targets if available
#     targets_path = os.path.join(path, "targets.parquet")
#     if os.path.exists(targets_path):
#         targets_df = pl.read_parquet(targets_path)

#         # Group by name and processing to reconstruct sources
#         if 'name' in targets_df.columns and 'target_type' in targets_df.columns:
#             # New format with explicit target metadata
#             for (name, processing, target_type), group in targets_df.group_by(['name', 'processing', 'target_type']):
#                 samples = group['sample'].to_numpy()
#                 targets_list = group['targets'].to_list()
#                 target_data = np.array(targets_list, dtype=np.float32)

#                 # Convert to strings for type safety
#                 name_str = str(name)
#                 processing_str = str(processing)
#                 target_type_str = str(target_type)

#                 if target_type_str == 'regression':
#                     ds.targets.add_regression_targets(name_str, target_data, samples, processing_str)
#                 elif target_type_str == 'classification':
#                     ds.targets.add_classification_targets(name_str, target_data.flatten().astype(int), samples, processing_str)
#                 elif target_type_str == 'multilabel':
#                     ds.targets.add_multilabel_targets(name_str, target_data, samples, processing_str)
#         else:
#             # Legacy format - assume regression
#             samples = targets_df['sample'].to_numpy()
#             targets_list = targets_df['targets'].to_list()
#             target_data = np.array(targets_list, dtype=np.float32)
#             processing = targets_df['processing'].to_list()[0] if 'processing' in targets_df.columns else "raw"
#             ds.targets.add_regression_targets("target", target_data, samples, processing)

#     # Load metadata if available
#     metadata_path = os.path.join(path, "metadata.parquet")
#     if os.path.exists(metadata_path):
#         ds.metadata.table = pl.read_parquet(metadata_path)

#     # Load predictions if available
#     predictions_path = os.path.join(path, "predictions.parquet")
#     if os.path.exists(predictions_path):
#         ds.predictions.table = pl.read_parquet(predictions_path)

#     # Load folds if available
#     folds_path = os.path.join(path, "folds.json")
#     if os.path.exists(folds_path):
#         with open(folds_path, 'r', encoding='utf-8') as f:
#             ds.folds.folds = json.load(f)

#     return ds
