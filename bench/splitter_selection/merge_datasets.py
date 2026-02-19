"""
Script to merge train/test splits into single datasets.

Creates nitro_regression_merged and nitro_classif_merged folders
with X.csv, Y.csv, M.csv files per subfolder.
"""

import os
import shutil
from pathlib import Path

import pandas as pd


def merge_train_test(source_folder: Path, dest_folder: Path) -> None:
    """Merge train and test CSV files from source to destination folder.

    Args:
        source_folder: Path to folder containing Xtrain.csv, Xtest.csv, etc.
        dest_folder: Path to destination folder for merged X.csv, Y.csv, M.csv
    """
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Columns to drop from metadata files (contain illegal/weird characters)
    metadata_columns_to_drop = ["trait_class"]

    for prefix in ["X", "Y", "M"]:
        train_file = source_folder / f"{prefix}train.csv"
        test_file = source_folder / f"{prefix}test.csv"

        if train_file.exists() and test_file.exists():
            # Read with semicolon separator (source files use ; as delimiter)
            df_train = pd.read_csv(train_file, sep=";")
            df_test = pd.read_csv(test_file, sep=";")
            df_merged = pd.concat([df_train, df_test], ignore_index=True)

            # Drop problematic columns from metadata files
            if prefix == "M":
                cols_to_drop = [c for c in metadata_columns_to_drop if c in df_merged.columns]
                if cols_to_drop:
                    df_merged = df_merged.drop(columns=cols_to_drop)
                    print(f"  Dropped columns from M: {cols_to_drop}")

            # Save merged file (using comma delimiter for clean output)
            output_file = dest_folder / f"{prefix}_train.csv"
            df_merged.to_csv(output_file, index=False, sep=";")
            print(f"  Created {output_file.name} ({len(df_merged)} rows)")
        else:
            print(f"  Warning: Missing {prefix}train.csv or {prefix}test.csv")

def process_dataset_folder(source_dir: Path, dest_dir: Path) -> None:
    """Process all subfolders in a dataset directory.

    Args:
        source_dir: Path to source directory (e.g., nitro_regression)
        dest_dir: Path to destination directory (e.g., nitro_regression_merged)
    """
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return

    print(f"\nProcessing {source_dir.name} -> {dest_dir.name}")
    print("=" * 50)

    # Clean destination if exists
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Process each subfolder
    for subfolder in sorted(source_dir.iterdir()):
        if subfolder.is_dir():
            print(f"\n{subfolder.name}:")
            dest_subfolder = dest_dir / subfolder.name
            merge_train_test(subfolder, dest_subfolder)

def main():
    """Main function to merge all datasets."""
    base_path = Path(__file__).parent

    # Define source and destination pairs
    datasets = [
        # ("nitro_regression", "nitro_regression_merged"),
        ("nitro_classif", "nitro_classif_merged"),
    ]

    for source_name, dest_name in datasets:
        source_dir = base_path / source_name
        dest_dir = base_path / dest_name
        process_dataset_folder(source_dir, dest_dir)

    print("\n" + "=" * 50)
    print("Done! Merged datasets created.")

if __name__ == "__main__":
    main()
