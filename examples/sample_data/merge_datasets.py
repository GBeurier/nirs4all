"""
Script to merge train and test datasets into a single train set.

Creates merged copies of datasets from:
- regression_denis -> regression_denis_merged
- classif_denis -> classif_denis_merged

Each merged folder contains: train_x.csv, train_y.csv, train_m.csv
(which are the concatenation of original Xtrain+Xtest, Ytrain+Ytest, Mtrain+Mtest)
"""

import os
import shutil
from pathlib import Path

import pandas as pd


def merge_dataset(source_folder: Path, dest_folder: Path) -> None:
    """
    Merge train and test files from source folder into destination folder.

    Args:
        source_folder: Path to the original dataset folder containing Xtrain, Xtest, etc.
        dest_folder: Path to the destination folder for merged files.
    """
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Define file pairs to merge (source_train, source_test, dest_name)
    file_pairs = [
        ("Xtrain.csv", "Xtest.csv", "train_x.csv"),
        ("Ytrain.csv", "Ytest.csv", "train_y.csv"),
        ("Mtrain.csv", "Mtest.csv", "train_m.csv"),
    ]

    for train_file, test_file, merged_name in file_pairs:
        train_path = source_folder / train_file
        test_path = source_folder / test_file

        if train_path.exists() and test_path.exists():
            # Read and concatenate
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            df_merged = pd.concat([df_train, df_test], ignore_index=True)

            # Save merged file
            merged_path = dest_folder / merged_name
            df_merged.to_csv(merged_path, index=False)
            print(f"  Created: {merged_path.name} ({len(df_merged)} rows)")
        else:
            missing = []
            if not train_path.exists():
                missing.append(train_file)
            if not test_path.exists():
                missing.append(test_file)
            print(f"  Warning: Missing files {missing} in {source_folder.name}")


def process_dataset_folder(source_root: Path, dest_root: Path) -> None:
    """
    Process all subfolders in a dataset root folder.

    Args:
        source_root: Path to the source root folder (e.g., regression_denis).
        dest_root: Path to the destination root folder (e.g., regression_denis_merged).
    """
    if not source_root.exists():
        print(f"Source folder not found: {source_root}")
        return

    # Clean destination if it exists
    if dest_root.exists():
        shutil.rmtree(dest_root)

    print(f"\nProcessing: {source_root.name} -> {dest_root.name}")
    print("-" * 50)

    # Process each subfolder
    subfolders = sorted([f for f in source_root.iterdir() if f.is_dir()])
    for subfolder in subfolders:
        print(f"\n{subfolder.name}:")
        dest_subfolder = dest_root / subfolder.name
        merge_dataset(subfolder, dest_subfolder)


def main():
    """Main entry point."""
    # Get the script's directory
    script_dir = Path(__file__).parent

    # Define source and destination folders
    datasets = [
        ("regression_denis", "regression_denis_merged"),
        ("classif_denis", "classif_denis_merged"),
    ]

    for source_name, dest_name in datasets:
        source_root = script_dir / source_name
        dest_root = script_dir / dest_name
        process_dataset_folder(source_root, dest_root)

    print("\n" + "=" * 50)
    print("Merging complete!")


if __name__ == "__main__":
    main()
