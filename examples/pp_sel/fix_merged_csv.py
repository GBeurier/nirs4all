"""Script to remove the trait_class column from all M.csv files in nitro_classif_merged."""

import os
import pandas as pd
from pathlib import Path

def fix_merged_csv_files(base_dir: str = "sample_data/nitro_classif_merged"):
    """Remove trait_class column from all M.csv files."""
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory {base_path} does not exist!")
        return

    # Find all M.csv files (train_m.csv, test_m.csv, val_m.csv)
    m_files = list(base_path.rglob("*_m.csv"))

    print(f"Found {len(m_files)} M.csv files to process")

    for file_path in m_files:
        try:
            df = pd.read_csv(file_path, sep=";")

            if "trait_class" in df.columns:
                df = df.drop(columns=["trait_class"])
                df.to_csv(file_path, index=False, sep=";")
                print(f"✓ Fixed: {file_path}")
            else:
                print(f"- Skipped (no trait_class): {file_path}")

        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")

if __name__ == "__main__":
    fix_merged_csv_files()
