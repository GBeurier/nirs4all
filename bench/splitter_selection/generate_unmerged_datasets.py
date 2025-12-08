"""
Generate Unmerged Datasets using SPXY Splitting
================================================

This script regenerates datasets from nitro_classif_merged and nitro_regression_merged
into train/test splits using the winning SPXY strategy from the splitter selection analysis.

Output structure:
    nitro_classif_unmerged/
        DatasetName/
            X_train.csv, Y_train.csv, M_train.csv
            X_test.csv, Y_test.csv, M_test.csv

    nitro_regression_unmerged/
        DatasetName/
            X_train.csv, Y_train.csv, M_train.csv
            X_test.csv, Y_test.csv, M_test.csv

The SPXY splitter considers both spectral (X) and target (Y) information for optimal
sample selection. It was the best strategy in the enhanced splitter selection analysis.

For cross-validation during training, use StratifiedGroupKFold with:
    - n_splits=3
    - shuffle=True
    - random_state=42
    - Groups: sample IDs (to keep repetitions together)
    - Stratification: binned target values (5 bins)

Usage:
    python generate_unmerged_datasets.py
    python generate_unmerged_datasets.py --test_size 0.2 --random_state 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent))

from splitter_strategies import Nirs4allSPXYSplitter, HAS_NIRS4ALL_SPLITTERS

###REUSE

# from sklearn.model_selection import StratifiedGroupKFold
# import pandas as pd
# import numpy as np

# # Load training data
# M_train = pd.read_csv('M_train.csv', sep=';')
# Y_train = pd.read_csv('Y_train.csv')

# # Get groups and binned targets
# groups = M_train['ID'].values
# y = Y_train.iloc[:, 0].values
# y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

# # Create StratifiedGroupKFold
# sgkf = StratifiedGroupKFold(
#     n_splits=3,
#     shuffle=True,
#     random_state=42
# )

# # Iterate over folds
# for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_bins, groups)):
#     X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
#     y_fold_train, y_fold_val = y[train_idx], y[val_idx]
#     # ... train your model









def detect_separator(file_path: Path) -> str:
    """Detect CSV separator by reading first line."""
    with open(file_path, 'r') as f:
        first_line = f.readline()
    if ';' in first_line:
        return ';'
    return ','


def load_merged_dataset(data_dir: Path) -> tuple:
    """
    Load X, Y, M from a merged dataset directory.

    Args:
        data_dir: Directory containing X.csv, Y.csv, M.csv

    Returns:
        X: Spectra array (n_samples, n_features)
        Y_df: Target DataFrame
        M_df: Metadata DataFrame
    """
    x_sep = detect_separator(data_dir / 'X.csv')
    m_sep = detect_separator(data_dir / 'M.csv')

    X_df = pd.read_csv(data_dir / 'X.csv', sep=x_sep)
    X = X_df.values.astype(np.float32)

    Y_df = pd.read_csv(data_dir / 'Y.csv')
    M_df = pd.read_csv(data_dir / 'M.csv', sep=m_sep)

    return X, Y_df, M_df


def load_original_dataset(data_dir: Path) -> tuple:
    """
    Load X, Y, M from original train/test split files.

    Args:
        data_dir: Directory containing Xtrain.csv, Xtest.csv, etc.

    Returns:
        X: Spectra array (n_samples, n_features)
        Y_df: Target DataFrame
        M_df: Metadata DataFrame
    """
    # Detect separators
    x_sep = detect_separator(data_dir / 'Xtrain.csv')
    m_sep = detect_separator(data_dir / 'Mtrain.csv')

    # Load and merge train/test
    X_train = pd.read_csv(data_dir / 'Xtrain.csv', sep=x_sep)
    X_test = pd.read_csv(data_dir / 'Xtest.csv', sep=x_sep)
    X_df = pd.concat([X_train, X_test], ignore_index=True)
    X = X_df.values.astype(np.float32)

    Y_train = pd.read_csv(data_dir / 'Ytrain.csv')
    Y_test = pd.read_csv(data_dir / 'Ytest.csv')
    Y_df = pd.concat([Y_train, Y_test], ignore_index=True)

    M_train = pd.read_csv(data_dir / 'Mtrain.csv', sep=m_sep)
    M_test = pd.read_csv(data_dir / 'Mtest.csv', sep=m_sep)
    M_df = pd.concat([M_train, M_test], ignore_index=True)

    return X, Y_df, M_df


def save_split_dataset(
    X: np.ndarray,
    Y_df: pd.DataFrame,
    M_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    output_dir: Path,
    x_sep: str = ';'
) -> None:
    """
    Save train/test split to separate CSV files.

    Args:
        X: Full spectra array
        Y_df: Full target DataFrame
        M_df: Full metadata DataFrame
        train_mask: Boolean mask for train samples
        test_mask: Boolean mask for test samples
        output_dir: Directory to save files
        x_sep: Separator for X files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train files
    X_train = pd.DataFrame(X[train_mask])
    X_train.to_csv(output_dir / 'X_train.csv', sep=x_sep, index=False)

    Y_train = Y_df.iloc[train_mask].reset_index(drop=True)
    Y_train.to_csv(output_dir / 'Y_train.csv', index=False)

    M_train = M_df.iloc[train_mask].reset_index(drop=True)
    M_train.to_csv(output_dir / 'M_train.csv', sep=';', index=False)

    # Save test files
    X_test = pd.DataFrame(X[test_mask])
    X_test.to_csv(output_dir / 'X_test.csv', sep=x_sep, index=False)

    Y_test = Y_df.iloc[test_mask].reset_index(drop=True)
    Y_test.to_csv(output_dir / 'Y_test.csv', index=False)

    M_test = M_df.iloc[test_mask].reset_index(drop=True)
    M_test.to_csv(output_dir / 'M_test.csv', sep=';', index=False)


def split_dataset_with_spxy(
    X: np.ndarray,
    Y_df: pd.DataFrame,
    M_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_folds: int = 3
) -> tuple:
    """
    Split a dataset using SPXY strategy.

    Args:
        X: Spectra array
        Y_df: Target DataFrame
        M_df: Metadata DataFrame
        test_size: Fraction for test set
        random_state: Random seed
        n_folds: Number of CV folds (for fold assignment info)

    Returns:
        train_mask: Boolean mask for train samples
        test_mask: Boolean mask for test samples
        split_result: SplitResult object with fold assignments
    """
    y = Y_df.iloc[:, 0].values.astype(np.float32)
    sample_ids = M_df['ID'].values

    # Create SPXY splitter
    splitter = Nirs4allSPXYSplitter(
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state,
        pca_components=None,  # Use all components
        metric='euclidean'
    )

    # Get split result
    split_result = splitter.split(X, y, sample_ids)

    # Create masks for all samples (including repetitions)
    train_mask = np.isin(sample_ids, split_result.train_ids)
    test_mask = np.isin(sample_ids, split_result.test_ids)

    return train_mask, test_mask, split_result


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    n_folds: int = 3,
    verbose: bool = True,
    use_original: bool = False
) -> dict:
    """
    Process a single dataset and create train/test split files.

    Args:
        input_dir: Directory containing X.csv/Y.csv/M.csv or Xtrain/Xtest etc.
        output_dir: Directory to save split files
        test_size: Fraction for test set
        random_state: Random seed
        n_folds: Number of CV folds
        verbose: Print progress
        use_original: If True, load from original train/test files instead of merged

    Returns:
        Dictionary with split statistics
    """
    dataset_name = input_dir.name

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print('='*60)

    # Load data
    if use_original:
        X, Y_df, M_df = load_original_dataset(input_dir)
    else:
        X, Y_df, M_df = load_merged_dataset(input_dir)
    sample_ids = M_df['ID'].values
    unique_ids = np.unique(sample_ids)

    if verbose:
        print(f"  Loaded: {X.shape[0]} samples, {len(unique_ids)} unique IDs")
        print(f"  Spectral features: {X.shape[1]}")

    # Apply SPXY split
    train_mask, test_mask, split_result = split_dataset_with_spxy(
        X, Y_df, M_df, test_size, random_state, n_folds
    )

    n_train = train_mask.sum()
    n_test = test_mask.sum()
    n_train_ids = len(split_result.train_ids)
    n_test_ids = len(split_result.test_ids)

    if verbose:
        print(f"  Train: {n_train} samples ({n_train_ids} unique IDs)")
        print(f"  Test: {n_test} samples ({n_test_ids} unique IDs)")
        print(f"  Actual test size: {n_test_ids / len(unique_ids):.1%}")

    # Save split files
    save_split_dataset(X, Y_df, M_df, train_mask, test_mask, output_dir)

    if verbose:
        print(f"  Saved to: {output_dir}")

    # Save fold assignments for reference
    fold_df = split_result.fold_assignments
    fold_df.to_csv(output_dir / 'fold_assignments.csv', index=False)

    # Save split info
    info = {
        'dataset': dataset_name,
        'n_total_samples': X.shape[0],
        'n_unique_ids': len(unique_ids),
        'n_train_samples': int(n_train),
        'n_train_ids': int(n_train_ids),
        'n_test_samples': int(n_test),
        'n_test_ids': int(n_test_ids),
        'test_size_requested': test_size,
        'test_size_actual': n_test_ids / len(unique_ids),
        'random_state': random_state,
        'splitter': 'Nirs4allSPXYSplitter',
        'strategy_info': split_result.strategy_info
    }

    # Save as JSON
    import json
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(info, f, indent=2, default=str)

    return info


def process_all_datasets(
    source_dir: Path,
    unmerged_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    n_folds: int = 3,
    verbose: bool = True,
    use_original: bool = False
) -> list:
    """
    Process all datasets in a directory.

    Args:
        source_dir: Directory containing dataset subdirectories
        unmerged_dir: Output directory for split datasets
        test_size: Fraction for test set
        random_state: Random seed
        n_folds: Number of CV folds
        verbose: Print progress
        use_original: If True, load from original train/test files

    Returns:
        List of split info dictionaries
    """
    results = []

    # Determine what files to look for
    if use_original:
        required_files = ['Xtrain.csv', 'Ytrain.csv', 'Mtrain.csv',
                          'Xtest.csv', 'Ytest.csv', 'Mtest.csv']
    else:
        required_files = ['X.csv', 'Y.csv', 'M.csv']

    # Find all subdirectories with required files
    for subdir in sorted(source_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if not all((subdir / f).exists() for f in required_files):
            if verbose:
                print(f"Skipping {subdir.name}: missing required files")
            continue

        output_dir = unmerged_dir / subdir.name

        try:
            info = process_dataset(
                subdir, output_dir,
                test_size=test_size,
                random_state=random_state,
                n_folds=n_folds,
                verbose=verbose,
                use_original=use_original
            )
            results.append(info)
        except Exception as e:
            print(f"ERROR processing {subdir.name}: {e}")
            import traceback
            traceback.print_exc()

    return results


def main(
    test_size: float = 0.2,
    random_state: int = 42,
    n_folds: int = 3,
    verbose: bool = True
):
    """
    Main function to generate unmerged datasets.

    Args:
        test_size: Fraction for test set
        random_state: Random seed
        n_folds: Number of CV folds for fold assignment reference
        verbose: Print progress
    """
    if not HAS_NIRS4ALL_SPLITTERS:
        print("ERROR: nirs4all splitters not available. Please install nirs4all.")
        sys.exit(1)

    base_dir = Path(__file__).parent

    print("\n" + "=" * 80)
    print("GENERATE UNMERGED DATASETS USING SPXY SPLITTING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Test size: {test_size:.0%}")
    print(f"  Random state: {random_state}")
    print(f"  CV folds (for reference): {n_folds}")
    print(f"  Splitter: SPXY (Sample set Partitioning based on joint X-Y distances)")

    # Process regression datasets (from merged)
    regression_merged = base_dir / 'nitro_regression_merged'
    regression_unmerged = base_dir / 'nitro_regression_unmerged'

    if regression_merged.exists():
        print(f"\n\n{'#' * 80}")
        print("PROCESSING REGRESSION DATASETS")
        print('#' * 80)
        regression_results = process_all_datasets(
            regression_merged, regression_unmerged,
            test_size=test_size,
            random_state=random_state,
            n_folds=n_folds,
            verbose=verbose,
            use_original=False
        )
        print(f"\n  ✓ Processed {len(regression_results)} regression datasets")
    else:
        print(f"\nWARNING: {regression_merged} not found, skipping regression datasets")
        regression_results = []

    # Process classification datasets (from original, since merged M.csv are corrupted)
    classif_original = base_dir / 'nitro_classif'
    classif_unmerged = base_dir / 'nitro_classif_unmerged'

    if classif_original.exists():
        print(f"\n\n{'#' * 80}")
        print("PROCESSING CLASSIFICATION DATASETS")
        print('#' * 80)
        classif_results = process_all_datasets(
            classif_original, classif_unmerged,
            test_size=test_size,
            random_state=random_state,
            n_folds=n_folds,
            verbose=verbose,
            use_original=True  # Use original train/test files
        )
        print(f"\n  ✓ Processed {len(classif_results)} classification datasets")
    else:
        print(f"\nWARNING: {classif_original} not found, skipping classification datasets")
        classif_results = []

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if regression_results:
        print(f"\nRegression datasets saved to: {regression_unmerged}")
        for r in regression_results:
            print(f"  - {r['dataset']}: {r['n_train_samples']} train / {r['n_test_samples']} test samples")

    if classif_results:
        print(f"\nClassification datasets saved to: {classif_unmerged}")
        for r in classif_results:
            print(f"  - {r['dataset']}: {r['n_train_samples']} train / {r['n_test_samples']} test samples")

    # Fold configuration recommendation
    print("\n\n" + "=" * 80)
    print("RECOMMENDED FOLD CONFIGURATION FOR TRAINING")
    print("=" * 80)
    print("""
To reproduce the cross-validation strategy from the splitter selection analysis,
use StratifiedGroupKFold during training:

    from sklearn.model_selection import StratifiedGroupKFold
    import pandas as pd
    import numpy as np

    # Load training data
    M_train = pd.read_csv('M_train.csv', sep=';')
    Y_train = pd.read_csv('Y_train.csv')

    # Get groups (sample IDs) and binned targets for stratification
    groups = M_train['ID'].values
    y = Y_train.iloc[:, 0].values
    y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

    # Create StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    # Iterate over folds
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_bins, groups)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        # ... train your model

Key parameters:
    - n_splits: 3 (as used in the analysis)
    - shuffle: True
    - random_state: 42 (for reproducibility)
    - groups: Sample IDs (keeps repetitions together)
    - stratification: 5-bin quantile discretization of target values
""")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate unmerged datasets using SPXY splitting strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses the SPXY splitting strategy (winner of the enhanced splitter
selection analysis) to create train/test splits for all datasets.

Output structure for each dataset:
    - X_train.csv, Y_train.csv, M_train.csv (training data)
    - X_test.csv, Y_test.csv, M_test.csv (test data)
    - fold_assignments.csv (CV fold reference)
    - split_info.json (split statistics)
        """
    )

    parser.add_argument(
        '--test_size', '-t',
        type=float,
        default=0.2,
        help='Fraction for test set (default: 0.2)'
    )
    parser.add_argument(
        '--random_state', '-r',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--n_folds', '-f',
        type=int,
        default=3,
        help='Number of CV folds for reference (default: 3)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    main(
        test_size=args.test_size,
        random_state=args.random_state,
        n_folds=args.n_folds,
        verbose=not args.quiet
    )
