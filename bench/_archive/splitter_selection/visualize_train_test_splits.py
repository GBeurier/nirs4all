"""
Visualization of Train/Test Splits
===================================

Creates a comprehensive visualization showing:
- Left: Spectral distribution (min/max/mean envelope) for train vs test
- Right: Y-value histograms for each dataset (train vs test overlap)

This helps verify that the SPXY splitting creates representative splits
with good coverage of both spectral and target spaces.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color palette - Teal/Coral modern palette
TRAIN_COLOR = '#0d9488'  # Teal
TRAIN_COLOR_DARK = '#0f766e'
TEST_COLOR = '#f97316'  # Orange/Coral
TEST_COLOR_DARK = '#ea580c'

def load_split_data(dataset_dir: Path) -> dict:
    """Load train/test X and Y data from a split dataset."""
    # Detect X separator
    x_train_path = dataset_dir / 'X_train.csv'
    with open(x_train_path) as f:
        first_line = f.readline()
    x_sep = ';' if ';' in first_line else ','

    # Load data
    X_train = pd.read_csv(dataset_dir / 'X_train.csv', sep=x_sep).values
    X_test = pd.read_csv(dataset_dir / 'X_test.csv', sep=x_sep).values
    Y_train = pd.read_csv(dataset_dir / 'Y_train.csv').iloc[:, 0].values
    Y_test = pd.read_csv(dataset_dir / 'Y_test.csv').iloc[:, 0].values

    return {
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'name': dataset_dir.name
    }

def plot_spectral_distribution(ax, X_train: np.ndarray, X_test: np.ndarray,
                                title: str = "Spectral Distribution"):
    """
    Plot spectral distribution showing envelope (min/max/mean) for train and test.
    """
    wavelengths = np.arange(X_train.shape[1])

    # Calculate statistics for train
    train_mean = np.mean(X_train, axis=0)
    train_min = np.min(X_train, axis=0)
    train_max = np.max(X_train, axis=0)
    train_p25 = np.percentile(X_train, 25, axis=0)
    train_p75 = np.percentile(X_train, 75, axis=0)

    # Calculate statistics for test
    test_mean = np.mean(X_test, axis=0)
    test_min = np.min(X_test, axis=0)
    test_max = np.max(X_test, axis=0)
    test_p25 = np.percentile(X_test, 25, axis=0)
    test_p75 = np.percentile(X_test, 75, axis=0)

    # Plot train envelope
    ax.fill_between(wavelengths, train_min, train_max,
                    alpha=0.15, color=TRAIN_COLOR, label='Train (min-max)')
    ax.fill_between(wavelengths, train_p25, train_p75,
                    alpha=0.3, color=TRAIN_COLOR, label='Train (IQR)')
    ax.plot(wavelengths, train_mean, color=TRAIN_COLOR_DARK, linewidth=2,
            label='Train mean', alpha=0.95)

    # Plot test envelope
    ax.fill_between(wavelengths, test_min, test_max,
                    alpha=0.15, color=TEST_COLOR, label='Test (min-max)')
    ax.fill_between(wavelengths, test_p25, test_p75,
                    alpha=0.3, color=TEST_COLOR, label='Test (IQR)')
    ax.plot(wavelengths, test_mean, color=TEST_COLOR_DARK, linewidth=2,
            label='Test mean', alpha=0.95)

    ax.set_xlabel('Wavelength index', fontsize=12)
    ax.set_ylabel('Absorbance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, len(wavelengths) - 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_y_histogram(ax, Y_train: np.ndarray, Y_test: np.ndarray,
                      dataset_name: str, show_legend: bool = True):
    """
    Plot overlapping histograms of Y values for train and test sets.
    """
    # Determine bins based on combined data
    all_y = np.concatenate([Y_train, Y_test])
    bins = np.linspace(all_y.min(), all_y.max(), 20)

    # Plot histograms
    ax.hist(Y_train, bins=bins, alpha=0.55, color=TRAIN_COLOR,
            label=f'Train (n={len(Y_train)})', density=True, edgecolor=TRAIN_COLOR_DARK, linewidth=0.8)
    ax.hist(Y_test, bins=bins, alpha=0.55, color=TEST_COLOR,
            label=f'Test (n={len(Y_test)})', density=True, edgecolor=TEST_COLOR_DARK, linewidth=0.8)

    # Add vertical lines for means
    train_mean = np.mean(Y_train)
    test_mean = np.mean(Y_test)
    ax.axvline(train_mean, color=TRAIN_COLOR_DARK, linestyle='--', linewidth=2, alpha=0.9)
    ax.axvline(test_mean, color=TEST_COLOR_DARK, linestyle='--', linewidth=2, alpha=0.9)

    # Title and labels
    short_name = dataset_name.replace('_', ' ')
    ax.set_title(short_name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Y value', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)

    if show_legend:
        ax.legend(fontsize=7, loc='upper right')

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_comprehensive_visualization(regression_dir: Path, classif_dir: Path,
                                         output_path: Path = None):
    """
    Create comprehensive train/test split visualization.

    Left: Large spectral distribution plot
    Right: Grid of Y histograms for all datasets
    """
    # Collect all datasets
    datasets = []

    # Load regression datasets
    if regression_dir.exists():
        for subdir in sorted(regression_dir.iterdir()):
            if subdir.is_dir() and (subdir / 'X_train.csv').exists():
                datasets.append(load_split_data(subdir))

    # Load classification datasets
    if classif_dir.exists():
        for subdir in sorted(classif_dir.iterdir()):
            if subdir.is_dir() and (subdir / 'X_train.csv').exists():
                datasets.append(load_split_data(subdir))

    if not datasets:
        print("No datasets found!")
        return

    print(f"Found {len(datasets)} datasets")

    # Create figure with specific layout
    # Left: 1 large spectral plot, Right: grid of histograms
    n_datasets = len(datasets)
    n_cols_hist = 2
    n_rows_hist = (n_datasets + 1) // 2

    fig = plt.figure(figsize=(32, 13.5))  # Wider 16:9ish aspect ratio

    # Use GridSpec for flexible layout - 6 columns: 4 for spectra, 2 for histograms
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows_hist, 6, figure=fig, wspace=0.3, hspace=0.4,
                  left=0.04, right=0.98, top=0.92, bottom=0.06)

    # Left side: Spectral plot (spans all rows, columns 0-3 = 4/6 = 2/3 of width)
    ax_spectral = fig.add_subplot(gs[:, 0:4])

    # Combine all X data for spectral plot
    X_train_all = np.vstack([d['X_train'] for d in datasets])
    X_test_all = np.vstack([d['X_test'] for d in datasets])

    plot_spectral_distribution(
        ax_spectral, X_train_all, X_test_all,
        title=f"Spectral Distribution: Train ({len(X_train_all)}) vs Test ({len(X_test_all)})"
    )

    # Right side: Y histograms (columns 4-5)
    for idx, data in enumerate(datasets):
        row = idx // n_cols_hist
        col = 4 + (idx % n_cols_hist)

        ax_hist = fig.add_subplot(gs[row, col])
        plot_y_histogram(
            ax_hist, data['Y_train'], data['Y_test'],
            data['name'],
            show_legend=(idx == 0)  # Only first histogram shows legend
        )

    # Super title
    fig.suptitle(
        'SPXY Train/Test Split Visualization\n'
        'Spectral coverage and target distribution overlap',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Add legend patch explanation at bottom
    train_patch = mpatches.Patch(color=TRAIN_COLOR, alpha=0.6, label='Train')
    test_patch = mpatches.Patch(color=TEST_COLOR, alpha=0.6, label='Test')
    fig.legend(handles=[train_patch, test_patch],
               loc='lower center', ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 0.005), framealpha=0.9)

    # Save
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved visualization to: {output_path}")

    plt.show()
    return fig

def main():
    """Main function."""
    base_dir = Path(__file__).parent
    regression_dir = base_dir / 'nitro_regression_unmerged'
    classif_dir = base_dir / 'nitro_classif_unmerged'
    output_path = base_dir / 'train_test_split_visualization.png'

    print("Creating train/test split visualization...")
    print(f"  Regression datasets: {regression_dir}")
    print(f"  Classification datasets: {classif_dir}")

    fig = create_comprehensive_visualization(
        regression_dir, classif_dir, output_path
    )

if __name__ == '__main__':
    main()
