"""
Study Report Generator - Automated Report Generation for NIRS Analysis
=======================================================================

This script automatically generates comprehensive reports from prediction parquet files
created by study_training.py. It generates both PDF reports and export packages with
all artifacts (models, predictions, charts, etc.).

The report includes:
- Cover page with logos and authors
- Dataset description and statistics
- Experimental protocol documentation
- Global results (histograms, candlestick plots)
- Ranking analysis (heatmaps, top-k tables)
- Individual model diagnostics (confusion matrices, scatter plots, ROC curves)
- Export summary with file descriptions
- Methodology annexes

Usage:
    python study_report.py --workspace wk --output reports/
    python study_report.py --workspace wk --filenames digestibility_custom2 hardness_0_8
    python study_report.py --workspace wk --format pdf --include-raw
"""

import argparse
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

# NIRS4All imports
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.utils.header_units import get_x_values_and_label


# ========================================
# CONFIGURATION
# ========================================

# Model filtering and renaming (reuse from predictions notebook)
EXCLUDE_MODELS = ["KernelPLS"]
MODEL_RENAME_MAP = {
    "dict": "nicon",
    "tabpfn": "transformer",
    "tunedtabpfn": "tuned_transformer",
}

# Authors information
AUTHORS = [
    "gregory.beurier@cirad.fr",
    "denis.cornet@cirad.fr",
    "lauriane.rouan@cirad.fr"
]

# Git repository
GIT_URL = "https://github.com/GBeurier/nirs4all"

# Logo paths (relative to bench directory)
LOGO_NIRS4ALL = Path(__file__).parent / "nirs4all_logo.png"
LOGO_CIRAD = Path(__file__).parent / "logo-cirad-en.jpg"

# Method documentation
METHOD_DOC_PATH = Path(__file__).parent / "study_method.md"


# ========================================
# PANDOC SUPPORT
# ========================================

def is_pandoc_available() -> bool:
    """Check if pandoc is available on the system.

    Returns:
        True if pandoc is available, False otherwise.
    """
    try:
        import pypandoc
        pypandoc.get_pandoc_version()
        return True
    except (ImportError, OSError):
        return False


def render_markdown_to_pdf_pandoc(markdown_path: Path, output_pdf_path: Path) -> bool:
    """Render markdown file to PDF using pandoc.

    This provides high-quality rendering of LaTeX equations, tables,
    and other markdown features.

    Args:
        markdown_path: Path to the markdown file.
        output_pdf_path: Path for the output PDF file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import pypandoc

        # Convert markdown to PDF with LaTeX engine for best equation rendering
        pypandoc.convert_file(
            str(markdown_path),
            'pdf',
            outputfile=str(output_pdf_path),
            extra_args=[
                '--pdf-engine=pdflatex',
                '-V', 'geometry:margin=1in',
                '-V', 'fontsize=10pt',
                '-V', 'documentclass=article',
                '--highlight-style=tango',
            ]
        )
        return True
    except Exception as e:
        print(f"    ⚠️ Pandoc rendering failed: {e}")
        # Try with xelatex as fallback (better unicode support)
        try:
            pypandoc.convert_file(
                str(markdown_path),
                'pdf',
                outputfile=str(output_pdf_path),
                extra_args=[
                    '--pdf-engine=xelatex',
                    '-V', 'geometry:margin=1in',
                    '-V', 'fontsize=10pt',
                ]
            )
            return True
        except Exception as e2:
            print(f"    ⚠️ Pandoc xelatex fallback also failed: {e2}")
            return False


# ========================================
# UTILITY FUNCTIONS
# ========================================

def apply_model_filters(predictions: Predictions, exclude_models: list, rename_map: dict,
                        strip_suffixes: Optional[list] = None) -> Predictions:
    """Filter out excluded models and rename model_classname values.

    Args:
        predictions: Predictions object to filter/rename.
        exclude_models: List of model_classname values to exclude (case-insensitive partial match).
        rename_map: Dict mapping old pattern to new names (case-insensitive partial match).
        strip_suffixes: List of suffixes to remove from model names (case-insensitive).

    Returns:
        Filtered and renamed Predictions object (modifies in place).
    """
    if strip_suffixes is None:
        strip_suffixes = ["classifier", "regressor"]

    df = predictions._storage._df
    original_count = len(df)

    # Exclude models
    if exclude_models:
        exclude_pattern = '(?i)(' + '|'.join(re.escape(m) for m in exclude_models) + ')'
        df = df.filter(~pl.col("model_classname").str.contains(exclude_pattern))
        excluded_count = original_count - len(df)
        if excluded_count > 0:
            print(f"  🚫 Excluded {excluded_count} predictions matching: {exclude_models}")

    # Strip suffixes
    if strip_suffixes:
        suffix_pattern = '(?i)(' + '|'.join(re.escape(s) for s in strip_suffixes) + ')$'
        df = df.with_columns([
            pl.col("model_classname").str.replace(suffix_pattern, "").alias("model_classname"),
            pl.col("model_name").str.replace(suffix_pattern, "").alias("model_name"),
        ])

    # Rename models
    if rename_map:
        for old_pattern, new_name in rename_map.items():
            pattern = f'(?i){re.escape(old_pattern)}'
            match_count = df.filter(pl.col("model_classname").str.contains(pattern)).height

            if match_count > 0:
                df = df.with_columns([
                    pl.when(pl.col("model_classname").str.contains(pattern))
                    .then(pl.lit(new_name))
                    .otherwise(pl.col("model_classname"))
                    .alias("model_classname"),
                    pl.when(pl.col("model_name").str.contains(pattern))
                    .then(pl.lit(new_name))
                    .otherwise(pl.col("model_name"))
                    .alias("model_name"),
                ])
                print(f"  ✏️ Renamed {match_count} predictions matching '{old_pattern}' → '{new_name}'")

    predictions._storage._df = df
    return predictions


def detect_task_type(predictions: Predictions) -> Tuple[bool, str, List[str], List[str]]:
    """Detect task type and return appropriate metrics.

    Args:
        predictions: Predictions object

    Returns:
        Tuple of (is_classification, rank_metric, display_metrics, chart_metrics)
    """
    task_types = predictions.get_unique_values('task_type')
    is_classification = any(t and 'classification' in str(t).lower() for t in task_types)

    if is_classification:
        rank_metric = 'balanced_accuracy'
        display_metrics = ['balanced_accuracy', 'accuracy', 'f1']
        chart_metrics = ['balanced_accuracy', 'accuracy', 'f1', 'precision', 'recall']
    else:
        rank_metric = 'rmse'
        display_metrics = ['rmse', 'mse', 'mae', 'r2']
        chart_metrics = ['rmse', 'mae', 'r2']

    return is_classification, rank_metric, display_metrics, chart_metrics


# ========================================
# REPORT GENERATION CLASSES
# ========================================

class ReportGenerator:
    """Generates comprehensive analysis reports."""

    def __init__(self, output_dir: Path, filename: str):
        """Initialize report generator.

        Args:
            output_dir: Output directory for reports
            filename: Dataset filename (without extension)
        """
        import tempfile
        self.output_dir = Path(output_dir)
        self.filename = filename
        # Use temporary directory for charts (will be moved to ZIP)
        self.temp_dir = tempfile.mkdtemp(prefix=f"nirs_report_{filename}_")
        self.charts_dir = Path(self.temp_dir) / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        self.figures: List[Tuple[str, Figure]] = []
        self.markdown_content = []

    def _add_dataset_visualizations(self, pdf: PdfPages, dataset_folder: Path, is_classification: bool):
        """Add dataset visualizations (spectra chart and y histogram).

        Args:
            pdf: PdfPages object
            dataset_folder: Path to dataset folder
            is_classification: Whether this is a classification task
        """
        from nirs4all.data import DatasetConfigs

        print("    📊 Generating dataset visualizations...")

        try:
            # Load dataset
            dataset_config = DatasetConfigs(str(dataset_folder))
            dataset = dataset_config.get_dataset_at(0)

            # Get data for train and test
            x_train = dataset.x({"partition": "train"}, layout="2d")
            y_train = dataset.y({"partition": "train"})
            x_test = dataset.x({"partition": "test"}, layout="2d")
            y_test = dataset.y({"partition": "test"})

            # Get headers if available
            headers = None
            header_unit = "cm-1"
            try:
                # Try to get headers from the dataset (first source, first processing)
                headers = dataset.headers(0)
                header_unit = dataset.header_unit(0) or "cm-1"
            except:
                pass

            # 1. Create 2D Spectra Chart (train/test)
            print("      📈 Creating 2D spectra chart...")
            fig_spectra = self._create_spectra_chart(x_train, y_train, x_test, y_test,
                                                      headers, header_unit, is_classification)
            if fig_spectra:
                pdf.savefig(fig_spectra, bbox_inches='tight')
                self._save_chart(fig_spectra, "dataset_spectra_2d")
                plt.close(fig_spectra)
                self.markdown_content.append("### Dataset Spectra (2D)\n\n")
                self.markdown_content.append("![Dataset Spectra](charts/dataset_spectra_2d.png)\n\n")

            # 2. Create Y Distribution Chart
            print("      📊 Creating y distribution chart...")
            fig_y = self._create_y_chart(y_train, y_test, is_classification)
            if fig_y:
                pdf.savefig(fig_y, bbox_inches='tight')
                self._save_chart(fig_y, "dataset_y_distribution")
                plt.close(fig_y)
                self.markdown_content.append("### Target Distribution (Y)\n\n")
                self.markdown_content.append("![Y Distribution](charts/dataset_y_distribution.png)\n\n")

        except Exception as e:
            print(f"      ⚠️  Could not generate dataset visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _create_spectra_chart(self, x_train: np.ndarray, y_train: np.ndarray,
                              x_test: np.ndarray, y_test: np.ndarray,
                              headers: Optional[list], header_unit: str,
                              is_classification: bool) -> Optional[Figure]:
        """Create 2D spectra visualization with train/test split.

        Args:
            x_train: Training spectra
            y_train: Training targets
            x_test: Test spectra
            y_test: Test targets
            headers: Feature headers (wavelengths)
            header_unit: Unit of headers
            is_classification: Whether this is a classification task

        Returns:
            Matplotlib Figure or None if creation failed
        """
        try:
            # Combine train and test data
            x_all = np.vstack([x_train, x_test])
            y_all = np.concatenate([y_train, y_test])

            # Sort by y values
            sorted_indices = np.argsort(y_all.flatten())
            x_sorted = x_all[sorted_indices]
            y_sorted = y_all.flatten()[sorted_indices]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Get x-values and label using centralized utility
            n_features = x_sorted.shape[1]
            x_values, x_label = get_x_values_and_label(headers, header_unit, n_features)

            # Create colormap
            if is_classification:
                unique_values = np.unique(y_sorted)
                n_unique = len(unique_values)
                if n_unique <= 10:
                    colormap = cm.get_cmap('tab10', n_unique)
                elif n_unique <= 20:
                    colormap = cm.get_cmap('tab20', n_unique)
                else:
                    colormap = cm.get_cmap('hsv', n_unique)
                value_to_index = {val: idx for idx, val in enumerate(unique_values)}
                y_normalized = np.array([value_to_index[val] / max(n_unique - 1, 1) for val in y_sorted])
            else:
                colormap = cm.get_cmap('viridis')
                y_min, y_max = y_sorted.min(), y_sorted.max()
                if y_max != y_min:
                    y_normalized = (y_sorted - y_min) / (y_max - y_min)
                else:
                    y_normalized = np.zeros_like(y_sorted)

            # Plot each spectrum as a line
            for i, (spectrum, y_val) in enumerate(zip(x_sorted, y_sorted)):
                color = colormap(y_normalized[i])
                ax.plot(x_values, spectrum, color=color, alpha=0.6, linewidth=0.5)

            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel('Intensity', fontsize=10)
            ax.set_title('Dataset Spectra (Train + Test)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add colorbar
            if is_classification:
                unique_values = np.unique(y_sorted)
                n_unique = len(unique_values)
                mappable = cm.ScalarMappable(cmap=colormap)
                mappable.set_array(np.arange(n_unique))
                cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=10)
                if n_unique <= 20:
                    cbar.ax.set_yticklabels([str(val) for val in unique_values])
            else:
                mappable = cm.ScalarMappable(cmap=colormap)
                mappable.set_array(y_sorted)
                cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=10)

            cbar.set_label('y', fontsize=8)
            cbar.ax.tick_params(labelsize=7)

            return fig

        except Exception as e:
            print(f"      ⚠️  Could not create spectra chart: {e}")
            return None

    def _create_y_chart(self, y_train: np.ndarray, y_test: np.ndarray,
                       is_classification: bool) -> Optional[Figure]:
        """Create y distribution histogram with train/test split.

        Args:
            y_train: Training targets
            y_test: Test targets
            is_classification: Whether this is a classification task

        Returns:
            Matplotlib Figure or None if creation failed
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            y_train_flat = y_train.flatten()
            y_test_flat = y_test.flatten()
            y_all_flat = np.concatenate([y_train_flat, y_test_flat])

            # Determine if categorical or continuous
            unique_values = np.unique(y_all_flat)
            is_categorical = len(unique_values) <= 20 or y_all_flat.dtype.kind in {'U', 'S', 'O'}

            # Use viridis colormap
            viridis_cmap = cm.get_cmap('viridis')
            train_color = viridis_cmap(0.9)  # Bright yellow-green for train
            test_color = viridis_cmap(0.1)   # Dark purple-blue for test

            if is_categorical:
                # Categorical data: grouped bar plot
                train_counts = np.zeros(len(unique_values))
                test_counts = np.zeros(len(unique_values))

                for i, val in enumerate(unique_values):
                    train_counts[i] = np.sum(y_train_flat == val)
                    test_counts[i] = np.sum(y_test_flat == val)

                x_pos = np.arange(len(unique_values))
                width = 0.8

                ax.bar(x_pos, train_counts, width, label='Train', color=train_color)
                ax.bar(x_pos, test_counts, width, bottom=train_counts, label='Test', color=test_color)

                ax.set_xlabel('Y Categories')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(val) for val in unique_values], rotation=45)
                title = 'Y Distribution: Train vs Test (Categorical)'
            else:
                # Continuous data: overlapping histograms
                bins = 30
                ax.hist(y_train_flat, bins=bins, label='Train', color=train_color, alpha=0.5, edgecolor='none')
                ax.hist(y_test_flat, bins=bins, label='Test', color=test_color, alpha=0.8, edgecolor='none')
                ax.set_xlabel('Y Values')
                title = 'Y Distribution: Train vs Test (Continuous)'

            ax.set_ylabel('Count')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add statistics
            train_stats = f'Train (n={len(y_train_flat)}):\nMean: {np.mean(y_train_flat):.3f}\nStd: {np.std(y_train_flat):.3f}'
            test_stats = f'Test (n={len(y_test_flat)}):\nMean: {np.mean(y_test_flat):.3f}\nStd: {np.std(y_test_flat):.3f}'

            ax.text(0.02, 0.98, train_stats, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=train_color, edgecolor='black'),
                    color='black')
            ax.text(0.02, 0.75, test_stats, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=test_color, edgecolor='white'),
                    color='white')

            return fig

        except Exception as e:
            print(f"      ⚠️  Could not create y chart: {e}")
            return None

    def _add_subtitle_page(self, pdf: PdfPages, title: str, description: str):
        """Add a subtitle page explaining the upcoming charts.

        Args:
            pdf: PdfPages object
            title: Title for the subtitle page
            description: Description text
        """
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis('off')
        ax.text(0.5, 0.6, title, ha='center', va='center',
                fontsize=14, fontweight='bold', transform=ax.transAxes,
                family='sans-serif')

        # Wrap description text
        from textwrap import wrap
        wrapped_lines = []
        for line in description.split('\n'):
            if line.strip():
                wrapped_lines.extend(wrap(line, width=70))
            else:
                wrapped_lines.append('')

        y_pos = 0.45
        for line in wrapped_lines:
            ax.text(0.5, y_pos, line, ha='center', va='top',
                    fontsize=9, transform=ax.transAxes, family='sans-serif')
            y_pos -= 0.03

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def add_cover_page(self, pdf: PdfPages):
        """Add cover page with logos and title.

        Args:
            pdf: PdfPages object
        """
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Add logos if they exist (keeping aspect ratio)
        y_pos = 0.9
        if LOGO_NIRS4ALL.exists():
            try:
                from PIL import Image
                img = Image.open(LOGO_NIRS4ALL)
                img_width, img_height = img.size
                aspect = img_height / img_width
                logo_width = 0.25
                logo_height = logo_width * aspect
                ax.imshow(img, extent=[0.1, 0.1 + logo_width, y_pos - logo_height, y_pos])
            except Exception as e:
                print(f"  ⚠️ Could not load nirs4all logo: {e}")

        if LOGO_CIRAD.exists():
            try:
                from PIL import Image
                img = Image.open(LOGO_CIRAD)
                img_width, img_height = img.size
                aspect = img_height / img_width
                logo_width = 0.25
                logo_height = logo_width * aspect
                ax.imshow(img, extent=[0.65, 0.65 + logo_width, y_pos - logo_height, y_pos])
            except Exception as e:
                print(f"  ⚠️ Could not load CIRAD logo: {e}")

        # Title
        ax.text(0.5, 0.6, 'NIRS Analysis Report', ha='center', va='center',
                fontsize=24, fontweight='bold', transform=ax.transAxes,
                family='sans-serif', color='#2c3e50')

        # Dataset name
        ax.text(0.5, 0.5, f'Dataset: {self.filename}', ha='center', va='center',
                fontsize=14, transform=ax.transAxes, family='sans-serif',
                color='#34495e')

        # Date
        date_str = datetime.now().strftime("%Y-%m-%d")
        ax.text(0.5, 0.4, f'Generated: {date_str}', ha='center', va='center',
                fontsize=10, transform=ax.transAxes, family='sans-serif',
                color='#7f8c8d')

        # Git URL
        ax.text(0.5, 0.35, GIT_URL, ha='center', va='center',
                fontsize=8, style='italic', color='#3498db', transform=ax.transAxes,
                family='monospace')

        # Authors
        y_authors = 0.25
        ax.text(0.5, y_authors, 'Authors:', ha='center', va='center',
                fontsize=10, fontweight='bold', transform=ax.transAxes,
                family='sans-serif')
        for i, author in enumerate(AUTHORS):
            ax.text(0.5, y_authors - 0.03 * (i+1), author, ha='center', va='center',
                    fontsize=8, transform=ax.transAxes, family='sans-serif')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Add to markdown
        self.markdown_content.append("# NIRS Analysis Report\n")
        self.markdown_content.append(f"\n**Dataset:** {self.filename}\n")
        self.markdown_content.append(f"**Generated:** {date_str}\n\n")
        self.markdown_content.append("**Authors:**\n")
        for author in AUTHORS:
            self.markdown_content.append(f"- {author}\n")
        self.markdown_content.append("\n---\n\n")

    def add_section_1_dataset_description(self, pdf: PdfPages, predictions: Predictions,
                                           is_classification: bool, dataset_folder: Optional[Path] = None):
        """Add Section 1: Dataset Description.

        Args:
            pdf: PdfPages object
            predictions: Predictions object
            is_classification: Whether this is a classification task
            dataset_folder: Path to dataset folder (optional) for loading actual data
        """
        print("  📊 Section 1: Dataset Description")

        # Create text page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        # Get dataset statistics
        datasets = predictions.get_datasets()
        dataset_name = datasets[0] if datasets else self.filename

        # Get unique values for statistics
        models = predictions.get_unique_values('model_name')
        n_predictions = len(predictions)

        # Statistics text
        y_pos = 0.9
        ax.text(0.5, y_pos, '1. Dataset Description', ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes,
                family='sans-serif')
        y_pos -= 0.08

        stats_text = f"""
Dataset Name: {dataset_name}

Number of predictions: {n_predictions}
Number of unique models: {len(models)}
Task type: {'Classification' if is_classification else 'Regression'}

Models evaluated:
"""
        for model in sorted(models)[:15]:  # Show first 15
            stats_text += f"  - {model}\n"

        if len(models) > 15:
            stats_text += f"  ... and {len(models) - 15} more models"

        ax.text(0.1, y_pos, stats_text, ha='left', va='top',
                fontsize=8, family='monospace', transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Add to markdown
        self.markdown_content.append("## Section 1 – Dataset Description\n\n")
        self.markdown_content.append(f"**Dataset Name:** {dataset_name}\n\n")
        self.markdown_content.append(f"**Number of predictions:** {n_predictions}\n\n")
        self.markdown_content.append(f"**Number of unique models:** {len(models)}\n\n")
        self.markdown_content.append(f"**Task type:** {'Classification' if is_classification else 'Regression'}\n\n")
        self.markdown_content.append("**Models evaluated:**\n\n")
        for model in sorted(models):
            self.markdown_content.append(f"- {model}\n")
        self.markdown_content.append("\n")

        # Add dataset visualizations if dataset folder is provided
        if dataset_folder and dataset_folder.exists():
            self._add_dataset_visualizations(pdf, dataset_folder, is_classification)

    def add_section_2_protocol(self, pdf: PdfPages):
        """Add Section 2: Experimental Protocol.

        Args:
            pdf: PdfPages object
        """
        print("  📋 Section 2: Experimental Protocol")

        # Create text page with better styling
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis('off')

        y_pos = 0.92
        ax.text(0.5, y_pos, '2. Experimental Protocol', ha='center', va='top',
                fontsize=16, fontweight='bold', transform=ax.transAxes,
                family='sans-serif', color='#2c3e50')
        y_pos -= 0.10

        protocol_sections = [
            ("Overview", "This analysis follows a multi-phase cascading architecture to\nevaluate model performance across different preprocessing and\nmodeling strategies."),
            ("Phase 1: Transfer Preprocessing Selection", "Evaluate large preprocessing search space and select\ntop-K configurations for downstream evaluation."),
            ("Phase 2: PLS Baseline with Finetuning", "Train PLS models with automated component tuning to\nidentify best preprocessing combinations."),
            ("Phase 3: Ensemble and Deep Learning", "• Ridge regression with regularization tuning\n• Gradient boosting (CatBoost) configurations\n• 1D-CNN architectures (NICON variants)"),
            ("Phase 4: Transformer-Based Methods", "• Dimensionality reduction (PCA, SVD, Wavelets)\n• Transformer model with hyperparameter tuning"),
            ("Validation Strategy", "All models use consistent cross-validation with SPXYGFold\nfor fair comparison and reproducibility.")
        ]

        for title, content in protocol_sections:
            ax.text(0.12, y_pos, title, ha='left', va='top',
                    fontsize=10, fontweight='bold', transform=ax.transAxes,
                    family='sans-serif', color='#34495e')
            y_pos -= 0.04

            for line in content.split('\n'):
                ax.text(0.15, y_pos, line, ha='left', va='top',
                        fontsize=8, transform=ax.transAxes,
                        family='sans-serif', color='#2c3e50')
                y_pos -= 0.03

            y_pos -= 0.02  # Extra space between sections

        # Footer note
        ax.text(0.5, 0.12, 'See Annex for complete methodology documentation.',
                ha='center', va='top', fontsize=7, style='italic',
                transform=ax.transAxes, family='sans-serif', color='#7f8c8d')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Add to markdown
        self.markdown_content.append("## Section 2 – Experimental Protocol\n\n")
        for title, content in protocol_sections:
            self.markdown_content.append(f"**{title}**\n\n{content}\n\n")

    def add_section_3_global_results(self, pdf: PdfPages, analyzer: PredictionAnalyzer,
                                      rank_metric: str, is_classification: bool,
                                      aggregation_key: Optional[str] = None, mode: str = 'aggregated'):
        """Add Section 3: Global Results.

        Args:
            pdf: PdfPages object
            analyzer: PredictionAnalyzer instance
            rank_metric: Metric for ranking
            is_classification: Whether this is a classification task
            aggregation_key: Column name to aggregate by (None for raw)
            mode: Analysis mode - 'raw', 'aggregated', or 'both'
        """
        print("  📊 Section 3: Global Results")

        # Title page
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis('off')
        ax.text(0.5, 0.5, '3. Global Results', ha='center', va='center',
                fontsize=18, fontweight='bold', transform=ax.transAxes,
                family='sans-serif', color='#2c3e50')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        self.markdown_content.append("## Section 3 – Global Results\n\n")

        # Determine what to show based on mode
        show_aggregated = mode in ['aggregated', 'both']
        show_raw = mode in ['raw', 'both']

        # Aggregated results
        if show_aggregated and aggregation_key:
            self.markdown_content.append(f"### Aggregated Predictions (by {aggregation_key})\n\n")

            # Add subtitle page for aggregated results
            if mode == 'both':
                self._add_subtitle_page(pdf,
                    f"Aggregated Results (by {aggregation_key})",
                    f"The following charts show aggregated predictions where multiple repetitions\n"
                    f"or runs are combined by {aggregation_key}. This provides a robust view\n"
                    f"of model performance by averaging or aggregating across experimental runs.")

            # Histogram
            print(f"    📊 Histogram (aggregated by {aggregation_key})")
            try:
                fig_hist = analyzer.plot_histogram(
                    display_metric=rank_metric,
                    display_partition='test',
                    aggregate=aggregation_key
                )
                if isinstance(fig_hist, list):
                    for i, fig in enumerate(fig_hist):
                        self._save_chart(fig, f"histogram_{rank_metric}_test_agg_{i}")
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        self.markdown_content.append(f"![Histogram {i}](charts/histogram_{rank_metric}_test_agg_{i}.png)\n\n")
                else:
                    self._save_chart(fig_hist, f"histogram_{rank_metric}_test_agg")
                    pdf.savefig(fig_hist, bbox_inches='tight')
                    plt.close(fig_hist)
                    self.markdown_content.append(f"![Histogram](charts/histogram_{rank_metric}_test_agg.png)\n\n")
            except Exception as e:
                print(f"    ⚠️ Could not create histogram: {e}")

            # Candlestick
            print(f"    📊 Candlestick (aggregated by {aggregation_key})")
            try:
                fig_candle = analyzer.plot_candlestick(
                    variable="model_classname",
                    display_metric=rank_metric,
                    display_partition='test',
                    aggregate=aggregation_key
                )
                if isinstance(fig_candle, list):
                    for i, fig in enumerate(fig_candle):
                        self._save_chart(fig, f"candlestick_{rank_metric}_test_agg_{i}")
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        self.markdown_content.append(f"![Candlestick {i}](charts/candlestick_{rank_metric}_test_agg_{i}.png)\n\n")
                else:
                    self._save_chart(fig_candle, f"candlestick_{rank_metric}_test_agg")
                    pdf.savefig(fig_candle, bbox_inches='tight')
                    plt.close(fig_candle)
                    self.markdown_content.append(f"![Candlestick](charts/candlestick_{rank_metric}_test_agg.png)\n\n")
            except Exception as e:
                print(f"    ⚠️ Could not create candlestick: {e}")

        # Raw results
        if show_raw:
            self.markdown_content.append("### Raw Predictions\n\n")

            # Add subtitle page for raw results
            if mode == 'both':
                self._add_subtitle_page(pdf,
                    "Raw Results (Individual Predictions)",
                    "The following charts show raw, unaggregated predictions where each\n"
                    "experimental run or repetition is treated independently. This provides\n"
                    "insight into individual model performance and variability.")

            # Histogram
            print("    📊 Histogram (raw)")
            try:
                fig_hist_raw = analyzer.plot_histogram(
                    display_metric=rank_metric,
                    display_partition='test'
                )
                if isinstance(fig_hist_raw, list):
                    for i, fig in enumerate(fig_hist_raw):
                        self._save_chart(fig, f"histogram_{rank_metric}_test_raw_{i}")
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        self.markdown_content.append(f"![Histogram Raw {i}](charts/histogram_{rank_metric}_test_raw_{i}.png)\n\n")
                else:
                    self._save_chart(fig_hist_raw, f"histogram_{rank_metric}_test_raw")
                    pdf.savefig(fig_hist_raw, bbox_inches='tight')
                    plt.close(fig_hist_raw)
                    self.markdown_content.append(f"![Histogram Raw](charts/histogram_{rank_metric}_test_raw.png)\n\n")
            except Exception as e:
                print(f"    ⚠️ Could not create raw histogram: {e}")

            # Candlestick
            print("    📊 Candlestick (raw)")
            try:
                fig_candle_raw = analyzer.plot_candlestick(
                    variable="model_classname",
                    display_metric=rank_metric,
                    display_partition='test'
                )
                if isinstance(fig_candle_raw, list):
                    for i, fig in enumerate(fig_candle_raw):
                        self._save_chart(fig, f"candlestick_{rank_metric}_test_raw_{i}")
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        self.markdown_content.append(f"![Candlestick Raw {i}](charts/candlestick_{rank_metric}_test_raw_{i}.png)\n\n")
                else:
                    self._save_chart(fig_candle_raw, f"candlestick_{rank_metric}_test_raw")
                    pdf.savefig(fig_candle_raw, bbox_inches='tight')
                    plt.close(fig_candle_raw)
                    self.markdown_content.append(f"![Candlestick Raw](charts/candlestick_{rank_metric}_test_raw.png)\n\n")
            except Exception as e:
                print(f"    ⚠️ Could not create raw candlestick: {e}")

    def add_section_4_ranking_analysis(self, pdf: PdfPages, analyzer: PredictionAnalyzer,
                                        rank_metric: str, aggregation_key: Optional[str] = None,
                                        mode: str = 'aggregated'):
        """Add Section 4: Ranking Analysis.

        Args:
            pdf: PdfPages object
            analyzer: PredictionAnalyzer instance
            rank_metric: Metric for ranking
            aggregation_key: Column name to aggregate by (None for raw)
            mode: Analysis mode - 'raw', 'aggregated', or 'both'
        """
        print("  📊 Section 4: Ranking Analysis")

        # Title page
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis('off')
        ax.text(0.5, 0.5, '4. Ranking Analysis', ha='center', va='center',
                fontsize=18, fontweight='bold', transform=ax.transAxes,
                family='sans-serif', color='#2c3e50')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        self.markdown_content.append("## Section 4 – Ranking Analysis\n\n")

        # Common heatmap config
        from nirs4all.visualization.charts.config import ChartConfig
        hm_config = ChartConfig(annotation_fontsize=18)

        # Determine what to show based on mode
        show_aggregated = mode in ['aggregated', 'both']
        show_raw = mode in ['raw', 'both']

        # Aggregated heatmaps
        if show_aggregated and aggregation_key:
            self.markdown_content.append(f"### Aggregated Predictions (by {aggregation_key})\n\n")

            # Add subtitle page for aggregated heatmaps
            if mode == 'both':
                self._add_subtitle_page(pdf,
                    f"Aggregated Heatmaps (by {aggregation_key})",
                    "The following heatmaps show model performance across different data partitions\n"
                    "(train/val/test) with aggregated predictions. Models are ranked and displayed\n"
                    "based on their validation or test performance.")

            # Heatmap ranked by VAL
            print(f"    📊 Heatmap (ranked by val, aggregated by {aggregation_key})")
            try:
                fig_hm_val = analyzer.plot_heatmap(
                    x_var="partition",
                    y_var="model_classname",
                    rank_metric=rank_metric,
                    display_metric=rank_metric,
                    show_counts=False,
                    rank_partition='val',
                    aggregate=aggregation_key,
                    column_scale=True,
                    top_k=20,
                    sort_by='value',
                    config=hm_config
                )
                self._save_chart(fig_hm_val, f"heatmap_{rank_metric}_rank_val_agg")
                pdf.savefig(fig_hm_val, bbox_inches='tight')
                plt.close(fig_hm_val)
                self.markdown_content.append(f"![Heatmap Val](charts/heatmap_{rank_metric}_rank_val_agg.png)\n\n")
            except Exception as e:
                print(f"    ⚠️ Could not create val heatmap: {e}")

            # Heatmap ranked by TEST
            print(f"    📊 Heatmap (ranked by test, aggregated by {aggregation_key})")
            try:
                fig_hm_test = analyzer.plot_heatmap(
                    x_var="partition",
                    y_var="model_classname",
                    rank_metric=rank_metric,
                    display_metric=rank_metric,
                    show_counts=False,
                    rank_partition='test',
                    aggregate=aggregation_key,
                    column_scale=True,
                    top_k=6,
                    sort_by='value',
                    config=hm_config
                )
                self._save_chart(fig_hm_test, f"heatmap_{rank_metric}_rank_test_agg")
                pdf.savefig(fig_hm_test, bbox_inches='tight')
                plt.close(fig_hm_test)
                self.markdown_content.append(f"![Heatmap Test](charts/heatmap_{rank_metric}_rank_test_agg.png)\n\n")
            except Exception as e:
                print(f"    ⚠️ Could not create test heatmap: {e}")

        # Raw heatmaps
        if show_raw:
            self.markdown_content.append("### Raw Predictions\n\n")

            # Add subtitle page for raw heatmaps
            if mode == 'both':
                self._add_subtitle_page(pdf,
                    "Raw Heatmaps (Individual Predictions)",
                    "The following heatmaps show model performance across different data partitions\n"
                    "(train/val/test) with raw, unaggregated predictions. Each experimental run\n"
                    "is shown independently.")

            # Heatmap ranked by VAL
            print("    📊 Heatmap (ranked by val, raw)")
            try:
                fig_hm_val_raw = analyzer.plot_heatmap(
                    x_var="partition",
                    y_var="model_classname",
                    rank_metric=rank_metric,
                    display_metric=rank_metric,
                    show_counts=False,
                    rank_partition='val',
                    column_scale=True,
                    top_k=20,
                    sort_by='value',
                    config=hm_config
                )
                self._save_chart(fig_hm_val_raw, f"heatmap_{rank_metric}_rank_val_raw")
                pdf.savefig(fig_hm_val_raw, bbox_inches='tight')
                plt.close(fig_hm_val_raw)
                self.markdown_content.append(f"![Heatmap Val Raw](charts/heatmap_{rank_metric}_rank_val_raw.png)\n\n")
            except Exception as e:
                print(f"    ⚠️ Could not create val raw heatmap: {e}")

            # Heatmap ranked by TEST
            print("    📊 Heatmap (ranked by test, raw)")
            try:
                fig_hm_test_raw = analyzer.plot_heatmap(
                    x_var="partition",
                    y_var="model_classname",
                    rank_metric=rank_metric,
                    display_metric=rank_metric,
                    show_counts=False,
                    rank_partition='test',
                    column_scale=True,
                    top_k=6,
                    sort_by='value',
                    config=hm_config
                )
                self._save_chart(fig_hm_test_raw, f"heatmap_{rank_metric}_rank_test_raw")
                pdf.savefig(fig_hm_test_raw, bbox_inches='tight')
                plt.close(fig_hm_test_raw)
                self.markdown_content.append(f"![Heatmap Test Raw](charts/heatmap_{rank_metric}_rank_test_raw.png)\n\n")
            except Exception as e:
                print(f"    ⚠️ Could not create test raw heatmap: {e}")

    def add_section_5_model_diagnostics(self, pdf: PdfPages, analyzer: PredictionAnalyzer,
                                         predictions: Predictions, rank_metric: str,
                                         is_classification: bool, aggregation_key: Optional[str] = None,
                                         mode: str = 'aggregated'):
        """Add Section 5: Individual Model Diagnostics.

        Args:
            pdf: PdfPages object
            analyzer: PredictionAnalyzer instance
            predictions: Predictions object
            rank_metric: Metric for ranking
            is_classification: Whether this is a classification task
            aggregation_key: Column name to aggregate by (None for raw)
            mode: Analysis mode - 'raw', 'aggregated', or 'both'
        """
        print("  📊 Section 5: Individual Model Diagnostics")

        # Title page
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis('off')
        ax.text(0.5, 0.5, '5. Individual Model Diagnostics', ha='center', va='center',
                fontsize=18, fontweight='bold', transform=ax.transAxes,
                family='sans-serif', color='#2c3e50')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        self.markdown_content.append("## Section 5 – Individual Model Diagnostics\n\n")

        # Determine what to show based on mode
        show_aggregated = mode in ['aggregated', 'both']
        show_raw = mode in ['raw', 'both']

        if is_classification:
            # Confusion matrices for top 6 models

            # Aggregated confusion matrices
            if show_aggregated and aggregation_key:
                agg_text = f" (aggregated by {aggregation_key})"

                if mode == 'both':
                    self._add_subtitle_page(pdf,
                        f"Confusion Matrices - Aggregated (by {aggregation_key})",
                        "The following confusion matrices show classification performance for the\\n"
                        "top-ranked models using aggregated predictions. True labels are shown on\\n"
                        "the Y-axis and predicted labels on the X-axis.")

                print(f"    📊 Confusion matrices (top 6){agg_text}")
                self.markdown_content.append(f"### Confusion Matrices{agg_text}\n\n")
                try:
                    fig_cm = analyzer.plot_confusion_matrix(
                        rank_metric=rank_metric,
                        display_metric=rank_metric,
                        display_partition='test',
                        rank_partition='val',
                        aggregate=aggregation_key,
                        k=6
                    )
                    if isinstance(fig_cm, list):
                        for i, fig in enumerate(fig_cm):
                            self._save_chart(fig, f"confusion_matrix_{rank_metric}_agg_{i}")
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            self.markdown_content.append(f"![Confusion Matrix {i}](charts/confusion_matrix_{rank_metric}_agg_{i}.png)\n\n")
                    else:
                        self._save_chart(fig_cm, f"confusion_matrix_{rank_metric}_agg")
                        pdf.savefig(fig_cm, bbox_inches='tight')
                        plt.close(fig_cm)
                        self.markdown_content.append(f"![Confusion Matrix](charts/confusion_matrix_{rank_metric}_agg.png)\n\n")
                except Exception as e:
                    print(f"    ⚠️ Could not create confusion matrices: {e}")

            # Raw confusion matrices
            if show_raw:
                agg_text = " (raw)"

                if mode == 'both':
                    self._add_subtitle_page(pdf,
                        "Confusion Matrices - Raw (Individual Predictions)",
                        "The following confusion matrices show classification performance for the\\n"
                        "top-ranked models using raw, unaggregated predictions. Each experimental\\n"
                        "run is treated independently.")

                print(f"    📊 Confusion matrices (top 6){agg_text}")
                self.markdown_content.append(f"### Confusion Matrices{agg_text}\n\n")
                try:
                    fig_cm_raw = analyzer.plot_confusion_matrix(
                        rank_metric=rank_metric,
                        display_metric=rank_metric,
                        display_partition='test',
                        rank_partition='val',
                        k=6
                    )
                    if isinstance(fig_cm_raw, list):
                        for i, fig in enumerate(fig_cm_raw):
                            self._save_chart(fig, f"confusion_matrix_{rank_metric}_raw_{i}")
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            self.markdown_content.append(f"![Confusion Matrix Raw {i}](charts/confusion_matrix_{rank_metric}_raw_{i}.png)\n\n")
                    else:
                        self._save_chart(fig_cm_raw, f"confusion_matrix_{rank_metric}_raw")
                        pdf.savefig(fig_cm_raw, bbox_inches='tight')
                        plt.close(fig_cm_raw)
                        self.markdown_content.append(f"![Confusion Matrix Raw](charts/confusion_matrix_{rank_metric}_raw.png)\n\n")
                except Exception as e:
                    print(f"    ⚠️ Could not create raw confusion matrices: {e}")

        else:
            # Top-K comparison for regression

            # Aggregated top-k
            if show_aggregated and aggregation_key:
                agg_text = f" (aggregated by {aggregation_key})"

                if mode == 'both':
                    self._add_subtitle_page(pdf,
                        f"Top Models Comparison - Aggregated (by {aggregation_key})",
                        "The following scatter plots show predicted vs. true values for the\\n"
                        "top-ranked models using aggregated predictions. Each point represents\\n"
                        "an aggregated sample across experimental runs.")

                print(f"    📊 Top-3 models comparison{agg_text}")
                self.markdown_content.append(f"### Top Models Comparison{agg_text}\n\n")
                try:
                    fig_top3 = analyzer.plot_top_k(
                        k=3,
                        rank_metric=rank_metric,
                        rank_partition='val',
                        display_partition='test',
                        aggregate=aggregation_key
                    )
                    if isinstance(fig_top3, list):
                        for i, fig in enumerate(fig_top3):
                            self._save_chart(fig, f"top3_test_rank_val_agg_{i}")
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            self.markdown_content.append(f"![Top 3 Models {i}](charts/top3_test_rank_val_agg_{i}.png)\n\n")
                    else:
                        self._save_chart(fig_top3, f"top3_test_rank_val_agg")
                        pdf.savefig(fig_top3, bbox_inches='tight')
                        plt.close(fig_top3)
                        self.markdown_content.append(f"![Top 3 Models](charts/top3_test_rank_val_agg.png)\n\n")
                except Exception as e:
                    print(f"    ⚠️ Could not create top-K comparison: {e}")

            # Raw top-k
            if show_raw:
                agg_text = " (raw)"

                if mode == 'both':
                    self._add_subtitle_page(pdf,
                        "Top Models Comparison - Raw (Individual Predictions)",
                        "The following scatter plots show predicted vs. true values for the\\n"
                        "top-ranked models using raw predictions. Each point represents an\\n"
                        "individual prediction from a single experimental run.")

                print(f"    📊 Top-3 models comparison{agg_text}")
                self.markdown_content.append(f"### Top Models Comparison{agg_text}\n\n")
                try:
                    fig_top3_raw = analyzer.plot_top_k(
                        k=3,
                        rank_metric=rank_metric,
                        rank_partition='val',
                        display_partition='test'
                    )
                    if isinstance(fig_top3_raw, list):
                        for i, fig in enumerate(fig_top3_raw):
                            self._save_chart(fig, f"top3_test_rank_val_raw_{i}")
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            self.markdown_content.append(f"![Top 3 Models Raw {i}](charts/top3_test_rank_val_raw_{i}.png)\n\n")
                    else:
                        self._save_chart(fig_top3_raw, f"top3_test_rank_val_raw")
                        pdf.savefig(fig_top3_raw, bbox_inches='tight')
                        plt.close(fig_top3_raw)
                        self.markdown_content.append(f"![Top 3 Models Raw](charts/top3_test_rank_val_raw.png)\n\n")
                except Exception as e:
                    print(f"    ⚠️ Could not create raw top-K comparison: {e}")

        # Top models table
        agg_text = f" (aggregated by {aggregation_key})" if aggregation_key else " (raw)"
        self.markdown_content.append(f"### Top 5 Models{agg_text}\n\n")
        try:
            top_models = predictions.top(n=5, rank_metric=rank_metric, rank_partition='val', by_repetition=aggregation_key)
            self.markdown_content.append("| Rank | Model | Score |\n")
            self.markdown_content.append("|------|-------|-------|\n")
            for i, model in enumerate(top_models, 1):
                score = model.get('rank_score', 'N/A')
                model_name = model.get('model_name', 'Unknown')
                if isinstance(score, (int, float)):
                    self.markdown_content.append(f"| {i} | {model_name} | {score:.4f} |\n")
                else:
                    self.markdown_content.append(f"| {i} | {model_name} | {score} |\n")
            self.markdown_content.append("\n")
        except Exception as e:
            print(f"    ⚠️ Could not create top models table: {e}")

    def add_section_6_export_summary(self, pdf: PdfPages, export_dir: Optional[Path]):
        """Add Section 6: Export Summary.

        Args:
            pdf: PdfPages object
            export_dir: Directory containing exported files
        """
        print("  📦 Section 6: Export Summary")

        # Only add to markdown, skip PDF pages
        self.markdown_content.append("## Section 6 – Export Summary\n\n")

        summary_text = """
Exported Files:

1. Charts and Visualizations
   - All charts from analysis sections
   - Available in PNG format (charts/ directory)

2. Predictions and Results
   - Top 3 best predictions as CSV files
   - Top models performance metrics
   - Ranking tables

3. Best Pipeline Export
   - Complete pipeline with trained models
   - All artifacts (scalers, transformers, etc.)
   - Pipeline configuration and metadata

4. Report Documents
   - This PDF report with methodology
   - Markdown version for easy editing

All files are packaged in a ZIP archive for easy sharing.
"""

        self.markdown_content.append(summary_text + "\n\n")

        # List exported files in markdown only
        if export_dir and export_dir.exists():
            self.markdown_content.append("### Exported Files\n\n")
            for file_path in sorted(export_dir.rglob("*")):
                if file_path.is_file():
                    rel_path = file_path.relative_to(export_dir)
                    self.markdown_content.append(f"- `{rel_path}`\n")
            self.markdown_content.append("\n")

    def add_annex_methodology(self, pdf: PdfPages):
        """Add Annex: Methodology Documentation integrally to PDF.

        Tries to use pandoc for high-quality rendering of LaTeX equations
        and tables. Falls back to matplotlib-based rendering if pandoc
        is not available.

        Args:
            pdf: PdfPages object
        """
        print("  📄 Annex: Methodology")

        # Add to markdown content regardless of rendering method
        self.markdown_content.append("## Annex – Methodology\n\n")

        if not METHOD_DOC_PATH.exists():
            print(f"    ⚠️ Method documentation not found: {METHOD_DOC_PATH}")
            return

        # Read the methodology content
        try:
            with open(METHOD_DOC_PATH, 'r', encoding='utf-8') as f:
                method_content = f.read()
            self.markdown_content.append(method_content + "\n\n")
        except Exception as e:
            print(f"    ⚠️ Could not read methodology file: {e}")
            return

        # Try pandoc rendering first (best quality for equations and tables)
        if is_pandoc_available():
            print("    🔧 Using pandoc for high-quality rendering...")
            success = self._render_methodology_with_pandoc(pdf, method_content)
            if success:
                return
            print("    ⚠️ Pandoc rendering failed, falling back to matplotlib...")

        # Fallback: matplotlib-based rendering (strips equations)
        print("    🔧 Using matplotlib rendering (equations simplified)...")
        self._render_methodology_with_matplotlib(pdf, method_content)

    def _render_methodology_with_pandoc(self, pdf: PdfPages, method_content: str) -> bool:
        """Render methodology using pandoc and merge into the PDF.

        Args:
            pdf: PdfPages object
            method_content: Markdown content to render

        Returns:
            True if successful, False otherwise
        """
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            try:
                from pypdf import PdfReader
            except ImportError:
                print("    ⚠️ PyPDF2/pypdf not available for PDF merging")
                return False

        # Create temporary files for pandoc conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_md = Path(temp_dir) / "methodology.md"
            temp_pdf = Path(temp_dir) / "methodology.pdf"

            # Add title to the markdown content
            full_content = "# Annex: Methodology\n\n" + method_content

            # Write markdown to temp file
            with open(temp_md, 'w', encoding='utf-8') as f:
                f.write(full_content)

            # Convert to PDF using pandoc
            if not render_markdown_to_pdf_pandoc(temp_md, temp_pdf):
                return False

            if not temp_pdf.exists():
                print("    ⚠️ Pandoc did not produce output PDF")
                return False

            # Read the generated PDF and add each page to the main PDF
            try:
                reader = PdfReader(str(temp_pdf))
                num_pages = len(reader.pages)
                print(f"    📄 Adding {num_pages} methodology pages from pandoc...")

                # For each page in the pandoc PDF, create a matplotlib figure
                # that embeds the PDF page as an image
                import subprocess

                # Convert PDF pages to images using pdftoppm if available
                try:
                    # Try using pdf2image if available
                    from pdf2image import convert_from_path
                    images = convert_from_path(str(temp_pdf), dpi=150)

                    for i, img in enumerate(images):
                        fig, ax = plt.subplots(figsize=(8.27, 11.69))
                        ax.axis('off')
                        ax.imshow(img)
                        ax.set_xlim(0, img.width)
                        ax.set_ylim(img.height, 0)
                        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)

                    print(f"    ✅ Added {len(images)} pages via pdf2image")
                    return True

                except ImportError:
                    # pdf2image not available, try alternative approach
                    print("    ℹ️ pdf2image not available, using alternative method...")

                    # Save pandoc PDF separately and note it in the report
                    annex_pdf_path = self.output_dir / f"{self.filename}_methodology_annex.pdf"
                    shutil.copy(temp_pdf, annex_pdf_path)
                    print(f"    📄 Saved methodology annex separately: {annex_pdf_path}")

                    # Add a reference page to the main PDF
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.axis('off')
                    ax.text(0.5, 0.6, 'Annex: Methodology', ha='center', va='center',
                            fontsize=18, fontweight='bold', transform=ax.transAxes,
                            family='sans-serif', color='#2c3e50')
                    ax.text(0.5, 0.45, f'See separate file: {annex_pdf_path.name}', ha='center', va='center',
                            fontsize=12, transform=ax.transAxes,
                            family='sans-serif', color='#7f8c8d')
                    ax.text(0.5, 0.35, '(Generated with pandoc for full LaTeX equation support)', ha='center', va='center',
                            fontsize=10, transform=ax.transAxes,
                            family='sans-serif', color='#95a5a6', style='italic')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    return True

            except Exception as e:
                print(f"    ⚠️ Error processing pandoc PDF: {e}")
                return False

        return False

    def _render_methodology_with_matplotlib(self, pdf: PdfPages, method_content: str):
        """Render methodology using matplotlib (fallback method).

        This method strips LaTeX equations as matplotlib cannot render them properly.

        Args:
            pdf: PdfPages object
            method_content: Markdown content to render
        """
        # Title page
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis('off')
        ax.text(0.5, 0.5, 'Annex: Methodology', ha='center', va='center',
                fontsize=18, fontweight='bold', transform=ax.transAxes,
                family='sans-serif', color='#2c3e50')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Strip or simplify LaTeX equations to avoid rendering issues
        # Replace LaTeX equations with plain text descriptions
        method_content_pdf = re.sub(r'\$\$[^\$]+\$\$', '[Mathematical formula - see markdown version]', method_content)
        method_content_pdf = re.sub(r'\$[^\$]+\$', '[formula]', method_content_pdf)

        # Use improved markdown rendering with cleaned content
        self._render_markdown_to_pdf(pdf, method_content_pdf)

    def _render_markdown_to_pdf(self, pdf: PdfPages, markdown_text: str):
        """Render markdown content as formatted PDF pages with proper pagination.

        Args:
            pdf: PdfPages object
            markdown_text: Markdown content to render
        """
        import re
        from textwrap import wrap

        lines = markdown_text.split('\n')
        current_fig = None
        current_ax = None
        y_position = 0.92
        line_spacing = 0.020  # Reduced line spacing
        left_margin = 0.08
        right_margin = 0.92
        text_width = right_margin - left_margin
        page_num = 0

        def start_new_page():
            nonlocal current_fig, current_ax, y_position, page_num
            if current_fig is not None:
                pdf.savefig(current_fig, bbox_inches='tight')
                plt.close(current_fig)

            current_fig, current_ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
            current_ax.axis('off')
            current_ax.set_xlim(0, 1)
            current_ax.set_ylim(0, 1)
            y_position = 0.92
            page_num += 1
            return current_fig, current_ax

        fig, ax = start_new_page()

        def check_page_break(needed_space=0.05):
            """Check if we need a new page based on remaining space."""
            nonlocal fig, ax, y_position
            if y_position < (0.08 + needed_space):
                fig, ax = start_new_page()

        def is_safe_to_render(text):
            """Check if text is safe to render (not problematic ASCII art)."""
            # Skip lines with heavy ASCII art characters that might cause issues
            problematic_chars = ['│', '├', '└', '┌', '┐', '┘', '─', '┬', '┴', '┼', '║', '═', '╔', '╗', '╚', '╝']
            # Allow some ASCII art but skip if too many problematic chars
            problem_count = sum(1 for char in text if char in problematic_chars)
            return problem_count < 5  # Allow up to 4 problematic chars per line

        in_code_block = False
        code_lines = []
        skip_ascii_art = False

        for i, line in enumerate(lines):
            # Handle code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block - check if it's ASCII art
                    if code_lines and not all(is_safe_to_render(l) for l in code_lines[:5]):
                        # Likely ASCII art, skip it but add a note
                        check_page_break(0.04)
                        ax.text(left_margin, y_position, '[Workflow diagram - see markdown version]',
                               ha='left', va='top', fontsize=7,
                               family='sans-serif', transform=ax.transAxes,
                               color='#7f8c8d', style='italic')
                        y_position -= line_spacing * 1.5
                    else:
                        # Regular code block - render it
                        check_page_break(min(len(code_lines), 20) * line_spacing * 0.8 + 0.05)
                        for code_line in code_lines[:30]:  # Limit to 30 lines
                            try:
                                # Clean the line of problematic characters
                                clean_line = code_line.replace('\t', '    ')
                                ax.text(left_margin + 0.02, y_position, clean_line,
                                       ha='left', va='top', fontsize=6,
                                       family='monospace', transform=ax.transAxes,
                                       color='#2c3e50', backgroundcolor='#f8f9fa')
                                y_position -= line_spacing * 0.7
                            except Exception as e:
                                # Skip problematic lines silently
                                print(f"    ⚠️ Skipped line in code block: {str(e)[:50]}")
                                continue
                    code_lines = []
                    in_code_block = False
                    y_position -= line_spacing  # Extra space after code block
                else:
                    # Start of code block
                    in_code_block = True
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            # Skip empty lines but add spacing
            if not line.strip():
                y_position -= line_spacing * 0.4
                continue

            # Determine styling based on markdown syntax
            fontsize = 8
            fontweight = 'normal'
            fontstyle = 'normal'
            color = '#2c3e50'
            x_pos = left_margin
            extra_space_before = 0
            extra_space_after = 0

            original_line = line

            # Headings
            if line.startswith('# '):
                fontsize = 12
                fontweight = 'bold'
                line = line[2:].strip()
                extra_space_before = line_spacing * 1.2
                extra_space_after = line_spacing * 0.4
                color = '#2c3e50'
            elif line.startswith('## '):
                fontsize = 10
                fontweight = 'bold'
                line = line[3:].strip()
                extra_space_before = line_spacing * 0.8
                extra_space_after = line_spacing * 0.25
                color = '#34495e'
            elif line.startswith('### '):
                fontsize = 9
                fontweight = 'bold'
                line = line[4:].strip()
                extra_space_before = line_spacing * 0.5
                extra_space_after = line_spacing * 0.15
                color = '#7f8c8d'
            elif line.startswith('#### '):
                fontsize = 8
                fontweight = 'bold'
                line = line[5:].strip()
                extra_space_before = line_spacing * 0.4
            # Lists
            elif re.match(r'^[\s]*[-*+]\s', line):
                bullet_indent = len(line) - len(line.lstrip())
                x_pos = left_margin + (bullet_indent / 100.0) * 0.5
                line = '  •  ' + line.lstrip('- *+').strip()
                fontsize = 8
            elif re.match(r'^[\s]*\d+\.\s', line):
                num_indent = len(line) - len(line.lstrip())
                x_pos = left_margin + (num_indent / 100.0) * 0.5
                match = re.match(r'^[\s]*(\d+)\.\s(.*)$', line)
                if match:
                    num, text = match.groups()
                    line = f'  {num}.  {text}'
                fontsize = 8

            # Handle inline formatting (bold, italic, code)
            # Simple handling - remove markers for PDF rendering
            line = re.sub(r'\*\*([^\*]+)\*\*', r'\1', line)  # Bold
            line = re.sub(r'\*([^\*]+)\*', r'\1', line)  # Italic
            line = re.sub(r'`([^`]+)`', r'\1', line)  # Inline code

            # Clean any remaining problematic characters
            line = line.replace('\t', '    ')

            # Skip lines that are still problematic
            if not is_safe_to_render(line):
                continue

            # Handle equations (LaTeX) - mark them for special rendering
            has_equation = '$' in line
            if has_equation:
                # For now, keep equations as-is and use a monospace font
                fontsize = 7
                family = 'monospace'
            else:
                family = 'sans-serif'

            # Apply extra spacing before
            if extra_space_before > 0:
                y_position -= extra_space_before
                check_page_break()

            # Wrap long lines
            if len(line) > 90:
                wrapped_lines = wrap(line, width=int(100 * (text_width / 0.84)))
            else:
                wrapped_lines = [line]

            # Check if we need a new page for all wrapped lines
            needed_space = len(wrapped_lines) * line_spacing + extra_space_after
            check_page_break(needed_space)

            # Render each wrapped line
            for wrapped_line in wrapped_lines:
                try:
                    ax.text(x_pos, y_position, wrapped_line,
                           ha='left', va='top', fontsize=fontsize,
                           fontweight=fontweight, fontstyle=fontstyle,
                           color=color, transform=ax.transAxes,
                           family=family)
                    y_position -= line_spacing
                except Exception as e:
                    # Skip problematic lines
                    print(f"    ⚠️ Skipped rendering line: {str(e)[:50]}")
                    y_position -= line_spacing
                    continue

            # Apply extra spacing after
            if extra_space_after > 0:
                y_position -= extra_space_after

        # Save final page
        if fig is not None:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        print(f"    📄 Rendered {page_num} methodology pages")

    def _save_chart(self, fig: Figure, name: str):
        """Save chart to charts directory.

        Args:
            fig: Matplotlib figure
            name: Chart name (without extension)
        """
        path = self.charts_dir / f"{name}.png"
        fig.savefig(str(path), dpi=150, bbox_inches='tight')

    def save_markdown(self):
        """Save markdown version of report."""
        md_path = self.output_dir / f"{self.filename}_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.writelines(self.markdown_content)
        print(f"  📄 Saved markdown: {md_path}")

    def create_zip_export(self, predictions_obj: Predictions, export_path: Path,
                         rank_metric: str, workspace_path: Path, aggregation_key: Optional[str] = None):
        """Create ZIP archive with all export files.

        Args:
            predictions_obj: Predictions object
            export_path: Output path for ZIP file
            rank_metric: Metric for ranking
            workspace_path: Path to workspace containing run directories
            aggregation_key: Column name to aggregate by (None for raw)
        """
        print(f"  📦 Creating ZIP export: {export_path}")

        # Create temp directory for predictions and pipeline exports
        temp_export_dir = self.output_dir / f"{self.filename}_temp"
        temp_export_dir.mkdir(exist_ok=True)
        predictions_dir = temp_export_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        pipeline_dir = temp_export_dir / "pipeline"
        pipeline_dir.mkdir(exist_ok=True)

        try:
            # Export top 3 predictions as CSV
            agg_text = f" (aggregated by {aggregation_key})" if aggregation_key else " (raw)"
            print(f"    💾 Exporting top 3 predictions as CSV{agg_text}...")
            top_3_models = predictions_obj.top(n=3, rank_metric=rank_metric,
                                               rank_partition='val', by_repetition=aggregation_key)
            for i, pred in enumerate(top_3_models, 1):
                model_name = pred.get('model_name', 'Unknown')
                score = pred.get('rank_score', 'N/A')
                y_true = pred.get('y_true')
                y_pred = pred.get('y_pred')

                if y_true is not None and y_pred is not None:
                    csv_path = predictions_dir / f"top_{i}_{model_name}_{rank_metric}_{score:.4f}.csv"
                    Predictions.save_predictions_to_csv(y_true=y_true, y_pred=y_pred,
                                                       filepath=str(csv_path))
                    print(f"      ✅ Exported: {csv_path.name}")

            # Export best pipeline with artifacts
            print("    📦 Exporting best pipeline with artifacts...")
            best_pred = top_3_models[0]
            config_name = best_pred.get('config_name', '')

            # Find the run directory containing this pipeline
            runs_dir = workspace_path.parent / "runs"
            if runs_dir.exists():
                for run_dir in runs_dir.iterdir():
                    if run_dir.is_dir():
                        for pipe_dir in run_dir.iterdir():
                            if pipe_dir.is_dir() and config_name in pipe_dir.name and not pipe_dir.name.startswith('_'):
                                # Found the pipeline directory
                                print(f"      📂 Found pipeline: {pipe_dir.name}")

                                # Copy full pipeline directory to export
                                exported_path = pipeline_dir / "best_model"
                                try:
                                    shutil.copytree(pipe_dir, exported_path, dirs_exist_ok=True)
                                    # Also copy binaries from run_dir if present
                                    binaries_src = run_dir / "_binaries"
                                    if binaries_src.exists():
                                        shutil.copytree(binaries_src, exported_path / "_binaries", dirs_exist_ok=True)
                                    print(f"      Exported pipeline to: {exported_path.name}")
                                except Exception as e:
                                    print(f"      Could not export pipeline: {e}")
                                break
                        else:
                            continue
                        break

            # Create ZIP
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add charts
                if self.charts_dir.exists():
                    for chart_file in self.charts_dir.glob("*.png"):
                        arcname = f"charts/{chart_file.name}"
                        zipf.write(chart_file, arcname)

                # Add reports
                pdf_path = self.output_dir / f"{self.filename}_report.pdf"
                if pdf_path.exists():
                    zipf.write(pdf_path, f"{self.filename}_report.pdf")

                md_path = self.output_dir / f"{self.filename}_report.md"
                if md_path.exists():
                    zipf.write(md_path, f"{self.filename}_report.md")

                # Add predictions CSVs
                for csv_file in predictions_dir.glob("*.csv"):
                    arcname = f"predictions/{csv_file.name}"
                    zipf.write(csv_file, arcname)

                # Add pipeline exports
                if pipeline_dir.exists():
                    for item in pipeline_dir.rglob("*"):
                        if item.is_file():
                            arcname = f"pipeline/{item.relative_to(pipeline_dir)}"
                            zipf.write(item, arcname)

            print(f"  ✅ ZIP export created: {export_path}")

        finally:
            # Clean up temp directory
            if temp_export_dir.exists():
                shutil.rmtree(temp_export_dir)


# ========================================
# MAIN REPORT GENERATION FUNCTION
# ========================================

def generate_report(workspace_path: Path, filename: str, output_dir: Path,
                   aggregation_key: Optional[str] = None, mode: str = 'aggregated',
                   dataset_folder: Optional[Path] = None,
                   exclude_models: Optional[List[str]] = None,
                   rename_map: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """Generate comprehensive report for a dataset.

    Args:
        workspace_path: Path to workspace directory
        filename: Dataset filename (without extension)
        output_dir: Output directory for reports
        aggregation_key: Column name to aggregate by (e.g., 'ID', 'sample_id'). None for raw predictions.
        mode: Analysis mode - 'raw', 'aggregated', or 'both'
        dataset_folder: Path to dataset folder (optional) for loading actual data and creating visualizations
        exclude_models: List of model names to exclude (case-insensitive partial match). Defaults to EXCLUDE_MODELS.
        rename_map: Dict mapping old pattern to new names. Defaults to MODEL_RENAME_MAP.

    Returns:
        Path to generated PDF report
    """
    print(f"\n{'='*70}")
    print(f"GENERATING REPORT: {filename}")
    print("=" * 70)

    # Load predictions
    predictions_path = workspace_path / f"{filename}.meta.parquet"
    if not predictions_path.exists():
        print(f"  ❌ Predictions file not found: {predictions_path}")
        return None

    print(f"  📂 Loading predictions from {predictions_path}")
    predictions = Predictions.load(path=str(predictions_path))
    print(f"  ➡️ Loaded {len(predictions)} predictions")

    if len(predictions) == 0:
        print(f"  ⚠️ No predictions found, skipping")
        return None

    # Apply filters (use defaults if not provided)
    effective_exclude = exclude_models if exclude_models is not None else EXCLUDE_MODELS
    effective_rename = rename_map if rename_map is not None else MODEL_RENAME_MAP
    predictions = apply_model_filters(predictions, effective_exclude, effective_rename)
    print(f"  ➡️ Using {len(predictions)} predictions after filtering")

    # Detect task type
    is_classification, rank_metric, display_metrics, chart_metrics = detect_task_type(predictions)
    print(f"  📋 Task type: {'Classification' if is_classification else 'Regression'}")
    print(f"  📏 Ranking metric: {rank_metric}")

    # Create analyzer
    analyzer = PredictionAnalyzer(predictions, output_dir=None)

    # Initialize report generator
    report = ReportGenerator(output_dir, filename)

    # Generate PDF report
    pdf_path = output_dir / f"{filename}_report.pdf"
    print(f"\n  📄 Generating PDF report: {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        # Cover page
        report.add_cover_page(pdf)

        # Section 1: Dataset Description
        report.add_section_1_dataset_description(pdf, predictions, is_classification, dataset_folder)

        # Section 2: Experimental Protocol
        report.add_section_2_protocol(pdf)

        # Section 3: Global Results
        report.add_section_3_global_results(pdf, analyzer, rank_metric, is_classification, aggregation_key, mode)

        # Section 4: Ranking Analysis
        report.add_section_4_ranking_analysis(pdf, analyzer, rank_metric, aggregation_key, mode)

        # Section 5: Individual Model Diagnostics
        report.add_section_5_model_diagnostics(pdf, analyzer, predictions, rank_metric, is_classification, aggregation_key, mode)

        # Section 6: Export Summary
        export_dir = output_dir / filename
        report.add_section_6_export_summary(pdf, export_dir)

        # Annex: Methodology
        report.add_annex_methodology(pdf)

        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = f'NIRS Analysis Report - {filename}'
        d['Author'] = ', '.join(AUTHORS)
        d['Subject'] = 'NIRS Spectroscopy Analysis'
        d['Keywords'] = 'NIRS, Machine Learning, Spectroscopy'
        d['CreationDate'] = datetime.now()

    print(f"  ✅ PDF report generated: {pdf_path}")

    # Save markdown version
    report.save_markdown()

    # Create ZIP export
    zip_path = output_dir / f"{filename}_export.zip"
    report.create_zip_export(predictions, zip_path, rank_metric, workspace_path, aggregation_key)

    # Cleanup temp directory for charts
    import shutil
    if Path(report.temp_dir).exists():
        shutil.rmtree(report.temp_dir, ignore_errors=True)
        print(f"  🧹 Cleaned up temporary files")

    print(f"\n✅ Report generation complete for {filename}")
    return pdf_path


# ========================================
# MAIN FUNCTION
# ========================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive analysis reports from prediction parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reports for all parquet files in workspace
  python study_report.py --workspace wk --output reports/

  # Generate reports for specific datasets
  python study_report.py --workspace wk --filenames digestibility_custom2 hardness_0_8

  # Include raw predictions analysis
  python study_report.py --workspace wk --include-raw

  # Specify output directory
  python study_report.py --workspace wk --output my_reports/

  # Include dataset visualizations (spectra and y distribution charts)
  python study_report.py --workspace wk --dataset-folder _datasets/digestibility_custom2
        """
    )
    parser.add_argument('--workspace', type=str, default='wk',
                       help='Workspace directory containing parquet files (default: wk)')
    parser.add_argument('--output', type=str, default='reports',
                       help='Output directory for reports (default: reports)')
    parser.add_argument('--filenames', nargs='+',
                       help='Specific dataset filenames to process (without extension)')
    parser.add_argument('--aggregation-key', type=str, default=None,
                       help='Aggregation key for predictions (e.g., "ID", "sample_id"). If not provided, uses raw predictions.')
    parser.add_argument('--mode', type=str, choices=['raw', 'aggregated', 'both'], default='aggregated',
                       help='Analysis mode: "raw" (no aggregation), "aggregated" (with aggregation), or "both" (default: aggregated)')
    parser.add_argument('--dataset-folder', type=str, default=None,
                       help='Path to dataset folder for loading actual data and creating visualizations')
    parser.add_argument('--exclude-models', nargs='*', default=None,
                       help='Models to exclude from report (case-insensitive partial match). Default: KernelPLS')
    parser.add_argument('--rename-models', nargs='*', default=None,
                       help='Model renaming as key=value pairs (e.g., "dict=nicon tabpfn=transformer")')

    args = parser.parse_args()

    # Setup paths
    workspace_path = Path(args.workspace)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_folder = Path(args.dataset_folder) if args.dataset_folder else None

    # Parse model filtering options
    exclude_models = args.exclude_models if args.exclude_models is not None else EXCLUDE_MODELS
    rename_map = {}
    if args.rename_models:
        for pair in args.rename_models:
            if '=' in pair:
                key, value = pair.split('=', 1)
                rename_map[key] = value
    else:
        rename_map = MODEL_RENAME_MAP

    print("=" * 70)
    print("NIRS ANALYSIS REPORT GENERATOR")
    print("=" * 70)
    print(f"Workspace: {workspace_path}")
    print(f"Output: {output_dir}")
    print(f"Mode: {args.mode}")
    if args.aggregation_key:
        print(f"Aggregation key: {args.aggregation_key}")
    else:
        print("Aggregation: None (raw predictions)")
    if dataset_folder:
        print(f"Dataset folder: {dataset_folder}")
    if exclude_models:
        print(f"Excluding models: {exclude_models}")
    if rename_map:
        print(f"Renaming models: {rename_map}")
    print()

    # Determine which files to process
    if args.filenames:
        filenames = args.filenames
        print(f"Processing {len(filenames)} specified datasets")
    else:
        # Find all parquet files in workspace
        parquet_files = list(workspace_path.glob("*.meta.parquet"))
        filenames = [f.stem.replace('.meta', '') for f in parquet_files]
        print(f"Found {len(filenames)} datasets in workspace")

    if not filenames:
        print("❌ No datasets found to process")
        return

    # Generate reports
    success_count = 0
    for filename in filenames:
        try:
            pdf_path = generate_report(
                workspace_path=workspace_path,
                filename=filename,
                output_dir=output_dir,
                aggregation_key=args.aggregation_key,
                mode=args.mode,
                dataset_folder=dataset_folder,
                exclude_models=exclude_models,
                rename_map=rename_map,
            )
            if pdf_path:
                success_count += 1
        except Exception as e:
            print(f"\n❌ Error generating report for {filename}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 70)
    print(f"REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"Successfully generated {success_count}/{len(filenames)} reports")
    print(f"Reports saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
