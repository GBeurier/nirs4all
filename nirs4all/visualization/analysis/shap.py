"""
SHAP Analyzer - Model explainability using SHAP values

This module provides SHAP-based explanations for NIRS models, including
spectral importance visualizations that highlight which wavelengths contribute
most to predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


class ShapAnalyzer:
    """
    SHAP-based model explainability analyzer for NIRS models.

    Provides explanations showing which wavelengths/features are most important
    for model predictions, with specialized visualizations for spectral data.
    """

    def __init__(self):
        """Initialize the SHAP analyzer."""
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Please install it with: pip install shap"
            )
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        self.data = None
        self.wavelengths = None
        self.feature_names = None

        # Binning parameters for aggregation
        self.bin_size = 20
        self.bin_stride = 10
        self.bin_aggregation = 'sum'  # 'sum', 'sum_abs', 'mean', 'mean_abs'

    def explain_model(
        self,
        model: Any,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        sample_indices: Optional[List[int]] = None,
        task_type: str = "regression",
        n_background: int = 100,
        explainer_type: str = "auto",
        output_dir: Optional[str] = None,
        visualizations: Optional[List[str]] = None,
        bin_size = 20,
        bin_stride = 10,
        bin_aggregation = 'sum',
        plots_visible = True
    ) -> Dict[str, Any]:
        """
        Explain model predictions using SHAP values.

        Args:
            model: Trained model to explain
            X: Input features (samples x features)
            y: Target values (optional, for reference)
            feature_names: Names of features/wavelengths
            sample_indices: Specific samples to explain (None = all)
            task_type: 'regression' or 'classification'
            n_background: Number of background samples for KernelExplainer
            explainer_type: 'auto', 'tree', 'deep', 'kernel', 'linear'
            output_dir: Directory to save visualizations
            visualizations: List of viz types to generate
            bin_size: Number of wavelengths per bin. Can be:
                     - int: same for all visualizations
                     - dict: {'spectral': 20, 'waterfall': 30, 'beeswarm': 50}
            bin_stride: Step size between bins. Can be:
                       - int: same for all visualizations
                       - dict: {'spectral': 10, 'waterfall': 15, 'beeswarm': 25}
            bin_aggregation: Aggregation method. Can be:
                            - str: same for all ('sum', 'sum_abs', 'mean', 'mean_abs')
                            - dict: {'spectral': 'sum', 'waterfall': 'mean', 'beeswarm': 'sum_abs'}

        Returns:
            Dictionary with SHAP results
        """
        logger.info("=" * 80)
        logger.info("SHAP Analysis Starting")
        logger.info("=" * 80)

        # Select samples if specified
        if sample_indices is not None:
            X_explain = X[sample_indices]
        else:
            X_explain = X

        logger.info(f"Analyzing {X_explain.shape[0]} samples with {X_explain.shape[1]} features")

        # Step 1: Select and create explainer
        logger.info(f"Creating SHAP explainer (type: {explainer_type})...")
        self.explainer = self._create_explainer(
            model, X, explainer_type, n_background, task_type
        )

        # Step 2: Compute SHAP values
        logger.info("Computing SHAP values...")

        # No need to reshape - the predict function wrapper handles it
        self.shap_values = self.explainer.shap_values(X_explain)

        # Handle multi-output case (e.g., multi-class classification)
        if isinstance(self.shap_values, list):
            logger.info(f"   Multi-output detected: {len(self.shap_values)} outputs")
            # For now, use first output
            self.shap_values = self.shap_values[0]

        # Flatten if needed (Keras may return 3D)
        if len(self.shap_values.shape) == 3 and self.shap_values.shape[2] == 1:
            self.shap_values = self.shap_values.squeeze(axis=2)

        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            self.base_value = self.explainer.expected_value
            if isinstance(self.base_value, np.ndarray):
                self.base_value = self.base_value[0]
        else:
            self.base_value = np.mean(X_explain)

        self.data = X_explain
        self.feature_names = feature_names

        # Extract wavelengths from feature names if they follow the λXXX.X format
        if feature_names and isinstance(feature_names, list) and len(feature_names) > 0:
            try:
                # Try to extract wavelengths from λXXX.X format
                if feature_names[0].startswith('λ'):
                    self.wavelengths = np.array([float(name[1:]) for name in feature_names])
            except (ValueError, IndexError):
                self.wavelengths = None

        # Normalize binning parameters to dict format
        # Allows single value or per-visualization configuration
        def normalize_param(param, default):
            if isinstance(param, dict):
                return param
            else:
                # Single value - use for all visualizations
                return {'spectral': param, 'waterfall': param, 'beeswarm': param}

        self.bin_size_dict = normalize_param(bin_size, 20)
        self.bin_stride_dict = normalize_param(bin_stride, 10)
        self.bin_aggregation_dict = normalize_param(bin_aggregation, 'sum')

        logger.success(f"SHAP values computed: shape={self.shap_values.shape}")
        logger.info(f"   Base value: {self.base_value:.4f}")
        logger.info("   Binning config:")
        for viz in ['spectral', 'waterfall', 'beeswarm']:
            if viz in self.bin_size_dict:
                logger.info(f"     {viz}: size={self.bin_size_dict[viz]}, stride={self.bin_stride_dict[viz]}, agg={self.bin_aggregation_dict[viz]}")

        # Step 3: Generate visualizations
        if output_dir and visualizations:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info("Generating visualizations...")

            if 'spectral' in visualizations:
                # Set binning for spectral
                self.bin_size = self.bin_size_dict.get('spectral', 20)
                self.bin_stride = self.bin_stride_dict.get('spectral', 10)
                self.bin_aggregation = self.bin_aggregation_dict.get('spectral', 'sum')

                self.plot_spectral_importance(
                    feature_names=feature_names,
                    output_path=str(output_path / "spectral_importance.png"),
                    plots_visible=plots_visible
                )
                logger.success("   Spectral importance")

            if 'summary' in visualizations:
                self.plot_summary(
                    feature_names=feature_names,
                    output_path=str(output_path / "summary.png"),
                    plots_visible=plots_visible
                )
                logger.success("   Summary plot")

            if 'waterfall' in visualizations:
                # Set binning for waterfall
                self.bin_size = self.bin_size_dict.get('waterfall', 20)
                self.bin_stride = self.bin_stride_dict.get('waterfall', 10)
                self.bin_aggregation = self.bin_aggregation_dict.get('waterfall', 'sum')

                self.plot_waterfall_binned(
                    sample_idx=0,
                    output_path=str(output_path / "waterfall_binned.png"),
                    plots_visible=plots_visible
                )
                logger.success("   Waterfall plot (binned)")

            if 'force' in visualizations:
                self.plot_force(
                    sample_idx=0,
                    feature_names=feature_names,
                    output_path=str(output_path / "force.html"),
                    plots_visible=plots_visible
                )
                logger.success("   Force plot")

            if 'beeswarm' in visualizations:
                # Set binning for beeswarm
                self.bin_size = self.bin_size_dict.get('beeswarm', 20)
                self.bin_stride = self.bin_stride_dict.get('beeswarm', 10)
                self.bin_aggregation = self.bin_aggregation_dict.get('beeswarm', 'sum')

                self.plot_beeswarm_binned(
                    output_path=str(output_path / "beeswarm_binned.png"),
                    plots_visible=plots_visible
                )
                logger.success("   Beeswarm plot (binned)")

        # Prepare results dictionary
        results = {
            'shap_values': self.shap_values,
            'base_value': self.base_value,
            'data': self.data,
            'feature_names': feature_names,
            'explainer_type': type(self.explainer).__name__,
            'n_samples': self.shap_values.shape[0],
            'n_features': self.shap_values.shape[1]
        }

        logger.success("SHAP analysis completed!")
        logger.info("=" * 80)

        return results

    def _create_explainer(
        self,
        model: Any,
        X: np.ndarray,
        explainer_type: str,
        n_background: int,
        task_type: str
    ):
        """Create appropriate SHAP explainer based on model type."""

        if explainer_type == "auto":
            # Auto-detect best explainer
            model_type = type(model).__name__

            # Tree-based models
            if any(tree_name in model_type for tree_name in
                   ['Tree', 'Forest', 'Gradient', 'XGB', 'LightGBM', 'CatBoost']):
                explainer_type = "tree"
            # Linear models
            elif any(linear_name in model_type for linear_name in
                     ['Linear', 'Ridge', 'Lasso', 'Elastic', 'PLS']):
                explainer_type = "linear"
            # Deep learning
            elif any(dl_name in model_type for dl_name in
                     ['Sequential', 'Functional', 'Model', 'Network']):
                explainer_type = "deep"
            else:
                explainer_type = "kernel"

            logger.info(f"   Auto-selected explainer: {explainer_type}")

        # Create explainer
        if explainer_type == "tree":
            try:
                return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            except Exception as e:
                error_msg = str(e).split('\n')[0]
                logger.warning(f"   TreeExplainer failed: {error_msg}, falling back to Kernel")
                explainer_type = "kernel"

        if explainer_type == "linear":
            try:
                return shap.LinearExplainer(model, X)
            except Exception as e:
                error_msg = str(e).split('\n')[0]
                logger.warning(f"   LinearExplainer failed: {error_msg}, falling back to Kernel")
                explainer_type = "kernel"

        if explainer_type == "deep":
            try:
                # Suppress verbose TensorFlow warnings during explainer creation
                import warnings
                import logging

                # Temporarily suppress TensorFlow logging
                tf_logger = logging.getLogger('tensorflow')
                old_level = tf_logger.level
                tf_logger.setLevel(logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=Warning)
                    explainer = shap.DeepExplainer(model, X[:n_background])

                # Restore logging
                tf_logger.setLevel(old_level)
                return explainer

            except Exception as e:
                # Clean error message without stack trace
                error_msg = str(e).split('\n')[0]  # First line only
                logger.warning(f"   DeepExplainer failed: {error_msg}, falling back to Kernel")
                explainer_type = "kernel"

        # Fallback to KernelExplainer (works with any model)
        if explainer_type == "kernel":
            # Sample background data
            if X.shape[0] > n_background:
                background_indices = np.random.choice(
                    X.shape[0], n_background, replace=False
                )
                background = X[background_indices]
            else:
                background = X

            # Create prediction function with proper wrapping for TensorFlow/Keras
            model_type = type(model).__name__
            is_keras = any(name in model_type for name in ['Sequential', 'Functional', 'Model'])

            if is_keras:
                # Keras/TensorFlow models with preprocessing are not directly compatible
                raise ValueError(
                    "\n" + "="*80 +
                    f"\n{CROSS}SHAP Error: Keras/TensorFlow models are not directly supported.\n" +
                    "\n" +
                    "The issue: Your Keras model was trained on preprocessed data, but SHAP\n" +
                    "needs to explain the raw features. The model expects a different input\n" +
                    "shape than the raw data provides.\n" +
                    "\n" +
                    "Solutions:\n" +
                    "1. Use a tree-based model instead (e.g., GradientBoostingRegressor,\n" +
                    "   RandomForest, XGBoost) - these work perfectly with SHAP\n" +
                    "2. Use PLSRegression or other sklearn models\n" +
                    "3. For Keras models, you'd need to include preprocessing inside the model\n" +
                    "   (not currently supported by the pipeline)\n" +
                    "\n" +
                    "Example with GradientBoost:\n" +
                    "  {\"model\": GradientBoostingRegressor(n_estimators=100), \"name\": \"GradBoost\"}\n" +
                    "\n" +
                    "=" *80
                )

            elif hasattr(model, 'predict_proba') and task_type == 'classification':
                predict_fn = model.predict_proba
            else:
                predict_fn = model.predict

            return shap.KernelExplainer(predict_fn, background)

        raise ValueError(f"Unknown explainer type: {explainer_type}")

    def plot_spectral_importance(
        self,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        plots_visible: bool = True
    ):
        """
        Create NIRS-specific spectral importance visualization with binned regions.

        Shows important spectral regions (not individual wavelengths) by binning
        wavelengths and aggregating SHAP values. This is more robust and meaningful
        for NIRS analysis than point-by-point importance.

        Uses self.bin_size, self.bin_stride, and self.bin_aggregation configured
        in explain_model().
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        # Calculate mean absolute SHAP values per feature
        mean_shap = np.abs(self.shap_values).mean(axis=0)

        # Create wavelength axis
        n_features = len(mean_shap)
        if feature_names and len(feature_names) == n_features:
            try:
                wavelengths = np.array([
                    float(name.replace('λ', '').replace('cm-1', '').strip())
                    for name in feature_names
                ])
            except Exception:
                wavelengths = np.arange(n_features)
        else:
            wavelengths = np.arange(n_features)

        # Sort by wavelength to ensure proper ordering
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        mean_shap = mean_shap[sort_idx]
        mean_spectrum = self.data.mean(axis=0)[sort_idx]

        # Create bins using configured parameters
        bin_starts = range(0, n_features - self.bin_size + 1, self.bin_stride)

        bin_centers = []
        bin_shap_sums = []
        bin_ranges = []

        for start in bin_starts:
            end = start + self.bin_size
            bin_wavelengths = wavelengths[start:end]
            bin_shap = mean_shap[start:end]

            bin_centers.append(bin_wavelengths.mean())

            # Use configured aggregation method
            if self.bin_aggregation == 'sum':
                bin_shap_sums.append(bin_shap.sum())
            elif self.bin_aggregation == 'sum_abs':
                bin_shap_sums.append(np.abs(bin_shap).sum())
            elif self.bin_aggregation == 'mean':
                bin_shap_sums.append(bin_shap.mean())
            elif self.bin_aggregation == 'mean_abs':
                bin_shap_sums.append(np.abs(bin_shap).mean())

            bin_ranges.append((bin_wavelengths.min(), bin_wavelengths.max()))

        bin_centers = np.array(bin_centers)
        bin_shap_sums = np.array(bin_shap_sums)

        # Create figure with 2 panels (spectrum + bar chart)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.5], hspace=0.3)

        # Plot 1: Mean spectrum with important regions highlighted
        ax1 = fig.add_subplot(gs[0])

        # Plot mean spectrum
        ax1.plot(wavelengths, mean_spectrum, 'k-', linewidth=2, alpha=0.8, label='Mean Spectrum')

        # Highlight important regions with colored bands (Viridis colormap)
        # The WIDTH of each band = wavelength range of the bin (changes with bin_size)
        # The COLOR intensity = importance (higher SHAP = darker purple)
        shap_norm = bin_shap_sums / bin_shap_sums.max()

        # Only show regions with some importance to avoid clutter
        importance_threshold = 0.2
        for i, (start, end) in enumerate(bin_ranges):
            if shap_norm[i] > importance_threshold:
                color_intensity = shap_norm[i]
                # Alpha varies from 0.15 (low) to 0.7 (high) for better visibility
                alpha_val = 0.15 + 0.55 * color_intensity
                ax1.axvspan(start, end,
                           alpha=alpha_val,
                           color=plt.cm.viridis(0.2 + 0.8 * color_intensity),
                           zorder=0,
                           linewidth=0)

        # Add a simple legend explaining the bands
        from matplotlib.patches import Patch
        legend_elements = [
            plt.Line2D([0], [0], color='k', linewidth=2, label='Mean Spectrum'),
            Patch(facecolor=plt.cm.viridis(0.9), alpha=0.6, label='High Importance'),
            Patch(facecolor=plt.cm.viridis(0.5), alpha=0.4, label='Medium Importance'),
            Patch(facecolor=plt.cm.viridis(0.3), alpha=0.25, label='Low Importance')
        ]
        ax1.legend(handles=legend_elements, loc='best', framealpha=0.9, fontsize=9)

        ax1.set_xlabel('Wavelength (cm-1)' if feature_names else 'Feature Index',
                      fontsize=12, fontweight='bold')
        ax1.set_ylabel('Absorbance', fontsize=12, fontweight='bold')
        ax1.set_title(f'Mean Spectrum with Important Regions (Band Width = Bin Size: {self.bin_size} wavelengths)',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Plot 2: Binned SHAP importance across wavelength regions
        ax2 = fig.add_subplot(gs[1])

        # Create bar plot with bins (Viridis colormap)
        colors = plt.cm.viridis(0.3 + 0.7 * shap_norm)
        bar_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 10

        bars = ax2.bar(bin_centers, bin_shap_sums, width=bar_width * 0.9,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add line to show trend
        ax2.plot(bin_centers, bin_shap_sums, 'b-', linewidth=2, alpha=0.5, zorder=0)

        # Add colorbar legend
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', pad=0.02, aspect=30)
        cbar.set_label('Relative Importance', rotation=270, labelpad=20, fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)

        ax2.set_xlabel('Wavelength (cm-1)' if feature_names else 'Feature Index',
                      fontsize=12, fontweight='bold')
        ax2.set_ylabel('Aggregated SHAP Importance', fontsize=12, fontweight='bold')

        # Show binning configuration in title
        agg_method = self.bin_aggregation.replace('_', ' ').title()
        ax2.set_title(f'Binned SHAP Importance (Bin: {self.bin_size}, Stride: {self.bin_stride}, Agg: {agg_method})',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"   Saved: {output_path}")

        if not plots_visible:
            plt.close(fig)
        else:
            plt.show()  # Blocking

    def plot_summary(
        self,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        max_display: int = 20,
        plots_visible: bool = True
    ):
        """Create SHAP summary plot showing feature importance."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.data,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"   Saved: {output_path}")

        if not plots_visible:
            plt.close()
        else:
            plt.show()  # Blocking

    def plot_beeswarm(
        self,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        max_display: int = 20,
        plots_visible: bool = True
    ):
        """Create SHAP beeswarm plot."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(
            shap.Explanation(
                values=self.shap_values,
                base_values=self.base_value,
                data=self.data,
                feature_names=feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"   Saved: {output_path}")

        if not plots_visible:
            plt.close()
        else:
            plt.show()  # Blocking

    def plot_waterfall(
        self,
        sample_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        max_display: int = 20,
        plots_visible: bool = True
    ):
        """Create SHAP waterfall plot for a single sample."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.base_value,
                data=self.data[sample_idx],
                feature_names=feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"   Saved: {output_path}")

        if not plots_visible:
            plt.close()
        else:
            plt.show()  # Blocking

    def plot_force(
        self,
        sample_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        plots_visible: bool = True
    ):
        """Create SHAP force plot for a single sample."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        # Force plot returns HTML
        force_plot = shap.force_plot(
            self.base_value,
            self.shap_values[sample_idx],
            self.data[sample_idx],
            feature_names=feature_names,
            show=False
        )

        if output_path:
            shap.save_html(output_path, force_plot)
        else:
            shap.plots.force(
                self.base_value,
                self.shap_values[sample_idx],
                self.data[sample_idx],
                feature_names=feature_names
            )
        if output_path:
            logger.info(f"   Saved: {output_path}")
        if plots_visible and output_path is None:
            plt.show()  # Blocking
            plt.close()

    def plot_dependence(
        self,
        feature_idx: int,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        interaction_index: Optional[int] = None,
        plots_visible: bool = True
    ):
        """Create SHAP dependence plot for a specific feature."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            self.data,
            feature_names=feature_names,
            interaction_index=interaction_index,
            show=False
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')


        if not plots_visible:
            plt.close()
        else:
            plt.show()  # Blocking


    def _aggregate_shap_bins(self, shap_values: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Aggregate SHAP values into bins based on configured parameters.

        Args:
            shap_values: SHAP values array (can be 1D or 2D)

        Returns:
            Tuple of (binned_shap_values, bin_labels)
        """
        is_1d = (len(shap_values.shape) == 1)
        if is_1d:
            shap_values = shap_values.reshape(1, -1)

        n_samples, n_features = shap_values.shape

        # Create bins
        bins = []
        bin_labels = []

        start = 0
        while start < n_features:
            end = min(start + self.bin_size, n_features)
            bins.append((start, end))

            # Create label
            if self.wavelengths is not None and self.feature_names is not None:
                # Use wavelength ranges
                wl_start = self.wavelengths[start]
                wl_end = self.wavelengths[end - 1]
                bin_labels.append(f"{wl_start:.1f}-{wl_end:.1f} cm-1")
            else:
                bin_labels.append(f"Bin {start}-{end}")

            start += self.bin_stride

        # Aggregate SHAP values per bin
        binned_values = np.zeros((n_samples, len(bins)))

        for i, (start, end) in enumerate(bins):
            bin_shap = shap_values[:, start:end]

            if self.bin_aggregation == 'sum':
                binned_values[:, i] = bin_shap.sum(axis=1)
            elif self.bin_aggregation == 'sum_abs':
                binned_values[:, i] = np.abs(bin_shap).sum(axis=1)
            elif self.bin_aggregation == 'mean':
                binned_values[:, i] = bin_shap.mean(axis=1)
            elif self.bin_aggregation == 'mean_abs':
                binned_values[:, i] = np.abs(bin_shap).mean(axis=1)
            else:
                raise ValueError(f"Unknown aggregation method: {self.bin_aggregation}")

        if is_1d:
            binned_values = binned_values.flatten()

        return binned_values, bin_labels

    def plot_beeswarm_binned(
        self,
        output_path: Optional[str] = None,
        max_display: int = 20,
        plots_visible: bool = True
    ):
        """
        Create SHAP beeswarm plot with binned features.

        Bins wavelengths/features according to bin_size and bin_stride parameters,
        then displays beeswarm plot for aggregated SHAP values.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        # Aggregate SHAP values into bins
        binned_shap, bin_labels = self._aggregate_shap_bins(self.shap_values)

        # Create binned data (average features in each bin)
        n_samples, n_features = self.data.shape
        bins = []
        start = 0
        while start < n_features:
            end = min(start + self.bin_size, n_features)
            bins.append((start, end))
            start += self.bin_stride

        binned_data = np.zeros((n_samples, len(bins)))
        for i, (start, end) in enumerate(bins):
            binned_data[:, i] = self.data[:, start:end].mean(axis=1)

        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(
            shap.Explanation(
                values=binned_shap,
                base_values=self.base_value,
                data=binned_data,
                feature_names=bin_labels
            ),
            max_display=max_display,
            show=False
        )

        title = f'SHAP Beeswarm Plot (Binned)\n'
        title += f'Bin Size: {self.bin_size}, Stride: {self.bin_stride}, '
        title += f'Aggregation: {self.bin_aggregation}'
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"   Saved: {output_path}")

        if plots_visible:
            plt.show()  # Blocking
        else:
            plt.close()

    def plot_waterfall_binned(
        self,
        sample_idx: int = 0,
        output_path: Optional[str] = None,
        max_display: int = 20,
        plots_visible: bool = True
    ):
        """
        Create SHAP waterfall plot with binned features for a single sample.

        Bins wavelengths/features according to bin_size and bin_stride parameters,
        then displays waterfall plot for aggregated SHAP values.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        # Aggregate SHAP values into bins for the selected sample
        binned_shap, bin_labels = self._aggregate_shap_bins(self.shap_values[sample_idx])

        # Create binned data for the sample
        n_features = self.data.shape[1]
        bins = []
        start = 0
        while start < n_features:
            end = min(start + self.bin_size, n_features)
            bins.append((start, end))
            start += self.bin_stride

        binned_data = np.zeros(len(bins))
        for i, (start, end) in enumerate(bins):
            binned_data[i] = self.data[sample_idx, start:end].mean()

        plt.figure(figsize=(12, 10))
        shap.plots.waterfall(
            shap.Explanation(
                values=binned_shap,
                base_values=self.base_value,
                data=binned_data,
                feature_names=bin_labels
            ),
            max_display=max_display,
            show=False
        )

        title = f'SHAP Waterfall Plot - Sample {sample_idx} (Binned)\n'
        title += f'Bin Size: {self.bin_size}, Stride: {self.bin_stride}, '
        title += f'Aggregation: {self.bin_aggregation}'
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"   Saved: {output_path}")

        if not plots_visible:
            plt.close()
        else:
            plt.show()  # Blocking

    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get feature importance ranking based on mean absolute SHAP values.

        Args:
            top_n: Return only top N features (None = all)

        Returns:
            Dictionary mapping feature index to importance score
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model first.")

        mean_shap = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_shap)[::-1]

        if top_n:
            indices = indices[:top_n]

        return {int(idx): float(mean_shap[idx]) for idx in indices}

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save SHAP results to disk using the new serializer."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import to_bytes

        data, _ = to_bytes(results, format_hint=None)
        with open(output_path, 'wb') as f:
            f.write(data)
        logger.info(f"Results saved to: {output_path}")

    @staticmethod
    def load_results(input_path: str) -> Dict[str, Any]:
        """Load SHAP results from disk using the new serializer."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import from_bytes

        with open(input_path, 'rb') as f:
            data = f.read()
        return from_bytes(data, 'cloudpickle')



