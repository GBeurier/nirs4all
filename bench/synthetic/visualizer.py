"""
Visualization tools for synthetic NIRS spectra.

Provides comprehensive plotting functions for analyzing and understanding
the generated synthetic spectra, including:
- Spectral overview plots (2D and 3D)
- Component library visualization
- Concentration distribution analysis
- Batch effect visualization
- Noise analysis
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class SyntheticSpectraVisualizer:
    """
    Comprehensive visualizer for synthetic NIRS spectra.

    Provides methods to visualize:
    - Generated spectra with concentration coloring
    - Component library and band structures
    - Concentration distributions
    - Batch effects
    - Noise characteristics
    - Statistical summaries

    Example:
        >>> from examples.synthetic import SyntheticNIRSGenerator, SyntheticSpectraVisualizer
        >>> generator = SyntheticNIRSGenerator(random_state=42)
        >>> X, Y, E = generator.generate(n_samples=500)
        >>> viz = SyntheticSpectraVisualizer(
        ...     X, Y, generator.wavelengths,
        ...     component_names=generator.library.component_names,
        ...     component_spectra=E
        ... )
        >>> viz.plot_all()
    """

    def __init__(
        self,
        spectra: np.ndarray,
        concentrations: np.ndarray,
        wavelengths: np.ndarray,
        component_names: Optional[List[str]] = None,
        component_spectra: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the visualizer.

        Args:
            spectra: Generated spectra (n_samples, n_wavelengths)
            concentrations: Concentration matrix (n_samples, n_components)
            wavelengths: Wavelength grid (n_wavelengths,)
            component_names: Names of components
            component_spectra: Pure component spectra (n_components, n_wavelengths)
            metadata: Optional metadata from generation
        """
        self.X = spectra
        self.Y = concentrations
        self.wavelengths = wavelengths
        self.n_samples, self.n_wavelengths = spectra.shape
        self.n_components = concentrations.shape[1]
        self.component_names = component_names or [f"Component {i}" for i in range(self.n_components)]
        self.E = component_spectra
        self.metadata = metadata or {}

    def plot_spectra_overview(
        self,
        n_display: int = 100,
        color_by: str = "component",
        component_idx: int = 0,
        alpha: float = 0.5,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (12, 6),
    ) -> plt.Figure:
        """
        Plot overview of generated spectra with concentration-based coloring.

        Args:
            n_display: Number of spectra to display
            color_by: Coloring method ('component', 'sample_index', 'random')
            component_idx: Index of component for color mapping (if color_by='component')
            alpha: Transparency of lines
            ax: Optional existing axes
            figsize: Figure size

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Select random subset of samples
        indices = np.random.choice(self.n_samples, min(n_display, self.n_samples), replace=False)
        indices = np.sort(indices)

        if color_by == "component":
            colors = self.Y[indices, component_idx]
            cmap = cm.viridis
            norm = Normalize(vmin=colors.min(), vmax=colors.max())
            label = f"Concentration: {self.component_names[component_idx]}"
        elif color_by == "sample_index":
            colors = indices
            cmap = cm.plasma
            norm = Normalize(vmin=0, vmax=self.n_samples)
            label = "Sample index"
        else:  # random
            colors = np.random.rand(len(indices))
            cmap = cm.tab20
            norm = Normalize(vmin=0, vmax=1)
            label = "Random color"

        for i, idx in enumerate(indices):
            ax.plot(
                self.wavelengths, self.X[idx],
                color=cmap(norm(colors[i])),
                alpha=alpha,
                linewidth=0.8
            )

        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_ylabel("Absorbance (a.u.)", fontsize=12)
        ax.set_title(f"Synthetic NIRS Spectra (n={len(indices)})", fontsize=14, fontweight='bold')

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(label, fontsize=10)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_spectra_3d(
        self,
        n_display: int = 100,
        color_by_component: int = 0,
        figsize: Tuple[float, float] = (14, 8),
        elevation: float = 25,
        azimuth: float = -60,
    ) -> plt.Figure:
        """
        Create 3D surface plot of spectra.

        Args:
            n_display: Number of spectra to display
            color_by_component: Component index for coloring
            figsize: Figure size
            elevation: 3D view elevation angle
            azimuth: 3D view azimuth angle

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Select and sort samples by target component
        indices = np.random.choice(self.n_samples, min(n_display, self.n_samples), replace=False)
        sort_order = np.argsort(self.Y[indices, color_by_component])
        indices = indices[sort_order]

        # Create mesh grid
        X_wl, Y_idx = np.meshgrid(self.wavelengths, np.arange(len(indices)))
        Z = self.X[indices]

        # Color by concentration
        colors = self.Y[indices, color_by_component]
        cmap = cm.viridis
        norm = Normalize(vmin=colors.min(), vmax=colors.max())

        # Plot each spectrum as a line
        for i, idx in enumerate(indices):
            ax.plot(
                self.wavelengths,
                [i] * len(self.wavelengths),
                self.X[idx],
                color=cmap(norm(colors[i])),
                alpha=0.7,
                linewidth=0.5
            )

        ax.set_xlabel("Wavelength (nm)", fontsize=10)
        ax.set_ylabel("Sample (sorted)", fontsize=10)
        ax.set_zlabel("Absorbance (a.u.)", fontsize=10)
        ax.set_title(f"3D Spectral View (sorted by {self.component_names[color_by_component]})",
                     fontsize=12, fontweight='bold')

        ax.view_init(elev=elevation, azim=azimuth)

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(f"{self.component_names[color_by_component]} concentration", fontsize=10)

        plt.tight_layout()
        return fig

    def plot_spectral_envelope(
        self,
        show_percentiles: bool = True,
        figsize: Tuple[float, float] = (12, 6),
    ) -> plt.Figure:
        """
        Plot spectral envelope showing mean, std, and percentiles.

        Args:
            show_percentiles: Whether to show 5th/95th percentile bands
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        mean_spec = self.X.mean(axis=0)
        std_spec = self.X.std(axis=0)
        min_spec = self.X.min(axis=0)
        max_spec = self.X.max(axis=0)

        # Plot min-max range
        ax.fill_between(self.wavelengths, min_spec, max_spec,
                        alpha=0.2, color='blue', label='Min-Max range')

        if show_percentiles:
            p5 = np.percentile(self.X, 5, axis=0)
            p95 = np.percentile(self.X, 95, axis=0)
            ax.fill_between(self.wavelengths, p5, p95,
                            alpha=0.3, color='blue', label='5th-95th percentile')

        # Plot mean ± std
        ax.fill_between(self.wavelengths, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.4, color='blue', label='Mean ± 1 std')

        ax.plot(self.wavelengths, mean_spec, 'b-', linewidth=2, label='Mean spectrum')

        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_ylabel("Absorbance (a.u.)", fontsize=12)
        ax.set_title("Spectral Envelope", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_component_library(
        self,
        normalize: bool = True,
        stacked: bool = False,
        figsize: Tuple[float, float] = (12, 6),
    ) -> plt.Figure:
        """
        Plot pure component spectra from the library.

        Args:
            normalize: Whether to normalize spectra to [0, 1]
            stacked: Whether to stack spectra vertically
            figsize: Figure size

        Returns:
            Figure object
        """
        if self.E is None:
            raise ValueError("Component spectra (E) not provided to visualizer")

        fig, ax = plt.subplots(figsize=figsize)
        cmap = cm.Set2

        for i in range(self.n_components):
            spectrum = self.E[i].copy()
            if normalize:
                spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-10)
            if stacked:
                spectrum = spectrum + i * 1.2

            ax.plot(self.wavelengths, spectrum,
                    color=cmap(i / max(self.n_components - 1, 1)),
                    linewidth=2,
                    label=self.component_names[i])

        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ylabel = "Normalized Absorbance" if normalize else "Absorbance (a.u.)"
        if stacked:
            ylabel += " (stacked)"
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title("Component Library - Pure Spectra", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', ncol=2 if self.n_components > 4 else 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_concentration_distributions(
        self,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Plot concentration distributions for all components.

        Args:
            figsize: Figure size (auto-calculated if None)

        Returns:
            Figure object
        """
        n_cols = min(3, self.n_components)
        n_rows = (self.n_components + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (4 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if self.n_components == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        cmap = cm.Set2

        for i in range(self.n_components):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            data = self.Y[:, i]
            color = cmap(i / max(self.n_components - 1, 1))

            ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
            ax.axvline(np.median(data), color='orange', linestyle=':', linewidth=2, label=f'Median: {np.median(data):.3f}')

            ax.set_xlabel("Concentration", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(self.component_names[i], fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(self.n_components, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle("Concentration Distributions", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_concentration_correlations(
        self,
        figsize: Tuple[float, float] = (10, 8),
    ) -> plt.Figure:
        """
        Plot concentration correlation matrix.

        Args:
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        corr_matrix = np.corrcoef(self.Y.T)

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation coefficient", fontsize=10)

        # Set ticks
        ax.set_xticks(np.arange(self.n_components))
        ax.set_yticks(np.arange(self.n_components))
        ax.set_xticklabels(self.component_names, rotation=45, ha='right')
        ax.set_yticklabels(self.component_names)

        # Add correlation values as text
        for i in range(self.n_components):
            for j in range(self.n_components):
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha='center', va='center', color=color, fontsize=9)

        ax.set_title("Concentration Correlation Matrix", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_batch_effects(
        self,
        batch_ids: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (14, 5),
    ) -> Optional[plt.Figure]:
        """
        Visualize batch effects in the spectra.

        Args:
            batch_ids: Array of batch IDs per sample (from metadata if None)
            figsize: Figure size

        Returns:
            Figure object or None if no batch info
        """
        if batch_ids is None:
            batch_ids = self.metadata.get("batch_ids")

        if batch_ids is None:
            print("No batch information available")
            return None

        unique_batches = np.unique(batch_ids)
        n_batches = len(unique_batches)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Mean spectrum per batch
        ax1 = axes[0]
        cmap = cm.tab10
        for i, batch_id in enumerate(unique_batches):
            mask = batch_ids == batch_id
            mean_spec = self.X[mask].mean(axis=0)
            ax1.plot(self.wavelengths, mean_spec,
                     color=cmap(i / max(n_batches - 1, 1)),
                     label=f'Batch {batch_id}',
                     linewidth=1.5)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Mean Absorbance")
        ax1.set_title("Mean Spectrum per Batch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Difference from overall mean
        ax2 = axes[1]
        overall_mean = self.X.mean(axis=0)
        for i, batch_id in enumerate(unique_batches):
            mask = batch_ids == batch_id
            batch_mean = self.X[mask].mean(axis=0)
            diff = batch_mean - overall_mean
            ax2.plot(self.wavelengths, diff,
                     color=cmap(i / max(n_batches - 1, 1)),
                     label=f'Batch {batch_id}',
                     linewidth=1.5)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Difference from Mean")
        ax2.set_title("Batch Deviation from Overall Mean")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: PCA colored by batch
        ax3 = axes[2]
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        for i, batch_id in enumerate(unique_batches):
            mask = batch_ids == batch_id
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=[cmap(i / max(n_batches - 1, 1))],
                        alpha=0.5,
                        s=20,
                        label=f'Batch {batch_id}')
        ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax3.set_title("PCA - Colored by Batch")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle("Batch Effects Analysis", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_noise_analysis(
        self,
        n_samples_for_residuals: int = 100,
        figsize: Tuple[float, float] = (14, 5),
    ) -> plt.Figure:
        """
        Analyze noise characteristics of the spectra.

        Args:
            n_samples_for_residuals: Number of samples for residual analysis
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Wavelength-dependent noise (std at each wavelength)
        ax1 = axes[0]
        std_per_wl = self.X.std(axis=0)
        ax1.plot(self.wavelengths, std_per_wl, 'b-', linewidth=1.5)
        ax1.fill_between(self.wavelengths, 0, std_per_wl, alpha=0.3)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Standard Deviation")
        ax1.set_title("Noise Level vs Wavelength")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Signal-dependent noise analysis
        ax2 = axes[1]
        # Compute local std using rolling window
        n_subset = min(n_samples_for_residuals, self.n_samples)
        subset_idx = np.random.choice(self.n_samples, n_subset, replace=False)

        local_means = []
        local_stds = []
        window_size = 10

        for idx in subset_idx:
            spectrum = self.X[idx]
            for i in range(0, len(spectrum) - window_size, window_size // 2):
                window = spectrum[i:i + window_size]
                local_means.append(window.mean())
                local_stds.append(window.std())

        local_means = np.array(local_means)
        local_stds = np.array(local_stds)

        ax2.scatter(local_means, local_stds, alpha=0.1, s=5)
        ax2.set_xlabel("Local Mean (Signal Level)")
        ax2.set_ylabel("Local Std (Noise Level)")
        ax2.set_title("Signal-Dependent Noise")
        ax2.grid(True, alpha=0.3)

        # Fit linear trend
        if len(local_means) > 10:
            coeffs = np.polyfit(local_means, local_stds, 1)
            x_fit = np.linspace(local_means.min(), local_means.max(), 100)
            y_fit = np.polyval(coeffs, x_fit)
            ax2.plot(x_fit, y_fit, 'r-', linewidth=2,
                     label=f'Linear fit: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}')
            ax2.legend()

        # Plot 3: Noise histogram
        ax3 = axes[2]
        # Estimate noise using first difference
        first_diff = np.diff(self.X, axis=1)
        noise_estimate = first_diff.flatten() / np.sqrt(2)

        ax3.hist(noise_estimate, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Fit Gaussian
        mu, sigma = noise_estimate.mean(), noise_estimate.std()
        x_gauss = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        y_gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2)
        ax3.plot(x_gauss, y_gauss, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.4f}, σ={sigma:.4f}')

        ax3.set_xlabel("Noise Value")
        ax3.set_ylabel("Density")
        ax3.set_title("Noise Distribution (1st Difference)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle("Noise Analysis", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_pca_analysis(
        self,
        n_components_max: int = 10,
        color_by_component: int = 0,
        figsize: Tuple[float, float] = (14, 5),
    ) -> plt.Figure:
        """
        PCA analysis of the synthetic spectra.

        Args:
            n_components_max: Maximum PCA components to compute
            color_by_component: Component index for coloring scatter plot
            figsize: Figure size

        Returns:
            Figure object
        """
        from sklearn.decomposition import PCA

        n_components = min(n_components_max, self.n_wavelengths, self.n_samples)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Scree plot
        ax1 = axes[0]
        ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_ * 100, alpha=0.7)
        ax1.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_) * 100,
                 'ro-', linewidth=2, markersize=6)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Variance Explained (%)")
        ax1.set_title("Scree Plot")
        ax1.set_xticks(range(1, n_components + 1))
        ax1.grid(True, alpha=0.3)
        ax1.axhline(95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
        ax1.legend()

        # Plot 2: PC1 vs PC2 colored by concentration
        ax2 = axes[1]
        colors = self.Y[:, color_by_component]
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis', alpha=0.5, s=20)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label(f"{self.component_names[color_by_component]}")
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax2.set_title(f"PCA Score Plot (colored by {self.component_names[color_by_component]})")
        ax2.grid(True, alpha=0.3)

        # Plot 3: First 3 loadings
        ax3 = axes[2]
        for i in range(min(3, n_components)):
            ax3.plot(self.wavelengths, pca.components_[i],
                     label=f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)',
                     linewidth=1.5)
        ax3.set_xlabel("Wavelength (nm)")
        ax3.set_ylabel("Loading")
        ax3.set_title("PCA Loadings")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle("PCA Analysis", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_all(
        self,
        show: bool = True,
        save_prefix: Optional[str] = None,
    ) -> List[plt.Figure]:
        """
        Generate all visualization plots.

        Args:
            show: Whether to call plt.show()
            save_prefix: If provided, save plots with this prefix

        Returns:
            List of Figure objects
        """
        figures = []

        # 1. Spectra overview
        fig = self.plot_spectra_overview(n_display=100)
        figures.append(("spectra_overview", fig))

        # 2. Spectral envelope
        fig = self.plot_spectral_envelope()
        figures.append(("spectral_envelope", fig))

        # 3. 3D view
        fig = self.plot_spectra_3d(n_display=100)
        figures.append(("spectra_3d", fig))

        # 4. Component library (if available)
        if self.E is not None:
            fig = self.plot_component_library(stacked=True)
            figures.append(("component_library", fig))

        # 5. Concentration distributions
        fig = self.plot_concentration_distributions()
        figures.append(("concentration_distributions", fig))

        # 6. Concentration correlations
        fig = self.plot_concentration_correlations()
        figures.append(("concentration_correlations", fig))

        # 7. Noise analysis
        fig = self.plot_noise_analysis()
        figures.append(("noise_analysis", fig))

        # 8. PCA analysis
        fig = self.plot_pca_analysis()
        figures.append(("pca_analysis", fig))

        # 9. Batch effects (if available)
        if "batch_ids" in self.metadata:
            fig = self.plot_batch_effects()
            if fig is not None:
                figures.append(("batch_effects", fig))

        # Save if prefix provided
        if save_prefix:
            for name, fig in figures:
                fig.savefig(f"{save_prefix}_{name}.png", dpi=150, bbox_inches='tight')
                print(f"Saved: {save_prefix}_{name}.png")

        if show:
            plt.show()

        return [fig for _, fig in figures]


# ============================================================================
# Convenience functions for quick plotting
# ============================================================================

def plot_synthetic_spectra(
    X: np.ndarray,
    Y: np.ndarray,
    wavelengths: np.ndarray,
    n_display: int = 100,
    color_by_component: int = 0,
    component_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    show: bool = True,
) -> plt.Figure:
    """
    Quick plot of synthetic spectra with concentration coloring.

    Args:
        X: Spectra (n_samples, n_wavelengths)
        Y: Concentrations (n_samples, n_components)
        wavelengths: Wavelength grid
        n_display: Number of spectra to display
        color_by_component: Which component to use for coloring
        component_names: Optional list of component names
        title: Optional custom title
        figsize: Figure size
        show: Whether to call plt.show()

    Returns:
        Figure object
    """
    viz = SyntheticSpectraVisualizer(
        X, Y, wavelengths,
        component_names=component_names
    )
    fig = viz.plot_spectra_overview(
        n_display=n_display,
        component_idx=color_by_component,
    )
    if title:
        fig.axes[0].set_title(title, fontsize=14, fontweight='bold')

    if show:
        plt.show()

    return fig


def plot_component_library(
    E: np.ndarray,
    wavelengths: np.ndarray,
    component_names: Optional[List[str]] = None,
    stacked: bool = True,
    figsize: Tuple[float, float] = (12, 6),
    show: bool = True,
) -> plt.Figure:
    """
    Quick plot of component library.

    Args:
        E: Component spectra (n_components, n_wavelengths)
        wavelengths: Wavelength grid
        component_names: Optional list of component names
        stacked: Whether to stack spectra vertically
        figsize: Figure size
        show: Whether to call plt.show()

    Returns:
        Figure object
    """
    n_components = E.shape[0]
    dummy_Y = np.eye(n_components)  # Dummy concentrations

    viz = SyntheticSpectraVisualizer(
        E, dummy_Y, wavelengths,
        component_names=component_names,
        component_spectra=E
    )
    fig = viz.plot_component_library(stacked=stacked)

    if show:
        plt.show()

    return fig


def plot_concentration_distributions(
    Y: np.ndarray,
    component_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Quick plot of concentration distributions.

    Args:
        Y: Concentrations (n_samples, n_components)
        component_names: Optional list of component names
        figsize: Figure size
        show: Whether to call plt.show()

    Returns:
        Figure object
    """
    n_components = Y.shape[1]
    dummy_X = np.zeros((Y.shape[0], 10))  # Dummy spectra
    dummy_wl = np.arange(10)

    viz = SyntheticSpectraVisualizer(
        dummy_X, Y, dummy_wl,
        component_names=component_names
    )
    fig = viz.plot_concentration_distributions(figsize=figsize)

    if show:
        plt.show()

    return fig


def plot_batch_effects(
    X: np.ndarray,
    wavelengths: np.ndarray,
    batch_ids: np.ndarray,
    figsize: Tuple[float, float] = (14, 5),
    show: bool = True,
) -> plt.Figure:
    """
    Quick plot of batch effects.

    Args:
        X: Spectra (n_samples, n_wavelengths)
        wavelengths: Wavelength grid
        batch_ids: Array of batch IDs per sample
        figsize: Figure size
        show: Whether to call plt.show()

    Returns:
        Figure object
    """
    n_samples = X.shape[0]
    dummy_Y = np.zeros((n_samples, 1))

    viz = SyntheticSpectraVisualizer(
        X, dummy_Y, wavelengths,
        metadata={"batch_ids": batch_ids}
    )
    fig = viz.plot_batch_effects()

    if show:
        plt.show()

    return fig


def plot_noise_analysis(
    X: np.ndarray,
    wavelengths: np.ndarray,
    figsize: Tuple[float, float] = (14, 5),
    show: bool = True,
) -> plt.Figure:
    """
    Quick noise analysis plot.

    Args:
        X: Spectra (n_samples, n_wavelengths)
        wavelengths: Wavelength grid
        figsize: Figure size
        show: Whether to call plt.show()

    Returns:
        Figure object
    """
    n_samples = X.shape[0]
    dummy_Y = np.zeros((n_samples, 1))

    viz = SyntheticSpectraVisualizer(X, dummy_Y, wavelengths)
    fig = viz.plot_noise_analysis()

    if show:
        plt.show()

    return fig
