"""
Synthetic NIRS Spectra Generator.

A physically-motivated synthetic NIRS spectra generator based on Beer-Lambert law,
with realistic instrumental effects and noise models.

Key features:
    - Voigt profile peak shapes (Gaussian + Lorentzian convolution)
    - Realistic NIR band positions from known spectroscopic databases
    - Configurable baseline, scattering, and instrumental effects
    - Batch/session effects for domain adaptation research
    - Controllable outlier/artifact generation

References:
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas for
      Interpretive Near-Infrared Spectroscopy. CRC Press.
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared Analysis.
      CRC Press.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .components import ComponentLibrary
from ._constants import (
    COMPLEXITY_PARAMS,
    DEFAULT_REALISTIC_COMPONENTS,
    DEFAULT_WAVELENGTH_END,
    DEFAULT_WAVELENGTH_START,
    DEFAULT_WAVELENGTH_STEP,
)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset


class SyntheticNIRSGenerator:
    """
    Generator for synthetic NIRS spectra with realistic instrumental effects.

    This generator implements a physically-motivated model based on Beer-Lambert law
    with additional effects for baseline, scattering, instrumental response, and noise.

    Model:
        A_i(λ) = L_i * Σ_k c_ik * ε_k(λ) + baseline_i(λ) + scatter_i(λ) + noise_i(λ)

    where:
        - c_ik: concentration of component k in sample i
        - ε_k(λ): molar absorptivity of component k (Voigt profiles)
        - L_i: optical path length factor
        - baseline: polynomial baseline drift
        - scatter: multiplicative/additive scattering effects
        - noise: wavelength-dependent Gaussian noise

    Attributes:
        wavelengths: Array of wavelength values in nm.
        n_wavelengths: Number of wavelength points.
        library: ComponentLibrary containing spectral components.
        E: Precomputed component spectra matrix (n_components, n_wavelengths).
        params: Dictionary of effect parameters based on complexity level.

    Args:
        wavelength_start: Start wavelength in nm.
        wavelength_end: End wavelength in nm.
        wavelength_step: Wavelength step in nm.
        component_library: Optional ComponentLibrary. If None, generates
            predefined components for realistic mode or random for simple mode.
        complexity: Complexity level controlling noise, scatter, etc.
            Options: 'simple', 'realistic', 'complex'.
        random_state: Random seed for reproducibility.

    Example:
        >>> generator = SyntheticNIRSGenerator(random_state=42)
        >>> X, Y, E = generator.generate(n_samples=1000)
        >>> print(X.shape, Y.shape, E.shape)
        (1000, 751) (1000, 5) (5, 751)

        >>> # Create a SpectroDataset directly
        >>> dataset = generator.create_dataset(n_train=800, n_test=200)

    See Also:
        ComponentLibrary: For managing spectral components.
        SyntheticDatasetBuilder: For fluent dataset construction (Phase 2).
    """

    def __init__(
        self,
        wavelength_start: float = DEFAULT_WAVELENGTH_START,
        wavelength_end: float = DEFAULT_WAVELENGTH_END,
        wavelength_step: float = DEFAULT_WAVELENGTH_STEP,
        component_library: Optional[ComponentLibrary] = None,
        complexity: Literal["simple", "realistic", "complex"] = "realistic",
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the synthetic NIRS generator.

        Args:
            wavelength_start: Start wavelength in nm.
            wavelength_end: End wavelength in nm.
            wavelength_step: Wavelength step in nm.
            component_library: Optional ComponentLibrary instance.
                If None, creates appropriate library based on complexity.
            complexity: Complexity level: 'simple', 'realistic', or 'complex'.
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If complexity is not a valid option.
        """
        if complexity not in COMPLEXITY_PARAMS:
            valid = list(COMPLEXITY_PARAMS.keys())
            raise ValueError(f"complexity must be one of {valid}, got '{complexity}'")

        self.wavelength_start = wavelength_start
        self.wavelength_end = wavelength_end
        self.wavelength_step = wavelength_step
        self.complexity = complexity
        self.rng = np.random.default_rng(random_state)
        self._random_state = random_state

        # Generate wavelength grid
        self.wavelengths = np.arange(
            wavelength_start, wavelength_end + wavelength_step, wavelength_step
        )
        self.n_wavelengths = len(self.wavelengths)

        # Set up component library
        if component_library is not None:
            self.library = component_library
        else:
            # Use predefined components for realistic/complex mode
            if complexity in ("realistic", "complex"):
                self.library = ComponentLibrary.from_predefined(
                    DEFAULT_REALISTIC_COMPONENTS,
                    random_state=random_state,
                )
            else:
                # Generate random components for simple mode
                self.library = ComponentLibrary(random_state=random_state)
                self.library.generate_random_library(n_components=5)

        # Precompute component spectra (pure component matrix E)
        self.E = self.library.compute_all(self.wavelengths)

        # Set complexity-dependent parameters
        self.params = COMPLEXITY_PARAMS[complexity].copy()

    def generate_concentrations(
        self,
        n_samples: int,
        method: Literal["dirichlet", "uniform", "lognormal", "correlated"] = "dirichlet",
        alpha: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate concentration matrix using specified distribution.

        Args:
            n_samples: Number of samples to generate.
            method: Concentration generation method:
                - 'dirichlet': Compositional data (concentrations sum to ~1).
                - 'uniform': Independent uniform [0, 1] values.
                - 'lognormal': Log-normal distributed, normalized.
                - 'correlated': Multivariate with specified correlations.
            alpha: Dirichlet concentration parameters (only for 'dirichlet' method).
                Shape: (n_components,). Higher values = more uniform distribution.
            correlation_matrix: Correlation structure for 'correlated' method.
                Shape: (n_components, n_components).

        Returns:
            Concentration matrix of shape (n_samples, n_components).

        Raises:
            ValueError: If method is unknown.

        Example:
            >>> generator = SyntheticNIRSGenerator(random_state=42)
            >>> C = generator.generate_concentrations(100, method='dirichlet')
            >>> print(C.shape, C.sum(axis=1).mean())  # Should sum to ~1
        """
        n_components = self.library.n_components

        if method == "dirichlet":
            if alpha is None:
                alpha = np.ones(n_components) * 2.0
            C = self.rng.dirichlet(alpha, size=n_samples)

        elif method == "uniform":
            C = self.rng.uniform(0, 1, size=(n_samples, n_components))

        elif method == "lognormal":
            C = self.rng.lognormal(mean=0, sigma=0.5, size=(n_samples, n_components))
            C = C / C.sum(axis=1, keepdims=True)  # Normalize

        elif method == "correlated":
            C = self._generate_correlated_concentrations(
                n_samples, n_components, correlation_matrix
            )

        else:
            valid = ["dirichlet", "uniform", "lognormal", "correlated"]
            raise ValueError(f"Unknown concentration method: '{method}'. Use one of {valid}")

        return C

    def _generate_correlated_concentrations(
        self,
        n_samples: int,
        n_components: int,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate correlated concentrations using Cholesky decomposition.

        Args:
            n_samples: Number of samples.
            n_components: Number of components.
            correlation_matrix: Desired correlation structure.

        Returns:
            Concentration matrix with specified correlations.
        """
        if correlation_matrix is None:
            # Create default correlation structure
            correlation_matrix = np.eye(n_components)
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    corr = self.rng.uniform(-0.3, 0.5)
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr

        # Ensure positive definiteness
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 0.01)
        correlation_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

        L = np.linalg.cholesky(correlation_matrix)
        Z = self.rng.standard_normal((n_samples, n_components))
        C = Z @ L.T

        # Transform to positive values and normalize
        C = np.abs(C)
        C = C / C.sum(axis=1, keepdims=True)

        return C

    def _apply_beer_lambert(self, C: np.ndarray) -> np.ndarray:
        """
        Apply Beer-Lambert law: A = C @ E.

        Args:
            C: Concentration matrix (n_samples, n_components).

        Returns:
            Absorbance matrix (n_samples, n_wavelengths).
        """
        return C @ self.E

    def _apply_path_length(self, A: np.ndarray) -> np.ndarray:
        """
        Apply random path length variation to absorbance.

        Args:
            A: Absorbance matrix (n_samples, n_wavelengths).

        Returns:
            Modified absorbance matrix.
        """
        n_samples = A.shape[0]
        L = self.rng.normal(1.0, self.params["path_length_std"], size=n_samples)
        L = np.maximum(L, 0.5)  # Ensure positive
        return A * L[:, np.newaxis]

    def _generate_baseline(self, n_samples: int) -> np.ndarray:
        """
        Generate polynomial baseline drift.

        Args:
            n_samples: Number of samples.

        Returns:
            Baseline array (n_samples, n_wavelengths).
        """
        x = (self.wavelengths - self.wavelengths.mean()) / (np.ptp(self.wavelengths) / 2)
        amp = self.params["baseline_amplitude"]

        baseline = np.zeros((n_samples, self.n_wavelengths))
        for i in range(n_samples):
            b0 = self.rng.normal(0, amp)
            b1 = self.rng.normal(0, amp * 0.5)
            b2 = self.rng.normal(0, amp * 0.3)
            b3 = self.rng.normal(0, amp * 0.1)
            baseline[i] = b0 + b1 * x + b2 * x**2 + b3 * x**3

        return baseline

    def _apply_global_slope(self, A: np.ndarray) -> np.ndarray:
        """
        Apply global slope effect commonly observed in NIR spectra.

        This simulates the typical upward trend in absorbance with increasing
        wavelength, caused by scattering effects and instrumental factors.

        Args:
            A: Absorbance matrix (n_samples, n_wavelengths).

        Returns:
            Modified absorbance matrix.
        """
        n_samples = A.shape[0]

        # Normalized wavelength (0 to 1 across the range)
        wl_range = np.ptp(self.wavelengths)
        x_norm = (self.wavelengths - self.wavelengths.min()) / wl_range

        # Generate slopes: mean + sample-specific variation
        slope_mean = self.params["global_slope_mean"]
        slope_std = self.params["global_slope_std"]
        slopes = self.rng.normal(slope_mean, slope_std, size=n_samples)

        # Scale to actual wavelength range (slope is per 1000nm)
        scale_factor = wl_range / 1000.0

        # Apply slope to each sample
        for i in range(n_samples):
            A[i] += slopes[i] * scale_factor * x_norm

        return A

    def _apply_scatter(self, A: np.ndarray) -> np.ndarray:
        """
        Apply multiplicative/additive scatter effects (SNV/MSC-like before correction).

        Args:
            A: Absorbance matrix (n_samples, n_wavelengths).

        Returns:
            Modified absorbance matrix.
        """
        n_samples = A.shape[0]

        # Multiplicative scatter
        alpha = self.rng.normal(1.0, self.params["scatter_alpha_std"], size=n_samples)
        alpha = np.maximum(alpha, 0.7)

        # Additive offset
        beta = self.rng.normal(0, self.params["scatter_beta_std"], size=n_samples)

        # Apply
        A_scattered = A * alpha[:, np.newaxis] + beta[:, np.newaxis]

        # Add tilt
        x = (self.wavelengths - self.wavelengths.mean()) / np.ptp(self.wavelengths)
        gamma = self.rng.normal(0, self.params["tilt_std"], size=n_samples)
        tilt = gamma[:, np.newaxis] * x[np.newaxis, :]
        A_scattered += tilt

        return A_scattered

    def _apply_wavelength_shift(self, A: np.ndarray) -> np.ndarray:
        """
        Apply wavelength calibration shifts/stretches via interpolation.

        Args:
            A: Absorbance matrix (n_samples, n_wavelengths).

        Returns:
            Modified absorbance matrix.
        """
        n_samples = A.shape[0]

        shifts = self.rng.normal(0, self.params["shift_std"], size=n_samples)
        stretches = self.rng.normal(1.0, self.params["stretch_std"], size=n_samples)

        A_shifted = np.zeros_like(A)
        for i in range(n_samples):
            wl_shifted = stretches[i] * self.wavelengths + shifts[i]
            A_shifted[i] = np.interp(self.wavelengths, wl_shifted, A[i])

        return A_shifted

    def _apply_instrumental_response(self, A: np.ndarray) -> np.ndarray:
        """
        Apply instrumental broadening via Gaussian convolution.

        Args:
            A: Absorbance matrix (n_samples, n_wavelengths).

        Returns:
            Modified absorbance matrix.
        """
        fwhm = self.params["instrumental_fwhm"]
        # Convert FWHM to sigma in wavelength step units
        sigma_wl = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_pts = sigma_wl / self.wavelength_step

        A_convolved = np.zeros_like(A)
        for i in range(A.shape[0]):
            A_convolved[i] = gaussian_filter1d(A[i], sigma_pts)

        return A_convolved

    def _add_noise(self, A: np.ndarray) -> np.ndarray:
        """
        Add wavelength-dependent Gaussian noise.

        Args:
            A: Absorbance matrix (n_samples, n_wavelengths).

        Returns:
            Modified absorbance matrix with noise.
        """
        sigma_base = self.params["noise_base"]
        sigma_signal = self.params["noise_signal_dep"]

        # Heteroscedastic noise: higher noise at higher absorbance
        sigma = sigma_base + sigma_signal * np.abs(A)
        noise = self.rng.normal(0, sigma)

        return A + noise

    def _add_artifacts(self, A: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Add random artifacts (spikes, dead bands, saturation).

        Args:
            A: Absorbance matrix (n_samples, n_wavelengths).
            metadata: Dictionary to store artifact information.

        Returns:
            Modified absorbance matrix.
        """
        n_samples = A.shape[0]
        artifact_prob = self.params["artifact_prob"]

        artifact_types: List[Optional[str]] = []
        for i in range(n_samples):
            if self.rng.random() < artifact_prob:
                artifact_type = self.rng.choice(["spike", "dead_band", "saturation"])
                artifact_types.append(artifact_type)

                if artifact_type == "spike":
                    n_spikes = self.rng.integers(1, 4)
                    spike_indices = self.rng.choice(
                        self.n_wavelengths, n_spikes, replace=False
                    )
                    spike_values = self.rng.uniform(0.5, 1.5, n_spikes)
                    A[i, spike_indices] += spike_values * np.sign(
                        self.rng.standard_normal(n_spikes)
                    )

                elif artifact_type == "dead_band":
                    start_idx = self.rng.integers(0, self.n_wavelengths - 20)
                    width = self.rng.integers(10, 30)
                    end_idx = min(start_idx + width, self.n_wavelengths)
                    A[i, start_idx:end_idx] += self.rng.normal(
                        0, 0.05, end_idx - start_idx
                    )

                elif artifact_type == "saturation":
                    threshold = self.rng.uniform(0.8, 1.2)
                    A[i] = np.clip(A[i], -np.inf, threshold)
            else:
                artifact_types.append(None)

        metadata["artifact_types"] = artifact_types
        return A

    def generate_batch_effects(
        self,
        n_batches: int,
        samples_per_batch: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate batch/session effects for domain adaptation research.

        Args:
            n_batches: Number of measurement batches/sessions.
            samples_per_batch: List of sample counts per batch.

        Returns:
            Tuple of:
                - batch_offsets: Wavelength-dependent offsets per batch.
                - batch_gains: Multiplicative gains per batch.
        """
        batch_offsets = []
        batch_gains = []

        for _ in range(n_batches):
            # Baseline offset per batch (slow drift)
            x = (self.wavelengths - self.wavelengths.mean()) / np.ptp(self.wavelengths)
            offset = self.rng.normal(0, 0.02) + self.rng.normal(0, 0.01) * x
            batch_offsets.append(offset)

            # Gain variation per batch
            gain = self.rng.normal(1.0, 0.03)
            batch_gains.append(gain)

        return np.array(batch_offsets), np.array(batch_gains)

    def generate(
        self,
        n_samples: int = 1000,
        concentration_method: Literal[
            "dirichlet", "uniform", "lognormal", "correlated"
        ] = "dirichlet",
        include_batch_effects: bool = False,
        n_batches: int = 1,
        return_metadata: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    ]:
        """
        Generate synthetic NIRS spectra.

        This is the main generation method that creates synthetic spectra
        by applying all physical effects in sequence.

        Args:
            n_samples: Number of spectra to generate.
            concentration_method: Method for generating concentrations.
                Options: 'dirichlet', 'uniform', 'lognormal', 'correlated'.
            include_batch_effects: Whether to add batch/session effects.
            n_batches: Number of batches (only if include_batch_effects=True).
            return_metadata: Whether to return additional metadata dictionary.

        Returns:
            If return_metadata=False:
                Tuple of (X, Y, E):
                    - X: Spectra matrix (n_samples, n_wavelengths)
                    - Y: Concentration matrix (n_samples, n_components)
                    - E: Component spectra (n_components, n_wavelengths)

            If return_metadata=True:
                Tuple of (X, Y, E, metadata):
                    - metadata: Dictionary with generation details

        Example:
            >>> generator = SyntheticNIRSGenerator(random_state=42)
            >>> X, Y, E = generator.generate(n_samples=500)
            >>> print(f"Spectra: {X.shape}, Targets: {Y.shape}")
            Spectra: (500, 751), Targets: (500, 5)

            >>> # With metadata
            >>> X, Y, E, meta = generator.generate(100, return_metadata=True)
            >>> print(meta.keys())
        """
        metadata: Dict[str, Any] = {
            "n_samples": n_samples,
            "n_components": self.library.n_components,
            "n_wavelengths": self.n_wavelengths,
            "component_names": self.library.component_names,
            "wavelengths": self.wavelengths.copy(),
            "complexity": self.complexity,
            "concentration_method": concentration_method,
        }

        # 1. Generate concentrations
        C = self.generate_concentrations(n_samples, method=concentration_method)

        # 2. Apply Beer-Lambert law
        A = self._apply_beer_lambert(C)

        # 3. Apply path length variation
        A = self._apply_path_length(A)

        # 4. Generate and add baseline
        baseline = self._generate_baseline(n_samples)
        A = A + baseline

        # 5. Apply global slope (typical NIR upward trend)
        A = self._apply_global_slope(A)

        # 6. Apply scatter effects
        A = self._apply_scatter(A)

        # 7. Apply batch effects if requested
        if include_batch_effects and n_batches > 1:
            samples_per_batch = [n_samples // n_batches] * n_batches
            samples_per_batch[-1] += n_samples % n_batches

            batch_offsets, batch_gains = self.generate_batch_effects(
                n_batches, samples_per_batch
            )

            batch_ids = []
            idx = 0
            for batch_id, n_in_batch in enumerate(samples_per_batch):
                batch_ids.extend([batch_id] * n_in_batch)
                A[idx : idx + n_in_batch] = (
                    A[idx : idx + n_in_batch] * batch_gains[batch_id]
                    + batch_offsets[batch_id]
                )
                idx += n_in_batch

            metadata["batch_ids"] = np.array(batch_ids)
            metadata["batch_offsets"] = batch_offsets
            metadata["batch_gains"] = batch_gains

        # 8. Apply wavelength shift/stretch
        A = self._apply_wavelength_shift(A)

        # 9. Apply instrumental response
        A = self._apply_instrumental_response(A)

        # 10. Add noise
        A = self._add_noise(A)

        # 11. Add artifacts
        A = self._add_artifacts(A, metadata)

        if return_metadata:
            return A, C, self.E.copy(), metadata
        else:
            return A, C, self.E.copy()

    def create_dataset(
        self,
        n_train: int = 800,
        n_test: int = 200,
        target_component: Optional[Union[str, int]] = None,
        **generate_kwargs: Any,
    ) -> SpectroDataset:
        """
        Create a SpectroDataset from synthetic spectra.

        This method generates synthetic spectra and wraps them in a
        SpectroDataset object ready for use with nirs4all pipelines.

        Args:
            n_train: Number of training samples.
            n_test: Number of test samples.
            target_component: Which component to use as target.
                - If None: uses all components as multi-output target.
                - If str: uses the component with that name.
                - If int: uses the component at that index.
            **generate_kwargs: Additional arguments passed to generate().

        Returns:
            SpectroDataset with train/test partitions.

        Example:
            >>> generator = SyntheticNIRSGenerator(random_state=42)
            >>> dataset = generator.create_dataset(
            ...     n_train=800,
            ...     n_test=200,
            ...     target_component="protein"
            ... )
            >>> print(f"Train: {dataset.n_train}, Test: {dataset.n_test}")
        """
        from nirs4all.data import SpectroDataset

        # Generate all samples
        n_total = n_train + n_test
        X, C, _E = self.generate(n_samples=n_total, **generate_kwargs)

        # Determine target
        if target_component is None:
            y = C
        elif isinstance(target_component, str):
            comp_idx = self.library.component_names.index(target_component)
            y = C[:, comp_idx]
        else:
            y = C[:, target_component]

        # Create dataset
        dataset = SpectroDataset(name="synthetic_nirs")

        # Create wavelength headers
        headers = [str(int(wl)) for wl in self.wavelengths]

        # Add training samples
        dataset.add_samples(
            X[:n_train],
            indexes={"partition": "train"},
            headers=headers,
            header_unit="nm",
        )
        dataset.add_targets(y[:n_train])

        # Add test samples
        dataset.add_samples(
            X[n_train:],
            indexes={"partition": "test"},
            headers=headers,
            header_unit="nm",
        )
        dataset.add_targets(y[n_train:])

        return dataset

    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (
            f"SyntheticNIRSGenerator("
            f"wavelengths={self.wavelength_start}-{self.wavelength_end}nm, "
            f"n_wavelengths={self.n_wavelengths}, "
            f"n_components={self.library.n_components}, "
            f"complexity='{self.complexity}')"
        )
