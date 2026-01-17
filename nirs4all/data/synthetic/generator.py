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
    - Instrument archetype simulation (Phase 2)
    - Measurement mode physics (transmittance, reflectance, ATR) (Phase 2)
    - Detector response and noise models (Phase 2)
    - Multi-sensor stitching (Phase 2)
    - Multi-scan averaging/denoising (Phase 2)

References:
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas for
      Interpretive Near-Infrared Spectroscopy. CRC Press.
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared Analysis.
      CRC Press.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

from .components import ComponentLibrary
from ._constants import (
    COMPLEXITY_PARAMS,
    DEFAULT_REALISTIC_COMPONENTS,
    DEFAULT_WAVELENGTH_END,
    DEFAULT_WAVELENGTH_START,
    DEFAULT_WAVELENGTH_STEP,
)
from .instruments import (
    EdgeArtifactsConfig,
    InstrumentArchetype,
    InstrumentSimulator,
    MultiScanConfig,
    MultiSensorConfig,
    SensorConfig,
    get_instrument_archetype,
    get_instrument_wavelengths,
)
from .measurement_modes import (
    MeasurementMode,
    MeasurementModeSimulator,
    create_transmittance_simulator,
)
from .detectors import (
    DetectorConfig,
    DetectorSimulator,
    get_default_noise_config,
)
from .environmental import (
    EnvironmentalEffectsConfig,
    TemperatureConfig,
    MoistureConfig,
)
from .scattering import (
    ScatteringEffectsConfig,
    ParticleSizeConfig,
    EMSCConfig,
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

    Phase 2 Features:
        - Instrument archetype simulation (FOSS, Bruker, etc.)
        - Measurement mode physics (transmittance, reflectance, ATR)
        - Detector response curves and noise models
        - Multi-sensor stitching (combining signals from different wavelength ranges)
        - Multi-scan averaging/denoising (simulating multiple scans per sample)

    Phase 3 Features:
        - Temperature effects on spectral bands (O-H, N-H, C-H shifts)
        - Moisture and water activity effects
        - Particle size effects (EMSC-style scattering)

    Attributes:
        wavelengths: Array of wavelength values in nm.
        n_wavelengths: Number of wavelength points.
        library: ComponentLibrary containing spectral components.
        E: Precomputed component spectra matrix (n_components, n_wavelengths).
        params: Dictionary of effect parameters based on complexity level.
        instrument: Optional InstrumentArchetype for realistic simulation.
        measurement_mode_simulator: Optional measurement mode simulator.

    Args:
        wavelength_start: Start wavelength in nm.
        wavelength_end: End wavelength in nm.
        wavelength_step: Wavelength step in nm.
        component_library: Optional ComponentLibrary. If None, generates
            predefined components for realistic mode or random for simple mode.
        complexity: Complexity level controlling noise, scatter, etc.
            Options: 'simple', 'realistic', 'complex'.
        instrument: Instrument archetype name or InstrumentArchetype object.
            If provided, uses instrument-specific wavelength range, detector, etc.
        measurement_mode: Measurement mode (transmittance, reflectance, etc.).
        multi_sensor_config: Configuration for multi-sensor stitching.
        multi_scan_config: Configuration for multi-scan averaging.
        environmental_config: Phase 3 configuration for temperature/moisture effects.
        scattering_effects_config: Phase 3 configuration for particle size/scattering.
        random_state: Random seed for reproducibility.

    Example:
        >>> generator = SyntheticNIRSGenerator(random_state=42)
        >>> X, Y, E = generator.generate(n_samples=1000)
        >>> print(X.shape, Y.shape, E.shape)
        (1000, 751) (1000, 5) (5, 751)

        >>> # With instrument simulation (Phase 2)
        >>> generator = SyntheticNIRSGenerator(
        ...     instrument="foss_xds",
        ...     measurement_mode="reflectance",
        ...     random_state=42
        ... )
        >>> X, Y, E = generator.generate(n_samples=500)

        >>> # With environmental effects (Phase 3)
        >>> from nirs4all.data.synthetic import EnvironmentalEffectsConfig
        >>> env_config = EnvironmentalEffectsConfig(
        ...     enable_temperature=True,
        ...     enable_moisture=True
        ... )
        >>> generator = SyntheticNIRSGenerator(
        ...     environmental_config=env_config,
        ...     random_state=42
        ... )
        >>> X, Y, E = generator.generate(n_samples=500, include_environmental_effects=True)

        >>> # Create a SpectroDataset directly
        >>> dataset = generator.create_dataset(n_train=800, n_test=200)

    See Also:
        ComponentLibrary: For managing spectral components.
        InstrumentArchetype: For instrument-specific simulation.
        MeasurementModeSimulator: For measurement mode physics.
        nirs4all.operators.augmentation.TemperatureAugmenter: For temperature effects.
        nirs4all.operators.augmentation.MoistureAugmenter: For moisture effects.
        nirs4all.operators.augmentation.ParticleSizeAugmenter: For particle size effects.
        nirs4all.operators.augmentation.EMSCDistortionAugmenter: For EMSC-style distortions.
    """

    def __init__(
        self,
        wavelength_start: float = DEFAULT_WAVELENGTH_START,
        wavelength_end: float = DEFAULT_WAVELENGTH_END,
        wavelength_step: float = DEFAULT_WAVELENGTH_STEP,
        wavelengths: Optional[np.ndarray] = None,
        instrument_wavelength_grid: Optional[str] = None,
        component_library: Optional[ComponentLibrary] = None,
        complexity: Literal["simple", "realistic", "complex"] = "realistic",
        instrument: Optional[Union[str, InstrumentArchetype]] = None,
        measurement_mode: Optional[Union[str, MeasurementMode]] = None,
        multi_sensor_config: Optional[MultiSensorConfig] = None,
        multi_scan_config: Optional[MultiScanConfig] = None,
        environmental_config: Optional[EnvironmentalEffectsConfig] = None,
        scattering_effects_config: Optional[ScatteringEffectsConfig] = None,
        edge_artifacts_config: Optional[EdgeArtifactsConfig] = None,
        custom_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the synthetic NIRS generator.

        Args:
            wavelength_start: Start wavelength in nm (ignored if wavelengths provided).
            wavelength_end: End wavelength in nm (ignored if wavelengths provided).
            wavelength_step: Wavelength step in nm (ignored if wavelengths provided).
            wavelengths: Custom wavelength array in nm. If provided, overrides
                wavelength_start/end/step parameters. Useful for matching real
                instrument wavelength grids.
            instrument_wavelength_grid: Name of predefined instrument wavelength grid
                (e.g., 'micronir_onsite', 'foss_xds'). If provided, uses the
                wavelength grid for that instrument. See get_instrument_wavelengths().
            component_library: Optional ComponentLibrary instance.
                If None, creates appropriate library based on complexity.
            complexity: Complexity level: 'simple', 'realistic', or 'complex'.
            instrument: Instrument archetype name (e.g., 'foss_xds', 'bruker_mpa')
                or InstrumentArchetype object. If provided, instrument-specific
                parameters are used for wavelength range and noise.
            measurement_mode: Measurement mode ('transmittance', 'reflectance',
                'atr', 'transflectance'). If None, uses 'transmittance'.
            multi_sensor_config: Configuration for multi-sensor stitching.
                If None and instrument has multi_sensor, uses instrument defaults.
            multi_scan_config: Configuration for multi-scan averaging.
                If None and instrument has multi_scan, uses instrument defaults.
            environmental_config: Phase 3 configuration for environmental effects
                (temperature and moisture). If provided, enables simulation of
                temperature-induced band shifts and moisture effects.
            scattering_effects_config: Phase 3 configuration for scattering effects
                (particle size and EMSC-style distortions). If provided, enables
                simulation of particle-size-dependent scattering using nirs4all
                operators (ParticleSizeAugmenter, EMSCDistortionAugmenter).
            edge_artifacts_config: Configuration for edge artifact effects
                (detector roll-off, stray light, truncated peaks, edge curvature).
                If provided, enables simulation of common spectral edge artifacts
                using nirs4all operators (DetectorRollOffAugmenter, StrayLightAugmenter,
                EdgeCurvatureAugmenter, TruncatedPeakAugmenter).
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If complexity is not a valid option.

        Example:
            >>> # Generate spectra matching MicroNIR instrument
            >>> gen = SyntheticNIRSGenerator(instrument_wavelength_grid="micronir_onsite")
            >>> X, Y, E = gen.generate(n_samples=500)
            >>> print(f"Wavelengths: {len(gen.wavelengths)} from {gen.wavelengths[0]:.0f} to {gen.wavelengths[-1]:.0f} nm")

            >>> # Generate with custom wavelength array
            >>> custom_wl = np.linspace(1000, 2000, 100)
            >>> gen = SyntheticNIRSGenerator(wavelengths=custom_wl)
            >>> X, Y, E = gen.generate(n_samples=500)
        """
        if complexity not in COMPLEXITY_PARAMS:
            valid = list(COMPLEXITY_PARAMS.keys())
            raise ValueError(f"complexity must be one of {valid}, got '{complexity}'")

        self.complexity = complexity
        self.rng = np.random.default_rng(random_state)
        self._random_state = random_state

        # Phase 6: Custom wavelength grid support
        # Priority: wavelengths > instrument_wavelength_grid > instrument > defaults
        custom_wavelength_grid: Optional[np.ndarray] = None

        if wavelengths is not None:
            # Direct custom wavelength array provided
            custom_wavelength_grid = np.asarray(wavelengths)
        elif instrument_wavelength_grid is not None:
            # Use predefined instrument wavelength grid
            custom_wavelength_grid = get_instrument_wavelengths(instrument_wavelength_grid)

        # Phase 2: Instrument archetype setup
        self.instrument: Optional[InstrumentArchetype] = None
        self.instrument_simulator: Optional[InstrumentSimulator] = None
        self.multi_sensor_config = multi_sensor_config
        self.multi_scan_config = multi_scan_config

        if instrument is not None:
            if isinstance(instrument, str):
                self.instrument = get_instrument_archetype(instrument)
            else:
                self.instrument = instrument

            # Use instrument's wavelength range if not explicitly specified
            # (and no custom wavelength grid provided)
            if custom_wavelength_grid is None:
                if wavelength_start == DEFAULT_WAVELENGTH_START:
                    wavelength_start = self.instrument.wavelength_range[0]
                if wavelength_end == DEFAULT_WAVELENGTH_END:
                    wavelength_end = self.instrument.wavelength_range[1]

            # Get multi-sensor config from instrument if not provided
            if self.multi_sensor_config is None and self.instrument.multi_sensor is not None:
                self.multi_sensor_config = self.instrument.multi_sensor

            # Get multi-scan config from instrument if not provided
            if self.multi_scan_config is None and self.instrument.multi_scan is not None:
                self.multi_scan_config = self.instrument.multi_scan

            # Create instrument simulator
            self.instrument_simulator = InstrumentSimulator(
                self.instrument, random_state=random_state
            )

        # Generate or use provided wavelength grid
        if custom_wavelength_grid is not None:
            self.wavelengths = custom_wavelength_grid
            # Infer start/end/step from custom grid
            self.wavelength_start = float(self.wavelengths[0])
            self.wavelength_end = float(self.wavelengths[-1])
            if len(self.wavelengths) > 1:
                self.wavelength_step = float(np.mean(np.diff(self.wavelengths)))
            else:
                self.wavelength_step = 1.0
        else:
            self.wavelength_start = wavelength_start
            self.wavelength_end = wavelength_end
            self.wavelength_step = wavelength_step
            self.wavelengths = np.arange(
                wavelength_start, wavelength_end + wavelength_step, wavelength_step
            )

        self.n_wavelengths = len(self.wavelengths)

        # Phase 2: Measurement mode setup
        self.measurement_mode: Optional[MeasurementMode] = None
        self.measurement_mode_simulator: Optional[MeasurementModeSimulator] = None

        if measurement_mode is not None:
            if isinstance(measurement_mode, str):
                self.measurement_mode = MeasurementMode(measurement_mode.lower())
            else:
                self.measurement_mode = measurement_mode

            # Create measurement mode simulator
            self.measurement_mode_simulator = create_transmittance_simulator(
                scattering_enabled=True, random_state=random_state
            )

        # Phase 2: Detector simulator
        self.detector_simulator: Optional[DetectorSimulator] = None
        if self.instrument is not None:
            detector_type = self.instrument.detector_type
            noise_config = get_default_noise_config(detector_type)
            detector_config = DetectorConfig(
                detector_type=detector_type,
                noise_model=noise_config,
                apply_response_curve=True,
            )
            self.detector_simulator = DetectorSimulator(detector_config, random_state)

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

        # Set complexity-dependent parameters, with optional custom overrides
        self.params = COMPLEXITY_PARAMS[complexity].copy()
        if custom_params is not None:
            # Merge custom params, allowing override of complexity defaults
            for key, value in custom_params.items():
                if key in self.params:
                    self.params[key] = value

        # Phase 3: Environmental effects configuration (uses operators)
        self.environmental_config = environmental_config
        self._temperature_op = None
        self._moisture_op = None
        if environmental_config is not None:
            self._init_environmental_operators()

        # Phase 3: Scattering effects configuration (uses operators)
        self.scattering_effects_config = scattering_effects_config
        self._particle_op = None
        self._emsc_op = None
        if scattering_effects_config is not None:
            self._init_scattering_operators()

        # Phase 6: Edge artifacts configuration (uses operators)
        self.edge_artifacts_config = edge_artifacts_config
        self._detector_rolloff_op = None
        self._stray_light_op = None
        self._edge_curvature_op = None
        self._truncated_peak_op = None
        if edge_artifacts_config is not None:
            self._init_edge_artifact_operators()

    def _init_environmental_operators(self) -> None:
        """
        Initialize environmental effect operators (temperature, moisture).

        Creates TemperatureAugmenter and MoistureAugmenter based on the
        environmental_config settings.
        """
        from nirs4all.operators.augmentation import (
            TemperatureAugmenter,
            MoistureAugmenter,
        )

        # Initialize temperature operator
        if self.environmental_config.enable_temperature:
            temp_config = self.environmental_config.temperature
            if temp_config is not None:
                # Determine temperature range based on config
                variation = temp_config.temperature_variation
                if variation > 0:
                    temperature_range = (-variation, variation)
                else:
                    temperature_range = None

                self._temperature_op = TemperatureAugmenter(
                    temperature_delta=temp_config.delta_temperature,
                    temperature_range=temperature_range,
                    reference_temperature=temp_config.reference_temperature,
                    enable_shift=temp_config.enable_shift,
                    enable_intensity=temp_config.enable_intensity,
                    enable_broadening=temp_config.enable_broadening,
                    region_specific=temp_config.region_specific,
                    random_state=self._random_state,
                )

        # Initialize moisture operator
        if self.environmental_config.enable_moisture:
            moisture_config = self.environmental_config.moisture
            if moisture_config is not None:
                self._moisture_op = MoistureAugmenter(
                    water_activity_delta=moisture_config.water_activity - moisture_config.reference_aw,
                    reference_water_activity=moisture_config.reference_aw,
                    free_water_fraction=moisture_config.free_water_fraction,
                    bound_water_shift=moisture_config.bound_water_shift,
                    moisture_content=moisture_config.moisture_content,
                    random_state=self._random_state,
                )

    def _init_scattering_operators(self) -> None:
        """
        Initialize scattering effect operators (particle size, EMSC).

        Creates ParticleSizeAugmenter and EMSCDistortionAugmenter based on
        the scattering_effects_config settings.
        """
        from nirs4all.operators.augmentation import (
            ParticleSizeAugmenter,
            EMSCDistortionAugmenter,
        )

        # Initialize particle size operator
        if self.scattering_effects_config.enable_particle_size:
            particle_config = self.scattering_effects_config.particle_size
            if particle_config is not None:
                dist = particle_config.distribution
                self._particle_op = ParticleSizeAugmenter(
                    mean_size_um=dist.mean_size_um,
                    size_variation_um=dist.std_size_um,
                    reference_size_um=particle_config.reference_size_um,
                    wavelength_exponent=particle_config.wavelength_exponent,
                    size_effect_strength=particle_config.size_effect_strength,
                    include_path_length=particle_config.include_path_length_effect,
                    path_length_sensitivity=particle_config.path_length_sensitivity,
                    random_state=self._random_state,
                )

        # Initialize EMSC distortion operator
        if self.scattering_effects_config.enable_emsc:
            emsc_config = self.scattering_effects_config.emsc
            if emsc_config is not None:
                # Convert std to range (approximate 95% CI)
                mult_half_range = 2.0 * emsc_config.multiplicative_scatter_std
                add_half_range = 2.0 * emsc_config.additive_scatter_std

                self._emsc_op = EMSCDistortionAugmenter(
                    multiplicative_range=(1.0 - mult_half_range, 1.0 + mult_half_range),
                    additive_range=(-add_half_range, add_half_range),
                    polynomial_order=emsc_config.polynomial_order,
                    polynomial_strength=emsc_config.wavelength_coef_std,
                    random_state=self._random_state,
                )

    def _init_edge_artifact_operators(self) -> None:
        """
        Initialize edge artifact operators.

        Creates DetectorRollOffAugmenter, StrayLightAugmenter, EdgeCurvatureAugmenter,
        and TruncatedPeakAugmenter based on edge_artifacts_config settings.
        """
        from nirs4all.operators.augmentation import (
            DetectorRollOffAugmenter,
            StrayLightAugmenter,
            EdgeCurvatureAugmenter,
            TruncatedPeakAugmenter,
        )

        config = self.edge_artifacts_config

        # Initialize detector roll-off operator
        if config.enable_detector_rolloff:
            self._detector_rolloff_op = DetectorRollOffAugmenter(
                detector_model=config.detector_model,
                effect_strength=config.rolloff_severity,
                random_state=self._random_state,
            )

        # Initialize stray light operator
        if config.enable_stray_light:
            self._stray_light_op = StrayLightAugmenter(
                stray_light_fraction=config.stray_fraction,
                random_state=self._random_state,
            )

        # Initialize edge curvature operator
        if config.enable_edge_curvature:
            # Average the left/right severity for overall strength
            avg_severity = (config.left_curvature_severity + config.right_curvature_severity) / 2.0
            # Calculate asymmetry if different
            if config.left_curvature_severity + config.right_curvature_severity > 0:
                asymmetry = (config.right_curvature_severity - config.left_curvature_severity) / \
                           (config.left_curvature_severity + config.right_curvature_severity)
            else:
                asymmetry = 0.0

            self._edge_curvature_op = EdgeCurvatureAugmenter(
                curvature_type=config.curvature_type,
                curvature_strength=avg_severity * 0.1,  # Scale to appropriate range
                asymmetry=asymmetry,
                random_state=self._random_state,
            )

        # Initialize truncated peak operator
        if config.enable_truncated_peaks:
            # Determine amplitude range from config values
            min_amp = min(config.left_peak_amplitude, config.right_peak_amplitude)
            max_amp = max(config.left_peak_amplitude, config.right_peak_amplitude)
            if min_amp == max_amp:
                min_amp = max(0.01, max_amp * 0.5)

            self._truncated_peak_op = TruncatedPeakAugmenter(
                amplitude_range=(min_amp, max_amp),
                left_edge=config.left_peak_amplitude > 0,
                right_edge=config.right_peak_amplitude > 0,
                random_state=self._random_state,
            )

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

    # ========================================================================
    # Phase 2: Multi-Sensor Stitching
    # ========================================================================

    def _apply_multi_sensor_stitching(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """
        Apply multi-sensor stitching effects.

        Real NIR instruments often use multiple sensors to cover a wide
        wavelength range (e.g., Si for 400-1100nm, InGaAs for 900-1700nm).
        This creates stitching artifacts at the junction regions.

        Args:
            spectra: Input spectra (n_samples, n_wavelengths).
            wavelengths: Wavelength array (nm).

        Returns:
            Spectra with multi-sensor stitching effects applied.
        """
        if self.multi_sensor_config is None:
            return spectra

        config = self.multi_sensor_config
        n_samples, n_wl = spectra.shape
        result = spectra.copy()

        # Check if enabled
        if not config.enabled:
            return result

        # Get sensor configurations
        sensors = config.sensors
        if len(sensors) < 2:
            return result

        # Get artifact intensity for noise level
        artifact_intensity = config.artifact_intensity if config.add_stitch_artifacts else 0.0

        # Process each junction between sensors
        for i in range(len(sensors) - 1):
            sensor1 = sensors[i]
            sensor2 = sensors[i + 1]

            # Find overlap region
            overlap_start = max(sensor1.wavelength_range[0], sensor2.wavelength_range[0])
            overlap_end = min(sensor1.wavelength_range[1], sensor2.wavelength_range[1])

            if overlap_start >= overlap_end:
                # No overlap - this is a hard transition
                junction_wl = (sensor1.wavelength_range[1] + sensor2.wavelength_range[0]) / 2
                junction_idx = np.argmin(np.abs(wavelengths - junction_wl))

                # Add stitching offset artifact
                if config.add_stitch_artifacts:
                    offset = self.rng.normal(0, artifact_intensity, n_samples)
                    result[:, junction_idx:] += offset[:, np.newaxis]

                    # Add stitching noise
                    noise_width = min(10, n_wl - junction_idx)
                    noise = self.rng.normal(0, artifact_intensity * 0.5, (n_samples, noise_width))
                    result[:, junction_idx:junction_idx + noise_width] += noise
            else:
                # Overlap region - apply blending
                result = self._blend_overlap_region(
                    result, wavelengths, overlap_start, overlap_end,
                    config.stitch_method, artifact_intensity
                )

        return result

    def _blend_overlap_region(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        overlap_start: float,
        overlap_end: float,
        method: str,
        noise_level: float,
    ) -> np.ndarray:
        """
        Blend spectra in the overlap region between two sensors.

        Args:
            spectra: Input spectra.
            wavelengths: Wavelength array.
            overlap_start: Start of overlap region (nm).
            overlap_end: End of overlap region (nm).
            method: Blending method ('weighted', 'average', 'first', 'last').
            noise_level: Amount of noise to add in overlap region.

        Returns:
            Blended spectra.
        """
        n_samples = spectra.shape[0]
        result = spectra.copy()

        # Find overlap indices
        overlap_mask = (wavelengths >= overlap_start) & (wavelengths <= overlap_end)
        overlap_indices = np.where(overlap_mask)[0]

        if len(overlap_indices) == 0:
            return result

        # Add sensor-specific characteristics in overlap region
        if method == "weighted":
            # Linear blend weights
            weights = np.linspace(0, 1, len(overlap_indices))
            # Apply slight gain difference between sensors
            gain_diff = self.rng.normal(0, 0.02, n_samples)
            for i, idx in enumerate(overlap_indices):
                result[:, idx] += gain_diff * (weights[i] - 0.5)
        elif method == "average":
            # Simulate averaging two sensor signals (adds noise)
            noise = self.rng.normal(0, noise_level * 0.5, (n_samples, len(overlap_indices)))
            result[:, overlap_mask] += noise
        elif method == "optimal":
            # Simulate optimal combination (lower noise than individual sensors)
            noise = self.rng.normal(0, noise_level * 0.3, (n_samples, len(overlap_indices)))
            result[:, overlap_mask] += noise

        # Add stitching artifacts
        stitching_noise = self.rng.normal(0, noise_level, (n_samples, len(overlap_indices)))
        result[:, overlap_mask] += stitching_noise

        return result

    # ========================================================================
    # Phase 2: Multi-Scan Averaging
    # ========================================================================

    def _simulate_multi_scan_averaging(
        self,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate the effect of averaging multiple scans per sample.

        Real NIR instruments often acquire multiple scans per sample and
        average them to reduce noise. This method simulates that process.

        Args:
            spectra: Input spectra (n_samples, n_wavelengths).

        Returns:
            Spectra after simulating multi-scan averaging.
        """
        if self.multi_scan_config is None:
            return spectra

        config = self.multi_scan_config

        # Check if enabled
        if not config.enabled:
            return spectra

        n_samples, n_wl = spectra.shape
        n_scans = config.n_scans

        # Generate individual scan noise
        # More scans = lower effective noise (sqrt(n) reduction)
        scan_noise_std = config.scan_to_scan_noise

        # Generate all scans for each sample
        all_scans = np.zeros((n_samples, n_scans, n_wl))

        for scan_idx in range(n_scans):
            # Each scan has independent noise
            scan_noise = self.rng.normal(0, scan_noise_std, (n_samples, n_wl))

            # Add wavelength jitter between scans (small shifts)
            if config.wavelength_jitter > 0:
                jitter = self.rng.normal(0, config.wavelength_jitter, n_samples)
                # Add as a small baseline shift effect
                scan_noise += jitter[:, np.newaxis] * 0.001

            all_scans[:, scan_idx, :] = spectra + scan_noise

        # Apply averaging/denoising method
        method = config.averaging_method.lower()

        if method == "mean":
            result = np.mean(all_scans, axis=1)
        elif method == "median":
            result = np.median(all_scans, axis=1)
        elif method == "weighted":
            # Weight scans by inverse variance (more stable scans get higher weight)
            scan_variances = np.var(all_scans, axis=2, keepdims=True)
            weights = 1.0 / (scan_variances + 1e-10)
            weights = weights / weights.sum(axis=1, keepdims=True)
            result = np.sum(all_scans * weights, axis=1)
        elif method == "savgol":
            # Apply Savitzky-Golay filter across scans
            result = np.mean(all_scans, axis=1)  # Start with mean
            for i in range(n_samples):
                if n_wl >= 11:  # Minimum points for window_length=11
                    result[i] = savgol_filter(result[i], window_length=11, polyorder=3)
        else:
            # Default to mean
            result = np.mean(all_scans, axis=1)

        # Apply outlier rejection if enabled
        if config.discard_outliers:
            result = self._reject_scan_outliers(
                all_scans, result, config.outlier_threshold
            )

        return result

    def _reject_scan_outliers(
        self,
        all_scans: np.ndarray,
        averaged: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Reject outlier scans and recalculate average.

        Args:
            all_scans: All scan data (n_samples, n_scans, n_wavelengths).
            averaged: Current averaged result.
            threshold: Z-score threshold for outlier detection.

        Returns:
            Averaged spectra with outliers rejected.
        """
        n_samples, n_scans, n_wl = all_scans.shape
        result = averaged.copy()

        for i in range(n_samples):
            # Calculate deviation from mean for each scan
            deviations = np.abs(all_scans[i] - averaged[i])
            mean_deviation = np.mean(deviations, axis=1)

            # Z-score for each scan
            z_scores = (mean_deviation - np.mean(mean_deviation)) / (np.std(mean_deviation) + 1e-10)

            # Mask outlier scans
            valid_scans = z_scores < threshold

            if np.sum(valid_scans) >= 2:  # Need at least 2 valid scans
                result[i] = np.mean(all_scans[i, valid_scans], axis=0)

        return result

    # ========================================================================
    # Phase 2: Detector Effects
    # ========================================================================

    def _apply_detector_effects(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """
        Apply detector-specific effects.

        Args:
            spectra: Input spectra.
            wavelengths: Wavelength array.

        Returns:
            Spectra with detector effects applied.
        """
        if self.detector_simulator is None:
            return spectra

        return self.detector_simulator.apply(spectra, wavelengths)

    def generate(
        self,
        n_samples: int = 1000,
        concentration_method: Literal[
            "dirichlet", "uniform", "lognormal", "correlated"
        ] = "dirichlet",
        include_batch_effects: bool = False,
        n_batches: int = 1,
        include_instrument_effects: bool = True,
        include_multi_sensor: bool = True,
        include_multi_scan: bool = True,
        include_environmental_effects: bool = True,
        include_scattering_effects: bool = True,
        include_edge_artifacts: bool = True,
        temperatures: Optional[np.ndarray] = None,
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
            include_instrument_effects: Whether to apply instrument-specific
                effects (detector response, noise). Only applies if instrument
                was specified during initialization.
            include_multi_sensor: Whether to apply multi-sensor stitching
                effects. Only applies if multi_sensor_config is set.
            include_multi_scan: Whether to simulate multi-scan averaging.
                Only applies if multi_scan_config is set.
            include_environmental_effects: Whether to apply Phase 3 temperature
                and moisture effects. Only applies if environmental_config is set.
            include_scattering_effects: Whether to apply Phase 3 particle size
                and EMSC-style scattering effects. Only applies if
                scattering_effects_config is set.
            include_edge_artifacts: Whether to apply edge artifact effects
                (detector roll-off, stray light, edge curvature, truncated peaks).
                Only applies if edge_artifacts_config is set.
            temperatures: Optional array of temperatures (°C) for each sample.
                If None and environmental effects are enabled, random temperatures
                are generated based on the configuration. Shape: (n_samples,).
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

            >>> # With instrument simulation (Phase 2)
            >>> generator = SyntheticNIRSGenerator(
            ...     instrument="foss_xds",
            ...     random_state=42
            ... )
            >>> X, Y, E = generator.generate(n_samples=500)

            >>> # With environmental effects (Phase 3)
            >>> from nirs4all.data.synthetic import EnvironmentalEffectsConfig
            >>> env_config = EnvironmentalEffectsConfig()
            >>> generator = SyntheticNIRSGenerator(
            ...     environmental_config=env_config,
            ...     random_state=42
            ... )
            >>> X, Y, E = generator.generate(n_samples=500, include_environmental_effects=True)

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
            "instrument": self.instrument.name if self.instrument else None,
            "multi_sensor": self.multi_sensor_config is not None,
            "multi_scan": self.multi_scan_config is not None,
            "environmental_effects": self.environmental_config is not None,
            "scattering_effects": self.scattering_effects_config is not None,
            "edge_artifacts": self.edge_artifacts_config is not None,
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

        # 10. Phase 2: Apply detector effects
        if include_instrument_effects and self.detector_simulator is not None:
            A = self._apply_detector_effects(A, self.wavelengths)
        else:
            # 10. Add noise (legacy method if no detector simulator)
            A = self._add_noise(A)

        # 11. Phase 2: Apply multi-sensor stitching effects
        if include_multi_sensor and self.multi_sensor_config is not None:
            A = self._apply_multi_sensor_stitching(A, self.wavelengths)
            metadata["multi_sensor_config"] = {
                "n_sensors": len(self.multi_sensor_config.sensors),
                "stitch_method": self.multi_sensor_config.stitch_method,
            }

        # 12. Phase 2: Simulate multi-scan averaging
        if include_multi_scan and self.multi_scan_config is not None:
            A = self._simulate_multi_scan_averaging(A)
            metadata["multi_scan_config"] = {
                "n_scans": self.multi_scan_config.n_scans,
                "averaging_method": self.multi_scan_config.averaging_method,
            }

        # 13. Phase 3: Apply environmental effects (temperature, moisture)
        if include_environmental_effects and self.environmental_config is not None:
            # Generate temperatures for metadata tracking
            if temperatures is None:
                temp_config = self.environmental_config.temperature
                if temp_config is not None:
                    base_temp = temp_config.sample_temperature
                    variation = temp_config.temperature_variation
                    if variation > 0:
                        temperatures = self.rng.normal(base_temp, variation, n_samples)
                    else:
                        temperatures = np.full(n_samples, base_temp)

            # Apply temperature and moisture operators
            if self._temperature_op is not None:
                A = self._temperature_op.transform(A, wavelengths=self.wavelengths)
            if self._moisture_op is not None:
                A = self._moisture_op.transform(A, wavelengths=self.wavelengths)

            metadata["environmental_config"] = {
                "enable_temperature": self.environmental_config.enable_temperature,
                "enable_moisture": self.environmental_config.enable_moisture,
                "temperatures": temperatures,
            }

        # 14. Phase 3: Apply scattering effects (particle size, EMSC)
        if include_scattering_effects and self.scattering_effects_config is not None:
            # Apply particle size and EMSC operators
            if self._particle_op is not None:
                A = self._particle_op.transform(A, wavelengths=self.wavelengths)
            if self._emsc_op is not None:
                A = self._emsc_op.transform(A, wavelengths=self.wavelengths)

            metadata["scattering_effects_config"] = {
                "enable_particle_size": self.scattering_effects_config.enable_particle_size,
                "enable_emsc": self.scattering_effects_config.enable_emsc,
            }

        # 15. Phase 6: Apply edge artifacts (detector roll-off, stray light, etc.)
        if include_edge_artifacts and self.edge_artifacts_config is not None:
            # Apply edge artifact operators
            if self._detector_rolloff_op is not None:
                A = self._detector_rolloff_op.transform(A, wavelengths=self.wavelengths)
            if self._stray_light_op is not None:
                A = self._stray_light_op.transform(A, wavelengths=self.wavelengths)
            if self._edge_curvature_op is not None:
                A = self._edge_curvature_op.transform(A, wavelengths=self.wavelengths)
            if self._truncated_peak_op is not None:
                A = self._truncated_peak_op.transform(A, wavelengths=self.wavelengths)

            metadata["edge_artifacts_config"] = {
                "enable_detector_rolloff": self.edge_artifacts_config.enable_detector_rolloff,
                "enable_stray_light": self.edge_artifacts_config.enable_stray_light,
                "enable_edge_curvature": self.edge_artifacts_config.enable_edge_curvature,
                "enable_truncated_peaks": self.edge_artifacts_config.enable_truncated_peaks,
            }

        # 16. Add artifacts
        A = self._add_artifacts(A, metadata)

        if return_metadata:
            return A, C, self.E.copy(), metadata
        else:
            return A, C, self.E.copy()

    def generate_from_concentrations(
        self,
        concentrations: np.ndarray,
        include_batch_effects: bool = False,
        n_batches: int = 1,
        include_instrument_effects: bool = True,
        include_environmental_effects: bool = True,
        include_scattering_effects: bool = True,
        include_edge_artifacts: bool = True,
        temperatures: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic NIRS spectra from pre-defined concentrations.

        This method allows generating spectra using externally-provided
        concentrations (e.g., from aggregate components) instead of
        random sampling.

        Args:
            concentrations: Concentration matrix (n_samples, n_components).
                Each row should sum to approximately 1.0.
            include_batch_effects: Whether to add batch/session effects.
            n_batches: Number of batches (only if include_batch_effects=True).
            include_instrument_effects: Whether to apply instrument-specific
                effects (detector response, noise).
            include_environmental_effects: Whether to apply temperature
                and moisture effects.
            include_scattering_effects: Whether to apply particle size
                and EMSC-style scattering effects.
            include_edge_artifacts: Whether to apply edge artifact effects.
            temperatures: Optional array of temperatures (°C) for each sample.

        Returns:
            Tuple of (X, metadata) where:
                - X: Spectra matrix (n_samples, n_wavelengths)
                - metadata: Dictionary with generation details

        Example:
            >>> # Generate from aggregate concentrations
            >>> C = np.array([[0.6, 0.3, 0.1], [0.5, 0.35, 0.15]])
            >>> generator = SyntheticNIRSGenerator(
            ...     component_library=ComponentLibrary.from_predefined(
            ...         ["starch", "protein", "moisture"]
            ...     )
            ... )
            >>> X, meta = generator.generate_from_concentrations(C)
        """
        n_samples = concentrations.shape[0]
        C = concentrations

        # Build metadata
        metadata: Dict[str, Any] = {
            "n_samples": n_samples,
            "n_components": self.library.n_components,
            "n_wavelengths": self.n_wavelengths,
            "component_names": self.library.component_names,
            "wavelengths": self.wavelengths.copy(),
            "complexity": self.complexity,
            "concentration_method": "external",
            "instrument": self.instrument.name if self.instrument else None,
        }

        # 2. Apply Beer-Lambert law
        A = self._apply_beer_lambert(C)

        # 3. Apply path length variation
        A = self._apply_path_length(A)

        # 4. Generate and add baseline
        baseline = self._generate_baseline(n_samples)
        A = A + baseline

        # 5. Apply global slope
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

        # 10. Apply detector effects or add noise
        if include_instrument_effects and self.detector_simulator is not None:
            A = self._apply_detector_effects(A, self.wavelengths)
        else:
            A = self._add_noise(A)

        # 11. Apply environmental effects
        if include_environmental_effects and self.environmental_config is not None:
            if temperatures is None:
                temp_config = self.environmental_config.temperature
                if temp_config is not None:
                    base_temp = temp_config.sample_temperature
                    variation = temp_config.temperature_variation
                    if variation > 0:
                        temperatures = self.rng.normal(base_temp, variation, n_samples)
                    else:
                        temperatures = np.full(n_samples, base_temp)

            # Phase 4: Use operators if enabled
            if self._use_operators and (self._temperature_op is not None or self._moisture_op is not None):
                if self._temperature_op is not None:
                    A = self._temperature_op.transform(A, wavelengths=self.wavelengths)
                if self._moisture_op is not None:
                    A = self._moisture_op.transform(A, wavelengths=self.wavelengths)
            elif self.environmental_simulator is not None:
                # Legacy: use internal simulator
                A = self.environmental_simulator.apply(A, self.wavelengths, sample_temperatures=temperatures)

            metadata["temperatures"] = temperatures
            metadata["use_operators"] = self._use_operators

        # 12. Apply scattering effects
        if include_scattering_effects and self.scattering_effects_config is not None:
            # Phase 4: Use operators if enabled
            if self._use_operators and (self._particle_op is not None or self._emsc_op is not None):
                if self._particle_op is not None:
                    A = self._particle_op.transform(A, wavelengths=self.wavelengths)
                if self._emsc_op is not None:
                    A = self._emsc_op.transform(A, wavelengths=self.wavelengths)
            elif self.scattering_effects_simulator is not None:
                # Legacy: use internal simulator
                A = self.scattering_effects_simulator.apply(A, self.wavelengths)

        # 13. Apply edge artifacts
        if include_edge_artifacts and self.edge_artifacts_config is not None:
            if self._detector_rolloff_op is not None:
                A = self._detector_rolloff_op.transform(A, wavelengths=self.wavelengths)
            if self._stray_light_op is not None:
                A = self._stray_light_op.transform(A, wavelengths=self.wavelengths)
            if self._edge_curvature_op is not None:
                A = self._edge_curvature_op.transform(A, wavelengths=self.wavelengths)
            if self._truncated_peak_op is not None:
                A = self._truncated_peak_op.transform(A, wavelengths=self.wavelengths)

        # 14. Add artifacts
        A = self._add_artifacts(A, metadata)

        return A, metadata

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
        parts = [
            f"wavelengths={self.wavelength_start}-{self.wavelength_end}nm",
            f"n_wavelengths={self.n_wavelengths}",
            f"n_components={self.library.n_components}",
            f"complexity='{self.complexity}'",
        ]
        if self.instrument:
            parts.append(f"instrument='{self.instrument.name}'")
        if self.multi_sensor_config:
            parts.append(f"multi_sensor=True({len(self.multi_sensor_config.sensors)} sensors)")
        if self.multi_scan_config:
            parts.append(f"multi_scan=True({self.multi_scan_config.n_scans} scans)")
        if self.environmental_config:
            env_effects = []
            if self.environmental_config.enable_temperature:
                env_effects.append("temp")
            if self.environmental_config.enable_moisture:
                env_effects.append("moisture")
            parts.append(f"environmental=True({'+'.join(env_effects)})")
        if self.scattering_effects_config:
            scatter_effects = []
            if self.scattering_effects_config.enable_particle_size:
                scatter_effects.append("particle")
            if self.scattering_effects_config.enable_emsc:
                scatter_effects.append("emsc")
            parts.append(f"scattering=True({'+'.join(scatter_effects)})")
        return f"SyntheticNIRSGenerator({', '.join(parts)})"

