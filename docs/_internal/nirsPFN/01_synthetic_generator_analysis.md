# Deep Analysis of the nirs4all Synthetic Generator for NIRS-PFN Training

## Executive Summary

This document provides a comprehensive analysis of the nirs4all synthetic NIRS data generator and evaluates its suitability for training a Prior-data Fitted Network (PFN) specialized for Near-Infrared Spectroscopy predictions. We identify current strengths, critical gaps, and propose extensions needed to generate the spectral diversity required for a foundation model.

---

## ⚠️ Important Context: Current SOTA Performance

**Empirical finding**: The pipeline `ASLSBaseline → PCA(0.99) → TabPFN` already significantly outperforms traditional methods (PLS, RF, etc.) on real NIRS datasets.

**Implication**: TabPFN's in-context learning mechanism is already effective for NIRS when given appropriate features. The primary bottleneck is **feature extraction**, not the learning algorithm. This suggests:

1. **Short-term priority**: Train a spectral encoder on synthetic data (see document 02)
2. **Generator requirements are different**: Focus on spectral shape diversity, not just concentration/task diversity
3. **Full NIRS-PFN may not be necessary**: Improve the synthetic→TabPFN feature bridge first

---

## 1. Current Generator Architecture

### 1.1 Physical Model Foundation

The nirs4all generator implements an **empirical chemometric model** inspired by Beer-Lambert law. This is specifically designed for **transmission-mode spectroscopy** where absorbance is the natural measurement space.

> ⚠️ **Important clarification**: The model below is formulated in **apparent absorbance space** as typically used in chemometrics. For diffuse reflectance, the relationship between chemistry and observed signal is fundamentally different (see Section 4.2.2 for Kubelka-Munk treatment).

**Signal path layers** (explicit hierarchy):

```
Layer 1: Latent absorption coefficients
    K_k(λ) = ε_k(λ)  [1/length·concentration]
    Component-specific molar absorptivities modeled as Voigt profiles

Layer 2: Beer-Lambert mixing (transmission geometry only)
    A_pure(λ) = L · Σ_k c_k · K_k(λ)  [dimensionless]
    Where L is optical path length [length] and c_k is concentration

Layer 3: Instrumental/physical perturbations (in absorbance domain)
    A_observed(λ) = A_pure(λ) · (1 + scatter_mult(λ)) + baseline(λ) + noise(λ)

Layer 4: Optional conversion for reflectance modes (see Section 4.2.2)
    R(λ) = f(K, S)  via Kubelka-Munk for diffuse reflectance
    A_apparent(λ) = -log₁₀(R)  typical chemometric representation
```

**Current implementation** (transmission/absorbance-centric):

$$A_i(\lambda) = L_i \cdot \sum_k c_{ik} \cdot \varepsilon_k(\lambda) + \text{baseline}_i(\lambda) + \text{scatter}_i(\lambda) + \text{noise}_i(\lambda)$$

Where:
- $c_{ik}$: Concentration of component $k$ in sample $i$ [mol/L or mass fraction]
- $\varepsilon_k(\lambda)$: Molar absorptivity modeled as Voigt profiles [AU·L/mol·cm]
- $L_i$: Optical path length factor [dimensionless, normalized around 1.0]
- baseline: Polynomial drift in absorbance domain [AU]
- scatter: Multiplicative and additive distortions mimicking SNV/MSC-correctable effects
- noise: Wavelength-dependent Gaussian noise [AU]

> 📝 **Note on scatter modeling**: The "scatter" terms here are empirical nuisance models that approximate the distortions chemometric preprocessing (SNV, MSC) attempts to remove. They are NOT physics-based Mie scattering simulations. See Section 3.1.3 for the distinction.

### 1.2 Core Components

| Component | Implementation Status | Strengths | Limitations |
|-----------|----------------------|-----------|-------------|
| **ComponentLibrary** | ✅ Implemented | 31 predefined compounds based on Workman & Weyer (2012) band assignments | Limited to predefined molecules; band positions are approximate |
| **NIRBand** | ✅ Implemented | Voigt profile (Gaussian ⊗ Lorentzian) captures thermal + pressure broadening | Single-peak model; no concentration-dependent shape changes |
| **Concentration Methods** | ✅ Implemented | Dirichlet, uniform, lognormal, correlated distributions | Sum-to-one constraint may not match all real scenarios |
| **Complexity Presets** | ✅ Implemented | simple, realistic, complex configurations | Only 3 discrete presets, no continuous control |
| **Random Component Generation** | ✅ Implemented | `add_random_component()` creates random bands in NIR zones | No physical constraints (overtones, combinations) |

> 📝 **References for predefined components**: Band positions are based on Workman Jr, J., & Weyer, L. (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*. CRC Press. These are approximate literature values, not measured spectra.

### 1.3 Instrumental Effects Implemented

The following effects operate in **absorbance domain** (Layer 3 of the signal path):

| Effect | Domain | Typical Magnitude | Purpose |
|--------|--------|-------------------|---------|
| **Path length variation** | Multiplicative on A | σ = 0.1-0.3 | Sample thickness variation |
| **Polynomial baseline** | Additive to A | 0-3rd order polynomials | Drift, background |
| **Wavelength-dependent slope** | Additive to A | Linear λ term | Scattering-induced trend |
| **Multiplicative scatter** | Multiplicative on A | ±10-30% | Particle density, packing |
| **Additive scatter** | Additive to A | 0.01-0.1 AU | Baseline offsets |
| **Wavelength shift/stretch** | On λ axis | ±1-5 nm | Calibration drift |
| **Instrumental broadening** | Gaussian convolution | σ = 1-5 nm | Resolution effects |
| **Heteroscedastic noise** | Additive to A | Signal-dependent | Detector noise |
| **Artifacts** | Various | Rare (1-5%) | Spikes, dead bands, saturation |
| **Batch effects** | Session-level shifts | Variable | Domain adaptation scenarios |

> ⚠️ **Clarification on overlap**:
> - "Polynomial baseline (0-3 order)" includes slope as 1st order; "global slope" is a specific common case
> - "Multiplicative/additive scatter" are **empirical** distortions that SNV/MSC preprocessing corrects, not physics-based scattering
> - "Instrumental broadening" convolves the **final spectrum**; Voigt band shapes model **molecular line shapes** at infinite resolution

### 1.4 Target Generation Capabilities

**Regression:**
- Linear combination of concentrations
- Scaling to arbitrary ranges
- Log/sqrt transforms
- Multi-output support

**Classification:**
- Configurable class separation
- Multiple separation methods (component, threshold, cluster)
- Class weight control

**Advanced (NonLinearTargetProcessor):**
- Polynomial/synergistic/antagonistic interactions
- Hidden factors (irreducible error)
- Multi-regime landscapes
- Confounders and temporal drift
- Heteroscedastic noise

---

## 2. Comparison with TabPFN's Training Data Requirements

### 2.1 What Makes TabPFN's Prior Work

TabPFN v2 was trained on synthetic tabular data with specific characteristics that enabled zero-shot generalization:

| TabPFN Prior Property | Why It Matters | NIRS Equivalent Needed |
|-----------------------|----------------|------------------------|
| **Diverse function families** | Covers linear, polynomial, tree-like, interactions | Multiple spectral-target relationships |
| **Variable noise levels** | Learns uncertainty estimation | Variable SNR, instrument quality |
| **Mixed feature types** | Handles real + categorical | Spectra + metadata + auxiliary |
| **Varied dimensionality** | Adapts to feature count | Different wavelength ranges |
| **Realistic correlations** | Captures feature dependencies | Spectral autocorrelation patterns |
| **Causal structures** | Understands variable relationships | Physics-based component mixing |

### 2.2 Current Coverage Assessment

```
NIRS4ALL GENERATOR COVERAGE FOR PFN TRAINING:

✅ WELL COVERED:
   - Beer-Lambert physics (fundamental spectroscopy law)
   - Common agricultural/food/pharma analytes (31 components)
   - Standard instrumental effects (baseline, scatter, noise)
   - Batch effects for domain adaptation
   - Non-linear target complexity

⚠️ PARTIALLY COVERED:
   - Wavelength ranges (fixed implementation, but configurable)
   - Instrument simulation (basic parameters only)
   - Sample matrix effects (limited)
   - Environmental factors (temperature, humidity)

❌ NOT COVERED:
   - Instrument-specific response functions
   - Reflectance vs. transmittance vs. transflectance modes
   - Sample presentation geometry variations
   - Real-world matrix interactions (Maillard, oxidation)
   - Moisture migration and temporal stability
   - Temperature-induced peak shifts
   - Particle size distribution effects
   - Depth of penetration variations
   - Multi-modal fusion (NIRS + RGB + FTIR)
```

### 2.3 TabPFN Architecture Alignment Considerations

> ⚠️ **Critical design decision**: The generator must align with downstream model architecture assumptions.

#### 2.3.1 Feature Exchangeability vs. Ordered Spectra

**TabPFN assumption**: Tabular data where column order is arbitrary (features are exchangeable).

**NIRS reality**: Spectra have strong **locality** and **ordering** structure:
- Adjacent wavelengths are highly correlated
- Peak-to-peak relationships encode chemistry
- Derivative information is spatially meaningful

**Implications for generator/model design**:

| Approach | Architecture | Generator Focus | Pros | Cons |
|----------|--------------|-----------------|------|------|
| **A: Engineered features** | TabPFN on PCA/peaks | Feature extraction diversity | Proven (current SOTA) | Loses spectral structure |
| **B: Spectral encoder** | Encoder → TabPFN | Spectral shape diversity | Preserves structure | Requires encoder training |
| **C: Spectral PFN** | Custom transformer with positional encoding | Full spectral diversity | End-to-end | High development cost |
| **D: Patch tokenization** | Wavelength patches as tokens | Within-patch + between-patch patterns | Vision transformer analog | Needs tuning |

**Recommendation**: For the spectral encoder approach (document 02), focus generator extensions on **spectral shape diversity** rather than feature-level diversity.

#### 2.3.2 Task Distribution Realism

**TabPFN prior focuses on**:
- Causal structure variety
- Dataset "quirks" (missingness, noise, mixed types)
- Variable sample sizes and feature counts

**NIRS datasets additionally have**:
- Hierarchical batch structure (labs → instruments → sessions → samples)
- Replicate measurements (2-3 per sample, averaged or not)
- Group-based splits (train on farms A,B; test on farm C)
- Reference method noise (wet chemistry error ~1-5%)
- Limit of detection / censoring (values below LOD)
- Operator effects (presentation angle, packing pressure)

**Generator gap**: Current implementation models batch effects but not the full hierarchical structure or dataset-level quirks.

#### 2.3.3 Recommendations for Architecture Alignment

1. **If using TabPFN on features** (current SOTA):
   - Generator diversity should emphasize variation in **engineered feature distributions**
   - Include PCA/derivative statistics as intermediate validation

2. **If training spectral encoder** (recommended next step):
   - Generator diversity should emphasize **spectral shape** variation
   - Encoder should learn to be invariant to nuisance effects
   - Contrastive training with augmentation can encode invariances

3. **If building full spectral PFN** (future work):
   - Need explicit wavelength positional encoding
   - Prior must cover wavelength-to-output relationships
   - Dataset-level quirks become critical

---

## 3. Critical Gaps for Foundation Model Training

### 3.1 Gap Analysis by Category

#### 3.1.1 Spectral Diversity

| Gap | Impact | Severity |
|-----|--------|----------|
| **Limited compound library** | Cannot generalize to novel analytes | 🔴 Critical |
| **No band broadening by matrix** | Misses hydrogen bonding effects | 🔴 Critical |
| **Fixed wavelength grid** | Cannot learn variable resolution | 🟡 Moderate |
| **No overtone/combination relationships** | Ignores physical constraints | 🟡 Moderate |

**Current**: 31 predefined components with fixed band positions; random component generator places bands without physical constraints.

**Needed**: Procedural generation of random chemically-plausible components with:
- Random number and positions of bands (within NIR-relevant zones)
- Physically constrained band relationships:

> ⚠️ **Critical: Wavenumber vs Wavelength**
>
> Harmonic relationships (overtones, combinations) are linear in **wavenumber** ($\tilde{\nu}$, cm⁻¹), NOT wavelength (nm).
>
> - **Fundamental at 3000 cm⁻¹** → 1st overtone at ~6000 cm⁻¹ (factor of 2)
> - In wavelength: 3333 nm → 1667 nm (NOT half!)
>
> The generator MUST define band placement rules in wavenumber space, then convert to wavelength:
> $$\lambda_{nm} = \frac{10^7}{\tilde{\nu}_{cm^{-1}}}$$
>
> For NIR (800-2500 nm = 4000-12500 cm⁻¹), overtone relationships are less strict than MIR due to anharmonicity, but the direction of the constraint still matters.

- Concentration-dependent band shape changes (currently not modeled)
- Matrix-induced band shifts (-50 to +50 nm for hydrogen bonding)

#### 3.1.2 Instrument Simulation

| Gap | Impact | Severity |
|-----|--------|----------|
| **No spectrophotometer archetypes** | Cannot learn instrument fingerprints | 🔴 Critical |
| **Fixed resolution model** | Misses high vs. low resolution trade-offs | 🟡 Moderate |
| **No detector nonlinearity** | Ignores saturation curves | 🟡 Moderate |
| **Missing measurement modes** | Reflectance/transmittance very different | 🔴 Critical |

**Current**: Single set of complexity parameters.
**Needed**: Parametric instrument archetypes:
```python
INSTRUMENT_ARCHETYPES = {
    "foss_nir": {
        "spectral_range": (1100, 2500),
        "resolution": 2,
        "detector": "InGaAs",
        "noise_model": "shot_dominated",
        "stray_light": 0.01,
    },
    "benchtop_ft": {
        "spectral_range": (800, 2500),
        "resolution": 0.5,
        "detector": "InGaAs",
        "noise_model": "thermal_dominated",
        "interferogram_artifacts": True,
    },
    "handheld_led": {
        "spectral_range": (1350, 1650),
        "resolution": 8,
        "detector": "InGaAs",
        "noise_model": "high_variance",
        "led_drift": True,
    },
    # ... more archetypes
}
```

#### 3.1.3 Sample Matrix Effects

| Gap | Impact | Severity |
|-----|--------|----------|
| **No particle size effects** | Major scattering driver ignored | 🔴 Critical |
| **No porosity/density simulation** | Path length varies with packing | 🟡 Moderate |
| **No moisture diffusion** | Time-dependent spectra not modeled | 🟡 Moderate |
| **No temperature effects** | Peak shifts and broadening | 🔴 Critical |

**Current**: Simple multiplicative/additive scatter model.
**Needed**: Physics-based scattering with:
- Mie scattering for particle size distributions
- Kubelka-Munk model for diffuse reflectance
- Temperature-dependent band shifts and intensities
- Water activity effects on hydrogen bonding

#### 3.1.4 Target Relationship Diversity

| Gap | Impact | Severity |
|-----|--------|----------|
| **Limited interaction types** | Polynomial/synergistic only | 🟡 Moderate |
| **No latent variable models** | PLS/factor model relationships | 🟡 Moderate |
| **No calibration transfer** | Domain shift patterns | 🟡 Moderate |

**Current**: NonLinearTargetProcessor with polynomial, synergistic, antagonistic.
**Needed**: Expanded function families:
- Sigmoidal/threshold relationships
- Mixture-of-experts (different models in different regimes)
- Hierarchical targets (class → continuous within class)
- Time-series dependencies

### 3.2 Validation Protocol for Generator Adequacy

> ⚠️ **Replacing subjective coverage percentages**
>
> Previous versions of this document used estimates like "25% coverage" which are not defensible without measurement. Instead, we define a **quantitative validation protocol**.

#### 3.2.1 Spectral Realism Scorecard (Automated Metrics)

For each domain/instrument combination, compare synthetic vs. real spectra distributions:

| Metric | Description | Target |
|--------|-------------|--------|
| **Correlation length** | Autocorrelation decay rate | Distribution overlap > 0.8 |
| **Derivative statistics** | Mean/std of 1st and 2nd derivatives | KS-test p > 0.05 |
| **Peak density** | Number of local maxima per 100 nm | Within 20% of real |
| **Baseline curvature** | Polynomial fit residuals | Distribution overlap > 0.7 |
| **SNR distribution** | Signal-to-noise ratio estimates | Within one order of magnitude |
| **Spectral range/resolution** | Wavelength coverage and sampling | Match target instrument |

#### 3.2.2 Adversarial Validation

Train a binary classifier to distinguish real vs. synthetic spectra:

```python
def adversarial_validation(real_spectra, synthetic_spectra):
    """
    Lower AUC indicates synthetic data is more realistic.

    Target: AUC < 0.6 (hard to distinguish)
    Acceptable: AUC < 0.7
    Poor: AUC > 0.8 (easily separable)
    """
    X = np.vstack([real_spectra, synthetic_spectra])
    y = np.hstack([np.ones(len(real_spectra)), np.zeros(len(synthetic_spectra))])

    # Use simple model to avoid overfitting
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=1000)
    auc_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')

    return auc_scores.mean()
```

#### 3.2.3 Task Transfer Metrics (TSTR/RTSR)

| Protocol | Description | Success Criterion |
|----------|-------------|-------------------|
| **TSTR** (Train Synthetic, Test Real) | Train PLS/CNN on synthetic, evaluate on real | R² > 0.7 × oracle R² |
| **RTSR** (Real Train, Synthetic Refine) | Pretrain on synthetic, fine-tune on real | Improves over real-only |
| **Ablation** | Systematically disable prior components | Identify which dimensions matter |

#### 3.2.4 Domain-Specific Validation Sets

| Domain | Benchmark Dataset | Samples | Key Challenge |
|--------|-------------------|---------|---------------|
| Agriculture | Corn (Cargill) | 80 | Small sample, multiple outputs |
| Food | Tecator (meat) | 215 | Wet samples, scattering |
| Pharma | Tablets (IDRC) | 654 | Powder compacts, blend uniformity |
| Grain | Wheat kernels | 155 | Intact samples, particle size |

> 📝 **Recommended workflow**:
> 1. Start with adversarial validation on each domain
> 2. Identify largest synthetic-real gaps via scorecard
> 3. Prioritize generator extensions based on measured gaps
> 4. Iterate until TSTR performance is acceptable

---

## 4. Proposed Extensions for NIRS-PFN

### 4.1 Phase 1: Enhanced Component Generation

#### 4.1.1 Procedural Component Generator

> ⚠️ **All band placement is done in wavenumber space (cm⁻¹) then converted to wavelength (nm)**.

```python
@dataclass
class ProceduralComponentConfig:
    """Configuration for random component generation."""

    # Band generation (in wavenumber space)
    n_bands_range: Tuple[int, int] = (2, 8)
    band_center_distribution: str = "nir_zones"  # or "uniform", "gaussian"

    # Physical constraints
    allow_overtones: bool = True  # Generate related bands at 2×, 3× wavenumber
    allow_combinations: bool = True  # Generate combination bands (ν₁ + ν₂)
    hydrogen_bonding_shift_cm: Tuple[float, float] = (-100, 100)  # cm⁻¹ shift range

    # Band properties (in wavenumber units)
    amplitude_distribution: str = "lognormal"
    sigma_cm_range: Tuple[float, float] = (20, 200)  # Gaussian HWHM in cm⁻¹
    gamma_cm_range: Tuple[float, float] = (0, 50)  # Lorentzian HWHM in cm⁻¹

    # Correlation structure
    concentration_correlation_prob: float = 0.3


# NIR zones in wavenumber space (cm⁻¹)
NIR_ZONES_WAVENUMBER = [
    (9000, 12500),   # 800-1100 nm: 3rd overtones, electronic
    (7700, 9000),    # 1100-1300 nm: 2nd overtones C-H
    (6450, 7150),    # 1400-1550 nm: 1st overtones O-H, N-H
    (5550, 6060),    # 1650-1800 nm: 1st overtones C-H
    (5000, 5260),    # 1900-2000 nm: Combination O-H
    (4545, 5000),    # 2000-2200 nm: Combination N-H
    (4000, 4545),    # 2200-2500 nm: Combination C-H
]


def wavenumber_to_wavelength(nu_cm: float) -> float:
    """Convert wavenumber (cm⁻¹) to wavelength (nm)."""
    return 1e7 / nu_cm


def wavelength_to_wavenumber(lambda_nm: float) -> float:
    """Convert wavelength (nm) to wavenumber (cm⁻¹)."""
    return 1e7 / lambda_nm


class ProceduralComponentGenerator:
    """Generate random but physically plausible spectral components."""

    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.default_rng(random_state)

    def generate_random_component(self, config: ProceduralComponentConfig) -> SpectralComponent:
        """Generate a single random component with physical constraints."""
        bands = []

        # Generate fundamental bands in wavenumber space
        n_bands = self.rng.integers(*config.n_bands_range)
        fundamental_wavenumbers = self._place_bands_in_nir_zones(n_bands)

        for fund_nu in fundamental_wavenumbers:
            # Main band (convert wavenumber to wavelength for NIRBand)
            fund_lambda = wavenumber_to_wavelength(fund_nu)
            bands.append(self._create_band(fund_lambda, fund_nu, config))

            # Optional 1st overtone at ~2× wavenumber
            # Due to anharmonicity, actual position is slightly less than 2×
            if config.allow_overtones and self.rng.random() < 0.4:
                # Anharmonicity correction: ν_overtone ≈ 2×ν - δ
                anharmonicity = self.rng.uniform(0.02, 0.05)  # 2-5% reduction
                overtone_nu = fund_nu * 2 * (1 - anharmonicity)
                overtone_lambda = wavenumber_to_wavelength(overtone_nu)

                # Only include if in NIR range (800-2500 nm)
                if 800 < overtone_lambda < 2500:
                    bands.append(self._create_band(
                        overtone_lambda, overtone_nu, config,
                        amplitude_factor=0.3  # Overtones are weaker
                    ))

            # Optional combination band (ν₁ + ν₂)
            if config.allow_combinations and len(fundamental_wavenumbers) > 1:
                if self.rng.random() < 0.3:
                    partner_nu = self.rng.choice([
                        nu for nu in fundamental_wavenumbers if nu != fund_nu
                    ])
                    combo_nu = fund_nu + partner_nu
                    combo_lambda = wavenumber_to_wavelength(combo_nu)

                    if 800 < combo_lambda < 2500:
                        bands.append(self._create_band(
                            combo_lambda, combo_nu, config,
                            amplitude_factor=0.2  # Combinations are weak
                        ))

        return SpectralComponent(
            name=f"synth_{self.rng.integers(0, 1000000):06d}",
            bands=bands,
            correlation_group=self.rng.integers(0, 20),
        )

    def _place_bands_in_nir_zones(self, n_bands: int) -> List[float]:
        """Place fundamental bands in NIR-relevant wavenumber zones."""
        wavenumbers = []
        for _ in range(n_bands):
            # Pick a zone and sample within it
            zone = NIR_ZONES_WAVENUMBER[self.rng.integers(len(NIR_ZONES_WAVENUMBER))]
            nu = self.rng.uniform(*zone)
            wavenumbers.append(nu)
        return wavenumbers

    def _create_band(
        self,
        center_nm: float,
        center_cm: float,
        config: ProceduralComponentConfig,
        amplitude_factor: float = 1.0,
    ) -> NIRBand:
        """Create a NIRBand with parameters sampled from config."""
        # Convert wavenumber widths to wavelength widths
        # dλ ≈ λ²/10⁷ × dν (approximate for small widths)
        sigma_cm = self.rng.uniform(*config.sigma_cm_range)
        gamma_cm = self.rng.uniform(*config.gamma_cm_range)

        # Approximate conversion: Δλ ≈ Δν × (λ/ν) = Δν × λ² / 10⁷
        sigma_nm = sigma_cm * (center_nm ** 2) / 1e7
        gamma_nm = gamma_cm * (center_nm ** 2) / 1e7

        amplitude = self.rng.lognormal(mean=-0.5, sigma=0.5) * amplitude_factor

        return NIRBand(
            center=center_nm,
            sigma=sigma_nm,
            gamma=gamma_nm,
            amplitude=amplitude,
            name=f"band_{center_cm:.0f}cm-1",
        )
```

#### 4.1.2 Application Domain Priors

```python
APPLICATION_PRIORS = {
    "agriculture": {
        "likely_components": ["water", "protein", "starch", "cellulose", "lipid", "chlorophyll"],
        "wavelength_range": (1100, 2500),
        "typical_complexity": "realistic",
        "matrix_type": "plant_tissue",
        "moisture_range": (0.05, 0.8),
    },
    "pharmaceutical": {
        "likely_components": ["cellulose", "starch", "paracetamol", "aspirin", "caffeine"],
        "wavelength_range": (1000, 2500),
        "typical_complexity": "complex",
        "matrix_type": "powder_compact",
        "particle_size_range": (1, 100),  # microns
    },
    "dairy": {
        "likely_components": ["water", "protein", "lipid", "lactose"],
        "wavelength_range": (1100, 2500),
        "typical_complexity": "realistic",
        "matrix_type": "emulsion",
        "fat_globule_size": (1, 10),  # microns
    },
    "petroleum": {
        "likely_components": ["aromatic", "alkane", "oil"],
        "wavelength_range": (800, 2500),
        "typical_complexity": "simple",
        "matrix_type": "liquid",
    },
    # ... 20+ more domains
}
```

### 4.2 Phase 2: Instrument Simulation Enhancement

#### 4.2.1 Instrument Archetype System

> 📝 **Data sources**: Parameters marked with 📋 are from manufacturer datasheets. Parameters marked with 📊 are illustrative placeholders that should be validated against real instrument data before use.

```python
@dataclass
class InstrumentArchetype:
    """Parameterized instrument simulation."""

    name: str
    category: str  # "benchtop", "process", "handheld", "lab"

    # Optical characteristics
    spectral_range: Tuple[float, float]  # nm, 📋 from datasheets
    spectral_resolution: float  # nm FWHM, 📋 from datasheets
    wavelength_accuracy: float  # nm, 📊 estimated

    # Detector
    detector_type: str  # "PbS", "InGaAs", "extended_InGaAs", "Si"
    detector_linearity: float  # 0.99 = 1% nonlinearity, 📊 estimated
    dark_current: float  # 📊 estimated

    # Noise model (compositional: shot + read + drift)
    shot_noise_fraction: float  # [0-1]
    read_noise_level: float  # AU
    drift_rate: float  # AU/hour

    # Measurement mode
    mode: str  # "reflectance", "transmittance", "transflectance", "ATR"
    path_length: Optional[float]  # mm, for transmission

    # Artifacts
    stray_light: float  # fraction, 📊 estimated
    etalon_effect: bool  # interference fringes
    water_vapor_bands: bool
    co2_bands: bool

    # Calibration stability
    wavelength_drift: float  # nm/°C, 📊 estimated
    intensity_drift: float  # %/°C, 📊 estimated


# Grounded instrument specifications with citations
INSTRUMENT_LIBRARY = {
    # FOSS XDS RapidContent Analyzer
    # Source: Metrohm datasheet (https://www.metrohm.com/content/dam/metrohm/shared/documents/manuals/89/89218001EN.pdf)
    "foss_xds": InstrumentArchetype(
        name="FOSS XDS RapidContent",
        category="benchtop",
        spectral_range=(400, 2500),  # 📋 datasheet: VIS-NIR range
        spectral_resolution=0.5,  # 📋 datasheet: 0.5 nm data interval
        wavelength_accuracy=0.1,  # 📊 estimated
        detector_type="Si+PbS",  # 📋 silicon (VIS) + lead sulfide (NIR)
        detector_linearity=0.999,  # 📊 estimated
        dark_current=1e-4,  # 📊 estimated
        shot_noise_fraction=0.3,
        read_noise_level=1e-4,
        drift_rate=0.001,
        mode="reflectance",  # 📋 spinning cup, diffuse reflectance
        path_length=None,
        stray_light=0.001,  # 📊 estimated
        etalon_effect=False,
        water_vapor_bands=False,  # internal purge
        co2_bands=False,
        wavelength_drift=0.01,  # 📊 estimated
        intensity_drift=0.1,  # 📊 estimated
    ),

    # VIAVI MicroNIR 1700
    # Source: VIAVI datasheet (https://www.viavisolutions.com/en-us/literature/micronir-1700ec-data-sheets-en.pdf)
    "viavi_micronir": InstrumentArchetype(
        name="VIAVI MicroNIR 1700",
        category="handheld",
        spectral_range=(908, 1676),  # 📋 datasheet
        spectral_resolution=12.5,  # 📋 datasheet: ~12.5 nm FWHM (6.2 nm sampling)
        wavelength_accuracy=1.0,  # 📊 estimated for handheld
        detector_type="InGaAs",  # 📋 128-element linear array
        detector_linearity=0.99,  # 📊 estimated
        dark_current=1e-3,  # 📊 estimated, higher for compact device
        shot_noise_fraction=0.5,
        read_noise_level=5e-4,
        drift_rate=0.005,  # 📊 higher drift for portable
        mode="reflectance",
        path_length=None,
        stray_light=0.01,  # 📊 estimated, higher for compact optics
        etalon_effect=False,
        water_vapor_bands=True,  # no purge in handheld
        co2_bands=False,
        wavelength_drift=0.1,  # 📊 temperature sensitivity
        intensity_drift=0.5,  # 📊 LED source variation
    ),

    # Si-Ware NeoSpectra Micro
    # Source: Photonic Solutions (https://photonicsolutions.co.uk/products/neospectra-micro-spectral-ft-ir-sensor/)
    "neospectra_micro": InstrumentArchetype(
        name="Si-Ware NeoSpectra Micro",
        category="embedded",
        spectral_range=(1350, 2500),  # 📋 approximate, module-dependent
        spectral_resolution=16,  # 📋 ~16 nm FWHM at 1550 nm
        wavelength_accuracy=2.0,  # 📊 MEMS-based, lower precision
        detector_type="InGaAs",  # 📋 single element with MEMS FT
        detector_linearity=0.98,  # 📊 estimated
        dark_current=1e-3,  # 📊 estimated
        shot_noise_fraction=0.4,
        read_noise_level=1e-3,
        drift_rate=0.01,  # 📊 MEMS stability
        mode="reflectance",
        path_length=None,
        stray_light=0.02,  # 📊 compact MEMS optics
        etalon_effect=True,  # MEMS interferometer artifacts possible
        water_vapor_bands=True,
        co2_bands=True,
        wavelength_drift=0.2,  # 📊 MEMS temperature sensitivity
        intensity_drift=1.0,  # 📊 tungsten lamp variation
    ),

    # Generic benchtop FT-NIR (illustrative, not a specific model)
    "benchtop_ft": InstrumentArchetype(
        name="Generic FT-NIR Benchtop",
        category="benchtop",
        spectral_range=(800, 2500),  # 📊 typical extended InGaAs range
        spectral_resolution=2.0,  # 📊 typical FT resolution
        wavelength_accuracy=0.05,  # 📊 interferometer accuracy
        detector_type="extended_InGaAs",
        detector_linearity=0.999,
        dark_current=1e-5,
        shot_noise_fraction=0.2,
        read_noise_level=5e-5,
        drift_rate=0.0005,
        mode="transmittance",  # fiber optic probe typical
        path_length=1.0,  # mm
        stray_light=0.0005,
        etalon_effect=True,  # interferogram artifacts possible
        water_vapor_bands=True,  # unless purged
        co2_bands=True,
        wavelength_drift=0.005,
        intensity_drift=0.05,
    ),
}
```

> ⚠️ **Usage note**: Noise model parameters (shot_noise_fraction, read_noise_level, drift_rate) are compositional and should be sampled from distributions rather than used as fixed values. The specific numbers above are starting points for prior design.

#### 4.2.2 Measurement Mode Simulation

> ⚠️ **Critical: Units and Variable Definitions**
>
> The following code uses explicit variable names to avoid confusion:
> - `K`: Absorption coefficient [1/length], NOT absorbance
> - `S`: Scattering coefficient [1/length] (Kubelka-Munk)
> - `L`: Path length [mm]
> - `A`: Absorbance [dimensionless, AU]
> - `R`: Reflectance [0-1]
> - `T`: Transmittance [0-1]

```python
class MeasurementModeSimulator:
    """
    Simulate different NIR measurement geometries.

    This class converts from absorption coefficients K(λ) to observed signals
    appropriate for each measurement mode.
    """

    def apply_transmittance(
        self,
        K: np.ndarray,  # Absorption coefficient [1/mm]
        path_length: float,  # [mm]
    ) -> np.ndarray:
        """
        Simulate transmission measurement (Beer-Lambert).

        A = K * L (absorbance is coefficient times path length)

        Returns:
            Absorbance [AU]
        """
        return K * path_length

    def apply_reflectance(
        self,
        K: np.ndarray,  # Absorption coefficient [1/mm]
        S: np.ndarray,  # Scattering coefficient [1/mm]
    ) -> np.ndarray:
        """
        Simulate diffuse reflectance via Kubelka-Munk.

        The Kubelka-Munk function: f(R∞) = (1-R∞)²/(2R∞) = K/S

        Note: For mixtures, K-M is NOT additive in K; this is a simplification
        valid only when scattering >> absorption or for thin samples.

        Returns:
            Apparent absorbance: -log₁₀(R) [AU]
        """
        # K/S ratio
        KS_ratio = K / np.maximum(S, 1e-6)

        # Solve Kubelka-Munk for reflectance
        # R∞ = 1 + KS - sqrt(KS² + 2*KS)
        R_inf = 1 + KS_ratio - np.sqrt(KS_ratio**2 + 2 * KS_ratio)
        R_inf = np.clip(R_inf, 1e-6, 1.0)

        # Convert to apparent absorbance (chemometric convention)
        return -np.log10(R_inf)

    def apply_transflectance(
        self,
        K: np.ndarray,  # Absorption coefficient [1/mm]
        path_length: float,  # One-way path length [mm]
    ) -> np.ndarray:
        """
        Simulate transflectance (double transmission through sample).

        Light passes through sample, reflects off backing, passes through again.

        Returns:
            Absorbance [AU] (approximately 2× transmission path)
        """
        return K * path_length * 2

    def apply_atr(
        self,
        K: np.ndarray,  # Absorption coefficient [1/mm]
        wavelengths: np.ndarray,  # [nm]
        n_crystal: float = 2.4,  # Refractive index of ATR crystal (ZnSe)
        n_sample: float = 1.5,  # Refractive index of sample
        angle_deg: float = 45.0,  # Incidence angle [degrees]
    ) -> np.ndarray:
        """
        Simulate ATR with wavelength-dependent penetration depth.

        Penetration depth: dp = λ / (2π n₁ √(sin²θ - (n₂/n₁)²))

        Returns:
            Effective absorbance [AU] (scaled by penetration depth)
        """
        theta = np.radians(angle_deg)
        n_ratio = n_sample / n_crystal

        # Check for total internal reflection condition
        sin_term = np.sin(theta)**2 - n_ratio**2
        if np.any(sin_term <= 0):
            raise ValueError("Angle too small for total internal reflection")

        # Penetration depth in nm
        dp_nm = wavelengths / (2 * np.pi * n_crystal * np.sqrt(sin_term))

        # Convert to mm and scale absorbance
        dp_mm = dp_nm / 1e6  # nm to mm
        return K * dp_mm
```

> 📝 **Simplifications and limitations**:
> - Kubelka-Munk assumes diffuse irradiation and isotropic scattering
> - Real K-M is not strictly additive for mixtures
> - ATR formula assumes single-bounce geometry
> - Temperature-dependent refractive indices are not modeled

### 4.3 Phase 3: Advanced Matrix and Environmental Effects

#### 4.3.1 Temperature Effects

```python
class TemperatureEffectSimulator:
    """Simulate temperature-dependent spectral changes."""

    def apply_temperature_effects(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        temperature: float,  # Celsius
        reference_temp: float = 25.0,
    ) -> np.ndarray:
        """Apply temperature-induced peak shifts and intensity changes."""

        delta_T = temperature - reference_temp

        # 1. Wavelength shift (typically -0.1 to -0.5 nm/°C for O-H bands)
        # Water bands shift to shorter wavelengths with increasing T
        water_regions = [(1400, 1500), (1900, 2000)]
        shift_coefficient = -0.3  # nm/°C

        for wl_start, wl_end in water_regions:
            mask = (wavelengths >= wl_start) & (wavelengths <= wl_end)
            if np.any(mask):
                # Shift this region
                shift = delta_T * shift_coefficient
                spectrum = self._apply_local_shift(spectrum, wavelengths, mask, shift)

        # 2. Intensity changes (hydrogen bonding decreases with T)
        # O-H bands decrease, free O-H increases
        intensity_factor = 1 - 0.002 * delta_T  # 0.2% per °C
        spectrum = spectrum * intensity_factor

        # 3. Band broadening (increased thermal motion)
        broadening_factor = 1 + 0.001 * abs(delta_T)
        spectrum = gaussian_filter1d(spectrum, sigma=broadening_factor)

        return spectrum
```

#### 4.3.2 Particle Size Effects

> ⚠️ **Design choice: EMSC-style vs. Mie simulation**
>
> Full Mie scattering produces complex oscillatory patterns that are:
> 1. Computationally expensive
> 2. Highly sensitive to particle size distribution parameters
> 3. May generate artifacts that don't match real spectra well
>
> For PFN training, we recommend **EMSC-style** (Extended Multiplicative Scatter Correction) models that approximate the same nuisance structure that chemometric preprocessing corrects. This is safer for learning transferable representations.

```python
class ParticleSizeSimulator:
    """
    Simulate scattering effects using EMSC-style empirical models.

    This models the spectral distortions that EMSC/MSC preprocessing
    attempts to correct, rather than full physics-based Mie scattering.
    """

    def apply_particle_size_effects(
        self,
        spectrum: np.ndarray,  # Absorbance [AU]
        wavelengths: np.ndarray,  # [nm]
        particle_size_params: Tuple[float, float],  # (mean, std) in microns
        particle_density: float = 1.5,  # g/cm³ (affects baseline level)
    ) -> np.ndarray:
        """
        Apply EMSC-style scattering effects parameterized by particle size.

        Model: A_observed = a + b*A_pure + c*λ + d*λ² + e*(1/λ)
        Where coefficients depend on particle size distribution.

        Returns:
            Modified spectrum with scattering effects [AU]
        """
        mean_size, std_size = particle_size_params

        # Normalize wavelengths for numerical stability
        lambda_norm = (wavelengths - wavelengths.mean()) / wavelengths.std()

        # EMSC-style components tied to particle size physics:

        # 1. Multiplicative scaling (path length varies with packing)
        #    Smaller particles → more surface area → different scattering
        mult_factor = 1.0 + 0.1 * (50 / mean_size - 1)  # deviation from reference 50 μm
        mult_factor = np.clip(mult_factor, 0.7, 1.5)

        # 2. Additive baseline offset (smaller particles → higher baseline)
        baseline_offset = 0.05 * (1 / np.sqrt(mean_size))

        # 3. Wavelength-dependent slope (Rayleigh-like: λ^-b with b ≈ 0-4)
        #    Exponent b increases with smaller particles
        b_exponent = np.clip(4 * (10 / mean_size), 0, 4)
        lambda_ref = 1700  # reference wavelength nm
        slope_effect = 0.02 * ((wavelengths / lambda_ref) ** (-b_exponent) - 1)

        # 4. Quadratic curvature (broad size distributions)
        curvature = 0.01 * (std_size / mean_size) * lambda_norm**2

        # 5. Random scatter variation (sample-to-sample)
        # This should be different for each sample in batch generation
        noise_scale = 0.005 * (std_size / mean_size)

        # Combine effects
        spectrum_out = mult_factor * spectrum + baseline_offset + slope_effect + curvature

        return spectrum_out

    def apply_emsc_reference(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        coefficients: Dict[str, float],
    ) -> np.ndarray:
        """
        Apply arbitrary EMSC-style transformation with explicit coefficients.

        This is useful for generating diverse scattering patterns during training.

        Args:
            coefficients: Dict with keys 'a' (offset), 'b' (mult), 'c' (linear),
                         'd' (quadratic), 'e' (inverse) as per EMSC model.
        """
        lambda_norm = (wavelengths - wavelengths.mean()) / wavelengths.std()
        lambda_inv = wavelengths.max() / wavelengths  # inverse wavelength term

        a = coefficients.get('a', 0.0)
        b = coefficients.get('b', 1.0)
        c = coefficients.get('c', 0.0)
        d = coefficients.get('d', 0.0)
        e = coefficients.get('e', 0.0)

        return a + b * spectrum + c * lambda_norm + d * lambda_norm**2 + e * (lambda_inv - 1)
```

> 📝 **When to use full Mie scattering**:
> If validation shows EMSC-style models are insufficient, implement Mie as a **rare component** of the prior (e.g., 5% of training samples) rather than the default. This prevents the model from learning "ripple detectors" that don't transfer to real data.

### 4.4 Phase 4: Training Data Distribution Design

#### 4.4.1 Prior Distribution Strategy: Conditional Hierarchy

> ⚠️ **Important**: Prior sampling should be **conditional**, not independent. Many variables are dependent:
> - Measurement mode depends on instrument category
> - Wavelength range/resolution depend on instrument
> - Matrix effects depend on domain (powder vs leaf vs liquid)
> - Temperature/humidity effects vary by application

**Generative DAG for conditional sampling**:

```
Domain (agriculture, pharma, food, ...)
    │
    ├─→ Instrument Category (benchtop, handheld, inline)
    │       │
    │       ├─→ Wavelength Range (constrained by detector)
    │       ├─→ Resolution (constrained by instrument type)
    │       ├─→ Measurement Mode (R/T/TR/ATR)
    │       └─→ Noise Model (shot/thermal/MEMS-specific)
    │
    ├─→ Matrix Type (powder, liquid, tissue, emulsion)
    │       │
    │       ├─→ Particle Size Distribution (if solid)
    │       ├─→ Scattering Model (EMSC parameters)
    │       └─→ Water Activity (affects H-bonding bands)
    │
    ├─→ Component Set (domain-specific likely analytes)
    │       │
    │       ├─→ Concentration Distributions
    │       └─→ Correlation Structure
    │
    └─→ Target Type (regression/classification)
            │
            ├─→ Nonlinearity (more common in biological)
            └─→ Hidden Factors (reference method noise)
```

**Implementation**:

```python
@dataclass
class NIRSPFNPriorConfig:
    """Configuration for NIRS-PFN training data generation with conditional sampling."""

    # === Top-level: Domain selection ===
    application_domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "agriculture": 0.25,
        "food": 0.20,
        "pharmaceutical": 0.15,
        "petrochemical": 0.10,
        "environmental": 0.10,
        "biomedical": 0.10,
        "general": 0.10,
    })

    # === Conditional: Instrument | Domain ===
    # Each domain has different instrument type distributions
    instrument_given_domain: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "agriculture": {"handheld": 0.4, "benchtop": 0.4, "inline": 0.2},
        "pharmaceutical": {"benchtop": 0.6, "inline": 0.3, "handheld": 0.1},
        "food": {"inline": 0.4, "benchtop": 0.4, "handheld": 0.2},
        "petrochemical": {"inline": 0.5, "benchtop": 0.4, "handheld": 0.1},
        # ... other domains
    })

    # === Conditional: Measurement Mode | Instrument ===
    mode_given_instrument: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "benchtop": {"reflectance": 0.5, "transmittance": 0.3, "transflectance": 0.15, "ATR": 0.05},
        "handheld": {"reflectance": 0.9, "transflectance": 0.1},
        "inline": {"transmittance": 0.5, "reflectance": 0.3, "transflectance": 0.2},
    })

    # === Conditional: Matrix Type | Domain ===
    matrix_given_domain: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "agriculture": {"plant_tissue": 0.4, "grain": 0.3, "powder": 0.2, "liquid": 0.1},
        "pharmaceutical": {"powder": 0.5, "tablet": 0.3, "liquid": 0.2},
        "food": {"emulsion": 0.3, "liquid": 0.3, "tissue": 0.2, "powder": 0.2},
        "petrochemical": {"liquid": 0.9, "other": 0.1},
    })

    # === Shared priors (less domain-dependent) ===
    n_samples_distribution: Tuple[int, int] = (50, 5000)  # Log-uniform
    n_components_distribution: Tuple[int, int] = (2, 15)
    nonlinearity_probability: float = 0.4
    hidden_factors_probability: float = 0.2

    # === Target priors ===
    task_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "regression_single": 0.5,
        "regression_multi": 0.15,
        "classification_binary": 0.2,
        "classification_multi": 0.15,
    })


def sample_from_prior(config: NIRSPFNPriorConfig, rng: np.random.Generator) -> Dict:
    """Sample a complete dataset configuration from the conditional prior."""

    # 1. Sample domain
    domain = rng.choice(
        list(config.application_domain_weights.keys()),
        p=list(config.application_domain_weights.values())
    )

    # 2. Sample instrument | domain
    inst_probs = config.instrument_given_domain.get(domain, {"benchtop": 1.0})
    instrument = rng.choice(list(inst_probs.keys()), p=list(inst_probs.values()))

    # 3. Sample mode | instrument
    mode_probs = config.mode_given_instrument.get(instrument, {"reflectance": 1.0})
    mode = rng.choice(list(mode_probs.keys()), p=list(mode_probs.values()))

    # 4. Sample matrix | domain
    matrix_probs = config.matrix_given_domain.get(domain, {"powder": 1.0})
    matrix = rng.choice(list(matrix_probs.keys()), p=list(matrix_probs.values()))

    # 5. Sample remaining parameters
    n_samples = int(rng.uniform(*np.log10(config.n_samples_distribution)))
    n_samples = 10 ** n_samples  # Log-uniform

    return {
        "domain": domain,
        "instrument": instrument,
        "mode": mode,
        "matrix": matrix,
        "n_samples": int(n_samples),
        "has_nonlinearity": rng.random() < config.nonlinearity_probability,
        "has_hidden_factors": rng.random() < config.hidden_factors_probability,
    }
```

#### 4.4.2 Curriculum Learning Strategy

```python
TRAINING_CURRICULUM = {
    "stage_1_fundamentals": {
        "description": "Pure Beer-Lambert law, no noise",
        "epochs": 10000,
        "complexity": "simple",
        "noise_scale": 0.0,
        "n_components": (2, 5),
        "task": "regression_single",
    },
    "stage_2_noise": {
        "description": "Add realistic noise and baseline",
        "epochs": 20000,
        "complexity": "realistic",
        "noise_scale": (0.001, 0.02),
        "n_components": (2, 8),
        "task": "regression_single",
    },
    "stage_3_instruments": {
        "description": "Variable instruments and measurement modes",
        "epochs": 30000,
        "complexity": ("simple", "realistic", "complex"),
        "instruments": "all",
        "measurement_modes": ["reflectance", "transmittance", "transflectance"],
        "task": ("regression_single", "classification_binary"),
    },
    "stage_4_full_prior": {
        "description": "Full prior distribution",
        "epochs": 100000,
        "prior_config": NIRSPFNPriorConfig(),  # Use full prior
    },
}
```

---

## 5. Implementation Status and Extensions

### 5.0 Implementation Status Truth Table

> 📝 **Key**: ✅ Implemented and tested | ⚠️ Partial/Basic | ❌ Not implemented | 📋 Proposed in this document

| Feature | Status | Module/Class | Tested/Validated |
|---------|--------|--------------|------------------|
| **Spectral Components** ||||
| NIRBand (Voigt profile) | ✅ | `components.NIRBand` | Unit tests |
| SpectralComponent | ✅ | `components.SpectralComponent` | Unit tests |
| ComponentLibrary | ✅ | `components.ComponentLibrary` | Unit tests |
| 31 predefined components | ✅ | `_constants.PREDEFINED_COMPONENTS` | Visual inspection |
| Random component generation | ✅ | `ComponentLibrary.add_random_component()` | Basic tests |
| Overtone/combination constraints | ❌ | — | — |
| Wavenumber-based band placement | ❌ | — | — |
| **Generator Core** ||||
| Beer-Lambert mixing | ✅ | `generator._apply_beer_lambert()` | Unit tests |
| Path length variation | ✅ | `generator._apply_path_length()` | Unit tests |
| Polynomial baseline | ✅ | `generator._generate_baseline()` | Unit tests |
| Wavelength shift/stretch | ✅ | `generator._apply_wavelength_effects()` | Unit tests |
| Multiplicative/additive scatter | ✅ | `generator._apply_scatter()` | Unit tests |
| Heteroscedastic noise | ✅ | `generator._add_noise()` | Unit tests |
| Batch effects | ✅ | `generator.generate(..., include_batch_effects=True)` | Integration tests |
| Artifacts (spikes, saturation) | ✅ | `generator._add_artifacts()` | Visual inspection |
| **Target Generation** ||||
| Linear regression targets | ✅ | `targets.generate_regression_targets()` | Unit tests |
| Classification targets | ✅ | `targets.ClassSeparationConfig` | Unit tests |
| Non-linear targets | ✅ | `targets.NonLinearTargetProcessor` | Integration tests |
| Multi-regime landscapes | ✅ | `targets.NonLinearTargetProcessor` | Integration tests |
| **Proposed Extensions (Section 4)** ||||
| ProceduralComponentGenerator | 📋 | Proposed | — |
| InstrumentArchetype system | 📋 | Proposed | — |
| Measurement mode simulation (R/T/TR/ATR) | 📋 | Proposed | — |
| Temperature effects | 📋 | Proposed | — |
| Particle size (EMSC-style) | 📋 | Proposed | — |
| Kubelka-Munk reflectance | 📋 | Proposed | — |
| Curriculum learning | 📋 | Proposed | — |
| Conditional prior sampling | 📋 | Proposed | — |

### 5.1 Component Library Extensions

| Extension | Current | Target | Effort |
|-----------|---------|--------|--------|
| Predefined components | 31 | 100+ | Medium |
| Procedural component generator | 0 | 1 fully parameterized | High |
| Domain-specific component sets | 0 | 10+ domains | Medium |
| Band interaction modeling | None | Overtones, combinations | Medium |
| Temperature-dependent bands | None | Full implementation | High |

### 5.2 Instrument Simulation Extensions

| Extension | Current | Target | Effort |
|-----------|---------|--------|--------|
| Instrument archetypes | 1 implicit | 20+ explicit | High |
| Measurement modes | 1 (absorption) | 4 (R, T, TR, ATR) | High |
| Detector models | None | 4 detector types | Medium |
| Spectral resolution models | Fixed | Continuous range | Medium |
| Environmental compensation | None | T, humidity, CO₂ | High |

### 5.3 Matrix Effect Extensions

| Extension | Current | Target | Effort |
|-----------|---------|--------|--------|
| Scattering models | MSC-like only | Mie + Kubelka-Munk | High |
| Particle size effects | None | Full simulation | High |
| Moisture effects | None | Water activity model | Medium |
| Matrix-band interactions | None | Hydrogen bonding shifts | High |

### 5.4 Training Infrastructure

| Extension | Current | Target | Effort |
|-----------|---------|--------|--------|
| Batch generation for training | Basic | GPU-accelerated | High |
| Prior config system | None | Full schema + sampling | Medium |
| Curriculum learning support | None | Multi-stage training | Medium |
| Validation datasets | Test fixtures | Benchmark suite | High |

---

## 6. Conclusions and Recommendations

### 6.1 Current State Assessment

The nirs4all synthetic generator provides a **solid foundation** with:
- ✅ Physically-grounded Beer-Lambert model (for transmission geometry)
- ✅ 31 predefined components based on literature band assignments
- ✅ Basic instrumental effects (scatter, baseline, noise)
- ✅ Advanced target complexity (non-linear, multi-regime)
- ✅ Multi-source support
- ✅ Random component generation (without physical constraints)

**Key gaps identified** (to be validated via adversarial validation):
- Measurement modes limited to transmission/absorbance-like model
- No conditional prior structure (domain → instrument dependencies)
- No temperature or explicit particle size effects
- Overtone/combination constraints not enforced

> 📝 **Next step**: Run adversarial validation on benchmark datasets (Section 3.2.2) to quantify the synthetic-real gap before prioritizing extensions.

### 6.2 Critical Path for NIRS-PFN

1. **Must Have (Phase 1)**: Procedural component generation with physical constraints
2. **Must Have (Phase 2)**: Instrument archetype library with measurement modes
3. **Should Have (Phase 3)**: Temperature and particle size effects
4. **Nice to Have (Phase 4)**: Full curriculum learning infrastructure

### 6.3 Effort Estimate

> ⚠️ **Note**: These estimates include implementation time but validation/tuning often takes longer. Budget 50-100% additional time for:
> - Collecting representative real datasets per domain/instrument
> - Tuning prior distributions so synthetic isn't trivially separable
> - Ablation studies to identify which effects improve transfer

| Phase | Scope | Implementation | Validation | Total |
|-------|-------|----------------|------------|-------|
| Phase 1 | Procedural components + domain priors | 3-4 weeks | 2 weeks | 5-6 weeks |
| Phase 2 | Instrument library + measurement modes | 4-5 weeks | 2-3 weeks | 6-8 weeks |
| Phase 3 | Matrix and environmental effects | 3-4 weeks | 2 weeks | 5-6 weeks |
| Phase 4 | Training infrastructure + curriculum | 2-3 weeks | 1-2 weeks | 3-5 weeks |
| **Total** | **Full NIRS-PFN generator** | **12-16 weeks** | **7-9 weeks** | **19-25 weeks** |

### 6.4 Risk Assessment for Generator Adequacy

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Spectral diversity insufficient | Medium | High | Validate on diverse real datasets early |
| Instrument simulation unrealistic | Medium | Medium | Collect real instrument spectra for validation |
| Target relationships too simple | Low | Medium | Current NL support is good, extend if needed |
| Computational cost too high | Medium | Medium | GPU-accelerated generation |
| Domain shift not captured | High | High | Include explicit domain shift scenarios |

---

## Appendix A: Current Predefined Spectral Absorbers

> 📝 **Terminology note**: This list includes both pure chemical compounds (e.g., ethanol) and complex materials (e.g., cotton). For materials, the "molar absorptivity" should be understood as an **effective absorption coefficient** representing the average spectral signature, not a true molar quantity.

| Category | Absorbers | Type |
|----------|-----------|------|
| **Water/Moisture** | water, moisture | Compound |
| **Proteins** | protein, nitrogen_compound, urea, amino_acid | Mixture/Compound |
| **Lipids** | lipid, oil, saturated_fat, unsaturated_fat | Mixture |
| **Carbohydrates** | starch, cellulose, glucose, fructose, sucrose, hemicellulose, lignin | Compound/Polymer |
| **Alcohols** | ethanol, methanol | Compound |
| **Acids** | acetic_acid, citric_acid, lactic_acid | Compound |
| **Pigments** | chlorophyll, carotenoid | Compound class |
| **Pharmaceutical** | caffeine, aspirin, paracetamol | Compound |
| **Hydrocarbons** | aromatic, alkane | Functional group class |
| **Fibers/Materials** | cotton, polyester | Material (effective absorber) |

**References**:
- Band positions are approximate values based on: Workman Jr, J., & Weyer, L. (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*. CRC Press.
- For materials (cotton, polyester), spectral signatures are composites of constituent functional groups.

## Appendix B: NIR Spectral Zones

| Zone (nm) | Zone (cm⁻¹) | Assignment | Components Affected |
|-----------|-------------|------------|---------------------|
| 800-1100 | 9000-12500 | 3rd overtones, electronic | Aromatics, pigments |
| 1100-1300 | 7700-9000 | 2nd overtones C-H | Lipids, hydrocarbons |
| 1400-1550 | 6450-7150 | 1st overtones O-H, N-H | Water, proteins |
| 1650-1800 | 5550-6060 | 1st overtones C-H | Lipids, carbohydrates |
| 1900-2000 | 5000-5260 | Combination O-H | Water, alcohols |
| 2000-2200 | 4545-5000 | Combination N-H | Proteins |
| 2200-2500 | 4000-4545 | Combination C-H | Lipids, carbohydrates |

> 📝 **Conversion**: $\lambda_{nm} = 10^7 / \tilde{\nu}_{cm^{-1}}$
>
> Harmonic relationships (overtones) are linear in wavenumber, not wavelength.
> For example: fundamental C-H stretch at ~3000 cm⁻¹ (3333 nm) → 1st overtone at ~6000 cm⁻¹ (1667 nm)

---

## Appendix C: Scope Clarification

The following items are mentioned in the gap analysis but are **out of scope** for the initial generator extensions:

| Item | Status | Rationale |
|------|--------|-----------|
| Multi-modal fusion (NIRS + RGB + FTIR) | Future work | Different data modalities require separate generators |
| Maillard reaction chemistry | Future work | Complex time-dependent reactions |
| Moisture migration/temporal stability | Future work | Requires time-series modeling |
| Real-world matrix interactions | Simplified | EMSC-style approximation is sufficient for encoder training |

These may become relevant if the spectral encoder approach (document 02) shows limitations related to these factors.

---

*Document prepared for nirs4all NIRS-PFN feasibility study*
*Version 2.0 - January 2026*
*Revised based on external review feedback*
