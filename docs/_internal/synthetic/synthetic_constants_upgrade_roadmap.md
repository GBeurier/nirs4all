# Synthetic Spectra Constants Upgrade Roadmap

## Document Information

| Property | Value |
|----------|-------|
| **Version** | 1.7 |
| **Date** | January 2026 |
| **Status** | Phases 1-7 Complete |
| **Author** | nirs4all Development Team |
| **Scope** | `nirs4all/data/synthetic/_constants.py` and related modules |

---

## Executive Summary

This roadmap addresses critical issues in the synthetic NIRS spectra generator's component library (`_constants.py`) and proposes new functionality for component discovery, aggregation, spectral fitting, and neural network training support. The primary issues identified are:

1. **Wavelength Range**: Components defined only for 1000-2500nm, missing 350-1000nm (Vis-NIR)
2. **Component Quality**: Duplicates, inconsistencies, missing normalization, metadata only in prose
3. **Missing Functionality**: No discovery API, no aggregate components, no fitting tools
4. **Wavelength Flexibility**: Cannot generate spectra matching arbitrary instrument wavelength grids
5. **Product-Level Generation**: No high-level API to generate diverse, realistic product samples for NN training

**Confidence**: High (issues clearly identified from code review and ChatGPT analysis)

---

## Section 1: Problem Analysis

### 1.1 Wavelength Range Issues

**Current State:**
```python
DEFAULT_WAVELENGTH_START: float = 1000.0  # nm
DEFAULT_WAVELENGTH_END: float = 2500.0    # nm
DEFAULT_WAVELENGTH_STEP: float = 2.0      # nm
```

**Problem**: The NIR spectrum is conventionally divided into:

| Region | Wavelength Range | Features |
|--------|-----------------|----------|
| **Vis-NIR (missing)** | 350-700 nm | Electronic transitions, pigments (chlorophyll, carotenoids, anthocyanins) |
| **Short-wave NIR (partial)** | 700-1100 nm | 3rd overtones, Si detector range, electronic tail |
| **NIR** | 1100-2500 nm | 1st/2nd overtones, combination bands |

**Impact**:
- Cannot generate realistic spectra for instruments with Si detectors (400-1100nm)
- Missing electronic absorption bands for biological samples
- Cannot simulate Vis-NIR spectrometers (common in agriculture, food)

**Scientific Reference**:
- Burns & Ciurczak (2007), Ch. 2: NIR conventionally 780-2500nm, but many applications use extended Vis-NIR (400-2500nm)
- Workman & Weyer (2012): Band assignments include electronic transitions in visible region

### 1.2 Component Definition Issues

**Issue A: Docstring vs Reality**
- Docstring claims "111 total" components
- Not verified programmatically
- Risk of documentation drift

**Issue B: Duplicates and Semantic Overlaps**
| Component A | Component B | Issue |
|-------------|-------------|-------|
| `polyester` | `pet` | PET is polyester; redundant definitions |
| `lipid` | `oil` | Conceptually overlapping |
| `saturated_fat` | `palmitic_acid`, `stearic_acid` | Fatty acids are saturated fats |
| `lutein` | `xanthophyll` | Lutein IS a xanthophyll |

**Issue C: Missing Amplitude Normalization**
- Comment states "normalized within component" but not enforced
- No validation that max(amplitude) == 1.0 or sum(amplitude) == 1.0
- Different components have different amplitude scales, making concentration-based mixing inconsistent

**Issue D: Category Metadata Only in Prose**
- Categories exist only in docstring and comments
- No programmatic access to category membership
- Cannot auto-generate documentation or filter by category

**Issue E: Validation Gaps**
- No uniqueness check for component names
- No validation of band parameters (positive sigma/gamma, reasonable centers)
- No cross-reference verification between component names and dict keys

**Confidence**: High (verified by reading _constants.py source code)

---

## Section 2: Wavelength Extension Design

### 2.1 Extended Wavelength Range

**Proposed Change:**
```python
# New defaults supporting Vis-NIR range
DEFAULT_WAVELENGTH_START: float = 350.0   # nm (was 1000.0)
DEFAULT_WAVELENGTH_END: float = 2500.0    # nm (unchanged)
DEFAULT_WAVELENGTH_STEP: float = 2.0      # nm (unchanged)
# Results in 1076 wavelength points (was 751)
```

### 2.2 New Spectral Zones

**Current NIR Zones (wavenumber.py)**:
```python
NIR_ZONES_WAVENUMBER = [
    (9000, 12500, "3rd_overtones"),      # ~800-1111 nm
    (7500, 9000, "2nd_overtones"),       # ~1111-1333 nm
    ...
]
```

**Proposed Extended Zones**:
```python
EXTENDED_SPECTRAL_ZONES = [
    # Visible region (new)
    (14285, 28571, "visible_electronic", "Electronic transitions, pigments"),  # 350-700 nm

    # Red-edge / Short-wave NIR (new)
    (12500, 14285, "red_edge", "Chlorophyll red edge, electronic tail"),       # 700-800 nm

    # Existing NIR zones
    (9000, 12500, "3rd_overtones", "3rd overtones C-H, O-H, N-H"),             # 800-1111 nm
    (7500, 9000, "2nd_overtones", "2nd overtones C-H"),                         # 1111-1333 nm
    (6450, 7150, "1st_overtones_OH_NH", "1st overtones O-H, N-H"),             # 1400-1550 nm
    (5550, 6060, "1st_overtones_CH", "1st overtones C-H"),                      # 1650-1800 nm
    (5000, 5260, "combination_OH", "O-H combination bands"),                    # 1900-2000 nm
    (4545, 5000, "combination_NH", "N-H combination bands"),                    # 2000-2200 nm
    (4000, 4545, "combination_CH", "C-H combination bands"),                    # 2200-2500 nm
]
```

### 2.3 New Components for Visible Region

**Required Additions** (electronic transitions, not vibrational):

| Component | Wavelength (nm) | Type | Chromophore |
|-----------|----------------|------|-------------|
| chlorophyll_a | 430, 662 | Electronic | Porphyrin ring |
| chlorophyll_b | 453, 642 | Electronic | Porphyrin ring |
| beta_carotene | 425, 450, 480 | Electronic | Polyene chain |
| lycopene | 443, 471, 502 | Electronic | Polyene chain |
| anthocyanin_red | 520-540 | Electronic | Flavylium cation |
| anthocyanin_purple | 560-580 | Electronic | Flavylium (pH-dependent) |
| hemoglobin_oxy | 414, 542, 577 | Electronic | Heme group (Soret, Q bands) |
| hemoglobin_deoxy | 430, 555 | Electronic | Heme group |
| melanin | 400-700 (broad) | Electronic | Indole polymer |

**Scientific References**:
- Lichtenthaler & Buschmann (2001): Chlorophylls and Carotenoids: Measurement and Characterization
- Giusti & Wrolstad (2001): Characterization and Measurement of Anthocyanins
- Horecker (1943): Absorption spectra of hemoglobin

### 2.4 Extended Bands for Existing Components

Many existing components need additional bands in the 350-1000nm region:

**Example: Water**
```python
# Current (1000-2500nm only)
"water": SpectralComponent(
    bands=[
        NIRBand(center=1450, ...),  # O-H 1st overtone
        NIRBand(center=1940, ...),  # O-H combination
        NIRBand(center=2500, ...),  # O-H stretch+bend
    ]
)

# Extended (350-2500nm)
"water": SpectralComponent(
    bands=[
        NIRBand(center=730, sigma=20, amplitude=0.05, name="O-H 3rd overtone"),  # NEW
        NIRBand(center=970, sigma=25, amplitude=0.15, name="O-H 2nd overtone"),  # NEW
        NIRBand(center=1450, sigma=25, amplitude=0.80, name="O-H 1st overtone"),
        NIRBand(center=1940, sigma=30, amplitude=1.00, name="O-H combination"),
        NIRBand(center=2500, sigma=50, amplitude=0.30, name="O-H stretch+bend"),
    ]
)
```

**Confidence**: High (well-established spectroscopic literature)

---

## Section 3: Component Quality Improvements

### 3.1 Metadata Schema

**Proposed Enhancement to SpectralComponent:**

```python
@dataclass
class SpectralComponent:
    """Spectral component with full metadata."""
    name: str
    bands: List[NIRBand]

    # New metadata fields
    category: str = ""                    # e.g., "carbohydrates", "proteins"
    subcategory: str = ""                 # e.g., "monosaccharides", "amino_acids"
    synonyms: List[str] = field(default_factory=list)  # e.g., ["vitamin C"] for ascorbic_acid
    formula: str = ""                     # Chemical formula, e.g., "C6H12O6"
    cas_number: str = ""                  # CAS registry number
    references: List[str] = field(default_factory=list)  # Literature citations
    tags: List[str] = field(default_factory=list)  # ["food", "pharma", "agriculture"]

    # Existing fields
    correlation_group: Optional[int] = None

    # New validation hooks
    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate component parameters."""
        # Check band parameters
        for band in self.bands:
            assert band.sigma > 0, f"Band {band.name}: sigma must be positive"
            assert band.gamma >= 0, f"Band {band.name}: gamma must be non-negative"
            assert band.amplitude >= 0, f"Band {band.name}: amplitude must be non-negative"
            assert 200 < band.center < 3000, f"Band {band.name}: center {band.center} outside valid range"
```

### 3.2 Amplitude Normalization Policy

**Decision Required**: Choose one normalization strategy:

| Strategy | Formula | Pros | Cons |
|----------|---------|------|------|
| **Max-normalized** | `max(amplitude) = 1.0` | Simple, intuitive | Sum varies by component |
| **Sum-normalized** | `sum(amplitude) = 1.0` | Consistent total absorption | Max varies |
| **As-is + metadata** | Keep current + add `amplitude_scale` field | Flexible | Requires extra handling |

**Recommendation**: Max-normalized (band with highest absorption = 1.0)

**Implementation**:
```python
def normalize_component_amplitudes(component: SpectralComponent, method: str = "max") -> SpectralComponent:
    """Normalize band amplitudes."""
    amplitudes = [b.amplitude for b in component.bands]

    if method == "max":
        factor = 1.0 / max(amplitudes)
    elif method == "sum":
        factor = 1.0 / sum(amplitudes)
    else:
        raise ValueError(f"Unknown method: {method}")

    normalized_bands = [
        NIRBand(
            center=b.center, sigma=b.sigma, gamma=b.gamma,
            amplitude=b.amplitude * factor, name=b.name
        )
        for b in component.bands
    ]
    return replace(component, bands=normalized_bands)
```

### 3.3 Duplicate Resolution

**Proposed Resolutions**:

| Issue | Resolution | Action |
|-------|------------|--------|
| `polyester` / `pet` | Keep `pet` as primary, make `polyester` an alias | Add `synonyms=["polyester"]` to `pet` |
| `lipid` / `oil` | Keep both but differentiate | `lipid` = biological fats, `oil` = liquid fats/oils |
| `lutein` / `xanthophyll` | Keep both, `xanthophyll` as category | `lutein` is specific, `xanthophyll` is generic |
| Individual fatty acids vs `saturated_fat` | Keep all, use hierarchy | `saturated_fat` = mixture template |

### 3.4 Category Taxonomy

**Proposed Categories** (hierarchical):

```
components/
├── water_related/
│   ├── water
│   └── moisture
├── proteins/
│   ├── general/
│   │   └── protein
│   ├── structural/
│   │   ├── collagen, keratin
│   └── food_proteins/
│       ├── casein, gluten, whey, albumin, zein, gelatin
├── lipids/
│   ├── triglycerides/
│   │   └── lipid, oil, cocoa_butter
│   ├── fatty_acids/
│   │   ├── saturated: palmitic_acid, stearic_acid
│   │   └── unsaturated: oleic_acid, linoleic_acid, linolenic_acid
│   └── sterols/
│       └── cholesterol, phospholipid
├── carbohydrates/
│   ├── monosaccharides/
│   │   └── glucose, fructose, xylose, arabinose, galactose, mannose
│   ├── disaccharides/
│   │   └── sucrose, lactose, maltose, trehalose
│   ├── oligosaccharides/
│   │   └── raffinose
│   └── polysaccharides/
│       └── starch, cellulose, hemicellulose, inulin, dietary_fiber, lignin
├── alcohols/
│   ├── simple/
│   │   └── methanol, ethanol, propanol, butanol, isopropanol
│   └── polyols/
│       └── glycerol, sorbitol, mannitol, xylitol
├── organic_acids/
│   ├── carboxylic/
│   │   └── formic_acid, acetic_acid, propionic_acid, butyric_acid
│   ├── dicarboxylic/
│   │   └── oxalic_acid, succinic_acid, fumaric_acid, malic_acid, tartaric_acid
│   └── hydroxy_acids/
│       └── lactic_acid, citric_acid, ascorbic_acid
├── pigments/
│   ├── chlorophylls/
│   │   └── chlorophyll (→ chlorophyll_a, chlorophyll_b)
│   ├── carotenoids/
│   │   └── carotenoid, beta_carotene, lycopene, lutein, xanthophyll
│   ├── anthocyanins/
│   │   └── anthocyanin
│   └── other/
│       └── melanin, tannins
├── pharmaceuticals/
│   ├── analgesics/
│   │   └── aspirin, paracetamol, ibuprofen, naproxen
│   └── excipients/
│       └── microcrystalline_cellulose
├── polymers/
│   ├── thermoplastics/
│   │   └── polyethylene, polypropylene, polystyrene, pvc, pmma, abs, pet
│   ├── elastomers/
│   │   └── natural_rubber
│   └── fibers/
│       └── nylon, polyester
├── solvents/
│   └── acetone, dmso, ethyl_acetate, toluene, chloroform, hexane
└── minerals/
    ├── carbonates/
    │   └── carbonates, gypsum
    └── silicates/
        └── kaolinite, montmorillonite, illite, talc, silica, goethite
```

**Confidence**: High (standard chemical taxonomy)

---

## Section 4: New API Functions

### 4.1 Component Discovery API

**Proposed Functions:**

```python
# nirs4all/data/synthetic/components.py

def available_components() -> List[str]:
    """Return list of all available component names."""
    return list(get_predefined_components().keys())


def get_component(name: str) -> SpectralComponent:
    """Get a single component by name."""
    components = get_predefined_components()
    if name not in components:
        raise ValueError(f"Unknown component: {name}. Use available_components() to list options.")
    return components[name]


def search_components(
    query: str = None,
    category: str = None,
    tags: List[str] = None,
    wavelength_range: Tuple[float, float] = None,
) -> List[str]:
    """Search components by various criteria."""
    components = get_predefined_components()
    results = []

    for name, comp in components.items():
        # Filter by query (fuzzy match on name/synonyms)
        if query and query.lower() not in name.lower():
            if not any(query.lower() in s.lower() for s in comp.synonyms):
                continue

        # Filter by category
        if category and comp.category != category:
            continue

        # Filter by tags
        if tags and not any(t in comp.tags for t in tags):
            continue

        # Filter by wavelength range (has bands in range)
        if wavelength_range:
            band_centers = [b.center for b in comp.bands]
            if not any(wavelength_range[0] <= c <= wavelength_range[1] for c in band_centers):
                continue

        results.append(name)

    return results


def list_categories() -> Dict[str, List[str]]:
    """Return dictionary of categories to component names."""
    components = get_predefined_components()
    categories = {}
    for name, comp in components.items():
        cat = comp.category or "uncategorized"
        categories.setdefault(cat, []).append(name)
    return categories


def component_info(name: str) -> str:
    """Return formatted information about a component."""
    comp = get_component(name)
    lines = [
        f"Component: {comp.name}",
        f"Category: {comp.category or 'N/A'}",
        f"Formula: {comp.formula or 'N/A'}",
        f"Bands ({len(comp.bands)}):",
    ]
    for band in sorted(comp.bands, key=lambda b: b.center):
        lines.append(f"  - {band.center:.0f} nm: {band.name} (amp={band.amplitude:.2f})")
    if comp.references:
        lines.append("References:")
        for ref in comp.references:
            lines.append(f"  - {ref}")
    return "\n".join(lines)
```

### 4.2 Aggregate Components

**Concept**: Pre-defined mixtures representing common sample types.

```python
# nirs4all/data/synthetic/_aggregates.py

@dataclass
class AggregateComponent:
    """Predefined mixture of components for common sample types."""
    name: str
    components: Dict[str, float]  # component_name -> relative weight
    description: str
    domain: str  # "agriculture", "food", "pharma", etc.
    variability: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # weight ranges


AGGREGATE_COMPONENTS = {
    # Agricultural
    "wheat_grain": AggregateComponent(
        name="wheat_grain",
        components={
            "starch": 0.65,
            "protein": 0.12,
            "moisture": 0.12,
            "lipid": 0.02,
            "cellulose": 0.08,
            "ash": 0.01,  # requires new mineral component
        },
        description="Typical wheat grain composition",
        domain="agriculture",
        variability={
            "protein": (0.08, 0.18),
            "moisture": (0.08, 0.15),
        }
    ),

    "corn_grain": AggregateComponent(
        name="corn_grain",
        components={
            "starch": 0.72,
            "protein": 0.09,
            "moisture": 0.11,
            "lipid": 0.04,
            "cellulose": 0.03,
        },
        description="Typical corn/maize grain composition",
        domain="agriculture",
    ),

    "soybean": AggregateComponent(
        name="soybean",
        components={
            "protein": 0.36,
            "lipid": 0.20,
            "starch": 0.05,
            "cellulose": 0.15,
            "moisture": 0.10,
        },
        description="Typical soybean composition",
        domain="agriculture",
    ),

    # Food
    "milk": AggregateComponent(
        name="milk",
        components={
            "water": 0.87,
            "casein": 0.028,
            "whey": 0.006,
            "lipid": 0.04,
            "lactose": 0.05,
        },
        description="Whole milk composition",
        domain="food",
    ),

    "cheese_cheddar": AggregateComponent(
        name="cheese_cheddar",
        components={
            "casein": 0.25,
            "lipid": 0.33,
            "moisture": 0.36,
            "lactose": 0.01,
        },
        description="Cheddar cheese composition",
        domain="food",
    ),

    "meat_beef": AggregateComponent(
        name="meat_beef",
        components={
            "water": 0.73,
            "protein": 0.22,
            "lipid": 0.03,
            "collagen": 0.01,
        },
        description="Lean beef composition",
        domain="food",
    ),

    # Pharmaceutical
    "tablet_excipient_base": AggregateComponent(
        name="tablet_excipient_base",
        components={
            "microcrystalline_cellulose": 0.40,
            "starch": 0.30,
            "lactose": 0.20,
            "moisture": 0.05,
        },
        description="Common tablet excipient mixture",
        domain="pharmaceutical",
    ),

    # Environmental
    "soil_agricultural": AggregateComponent(
        name="soil_agricultural",
        components={
            "silica": 0.60,
            "kaolinite": 0.15,
            "organic_matter": 0.05,  # requires new component
            "moisture": 0.20,
        },
        description="Typical agricultural soil",
        domain="environmental",
    ),

    # Plant tissue
    "leaf_green": AggregateComponent(
        name="leaf_green",
        components={
            "chlorophyll": 0.005,
            "carotenoid": 0.001,
            "cellulose": 0.20,
            "protein": 0.05,
            "water": 0.70,
        },
        description="Green leaf tissue",
        domain="agriculture",
    ),
}


def get_aggregate(name: str) -> AggregateComponent:
    """Get an aggregate component definition."""
    if name not in AGGREGATE_COMPONENTS:
        available = list(AGGREGATE_COMPONENTS.keys())
        raise ValueError(f"Unknown aggregate: {name}. Available: {available}")
    return AGGREGATE_COMPONENTS[name]


def list_aggregates(domain: str = None) -> List[str]:
    """List available aggregate components."""
    if domain:
        return [n for n, a in AGGREGATE_COMPONENTS.items() if a.domain == domain]
    return list(AGGREGATE_COMPONENTS.keys())


def expand_aggregate(
    name: str,
    variability: bool = False,
    random_state: int = None,
) -> Dict[str, float]:
    """
    Expand an aggregate into component weights.

    Args:
        name: Aggregate component name
        variability: If True, sample from variability ranges
        random_state: Random seed for variability sampling

    Returns:
        Dictionary of component_name -> weight
    """
    agg = get_aggregate(name)

    if not variability:
        return agg.components.copy()

    rng = np.random.default_rng(random_state)
    result = {}

    for comp_name, base_weight in agg.components.items():
        if comp_name in agg.variability:
            low, high = agg.variability[comp_name]
            result[comp_name] = rng.uniform(low, high)
        else:
            result[comp_name] = base_weight

    # Renormalize if needed
    total = sum(result.values())
    if abs(total - 1.0) > 0.01:  # Allow some deviation
        result = {k: v / total for k, v in result.items()}

    return result
```

### 4.3 Spectral Fitting Tools

**Concept**: Fit combinations of known components to match observed spectra.

```python
# nirs4all/data/synthetic/fitter.py (new section)

class ComponentFitter:
    """
    Fit linear combinations of spectral components to observed spectra.

    Solves: spectrum ≈ Σ(c_i * component_i(λ)) + baseline

    Uses non-negative least squares (NNLS) to ensure positive concentrations.
    """

    def __init__(
        self,
        component_names: List[str] = None,
        wavelengths: np.ndarray = None,
        fit_baseline: bool = True,
        baseline_order: int = 2,
    ):
        """
        Initialize fitter.

        Args:
            component_names: Components to fit. If None, uses all available.
            wavelengths: Wavelength grid. If None, uses default (350-2500nm, 2nm step).
            fit_baseline: Include polynomial baseline in fit.
            baseline_order: Polynomial order for baseline.
        """
        self.component_names = component_names or available_components()
        self.wavelengths = wavelengths
        self.fit_baseline = fit_baseline
        self.baseline_order = baseline_order

        # Build design matrix lazily
        self._design_matrix = None
        self._component_library = None

    def _build_design_matrix(self):
        """Build design matrix from components."""
        if self.wavelengths is None:
            self.wavelengths = np.arange(350, 2501, 2)

        components = get_predefined_components()
        self._component_library = ComponentLibrary()

        for name in self.component_names:
            if name in components:
                self._component_library.add_component(components[name])

        # Compute all component spectra: shape (n_components, n_wavelengths)
        component_spectra = self._component_library.compute_all(self.wavelengths)

        # Transpose to (n_wavelengths, n_components) for design matrix
        X = component_spectra.T

        # Add baseline terms if requested
        if self.fit_baseline:
            baseline_terms = []
            for order in range(self.baseline_order + 1):
                # Normalize wavelengths to [0, 1] for numerical stability
                normalized = (self.wavelengths - self.wavelengths[0]) / (self.wavelengths[-1] - self.wavelengths[0])
                baseline_terms.append(normalized ** order)
            X = np.column_stack([X, np.column_stack(baseline_terms)])

        self._design_matrix = X

    def fit(
        self,
        spectrum: np.ndarray,
        method: str = "nnls",
    ) -> "FitResult":
        """
        Fit components to a single spectrum.

        Args:
            spectrum: Observed spectrum, shape (n_wavelengths,)
            method: "nnls" (non-negative least squares) or "lsq" (unconstrained)

        Returns:
            FitResult with concentrations, residuals, and fit quality metrics.
        """
        if self._design_matrix is None:
            self._build_design_matrix()

        X = self._design_matrix
        y = spectrum

        if method == "nnls":
            from scipy.optimize import nnls
            coefficients, residual_norm = nnls(X, y)
        elif method == "lsq":
            coefficients, residual_norm, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residual_norm = np.sqrt(residual_norm[0]) if len(residual_norm) > 0 else 0
        else:
            raise ValueError(f"Unknown method: {method}")

        # Split coefficients into component weights and baseline
        n_components = len(self.component_names)
        component_weights = coefficients[:n_components]
        baseline_coeffs = coefficients[n_components:] if self.fit_baseline else None

        # Compute fitted spectrum
        fitted = X @ coefficients
        residuals = y - fitted

        # Compute R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return FitResult(
            component_names=self.component_names,
            concentrations=component_weights,
            baseline_coefficients=baseline_coeffs,
            fitted_spectrum=fitted,
            residuals=residuals,
            r_squared=r_squared,
            rmse=np.sqrt(np.mean(residuals ** 2)),
        )

    def fit_batch(
        self,
        spectra: np.ndarray,
        n_jobs: int = -1,
    ) -> List["FitResult"]:
        """
        Fit components to multiple spectra.

        Args:
            spectra: Observed spectra, shape (n_samples, n_wavelengths)
            n_jobs: Number of parallel jobs (-1 = all cores)

        Returns:
            List of FitResult objects.
        """
        from joblib import Parallel, delayed

        if self._design_matrix is None:
            self._build_design_matrix()

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.fit)(spectrum) for spectrum in spectra
        )
        return results

    def suggest_components(
        self,
        spectrum: np.ndarray,
        top_n: int = 5,
        threshold: float = 0.01,
    ) -> List[Tuple[str, float]]:
        """
        Suggest which components are likely present in a spectrum.

        Returns list of (component_name, estimated_concentration) tuples,
        sorted by concentration descending.
        """
        result = self.fit(spectrum)

        suggestions = [
            (name, conc)
            for name, conc in zip(result.component_names, result.concentrations)
            if conc > threshold
        ]

        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_n]


@dataclass
class FitResult:
    """Result of fitting components to a spectrum."""
    component_names: List[str]
    concentrations: np.ndarray
    baseline_coefficients: Optional[np.ndarray]
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    r_squared: float
    rmse: float

    def to_dict(self) -> Dict[str, float]:
        """Return concentrations as dictionary."""
        return dict(zip(self.component_names, self.concentrations))

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Fit Quality: R² = {self.r_squared:.4f}, RMSE = {self.rmse:.6f}",
            "Component Concentrations (top 10):",
        ]
        sorted_pairs = sorted(
            zip(self.component_names, self.concentrations),
            key=lambda x: x[1],
            reverse=True
        )
        for name, conc in sorted_pairs[:10]:
            if conc > 0.001:
                lines.append(f"  {name}: {conc:.4f}")
        return "\n".join(lines)
```

**Confidence**: High (standard spectral unmixing approach, well-established mathematically)

---

### 4.4 Custom Wavelength Support

**Problem**: Current generator only supports regularly-spaced wavelength grids. Real instruments have:
- Irregular wavelength spacing (especially FT-NIR, filter-based instruments)
- Instrument-specific wavelength calibrations
- Different resolutions in different spectral regions

**Use Cases**:
1. Generate synthetic data matching exact wavelengths of a real instrument
2. Combine synthetic and real data for transfer learning
3. Simulate specific commercial instruments (MicroNIR, SCiO, etc.)

**Proposed API**:

```python
# nirs4all/data/synthetic/generator.py (modification)

class SyntheticNIRSGenerator:
    def __init__(
        self,
        # Existing regular grid parameters (backward compatible)
        wavelength_start: float = 350.0,
        wavelength_end: float = 2500.0,
        wavelength_step: float = 2.0,

        # NEW: Custom wavelength support
        wavelengths: Optional[np.ndarray] = None,  # Overrides start/end/step if provided

        # Existing parameters...
        component_library: Optional[ComponentLibrary] = None,
        complexity: str = "simple",
        instrument: Optional[str] = None,
        ...
    ):
        """
        Initialize generator.

        Args:
            wavelengths: Custom wavelength array (nm). If provided, overrides
                         wavelength_start/end/step. Enables matching exact
                         instrument wavelength grids.

        Example:
            # Match a real instrument's wavelength grid
            real_wavelengths = np.array([908, 912, 916, ..., 1676])  # From instrument
            gen = SyntheticNIRSGenerator(wavelengths=real_wavelengths)
            X_synth, C, E = gen.generate(n_samples=1000)
            # X_synth.shape[1] matches len(real_wavelengths)
        """
        if wavelengths is not None:
            self.wavelengths = np.asarray(wavelengths)
            self._custom_wavelengths = True
        else:
            self.wavelengths = np.arange(wavelength_start, wavelength_end + wavelength_step, wavelength_step)
            self._custom_wavelengths = False


# Top-level convenience function
def generate(
    n_samples: int = 1000,
    *,
    wavelengths: Optional[np.ndarray] = None,  # NEW parameter
    wavelength_start: float = 350.0,
    wavelength_end: float = 2500.0,
    wavelength_step: float = 2.0,
    ...
) -> SpectroDataset:
    """
    Generate synthetic NIRS dataset.

    Args:
        wavelengths: Custom wavelength array. If provided, generates spectra
                     at exactly these wavelengths (useful for matching real
                     instrument grids).

    Example:
        # Generate data matching MicroNIR wavelengths
        micronir_wl = np.linspace(908, 1676, 125)  # 125 channels
        dataset = nirs4all.generate(n_samples=500, wavelengths=micronir_wl)
    """
```

**Instrument Integration**:

```python
# nirs4all/data/synthetic/instruments.py (addition)

# Predefined wavelength grids for common instruments
INSTRUMENT_WAVELENGTHS = {
    "micronir_onsite": np.linspace(908, 1676, 125),
    "scio": np.linspace(740, 1070, 331),
    "neospectra_micro": np.linspace(1350, 2500, 228),
    "foss_xds": np.arange(400, 2498, 2),
    "bruker_mpa": np.arange(800, 2778, 4),  # FT-NIR
    "asd_fieldspec": np.arange(350, 2500, 1),  # Full range, high resolution
}


def get_instrument_wavelengths(instrument: str) -> np.ndarray:
    """Get wavelength grid for a known instrument."""
    if instrument not in INSTRUMENT_WAVELENGTHS:
        available = list(INSTRUMENT_WAVELENGTHS.keys())
        raise ValueError(f"Unknown instrument: {instrument}. Available: {available}")
    return INSTRUMENT_WAVELENGTHS[instrument].copy()


# Integration with generator
class SyntheticNIRSGenerator:
    def __init__(
        self,
        ...
        instrument: Optional[str] = None,
        match_instrument_wavelengths: bool = True,  # NEW
        ...
    ):
        """
        If instrument is specified and match_instrument_wavelengths=True,
        automatically uses that instrument's wavelength grid.
        """
        if instrument and match_instrument_wavelengths:
            self.wavelengths = get_instrument_wavelengths(instrument)
```

**Builder API Extension**:

```python
# nirs4all/data/synthetic/builder.py (addition)

class SyntheticDatasetBuilder:
    def with_wavelengths(
        self,
        wavelengths: Optional[np.ndarray] = None,
        from_instrument: Optional[str] = None,
        from_dataset: Optional[SpectroDataset] = None,
    ) -> 'SyntheticDatasetBuilder':
        """
        Configure wavelength grid.

        Args:
            wavelengths: Explicit wavelength array
            from_instrument: Use wavelengths from known instrument
            from_dataset: Copy wavelengths from existing dataset

        Example:
            # Match existing dataset
            builder.with_wavelengths(from_dataset=real_dataset)

            # Match known instrument
            builder.with_wavelengths(from_instrument="micronir_onsite")

            # Explicit wavelengths
            builder.with_wavelengths(wavelengths=my_wavelengths)
        """
        if wavelengths is not None:
            self._wavelengths = np.asarray(wavelengths)
        elif from_instrument is not None:
            self._wavelengths = get_instrument_wavelengths(from_instrument)
        elif from_dataset is not None:
            self._wavelengths = from_dataset.wavelengths.copy()
        return self
```

**Impact on Component Evaluation**:

When using custom wavelengths, component spectra are computed by interpolation:

```python
def _compute_component_at_wavelengths(
    component: SpectralComponent,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Compute component spectrum at arbitrary wavelengths.

    Uses interpolation from a high-resolution internal grid.
    """
    # Compute on fine grid first (0.5nm resolution)
    fine_grid = np.arange(
        max(200, wavelengths.min() - 50),
        min(3000, wavelengths.max() + 50),
        0.5
    )
    fine_spectrum = component.compute(fine_grid)

    # Interpolate to requested wavelengths
    from scipy.interpolate import interp1d
    interpolator = interp1d(
        fine_grid, fine_spectrum,
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )
    return interpolator(wavelengths)
```

**Confidence**: High (straightforward interpolation, common need for real-world applications)

---

### 4.5 Product Generator for Neural Network Training

**Goal**: Generate diverse, realistic product samples with controlled variability for training neural networks. The generator should produce samples that span the realistic variation space of real products.

**Design Philosophy**:
- **Plasticity**: High variability to train robust models
- **Realism**: Compositions within realistic bounds for each product type
- **Control**: Specify which properties to vary and by how much
- **Correlation Structure**: Maintain realistic correlations between components

**Core Concept: ProductTemplate**

```python
# nirs4all/data/synthetic/products.py (new file)

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import numpy as np


class VariationType(Enum):
    """Types of variability for component concentrations."""
    FIXED = "fixed"              # No variation
    UNIFORM = "uniform"          # Uniform within range
    NORMAL = "normal"            # Gaussian around mean
    LOGNORMAL = "lognormal"      # Log-normal (always positive)
    CORRELATED = "correlated"    # Varies with another component
    COMPUTED = "computed"        # Computed from other components


@dataclass
class ComponentVariation:
    """Specification for how a component varies."""
    component: str                              # Component name
    variation_type: VariationType               # How to vary
    mean: float = 0.0                           # Central value (fraction)
    std: float = 0.0                            # Standard deviation
    range: Tuple[float, float] = (0.0, 1.0)     # Min/max bounds
    correlation_with: Optional[str] = None      # For correlated variation
    correlation_strength: float = 0.0           # Correlation coefficient
    compute_fn: Optional[Callable] = None       # For computed variation


@dataclass
class ProductTemplate:
    """
    Template for generating diverse product samples.

    Defines the composition space of a product type with variability ranges
    for each component, enabling generation of diverse training samples.
    """
    name: str
    description: str
    domain: str                                 # "food", "agriculture", "pharma", etc.
    category: str                               # "dairy", "grain", "meat", etc.

    # Base composition (mean values, must sum to ~1.0)
    base_composition: Dict[str, float]

    # Variability specification for each component
    variations: Dict[str, ComponentVariation] = field(default_factory=dict)

    # Components that should covary (correlation groups)
    correlation_groups: List[List[str]] = field(default_factory=list)

    # Constraints
    min_total: float = 0.95                     # Minimum sum of components
    max_total: float = 1.05                     # Maximum sum of components
    renormalize: bool = True                    # Force sum to 1.0

    # Spectral effects
    include_moisture_effects: bool = True
    include_temperature_variation: bool = False
    temperature_range: Tuple[float, float] = (20, 25)

    # Tags for filtering
    tags: List[str] = field(default_factory=list)


# Predefined product templates
PRODUCT_TEMPLATES = {
    # ================================================================
    # DAIRY PRODUCTS
    # ================================================================
    "milk_variable_fat": ProductTemplate(
        name="milk_variable_fat",
        description="Milk with variable fat content (skim to whole)",
        domain="food",
        category="dairy",
        base_composition={
            "water": 0.87,
            "casein": 0.026,
            "whey": 0.006,
            "lipid": 0.035,
            "lactose": 0.048,
        },
        variations={
            "lipid": ComponentVariation(
                component="lipid",
                variation_type=VariationType.UNIFORM,
                range=(0.001, 0.06),  # Skim (0.1%) to whole (6%)
            ),
            "water": ComponentVariation(
                component="water",
                variation_type=VariationType.COMPUTED,
                compute_fn=lambda c: 1.0 - sum(v for k, v in c.items() if k != "water"),
            ),
        },
        tags=["dairy", "liquid", "fat_variable"],
    ),

    "cheese_variable_moisture": ProductTemplate(
        name="cheese_variable_moisture",
        description="Cheese with variable moisture (fresh to aged)",
        domain="food",
        category="dairy",
        base_composition={
            "casein": 0.25,
            "lipid": 0.33,
            "moisture": 0.36,
            "lactose": 0.02,
        },
        variations={
            "moisture": ComponentVariation(
                component="moisture",
                variation_type=VariationType.UNIFORM,
                range=(0.30, 0.55),  # Aged to fresh
            ),
            "casein": ComponentVariation(
                component="casein",
                variation_type=VariationType.CORRELATED,
                correlation_with="moisture",
                correlation_strength=-0.7,  # More moisture = less protein
                range=(0.18, 0.35),
            ),
        },
        tags=["dairy", "solid", "moisture_variable"],
    ),

    # ================================================================
    # MEAT PRODUCTS
    # ================================================================
    "meat_variable_fat": ProductTemplate(
        name="meat_variable_fat",
        description="Meat with variable fat content (lean to marbled)",
        domain="food",
        category="meat",
        base_composition={
            "water": 0.70,
            "protein": 0.20,
            "lipid": 0.08,
            "collagen": 0.02,
        },
        variations={
            "lipid": ComponentVariation(
                component="lipid",
                variation_type=VariationType.LOGNORMAL,
                mean=0.08,
                std=0.05,
                range=(0.01, 0.35),  # Very lean to heavily marbled
            ),
            "protein": ComponentVariation(
                component="protein",
                variation_type=VariationType.CORRELATED,
                correlation_with="lipid",
                correlation_strength=-0.6,
                range=(0.15, 0.24),
            ),
            "water": ComponentVariation(
                component="water",
                variation_type=VariationType.CORRELATED,
                correlation_with="lipid",
                correlation_strength=-0.8,
                range=(0.50, 0.78),
            ),
        },
        tags=["meat", "protein_source", "fat_variable"],
    ),

    # ================================================================
    # GRAIN PRODUCTS
    # ================================================================
    "wheat_variable_protein": ProductTemplate(
        name="wheat_variable_protein",
        description="Wheat grain with variable protein (feed to bread quality)",
        domain="agriculture",
        category="grain",
        base_composition={
            "starch": 0.65,
            "protein": 0.12,
            "moisture": 0.12,
            "lipid": 0.02,
            "cellulose": 0.08,
        },
        variations={
            "protein": ComponentVariation(
                component="protein",
                variation_type=VariationType.NORMAL,
                mean=0.12,
                std=0.025,
                range=(0.08, 0.18),  # Feed wheat to high-protein bread wheat
            ),
            "starch": ComponentVariation(
                component="starch",
                variation_type=VariationType.CORRELATED,
                correlation_with="protein",
                correlation_strength=-0.85,
                range=(0.58, 0.72),
            ),
            "moisture": ComponentVariation(
                component="moisture",
                variation_type=VariationType.NORMAL,
                mean=0.12,
                std=0.02,
                range=(0.08, 0.16),
            ),
        },
        correlation_groups=[["protein", "starch"]],
        tags=["grain", "agriculture", "protein_variable"],
    ),

    # ================================================================
    # PHARMACEUTICAL
    # ================================================================
    "tablet_variable_api": ProductTemplate(
        name="tablet_variable_api",
        description="Pharmaceutical tablet with variable API content",
        domain="pharmaceutical",
        category="solid_dosage",
        base_composition={
            "microcrystalline_cellulose": 0.40,
            "starch": 0.25,
            "lactose": 0.20,
            "paracetamol": 0.10,  # API
            "moisture": 0.05,
        },
        variations={
            "paracetamol": ComponentVariation(
                component="paracetamol",
                variation_type=VariationType.NORMAL,
                mean=0.10,
                std=0.02,
                range=(0.05, 0.20),
            ),
            "microcrystalline_cellulose": ComponentVariation(
                component="microcrystalline_cellulose",
                variation_type=VariationType.COMPUTED,
                compute_fn=lambda c: max(0.30, 0.95 - sum(
                    v for k, v in c.items()
                    if k not in ["microcrystalline_cellulose", "water"]
                )),
            ),
        },
        tags=["pharma", "tablet", "api_variable"],
    ),

    # ================================================================
    # SPECIAL: HIGH VARIABILITY FOR NN TRAINING
    # ================================================================
    "food_cholesterol_variable": ProductTemplate(
        name="food_cholesterol_variable",
        description="Food product with variable cholesterol for NN training",
        domain="food",
        category="mixed",
        base_composition={
            "water": 0.60,
            "protein": 0.15,
            "lipid": 0.15,
            "cholesterol": 0.005,  # 500 mg/100g base
            "starch": 0.08,
        },
        variations={
            "cholesterol": ComponentVariation(
                component="cholesterol",
                variation_type=VariationType.LOGNORMAL,
                mean=0.005,
                std=0.003,
                range=(0.0001, 0.02),  # 10 mg to 2000 mg per 100g
            ),
            "lipid": ComponentVariation(
                component="lipid",
                variation_type=VariationType.CORRELATED,
                correlation_with="cholesterol",
                correlation_strength=0.6,  # Higher fat often = higher cholesterol
                range=(0.02, 0.40),
            ),
            "protein": ComponentVariation(
                component="protein",
                variation_type=VariationType.UNIFORM,
                range=(0.05, 0.30),
            ),
            "starch": ComponentVariation(
                component="starch",
                variation_type=VariationType.UNIFORM,
                range=(0.0, 0.40),
            ),
        },
        include_moisture_effects=True,
        tags=["food", "cholesterol", "nn_training", "high_variability"],
    ),
}
```

**ProductGenerator Class**:

```python
class ProductGenerator:
    """
    Generate diverse product samples from templates.

    Designed for neural network training where diverse, realistic
    samples are needed to learn robust representations.
    """

    def __init__(
        self,
        template: Union[str, ProductTemplate],
        wavelengths: Optional[np.ndarray] = None,
        complexity: str = "realistic",
        random_state: Optional[int] = None,
    ):
        """
        Initialize product generator.

        Args:
            template: Product template name or ProductTemplate object
            wavelengths: Custom wavelength grid (optional)
            complexity: Spectral complexity ("simple", "realistic", "complex")
            random_state: Random seed for reproducibility
        """
        if isinstance(template, str):
            if template not in PRODUCT_TEMPLATES:
                available = list(PRODUCT_TEMPLATES.keys())
                raise ValueError(f"Unknown template: {template}. Available: {available}")
            self.template = PRODUCT_TEMPLATES[template]
        else:
            self.template = template

        self.wavelengths = wavelengths
        self.complexity = complexity
        self.rng = np.random.default_rng(random_state)

        # Build underlying NIRS generator
        self._build_generator()

    def generate(
        self,
        n_samples: int,
        return_compositions: bool = False,
        return_metadata: bool = False,
    ) -> Union[SpectroDataset, Tuple]:
        """
        Generate diverse product samples.

        Args:
            n_samples: Number of samples to generate
            return_compositions: Also return component concentrations
            return_metadata: Also return sample metadata

        Returns:
            SpectroDataset or tuple with additional arrays
        """
        # Sample compositions according to template variability
        compositions = self._sample_compositions(n_samples)

        # Generate spectra from compositions
        spectra = self._generate_spectra(compositions)

        # Create dataset
        dataset = self._create_dataset(spectra, compositions)

        if return_compositions and return_metadata:
            return dataset, compositions, self._generate_metadata(n_samples)
        elif return_compositions:
            return dataset, compositions
        elif return_metadata:
            return dataset, self._generate_metadata(n_samples)
        return dataset

    def _sample_compositions(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Sample component concentrations according to template."""
        compositions = {}
        template = self.template

        # First pass: sample independent and uniform variations
        for comp_name, base_value in template.base_composition.items():
            if comp_name in template.variations:
                var = template.variations[comp_name]

                if var.variation_type == VariationType.FIXED:
                    compositions[comp_name] = np.full(n_samples, base_value)

                elif var.variation_type == VariationType.UNIFORM:
                    compositions[comp_name] = self.rng.uniform(
                        var.range[0], var.range[1], n_samples
                    )

                elif var.variation_type == VariationType.NORMAL:
                    samples = self.rng.normal(var.mean, var.std, n_samples)
                    compositions[comp_name] = np.clip(samples, var.range[0], var.range[1])

                elif var.variation_type == VariationType.LOGNORMAL:
                    # Convert to log-space parameters
                    log_mean = np.log(var.mean**2 / np.sqrt(var.mean**2 + var.std**2))
                    log_std = np.sqrt(np.log(1 + var.std**2 / var.mean**2))
                    samples = self.rng.lognormal(log_mean, log_std, n_samples)
                    compositions[comp_name] = np.clip(samples, var.range[0], var.range[1])

            else:
                # No variation specified, use base value with small noise
                noise = self.rng.normal(0, base_value * 0.02, n_samples)
                compositions[comp_name] = np.clip(base_value + noise, 0, 1)

        # Second pass: correlated variations
        for comp_name, var in template.variations.items():
            if var.variation_type == VariationType.CORRELATED:
                if var.correlation_with not in compositions:
                    raise ValueError(f"Correlation target {var.correlation_with} not found")

                target = compositions[var.correlation_with]
                # Generate correlated samples
                base = template.base_composition[comp_name]
                target_base = template.base_composition[var.correlation_with]

                # Standardize target
                target_std = (target - target.mean()) / (target.std() + 1e-8)

                # Generate correlated noise
                corr = var.correlation_strength
                independent = self.rng.normal(0, 1, n_samples)
                correlated = corr * target_std + np.sqrt(1 - corr**2) * independent

                # Scale to desired range
                mid = (var.range[0] + var.range[1]) / 2
                spread = (var.range[1] - var.range[0]) / 4
                samples = mid + spread * correlated
                compositions[comp_name] = np.clip(samples, var.range[0], var.range[1])

        # Third pass: computed variations
        for comp_name, var in template.variations.items():
            if var.variation_type == VariationType.COMPUTED:
                for i in range(n_samples):
                    sample_comp = {k: v[i] for k, v in compositions.items()}
                    compositions[comp_name][i] = var.compute_fn(sample_comp)

        # Renormalize if requested
        if template.renormalize:
            totals = sum(compositions.values())
            for comp_name in compositions:
                compositions[comp_name] = compositions[comp_name] / totals

        return compositions

    def generate_dataset_for_target(
        self,
        n_samples: int,
        target_component: str,
        target_range: Optional[Tuple[float, float]] = None,
    ) -> SpectroDataset:
        """
        Generate dataset where target is a specific component concentration.

        Args:
            n_samples: Number of samples
            target_component: Component to use as regression target
            target_range: Optional range to scale target to (e.g., (0, 100) for percentage)

        Returns:
            SpectroDataset with y = component concentration
        """
        dataset, compositions = self.generate(n_samples, return_compositions=True)

        if target_component not in compositions:
            raise ValueError(f"Component {target_component} not in template")

        y = compositions[target_component]

        if target_range:
            # Scale to target range (e.g., fraction to percentage)
            y = y * (target_range[1] - target_range[0]) + target_range[0]

        # Update dataset with correct target
        dataset._y = y.reshape(-1, 1)
        dataset._target_names = [target_component]

        return dataset


# Convenience functions
def list_product_templates(
    domain: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[str]:
    """List available product templates."""
    results = []
    for name, template in PRODUCT_TEMPLATES.items():
        if domain and template.domain != domain:
            continue
        if category and template.category != category:
            continue
        if tags and not any(t in template.tags for t in tags):
            continue
        results.append(name)
    return results


def get_product_template(name: str) -> ProductTemplate:
    """Get a product template by name."""
    if name not in PRODUCT_TEMPLATES:
        available = list(PRODUCT_TEMPLATES.keys())
        raise ValueError(f"Unknown template: {name}. Available: {available}")
    return PRODUCT_TEMPLATES[name]


def generate_product_samples(
    template: str,
    n_samples: int,
    target: Optional[str] = None,
    target_range: Tuple[float, float] = (0, 100),
    wavelengths: Optional[np.ndarray] = None,
    complexity: str = "realistic",
    random_state: Optional[int] = None,
) -> SpectroDataset:
    """
    Convenience function to generate product samples.

    Example:
        # Generate food samples with cholesterol as target
        dataset = generate_product_samples(
            template="food_cholesterol_variable",
            n_samples=10000,
            target="cholesterol",
            target_range=(0, 2000),  # mg/100g
            random_state=42
        )

        # Train neural network
        X_train, y_train = dataset.x({"partition": "train"}), dataset.y({"partition": "train"})
    """
    gen = ProductGenerator(
        template=template,
        wavelengths=wavelengths,
        complexity=complexity,
        random_state=random_state,
    )

    if target:
        return gen.generate_dataset_for_target(n_samples, target, target_range)
    else:
        return gen.generate(n_samples)
```

**High-Level Category Generator**:

```python
class CategoryGenerator:
    """
    Generate diverse samples from a product category.

    Combines multiple templates from the same category to create
    maximum diversity for neural network training.
    """

    def __init__(
        self,
        category: str,
        domain: Optional[str] = None,
        wavelengths: Optional[np.ndarray] = None,
        complexity: str = "realistic",
        random_state: Optional[int] = None,
    ):
        """
        Initialize category generator.

        Args:
            category: Product category ("dairy", "meat", "grain", etc.)
            domain: Optional domain filter ("food", "agriculture", etc.)
            wavelengths: Custom wavelength grid
            complexity: Spectral complexity level
            random_state: Random seed
        """
        self.category = category
        self.domain = domain
        self.wavelengths = wavelengths
        self.complexity = complexity
        self.rng = np.random.default_rng(random_state)

        # Find all templates for this category
        self.templates = self._find_templates()
        if not self.templates:
            raise ValueError(f"No templates found for category: {category}")

    def _find_templates(self) -> List[ProductTemplate]:
        """Find all templates matching category/domain."""
        templates = []
        for name, template in PRODUCT_TEMPLATES.items():
            if template.category == self.category:
                if self.domain is None or template.domain == self.domain:
                    templates.append(template)
        return templates

    def generate(
        self,
        n_samples: int,
        samples_per_template: Optional[int] = None,
        return_template_labels: bool = False,
    ) -> Union[SpectroDataset, Tuple[SpectroDataset, np.ndarray]]:
        """
        Generate diverse samples from all templates in category.

        Args:
            n_samples: Total samples to generate
            samples_per_template: If set, generate this many per template
            return_template_labels: Return array indicating source template

        Returns:
            SpectroDataset (and optionally template labels)
        """
        if samples_per_template:
            n_per = samples_per_template
        else:
            n_per = n_samples // len(self.templates)

        all_spectra = []
        all_compositions = []
        template_labels = []

        for i, template in enumerate(self.templates):
            gen = ProductGenerator(
                template=template,
                wavelengths=self.wavelengths,
                complexity=self.complexity,
                random_state=self.rng.integers(0, 2**31),
            )
            dataset, comps = gen.generate(n_per, return_compositions=True)

            all_spectra.append(dataset.x())
            all_compositions.append(comps)
            template_labels.extend([i] * n_per)

        # Combine all samples
        X = np.vstack(all_spectra)
        template_labels = np.array(template_labels)

        # Shuffle
        indices = self.rng.permutation(len(X))
        X = X[indices]
        template_labels = template_labels[indices]

        # Create combined dataset
        dataset = self._create_combined_dataset(X)

        if return_template_labels:
            return dataset, template_labels
        return dataset
```

**Integration with nirs4all.generate API**:

```python
# nirs4all/api/generate.py (additions)

def product(
    template: str,
    n_samples: int = 1000,
    target: Optional[str] = None,
    target_range: Tuple[float, float] = (0, 100),
    wavelengths: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> SpectroDataset:
    """
    Generate synthetic product samples from a template.

    Example:
        # Generate wheat samples with protein as target
        dataset = nirs4all.generate.product(
            template="wheat_variable_protein",
            n_samples=5000,
            target="protein",
            target_range=(8, 18),  # Percentage
        )
    """
    return generate_product_samples(
        template=template,
        n_samples=n_samples,
        target=target,
        target_range=target_range,
        wavelengths=wavelengths,
        random_state=random_state,
        **kwargs
    )


def category(
    category: str,
    n_samples: int = 1000,
    domain: Optional[str] = None,
    wavelengths: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> SpectroDataset:
    """
    Generate diverse samples from a product category.

    Combines all templates in the category for maximum diversity.

    Example:
        # Generate diverse dairy samples for NN training
        dataset = nirs4all.generate.category(
            category="dairy",
            n_samples=10000,
            random_state=42
        )
    """
    gen = CategoryGenerator(
        category=category,
        domain=domain,
        wavelengths=wavelengths,
        random_state=random_state,
        **kwargs
    )
    return gen.generate(n_samples)
```

**Usage Examples for NN Training**:

```python
import nirs4all
import numpy as np

# Example 1: Generate food samples with variable cholesterol
dataset = nirs4all.generate.product(
    template="food_cholesterol_variable",
    n_samples=10000,
    target="cholesterol",
    target_range=(0, 2000),  # mg per 100g
    random_state=42
)

X_train = dataset.x({"partition": "train"})
y_train = dataset.y({"partition": "train"})
# Train neural network: model.fit(X_train, y_train)


# Example 2: Generate diverse dairy products
dataset = nirs4all.generate.category(
    category="dairy",
    n_samples=20000,
    random_state=42
)


# Example 3: Match specific instrument wavelengths
micronir_wavelengths = np.linspace(908, 1676, 125)

dataset = nirs4all.generate.product(
    template="meat_variable_fat",
    n_samples=5000,
    target="lipid",
    target_range=(0, 40),  # Percentage fat
    wavelengths=micronir_wavelengths,
    random_state=42
)


# Example 4: Create custom high-variability template
from nirs4all.data.synthetic.products import (
    ProductTemplate, ComponentVariation, VariationType, ProductGenerator
)

custom_template = ProductTemplate(
    name="custom_mixed_food",
    description="High variability food for NN training",
    domain="food",
    category="mixed",
    base_composition={
        "water": 0.50,
        "protein": 0.15,
        "lipid": 0.20,
        "starch": 0.10,
        "cellulose": 0.05,
    },
    variations={
        "protein": ComponentVariation(
            component="protein",
            variation_type=VariationType.UNIFORM,
            range=(0.02, 0.40),  # 2% to 40% protein
        ),
        "lipid": ComponentVariation(
            component="lipid",
            variation_type=VariationType.LOGNORMAL,
            mean=0.15,
            std=0.10,
            range=(0.01, 0.50),
        ),
        "starch": ComponentVariation(
            component="starch",
            variation_type=VariationType.UNIFORM,
            range=(0.0, 0.60),
        ),
    },
    tags=["custom", "nn_training", "high_variability"],
)

gen = ProductGenerator(template=custom_template, random_state=42)
dataset = gen.generate_dataset_for_target(
    n_samples=50000,
    target_component="protein",
    target_range=(0, 100)
)
```

**Confidence**: High (well-defined architecture, covers key use cases for NN training)

---

## Section 5: Validation Framework

### 5.1 Component Validation

```python
# nirs4all/data/synthetic/validation.py (addition)

def validate_predefined_components() -> List[str]:
    """
    Validate all predefined components.

    Returns:
        List of validation warnings/errors (empty if all valid).
    """
    issues = []
    components = get_predefined_components()

    # Check total count matches docstring
    expected_count = 121
    actual_count = len(components)
    if actual_count != expected_count:
        issues.append(f"Component count mismatch: expected {expected_count}, got {actual_count}")

    # Check for uniqueness
    names = list(components.keys())
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        issues.append(f"Duplicate component names: {duplicates}")

    # Validate each component
    for name, comp in components.items():
        # Check name matches key
        if comp.name != name:
            issues.append(f"Component '{name}': name mismatch (comp.name='{comp.name}')")

        # Check bands
        if not comp.bands:
            issues.append(f"Component '{name}': no bands defined")
            continue

        for band in comp.bands:
            # Positive parameters
            if band.sigma <= 0:
                issues.append(f"Component '{name}', band '{band.name}': sigma must be positive")
            if band.gamma < 0:
                issues.append(f"Component '{name}', band '{band.name}': gamma must be non-negative")
            if band.amplitude < 0:
                issues.append(f"Component '{name}', band '{band.name}': amplitude must be non-negative")

            # Reasonable wavelength range
            if not (200 < band.center < 3000):
                issues.append(f"Component '{name}', band '{band.name}': center {band.center} outside 200-3000nm")

        # Check amplitude normalization (max should be ~1.0)
        amplitudes = [b.amplitude for b in comp.bands]
        max_amp = max(amplitudes)
        if abs(max_amp - 1.0) > 0.1:
            issues.append(f"Component '{name}': max amplitude {max_amp:.2f} not normalized to 1.0")

    return issues


def validate_component_coverage(
    wavelength_range: Tuple[float, float] = (350, 2500),
) -> Dict[str, List[str]]:
    """
    Check which components have bands in the given wavelength range.

    Returns:
        Dictionary with 'covered' and 'not_covered' component lists.
    """
    components = get_predefined_components()
    covered = []
    not_covered = []

    for name, comp in components.items():
        band_centers = [b.center for b in comp.bands]
        has_coverage = any(
            wavelength_range[0] <= c <= wavelength_range[1]
            for c in band_centers
        )
        if has_coverage:
            covered.append(name)
        else:
            not_covered.append(name)

    return {"covered": covered, "not_covered": not_covered}
```

### 5.2 Integration Tests

```python
# tests/unit/data/synthetic/test_constants_validation.py

class TestComponentValidation:
    """Tests for component library validation."""

    def test_component_count_matches_docstring(self):
        """Verify component count matches documentation."""
        components = get_predefined_components()
        # Update this when adding/removing components
        assert len(components) == 121, f"Expected 121 components, got {len(components)}"

    def test_no_duplicate_names(self):
        """Verify no duplicate component names."""
        components = get_predefined_components()
        names = list(components.keys())
        assert len(names) == len(set(names)), "Duplicate component names found"

    def test_all_components_have_valid_bands(self):
        """Verify all components have valid band parameters."""
        issues = validate_predefined_components()
        assert not issues, f"Validation issues:\n" + "\n".join(issues)

    def test_all_categories_populated(self):
        """Verify all documented categories have components."""
        categories = list_categories()
        expected = [
            "water_related", "proteins", "lipids", "carbohydrates",
            "alcohols", "organic_acids", "pigments", "pharmaceuticals",
            "polymers", "solvents", "minerals"
        ]
        for cat in expected:
            assert cat in categories, f"Missing category: {cat}"
            assert len(categories[cat]) > 0, f"Empty category: {cat}"

    def test_wavelength_coverage_350_2500(self):
        """Verify most components have bands in extended range."""
        coverage = validate_component_coverage((350, 2500))
        # Allow up to 5% without coverage (e.g., minerals with only MIR bands)
        coverage_ratio = len(coverage["covered"]) / (len(coverage["covered"]) + len(coverage["not_covered"]))
        assert coverage_ratio >= 0.95, f"Only {coverage_ratio:.1%} components have bands in 350-2500nm"
```

**Confidence**: High (straightforward validation logic)

---

## Section 6: Implementation Phases

### Phase 1: Component Quality (2-3 weeks)

**Tasks:**
1. Add metadata fields to `SpectralComponent` dataclass
2. Implement amplitude normalization (max=1.0)
3. Add category metadata to all 121 components
4. Resolve duplicates (aliases, not deletion)
5. Add validation framework
6. Write tests for validation

**Files Modified:**
- `nirs4all/data/synthetic/components.py`
- `nirs4all/data/synthetic/_constants.py`
- `nirs4all/data/synthetic/validation.py`
- `tests/unit/data/synthetic/test_constants_validation.py`

**Risk**: Low (metadata additions are backward-compatible)

### Phase 2: Wavelength Extension (3-4 weeks)

**Tasks:**
1. Update default wavelength range to 350-2500nm
2. Add visible-region spectral zones
3. Add 10+ new electronic absorption components (chlorophylls, hemoglobin, etc.)
4. Extend existing components with 2nd/3rd overtone bands
5. Update wavenumber.py with visible region zones
6. Update unit tests

**Files Modified:**
- `nirs4all/data/synthetic/_constants.py`
- `nirs4all/data/synthetic/wavenumber.py`
- `tests/unit/data/synthetic/test_wavenumber.py`
- `tests/unit/data/synthetic/test_components.py`

**Risk**: Medium (may affect downstream code expecting 1000-2500nm)

### Phase 3: Discovery API (1-2 weeks)

**Tasks:**
1. Implement `available_components()`, `get_component()`, `search_components()`
2. Implement `list_categories()`, `component_info()`
3. Export new functions from `__init__.py`
4. Write documentation and examples
5. Write unit tests

**Files Modified:**
- `nirs4all/data/synthetic/components.py`
- `nirs4all/data/synthetic/__init__.py`
- `docs/source/api/synthetic.rst`

**Risk**: Low (new functionality, no breaking changes)

### Phase 4: Aggregate Components (2 weeks)

**Tasks:**
1. Create `_aggregates.py` module
2. Define 15+ aggregate components (agriculture, food, pharma)
3. Implement `get_aggregate()`, `list_aggregates()`, `expand_aggregate()`
4. Integration with `SyntheticDatasetBuilder`
5. Write documentation and examples

**Files Modified:**
- `nirs4all/data/synthetic/_aggregates.py` (new)
- `nirs4all/data/synthetic/builder.py`
- `nirs4all/data/synthetic/__init__.py`

**Risk**: Low (new functionality)

### Phase 5: Spectral Fitting Tools (2-3 weeks)

**Tasks:**
1. Implement `ComponentFitter` class
2. Implement `FitResult` dataclass
3. Add NNLS-based fitting with baseline
4. Add batch fitting with parallel processing
5. Add `suggest_components()` method
6. Write comprehensive tests
7. Write documentation and tutorial

**Files Modified:**
- `nirs4all/data/synthetic/fitter.py`
- `nirs4all/data/synthetic/__init__.py`
- `docs/source/tutorials/spectral_fitting.rst` (new)

**Risk**: Medium (computational complexity, edge cases)

### Phase 6: Custom Wavelength Support (2 weeks)

**Goal**: Enable generation with user-provided wavelength arrays to match real instrument grids.

**Tasks:**
1. Add `wavelengths` parameter to `SyntheticNIRSGenerator.__init__()`
2. Implement cubic interpolation for component evaluation at arbitrary wavelengths
3. Add `INSTRUMENT_WAVELENGTHS` dictionary with common instruments
4. Implement `get_instrument_wavelengths()` function
5. Add `match_instrument_wavelengths` option to generator
6. Add `with_wavelengths()` method to `SyntheticDatasetBuilder`
7. Update `nirs4all.generate()` to accept `wavelengths` parameter
8. Write comprehensive tests for edge cases (sparse grids, extrapolation)
9. Write documentation with examples

**Files Modified:**
- `nirs4all/data/synthetic/generator.py`
- `nirs4all/data/synthetic/instruments.py`
- `nirs4all/data/synthetic/builder.py`
- `nirs4all/data/synthetic/components.py` (interpolation logic)
- `nirs4all/api/generate.py`
- `nirs4all/data/synthetic/__init__.py`
- `tests/unit/data/synthetic/test_custom_wavelengths.py` (new)

**Known Instrument Grids to Include:**
- MicroNIR OnSite (908-1676nm, 125 channels)
- SCiO (740-1070nm, 331 channels)
- NeoSpectra Micro (1350-2500nm, 228 channels)
- FOSS XDS (400-2498nm, 2nm step)
- Bruker MPA (FT-NIR, 800-2778nm)
- ASD FieldSpec (350-2500nm, 1nm step)

**Risk**: Low (additive feature, no breaking changes if wavelengths=None uses default)

### Phase 7: Product Generator for NN Training (3-4 weeks)

**Goal**: High-level API to generate diverse, realistic product samples with controlled variability for training neural networks.

**Tasks:**
1. Create `nirs4all/data/synthetic/products.py` module
2. Implement `VariationType` enum (FIXED, UNIFORM, NORMAL, LOGNORMAL, CORRELATED, COMPUTED)
3. Implement `ComponentVariation` dataclass
4. Implement `ProductTemplate` dataclass with variability specification
5. Define 15+ predefined product templates:
   - Dairy: milk_variable_fat, cheese_variable_moisture
   - Meat: meat_variable_fat
   - Grain: wheat_variable_protein, corn_grain, soybean
   - Pharma: tablet_variable_api
   - Special: food_cholesterol_variable (high variability for NN)
6. Implement `ProductGenerator` class with:
   - `_sample_compositions()` respecting correlations
   - `generate()` returning SpectroDataset
   - `generate_dataset_for_target()` for specific target component
7. Implement `CategoryGenerator` for combining multiple templates
8. Add convenience functions: `list_product_templates()`, `get_product_template()`, `generate_product_samples()`
9. Integrate with `nirs4all.generate` API:
   - `nirs4all.generate.product(template, n_samples, target, ...)`
   - `nirs4all.generate.category(category, n_samples, ...)`
10. Write comprehensive tests
11. Write tutorial for NN training use cases
12. Add examples in `examples/developer/synthetic_nn_training.py`

**Files Created:**
- `nirs4all/data/synthetic/products.py` (new)
- `tests/unit/data/synthetic/test_products.py` (new)
- `docs/source/tutorials/synthetic_for_nn.rst` (new)
- `examples/developer/synthetic_nn_training.py` (new)

**Files Modified:**
- `nirs4all/api/generate.py`
- `nirs4all/data/synthetic/__init__.py`

**Key Features for NN Training:**
- **Plasticity**: Wide variability ranges to train robust models
- **Correlation preservation**: Realistic component correlations (protein-starch, fat-water)
- **Composition constraints**: Sum to 1.0, realistic bounds
- **Target flexibility**: Any component can be the regression target
- **Scalability**: Generate 10k-100k samples efficiently
- **Instrument matching**: Combine with custom wavelengths for specific instruments

**Risk**: Medium (complex sampling logic, correlation handling)

---

## Section 7: Backward Compatibility

### 7.1 Breaking Changes

| Change | Impact | Mitigation |
|--------|--------|------------|
| Default wavelength 350→2500nm | Generated spectra have different shape | Provide `wavelength_start` parameter |
| Component count change | Tests may fail | Update test assertions |
| Amplitude normalization | Existing code may expect different values | Normalize consistently |

### 7.2 Deprecation Strategy

```python
# Example: Handle old wavelength range
def get_wavelength_range() -> Tuple[float, float]:
    """Get default wavelength range."""
    import warnings
    # If user expects old range, warn them
    if _using_legacy_range:
        warnings.warn(
            "Default wavelength range changed from (1000, 2500) to (350, 2500). "
            "Set wavelength_start=1000 explicitly for old behavior.",
            DeprecationWarning
        )
    return (350.0, 2500.0)
```

---

## Section 8: Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Wavelength range | 350-2500nm | ✅ Complete (Phase 2) |
| Component count | 130+ | ✅ 121 components with visible-region additions |
| Components with metadata | 100% | ✅ Complete (Phase 1) |
| Validation passing | 100% | ✅ Complete (Phase 1) |
| Test coverage | 95% | ✅ ~95% synthetic module coverage |
| API functions | 15+ | ✅ 20+ functions exported |
| Aggregate components | 15+ | ✅ 26 aggregates (Phase 4) |
| Known instrument grids | 6+ | ✅ 13 wavelength grids + 19 archetypes (Phase 6) |
| Custom wavelength support | Yes | ✅ Complete (Phase 6) |
| Spectral fitting (ComponentFitter) | Yes | ✅ Complete (Phase 5) |
| Product templates | 15+ | ✅ 18 templates (Phase 7) |
| NN training examples | 3+ | ⏳ Pending (tutorial & examples) |

---

## Section 9: References

### Spectroscopy References

1. **Workman & Weyer (2012)**: Practical Guide and Spectral Atlas for Interpretive NIR Spectroscopy
2. **Burns & Ciurczak (2007)**: Handbook of Near-Infrared Analysis, 3rd ed.
3. **Siesler et al. (2002)**: Near-Infrared Spectroscopy: Principles, Instruments, Applications
4. **Lichtenthaler & Buschmann (2001)**: Chlorophylls and Carotenoids: Measurement and Characterization
5. **Horecker (1943)**: The absorption spectra of hemoglobin and its derivatives

### Implementation References

6. **Bro & De Jong (1997)**: A fast non-negativity-constrained least squares algorithm (NNLS)
7. **Keshava & Mustard (2002)**: Spectral unmixing (IEEE Signal Processing)

---

## Appendix A: Component Extension Examples

### A.1 Water (Extended)

```python
"water": SpectralComponent(
    name="water",
    category="water_related",
    formula="H2O",
    bands=[
        # 3rd overtone (visible-NIR region)
        NIRBand(center=730, sigma=18, gamma=2, amplitude=0.05,
                name="O-H 3rd overtone"),

        # 2nd overtone (short-wave NIR)
        NIRBand(center=970, sigma=22, gamma=3, amplitude=0.15,
                name="O-H 2nd overtone"),

        # 1st overtone (existing, unchanged)
        NIRBand(center=1450, sigma=25, gamma=3, amplitude=0.80,
                name="O-H 1st overtone"),

        # Combination (existing, unchanged)
        NIRBand(center=1940, sigma=30, gamma=4, amplitude=1.00,
                name="O-H combination"),

        # Stretch + bend (existing, unchanged)
        NIRBand(center=2500, sigma=50, gamma=5, amplitude=0.30,
                name="O-H stretch + bend"),
    ],
    correlation_group=1,
    references=["Workman2012 p.35", "Burns2007 p.358"],
    tags=["universal", "moisture"],
)
```

### A.2 Chlorophyll A (New)

```python
"chlorophyll_a": SpectralComponent(
    name="chlorophyll_a",
    category="pigments",
    subcategory="chlorophylls",
    formula="C55H72MgN4O5",
    bands=[
        # Soret band (blue absorption)
        NIRBand(center=430, sigma=15, gamma=3, amplitude=1.00,
                name="Soret band"),

        # Q band (red absorption)
        NIRBand(center=662, sigma=12, gamma=2, amplitude=0.65,
                name="Q band"),

        # Red-edge tail (extends into NIR)
        NIRBand(center=700, sigma=20, gamma=5, amplitude=0.10,
                name="Red edge tail"),
    ],
    correlation_group=7,  # Pigments group
    references=["Lichtenthaler2001", "Workman2012 p.375"],
    tags=["plant", "agriculture", "visible"],
)
```

---

## Appendix B: Checklist

### Phase 1 Checklist ✅ COMPLETE
- [x] Add `category`, `subcategory`, `synonyms`, `formula`, `references`, `tags` to SpectralComponent
- [x] Add `validate()` method to SpectralComponent (instead of `__post_init__`)
- [x] Implement `normalize_component_amplitudes()` function
- [x] Update all 121 components with category metadata
- [x] Resolve polyester/pet, lutein/xanthophyll duplicates (via synonyms)
- [x] Implement `validate_predefined_components()`
- [x] Write validation tests
- [x] Update docstrings with computed counts
- [x] Implement Discovery API: `available_components()`, `get_component()`, `search_components()`, `list_categories()`, `component_info()`

### Phase 2 Checklist ✅ COMPLETE
- [x] Change DEFAULT_WAVELENGTH_START from 1000 to 350
- [x] Add visible region to EXTENDED_SPECTRAL_ZONES in wavenumber.py
- [x] Add chlorophyll_a, chlorophyll_b components
- [x] Add beta_carotene component (lycopene already existed, updated with visible bands)
- [x] Add hemoglobin_oxy, hemoglobin_deoxy components
- [x] Add anthocyanin_red, anthocyanin_purple components
- [x] Update melanin component with broad visible absorption
- [x] Extend water with 2nd/3rd overtones
- [x] Extend protein with 2nd/3rd overtone bands
- [x] Update wavenumber.py with visible zones (VISIBLE_ZONES_WAVENUMBER, EXTENDED_SPECTRAL_ZONES)
- [x] Add helper functions: is_visible_region(), is_nir_region(), classify_wavelength_extended(), get_all_zones_extended()
- [x] Added new hemoproteins: myoglobin, cytochrome_c, bilirubin
- [x] Update unit tests for new wavelength range and components

### Phase 3 Checklist ✅ COMPLETE (implemented during Phase 1)
- [x] Implement `available_components()`
- [x] Implement `get_component()`
- [x] Implement `search_components()`
- [x] Implement `list_categories()`
- [x] Implement `component_info()`
- [x] Export from `__init__.py`
- [x] Write documentation
- [x] Write tests

### Phase 4 Checklist ✅ COMPLETE
- [x] Create `_aggregates.py`
- [x] Define wheat_grain, corn_grain, soybean aggregates
- [x] Define milk, cheese_cheddar, meat_beef aggregates
- [x] Define tablet_excipient_base aggregate
- [x] Define soil_agricultural, leaf_green aggregates
- [x] Implement `get_aggregate()`, `list_aggregates()`, `expand_aggregate()`
- [x] Implement `aggregate_info()`, `list_aggregate_domains()`, `list_aggregate_categories()`, `validate_aggregates()`
- [x] Add `with_aggregate()` method to SyntheticDatasetBuilder
- [x] Add `generate_from_concentrations()` method to SyntheticNIRSGenerator
- [x] Export from `__init__.py`
- [x] Write comprehensive unit tests (45 tests passing)

### Phase 5 Checklist ✅ COMPLETE
- [x] Implement `ComponentFitter` class
- [x] Implement `ComponentFitResult` dataclass
- [x] Add NNLS fitting with `fit()` method
- [x] Add polynomial baseline fitting
- [x] Add batch fitting with `fit_batch()` method and parallel processing (joblib)
- [x] Add `suggest_components()` method
- [x] Add `get_concentration_matrix()` convenience method
- [x] Add `fit_components()` convenience function
- [x] Export from `__init__.py`
- [x] Write comprehensive unit tests (15 tests in test_fitter.py)

### Phase 6 Checklist ✅ COMPLETE
- [x] Add `wavelengths` parameter to `SyntheticNIRSGenerator.__init__()`
- [x] Add `instrument_wavelength_grid` parameter to `SyntheticNIRSGenerator.__init__()`
- [x] Add `INSTRUMENT_WAVELENGTHS` dictionary with 13 instruments:
  - MicroNIR OnSite, SCiO, NeoSpectra Micro, LinkSquare (handheld)
  - FOSS XDS, FOSS NIRS DS2500 (benchtop dispersive)
  - Bruker MPA (FT-NIR)
  - ASD FieldSpec (field portable)
  - ABB FTPA2000 (process NIR)
  - TI DLP NIRscan, Hamamatsu C14384MA (embedded/MEMS)
  - BUCHI NIRFlex, Thermo Antaris (specialty)
- [x] Implement `get_instrument_wavelengths()` function
- [x] Implement `list_instrument_wavelength_grids()` function
- [x] Implement `get_instrument_wavelength_info()` function
- [x] Add `with_wavelengths()` to `SyntheticDatasetBuilder` with `instrument_grid` option
- [x] Update `_create_generator()` to pass custom wavelengths
- [x] Handle custom wavelength grids (priority: wavelengths > instrument_grid > defaults)
- [x] Export from `__init__.py`
- [x] Write comprehensive unit tests (24 tests in test_custom_wavelengths.py)
  - Tested: instrument grids, custom arrays, generator integration, builder integration
  - Edge cases: non-uniform grids, sparse grids, high-resolution grids

### Phase 7 Checklist (Product Generator for NN Training)
- [x] Create `products.py` module
- [x] Implement `VariationType` enum (FIXED, UNIFORM, NORMAL, LOGNORMAL, CORRELATED, COMPUTED)
- [x] Implement `ComponentVariation` dataclass with validation
- [x] Implement `ProductTemplate` dataclass with component name validation
- [x] Define dairy templates: milk_variable_fat, cheese_variable_moisture, yogurt_variable_fat
- [x] Define meat templates: meat_variable_fat, meat_variable_protein
- [x] Define grain templates: wheat_variable_protein, corn_grain, soybean, rice_grain, barley_grain
- [x] Define pharma templates: tablet_variable_api, tablet_moisture_stability, capsule_blend_uniformity
- [x] Define high-variability NN templates: food_cholesterol_variable, universal_fat_predictor, universal_protein_predictor, universal_moisture_predictor
- [x] Define fruit template: fruit_sugar_variable
- [x] Implement `ProductGenerator._sample_compositions()` with correlation handling
- [x] Implement `ProductGenerator.generate()` with train/test splitting
- [x] Implement `ProductGenerator.generate_dataset_for_target()` with target scaling
- [x] Implement `CategoryGenerator` for multi-template generation with shuffle option
- [x] Add `list_product_templates()`, `get_product_template()`, `generate_product_samples()` convenience functions
- [x] Add `list_product_categories()`, `list_product_domains()`, `product_template_info()` discovery functions
- [x] Add `nirs4all.generate.product()` API with target_range support
- [x] Add `nirs4all.generate.category()` API with samples_per_template option
- [x] Export all classes and functions from `synthetic/__init__.py`
- [x] Write 55 unit tests covering all functionality
  - VariationType enum tests
  - ComponentVariation validation tests
  - ProductTemplate validation tests
  - Predefined templates validation (18 templates defined)
  - ProductGenerator composition sampling tests
  - CategoryGenerator multi-template tests
  - Generate API integration tests
  - Composition sampling correlation tests
- [ ] Write NN training tutorial
- [ ] Add example: `examples/developer/synthetic_nn_training.py`

---

*Document version 1.7 - January 2026*

**Version 1.7 Updates:**
- Updated Success Metrics table with actual counts (121 components, 26 aggregates, 13 wavelength grids + 19 archetypes, 18 templates)
- Updated Phase 7 checklist template count (17→18)
- Updated implementation notes with correct counts
- Updated _constants.py docstring to match actual component count (111→121)
- Verified all implementation phases are complete (Phases 1-7)

**Phase 5-7 Implementation Notes:**
- Phase 5 (Spectral Fitting): `ComponentFitter` class provides NNLS-based fitting for component unmixing with polynomial baseline support
- Phase 6 (Custom Wavelengths): Full support for arbitrary wavelength grids, including 13 predefined instrument wavelength grids + 19 instrument archetypes for simulation
- Phase 7 (Product Generator): High-level API for generating diverse, realistic product samples for NN training. Key features:
  - `ProductGenerator` class with 18 predefined product templates (dairy, meat, grain, pharma, high-variability)
  - Support for 6 variation types: FIXED, UNIFORM, NORMAL, LOGNORMAL, CORRELATED, COMPUTED
  - Correlation handling preserves realistic component relationships
  - `CategoryGenerator` for combining multiple templates into diverse training datasets
  - Integration with `nirs4all.generate.product()` and `nirs4all.generate.category()` APIs
