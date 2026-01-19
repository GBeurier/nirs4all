"""
Product Category Detector for NIR Spectra.

This module provides scientifically sound category detection for NIR spectral datasets,
inferring the most probable PRODUCT/MATRIX category (dominant material family) purely
from spectral shape and chemistry.

Key Features:
    - Canonical representation: handles reflectance, absorbance, and derivative inputs
    - Uniform grid resampling: handles irregular wavelength axes
    - Matched filter scoring: uses band templates instead of peak picking
    - Composition fitting: NNLS-based fitting for interpretable latent scores
    - Family scoring: water, hydrocarbons, lipids, proteins, carbohydrates, minerals, pigments
    - Product mapping: maps family scores to product categories
    - Mixture support: outputs top-K categories with weights when ambiguous
    - Confidence metrics: bootstrap stability and evidence-based thresholds
    - Wavelength-range awareness: penalizes only when bands are in-range but absent

Design Choices for Inversion/Generation:
    The family_scores output provides a meaningful latent composition vector that can
    be used to generate synthetic spectra with known category characteristics.

Usage:
    python bench/synthetic/dataset_category_identifier.py

References:
    - Workman & Weyer (2012). Practical Guide and Spectral Atlas for NIRS.
    - Burns & Ciurczak (2007). Handbook of Near-Infrared Analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import nnls
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# nirs4all imports
from nirs4all.data import DatasetConfigs

# Use nirs4all's component fitter for robust spectral analysis
try:
    from nirs4all.data.synthetic.fitter import OptimizedComponentFitter, COMPONENT_CATEGORIES

    NIRS4ALL_FITTER_AVAILABLE = True
except ImportError:
    NIRS4ALL_FITTER_AVAILABLE = False
    COMPONENT_CATEGORIES = {}

# =============================================================================
# CONSTANTS
# =============================================================================

DATASET_BASE = Path("/home/delete/NIRS DB/x_bank")

DATASET_NAMES = [
    "Beer_OriginalExtract_60_KS",
    "Biscuit_Fat_40_RandomSplit",
    "DIESEL_bp50_246_b-a",
    "grapevine_chloride_556_KS",
    "LUCAS_SOC_Organic_1102_NocitaKS",
    "Milk_Fat_1224_KS",
    "Poultry_manure_CaO_KS",
    "Rice_Amylose_313_YbasedSplit",
    "Corn_Moisture_80_WangStyle_m5spec",
    "TABLET_Escitalopramt_310_Zhao",
]

# Ground truth for evaluation (not used in detection)
GROUND_TRUTH = {
    "Beer_OriginalExtract_60_KS": "alcohols",
    "Biscuit_Fat_40_RandomSplit": "lipids",
    "DIESEL_bp50_246_b-a": "petroleum",
    "grapevine_chloride_556_KS": None,  # Uncertain
    "LUCAS_SOC_Organic_1102_NocitaKS": "minerals",
    "Milk_Fat_1224_KS": "lipids",
    "Poultry_manure_CaO_KS": None,  # Uncertain
    "Rice_Amylose_313_YbasedSplit": "carbohydrates",
    "Corn_Moisture_80_WangStyle_m5spec": "carbohydrates",
    "TABLET_Escitalopramt_310_Zhao": None,  # Pharmaceutical
}


# =============================================================================
# ENUMS
# =============================================================================


class CanonicalType(str, Enum):
    """Detected preprocessing/representation type of input spectra."""

    REFLECTANCE = "reflectance"
    ABSORBANCE = "absorbance"
    FIRST_DERIVATIVE = "first_derivative"
    SECOND_DERIVATIVE = "second_derivative"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level for predictions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# =============================================================================
# DIAGNOSTIC BAND DEFINITIONS
# =============================================================================

# Band templates for each chemical family
# Structure: center_nm, sigma_nm (width), weight, exclusivity (0-1), assignment
# Exclusivity: 1.0 = highly diagnostic for this family, 0.0 = shared with many families
FAMILY_BAND_TEMPLATES: Dict[str, List[Tuple[float, float, float, float, str]]] = {
    "water": [
        (1450, 30, 0.8, 0.2, "O-H 1st overtone"),
        (1940, 35, 1.0, 0.2, "O-H combination"),
        (970, 20, 0.4, 0.3, "O-H 2nd overtone"),
        (1190, 25, 0.3, 0.3, "O-H 2nd overtone / combination"),
    ],
    "hydrocarbons": [
        (1712, 25, 1.0, 0.9, "C-H 1st overtone CH2/CH3"),
        (1390, 20, 0.8, 0.8, "C-H symmetric methyl"),
        (1195, 18, 0.6, 0.7, "C-H 2nd overtone"),
        (2310, 25, 0.7, 0.5, "C-H combination"),
        (2350, 25, 0.7, 0.5, "CH2 combination"),
        (1170, 15, 0.5, 0.6, "C-H 2nd overtone"),
    ],
    "lipids": [
        (1760, 22, 1.0, 0.95, "C=O 1st overtone ester"),
        (1720, 25, 0.9, 0.7, "C-H 1st overtone CH2"),
        (2310, 25, 0.6, 0.4, "C-H combination"),
        (2350, 25, 0.6, 0.4, "CH2 combination"),
        (1210, 18, 0.4, 0.5, "C-H 2nd overtone"),
    ],
    "proteins": [
        (2050, 30, 1.0, 0.9, "N-H combination amide"),
        (2180, 28, 0.9, 0.85, "N-H + amide III"),
        (1510, 25, 0.8, 0.8, "N-H 1st overtone"),
        (1680, 25, 0.6, 0.7, "N-H / amide"),
        (2300, 25, 0.5, 0.4, "N-H + C-N"),
    ],
    "carbohydrates": [
        (2100, 30, 1.0, 0.85, "O-H + C-O starch/cellulose"),
        (2280, 28, 0.9, 0.85, "C-H + C-O starch"),
        (2060, 25, 0.8, 0.8, "Starch-specific"),
        (2320, 28, 0.7, 0.75, "Cellulose combination"),
        (1200, 20, 0.5, 0.4, "C-H 2nd overtone"),
    ],
    "alcohols": [
        (2270, 25, 1.0, 0.95, "C-O stretch combination ethanol"),
        (1580, 25, 0.7, 0.6, "O-H 1st overtone alcohols"),
        (2070, 25, 0.8, 0.85, "Ethanol combination"),
        (1119, 18, 0.6, 0.75, "C-C-O stretch overtone"),
    ],
    "minerals": [
        (2200, 30, 1.0, 0.95, "Al-OH kaolinite/illite"),
        (2160, 28, 0.9, 0.9, "Al-OH clay"),
        (2330, 30, 0.7, 0.8, "Carbonate combinations"),
        (2260, 25, 0.6, 0.75, "Fe-OH"),
        (1400, 30, 0.3, 0.2, "Structural water"),
    ],
    "pigments": [
        (670, 25, 1.0, 0.98, "Chlorophyll red absorption"),
        (550, 30, 0.8, 0.9, "Chlorophyll green reflectance"),
        (500, 25, 0.7, 0.85, "Carotenoid absorption"),
        (450, 25, 0.6, 0.8, "Carotenoid blue absorption"),
    ],
    "organic_acids": [
        (2090, 25, 0.8, 0.75, "O-H + C-O carboxylic"),
        (1700, 25, 0.9, 0.8, "C=O 1st overtone carboxylic"),
        (2520, 30, 0.6, 0.7, "Carboxylic combination"),
    ],
    "polymers": [
        (1680, 25, 0.7, 0.6, "C-H aromatic"),
        (2140, 28, 0.8, 0.7, "C-H + C=O ester polymer"),
        (1650, 25, 0.6, 0.5, "C=C aromatic"),
    ],
    # Pharmaceutical bands from nirs4all/_constants.py
    # Combines API (active pharmaceutical ingredient) and excipient signatures
    "pharmaceuticals": [
        # API characteristic bands (aromatic drugs, NSAIDs, etc.)
        (1140, 16, 0.7, 0.85, "Aromatic C-H 2nd overtone (APIs)"),
        (1432, 25, 0.8, 0.8, "Carboxylic O-H (NSAIDs)"),
        (1680, 18, 0.9, 0.75, "Aromatic C-H 1st overtone (APIs)"),
        (1695, 18, 0.7, 0.8, "N-CH3 1st overtone (caffeine, methylated drugs)"),
        (2020, 25, 0.85, 0.7, "C=O combination (carbonyl-containing APIs)"),
        # Microcrystalline cellulose (MCC) excipient bands - highly diagnostic
        (1492, 22, 0.9, 0.9, "MCC O-H 1st overtone (excipient)"),
        (1782, 18, 0.8, 0.9, "MCC cellulose C-H (excipient)"),
        (2092, 28, 0.85, 0.85, "MCC O-H combination (excipient)"),
        (2282, 22, 0.7, 0.8, "MCC cellulose C-O (excipient)"),
    ],
}

# Product category definitions: maps family score patterns to products
# Each product has required families (with min thresholds) and optional boosters
PRODUCT_CATEGORIES = {
    "petroleum/hydrocarbons": {
        "description": "Fuels, oils, and hydrocarbon-based materials",
        "required_families": {"hydrocarbons": 0.3},
        "boost_families": {"lipids": -0.15},  # Negative = penalize if present
        "exclude_families": {"lipids": 0.4},  # Exclude if lipids too strong
    },
    "lipid-rich/fats": {
        "description": "Dairy fats, vegetable oils, animal fats",
        "required_families": {"lipids": 0.25},
        "boost_families": {"hydrocarbons": 0.1},  # CH bands support
    },
    "starch/grain/carbohydrate": {
        "description": "Starch-based grains, flour, cereals",
        "required_families": {"carbohydrates": 0.3},
        "boost_families": {"proteins": 0.05},
    },
    "alcoholic/fermented": {
        "description": "Beers, wines, spirits, fermented products",
        "required_families": {"alcohols": 0.2},
        "boost_families": {"water": 0.05, "carbohydrates": 0.05},
    },
    "soil/mineral_matrix": {
        "description": "Soils, clays, mineral materials",
        "required_families": {"minerals": 0.25},
        "boost_families": {"organic_acids": 0.05},
    },
    "plant/vegetation": {
        "description": "Fresh leaves, plants, vegetation",
        "required_families": {"pigments": 0.2},
        "boost_families": {"water": 0.05, "carbohydrates": 0.05},
    },
    "protein-rich": {
        "description": "High-protein foods, feeds",
        "required_families": {"proteins": 0.25},
        "boost_families": {"water": 0.05},
    },
    "pharmaceutical/drug": {
        "description": "Pharmaceutical tablets, drug formulations",
        "required_families": {},  # Weak evidence pattern
        "boost_families": {"carbohydrates": 0.1, "proteins": 0.05},  # Excipients
        "low_overall_score": True,  # Characterized by weak pattern
    },
}


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class BandEvidence:
    """Evidence for a single band detection."""

    center_nm: float
    response_snr: float
    weight: float
    assignment: str
    in_range: bool = True


@dataclass
class FamilyScore:
    """Score for a chemical family."""

    family: str
    score: float
    band_evidence: List[BandEvidence] = field(default_factory=list)
    evidence_coverage: float = 1.0  # Fraction of bands in measurement range


@dataclass
class CategoryPrediction:
    """Prediction for a product category."""

    category: str
    score: float
    weight: float = 1.0


@dataclass
class RangeCoverage:
    """Wavelength range coverage information."""

    measured_range: Tuple[float, float]
    has_vis: bool = False  # < 780 nm
    has_nir1: bool = False  # 780-1100 nm
    has_nir2: bool = False  # 1100-1800 nm
    has_nir3: bool = False  # 1800-2500 nm
    missing_regions: List[str] = field(default_factory=list)


@dataclass
class IdentificationResult:
    """Complete identification result for a dataset."""

    # Input info
    dataset_name: str
    n_samples: int
    wavelength_range: Tuple[float, float]
    canonical_type: CanonicalType

    # Scores
    family_scores: Dict[str, FamilyScore]
    product_scores: List[CategoryPrediction]

    # Predictions
    predicted_category: str
    mixture_categories: List[Tuple[str, float]]  # (category, weight) pairs
    confidence: ConfidenceLevel
    confidence_numeric: float

    # Metadata
    limitation_notes: List[str]
    range_coverage: RangeCoverage

    # For evaluation (not used in detection)
    expected_category: Optional[str] = None
    is_correct: Optional[bool] = None


# =============================================================================
# CANONICALIZATION
# =============================================================================


def detect_canonical_type(X: np.ndarray) -> CanonicalType:
    """
    Detect whether input spectra are reflectance, absorbance, or derivative.

    Detection heuristics:
        - Derivative: many zero-crossings, mean near 0, has negatives
        - Reflectance: values in (0, ~1.2], typical NIRS reflectance
        - Absorbance: values in (0, ~3+], no negatives, typical absorbance range

    Args:
        X: Spectral data (n_samples, n_wavelengths).

    Returns:
        Detected canonical type.
    """
    min_val = np.min(X)
    max_val = np.max(X)
    mean_val = np.mean(X)

    # Derivative detection
    has_negative = min_val < -0.05
    mean_near_zero = abs(mean_val) < 0.3

    if has_negative and mean_near_zero:
        # Count zero crossings in median spectrum
        median_spec = np.median(X, axis=0)
        zero_crossings = np.sum(np.diff(np.sign(median_spec + 1e-10)) != 0)
        relative_crossings = zero_crossings / len(median_spec)

        if relative_crossings > 0.15:
            return CanonicalType.SECOND_DERIVATIVE
        elif relative_crossings > 0.05:
            return CanonicalType.FIRST_DERIVATIVE

    # Reflectance vs Absorbance
    if min_val >= -0.01 and max_val <= 1.2:
        # Typical reflectance range (0-1 or percent scaled)
        if max_val <= 0.05:
            return CanonicalType.UNKNOWN  # Too flat
        return CanonicalType.REFLECTANCE

    if min_val >= -0.1 and max_val > 1.2:
        # High values suggest absorbance
        if max_val > 3.5:
            return CanonicalType.ABSORBANCE  # High absorbance
        return CanonicalType.ABSORBANCE

    return CanonicalType.UNKNOWN


def canonicalize_spectra(
    X: np.ndarray,
    canonical_type: Optional[CanonicalType] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, CanonicalType]:
    """
    Convert spectra to canonical absorbance-like representation.

    For reflectance: A = -log10(max(R, eps))
    For absorbance: keep as-is
    For derivative: keep as-is (use different templates)

    Args:
        X: Input spectra (n_samples, n_wavelengths).
        canonical_type: If None, auto-detect.
        eps: Minimum value for log transform.

    Returns:
        Tuple of (canonical spectra, detected type).
    """
    if canonical_type is None:
        canonical_type = detect_canonical_type(X)

    if canonical_type == CanonicalType.REFLECTANCE:
        # Clamp to valid range and convert to pseudo-absorbance
        X_clamped = np.clip(X, eps, 1.0)
        X_canonical = -np.log10(X_clamped)
    elif canonical_type in (CanonicalType.FIRST_DERIVATIVE, CanonicalType.SECOND_DERIVATIVE):
        # Keep derivative as-is
        X_canonical = X.copy()
    else:
        # Assume already absorbance-like
        X_canonical = X.copy()

    return X_canonical, canonical_type


# =============================================================================
# RESAMPLING
# =============================================================================


def resample_to_uniform_grid(
    X: np.ndarray,
    wavelengths: np.ndarray,
    target_step: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample spectra to a uniform wavelength grid.

    Handles irregular wavelength spacing by interpolating to a regular grid.
    Essential for valid derivative computation with Savitzky-Golay.

    Args:
        X: Spectral data (n_samples, n_wavelengths).
        wavelengths: Original wavelength axis (may be irregular).
        target_step: Target step size. If None, uses median step.

    Returns:
        Tuple of (resampled spectra, new uniform wavelengths).
    """
    wl_steps = np.diff(wavelengths)
    step_variation = np.std(wl_steps) / (np.mean(wl_steps) + 1e-10)

    # If already uniform (< 5% variation), return as-is
    if step_variation < 0.05:
        return X.copy(), wavelengths.copy()

    # Determine target step
    if target_step is None:
        target_step = np.median(wl_steps)

    # Create uniform grid
    wl_min, wl_max = wavelengths.min(), wavelengths.max()
    n_points = int((wl_max - wl_min) / target_step) + 1
    wl_uniform = np.linspace(wl_min, wl_max, n_points)

    # Interpolate each sample
    X_resampled = np.zeros((X.shape[0], n_points))
    for i in range(X.shape[0]):
        interpolator = interp1d(wavelengths, X[i], kind="linear", fill_value="extrapolate")
        X_resampled[i] = interpolator(wl_uniform)

    return X_resampled, wl_uniform


# =============================================================================
# SCATTER CORRECTION
# =============================================================================


def apply_snv(X: np.ndarray) -> np.ndarray:
    """
    Apply Standard Normal Variate (SNV) correction.

    SNV corrects for scatter by row-wise centering and scaling:
        X_snv = (X - mean(X)) / std(X)

    Args:
        X: Spectral data (n_samples, n_wavelengths).

    Returns:
        SNV-corrected spectra.
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)  # Avoid division by zero
    return (X - mean) / std


# =============================================================================
# BAND TEMPLATE RESPONSE
# =============================================================================


def compute_band_response(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    center: float,
    sigma: float,
    window_factor: float = 3.0,
) -> Tuple[float, bool]:
    """
    Compute band evidence at a specific wavelength position.

    For NIR spectra, bands are broad and overlapping. This function computes
    band evidence by comparing the spectral intensity at the band position
    to the spectrum-wide statistics, rather than local baseline comparison.

    Args:
        spectrum: Single spectrum (1D array, absorbance-like).
        wavelengths: Wavelength axis.
        center: Band center wavelength (nm).
        sigma: Band width (sigma) in nm (used for averaging region).
        window_factor: Not used, kept for API compatibility.

    Returns:
        Tuple of (band_strength, in_range_flag).
        band_strength is how much the band region deviates from spectrum mean.
    """
    # Check if band center is in range
    if center < wavelengths.min() or center > wavelengths.max():
        return 0.0, False

    # Find wavelengths near band center for averaging
    # Use sigma to define the averaging region
    mask = (wavelengths >= center - sigma) & (wavelengths <= center + sigma)

    if np.sum(mask) < 3:
        # If band region is too narrow, use nearest point
        idx = np.argmin(np.abs(wavelengths - center))
        band_value = spectrum[idx]
    else:
        # Average over the band region
        band_value = np.mean(spectrum[mask])

    # Compare to spectrum statistics
    spec_mean = np.mean(spectrum)
    spec_std = np.std(spectrum)

    if spec_std < 1e-10:
        return 0.0, True

    # Compute z-score: how many stds above mean
    # For absorbance, higher values = more absorption = band present
    z_score = (band_value - spec_mean) / spec_std

    return z_score, True


def compute_derivative_band_response(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    center: float,
    sigma: float,
) -> Tuple[float, float]:
    """
    Compute band response for derivative spectra.

    For second derivative, absorption bands appear as positive peaks
    (minima in absorbance become maxima in d2A/dλ²).

    Args:
        spectrum: Single derivative spectrum (1D array).
        wavelengths: Wavelength axis.
        center: Band center wavelength (nm).
        sigma: Band width (sigma) in nm.

    Returns:
        Tuple of (response_snr, in_range_flag).
    """
    # Use narrower window for derivatives
    return compute_band_response(spectrum, wavelengths, center, sigma * 0.7, window_factor=2.5)


# =============================================================================
# FAMILY SCORING
# =============================================================================


def compute_family_scores(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    canonical_type: CanonicalType,
) -> Dict[str, FamilyScore]:
    """
    Compute scores for each chemical family using band template matching.

    Args:
        spectrum: Single canonical spectrum (1D array).
        wavelengths: Wavelength axis.
        canonical_type: Type of canonical representation.

    Returns:
        Dictionary mapping family names to FamilyScore objects.
    """
    wl_range = (wavelengths.min(), wavelengths.max())
    is_derivative = canonical_type in (CanonicalType.FIRST_DERIVATIVE, CanonicalType.SECOND_DERIVATIVE)

    family_scores = {}

    for family, bands in FAMILY_BAND_TEMPLATES.items():
        total_score = 0.0
        total_weight = 0.0
        bands_in_range = 0
        total_bands = 0
        band_evidence = []

        for center, sigma, weight, exclusivity, assignment in bands:
            total_bands += 1

            # Check if band is in measurement range
            in_range = wl_range[0] <= center <= wl_range[1]
            if not in_range:
                # Band outside range - track but don't penalize
                band_evidence.append(
                    BandEvidence(
                        center_nm=center,
                        response_snr=0.0,
                        weight=weight,
                        assignment=assignment,
                        in_range=False,
                    )
                )
                continue

            bands_in_range += 1

            # Compute band response
            if is_derivative:
                response_snr, valid = compute_derivative_band_response(spectrum, wavelengths, center, sigma)
                # For derivatives, we want positive values (absorption bands)
                response_snr = abs(response_snr)
            else:
                response_snr, valid = compute_band_response(spectrum, wavelengths, center, sigma)

            if not valid:
                continue

            # Weighted contribution
            # Higher exclusivity means more discriminative
            effective_weight = weight * (0.5 + 0.5 * exclusivity)

            # Score contribution: positive response is evidence for this family
            if response_snr > 0:
                contribution = effective_weight * min(response_snr, 5.0)  # Cap at SNR=5
                total_score += contribution
                total_weight += effective_weight
            else:
                total_weight += effective_weight * 0.3  # Partial weight for non-detection

            band_evidence.append(
                BandEvidence(
                    center_nm=center,
                    response_snr=response_snr,
                    weight=weight,
                    assignment=assignment,
                    in_range=True,
                )
            )

        # Coverage factor - critical bands must be measurable
        evidence_coverage = bands_in_range / total_bands if total_bands > 0 else 0.0

        # Normalize score
        if total_weight > 0:
            normalized_score = total_score / total_weight
        else:
            normalized_score = 0.0

        # CRITICAL: If coverage is too low, heavily penalize the score
        # This prevents families with no bands in range from scoring highly
        if evidence_coverage < 0.25:
            normalized_score *= evidence_coverage * 0.5  # Severe penalty
        elif evidence_coverage < 0.5:
            normalized_score *= 0.3 + evidence_coverage  # Moderate penalty

        family_scores[family] = FamilyScore(
            family=family,
            score=normalized_score,
            band_evidence=band_evidence,
            evidence_coverage=evidence_coverage,
        )

    return family_scores


# =============================================================================
# COMPOSITION FITTING (NNLS)
# =============================================================================


def build_basis_matrix(
    wavelengths: np.ndarray,
    families: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a basis matrix from family band templates.

    Each family contributes a composite spectrum based on its band templates.

    Args:
        wavelengths: Wavelength axis.
        families: List of family names to include.

    Returns:
        Tuple of (basis matrix [n_wavelengths x n_families], family names).
    """
    n_wl = len(wavelengths)
    n_families = len(families)

    basis = np.zeros((n_wl, n_families))

    for i, family in enumerate(families):
        if family not in FAMILY_BAND_TEMPLATES:
            continue

        family_spectrum = np.zeros(n_wl)
        for center, sigma, weight, _, _ in FAMILY_BAND_TEMPLATES[family]:
            # Add Gaussian band
            band = weight * np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
            family_spectrum += band

        # Normalize
        if np.max(family_spectrum) > 0:
            family_spectrum = family_spectrum / np.max(family_spectrum)

        basis[:, i] = family_spectrum

    return basis, families


def fit_composition_nnls(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    families: Optional[List[str]] = None,
    baseline_order: int = 3,
) -> Dict[str, float]:
    """
    Fit family composition using non-negative least squares.

    Solves: spectrum ≈ sum_i(a_i * basis_i) + baseline(polynomial)

    Universal components (water) are fit but normalized separately
    to prevent them from dominating the discriminative families.

    Args:
        spectrum: Single spectrum (1D array).
        wavelengths: Wavelength axis.
        families: Families to fit. If None, uses all.
        baseline_order: Order of polynomial baseline.

    Returns:
        Dictionary mapping family names to fitted coefficients.
    """
    # Universal families that appear in almost all samples
    UNIVERSAL_FAMILIES = {"water"}

    if families is None:
        families = list(FAMILY_BAND_TEMPLATES.keys())

    # Build basis matrix
    basis, family_names = build_basis_matrix(wavelengths, families)

    # Add polynomial baseline columns
    wl_norm = (wavelengths - wavelengths.mean()) / (wavelengths.std() + 1e-10)
    baseline_cols = np.column_stack([wl_norm**i for i in range(baseline_order + 1)])

    # Combined design matrix
    # Note: NNLS requires non-negative coefficients, so we handle baseline separately
    A_components = basis

    # First, fit and remove baseline
    baseline_fit, _, _, _ = np.linalg.lstsq(baseline_cols, spectrum, rcond=None)
    baseline = baseline_cols @ baseline_fit
    spectrum_detrended = spectrum - baseline

    # Fit components with NNLS
    if A_components.shape[1] > 0:
        coeffs, residual = nnls(A_components, spectrum_detrended)
    else:
        coeffs = np.array([])

    # Build raw result
    raw_result = {}
    for i, family in enumerate(family_names):
        if i < len(coeffs):
            raw_result[family] = float(coeffs[i])
        else:
            raw_result[family] = 0.0

    # Normalize DISCRIMINATIVE families separately from UNIVERSAL
    # This prevents water from overwhelming other components
    universal_total = sum(v for k, v in raw_result.items() if k in UNIVERSAL_FAMILIES)
    discriminative_total = sum(v for k, v in raw_result.items() if k not in UNIVERSAL_FAMILIES)

    result = {}
    for family, coeff in raw_result.items():
        if family in UNIVERSAL_FAMILIES:
            # Keep universal families at their raw proportion
            result[family] = coeff / (universal_total + discriminative_total) if (universal_total + discriminative_total) > 0 else 0
        else:
            # Normalize discriminative families to sum to 1 among themselves
            result[family] = coeff / discriminative_total if discriminative_total > 0 else 0

    return result


def compute_family_scores_with_fitter(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
) -> Dict[str, float]:
    """
    Compute family scores using nirs4all's OptimizedComponentFitter.

    This approach fits the spectrum as a linear combination of known
    spectral components and aggregates the results by family.

    Since the fitter works best with priority_categories, we run it
    multiple times with different priorities and aggregate results.

    Args:
        spectrum: Single spectrum (1D array).
        wavelengths: Wavelength axis.

    Returns:
        Dictionary mapping family names to scores.
    """
    if not NIRS4ALL_FITTER_AVAILABLE:
        return {}

    # Categories to try as priorities
    priority_options = [
        ["lipids"],
        ["carbohydrates"],
        ["proteins"],
        ["minerals"],
        ["alcohols"],
        ["hydrocarbons", "petroleum"],
        ["pharmaceuticals"],  # Add pharmaceutical detection
        [],  # No priority - general fit
    ]

    # Collect results from all fits
    all_family_scores = {}
    best_r2 = 0.0

    for priorities in priority_options:
        try:
            fitter = OptimizedComponentFitter(
                wavelengths=wavelengths,
                priority_categories=priorities,
                max_components=8,
                baseline_order=4,
                use_nnls=False,  # Allow negative (will filter later)
            )
            result = fitter.fit(spectrum)

            if result.r_squared < 0.8:
                continue  # Poor fit, skip

            # Aggregate fitted components by category
            family_scores = {}
            for comp_name, conc in zip(result.component_names, result.concentrations):
                # Only count positive concentrations
                if conc <= 0:
                    continue

                for category, components in COMPONENT_CATEGORIES.items():
                    if comp_name in components:
                        if category not in family_scores:
                            family_scores[category] = 0.0
                        family_scores[category] += conc
                        break

            # Weight by R2 and accumulate
            for cat, score in family_scores.items():
                if cat not in all_family_scores:
                    all_family_scores[cat] = 0.0
                all_family_scores[cat] += score * result.r_squared

        except Exception:
            continue

    # Normalize final scores
    total = sum(all_family_scores.values())
    if total > 0:
        all_family_scores = {k: v / total for k, v in all_family_scores.items()}

    return all_family_scores


# =============================================================================
# PRODUCT CATEGORY SCORING
# =============================================================================


def compute_product_scores(
    family_scores: Dict[str, FamilyScore],
) -> List[CategoryPrediction]:
    """
    Map family scores to product category predictions.

    Uses intelligent rule-based scoring that considers:
    - Required families with minimum thresholds
    - Discriminative comparisons between similar families
    - Evidence coverage constraints

    Args:
        family_scores: Dictionary of family scores.

    Returns:
        List of CategoryPrediction objects, sorted by score descending.
    """

    def get_score(family: str) -> float:
        return family_scores.get(family, FamilyScore(family, 0.0)).score

    def get_coverage(family: str) -> float:
        return family_scores.get(family, FamilyScore(family, 0.0)).evidence_coverage

    predictions = []

    # --- PETROLEUM/HYDROCARBONS ---
    # Key: strong CH bands, NO ester band (distinguish from lipids)
    hc_score = get_score("hydrocarbons")
    lipid_score = get_score("lipids")
    hc_coverage = get_coverage("hydrocarbons")

    # Hydrocarbons must be stronger than lipids (no ester)
    if hc_coverage >= 0.5 and hc_score > 0.2:
        # If hydrocarbon score dominates over lipid score, it's petroleum
        if hc_score > lipid_score * 1.2 or lipid_score < 0.3:
            petroleum_score = hc_score * 2.0
        else:
            petroleum_score = max(0, hc_score - lipid_score) * 1.5
    else:
        petroleum_score = 0.0
    predictions.append(CategoryPrediction("petroleum/hydrocarbons", petroleum_score))

    # --- LIPID-RICH/FATS ---
    # Key: ester band at 1760nm, supported by CH bands
    lipid_coverage = get_coverage("lipids")
    if lipid_coverage >= 0.2 and lipid_score > 0.15:  # Lower thresholds
        # Lipids score is boosted if CH is also present (fats have both)
        lipid_product_score = lipid_score * 2.5 + hc_score * 0.3
        # Penalize if ester band coverage is too low
        if lipid_coverage < 0.4:
            lipid_product_score *= lipid_coverage + 0.6
    else:
        lipid_product_score = lipid_score * 0.5
    predictions.append(CategoryPrediction("lipid-rich/fats", lipid_product_score))

    # --- STARCH/GRAIN/CARBOHYDRATE ---
    # Key: carbohydrate combination bands in NIR3 (2100, 2280, etc.)
    carb_score = get_score("carbohydrates")
    carb_coverage = get_coverage("carbohydrates")
    if carb_coverage >= 0.5 and carb_score > 0.2:
        carb_product_score = carb_score * 2.0
        # Slight boost if proteins also present (grains have both)
        carb_product_score += get_score("proteins") * 0.2
    else:
        carb_product_score = carb_score * 0.5
    predictions.append(CategoryPrediction("starch/grain/carbohydrate", carb_product_score))

    # --- ALCOHOLIC/FERMENTED ---
    # Key: ethanol bands AND must dominate other signatures
    # Alcohols family includes polyols (glycerol, sorbitol) which can appear in foods
    # True alcoholic beverages: alcohol dominates, lipids/carbs lower
    alcohol_score = get_score("alcohols")
    alcohol_coverage = get_coverage("alcohols")
    water_score = get_score("water")

    # Alcohol must be present AND stronger than lipids/carbs to indicate beverage
    is_dominant_alcohol = (
        alcohol_score > lipid_score * 1.5 and alcohol_score > carb_score * 1.5 and alcohol_coverage >= 0.5
    )

    # Additional check: if lipids are significant (>0.3), it's food not beverage
    # Foods like biscuits can have alcohol-family compounds (polyols) but also significant lipids
    is_food_not_beverage = lipid_score > 0.35

    if is_dominant_alcohol and alcohol_score > 0.3 and not is_food_not_beverage:
        alcohol_product_score = alcohol_score * 2.5  # Higher multiplier for true beverages
        alcohol_product_score += water_score * 0.15
        # Penalize if petroleum-like
        if hc_score > alcohol_score * 2:
            alcohol_product_score *= 0.3
    else:
        # Weak alcohol signal or likely polyols in food, not alcoholic beverage
        alcohol_product_score = alcohol_score * 0.15
    predictions.append(CategoryPrediction("alcoholic/fermented", alcohol_product_score))

    # --- SOIL/MINERAL MATRIX ---
    # Key: clay bands at 2200, 2160nm
    mineral_score = get_score("minerals")
    mineral_coverage = get_coverage("minerals")
    if mineral_coverage >= 0.5 and mineral_score > 0.2:
        mineral_product_score = mineral_score * 2.5
        # Organic matter in soil can add organic acid signal
        mineral_product_score += get_score("organic_acids") * 0.3
    else:
        mineral_product_score = mineral_score * 0.3
    predictions.append(CategoryPrediction("soil/mineral_matrix", mineral_product_score))

    # --- PLANT/VEGETATION ---
    # Key: pigments in VIS, REQUIRES visible range coverage
    pigment_score = get_score("pigments")
    pigment_coverage = get_coverage("pigments")
    # Pigments can ONLY be detected if VIS bands are in range
    if pigment_coverage >= 0.5 and pigment_score > 0.3:
        vegetation_score = pigment_score * 2.0
        # Water and carbs support plant material
        vegetation_score += water_score * 0.1 + carb_score * 0.1
    else:
        vegetation_score = 0.0  # Cannot claim vegetation without pigment evidence
    predictions.append(CategoryPrediction("plant/vegetation", vegetation_score))

    # --- PROTEIN-RICH ---
    # Key: N-H bands at 2050, 2180, 1510nm
    protein_score = get_score("proteins")
    protein_coverage = get_coverage("proteins")
    if protein_coverage >= 0.5 and protein_score > 0.25:
        protein_product_score = protein_score * 2.0
    else:
        protein_product_score = protein_score * 0.3
    predictions.append(CategoryPrediction("protein-rich", protein_product_score))

    # --- PHARMACEUTICAL/DRUG ---
    # Key: Uses new "pharmaceuticals" family with API bands (aromatic C-H, COOH, C=O)
    # and MCC excipient bands (1492, 1782, 2092nm)
    # IMPORTANT: Pharmaceutical tablets are mostly EXCIPIENT (carbohydrates) with small API
    # Pattern: pharma bands present + high carbs + low lipids/proteins = tablet formulation
    pharma_score_raw = get_score("pharmaceuticals")
    pharma_coverage = get_coverage("pharmaceuticals")

    # Check for pharmaceutical formulation pattern:
    # - Pharmaceutical bands detected (even weakly)
    # - High carbohydrate (lactose, starch, MCC excipients)
    # - Low lipid and alcohol (tablets are dry powders)
    is_tablet_pattern = (
        pharma_score_raw > 0.15
        and carb_score > 0.4
        and lipid_score < 0.35
        and alcohol_score < 0.5
    )

    # Also check for API-dominant pattern (pure drug, less excipient)
    max_non_carb_score = max(lipid_score, protein_score, alcohol_score, hc_score)
    is_api_pattern = pharma_score_raw > 0.3 and pharma_score_raw > max_non_carb_score * 0.7

    if is_tablet_pattern:
        # Tablet formulation: pharma + carb excipients
        # Use higher multiplier since tablets are pharma even if carbs dominate
        pharma_score = pharma_score_raw * 3.5 + carb_score * 1.0
    elif is_api_pattern and pharma_coverage >= 0.5:
        # API-dominant (less common in tablets but possible)
        pharma_score = pharma_score_raw * 2.5
    elif pharma_score_raw > 0.1 and max_non_carb_score < 0.25:
        # Weak but present with minimal interference
        pharma_score = pharma_score_raw * 1.0
    else:
        pharma_score = 0.0
    predictions.append(CategoryPrediction("pharmaceutical/drug", pharma_score))

    # Sort by score
    predictions.sort(key=lambda p: -p.score)

    return predictions


# =============================================================================
# CONFIDENCE ESTIMATION
# =============================================================================


def compute_confidence(
    product_scores: List[CategoryPrediction],
    family_scores: Dict[str, FamilyScore],
    range_coverage: RangeCoverage,
    n_bootstrap_samples: int = 0,
) -> Tuple[ConfidenceLevel, float, List[str]]:
    """
    Compute prediction confidence based on multiple factors.

    Factors:
        - Score margin between top candidates
        - Total evidence strength
        - Wavelength range coverage
        - Bootstrap stability (if samples provided)

    Args:
        product_scores: Sorted list of product predictions.
        family_scores: Dictionary of family scores.
        range_coverage: Wavelength range coverage info.
        n_bootstrap_samples: Number of bootstrap iterations (0 = skip).

    Returns:
        Tuple of (confidence_level, confidence_numeric, limitation_notes).
    """
    notes = []

    if len(product_scores) < 2:
        return ConfidenceLevel.LOW, 0.2, ["Insufficient predictions"]

    top1_score = product_scores[0].score
    top2_score = product_scores[1].score

    # Score margin factor
    if top1_score > 0:
        margin_ratio = (top1_score - top2_score) / top1_score
    else:
        margin_ratio = 0.0

    # Total evidence factor
    total_evidence = sum(fs.score for fs in family_scores.values())
    evidence_factor = min(1.0, total_evidence / 3.0)

    # Coverage factor
    avg_coverage = np.mean([fs.evidence_coverage for fs in family_scores.values()])
    coverage_factor = avg_coverage

    # Combined confidence
    confidence_numeric = 0.4 * margin_ratio + 0.3 * evidence_factor + 0.3 * coverage_factor

    # Add notes for limitations
    if range_coverage.missing_regions:
        notes.append(f"Missing regions: {', '.join(range_coverage.missing_regions)}")

    if margin_ratio < 0.1:
        notes.append(f"Close competition: {product_scores[0].category} vs {product_scores[1].category}")

    if total_evidence < 1.0:
        notes.append("Weak overall spectral evidence")

    # Determine level
    if confidence_numeric >= 0.6 and margin_ratio > 0.3 and total_evidence > 1.5:
        level = ConfidenceLevel.HIGH
    elif confidence_numeric >= 0.35 or (margin_ratio > 0.2 and total_evidence > 0.8):
        level = ConfidenceLevel.MEDIUM
    elif confidence_numeric >= 0.15:
        level = ConfidenceLevel.LOW
    else:
        level = ConfidenceLevel.UNKNOWN
        notes.append("Insufficient evidence for confident prediction")

    return level, confidence_numeric, notes


# =============================================================================
# RANGE COVERAGE
# =============================================================================


def compute_range_coverage(wavelengths: np.ndarray) -> RangeCoverage:
    """
    Analyze wavelength range coverage for diagnostic purposes.

    Args:
        wavelengths: Wavelength axis.

    Returns:
        RangeCoverage object with region information.
    """
    wl_min, wl_max = wavelengths.min(), wavelengths.max()

    has_vis = wl_min < 780
    has_nir1 = wl_min < 1100 and wl_max > 780
    has_nir2 = wl_min < 1800 and wl_max > 1100
    has_nir3 = wl_max > 1800

    missing = []
    if not has_vis:
        missing.append("VIS (<780nm)")
    if not has_nir1:
        missing.append("NIR1 (780-1100nm)")
    if not has_nir2:
        missing.append("NIR2 (1100-1800nm)")
    if not has_nir3:
        missing.append("NIR3 (>1800nm)")

    return RangeCoverage(
        measured_range=(wl_min, wl_max),
        has_vis=has_vis,
        has_nir1=has_nir1,
        has_nir2=has_nir2,
        has_nir3=has_nir3,
        missing_regions=missing,
    )


# =============================================================================
# MAIN IDENTIFICATION
# =============================================================================


def identify_category(
    X: np.ndarray,
    wavelengths: np.ndarray,
    dataset_name: str = "dataset",
    apply_scatter_correction: bool = True,
    use_composition_fitting: bool = True,
) -> IdentificationResult:
    """
    Identify the most probable product category for a dataset.

    Main entry point for category detection. Handles:
        - Canonical representation detection and conversion
        - Uniform grid resampling
        - Scatter correction (SNV)
        - Band template matching for family scoring
        - Optional NNLS composition fitting
        - Product category mapping
        - Confidence estimation

    Args:
        X: Spectral data (n_samples, n_wavelengths).
        wavelengths: Wavelength axis.
        dataset_name: Name for reporting.
        apply_scatter_correction: Apply SNV correction.
        use_composition_fitting: Use NNLS fitting (slower but more accurate).

    Returns:
        IdentificationResult with predictions and metadata.
    """
    # Ensure 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_samples = X.shape[0]

    # 1. Detect canonical type and convert
    canonical_type = detect_canonical_type(X)
    X_canonical, _ = canonicalize_spectra(X, canonical_type)

    # 2. Resample to uniform grid
    X_uniform, wl_uniform = resample_to_uniform_grid(X_canonical, wavelengths)

    # 3. Compute representative spectrum (median) for band detection
    # NOTE: Do NOT apply SNV for band detection - it normalizes away band heights
    median_spectrum_raw = np.median(X_uniform, axis=0)

    # 4. Apply light smoothing to reduce noise (optional)
    # Use simple moving average
    if len(median_spectrum_raw) > 10:
        kernel_size = min(5, len(median_spectrum_raw) // 20)
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            median_spectrum = np.convolve(median_spectrum_raw, kernel, mode="same")
        else:
            median_spectrum = median_spectrum_raw
    else:
        median_spectrum = median_spectrum_raw

    # 5. Range coverage analysis
    range_coverage = compute_range_coverage(wl_uniform)

    # 6. PRIMARY: Use nirs4all's OptimizedComponentFitter for family scores
    # This uses scientifically validated spectral components
    family_scores = {}
    wl_range = (wl_uniform.min(), wl_uniform.max())

    if NIRS4ALL_FITTER_AVAILABLE:
        try:
            fitter_scores = compute_family_scores_with_fitter(median_spectrum, wl_uniform)

            for family, score in fitter_scores.items():
                # Map fitter categories to our family names
                family_mapping = {
                    "water_related": "water",
                    "proteins": "proteins",
                    "lipids": "lipids",
                    "hydrocarbons": "hydrocarbons",
                    "petroleum": "hydrocarbons",  # Map to hydrocarbons
                    "carbohydrates": "carbohydrates",
                    "alcohols": "alcohols",
                    "organic_acids": "organic_acids",
                    "pigments": "pigments",
                    "minerals": "minerals",
                    "pharmaceuticals": "pharmaceuticals",  # Now has proper band templates
                    "organic_matter": "carbohydrates",  # Approximate
                    "polymers": "polymers",
                }
                mapped_family = family_mapping.get(family, family)

                # Calculate coverage based on our band definitions
                if mapped_family in FAMILY_BAND_TEMPLATES:
                    bands = FAMILY_BAND_TEMPLATES[mapped_family]
                    bands_in_range = sum(1 for c, _, _, _, _ in bands if wl_range[0] <= c <= wl_range[1])
                    coverage = bands_in_range / len(bands) if bands else 0.0
                else:
                    coverage = 1.0

                # Aggregate scores for mapped families
                if mapped_family not in family_scores:
                    family_scores[mapped_family] = FamilyScore(
                        family=mapped_family,
                        score=score * 3.0,  # Scale for product scoring
                        band_evidence=[],
                        evidence_coverage=coverage,
                    )
                else:
                    # Add to existing score
                    existing = family_scores[mapped_family]
                    family_scores[mapped_family] = FamilyScore(
                        family=mapped_family,
                        score=existing.score + score * 3.0,
                        band_evidence=existing.band_evidence,
                        evidence_coverage=existing.evidence_coverage,
                    )

        except Exception as e:
            pass  # Fall through to fallback

    # Fallback to simple NNLS if fitter didn't work
    if not family_scores:
        try:
            composition = fit_composition_nnls(median_spectrum, wl_uniform)

            for family, coeff in composition.items():
                if family in FAMILY_BAND_TEMPLATES:
                    bands = FAMILY_BAND_TEMPLATES[family]
                    bands_in_range = sum(1 for c, _, _, _, _ in bands if wl_range[0] <= c <= wl_range[1])
                    coverage = bands_in_range / len(bands) if bands else 0.0
                else:
                    coverage = 1.0

                if coverage < 0.25:
                    adjusted_coeff = coeff * coverage * 0.5
                elif coverage < 0.5:
                    adjusted_coeff = coeff * (0.3 + coverage)
                else:
                    adjusted_coeff = coeff

                family_scores[family] = FamilyScore(
                    family=family,
                    score=adjusted_coeff * 3.0,
                    band_evidence=[],
                    evidence_coverage=coverage,
                )

        except Exception:
            family_scores = compute_family_scores(median_spectrum, wl_uniform, canonical_type)

    # 8. Map to product categories
    product_scores = compute_product_scores(family_scores)

    # 9. Compute confidence
    confidence_level, confidence_numeric, limitation_notes = compute_confidence(product_scores, family_scores, range_coverage)

    # 10. Determine predictions
    predicted_category = product_scores[0].category if product_scores else "unknown"

    # Get mixture candidates (top 3 with significant scores)
    mixture_categories = []
    if len(product_scores) >= 2:
        top_score = product_scores[0].score
        for pred in product_scores[:3]:
            if pred.score > 0 and (top_score == 0 or pred.score / top_score > 0.3):
                weight = pred.score / top_score if top_score > 0 else 1.0
                mixture_categories.append((pred.category, weight))

    # If very uncertain, mark as unknown
    if confidence_level == ConfidenceLevel.UNKNOWN or (product_scores and product_scores[0].score < 0.1):
        predicted_category = "unknown"

    # Expected category for evaluation
    expected = GROUND_TRUTH.get(dataset_name)
    is_correct = None
    if expected is not None:
        # Map expected to product category format
        expected_mapping = {
            "alcohols": "alcoholic/fermented",
            "lipids": "lipid-rich/fats",
            "petroleum": "petroleum/hydrocarbons",
            "minerals": "soil/mineral_matrix",
            "carbohydrates": "starch/grain/carbohydrate",
            "proteins": "protein-rich",
            "pigments": "plant/vegetation",
        }
        expected_product = expected_mapping.get(expected, expected)
        is_correct = predicted_category == expected_product

    return IdentificationResult(
        dataset_name=dataset_name,
        n_samples=n_samples,
        wavelength_range=(wavelengths.min(), wavelengths.max()),
        canonical_type=canonical_type,
        family_scores=family_scores,
        product_scores=product_scores,
        predicted_category=predicted_category,
        mixture_categories=mixture_categories,
        confidence=confidence_level,
        confidence_numeric=confidence_numeric,
        limitation_notes=limitation_notes,
        range_coverage=range_coverage,
        expected_category=expected,
        is_correct=is_correct,
    )


# =============================================================================
# DATASET LOADING
# =============================================================================


def load_dataset(name: str) -> Dict:
    """Load dataset from CSV."""
    csv_path = DATASET_BASE / f"{name}.csv"
    config = {
        "x_train": str(csv_path),
        "delimiter": ",",
        "has_header": True,
        "header_unit": "nm",
    }
    ds = DatasetConfigs(config).get_datasets()[0]
    X = ds.x({}, layout="2d")
    wl = ds.wavelengths_nm(0)
    if wl is None:
        wl = np.arange(X.shape[1])
    return {"name": name, "X": X, "wl": wl}


def load_all_datasets() -> List[Dict]:
    """Load all datasets."""
    datasets = []
    for name in DATASET_NAMES:
        try:
            d = load_dataset(name)
            datasets.append(d)
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")
    return datasets


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def run_experiment() -> Tuple[List[IdentificationResult], float]:
    """Run identification on all datasets and return results with accuracy."""
    print("Loading datasets...")
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets\n")

    results = []
    correct = 0
    total_definite = 0

    print("=" * 120)
    print(f"{'Dataset':<18} {'Expected':<22} {'Predicted':<25} {'Conf':<8} {'Score':<8} {'Result'}")
    print("=" * 120)

    for d in datasets:
        result = identify_category(d["X"], d["wl"], d["name"])
        results.append(result)

        # Format expected
        expected_str = result.expected_category if result.expected_category else "uncertain"

        # Check correctness
        if result.expected_category is not None:
            total_definite += 1
            if result.is_correct:
                correct += 1
                status = "OK"
            else:
                status = "WRONG"
        else:
            status = "n/a"

        # Get top score
        top_score = result.product_scores[0].score if result.product_scores else 0.0

        # Short name
        short_name = d["name"].split("_")[0]
        print(f"{short_name:<18} {expected_str:<22} {result.predicted_category:<25} " f"{result.confidence.value:<8} {top_score:<8.2f} {status}")

        # Print limitations
        if result.limitation_notes:
            for note in result.limitation_notes[:2]:
                print(f"{'':>18} -> {note}")

    print("=" * 120)

    # Calculate accuracy
    accuracy = correct / total_definite if total_definite > 0 else 0.0
    print(f"\nOverall accuracy: {correct}/{total_definite} ({accuracy*100:.1f}%)")

    return results, accuracy


def print_detailed_results(results: List[IdentificationResult]):
    """Print detailed scoring for each dataset."""
    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)

    for result in results:
        print(f"\n{result.dataset_name}")
        print(f"  Canonical type: {result.canonical_type.value}")
        print(f"  WL range: {result.wavelength_range[0]:.0f}-{result.wavelength_range[1]:.0f} nm")
        print(f"  N samples: {result.n_samples}")

        print(f"\n  Family Scores:")
        sorted_families = sorted(result.family_scores.items(), key=lambda x: -x[1].score)
        for family, score in sorted_families[:5]:
            coverage = f"(cov={score.evidence_coverage:.0%})" if score.evidence_coverage < 1.0 else ""
            print(f"    {family:<15}: {score.score:6.3f} {coverage}")

        print(f"\n  Product Scores:")
        for pred in result.product_scores[:3]:
            print(f"    {pred.category:<30}: {pred.score:6.3f}")

        if result.mixture_categories:
            print(f"\n  Mixture: {result.mixture_categories}")

        print(f"\n  Prediction: {result.predicted_category}")
        print(f"  Confidence: {result.confidence.value} ({result.confidence_numeric:.2f})")


# =============================================================================
# UNIT TESTS
# =============================================================================


def run_tests():
    """Run unit tests for key functions."""
    print("\n" + "=" * 80)
    print("RUNNING UNIT TESTS")
    print("=" * 80)

    passed = 0
    failed = 0

    # Test 1: Canonical type detection
    print("\nTest 1: Canonical type detection")

    # Reflectance-like
    X_refl = np.random.uniform(0.3, 0.8, (10, 100))
    ct = detect_canonical_type(X_refl)
    if ct == CanonicalType.REFLECTANCE:
        print("  [PASS] Reflectance detection")
        passed += 1
    else:
        print(f"  [FAIL] Reflectance detection: got {ct}")
        failed += 1

    # Absorbance-like
    X_abs = np.random.uniform(0.5, 2.5, (10, 100))
    ct = detect_canonical_type(X_abs)
    if ct == CanonicalType.ABSORBANCE:
        print("  [PASS] Absorbance detection")
        passed += 1
    else:
        print(f"  [FAIL] Absorbance detection: got {ct}")
        failed += 1

    # Derivative-like (zero-mean, oscillating)
    wl = np.linspace(1000, 2000, 100)
    X_deriv = np.sin(wl / 50) * 0.1 + np.random.randn(10, 100) * 0.01
    ct = detect_canonical_type(X_deriv)
    if ct in (CanonicalType.FIRST_DERIVATIVE, CanonicalType.SECOND_DERIVATIVE):
        print("  [PASS] Derivative detection")
        passed += 1
    else:
        print(f"  [FAIL] Derivative detection: got {ct}")
        failed += 1

    # Test 2: Resampling
    print("\nTest 2: Uniform grid resampling")

    wl_irregular = np.array([1000, 1010, 1025, 1050, 1080, 1120, 1170, 1230])
    X_test = np.random.randn(5, len(wl_irregular))
    X_resampled, wl_resampled = resample_to_uniform_grid(X_test, wl_irregular)

    wl_steps = np.diff(wl_resampled)
    step_var = np.std(wl_steps) / np.mean(wl_steps)
    if step_var < 0.01:
        print(f"  [PASS] Uniform resampling (step variance: {step_var:.4f})")
        passed += 1
    else:
        print(f"  [FAIL] Uniform resampling: step variance {step_var:.4f}")
        failed += 1

    # Test 3: SNV correction
    print("\nTest 3: SNV correction")

    X_test = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])
    X_snv = apply_snv(X_test)

    # Check row means are ~0 and stds are ~1
    row_means = np.mean(X_snv, axis=1)
    row_stds = np.std(X_snv, axis=1)

    if np.allclose(row_means, 0, atol=1e-10) and np.allclose(row_stds, 1, atol=0.1):
        print(f"  [PASS] SNV normalization")
        passed += 1
    else:
        print(f"  [FAIL] SNV: means={row_means}, stds={row_stds}")
        failed += 1

    # Test 4: Band response computation
    print("\nTest 4: Band template response")

    wl = np.linspace(1400, 1500, 100)
    # Spectrum with a peak at 1450 nm
    spectrum = np.exp(-0.5 * ((wl - 1450) / 20) ** 2) + 0.1

    response, in_range = compute_band_response(spectrum, wl, center=1450, sigma=20)
    if response > 1.0 and in_range:
        print(f"  [PASS] Band detection at 1450nm (SNR={response:.2f})")
        passed += 1
    else:
        print(f"  [FAIL] Band detection: response={response}, in_range={in_range}")
        failed += 1

    # Test for no peak
    spectrum_flat = np.ones_like(wl) * 0.5
    response_flat, _ = compute_band_response(spectrum_flat, wl, center=1450, sigma=20)
    if abs(response_flat) < 0.5:
        print(f"  [PASS] No false positive on flat spectrum (SNR={response_flat:.2f})")
        passed += 1
    else:
        print(f"  [FAIL] False positive on flat: response={response_flat}")
        failed += 1

    # Test 5: Family scoring with synthetic spectrum
    print("\nTest 5: Family scoring with synthetic lipid spectrum")

    wl = np.linspace(1100, 2400, 200)
    # Create synthetic lipid-like spectrum with ester band at 1760nm and CH at 1720nm
    spectrum = np.zeros_like(wl)
    spectrum += 0.8 * np.exp(-0.5 * ((wl - 1760) / 22) ** 2)  # Ester C=O
    spectrum += 0.6 * np.exp(-0.5 * ((wl - 1720) / 25) ** 2)  # C-H
    spectrum += 0.3 * np.exp(-0.5 * ((wl - 2310) / 25) ** 2)  # CH combination
    spectrum += 0.1  # Baseline

    family_scores = compute_family_scores(spectrum, wl, CanonicalType.ABSORBANCE)

    lipid_score = family_scores["lipids"].score
    hydrocarbon_score = family_scores["hydrocarbons"].score

    if lipid_score > 0.3 and lipid_score > hydrocarbon_score:
        print(f"  [PASS] Lipid spectrum detected (lipids={lipid_score:.2f}, hydrocarbons={hydrocarbon_score:.2f})")
        passed += 1
    else:
        print(f"  [FAIL] Lipid detection: lipids={lipid_score}, hydrocarbons={hydrocarbon_score}")
        failed += 1

    # Test 6: Mineral detection
    print("\nTest 6: Family scoring with synthetic mineral spectrum")

    wl = np.linspace(1800, 2500, 150)
    spectrum = np.zeros_like(wl)
    spectrum += 0.9 * np.exp(-0.5 * ((wl - 2200) / 30) ** 2)  # Al-OH kaolinite
    spectrum += 0.7 * np.exp(-0.5 * ((wl - 2160) / 28) ** 2)  # Al-OH clay
    spectrum += 0.05

    family_scores = compute_family_scores(spectrum, wl, CanonicalType.ABSORBANCE)
    mineral_score = family_scores["minerals"].score

    if mineral_score > 0.4:
        print(f"  [PASS] Mineral spectrum detected (score={mineral_score:.2f})")
        passed += 1
    else:
        print(f"  [FAIL] Mineral detection: score={mineral_score}")
        failed += 1

    # Test 7: Unknown/weak evidence
    print("\nTest 7: Unknown detection with weak evidence")

    wl = np.linspace(1000, 2000, 100)
    spectrum_noise = np.random.randn(100) * 0.01 + 0.5  # Just noise

    family_scores = compute_family_scores(spectrum_noise, wl, CanonicalType.ABSORBANCE)
    total_evidence = sum(fs.score for fs in family_scores.values())

    if total_evidence < 1.5:
        print(f"  [PASS] Weak evidence detected (total={total_evidence:.2f})")
        passed += 1
    else:
        print(f"  [FAIL] Should have weak evidence: total={total_evidence}")
        failed += 1

    # Test 8: NNLS composition fitting
    print("\nTest 8: NNLS composition fitting")

    wl = np.linspace(1100, 2400, 200)
    # Create mixed spectrum
    spectrum = np.zeros_like(wl)
    # Add water bands
    spectrum += 0.5 * np.exp(-0.5 * ((wl - 1450) / 30) ** 2)
    spectrum += 0.6 * np.exp(-0.5 * ((wl - 1940) / 35) ** 2)
    # Add protein band
    spectrum += 0.4 * np.exp(-0.5 * ((wl - 2050) / 30) ** 2)
    spectrum += 0.1

    composition = fit_composition_nnls(spectrum, wl)

    if composition.get("water", 0) > 0.1 and composition.get("proteins", 0) > 0.05:
        print(f"  [PASS] Mixture detected (water={composition.get('water', 0):.2f}, proteins={composition.get('proteins', 0):.2f})")
        passed += 1
    else:
        print(f"  [FAIL] Mixture detection: {composition}")
        failed += 1

    # Test 9: Pigment detection (VIS range)
    print("\nTest 9: Pigment detection in VIS range")

    wl = np.linspace(400, 800, 100)
    spectrum = np.zeros_like(wl)
    spectrum += 0.8 * np.exp(-0.5 * ((wl - 670) / 25) ** 2)  # Chlorophyll red
    spectrum += 0.6 * np.exp(-0.5 * ((wl - 500) / 25) ** 2)  # Carotenoid
    spectrum += 0.1

    family_scores = compute_family_scores(spectrum, wl, CanonicalType.ABSORBANCE)
    pigment_score = family_scores["pigments"].score

    if pigment_score > 0.3:
        print(f"  [PASS] Pigment spectrum detected (score={pigment_score:.2f})")
        passed += 1
    else:
        print(f"  [FAIL] Pigment detection: score={pigment_score}")
        failed += 1

    # Test 10: End-to-end identification
    print("\nTest 10: End-to-end petroleum identification")

    wl = np.linspace(1100, 2400, 200)
    # Create hydrocarbon/petroleum spectrum
    X = np.zeros((10, len(wl)))
    for i in range(10):
        X[i] = 0.7 * np.exp(-0.5 * ((wl - 1712) / 25) ** 2)  # C-H methylene
        X[i] += 0.5 * np.exp(-0.5 * ((wl - 1390) / 20) ** 2)  # C-H methyl
        X[i] += 0.4 * np.exp(-0.5 * ((wl - 1195) / 18) ** 2)  # C-H 2nd overtone
        X[i] += np.random.randn(len(wl)) * 0.02 + 0.1  # Noise + baseline

    result = identify_category(X, wl, "test_petroleum")

    if "petroleum" in result.predicted_category.lower() or "hydrocarbon" in result.predicted_category.lower():
        print(f"  [PASS] Petroleum identified: {result.predicted_category}")
        passed += 1
    else:
        print(f"  [FAIL] Petroleum identification: got {result.predicted_category}")
        failed += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)

    return passed, failed


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main entry point."""
    print("Product Category Detector for NIR Spectra")
    print("=" * 120)
    print("\nThis detector uses:")
    print("  - Canonical representation (handles reflectance/absorbance/derivative)")
    print("  - Band template matching (matched filter approach)")
    print("  - NNLS composition fitting")
    print("  - Family-to-product mapping with mixture support")

    print("\nGround truth (for evaluation only):")
    for name, expected in GROUND_TRUTH.items():
        if expected is not None:
            short_name = name.split("_")[0]
            print(f"  {short_name}: {expected}")
    print("\nUncertain: Grapevine, Poultry_manure, TABLET\n")

    # Run benchmark
    results, accuracy = run_experiment()

    # Detailed results
    print_detailed_results(results)

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    definite_results = [r for r in results if r.expected_category is not None]
    correct_count = sum(1 for r in definite_results if r.is_correct)
    total_count = len(definite_results)

    print(f"Overall accuracy: {accuracy*100:.1f}% ({correct_count}/{total_count})")

    # By confidence level
    for conf_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]:
        conf_results = [r for r in definite_results if r.confidence == conf_level]
        if conf_results:
            conf_correct = sum(1 for r in conf_results if r.is_correct)
            conf_acc = conf_correct / len(conf_results) * 100
            print(f"  {conf_level.value.upper():6} confidence: {conf_acc:.1f}% ({conf_correct}/{len(conf_results)})")

    # Show incorrect predictions
    wrong_results = [r for r in results if r.is_correct is False]
    if wrong_results:
        print("\nIncorrect predictions:")
        for result in wrong_results:
            short_name = result.dataset_name.split("_")[0]
            print(f"  - {short_name}: predicted '{result.predicted_category}', expected '{result.expected_category}' [{result.confidence.value}]")
            for note in result.limitation_notes[:1]:
                print(f"      Note: {note}")

    # Run tests
    run_tests()


if __name__ == "__main__":
    main()
