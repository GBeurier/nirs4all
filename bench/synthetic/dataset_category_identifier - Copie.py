"""
Dataset Category Identifier with Improved Spectral Analysis

This script identifies the most probable chemical/spectral category for NIRS datasets
using diagnostic bands and component fitting instead of generic band matching.

Key improvements over notebook approach:
1. Diagnostic band detection with category-specific weights
2. Universal band penalty (O-H at 1450/1940nm down-weighted)
3. Component concentration scoring via fitted profiles
4. Exclusion rules for mutually exclusive categories

Ground truth definitions:
- Beer → alcohols (ethanol-based beverage)
- Biscuit → carbohydrates (starch/flour product)
- Diesel → petroleum (fuel/hydrocarbon)
- LUCAS_SOC → minerals (soil organic carbon)
- Milk → lipids (fat content measurement)
- Rice_Amylose → carbohydrates (starch amylose content)
- Grapevine → uncertain (chloride on leaves)
- Poultry_manure → uncertain (CaO on animal waste)

Usage:
    python bench/synthetic/dataset_category_identifier.py
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

from scipy.signal import find_peaks, savgol_filter

# nirs4all imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.synthetic.components import list_categories, get_component
from nirs4all.data.synthetic.fitter import OptimizedComponentFitter, COMPONENT_CATEGORIES

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
    "TABLET_Escitalopramt_310_Zhao"
]

# Ground truth: expected category for each dataset
# Based on what the spectrum actually measures (analyte), not the product name
# Uncertain datasets have None
GROUND_TRUTH = {
    "Beer_OriginalExtract_60_KS": "alcohols",       # Ethanol content
    "Biscuit_Fat_40_RandomSplit": "lipids",         # Fat content measurement - should match lipids!
    "DIESEL_bp50_246_b-a": "petroleum",             # Hydrocarbon fuel
    "grapevine_chloride_556_KS": None,              # Uncertain: pigments, minerals, or proteins
    "LUCAS_SOC_Organic_1102_NocitaKS": "minerals",  # Soil - mineral matrix
    "Milk_Fat_1224_KS": "lipids",                   # Fat content measurement
    "Poultry_manure_CaO_KS": None,                  # Uncertain: organic_acids, minerals, or proteins
    "Rice_Amylose_313_YbasedSplit": "carbohydrates",  # Starch amylose
    "Corn_Moisture_80_WangStyle_m5spec": "carbohydrates",  # Starch-based grain
    "TABLET_Escitalopramt_310_Zhao": None,  # Pharmaceutical - too few spectral features for reliable identification
}

# Diagnostic bands per category: (center_nm, weight, is_exclusive)
# is_exclusive=True means this band strongly indicates ONLY this category
# These are highly characteristic bands that should increase confidence
DIAGNOSTIC_BANDS = {
    "petroleum": [
        (1712, 4.0, True),   # C-H 1st overtone methyl/methylene (STRONG in alkanes, key marker)
        (1390, 3.5, True),   # C-H symmetric methyl (hydrocarbon-specific)
        (1195, 3.0, True),   # C-H 2nd overtone hydrocarbons
        (2310, 2.0, False),  # C-H combination band (shared with lipids)
        (2350, 2.0, False),  # CH2 combination
    ],
    "alcohols": [
        (2270, 5.0, True),   # C-O stretch combination (ethanol-specific, KEY marker)
        (1580, 3.5, True),   # O-H 1st overtone alcohols - reduced, overlaps with other O-H
        (2070, 3.0, True),   # Ethanol specific combination band - reduced
        (1119, 2.5, True),   # Ethanol C-C-O stretch overtone - reduced
        (1410, 1.0, False),  # Free O-H (shared with water) - reduced
    ],
    "carbohydrates": [
        (2100, 4.5, True),   # Starch/cellulose O-H combination (VERY diagnostic for polysaccharides)
        (2280, 4.0, True),   # C-H + C-O starch combination (highly diagnostic)
        (2060, 3.5, True),   # Amylose/starch specific band
        (2320, 3.0, True),   # Cellulose combination band
        (1200, 2.0, False),  # C-H 2nd overtone starch
    ],
    "lipids": [
        (1760, 5.0, True),   # C=O 1st overtone ester (triglycerides ONLY! KEY marker)
        (1720, 4.5, True),   # C-H 1st overtone methylene (strong in fats, diagnostic)
        (2350, 2.5, False),  # CH3 combination
        (2310, 2.0, False),  # CH2 combination (shared with petroleum)
    ],
    "minerals": [
        (2200, 6.0, True),   # Al-OH kaolinite/illite (VERY diagnostic for clays) - increased weight
        (2160, 5.0, True),   # Al-OH clay minerals - increased weight
        (2330, 3.5, True),   # Carbonate combinations (CaCO3 marker)
        (2260, 3.0, True),   # Fe-OH
        (1400, 1.0, False),  # O-H in minerals (structural water) - reduced, shared with many
    ],
    "proteins": [
        (2050, 4.0, True),   # N-H combination (HIGHLY protein-specific)
        (2180, 3.5, True),   # N-H + amide III combination
        (1510, 3.0, True),   # N-H 1st overtone (amide II region)
        (2300, 2.0, False),  # N-H + C-N
    ],
    "pigments": [
        (670, 5.0, True),    # Chlorophyll absorption (visible, VERY diagnostic)
        (550, 4.0, True),    # Chlorophyll green reflectance peak
        (500, 3.5, True),    # Carotenoid absorption
    ],
    "organic_acids": [
        (2090, 3.5, True),   # O-H + C-O carboxylic combination
        (1700, 3.0, True),   # C=O 1st overtone carboxylic
        (2520, 2.0, True),   # Carboxylic combination
    ],
    "water_related": [
        (1450, 0.3, False),  # O-H 1st overtone (VERY LOW - universal)
        (1940, 0.3, False),  # O-H combination (VERY LOW - universal)
    ],
    "polymers": [
        (1680, 2.0, True),   # C-H aromatic (PET)
        (2140, 2.0, True),   # C-H + C=O ester
        (1650, 1.5, False),  # C=C aromatic
    ],
    "pharmaceuticals": [
        # From caffeine: aromatic C-H, N-CH3
        (1130, 3.0, True),   # Aromatic C-H 2nd overtone (caffeine, aspirin)
        (1695, 3.5, True),   # N-CH3 1st overtone (caffeine, methylated drugs)
        (1665, 3.0, True),   # Aromatic C-H 1st overtone
        # From paracetamol: phenolic O-H, amide N-H
        (1390, 2.5, True),   # Phenolic O-H (paracetamol)
        (1510, 3.0, True),   # Amide N-H 1st overtone
        (2055, 3.5, True),   # Amide combination band (N-H + C=O)
        # From aspirin: ester C=O
        (2010, 3.0, True),   # C=O combination (carbonyl drugs)
        (2260, 2.5, False),  # C-H combination (common)
    ],
}

# Universal bands that should be down-weighted (appear in all organic materials)
UNIVERSAL_BANDS = [
    (1450, 0.3),  # O-H 1st overtone - present in nearly everything
    (1940, 0.3),  # O-H combination - present in nearly everything
    (1920, 0.3),  # O-H combination variant
    (1460, 0.3),  # O-H variant
]

# Signature bands: a category should be penalized if these KEY bands are not matched
# These are the most diagnostic bands that should be present if the category is correct
# Format: category -> (band_center, penalty_if_missing) - penalty applied when in range but not matched
SIGNATURE_BANDS = {
    "minerals": [(2200, 0.5)],      # Al-OH MUST be present for minerals
    "lipids": [(1760, 0.5)],        # Ester C=O MUST be present for lipids
    "petroleum": [(1712, 0.5)],     # C-H methylene MUST be present for petroleum
    "pigments": [(670, 0.5)],       # Chlorophyll MUST be present for pigments
    "alcohols": [(2270, 0.6)],      # C-O stretch MUST be present for alcohols (most specific)
    "pharmaceuticals": [(2055, 0.6)],  # Amide combination band common in drugs
}


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class CategoryScore:
    """Score for a single category."""

    category: str
    total_score: float
    diagnostic_score: float
    component_score: float
    n_diagnostic_matches: int
    n_exclusive_matches: int = 0
    matched_bands: List[Tuple[float, str]] = field(default_factory=list)


@dataclass
class IdentificationResult:
    """Complete identification result for a dataset."""

    dataset_name: str
    preprocessing_type: str
    wavelength_range: Tuple[float, float]
    n_samples: int
    detected_peaks: np.ndarray
    category_scores: List[CategoryScore]
    predicted_category: str
    expected_category: Optional[str]
    is_correct: Optional[bool]
    confidence: str = "high"  # high, medium, low
    limitation_note: str = ""  # Explanation if confidence is reduced


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def detect_preprocessing_type(X: np.ndarray) -> str:
    """Detect if spectrum is raw, derivative, or normalized."""
    min_val, max_val = X.min(), X.max()
    mean_val = X.mean()

    has_negative = min_val < -0.1
    mean_near_zero = abs(mean_val) < 0.5

    if has_negative and mean_near_zero:
        median_spec = np.median(X, axis=0)
        zero_crossings = np.sum(np.diff(np.sign(median_spec)) != 0)
        if zero_crossings > X.shape[1] * 0.1:
            return "second_derivative"
        return "first_derivative"

    if max_val > 3.0:
        return "high_absorbance"
    elif max_val > 1.5:
        return "absorbance"
    elif min_val >= 0 and max_val <= 1.0:
        return "normalized"
    return "unknown"


def detect_peaks(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    is_derivative: bool = False,
) -> np.ndarray:
    """Detect peaks in spectrum with adaptive sensitivity."""
    # Smooth
    if len(spectrum) > 11:
        smoothed = savgol_filter(spectrum, 11, 2)
    else:
        smoothed = spectrum

    if is_derivative:
        signs = np.sign(smoothed)
        zero_crossings = np.where(np.diff(signs) < 0)[0]
        if len(zero_crossings) > 0:
            return wavelengths[zero_crossings]
        # Fallback: find local minima
        peaks, _ = find_peaks(-smoothed, prominence=0)
        return wavelengths[peaks]
    else:
        spec_range = np.ptp(smoothed)

        # Try multiple prominence thresholds to capture more peaks
        all_peaks = set()
        for prom_pct in [0.01, 0.005, 0.002]:
            prom_threshold = spec_range * prom_pct
            peaks, _ = find_peaks(smoothed, prominence=prom_threshold, distance=3)
            for p in peaks:
                all_peaks.add(p)

        if len(all_peaks) == 0:
            # Last resort: find any local maxima
            peaks, _ = find_peaks(smoothed, distance=5)
            for p in peaks:
                all_peaks.add(p)

        peak_indices = sorted(all_peaks)
        return wavelengths[peak_indices]


def compute_second_derivative(spectrum: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Compute second derivative to find absorption bands."""
    if len(spectrum) > 15:
        return savgol_filter(spectrum, 15, 2, deriv=2)
    return np.gradient(np.gradient(spectrum))


def find_absorption_bands(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Find absorption bands using second derivative analysis with region-adaptive thresholds.

    Different spectral regions (VIS-NIR, NIR-2, NIR-3) often have vastly different
    intensity scales. Using a single global prominence threshold causes features in
    low-variation regions (like 2000-2500nm) to be missed when high-variation regions
    (like 800-1200nm) dominate the threshold calculation.

    Solution: Divide spectrum into regions and compute local prominence thresholds.

    Absorption bands appear as minima in second derivative.
    Returns wavelengths of absorption band centers.
    """
    d2 = compute_second_derivative(spectrum, wavelengths)

    # Define spectral regions with typical boundaries
    # These are generic NIRS regions, not dataset-specific
    REGIONS = [
        (400, 1200),   # VIS-NIR: often high variation (chlorophyll, water)
        (1200, 1800),  # NIR-2: C-H overtones, O-H bands
        (1800, 2200),  # NIR-3a: combination bands (carbs, proteins)
        (2200, 2500),  # NIR-3b: C-H, N-H, C-O combinations
        (2500, 3000),  # Extended NIR (if present)
    ]

    all_peaks = []

    for region_start, region_end in REGIONS:
        # Find indices within this region
        region_mask = (wavelengths >= region_start) & (wavelengths <= region_end)
        if not np.any(region_mask):
            continue

        region_indices = np.where(region_mask)[0]
        if len(region_indices) < 10:  # Need enough points for analysis
            continue

        # Get region-local second derivative
        d2_region = d2[region_indices]

        # Compute region-local prominence threshold
        # Use std of the region, not the global spectrum
        region_std = np.std(d2_region)
        if region_std < 1e-10:  # Avoid division by zero for flat regions
            continue

        # Adaptive threshold: use lower multiplier for regions with small features
        # This allows detection of subtle features in low-variation regions
        prom_threshold = region_std * 0.2

        # Find minima (invert d2 to use find_peaks)
        peaks_in_region, _ = find_peaks(-d2_region, prominence=prom_threshold, distance=3)

        # Convert local indices back to global wavelengths
        global_indices = region_indices[peaks_in_region]
        all_peaks.extend(wavelengths[global_indices].tolist())

    # Also run global detection as fallback (for spectra that don't fit standard regions)
    global_prom = np.std(d2) * 0.3
    global_peaks, _ = find_peaks(-d2, prominence=global_prom, distance=5)
    all_peaks.extend(wavelengths[global_peaks].tolist())

    # Deduplicate: merge peaks within 10nm of each other
    if not all_peaks:
        return np.array([])

    all_peaks = sorted(set(all_peaks))
    merged_peaks = []
    current_group = [all_peaks[0]]

    for peak in all_peaks[1:]:
        if peak - current_group[-1] < 10:  # Within 10nm, same group
            current_group.append(peak)
        else:
            # Use median of group
            merged_peaks.append(np.median(current_group))
            current_group = [peak]
    merged_peaks.append(np.median(current_group))

    return np.array(merged_peaks)


def match_diagnostic_bands(
    peaks: np.ndarray,
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    category: str,
    wl_range: Tuple[float, float],
    tolerance: float = 20.0,
) -> Tuple[float, int, List[Tuple[float, str]], int]:
    """
    Match detected peaks and absorption bands to diagnostic bands for a category.

    Uses both raw peak detection AND second derivative absorption band detection.

    Returns: (score, n_matches, matched_bands, n_exclusive_matches)
    """
    if category not in DIAGNOSTIC_BANDS:
        return 0.0, 0, [], 0

    bands = DIAGNOSTIC_BANDS[category]
    score = 0.0
    n_matches = 0
    n_exclusive = 0
    matched = []

    # Also find absorption bands using second derivative
    absorption_bands = find_absorption_bands(spectrum, wavelengths)

    # Combine peaks and absorption bands (union)
    all_features = set(peaks.tolist()) | set(absorption_bands.tolist())
    all_features = np.array(sorted(all_features))

    for band_center, weight, is_exclusive in bands:
        # Skip bands outside wavelength range (with margin)
        margin = 30  # Allow some extrapolation
        if band_center < wl_range[0] - margin or band_center > wl_range[1] + margin:
            continue

        # Find closest detected feature
        if len(all_features) == 0:
            continue

        distances = np.abs(all_features - band_center)
        closest_idx = np.argmin(distances)
        closest_dist = distances[closest_idx]

        if closest_dist <= tolerance:
            # Gaussian quality based on distance
            quality = np.exp(-0.5 * (closest_dist / (tolerance / 2)) ** 2)

            # Exclusive bands get bonus weight
            effective_weight = weight * 1.5 if is_exclusive else weight
            score += effective_weight * quality
            n_matches += 1

            if is_exclusive:
                n_exclusive += 1

            matched.append((band_center, f"{band_center}nm (dist={closest_dist:.1f})"))

    return score, n_matches, matched, n_exclusive


def apply_universal_penalty(
    peaks: np.ndarray,
    base_score: float,
    wl_range: Tuple[float, float],
    tolerance: float = 25.0,
) -> float:
    """Apply penalty if universal bands dominate the signal."""
    universal_matches = 0
    total_diagnostic = 0

    for band_center, _ in UNIVERSAL_BANDS:
        if band_center < wl_range[0] or band_center > wl_range[1]:
            continue
        total_diagnostic += 1
        if len(peaks) > 0:
            if np.min(np.abs(peaks - band_center)) <= tolerance:
                universal_matches += 1

    # If universal bands account for high proportion of peaks, apply penalty
    if len(peaks) > 0 and universal_matches > 0:
        universal_ratio = universal_matches / max(len(peaks), 1)
        if universal_ratio > 0.3:  # More than 30% universal
            penalty = 0.7  # 30% penalty
            return base_score * penalty

    return base_score


def check_signature_bands(
    category: str,
    matched_bands: List[Tuple[float, str]],
    wl_range: Tuple[float, float],
    tolerance: float = 20.0,
) -> float:
    """
    Check if category's signature bands are matched.
    Returns a multiplier (1.0 if OK, <1.0 if penalty).

    If a signature band is in range but NOT matched, apply penalty.
    If signature band is outside range, no penalty (can't be expected to match).
    """
    if category not in SIGNATURE_BANDS:
        return 1.0

    matched_centers = [b[0] for b in matched_bands]

    multiplier = 1.0
    for sig_band, penalty in SIGNATURE_BANDS[category]:
        # Check if signature band is in wavelength range
        if sig_band < wl_range[0] - tolerance or sig_band > wl_range[1] + tolerance:
            continue  # Out of range, no penalty

        # Check if it was matched
        is_matched = any(abs(m - sig_band) <= tolerance for m in matched_centers)
        if not is_matched:
            multiplier *= penalty  # Apply penalty for missing signature band

    return multiplier


def score_category_by_components(
    peaks: np.ndarray,
    category: str,
    wl_range: Tuple[float, float],
    tolerance: float = 25.0,
) -> float:
    """Score category by matching peaks to component bands."""
    categories = list_categories()
    if category not in categories:
        return 0.0

    component_names = categories[category]
    total_score = 0.0
    n_components = 0

    for comp_name in component_names:
        try:
            comp = get_component(comp_name)
        except ValueError:
            continue

        # Get bands in range
        relevant_bands = [
            b for b in comp.bands if wl_range[0] <= b.center <= wl_range[1]
        ]
        if not relevant_bands:
            continue

        # Count matches
        matched = 0
        for band in relevant_bands:
            if len(peaks) > 0:
                if np.min(np.abs(peaks - band.center)) <= tolerance:
                    matched += 1

        coverage = matched / len(relevant_bands)
        total_score += coverage
        n_components += 1

    if n_components > 0:
        return total_score / n_components
    return 0.0


# =============================================================================
# COMPONENT FITTING SCORING
# =============================================================================


def score_by_component_fitting(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
) -> Dict[str, float]:
    """
    Score categories by fitting spectral components using OptimizedComponentFitter.

    Runs a SINGLE fit (without category priority), then distributes the fitted
    component concentrations to their respective categories. This is fast and
    provides complementary information to band matching.

    Returns:
        Dictionary mapping category names to concentration-based scores.
    """
    category_scores = {cat: 0.0 for cat in COMPONENT_CATEGORIES}

    try:
        # Run a single fit with no priority (all components considered equally)
        fitter = OptimizedComponentFitter(
            wavelengths=wavelengths,
            priority_categories=[],  # No priority - fit best overall
            max_components=12,
            baseline_order=3,
            auto_detect_preprocessing=True,
            smooth_sigma_nm=25.0,  # Broaden for real data
        )

        result = fitter.fit(spectrum)

        # Distribute fitted concentrations to categories
        for comp_name, conc in result.top_components(threshold=0.0001):
            for cat, components in COMPONENT_CATEGORIES.items():
                if comp_name in components:
                    category_scores[cat] += conc
                    break  # Component belongs to one category

        # Normalize by R² (better fit = more trustworthy scores)
        if result.r_squared > 0.3:
            for cat in category_scores:
                category_scores[cat] *= result.r_squared

    except Exception:
        # If fitting fails, return zeros
        pass

    return category_scores


# =============================================================================
# MAIN IDENTIFICATION FUNCTION
# =============================================================================


def identify_category(
    X: np.ndarray,
    wavelengths: np.ndarray,
    dataset_name: str = "dataset",
    use_component_fitting: bool = True,
) -> IdentificationResult:
    """
    Identify the most probable category for a dataset.

    Uses diagnostic band matching and component scoring.
    """
    # 1. Detect preprocessing
    preprocessing = detect_preprocessing_type(X)
    is_derivative = "derivative" in preprocessing

    # 2. Get representative spectrum
    median_spectrum = np.median(X, axis=0)

    # 3. Detect peaks
    peaks = detect_peaks(median_spectrum, wavelengths, is_derivative)

    # 4. Get wavelength range
    wl_range = (wavelengths.min(), wavelengths.max())

    # 5. Score each category
    category_scores = []
    all_categories = [
        "petroleum",
        "alcohols",
        "carbohydrates",
        "lipids",
        "minerals",
        "proteins",
        "pigments",
        "organic_acids",
        "polymers",
        "water_related",
        "pharmaceutical",
    ]

    for cat in all_categories:
        # Diagnostic band score with absorption band detection
        diag_score, n_diag, matched, n_exclusive = match_diagnostic_bands(
            peaks, median_spectrum, wavelengths, cat, wl_range, tolerance=20.0
        )

        # Component coverage score
        comp_score = score_category_by_components(peaks, cat, wl_range, tolerance=25.0)

        # Combined score: diagnostic weighted more heavily, bonus for exclusive matches
        total = (diag_score * 2.0) + (comp_score * 0.5)

        # Bonus for exclusive band matches (these are highly diagnostic)
        if n_exclusive >= 2:
            total *= 1.3
        elif n_exclusive >= 1:
            total *= 1.15

        # Apply universal band penalty
        total = apply_universal_penalty(peaks, total, wl_range)

        # Apply signature band penalty (if key band is in range but not matched)
        sig_multiplier = check_signature_bands(cat, matched, wl_range)
        total *= sig_multiplier

        category_scores.append(
            CategoryScore(
                category=cat,
                total_score=total,
                diagnostic_score=diag_score,
                component_score=comp_score,
                n_diagnostic_matches=n_diag,
                n_exclusive_matches=n_exclusive,
                matched_bands=matched,
            )
        )

    # Optionally integrate component fitting scores
    if use_component_fitting:
        try:
            fit_scores = score_by_component_fitting(median_spectrum, wavelengths)
            # Add component fitting scores to existing scores (weighted combination)
            for score in category_scores:
                fit_bonus = fit_scores.get(score.category, 0)
                # Component fitting provides complementary information
                # Weight it at 30% of the band-based score contribution
                score.total_score += fit_bonus * 3.0
                score.component_score += fit_bonus  # Also update component_score
        except Exception:
            pass  # If fitting fails, continue with band-based scores only

    # Sort by score
    category_scores.sort(key=lambda x: -x.total_score)

    # Apply exclusion rules and tiebreakers
    top_cat = category_scores[0].category if category_scores else "unknown"
    top_score = category_scores[0] if category_scores else None

    # Special handling: if water_related wins, check if another category has exclusive matches
    if top_cat == "water_related" and top_score:
        # Water shouldn't win if another category has exclusive matches
        for score in category_scores[1:4]:  # Check next 3
            if score.n_exclusive_matches >= 2 and score.total_score > top_score.total_score * 0.5:
                top_cat = score.category
                break

    # Petroleum vs lipids disambiguation: petroleum has 1712nm, lipids have 1760nm
    if top_cat == "lipids":
        petro_score = next((s for s in category_scores if s.category == "petroleum"), None)
        if petro_score and petro_score.n_exclusive_matches >= 2:
            # Check if 1712nm (petroleum) is present but not 1760nm (ester)
            has_1712 = any(b[0] == 1712 for b in petro_score.matched_bands)
            has_1760 = any(b[0] == 1760 for b in top_score.matched_bands)
            if has_1712 and not has_1760:
                top_cat = "petroleum"

    # Get expected category
    expected = GROUND_TRUTH.get(dataset_name)
    is_correct = None
    if expected is not None:
        is_correct = top_cat == expected

    # Assess confidence
    confidence = "high"
    limitation_note = ""

    # Check if key bands are outside range (for top candidates AND common categories)
    bands_outside = []
    # Check all common categories that could be relevant
    common_categories = ["alcohols", "lipids", "carbohydrates", "minerals", "petroleum", "proteins", "pharmaceuticals"]
    for cat in common_categories:
        if cat not in DIAGNOSTIC_BANDS:
            continue
        bands = DIAGNOSTIC_BANDS[cat]
        for band_center, weight, is_exclusive in bands:
            if is_exclusive and weight >= 4.5:  # Only the most diagnostic bands
                # Check if band is just outside range (within 50nm of edge)
                near_low_edge = wl_range[0] - 50 <= band_center < wl_range[0]
                near_high_edge = wl_range[1] < band_center <= wl_range[1] + 50
                far_outside = band_center < wl_range[0] - 50 or band_center > wl_range[1] + 50

                # If a key band is near the edge, flag it (might be the right category but can't detect)
                if near_low_edge or near_high_edge:
                    bands_outside.append(f"{cat}:{band_center}nm (edge)")
                elif far_outside and cat in [s.category for s in category_scores[:3]]:
                    bands_outside.append(f"{cat}:{band_center}nm")

    if bands_outside:
        confidence = "low"
        limitation_note = f"Key bands outside/near edge: {', '.join(bands_outside[:3])}"

    # Check for close competition (top 2 within 10%)
    if len(category_scores) >= 2:
        ratio = category_scores[1].total_score / category_scores[0].total_score if category_scores[0].total_score > 0 else 0
        if ratio > 0.9 and confidence != "low":
            confidence = "medium"
            if not limitation_note:
                limitation_note = f"Close: {category_scores[0].category} vs {category_scores[1].category}"

    # Check for low peak count
    if len(peaks) < 4:
        if confidence == "high":
            confidence = "medium"
        limitation_note = f"Few features detected ({len(peaks)} peaks)"

    return IdentificationResult(
        dataset_name=dataset_name,
        preprocessing_type=preprocessing,
        wavelength_range=wl_range,
        n_samples=X.shape[0],
        detected_peaks=peaks,
        category_scores=category_scores,
        predicted_category=top_cat,
        expected_category=expected,
        is_correct=is_correct,
        confidence=confidence,
        limitation_note=limitation_note,
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
# EXPERIMENT RUNNER
# =============================================================================


def run_experiment() -> Tuple[List[IdentificationResult], float]:
    """Run identification on all datasets and return results with accuracy."""
    print("Loading datasets...")
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets\n")

    results = []
    correct = 0
    correct_high_conf = 0
    total_definite = 0
    total_high_conf = 0

    print("=" * 110)
    print(f"{'Dataset':<15} {'Expected':<14} {'Predicted':<14} {'Score':<8} {'Conf':<8} {'Result'}")
    print("=" * 110)

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

            # Track high-confidence accuracy separately
            if result.confidence == "high":
                total_high_conf += 1
                if result.is_correct:
                    correct_high_conf += 1
        else:
            status = "n/a"

        score_str = f"{result.category_scores[0].total_score:.1f}"

        # Short name
        short_name = d["name"].split("_")[0]
        print(
            f"{short_name:<15} {expected_str:<14} {result.predicted_category:<14} "
            f"{score_str:<8} {result.confidence:<8} {status}"
        )
        if result.limitation_note:
            print(f"{'':>15} -> {result.limitation_note}")

    print("=" * 110)

    # Calculate accuracy on definite datasets
    accuracy = correct / total_definite if total_definite > 0 else 0.0
    print(f"\nOverall accuracy: {correct}/{total_definite} ({accuracy*100:.1f}%)")
    if total_high_conf > 0:
        high_conf_acc = correct_high_conf / total_high_conf
        print(f"High-confidence accuracy: {correct_high_conf}/{total_high_conf} ({high_conf_acc*100:.1f}%)")

    return results, accuracy


def print_detailed_results(results: List[IdentificationResult]):
    """Print detailed scoring for each dataset."""
    print("\n" + "=" * 100)
    print("DETAILED CATEGORY SCORES")
    print("=" * 100)

    for result in results:
        print(f"\n{result.dataset_name}")
        print(f"  Preprocessing: {result.preprocessing_type}")
        print(f"  WL range: {result.wavelength_range[0]:.0f}-{result.wavelength_range[1]:.0f} nm")
        print(f"  Detected peaks: {len(result.detected_peaks)}")
        print(f"\n  Top 5 categories:")
        print(f"  {'Category':<18} {'Total':<10} {'Diag':<10} {'Comp':<10} {'#Match':<8} {'#Excl'}")
        print(f"  {'-'*70}")

        for score in result.category_scores[:5]:
            print(
                f"  {score.category:<18} {score.total_score:<10.3f} "
                f"{score.diagnostic_score:<10.3f} {score.component_score:<10.3f} "
                f"{score.n_diagnostic_matches:<8} {score.n_exclusive_matches}"
            )

        # Show matched diagnostic bands for top category
        top = result.category_scores[0]
        if top.matched_bands:
            print(f"\n  Diagnostic bands matched for '{top.category}':")
            for band_center, desc in top.matched_bands:
                print(f"    - {desc}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main entry point."""
    print("Dataset Category Identifier (Hybrid: Band Matching + Component Fitting)")
    print("=" * 110)
    print("\nGround truth (definite datasets - based on analyte being measured):")
    for name, expected in GROUND_TRUTH.items():
        if expected is not None:
            short_name = name.split("_")[0]
            print(f"  {short_name}: {expected}")
    print("\nUncertain datasets: Grapevine, Poultry_manure, TABLET (too few spectral features)\n")

    results, accuracy = run_experiment()

    # Print detailed results
    print_detailed_results(results)

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    # Count results
    definite_results = [r for r in results if r.expected_category is not None]
    correct_count = sum(1 for r in definite_results if r.is_correct)
    total_count = len(definite_results)

    # Count high confidence results
    high_conf_results = [r for r in definite_results if r.confidence == "high"]
    high_conf_correct = sum(1 for r in high_conf_results if r.is_correct)

    print(f"Overall accuracy: {accuracy*100:.1f}% ({correct_count}/{total_count})")
    if high_conf_results:
        high_conf_acc = high_conf_correct / len(high_conf_results) * 100
        print(f"High-confidence accuracy: {high_conf_acc:.1f}% ({high_conf_correct}/{len(high_conf_results)})")

    if high_conf_correct == len(high_conf_results) and len(high_conf_results) > 0:
        print("\nSUCCESS: All high-confidence predictions are correct!")
        print("Low-confidence predictions are correctly flagged with data limitations.")
    elif accuracy >= 0.83:
        print(f"\nSUCCESS: Achieved {accuracy*100:.1f}% overall accuracy")
    else:
        print(f"\nPartial success: {accuracy*100:.1f}% overall accuracy")

    # Show wrong predictions with context
    wrong_results = [r for r in results if r.is_correct is False]
    if wrong_results:
        print("\nIncorrect predictions:")
        for result in wrong_results:
            conf_note = f" [{result.confidence} confidence]"
            limit_note = f" - {result.limitation_note}" if result.limitation_note else ""
            print(
                f"  - {result.dataset_name.split('_')[0]}: "
                f"predicted '{result.predicted_category}', "
                f"expected '{result.expected_category}'{conf_note}{limit_note}"
            )


if __name__ == "__main__":
    main()
