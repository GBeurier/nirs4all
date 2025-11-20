"""
Optimized Multi-Layer Preprocessing for NIRS

PHILOSOPHY: Each layer should provide a DIFFERENT VIEW of the chemical information
- Minimize redundancy between layers
- Maximize complementary information
- Order matters in sequential preprocessing
"""
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer, Gaussian as Gauss,
    StandardNormalVariate as SNV, SavitzkyGolay as SavGol, Haar, MultiplicativeScatterCorrection as MSC,
    RobustStandardNormalVariate as RSNV, LocalStandardNormalVariate as LSNV, Wavelet
)

# ============================================================
# RECOMMENDED APPROACH (8-10 layers)
# ============================================================

preprocessing_layers_optimal = [
    # Layer 1: RAW BASELINE INFORMATION
    # Captures absolute absorbance, path length effects, overall intensity
    "raw_signal",

    # Layer 2: SCATTER-CORRECTED BASELINE
    # Removes multiplicative scatter but keeps baseline shape
    # MSC is better than SNV for baseline preservation
    MSC,  # or EMSC if you have it

    # Layer 3: NORMALIZED + SMOOTHED
    # Standard preprocessing - removes scatter + noise
    # This is your "safe" preprocessing
    [SNV, SavGol],  # NOT MSC+SNV! Pick one scatter correction

    # Layer 4: FIRST DERIVATIVE (CRITICAL!)
    # Removes baseline, enhances peaks, wavelength-shift invariant
    # This should be your STRONGEST signal for most NIRS tasks
    [MSC, FstDer],  # MSC first to reduce noise in derivative
    # Alternative: [SNV, SavGol_derivative(deriv=1)]

    # Layer 5: SECOND DERIVATIVE
    # Removes baseline + scatter in one step, resolves overlapping peaks
    # Very sensitive to noise, so preprocess carefully
    [SNV, SndDer],  # or [MSC, SavGol(deriv=2)]

    # Layer 6: SNV-DETREND COMBO (if you have it)
    # Combined scatter + baseline correction
    # Different from doing them separately
    [SNV, Detrend],  # Detrend AFTER SNV, not before

    # Layer 7: WAVELET DECOMPOSITION - High Frequency
    # Captures fine structure and noise patterns
    Wavelet('db4'),  # Daubechies-4: good for NIRS peaks

    # Layer 8: WAVELET DECOMPOSITION - Low Frequency
    # Captures baseline and broad absorption features
    Wavelet('sym5'),  # Symlet-5: good for smooth baseline

    # Layer 9: AREA NORMALIZATION (if relevant)
    # Normalizes total area under curve (good for concentration-independent features)
    [AreaNormalization, SavGol],  # Custom or use StandardScaler per spectrum

    # Layer 10: ROBUST SCATTER CORRECTION
    # Local scatter correction (good for heterogeneous samples)
    [RSNV, SavGol],  # or LSNV depending on your implementation
]

# Final scaling
final_scaler = RobustScaler()  # NOT MinMaxScaler(clip=True)!
y_scaler = StandardScaler()     # Consider StandardScaler for Y instead of MinMax


# ============================================================
# MINIMAL APPROACH (5 layers - if computational cost is high)
# ============================================================

preprocessing_layers_minimal = [
    # Layer 1: Raw baseline
    "raw_signal",

    # Layer 2: Standard preprocessing
    [MSC, SavGol],

    # Layer 3: First derivative (most important!)
    [SNV, FstDer],

    # Layer 4: Second derivative
    [MSC, SndDer],

    # Layer 5: Wavelet
    Wavelet('db6'),
]


# ============================================================
# TASK-SPECIFIC APPROACHES
# ============================================================

# For PROTEIN/NITROGEN prediction (N-H bonds, 2000-2200nm)
preprocessing_protein = [
    "raw_signal",
    MSC,
    [SNV, SavGol],
    [MSC, FstDer],  # Critical for protein
    [SNV, SndDer],   # Resolves amide peaks
    Wavelet('coif3'),
]

# For MOISTURE prediction (O-H bonds, 1400-1500nm, 1900-2000nm)
preprocessing_moisture = [
    "raw_signal",
    [MSC, SavGol],
    [SNV, FstDer],  # Water peaks show up strongly
    [MSC, SndDer],
    [RSNV, SavGol],  # Local scatter for heterogeneous moisture
    Wavelet('haar'),  # Sharp edges for water absorption
]

# For FAT/OIL prediction (C-H bonds, 1700-1800nm, 2300-2400nm)
preprocessing_fat = [
    "raw_signal",
    MSC,  # Important for fat scatter
    [SNV, SavGol],
    [MSC, FstDer],
    [SNV, SndDer],
    Wavelet('db8'),  # Smooth wavelets for fat peaks
    [AreaNormalization, MSC],  # Fat often needs area normalization
]


# ============================================================
# PREPROCESSING ORDER RULES (VERY IMPORTANT!)
# ============================================================

"""
CORRECT ORDER in sequential preprocessing:

1. DETREND (baseline removal)
   ↓
2. MSC or SNV (scatter correction)
   ↓
3. SAVGOL or SMOOTHING (noise reduction)
   ↓
4. DERIVATIVES (if using)

WRONG ORDERS (don't do these):
❌ [SNV, MSC] - both do scatter correction, redundant
❌ [SavGol, MSC] - MSC needs raw baseline, do MSC first
❌ [FstDer, MSC] - Derivative magnifies noise, smooth/correct first
❌ [SavGol, Detrend] - SavGol already removes trends
❌ [Detrend, SavGol, Detrend] - Redundant detrending

CORRECT EXAMPLES:
✓ [MSC, SavGol]
✓ [Detrend, SNV, SavGol]
✓ [MSC, SavGol, FstDer]
✓ [SNV, FstDer]
✓ [Detrend, MSC, SndDer]
"""


# ============================================================
# ADVANCED: WAVELENGTH-REGION-SPECIFIC PREPROCESSING
# ============================================================

def region_specific_preprocessing(spectrum, wavelengths):
    """
    Apply different preprocessing to different spectral regions.
    Example: Water bands need different treatment than protein bands.
    """
    # This is pseudocode - adapt to your framework

    # Region 1: 1100-1400nm (C-H, good SNR)
    region1 = spectrum[:, wavelengths < 1400]
    region1_processed = apply_preprocessing(region1, [SNV, FstDer])

    # Region 2: 1400-1600nm (water, O-H, high absorption)
    region2 = spectrum[:, (wavelengths >= 1400) & (wavelengths < 1600)]
    region2_processed = apply_preprocessing(region2, [MSC, SavGol])  # Don't use derivatives in high absorption

    # Region 3: 1600-2400nm (protein, fat, lower SNR)
    region3 = spectrum[:, wavelengths >= 1600]
    region3_processed = apply_preprocessing(region3, [RSNV, SndDer])

    return concatenate([region1_processed, region2_processed, region3_processed])


# ============================================================
# SCALING RECOMMENDATIONS
# ============================================================

"""
AFTER preprocessing layers, apply global scaling:

1. For features (X):
   - RobustScaler() if you have outliers (recommended)
   - StandardScaler() if data is clean
   - MinMaxScaler() only if model requires [0,1] range
   - NEVER use clip=True (hides problems)

2. For target (Y):
   - StandardScaler() for most cases
   - MinMaxScaler() if target is bounded (e.g., percentages)
   - Log transform if target is right-skewed
   - Consider quantile transform for non-normal distributions

3. Special cases:
   - If using sigmoid output: MinMaxScaler(feature_range=(0.1, 0.9))
     Avoids saturation at 0 and 1
   - If using tanh output: StandardScaler() or MinMaxScaler((-0.9, 0.9))
"""

from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer

# Recommended scaling
X_scaler = RobustScaler()  # Less sensitive to outliers than StandardScaler
y_scaler = StandardScaler()  # For normally distributed targets

# If your target is percentage (e.g., protein %)
y_scaler_percentage = MinMaxScaler(feature_range=(0.05, 0.95))  # Avoid exact 0/1

# If your target is right-skewed (e.g., fat content)
import numpy as np
y_log_transformed = np.log1p(y)  # Then use StandardScaler
y_scaler_skewed = StandardScaler()


# ============================================================
# VALIDATION: Check Your Preprocessing
# ============================================================

def validate_preprocessing_layers(layers):
    """
    Check if your preprocessing layers have too much redundancy.
    """
    issues = []

    # Check for scatter correction redundancy
    scatter_methods = ['SNV', 'MSC', 'LSNV', 'RSNV']
    scatter_count = sum([
        1 for layer in layers
        if isinstance(layer, list) and
        sum([any(s in str(p) for s in scatter_methods) for p in layer]) > 1
    ])
    if scatter_count > 0:
        issues.append("⚠️  Multiple scatter corrections in same pipeline (SNV+MSC)")

    # Check for detrending redundancy
    has_savgol_then_detrend = False
    for layer in layers:
        if isinstance(layer, list):
            has_savgol = any('SavGol' in str(p) for p in layer)
            has_detrend = any('Detrend' in str(p) for p in layer)
            if has_savgol and has_detrend:
                issues.append("⚠️  SavGol + Detrend in same pipeline (redundant)")

    # Check for derivative diversity
    has_fstder = any('FstDer' in str(layer) for layer in layers)
    has_sndder = any('SndDer' in str(layer) for layer in layers)
    if not has_fstder:
        issues.append("❌ Missing first derivative (critical for NIRS!)")
    if not has_sndder:
        issues.append("⚠️  Missing second derivative (good for overlapping peaks)")

    # Check wavelet diversity
    wavelet_count = sum(1 for layer in layers if 'Wavelet' in str(layer))
    if wavelet_count < 2:
        issues.append("⚠️  Only one wavelet family (try db4, sym5, coif3)")

    return issues


# Example usage
issues = validate_preprocessing_layers(preprocessing_layers_optimal)
for issue in issues:
    print(issue)




# ============================================================
# TIPS & TRICKS SUMMARY
# * Convertir d’abord en absorbance si tu pars en réflectance: (A=\log(1/R)). Beaucoup de corrections supposent l’absorbance.
# * “Detrend → MSC/SNV → lissage → dérivées” est correct, mais si tu utilises Savitzky-Golay **avec dérivée**, le lissage est déjà inclus. Évite “SavGol puis Detrend”.
# * Ne cumule jamais deux corrections de diffusion dans **une même** branche (SNV+MSC, MSC+RSNV, etc.).
# * Après concaténation multi-couches pour NN, fais un **z-score par canal** (pas un seul scaler global sur tout empilé). Évite de re-normaliser une couche déjà SNV.

# # Extensions à haute valeur

# ## Pré-étapes (avant les couches)

# * Harmonisation instrument: rééchantillonnage sur grille commune, **alignement spectral** (si léger décalage), masques bandes saturées (≈ 1350–1450 et 1850–1950 nm selon capteur).
# * Détection/filtrage d’outliers: leverage PLS rapide, Q-stat, ou IQR par bande.

# ## Familles de transformations (à piocher pour construire 6–12 canaux)

# ### Baseline / dérive

# * Detrend poly (ordre 1–2)
# * ALS / airPLS (baseline asymétrique)
# * Rubber-band / Top-hat morphologique

# ### Diffusion / scatter

# * SNV, MSC
# * **EMSC** (référence + termes polynomiaux ou eau)
# * **LSNV/RSNV** (locaux, fenêtre 10–50 points)
# * **PMSC** (piecewise MSC) si hétérogénéité forte

# ### Lissage / débruitage

# * Savitzky-Golay (fenêtre 11–25 pts à 2 nm, polyorder 2–3)
# * Norris–Williams
# * Whittaker–Eilers
# * Dénombrement ondelettes (seuils doux dBn/symN)

# ### Dérivées

# * SG dérivée 1 (pics, invariance décalage)
# * SG dérivée 2 (pics chevauchés, plus bruit)
# * Dérivée ondelette continue autour des bandes d’intérêt

# ### Normalisations

# * Vector/length (norme L2)
# * Area normalization (attention aux bandes d’eau)
# * Pareto / VAST si tu veux pondérer par variance

# ### Ondelettes multi-résolution

# * db4/sym5/coif3 détaillés (hautes fréquences)
# * db8/sym8 approximations (basses fréquences)
# * **Wavelet packet**: énergie par sous-bandes fixes

# ### Spécifiques NIRS

# * **Continuum removal** (formes de bandes)
# * **Kubelka–Munk** si besoin (peu courant en proche IR mais possible)
# * **Band-pass** physiques: masquages sélectifs ou moyenne locale par fenêtres centrées sur 1ᵉ/2ᵉ harmoniques CH, OH, NH

# ### Transfert/calibration (à utiliser avec prudence et CV stricte)

# * **OSC** (orthogonal à Y) et **EPO** (external parameter orthogonalization): risque de fuite si CV mal faite
# * **DS/PDS** (direct/piecewise direct standardization) entre instruments
# * **EMSC** avec spectre de référence du nouvel instrument

# ### Augmentations (pour NN)

# * Jitter multiplicatif/additif faible (scatter synthétique)
# * Jitter spectral (±1–2 pas) + légère ré-interpolation
# * Dropout de bandes étroites, bruit blanc calibré SNR

# # Collections prêtes à l’emploi (couches à concaténer en “channels” pour NN)

# ## Backbone général (8–10 canaux)

# 1. Raw absorbance
# 2. MSC
# 3. SNV + SG(lissage)
# 4. MSC + SG(deriv=1)
# 5. SNV + SG(deriv=2)
# 6. SNV + Detrend
# 7. Wavelet db4 (détails)
# 8. Wavelet sym5 (approximations)
# 9. Continuum removal + vector norm
# 10. RSNV local + SG(lissage)

# ## Minimal robuste (5 canaux)

# 1. Raw
# 2. MSC + SG(lissage)
# 3. SNV + SG(deriv=1)
# 4. MSC + SG(deriv=2)
# 5. Wavelet db6

# ## Protéines / azote

# Raw → MSC → SNV+SG → MSC+SG(d1) → SNV+SG(d2) → Wavelet coif3 → Continuum removal local autour 2050–2250 nm

# ## Eau / humidité

# Raw → MSC+SG → SNV+SG(d1) léger → MSC+SG(d2) prudent → RSNV local → Wavelet haar → Masques bandes saturées

# ## Lipides / huiles

# Raw → MSC → SNV+SG → MSC+SG(d1) → SNV+SG(d2) → Wavelet db8 → Area norm + MSC (si variation de quantité)

# ## Cellulose/lignine (matrices végétales)

# Raw → Detrend → MSC → SG(d1) → SG(d2) sur 1600–1800 et 2100–2350 nm → Wavelet sym8 → Vector norm

# ## Transfert entre instruments

# Raw → EMSC(réf.) → PDS ou LSNV → SG(d1) → Wavelet db4 → Vector norm
# (Tester aussi: Raw → DS → MSC → SG(d1))

# ## Prétraitement régional (multi-branche)

# * 1100–1400 nm: SNV + SG(d1)
# * 1400–1600 nm: MSC + SG(lissage), pas de dérivée si absorption forte
# * 1600–2400 nm: RSNV + SG(d2)
#   Concaténer les sorties. Option: pondération par SNR régional.

# # Règles d’ordre

# * Si dérivées SG: **MSC/SNV → SG(deriv=k)** suffit.
# * Detrend **avant** MSC/SNV si tu l’utilises; sinon inutile avec SG(d1/d2) bien paramétré.
# * Normalisations de type vector/area **après** corrections et dérivées.
# * Ondelettes en fin de pipeline de leur branche, sur signal déjà “propre”.

# # Paramètres de départ raisonnables

# * SG lissage: fenêtre 15–21, poly 2–3
# * SG d1: fenêtre 11–17, poly 2–3
# * SG d2: fenêtre 21–31, poly 3
# * RSNV/LSNV: fenêtre locale 25–75 points selon résolution
# * Ondelettes: 3–5 niveaux; seuillage doux universel ou SURE

# # Mise à l’échelle

# * **Par canal**: StandardScaler ou RobustScaler.
# * Évite de re-centrer une couche déjà SNV.
# * Y: StandardScaler pour régression non bornée. MinMax seulement si borné.

# # Contrôles de redondance et utilité

# * Redondance inter-canaux: RV-coefficient, angles de sous-espaces, corrélations de canaux.
# * Utilité vs Y: MI (mutual information) ou ΔRMSECV par ablation de canal.
# * Coût-bénéfice: bruit amplifié (facteur de bruit) des dérivées vs gain de résolution de pics.

# # Pièges courants

# * SNV **puis** scaler global sur toutes les features concaténées: double normalisation non souhaitée.
# * Area norm avec fortes bandes d’eau: biaise l’échelle.
# * OSC/EPO sans CV imbriquée: fuite d’info.

# # Conclusion

# Ton schéma est solide. Intègre EMSC, PDS/DS, ALS/airPLS, continuum removal, et un duo d’ondelettes complémentaires. Normalise par canal. Limite à 6–12 canaux utiles en mesurant redondance et gain prédictif.




Technique                          |  Main Usage/Effect
-----------------------------------+----------------------------------------------------------------------------------------------------------------------------------
Savitzky-Golay Smoothing           |  Reduces spectral noise; preserves peaks and features .
Derivative Spectroscopy            |  Removes baseline effects; resolves overlapping bands .
Standard Normal Variate (SNV)      |  Corrects scatter; normalizes each spectrum .
Multiplicative Scatter Correction  |  Removes scatter artifacts from particle size/path length .
Local/Robust SNV (LSNV, RNV)       |  Enhanced scatter correction for difficult outliers .
Detrending                         |  Removes global or polynomial trends in spectra .
Baseline Correction                |  Subtracts or fits baseline drift with polynomial or other methods .
Mean Centering/Autoscaling         |  Adjusts spectral features to centered/scaled form .
Normalization (e.g., area)         |  Adjusts all spectra to same overall intensity .
Wavelength Selection               |  Focuses analysis on most relevant regions .
Haar Wavelet Transform             |  Sometimes usedfor noise reduction and feature extraction; less common than above methods but useful in some advanced pipelines .
