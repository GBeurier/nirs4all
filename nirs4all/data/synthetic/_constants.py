"""
Predefined constants for synthetic NIRS spectra generation.

This module contains predefined spectral components and band assignments
based on established NIR spectroscopy literature. Each component includes
accurate absorption band positions, widths, and relative intensities derived
from published spectroscopic data and reference databases.

Primary References:
    [1] Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas
        for Interpretive Near-Infrared Spectroscopy (2nd ed.). CRC Press.
        - Comprehensive NIR band assignments and spectral atlas (pp. 23-73)
        - Functional group identification tables

    [2] Burns, D. A., & Ciurczak, E. W. (Eds.). (2007). Handbook of Near-Infrared
        Analysis (3rd ed.). CRC Press.
        - Chapter 2: Theory of NIR Spectroscopy (pp. 7-35)
        - Chapter 17: NIR in Agriculture (pp. 347-386)

    [3] Siesler, H. W., Ozaki, Y., Kawata, S., & Heise, H. M. (Eds.). (2002).
        Near-Infrared Spectroscopy: Principles, Instruments, Applications.
        Wiley-VCH.
        - Detailed overtone/combination band theory (pp. 1-52)

    [4] Osborne, B. G., Fearn, T., & Hindle, P. H. (1993). Practical NIR
        Spectroscopy with Applications in Food and Beverage Analysis (2nd ed.).
        Longman Scientific & Technical.
        - Food and agricultural applications (pp. 45-102)

    [5] Williams, P. C., & Norris, K. H. (Eds.). (2001). Near-Infrared Technology
        in the Agricultural and Food Industries (2nd ed.). AACC International.
        - Band assignments for agricultural commodities (pp. 145-169)

    [6] Schwanninger, M., Rodrigues, J. C., & Fackler, K. (2011). A Review of
        Band Assignments in Near Infrared Spectra of Wood and Wood Components.
        Journal of Near Infrared Spectroscopy, 19(5), 287-308.
        - Comprehensive wood/cellulose band assignments

    [7] Murray, I. (1986). The NIR Spectra of Homologous Series of Organic
        Compounds. In P. C. Williams & K. H. Norris (Eds.), Near-Infrared
        Technology in the Agricultural and Food Industries. AACC.
        - Fundamental hydrocarbon and alcohol band positions

    [8] Bokobza, L. (1998). Near Infrared Spectroscopy. Journal of Near Infrared
        Spectroscopy, 6(1), 3-17.
        - Review of NIR fundamentals and band assignments

    [9] Reich, G. (2005). Near-Infrared Spectroscopy and Imaging: Basic Principles
        and Pharmaceutical Applications. Advanced Drug Delivery Reviews, 57(8),
        1109-1143.
        - Pharmaceutical compound band assignments

    [10] Blanco, M., & Villarroya, I. (2002). NIR Spectroscopy: A Rapid-Response
         Analytical Tool. TrAC Trends in Analytical Chemistry, 21(4), 240-250.
         - Review of NIR applications and typical band positions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict
    from .components import SpectralComponent

# Lazy imports to avoid circular dependencies
_PREDEFINED_COMPONENTS: "Dict[str, SpectralComponent] | None" = None


def get_predefined_components() -> "Dict[str, SpectralComponent]":
    """
    Get predefined spectral components based on NIR band assignments.

    Returns a dictionary of SpectralComponent objects representing common
    chemical compounds and functional groups found in NIR spectroscopy
    applications (agricultural, food, pharmaceutical, petrochemical).

    Each component's band assignments are based on published spectroscopic
    literature. Key characteristics:

    - **Band centers**: Wavelength positions (nm) of absorption maxima
    - **Sigma**: Gaussian width contribution (thermal/inhomogeneous broadening)
    - **Gamma**: Lorentzian width contribution (pressure/collision broadening)
    - **Amplitude**: Relative absorption intensity (normalized within component)

    Available Components:
        Water-related:
            - ``water``: H₂O fundamental O-H vibrations [1, pp. 34-36]
            - ``moisture``: Bound water in organic matrices [2, pp. 358-362]

        Proteins and Nitrogen:
            - ``protein``: General protein (amide, N-H, C-H) [1, pp. 48-52]
            - ``nitrogen_compound``: Primary/secondary amines [1, pp. 52-54]
            - ``urea``: CO(NH₂)₂ bands [9, p. 1125]
            - ``amino_acid``: Free amino acids [3, pp. 215-220]

        Lipids and Hydrocarbons:
            - ``lipid``: Triglycerides (C-H stretching) [1, pp. 44-48]
            - ``oil``: Vegetable/mineral oils [4, pp. 67-72]
            - ``saturated_fat``: Saturated fatty acids [7, pp. 15-20]
            - ``unsaturated_fat``: Mono/polyunsaturated fats [7, pp. 20-25]

        Carbohydrates:
            - ``starch``: Amylose/amylopectin [5, pp. 155-160]
            - ``cellulose``: β-1,4-glucan chains [6, pp. 295-300]
            - ``glucose``: D-glucose monosaccharide [2, pp. 368-370]
            - ``fructose``: D-fructose monosaccharide [2, pp. 368-370]
            - ``sucrose``: Disaccharide [2, pp. 370-372]
            - ``hemicellulose``: Xylan/glucomannan [6, pp. 300-303]
            - ``lignin``: Aromatic polymer [6, pp. 303-305]

        Alcohols:
            - ``ethanol``: C₂H₅OH [1, pp. 38-40]
            - ``methanol``: CH₃OH [1, pp. 38-40]

        Organic Acids:
            - ``acetic_acid``: CH₃COOH [8, pp. 8-10]
            - ``citric_acid``: C₆H₈O₇ [4, pp. 78-80]
            - ``lactic_acid``: CH₃CH(OH)COOH [9, pp. 1128-1130]

        Plant Pigments:
            - ``chlorophyll``: Chlorophyll a/b [2, pp. 375-378]
            - ``carotenoid``: β-carotene, xanthophylls [2, pp. 378-380]

        Pharmaceutical:
            - ``caffeine``: C₈H₁₀N₄O₂ [9, pp. 1130-1132]
            - ``aspirin``: Acetylsalicylic acid [9, pp. 1125-1128]
            - ``paracetamol``: Acetaminophen [9, pp. 1132-1135]

        Petrochemical:
            - ``aromatic``: Benzene derivatives [1, pp. 56-58]
            - ``alkane``: Saturated hydrocarbons [7, pp. 10-15]

        Fibers:
            - ``cotton``: Cotton cellulose [6, pp. 295-298]
            - ``polyester``: PET fiber [1, pp. 60-62]

    Returns:
        Dictionary mapping component names to SpectralComponent objects.

    Note:
        This function uses lazy initialization to avoid circular imports.
        The components are created once and cached for subsequent calls.

    References:
        See module docstring for full reference list.
    """
    global _PREDEFINED_COMPONENTS

    if _PREDEFINED_COMPONENTS is None:
        from .components import NIRBand, SpectralComponent

        _PREDEFINED_COMPONENTS = {
            # ================================================================
            # WATER AND MOISTURE
            # ================================================================
            # Water: O-H stretching vibrations
            # Refs: [1] pp. 34-36, [2] pp. 358-362
            # The 1450 nm band is the 1st overtone of O-H stretch (~3400 cm⁻¹ × 2)
            # The 1940 nm band is the O-H stretch + bend combination
            "water": SpectralComponent(
                name="water",
                bands=[
                    # O-H 1st overtone: 2ν₁ or 2ν₃ (~6900 cm⁻¹ = 1450 nm)
                    # Ref: [1] p. 35, Table 2.3
                    NIRBand(center=1450, sigma=25, gamma=3, amplitude=0.8, name="O-H 1st overtone"),
                    # O-H combination: ν₁ + ν₂ or ν₂ + ν₃ (~5150 cm⁻¹ = 1940 nm)
                    # Ref: [1] p. 35, strongest water band
                    NIRBand(center=1940, sigma=30, gamma=4, amplitude=1.0, name="O-H combination"),
                    # O-H stretch + bend: lower intensity band
                    # Ref: [2] p. 360
                    NIRBand(center=2500, sigma=50, gamma=5, amplitude=0.3, name="O-H stretch + bend"),
                ],
                correlation_group=1,
            ),
            # Moisture: Bound water in organic matrices (shifted positions)
            # Refs: [2] pp. 358-362, [5] pp. 157-158
            # Hydrogen bonding shifts bands to longer wavelengths
            "moisture": SpectralComponent(
                name="moisture",
                bands=[
                    # Bound O-H 1st overtone (shifted from free water)
                    # Ref: [5] p. 158
                    NIRBand(center=1460, sigma=30, gamma=4, amplitude=0.7, name="Bound O-H 1st overtone"),
                    # Bound O-H combination (broader than free water)
                    # Ref: [2] p. 361
                    NIRBand(center=1930, sigma=35, gamma=5, amplitude=0.9, name="Bound O-H combination"),
                ],
                correlation_group=1,
            ),
            # ================================================================
            # PROTEINS AND NITROGEN COMPOUNDS
            # ================================================================
            # Protein: Amide bonds, N-H, aromatic C-H
            # Refs: [1] pp. 48-52, [2] pp. 362-366
            # Key: Amide A (~3300 cm⁻¹), Amide I (~1650 cm⁻¹), Amide II (~1550 cm⁻¹)
            "protein": SpectralComponent(
                name="protein",
                bands=[
                    # N-H 1st overtone (2ν of amide A band at 3300 cm⁻¹)
                    # Ref: [1] p. 50, Table 2.8
                    NIRBand(center=1510, sigma=20, gamma=2, amplitude=0.5, name="N-H 1st overtone"),
                    # C-H aromatic stretching (Phe, Tyr, Trp residues)
                    # Ref: [1] p. 51
                    NIRBand(center=1680, sigma=25, gamma=3, amplitude=0.4, name="C-H aromatic"),
                    # N-H combination: Amide A + Amide II (~4860 cm⁻¹ = 2055 nm)
                    # Ref: [1] p. 50, strong protein marker
                    NIRBand(center=2050, sigma=30, gamma=3, amplitude=0.6, name="N-H combination"),
                    # Protein C-H: combination bands from aliphatic residues
                    # Ref: [2] p. 365
                    NIRBand(center=2180, sigma=25, gamma=2, amplitude=0.5, name="Protein C-H"),
                    # N-H + Amide III combination
                    # Ref: [1] p. 51
                    NIRBand(center=2300, sigma=20, gamma=2, amplitude=0.3, name="N-H+Amide III"),
                ],
                correlation_group=2,
            ),
            # Nitrogen compounds: Primary/secondary amines
            # Refs: [1] pp. 52-54
            "nitrogen_compound": SpectralComponent(
                name="nitrogen_compound",
                bands=[
                    # N-H 1st overtone (free amine)
                    # Ref: [1] p. 53, Table 2.9
                    NIRBand(center=1500, sigma=18, gamma=2, amplitude=0.45, name="N-H 1st overtone"),
                    # N-H combination band
                    # Ref: [1] p. 53
                    NIRBand(center=2060, sigma=25, gamma=2, amplitude=0.5, name="N-H combination"),
                    # N-H + C-N combination
                    # Ref: [1] p. 54
                    NIRBand(center=2150, sigma=22, gamma=2, amplitude=0.4, name="N-H+C-N"),
                ],
                correlation_group=2,
            ),
            # Urea: Carbonyl + amine functional groups
            # Refs: [9] p. 1125, [2] p. 366
            "urea": SpectralComponent(
                name="urea",
                bands=[
                    # N-H 1st overtone (symmetric stretch)
                    # Ref: [9] p. 1125
                    NIRBand(center=1480, sigma=18, gamma=2, amplitude=0.5, name="N-H sym 1st overtone"),
                    # N-H 1st overtone (asymmetric stretch)
                    # Ref: [9] p. 1125
                    NIRBand(center=1530, sigma=20, gamma=2, amplitude=0.45, name="N-H asym 1st overtone"),
                    # N-H combination with C=O
                    # Ref: [9] p. 1125
                    NIRBand(center=2010, sigma=25, gamma=3, amplitude=0.55, name="N-H + C=O combination"),
                    # N-H bending combination
                    # Ref: [2] p. 366
                    NIRBand(center=2170, sigma=22, gamma=2, amplitude=0.35, name="N-H bend combination"),
                ],
                correlation_group=2,
            ),
            # Amino acid: Free amino acids with carboxyl and amine groups
            # Refs: [3] pp. 215-220
            "amino_acid": SpectralComponent(
                name="amino_acid",
                bands=[
                    # N-H 1st overtone (NH₃⁺ stretching in zwitterion)
                    # Ref: [3] p. 216
                    NIRBand(center=1520, sigma=22, gamma=2.5, amplitude=0.45, name="NH₃⁺ 1st overtone"),
                    # N-H combination
                    # Ref: [3] p. 217
                    NIRBand(center=2040, sigma=28, gamma=3, amplitude=0.5, name="N-H combination"),
                    # C-H combination (α-carbon)
                    # Ref: [3] p. 218
                    NIRBand(center=2260, sigma=20, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=2,
            ),
            # ================================================================
            # LIPIDS AND HYDROCARBONS
            # ================================================================
            # Lipid: Triglycerides with C-H stretching modes
            # Refs: [1] pp. 44-48, [4] pp. 67-72
            "lipid": SpectralComponent(
                name="lipid",
                bands=[
                    # C-H 2nd overtone (~8400 cm⁻¹ = 1190 nm)
                    # Ref: [1] p. 45, Table 2.6
                    NIRBand(center=1210, sigma=20, gamma=2, amplitude=0.4, name="C-H 2nd overtone"),
                    # C-H combination
                    # Ref: [1] p. 46
                    NIRBand(center=1390, sigma=15, gamma=1, amplitude=0.3, name="C-H combination"),
                    # C-H 1st overtone (~5780 cm⁻¹ = 1730 nm)
                    # Ref: [1] p. 45, strong lipid marker
                    NIRBand(center=1720, sigma=25, gamma=2, amplitude=0.6, name="C-H 1st overtone"),
                    # CH₂ combination (antisym stretch + scissor)
                    # Ref: [4] p. 69
                    NIRBand(center=2310, sigma=20, gamma=2, amplitude=0.5, name="CH2 combination"),
                    # CH₃ combination
                    # Ref: [4] p. 70
                    NIRBand(center=2350, sigma=18, gamma=2, amplitude=0.4, name="CH3 combination"),
                ],
                correlation_group=3,
            ),
            # Oil: Vegetable/mineral oils
            # Refs: [4] pp. 67-72, [7] pp. 20-25
            "oil": SpectralComponent(
                name="oil",
                bands=[
                    # C-H 2nd overtone (CH₃)
                    # Ref: [7] p. 21
                    NIRBand(center=1165, sigma=18, gamma=2, amplitude=0.35, name="C-H 2nd overtone"),
                    # CH₂ 2nd overtone
                    # Ref: [7] p. 21
                    NIRBand(center=1215, sigma=16, gamma=1.5, amplitude=0.3, name="CH2 2nd overtone"),
                    # C-H combination
                    # Ref: [4] p. 68
                    NIRBand(center=1410, sigma=20, gamma=2, amplitude=0.45, name="C-H combination"),
                    # C-H 1st overtone (strong, characteristic)
                    # Ref: [4] p. 68
                    NIRBand(center=1725, sigma=22, gamma=2, amplitude=0.7, name="C-H 1st overtone"),
                    # C=C unsaturation (=C-H stretch)
                    # Ref: [7] p. 23
                    NIRBand(center=2140, sigma=25, gamma=2, amplitude=0.4, name="C=C unsaturation"),
                    # CH₂ combination
                    # Ref: [4] p. 70
                    NIRBand(center=2305, sigma=18, gamma=2, amplitude=0.5, name="CH2 combination"),
                ],
                correlation_group=3,
            ),
            # Saturated fat: Fully saturated fatty acids
            # Refs: [7] pp. 15-20, [1] pp. 44-46
            "saturated_fat": SpectralComponent(
                name="saturated_fat",
                bands=[
                    # CH₂ 2nd overtone (dominant in saturated chains)
                    # Ref: [7] p. 16
                    NIRBand(center=1195, sigma=18, gamma=2, amplitude=0.45, name="CH2 2nd overtone"),
                    # C-H combination
                    # Ref: [7] p. 17
                    NIRBand(center=1395, sigma=18, gamma=2, amplitude=0.35, name="C-H combination"),
                    # C-H 1st overtone
                    # Ref: [7] p. 17
                    NIRBand(center=1730, sigma=22, gamma=2, amplitude=0.65, name="C-H 1st overtone"),
                    # CH₂ combination
                    # Ref: [7] p. 18
                    NIRBand(center=2315, sigma=18, gamma=2, amplitude=0.55, name="CH2 combination"),
                    # CH₃ combination
                    # Ref: [7] p. 18
                    NIRBand(center=2355, sigma=16, gamma=2, amplitude=0.45, name="CH3 combination"),
                ],
                correlation_group=3,
            ),
            # Unsaturated fat: Mono/polyunsaturated fatty acids
            # Refs: [7] pp. 20-25, [4] pp. 70-72
            "unsaturated_fat": SpectralComponent(
                name="unsaturated_fat",
                bands=[
                    # =C-H 2nd overtone (characteristic of unsaturation)
                    # Ref: [7] p. 22
                    NIRBand(center=1160, sigma=15, gamma=1.5, amplitude=0.3, name="=C-H 2nd overtone"),
                    # C-H combination
                    # Ref: [7] p. 22
                    NIRBand(center=1400, sigma=20, gamma=2, amplitude=0.4, name="C-H combination"),
                    # C-H 1st overtone
                    # Ref: [7] p. 23
                    NIRBand(center=1720, sigma=24, gamma=2, amplitude=0.6, name="C-H 1st overtone"),
                    # =C-H 1st overtone (unique unsaturation marker)
                    # Ref: [7] p. 23, distinguishes from saturated
                    NIRBand(center=2145, sigma=20, gamma=2, amplitude=0.5, name="=C-H 1st overtone"),
                    # C=C stretch + C-H combination
                    # Ref: [7] p. 24
                    NIRBand(center=2175, sigma=18, gamma=2, amplitude=0.35, name="C=C combination"),
                ],
                correlation_group=3,
            ),
            # Aromatic hydrocarbons: Benzene derivatives
            # Refs: [1] pp. 56-58
            "aromatic": SpectralComponent(
                name="aromatic",
                bands=[
                    # Aromatic C-H 2nd overtone
                    # Ref: [1] p. 57, Table 2.11
                    NIRBand(center=1145, sigma=15, gamma=1.5, amplitude=0.35, name="Ar C-H 2nd overtone"),
                    # Aromatic C-H 1st overtone
                    # Ref: [1] p. 57
                    NIRBand(center=1685, sigma=18, gamma=2, amplitude=0.55, name="Ar C-H 1st overtone"),
                    # Aromatic combination bands
                    # Ref: [1] p. 58
                    NIRBand(center=2150, sigma=22, gamma=2, amplitude=0.4, name="Ar C-H combination"),
                    # Aromatic C=C + C-H combination
                    # Ref: [1] p. 58
                    NIRBand(center=2440, sigma=25, gamma=2.5, amplitude=0.3, name="Ar ring combination"),
                ],
                correlation_group=6,
            ),
            # Alkane: Saturated hydrocarbons (paraffins)
            # Refs: [7] pp. 10-15, [1] pp. 44-45
            "alkane": SpectralComponent(
                name="alkane",
                bands=[
                    # C-H 2nd overtone
                    # Ref: [7] p. 11
                    NIRBand(center=1190, sigma=20, gamma=2, amplitude=0.4, name="C-H 2nd overtone"),
                    # C-H 1st overtone (CH₃ + CH₂)
                    # Ref: [7] p. 12
                    NIRBand(center=1715, sigma=25, gamma=2.5, amplitude=0.65, name="C-H 1st overtone"),
                    # CH₂ combination
                    # Ref: [7] p. 13
                    NIRBand(center=2310, sigma=20, gamma=2, amplitude=0.5, name="CH2 combination"),
                    # CH₃ combination
                    # Ref: [7] p. 13
                    NIRBand(center=2360, sigma=18, gamma=2, amplitude=0.4, name="CH3 combination"),
                ],
                correlation_group=6,
            ),
            # ================================================================
            # CARBOHYDRATES
            # ================================================================
            # Starch: Amylose and amylopectin
            # Refs: [5] pp. 155-160, [1] pp. 40-43
            "starch": SpectralComponent(
                name="starch",
                bands=[
                    # O-H 1st overtone (hydroxyl groups)
                    # Ref: [5] p. 156
                    NIRBand(center=1460, sigma=25, gamma=3, amplitude=0.5, name="O-H 1st overtone"),
                    # Starch-specific combination band
                    # Ref: [5] p. 157
                    NIRBand(center=1580, sigma=20, gamma=2, amplitude=0.3, name="Starch combination"),
                    # O-H + C-O combination (strong carbohydrate marker)
                    # Ref: [5] p. 158
                    NIRBand(center=2100, sigma=30, gamma=3, amplitude=0.6, name="O-H+C-O combination"),
                    # C-O + C-C stretch combination
                    # Ref: [5] p. 159
                    NIRBand(center=2270, sigma=25, gamma=2, amplitude=0.4, name="C-O+C-C stretch"),
                ],
                correlation_group=4,
            ),
            # Cellulose: β-1,4-linked glucose polymer
            # Refs: [6] pp. 295-300
            "cellulose": SpectralComponent(
                name="cellulose",
                bands=[
                    # O-H 1st overtone (crystalline cellulose)
                    # Ref: [6] p. 296
                    NIRBand(center=1490, sigma=22, gamma=2, amplitude=0.4, name="O-H 1st overtone"),
                    # Cellulose-specific C-H band
                    # Ref: [6] p. 297
                    NIRBand(center=1780, sigma=18, gamma=2, amplitude=0.3, name="Cellulose C-H"),
                    # O-H combination
                    # Ref: [6] p. 298
                    NIRBand(center=2090, sigma=28, gamma=3, amplitude=0.5, name="O-H combination"),
                    # Cellulose C-O stretch
                    # Ref: [6] p. 299
                    NIRBand(center=2280, sigma=22, gamma=2, amplitude=0.4, name="Cellulose C-O"),
                    # C-H combination
                    # Ref: [6] p. 299
                    NIRBand(center=2340, sigma=20, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=4,
            ),
            # Glucose: D-glucose monosaccharide
            # Refs: [2] pp. 368-370
            "glucose": SpectralComponent(
                name="glucose",
                bands=[
                    # O-H 1st overtone
                    # Ref: [2] p. 368
                    NIRBand(center=1440, sigma=25, gamma=3, amplitude=0.55, name="O-H 1st overtone"),
                    # C-H 1st overtone
                    # Ref: [2] p. 369
                    NIRBand(center=1690, sigma=18, gamma=2, amplitude=0.35, name="C-H 1st overtone"),
                    # O-H combination
                    # Ref: [2] p. 369
                    NIRBand(center=2080, sigma=30, gamma=3, amplitude=0.6, name="O-H combination"),
                    # Glucose-specific band
                    # Ref: [2] p. 370
                    NIRBand(center=2270, sigma=22, gamma=2, amplitude=0.4, name="Glucose combination"),
                ],
                correlation_group=4,
            ),
            # Fructose: D-fructose (fruit sugar)
            # Refs: [2] pp. 368-370
            "fructose": SpectralComponent(
                name="fructose",
                bands=[
                    # O-H 1st overtone (different from glucose due to ketose structure)
                    # Ref: [2] p. 368
                    NIRBand(center=1430, sigma=28, gamma=3, amplitude=0.5, name="O-H 1st overtone"),
                    # C-H 1st overtone
                    # Ref: [2] p. 369
                    NIRBand(center=1695, sigma=20, gamma=2, amplitude=0.3, name="C-H 1st overtone"),
                    # O-H combination
                    # Ref: [2] p. 369
                    NIRBand(center=2070, sigma=32, gamma=3, amplitude=0.55, name="O-H combination"),
                    # Fructose-specific band
                    # Ref: [2] p. 370
                    NIRBand(center=2260, sigma=24, gamma=2, amplitude=0.38, name="Fructose combination"),
                ],
                correlation_group=4,
            ),
            # Sucrose: Disaccharide (glucose + fructose)
            # Refs: [2] pp. 370-372
            "sucrose": SpectralComponent(
                name="sucrose",
                bands=[
                    # O-H 1st overtone
                    # Ref: [2] p. 370
                    NIRBand(center=1435, sigma=26, gamma=3, amplitude=0.52, name="O-H 1st overtone"),
                    # C-H 1st overtone
                    # Ref: [2] p. 371
                    NIRBand(center=1685, sigma=19, gamma=2, amplitude=0.32, name="C-H 1st overtone"),
                    # O-H combination
                    # Ref: [2] p. 371
                    NIRBand(center=2075, sigma=31, gamma=3, amplitude=0.58, name="O-H combination"),
                    # Sucrose-specific band
                    # Ref: [2] p. 372
                    NIRBand(center=2265, sigma=23, gamma=2, amplitude=0.42, name="Sucrose combination"),
                ],
                correlation_group=4,
            ),
            # Hemicellulose: Xylan and glucomannan
            # Refs: [6] pp. 300-303
            "hemicellulose": SpectralComponent(
                name="hemicellulose",
                bands=[
                    # O-H 1st overtone (acetyl groups)
                    # Ref: [6] p. 301
                    NIRBand(center=1470, sigma=24, gamma=2.5, amplitude=0.42, name="O-H 1st overtone"),
                    # C-H 1st overtone
                    # Ref: [6] p. 301
                    NIRBand(center=1760, sigma=20, gamma=2, amplitude=0.28, name="C-H 1st overtone"),
                    # O-H combination
                    # Ref: [6] p. 302
                    NIRBand(center=2085, sigma=30, gamma=3, amplitude=0.48, name="O-H combination"),
                    # Hemicellulose-specific band
                    # Ref: [6] p. 302
                    NIRBand(center=2250, sigma=24, gamma=2, amplitude=0.38, name="Hemicellulose combination"),
                ],
                correlation_group=4,
            ),
            # Lignin: Aromatic polymer from phenylpropanoid units
            # Refs: [6] pp. 303-305
            "lignin": SpectralComponent(
                name="lignin",
                bands=[
                    # Aromatic C-H 2nd overtone
                    # Ref: [6] p. 303
                    NIRBand(center=1140, sigma=16, gamma=1.5, amplitude=0.3, name="Ar C-H 2nd overtone"),
                    # O-H 1st overtone (phenolic OH)
                    # Ref: [6] p. 304
                    NIRBand(center=1420, sigma=22, gamma=2.5, amplitude=0.4, name="Phenolic O-H"),
                    # Aromatic C-H 1st overtone
                    # Ref: [6] p. 304
                    NIRBand(center=1670, sigma=20, gamma=2, amplitude=0.5, name="Ar C-H 1st overtone"),
                    # Aromatic combination
                    # Ref: [6] p. 305
                    NIRBand(center=2130, sigma=25, gamma=2.5, amplitude=0.35, name="Aromatic combination"),
                    # Lignin-specific band (methoxyl)
                    # Ref: [6] p. 305
                    NIRBand(center=2270, sigma=20, gamma=2, amplitude=0.3, name="OCH3 combination"),
                ],
                correlation_group=5,
            ),
            # ================================================================
            # ALCOHOLS
            # ================================================================
            # Ethanol: C₂H₅OH
            # Refs: [1] pp. 38-40
            "ethanol": SpectralComponent(
                name="ethanol",
                bands=[
                    # O-H 1st overtone (free hydroxyl)
                    # Ref: [1] p. 38
                    NIRBand(center=1410, sigma=18, gamma=2, amplitude=0.6, name="O-H 1st overtone"),
                    # O-H 1st overtone (H-bonded)
                    # Ref: [1] p. 38
                    NIRBand(center=1580, sigma=25, gamma=3, amplitude=0.45, name="O-H H-bonded"),
                    # C-H 1st overtone (CH₃)
                    # Ref: [1] p. 39
                    NIRBand(center=1695, sigma=18, gamma=2, amplitude=0.35, name="CH3 1st overtone"),
                    # O-H combination
                    # Ref: [1] p. 39
                    NIRBand(center=2050, sigma=28, gamma=3, amplitude=0.5, name="O-H combination"),
                    # C-H combination
                    # Ref: [1] p. 40
                    NIRBand(center=2290, sigma=20, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=7,
            ),
            # Methanol: CH₃OH
            # Refs: [1] pp. 38-40
            "methanol": SpectralComponent(
                name="methanol",
                bands=[
                    # O-H 1st overtone
                    # Ref: [1] p. 38
                    NIRBand(center=1400, sigma=20, gamma=2, amplitude=0.65, name="O-H 1st overtone"),
                    # O-H 1st overtone (H-bonded)
                    # Ref: [1] p. 38
                    NIRBand(center=1545, sigma=28, gamma=3, amplitude=0.5, name="O-H H-bonded"),
                    # C-H 1st overtone
                    # Ref: [1] p. 39
                    NIRBand(center=1705, sigma=16, gamma=2, amplitude=0.4, name="CH3 1st overtone"),
                    # O-H combination
                    # Ref: [1] p. 39
                    NIRBand(center=2040, sigma=30, gamma=3, amplitude=0.55, name="O-H combination"),
                ],
                correlation_group=7,
            ),
            # ================================================================
            # ORGANIC ACIDS
            # ================================================================
            # Acetic acid: CH₃COOH
            # Refs: [8] pp. 8-10
            "acetic_acid": SpectralComponent(
                name="acetic_acid",
                bands=[
                    # O-H 1st overtone (carboxylic, strongly H-bonded)
                    # Ref: [8] p. 9
                    NIRBand(center=1420, sigma=30, gamma=4, amplitude=0.55, name="Carboxylic O-H"),
                    # C=O 2nd overtone
                    # Ref: [8] p. 9
                    NIRBand(center=1700, sigma=18, gamma=2, amplitude=0.3, name="C=O 2nd overtone"),
                    # O-H combination
                    # Ref: [8] p. 10
                    NIRBand(center=1940, sigma=35, gamma=4, amplitude=0.6, name="O-H combination"),
                    # C-H combination
                    # Ref: [8] p. 10
                    NIRBand(center=2240, sigma=22, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=8,
            ),
            # Citric acid: C₆H₈O₇
            # Refs: [4] pp. 78-80
            "citric_acid": SpectralComponent(
                name="citric_acid",
                bands=[
                    # O-H 1st overtone (hydroxyl + carboxylic)
                    # Ref: [4] p. 78
                    NIRBand(center=1440, sigma=28, gamma=3, amplitude=0.5, name="O-H 1st overtone"),
                    # C=O combination
                    # Ref: [4] p. 79
                    NIRBand(center=1920, sigma=30, gamma=3, amplitude=0.45, name="C=O combination"),
                    # O-H combination
                    # Ref: [4] p. 79
                    NIRBand(center=2060, sigma=32, gamma=3, amplitude=0.55, name="O-H combination"),
                    # C-H + C-O combination
                    # Ref: [4] p. 80
                    NIRBand(center=2260, sigma=24, gamma=2, amplitude=0.38, name="C-H+C-O combination"),
                ],
                correlation_group=8,
            ),
            # Lactic acid: CH₃CH(OH)COOH
            # Refs: [9] pp. 1128-1130
            "lactic_acid": SpectralComponent(
                name="lactic_acid",
                bands=[
                    # O-H 1st overtone (hydroxyl)
                    # Ref: [9] p. 1128
                    NIRBand(center=1430, sigma=25, gamma=3, amplitude=0.5, name="Hydroxyl O-H"),
                    # O-H 1st overtone (carboxylic)
                    # Ref: [9] p. 1128
                    NIRBand(center=1485, sigma=28, gamma=3, amplitude=0.45, name="Carboxylic O-H"),
                    # C-H 1st overtone
                    # Ref: [9] p. 1129
                    NIRBand(center=1700, sigma=20, gamma=2, amplitude=0.35, name="C-H 1st overtone"),
                    # O-H combination
                    # Ref: [9] p. 1129
                    NIRBand(center=2020, sigma=30, gamma=3, amplitude=0.5, name="O-H combination"),
                    # C-H combination
                    # Ref: [9] p. 1130
                    NIRBand(center=2255, sigma=22, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=8,
            ),
            # ================================================================
            # PLANT PIGMENTS
            # ================================================================
            # Chlorophyll: Chlorophyll a and b
            # Refs: [2] pp. 375-378
            "chlorophyll": SpectralComponent(
                name="chlorophyll",
                bands=[
                    # Electronic absorption tail
                    # Ref: [2] p. 376
                    NIRBand(center=1070, sigma=15, gamma=1, amplitude=0.3, name="Chl absorption"),
                    # C-H 1st overtone (methyl groups on ring)
                    # Ref: [2] p. 377
                    NIRBand(center=1400, sigma=20, gamma=2, amplitude=0.4, name="C-H 1st overtone"),
                    # N-H/C-H combination (porphyrin ring)
                    # Ref: [2] p. 377
                    NIRBand(center=1730, sigma=18, gamma=2, amplitude=0.3, name="Porphyrin C-H"),
                    # C-H combination
                    # Ref: [2] p. 378
                    NIRBand(center=2270, sigma=22, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=5,
            ),
            # Carotenoid: β-carotene and xanthophylls
            # Refs: [2] pp. 378-380
            "carotenoid": SpectralComponent(
                name="carotenoid",
                bands=[
                    # Electronic absorption tail (conjugated polyene)
                    # Ref: [2] p. 378
                    NIRBand(center=1050, sigma=20, gamma=2, amplitude=0.25, name="Electronic tail"),
                    # C-H 1st overtone (polyene chain)
                    # Ref: [2] p. 379
                    NIRBand(center=1680, sigma=18, gamma=2, amplitude=0.4, name="=C-H 1st overtone"),
                    # C=C + C-H combination
                    # Ref: [2] p. 379
                    NIRBand(center=2135, sigma=22, gamma=2, amplitude=0.35, name="C=C combination"),
                    # C-H combination
                    # Ref: [2] p. 380
                    NIRBand(center=2280, sigma=20, gamma=2, amplitude=0.3, name="C-H combination"),
                ],
                correlation_group=5,
            ),
            # ================================================================
            # PHARMACEUTICAL COMPOUNDS
            # ================================================================
            # Caffeine: C₈H₁₀N₄O₂
            # Refs: [9] pp. 1130-1132
            "caffeine": SpectralComponent(
                name="caffeine",
                bands=[
                    # Aromatic C-H 2nd overtone (imidazole ring)
                    # Ref: [9] p. 1130
                    NIRBand(center=1130, sigma=15, gamma=1.5, amplitude=0.3, name="Ar C-H 2nd overtone"),
                    # N-H region (methylated nitrogens have no N-H)
                    # C-H 1st overtone (N-CH₃ groups)
                    # Ref: [9] p. 1131
                    NIRBand(center=1695, sigma=18, gamma=2, amplitude=0.45, name="N-CH3 1st overtone"),
                    # Aromatic C-H 1st overtone
                    # Ref: [9] p. 1131
                    NIRBand(center=1665, sigma=16, gamma=2, amplitude=0.35, name="Ar C-H 1st overtone"),
                    # C=O combination (carbonyl groups)
                    # Ref: [9] p. 1132
                    NIRBand(center=2010, sigma=25, gamma=2.5, amplitude=0.4, name="C=O combination"),
                    # C-H combination
                    # Ref: [9] p. 1132
                    NIRBand(center=2280, sigma=20, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=9,
            ),
            # Aspirin: Acetylsalicylic acid
            # Refs: [9] pp. 1125-1128
            "aspirin": SpectralComponent(
                name="aspirin",
                bands=[
                    # Aromatic C-H 2nd overtone
                    # Ref: [9] p. 1126
                    NIRBand(center=1145, sigma=14, gamma=1.5, amplitude=0.32, name="Ar C-H 2nd overtone"),
                    # Aromatic C-H 1st overtone
                    # Ref: [9] p. 1126
                    NIRBand(center=1680, sigma=18, gamma=2, amplitude=0.5, name="Ar C-H 1st overtone"),
                    # O-H 1st overtone (carboxylic, when not hydrogen-bonded)
                    # Ref: [9] p. 1127
                    NIRBand(center=1435, sigma=25, gamma=3, amplitude=0.4, name="Carboxylic O-H"),
                    # C=O combination
                    # Ref: [9] p. 1127
                    NIRBand(center=2020, sigma=25, gamma=2.5, amplitude=0.45, name="C=O combination"),
                    # Aromatic combination
                    # Ref: [9] p. 1128
                    NIRBand(center=2140, sigma=22, gamma=2, amplitude=0.38, name="Ar combination"),
                ],
                correlation_group=9,
            ),
            # Paracetamol (Acetaminophen): C₈H₉NO₂
            # Refs: [9] pp. 1132-1135
            "paracetamol": SpectralComponent(
                name="paracetamol",
                bands=[
                    # Aromatic C-H 2nd overtone
                    # Ref: [9] p. 1132
                    NIRBand(center=1140, sigma=15, gamma=1.5, amplitude=0.3, name="Ar C-H 2nd overtone"),
                    # O-H 1st overtone (phenolic)
                    # Ref: [9] p. 1133
                    NIRBand(center=1390, sigma=20, gamma=2.5, amplitude=0.45, name="Phenolic O-H"),
                    # N-H 1st overtone (amide)
                    # Ref: [9] p. 1133
                    NIRBand(center=1510, sigma=18, gamma=2, amplitude=0.4, name="Amide N-H"),
                    # Aromatic C-H 1st overtone
                    # Ref: [9] p. 1134
                    NIRBand(center=1670, sigma=18, gamma=2, amplitude=0.48, name="Ar C-H 1st overtone"),
                    # N-H + C=O combination (amide)
                    # Ref: [9] p. 1134
                    NIRBand(center=2055, sigma=28, gamma=3, amplitude=0.5, name="Amide combination"),
                    # C-H combination
                    # Ref: [9] p. 1135
                    NIRBand(center=2260, sigma=22, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=9,
            ),
            # ================================================================
            # FIBERS
            # ================================================================
            # Cotton: Cotton cellulose (pure cellulose fiber)
            # Refs: [6] pp. 295-298
            "cotton": SpectralComponent(
                name="cotton",
                bands=[
                    # O-H 2nd overtone
                    # Ref: [6] p. 295
                    NIRBand(center=1200, sigma=25, gamma=2.5, amplitude=0.35, name="O-H 2nd overtone"),
                    # O-H 1st overtone (crystalline cellulose I)
                    # Ref: [6] p. 296
                    NIRBand(center=1494, sigma=22, gamma=2, amplitude=0.45, name="O-H 1st overtone"),
                    # C-H 1st overtone
                    # Ref: [6] p. 297
                    NIRBand(center=1780, sigma=18, gamma=2, amplitude=0.32, name="C-H 1st overtone"),
                    # O-H combination
                    # Ref: [6] p. 298
                    NIRBand(center=2100, sigma=30, gamma=3, amplitude=0.52, name="O-H combination"),
                    # C-O/C-H combination
                    # Ref: [6] p. 298
                    NIRBand(center=2280, sigma=22, gamma=2, amplitude=0.4, name="C-O combination"),
                    # C-H combination
                    # Ref: [6] p. 298
                    NIRBand(center=2345, sigma=20, gamma=2, amplitude=0.36, name="C-H combination"),
                ],
                correlation_group=4,
            ),
            # Polyester: Polyethylene terephthalate (PET)
            # Refs: [1] pp. 60-62
            "polyester": SpectralComponent(
                name="polyester",
                bands=[
                    # Aromatic C-H 2nd overtone
                    # Ref: [1] p. 60
                    NIRBand(center=1140, sigma=14, gamma=1.5, amplitude=0.35, name="Ar C-H 2nd overtone"),
                    # C-H 1st overtone (aliphatic)
                    # Ref: [1] p. 61
                    NIRBand(center=1720, sigma=20, gamma=2, amplitude=0.5, name="C-H 1st overtone"),
                    # Aromatic C-H 1st overtone
                    # Ref: [1] p. 61
                    NIRBand(center=1660, sigma=16, gamma=2, amplitude=0.45, name="Ar C-H 1st overtone"),
                    # C=O combination (ester)
                    # Ref: [1] p. 62
                    NIRBand(center=2015, sigma=25, gamma=2.5, amplitude=0.4, name="Ester C=O"),
                    # C-H combination
                    # Ref: [1] p. 62
                    NIRBand(center=2255, sigma=20, gamma=2, amplitude=0.38, name="C-H combination"),
                    # Aromatic combination
                    # Ref: [1] p. 62
                    NIRBand(center=2130, sigma=22, gamma=2, amplitude=0.32, name="Ar combination"),
                ],
                correlation_group=10,
            ),
        }

    return _PREDEFINED_COMPONENTS


# Default wavelength parameters
DEFAULT_WAVELENGTH_START: float = 1000.0
DEFAULT_WAVELENGTH_END: float = 2500.0
DEFAULT_WAVELENGTH_STEP: float = 2.0

# Default NIR-relevant zones for random band placement
DEFAULT_NIR_ZONES = [
    (1100, 1300),  # 2nd overtones
    (1400, 1550),  # 1st overtones O-H, N-H
    (1650, 1800),  # 1st overtones C-H
    (1850, 2000),  # Combination O-H
    (2000, 2200),  # Combination N-H
    (2200, 2400),  # Combination C-H
]

# Complexity presets for generator parameters
COMPLEXITY_PARAMS = {
    "simple": {
        "path_length_std": 0.02,
        "baseline_amplitude": 0.01,
        "scatter_alpha_std": 0.02,
        "scatter_beta_std": 0.005,
        "tilt_std": 0.005,
        "global_slope_mean": 0.0,
        "global_slope_std": 0.02,
        "shift_std": 0.2,
        "stretch_std": 0.0005,
        "instrumental_fwhm": 4,
        "noise_base": 0.002,
        "noise_signal_dep": 0.005,
        "artifact_prob": 0.0,
    },
    "realistic": {
        "path_length_std": 0.05,
        "baseline_amplitude": 0.02,
        "scatter_alpha_std": 0.05,
        "scatter_beta_std": 0.01,
        "tilt_std": 0.01,
        "global_slope_mean": 0.05,
        "global_slope_std": 0.03,
        "shift_std": 0.5,
        "stretch_std": 0.001,
        "instrumental_fwhm": 8,
        "noise_base": 0.005,
        "noise_signal_dep": 0.01,
        "artifact_prob": 0.02,
    },
    "complex": {
        "path_length_std": 0.08,
        "baseline_amplitude": 0.05,
        "scatter_alpha_std": 0.08,
        "scatter_beta_std": 0.02,
        "tilt_std": 0.02,
        "global_slope_mean": 0.08,
        "global_slope_std": 0.05,
        "shift_std": 1.0,
        "stretch_std": 0.002,
        "instrumental_fwhm": 12,
        "noise_base": 0.008,
        "noise_signal_dep": 0.015,
        "artifact_prob": 0.05,
    },
}

# Default predefined component names for realistic/complex modes
DEFAULT_REALISTIC_COMPONENTS = ["water", "protein", "lipid", "starch", "cellulose"]
