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

    [11] Martelo-Vidal, M. J., & Vázquez, M. (2014). Determination of Polyphenols,
         Sugars, Glycerol and Ethanol in Wine by NIR Spectroscopy. Czech Journal
         of Food Sciences, 32(1), 37-47.
         - Wine/fermentation component band assignments

    [12] Luypaert, J., Massart, D. L., & Vander Heyden, Y. (2007). Near-Infrared
         Spectroscopy Applications in Pharmaceutical Analysis. Talanta, 72(3),
         865-883.
         - Lactose and pharmaceutical excipient bands

    [13] Khayamim, F., et al. (2015). Using Visible and Near Infrared Spectroscopy
         to Estimate Carbonates and Gypsum in Soils. Journal of Earth System
         Science, 124(8), 1755-1766. DOI: 10.1007/s12145-015-0244-9
         - Soil mineral (carbonate, gypsum) band assignments

    [14] Shenk, J. S., Workman Jr, J., & Westerhaus, M. O. (2008). Application of
         NIR Spectroscopy to Agricultural Products. In D. A. Burns & E. W. Ciurczak
         (Eds.), Handbook of Near-Infrared Analysis (3rd ed., pp. 347-386). CRC Press.
         - Agricultural commodity band assignments

    [15] Lachenal, G. (1995). Polymer Applications. In H. W. Siesler & K. Holland-Moritz
         (Eds.), Infrared and Raman Spectroscopy of Polymers. Marcel Dekker.
         - Polymer (PE, PS, rubber) NIR band assignments
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

    Available Components (111 total):
        Water-related (2):
            - ``water``: H₂O fundamental O-H vibrations [1, pp. 34-36]
            - ``moisture``: Bound water in organic matrices [2, pp. 358-362]

        Proteins and Nitrogen (12):
            - ``protein``: General protein (amide, N-H, C-H) [1, pp. 48-52]
            - ``nitrogen_compound``: Primary/secondary amines [1, pp. 52-54]
            - ``urea``: CO(NH₂)₂ bands [9, p. 1125]
            - ``amino_acid``: Free amino acids [3, pp. 215-220]
            - ``casein``: Milk protein [4, pp. 85-88]
            - ``gluten``: Wheat protein complex [5, pp. 155-160]
            - ``albumin``: Globular protein (egg white, serum)
            - ``collagen``: Fibrous structural protein
            - ``keratin``: Structural protein (hair, nails)
            - ``zein``: Corn protein (prolamin)
            - ``gelatin``: Denatured collagen
            - ``whey``: Milk serum proteins

        Lipids and Hydrocarbons (15):
            - ``lipid``: Triglycerides (C-H stretching) [1, pp. 44-48]
            - ``oil``: Vegetable/mineral oils [4, pp. 67-72]
            - ``saturated_fat``: Saturated fatty acids [7, pp. 15-20]
            - ``unsaturated_fat``: Mono/polyunsaturated fats [7, pp. 20-25]
            - ``aromatic``: Benzene derivatives [1, pp. 56-58]
            - ``alkane``: Saturated hydrocarbons [7, pp. 10-15]
            - ``waxes``: Cuticular waxes [7, pp. 15-20]
            - ``oleic_acid``: Monounsaturated fatty acid (C18:1)
            - ``linoleic_acid``: Polyunsaturated fatty acid (C18:2)
            - ``linolenic_acid``: Polyunsaturated fatty acid (C18:3)
            - ``palmitic_acid``: Saturated fatty acid (C16:0)
            - ``stearic_acid``: Saturated fatty acid (C18:0)
            - ``phospholipid``: Lecithin-like membrane lipids
            - ``cholesterol``: Sterol lipid
            - ``cocoa_butter``: Triglyceride mix

        Carbohydrates (18):
            - ``starch``: Amylose/amylopectin [5, pp. 155-160]
            - ``cellulose``: β-1,4-glucan chains [6, pp. 295-300]
            - ``glucose``: D-glucose monosaccharide [2, pp. 368-370]
            - ``fructose``: D-fructose monosaccharide [2, pp. 368-370]
            - ``sucrose``: Disaccharide [2, pp. 370-372]
            - ``hemicellulose``: Xylan/glucomannan [6, pp. 300-303]
            - ``lignin``: Aromatic polymer [6, pp. 303-305]
            - ``lactose``: Milk sugar [12], [4, pp. 85-88]
            - ``cotton``: Cotton cellulose [6, pp. 295-298]
            - ``dietary_fiber``: Plant cell wall [6], [5]
            - ``maltose``: Malt sugar (glucose-glucose disaccharide)
            - ``raffinose``: Trisaccharide (galactose-glucose-fructose)
            - ``inulin``: Fructose polymer (dietary fiber)
            - ``xylose``: Pentose monosaccharide
            - ``arabinose``: Pentose monosaccharide
            - ``galactose``: Hexose monosaccharide
            - ``mannose``: Hexose monosaccharide
            - ``trehalose``: Non-reducing disaccharide

        Alcohols and Polyols (9):
            - ``ethanol``: C₂H₅OH [1, pp. 38-40]
            - ``methanol``: CH₃OH [1, pp. 38-40]
            - ``glycerol``: Polyol from fermentation [11]
            - ``propanol``: Propyl alcohol
            - ``butanol``: Butyl alcohol
            - ``sorbitol``: Sugar alcohol
            - ``mannitol``: Sugar alcohol
            - ``xylitol``: Sugar alcohol
            - ``isopropanol``: Isopropyl alcohol

        Organic Acids (12):
            - ``acetic_acid``: CH₃COOH [8, pp. 8-10]
            - ``citric_acid``: C₆H₈O₇ [4, pp. 78-80]
            - ``lactic_acid``: CH₃CH(OH)COOH [9, pp. 1128-1130]
            - ``malic_acid``: Fruit acid [4, pp. 78-80]
            - ``tartaric_acid``: Grape/wine acid [11]
            - ``formic_acid``: HCOOH
            - ``oxalic_acid``: (COOH)₂
            - ``succinic_acid``: Dicarboxylic acid
            - ``fumaric_acid``: Unsaturated dicarboxylic acid
            - ``propionic_acid``: CH₃CH₂COOH
            - ``butyric_acid``: Short-chain fatty acid
            - ``ascorbic_acid``: Vitamin C

        Plant Pigments and Phenolics (8):
            - ``chlorophyll``: Chlorophyll a/b [2, pp. 375-378]
            - ``carotenoid``: β-carotene, xanthophylls [2, pp. 378-380]
            - ``tannins``: Phenolic compounds [6], [11]
            - ``anthocyanin``: Red-purple plant pigment
            - ``lycopene``: Red carotenoid (tomatoes)
            - ``lutein``: Yellow carotenoid (xanthophyll)
            - ``xanthophyll``: General yellow pigments
            - ``melanin``: Brown-black pigment

        Pharmaceutical (10):
            - ``caffeine``: C₈H₁₀N₄O₂ [9, pp. 1130-1132]
            - ``aspirin``: Acetylsalicylic acid [9, pp. 1125-1128]
            - ``paracetamol``: Acetaminophen [9, pp. 1132-1135]
            - ``ibuprofen``: Anti-inflammatory drug
            - ``naproxen``: NSAID drug
            - ``diclofenac``: NSAID drug
            - ``metformin``: Diabetes drug
            - ``omeprazole``: Proton pump inhibitor
            - ``amoxicillin``: Antibiotic
            - ``microcrystalline_cellulose``: Pharmaceutical excipient

        Fibers and Textiles (2):
            - ``polyester``: PET fiber [1, pp. 60-62]
            - ``nylon``: Polyamide fiber [1, pp. 60-62]

        Polymers and Plastics (10):
            - ``polyethylene``: HDPE/LDPE plastic [15], [1, pp. 58-60]
            - ``polystyrene``: Aromatic polymer [15], [1, pp. 56-58]
            - ``natural_rubber``: cis-1,4-polyisoprene [15]
            - ``pmma``: Polymethyl methacrylate (acrylic)
            - ``pvc``: Polyvinyl chloride
            - ``polypropylene``: PP plastic
            - ``pet``: Polyethylene terephthalate
            - ``ptfe``: Polytetrafluoroethylene (Teflon)
            - ``abs``: Acrylonitrile butadiene styrene

        Solvents (6):
            - ``acetone``: Ketone solvent [1, pp. 42-44]
            - ``dmso``: Dimethyl sulfoxide
            - ``ethyl_acetate``: Ester solvent
            - ``toluene``: Aromatic solvent
            - ``chloroform``: Halogenated solvent
            - ``hexane``: Alkane solvent

        Soil Minerals (8):
            - ``carbonates``: CaCO₃, MgCO₃ [13]
            - ``gypsum``: CaSO₄·2H₂O [13]
            - ``kaolinite``: Clay mineral [13]
            - ``montmorillonite``: Smectite clay
            - ``illite``: Mica-like clay
            - ``goethite``: Iron oxyhydroxide
            - ``talc``: Magnesium silicate
            - ``silica``: Silicon dioxide

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
            # ================================================================
            # POLYMERS AND PLASTICS
            # ================================================================
            # Polyethylene: High/low density polyethylene (HDPE/LDPE)
            # Refs: [15], [1] pp. 58-60
            # Pure -CH2- polymer with strong methylene bands
            "polyethylene": SpectralComponent(
                name="polyethylene",
                bands=[
                    # CH2 2nd overtone
                    # Ref: [15], [1] p. 58
                    NIRBand(center=1190, sigma=18, gamma=1.5, amplitude=0.35, name="CH2 2nd overtone"),
                    # CH2 1st overtone (very strong)
                    # Ref: [15], characteristic PE band
                    NIRBand(center=1720, sigma=22, gamma=2, amplitude=0.7, name="CH2 1st overtone"),
                    # CH2 combination
                    # Ref: [1] p. 59
                    NIRBand(center=2310, sigma=20, gamma=2, amplitude=0.55, name="CH2 combination"),
                    # CH2 combination (additional)
                    # Ref: [1] p. 59
                    NIRBand(center=2355, sigma=18, gamma=2, amplitude=0.45, name="CH2 combination 2"),
                ],
                correlation_group=11,
            ),
            # Polystyrene: Aromatic polymer (styrene units)
            # Refs: [15], [1] pp. 56-58
            "polystyrene": SpectralComponent(
                name="polystyrene",
                bands=[
                    # Aromatic C-H 2nd overtone
                    # Ref: [15]
                    NIRBand(center=1145, sigma=15, gamma=1.5, amplitude=0.4, name="Ar C-H 2nd overtone"),
                    # Aromatic C-H 1st overtone
                    # Ref: [1] p. 57
                    NIRBand(center=1680, sigma=18, gamma=2, amplitude=0.55, name="Ar C-H 1st overtone"),
                    # Aliphatic C-H 1st overtone (backbone)
                    # Ref: [15]
                    NIRBand(center=1720, sigma=20, gamma=2, amplitude=0.35, name="C-H 1st overtone"),
                    # Aromatic C-H combination
                    # Ref: [1] p. 58
                    NIRBand(center=2170, sigma=22, gamma=2, amplitude=0.45, name="Ar C-H combination"),
                    # C-H combination
                    # Ref: [15]
                    NIRBand(center=2300, sigma=20, gamma=2, amplitude=0.4, name="C-H combination"),
                ],
                correlation_group=11,
            ),
            # Natural rubber: cis-1,4-polyisoprene
            # Refs: [15], [1] pp. 58-60
            "natural_rubber": SpectralComponent(
                name="natural_rubber",
                bands=[
                    # =C-H 2nd overtone (vinyl)
                    # Ref: [15]
                    NIRBand(center=1160, sigma=16, gamma=1.5, amplitude=0.3, name="=C-H 2nd overtone"),
                    # C-H 1st overtone (methyl + methylene)
                    # Ref: [15]
                    NIRBand(center=1720, sigma=24, gamma=2, amplitude=0.55, name="C-H 1st overtone"),
                    # =C-H 1st overtone (isoprene)
                    # Ref: [15]
                    NIRBand(center=2130, sigma=22, gamma=2, amplitude=0.35, name="=C-H 1st overtone"),
                    # C-H combination
                    # Ref: [15]
                    NIRBand(center=2250, sigma=20, gamma=2, amplitude=0.45, name="C-H combination"),
                    # CH3 combination
                    # Ref: [15]
                    NIRBand(center=2350, sigma=18, gamma=2, amplitude=0.38, name="CH3 combination"),
                ],
                correlation_group=11,
            ),
            # Nylon (Polyamide): Nylon 6,6 / Nylon 6
            # Refs: [1] pp. 60-62, [15]
            "nylon": SpectralComponent(
                name="nylon",
                bands=[
                    # N-H 1st overtone (amide)
                    # Ref: [1] p. 60
                    NIRBand(center=1500, sigma=20, gamma=2, amplitude=0.5, name="N-H 1st overtone"),
                    # C-H 1st overtone
                    # Ref: [15]
                    NIRBand(center=1720, sigma=22, gamma=2, amplitude=0.45, name="C-H 1st overtone"),
                    # N-H combination (amide II)
                    # Ref: [1] p. 61
                    NIRBand(center=2050, sigma=28, gamma=3, amplitude=0.55, name="N-H combination"),
                    # C-H combination
                    # Ref: [15]
                    NIRBand(center=2295, sigma=20, gamma=2, amplitude=0.4, name="C-H combination"),
                ],
                correlation_group=11,
            ),
            # ================================================================
            # DAIRY COMPONENTS
            # ================================================================
            # Lactose: Milk sugar (galactose + glucose disaccharide)
            # Refs: [12], [4] pp. 85-88
            "lactose": SpectralComponent(
                name="lactose",
                bands=[
                    # O-H 1st overtone
                    # Ref: [12]
                    NIRBand(center=1450, sigma=26, gamma=3, amplitude=0.5, name="O-H 1st overtone"),
                    # C-H 1st overtone
                    # Ref: [12]
                    NIRBand(center=1690, sigma=18, gamma=2, amplitude=0.32, name="C-H 1st overtone"),
                    # O-H combination (water-like)
                    # Ref: [4] p. 86
                    NIRBand(center=1940, sigma=32, gamma=4, amplitude=0.65, name="O-H combination"),
                    # O-H + C-O combination (carbohydrate)
                    # Ref: [12]
                    NIRBand(center=2100, sigma=30, gamma=3, amplitude=0.55, name="O-H+C-O combination"),
                    # C-H combination
                    # Ref: [4] p. 87
                    NIRBand(center=2270, sigma=22, gamma=2, amplitude=0.4, name="C-H combination"),
                ],
                correlation_group=4,
            ),
            # Casein: Major milk protein
            # Refs: [4] pp. 85-88, [2] pp. 380-382
            "casein": SpectralComponent(
                name="casein",
                bands=[
                    # N-H 1st overtone (amide)
                    # Ref: [4] p. 86
                    NIRBand(center=1510, sigma=20, gamma=2, amplitude=0.48, name="N-H 1st overtone"),
                    # C-H aromatic
                    # Ref: [2] p. 381
                    NIRBand(center=1680, sigma=24, gamma=3, amplitude=0.38, name="C-H aromatic"),
                    # N-H combination
                    # Ref: [4] p. 87
                    NIRBand(center=2050, sigma=30, gamma=3, amplitude=0.55, name="N-H combination"),
                    # Protein C-H
                    # Ref: [2] p. 381
                    NIRBand(center=2180, sigma=24, gamma=2, amplitude=0.45, name="Protein C-H"),
                ],
                correlation_group=2,
            ),
            # ================================================================
            # SOLVENTS
            # ================================================================
            # Acetone: Ketone solvent (propan-2-one)
            # Refs: [1] pp. 42-44
            "acetone": SpectralComponent(
                name="acetone",
                bands=[
                    # C=O 1st overtone (characteristic ketone band)
                    # Ref: [1] p. 42
                    NIRBand(center=1690, sigma=20, gamma=2.5, amplitude=0.5, name="C=O 1st overtone"),
                    # C-H 1st overtone (methyl)
                    # Ref: [1] p. 43
                    NIRBand(center=1710, sigma=18, gamma=2, amplitude=0.4, name="CH3 1st overtone"),
                    # C=O + C-H combination
                    # Ref: [1] p. 43
                    NIRBand(center=2100, sigma=25, gamma=3, amplitude=0.45, name="C=O combination"),
                    # C-H combination
                    # Ref: [1] p. 44
                    NIRBand(center=2300, sigma=22, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=12,
            ),
            # ================================================================
            # PLANT PHENOLICS
            # ================================================================
            # Tannins: Phenolic compounds in plants
            # Refs: [6] pp. 303-305, [11]
            "tannins": SpectralComponent(
                name="tannins",
                bands=[
                    # Phenolic O-H 1st overtone
                    # Ref: [6] p. 304
                    NIRBand(center=1420, sigma=25, gamma=3, amplitude=0.4, name="Phenolic O-H"),
                    # Aromatic C-H
                    # Ref: [6] p. 304
                    NIRBand(center=1670, sigma=20, gamma=2, amplitude=0.35, name="Ar C-H"),
                    # Phenolic C-O combination
                    # Ref: [11]
                    NIRBand(center=2056, sigma=22, gamma=2.5, amplitude=0.38, name="C-O phenol"),
                    # Aromatic combination
                    # Ref: [6] p. 305
                    NIRBand(center=2270, sigma=22, gamma=2, amplitude=0.32, name="Ar combination"),
                ],
                correlation_group=5,
            ),
            # Waxes: Cuticular waxes (long-chain esters/alkanes)
            # Refs: [7] pp. 15-20, [14]
            "waxes": SpectralComponent(
                name="waxes",
                bands=[
                    # C-H 2nd overtone (long chain)
                    # Ref: [7] p. 16
                    NIRBand(center=1190, sigma=18, gamma=2, amplitude=0.35, name="C-H 2nd overtone"),
                    # C-H 1st overtone
                    # Ref: [7] p. 17
                    NIRBand(center=1720, sigma=22, gamma=2, amplitude=0.6, name="C-H 1st overtone"),
                    # C-H combination
                    # Ref: [7] p. 18
                    NIRBand(center=2310, sigma=20, gamma=2, amplitude=0.5, name="CH2 combination"),
                    # C-H combination (additional)
                    # Ref: [14]
                    NIRBand(center=2350, sigma=18, gamma=2, amplitude=0.42, name="CH3 combination"),
                ],
                correlation_group=3,
            ),
            # ================================================================
            # FERMENTATION / BEVERAGES
            # ================================================================
            # Glycerol: Polyol produced during fermentation
            # Refs: [11], [4] pp. 90-92
            "glycerol": SpectralComponent(
                name="glycerol",
                bands=[
                    # O-H 1st overtone (polyol, multiple OH)
                    # Ref: [11]
                    NIRBand(center=1450, sigma=28, gamma=3, amplitude=0.55, name="O-H 1st overtone"),
                    # O-H hydrogen bonded
                    # Ref: [11]
                    NIRBand(center=1580, sigma=25, gamma=3, amplitude=0.4, name="O-H H-bonded"),
                    # C-H 1st overtone
                    # Ref: [4] p. 91
                    NIRBand(center=1700, sigma=20, gamma=2, amplitude=0.35, name="C-H 1st overtone"),
                    # O-H combination
                    # Ref: [11]
                    NIRBand(center=2060, sigma=30, gamma=3, amplitude=0.5, name="O-H combination"),
                    # C-H combination
                    # Ref: [4] p. 91
                    NIRBand(center=2280, sigma=22, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=7,
            ),
            # Malic acid: Fruit acid common in apples, grapes
            # Refs: [4] pp. 78-80, [11]
            "malic_acid": SpectralComponent(
                name="malic_acid",
                bands=[
                    # O-H 1st overtone (hydroxyl + carboxylic)
                    # Ref: [4] p. 78
                    NIRBand(center=1440, sigma=28, gamma=3, amplitude=0.48, name="O-H 1st overtone"),
                    # C=O combination (carboxylic)
                    # Ref: [11]
                    NIRBand(center=1920, sigma=30, gamma=3, amplitude=0.42, name="C=O combination"),
                    # O-H combination
                    # Ref: [4] p. 79
                    NIRBand(center=2050, sigma=32, gamma=3, amplitude=0.52, name="O-H combination"),
                    # C-H combination
                    # Ref: [11]
                    NIRBand(center=2255, sigma=22, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=8,
            ),
            # Tartaric acid: Major grape/wine acid
            # Refs: [11], [4] pp. 78-80
            "tartaric_acid": SpectralComponent(
                name="tartaric_acid",
                bands=[
                    # O-H 1st overtone
                    # Ref: [11]
                    NIRBand(center=1435, sigma=28, gamma=3, amplitude=0.5, name="O-H 1st overtone"),
                    # C=O combination
                    # Ref: [11]
                    NIRBand(center=1910, sigma=30, gamma=3, amplitude=0.4, name="C=O combination"),
                    # O-H combination
                    # Ref: [4] p. 79
                    NIRBand(center=2040, sigma=32, gamma=3, amplitude=0.5, name="O-H combination"),
                    # C-H combination
                    # Ref: [11]
                    NIRBand(center=2260, sigma=22, gamma=2, amplitude=0.36, name="C-H combination"),
                ],
                correlation_group=8,
            ),
            # ================================================================
            # SOIL MINERALS
            # ================================================================
            # Carbonates: CaCO3, MgCO3 (calcite, dolomite)
            # Refs: [13]
            "carbonates": SpectralComponent(
                name="carbonates",
                bands=[
                    # CO3 combination band
                    # Ref: [13]
                    NIRBand(center=2330, sigma=25, gamma=2.5, amplitude=0.4, name="CO3 combination"),
                    # CO3 combination (higher wavelength)
                    # Ref: [13]
                    NIRBand(center=2525, sigma=30, gamma=3, amplitude=0.3, name="CO3 combination 2"),
                ],
                correlation_group=13,
            ),
            # Gypsum: CaSO4·2H2O (hydrated calcium sulfate)
            # Refs: [13]
            "gypsum": SpectralComponent(
                name="gypsum",
                bands=[
                    # Crystal water / SO4 combination
                    # Ref: [13]
                    NIRBand(center=1740, sigma=25, gamma=2.5, amplitude=0.35, name="Crystal water"),
                    # O-H combination (structural water)
                    # Ref: [13]
                    NIRBand(center=1900, sigma=32, gamma=3, amplitude=0.5, name="O-H combination"),
                    # SO4 / water combination
                    # Ref: [13]
                    NIRBand(center=2200, sigma=28, gamma=2.5, amplitude=0.38, name="SO4 combination"),
                ],
                correlation_group=13,
            ),
            # Kaolinite: Clay mineral Al2Si2O5(OH)4
            # Refs: [13], common soil NIR feature
            "kaolinite": SpectralComponent(
                name="kaolinite",
                bands=[
                    # Al-OH 1st overtone
                    # Ref: [13]
                    NIRBand(center=1400, sigma=20, gamma=2, amplitude=0.4, name="Al-OH 1st overtone"),
                    # Al-OH combination
                    # Ref: [13]
                    NIRBand(center=2160, sigma=25, gamma=2.5, amplitude=0.5, name="Al-OH combination"),
                    # Al-OH combination (characteristic doublet)
                    # Ref: [13]
                    NIRBand(center=2200, sigma=22, gamma=2, amplitude=0.55, name="Al-OH combination 2"),
                ],
                correlation_group=13,
            ),
            # ================================================================
            # ADDITIONAL AGRICULTURAL
            # ================================================================
            # Gluten: Wheat protein complex
            # Refs: [5] pp. 155-160, [14]
            "gluten": SpectralComponent(
                name="gluten",
                bands=[
                    # N-H 1st overtone (amide)
                    # Ref: [5] p. 156
                    NIRBand(center=1505, sigma=20, gamma=2, amplitude=0.5, name="N-H 1st overtone"),
                    # C-H aromatic (Phe, Tyr)
                    # Ref: [14]
                    NIRBand(center=1680, sigma=24, gamma=3, amplitude=0.4, name="C-H aromatic"),
                    # N-H combination
                    # Ref: [5] p. 158
                    NIRBand(center=2050, sigma=30, gamma=3, amplitude=0.6, name="N-H combination"),
                    # Protein C-H combination
                    # Ref: [14]
                    NIRBand(center=2180, sigma=24, gamma=2, amplitude=0.48, name="Protein C-H"),
                    # N-H + Amide III
                    # Ref: [5] p. 159
                    NIRBand(center=2290, sigma=20, gamma=2, amplitude=0.35, name="N-H+Amide III"),
                ],
                correlation_group=2,
            ),
            # Fiber (dietary): Mixed plant cell wall material
            # Refs: [6], [5] pp. 160-165
            "dietary_fiber": SpectralComponent(
                name="dietary_fiber",
                bands=[
                    # O-H 1st overtone (cellulose-like)
                    # Ref: [6] p. 296
                    NIRBand(center=1490, sigma=24, gamma=2.5, amplitude=0.45, name="O-H 1st overtone"),
                    # C-H 1st overtone
                    # Ref: [6] p. 297
                    NIRBand(center=1770, sigma=20, gamma=2, amplitude=0.32, name="C-H 1st overtone"),
                    # O-H combination
                    # Ref: [5] p. 162
                    NIRBand(center=2090, sigma=30, gamma=3, amplitude=0.52, name="O-H combination"),
                    # C-H combination
                    # Ref: [6] p. 299
                    NIRBand(center=2275, sigma=22, gamma=2, amplitude=0.42, name="C-H combination"),
                    # C-H combination (hemicellulose region)
                    # Ref: [5] p. 163
                    NIRBand(center=2340, sigma=20, gamma=2, amplitude=0.35, name="C-H combination 2"),
                ],
                correlation_group=4,
            ),
            # ================================================================
            # EXTENDED CARBOHYDRATES (Phase 1.4 additions)
            # ================================================================
            # Maltose: Malt sugar (glucose-glucose disaccharide)
            # Refs: [2] pp. 368-372
            "maltose": SpectralComponent(
                name="maltose",
                bands=[
                    NIRBand(center=1445, sigma=26, gamma=3, amplitude=0.52, name="O-H 1st overtone"),
                    NIRBand(center=1688, sigma=18, gamma=2, amplitude=0.33, name="C-H 1st overtone"),
                    NIRBand(center=2078, sigma=30, gamma=3, amplitude=0.58, name="O-H combination"),
                    NIRBand(center=2268, sigma=22, gamma=2, amplitude=0.40, name="Maltose combination"),
                ],
                correlation_group=4,
            ),
            # Raffinose: Trisaccharide (galactose-glucose-fructose)
            "raffinose": SpectralComponent(
                name="raffinose",
                bands=[
                    NIRBand(center=1448, sigma=27, gamma=3, amplitude=0.50, name="O-H 1st overtone"),
                    NIRBand(center=1692, sigma=19, gamma=2, amplitude=0.32, name="C-H 1st overtone"),
                    NIRBand(center=2082, sigma=31, gamma=3, amplitude=0.55, name="O-H combination"),
                    NIRBand(center=2272, sigma=23, gamma=2, amplitude=0.38, name="Raffinose combination"),
                ],
                correlation_group=4,
            ),
            # Inulin: Fructose polymer (dietary fiber)
            "inulin": SpectralComponent(
                name="inulin",
                bands=[
                    NIRBand(center=1432, sigma=28, gamma=3, amplitude=0.48, name="O-H 1st overtone"),
                    NIRBand(center=1698, sigma=20, gamma=2, amplitude=0.30, name="C-H 1st overtone"),
                    NIRBand(center=2068, sigma=32, gamma=3, amplitude=0.52, name="O-H combination"),
                    NIRBand(center=2258, sigma=24, gamma=2, amplitude=0.36, name="Inulin combination"),
                ],
                correlation_group=4,
            ),
            # Xylose: Pentose monosaccharide
            "xylose": SpectralComponent(
                name="xylose",
                bands=[
                    NIRBand(center=1438, sigma=25, gamma=3, amplitude=0.50, name="O-H 1st overtone"),
                    NIRBand(center=1686, sigma=18, gamma=2, amplitude=0.32, name="C-H 1st overtone"),
                    NIRBand(center=2072, sigma=30, gamma=3, amplitude=0.55, name="O-H combination"),
                    NIRBand(center=2262, sigma=22, gamma=2, amplitude=0.38, name="Xylose combination"),
                ],
                correlation_group=4,
            ),
            # Arabinose: Pentose monosaccharide
            "arabinose": SpectralComponent(
                name="arabinose",
                bands=[
                    NIRBand(center=1442, sigma=25, gamma=3, amplitude=0.48, name="O-H 1st overtone"),
                    NIRBand(center=1684, sigma=18, gamma=2, amplitude=0.30, name="C-H 1st overtone"),
                    NIRBand(center=2076, sigma=30, gamma=3, amplitude=0.52, name="O-H combination"),
                    NIRBand(center=2266, sigma=22, gamma=2, amplitude=0.36, name="Arabinose combination"),
                ],
                correlation_group=4,
            ),
            # Galactose: Hexose monosaccharide
            "galactose": SpectralComponent(
                name="galactose",
                bands=[
                    NIRBand(center=1442, sigma=25, gamma=3, amplitude=0.54, name="O-H 1st overtone"),
                    NIRBand(center=1692, sigma=18, gamma=2, amplitude=0.34, name="C-H 1st overtone"),
                    NIRBand(center=2082, sigma=30, gamma=3, amplitude=0.58, name="O-H combination"),
                    NIRBand(center=2272, sigma=22, gamma=2, amplitude=0.40, name="Galactose combination"),
                ],
                correlation_group=4,
            ),
            # Mannose: Hexose monosaccharide
            "mannose": SpectralComponent(
                name="mannose",
                bands=[
                    NIRBand(center=1444, sigma=25, gamma=3, amplitude=0.52, name="O-H 1st overtone"),
                    NIRBand(center=1688, sigma=18, gamma=2, amplitude=0.33, name="C-H 1st overtone"),
                    NIRBand(center=2078, sigma=30, gamma=3, amplitude=0.56, name="O-H combination"),
                    NIRBand(center=2268, sigma=22, gamma=2, amplitude=0.39, name="Mannose combination"),
                ],
                correlation_group=4,
            ),
            # Trehalose: Non-reducing disaccharide
            "trehalose": SpectralComponent(
                name="trehalose",
                bands=[
                    NIRBand(center=1440, sigma=26, gamma=3, amplitude=0.50, name="O-H 1st overtone"),
                    NIRBand(center=1686, sigma=18, gamma=2, amplitude=0.32, name="C-H 1st overtone"),
                    NIRBand(center=2076, sigma=30, gamma=3, amplitude=0.54, name="O-H combination"),
                    NIRBand(center=2266, sigma=22, gamma=2, amplitude=0.38, name="Trehalose combination"),
                ],
                correlation_group=4,
            ),
            # ================================================================
            # EXTENDED PROTEINS (Phase 1.4 additions)
            # ================================================================
            # Albumin: Globular protein (egg white, serum)
            "albumin": SpectralComponent(
                name="albumin",
                bands=[
                    NIRBand(center=1512, sigma=20, gamma=2, amplitude=0.52, name="N-H 1st overtone"),
                    NIRBand(center=1682, sigma=24, gamma=3, amplitude=0.42, name="C-H aromatic"),
                    NIRBand(center=2055, sigma=30, gamma=3, amplitude=0.62, name="N-H combination"),
                    NIRBand(center=2182, sigma=24, gamma=2, amplitude=0.50, name="Protein C-H"),
                    NIRBand(center=2295, sigma=20, gamma=2, amplitude=0.35, name="N-H+Amide III"),
                ],
                correlation_group=2,
            ),
            # Collagen: Fibrous structural protein
            "collagen": SpectralComponent(
                name="collagen",
                bands=[
                    NIRBand(center=1508, sigma=22, gamma=2.5, amplitude=0.48, name="N-H 1st overtone"),
                    NIRBand(center=1560, sigma=20, gamma=2, amplitude=0.35, name="N-H H-bonded"),
                    NIRBand(center=2048, sigma=32, gamma=3, amplitude=0.58, name="N-H combination"),
                    NIRBand(center=2175, sigma=25, gamma=2, amplitude=0.45, name="Protein C-H"),
                    NIRBand(center=2285, sigma=22, gamma=2, amplitude=0.38, name="N-H+Amide III"),
                ],
                correlation_group=2,
            ),
            # Keratin: Structural protein (hair, nails)
            "keratin": SpectralComponent(
                name="keratin",
                bands=[
                    NIRBand(center=1520, sigma=22, gamma=2.5, amplitude=0.50, name="N-H 1st overtone"),
                    NIRBand(center=1685, sigma=22, gamma=2.5, amplitude=0.40, name="C-H aromatic"),
                    NIRBand(center=2060, sigma=30, gamma=3, amplitude=0.55, name="N-H combination"),
                    NIRBand(center=2185, sigma=24, gamma=2, amplitude=0.48, name="Protein C-H"),
                ],
                correlation_group=2,
            ),
            # Zein: Corn protein (prolamin)
            "zein": SpectralComponent(
                name="zein",
                bands=[
                    NIRBand(center=1515, sigma=21, gamma=2, amplitude=0.48, name="N-H 1st overtone"),
                    NIRBand(center=1676, sigma=23, gamma=2.5, amplitude=0.38, name="C-H aromatic"),
                    NIRBand(center=2052, sigma=30, gamma=3, amplitude=0.55, name="N-H combination"),
                    NIRBand(center=2178, sigma=24, gamma=2, amplitude=0.46, name="Protein C-H"),
                ],
                correlation_group=2,
            ),
            # Gelatin: Denatured collagen
            "gelatin": SpectralComponent(
                name="gelatin",
                bands=[
                    NIRBand(center=1505, sigma=22, gamma=2.5, amplitude=0.46, name="N-H 1st overtone"),
                    NIRBand(center=2045, sigma=32, gamma=3.5, amplitude=0.55, name="N-H combination"),
                    NIRBand(center=2172, sigma=26, gamma=2.5, amplitude=0.44, name="Protein C-H"),
                ],
                correlation_group=2,
            ),
            # Whey: Milk serum proteins
            "whey": SpectralComponent(
                name="whey",
                bands=[
                    NIRBand(center=1514, sigma=20, gamma=2, amplitude=0.50, name="N-H 1st overtone"),
                    NIRBand(center=1680, sigma=24, gamma=3, amplitude=0.40, name="C-H aromatic"),
                    NIRBand(center=2052, sigma=30, gamma=3, amplitude=0.58, name="N-H combination"),
                    NIRBand(center=2180, sigma=24, gamma=2, amplitude=0.48, name="Protein C-H"),
                ],
                correlation_group=2,
            ),
            # ================================================================
            # EXTENDED LIPIDS (Phase 1.4 additions)
            # ================================================================
            # Oleic acid: Monounsaturated fatty acid (C18:1)
            "oleic_acid": SpectralComponent(
                name="oleic_acid",
                bands=[
                    NIRBand(center=1162, sigma=15, gamma=1.5, amplitude=0.30, name="=C-H 2nd overtone"),
                    NIRBand(center=1722, sigma=22, gamma=2, amplitude=0.60, name="C-H 1st overtone"),
                    NIRBand(center=2142, sigma=20, gamma=2, amplitude=0.45, name="=C-H 1st overtone"),
                    NIRBand(center=2308, sigma=18, gamma=2, amplitude=0.50, name="CH2 combination"),
                ],
                correlation_group=3,
            ),
            # Linoleic acid: Polyunsaturated fatty acid (C18:2)
            "linoleic_acid": SpectralComponent(
                name="linoleic_acid",
                bands=[
                    NIRBand(center=1158, sigma=15, gamma=1.5, amplitude=0.32, name="=C-H 2nd overtone"),
                    NIRBand(center=1718, sigma=22, gamma=2, amplitude=0.58, name="C-H 1st overtone"),
                    NIRBand(center=2138, sigma=20, gamma=2, amplitude=0.48, name="=C-H 1st overtone"),
                    NIRBand(center=2172, sigma=18, gamma=2, amplitude=0.38, name="C=C combination"),
                    NIRBand(center=2305, sigma=18, gamma=2, amplitude=0.48, name="CH2 combination"),
                ],
                correlation_group=3,
            ),
            # Linolenic acid: Polyunsaturated fatty acid (C18:3)
            "linolenic_acid": SpectralComponent(
                name="linolenic_acid",
                bands=[
                    NIRBand(center=1155, sigma=15, gamma=1.5, amplitude=0.35, name="=C-H 2nd overtone"),
                    NIRBand(center=1715, sigma=22, gamma=2, amplitude=0.55, name="C-H 1st overtone"),
                    NIRBand(center=2135, sigma=20, gamma=2, amplitude=0.52, name="=C-H 1st overtone"),
                    NIRBand(center=2168, sigma=18, gamma=2, amplitude=0.42, name="C=C combination"),
                    NIRBand(center=2302, sigma=18, gamma=2, amplitude=0.45, name="CH2 combination"),
                ],
                correlation_group=3,
            ),
            # Palmitic acid: Saturated fatty acid (C16:0)
            "palmitic_acid": SpectralComponent(
                name="palmitic_acid",
                bands=[
                    NIRBand(center=1196, sigma=18, gamma=2, amplitude=0.42, name="CH2 2nd overtone"),
                    NIRBand(center=1732, sigma=22, gamma=2, amplitude=0.65, name="C-H 1st overtone"),
                    NIRBand(center=2318, sigma=18, gamma=2, amplitude=0.55, name="CH2 combination"),
                    NIRBand(center=2358, sigma=16, gamma=2, amplitude=0.45, name="CH3 combination"),
                ],
                correlation_group=3,
            ),
            # Stearic acid: Saturated fatty acid (C18:0)
            "stearic_acid": SpectralComponent(
                name="stearic_acid",
                bands=[
                    NIRBand(center=1194, sigma=18, gamma=2, amplitude=0.44, name="CH2 2nd overtone"),
                    NIRBand(center=1728, sigma=22, gamma=2, amplitude=0.68, name="C-H 1st overtone"),
                    NIRBand(center=2315, sigma=18, gamma=2, amplitude=0.58, name="CH2 combination"),
                    NIRBand(center=2355, sigma=16, gamma=2, amplitude=0.48, name="CH3 combination"),
                ],
                correlation_group=3,
            ),
            # Phospholipid: Lecithin-like membrane lipids
            "phospholipid": SpectralComponent(
                name="phospholipid",
                bands=[
                    NIRBand(center=1205, sigma=20, gamma=2, amplitude=0.38, name="C-H 2nd overtone"),
                    NIRBand(center=1725, sigma=24, gamma=2.5, amplitude=0.55, name="C-H 1st overtone"),
                    NIRBand(center=2305, sigma=20, gamma=2, amplitude=0.48, name="CH2 combination"),
                    NIRBand(center=2165, sigma=22, gamma=2.5, amplitude=0.35, name="P-O combination"),
                ],
                correlation_group=3,
            ),
            # Cholesterol: Sterol lipid
            "cholesterol": SpectralComponent(
                name="cholesterol",
                bands=[
                    NIRBand(center=1390, sigma=18, gamma=2, amplitude=0.35, name="O-H 1st overtone"),
                    NIRBand(center=1708, sigma=22, gamma=2, amplitude=0.50, name="C-H 1st overtone"),
                    NIRBand(center=2298, sigma=20, gamma=2, amplitude=0.45, name="C-H combination"),
                ],
                correlation_group=3,
            ),
            # Cocoa butter: Triglyceride mix
            "cocoa_butter": SpectralComponent(
                name="cocoa_butter",
                bands=[
                    NIRBand(center=1210, sigma=20, gamma=2, amplitude=0.40, name="C-H 2nd overtone"),
                    NIRBand(center=1728, sigma=24, gamma=2, amplitude=0.65, name="C-H 1st overtone"),
                    NIRBand(center=2312, sigma=20, gamma=2, amplitude=0.52, name="CH2 combination"),
                    NIRBand(center=2352, sigma=18, gamma=2, amplitude=0.42, name="CH3 combination"),
                ],
                correlation_group=3,
            ),
            # ================================================================
            # EXTENDED ALCOHOLS (Phase 1.4 additions)
            # ================================================================
            # Propanol: Propyl alcohol
            "propanol": SpectralComponent(
                name="propanol",
                bands=[
                    NIRBand(center=1415, sigma=18, gamma=2, amplitude=0.58, name="O-H 1st overtone"),
                    NIRBand(center=1575, sigma=25, gamma=3, amplitude=0.42, name="O-H H-bonded"),
                    NIRBand(center=1698, sigma=18, gamma=2, amplitude=0.38, name="C-H 1st overtone"),
                    NIRBand(center=2055, sigma=28, gamma=3, amplitude=0.48, name="O-H combination"),
                    NIRBand(center=2295, sigma=20, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=7,
            ),
            # Butanol: Butyl alcohol
            "butanol": SpectralComponent(
                name="butanol",
                bands=[
                    NIRBand(center=1418, sigma=18, gamma=2, amplitude=0.55, name="O-H 1st overtone"),
                    NIRBand(center=1572, sigma=25, gamma=3, amplitude=0.40, name="O-H H-bonded"),
                    NIRBand(center=1702, sigma=20, gamma=2, amplitude=0.42, name="C-H 1st overtone"),
                    NIRBand(center=2058, sigma=28, gamma=3, amplitude=0.46, name="O-H combination"),
                    NIRBand(center=2298, sigma=20, gamma=2, amplitude=0.40, name="C-H combination"),
                ],
                correlation_group=7,
            ),
            # Sorbitol: Sugar alcohol
            "sorbitol": SpectralComponent(
                name="sorbitol",
                bands=[
                    NIRBand(center=1445, sigma=28, gamma=3, amplitude=0.55, name="O-H 1st overtone"),
                    NIRBand(center=1585, sigma=25, gamma=3, amplitude=0.42, name="O-H H-bonded"),
                    NIRBand(center=1695, sigma=20, gamma=2, amplitude=0.35, name="C-H 1st overtone"),
                    NIRBand(center=2065, sigma=30, gamma=3, amplitude=0.52, name="O-H combination"),
                    NIRBand(center=2282, sigma=22, gamma=2, amplitude=0.40, name="C-H combination"),
                ],
                correlation_group=7,
            ),
            # Mannitol: Sugar alcohol
            "mannitol": SpectralComponent(
                name="mannitol",
                bands=[
                    NIRBand(center=1448, sigma=28, gamma=3, amplitude=0.52, name="O-H 1st overtone"),
                    NIRBand(center=1582, sigma=25, gamma=3, amplitude=0.40, name="O-H H-bonded"),
                    NIRBand(center=1698, sigma=20, gamma=2, amplitude=0.33, name="C-H 1st overtone"),
                    NIRBand(center=2068, sigma=30, gamma=3, amplitude=0.50, name="O-H combination"),
                    NIRBand(center=2285, sigma=22, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=7,
            ),
            # Xylitol: Sugar alcohol
            "xylitol": SpectralComponent(
                name="xylitol",
                bands=[
                    NIRBand(center=1442, sigma=27, gamma=3, amplitude=0.54, name="O-H 1st overtone"),
                    NIRBand(center=1578, sigma=25, gamma=3, amplitude=0.42, name="O-H H-bonded"),
                    NIRBand(center=1692, sigma=19, gamma=2, amplitude=0.34, name="C-H 1st overtone"),
                    NIRBand(center=2062, sigma=30, gamma=3, amplitude=0.52, name="O-H combination"),
                    NIRBand(center=2278, sigma=22, gamma=2, amplitude=0.40, name="C-H combination"),
                ],
                correlation_group=7,
            ),
            # Isopropanol: Isopropyl alcohol
            "isopropanol": SpectralComponent(
                name="isopropanol",
                bands=[
                    NIRBand(center=1412, sigma=18, gamma=2, amplitude=0.58, name="O-H 1st overtone"),
                    NIRBand(center=1568, sigma=24, gamma=3, amplitude=0.44, name="O-H H-bonded"),
                    NIRBand(center=1690, sigma=18, gamma=2, amplitude=0.40, name="C-H 1st overtone"),
                    NIRBand(center=2048, sigma=28, gamma=3, amplitude=0.50, name="O-H combination"),
                    NIRBand(center=2288, sigma=20, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=7,
            ),
            # ================================================================
            # EXTENDED ORGANIC ACIDS (Phase 1.4 additions)
            # ================================================================
            # Formic acid: HCOOH
            "formic_acid": SpectralComponent(
                name="formic_acid",
                bands=[
                    NIRBand(center=1425, sigma=30, gamma=4, amplitude=0.52, name="Carboxylic O-H"),
                    NIRBand(center=1695, sigma=18, gamma=2, amplitude=0.28, name="C=O 2nd overtone"),
                    NIRBand(center=1935, sigma=35, gamma=4, amplitude=0.58, name="O-H combination"),
                ],
                correlation_group=8,
            ),
            # Oxalic acid: (COOH)2
            "oxalic_acid": SpectralComponent(
                name="oxalic_acid",
                bands=[
                    NIRBand(center=1435, sigma=32, gamma=4, amplitude=0.50, name="Carboxylic O-H"),
                    NIRBand(center=1705, sigma=20, gamma=2.5, amplitude=0.35, name="C=O 2nd overtone"),
                    NIRBand(center=1945, sigma=35, gamma=4, amplitude=0.55, name="O-H combination"),
                ],
                correlation_group=8,
            ),
            # Succinic acid: Dicarboxylic acid
            "succinic_acid": SpectralComponent(
                name="succinic_acid",
                bands=[
                    NIRBand(center=1432, sigma=30, gamma=3.5, amplitude=0.48, name="Carboxylic O-H"),
                    NIRBand(center=1705, sigma=20, gamma=2, amplitude=0.32, name="C=O 2nd overtone"),
                    NIRBand(center=1942, sigma=35, gamma=4, amplitude=0.52, name="O-H combination"),
                    NIRBand(center=2245, sigma=22, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=8,
            ),
            # Fumaric acid: Unsaturated dicarboxylic
            "fumaric_acid": SpectralComponent(
                name="fumaric_acid",
                bands=[
                    NIRBand(center=1428, sigma=30, gamma=3.5, amplitude=0.48, name="Carboxylic O-H"),
                    NIRBand(center=1700, sigma=20, gamma=2, amplitude=0.30, name="C=O 2nd overtone"),
                    NIRBand(center=1938, sigma=35, gamma=4, amplitude=0.50, name="O-H combination"),
                    NIRBand(center=2135, sigma=20, gamma=2, amplitude=0.28, name="C=C combination"),
                ],
                correlation_group=8,
            ),
            # Propionic acid: CH3CH2COOH
            "propionic_acid": SpectralComponent(
                name="propionic_acid",
                bands=[
                    NIRBand(center=1422, sigma=30, gamma=4, amplitude=0.52, name="Carboxylic O-H"),
                    NIRBand(center=1698, sigma=18, gamma=2, amplitude=0.30, name="C=O 2nd overtone"),
                    NIRBand(center=1938, sigma=35, gamma=4, amplitude=0.55, name="O-H combination"),
                    NIRBand(center=2242, sigma=22, gamma=2, amplitude=0.36, name="C-H combination"),
                ],
                correlation_group=8,
            ),
            # Butyric acid: Short-chain fatty acid
            "butyric_acid": SpectralComponent(
                name="butyric_acid",
                bands=[
                    NIRBand(center=1420, sigma=30, gamma=4, amplitude=0.50, name="Carboxylic O-H"),
                    NIRBand(center=1702, sigma=18, gamma=2, amplitude=0.32, name="C=O 2nd overtone"),
                    NIRBand(center=1940, sigma=35, gamma=4, amplitude=0.52, name="O-H combination"),
                    NIRBand(center=2248, sigma=22, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=8,
            ),
            # Ascorbic acid: Vitamin C
            "ascorbic_acid": SpectralComponent(
                name="ascorbic_acid",
                bands=[
                    NIRBand(center=1445, sigma=28, gamma=3, amplitude=0.50, name="O-H 1st overtone"),
                    NIRBand(center=1565, sigma=25, gamma=3, amplitude=0.38, name="O-H H-bonded"),
                    NIRBand(center=1918, sigma=32, gamma=3.5, amplitude=0.48, name="C=O combination"),
                    NIRBand(center=2062, sigma=30, gamma=3, amplitude=0.52, name="O-H combination"),
                ],
                correlation_group=8,
            ),
            # ================================================================
            # EXTENDED PHARMACEUTICALS (Phase 1.4 additions)
            # ================================================================
            # Ibuprofen: Anti-inflammatory drug
            "ibuprofen": SpectralComponent(
                name="ibuprofen",
                bands=[
                    NIRBand(center=1148, sigma=15, gamma=1.5, amplitude=0.32, name="Ar C-H 2nd overtone"),
                    NIRBand(center=1432, sigma=25, gamma=3, amplitude=0.38, name="Carboxylic O-H"),
                    NIRBand(center=1682, sigma=18, gamma=2, amplitude=0.48, name="Ar C-H 1st overtone"),
                    NIRBand(center=1722, sigma=20, gamma=2, amplitude=0.42, name="C-H 1st overtone"),
                    NIRBand(center=2025, sigma=25, gamma=2.5, amplitude=0.42, name="C=O combination"),
                ],
                correlation_group=9,
            ),
            # Naproxen: NSAID drug
            "naproxen": SpectralComponent(
                name="naproxen",
                bands=[
                    NIRBand(center=1145, sigma=14, gamma=1.5, amplitude=0.35, name="Ar C-H 2nd overtone"),
                    NIRBand(center=1430, sigma=25, gamma=3, amplitude=0.40, name="Carboxylic O-H"),
                    NIRBand(center=1678, sigma=18, gamma=2, amplitude=0.50, name="Ar C-H 1st overtone"),
                    NIRBand(center=2028, sigma=25, gamma=2.5, amplitude=0.45, name="C=O combination"),
                    NIRBand(center=2140, sigma=22, gamma=2, amplitude=0.38, name="Ar combination"),
                ],
                correlation_group=9,
            ),
            # Diclofenac: NSAID drug
            "diclofenac": SpectralComponent(
                name="diclofenac",
                bands=[
                    NIRBand(center=1142, sigma=15, gamma=1.5, amplitude=0.33, name="Ar C-H 2nd overtone"),
                    NIRBand(center=1505, sigma=20, gamma=2, amplitude=0.42, name="N-H 1st overtone"),
                    NIRBand(center=1675, sigma=18, gamma=2, amplitude=0.48, name="Ar C-H 1st overtone"),
                    NIRBand(center=2032, sigma=26, gamma=2.5, amplitude=0.44, name="C=O combination"),
                ],
                correlation_group=9,
            ),
            # Metformin: Diabetes drug
            "metformin": SpectralComponent(
                name="metformin",
                bands=[
                    NIRBand(center=1485, sigma=20, gamma=2, amplitude=0.45, name="N-H 1st overtone"),
                    NIRBand(center=1545, sigma=22, gamma=2.5, amplitude=0.40, name="N-H asym"),
                    NIRBand(center=2045, sigma=28, gamma=3, amplitude=0.52, name="N-H combination"),
                    NIRBand(center=2168, sigma=22, gamma=2, amplitude=0.38, name="C-N combination"),
                ],
                correlation_group=9,
            ),
            # Omeprazole: Proton pump inhibitor
            "omeprazole": SpectralComponent(
                name="omeprazole",
                bands=[
                    NIRBand(center=1148, sigma=15, gamma=1.5, amplitude=0.30, name="Ar C-H 2nd overtone"),
                    NIRBand(center=1498, sigma=20, gamma=2, amplitude=0.42, name="N-H 1st overtone"),
                    NIRBand(center=1672, sigma=18, gamma=2, amplitude=0.45, name="Ar C-H 1st overtone"),
                    NIRBand(center=2038, sigma=26, gamma=2.5, amplitude=0.48, name="S=O combination"),
                ],
                correlation_group=9,
            ),
            # Amoxicillin: Antibiotic
            "amoxicillin": SpectralComponent(
                name="amoxicillin",
                bands=[
                    NIRBand(center=1148, sigma=15, gamma=1.5, amplitude=0.28, name="Ar C-H 2nd overtone"),
                    NIRBand(center=1390, sigma=20, gamma=2.5, amplitude=0.40, name="Phenolic O-H"),
                    NIRBand(center=1512, sigma=20, gamma=2, amplitude=0.45, name="N-H 1st overtone"),
                    NIRBand(center=1675, sigma=18, gamma=2, amplitude=0.42, name="Ar C-H 1st overtone"),
                    NIRBand(center=2050, sigma=28, gamma=3, amplitude=0.50, name="Amide combination"),
                ],
                correlation_group=9,
            ),
            # Microcrystalline cellulose: Pharmaceutical excipient
            "microcrystalline_cellulose": SpectralComponent(
                name="microcrystalline_cellulose",
                bands=[
                    NIRBand(center=1492, sigma=22, gamma=2, amplitude=0.42, name="O-H 1st overtone"),
                    NIRBand(center=1782, sigma=18, gamma=2, amplitude=0.32, name="Cellulose C-H"),
                    NIRBand(center=2092, sigma=28, gamma=3, amplitude=0.52, name="O-H combination"),
                    NIRBand(center=2282, sigma=22, gamma=2, amplitude=0.42, name="Cellulose C-O"),
                    NIRBand(center=2342, sigma=20, gamma=2, amplitude=0.36, name="C-H combination"),
                ],
                correlation_group=4,
            ),
            # ================================================================
            # EXTENDED POLYMERS (Phase 1.4 additions)
            # ================================================================
            # PMMA: Polymethyl methacrylate (acrylic)
            "pmma": SpectralComponent(
                name="pmma",
                bands=[
                    NIRBand(center=1190, sigma=18, gamma=2, amplitude=0.38, name="C-H 2nd overtone"),
                    NIRBand(center=1718, sigma=22, gamma=2, amplitude=0.52, name="C-H 1st overtone"),
                    NIRBand(center=2015, sigma=25, gamma=2.5, amplitude=0.42, name="Ester C=O"),
                    NIRBand(center=2298, sigma=20, gamma=2, amplitude=0.45, name="C-H combination"),
                ],
                correlation_group=11,
            ),
            # PVC: Polyvinyl chloride
            "pvc": SpectralComponent(
                name="pvc",
                bands=[
                    NIRBand(center=1188, sigma=17, gamma=1.8, amplitude=0.35, name="C-H 2nd overtone"),
                    NIRBand(center=1715, sigma=22, gamma=2, amplitude=0.48, name="C-H 1st overtone"),
                    NIRBand(center=2295, sigma=20, gamma=2, amplitude=0.42, name="C-H combination"),
                ],
                correlation_group=11,
            ),
            # PP: Polypropylene
            "polypropylene": SpectralComponent(
                name="polypropylene",
                bands=[
                    NIRBand(center=1185, sigma=18, gamma=1.8, amplitude=0.38, name="C-H 2nd overtone"),
                    NIRBand(center=1392, sigma=15, gamma=1.5, amplitude=0.30, name="C-H combination"),
                    NIRBand(center=1718, sigma=22, gamma=2, amplitude=0.58, name="C-H 1st overtone"),
                    NIRBand(center=2305, sigma=18, gamma=2, amplitude=0.48, name="CH2 combination"),
                    NIRBand(center=2348, sigma=16, gamma=2, amplitude=0.42, name="CH3 combination"),
                ],
                correlation_group=11,
            ),
            # PET: Polyethylene terephthalate (separate from polyester for clarity)
            "pet": SpectralComponent(
                name="pet",
                bands=[
                    NIRBand(center=1138, sigma=14, gamma=1.5, amplitude=0.36, name="Ar C-H 2nd overtone"),
                    NIRBand(center=1658, sigma=16, gamma=2, amplitude=0.46, name="Ar C-H 1st overtone"),
                    NIRBand(center=1722, sigma=20, gamma=2, amplitude=0.52, name="C-H 1st overtone"),
                    NIRBand(center=2018, sigma=25, gamma=2.5, amplitude=0.42, name="Ester C=O"),
                    NIRBand(center=2132, sigma=22, gamma=2, amplitude=0.34, name="Ar combination"),
                ],
                correlation_group=11,
            ),
            # PTFE: Polytetrafluoroethylene (Teflon)
            "ptfe": SpectralComponent(
                name="ptfe",
                bands=[
                    NIRBand(center=2180, sigma=25, gamma=2.5, amplitude=0.35, name="C-F combination"),
                    NIRBand(center=2365, sigma=22, gamma=2, amplitude=0.30, name="C-F overtone"),
                ],
                correlation_group=11,
            ),
            # ABS: Acrylonitrile butadiene styrene
            "abs": SpectralComponent(
                name="abs",
                bands=[
                    NIRBand(center=1145, sigma=15, gamma=1.5, amplitude=0.38, name="Ar C-H 2nd overtone"),
                    NIRBand(center=1158, sigma=16, gamma=1.5, amplitude=0.28, name="=C-H 2nd overtone"),
                    NIRBand(center=1682, sigma=18, gamma=2, amplitude=0.52, name="Ar C-H 1st overtone"),
                    NIRBand(center=1718, sigma=20, gamma=2, amplitude=0.45, name="C-H 1st overtone"),
                    NIRBand(center=2172, sigma=22, gamma=2, amplitude=0.42, name="Ar C-H combination"),
                ],
                correlation_group=11,
            ),
            # ================================================================
            # EXTENDED MINERALS (Phase 1.4 additions)
            # ================================================================
            # Montmorillonite: Smectite clay
            "montmorillonite": SpectralComponent(
                name="montmorillonite",
                bands=[
                    NIRBand(center=1410, sigma=22, gamma=2.5, amplitude=0.42, name="Al-OH 1st overtone"),
                    NIRBand(center=1910, sigma=32, gamma=3.5, amplitude=0.45, name="H-O-H combination"),
                    NIRBand(center=2210, sigma=25, gamma=2.5, amplitude=0.52, name="Al-OH combination"),
                ],
                correlation_group=13,
            ),
            # Illite: Mica-like clay
            "illite": SpectralComponent(
                name="illite",
                bands=[
                    NIRBand(center=1405, sigma=20, gamma=2.5, amplitude=0.40, name="Al-OH 1st overtone"),
                    NIRBand(center=2205, sigma=24, gamma=2.5, amplitude=0.50, name="Al-OH combination"),
                    NIRBand(center=2345, sigma=22, gamma=2, amplitude=0.35, name="Fe-OH combination"),
                ],
                correlation_group=13,
            ),
            # Goethite: Iron oxyhydroxide
            "goethite": SpectralComponent(
                name="goethite",
                bands=[
                    NIRBand(center=1420, sigma=25, gamma=3, amplitude=0.38, name="Fe-OH 1st overtone"),
                    NIRBand(center=1920, sigma=30, gamma=3, amplitude=0.42, name="Fe-OH combination"),
                    NIRBand(center=2260, sigma=25, gamma=2.5, amplitude=0.45, name="Fe-OH combination 2"),
                ],
                correlation_group=13,
            ),
            # Talc: Magnesium silicate
            "talc": SpectralComponent(
                name="talc",
                bands=[
                    NIRBand(center=1395, sigma=18, gamma=2, amplitude=0.42, name="Mg-OH 1st overtone"),
                    NIRBand(center=2315, sigma=22, gamma=2.5, amplitude=0.52, name="Mg-OH combination"),
                    NIRBand(center=2390, sigma=20, gamma=2, amplitude=0.38, name="Mg-OH combination 2"),
                ],
                correlation_group=13,
            ),
            # Silica: Silicon dioxide
            "silica": SpectralComponent(
                name="silica",
                bands=[
                    NIRBand(center=1380, sigma=20, gamma=2, amplitude=0.35, name="Si-OH 1st overtone"),
                    NIRBand(center=1900, sigma=35, gamma=4, amplitude=0.40, name="H2O combination"),
                    NIRBand(center=2220, sigma=25, gamma=2.5, amplitude=0.45, name="Si-OH combination"),
                ],
                correlation_group=13,
            ),
            # ================================================================
            # EXTENDED PIGMENTS (Phase 1.4 additions)
            # ================================================================
            # Anthocyanin: Red-purple plant pigment
            "anthocyanin": SpectralComponent(
                name="anthocyanin",
                bands=[
                    NIRBand(center=1040, sigma=18, gamma=2, amplitude=0.25, name="Electronic tail"),
                    NIRBand(center=1425, sigma=22, gamma=2.5, amplitude=0.38, name="Phenolic O-H"),
                    NIRBand(center=1672, sigma=18, gamma=2, amplitude=0.42, name="Ar C-H 1st overtone"),
                    NIRBand(center=2055, sigma=22, gamma=2.5, amplitude=0.35, name="Phenolic combination"),
                ],
                correlation_group=5,
            ),
            # Lycopene: Red carotenoid (tomatoes)
            "lycopene": SpectralComponent(
                name="lycopene",
                bands=[
                    NIRBand(center=1055, sigma=22, gamma=2.5, amplitude=0.28, name="Electronic tail"),
                    NIRBand(center=1685, sigma=18, gamma=2, amplitude=0.42, name="=C-H 1st overtone"),
                    NIRBand(center=2138, sigma=22, gamma=2, amplitude=0.38, name="C=C combination"),
                    NIRBand(center=2282, sigma=20, gamma=2, amplitude=0.32, name="C-H combination"),
                ],
                correlation_group=5,
            ),
            # Lutein: Yellow carotenoid (xanthophyll)
            "lutein": SpectralComponent(
                name="lutein",
                bands=[
                    NIRBand(center=1048, sigma=20, gamma=2, amplitude=0.26, name="Electronic tail"),
                    NIRBand(center=1415, sigma=20, gamma=2, amplitude=0.35, name="O-H 1st overtone"),
                    NIRBand(center=1678, sigma=18, gamma=2, amplitude=0.40, name="=C-H 1st overtone"),
                    NIRBand(center=2132, sigma=22, gamma=2, amplitude=0.35, name="C=C combination"),
                ],
                correlation_group=5,
            ),
            # Xanthophyll: General yellow pigments
            "xanthophyll": SpectralComponent(
                name="xanthophyll",
                bands=[
                    NIRBand(center=1052, sigma=20, gamma=2, amplitude=0.25, name="Electronic tail"),
                    NIRBand(center=1418, sigma=20, gamma=2, amplitude=0.36, name="O-H 1st overtone"),
                    NIRBand(center=1682, sigma=18, gamma=2, amplitude=0.42, name="=C-H 1st overtone"),
                    NIRBand(center=2135, sigma=22, gamma=2, amplitude=0.36, name="C=C combination"),
                ],
                correlation_group=5,
            ),
            # Melanin: Brown-black pigment
            "melanin": SpectralComponent(
                name="melanin",
                bands=[
                    NIRBand(center=1100, sigma=30, gamma=3, amplitude=0.35, name="Electronic absorption"),
                    NIRBand(center=1510, sigma=22, gamma=2.5, amplitude=0.40, name="N-H 1st overtone"),
                    NIRBand(center=1680, sigma=20, gamma=2, amplitude=0.38, name="Ar C-H 1st overtone"),
                    NIRBand(center=2055, sigma=28, gamma=3, amplitude=0.42, name="N-H combination"),
                ],
                correlation_group=5,
            ),
            # ================================================================
            # EXTENDED SOLVENTS (Phase 1.4 additions)
            # ================================================================
            # Dimethyl sulfoxide (DMSO)
            "dmso": SpectralComponent(
                name="dmso",
                bands=[
                    NIRBand(center=1700, sigma=20, gamma=2, amplitude=0.40, name="C-H 1st overtone"),
                    NIRBand(center=2020, sigma=25, gamma=2.5, amplitude=0.45, name="S=O combination"),
                    NIRBand(center=2290, sigma=20, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=12,
            ),
            # Ethyl acetate: Ester solvent
            "ethyl_acetate": SpectralComponent(
                name="ethyl_acetate",
                bands=[
                    NIRBand(center=1695, sigma=18, gamma=2, amplitude=0.42, name="C-H 1st overtone"),
                    NIRBand(center=1720, sigma=20, gamma=2, amplitude=0.38, name="C-H 1st overtone"),
                    NIRBand(center=2010, sigma=25, gamma=2.5, amplitude=0.40, name="Ester C=O"),
                    NIRBand(center=2285, sigma=20, gamma=2, amplitude=0.38, name="C-H combination"),
                ],
                correlation_group=12,
            ),
            # Toluene: Aromatic solvent
            "toluene": SpectralComponent(
                name="toluene",
                bands=[
                    NIRBand(center=1142, sigma=14, gamma=1.5, amplitude=0.38, name="Ar C-H 2nd overtone"),
                    NIRBand(center=1678, sigma=17, gamma=2, amplitude=0.55, name="Ar C-H 1st overtone"),
                    NIRBand(center=1705, sigma=18, gamma=2, amplitude=0.42, name="CH3 1st overtone"),
                    NIRBand(center=2145, sigma=20, gamma=2, amplitude=0.42, name="Ar C-H combination"),
                ],
                correlation_group=12,
            ),
            # Chloroform: Halogenated solvent
            "chloroform": SpectralComponent(
                name="chloroform",
                bands=[
                    NIRBand(center=1695, sigma=18, gamma=2, amplitude=0.38, name="C-H 1st overtone"),
                    NIRBand(center=2250, sigma=22, gamma=2.5, amplitude=0.32, name="C-Cl combination"),
                ],
                correlation_group=12,
            ),
            # Hexane: Alkane solvent
            "hexane": SpectralComponent(
                name="hexane",
                bands=[
                    NIRBand(center=1192, sigma=18, gamma=2, amplitude=0.40, name="C-H 2nd overtone"),
                    NIRBand(center=1718, sigma=24, gamma=2.5, amplitude=0.62, name="C-H 1st overtone"),
                    NIRBand(center=2308, sigma=20, gamma=2, amplitude=0.52, name="CH2 combination"),
                    NIRBand(center=2358, sigma=18, gamma=2, amplitude=0.42, name="CH3 combination"),
                ],
                correlation_group=12,
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
