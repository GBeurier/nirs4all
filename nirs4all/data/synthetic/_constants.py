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

    Available Components (121 total):
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

        Pigments (18):
            Chlorophylls:
            - ``chlorophyll``: Chlorophyll a/b combined [2, pp. 375-378]
            - ``chlorophyll_a``: Primary photosynthetic pigment (Soret 430nm, Q 662nm)
            - ``chlorophyll_b``: Accessory photosynthetic pigment (Soret 453nm, Q 642nm)
            Carotenoids:
            - ``carotenoid``: β-carotene and xanthophylls [2, pp. 378-380]
            - ``beta_carotene``: Orange carotenoid (λmax 425, 450, 478nm)
            - ``lycopene``: Red carotenoid (tomatoes)
            - ``lutein``: Yellow carotenoid (xanthophyll)
            - ``xanthophyll``: General yellow pigments
            Flavonoids:
            - ``anthocyanin``: Red-purple plant pigment
            - ``anthocyanin_red``: Red anthocyanin (520-540nm)
            - ``anthocyanin_purple``: Purple anthocyanin (560-580nm, pH-dependent)
            Hemoproteins (visible-region electronic transitions):
            - ``hemoglobin_oxy``: Oxygenated hemoglobin (Soret 414nm, Q 542/577nm)
            - ``hemoglobin_deoxy``: Deoxygenated hemoglobin (Soret 430nm, Q 555nm)
            - ``myoglobin``: Muscle oxygen-binding protein
            - ``cytochrome_c``: Electron transport hemoprotein
            Other pigments:
            - ``bilirubin``: Bile pigment (heme degradation product)
            - ``melanin``: Brown-black biopolymer pigment
            - ``tannins``: Polyphenolic compounds [6], [11]

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

        Polymers (11):
            - ``polyethylene``: HDPE/LDPE plastic [15], [1, pp. 58-60]
            - ``polystyrene``: Aromatic polymer [15], [1, pp. 56-58]
            - ``polypropylene``: PP plastic
            - ``pvc``: Polyvinyl chloride
            - ``pet``: Polyethylene terephthalate
            - ``polyester``: PET fiber [1, pp. 60-62]
            - ``nylon``: Polyamide fiber [1, pp. 60-62]
            - ``pmma``: Polymethyl methacrylate (acrylic)
            - ``ptfe``: Polytetrafluoroethylene (Teflon)
            - ``abs``: Acrylonitrile butadiene styrene
            - ``natural_rubber``: cis-1,4-polyisoprene [15]

        Solvents (6):
            - ``acetone``: Ketone solvent [1, pp. 42-44]
            - ``dmso``: Dimethyl sulfoxide
            - ``ethyl_acetate``: Ester solvent
            - ``toluene``: Aromatic solvent
            - ``chloroform``: Halogenated solvent
            - ``hexane``: Alkane solvent

        Minerals (8):
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
            # WATER AND MOISTURE (Phase 2 extended with 2nd/3rd overtones)
            # ================================================================
            # Water: O-H stretching vibrations
            # Refs: [1] pp. 34-36, [2] pp. 358-362
            # Phase 2: Extended to include 2nd overtone (970 nm) and 3rd overtone (720 nm)
            "water": SpectralComponent(
                name="water",
                bands=[
                    # O-H 3rd overtone: 4ν (~13700 cm⁻¹ = 730 nm)
                    # Ref: Curcio & Petty (1951), very weak
                    NIRBand(center=730, sigma=18, gamma=2, amplitude=0.08, name="O-H 3rd overtone"),
                    # O-H 2nd overtone: 3ν (~10300 cm⁻¹ = 970 nm)
                    # Ref: [1] p. 35, weak band
                    NIRBand(center=970, sigma=22, gamma=2.5, amplitude=0.18, name="O-H 2nd overtone"),
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
                category="water_related",
                formula="H2O",
                cas_number="7732-18-5",
                references=["Workman2012 pp.34-36", "Burns2007 pp.358-362"],
                tags=["universal", "moisture", "food", "agriculture", "pharma"],
            ),
            # Moisture: Bound water in organic matrices (shifted positions)
            # Refs: [2] pp. 358-362, [5] pp. 157-158
            # Phase 2: Extended with 2nd/3rd overtones (shifted by H-bonding)
            "moisture": SpectralComponent(
                name="moisture",
                bands=[
                    # Bound O-H 3rd overtone (shifted from free water)
                    NIRBand(center=740, sigma=20, gamma=2.5, amplitude=0.06, name="Bound O-H 3rd overtone"),
                    # Bound O-H 2nd overtone (shifted from free water)
                    NIRBand(center=985, sigma=25, gamma=3, amplitude=0.14, name="Bound O-H 2nd overtone"),
                    # Bound O-H 1st overtone (shifted from free water)
                    # Ref: [5] p. 158
                    NIRBand(center=1460, sigma=30, gamma=4, amplitude=0.78, name="Bound O-H 1st overtone"),
                    # Bound O-H combination (broader than free water)
                    # Ref: [2] p. 361
                    NIRBand(center=1930, sigma=35, gamma=5, amplitude=1.0, name="Bound O-H combination"),
                ],
                correlation_group=1,
                category="water_related",
                synonyms=["bound water", "hydration"],
                references=["Burns2007 pp.358-362", "Williams2001 pp.157-158"],
                tags=["universal", "moisture", "food", "agriculture"],
            ),
            # ================================================================
            # PROTEINS AND NITROGEN COMPOUNDS (Phase 2 extended with 2nd/3rd overtones)
            # ================================================================
            # Protein: Amide bonds, N-H, aromatic C-H
            # Refs: [1] pp. 48-52, [2] pp. 362-366
            # Phase 2: Extended with 2nd and 3rd overtone bands
            "protein": SpectralComponent(
                name="protein",
                bands=[
                    # N-H 3rd overtone: 4ν (~750 nm)
                    # Very weak band in visible-NIR transition region
                    NIRBand(center=750, sigma=18, gamma=2, amplitude=0.05, name="N-H 3rd overtone"),
                    # Aromatic C-H 3rd overtone (Phe, Tyr, Trp)
                    # ~875 nm for aromatic C-H
                    NIRBand(center=880, sigma=15, gamma=2, amplitude=0.08, name="Ar C-H 3rd overtone"),
                    # N-H 2nd overtone: 3ν (~1030 nm)
                    # Ref: [1] p. 50, weak band
                    NIRBand(center=1030, sigma=18, gamma=2, amplitude=0.18, name="N-H 2nd overtone"),
                    # Aromatic C-H 2nd overtone (Phe, Tyr, Trp residues)
                    # ~1140 nm characteristic band
                    NIRBand(center=1140, sigma=14, gamma=1.5, amplitude=0.22, name="Ar C-H 2nd overtone"),
                    # N-H 1st overtone (2ν of amide A band at 3300 cm⁻¹)
                    # Ref: [1] p. 50, Table 2.8
                    NIRBand(center=1510, sigma=20, gamma=2, amplitude=0.83, name="N-H 1st overtone"),
                    # C-H aromatic stretching (Phe, Tyr, Trp residues)
                    # Ref: [1] p. 51
                    NIRBand(center=1680, sigma=25, gamma=3, amplitude=0.67, name="C-H aromatic"),
                    # N-H combination: Amide A + Amide II (~4860 cm⁻¹ = 2055 nm)
                    # Ref: [1] p. 50, strong protein marker
                    NIRBand(center=2050, sigma=30, gamma=3, amplitude=1.0, name="N-H combination"),
                    # Protein C-H: combination bands from aliphatic residues
                    # Ref: [2] p. 365
                    NIRBand(center=2180, sigma=25, gamma=2, amplitude=0.83, name="Protein C-H"),
                    # N-H + Amide III combination
                    # Ref: [1] p. 51
                    NIRBand(center=2300, sigma=20, gamma=2, amplitude=0.5, name="N-H+Amide III"),
                ],
                correlation_group=2,
                category="proteins",
                references=["Workman2012 pp.48-52", "Burns2007 pp.362-366"],
                tags=["food", "agriculture", "pharma", "biological"],
            ),
            # Nitrogen compounds: Primary/secondary amines
            # Refs: [1] pp. 52-54
            "nitrogen_compound": SpectralComponent(
                name="nitrogen_compound",
                bands=[
                    # N-H 1st overtone (free amine)
                    # Ref: [1] p. 53, Table 2.9
                    NIRBand(center=1500, sigma=18, gamma=2, amplitude=0.9, name="N-H 1st overtone"),
                    # N-H combination band
                    # Ref: [1] p. 53
                    NIRBand(center=2060, sigma=25, gamma=2, amplitude=1.0, name="N-H combination"),
                    # N-H + C-N combination
                    # Ref: [1] p. 54
                    NIRBand(center=2150, sigma=22, gamma=2, amplitude=0.8, name="N-H+C-N"),
                ],
                correlation_group=2,
                category="proteins",
                subcategory="amines",
                references=["Workman2012 pp.52-54"],
                tags=["agriculture", "chemical"],
            ),
            # Urea: Carbonyl + amine functional groups
            # Refs: [9] p. 1125, [2] p. 366
            "urea": SpectralComponent(
                name="urea",
                bands=[
                    # N-H 1st overtone (symmetric stretch)
                    # Ref: [9] p. 1125
                    NIRBand(center=1480, sigma=18, gamma=2, amplitude=0.91, name="N-H sym 1st overtone"),
                    # N-H 1st overtone (asymmetric stretch)
                    # Ref: [9] p. 1125
                    NIRBand(center=1530, sigma=20, gamma=2, amplitude=0.82, name="N-H asym 1st overtone"),
                    # N-H combination with C=O
                    # Ref: [9] p. 1125
                    NIRBand(center=2010, sigma=25, gamma=3, amplitude=1.0, name="N-H + C=O combination"),
                    # N-H bending combination
                    # Ref: [2] p. 366
                    NIRBand(center=2170, sigma=22, gamma=2, amplitude=0.64, name="N-H bend combination"),
                ],
                correlation_group=2,
                category="proteins",
                subcategory="amines",
                formula="CH4N2O",
                cas_number="57-13-6",
                references=["Reich2005 p.1125", "Burns2007 p.366"],
                tags=["agriculture", "pharma", "fertilizer"],
            ),
            # Amino acid: Free amino acids with carboxyl and amine groups
            # Refs: [3] pp. 215-220
            "amino_acid": SpectralComponent(
                name="amino_acid",
                bands=[
                    # N-H 1st overtone (NH₃⁺ stretching in zwitterion)
                    # Ref: [3] p. 216
                    NIRBand(center=1520, sigma=22, gamma=2.5, amplitude=0.9, name="NH₃⁺ 1st overtone"),
                    # N-H combination
                    # Ref: [3] p. 217
                    NIRBand(center=2040, sigma=28, gamma=3, amplitude=1.0, name="N-H combination"),
                    # C-H combination (α-carbon)
                    # Ref: [3] p. 218
                    NIRBand(center=2260, sigma=20, gamma=2, amplitude=0.7, name="C-H combination"),
                ],
                correlation_group=2,
                category="proteins",
                subcategory="amino_acids",
                references=["Siesler2002 pp.215-220"],
                tags=["food", "pharma", "biological"],
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
            # PLANT PIGMENTS (Phase 2 extended with visible-region bands)
            # ================================================================
            # Chlorophyll: Chlorophyll a and b combined
            # Refs: [2] pp. 375-378, Wellburn (1994) spectral measurements
            # Phase 2: Added visible-region electronic transitions (Soret & Q bands)
            "chlorophyll": SpectralComponent(
                name="chlorophyll",
                bands=[
                    # Visible region - Soret band (blue absorption, mixed a+b)
                    # Chlorophyll a Soret ~430 nm, b Soret ~453 nm
                    NIRBand(center=435, sigma=15, gamma=3, amplitude=0.9, name="Soret band (blue)"),
                    # Visible region - Q band (red absorption, mixed a+b)
                    # Chlorophyll a Q ~662 nm, b Q ~642 nm
                    NIRBand(center=655, sigma=12, gamma=2, amplitude=1.0, name="Q band (red)"),
                    # Red edge - electronic tail ~700-750 nm
                    NIRBand(center=720, sigma=20, gamma=3, amplitude=0.35, name="Red edge tail"),
                    # NIR - Electronic absorption tail
                    # Ref: [2] p. 376
                    NIRBand(center=1070, sigma=15, gamma=1, amplitude=0.2, name="NIR electronic tail"),
                    # C-H 1st overtone (methyl groups on ring)
                    # Ref: [2] p. 377
                    NIRBand(center=1400, sigma=20, gamma=2, amplitude=0.3, name="C-H 1st overtone"),
                    # N-H/C-H combination (porphyrin ring)
                    # Ref: [2] p. 377
                    NIRBand(center=1730, sigma=18, gamma=2, amplitude=0.25, name="Porphyrin C-H"),
                    # C-H combination
                    # Ref: [2] p. 378
                    NIRBand(center=2270, sigma=22, gamma=2, amplitude=0.25, name="C-H combination"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="chlorophylls",
                formula="C55H72MgN4O5",
                tags=["agriculture", "biological", "vis-nir", "photosynthesis"],
            ),
            # Chlorophyll a: Primary photosynthetic pigment
            # Refs: Wellburn (1994), Lichtenthaler (1987)
            # Soret band: 430 nm, Q band: 662 nm
            "chlorophyll_a": SpectralComponent(
                name="chlorophyll_a",
                bands=[
                    # Visible - Soret band (blue absorption)
                    NIRBand(center=430, sigma=14, gamma=3, amplitude=0.85, name="Soret band"),
                    # Visible - Q band (red absorption, strongest)
                    NIRBand(center=662, sigma=10, gamma=2, amplitude=1.0, name="Q band"),
                    # Red edge tail
                    NIRBand(center=715, sigma=18, gamma=2.5, amplitude=0.3, name="Red edge"),
                    # NIR vibrational bands (weaker)
                    NIRBand(center=1070, sigma=15, gamma=1, amplitude=0.18, name="NIR electronic"),
                    NIRBand(center=1400, sigma=20, gamma=2, amplitude=0.25, name="C-H 1st overtone"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="chlorophylls",
                formula="C55H72MgN4O5",
                cas_number="479-61-8",
                tags=["agriculture", "biological", "vis-nir", "photosynthesis"],
            ),
            # Chlorophyll b: Accessory photosynthetic pigment
            # Refs: Wellburn (1994), Lichtenthaler (1987)
            # Soret band: 453 nm, Q band: 642 nm
            "chlorophyll_b": SpectralComponent(
                name="chlorophyll_b",
                bands=[
                    # Visible - Soret band (blue absorption, shifted from a)
                    NIRBand(center=453, sigma=15, gamma=3, amplitude=0.95, name="Soret band"),
                    # Visible - Q band (red absorption, strongest)
                    NIRBand(center=642, sigma=11, gamma=2, amplitude=1.0, name="Q band"),
                    # Red edge tail
                    NIRBand(center=705, sigma=18, gamma=2.5, amplitude=0.28, name="Red edge"),
                    # NIR vibrational bands
                    NIRBand(center=1065, sigma=15, gamma=1, amplitude=0.16, name="NIR electronic"),
                    NIRBand(center=1395, sigma=20, gamma=2, amplitude=0.22, name="C-H 1st overtone"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="chlorophylls",
                formula="C55H70MgN4O6",
                cas_number="519-62-0",
                tags=["agriculture", "biological", "vis-nir", "photosynthesis"],
            ),
            # Carotenoid: β-carotene and xanthophylls (general)
            # Refs: [2] pp. 378-380, Britton et al. (2004) Carotenoids Handbook
            # Phase 2: Added visible-region electronic transitions
            "carotenoid": SpectralComponent(
                name="carotenoid",
                bands=[
                    # Visible - conjugated polyene absorptions (3 peaks typical)
                    NIRBand(center=425, sigma=12, gamma=2, amplitude=0.75, name="π-π* transition 1"),
                    NIRBand(center=450, sigma=14, gamma=2.5, amplitude=1.0, name="π-π* transition 2 (max)"),
                    NIRBand(center=480, sigma=14, gamma=2.5, amplitude=0.85, name="π-π* transition 3"),
                    # NIR - Electronic absorption tail (conjugated polyene)
                    NIRBand(center=1050, sigma=20, gamma=2, amplitude=0.18, name="Electronic tail"),
                    # C-H 1st overtone (polyene chain)
                    # Ref: [2] p. 379
                    NIRBand(center=1680, sigma=18, gamma=2, amplitude=0.3, name="=C-H 1st overtone"),
                    # C=C + C-H combination
                    NIRBand(center=2135, sigma=22, gamma=2, amplitude=0.25, name="C=C combination"),
                    # C-H combination
                    NIRBand(center=2280, sigma=20, gamma=2, amplitude=0.22, name="C-H combination"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="carotenoids",
                tags=["food", "agriculture", "biological", "vis-nir"],
            ),
            # Beta-carotene: Orange carotenoid pigment
            # Refs: Britton et al. (2004), λmax: 425, 450, 478 nm in hexane
            "beta_carotene": SpectralComponent(
                name="beta_carotene",
                bands=[
                    # Visible - three characteristic peaks
                    NIRBand(center=425, sigma=11, gamma=2, amplitude=0.72, name="π-π* (0-2)"),
                    NIRBand(center=450, sigma=13, gamma=2.5, amplitude=1.0, name="π-π* (0-1) max"),
                    NIRBand(center=478, sigma=14, gamma=2.5, amplitude=0.82, name="π-π* (0-0)"),
                    # NIR vibrational
                    NIRBand(center=1050, sigma=20, gamma=2, amplitude=0.15, name="Electronic tail"),
                    NIRBand(center=1685, sigma=18, gamma=2, amplitude=0.28, name="=C-H 1st overtone"),
                    NIRBand(center=2140, sigma=22, gamma=2, amplitude=0.22, name="C=C combination"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="carotenoids",
                formula="C40H56",
                cas_number="7235-40-7",
                tags=["food", "agriculture", "vis-nir"],
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
            # EXTENDED PIGMENTS (Phase 2 with visible-region bands)
            # ================================================================
            # Anthocyanin: Red-purple plant pigment
            # Refs: Giusti & Wrolstad (2001), λmax 520 nm (pelargonidin) to 540 nm (cyanidin)
            "anthocyanin": SpectralComponent(
                name="anthocyanin",
                bands=[
                    # Visible - main absorption (red-orange region)
                    NIRBand(center=520, sigma=25, gamma=3, amplitude=1.0, name="Visible absorption max"),
                    NIRBand(center=280, sigma=15, gamma=2, amplitude=0.55, name="UV band (aromatic)"),
                    # NIR bands
                    NIRBand(center=1040, sigma=18, gamma=2, amplitude=0.18, name="Electronic tail"),
                    NIRBand(center=1425, sigma=22, gamma=2.5, amplitude=0.28, name="Phenolic O-H"),
                    NIRBand(center=1672, sigma=18, gamma=2, amplitude=0.32, name="Ar C-H 1st overtone"),
                    NIRBand(center=2055, sigma=22, gamma=2.5, amplitude=0.25, name="Phenolic combination"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="flavonoids",
                tags=["food", "biological", "vis-nir"],
            ),
            # Anthocyanin red variant: Pelargonidin-based (strawberries)
            "anthocyanin_red": SpectralComponent(
                name="anthocyanin_red",
                bands=[
                    # Visible - pelargonidin λmax ~500-520 nm
                    NIRBand(center=508, sigma=22, gamma=3, amplitude=1.0, name="Visible max (red)"),
                    NIRBand(center=275, sigma=14, gamma=2, amplitude=0.50, name="UV band"),
                    # NIR bands
                    NIRBand(center=1035, sigma=18, gamma=2, amplitude=0.16, name="Electronic tail"),
                    NIRBand(center=1420, sigma=22, gamma=2.5, amplitude=0.26, name="Phenolic O-H"),
                    NIRBand(center=1670, sigma=18, gamma=2, amplitude=0.30, name="Ar C-H 1st overtone"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="flavonoids",
                tags=["food", "vis-nir"],
            ),
            # Anthocyanin purple variant: Delphinidin-based (grapes, blueberries)
            "anthocyanin_purple": SpectralComponent(
                name="anthocyanin_purple",
                bands=[
                    # Visible - delphinidin λmax ~540-560 nm (more purple)
                    NIRBand(center=548, sigma=26, gamma=3, amplitude=1.0, name="Visible max (purple)"),
                    NIRBand(center=285, sigma=15, gamma=2, amplitude=0.52, name="UV band"),
                    # NIR bands
                    NIRBand(center=1045, sigma=18, gamma=2, amplitude=0.17, name="Electronic tail"),
                    NIRBand(center=1430, sigma=22, gamma=2.5, amplitude=0.27, name="Phenolic O-H"),
                    NIRBand(center=1675, sigma=18, gamma=2, amplitude=0.31, name="Ar C-H 1st overtone"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="flavonoids",
                tags=["food", "vis-nir"],
            ),
            # Lycopene: Red carotenoid (tomatoes)
            # Refs: Britton et al. (2004), λmax: 443, 471, 502 nm
            "lycopene": SpectralComponent(
                name="lycopene",
                bands=[
                    # Visible - three characteristic peaks (all-trans lycopene)
                    NIRBand(center=443, sigma=12, gamma=2.5, amplitude=0.70, name="π-π* (0-2)"),
                    NIRBand(center=471, sigma=14, gamma=2.5, amplitude=1.0, name="π-π* (0-1) max"),
                    NIRBand(center=502, sigma=15, gamma=3, amplitude=0.88, name="π-π* (0-0)"),
                    # NIR bands
                    NIRBand(center=1055, sigma=22, gamma=2.5, amplitude=0.20, name="Electronic tail"),
                    NIRBand(center=1685, sigma=18, gamma=2, amplitude=0.30, name="=C-H 1st overtone"),
                    NIRBand(center=2138, sigma=22, gamma=2, amplitude=0.25, name="C=C combination"),
                    NIRBand(center=2282, sigma=20, gamma=2, amplitude=0.22, name="C-H combination"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="carotenoids",
                formula="C40H56",
                cas_number="502-65-8",
                tags=["food", "vis-nir"],
            ),
            # Lutein: Yellow carotenoid (xanthophyll)
            # Refs: Britton et al. (2004), λmax: 421, 445, 474 nm
            "lutein": SpectralComponent(
                name="lutein",
                bands=[
                    # Visible - three characteristic peaks
                    NIRBand(center=421, sigma=11, gamma=2, amplitude=0.68, name="π-π* (0-2)"),
                    NIRBand(center=445, sigma=13, gamma=2.5, amplitude=1.0, name="π-π* (0-1) max"),
                    NIRBand(center=474, sigma=14, gamma=2.5, amplitude=0.80, name="π-π* (0-0)"),
                    # NIR bands
                    NIRBand(center=1048, sigma=20, gamma=2, amplitude=0.18, name="Electronic tail"),
                    NIRBand(center=1415, sigma=20, gamma=2, amplitude=0.25, name="O-H 1st overtone"),
                    NIRBand(center=1678, sigma=18, gamma=2, amplitude=0.28, name="=C-H 1st overtone"),
                    NIRBand(center=2132, sigma=22, gamma=2, amplitude=0.24, name="C=C combination"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="xanthophylls",
                formula="C40H56O2",
                cas_number="127-40-2",
                synonyms=["xanthophyll"],
                tags=["food", "biological", "vis-nir"],
            ),
            # Xanthophyll: General yellow pigments (alias for lutein)
            "xanthophyll": SpectralComponent(
                name="xanthophyll",
                bands=[
                    # Visible - similar to lutein
                    NIRBand(center=420, sigma=11, gamma=2, amplitude=0.65, name="π-π* (0-2)"),
                    NIRBand(center=444, sigma=13, gamma=2.5, amplitude=1.0, name="π-π* (0-1) max"),
                    NIRBand(center=472, sigma=14, gamma=2.5, amplitude=0.78, name="π-π* (0-0)"),
                    # NIR bands
                    NIRBand(center=1052, sigma=20, gamma=2, amplitude=0.17, name="Electronic tail"),
                    NIRBand(center=1418, sigma=20, gamma=2, amplitude=0.26, name="O-H 1st overtone"),
                    NIRBand(center=1682, sigma=18, gamma=2, amplitude=0.29, name="=C-H 1st overtone"),
                    NIRBand(center=2135, sigma=22, gamma=2, amplitude=0.25, name="C=C combination"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="carotenoids",
                synonyms=["lutein"],
                tags=["biological", "vis-nir"],
            ),
            # Melanin: Brown-black pigment (eumelanin)
            # Refs: Zonios et al. (2008), broad absorption decreasing with wavelength
            "melanin": SpectralComponent(
                name="melanin",
                bands=[
                    # Visible - broad absorption (decreases with wavelength)
                    # Melanin absorbs broadly across visible spectrum
                    NIRBand(center=400, sigma=50, gamma=5, amplitude=1.0, name="Blue absorption"),
                    NIRBand(center=500, sigma=60, gamma=6, amplitude=0.75, name="Green absorption"),
                    NIRBand(center=600, sigma=70, gamma=6, amplitude=0.55, name="Red absorption"),
                    NIRBand(center=700, sigma=80, gamma=6, amplitude=0.40, name="Red edge"),
                    # NIR bands
                    NIRBand(center=1100, sigma=30, gamma=3, amplitude=0.25, name="Electronic tail"),
                    NIRBand(center=1510, sigma=22, gamma=2.5, amplitude=0.28, name="N-H 1st overtone"),
                    NIRBand(center=1680, sigma=20, gamma=2, amplitude=0.26, name="Ar C-H 1st overtone"),
                    NIRBand(center=2055, sigma=28, gamma=3, amplitude=0.30, name="N-H combination"),
                ],
                correlation_group=5,
                category="pigments",
                subcategory="biopolymers",
                tags=["biological", "vis-nir", "medical"],
            ),
            # ================================================================
            # HEMOGLOBIN AND BLOOD PIGMENTS (Phase 2 addition)
            # ================================================================
            # Hemoglobin (oxyhemoglobin): Oxygenated blood
            # Refs: Prahl (1999), λmax: 414 (Soret), 542, 577 nm
            "hemoglobin_oxy": SpectralComponent(
                name="hemoglobin_oxy",
                bands=[
                    # Visible - Soret band (intense)
                    NIRBand(center=414, sigma=12, gamma=3, amplitude=1.0, name="Soret band"),
                    # Visible - Q bands (α and β)
                    NIRBand(center=542, sigma=10, gamma=2, amplitude=0.25, name="Q band β"),
                    NIRBand(center=577, sigma=10, gamma=2, amplitude=0.28, name="Q band α"),
                    # Red edge and NIR
                    NIRBand(center=700, sigma=30, gamma=3, amplitude=0.08, name="Red edge tail"),
                    NIRBand(center=920, sigma=40, gamma=4, amplitude=0.04, name="NIR tail"),
                ],
                correlation_group=14,
                category="pigments",
                subcategory="hemoproteins",
                formula="C2952H4664N812O832S8Fe4",
                tags=["medical", "biological", "vis-nir", "blood"],
            ),
            # Hemoglobin (deoxyhemoglobin): Deoxygenated blood
            # Refs: Prahl (1999), λmax: 430 (Soret), 555 nm
            "hemoglobin_deoxy": SpectralComponent(
                name="hemoglobin_deoxy",
                bands=[
                    # Visible - Soret band (shifted from oxy)
                    NIRBand(center=430, sigma=14, gamma=3.5, amplitude=1.0, name="Soret band"),
                    # Visible - single broad Q band
                    NIRBand(center=555, sigma=18, gamma=3, amplitude=0.32, name="Q band"),
                    # Red edge and NIR (stronger in deoxy)
                    NIRBand(center=680, sigma=35, gamma=4, amplitude=0.12, name="Red edge"),
                    NIRBand(center=760, sigma=30, gamma=3.5, amplitude=0.15, name="NIR absorption"),
                    NIRBand(center=920, sigma=45, gamma=4, amplitude=0.08, name="NIR tail"),
                ],
                correlation_group=14,
                category="pigments",
                subcategory="hemoproteins",
                formula="C2952H4664N812O832S8Fe4",
                tags=["medical", "biological", "vis-nir", "blood"],
            ),
            # Myoglobin: Muscle oxygen carrier
            # Refs: Similar to hemoglobin, single heme group
            "myoglobin": SpectralComponent(
                name="myoglobin",
                bands=[
                    # Visible - Soret band
                    NIRBand(center=418, sigma=13, gamma=3, amplitude=1.0, name="Soret band"),
                    # Q bands
                    NIRBand(center=544, sigma=12, gamma=2.5, amplitude=0.22, name="Q band β"),
                    NIRBand(center=580, sigma=11, gamma=2.5, amplitude=0.26, name="Q band α"),
                    # NIR
                    NIRBand(center=760, sigma=35, gamma=4, amplitude=0.06, name="NIR tail"),
                ],
                correlation_group=14,
                category="pigments",
                subcategory="hemoproteins",
                formula="C738H1166N210O208S2Fe",
                cas_number="100684-32-0",
                tags=["biological", "vis-nir", "meat"],
            ),
            # Cytochrome c: Electron transfer protein
            # Refs: Margoliash & Schejter (1966)
            "cytochrome_c": SpectralComponent(
                name="cytochrome_c",
                bands=[
                    # Visible - Soret band
                    NIRBand(center=410, sigma=11, gamma=2.5, amplitude=1.0, name="Soret band"),
                    # Q bands (reduced form)
                    NIRBand(center=520, sigma=12, gamma=2.5, amplitude=0.18, name="Q band β"),
                    NIRBand(center=550, sigma=10, gamma=2, amplitude=0.35, name="Q band α"),
                ],
                correlation_group=14,
                category="pigments",
                subcategory="hemoproteins",
                tags=["biological", "vis-nir"],
            ),
            # Bilirubin: Bile pigment (yellow)
            # Refs: λmax 453 nm
            "bilirubin": SpectralComponent(
                name="bilirubin",
                bands=[
                    # Visible - main absorption
                    NIRBand(center=453, sigma=25, gamma=4, amplitude=1.0, name="Visible max"),
                    NIRBand(center=380, sigma=20, gamma=3, amplitude=0.45, name="UV shoulder"),
                    # NIR
                    NIRBand(center=1025, sigma=25, gamma=3, amplitude=0.12, name="Electronic tail"),
                ],
                correlation_group=14,
                category="pigments",
                subcategory="bile_pigments",
                formula="C33H36N4O6",
                cas_number="635-65-4",
                tags=["medical", "biological", "vis-nir"],
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

        # Enrich components with metadata
        _enrich_components_with_metadata(_PREDEFINED_COMPONENTS)

    return _PREDEFINED_COMPONENTS


# ============================================================================
# Component Metadata (Phase 1 enhancement)
# ============================================================================
# This metadata is applied to components during initialization.
# Format: {component_name: {metadata_field: value, ...}}

_COMPONENT_METADATA = {
    # Water-related (already have inline metadata, but keep mapping for completeness)
    "water": {"category": "water_related", "formula": "H2O", "cas_number": "7732-18-5", "tags": ["universal", "moisture", "food", "agriculture", "pharma"]},
    "moisture": {"category": "water_related", "synonyms": ["bound water", "hydration"], "tags": ["universal", "moisture", "food", "agriculture"]},

    # Proteins
    "protein": {"category": "proteins", "tags": ["food", "agriculture", "pharma", "biological"]},
    "nitrogen_compound": {"category": "proteins", "subcategory": "amines", "tags": ["agriculture", "chemical"]},
    "urea": {"category": "proteins", "subcategory": "amines", "formula": "CH4N2O", "cas_number": "57-13-6", "tags": ["agriculture", "pharma", "fertilizer"]},
    "amino_acid": {"category": "proteins", "subcategory": "amino_acids", "tags": ["food", "pharma", "biological"]},
    "casein": {"category": "proteins", "subcategory": "milk_proteins", "tags": ["food", "dairy"]},
    "gluten": {"category": "proteins", "subcategory": "plant_proteins", "tags": ["food", "agriculture", "grain"]},
    "albumin": {"category": "proteins", "subcategory": "globular_proteins", "tags": ["food", "biological"]},
    "collagen": {"category": "proteins", "subcategory": "structural_proteins", "tags": ["biological", "pharma"]},
    "keratin": {"category": "proteins", "subcategory": "structural_proteins", "tags": ["biological"]},
    "zein": {"category": "proteins", "subcategory": "plant_proteins", "tags": ["food", "agriculture", "corn"]},
    "gelatin": {"category": "proteins", "subcategory": "derived_proteins", "tags": ["food", "pharma"]},
    "whey": {"category": "proteins", "subcategory": "milk_proteins", "tags": ["food", "dairy"]},

    # Lipids
    "lipid": {"category": "lipids", "tags": ["food", "biological"]},
    "oil": {"category": "lipids", "subcategory": "triglycerides", "tags": ["food", "agriculture"]},
    "saturated_fat": {"category": "lipids", "subcategory": "fatty_acids", "tags": ["food"]},
    "unsaturated_fat": {"category": "lipids", "subcategory": "fatty_acids", "tags": ["food"]},
    "aromatic": {"category": "lipids", "subcategory": "hydrocarbons", "tags": ["chemical", "petrochemical"]},
    "alkane": {"category": "lipids", "subcategory": "hydrocarbons", "tags": ["chemical", "petrochemical"]},
    "waxes": {"category": "lipids", "subcategory": "waxes", "tags": ["agriculture", "cosmetics"]},
    "oleic_acid": {"category": "lipids", "subcategory": "fatty_acids", "formula": "C18H34O2", "cas_number": "112-80-1", "tags": ["food"]},
    "linoleic_acid": {"category": "lipids", "subcategory": "fatty_acids", "formula": "C18H32O2", "cas_number": "60-33-3", "tags": ["food"]},
    "linolenic_acid": {"category": "lipids", "subcategory": "fatty_acids", "formula": "C18H30O2", "cas_number": "463-40-1", "tags": ["food"]},
    "palmitic_acid": {"category": "lipids", "subcategory": "fatty_acids", "formula": "C16H32O2", "cas_number": "57-10-3", "tags": ["food"]},
    "stearic_acid": {"category": "lipids", "subcategory": "fatty_acids", "formula": "C18H36O2", "cas_number": "57-11-4", "tags": ["food"]},
    "phospholipid": {"category": "lipids", "subcategory": "membrane_lipids", "tags": ["biological"]},
    "cholesterol": {"category": "lipids", "subcategory": "sterols", "formula": "C27H46O", "cas_number": "57-88-5", "tags": ["biological", "pharma"]},
    "cocoa_butter": {"category": "lipids", "subcategory": "fats", "tags": ["food", "cosmetics"]},

    # Carbohydrates
    "starch": {"category": "carbohydrates", "subcategory": "polysaccharides", "formula": "(C6H10O5)n", "tags": ["food", "agriculture"]},
    "cellulose": {"category": "carbohydrates", "subcategory": "polysaccharides", "formula": "(C6H10O5)n", "cas_number": "9004-34-6", "tags": ["agriculture", "fiber"]},
    "glucose": {"category": "carbohydrates", "subcategory": "monosaccharides", "formula": "C6H12O6", "cas_number": "50-99-7", "tags": ["food", "biological"]},
    "fructose": {"category": "carbohydrates", "subcategory": "monosaccharides", "formula": "C6H12O6", "cas_number": "57-48-7", "tags": ["food"]},
    "sucrose": {"category": "carbohydrates", "subcategory": "disaccharides", "formula": "C12H22O11", "cas_number": "57-50-1", "synonyms": ["sugar", "table sugar"], "tags": ["food"]},
    "hemicellulose": {"category": "carbohydrates", "subcategory": "polysaccharides", "tags": ["agriculture", "fiber"]},
    "lignin": {"category": "carbohydrates", "subcategory": "polyphenols", "tags": ["agriculture", "wood"]},
    "lactose": {"category": "carbohydrates", "subcategory": "disaccharides", "formula": "C12H22O11", "cas_number": "63-42-3", "synonyms": ["milk sugar"], "tags": ["food", "dairy", "pharma"]},
    "cotton": {"category": "carbohydrates", "subcategory": "fibers", "tags": ["textile", "fiber"]},
    "dietary_fiber": {"category": "carbohydrates", "subcategory": "fiber", "tags": ["food", "agriculture"]},
    "maltose": {"category": "carbohydrates", "subcategory": "disaccharides", "formula": "C12H22O11", "cas_number": "69-79-4", "synonyms": ["malt sugar"], "tags": ["food"]},
    "raffinose": {"category": "carbohydrates", "subcategory": "oligosaccharides", "formula": "C18H32O16", "cas_number": "512-69-6", "tags": ["food"]},
    "inulin": {"category": "carbohydrates", "subcategory": "polysaccharides", "tags": ["food", "fiber"]},
    "xylose": {"category": "carbohydrates", "subcategory": "monosaccharides", "formula": "C5H10O5", "cas_number": "58-86-6", "tags": ["food"]},
    "arabinose": {"category": "carbohydrates", "subcategory": "monosaccharides", "formula": "C5H10O5", "cas_number": "147-81-9", "tags": ["food"]},
    "galactose": {"category": "carbohydrates", "subcategory": "monosaccharides", "formula": "C6H12O6", "cas_number": "59-23-4", "tags": ["food", "dairy"]},
    "mannose": {"category": "carbohydrates", "subcategory": "monosaccharides", "formula": "C6H12O6", "cas_number": "3458-28-4", "tags": ["food"]},
    "trehalose": {"category": "carbohydrates", "subcategory": "disaccharides", "formula": "C12H22O11", "cas_number": "99-20-7", "tags": ["food"]},

    # Alcohols
    "ethanol": {"category": "alcohols", "formula": "C2H6O", "cas_number": "64-17-5", "tags": ["food", "solvent", "beverage"]},
    "methanol": {"category": "alcohols", "formula": "CH4O", "cas_number": "67-56-1", "synonyms": ["wood alcohol"], "tags": ["solvent", "chemical"]},
    "glycerol": {"category": "alcohols", "subcategory": "polyols", "formula": "C3H8O3", "cas_number": "56-81-5", "synonyms": ["glycerin"], "tags": ["food", "pharma", "cosmetics"]},
    "propanol": {"category": "alcohols", "formula": "C3H8O", "cas_number": "71-23-8", "tags": ["solvent", "chemical"]},
    "butanol": {"category": "alcohols", "formula": "C4H10O", "cas_number": "71-36-3", "tags": ["solvent", "chemical"]},
    "sorbitol": {"category": "alcohols", "subcategory": "sugar_alcohols", "formula": "C6H14O6", "cas_number": "50-70-4", "tags": ["food", "pharma"]},
    "mannitol": {"category": "alcohols", "subcategory": "sugar_alcohols", "formula": "C6H14O6", "cas_number": "69-65-8", "tags": ["food", "pharma"]},
    "xylitol": {"category": "alcohols", "subcategory": "sugar_alcohols", "formula": "C5H12O5", "cas_number": "87-99-0", "tags": ["food"]},
    "isopropanol": {"category": "alcohols", "formula": "C3H8O", "cas_number": "67-63-0", "synonyms": ["isopropyl alcohol", "IPA"], "tags": ["solvent", "pharma"]},

    # Organic acids
    "acetic_acid": {"category": "organic_acids", "formula": "C2H4O2", "cas_number": "64-19-7", "synonyms": ["vinegar"], "tags": ["food", "chemical"]},
    "citric_acid": {"category": "organic_acids", "formula": "C6H8O7", "cas_number": "77-92-9", "tags": ["food", "pharma"]},
    "lactic_acid": {"category": "organic_acids", "formula": "C3H6O3", "cas_number": "50-21-5", "tags": ["food", "pharma", "biological"]},
    "malic_acid": {"category": "organic_acids", "formula": "C4H6O5", "cas_number": "6915-15-7", "tags": ["food"]},
    "tartaric_acid": {"category": "organic_acids", "formula": "C4H6O6", "cas_number": "87-69-4", "tags": ["food", "wine"]},
    "formic_acid": {"category": "organic_acids", "formula": "CH2O2", "cas_number": "64-18-6", "tags": ["chemical"]},
    "oxalic_acid": {"category": "organic_acids", "formula": "C2H2O4", "cas_number": "144-62-7", "tags": ["chemical"]},
    "succinic_acid": {"category": "organic_acids", "formula": "C4H6O4", "cas_number": "110-15-6", "tags": ["food", "pharma"]},
    "fumaric_acid": {"category": "organic_acids", "formula": "C4H4O4", "cas_number": "110-17-8", "tags": ["food"]},
    "propionic_acid": {"category": "organic_acids", "formula": "C3H6O2", "cas_number": "79-09-4", "tags": ["food", "preservative"]},
    "butyric_acid": {"category": "organic_acids", "formula": "C4H8O2", "cas_number": "107-92-6", "tags": ["food", "dairy"]},
    "ascorbic_acid": {"category": "organic_acids", "formula": "C6H8O6", "cas_number": "50-81-7", "synonyms": ["vitamin C"], "tags": ["food", "pharma"]},

    # Pigments (Phase 2 extended with visible-region components)
    "chlorophyll": {"category": "pigments", "subcategory": "chlorophylls", "formula": "C55H72MgN4O5", "tags": ["agriculture", "biological", "vis-nir", "photosynthesis"]},
    "chlorophyll_a": {"category": "pigments", "subcategory": "chlorophylls", "formula": "C55H72MgN4O5", "cas_number": "479-61-8", "tags": ["agriculture", "biological", "vis-nir", "photosynthesis"]},
    "chlorophyll_b": {"category": "pigments", "subcategory": "chlorophylls", "formula": "C55H70MgN4O6", "cas_number": "519-62-0", "tags": ["agriculture", "biological", "vis-nir", "photosynthesis"]},
    "carotenoid": {"category": "pigments", "subcategory": "carotenoids", "tags": ["food", "agriculture", "biological", "vis-nir"]},
    "beta_carotene": {"category": "pigments", "subcategory": "carotenoids", "formula": "C40H56", "cas_number": "7235-40-7", "tags": ["food", "agriculture", "vis-nir"]},
    "tannins": {"category": "pigments", "subcategory": "polyphenols", "tags": ["food", "wine", "agriculture"]},
    "anthocyanin": {"category": "pigments", "subcategory": "flavonoids", "tags": ["food", "biological", "vis-nir"]},
    "anthocyanin_red": {"category": "pigments", "subcategory": "flavonoids", "tags": ["food", "vis-nir"]},
    "anthocyanin_purple": {"category": "pigments", "subcategory": "flavonoids", "tags": ["food", "vis-nir"]},
    "lycopene": {"category": "pigments", "subcategory": "carotenoids", "formula": "C40H56", "cas_number": "502-65-8", "tags": ["food", "vis-nir"]},
    "lutein": {"category": "pigments", "subcategory": "xanthophylls", "formula": "C40H56O2", "cas_number": "127-40-2", "synonyms": ["xanthophyll"], "tags": ["food", "biological", "vis-nir"]},
    "xanthophyll": {"category": "pigments", "subcategory": "carotenoids", "synonyms": ["lutein"], "tags": ["biological", "vis-nir"]},
    "melanin": {"category": "pigments", "subcategory": "biopolymers", "tags": ["biological", "medical", "vis-nir"]},
    # Hemoproteins and blood pigments (Phase 2)
    "hemoglobin_oxy": {"category": "pigments", "subcategory": "hemoproteins", "formula": "C2952H4664N812O832S8Fe4", "tags": ["medical", "biological", "vis-nir", "blood"]},
    "hemoglobin_deoxy": {"category": "pigments", "subcategory": "hemoproteins", "formula": "C2952H4664N812O832S8Fe4", "tags": ["medical", "biological", "vis-nir", "blood"]},
    "myoglobin": {"category": "pigments", "subcategory": "hemoproteins", "formula": "C738H1166N210O208S2Fe", "cas_number": "100684-32-0", "tags": ["biological", "vis-nir", "meat"]},
    "cytochrome_c": {"category": "pigments", "subcategory": "hemoproteins", "tags": ["biological", "vis-nir"]},
    "bilirubin": {"category": "pigments", "subcategory": "bile_pigments", "formula": "C33H36N4O6", "cas_number": "635-65-4", "tags": ["medical", "biological", "vis-nir"]},

    # Pharmaceuticals
    "caffeine": {"category": "pharmaceuticals", "formula": "C8H10N4O2", "cas_number": "58-08-2", "tags": ["pharma", "food", "beverage"]},
    "aspirin": {"category": "pharmaceuticals", "formula": "C9H8O4", "cas_number": "50-78-2", "synonyms": ["acetylsalicylic acid"], "tags": ["pharma"]},
    "paracetamol": {"category": "pharmaceuticals", "formula": "C8H9NO2", "cas_number": "103-90-2", "synonyms": ["acetaminophen"], "tags": ["pharma"]},
    "ibuprofen": {"category": "pharmaceuticals", "formula": "C13H18O2", "cas_number": "15687-27-1", "tags": ["pharma"]},
    "naproxen": {"category": "pharmaceuticals", "formula": "C14H14O3", "cas_number": "22204-53-1", "tags": ["pharma"]},
    "diclofenac": {"category": "pharmaceuticals", "formula": "C14H11Cl2NO2", "cas_number": "15307-86-5", "tags": ["pharma"]},
    "metformin": {"category": "pharmaceuticals", "formula": "C4H11N5", "cas_number": "657-24-9", "tags": ["pharma"]},
    "omeprazole": {"category": "pharmaceuticals", "formula": "C17H19N3O3S", "cas_number": "73590-58-6", "tags": ["pharma"]},
    "amoxicillin": {"category": "pharmaceuticals", "formula": "C16H19N3O5S", "cas_number": "26787-78-0", "tags": ["pharma", "antibiotic"]},
    "microcrystalline_cellulose": {"category": "pharmaceuticals", "subcategory": "excipients", "cas_number": "9004-34-6", "synonyms": ["MCC", "Avicel"], "tags": ["pharma"]},

    # Polymers
    "polyethylene": {"category": "polymers", "subcategory": "polyolefins", "synonyms": ["PE", "HDPE", "LDPE"], "tags": ["plastic", "packaging"]},
    "polystyrene": {"category": "polymers", "subcategory": "aromatics", "synonyms": ["PS"], "tags": ["plastic", "packaging"]},
    "natural_rubber": {"category": "polymers", "subcategory": "elastomers", "synonyms": ["polyisoprene"], "tags": ["rubber", "material"]},
    "nylon": {"category": "polymers", "subcategory": "polyamides", "synonyms": ["PA", "polyamide"], "tags": ["textile", "plastic"]},
    "polyester": {"category": "polymers", "subcategory": "polyesters", "synonyms": ["PET"], "tags": ["textile", "plastic"]},
    "pmma": {"category": "polymers", "subcategory": "acrylics", "synonyms": ["acrylic", "Plexiglas"], "tags": ["plastic"]},
    "pvc": {"category": "polymers", "formula": "(C2H3Cl)n", "synonyms": ["polyvinyl chloride"], "tags": ["plastic"]},
    "polypropylene": {"category": "polymers", "subcategory": "polyolefins", "synonyms": ["PP"], "tags": ["plastic", "packaging"]},
    "pet": {"category": "polymers", "subcategory": "polyesters", "synonyms": ["polyester", "polyethylene terephthalate"], "tags": ["plastic", "packaging", "textile"]},
    "ptfe": {"category": "polymers", "subcategory": "fluoropolymers", "synonyms": ["Teflon"], "tags": ["plastic"]},
    "abs": {"category": "polymers", "subcategory": "copolymers", "synonyms": ["acrylonitrile butadiene styrene"], "tags": ["plastic"]},

    # Solvents
    "acetone": {"category": "solvents", "formula": "C3H6O", "cas_number": "67-64-1", "synonyms": ["propanone"], "tags": ["solvent", "chemical"]},
    "dmso": {"category": "solvents", "formula": "C2H6OS", "cas_number": "67-68-5", "synonyms": ["dimethyl sulfoxide"], "tags": ["solvent", "pharma"]},
    "ethyl_acetate": {"category": "solvents", "formula": "C4H8O2", "cas_number": "141-78-6", "tags": ["solvent"]},
    "toluene": {"category": "solvents", "formula": "C7H8", "cas_number": "108-88-3", "synonyms": ["methylbenzene"], "tags": ["solvent", "petrochemical"]},
    "chloroform": {"category": "solvents", "formula": "CHCl3", "cas_number": "67-66-3", "synonyms": ["trichloromethane"], "tags": ["solvent"]},
    "hexane": {"category": "solvents", "formula": "C6H14", "cas_number": "110-54-3", "tags": ["solvent", "petrochemical"]},

    # Minerals
    "carbonates": {"category": "minerals", "subcategory": "carbonates", "tags": ["soil", "geology"]},
    "gypsum": {"category": "minerals", "subcategory": "sulfates", "formula": "CaSO4·2H2O", "cas_number": "10101-41-4", "tags": ["soil", "construction"]},
    "kaolinite": {"category": "minerals", "subcategory": "clays", "formula": "Al2Si2O5(OH)4", "tags": ["soil", "geology"]},
    "montmorillonite": {"category": "minerals", "subcategory": "clays", "tags": ["soil", "geology"]},
    "illite": {"category": "minerals", "subcategory": "clays", "tags": ["soil", "geology"]},
    "goethite": {"category": "minerals", "subcategory": "iron_oxides", "formula": "FeO(OH)", "cas_number": "1310-14-1", "tags": ["soil", "geology"]},
    "talc": {"category": "minerals", "subcategory": "silicates", "formula": "Mg3Si4O10(OH)2", "cas_number": "14807-96-6", "tags": ["cosmetics", "geology"]},
    "silica": {"category": "minerals", "subcategory": "silicates", "formula": "SiO2", "cas_number": "7631-86-9", "synonyms": ["silicon dioxide"], "tags": ["geology", "pharma"]},
}


def _enrich_components_with_metadata(components: "Dict[str, SpectralComponent]") -> None:
    """
    Enrich components with metadata from the metadata mapping.

    This function modifies components in-place, adding category, subcategory,
    formula, cas_number, synonyms, and tags from _COMPONENT_METADATA.
    It also normalizes band amplitudes to ensure max amplitude = 1.0.
    """
    for name, comp in components.items():
        # Apply metadata if available
        if name in _COMPONENT_METADATA:
            meta = _COMPONENT_METADATA[name]
            if "category" in meta and not comp.category:
                comp.category = meta["category"]
            if "subcategory" in meta and not comp.subcategory:
                comp.subcategory = meta["subcategory"]
            if "formula" in meta and not comp.formula:
                comp.formula = meta["formula"]
            if "cas_number" in meta and not comp.cas_number:
                comp.cas_number = meta["cas_number"]
            if "synonyms" in meta and not comp.synonyms:
                comp.synonyms = meta["synonyms"]
            if "tags" in meta and not comp.tags:
                comp.tags = meta["tags"]
            if "references" in meta and not comp.references:
                comp.references = meta["references"]

        # Normalize band amplitudes (max = 1.0)
        if comp.bands:
            max_amp = max(band.amplitude for band in comp.bands)
            if max_amp > 0 and abs(max_amp - 1.0) > 0.01:
                for band in comp.bands:
                    band.amplitude = band.amplitude / max_amp


# Default wavelength parameters
# Phase 2 Extension: Extended to include Vis-NIR region (350-2500nm)
# This enables generation of spectra for:
# - Si detector instruments (400-1100nm)
# - Vis-NIR spectrometers common in agriculture/food
# - Electronic absorption bands for biological samples
DEFAULT_WAVELENGTH_START: float = 350.0
DEFAULT_WAVELENGTH_END: float = 2500.0
DEFAULT_WAVELENGTH_STEP: float = 2.0

# Default spectral zones for random band placement
# Includes both visible (electronic transitions) and NIR (vibrational) regions
DEFAULT_NIR_ZONES = [
    # Visible region - electronic transitions
    (400, 500),    # Blue region - pigment absorptions (chlorophyll Soret, carotenoids)
    (500, 600),    # Green region - anthocyanins, flavonoids
    (600, 700),    # Red region - chlorophyll Q band
    (700, 800),    # Red edge - chlorophyll tail, electronic transitions
    # Short-wave NIR - 3rd overtones
    (800, 1000),   # 3rd overtones C-H, O-H, N-H
    (1000, 1100),  # 3rd overtones (Si detector limit)
    # NIR - 2nd and 1st overtones
    (1100, 1300),  # 2nd overtones
    (1400, 1550),  # 1st overtones O-H, N-H
    (1650, 1800),  # 1st overtones C-H
    # NIR - Combination bands
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
