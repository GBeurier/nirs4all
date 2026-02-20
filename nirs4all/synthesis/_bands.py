"""
Exhaustive NIR spectral band assignments dictionary.

This module provides a comprehensive reference of NIR absorption bands organized by:
    - Functional group (O-H, N-H, C-H, C=O, etc.)
    - Overtone level (fundamental, 1st, 2nd, 3rd overtone, combinations)
    - Chemical context (free vs. bonded, aromatic vs. aliphatic)

Band assignments are based on established NIR spectroscopy literature and enable:
    - Band-level spectral analysis and interpretation
    - Synthetic spectra generation with physically accurate peaks
    - Spectral fitting and deconvolution
    - Feature selection based on chemical knowledge

Theory Background:
    NIR spectroscopy (750-2500 nm, 4000-13300 cm⁻¹) observes overtones and
    combination bands of molecular vibrations. Key relationships:

    - **Fundamental frequency (ν₀)**: IR region (~3000 cm⁻¹ for C-H, ~3400 cm⁻¹ for O-H)
    - **1st overtone (2ν)**: ~2× fundamental frequency → NIR region
    - **2nd overtone (3ν)**: ~3× fundamental frequency → shorter wavelength
    - **3rd overtone (4ν)**: ~4× fundamental frequency → visible-NIR edge
    - **Combination bands**: ν₁ + ν₂ (stretch + bend, etc.)

    Anharmonicity causes overtone frequencies to be slightly less than
    integer multiples of the fundamental.

Primary References:
    [1] Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas
        for Interpretive Near-Infrared Spectroscopy (2nd ed.). CRC Press.
        - Comprehensive NIR band assignments and spectral atlas (pp. 23-73)
        - Table 2.3: Water bands, Table 2.6-2.7: C-H bands, Table 2.8-2.9: N-H bands

    [2] Burns, D. A., & Ciurczak, E. W. (Eds.). (2007). Handbook of Near-Infrared
        Analysis (3rd ed.). CRC Press.
        - Chapter 2: Theory of NIR Spectroscopy (pp. 7-35)
        - Overtone and combination band theory

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

    [10] Clark, R. N., et al. (1990). High Spectral Resolution Reflectance
         Spectroscopy of Minerals. Journal of Geophysical Research, 95(B8),
         12653-12680.
         - Mineral (clay, carbonate, sulfate) band assignments

    [11] Curcio, J. A., & Petty, C. C. (1951). The Near Infrared Absorption
         Spectrum of Liquid Water. Journal of the Optical Society of America,
         41(5), 302-304.
         - Water overtone bands

    [12] Britton, G., et al. (2004). Carotenoids Handbook. Birkhäuser.
         - Carotenoid visible absorption bands

    [13] Wellburn, A. R. (1994). The Spectral Determination of Chlorophylls a
         and b. Journal of Plant Physiology, 144(3), 307-313.
         - Chlorophyll absorption maxima

    [14] Prahl, S. (1999). Optical Absorption of Hemoglobin. Oregon Medical
         Laser Center. https://omlc.org/spectra/hemoglobin/
         - Hemoglobin absorption spectra
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class BandAssignment:
    """
    Represents a single NIR band assignment.

    This class stores the physical and chemical properties of an absorption band,
    including its position, width, and molecular origin.

    Attributes:
        center: Central wavelength in nm.
        wavenumber: Central wavenumber in cm⁻¹ (calculated from center).
        functional_group: Chemical functional group (e.g., "O-H", "C-H", "N-H").
        overtone_level: Overtone designation ("fundamental", "1st", "2nd", "3rd", "combination").
        assignment: Specific vibrational mode assignment (e.g., "2ν₁", "ν₁+ν₃").
        description: Human-readable description of the band.
        sigma_range: Typical Gaussian width range (min, max) in nm.
        gamma_range: Typical Lorentzian width range (min, max) in nm.
        intensity: Relative intensity category ("very_weak", "weak", "medium", "strong", "very_strong").
        chemical_context: Additional context (e.g., "free", "H-bonded", "aromatic", "aliphatic").
        affected_by: Factors that shift/modify the band (e.g., ["H-bonding", "temperature"]).
        common_compounds: Example compounds showing this band.
        references: Literature citations for the band assignment.
        tags: Classification tags for filtering (e.g., ["water", "carbohydrate"]).

    Example:
        >>> band = BandAssignment(
        ...     center=1450,
        ...     functional_group="O-H",
        ...     overtone_level="1st",
        ...     assignment="2ν₁ (O-H stretch)",
        ...     description="O-H 1st overtone, free hydroxyl",
        ...     sigma_range=(20, 30),
        ...     intensity="strong",
        ... )
    """

    center: float
    functional_group: str
    overtone_level: str
    assignment: str = ""
    description: str = ""
    sigma_range: tuple[float, float] = (15, 30)
    gamma_range: tuple[float, float] = (0, 5)
    intensity: str = "medium"
    chemical_context: str = ""
    affected_by: list[str] = field(default_factory=list)
    common_compounds: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def wavenumber(self) -> float:
        """Convert wavelength (nm) to wavenumber (cm⁻¹)."""
        return 1e7 / self.center

    def to_nir_band(self, amplitude: float = 1.0, sigma: float | None = None, gamma: float | None = None):
        """
        Convert to NIRBand object for spectral generation.

        Args:
            amplitude: Peak amplitude (default 1.0).
            sigma: Gaussian width in nm. If None, uses midpoint of sigma_range.
            gamma: Lorentzian width in nm. If None, uses midpoint of gamma_range.

        Returns:
            NIRBand object configured with this band's parameters.
        """
        from .components import NIRBand

        if sigma is None:
            sigma = (self.sigma_range[0] + self.sigma_range[1]) / 2
        if gamma is None:
            gamma = (self.gamma_range[0] + self.gamma_range[1]) / 2

        return NIRBand(
            center=self.center,
            sigma=sigma,
            gamma=gamma,
            amplitude=amplitude,
            name=self.description or f"{self.functional_group} {self.overtone_level}",
        )

    def info(self) -> str:
        """Return formatted information about the band."""
        lines = [
            f"Band: {self.functional_group} {self.overtone_level}",
            f"  Center: {self.center:.0f} nm ({self.wavenumber:.0f} cm⁻¹)",
            f"  Assignment: {self.assignment}" if self.assignment else "",
            f"  Description: {self.description}" if self.description else "",
            f"  Intensity: {self.intensity}",
            f"  Sigma range: {self.sigma_range[0]:.0f}-{self.sigma_range[1]:.0f} nm",
            f"  Context: {self.chemical_context}" if self.chemical_context else "",
        ]
        if self.affected_by:
            lines.append(f"  Affected by: {', '.join(self.affected_by)}")
        if self.common_compounds:
            lines.append(f"  Found in: {', '.join(self.common_compounds[:5])}")
        if self.references:
            lines.append(f"  References: {', '.join(self.references)}")
        return "\n".join(line for line in lines if line)

# =============================================================================
# NIR BAND ASSIGNMENTS DICTIONARY
# =============================================================================
# Organized by functional group and overtone level
# All wavelengths in nm, wavenumbers in cm⁻¹

NIR_BANDS: dict[str, dict[str, BandAssignment]] = {
    # =========================================================================
    # O-H BANDS (Hydroxyl groups)
    # Fundamental: ~3400 cm⁻¹ (2941 nm) - IR region
    # =========================================================================
    "O-H": {
        # 3rd overtone: 4ν (~13600 cm⁻¹)
        "3rd_overtone_free": BandAssignment(
            center=730,
            functional_group="O-H",
            overtone_level="3rd",
            assignment="4ν (O-H stretch)",
            description="O-H 3rd overtone, free hydroxyl",
            sigma_range=(15, 22),
            gamma_range=(1, 3),
            intensity="very_weak",
            chemical_context="free",
            affected_by=["H-bonding", "temperature"],
            common_compounds=["water", "alcohols"],
            references=["Curcio1951", "Workman2012 p.35"],
            tags=["water", "alcohol", "visible-NIR"],
        ),
        "3rd_overtone_bound": BandAssignment(
            center=740,
            functional_group="O-H",
            overtone_level="3rd",
            assignment="4ν (O-H stretch, H-bonded)",
            description="O-H 3rd overtone, hydrogen bonded",
            sigma_range=(18, 25),
            gamma_range=(2, 4),
            intensity="very_weak",
            chemical_context="H-bonded",
            affected_by=["H-bonding", "matrix effects"],
            common_compounds=["bound water", "carbohydrates"],
            references=["Burns2007 p.361"],
            tags=["water", "carbohydrate", "visible-NIR"],
        ),
        # 2nd overtone: 3ν (~10300 cm⁻¹)
        "2nd_overtone_free": BandAssignment(
            center=970,
            functional_group="O-H",
            overtone_level="2nd",
            assignment="3ν (O-H stretch)",
            description="O-H 2nd overtone, free hydroxyl",
            sigma_range=(18, 25),
            gamma_range=(2, 4),
            intensity="weak",
            chemical_context="free",
            affected_by=["H-bonding", "temperature"],
            common_compounds=["water", "alcohols"],
            references=["Workman2012 p.35"],
            tags=["water", "alcohol"],
        ),
        "2nd_overtone_bound": BandAssignment(
            center=985,
            functional_group="O-H",
            overtone_level="2nd",
            assignment="3ν (O-H stretch, H-bonded)",
            description="O-H 2nd overtone, hydrogen bonded",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="weak",
            chemical_context="H-bonded",
            affected_by=["H-bonding", "matrix effects"],
            common_compounds=["bound water", "carbohydrates"],
            references=["Burns2007 p.361"],
            tags=["water", "carbohydrate"],
        ),
        # 1st overtone: 2ν (~6900 cm⁻¹)
        "1st_overtone_free": BandAssignment(
            center=1410,
            functional_group="O-H",
            overtone_level="1st",
            assignment="2ν₁ or 2ν₃ (O-H stretch)",
            description="O-H 1st overtone, free hydroxyl (alcohols)",
            sigma_range=(15, 22),
            gamma_range=(1, 3),
            intensity="strong",
            chemical_context="free",
            affected_by=["H-bonding", "concentration"],
            common_compounds=["alcohols", "phenols"],
            references=["Workman2012 pp.38-40", "Murray1986"],
            tags=["alcohol", "phenol"],
        ),
        "1st_overtone_water": BandAssignment(
            center=1450,
            functional_group="O-H",
            overtone_level="1st",
            assignment="2ν₁ or 2ν₃ (O-H stretch)",
            description="O-H 1st overtone, water",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="strong",
            chemical_context="water",
            affected_by=["temperature", "solutes"],
            common_compounds=["water", "aqueous solutions"],
            references=["Workman2012 p.35", "Curcio1951"],
            tags=["water", "universal"],
        ),
        "1st_overtone_bound": BandAssignment(
            center=1460,
            functional_group="O-H",
            overtone_level="1st",
            assignment="2ν (O-H stretch, H-bonded)",
            description="O-H 1st overtone, bound/matrix water",
            sigma_range=(25, 35),
            gamma_range=(3, 5),
            intensity="strong",
            chemical_context="H-bonded/bound",
            affected_by=["matrix effects", "hydration state"],
            common_compounds=["bound water", "hydrates"],
            references=["Williams2001 p.158", "Burns2007 p.361"],
            tags=["water", "moisture"],
        ),
        "1st_overtone_carbohydrate": BandAssignment(
            center=1440,
            functional_group="O-H",
            overtone_level="1st",
            assignment="2ν (O-H stretch)",
            description="O-H 1st overtone, carbohydrate hydroxyl",
            sigma_range=(22, 28),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="carbohydrate",
            affected_by=["H-bonding", "crystallinity"],
            common_compounds=["glucose", "sucrose", "starch", "cellulose"],
            references=["Burns2007 pp.368-372", "Williams2001"],
            tags=["carbohydrate", "sugar"],
        ),
        "1st_overtone_phenolic": BandAssignment(
            center=1420,
            functional_group="O-H",
            overtone_level="1st",
            assignment="2ν (phenolic O-H stretch)",
            description="O-H 1st overtone, phenolic hydroxyl",
            sigma_range=(20, 28),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="phenolic",
            affected_by=["H-bonding", "aromatic ring"],
            common_compounds=["lignin", "tannins", "paracetamol"],
            references=["Schwanninger2011 p.304"],
            tags=["phenol", "lignin", "polyphenol"],
        ),
        "1st_overtone_carboxylic": BandAssignment(
            center=1425,
            functional_group="O-H",
            overtone_level="1st",
            assignment="2ν (carboxylic O-H stretch)",
            description="O-H 1st overtone, carboxylic acid (strongly H-bonded)",
            sigma_range=(25, 35),
            gamma_range=(3, 5),
            intensity="medium",
            chemical_context="carboxylic",
            affected_by=["dimerization", "H-bonding"],
            common_compounds=["acetic acid", "citric acid", "fatty acids"],
            references=["Bokobza1998 p.9", "Osborne1993 pp.78-80"],
            tags=["organic_acid", "carboxylic"],
        ),
        # Combination bands
        "combination_water": BandAssignment(
            center=1940,
            functional_group="O-H",
            overtone_level="combination",
            assignment="ν₁ + ν₂ or ν₂ + ν₃ (stretch + bend)",
            description="O-H combination, water (strongest water band)",
            sigma_range=(28, 40),
            gamma_range=(3, 6),
            intensity="very_strong",
            chemical_context="water",
            affected_by=["temperature", "solutes"],
            common_compounds=["water", "aqueous solutions"],
            references=["Workman2012 p.35", "Curcio1951"],
            tags=["water", "universal", "diagnostic"],
        ),
        "combination_bound": BandAssignment(
            center=1930,
            functional_group="O-H",
            overtone_level="combination",
            assignment="ν₁ + ν₂ (O-H stretch + bend)",
            description="O-H combination, bound water (broader)",
            sigma_range=(32, 45),
            gamma_range=(4, 7),
            intensity="very_strong",
            chemical_context="H-bonded/bound",
            affected_by=["matrix effects"],
            common_compounds=["bound water", "hydrates"],
            references=["Burns2007 p.361"],
            tags=["water", "moisture"],
        ),
        "combination_alcohol": BandAssignment(
            center=2050,
            functional_group="O-H",
            overtone_level="combination",
            assignment="ν(O-H) + δ(O-H) or ν(O-H) + ν(C-O)",
            description="O-H combination, alcohols and polyols",
            sigma_range=(25, 35),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="alcohol",
            affected_by=["H-bonding", "carbon chain"],
            common_compounds=["ethanol", "methanol", "glycerol"],
            references=["Workman2012 pp.38-40"],
            tags=["alcohol", "polyol"],
        ),
        "combination_carbohydrate": BandAssignment(
            center=2100,
            functional_group="O-H",
            overtone_level="combination",
            assignment="ν(O-H) + ν(C-O) combination",
            description="O-H + C-O combination, carbohydrates",
            sigma_range=(28, 35),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="carbohydrate",
            affected_by=["glycosidic linkage", "crystallinity"],
            common_compounds=["starch", "cellulose", "sugars"],
            references=["Williams2001 p.158", "Schwanninger2011 p.298"],
            tags=["carbohydrate", "diagnostic"],
        ),
        "stretch_bend": BandAssignment(
            center=2500,
            functional_group="O-H",
            overtone_level="combination",
            assignment="ν(O-H) + 2δ(O-H)",
            description="O-H stretch + 2× bend combination",
            sigma_range=(40, 60),
            gamma_range=(4, 7),
            intensity="weak",
            chemical_context="general",
            affected_by=["H-bonding"],
            common_compounds=["water", "carbohydrates"],
            references=["Burns2007 p.360"],
            tags=["water"],
        ),
    },

    # =========================================================================
    # N-H BANDS (Amine, amide groups)
    # Fundamental: ~3300 cm⁻¹ (3030 nm) - IR region
    # =========================================================================
    "N-H": {
        # 3rd overtone: 4ν (~13200 cm⁻¹)
        "3rd_overtone": BandAssignment(
            center=750,
            functional_group="N-H",
            overtone_level="3rd",
            assignment="4ν (N-H stretch)",
            description="N-H 3rd overtone",
            sigma_range=(15, 22),
            gamma_range=(1, 3),
            intensity="very_weak",
            chemical_context="general",
            common_compounds=["proteins", "amines"],
            references=["Workman2012 p.50"],
            tags=["protein", "amine", "visible-NIR"],
        ),
        # 2nd overtone: 3ν (~9900 cm⁻¹)
        "2nd_overtone": BandAssignment(
            center=1030,
            functional_group="N-H",
            overtone_level="2nd",
            assignment="3ν (N-H stretch)",
            description="N-H 2nd overtone",
            sigma_range=(15, 22),
            gamma_range=(1, 3),
            intensity="weak",
            chemical_context="general",
            affected_by=["H-bonding"],
            common_compounds=["proteins", "amines", "amides"],
            references=["Workman2012 p.50"],
            tags=["protein", "amine"],
        ),
        # 1st overtone: 2ν (~6600 cm⁻¹)
        "1st_overtone_free_amine": BandAssignment(
            center=1500,
            functional_group="N-H",
            overtone_level="1st",
            assignment="2ν (N-H stretch, free amine)",
            description="N-H 1st overtone, free primary/secondary amine",
            sigma_range=(15, 22),
            gamma_range=(1, 3),
            intensity="strong",
            chemical_context="free amine",
            affected_by=["H-bonding", "amine type"],
            common_compounds=["primary amines", "secondary amines"],
            references=["Workman2012 pp.52-54"],
            tags=["amine"],
        ),
        "1st_overtone_amide": BandAssignment(
            center=1510,
            functional_group="N-H",
            overtone_level="1st",
            assignment="2ν (Amide A, N-H stretch)",
            description="N-H 1st overtone, amide (protein backbone)",
            sigma_range=(18, 25),
            gamma_range=(2, 3),
            intensity="strong",
            chemical_context="amide",
            affected_by=["protein conformation", "H-bonding"],
            common_compounds=["proteins", "peptides", "nylon"],
            references=["Workman2012 p.50", "Burns2007 pp.362-366"],
            tags=["protein", "amide", "diagnostic"],
        ),
        "1st_overtone_amide_sym": BandAssignment(
            center=1480,
            functional_group="N-H",
            overtone_level="1st",
            assignment="2ν (N-H symmetric stretch)",
            description="N-H 1st overtone, symmetric (urea, guanidine)",
            sigma_range=(15, 22),
            gamma_range=(1, 3),
            intensity="strong",
            chemical_context="symmetric NH₂",
            common_compounds=["urea", "guanidine", "metformin"],
            references=["Reich2005 p.1125"],
            tags=["urea", "pharma"],
        ),
        "1st_overtone_amide_asym": BandAssignment(
            center=1530,
            functional_group="N-H",
            overtone_level="1st",
            assignment="2ν (N-H asymmetric stretch)",
            description="N-H 1st overtone, asymmetric",
            sigma_range=(18, 25),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="asymmetric NH₂",
            common_compounds=["urea", "primary amides"],
            references=["Reich2005 p.1125"],
            tags=["urea", "amide"],
        ),
        "1st_overtone_H_bonded": BandAssignment(
            center=1560,
            functional_group="N-H",
            overtone_level="1st",
            assignment="2ν (N-H stretch, H-bonded)",
            description="N-H 1st overtone, hydrogen bonded",
            sigma_range=(20, 28),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="H-bonded",
            affected_by=["protein secondary structure"],
            common_compounds=["collagen", "gelatin"],
            references=["Burns2007 p.365"],
            tags=["protein", "collagen"],
        ),
        # Combination bands
        "combination_amide": BandAssignment(
            center=2050,
            functional_group="N-H",
            overtone_level="combination",
            assignment="Amide A + Amide II (N-H stretch + N-H bend)",
            description="N-H combination, strong protein marker",
            sigma_range=(25, 35),
            gamma_range=(2, 4),
            intensity="strong",
            chemical_context="amide",
            affected_by=["protein conformation"],
            common_compounds=["proteins", "peptides"],
            references=["Workman2012 p.50", "Burns2007 p.364"],
            tags=["protein", "diagnostic"],
        ),
        "combination_amine": BandAssignment(
            center=2060,
            functional_group="N-H",
            overtone_level="combination",
            assignment="ν(N-H) + δ(N-H)",
            description="N-H combination, amines",
            sigma_range=(22, 30),
            gamma_range=(2, 3),
            intensity="strong",
            chemical_context="amine",
            common_compounds=["amines", "amino acids"],
            references=["Workman2012 pp.52-54"],
            tags=["amine"],
        ),
        "combination_CN": BandAssignment(
            center=2150,
            functional_group="N-H",
            overtone_level="combination",
            assignment="ν(N-H) + ν(C-N)",
            description="N-H + C-N combination",
            sigma_range=(20, 28),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="general",
            common_compounds=["amines", "amides"],
            references=["Workman2012 p.54"],
            tags=["amine", "amide"],
        ),
        "combination_amide_III": BandAssignment(
            center=2300,
            functional_group="N-H",
            overtone_level="combination",
            assignment="ν(N-H) + Amide III",
            description="N-H + Amide III combination",
            sigma_range=(18, 25),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="amide",
            affected_by=["protein secondary structure"],
            common_compounds=["proteins"],
            references=["Workman2012 p.51"],
            tags=["protein"],
        ),
    },

    # =========================================================================
    # C-H BANDS (Aliphatic)
    # Fundamental: ~2900 cm⁻¹ (3448 nm) - IR region
    # =========================================================================
    "C-H_aliphatic": {
        # 3rd overtone: 4ν (~11600 cm⁻¹)
        "3rd_overtone": BandAssignment(
            center=890,
            functional_group="C-H",
            overtone_level="3rd",
            assignment="4ν (C-H stretch)",
            description="C-H 3rd overtone, aliphatic (CH₃, CH₂)",
            sigma_range=(18, 25),
            gamma_range=(1, 3),
            intensity="very_weak",
            chemical_context="aliphatic",
            common_compounds=["lipids", "alkanes", "polymers"],
            references=["Workman2012 p.45"],
            tags=["lipid", "hydrocarbon"],
        ),
        # 2nd overtone: 3ν (~8700 cm⁻¹)
        "2nd_overtone_CH3": BandAssignment(
            center=1165,
            functional_group="C-H",
            overtone_level="2nd",
            assignment="3ν (CH₃ stretch)",
            description="C-H 2nd overtone, methyl (CH₃)",
            sigma_range=(15, 20),
            gamma_range=(1, 2),
            intensity="weak",
            chemical_context="methyl",
            common_compounds=["oils", "lipids", "polymers"],
            references=["Murray1986 p.21", "Workman2012 p.45"],
            tags=["lipid", "methyl"],
        ),
        "2nd_overtone_CH2": BandAssignment(
            center=1195,
            functional_group="C-H",
            overtone_level="2nd",
            assignment="3ν (CH₂ stretch)",
            description="C-H 2nd overtone, methylene (CH₂)",
            sigma_range=(16, 22),
            gamma_range=(1, 2),
            intensity="weak",
            chemical_context="methylene",
            common_compounds=["lipids", "fatty acids", "polyethylene"],
            references=["Murray1986 p.16", "Workman2012 p.45"],
            tags=["lipid", "methylene", "diagnostic"],
        ),
        "2nd_overtone_general": BandAssignment(
            center=1210,
            functional_group="C-H",
            overtone_level="2nd",
            assignment="3ν (C-H stretch, mixed)",
            description="C-H 2nd overtone, general aliphatic",
            sigma_range=(18, 25),
            gamma_range=(1, 3),
            intensity="weak",
            chemical_context="general aliphatic",
            common_compounds=["lipids", "carbohydrates"],
            references=["Workman2012 p.46"],
            tags=["lipid", "carbohydrate"],
        ),
        # 1st overtone: 2ν (~5800 cm⁻¹)
        "1st_overtone_CH3": BandAssignment(
            center=1695,
            functional_group="C-H",
            overtone_level="1st",
            assignment="2ν (CH₃ stretch)",
            description="C-H 1st overtone, methyl (CH₃)",
            sigma_range=(15, 22),
            gamma_range=(1, 3),
            intensity="medium",
            chemical_context="methyl",
            common_compounds=["lipids", "alcohols"],
            references=["Murray1986", "Workman2012 p.45"],
            tags=["lipid", "methyl"],
        ),
        "1st_overtone_CH2": BandAssignment(
            center=1720,
            functional_group="C-H",
            overtone_level="1st",
            assignment="2ν (CH₂ stretch)",
            description="C-H 1st overtone, methylene (CH₂) - strong lipid marker",
            sigma_range=(20, 28),
            gamma_range=(2, 3),
            intensity="strong",
            chemical_context="methylene",
            affected_by=["chain length", "unsaturation"],
            common_compounds=["lipids", "fatty acids", "oils", "polyethylene"],
            references=["Workman2012 p.45", "Murray1986 p.17"],
            tags=["lipid", "diagnostic"],
        ),
        "1st_overtone_general": BandAssignment(
            center=1730,
            functional_group="C-H",
            overtone_level="1st",
            assignment="2ν (C-H stretch, saturated)",
            description="C-H 1st overtone, saturated hydrocarbons",
            sigma_range=(20, 28),
            gamma_range=(2, 3),
            intensity="strong",
            chemical_context="saturated",
            common_compounds=["saturated fats", "alkanes", "waxes"],
            references=["Murray1986 pp.15-20"],
            tags=["lipid", "saturated"],
        ),
        # Combination bands
        "combination_CH2_antisym_scissor": BandAssignment(
            center=2310,
            functional_group="C-H",
            overtone_level="combination",
            assignment="ν(CH₂ antisym) + δ(CH₂ scissor)",
            description="CH₂ combination (antisymmetric stretch + scissor)",
            sigma_range=(18, 25),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="methylene",
            common_compounds=["lipids", "polymers", "petroleum"],
            references=["Osborne1993 p.69", "Workman2012 p.46"],
            tags=["lipid", "diagnostic"],
        ),
        "combination_CH3": BandAssignment(
            center=2350,
            functional_group="C-H",
            overtone_level="combination",
            assignment="ν(CH₃) + δ(CH₃)",
            description="CH₃ combination (stretch + deformation)",
            sigma_range=(16, 22),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="methyl",
            common_compounds=["lipids", "alkanes"],
            references=["Osborne1993 p.70", "Murray1986 p.18"],
            tags=["lipid", "methyl"],
        ),
        "combination_general": BandAssignment(
            center=1390,
            functional_group="C-H",
            overtone_level="combination",
            assignment="ν(C-H) + δ(C-H)",
            description="C-H combination (stretch + deformation)",
            sigma_range=(14, 20),
            gamma_range=(1, 2),
            intensity="weak",
            chemical_context="general",
            common_compounds=["lipids", "polypropylene"],
            references=["Workman2012 p.46"],
            tags=["lipid"],
        ),
    },

    # =========================================================================
    # C-H BANDS (Aromatic)
    # Fundamental: ~3050 cm⁻¹ (3279 nm) - IR region (slightly higher than aliphatic)
    # =========================================================================
    "C-H_aromatic": {
        # 3rd overtone: 4ν (~12200 cm⁻¹)
        "3rd_overtone": BandAssignment(
            center=875,
            functional_group="Ar C-H",
            overtone_level="3rd",
            assignment="4ν (aromatic C-H stretch)",
            description="Aromatic C-H 3rd overtone",
            sigma_range=(12, 18),
            gamma_range=(1, 3),
            intensity="very_weak",
            chemical_context="aromatic",
            common_compounds=["proteins (Phe, Tyr, Trp)", "polystyrene"],
            references=["Workman2012 p.57"],
            tags=["aromatic", "protein"],
        ),
        # 2nd overtone: 3ν (~9150 cm⁻¹)
        "2nd_overtone": BandAssignment(
            center=1140,
            functional_group="Ar C-H",
            overtone_level="2nd",
            assignment="3ν (aromatic C-H stretch)",
            description="Aromatic C-H 2nd overtone - diagnostic for aromatics",
            sigma_range=(12, 18),
            gamma_range=(1, 2),
            intensity="weak",
            chemical_context="aromatic",
            affected_by=["ring substitution"],
            common_compounds=["polystyrene", "PET", "lignin", "caffeine", "aromatic APIs"],
            references=["Workman2012 pp.56-58"],
            tags=["aromatic", "diagnostic"],
        ),
        "2nd_overtone_protein": BandAssignment(
            center=1145,
            functional_group="Ar C-H",
            overtone_level="2nd",
            assignment="3ν (aromatic C-H, Phe/Tyr/Trp)",
            description="Aromatic C-H 2nd overtone in proteins",
            sigma_range=(12, 16),
            gamma_range=(1, 2),
            intensity="weak",
            chemical_context="aromatic amino acids",
            common_compounds=["proteins", "peptides"],
            references=["Workman2012 p.51"],
            tags=["protein", "aromatic"],
        ),
        # 1st overtone: 2ν (~6100 cm⁻¹)
        "1st_overtone": BandAssignment(
            center=1680,
            functional_group="Ar C-H",
            overtone_level="1st",
            assignment="2ν (aromatic C-H stretch)",
            description="Aromatic C-H 1st overtone - strong aromatic marker",
            sigma_range=(15, 22),
            gamma_range=(1, 3),
            intensity="medium",
            chemical_context="aromatic",
            affected_by=["ring substitution", "conjugation"],
            common_compounds=["polystyrene", "PET", "lignin", "aromatic compounds"],
            references=["Workman2012 pp.56-58"],
            tags=["aromatic", "diagnostic"],
        ),
        "1st_overtone_protein": BandAssignment(
            center=1685,
            functional_group="Ar C-H",
            overtone_level="1st",
            assignment="2ν (aromatic C-H, Phe/Tyr/Trp residues)",
            description="Aromatic C-H 1st overtone in proteins",
            sigma_range=(20, 28),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="aromatic amino acids",
            common_compounds=["proteins", "gluten", "casein"],
            references=["Workman2012 p.51", "Burns2007 p.365"],
            tags=["protein"],
        ),
        # Combination bands
        "combination": BandAssignment(
            center=2150,
            functional_group="Ar C-H",
            overtone_level="combination",
            assignment="ν(Ar C-H) + ring modes",
            description="Aromatic C-H combination",
            sigma_range=(20, 28),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="aromatic",
            common_compounds=["polystyrene", "PET", "lignin"],
            references=["Workman2012 p.58"],
            tags=["aromatic"],
        ),
        "ring_combination": BandAssignment(
            center=2440,
            functional_group="Ar C-H",
            overtone_level="combination",
            assignment="Aromatic ring combination",
            description="Aromatic ring C=C + C-H combination",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="weak",
            chemical_context="aromatic",
            common_compounds=["aromatics", "PAH"],
            references=["Workman2012 p.58"],
            tags=["aromatic"],
        ),
    },

    # =========================================================================
    # =C-H BANDS (Olefinic/Vinyl)
    # Fundamental: ~3020 cm⁻¹ (3311 nm)
    # =========================================================================
    "C-H_olefinic": {
        "2nd_overtone": BandAssignment(
            center=1160,
            functional_group="=C-H",
            overtone_level="2nd",
            assignment="3ν (=C-H stretch)",
            description="Olefinic =C-H 2nd overtone - unsaturation marker",
            sigma_range=(13, 18),
            gamma_range=(1, 2),
            intensity="weak",
            chemical_context="unsaturated",
            affected_by=["degree of unsaturation"],
            common_compounds=["unsaturated fats", "oleic acid", "rubber"],
            references=["Murray1986 p.22", "Workman2012 p.47"],
            tags=["unsaturated", "lipid", "diagnostic"],
        ),
        "1st_overtone": BandAssignment(
            center=2145,
            functional_group="=C-H",
            overtone_level="1st",
            assignment="2ν (=C-H stretch)",
            description="Olefinic =C-H 1st overtone - key unsaturation marker",
            sigma_range=(18, 25),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="unsaturated",
            affected_by=["cis/trans isomerism", "conjugation"],
            common_compounds=["unsaturated fats", "oleic acid", "linoleic acid"],
            references=["Murray1986 p.23", "Osborne1993 p.70"],
            tags=["unsaturated", "lipid", "diagnostic"],
        ),
        "combination_CC": BandAssignment(
            center=2175,
            functional_group="=C-H",
            overtone_level="combination",
            assignment="ν(C=C) + ν(=C-H)",
            description="C=C + =C-H combination",
            sigma_range=(16, 22),
            gamma_range=(2, 3),
            intensity="weak",
            chemical_context="unsaturated",
            common_compounds=["unsaturated fats", "rubber"],
            references=["Murray1986 p.24"],
            tags=["unsaturated"],
        ),
    },

    # =========================================================================
    # C=O BANDS (Carbonyl groups)
    # Fundamental: ~1715 cm⁻¹ (5831 nm) - IR region
    # =========================================================================
    "C=O": {
        "2nd_overtone": BandAssignment(
            center=1700,
            functional_group="C=O",
            overtone_level="2nd",
            assignment="3ν (C=O stretch)",
            description="C=O 2nd overtone, carbonyl",
            sigma_range=(16, 24),
            gamma_range=(2, 3),
            intensity="weak",
            chemical_context="general carbonyl",
            common_compounds=["ketones", "aldehydes", "carboxylic acids"],
            references=["Bokobza1998 p.9", "Workman2012 p.42"],
            tags=["carbonyl", "ketone"],
        ),
        "1st_overtone_ketone": BandAssignment(
            center=1690,
            functional_group="C=O",
            overtone_level="1st",
            assignment="2ν (C=O stretch, ketone)",
            description="C=O 1st overtone, ketone",
            sigma_range=(18, 25),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="ketone",
            common_compounds=["acetone", "ketones"],
            references=["Workman2012 pp.42-44"],
            tags=["ketone", "solvent"],
        ),
        "combination_ester": BandAssignment(
            center=2015,
            functional_group="C=O",
            overtone_level="combination",
            assignment="ν(C=O) + ν(C-O) or δ(C-O)",
            description="C=O combination, ester",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="ester",
            common_compounds=["polyester", "PET", "ethyl acetate"],
            references=["Workman2012 p.62"],
            tags=["ester", "polymer"],
        ),
        "combination_general": BandAssignment(
            center=2020,
            functional_group="C=O",
            overtone_level="combination",
            assignment="ν(C=O) + δ combination",
            description="C=O combination, general",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="general",
            common_compounds=["esters", "acids", "amides"],
            references=["Reich2005 p.1127"],
            tags=["carbonyl"],
        ),
        "combination_carboxylic": BandAssignment(
            center=1920,
            functional_group="C=O",
            overtone_level="combination",
            assignment="ν(C=O) + δ(C-O-H)",
            description="C=O combination, carboxylic acid",
            sigma_range=(25, 35),
            gamma_range=(3, 4),
            intensity="medium",
            chemical_context="carboxylic acid",
            common_compounds=["citric acid", "malic acid", "tartaric acid"],
            references=["Osborne1993 pp.78-80"],
            tags=["organic_acid"],
        ),
    },

    # =========================================================================
    # METAL-HYDROXYL BANDS (Minerals)
    # =========================================================================
    "Al-OH": {
        "1st_overtone": BandAssignment(
            center=1400,
            functional_group="Al-OH",
            overtone_level="1st",
            assignment="2ν (Al-OH stretch)",
            description="Al-OH 1st overtone, clay minerals",
            sigma_range=(18, 25),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="clay mineral",
            common_compounds=["kaolinite", "illite", "montmorillonite"],
            references=["Clark1990", "Khayamim2015"],
            tags=["mineral", "clay", "soil"],
        ),
        "combination_1": BandAssignment(
            center=2160,
            functional_group="Al-OH",
            overtone_level="combination",
            assignment="ν(Al-OH) + δ(Al-OH)",
            description="Al-OH combination, clay minerals",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="clay mineral",
            common_compounds=["kaolinite"],
            references=["Clark1990"],
            tags=["mineral", "clay", "soil"],
        ),
        "combination_2": BandAssignment(
            center=2200,
            functional_group="Al-OH",
            overtone_level="combination",
            assignment="ν(Al-OH) + δ(Al-OH) doublet",
            description="Al-OH combination (characteristic kaolinite doublet)",
            sigma_range=(20, 28),
            gamma_range=(2, 3),
            intensity="strong",
            chemical_context="clay mineral",
            common_compounds=["kaolinite", "illite", "montmorillonite"],
            references=["Clark1990", "Khayamim2015"],
            tags=["mineral", "clay", "diagnostic"],
        ),
    },

    "Fe-OH": {
        "1st_overtone": BandAssignment(
            center=1420,
            functional_group="Fe-OH",
            overtone_level="1st",
            assignment="2ν (Fe-OH stretch)",
            description="Fe-OH 1st overtone, iron oxyhydroxides",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="iron hydroxide",
            common_compounds=["goethite", "ferrihydrite"],
            references=["Clark1990"],
            tags=["mineral", "iron", "soil"],
        ),
        "combination_1": BandAssignment(
            center=1920,
            functional_group="Fe-OH",
            overtone_level="combination",
            assignment="ν(Fe-OH) + δ(Fe-OH)",
            description="Fe-OH combination",
            sigma_range=(28, 35),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="iron hydroxide",
            common_compounds=["goethite"],
            references=["Clark1990"],
            tags=["mineral", "iron"],
        ),
        "combination_2": BandAssignment(
            center=2260,
            functional_group="Fe-OH",
            overtone_level="combination",
            assignment="ν(Fe-OH) + δ(Fe-OH)",
            description="Fe-OH combination 2",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="iron hydroxide",
            common_compounds=["goethite", "illite"],
            references=["Clark1990"],
            tags=["mineral", "iron"],
        ),
    },

    "Mg-OH": {
        "1st_overtone": BandAssignment(
            center=1395,
            functional_group="Mg-OH",
            overtone_level="1st",
            assignment="2ν (Mg-OH stretch)",
            description="Mg-OH 1st overtone, magnesium silicates",
            sigma_range=(16, 22),
            gamma_range=(1, 3),
            intensity="medium",
            chemical_context="magnesium silicate",
            common_compounds=["talc", "serpentine"],
            references=["Clark1990"],
            tags=["mineral"],
        ),
        "combination_1": BandAssignment(
            center=2315,
            functional_group="Mg-OH",
            overtone_level="combination",
            assignment="ν(Mg-OH) + δ(Mg-OH)",
            description="Mg-OH combination",
            sigma_range=(20, 26),
            gamma_range=(2, 3),
            intensity="strong",
            chemical_context="magnesium silicate",
            common_compounds=["talc"],
            references=["Clark1990"],
            tags=["mineral", "diagnostic"],
        ),
        "combination_2": BandAssignment(
            center=2390,
            functional_group="Mg-OH",
            overtone_level="combination",
            assignment="ν(Mg-OH) + δ(Mg-OH)",
            description="Mg-OH combination 2",
            sigma_range=(18, 24),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="magnesium silicate",
            common_compounds=["talc"],
            references=["Clark1990"],
            tags=["mineral"],
        ),
    },

    "Si-OH": {
        "1st_overtone": BandAssignment(
            center=1380,
            functional_group="Si-OH",
            overtone_level="1st",
            assignment="2ν (Si-OH stretch)",
            description="Si-OH 1st overtone, silanol",
            sigma_range=(18, 25),
            gamma_range=(2, 3),
            intensity="medium",
            chemical_context="silica",
            common_compounds=["silica gel", "glass"],
            references=["Clark1990"],
            tags=["mineral", "silica"],
        ),
        "combination": BandAssignment(
            center=2220,
            functional_group="Si-OH",
            overtone_level="combination",
            assignment="ν(Si-OH) + δ(Si-OH)",
            description="Si-OH combination",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="silica",
            common_compounds=["silica"],
            references=["Clark1990"],
            tags=["mineral", "silica"],
        ),
    },

    # =========================================================================
    # CARBONATE BANDS
    # =========================================================================
    "CO3": {
        "combination_1": BandAssignment(
            center=2330,
            functional_group="CO₃",
            overtone_level="combination",
            assignment="ν₁ + ν₃ (CO₃ combination)",
            description="Carbonate combination band",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="carbonate",
            common_compounds=["calcite", "dolomite", "limestone"],
            references=["Clark1990", "Khayamim2015"],
            tags=["mineral", "carbonate", "soil"],
        ),
        "combination_2": BandAssignment(
            center=2525,
            functional_group="CO₃",
            overtone_level="combination",
            assignment="ν₁ + 2ν₂ (CO₃ combination)",
            description="Carbonate combination band 2",
            sigma_range=(28, 38),
            gamma_range=(3, 5),
            intensity="weak",
            chemical_context="carbonate",
            common_compounds=["calcite", "dolomite"],
            references=["Clark1990"],
            tags=["mineral", "carbonate"],
        ),
    },

    # =========================================================================
    # SULFATE BANDS
    # =========================================================================
    "SO4": {
        "H2O_combination": BandAssignment(
            center=1740,
            functional_group="SO₄·H₂O",
            overtone_level="combination",
            assignment="Crystal water / SO₄ combination",
            description="Hydrated sulfate combination",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="hydrated sulfate",
            common_compounds=["gypsum"],
            references=["Khayamim2015"],
            tags=["mineral", "sulfate"],
        ),
        "combination": BandAssignment(
            center=2200,
            functional_group="SO₄",
            overtone_level="combination",
            assignment="SO₄ / H₂O combination",
            description="Sulfate combination",
            sigma_range=(25, 32),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="sulfate",
            common_compounds=["gypsum"],
            references=["Khayamim2015"],
            tags=["mineral", "sulfate"],
        ),
    },

    # =========================================================================
    # S=O BANDS (Sulfoxide, Sulfone)
    # =========================================================================
    "S=O": {
        "combination": BandAssignment(
            center=2030,
            functional_group="S=O",
            overtone_level="combination",
            assignment="ν(S=O) combination",
            description="S=O combination, sulfoxide/sulfone",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="sulfoxide",
            common_compounds=["DMSO", "omeprazole"],
            references=["Reich2005"],
            tags=["pharma", "solvent"],
        ),
    },

    # =========================================================================
    # C-F BANDS (Fluorocarbons)
    # =========================================================================
    "C-F": {
        "combination_1": BandAssignment(
            center=2180,
            functional_group="C-F",
            overtone_level="combination",
            assignment="C-F combination",
            description="C-F combination band",
            sigma_range=(22, 30),
            gamma_range=(2, 4),
            intensity="weak",
            chemical_context="fluorocarbon",
            common_compounds=["PTFE", "Teflon"],
            references=["Workman2012"],
            tags=["polymer", "fluorocarbon"],
        ),
        "overtone": BandAssignment(
            center=2365,
            functional_group="C-F",
            overtone_level="1st",
            assignment="2ν (C-F stretch)",
            description="C-F overtone",
            sigma_range=(20, 26),
            gamma_range=(2, 3),
            intensity="weak",
            chemical_context="fluorocarbon",
            common_compounds=["PTFE"],
            references=["Workman2012"],
            tags=["polymer", "fluorocarbon"],
        ),
    },

    # =========================================================================
    # C-Cl BANDS (Chlorocarbons)
    # =========================================================================
    "C-Cl": {
        "combination": BandAssignment(
            center=2250,
            functional_group="C-Cl",
            overtone_level="combination",
            assignment="C-Cl combination",
            description="C-Cl combination band",
            sigma_range=(20, 28),
            gamma_range=(2, 4),
            intensity="weak",
            chemical_context="chlorocarbon",
            common_compounds=["chloroform", "PVC"],
            references=["Workman2012"],
            tags=["solvent", "polymer"],
        ),
    },

    # =========================================================================
    # P-O BANDS (Phosphate)
    # =========================================================================
    "P-O": {
        "combination": BandAssignment(
            center=2165,
            functional_group="P-O",
            overtone_level="combination",
            assignment="P-O combination",
            description="P-O combination, phosphate/phospholipid",
            sigma_range=(20, 28),
            gamma_range=(2, 4),
            intensity="weak",
            chemical_context="phosphate",
            common_compounds=["phospholipids", "ATP"],
            references=["Siesler2002"],
            tags=["biological", "phospholipid"],
        ),
    },

    # =========================================================================
    # VISIBLE REGION - ELECTRONIC TRANSITIONS
    # =========================================================================
    "Porphyrin": {
        "soret_chlorophyll_a": BandAssignment(
            center=430,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Soret band (B band), π→π*",
            description="Chlorophyll a Soret band (blue absorption)",
            sigma_range=(12, 18),
            gamma_range=(2, 4),
            intensity="very_strong",
            chemical_context="chlorophyll a",
            common_compounds=["chlorophyll a"],
            references=["Wellburn1994", "Lichtenthaler1987"],
            tags=["pigment", "chlorophyll", "visible"],
        ),
        "soret_chlorophyll_b": BandAssignment(
            center=453,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Soret band (B band), π→π*",
            description="Chlorophyll b Soret band (blue absorption)",
            sigma_range=(13, 18),
            gamma_range=(2, 4),
            intensity="very_strong",
            chemical_context="chlorophyll b",
            common_compounds=["chlorophyll b"],
            references=["Wellburn1994"],
            tags=["pigment", "chlorophyll", "visible"],
        ),
        "Q_chlorophyll_a": BandAssignment(
            center=662,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Q band (Qy), π→π*",
            description="Chlorophyll a Q band (red absorption)",
            sigma_range=(8, 14),
            gamma_range=(1, 3),
            intensity="very_strong",
            chemical_context="chlorophyll a",
            common_compounds=["chlorophyll a"],
            references=["Wellburn1994"],
            tags=["pigment", "chlorophyll", "visible", "diagnostic"],
        ),
        "Q_chlorophyll_b": BandAssignment(
            center=642,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Q band (Qy), π→π*",
            description="Chlorophyll b Q band (red absorption)",
            sigma_range=(9, 14),
            gamma_range=(1, 3),
            intensity="very_strong",
            chemical_context="chlorophyll b",
            common_compounds=["chlorophyll b"],
            references=["Wellburn1994"],
            tags=["pigment", "chlorophyll", "visible"],
        ),
        "soret_hemoglobin_oxy": BandAssignment(
            center=414,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Soret band, oxyhemoglobin",
            description="Oxyhemoglobin Soret band",
            sigma_range=(10, 15),
            gamma_range=(2, 4),
            intensity="very_strong",
            chemical_context="oxyhemoglobin",
            common_compounds=["oxyhemoglobin"],
            references=["Prahl1999"],
            tags=["blood", "hemoglobin", "visible", "medical"],
        ),
        "soret_hemoglobin_deoxy": BandAssignment(
            center=430,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Soret band, deoxyhemoglobin",
            description="Deoxyhemoglobin Soret band",
            sigma_range=(12, 18),
            gamma_range=(2, 5),
            intensity="very_strong",
            chemical_context="deoxyhemoglobin",
            common_compounds=["deoxyhemoglobin"],
            references=["Prahl1999"],
            tags=["blood", "hemoglobin", "visible", "medical"],
        ),
        "Q_hemoglobin_oxy_alpha": BandAssignment(
            center=577,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Q band α, oxyhemoglobin",
            description="Oxyhemoglobin Q band α",
            sigma_range=(8, 12),
            gamma_range=(1, 3),
            intensity="medium",
            chemical_context="oxyhemoglobin",
            common_compounds=["oxyhemoglobin"],
            references=["Prahl1999"],
            tags=["blood", "hemoglobin", "visible"],
        ),
        "Q_hemoglobin_oxy_beta": BandAssignment(
            center=542,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Q band β, oxyhemoglobin",
            description="Oxyhemoglobin Q band β",
            sigma_range=(8, 12),
            gamma_range=(1, 3),
            intensity="medium",
            chemical_context="oxyhemoglobin",
            common_compounds=["oxyhemoglobin"],
            references=["Prahl1999"],
            tags=["blood", "hemoglobin", "visible"],
        ),
        "Q_hemoglobin_deoxy": BandAssignment(
            center=555,
            functional_group="Porphyrin",
            overtone_level="electronic",
            assignment="Q band, deoxyhemoglobin",
            description="Deoxyhemoglobin Q band (single broad)",
            sigma_range=(15, 22),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="deoxyhemoglobin",
            common_compounds=["deoxyhemoglobin"],
            references=["Prahl1999"],
            tags=["blood", "hemoglobin", "visible"],
        ),
    },

    "Carotenoid": {
        "pi_pi_1": BandAssignment(
            center=425,
            functional_group="Carotenoid",
            overtone_level="electronic",
            assignment="π→π* (0-2 vibronic)",
            description="Carotenoid absorption (shoulder)",
            sigma_range=(10, 15),
            gamma_range=(2, 3),
            intensity="strong",
            chemical_context="carotenoid",
            common_compounds=["β-carotene", "lutein", "zeaxanthin"],
            references=["Britton2004"],
            tags=["pigment", "carotenoid", "visible"],
        ),
        "pi_pi_2": BandAssignment(
            center=450,
            functional_group="Carotenoid",
            overtone_level="electronic",
            assignment="π→π* (0-1 vibronic, λmax)",
            description="Carotenoid main absorption peak",
            sigma_range=(12, 16),
            gamma_range=(2, 4),
            intensity="very_strong",
            chemical_context="carotenoid",
            common_compounds=["β-carotene", "lutein"],
            references=["Britton2004"],
            tags=["pigment", "carotenoid", "visible", "diagnostic"],
        ),
        "pi_pi_3": BandAssignment(
            center=478,
            functional_group="Carotenoid",
            overtone_level="electronic",
            assignment="π→π* (0-0 vibronic)",
            description="Carotenoid absorption (shoulder)",
            sigma_range=(12, 16),
            gamma_range=(2, 4),
            intensity="strong",
            chemical_context="carotenoid",
            common_compounds=["β-carotene", "lutein"],
            references=["Britton2004"],
            tags=["pigment", "carotenoid", "visible"],
        ),
        "lycopene_max": BandAssignment(
            center=471,
            functional_group="Carotenoid",
            overtone_level="electronic",
            assignment="π→π* (lycopene λmax)",
            description="Lycopene main absorption peak (tomato red)",
            sigma_range=(12, 18),
            gamma_range=(2, 4),
            intensity="very_strong",
            chemical_context="lycopene",
            common_compounds=["lycopene", "tomato"],
            references=["Britton2004"],
            tags=["pigment", "carotenoid", "visible"],
        ),
    },

    "Anthocyanin": {
        "red": BandAssignment(
            center=520,
            functional_group="Anthocyanin",
            overtone_level="electronic",
            assignment="π→π* (visible absorption)",
            description="Anthocyanin red absorption (pH dependent)",
            sigma_range=(22, 30),
            gamma_range=(3, 5),
            intensity="very_strong",
            chemical_context="anthocyanin red",
            affected_by=["pH", "copigmentation"],
            common_compounds=["cyanidin", "pelargonidin", "berries"],
            references=["Giusti2001"],
            tags=["pigment", "anthocyanin", "visible"],
        ),
        "purple": BandAssignment(
            center=550,
            functional_group="Anthocyanin",
            overtone_level="electronic",
            assignment="π→π* (visible absorption)",
            description="Anthocyanin purple absorption",
            sigma_range=(24, 32),
            gamma_range=(3, 5),
            intensity="very_strong",
            chemical_context="anthocyanin purple",
            affected_by=["pH"],
            common_compounds=["delphinidin", "grapes", "blueberries"],
            references=["Giusti2001"],
            tags=["pigment", "anthocyanin", "visible"],
        ),
    },

    "Red_edge": {
        "chlorophyll_tail": BandAssignment(
            center=720,
            functional_group="Red edge",
            overtone_level="electronic",
            assignment="Electronic tail (chlorophyll)",
            description="Chlorophyll red edge / far-red absorption",
            sigma_range=(18, 28),
            gamma_range=(2, 4),
            intensity="medium",
            chemical_context="chlorophyll",
            common_compounds=["chlorophyll", "vegetation"],
            references=["Wellburn1994"],
            tags=["vegetation", "chlorophyll", "visible-NIR"],
        ),
        "hemoglobin_deoxy": BandAssignment(
            center=760,
            functional_group="Red edge",
            overtone_level="electronic",
            assignment="NIR absorption (deoxyhemoglobin)",
            description="Deoxyhemoglobin NIR absorption",
            sigma_range=(25, 35),
            gamma_range=(3, 5),
            intensity="weak",
            chemical_context="deoxyhemoglobin",
            common_compounds=["deoxyhemoglobin"],
            references=["Prahl1999"],
            tags=["blood", "hemoglobin", "visible-NIR"],
        ),
    },
}

# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_band(functional_group: str, band_key: str) -> BandAssignment:
    """
    Get a specific band assignment by functional group and key.

    Args:
        functional_group: Functional group name (e.g., "O-H", "C-H_aliphatic").
        band_key: Band key within the group (e.g., "1st_overtone_water").

    Returns:
        BandAssignment object.

    Raises:
        KeyError: If functional group or band key not found.

    Example:
        >>> band = get_band("O-H", "1st_overtone_water")
        >>> print(f"{band.center} nm: {band.description}")
        1450 nm: O-H 1st overtone, water
    """
    if functional_group not in NIR_BANDS:
        raise KeyError(f"Unknown functional group: '{functional_group}'. "
                       f"Available: {list(NIR_BANDS.keys())}")
    if band_key not in NIR_BANDS[functional_group]:
        raise KeyError(f"Unknown band key '{band_key}' in '{functional_group}'. "
                       f"Available: {list(NIR_BANDS[functional_group].keys())}")
    return NIR_BANDS[functional_group][band_key]

def list_functional_groups() -> list[str]:
    """
    List all available functional groups.

    Returns:
        Sorted list of functional group names.

    Example:
        >>> groups = list_functional_groups()
        >>> print(groups[:5])
        ['Al-OH', 'Anthocyanin', 'C-Cl', 'C-F', 'C-H_aliphatic']
    """
    return sorted(NIR_BANDS.keys())

def list_bands(functional_group: str | None = None) -> list[str]:
    """
    List available bands, optionally filtered by functional group.

    Args:
        functional_group: If provided, list only bands for this group.

    Returns:
        List of band keys (if functional_group specified) or
        list of "group/key" strings (if no filter).

    Example:
        >>> # List all O-H bands
        >>> oh_bands = list_bands("O-H")
        >>> print(oh_bands[:3])
        ['1st_overtone_bound', '1st_overtone_carbohydrate', '1st_overtone_carboxylic']
        >>>
        >>> # List all bands
        >>> all_bands = list_bands()
    """
    if functional_group is not None:
        if functional_group not in NIR_BANDS:
            raise KeyError(f"Unknown functional group: '{functional_group}'")
        return sorted(NIR_BANDS[functional_group].keys())

    result = []
    for group, bands in NIR_BANDS.items():
        for key in bands:
            result.append(f"{group}/{key}")
    return sorted(result)

def get_bands_in_range(
    wavelength_min: float,
    wavelength_max: float,
    functional_groups: list[str] | None = None,
) -> list[BandAssignment]:
    """
    Get all bands with centers in a wavelength range.

    Args:
        wavelength_min: Minimum wavelength (nm).
        wavelength_max: Maximum wavelength (nm).
        functional_groups: Optional filter for specific functional groups.

    Returns:
        List of BandAssignment objects sorted by center wavelength.

    Example:
        >>> # Get all bands in the 1st overtone region
        >>> bands = get_bands_in_range(1400, 1600)
        >>> for b in bands[:3]:
        ...     print(f"{b.center} nm: {b.functional_group} {b.description}")
    """
    result = []

    groups_to_search = functional_groups if functional_groups else NIR_BANDS.keys()

    for group in groups_to_search:
        if group not in NIR_BANDS:
            continue
        for band in NIR_BANDS[group].values():
            if wavelength_min <= band.center <= wavelength_max:
                result.append(band)

    return sorted(result, key=lambda b: b.center)

def get_bands_by_tag(tag: str) -> list[BandAssignment]:
    """
    Get all bands with a specific tag.

    Args:
        tag: Tag to filter by (e.g., "water", "protein", "diagnostic").

    Returns:
        List of BandAssignment objects with the specified tag.

    Example:
        >>> # Get all diagnostic bands
        >>> diagnostic = get_bands_by_tag("diagnostic")
        >>> print(len(diagnostic))
    """
    result = []
    for group_bands in NIR_BANDS.values():
        for band in group_bands.values():
            if tag in band.tags:
                result.append(band)
    return sorted(result, key=lambda b: b.center)

def get_bands_by_overtone(overtone_level: str) -> list[BandAssignment]:
    """
    Get all bands of a specific overtone level.

    Args:
        overtone_level: Overtone level ("1st", "2nd", "3rd", "combination", "electronic").

    Returns:
        List of BandAssignment objects of the specified overtone level.

    Example:
        >>> # Get all 1st overtone bands
        >>> first_overtones = get_bands_by_overtone("1st")
        >>> print(f"Found {len(first_overtones)} 1st overtone bands")
    """
    result = []
    for group_bands in NIR_BANDS.values():
        for band in group_bands.values():
            if band.overtone_level == overtone_level:
                result.append(band)
    return sorted(result, key=lambda b: b.center)

def get_bands_by_compound(compound: str) -> list[BandAssignment]:
    """
    Get all bands commonly found in a specific compound.

    Args:
        compound: Compound name (e.g., "water", "protein", "cellulose").

    Returns:
        List of BandAssignment objects where the compound appears in common_compounds.

    Example:
        >>> # Get bands found in water
        >>> water_bands = get_bands_by_compound("water")
        >>> for b in water_bands:
        ...     print(f"{b.center} nm: {b.description}")
    """
    result = []
    compound_lower = compound.lower()
    for group_bands in NIR_BANDS.values():
        for band in group_bands.values():
            if any(compound_lower in c.lower() for c in band.common_compounds):
                result.append(band)
    return sorted(result, key=lambda b: b.center)

def generate_band_spectrum(
    band: BandAssignment,
    wavelengths: np.ndarray,
    amplitude: float = 1.0,
    sigma: float | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    """
    Generate a spectrum for a single band.

    Args:
        band: BandAssignment to generate spectrum for.
        wavelengths: Array of wavelengths in nm.
        amplitude: Peak amplitude.
        sigma: Gaussian width (default: midpoint of sigma_range).
        gamma: Lorentzian width (default: midpoint of gamma_range).

    Returns:
        Array of absorbance values at each wavelength.

    Example:
        >>> band = get_band("O-H", "1st_overtone_water")
        >>> wl = np.arange(1300, 1600, 1)
        >>> spectrum = generate_band_spectrum(band, wl, amplitude=0.8)
    """
    nir_band = band.to_nir_band(amplitude=amplitude, sigma=sigma, gamma=gamma)
    return np.asarray(nir_band.compute(wavelengths))

def band_info(functional_group: str, band_key: str) -> str:
    """
    Return formatted information about a specific band.

    Args:
        functional_group: Functional group name.
        band_key: Band key within the group.

    Returns:
        Human-readable string with band details.

    Example:
        >>> print(band_info("O-H", "1st_overtone_water"))
    """
    band = get_band(functional_group, band_key)
    return band.info()

def list_all_tags() -> list[str]:
    """
    List all unique tags used across all bands.

    Returns:
        Sorted list of unique tag names.

    Example:
        >>> tags = list_all_tags()
        >>> print(tags[:10])
    """
    tags = set()
    for group_bands in NIR_BANDS.values():
        for band in group_bands.values():
            tags.update(band.tags)
    return sorted(tags)

def validate_bands() -> list[str]:
    """
    Validate all band assignments.

    Returns:
        List of validation issues (empty if all valid).

    Example:
        >>> issues = validate_bands()
        >>> if issues:
        ...     for issue in issues:
        ...         print(f"Warning: {issue}")
    """
    issues = []

    for group, bands in NIR_BANDS.items():
        for key, band in bands.items():
            # Check wavelength range
            if band.center < 200 or band.center > 3000:
                issues.append(f"{group}/{key}: center {band.center} nm outside valid range (200-3000)")

            # Check sigma range
            if band.sigma_range[0] >= band.sigma_range[1]:
                issues.append(f"{group}/{key}: invalid sigma_range {band.sigma_range}")

            # Check gamma range
            if band.gamma_range[0] > band.gamma_range[1]:
                issues.append(f"{group}/{key}: invalid gamma_range {band.gamma_range}")

            # Check intensity value
            valid_intensities = ["very_weak", "weak", "medium", "strong", "very_strong"]
            if band.intensity not in valid_intensities:
                issues.append(f"{group}/{key}: invalid intensity '{band.intensity}'")

    return issues

def summary() -> str:
    """
    Return a summary of the band assignments dictionary.

    Returns:
        Human-readable summary string.

    Example:
        >>> print(summary())
    """
    total_bands = sum(len(bands) for bands in NIR_BANDS.values())
    all_tags = list_all_tags()

    lines = [
        "NIR Band Assignments Summary",
        "=" * 40,
        f"Total functional groups: {len(NIR_BANDS)}",
        f"Total band assignments: {total_bands}",
        f"Unique tags: {len(all_tags)}",
        "",
        "Bands by functional group:",
    ]

    for group in sorted(NIR_BANDS.keys()):
        lines.append(f"  {group}: {len(NIR_BANDS[group])} bands")

    lines.extend([
        "",
        "Bands by overtone level:",
    ])

    for level in ["3rd", "2nd", "1st", "combination", "electronic"]:
        count = len(get_bands_by_overtone(level))
        if count > 0:
            lines.append(f"  {level}: {count} bands")

    return "\n".join(lines)
