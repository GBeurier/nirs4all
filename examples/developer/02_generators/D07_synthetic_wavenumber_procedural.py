"""
D07 - Synthetic Generator: Wavenumber Utilities & Procedural Components
========================================================================

Learn to use wavenumber-based calculations and procedural component generation
for physically-accurate synthetic NIR spectral data.

This tutorial covers:

* Wavenumber ↔ wavelength conversions
* NIR spectral zones classification
* Overtone calculation with anharmonicity
* Combination band calculation
* Hydrogen bonding effects
* Procedural component generation
* Functional group types and properties
* Creating chemically-plausible synthetic spectra

Prerequisites
-------------
Complete user examples U05, U06, and developer examples D05, D06 first.

Duration: ~5 minutes
Difficulty: ★★★★☆

Phase 1 Feature
---------------
This example demonstrates Phase 1 features from the Synthetic Generator
Enhancement Roadmap: wavenumber utilities and procedural component generation.
"""

# Standard library imports
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Third-party imports
import numpy as np

# NIRS4All imports
import nirs4all
from nirs4all.synthesis import (
    FUNCTIONAL_GROUP_PROPERTIES,
    FUNDAMENTAL_VIBRATIONS,
    NIR_ZONES_WAVENUMBER,
    # Library and generator
    ComponentLibrary,
    # Procedural generation
    FunctionalGroupType,
    ProceduralComponentConfig,
    ProceduralComponentGenerator,
    SyntheticNIRSGenerator,
    apply_hydrogen_bonding_shift,
    calculate_combination_band,
    calculate_overtone_position,
    classify_wavelength_zone,
    wavelength_to_wavenumber,
    # Wavenumber utilities
    wavenumber_to_wavelength,
)

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from example_utils import get_example_output_path, print_output_location, save_array_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D07 Wavenumber & Procedural Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "D07_synthetic_wavenumber_procedural"

# =============================================================================
# Section 1: Wavenumber ↔ Wavelength Conversion
# =============================================================================
print("\n" + "=" * 60)
print("D07 - Wavenumber Utilities & Procedural Components")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Wavenumber ↔ Wavelength Conversion")
print("-" * 60)

# Basic conversions
print("\n📊 Wavenumber ↔ Wavelength Relationships:")
print("   Formula: λ (nm) = 10⁷ / ν̃ (cm⁻¹)")
print()

# Example conversions
wavelengths_nm = [1000, 1450, 1700, 1940, 2100, 2300]
print(f"   {'Wavelength (nm)':<18} {'Wavenumber (cm⁻¹)':<20} {'Zone'}")
print(f"   {'-'*18} {'-'*20} {'-'*25}")

for wl in wavelengths_nm:
    wn = wavelength_to_wavenumber(wl)
    zone = classify_wavelength_zone(wl)
    print(f"   {wl:<18} {wn:<20.1f} {zone}")

# Reverse conversion
print("\n📊 Wavenumber to Wavelength:")
wavenumbers_cm = [10000, 6900, 5800, 5150, 4762, 4350]
for wn in wavenumbers_cm:
    wl = wavenumber_to_wavelength(wn)
    print(f"   {wn:.0f} cm⁻¹ → {wl:.1f} nm")

# =============================================================================
# Section 2: NIR Spectral Zones
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: NIR Spectral Zones")
print("-" * 60)

print("\n📊 NIR Spectral Zones (in wavenumber space):")
print(f"   {'Zone':<25} {'Wavenumber (cm⁻¹)':<20} {'Wavelength (nm)':<20}")
print(f"   {'-'*25} {'-'*20} {'-'*20}")

for wn_min, wn_max, zone_name in NIR_ZONES_WAVENUMBER:
    wl_start = wavenumber_to_wavelength(wn_max)  # Higher wn = shorter wl
    wl_end = wavenumber_to_wavelength(wn_min)
    print(f"   {zone_name:<25} {wn_min}-{wn_max:<12} {wl_start:.0f}-{wl_end:.0f}")

# =============================================================================
# Section 3: Fundamental Vibrations
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Fundamental Vibrations Database")
print("-" * 60)

print("\n📊 Fundamental Vibrations (22 entries):")
print(f"   {'Vibration':<25} {'Wavenumber (cm⁻¹)':<20} {'Wavelength (nm)'}")
print(f"   {'-'*25} {'-'*20} {'-'*15}")

# Show selected fundamentals by category
categories = {
    "O-H stretches": ["O-H_stretch_free", "O-H_stretch_hbond"],
    "C-H stretches": ["C-H_stretch_CH3_asym", "C-H_stretch_CH2_asym", "C-H_stretch_aromatic"],
    "N-H stretches": ["N-H_stretch_primary", "N-H_stretch_secondary"],
    "Bending modes": ["O-H_bend", "C-H_bend", "N-H_bend"],
}

for cat_name, vibrations in categories.items():
    print(f"\n   {cat_name}:")
    for vib in vibrations:
        wn = FUNDAMENTAL_VIBRATIONS.get(vib)
        if wn:
            wl = wavenumber_to_wavelength(wn)
            print(f"     {vib:<23} {wn:<20} {wl:.1f}")

# =============================================================================
# Section 4: Overtone Calculation
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Overtone Calculation with Anharmonicity")
print("-" * 60)

print("\n📊 Overtone Formula: ν̃ₙ = n × ν̃₀ × (1 - n × χ)")
print("   where n = quantum number, χ = anharmonicity (~0.02)")
print()

# Calculate O-H overtone series
print("   O-H stretch overtone series (free O-H, 3650 cm⁻¹):")
print(f"   {'n':<4} {'Type':<20} {'Wavelength (nm)':<18} {'Amplitude'}")
print(f"   {'-'*4} {'-'*20} {'-'*18} {'-'*10}")

for n in range(1, 5):
    result = calculate_overtone_position("O-H_stretch_free", n)
    type_name = {1: "Fundamental", 2: "1st overtone", 3: "2nd overtone", 4: "3rd overtone"}[n]
    print(f"   {n:<4} {type_name:<20} {result.wavelength_nm:<18.1f} {result.amplitude_factor:.4f}")

# C-H overtone series
print("\n   C-H stretch overtone series (CH3 asymmetric, 2960 cm⁻¹):")
for n in range(1, 5):
    result = calculate_overtone_position("C-H_stretch_CH3_asym", n)
    type_name = {1: "Fundamental", 2: "1st overtone", 3: "2nd overtone", 4: "3rd overtone"}[n]
    print(f"   {n:<4} {type_name:<20} {result.wavelength_nm:<18.1f} {result.amplitude_factor:.4f}")

# Custom anharmonicity
print("\n   Custom anharmonicity example:")
result_low = calculate_overtone_position(3400, 2, anharmonicity=0.01)
result_high = calculate_overtone_position(3400, 2, anharmonicity=0.03)
print("   Same fundamental (3400 cm⁻¹) with different χ:")
print(f"     χ = 0.01: 1st overtone at {result_low.wavelength_nm:.1f} nm")
print(f"     χ = 0.03: 1st overtone at {result_high.wavelength_nm:.1f} nm")

# =============================================================================
# Section 5: Combination Bands
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Combination Band Calculation")
print("-" * 60)

print("\n📊 Combination bands arise from simultaneous excitation of two modes")
print("   ν̃_comb ≈ ν̃₁ + ν̃₂")
print()

# O-H stretch + bend combination
result = calculate_combination_band(["O-H_stretch_free", "O-H_bend"])
print("   O-H stretch + O-H bend:")
print(f"     Modes: 3650 + 1640 = {3650 + 1640} cm⁻¹")
print(f"     Combination band: {result.wavelength_nm:.1f} nm")
print("     (Observed: ~1890 nm)")

# C-H combination
result = calculate_combination_band(["C-H_stretch_CH3_asym", "C-H_bend"])
print("\n   C-H stretch + C-H bend:")
print(f"     Modes: 2960 + 1465 = {2960 + 1465} cm⁻¹")
print(f"     Combination band: {result.wavelength_nm:.1f} nm")
print("     (Observed: ~2250 nm)")

# N-H combination
result = calculate_combination_band(["N-H_stretch_primary", "N-H_bend"])
print("\n   N-H stretch + N-H bend:")
print(f"     Modes: 3400 + 1600 = {3400 + 1600} cm⁻¹")
print(f"     Combination band: {result.wavelength_nm:.1f} nm")

# =============================================================================
# Section 6: Hydrogen Bonding Effects
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Hydrogen Bonding Effects")
print("-" * 60)

print("\n📊 H-bonding shifts bands to lower wavenumbers (longer wavelengths)")
print("   Typical shift: 100-300 cm⁻¹ depending on H-bond strength")
print()

# O-H H-bonding
free_oh = 3650  # cm⁻¹
print(f"   Free O-H stretch: {free_oh} cm⁻¹ ({wavenumber_to_wavelength(free_oh):.0f} nm)")

for h_bond_strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
    bonded = apply_hydrogen_bonding_shift(free_oh, h_bond_strength=h_bond_strength)
    wl = wavenumber_to_wavelength(bonded)
    shift = free_oh - bonded
    print(f"   H-bond strength {h_bond_strength:.2f}: {bonded:.0f} cm⁻¹ ({wl:.0f} nm), shift = -{shift:.0f} cm⁻¹")

# =============================================================================
# Section 7: Functional Group Types
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Functional Group Types")
print("-" * 60)

print("\n📊 Available Functional Group Types:")
for fg_type in FunctionalGroupType:
    props = FUNCTIONAL_GROUP_PROPERTIES.get(fg_type, {})
    fundamental = props.get('fundamental_cm', 'N/A')
    hbond = props.get('h_bond_susceptibility', 0)
    amplitude = props.get('typical_amplitude', 1.0)
    print(f"   {fg_type.name:<15} fundamental: {fundamental:<8} H-bond: {hbond:<5} amplitude: {amplitude}")

# =============================================================================
# Section 8: Procedural Component Generation
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Procedural Component Generation")
print("-" * 60)

# Create generator
generator = ProceduralComponentGenerator(random_state=42)

# Generate simple component
print("\n📊 Generating simple component (3 fundamental bands):")
simple_config = ProceduralComponentConfig(
    n_fundamental_bands=3,
    include_overtones=True,
    max_overtone_order=2,
    include_combinations=False,
    wavelength_range=(900, 2500),
)

component1 = generator.generate_component(
    name="simple_alcohol",
    config=simple_config,
    functional_groups=[FunctionalGroupType.HYDROXYL, FunctionalGroupType.METHYL],
)

print(f"   Component: {component1.name}")
print(f"   Total bands: {len(component1.bands)}")
for i, band in enumerate(component1.bands[:5]):
    print(f"     Band {i+1}: {band.center:.1f} nm, amplitude {band.amplitude:.3f}")
if len(component1.bands) > 5:
    print(f"     ... and {len(component1.bands) - 5} more bands")

# Generate complex component
print("\n📊 Generating complex component (with combinations):")
complex_config = ProceduralComponentConfig(
    n_fundamental_bands=5,
    include_overtones=True,
    max_overtone_order=3,
    include_combinations=True,
    max_combinations=3,
    h_bond_strength=0.6,
    h_bond_variability=0.15,
    wavelength_range=(900, 2500),
)

component2 = generator.generate_component(
    name="complex_carbohydrate",
    config=complex_config,
    functional_groups=[
        FunctionalGroupType.HYDROXYL,
        FunctionalGroupType.HYDROXYL,
        FunctionalGroupType.METHYLENE,
    ],
)

print(f"   Component: {component2.name}")
print(f"   Total bands: {len(component2.bands)}")
for i, band in enumerate(component2.bands[:8]):
    print(f"     Band {i+1}: {band.center:.1f} nm, amplitude {band.amplitude:.3f}")
if len(component2.bands) > 8:
    print(f"     ... and {len(component2.bands) - 8} more bands")

# =============================================================================
# Section 9: Integration with Generator
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Integration with Generator")
print("-" * 60)

# Create library with procedural components
library = ComponentLibrary(random_state=42)

# Add predefined components
predefined = ComponentLibrary.from_predefined(["water", "starch"])
for comp in predefined.components.values():
    library.add_component(comp)

# Add procedurally generated components
proc_gen = ProceduralComponentGenerator(random_state=42)
for i in range(3):
    config = ProceduralComponentConfig(
        n_fundamental_bands=3 + i,
        include_overtones=True,
        max_overtone_order=2,
    )
    component = proc_gen.generate_component(
        name=f"procedural_{i+1}",
        config=config,
    )
    library.add_component(component)

print("\n📊 Library with mixed components:")
print(f"   Total components: {len(library.component_names)}")
for name in library.component_names:
    comp = library.components[name]
    print(f"     - {name}: {len(comp.bands)} bands")

# Generate spectra
generator = SyntheticNIRSGenerator(
    component_library=library,
    wavelength_start=1000,
    wavelength_end=2500,
    complexity="realistic",
    random_state=42,
)

X, C, *rest = generator.generate(n_samples=100)
print(f"\n   Generated: {X.shape[0]} spectra, {X.shape[1]} wavelengths")
print(f"   Concentrations: {C.shape}")

# =============================================================================
# Section 10: Visualization
# =============================================================================
print("\n" + "-" * 60)
print("Section 10: Visualization")
print("-" * 60)

# Save summary
summary_path = save_array_summary(
    {
        "X (spectra)": X,
        "C (concentrations)": C,
    },
    EXAMPLE_NAME
)
print_output_location(summary_path, "Data summary")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Overtone series
ax1 = axes[0, 0]
wavelengths = np.linspace(800, 3500, 1000)
for vib_name, color, label in [
    ("O-H_stretch_free", "b", "O-H"),
    ("C-H_stretch_CH3_asym", "r", "C-H"),
    ("N-H_stretch_primary", "g", "N-H"),
]:
    positions = []
    amplitudes = []
    for n in range(1, 5):
        result = calculate_overtone_position(vib_name, n)
        positions.append(result.wavelength_nm)
        amplitudes.append(result.amplitude_factor)
    ax1.stem(positions, amplitudes, linefmt=f'{color}-', markerfmt=f'{color}o', label=label)

ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Relative Amplitude")
ax1.set_title("Overtone Series (Fundamental to 3rd)")
ax1.set_xlim(800, 3500)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axvspan(900, 2500, alpha=0.1, color='yellow', label='NIR region')

# Plot 2: NIR zones
ax2 = axes[0, 1]
zone_colors_cmap = plt.cm.get_cmap('viridis')
zone_colors = [zone_colors_cmap(x) for x in np.linspace(0.2, 0.8, len(NIR_ZONES_WAVENUMBER))]
for y_pos, ((wn_min, wn_max, zone_name), color) in enumerate(zip(NIR_ZONES_WAVENUMBER, zone_colors, strict=False)):
    wl_start = wavenumber_to_wavelength(wn_max)
    wl_end = wavenumber_to_wavelength(wn_min)
    ax2.barh(y_pos, wl_end - wl_start, left=wl_start, height=0.8, color=color, alpha=0.7)
    ax2.text((wl_start + wl_end) / 2, y_pos, zone_name.replace('_', '\n'),
             ha='center', va='center', fontsize=8)

ax2.set_xlabel("Wavelength (nm)")
ax2.set_title("NIR Spectral Zones")
ax2.set_yticks([])
ax2.set_xlim(700, 2600)

# Plot 3: H-bonding effect
ax3 = axes[0, 2]
strengths = np.linspace(0, 1, 20)
free_oh = 3650
shifted = [apply_hydrogen_bonding_shift(free_oh, s) for s in strengths]
wavelengths_shifted = [wavenumber_to_wavelength(wn) for wn in shifted]
ax3.plot(strengths, wavelengths_shifted, 'b-', linewidth=2)
ax3.scatter([0, 0.5, 1],
           [wavenumber_to_wavelength(apply_hydrogen_bonding_shift(free_oh, s)) for s in [0, 0.5, 1]],
           s=100, c='red', zorder=5)
ax3.set_xlabel("H-bond Strength")
ax3.set_ylabel("O-H Stretch Wavelength (nm)")
ax3.set_title("H-bonding Effect on O-H Band")
ax3.grid(True, alpha=0.3)
ax3.annotate("Free O-H", (0.05, wavenumber_to_wavelength(free_oh)), fontsize=9)
ax3.annotate("Strong H-bond", (0.75, wavenumber_to_wavelength(apply_hydrogen_bonding_shift(free_oh, 1))), fontsize=9)

# Plot 4: Procedural component bands
ax4 = axes[1, 0]
band_centers = [b.center for b in component2.bands]
band_amplitudes = [b.amplitude for b in component2.bands]
ax4.stem(band_centers, band_amplitudes, linefmt='g-', markerfmt='go')
ax4.set_xlabel("Wavelength (nm)")
ax4.set_ylabel("Amplitude")
ax4.set_title(f"Procedural Component: {component2.name}")
ax4.set_xlim(900, 2600)
ax4.grid(True, alpha=0.3)

# Plot 5: Generated spectra
ax5 = axes[1, 1]
wavelengths_gen = np.linspace(1000, 2500, X.shape[1])
for i in range(min(20, X.shape[0])):
    ax5.plot(wavelengths_gen, X[i], alpha=0.4, linewidth=0.7)
ax5.set_xlabel("Wavelength (nm)")
ax5.set_ylabel("Absorbance")
ax5.set_title("Generated Spectra (Procedural + Predefined)")
ax5.grid(True, alpha=0.3)

# Plot 6: Concentration correlations
ax6 = axes[1, 2]
component_names = library.component_names[:3]  # First 3
for i, name in enumerate(component_names):
    ax6.scatter(C[:, i], C[:, (i+1) % len(component_names)],
               alpha=0.5, s=20, label=f"{name}")
ax6.set_xlabel("Component Concentration 1")
ax6.set_ylabel("Component Concentration 2")
ax6.set_title("Concentration Distribution")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
plot_path = get_example_output_path(EXAMPLE_NAME, "wavenumber_procedural_overview.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print_output_location(plot_path, "Overview plot")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Wavenumber Utilities (Phase 1):

  Conversions:
    wavenumber_to_wavelength(wn)    10⁷/wn → nm
    wavelength_to_wavenumber(wl)    10⁷/wl → cm⁻¹

  NIR Zones:
    NIR_ZONES_WAVENUMBER            7 zones with (start, end) cm⁻¹
    classify_wavelength_zone(wl)    Classify wavelength into zone

  Fundamentals:
    FUNDAMENTAL_VIBRATIONS          22 fundamental vibrations (cm⁻¹)

  Overtones:
    calculate_overtone_position(vib, n, anharmonicity)
      Returns: OvertoneResult(wavenumber_cm, wavelength_nm, amplitude_factor)
      Note: n=1 is fundamental, n=2 is 1st overtone

  Combinations:
    calculate_combination_band([vib1, vib2, ...])
      Returns: CombinationResult(wavenumber_cm, wavelength_nm, amplitude_factor)

  H-bonding:
    apply_hydrogen_bonding_shift(wn, strength)
      Shifts to lower wavenumber (longer wavelength)

Procedural Component Generation (Phase 1):

  Functional Groups:
    FunctionalGroupType enum        10 types (HYDROXYL, AMINE, METHYL, ...)
    FUNCTIONAL_GROUP_PROPERTIES     Physical properties for each type

  Configuration:
    ProceduralComponentConfig(
        n_fundamental_bands,        Number of fundamental vibrations
        include_overtones,          Generate overtone bands
        max_overtone_order,         Up to which overtone (2, 3, 4)
        include_combinations,       Generate combination bands
        h_bond_strength,            Average H-bonding (0-1)
        ...
    )

  Generation:
    generator = ProceduralComponentGenerator(random_state=42)
    component = generator.generate_component(
        name="my_compound",
        config=config,
        functional_groups=[FunctionalGroupType.HYDROXYL, ...]
    )

  Integration:
    library.add_component(component)
    generator = SyntheticNIRSGenerator(component_library=library, ...)
""")

if args.show:
    plt.show()
