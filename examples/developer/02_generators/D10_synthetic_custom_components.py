"""
D10 - Synthetic Generator: Custom Components
=============================================

Learn to extend the synthetic generator with custom spectral components.

This tutorial covers:

* Creating custom NIR bands and spectral components
* Building custom component libraries
* Using the core SyntheticNIRSGenerator directly
* Custom concentration distributions

Prerequisites
-------------
Complete user examples U09 and U10 first.

Duration: ~5 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# NIRS4All imports
from nirs4all.data.synthetic import (
    SyntheticNIRSGenerator,
    ComponentLibrary,
    SpectralComponent,
    NIRBand,
    get_predefined_components,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D10 Custom Components Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Understanding NIRBand
# =============================================================================
print("\n" + "=" * 60)
print("D10 - Custom Spectral Components")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Understanding NIRBand")
print("-" * 60)

# An NIRBand represents a single absorption band using Voigt profile
# (convolution of Gaussian and Lorentzian)

water_oh_band = NIRBand(
    center=1450,        # Band center in nm
    sigma=25,           # Gaussian width (FWHM component)
    gamma=3,            # Lorentzian width (affects tails)
    amplitude=0.8,      # Peak height
    name="O-H 1st overtone"
)

print(f"\n📊 NIRBand example:")
print(f"   Band: {water_oh_band.name}")
print(f"   Center: {water_oh_band.center} nm")
print(f"   Sigma (Gaussian): {water_oh_band.sigma}")
print(f"   Gamma (Lorentzian): {water_oh_band.gamma}")
print(f"   Amplitude: {water_oh_band.amplitude}")

# Compute the band shape
wavelengths = np.linspace(1300, 1600, 300)
# Note: In practice, the generator computes this internally
print(f"\n   Voigt profile combines Gaussian peak with Lorentzian tails")
print(f"   - Higher sigma = broader Gaussian core")
print(f"   - Higher gamma = heavier tails")


# =============================================================================
# Section 2: Creating Custom SpectralComponent
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Creating Custom SpectralComponent")
print("-" * 60)

# A SpectralComponent groups multiple bands belonging to one chemical species

# Example: Define a pharmaceutical compound (aspirin-like)
aspirin_like = SpectralComponent(
    name="aspirin_like",
    bands=[
        NIRBand(center=1520, sigma=15, gamma=1.5, amplitude=0.4, name="C-H aromatic"),
        NIRBand(center=1680, sigma=20, gamma=2.0, amplitude=0.5, name="C=O ester"),
        NIRBand(center=2020, sigma=25, gamma=2.5, amplitude=0.35, name="C-H combination"),
        NIRBand(center=2200, sigma=18, gamma=2.0, amplitude=0.3, name="O-H bend"),
    ],
    correlation_group=10  # Unique ID for correlation grouping
)

print(f"\n📊 Custom component: {aspirin_like.name}")
print(f"   Number of bands: {len(aspirin_like.bands)}")
print(f"   Correlation group: {aspirin_like.correlation_group}")
for band in aspirin_like.bands:
    print(f"   - {band.center} nm: {band.name}")


# =============================================================================
# Section 3: Exploring Predefined Components
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Predefined Components Reference")
print("-" * 60)

# Get all predefined components
predefined = get_predefined_components()

print(f"\n📊 Predefined components ({len(predefined)} total):")
for name, component in predefined.items():
    bands_info = ", ".join([f"{b.center}nm" for b in component.bands])
    print(f"\n   {name}:")
    print(f"   Bands: {bands_info}")


# =============================================================================
# Section 4: Building Custom Component Library
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Building Custom Component Library")
print("-" * 60)

# Create a library for pharmaceutical tablet analysis
pharma_library = ComponentLibrary(random_state=42)

# Add our custom compound
pharma_library.add_component(aspirin_like)

# Add predefined components as excipients - get from a predefined library
predefined = ComponentLibrary.from_predefined(["starch", "cellulose"])
for component in predefined.components.values():
    pharma_library.add_component(component)

# Add a random "unknown" impurity
pharma_library.add_random_component(
    name="impurity",
    n_bands=2,
    wavelength_range=(1400, 2200)
)

print(f"\n📊 Custom pharmaceutical library:")
print(f"   Components: {pharma_library.component_names}")
print(f"   Total bands: {sum(len(c.bands) for c in pharma_library.components.values())}")


# =============================================================================
# Section 5: Using Custom Library with Generator
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Generation with Custom Library")
print("-" * 60)

# Create generator with custom library
generator = SyntheticNIRSGenerator(
    wavelength_start=1100,
    wavelength_end=2400,
    wavelength_step=2,
    component_library=pharma_library,
    complexity="realistic",
    random_state=42,
)

# Generate data
X, C, E = generator.generate(n_samples=500)

print(f"\n📊 Generated data:")
print(f"   Spectra shape: {X.shape}")
print(f"   Concentrations shape: {C.shape}")
print(f"   Pure component spectra: {E.shape}")
print(f"\n   Component concentrations (first 5 samples):")
for i, name in enumerate(pharma_library.component_names):
    conc = C[:5, i]
    print(f"   {name}: {np.array2string(conc, precision=3)}")


# =============================================================================
# Section 6: Custom Concentration Distributions
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Custom Concentration Distributions")
print("-" * 60)

# Different concentration methods for different scenarios

# Dirichlet with custom alpha (controls concentration shape)
C_dirichlet = generator.generate_concentrations(
    n_samples=100,
    method="dirichlet",
    alpha=np.array([0.5, 2.0, 3.0, 0.3])  # Favor starch/cellulose
)
print(f"\n📊 Dirichlet (alpha=[0.5, 2.0, 3.0, 0.3]):")
print(f"   Mean concentrations: {C_dirichlet.mean(axis=0).round(3)}")

# Correlated concentrations
# Example: aspirin and impurity are inversely correlated
correlation_matrix = np.array([
    [1.0,  0.2, -0.1, -0.6],   # aspirin
    [0.2,  1.0,  0.8,  0.1],   # starch
    [-0.1, 0.8,  1.0,  0.2],   # cellulose (correlated with starch)
    [-0.6, 0.1,  0.2,  1.0],   # impurity (negatively correlated with aspirin)
])

C_correlated = generator.generate_concentrations(
    n_samples=1000,
    method="correlated",
    correlation_matrix=correlation_matrix,
)

# Verify correlation structure
actual_corr = np.corrcoef(C_correlated.T)
print(f"\n📊 Correlated concentrations:")
print(f"   Target aspirin-impurity correlation: -0.6")
print(f"   Actual aspirin-impurity correlation: {actual_corr[0, 3]:.3f}")
print(f"   Target starch-cellulose correlation: 0.8")
print(f"   Actual starch-cellulose correlation: {actual_corr[1, 2]:.3f}")


# =============================================================================
# Section 7: Visualization (if requested)
# =============================================================================
if args.plots:
    print("\n" + "-" * 60)
    print("Section 7: Visualization")
    print("-" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Pure component spectra
    ax1 = axes[0, 0]
    wavelengths = generator.wavelengths
    for i, name in enumerate(pharma_library.component_names):
        ax1.plot(wavelengths, E[i], label=name, linewidth=1.5)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Absorptivity (a.u.)")
    ax1.set_title("Pure Component Spectra")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Generated spectra colored by aspirin concentration
    ax2 = axes[0, 1]
    indices = np.random.choice(500, 100, replace=False)
    colors = C[indices, 0]  # Color by aspirin concentration
    for i, idx in enumerate(indices):
        ax2.plot(wavelengths, X[idx],
                color=plt.cm.viridis(colors[i] / colors.max()),
                alpha=0.5, linewidth=0.5)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Absorbance (a.u.)")
    ax2.set_title("Generated Spectra (colored by aspirin conc.)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Concentration distributions
    ax3 = axes[1, 0]
    for i, name in enumerate(pharma_library.component_names):
        ax3.hist(C[:, i], bins=30, alpha=0.5, label=name)
    ax3.set_xlabel("Concentration")
    ax3.set_ylabel("Count")
    ax3.set_title("Concentration Distributions")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Concentration correlations
    ax4 = axes[1, 1]
    ax4.scatter(C[:, 0], C[:, 3], alpha=0.3, s=10)
    ax4.set_xlabel("Aspirin Concentration")
    ax4.set_ylabel("Impurity Concentration")
    ax4.set_title(f"Aspirin vs Impurity (corr={actual_corr[0, 3]:.2f})")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("synthetic_custom_components.png", dpi=150, bbox_inches="tight")
    print("   Saved: synthetic_custom_components.png")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Custom Component Creation:

  NIRBand(center, sigma, gamma, amplitude, name)
      Single absorption band with Voigt profile
      - center: Peak position in nm
      - sigma: Gaussian width (core shape)
      - gamma: Lorentzian width (tail heaviness)
      - amplitude: Peak height

  SpectralComponent(name, bands, correlation_group)
      Chemical species with multiple absorption bands
      - bands: List of NIRBand objects
      - correlation_group: ID for inter-component correlation

  ComponentLibrary(random_state)
      Collection of spectral components
      - add_component(component): Add custom component
      - add_from_predefined(names): Add predefined components
      - add_random_component(name, n_bands): Add random bands

Concentration Methods:

  "dirichlet"    Sum-to-one constraint, alpha controls shape
  "uniform"      Independent uniform distributions
  "lognormal"    Skewed distributions (biological data)
  "correlated"   Specify inter-component correlations

Predefined Components:

  water, protein, lipid, starch, cellulose,
  chlorophyll, oil, nitrogen_compound

Use Cases for Custom Components:

  - Pharmaceutical tablet analysis
  - Food quality with specific ingredients
  - Polymer analysis with known additives
  - Petrochemical blends
  - Any application-specific spectral libraries
""")

if args.show:
    plt.show()
