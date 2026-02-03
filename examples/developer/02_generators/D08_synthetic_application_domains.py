"""
D08 - Synthetic Generator: Application Domains
===============================================

Learn to use application domain priors for domain-aware synthetic data generation.

This tutorial covers:

* Domain categories and available domains
* Getting domain configurations
* Concentration priors
* Domain-aware library creation
* Using domains with generators
* Cross-domain comparison

Prerequisites
-------------
Complete user examples U05, U06, and developer examples D05-D07 first.

Duration: ~5 minutes
Difficulty: ★★★★☆

Phase 1 Feature
---------------
This example demonstrates Phase 1 features from the Synthetic Generator
Enhancement Roadmap: application domain priors with 20 predefined domains.
"""

# Standard library imports
import argparse
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# NIRS4All imports
import nirs4all
from nirs4all.synthesis import (
    # Domain utilities
    DomainCategory,
    get_domain_config,
    list_domains,
    get_domain_components,
    get_domains_for_component,
    create_domain_aware_library,
    APPLICATION_DOMAINS,
    # Generator
    ComponentLibrary,
    SyntheticNIRSGenerator,
    SyntheticDatasetBuilder,
)

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from example_utils import get_example_output_path, print_output_location, save_array_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D08 Application Domains Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "D08_synthetic_application_domains"


# =============================================================================
# Section 1: Domain Categories
# =============================================================================
print("\n" + "=" * 60)
print("D08 - Application Domains")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Domain Categories")
print("-" * 60)

print("\n📊 Available Domain Categories:")
for category in DomainCategory:
    domains_in_cat = list_domains(category=category)
    print(f"   {category.name:<15} ({len(domains_in_cat)} domains)")
    for domain in domains_in_cat:
        print(f"      - {domain}")


# =============================================================================
# Section 2: Listing Domains
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Listing All Domains")
print("-" * 60)

all_domains = list_domains()
print(f"\n📊 Total domains available: {len(all_domains)}")
print()

# Group by category
for category in DomainCategory:
    domains = list_domains(category=category)
    if domains:
        print(f"   {category.name}:")
        for domain in domains:
            config = get_domain_config(domain)
            print(f"      {domain:<25} - {config.description[:40]}...")


# =============================================================================
# Section 3: Domain Configuration Details
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Domain Configuration Details")
print("-" * 60)

# Get grain domain
grain = get_domain_config("agriculture_grain")

print(f"\n📊 Domain: {grain.name}")
print(f"   Category: {grain.category.name}")
print(f"   Description: {grain.description}")
print()
print(f"   Wavelength range: {grain.wavelength_range[0]}-{grain.wavelength_range[1]} nm")
print(f"   Measurement mode: {grain.measurement_mode}")
print(f"   Complexity: {grain.complexity}")
print(f"   Noise level: {grain.noise_level}")
print()
print(f"   Typical components ({len(grain.typical_components)}):")
for comp in grain.typical_components[:8]:
    weight = grain.component_weights.get(comp, 1.0) if grain.component_weights else 1.0
    print(f"      - {comp} (weight: {weight:.1f})")
if len(grain.typical_components) > 8:
    print(f"      ... and {len(grain.typical_components) - 8} more")

print()
print(f"   Sample types: {', '.join(grain.typical_sample_types or [])}")


# =============================================================================
# Section 4: Comparing Domains
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Comparing Domain Configurations")
print("-" * 60)

print("\n📊 Domain Comparison Table:")
print(f"   {'Domain':<25} {'Category':<15} {'Components':<12} {'Wavelength':<15} {'Mode'}")
print(f"   {'-'*25} {'-'*15} {'-'*12} {'-'*15} {'-'*15}")

compare_domains = [
    "agriculture_grain", "food_dairy", "pharma_tablets",
    "petrochem_fuels", "environmental_water", "biomedical_tissue"
]

for domain_name in compare_domains:
    config = get_domain_config(domain_name)
    wl_range = f"{config.wavelength_range[0]}-{config.wavelength_range[1]}"
    print(f"   {domain_name:<25} {config.category.name:<15} {len(config.typical_components):<12} {wl_range:<15} {config.measurement_mode}")


# =============================================================================
# Section 5: Finding Components
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Component-Domain Relationships")
print("-" * 60)

# Get components for a domain
print("\n📊 Components for specific domains:")
for domain_name in ["agriculture_grain", "food_dairy", "pharma_tablets"]:
    components = get_domain_components(domain_name)
    print(f"   {domain_name}:")
    print(f"      {', '.join(components[:6])}")
    if len(components) > 6:
        print(f"      ... and {len(components) - 6} more")

# Find domains that use specific components
print("\n📊 Domains using specific components:")
for component in ["protein", "water", "starch", "lipid"]:
    domains = get_domains_for_component(component)
    if domains:
        print(f"   {component}: {', '.join(domains)}")


# =============================================================================
# Section 6: Domain-Aware Library Creation
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Domain-Aware Library Creation")
print("-" * 60)

# Create libraries for different domains
print("\n📊 Creating domain-aware libraries:")
print("   Note: create_domain_aware_library returns (component_names, concentrations)")

for domain_name in ["agriculture_grain", "food_dairy", "petrochem_fuels"]:
    component_names, concentrations = create_domain_aware_library(domain_name, n_samples=50)
    print(f"   {domain_name}:")
    print(f"      Components: {len(component_names)}")
    print(f"      Names: {', '.join(component_names[:5])}")
    if len(component_names) > 5:
        print(f"            ... and {len(component_names) - 5} more")
    print(f"      Concentrations shape: {concentrations.shape}")


# =============================================================================
# Section 7: Using Domains with Generator
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Domain-Aware Data Generation")
print("-" * 60)

# Generate data for grain analysis using domain configuration
grain_domain = get_domain_config("agriculture_grain")

# Create ComponentLibrary from domain's typical components
grain_library = ComponentLibrary.from_predefined(grain_domain.typical_components[:5])

generator = SyntheticNIRSGenerator(
    component_library=grain_library,
    wavelength_start=grain_domain.wavelength_range[0],
    wavelength_end=grain_domain.wavelength_range[1],
    complexity="realistic",
    random_state=42,
)

X_grain, C_grain, *rest_grain = generator.generate(n_samples=200)

print(f"\n📊 Generated grain analysis dataset:")
print(f"   Samples: {X_grain.shape[0]}")
print(f"   Wavelengths: {X_grain.shape[1]}")
print(f"   Components: {C_grain.shape[1]}")
print(f"   Component names: {grain_library.component_names}")

# Generate dairy data for comparison
dairy_domain = get_domain_config("food_dairy")
dairy_library = ComponentLibrary.from_predefined(dairy_domain.typical_components[:5])

generator_dairy = SyntheticNIRSGenerator(
    component_library=dairy_library,
    wavelength_start=dairy_domain.wavelength_range[0],
    wavelength_end=dairy_domain.wavelength_range[1],
    complexity="realistic",
    random_state=42,
)

X_dairy, C_dairy, *rest_dairy = generator_dairy.generate(n_samples=200)

print(f"\n📊 Generated dairy analysis dataset:")
print(f"   Samples: {X_dairy.shape[0]}")
print(f"   Wavelengths: {X_dairy.shape[1]}")
print(f"   Components: {C_dairy.shape[1]}")
print(f"   Component names: {dairy_library.component_names}")


# =============================================================================
# Section 8: Concentration Priors
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Concentration Priors")
print("-" * 60)

print("\n📊 Concentration priors define realistic ranges for components")
print()

# Check priors for grain domain
if hasattr(grain, 'concentration_priors') and grain.concentration_priors:
    print(f"   Grain domain priors:")
    for comp_name, prior in grain.concentration_priors.items():
        print(f"      {comp_name}: {prior}")
else:
    print("   Grain domain: Using default uniform priors")

# Demonstrate concentration ranges
print("\n📊 Typical concentration ranges by domain:")
print(f"   {'Domain':<20} {'Component':<15} {'Typical Range'}")
print(f"   {'-'*20} {'-'*15} {'-'*20}")

domain_ranges = [
    ("agriculture_grain", "starch", "50-80%"),
    ("agriculture_grain", "protein", "8-15%"),
    ("agriculture_grain", "moisture", "10-14%"),
    ("food_dairy", "fat", "0-40%"),
    ("food_dairy", "protein", "3-5%"),
    ("pharma_tablets", "api", "1-90%"),
    ("petrochem_fuels", "octane", "87-93"),
]

for domain, comp, range_str in domain_ranges:
    print(f"   {domain:<20} {comp:<15} {range_str}")


# =============================================================================
# Section 9: Extended Component Library
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Extended Component Library (111 Components)")
print("-" * 60)

from nirs4all.synthesis._constants import get_predefined_components

components = get_predefined_components()
print(f"\n📊 Total predefined components: {len(components)}")

# Categorize by type
categories = {
    "Carbohydrates": ["starch", "cellulose", "glucose", "maltose", "sucrose", "fructose"],
    "Proteins": ["protein", "casein", "gluten", "albumin", "collagen"],
    "Lipids": ["lipid", "oil", "oleic_acid", "linoleic_acid"],
    "Alcohols": ["ethanol", "methanol", "glycerol"],
    "Polymers": ["polyethylene", "polystyrene", "pmma", "pet"],
    "Pharmaceuticals": ["aspirin", "paracetamol", "ibuprofen"],
    "Pigments": ["chlorophyll", "carotenoid", "anthocyanin"],
}

print("\n   Component categories:")
for cat_name, examples in categories.items():
    available = [c for c in examples if c in components]
    print(f"   {cat_name}:")
    print(f"      {', '.join(available)}")


# =============================================================================
# Section 10: Cross-Domain Analysis
# =============================================================================
print("\n" + "-" * 60)
print("Section 10: Cross-Domain Analysis")
print("-" * 60)

# Generate datasets for multiple domains
domain_datasets = {}

for domain_name in ["agriculture_grain", "food_dairy", "petrochem_fuels"]:
    domain_config = get_domain_config(domain_name)

    # Create ComponentLibrary from domain's typical components
    library = ComponentLibrary.from_predefined(domain_config.typical_components[:5])

    generator = SyntheticNIRSGenerator(
        component_library=library,
        wavelength_start=domain_config.wavelength_range[0],
        wavelength_end=domain_config.wavelength_range[1],
        complexity="realistic",
        random_state=42,
    )

    X, C, *rest = generator.generate(n_samples=100)
    domain_datasets[domain_name] = {
        'X': X, 'C': C,
        'wavelengths': np.linspace(
            domain_config.wavelength_range[0],
            domain_config.wavelength_range[1],
            X.shape[1]
        ),
        'components': library.component_names,
    }

print(f"\n📊 Generated datasets for {len(domain_datasets)} domains:")
for domain_name, data in domain_datasets.items():
    print(f"   {domain_name}:")
    print(f"      Shape: {data['X'].shape}")
    print(f"      Wavelength: {data['wavelengths'][0]:.0f}-{data['wavelengths'][-1]:.0f} nm")


# =============================================================================
# Section 11: Visualization
# =============================================================================
print("\n" + "-" * 60)
print("Section 11: Visualization")
print("-" * 60)

# Save summary
summary_path = save_array_summary(
    {
        "X_grain (grain spectra)": X_grain,
        "C_grain (grain concentrations)": C_grain,
        "X_dairy (dairy spectra)": X_dairy,
        "C_dairy (dairy concentrations)": C_dairy,
    },
    EXAMPLE_NAME
)
print_output_location(summary_path, "Data summary")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Domain comparison - spectra
ax1 = axes[0, 0]
colors = ['blue', 'green', 'red']
for i, (domain_name, data) in enumerate(domain_datasets.items()):
    mean_spec = data['X'].mean(axis=0)
    ax1.plot(data['wavelengths'], mean_spec, color=colors[i],
             label=domain_name.split('_')[0].title(), linewidth=1.5)

ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Mean Absorbance")
ax1.set_title("Cross-Domain Mean Spectra")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Domain wavelength ranges
ax2 = axes[0, 1]
domain_list = ["agriculture_grain", "agriculture_forage", "food_dairy",
               "food_meat", "pharma_tablets", "petrochem_fuels",
               "environmental_water", "biomedical_tissue"]
y_pos = np.arange(len(domain_list))
for i, domain_name in enumerate(domain_list):
    try:
        config = get_domain_config(domain_name)
        wl_start, wl_end = config.wavelength_range
        ax2.barh(i, wl_end - wl_start, left=wl_start, height=0.6, alpha=0.7)
        ax2.text(wl_start + 10, i, domain_name.replace('_', ' '),
                 va='center', ha='left', fontsize=8)
    except Exception:
        pass

ax2.set_xlabel("Wavelength (nm)")
ax2.set_title("Domain Wavelength Ranges")
ax2.set_yticks([])
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Grain spectra
ax3 = axes[0, 2]
wavelengths_grain = np.linspace(grain_domain.wavelength_range[0], grain_domain.wavelength_range[1], X_grain.shape[1])
for i in range(min(30, X_grain.shape[0])):
    ax3.plot(wavelengths_grain, X_grain[i], alpha=0.3, linewidth=0.5)
ax3.set_xlabel("Wavelength (nm)")
ax3.set_ylabel("Absorbance")
ax3.set_title("Grain Domain Spectra")
ax3.grid(True, alpha=0.3)

# Plot 4: Dairy spectra
ax4 = axes[1, 0]
wavelengths_dairy = np.linspace(dairy_domain.wavelength_range[0], dairy_domain.wavelength_range[1], X_dairy.shape[1])
for i in range(min(30, X_dairy.shape[0])):
    ax4.plot(wavelengths_dairy, X_dairy[i], alpha=0.3, linewidth=0.5, color='green')
ax4.set_xlabel("Wavelength (nm)")
ax4.set_ylabel("Absorbance")
ax4.set_title("Dairy Domain Spectra")
ax4.grid(True, alpha=0.3)

# Plot 5: Component count by domain
ax5 = axes[1, 1]
domain_names = list(APPLICATION_DOMAINS.keys())[:10]
component_counts = []
for d in domain_names:
    try:
        config = get_domain_config(d)
        component_counts.append(len(config.typical_components))
    except Exception:
        component_counts.append(0)

bars = ax5.bar(range(len(domain_names)), component_counts, color='steelblue', alpha=0.7)
ax5.set_xticks(range(len(domain_names)))
ax5.set_xticklabels([d.replace('_', '\n') for d in domain_names], rotation=0, fontsize=7)
ax5.set_ylabel("Number of Components")
ax5.set_title("Components per Domain")
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Categories pie chart
ax6 = axes[1, 2]
category_counts = {}
for cat in DomainCategory:
    count = len(list_domains(category=cat))
    if count > 0:
        category_counts[cat.name] = count

ax6.pie(category_counts.values(), labels=category_counts.keys(),
        autopct='%1.0f%%', startangle=90)
ax6.set_title("Domains by Category")

plt.tight_layout()

# Save plot
plot_path = get_example_output_path(EXAMPLE_NAME, "application_domains_overview.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print_output_location(plot_path, "Overview plot")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Application Domains (Phase 1):

  Domain Categories (8):
    DomainCategory enum             AGRICULTURE, FOOD, PHARMACEUTICAL,
                                    PETROCHEMICAL, ENVIRONMENTAL,
                                    BIOMEDICAL, INDUSTRIAL, RESEARCH

  Domain Listing:
    list_domains()                  All 20 domains
    list_domains(category=...)      Filter by category

  Domain Configuration:
    get_domain_config(name)         Full DomainConfig object

    DomainConfig fields:
      .name                         Display name
      .category                     DomainCategory enum
      .description                  Detailed description
      .typical_components           List of component names
      .component_weights            Relative importance
      .concentration_priors         Statistical priors
      .wavelength_range             (start, end) nm
      .measurement_mode             "reflectance", "transmission"
      .noise_level                  "low", "medium", "high"
      .complexity                   "simple", "realistic", "complex"

  Component Queries:
    get_domain_components(domain)   Components for a domain
    get_domains_for_component(comp) Domains using a component

  Domain-Aware Library:
    create_domain_aware_library(domain_name)
      Returns ComponentLibrary with domain-typical components

  Integration with Generator:
    domain = get_domain_config("agriculture_grain")
    library = create_domain_aware_library("agriculture_grain")
    generator = SyntheticNIRSGenerator(
        component_library=library,
        wavelength_start=domain.wavelength_range[0],
        wavelength_end=domain.wavelength_range[1],
        complexity=domain.complexity,
        ...
    )

Extended Component Library (Phase 1):

  111 predefined components covering:
    - Carbohydrates (15): starch, cellulose, glucose, maltose, ...
    - Proteins (10): protein, casein, gluten, albumin, ...
    - Lipids (12): lipid, oil, oleic_acid, linoleic_acid, ...
    - Alcohols (8): ethanol, methanol, glycerol, ...
    - Pharmaceuticals (10): aspirin, paracetamol, ibuprofen, ...
    - Polymers (8): polyethylene, polystyrene, pmma, ...
    - Minerals (5): kaolinite, montmorillonite, silica, ...
    - Pigments (7): chlorophyll, carotenoid, anthocyanin, ...
""")

if args.show:
    plt.show()
