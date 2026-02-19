"""
Orthogonal Signal Correction (OSC) and External Parameter Orthogonalization (EPO)

This example demonstrates advanced preprocessing methods for removing unwanted
variation from spectral data:

1. OSC - Removes variation orthogonal to the target variable (Y-orthogonal variation)
2. EPO - Removes variation correlated with external parameters (temperature, batch, etc.)

These methods improve model interpretability and can enhance prediction performance
by focusing on relevant spectral variation.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.transforms import EPO, OSC

print("=" * 80)
print("Orthogonal Signal Correction (OSC) and EPO Examples")
print("=" * 80)

# ==============================================================================
# Example 1: Basic OSC Usage
# ==============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic OSC Usage")
print("=" * 80)

# Create synthetic data with Y-related and Y-orthogonal components
np.random.seed(42)
n_samples, n_features = 200, 100

# Generate target variable
y = np.random.randn(n_samples)

# Create spectral data:
# - Y-related component (should be preserved)
# - Y-orthogonal component (should be removed by OSC)
X_related = np.outer(y, np.random.randn(n_features)) * 2.0
X_orthogonal = np.random.randn(n_samples, n_features) * 1.5
X = X_related + X_orthogonal + np.random.randn(n_samples, n_features) * 0.2

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")

# Apply OSC with 2 orthogonal components
print("\nApplying OSC(n_components=2)...")
osc = OSC(n_components=2, scale=True)
X_filtered = osc.fit_transform(X, y)

print(f"Orthogonal weights shape: {osc.W_ortho_.shape}")
print(f"Orthogonal loadings shape: {osc.P_ortho_.shape}")
print(f"Actual components extracted: {osc.n_components_}")

# Measure Y-correlation before and after
corr_before = np.corrcoef(X.mean(axis=1), y)[0, 1]
corr_after = np.corrcoef(X_filtered.mean(axis=1), y)[0, 1]

print("\nY-correlation (average features):")
print(f"  Before OSC: {corr_before:.4f}")
print(f"  After OSC:  {corr_after:.4f}")
print(f"  Change:     {corr_after - corr_before:+.4f}")

# Measure data change
difference = np.linalg.norm(X - X_filtered) / np.linalg.norm(X)
print(f"\nRelative change in data: {difference:.2%}")

# ==============================================================================
# Example 2: OSC in Pipeline with PLS
# ==============================================================================
print("\n" + "=" * 80)
print("Example 2: OSC in Pipeline with PLS Regression")
print("=" * 80)

# Compare PLS performance with and without OSC
from sklearn.pipeline import Pipeline

# Pipeline without OSC
pipeline_no_osc = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression(n_components=10))])

# Pipeline with OSC
pipeline_with_osc = Pipeline([("osc", OSC(n_components=2)), ("scaler", StandardScaler()), ("pls", PLSRegression(n_components=10))])

# Cross-validation
print("\nCross-validation (5-fold, negative MSE):")

scores_no_osc = cross_val_score(pipeline_no_osc, X, y, cv=5, scoring="neg_mean_squared_error")
rmse_no_osc = np.sqrt(-scores_no_osc.mean())

scores_with_osc = cross_val_score(pipeline_with_osc, X, y, cv=5, scoring="neg_mean_squared_error")
rmse_with_osc = np.sqrt(-scores_with_osc.mean())

print(f"  Without OSC: RMSE = {rmse_no_osc:.4f} ± {scores_no_osc.std():.4f}")
print(f"  With OSC:    RMSE = {rmse_with_osc:.4f} ± {scores_with_osc.std():.4f}")
print(f"  Improvement: {(1 - rmse_with_osc / rmse_no_osc) * 100:+.1f}%")

# ==============================================================================
# Example 3: EPO for External Parameter Correction
# ==============================================================================
print("\n" + "=" * 80)
print("Example 3: EPO for External Parameter Correction")
print("=" * 80)

# Create synthetic data with temperature effect
np.random.seed(123)
n_samples = 200
n_features = 100

# External parameter: temperature (ranging from 15°C to 30°C)
temperature = 15 + 15 * np.random.rand(n_samples)

# True property of interest (independent of temperature)
true_property = 5 + 2 * np.sin(np.linspace(0, 4 * np.pi, n_samples))

# Create spectra:
# - Chemical component (related to true property)
# - Temperature component (unwanted interference)
X_chemical = np.outer(true_property, np.ones(n_features)) + np.random.randn(n_samples, n_features) * 0.3

# Temperature effect: shifts baseline and adds systematic bias
temp_effect = np.outer(temperature - temperature.mean(), np.ones(n_features)) * 0.5
X = X_chemical + temp_effect + np.random.randn(n_samples, n_features) * 0.2

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Temperature range: [{temperature.min():.1f}°C, {temperature.max():.1f}°C]")
print(f"True property range: [{true_property.min():.2f}, {true_property.max():.2f}]")

# Measure correlation with temperature before EPO
temp_corr_before = np.abs(np.corrcoef(X[:, 0], temperature)[0, 1])
print(f"\nCorrelation with temperature (first feature): {temp_corr_before:.4f}")

# Apply EPO to remove temperature effect
print("\nApplying EPO to remove temperature effect...")
epo = EPO(scale=True)
X_epo_filtered = epo.fit_transform(X, temperature)

# Measure correlation after EPO
temp_corr_after = np.abs(np.corrcoef(X_epo_filtered[:, 0], temperature)[0, 1])
print(f"Correlation with temperature after EPO: {temp_corr_after:.4f}")
print(f"Reduction: {(1 - temp_corr_after / temp_corr_before) * 100:.1f}%")

# Verify that true property correlation is preserved
prop_corr_before = np.corrcoef(X.mean(axis=1), true_property)[0, 1]
prop_corr_after = np.corrcoef(X_epo_filtered.mean(axis=1), true_property)[0, 1]

print("\nCorrelation with true property:")
print(f"  Before EPO: {prop_corr_before:.4f}")
print(f"  After EPO:  {prop_corr_after:.4f}")
print("  (Should be preserved)")

# ==============================================================================
# Example 4: Sequential EPO + OSC
# ==============================================================================
print("\n" + "=" * 80)
print("Example 4: Sequential EPO and OSC")
print("=" * 80)
print("\nCombining EPO (external parameter removal) and OSC (Y-orthogonal removal)")

# First apply EPO to remove temperature
epo = EPO()
X_epo = epo.fit_transform(X, temperature)

# Then apply OSC to remove remaining Y-orthogonal variation
osc = OSC(n_components=2)
X_final = osc.fit_transform(X_epo, true_property)

print("\nData transformation:")
print(f"  Original:           {np.linalg.norm(X):.2f}")
print(f"  After EPO:          {np.linalg.norm(X_epo):.2f}")
print(f"  After EPO + OSC:    {np.linalg.norm(X_final):.2f}")

# Measure correlations
temp_corr_final = np.abs(np.corrcoef(X_final[:, 0], temperature)[0, 1])
prop_corr_final = np.corrcoef(X_final.mean(axis=1), true_property)[0, 1]

print("\nFinal correlations:")
print(f"  With temperature: {temp_corr_final:.4f} (removed by EPO)")
print(f"  With property:    {prop_corr_final:.4f} (preserved)")

# ==============================================================================
# Example 5: Comparing Different Numbers of OSC Components
# ==============================================================================
print("\n" + "=" * 80)
print("Example 5: Effect of Number of OSC Components")
print("=" * 80)

# Test different numbers of components
component_range = [1, 2, 3, 5, 10]
print("\nTesting OSC with different numbers of components:")

for n_comp in component_range:
    osc = OSC(n_components=n_comp)
    X_osc = osc.fit_transform(X, true_property)

    # Measure Y-correlation preservation
    corr = np.corrcoef(X_osc.mean(axis=1), true_property)[0, 1]
    corr_orig = np.corrcoef(X.mean(axis=1), true_property)[0, 1]

    # Measure data change
    change = np.linalg.norm(X - X_osc) / np.linalg.norm(X)

    print(f"  n_components={n_comp:2d}: correlation={corr:.4f} (original={corr_orig:.4f}), change={change:.2%}")

print("\nNote: Too many components can remove Y-relevant information!")

# ==============================================================================
# Summary and Recommendations
# ==============================================================================
print("\n" + "=" * 80)
print("Summary and Recommendations")
print("=" * 80)

print("""
OSC (Orthogonal Signal Correction):
- Use when: You have systematic variation unrelated to target variable
- Benefits: Improves model interpretability, can enhance prediction
- Caution: Removing too many components may remove Y-relevant information
- Typical: 1-3 components
- Selection: Use cross-validation to select optimal n_components

EPO (External Parameter Orthogonalization):
- Use when: You know specific interference sources (temperature, batch, etc.)
- Benefits: Removes known systematic effects without using Y
- Advantage: Preserves all Y-relevant information (even if correlated with external parameter)
- Requirement: Need external parameter measurements during calibration

Combined EPO + OSC:
- Use EPO first to remove known effects
- Then use OSC to remove remaining Y-orthogonal variation
- Provides comprehensive preprocessing pipeline

Integration with nirs4all:
- Both OSC and EPO work seamlessly in sklearn pipelines
- Compatible with nirs4all.run() for automatic pipeline evaluation
- Support step caching for efficient computation with generators
""")

print("\nDone!")
