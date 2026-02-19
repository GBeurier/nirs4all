"""
D01 - Transfer Analysis: Calibration Transfer Preprocessing Selection
======================================================================

TransferPreprocessingSelector automatically identifies the best
preprocessing for calibration transfer between instruments.

This tutorial covers:

* Calibration transfer problem
* TransferPreprocessingSelector usage
* Preprocessing comparison for transfer
* Best transfer strategy selection
* Visualization of transfer results

Prerequisites
-------------
- 01_quickstart/U02_basic_regression for pipeline basics
- Understanding of instrument variability

Next Steps
----------
See D02_retrain_modes for model retraining strategies.

Duration: ~5 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import Detrend, FirstDerivative, SavitzkyGolay, SecondDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D01 Transfer Analysis Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D01 - Transfer Analysis: Preprocessing for Calibration Transfer")
print("=" * 60)

print("""
Calibration Transfer Problem:
  - Train model on Instrument A (master)
  - Apply model to Instrument B (slave)
  - Spectral differences cause prediction errors

Solutions:
  1. Transfer preprocessing (standardize both instruments)
  2. Model adaptation (fine-tune on slave data)
  3. Domain adaptation (learn invariant features)

TransferPreprocessingSelector helps find option 1.
""")

# =============================================================================
# Section 1: The Transfer Problem
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: The Calibration Transfer Problem")
print("-" * 60)

print("""
Instrument differences cause spectral shifts:

    Master Instrument          Slave Instrument
    ┌──────────────────┐       ┌──────────────────┐
    │    Spectrum A    │       │    Spectrum A'   │
    │  ╭──────────╮    │   →   │  ╭──────────╮    │
    │  │          │    │       │  │    ↗     │    │  ← shifted!
    │  ╰──────────╯    │       │  ╰──────────╯    │
    └──────────────────┘       └──────────────────┘

Good preprocessing can reduce these differences.
""")

# Simulate transfer scenario with synthetic data
np.random.seed(42)
n_samples = 100
n_features = 200

# Master instrument data
X_master = np.random.randn(n_samples, n_features)
y = np.sum(X_master[:, :10], axis=1) + np.random.randn(n_samples) * 0.1

# Slave instrument: shifted baseline + noise
X_slave = X_master + 0.5 + np.random.randn(n_samples, n_features) * 0.2

print("Simulated data:")
print(f"  Master: {X_master.shape}")
print(f"  Slave:  {X_slave.shape} (with baseline shift)")

# =============================================================================
# Section 2: TransferPreprocessingSelector
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: TransferPreprocessingSelector")
print("-" * 60)

print("""
TransferPreprocessingSelector evaluates preprocessing options:

    from nirs4all.operators.transfer import TransferPreprocessingSelector

    selector = TransferPreprocessingSelector(
        preprocessings=[SNV, MSC, Detrend, FirstDerivative],
        model=PLSRegression(n_components=5),
        metric='rmse',
        cv=5
    )

    best_preproc = selector.fit(X_master, y, X_slave, y_slave)
""")

try:
    from nirs4all.operators.transfer import TransferPreprocessingSelector

    selector = TransferPreprocessingSelector(
        preprocessings=[
            None,  # No preprocessing
            SNV(),
            MSC(),
            Detrend(),
            FirstDerivative(),
            StandardScaler(),
        ],
        model=PLSRegression(n_components=5),
        metric='rmse'
    )

    print("TransferPreprocessingSelector created")
    print("  Comparing 6 preprocessing options")

except ImportError:
    print("TransferPreprocessingSelector not available")
    print("Demonstrating concept with manual comparison...")

# =============================================================================
# Section 3: Manual Transfer Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Manual Transfer Comparison")
print("-" * 60)

print("""
Compare preprocessing strategies for transfer:
""")

preprocessings = {
    'none': None,
    'snv': SNV(),
    'msc': MSC(),
    'detrend': Detrend(),
    '1st_deriv': FirstDerivative(),
    'standard': StandardScaler(),
}

from sklearn.metrics import mean_squared_error

results = {}
for name, preproc in preprocessings.items():
    # Apply preprocessing
    if preproc is None:
        X_train_pp = X_master.copy()
        X_test_pp = X_slave.copy()
    else:
        preproc_fit = preproc.fit(X_master)
        X_train_pp = preproc_fit.transform(X_master)
        X_test_pp = preproc_fit.transform(X_slave)

    # Train and predict
    model = PLSRegression(n_components=5)
    model.fit(X_train_pp, y)
    y_pred = model.predict(X_test_pp).ravel()

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    results[name] = rmse

print("\n📊 Transfer Performance (lower RMSE = better):")
print("-" * 40)
for name, rmse in sorted(results.items(), key=lambda x: x[1]):
    marker = "🏆" if rmse == min(results.values()) else "  "
    print(f"{marker} {name:<12}: RMSE = {rmse:.4f}")

best_preproc = min(results, key=results.get)
print(f"\n✓ Best for transfer: {best_preproc}")

# =============================================================================
# Section 4: Pipeline-Based Transfer Evaluation
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Pipeline-Based Transfer Evaluation")
print("-" * 60)

print("""
Use nirs4all pipeline for systematic evaluation:
""")

pipeline_transfer = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"_or_": [
        [],  # No preprocessing
        [SNV()],
        [MSC()],
        [Detrend()],
        [FirstDerivative()],
        [SNV(), FirstDerivative()],  # Combined
    ]},
    PLSRegression(n_components=5),
]

result = nirs4all.run(
    pipeline=pipeline_transfer,
    dataset="sample_data/regression",
    name="TransferEvaluation",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nEvaluated {result.num_predictions} preprocessing variants")

# =============================================================================
# Section 5: Preprocessing Robustness Metrics
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Preprocessing Robustness Metrics")
print("-" * 60)

print("""
Evaluate preprocessing robustness:

1. Same-instrument performance (baseline)
2. Cross-instrument performance (transfer)
3. Robustness ratio = transfer / baseline

Lower ratio = more robust preprocessing.
""")

# Compute robustness metrics
robustness = {}
for name, preproc in preprocessings.items():
    if preproc is None:
        X_pp = X_master.copy()
    else:
        preproc_fit = preproc.fit(X_master)
        X_pp = preproc_fit.transform(X_master)

    # Same-instrument CV score
    from sklearn.model_selection import cross_val_score
    model = PLSRegression(n_components=5)
    cv_scores = cross_val_score(model, X_pp, y, cv=3, scoring='neg_root_mean_squared_error')
    baseline = -cv_scores.mean()

    # Transfer score
    transfer = results[name]

    # Robustness ratio
    ratio = transfer / baseline if baseline > 0 else float('inf')
    robustness[name] = {'baseline': baseline, 'transfer': transfer, 'ratio': ratio}

print("\n📊 Robustness Analysis:")
print("-" * 55)
print(f"{'Preprocessing':<12} {'Baseline':<10} {'Transfer':<10} {'Ratio':<10}")
print("-" * 55)
for name, metrics in sorted(robustness.items(), key=lambda x: x[1]['ratio']):
    print(f"{name:<12} {metrics['baseline']:<10.4f} {metrics['transfer']:<10.4f} {metrics['ratio']:<10.2f}")

# =============================================================================
# Section 6: Visualization of Transfer Effects
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Visualization of Transfer Effects")
print("-" * 60)

print("""
Visualize spectral differences and preprocessing effects:
""")

if args.plots or args.show:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    # Raw spectra comparison
    ax = axes[0, 0]
    ax.plot(X_master[0], label='Master', alpha=0.7)
    ax.plot(X_slave[0], label='Slave', alpha=0.7)
    ax.set_title('Raw Spectra')
    ax.legend()

    # After SNV
    snv = SNV()
    X_master_snv = snv.fit_transform(X_master)
    X_slave_snv = snv.transform(X_slave)
    ax = axes[0, 1]
    ax.plot(X_master_snv[0], label='Master', alpha=0.7)
    ax.plot(X_slave_snv[0], label='Slave', alpha=0.7)
    ax.set_title('After SNV')
    ax.legend()

    # After First Derivative
    deriv = FirstDerivative()
    X_master_deriv = deriv.fit_transform(X_master)
    X_slave_deriv = deriv.transform(X_slave)
    ax = axes[0, 2]
    ax.plot(X_master_deriv[0], label='Master', alpha=0.7)
    ax.plot(X_slave_deriv[0], label='Slave', alpha=0.7)
    ax.set_title('After 1st Derivative')
    ax.legend()

    # Performance comparison bar chart
    ax = axes[1, 0]
    names = list(results.keys())
    rmses = [results[n] for n in names]
    colors = ['green' if n == best_preproc else 'steelblue' for n in names]
    ax.barh(names, rmses, color=colors)
    ax.set_xlabel('RMSE (Transfer)')
    ax.set_title('Transfer Performance')

    # Robustness ratio
    ax = axes[1, 1]
    ratios = [robustness[n]['ratio'] for n in names]
    ax.barh(names, ratios, color='coral')
    ax.set_xlabel('Degradation Ratio')
    ax.set_title('Robustness (lower = better)')
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)

    # Hide last subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('transfer_analysis.png', dpi=100)
    print("Saved: transfer_analysis.png")

# =============================================================================
# Section 7: Transfer Strategies
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Transfer Strategies")
print("-" * 60)

print("""
📋 Calibration Transfer Strategies:

┌─────────────────────┬───────────────────────────────────────┐
│ Strategy            │ When to Use                           │
├─────────────────────┼───────────────────────────────────────┤
│ Preprocessing Only  │ Small instrument differences          │
│ (SNV, MSC, etc.)    │ No labeled data on slave              │
├─────────────────────┼───────────────────────────────────────┤
│ Slope/Bias Correct  │ Linear offset between instruments     │
│                     │ Few labeled samples on slave          │
├─────────────────────┼───────────────────────────────────────┤
│ Fine-tuning         │ Moderate differences                  │
│                     │ 10-50 labeled samples on slave        │
├─────────────────────┼───────────────────────────────────────┤
│ Full Retrain        │ Large differences                     │
│                     │ Sufficient labeled data on slave      │
└─────────────────────┴───────────────────────────────────────┘
""")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
What we learned:
1. Calibration transfer is a common NIRS challenge
2. Preprocessing can reduce instrument differences
3. TransferPreprocessingSelector automates comparison
4. Robustness ratio measures preprocessing stability
5. Best preprocessing for this data: {best_preproc}

Key metrics:
- Transfer RMSE: Lower is better
- Robustness ratio: Closer to 1 is better
- Baseline vs Transfer gap

Preprocessing ranking for transfer:
{', '.join(sorted(results.keys(), key=lambda x: results[x]))}

Next: D02_retrain_modes.py - Model retraining strategies
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
