"""
U06 - TabPFN for NIRS: a fixed recipe without HPO
==================================================

Use the TabPFN tabular foundation model on NIRS spectra with a fixed
preprocessing chain (no per-dataset HPO).

This tutorial covers:

* What :class:`TabPFNNIRSRegressor` does (recipe + rationale)
* When to use it vs PLS / AOM-PLS / CNN models
* Quick fit/predict on the bundled sample dataset
* How to tweak ``n_estimators``, ``max_features``, and other knobs

The recipe at a glance
----------------------
``SavitzkyGolay(11, 2, 1)`` ▶ ``OSC()`` ▶ centering ▶
``TabPFNRegressor(n_estimators=16)``

It was selected by extracting the most-frequent winning preprocessing
motif from the TabPFN paper logs (deriv-1 SG with OSC correction) and
empirically validating ``n_estimators=16`` on a 57-dataset NIRS cohort.
On that cohort, the median test-set RMSE lands at +2.5% versus the
TabPFN paper's 72-chain per-dataset HPO baseline ``TabPFN_opt`` -- at a
single TabPFN training per dataset.

Prerequisites
-------------
- ``pip install tabpfn`` (optional dependency)
- A working GPU is recommended for n_train > 200 (TabPFN runs on CPU but
  much faster on CUDA / MPS).

Next Steps
----------
For HPO across multiple chains (PLS variants, AOM, etc.), see
:ref:`U02_hyperparameter_tuning`. For ensemble / stacking, see
:ref:`U03_stacking_ensembles`.

Duration: ~3 minutes
Difficulty: ★★☆☆☆
"""

# Standard library
import argparse

# Third-party
import numpy as np
from sklearn.model_selection import ShuffleSplit

# nirs4all
import nirs4all
from nirs4all.operators.models import TabPFNNIRSRegressor

parser = argparse.ArgumentParser(description="U06 TabPFN NIRS recipe")
parser.add_argument("--plots", action="store_true", help="Generate plots")
parser.add_argument("--show", action="store_true", help="Display plots interactively")
args = parser.parse_args()


# =============================================================================
# Section 1: Why TabPFNNIRSRegressor?
# =============================================================================

print("\n" + "=" * 60)
print("U06 - TabPFN for NIRS")
print("=" * 60)
print("""
TabPFN is a transformer-based tabular foundation model that achieves
state-of-the-art on many small-to-medium tabular regression / classification
tasks without per-dataset hyperparameter tuning.

On NIRS spectra, the choice of preprocessing matters more than for tabular
data. The TabPFN paper runs an HPO sweep over 72 preprocessing chains per
dataset and keeps the best (``TabPFN_opt``); this is expensive.

``TabPFNNIRSRegressor`` instead uses ONE fixed preprocessing recipe selected
empirically. It matches ``TabPFN_opt`` within ~2.5% RMSE in median on a
57-dataset NIRS cohort, at a fraction of the cost (no HPO, one TabPFN fit).
""")


# =============================================================================
# Section 2: The recipe under the hood
# =============================================================================

print("-" * 60)
print("Section 2: Fixed recipe (no HPO)")
print("-" * 60)

est = TabPFNNIRSRegressor()
print("Default hyperparameters:")
for k, v in est.get_params().items():
    print(f"  {k:22s} = {v}")
print("""
Internally the estimator applies in sequence:
  1. SavitzkyGolay(window_length=11, polyorder=2, deriv=1)
  2. OSC(n_components=1)
  3. linspace subsample to <= 2000 columns (TabPFN v3 hard limit)
  4. center (with_std=False)
  5. TabPFNRegressor(n_estimators=16, ignore_pretraining_limits=True)
""")


# =============================================================================
# Section 3: Run on the bundled sample dataset via nirs4all.run()
# =============================================================================

print("-" * 60)
print("Section 3: Pipeline integration")
print("-" * 60)

pipeline = [
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    {"model": TabPFNNIRSRegressor(n_estimators=8)},  # n_est=8 for example speed
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="examples/sample_data/regression",
    verbose=0,
    random_state=42,
    name="U06_TabPFN_NIRS",
)
print(f"  best RMSE = {result.best_rmse:.4f}")
print(f"  best R^2  = {result.best_r2:.4f}")


# =============================================================================
# Section 4: Direct sklearn-style usage
# =============================================================================

print("\n" + "-" * 60)
print("Section 4: Direct sklearn-style fit/predict")
print("-" * 60)

# Synthetic NIRS-like spectra for a self-contained example
rng = np.random.default_rng(42)
n_train, n_test, p = 150, 60, 600
wavelengths = np.linspace(0.0, 1.0, p)
baseline = np.exp(-((wavelengths - 0.5) ** 2) / 0.01)


def _spectra(n: int) -> np.ndarray:
    return rng.normal(0.5, 0.05, (n, p)) + baseline + 0.1 * rng.standard_normal((n, p))


peak = int(np.argmax(wavelengths > 0.5))
X_train = _spectra(n_train)
X_test = _spectra(n_test)
# A non-linear y that TabPFN can model but PLS would only partially capture
y_train = 2.0 * X_train[:, peak] + 0.5 * X_train[:, peak] ** 2 + 0.05 * rng.standard_normal(n_train)
y_test = 2.0 * X_test[:, peak] + 0.5 * X_test[:, peak] ** 2 + 0.05 * rng.standard_normal(n_test)

est = TabPFNNIRSRegressor(n_estimators=8)  # n_est=8 for example speed
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
print(f"  Synthetic NIRS test-set RMSE = {rmse:.4f}")
print(f"  n_features_in   = {est.n_features_in_}")
print(f"  n_features_used = {est.n_features_used_}")
print(f"  subsample applied = {est.subsample_idx_ is not None}")


# =============================================================================
# Section 5: Tuning knobs
# =============================================================================

print("\n" + "-" * 60)
print("Section 5: Tuning knobs (when you might want them)")
print("-" * 60)
print("""
  n_estimators (default 16)
      TabPFN ensemble size. 4 is faster but noisier; 16 is the empirical
      sweet spot; 32 makes no measurable difference on the NIRS cohort.

  max_features (default 2000)
      TabPFN v3 hard limit. Wider spectra are subsampled via np.linspace
      indexing. Lowering this trades accuracy on wide spectra for speed.

  sg_window_length / sg_polyorder / sg_deriv (default 11 / 2 / 1)
      SavitzkyGolay smoothing. Empirically the deriv=1 with window 11 is
      the most universally robust SG kernel for the NIRS cohort tested.

  osc_n_components (default 1)
      OSC orthogonal-signal-correction components. On a few datasets where
      n is very small (<60), OSC can over-fit and SAFER is to drop it
      entirely (use a vanilla SG pipeline instead).

  device / model_path
      Standard TabPFN knobs. ``device='cuda'`` recommended on GPU.
""")

if args.plots:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(wavelengths, X_train[:5].T, alpha=0.6)
    ax[0].set_title("Sample synthetic spectra (5 train samples)")
    ax[0].set_xlabel("wavelength (normalized)")
    ax[1].scatter(y_test, y_pred, alpha=0.7)
    lo, hi = float(y_test.min()), float(y_test.max())
    ax[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax[1].set_xlabel("y_true")
    ax[1].set_ylabel("y_pred")
    ax[1].set_title(f"TabPFNNIRSRegressor test predictions (RMSE={rmse:.3f})")
    plt.tight_layout()
    if args.show:
        plt.show()
    else:
        out = "U06_tabpfn_nirs.png"
        plt.savefig(out, dpi=130)
        print(f"  Plot saved to {out}")

print("\nDone.")
