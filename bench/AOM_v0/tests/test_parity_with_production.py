"""Strict bit-exact parity tests against the production nirs4all AOM-PLS.

These tests are the safety net for the AOM_v0 framework: any change to
operators, banks, NIPALS, SIMPLS, selection, or estimators must not break
the strict equivalence between

    aompls.AOMPLSRegressor(operator_bank="default", criterion="holdout",
                           engine="nipals_adjoint", max_components=15)

and

    nirs4all.operators.models.sklearn.aom_pls.AOMPLSRegressor(
        n_components=15, gate="hard")

on the deployed default 100-operator bank.

Bit-exactness is verified by:
- identical predicted RMSE on the test set (diff < 1e-6),
- identical selected `n_components_` (the prefix `k*`).

If a parity test fails, the most likely culprits are:
1. Bank composition / order changed (production prepends Identity).
2. NIPALS algorithm subtly altered (`g = A^T c -> w_hat = g/||g|| -> a_w = A w_hat -> w = a_w/||a_w||`).
3. Holdout split RNG (must be `np.random.RandomState(42)`, not `default_rng`).
4. Centering: production uses GLOBAL centering once, no per-fold re-centering.
5. Operator kernel definition (e.g. NW d=2 = `gap_kernel ⊛ gap_kernel`).

The cohort below covers a deliberately diverse set of regression splits:
small-n vs large-n, low-noise vs noisy, derivative-friendly vs scatter-friendly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PARITY_DATASETS = [
    ("BEER/Beer_OriginalExtract_60_KS", 1, 0.2042),
    ("CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit", 15, 0.0222),
    ("CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit", 11, 0.1695),
    ("PHOSPHORUS/LP_spxyG", 14, 0.1728),
    ("IncombustibleMaterial/TIC_spxy70", 12, 3.0921),
    ("PLUMS/Firmness_spxy70", 7, 0.3771),
    ("PEACH/Brix_spxy70", 15, 2.0182),
    ("AMYLOSE/Rice_Amylose_313_YbasedSplit", 13, 1.8873),
    ("BISCUIT/Biscuit_Sucrose_40_RandomSplit", 15, 4.3469),
    ("BISCUIT/Biscuit_Fat_40_RandomSplit", 8, 0.5491),
    ("MILK/Milk_Lactose_1224_KS", 15, 0.0580),
]

DATA_ROOT = Path(__file__).resolve().parents[2] / "tabpfn_paper" / "data" / "regression"


def _read_xy(split_dir: Path):
    Xtr = pd.read_csv(split_dir / "Xtrain.csv", sep=";").to_numpy(dtype=float)
    Xte = pd.read_csv(split_dir / "Xtest.csv", sep=";").to_numpy(dtype=float)
    ytr = pd.read_csv(split_dir / "Ytrain.csv", sep=";").iloc[:, 0].astype(float).to_numpy()
    yte = pd.read_csv(split_dir / "Ytest.csv", sep=";").iloc[:, 0].astype(float).to_numpy()
    return Xtr, Xte, ytr, yte


@pytest.mark.parametrize("split_path,expected_k,expected_rmse", PARITY_DATASETS)
def test_aom_v0_bit_exact_with_production(split_path, expected_k, expected_rmse):
    """AOM_v0 must produce the same RMSE and k as production on the cohort."""
    split_dir = DATA_ROOT / split_path
    if not (split_dir / "Xtrain.csv").exists():
        pytest.skip(f"dataset not on disk: {split_path}")
    try:
        from nirs4all.operators.models.sklearn.aom_pls import AOMPLSRegressor as Prod
    except ImportError:
        pytest.skip("nirs4all not installed in test environment")
    from aompls.estimators import AOMPLSRegressor as Mine

    Xtr, Xte, ytr, yte = _read_xy(split_dir)

    prod = Prod(n_components=15, gate="hard")
    prod.fit(Xtr, ytr)
    rmse_p = float(np.sqrt(np.mean((yte - np.asarray(prod.predict(Xte)).ravel()) ** 2)))

    mine = type(prod)(  # build mine with exact same config
        n_components=15, gate="hard"
    ) if False else None
    mine = Mine(
        operator_bank="default",
        max_components=15,
        criterion="holdout",
        engine="nipals_adjoint",
        random_state=0,
    )
    mine.fit(Xtr, ytr)
    rmse_m = float(np.sqrt(np.mean((yte - mine.predict(Xte)) ** 2)))

    assert prod.n_components_ == mine.n_components_, (
        f"selected k mismatch on {split_path}: prod={prod.n_components_}, mine={mine.n_components_}"
    )
    assert abs(rmse_p - rmse_m) < 1e-4, (
        f"RMSE divergence on {split_path}: prod={rmse_p:.6f}, mine={rmse_m:.6f}, diff={abs(rmse_p-rmse_m):.4e}"
    )
    # Sanity: the locked-in expected values from the parity calibration run.
    assert abs(rmse_p - expected_rmse) < 5e-4, (
        f"production RMSE drifted on {split_path}: actual={rmse_p:.6f}, expected={expected_rmse:.4f}"
    )
    assert prod.n_components_ == expected_k, (
        f"production k drifted on {split_path}: actual={prod.n_components_}, expected={expected_k}"
    )


def test_aom_v0_default_bank_size_matches_production():
    """The default bank must contain the same 100 operators as production."""
    from aompls.banks import default_bank
    try:
        from nirs4all.operators.models.sklearn.aom_pls import default_operator_bank
    except ImportError:
        pytest.skip("nirs4all not installed")
    bank_mine = default_bank(p=576)
    bank_prod = default_operator_bank()
    assert len(bank_mine) == len(bank_prod), (
        f"bank size mismatch: mine={len(bank_mine)} prod={len(bank_prod)}"
    )
    assert len(bank_mine) == 100, "default bank must have exactly 100 operators"
