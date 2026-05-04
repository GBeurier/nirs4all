"""Cohort and dataset loading for nicon_v2.

The cohort manifest is the same CSV used by ``bench/AOM_v0/benchmarks/cohort_regression.csv``
and ``bench/AOM_v0/Ridge/benchmark_runs/curated/results.csv``. We do **not** duplicate the
file; we read it directly so any future cohort updates flow through automatically.

Reference RMSEPs (paper Ridge / PLS / TabPFN raw / TabPFN opt / CNN / CatBoost) are kept
on each ``DatasetSpec`` so that result rows can compute relative metrics offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# After the 2026-04-30 move, this file lives at
#   <NIRS4ALL_PKG_ROOT>/bench/nicon_v2/nicon_v2/datasets.py
# so parents[3] resolves to the nirs4all package root (the one that contains `bench/`).
NIRS4ALL_PKG_ROOT = Path(__file__).resolve().parents[3]   # /home/delete/nirs4all/nirs4all
REPO_ROOT = NIRS4ALL_PKG_ROOT.parent                      # /home/delete/nirs4all
COHORT_REGRESSION_CSV = NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "benchmarks" / "cohort_regression.csv"
AOM_RIDGE_CURATED_CSV = NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "Ridge" / "benchmark_runs" / "curated" / "results.csv"

SMOKE_DATASETS = (
    "ALPINE_P_291_KS",
    "Beer_OriginalExtract_60_KS",
    "Rice_Amylose_313_YbasedSplit",
)
EXTENDED_SMOKE_DATASETS = SMOKE_DATASETS + (
    "Biscuit_Fat_40_RandomSplit",
    "Corn_Oil_80_ZhengChenPelegYbaseSplit",
    "DIESEL_bp50_246_b-a",
)

# 10-dataset diversity-balanced cohort, hand-picked by the user (2026-04-30 v2)
# from the TabPFN paper analysis. n_train ∈ [40, 3734], p ∈ [196, 2151], spans
# 9 chemometric domains (manure, leaves, soil, ALPINE, beer, COLZA, grapevine,
# IncombustibleMaterial, ECOSIS_LeafTraits). Use this for fast hypothesis
# iteration before the full 61-dataset publication run.
REPRESENTATIVE_DATASETS = (
    "All_manure_MgO_SPXY_strat_Manure_type",
    "An_spxyG70_30_byCultivar_NeoSpectra",
    "TIC_spxy70",
    "Chla+b_spxyG_species",
    "ALPINE_P_291_KS",
    "Beer_OriginalExtract_60_YbaseSplit",
    "All_manure_Total_N_SPXY_strat_Manure_type",
    "Chla+b_spxyG_block2deg",
    "N_woOutlier",
    "grapevine_chloride_556_KS",
)


@dataclass(frozen=True)
class DatasetSpec:
    database_name: str
    dataset: str
    n_train: int
    n_test: int
    n_features: int
    train_path: Path
    test_path: Path
    ytrain_path: Path
    ytest_path: Path
    ref_rmse_pls: float | None = None
    ref_rmse_paper_ridge: float | None = None
    ref_rmse_tabpfn_raw: float | None = None
    ref_rmse_tabpfn_opt: float | None = None
    ref_rmse_cnn: float | None = None
    ref_rmse_catboost: float | None = None
    ref_rmse_aom_ridge_branch_global: float | None = None
    ref_rmse_aom_ridge_curated_best: float | None = None
    status: str = "ok"
    reason: str = ""


def _resolve(rel: str) -> Path:
    """Resolve a path stored in the cohort CSV (relative to nirs4all package root)."""
    p = (NIRS4ALL_PKG_ROOT / rel).resolve()
    return p


def _coerce_numeric(df: pd.DataFrame) -> np.ndarray:
    return df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def load_csv_array(path: Path | str) -> np.ndarray:
    """Load an X CSV (semicolon-separated), imputing column-wise mean for residual NaNs."""
    df = pd.read_csv(path, sep=";")
    arr = _coerce_numeric(df)
    if np.isnan(arr).any():
        col_mean = np.nanmean(arr, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        idx = np.where(np.isnan(arr))
        arr[idx] = np.take(col_mean, idx[1])
    return arr


def load_csv_target(path: Path | str) -> np.ndarray:
    """Load a y CSV; supports single-column or first-column-of-many layouts."""
    df = pd.read_csv(path, sep=";")
    return df.iloc[:, 0].astype(float).to_numpy()


def _load_aom_ridge_curated_lookup() -> dict[str, float]:
    """Best-per-dataset RMSEP from the AOM-Ridge curated benchmark CSV.

    Used to populate ``ref_rmse_aom_ridge_curated_best``. Returns ``{}`` if the file
    is missing (e.g. fresh checkout).
    """
    if not AOM_RIDGE_CURATED_CSV.is_file():
        return {}
    try:
        df = pd.read_csv(AOM_RIDGE_CURATED_CSV)
    except Exception:
        return {}
    if df.empty or "rmsep" not in df.columns or "dataset" not in df.columns:
        return {}
    df = df[df["status"].astype(str).str.lower() == "ok"]
    grp = df.groupby("dataset")["rmsep"].min()
    return {k: float(v) for k, v in grp.items() if pd.notna(v)}


def load_cohort_manifest(cohort: str = "smoke", csv_path: Path | None = None) -> list[DatasetSpec]:
    """Load and filter the cohort manifest.

    ``cohort`` ∈ {smoke, extended_smoke, curated, full, all}.
    """
    csv_path = csv_path or COHORT_REGRESSION_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cohort CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"]
    aom_ridge_best = _load_aom_ridge_curated_lookup()

    specs: list[DatasetSpec] = []
    for row in df.itertuples(index=False):
        spec = DatasetSpec(
            database_name=str(getattr(row, "database_name")),
            dataset=str(getattr(row, "dataset")),
            n_train=int(getattr(row, "n_train")),
            n_test=int(getattr(row, "n_test")),
            n_features=int(getattr(row, "p")),
            train_path=_resolve(str(getattr(row, "train_path"))),
            test_path=_resolve(str(getattr(row, "test_path"))),
            ytrain_path=_resolve(str(getattr(row, "ytrain_path"))),
            ytest_path=_resolve(str(getattr(row, "ytest_path"))),
            ref_rmse_pls=_safe_float(getattr(row, "ref_rmse_pls", None)),
            ref_rmse_paper_ridge=_safe_float(getattr(row, "ref_rmse_ridge", None)),
            ref_rmse_tabpfn_raw=_safe_float(getattr(row, "ref_rmse_tabpfn_raw", None)),
            ref_rmse_tabpfn_opt=_safe_float(getattr(row, "ref_rmse_tabpfn_opt", None)),
            ref_rmse_cnn=_safe_float(getattr(row, "ref_rmse_cnn", None)),
            ref_rmse_catboost=_safe_float(getattr(row, "ref_rmse_catboost", None)),
            ref_rmse_aom_ridge_curated_best=aom_ridge_best.get(str(getattr(row, "dataset"))),
            status=str(getattr(row, "status", "ok")),
            reason=str(getattr(row, "reason", "") or ""),
        )
        specs.append(spec)

    name_filter = _resolve_cohort_filter(cohort)
    if name_filter is not None:
        specs = [s for s in specs if s.dataset in name_filter]
    return specs


def _resolve_cohort_filter(cohort: str) -> set[str] | None:
    if cohort in {"all", "full"}:
        return None
    if cohort == "smoke":
        return set(SMOKE_DATASETS)
    if cohort == "extended_smoke":
        return set(EXTENDED_SMOKE_DATASETS)
    if cohort == "representative":
        return set(REPRESENTATIVE_DATASETS)
    if cohort == "curated":
        # Mirror the AOM-Ridge curated subset by reading the curated results CSV.
        names = set(_load_aom_ridge_curated_lookup().keys())
        if names:
            return names
    raise ValueError(f"Unknown cohort: {cohort!r}")


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def load_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the (Xtrain, ytrain, Xtest, ytest) arrays for a single dataset spec."""
    X_train = load_csv_array(spec.train_path)
    X_test = load_csv_array(spec.test_path)
    y_train = load_csv_target(spec.ytrain_path)
    y_test = load_csv_target(spec.ytest_path)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"feature mismatch: train={X_train.shape[1]} test={X_test.shape[1]} for {spec.dataset}"
        )
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"row mismatch: X_train={X_train.shape[0]} y_train={y_train.shape[0]} for {spec.dataset}"
        )
    return X_train, y_train, X_test, y_test


def iter_datasets(specs: Iterable[DatasetSpec]) -> Iterable[tuple[DatasetSpec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    for spec in specs:
        X_train, y_train, X_test, y_test = load_dataset(spec)
        yield spec, X_train, y_train, X_test, y_test
