"""Resumable smoke benchmark for AOM-Ridge.

Reads ``bench/AOM_v0/benchmarks/cohort_regression.csv`` (built by the existing
AOM-PLS benchmark tooling), filters to a small representative subset (or the
full cohort), and runs every requested AOM-Ridge variant on every dataset and
seed. Results are appended row-by-row so the runner is fully resumable.

Cross-validation defaults to ``SPXYFold`` from nirs4all when available, which
is the standard NIRS-aware splitter for AOM_v0; the user can override with a
plain ``KFold`` via ``--cv-kind kfold``.

Usage:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge python \\
  bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py \\
  --workspace bench/AOM_v0/Ridge/benchmark_runs/smoke \\
  --cohort smoke --variants smoke --cv 3
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AOM_ROOT = ROOT.parent
REPO_ROOT = AOM_ROOT.parent.parent
for path in (ROOT, AOM_ROOT, REPO_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from aomridge.aom_ridge_pls import AOMRidgePLS, AOMRidgePLSCV  # noqa: E402
from aomridge.cv import RepeatedSPXYFold  # noqa: E402
from aomridge.estimators import AOMRidgeRegressor  # noqa: E402
from aomridge.local_ridge import AOMLocalRidge  # noqa: E402
from aomridge.multi_branch_mkl import AOMMultiBranchMKL  # noqa: E402
from aomridge.split_aware_cv import detect_split_kind, make_inner_cv  # noqa: E402

CODE_VERSION = "AOM_v0/Ridge/0.1.0"

RESULT_COLUMNS = [
    "dataset_group",
    "dataset",
    "task",
    "variant",
    "status",
    "error",
    "selection",
    "operator_bank",
    "alpha",
    "alpha_index",
    "alpha_at_boundary",
    "grid_expansions",
    "cv_min_score",
    "block_scaling",
    "scale_power",
    "x_scale",
    "active_operator_names",
    "selected_operator_names",
    "rmsep",
    "mae",
    "r2",
    "ref_rmse_ridge",
    "ref_rmse_pls",
    "relative_rmsep_vs_ridge_raw",
    "relative_rmsep_vs_paper_ridge",
    "relative_rmsep_vs_pls_standard",
    "fit_time_s",
    "predict_time_s",
    "random_state",
    "version",
    # Ridge-PLS-specific diagnostics (empty for non Ridge-PLS variants).
    "best_n_components",
    "effective_components",
    "alpha_effective",
    "alpha_factor",
    "ridgepls_diagnostics",
]

SMOKE_DATASETS = [
    "Beer_OriginalExtract_60_KS",
    "Rice_Amylose_313_YbasedSplit",
    "ALPINE_P_291_KS",
]

# Larger smoke set used once iter results stabilise
EXTENDED_SMOKE_DATASETS = SMOKE_DATASETS + [
    "Tablet5_KS",
    "Tablet9_KS",
    "Tleaf_grp70_30",
]


# ----------------------------------------------------------------------
# Data loading (mirrors AOM-PLS benchmark conventions)
# ----------------------------------------------------------------------


def _coerce_numeric(df: pd.DataFrame) -> np.ndarray:
    return df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _load_csv_array(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    return _coerce_numeric(df)


def _load_csv_target(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(float).to_numpy()
    return df.iloc[:, 0].astype(float).to_numpy()


# ----------------------------------------------------------------------
# CV builders
# ----------------------------------------------------------------------


def _build_cv(kind: str, n_splits: int, seed: int):
    if kind == "spxy":
        try:
            from nirs4all.operators.splitters import SPXYFold
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "SPXYFold requires nirs4all to be installed in the environment"
            ) from exc
        return SPXYFold(n_splits=n_splits, random_state=seed)
    if kind == "kfold":
        from sklearn.model_selection import KFold

        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"unknown cv kind: {kind!r}")


# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------


@dataclass
class Variant:
    label: str
    selection: str
    operator_bank: str = "compact"
    block_scaling: str = "rms"
    branch_preproc: str | None = None       # external row-wise preproc (snv/msc/osc/asls/emsc2)
    extra: dict[str, object] = field(default_factory=dict)
    cv_factory: object = None               # callable(seed) -> sklearn-compatible splitter


SMOKE_VARIANTS: list[Variant] = [
    # Baseline (no AOM): identity + center
    Variant("Ridge-raw", selection="superblock", operator_bank="identity",
            block_scaling="none"),
    # Same baseline + StandardScaler (paper-style preprocessing)
    Variant("Ridge-raw-stdscale", selection="superblock", operator_bank="identity",
            block_scaling="none", extra={"x_scale": "feature_std"}),
    # Best variants from iter 1 — kept for regression comparison
    Variant("AOMRidge-superblock-compact-none", selection="superblock",
            block_scaling="none"),
    Variant("AOMRidge-global-compact-none", selection="global",
            block_scaling="none"),
    Variant("AOMRidge-active-compact-none", selection="active_superblock",
            block_scaling="none", extra={"active_top_m": 6}),
    # Iter 2 — feature_std variants (Codex backlog item #6)
    Variant("AOMRidge-superblock-compact-none-stdscale", selection="superblock",
            block_scaling="none", extra={"x_scale": "feature_std"}),
    Variant("AOMRidge-global-compact-none-stdscale", selection="global",
            block_scaling="none", extra={"x_scale": "feature_std"}),
    Variant("AOMRidge-active-compact-none-stdscale", selection="active_superblock",
            block_scaling="none", extra={"active_top_m": 6, "x_scale": "feature_std"}),
    # Iter 2b — family-balanced active (Codex backlog item #4)
    Variant("AOMRidge-active-compact-none-blend-fam", selection="active_superblock",
            block_scaling="none", extra={
                "active_top_m": 6,
                "active_score_method": "blend",
                "active_max_per_family": 1,
            }),
]

FULL_VARIANTS: list[Variant] = SMOKE_VARIANTS + [
    Variant("AOMRidge-superblock-default-none", selection="superblock",
            operator_bank="default", block_scaling="none"),
    Variant("AOMRidge-global-default", selection="global",
            operator_bank="default"),
    Variant("AOMRidge-active-default-none", selection="active_superblock",
            operator_bank="default", block_scaling="none",
            extra={"active_top_m": 12}),
]

# ``selection="ridge_pls"`` is a marker handled by ``_run_variant``: it
# dispatches to ``AOMRidgePLS`` (Sprint 1, fixed alpha) or to ``AOMRidgePLSCV``
# (Sprint 2, alpha + n_components grid) based on the ``extra`` keys provided.
# ``selection="aom_pls"`` wraps ``AOMPLSRegressor`` from ``aompls.estimators``.
LEAN_VARIANTS: list[Variant] = [
    # Baseline regressions for comparison.
    Variant("Ridge-raw", selection="superblock", operator_bank="identity",
            block_scaling="none"),
    Variant("AOMRidge-global-compact-none", selection="global",
            block_scaling="none"),
    # Sprint 1 - fixed (H, alpha) Ridge-PLS at the default settings.
    Variant("AOMRidgePLS-compact-h10-a1", selection="ridge_pls",
            operator_bank="compact", block_scaling="frobenius",
            extra={"n_components": 10, "ridge_alpha": 1.0}),
    Variant("AOMRidgePLS-compact-h15-a1", selection="ridge_pls",
            operator_bank="compact", block_scaling="frobenius",
            extra={"n_components": 15, "ridge_alpha": 1.0}),
    # Sprint 2 - original CV variant kept for regression comparison.
    Variant("AOMRidgePLS-compact-cv3-rep3", selection="ridge_pls",
            operator_bank="compact", block_scaling="frobenius",
            extra={
                "n_components_grid": (2, 5, 10, 15, 20),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 9).tolist(),
                "cv_splits": 3,
            }),
    # Round-2 fixes -------------------------------------------------------
    # 1) Real RepeatedSPXYFold(n_splits=3, n_repeats=3) + relative alpha grid
    #    + 1-SE rule + n_components_grid (2, 3, 5, 7, 10, 15, 20, 30).
    Variant("AOMRidgePLS-compact-cv3-rep3-spxy", selection="ridge_pls",
            operator_bank="compact", block_scaling="frobenius",
            extra={
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 9).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
            }),
    # 2) Strategy B: H_max=30 (capped per fold by the CV) + large relative
    #    alpha grid (25 factors) + 1-SE rule.
    Variant("AOMRidgePLS-compact-Hmax-relative", selection="ridge_pls",
            operator_bank="compact", block_scaling="frobenius",
            extra={
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
            }),
    # 3) Same recipe on response_dedup bank.
    Variant("AOMRidgePLS-response_dedup-cv-relative", selection="ridge_pls",
            operator_bank="response_dedup", block_scaling="frobenius",
            extra={
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
            }),
    # 4) Same recipe on family_pruned bank.
    Variant("AOMRidgePLS-family_pruned-cv-relative", selection="ridge_pls",
            operator_bank="family_pruned", block_scaling="frobenius",
            extra={
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
            }),
    # 5) column_scaling=True + relative alpha.
    Variant("AOMRidgePLS-compact-colscale-cv-relative", selection="ridge_pls",
            operator_bank="compact", block_scaling="frobenius",
            extra={
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
                "column_scaling": True,
            }),
    # 6) In-run AOM-PLS baseline (Codex item 14): wraps AOMPLSRegressor with a
    #    CV over n_components for fair in-run comparison.
    Variant("AOM-PLS-compact-CV", selection="aom_pls",
            operator_bank="compact", block_scaling="none",
            extra={"max_components": 30, "cv": 3}),
    # 7) RidgePLS x branch preprocessing (SNV/MSC/OSC/ASLS/EMSC) — same recipe
    #    as Hmax-relative, with row-wise NIRS preprocessor applied externally.
    *(
        Variant(f"AOMRidgePLS-compact-Hmax-relative-{branch}", selection="ridge_pls",
                operator_bank="compact", block_scaling="frobenius",
                branch_preproc=branch,
                extra={
                    "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                    "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                    "cv_kind": "spxy_repeated",
                    "cv_splits": 3,
                    "cv_repeats": 3,
                    "ridge_alpha_mode": "relative_to_score_variance",
                    "selection_rule": "1se",
                    "scoring": "rmse_mean",
                })
        for branch in ("snv", "msc", "osc", "asls", "emsc2")
    ),
    # 8) AOM-Ridge x branch preprocessing — top variant + each preproc.
    *(
        Variant(f"AOMRidge-global-compact-none-{branch}", selection="global",
                block_scaling="none", branch_preproc=branch)
        for branch in ("snv", "msc", "osc", "asls", "emsc2")
    ),
    # Phase H2 — branch_global with the 10-branch list (Codex review #2).
    Variant("AOMRidge-branch_global-compact-10branches-1se", selection="branch_global",
            block_scaling="none",
            extra={
                "branches": ("none", "snv", "msc", "emsc1", "emsc2",
                             "asls_soft", "asls", "asls_hard", "snv_asls", "msc_asls"),
                "selection_rule": "1se",
            }),
    Variant("AOMRidge-branch_global-compact-10branches", selection="branch_global",
            block_scaling="none",
            extra={
                "branches": ("none", "snv", "msc", "emsc1", "emsc2",
                             "asls_soft", "asls", "asls_hard", "snv_asls", "msc_asls"),
            }),
    # Phase H3 — split-aware inner CV (rebuilds inner CV to mirror test split).
    Variant("AOMRidge-global-compact-none-split_aware", selection="global",
            block_scaling="none",
            extra={
                "split_aware_cv": True,
                "split_kind": "auto",
                "scoring": "rmse_pooled_trimmed",
            }),
    Variant("AOMRidge-branch_global-compact-split_aware", selection="branch_global",
            block_scaling="none",
            extra={
                "branches": ("none", "snv", "msc", "asls"),
                "split_aware_cv": True,
                "split_kind": "auto",
            }),
    # Phase H4 — Soft Multi-Branch MKL across preprocessing branches.
    Variant("AOMRidge-MultiBranchMKL-compact-shrink03", selection="multi_branch_mkl",
            operator_bank="compact", block_scaling="none",
            extra={
                "branches": ("none", "snv", "msc", "asls", "emsc2"),
                "shrinkage_to_identity": 0.3,
            }),
    Variant("AOMRidge-MultiBranchMKL-compact-shrink05", selection="multi_branch_mkl",
            operator_bank="compact", block_scaling="none",
            extra={
                "branches": ("none", "snv", "msc", "asls", "emsc2"),
                "shrinkage_to_identity": 0.5,
            }),
    # Phase H5 — local Ridge in AOM score space.
    Variant("AOMRidge-Local-compact-knn50", selection="local_ridge",
            operator_bank="compact", block_scaling="none",
            extra={
                "k_grid": (50,),
                "local_weight_beta": 1.0,
                "distance_branches": ("none", "snv", "msc"),
            }),
    Variant("AOMRidge-Local-compact-cv-blended", selection="local_ridge",
            operator_bank="compact", block_scaling="none",
            extra={
                "k_grid": (10, 20, 50, 100),
                "local_weight_beta": "auto",
                "distance_branches": ("none", "snv", "msc"),
            }),
]


# Final headline set chosen from diverse-cohort iter3 winners: best single
# variants by median delta + RidgePLS-EMSC2 for the large-n ECOSIS sweet spot.
HEADLINE_VARIANTS: list[Variant] = [
    Variant("Ridge-raw", selection="superblock", operator_bank="identity",
            block_scaling="none"),
    Variant("AOMRidge-global-compact-none", selection="global",
            block_scaling="none"),
    Variant("AOMRidge-global-compact-none-msc", selection="global",
            block_scaling="none", branch_preproc="msc"),
    Variant("AOMRidge-global-compact-none-snv", selection="global",
            block_scaling="none", branch_preproc="snv"),
    Variant("AOMRidge-global-compact-none-asls", selection="global",
            block_scaling="none", branch_preproc="asls"),
    Variant("AOMRidgePLS-compact-colscale-cv-relative", selection="ridge_pls",
            operator_bank="compact", block_scaling="frobenius",
            extra={
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
                "column_scaling": True,
            }),
    Variant("AOMRidgePLS-compact-Hmax-relative-emsc2", selection="ridge_pls",
            operator_bank="compact", block_scaling="frobenius",
            branch_preproc="emsc2",
            extra={
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
            }),
    Variant("AOM-PLS-compact-CV", selection="aom_pls",
            operator_bank="compact", block_scaling="none",
            extra={"max_components": 30, "cv": 3}),
]


def _variant_to_spec(variant: Variant) -> dict[str, object]:
    """Convert a ``Variant`` dataclass to the spec dict consumed by
    ``AOMRidgeAutoSelector``."""
    spec: dict[str, object] = {
        "label": variant.label,
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "block_scaling": variant.block_scaling,
        "extra": dict(variant.extra),
    }
    if variant.branch_preproc:
        spec["branch_preproc"] = variant.branch_preproc
    return spec


# Auto-select variant: outer-CV picks the best HEADLINE variant per dataset.
# Excludes itself from the candidate set to avoid recursion.
HEADLINE_VARIANTS.append(
    Variant("AOMRidge-AutoSelect-headline-spxy3", selection="auto_select",
            block_scaling="none",
            extra={
                "candidates": "headline",
                "outer_cv_kind": "spxy",
                "outer_cv_splits": 3,
            }),
)

# Blender variant: convex non-negative blend of HEADLINE variant OOF
# predictions. Excludes auto_select and itself from the candidate set.
HEADLINE_VARIANTS.append(
    Variant("AOMRidge-Blender-headline-spxy3", selection="blender",
            block_scaling="none",
            extra={
                "candidates": "headline",
                "outer_cv_kind": "spxy",
                "outer_cv_splits": 3,
                "regularizer": 0.01,
            }),
)


_H_EXTRAS_LABELS = {
    "Ridge-raw",
    "AOMRidge-global-compact-none",
    "AOMRidge-global-compact-none-msc",
    "AOMRidge-branch_global-compact-10branches-1se",
    "AOMRidge-branch_global-compact-10branches",
    "AOMRidge-global-compact-none-split_aware",
    "AOMRidge-branch_global-compact-split_aware",
    "AOMRidge-MultiBranchMKL-compact-shrink03",
    "AOMRidge-MultiBranchMKL-compact-shrink05",
    "AOMRidge-Local-compact-knn50",
    "AOMRidge-Local-compact-cv-blended",
}

# Faster subset: only the most informative non-redundant new variants.
# Drops the second branch_global flavour (redundant w/ -1se), the
# split-aware-branch combo (slow on top of slow), the second MultiBranchMKL
# shrinkage (only differs by hyperparameter), and the cv-blended Local
# (slower than knn50 with similar performance).
_H_FAST_LABELS = {
    "AOMRidge-global-compact-none-split_aware",
    "AOMRidge-MultiBranchMKL-compact-shrink03",
    "AOMRidge-Local-compact-knn50",
    "AOMRidge-branch_global-compact-10branches-1se",
}

# Fastest subset (drops branch_global-10branches-1se which scales O(n^2)).
_H_FASTEST_LABELS = {
    "AOMRidge-global-compact-none-split_aware",
    "AOMRidge-MultiBranchMKL-compact-shrink03",
    "AOMRidge-Local-compact-knn50",
}

# Local-only — for big datasets where Local AOM-Ridge is the only fast option.
_LOCAL_ONLY_LABELS = {
    "Ridge-raw",
    "AOMRidge-Local-compact-knn50",
}

# Giants extras — split_aware + MultiBranchMKL + Local-cv-blended on giants
_GIANTS_EXTRAS_LABELS = {
    "Ridge-raw",
    "AOMRidge-global-compact-none-split_aware",
    "AOMRidge-MultiBranchMKL-compact-shrink03",
    "AOMRidge-Local-compact-cv-blended",
}

# H ablations — extra hyperparameter variants for the H-extension table.
_H_ABLATIONS_LABELS = {
    "Ridge-raw",
    "AOMRidge-MultiBranchMKL-compact-shrink05",
    "AOMRidge-Local-compact-cv-blended",
}

# Top-5 fast — fill missing cells in the per-variant table on remaining datasets
_TOP5_FAST_LABELS = {
    "Ridge-raw",
    "AOMRidge-global-compact-none-split_aware",
    "AOMRidge-MultiBranchMKL-compact-shrink03",
    "AOMRidge-Local-compact-knn50",
    "AOMRidge-Local-compact-cv-blended",
}


def _resolve_variants(name: str) -> list[Variant]:
    if name == "smoke":
        return SMOKE_VARIANTS
    if name == "lean":
        return LEAN_VARIANTS
    if name == "headline":
        return HEADLINE_VARIANTS
    if name == "full":
        return FULL_VARIANTS
    if name == "h_extras":
        return [v for v in LEAN_VARIANTS if v.label in _H_EXTRAS_LABELS]
    if name == "h_fast":
        return [v for v in LEAN_VARIANTS if v.label in _H_FAST_LABELS]
    if name == "h_fastest":
        return [v for v in LEAN_VARIANTS if v.label in _H_FASTEST_LABELS]
    if name == "local_only":
        return [v for v in LEAN_VARIANTS if v.label in _LOCAL_ONLY_LABELS]
    if name == "giants_extras":
        return [v for v in LEAN_VARIANTS if v.label in _GIANTS_EXTRAS_LABELS]
    if name == "h_ablations":
        return [v for v in LEAN_VARIANTS if v.label in _H_ABLATIONS_LABELS]
    if name == "top5_fast":
        return [v for v in LEAN_VARIANTS if v.label in _TOP5_FAST_LABELS]
    raise ValueError(f"unknown variants set: {name!r}")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true).ravel() - np.asarray(y_pred).ravel())))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ----------------------------------------------------------------------
# Single-variant runner
# ----------------------------------------------------------------------


def _existing_keys(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    df = pd.read_csv(results_path, dtype=str)
    if df.empty:
        return set()
    return {(row["dataset_group"], row["dataset"], row["variant"], row["random_state"])
            for _, row in df.iterrows()}


def _append_row(results_path: Path, row: dict[str, object]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _build_ridge_pls(variant: Variant, seed: int, cv_obj):
    """Construct ``AOMRidgePLS`` or ``AOMRidgePLSCV`` from a Ridge-PLS variant.

    ``variant.extra`` may include either a fixed ``n_components`` /
    ``ridge_alpha`` pair (-> ``AOMRidgePLS``) or grid keys
    ``n_components_grid`` / ``ridge_alpha_grid`` (-> ``AOMRidgePLSCV``).

    The CV splitter for the wrapper is resolved via ``cv_kind``:
    ``"spxy_repeated"`` -> ``RepeatedSPXYFold(n_splits, n_repeats, seed)``,
    ``"kfold_int"`` -> integer ``cv_splits`` (shuffled KFold inside the
    estimator), or omitted -> the outer ``cv_obj``. ``cv_splits`` is the
    number of folds per repeat; ``cv_repeats`` defaults to 1.
    """
    extra = dict(variant.extra)
    base_kwargs = {
        "operator_bank": variant.operator_bank,
        "block_scaling": variant.block_scaling,
        "random_state": seed,
    }
    cv_kind = extra.pop("cv_kind", None)
    cv_splits = int(extra.pop("cv_splits", 0)) or 0
    cv_repeats = int(extra.pop("cv_repeats", 1)) or 1
    if "n_components_grid" in extra or "ridge_alpha_grid" in extra:
        if cv_kind == "spxy_repeated":
            if cv_splits < 2:
                raise ValueError(
                    "cv_splits must be >= 2 when cv_kind='spxy_repeated'"
                )
            cv_for_inner = RepeatedSPXYFold(
                n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed,
            )
        elif cv_splits > 0:
            cv_for_inner = cv_splits
        else:
            cv_for_inner = cv_obj
        return AOMRidgePLSCV(
            **base_kwargs,
            n_components_grid=tuple(extra.pop("n_components_grid", (2, 5, 10, 15, 20))),
            ridge_alpha_grid=extra.pop(
                "ridge_alpha_grid", np.logspace(-4.0, 4.0, 9).tolist(),
            ),
            cv=cv_for_inner,
            **extra,
        )
    return AOMRidgePLS(**base_kwargs, cv=cv_obj, **extra)


def _run_variant(
    variant: Variant,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int,
    cv_obj,
    cv_splits: int = 3,
    dataset_name: str | None = None,
) -> dict[str, object]:
    # Optional per-variant CV override (e.g. RepeatedSPXYFold).
    if variant.cv_factory is not None:
        cv_obj = variant.cv_factory(seed)
    # Phase H3 — split-aware inner CV: detect the test split protocol from the
    # dataset name (Y-based / grouped / SPXY) and rebuild the inner CV so the
    # alpha selected on training folds corresponds to a fold geometry that
    # mirrors the eventual train/test partition. Done before any branch
    # preprocessing so the rebuilt CV reflects the unprocessed X geometry.
    extra_view = dict(variant.extra)
    detected_split_kind: str | None = None
    if extra_view.pop("split_aware_cv", False):
        split_kind_arg = str(extra_view.pop("split_kind", "auto"))
        if split_kind_arg == "auto":
            detected_split_kind = detect_split_kind(dataset_name or "", Xtr, ytr)
        else:
            detected_split_kind = split_kind_arg
        cv_obj = make_inner_cv(
            detected_split_kind, n_splits=cv_splits, random_state=seed,
        )
    # Optional row-wise NIRS preproc (SNV / MSC / OSC / ASLS / EMSC) applied
    # outside the estimator. Stateless preprocs (SNV) are fully fold-safe;
    # stateful ones (MSC reference, OSC fit) are computed on the full train
    # so they share the standard NIRS-pipeline assumption used by AOM-PLS.
    branch_preproc = None
    if variant.branch_preproc:
        from aomridge.branches import (
            fit_transform_branch as _fit_transform_branch,
        )
        from aomridge.branches import (
            make_branch_preproc as _make_preproc,
        )
        branch_preproc = _make_preproc(variant.branch_preproc)
        if branch_preproc is not None:
            Xtr = _fit_transform_branch(branch_preproc, np.asarray(Xtr, dtype=float),
                                        np.asarray(ytr, dtype=float))
            Xte = np.asarray(branch_preproc.transform(np.asarray(Xte, dtype=float)),
                             dtype=float)
    if variant.selection in ("auto_select", "blender"):
        if variant.selection == "auto_select":
            from aomridge.auto_selector import AOMRidgeAutoSelector as _Selector
        else:
            from aomridge.blender import AOMRidgeBlender as _Selector

        extra = dict(variant.extra)
        cand_key = extra.pop("candidates", "headline")
        outer_kind = str(extra.pop("outer_cv_kind", "spxy"))
        outer_splits = int(extra.pop("outer_cv_splits", 3))
        outer_repeats = int(extra.pop("outer_cv_repeats", 1))
        n_jobs = int(extra.pop("n_jobs", 1))
        if cand_key == "headline":
            cand_specs = [
                _variant_to_spec(v) for v in HEADLINE_VARIANTS
                if v.label != variant.label
                and v.selection not in ("auto_select", "blender")
            ]
        elif isinstance(cand_key, list):
            cand_specs = list(cand_key)
        else:
            raise ValueError(
                f"unknown candidates key {cand_key!r}; expected 'headline' or list"
            )
        est = _Selector(
            candidates=cand_specs,
            outer_cv=outer_splits,
            outer_cv_kind=outer_kind,
            outer_cv_repeats=outer_repeats,
            random_state=seed,
            n_jobs=n_jobs,
            **extra,
        )
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        yhat = est.predict(Xte)
        predict_time = time.perf_counter() - t1
        diag_blob: dict[str, object] = {
            "selected_variant_label": est.selected_variant_label_,
            "selected_variant_index": int(est.selected_variant_index_),
            "candidate_labels": [
                str(c.get("label", f"candidate_{i}"))
                for i, c in enumerate(est.candidates_)
            ],
            "cv_scores": [float(s) for s in est.cv_scores_],
        }
        if variant.selection == "blender":
            diag_blob["weights"] = [float(w) for w in est.weights_]
            diag_blob["regularizer"] = float(est.regularizer)
        return {
            "selection": variant.selection,
            "operator_bank": variant.operator_bank,
            "alpha": 0.0,
            "alpha_index": None,
            "alpha_at_boundary": None,
            "grid_expansions": 0,
            "cv_min_score": float(min(est.cv_scores_)) if est.cv_scores_ else None,
            "block_scaling": variant.block_scaling,
            "scale_power": 1.0,
            "x_scale": "center",
            "active_operator_names": "",
            "selected_operator_names": json.dumps(
                [est.selected_variant_label_]
            ),
            "rmsep": rmse(yte, yhat),
            "mae": mae(yte, yhat),
            "r2": r2(yte, yhat),
            "fit_time_s": float(fit_time),
            "predict_time_s": float(predict_time),
            "best_n_components": "",
            "effective_components": "",
            "alpha_effective": "",
            "alpha_factor": "",
            "ridgepls_diagnostics": json.dumps(diag_blob),
        }
    if variant.selection == "ridge_pls":
        est = _build_ridge_pls(variant, seed=seed, cv_obj=cv_obj)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        yhat = est.predict(Xte)
        predict_time = time.perf_counter() - t1
        diag = est.get_diagnostics()
        # Ridge-PLS diagnostics: factor (user-supplied alpha pre-scaling),
        # effective alpha, best_n_components, effective_components.
        if isinstance(est, AOMRidgePLSCV):
            best_h = int(diag.get("best_n_components", est.best_n_components_))
            alpha_factor = float(diag.get("best_ridge_alpha", est.best_ridge_alpha_))
        else:
            best_h = int(est.n_components_)
            alpha_factor = float(variant.extra.get("ridge_alpha", 0.0))
        alpha_effective = float(diag.get("ridge_alpha", 0.0))
        eff_components = float(diag.get("effective_components", float("nan")))
        # Compact serialisable diagnostic blob.
        ridgepls_blob = json.dumps({
            "shrinkage_factors": diag.get("shrinkage_factors", []),
            "block_importance": diag.get("block_importance", []),
            "score_diag": diag.get("score_diag", []),
            "alpha_per_component": diag.get("alpha_per_component", []),
            "selection_rule": diag.get("selection_rule", "min"),
            "scoring": diag.get("scoring", "rmse_mean"),
            "ridge_alpha_mode": diag.get("ridge_alpha_mode", "absolute"),
        })
        return {
            "selection": variant.selection,
            "operator_bank": variant.operator_bank,
            "alpha": alpha_effective,
            "alpha_index": None,
            "alpha_at_boundary": None,
            "grid_expansions": 0,
            "cv_min_score": diag.get("best_score"),
            "block_scaling": variant.block_scaling,
            "scale_power": 1.0,
            "x_scale": "center",
            "active_operator_names": "",
            "selected_operator_names": json.dumps(diag.get("operator_names", [])),
            "rmsep": rmse(yte, yhat),
            "mae": mae(yte, yhat),
            "r2": r2(yte, yhat),
            "fit_time_s": float(fit_time),
            "predict_time_s": float(predict_time),
            "best_n_components": best_h,
            "effective_components": eff_components,
            "alpha_effective": alpha_effective,
            "alpha_factor": alpha_factor,
            "ridgepls_diagnostics": ridgepls_blob,
        }
    if variant.selection == "aom_pls":
        from aompls.estimators import AOMPLSRegressor

        extra = dict(variant.extra)
        max_components = int(extra.pop("max_components", 30))
        cv_inner = int(extra.pop("cv", 3))
        est = AOMPLSRegressor(
            n_components="auto",
            max_components=max_components,
            operator_bank=variant.operator_bank,
            cv=cv_inner,
            random_state=seed,
            **extra,
        )
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        yhat = est.predict(Xte)
        predict_time = time.perf_counter() - t1
        # AOMPLSRegressor exposes ``n_components_`` and ``selected_operator_``.
        sel_op = getattr(est, "selected_operator_", None)
        sel_op_name = getattr(sel_op, "name", "") if sel_op is not None else ""
        return {
            "selection": variant.selection,
            "operator_bank": variant.operator_bank,
            "alpha": 0.0,
            "alpha_index": None,
            "alpha_at_boundary": None,
            "grid_expansions": 0,
            "cv_min_score": None,
            "block_scaling": variant.block_scaling,
            "scale_power": 1.0,
            "x_scale": "center",
            "active_operator_names": "",
            "selected_operator_names": json.dumps([sel_op_name] if sel_op_name else []),
            "rmsep": rmse(yte, yhat),
            "mae": mae(yte, yhat),
            "r2": r2(yte, yhat),
            "fit_time_s": float(fit_time),
            "predict_time_s": float(predict_time),
            "best_n_components": int(getattr(est, "n_components_", max_components)),
            "effective_components": float(getattr(est, "n_components_", max_components)),
            "alpha_effective": 0.0,
            "alpha_factor": 0.0,
            "ridgepls_diagnostics": "",
        }
    if variant.selection == "multi_branch_mkl":
        # Phase H4 — Soft Multi-Branch MKL across preprocessing branches.
        kwargs = {
            "operator_bank": variant.operator_bank,
            "block_scaling": variant.block_scaling,
            "cv": cv_obj,
            "random_state": seed,
        }
        kwargs.update(extra_view)
        est = AOMMultiBranchMKL(**kwargs)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        yhat = est.predict(Xte)
        predict_time = time.perf_counter() - t1
        diag = est.get_diagnostics()
        return {
            "selection": variant.selection,
            "operator_bank": variant.operator_bank,
            "alpha": float(diag["alpha"]),
            "alpha_index": diag.get("alpha_index"),
            "alpha_at_boundary": diag.get("alpha_at_boundary"),
            "grid_expansions": 0,
            "cv_min_score": diag.get("cv_min_score"),
            "block_scaling": variant.block_scaling,
            "scale_power": 1.0,
            "x_scale": "center",
            "active_operator_names": "",
            # Pack branch weights into the selected_operator_names slot so
            # downstream summaries surface the multi-branch decision.
            "selected_operator_names": json.dumps(diag["branch_weights"]),
            "rmsep": rmse(yte, yhat),
            "mae": mae(yte, yhat),
            "r2": r2(yte, yhat),
            "fit_time_s": float(fit_time),
            "predict_time_s": float(predict_time),
        }
    if variant.selection == "local_ridge":
        # Phase H5 — local Ridge in AOM score space.
        kwargs = {
            "operator_bank": variant.operator_bank,
            "block_scaling": variant.block_scaling,
            "cv": cv_obj,
            "random_state": seed,
        }
        kwargs.update(extra_view)
        est = AOMLocalRidge(**kwargs)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        yhat = est.predict(Xte)
        predict_time = time.perf_counter() - t1
        diag = est.get_diagnostics()
        summary = {
            "branch": diag["selected_branch"],
            "k": diag["selected_k"],
            "beta": diag["selected_beta"],
        }
        return {
            "selection": variant.selection,
            "operator_bank": variant.operator_bank,
            "alpha": float(diag["selected_alpha"]),
            "alpha_index": diag.get("selected_alpha_index"),
            "alpha_at_boundary": None,
            "grid_expansions": 0,
            "cv_min_score": None,
            "block_scaling": variant.block_scaling,
            "scale_power": 1.0,
            "x_scale": "center",
            "active_operator_names": "",
            "selected_operator_names": json.dumps(summary),
            "rmsep": rmse(yte, yhat),
            "mae": mae(yte, yhat),
            "r2": r2(yte, yhat),
            "fit_time_s": float(fit_time),
            "predict_time_s": float(predict_time),
        }
    kwargs = {
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "block_scaling": variant.block_scaling,
        "cv": cv_obj,
        "random_state": seed,
    }
    # ``extra_view`` already had any split-aware-cv flags consumed.
    kwargs.update(extra_view)
    est = AOMRidgeRegressor(**kwargs)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    yhat = est.predict(Xte)
    predict_time = time.perf_counter() - t1
    diag = est.get_diagnostics()
    out = {
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "alpha": float(diag["alpha"]),
        "alpha_index": diag.get("alpha_index"),
        "alpha_at_boundary": diag.get("alpha_at_boundary"),
        "grid_expansions": diag.get("grid_expansions", 0),
        "cv_min_score": diag.get("cv_min_score"),
        "block_scaling": variant.block_scaling,
        "scale_power": float(diag.get("scale_power", 1.0)),
        "x_scale": diag.get("x_scale", "center"),
        "active_operator_names": json.dumps(
            diag.get("active_operator_names", [])
        ) if variant.selection == "active_superblock" else "",
        "selected_operator_names": json.dumps(diag["selected_operator_names"]),
        "rmsep": rmse(yte, yhat),
        "mae": mae(yte, yhat),
        "r2": r2(yte, yhat),
        "fit_time_s": float(fit_time),
        "predict_time_s": float(predict_time),
    }
    return out


def run_dataset(
    cohort_row: pd.Series,
    variants: Sequence[Variant],
    results_path: Path,
    seeds: Sequence[int],
    cv_kind: str,
    cv_splits: int,
    existing_keys: set,
) -> int:
    Xtr = _load_csv_array(cohort_row["train_path"])
    Xte = _load_csv_array(cohort_row["test_path"])
    ytr = _load_csv_target(cohort_row["ytrain_path"])
    yte = _load_csv_target(cohort_row["ytest_path"])
    ref_pls = cohort_row.get("ref_rmse_pls", "")
    ref_ridge = cohort_row.get("ref_rmse_ridge", "")
    n_added = 0
    for seed in seeds:
        cv_obj = _build_cv(cv_kind, cv_splits, seed)
        # Compute the in-cohort raw-Ridge RMSE first so we can report ratios.
        raw_label = "Ridge-raw"
        raw_row_key = (
            cohort_row["database_name"], cohort_row["dataset"], raw_label, str(seed),
        )
        if raw_row_key in existing_keys:
            raw_rmsep = None
        else:
            raw_variant = next(v for v in variants if v.label == raw_label)
            try:
                raw_metrics = _run_variant(
                    raw_variant, Xtr, ytr, Xte, yte, seed=seed, cv_obj=cv_obj,
                    cv_splits=cv_splits, dataset_name=str(cohort_row["dataset"]),
                )
                raw_row = _row_record(
                    cohort_row, raw_label, seed, raw_variant, raw_metrics,
                    ref_pls=ref_pls, ref_ridge=ref_ridge, raw_rmsep=None,
                )
                _append_row(results_path, raw_row)
                existing_keys.add(raw_row_key)
                raw_rmsep = float(raw_metrics["rmsep"])
                n_added += 1
            except Exception as exc:
                _append_row(results_path, _error_record(
                    cohort_row, raw_label, seed, raw_variant, exc, ref_pls, ref_ridge,
                ))
                existing_keys.add(raw_row_key)
                raw_rmsep = None
                n_added += 1
        # Run remaining variants
        for variant in variants:
            if variant.label == raw_label:
                continue
            key = (cohort_row["database_name"], cohort_row["dataset"], variant.label, str(seed))
            if key in existing_keys:
                continue
            try:
                metrics = _run_variant(
                    variant, Xtr, ytr, Xte, yte, seed=seed, cv_obj=cv_obj,
                    cv_splits=cv_splits, dataset_name=str(cohort_row["dataset"]),
                )
                row = _row_record(
                    cohort_row, variant.label, seed, variant, metrics,
                    ref_pls=ref_pls, ref_ridge=ref_ridge, raw_rmsep=raw_rmsep,
                )
            except Exception as exc:
                row = _error_record(
                    cohort_row, variant.label, seed, variant, exc, ref_pls, ref_ridge,
                )
            _append_row(results_path, row)
            existing_keys.add(key)
            n_added += 1
    return n_added


def _row_record(
    cohort_row: pd.Series,
    label: str,
    seed: int,
    variant: Variant,
    metrics: dict[str, object],
    ref_pls,
    ref_ridge,
    raw_rmsep: float | None,
) -> dict[str, object]:
    rmsep = float(metrics["rmsep"])
    rel_ridge_raw = ""
    if raw_rmsep is not None and raw_rmsep > 0:
        rel_ridge_raw = rmsep / raw_rmsep
    rel_paper_ridge = ""
    if pd.notna(ref_ridge) and float(ref_ridge) > 0:
        rel_paper_ridge = rmsep / float(ref_ridge)
    rel_pls = ""
    if pd.notna(ref_pls) and float(ref_pls) > 0:
        rel_pls = rmsep / float(ref_pls)
    return {
        "dataset_group": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "regression",
        "variant": label,
        "status": "ok",
        "error": "",
        "selection": metrics["selection"],
        "operator_bank": metrics["operator_bank"],
        "alpha": metrics["alpha"],
        "alpha_index": metrics.get("alpha_index"),
        "alpha_at_boundary": metrics.get("alpha_at_boundary"),
        "grid_expansions": metrics.get("grid_expansions", 0),
        "cv_min_score": metrics.get("cv_min_score"),
        "block_scaling": metrics["block_scaling"],
        "scale_power": metrics.get("scale_power", 1.0),
        "x_scale": metrics.get("x_scale", "center"),
        "active_operator_names": metrics["active_operator_names"],
        "selected_operator_names": metrics["selected_operator_names"],
        "rmsep": rmsep,
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "ref_rmse_ridge": ref_ridge if pd.notna(ref_ridge) else "",
        "ref_rmse_pls": ref_pls if pd.notna(ref_pls) else "",
        "relative_rmsep_vs_ridge_raw": rel_ridge_raw,
        "relative_rmsep_vs_paper_ridge": rel_paper_ridge,
        "relative_rmsep_vs_pls_standard": rel_pls,
        "fit_time_s": metrics["fit_time_s"],
        "predict_time_s": metrics["predict_time_s"],
        "random_state": seed,
        "version": CODE_VERSION,
        "best_n_components": metrics.get("best_n_components", ""),
        "effective_components": metrics.get("effective_components", ""),
        "alpha_effective": metrics.get("alpha_effective", ""),
        "alpha_factor": metrics.get("alpha_factor", ""),
        "ridgepls_diagnostics": metrics.get("ridgepls_diagnostics", ""),
    }


def _error_record(
    cohort_row: pd.Series,
    label: str,
    seed: int,
    variant: Variant,
    exc: Exception,
    ref_pls,
    ref_ridge,
) -> dict[str, object]:
    return {
        "dataset_group": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "regression",
        "variant": label,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "alpha": "",
        "alpha_index": "",
        "alpha_at_boundary": "",
        "grid_expansions": "",
        "cv_min_score": "",
        "block_scaling": variant.block_scaling,
        "scale_power": variant.extra.get("scale_power", 1.0),
        "x_scale": variant.extra.get("x_scale", "center"),
        "active_operator_names": "",
        "selected_operator_names": "",
        "rmsep": "",
        "mae": "",
        "r2": "",
        "ref_rmse_ridge": ref_ridge if pd.notna(ref_ridge) else "",
        "ref_rmse_pls": ref_pls if pd.notna(ref_pls) else "",
        "relative_rmsep_vs_ridge_raw": "",
        "relative_rmsep_vs_paper_ridge": "",
        "relative_rmsep_vs_pls_standard": "",
        "fit_time_s": "",
        "predict_time_s": "",
        "random_state": seed,
        "version": CODE_VERSION,
        "best_n_components": "",
        "effective_components": "",
        "alpha_effective": "",
        "alpha_factor": "",
        "ridgepls_diagnostics": "",
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _select_cohort_rows(cohort_path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(cohort_path)
    df_ok = df[df["status"] == "ok"].copy()
    if name == "smoke":
        preferred = df_ok[df_ok["dataset"].isin(SMOKE_DATASETS)]
        if not preferred.empty:
            return preferred
        return df_ok.head(3)
    if name == "smoke6":
        preferred = df_ok[df_ok["dataset"].isin(EXTENDED_SMOKE_DATASETS)]
        return preferred
    if name == "full":
        return df_ok
    raise ValueError(f"unknown cohort selection: {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AOM-Ridge benchmark runner")
    parser.add_argument("--workspace", required=True, help="output workspace")
    parser.add_argument(
        "--cohort", default="smoke", choices=["smoke", "smoke6", "full"],
        help="dataset cohort selection",
    )
    parser.add_argument(
        "--variants", default="smoke",
        choices=["smoke", "lean", "headline", "full", "h_extras", "h_fast", "h_fastest",
                 "local_only", "giants_extras", "h_ablations", "top5_fast"],
        help="variant set",
    )
    parser.add_argument("--cv", type=int, default=3, help="CV split count")
    parser.add_argument(
        "--cv-kind", default="spxy", choices=["spxy", "kfold"],
        help="splitter kind for inner CV",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--cohort-path",
        default="bench/AOM_v0/benchmarks/cohort_regression.csv",
        help="path to the AOM_v0 regression cohort CSV",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "results.csv"

    cohort = _select_cohort_rows(args.cohort_path, args.cohort)
    print(
        f"[aomridge] {len(cohort)} datasets, variants={args.variants}, "
        f"cv={args.cv_kind}({args.cv})"
    )
    variants = _resolve_variants(args.variants)
    existing = _existing_keys(results_path)
    total = 0
    for _, row in cohort.iterrows():
        try:
            n = run_dataset(
                cohort_row=row,
                variants=variants,
                results_path=results_path,
                seeds=args.seeds,
                cv_kind=args.cv_kind,
                cv_splits=args.cv,
                existing_keys=existing,
            )
        except Exception as exc:
            print(f"[aomridge] dataset {row['dataset']} failed: {exc}")
            n = 0
        total += n
        print(f"[aomridge] {row['database_name']}/{row['dataset']} +{n} rows")
    print(f"[aomridge] wrote {total} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
